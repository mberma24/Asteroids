"""Feed-forward and recurrent PPO training on the shared Asteroids curriculum."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum import (CurriculumManager, load_curriculum, reward_matches,
                         task_hash, task_hash_matches)
from .environment import (ASTEROID_FEATURES, PROJECTILE_FEATURES, TEAMMATE_FEATURES,
                          AsteroidsRLEnv, ship_feature_count)
from .evaluation import evaluate_policy, save_evaluation
from .ppo_support import (PPOChampionTracker, SnapshotPolicy, format_episode_block,
                          observation_layout,
                          prune_ppo_checkpoints, truncate_log)


def require_ppo() -> tuple[Any, Any, Any, Any, Any]:
    """Import the optional stack only when a PPO command is actually used."""
    try:
        import torch
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise SystemExit(
            "PPO dependencies are not installed. Run: "
            ".venv/bin/pip install -e '.[ppo]'"
        ) from exc
    return torch, PPO, RecurrentPPO, BaseCallback, (DummyVecEnv, SubprocVecEnv)


def _task_layout(spec, env: AsteroidsRLEnv, curriculum_path: str | Path) -> dict:
    layout = observation_layout(env)
    layout.update({
        "task_hash": task_hash(spec),
        "reward": asdict(spec.reward), "curriculum": str(curriculum_path),
    })
    return layout


def widen_policy(target, source, torch) -> int:
    """Copy a narrower policy into a wider one, zero-filling the new inputs.

    Teammate features are appended to the end of the observation, so every weight the old
    policy learned still lines up with the same input; only the new columns are added, and
    they start at zero so the transferred policy behaves identically until it learns to use
    them. Without this a co-operative run could not start from a solo model at all -- the
    observation is a different width, and nothing else about the network has changed.
    """
    source_state = source.policy.state_dict()
    widened = 0
    with torch.no_grad():
        for name, parameter in target.policy.state_dict().items():
            old = source_state.get(name)
            if old is None:
                continue
            if old.shape == parameter.shape:
                parameter.copy_(old)
            elif (old.dim() == parameter.dim() == 2
                  and old.shape[0] == parameter.shape[0]
                  and old.shape[1] < parameter.shape[1]):
                parameter.zero_()
                parameter[:, :old.shape[1]].copy_(old)
                widened += 1
    return widened


def _stage_env(spec, index: int, layout: dict,
               companion_policy=None) -> AsteroidsRLEnv:
    stage = spec.stages[index]
    reward = (spec.reward if stage.miss_penalty is None else
              replace(spec.reward, miss_penalty=stage.miss_penalty))
    return AsteroidsRLEnv(
        stage.game_config(spec.base), frame_skip=4, max_decisions=stage.max_decisions,
        no_hit_seconds=stage.no_hit_seconds,
        history_frames=int(layout.get("history_frames", 0)),
        history_long_frames=int(layout.get("history_long_frames", 0)),
        history_long_stride=int(layout.get("history_long_stride", 8)),
        max_projectiles=int(layout.get("max_projectiles", 8)), reward_config=reward,
        completion=stage.completion, max_teammates=spec.max_teammates,
        companion_policy=companion_policy, rotate_agent_slot=True,
        observation_version=int(layout.get("version", 4)))


class PPOController:
    """Uniform stateful inference adapter for ordinary and recurrent PPO checkpoints."""

    def __init__(self, checkpoint: str | Path, *, device: str = "auto"):
        _, PPO, RecurrentPPO, _, _ = require_ppo()
        self.checkpoint = Path(checkpoint)
        self.metadata = json.loads(
            (self.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        self.recurrent = bool(self.metadata.get("recurrent"))
        algorithm = RecurrentPPO if self.recurrent else PPO
        self.model = algorithm.load(self.checkpoint / "model.zip", device=device)
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        self.state = None
        self.episode_start[:] = True

    def __call__(self, observation: np.ndarray) -> int:
        if self.recurrent:
            action, self.state = self.model.predict(
                observation, state=self.state, episode_start=self.episode_start,
                deterministic=True)
            self.episode_start[:] = False
        else:
            action, _ = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


def evaluate_ppo_checkpoint(checkpoint: str | Path, output: str | Path, *,
                            episodes: int, seed: int,
                            stage_index: int | None = None) -> dict:
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    layout = metadata.get("observation_layout") or {}
    curriculum_path = layout.get("curriculum")
    if not curriculum_path:
        raise ValueError("PPO checkpoint does not record its curriculum")
    spec = load_curriculum(curriculum_path)
    if stage_index is None:
        state_path = checkpoint / "curriculum_state.json"
        state = (json.loads(state_path.read_text(encoding="utf-8"))
                 if state_path.is_file() else {})
        stage_index = int(state.get("stage", 0))
    if not 0 <= stage_index < len(spec.stages):
        raise ValueError(f"stage must be between 1 and {len(spec.stages)}")
    env = _stage_env(spec, stage_index, layout)
    controller = PPOController(checkpoint)
    report = evaluate_policy(env, controller, list(range(seed, seed + episodes)))
    report.update({"algorithm": metadata.get("algorithm"), "stage": stage_index,
                   "stage_name": spec.stages[stage_index].name,
                   "checkpoint": str(checkpoint)})
    save_evaluation(report, output)
    return report


def retention_stages(spec, stage: int, rotation: int) -> set[int]:
    """Which prior stages to re-score this time round.

    A deterministic rotation rather than a random draw, so coverage is even and a run stays
    reproducible from its seed. With a sample of ten and forty prior stages, every stage is
    revisited every fourth evaluation.
    """
    limit = int(getattr(spec, "retention_stage_limit", 0) or 0)
    prior = list(range(min(stage, limit) if limit > 0 else stage))
    if spec.retention_sample <= 0 or len(prior) <= spec.retention_sample:
        return set(prior)
    start = (rotation * spec.retention_sample) % len(prior)
    return {prior[(start + offset) % len(prior)] for offset in range(spec.retention_sample)}


# --- Parallel evaluation -------------------------------------------------------------
#
# Training runs `--parallel-envs` environments at once; evaluation used to loop seeds on a
# single env, so every evaluation left all but one worker idle. Measured 2026-08-24, that
# was 49.5s of every 113.7s -- 44% of wall-clock spent using one core. Seeds are independent
# episodes of a deterministic policy, so fanning them across processes changes nothing about
# the numbers: same seeds, same episodes, same aggregates.
#
# The serial path is kept and is still used for recurrent policies (whose hidden state makes
# an episode order-dependent) and for team rounds with a companion policy (which is not
# picklable). Those are correctness cases, not performance ones.

_EVAL_WORKER: dict[str, Any] = {}


def eval_worker_count(requested: int = 0) -> int:
    """How many processes to fan evaluation across. 0 means auto."""
    if requested > 0:
        return requested
    override = os.environ.get("PPO_EVAL_WORKERS")
    if override and override.isdigit() and int(override) > 0:
        return int(override)
    return max(1, min(8, os.cpu_count() or 1))


def _eval_worker_init(spec, layout: dict, model_path: str) -> None:
    _, PPO, _, _, _ = require_ppo()
    _EVAL_WORKER.clear()
    _EVAL_WORKER["spec"] = spec
    _EVAL_WORKER["layout"] = layout
    _EVAL_WORKER["envs"] = {}
    _EVAL_WORKER["model"] = PPO.load(model_path, device="cpu")


def _eval_worker_episode(job: tuple[int, int]) -> dict:
    """One held-out episode. Returns the same metrics dict the serial path collects."""
    index, seed = job
    envs = _EVAL_WORKER["envs"]
    if index not in envs:
        envs[index] = _stage_env(_EVAL_WORKER["spec"], index, _EVAL_WORKER["layout"])
    env = envs[index]
    model = _EVAL_WORKER["model"]

    def policy(observation):
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    return evaluate_policy(env, policy, [seed])["episodes"][0]


def evaluate_ppo_stages(model, recurrent: bool, spec, layout: dict, stage: int,
                        eval_seed: int, rotation: int = 0,
                        companion_snapshot=None, workers: int = 0) -> list[dict]:
    import statistics

    companion = (SnapshotPolicy(companion_snapshot, recurrent)
                 if companion_snapshot and spec.max_teammates else None)
    scored = retention_stages(spec, stage, rotation)
    panel = rotation % max(1, spec.evaluation_panels)
    panel_seed = eval_seed + panel * spec.evaluation_episodes

    # Which stages are scored this round, and over which seeds.
    plan: list[tuple[int, int, int]] = []
    for index in range(stage + 1):
        if index != stage and index not in scored:
            continue
        count = (spec.evaluation_episodes if index == stage
                 else spec.retention_evaluation_episodes)
        start_seed = panel_seed if index == stage else eval_seed
        plan.append((index, start_seed, count))

    jobs = [(index, seed) for index, start_seed, count in plan
            for seed in range(start_seed, start_seed + count)]
    raw_by_index: dict[int, list[dict]] = {index: [] for index, _, _ in plan}

    # A recurrent policy carries hidden state, so its episodes are order-dependent; a
    # companion policy is not picklable. Both stay on the serial path for correctness.
    parallel = (not recurrent and companion is None
                and len(jobs) > 1 and eval_worker_count(workers) > 1)

    def evaluate_serially() -> None:
        state = None
        episode_start = np.ones((1,), dtype=bool)

        def policy(observation):
            nonlocal state
            if recurrent:
                action, state = model.predict(
                    observation, state=state, episode_start=episode_start,
                    deterministic=True)
                episode_start[:] = False
                return int(np.asarray(action).item())
            action, _ = model.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        envs: dict[int, AsteroidsRLEnv] = {}
        for index, seed in jobs:
            if index not in envs:
                envs[index] = _stage_env(spec, index, layout, companion)
            # evaluate_policy owns episode resets, so evaluate each seed separately to
            # reset LSTM state between episodes.
            state = None
            episode_start[:] = True
            raw_by_index[index].append(
                evaluate_policy(envs[index], policy, [seed])["episodes"][0])

    if parallel:
        try:
            with tempfile.TemporaryDirectory() as scratch:
                model_path = str(Path(scratch) / "eval_model.zip")
                model.save(model_path)
                count = min(eval_worker_count(workers), len(jobs))
                # "spawn", not the Linux default "fork": this process already owns torch
                # threads and the SubprocVecEnv workers, and forking a multithreaded process
                # deadlocks. SubprocVecEnv is constructed with start_method="spawn" here for
                # the same reason.
                with ProcessPoolExecutor(max_workers=count,
                                         mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_eval_worker_init,
                                         initargs=(spec, layout, model_path)) as pool:
                    for (index, _seed), episode in zip(
                            jobs, pool.map(_eval_worker_episode, jobs)):
                        raw_by_index[index].append(episode)
        except (PermissionError, NotImplementedError):
            # Sandboxes and a few Python builds expose multiprocessing but prohibit the
            # named-semaphore sysconf used by ProcessPoolExecutor. Evaluation is a
            # correctness path, so transparently retain the exact serial implementation.
            raw_by_index = {index: [] for index, _, _ in plan}
            evaluate_serially()
    else:
        evaluate_serially()

    numeric = ("survival_time", "wave", "reward", "asteroid_reward", "shots_fired",
               "asteroids_destroyed", "accuracy", "resolved_accuracy", "waves_cleared",
               "mean_wave_clear_time", "large_destroyed", "medium_destroyed",
               "small_destroyed")
    results = []
    for index, stage_spec in enumerate(spec.stages[:stage + 1]):
        raw = raw_by_index.get(index)
        if not raw:
            # Not scored this round. Zero episodes means "no evidence", which every
            # retention check treats as neutral rather than as a failure.
            results.append({"completion_rate": 0.0, "episodes": 0, "mean_accuracy": 0.0,
                            "mean_wave": 0.0, "stage": stage_spec.name})
            continue
        clear_rate = sum(bool(x.get("completed_stage")) for x in raw) / len(raw)
        limit = stage_spec.max_seconds
        survival_fraction = sum(min(1.0, float(x["survival_time"]) / limit)
                                for x in raw) / len(raw)
        aggregate = {
            "episodes": len(raw),
            "survival_rate_to_limit": sum(x["survived_to_limit"] for x in raw) / len(raw),
            "clear_rate": clear_rate,
            "survival_fraction": survival_fraction,
            # A survival round is scored on how long the ship lives, because that is the
            # objective; clearing the limit is just the top of that scale. The binary was
            # both the wrong quantity and a far noisier one -- on round 11 it read 69.5%
            # where the fraction of time actually survived was 83.0%, and every promotion,
            # retention check and champion comparison was driven by the 69.5%.
            "completion_rate": survival_fraction if stage_spec.survival else clear_rate,
        }
        for name in numeric:
            values = [float(item[name]) for item in raw]
            aggregate.update({f"mean_{name}": statistics.fmean(values),
                              f"median_{name}": statistics.median(values),
                              f"min_{name}": min(values), f"max_{name}": max(values)})
        results.append({"stage": index, "name": stage_spec.name,
                        "validation_panel": panel if index == stage else None,
                        "seed_start": panel_seed if index == stage else eval_seed,
                        **aggregate})
    return results


@dataclass(slots=True)
class PPOTrainSettings:
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lstm_hidden_size: int = 256
    target_kl: float | None = None
    entropy_floor: float | None = None
    """Policy entropy in nats to hold by adapting `ent_coef`; None leaves it constant."""


def set_effective_learning_rate(model, rate: float) -> None:
    """Set every SB3 source of truth for a constant learning rate."""
    model.learning_rate = float(rate)
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(rate)


class EntropyFloorController:
    """Hold policy entropy near a target by adapting the entropy bonus between updates.

    A fixed `ent_coef` has failed in both directions on this curriculum. At 0.01 entropy
    pinned flat at 1.07 nats and the run stalled; at 0.0025 entropy fell monotonically and the
    per-round clear slope decayed to zero in lockstep with it -- 0.711 nats while round 23 was
    still learning at +0.0134 per 1k episodes, 0.588 by the time round 26 sat at +0.0000 over
    112,500 episodes. A rung that was deliberately *easier* learned nothing either, so the
    binding constraint is exploration rather than difficulty.

    So the coefficient is a control input, not a constant. The loop only ever adds exploration
    pressure: it is clamped below by the configured base, so enabling a floor can never leave a
    run less explored than the same run without one.
    """

    EMA_ALPHA = 0.1
    """~10-update horizon. At n_steps=256 across 8 envs that spans roughly 20k decisions --
    long enough to reject per-rollout noise, short enough to react inside one eval interval,
    and enough to absorb the step change in entropy that a curriculum promotion causes."""
    DEADBAND = 0.05
    MAX_MULTIPLE = 8.0
    ABSOLUTE_MAX = 0.05
    GAIN = 0.2
    MAX_LOG_STEP = 0.05
    """At most ~5% per update, so base to the ceiling takes ~42 updates. Fast enough to arrest
    a collapse that took thousands of episodes, and far slower than the policy's own entropy
    response -- outrunning that lag is the usual cause of a controller oscillating."""
    WARMUP_UPDATES = 5

    def __init__(self, base: float, floor: float, *, coefficient: float | None = None,
                 smoothed: float | None = None, updates: int = 0):
        self.base = float(base)
        self.floor = float(floor)
        self.ceiling = min(self.base * self.MAX_MULTIPLE, self.ABSOLUTE_MAX)
        self.coefficient = self.base if coefficient is None else float(coefficient)
        self.smoothed = None if smoothed is None else float(smoothed)
        self.updates = int(updates)

    def update(self, entropy: float, *, approx_kl: float | None = None,
               target_kl: float | None = None) -> float:
        """Fold in one update's measured entropy, in nats, and return the new coefficient."""
        entropy = float(entropy)
        self.smoothed = (entropy if self.smoothed is None
                         else self.smoothed + self.EMA_ALPHA * (entropy - self.smoothed))
        self.updates += 1
        if self.updates <= self.WARMUP_UPDATES:
            return self.coefficient          # seed the average before acting on it
        error = max(-1.0, min(1.0, (self.floor - self.smoothed) / self.floor))
        if abs(error) <= self.DEADBAND:
            return self.coefficient
        # Stay subordinate to the KL cap. SB3 truncates its epoch loop past 1.5 * target_kl,
        # and pushing the policy toward uniform raises approx_kl directly, so climbing while
        # already over target would buy exploration by cutting updates short -- which is the
        # stall this exists to fix. Coming back down is always allowed.
        if error > 0 and approx_kl is not None and target_kl and approx_kl > target_kl:
            return self.coefficient
        step = max(-self.MAX_LOG_STEP, min(self.MAX_LOG_STEP, self.GAIN * error))
        self.coefficient = min(max(self.coefficient * math.exp(step), self.base), self.ceiling)
        return self.coefficient

    def state(self) -> dict:
        return {"floor": self.floor, "base": self.base, "coefficient": self.coefficient,
                "smoothed_entropy": self.smoothed, "updates": self.updates}


def entropy_floor_controller(base: float, floor: float | None,
                             restored: dict | None = None) -> EntropyFloorController | None:
    """Build a controller, or None when no floor is in force.

    A floor of zero is the documented way to switch one off, because the setting persists into
    the checkpoint and would otherwise be sticky across every later resume.
    """
    if floor is None or float(floor) <= 0.0:
        return None
    controller = EntropyFloorController(base=base, floor=float(floor))
    try:
        # Only resume mid-flight state that belongs to this exact loop; a changed flag means
        # the stored coefficient describes a different controller.
        if (restored and float(restored["floor"]) == controller.floor
                and float(restored["base"]) == controller.base):
            controller.coefficient = min(max(float(restored["coefficient"]), controller.base),
                                         controller.ceiling)
            stored = restored.get("smoothed_entropy")
            controller.smoothed = None if stored is None else float(stored)
            controller.updates = int(restored.get("updates", 0))
    except (TypeError, ValueError, KeyError):
        return EntropyFloorController(base=base, floor=float(floor))
    return controller


def _update_record(values: dict, *, episode: int, steps: int, wall_seconds: float,
                   learning_rate: float, ent_coef: float,
                   controller: EntropyFloorController | None = None,
                   target_kl: float | None = None) -> dict:
    """One `ppo_updates.jsonl` line, applying the entropy floor if one is active."""
    wanted = ("train/approx_kl", "train/clip_fraction", "train/entropy_loss",
              "train/policy_gradient_loss", "train/value_loss",
              "train/explained_variance", "train/loss")
    record = {
        "episode": episode, "environment_steps": int(steps),
        "effective_learning_rate": float(learning_rate),
        "wall_seconds": wall_seconds,
    }
    record.update({name.removeprefix("train/"): float(values[name])
                   for name in wanted if name in values})
    # SB3 logs entropy_loss as *negative* mean entropy, so flip it: everything downstream --
    # the controller included -- works in positive nats.
    entropy = (-float(values["train/entropy_loss"])
               if "train/entropy_loss" in values else None)
    if entropy is not None:
        record["entropy"] = entropy
    record["ent_coef"] = float(ent_coef)
    if controller is not None and entropy is not None:
        record["ent_coef"] = controller.update(
            entropy, approx_kl=record.get("approx_kl"), target_kl=target_kl)
        record.update({"entropy_floor": controller.floor,
                       "entropy_coefficient_base": controller.base,
                       "smoothed_entropy": controller.smoothed,
                       "entropy_controller_updates": controller.updates})
    return record


def train_ppo_curriculum(curriculum_path: str | Path, output_dir: str | Path, *,
                         episodes: int, recurrent: bool, seed: int = 0,
                         parallel_envs: int = 8, history_frames: int = 8,
                         history_long_frames: int = 8, history_long_stride: int = 8,
                         eval_every: int = 250, eval_seed: int = 10_000,
                         keep_checkpoints: int = 3, champion_patience: int = 4,
                         resume: str | Path | None = None,
                         initialize_from: str | Path | None = None,
                         start_stage: int = 0, device: str = "auto",
                         settings: PPOTrainSettings | None = None,
                         learning_rate: float | None = None,
                         ent_coef: float | None = None,
                         vf_coef: float | None = None,
                         entropy_floor: float | None = None,
                         stop_when_mastered: bool = False,
                         encoder: str = "mlp") -> Path:
    """Train PPO until ``episodes`` additional vector episodes have completed."""
    if resume and initialize_from:
        raise ValueError("resume and initialize_from are mutually exclusive")
    torch, PPO, RecurrentPPO, BaseCallback, vec_classes = require_ppo()
    from .gym_env import CurriculumGymEnv

    supplied_settings = settings is not None
    settings = settings or PPOTrainSettings()
    spec = load_curriculum(curriculum_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "curriculum_state.json"
    saved_state: dict[str, Any] = {}
    start_episode = start_steps = 0
    resume_metadata: dict[str, Any] = {}
    if resume:
        resume = Path(resume)
        resume_metadata = json.loads(
            (resume / "metadata.json").read_text(encoding="utf-8"))
        if bool(resume_metadata.get("recurrent")) != recurrent:
            raise ValueError("cannot resume a different PPO architecture")
        if not supplied_settings and resume_metadata.get("settings"):
            settings = PPOTrainSettings(**resume_metadata["settings"])
        start_episode = int(resume_metadata.get("episodes", 0))
        start_steps = int(resume_metadata.get("environment_steps", 0))
        resume_state = resume / "curriculum_state.json"
        if not resume_state.is_file():
            resume_state = resume.parent / "curriculum_state.json"
        if resume_state.is_file():
            saved_state = json.loads(resume_state.read_text(encoding="utf-8"))
    initial_stage = int(saved_state.get("stage", start_stage))
    if not 0 <= initial_stage < len(spec.stages):
        raise ValueError(f"start_stage must be between 0 and {len(spec.stages) - 1}")
    manager = CurriculumManager(
        spec, seed, stage=initial_stage,
        streak=int(saved_state.get("streak", 0)),
        mastered=bool(saved_state.get("mastered", False)),
        promotion_history=saved_state.get("promotion_history"),
        promotion_samples=saved_state.get("promotion_samples"))
    if spec.promotion_pool and "promotion_samples" not in saved_state and resume:
        from .curriculum import promotion_samples_from_log
        for source_log in (destination / "evaluation.jsonl",
                           resume.parent / "evaluation.jsonl"):
            manager.promotion_samples = promotion_samples_from_log(
                source_log, manager.stage, spec.promotion_window,
                through_episode=start_episode)
            if manager.promotion_samples:
                break
        manager.streak = len(manager.promotion_samples)
    recovery_stage = saved_state.get("recovery_stage")
    recovery_stage = None if recovery_stage is None else int(recovery_stage)

    prototype = _stage_env(spec, 0, {
        "history_frames": history_frames, "history_long_frames": history_long_frames,
        "history_long_stride": history_long_stride, "max_projectiles": 8,
        "version": spec.observation_version})
    layout = _task_layout(spec, prototype, curriculum_path)
    if resume:
        stored_layout = resume_metadata.get("observation_layout") or {}
        compared = ("version", "max_asteroids", "max_projectiles", "history_frames",
                    "history_long_frames", "history_long_stride", "history_offsets",
                    "mobile", "actions")
        if (any(stored_layout.get(key) != layout.get(key) for key in compared)
                or not task_hash_matches(stored_layout.get("task_hash"), spec)
                or not reward_matches(stored_layout.get("reward"), spec.reward)):
            raise ValueError("PPO checkpoint observation or curriculum manifest does not match")
    initialize_metadata: dict[str, Any] = {}
    expandable = False
    widen_by = 0
    if initialize_from:
        initialize_from = Path(initialize_from)
        initialize_metadata = json.loads(
            (initialize_from / "metadata.json").read_text(encoding="utf-8"))
        if bool(initialize_metadata.get("recurrent")) != recurrent:
            raise ValueError("cannot initialize from a different PPO architecture")
        if not supplied_settings and initialize_metadata.get("settings"):
            settings = PPOTrainSettings(**initialize_metadata["settings"])
        stored_layout = initialize_metadata.get("observation_layout") or {}
        structural = ("max_asteroids", "max_projectiles", "history_frames",
                      "history_long_frames", "history_long_stride", "history_offsets",
                      "mobile", "actions")
        widen_by = (prototype.observation_size
                    - int(initialize_metadata.get("observation_size", -1)))
        teammate_growth = (int(layout.get("max_teammates", 0))
                           - int(stored_layout.get("max_teammates", 0)))
        # A solo model can seed a co-operative run: teammate slots are appended last, so the
        # existing input weights keep their meaning and only new columns need adding.
        global_growth = int(layout.get("global_features", 0)) - int(
            stored_layout.get("global_features", 0))
        expandable = (widen_by > 0 and widen_by ==
                      teammate_growth * TEAMMATE_FEATURES + global_growth)
        if (any(stored_layout.get(key) != layout.get(key) for key in structural)
                or (not expandable and int(initialize_metadata.get("observation_size", -1))
                    != prototype.observation_size)
                or int(initialize_metadata.get("num_actions", -1)) != prototype.num_actions):
            raise ValueError("PPO initialization checkpoint observation/action layout does not match")
    n_envs = max(1, int(parallel_envs))
    # Companion ships load this snapshot rather than the live model, which cannot cross a
    # process boundary. Only written when the curriculum actually has multi-ship rounds.
    companion_snapshot = (destination / "companion.zip") if spec.max_teammates else None
    factories = [
        (lambda rank=rank: CurriculumGymEnv(
            curriculum_path, rank=rank, num_envs=n_envs, seed=seed,
            episode_offset=start_episode, stage=manager.stage,
            history_frames=history_frames, history_long_frames=history_long_frames,
            history_long_stride=history_long_stride, eval_seed=eval_seed,
            companion_snapshot=companion_snapshot))
        for rank in range(n_envs)
    ]
    DummyVecEnv, SubprocVecEnv = vec_classes
    vec_env = (DummyVecEnv(factories) if n_envs == 1 else
               SubprocVecEnv(factories, start_method="spawn"))

    auto_device = device == "auto"
    if auto_device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    algorithm = RecurrentPPO if recurrent else PPO
    policy = "MlpLstmPolicy" if recurrent else "MlpPolicy"
    policy_kwargs: dict[str, Any]
    if recurrent:
        policy_kwargs = {"lstm_hidden_size": settings.lstm_hidden_size,
                         "n_lstm_layers": 1, "shared_lstm": False,
                         "enable_critic_lstm": True,
                         "net_arch": {"pi": [256], "vf": [256]}}
    else:
        policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    if encoder == "set":
        # The asteroid and projectile blocks are sets, not fixed slots: they are re-sorted by
        # distance every decision, so a plain MLP sees its own inputs permuted underneath it.
        from .networks import SetFeaturesExtractor
        slots = len(prototype.history_slots)
        policy_kwargs = dict(policy_kwargs)
        policy_kwargs["features_extractor_class"] = SetFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "ship_features": ship_feature_count(prototype.config),
            "asteroid_slots": prototype.max_asteroids,
            "asteroid_features": ASTEROID_FEATURES + 2 * slots,
            "projectile_slots": prototype.max_projectiles,
            "projectile_features": PROJECTILE_FEATURES,
            "teammate_slots": prototype.max_teammates,
            "teammate_features": TEAMMATE_FEATURES,
            "global_features": int(layout.get("global_features", 0)),
        }
    common = dict(
        learning_rate=settings.learning_rate, n_steps=settings.n_steps,
        batch_size=settings.batch_size, n_epochs=settings.n_epochs,
        gamma=settings.gamma, gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range, ent_coef=settings.ent_coef,
        vf_coef=settings.vf_coef, max_grad_norm=settings.max_grad_norm,
        policy_kwargs=policy_kwargs, device=device, seed=seed, verbose=0,
        target_kl=settings.target_kl)
    try:
        if resume or initialize_from:
            source = Path(resume or initialize_from)
            if initialize_from and expandable:
                narrow = algorithm.load(source / "model.zip", device="cpu")
                model = algorithm(policy, vec_env, **common)
                widened = widen_policy(model, narrow, torch)
                print(f"transferred a narrower policy, zero-filling {widen_by} new inputs "
                      f"across {widened} layers", flush=True)
            else:
                model = algorithm.load(source / "model.zip", env=vec_env, device=device)
                set_effective_learning_rate(model, settings.learning_rate)
        else:
            model = algorithm(policy, vec_env, **common)
    except (RuntimeError, NotImplementedError):
        if not auto_device or device != "mps":
            vec_env.close()
            raise
        print("MPS does not support this PPO operation; falling back to CPU", flush=True)
        device = "cpu"
        common["device"] = device
        if resume or initialize_from:
            source = Path(resume or initialize_from)
            model = algorithm.load(source / "model.zip", env=vec_env, device=device)
            set_effective_learning_rate(model, settings.learning_rate)
        else:
            model = algorithm(policy, vec_env, **common)

    if initialize_from:
        # Transfer perception and policy, but do not carry task-specific Adam moments or
        # timestep schedules into a curriculum with different motion statistics.
        model.policy.optimizer.state.clear()
        set_effective_learning_rate(model, settings.learning_rate)

    if learning_rate is not None:
        # `model.learning_rate = x` alone does nothing to a loaded model: SB3 reads the rate
        # from `lr_schedule`, which `load` restores from the checkpoint. The schedule has to
        # be rebuilt and the live optimizer groups set for an override to take effect.
        set_effective_learning_rate(model, learning_rate)
        print(f"learning rate overridden to {learning_rate:g}", flush=True)

    if ent_coef is not None:
        # Unlike the Adam rate, SB3 reads `ent_coef` straight off the algorithm at every
        # train() call, so assigning it is enough. Keep `settings` in step so the value is
        # what a later --resume restores.
        settings.ent_coef = float(ent_coef)
        model.ent_coef = float(ent_coef)
        print(f"entropy coefficient overridden to {ent_coef:g}", flush=True)

    if vf_coef is not None:
        # Read off the algorithm at every train() call, like `ent_coef`. Raising it buys the
        # critic a larger share of the update: measured 2026-08-24, v7's value function had
        # explained_variance 0.479 with excursions below zero, so GAE advantages were built
        # on a critic that sometimes did worse than predicting the mean.
        settings.vf_coef = float(vf_coef)
        model.vf_coef = float(vf_coef)
        print(f"value coefficient overridden to {vf_coef:g}", flush=True)

    if entropy_floor is not None:
        settings.entropy_floor = float(entropy_floor) if entropy_floor > 0 else None
    entropy_controller = entropy_floor_controller(
        base=float(settings.ent_coef), floor=settings.entropy_floor,
        restored=(resume_metadata or {}).get("entropy_controller"))
    if entropy_controller is not None:
        model.ent_coef = entropy_controller.coefficient
        print(f"entropy floor {entropy_controller.floor:g} nats; coefficient adapts in "
              f"[{entropy_controller.base:g}, {entropy_controller.ceiling:g}]", flush=True)

    effective_learning_rate = float(
        learning_rate if learning_rate is not None else settings.learning_rate)

    training_log = destination / "training.jsonl"
    evaluation_log = destination / "evaluation.jsonl"
    if resume:
        removed = (truncate_log(training_log, start_episode)
                   + truncate_log(evaluation_log, start_episode))
        if removed:
            print(json.dumps({"discarded_unsaved_log_records": removed}), flush=True)
    champion = PPOChampionTracker(
        destination, spec.retention_completion, patience=champion_patience,
        retention_floor=spec.retention_floor,
        learning_rate=effective_learning_rate,
        # A floor of initial/8 was reached inside a single curriculum round, after which
        # the run could not move. Cutting the rate is the right response to instability, not
        # to a hard stage where progress is genuinely slow.
        minimum_learning_rate=effective_learning_rate / 3,
        promotion_completion=spec.promotion_completion,
        clear_target=spec.promotion_clear_rate,
        accuracy_targets=tuple(spec.promotion_accuracy if s.promotion_accuracy is None
                               else s.promotion_accuracy for s in spec.stages))
    champion.bootstrap(evaluation_log)
    started = time.monotonic()
    target_episode = start_episode + int(episodes)

    class CurriculumCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.episode = start_episode
            self.pending: list[dict] = []
            self.next_eval = ((start_episode // eval_every) + 1) * eval_every
            self.last_checkpoint = destination
            self.finished = False
            """Set when --stop-when-mastered fires, so the budget is not spent needlessly."""

        def _save(self, evaluation: dict | None = None) -> Path:
            checkpoint = destination / f"checkpoint_{self.episode:06d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            self.model.save(checkpoint / "model.zip")
            metadata = {
                        "encoder": encoder,
                "algorithm": "recurrent_ppo" if recurrent else "ppo",
                "recurrent": recurrent, "episodes": self.episode,
                "environment_steps": int(self.num_timesteps),
                "observation_size": prototype.observation_size,
                "num_actions": prototype.num_actions,
                "observation_layout": layout, "settings": asdict(settings),
                "configured_learning_rate": settings.learning_rate,
                "effective_learning_rate": effective_learning_rate,
                # `train-forever.sh` re-execs with RESUME= on every restart, so the live
                # coefficient has to survive one. Without this each restart would drop back
                # to base and start collapsing again.
                "entropy_controller": (None if entropy_controller is None
                                       else entropy_controller.state()),
                "parent_checkpoint": str(initialize_from) if initialize_from else None,
                "parallel_envs": n_envs, "device": str(device),
                "wall_seconds": time.monotonic() - started,
                "segment_start_steps": start_steps,
                "decisions_per_second": ((int(self.num_timesteps) - start_steps) /
                                         max(time.monotonic() - started, 1e-6)),
            }
            (checkpoint / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            state = {"stage": manager.stage, "streak": manager.streak,
                     "promotion_history": manager.promotion_history,
                     "promotion_samples": manager.promotion_samples,
                     "mastered": manager.mastered,
                     "recovery_stage": recovery_stage}
            state_text = json.dumps(state, indent=2) + "\n"
            state_path.write_text(state_text, encoding="utf-8")
            (checkpoint / "curriculum_state.json").write_text(
                state_text, encoding="utf-8")
            self.last_checkpoint = checkpoint
            if evaluation is not None:
                action = champion.consider(
                    evaluation, checkpoint, allow_recovery=evaluation["training_stage"] > 0)
                evaluation.update({"champion_action": action,
                                   "champion_episode": champion.state.get("episode")})
                with evaluation_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(evaluation) + "\n")
            prune_ppo_checkpoints(destination, keep_checkpoints)
            return checkpoint

        def _evaluate(self) -> None:
            nonlocal recovery_stage
            evaluated_stage = manager.stage
            if companion_snapshot is not None:
                # Refresh before scoring, so companions in both training and evaluation are
                # the same policy the learner currently is.
                self.model.save(companion_snapshot)
            results = evaluate_ppo_stages(
                self.model, recurrent, spec, layout, evaluated_stage, eval_seed,
                rotation=self.episode // max(1, eval_every),
                companion_snapshot=companion_snapshot)
            promoted = manager.consider_promotion(results)
            evaluation = {
                "episode": self.episode,
                "environment_steps": int(self.num_timesteps),
                "training_stage": evaluated_stage, "next_training_stage": manager.stage,
                "promotion_streak": manager.streak, "promoted": promoted,
                "promotion_pool": manager.promotion_pool,
                "promotion_completion_target": spec.promotion_completion,
                "promotion_clear_rate_target": spec.promotion_clear_rate,
                "stages": results,
            }
            current = results[evaluated_stage]
            target = (spec.promotion_accuracy if
                      spec.stages[evaluated_stage].promotion_accuracy is None else
                      spec.stages[evaluated_stage].promotion_accuracy)
            evaluation["promotion_accuracy_target"] = target
            self._save(evaluation)
            forgotten = champion.forgotten_stage(evaluation)
            recovery_stage = (forgotten if
                              evaluation.get("champion_action") == "recover" else None)
            self.training_env.env_method(
                "set_curriculum_state", manager.stage, recovery_stage)
            rate = (int(self.num_timesteps) - start_steps) / max(
                time.monotonic() - started, 1e-6)
            print(f"  PPO eval @ {self.episode}: stage {evaluated_stage + 1} "
                  f"completion {current['completion_rate']:.1%}, "
                  f"clear {current['clear_rate']:.1%}, "
                  f"accuracy {current['mean_accuracy']:.3f}" +
                  ((f", pool {manager.promotion_pool['evaluations']}/"
                    f"{manager.promotion_pool['required_evaluations']} "
                    f"clear {manager.promotion_pool['clear_rate']:.1%}")
                   if manager.promotion_pool else "") +
                  f", {rate:.0f} decisions/s" +
                  (f" - PROMOTED to stage {manager.stage + 1}" if promoted else ""),
                  flush=True)
            print(f"  watch champion: ./run.sh preview {destination}", flush=True)
            if stop_when_mastered and manager.mastered:
                print(f"  final stage {manager.stage + 1} mastered; stopping", flush=True)
                self.finished = True

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                metrics = info.get("episode_metrics")
                if not metrics or self.episode >= target_episode:
                    continue
                self.episode += 1
                record = {"episode": self.episode,
                          "environment_steps": int(self.num_timesteps),
                          "curriculum_stage": int(info.get("curriculum_stage", 0)),
                          "stage_name": spec.stages[int(info.get("curriculum_stage", 0))].name,
                          **metrics}
                with training_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                self.pending.append(record)
                if len(self.pending) >= 250 or self.episode == target_episode:
                    print(format_episode_block(self.pending), flush=True)
                    self.pending = []
            while self.episode >= self.next_eval and self.next_eval <= target_episode:
                self._evaluate()
                self.next_eval += eval_every
            return self.episode < target_episode and not self.finished

        def _on_rollout_end(self) -> None:
            values = getattr(self.model.logger, "name_to_value", {})
            wanted = ("train/approx_kl", "train/clip_fraction", "train/entropy_loss",
                      "train/policy_gradient_loss", "train/value_loss",
                      "train/explained_variance", "train/loss")
            if not any(name in values for name in wanted):
                return
            record = _update_record(
                values, episode=self.episode, steps=int(self.num_timesteps),
                wall_seconds=time.monotonic() - started,
                learning_rate=float(self.model.policy.optimizer.param_groups[0]["lr"]),
                ent_coef=float(self.model.ent_coef), controller=entropy_controller,
                target_kl=settings.target_kl)
            if entropy_controller is not None:
                # SB3 re-reads this at every train() call, so assigning it here -- on the
                # rollout end that immediately precedes the update -- takes effect at once.
                self.model.ent_coef = record["ent_coef"]
            with (destination / "ppo_updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

        def _on_training_end(self) -> None:
            if self.episode and self.last_checkpoint.name != f"checkpoint_{self.episode:06d}":
                self._save()

    callback = CurriculumCallback()
    # The callback, not this loose upper bound, defines the exact episode budget.
    max_timesteps = max(1, episodes) * max(stage.max_decisions for stage in spec.stages)
    try:
        model.learn(total_timesteps=max_timesteps, callback=callback,
                    reset_num_timesteps=not bool(resume), progress_bar=False)
    finally:
        vec_env.close()
    return callback.last_checkpoint
