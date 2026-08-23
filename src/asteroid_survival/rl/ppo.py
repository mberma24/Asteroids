"""Feed-forward and recurrent PPO training on the shared Asteroids curriculum."""
from __future__ import annotations

import json
import time
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
        companion_policy=companion_policy,
        global_features=int(layout.get("version", 4)) >= 5)


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
    prior = list(range(stage))
    if spec.retention_sample <= 0 or len(prior) <= spec.retention_sample:
        return set(prior)
    start = (rotation * spec.retention_sample) % len(prior)
    return {prior[(start + offset) % len(prior)] for offset in range(spec.retention_sample)}


def evaluate_ppo_stages(model, recurrent: bool, spec, layout: dict, stage: int,
                        eval_seed: int, rotation: int = 0,
                        companion_snapshot=None) -> list[dict]:
    results = []
    companion = (SnapshotPolicy(companion_snapshot, recurrent)
                 if companion_snapshot and spec.max_teammates else None)
    scored = retention_stages(spec, stage, rotation)
    for index, stage_spec in enumerate(spec.stages[:stage + 1]):
        if index != stage and index not in scored:
            # Not scored this round. Zero episodes means "no evidence", which every
            # retention check treats as neutral rather than as a failure.
            results.append({"completion_rate": 0.0, "episodes": 0, "mean_accuracy": 0.0,
                            "mean_wave": 0.0, "stage": stage_spec.name})
            continue
        count = (spec.evaluation_episodes if index == stage else
                 spec.retention_evaluation_episodes)
        env = _stage_env(spec, index, layout, companion)
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

        # evaluate_policy owns episode resets, so evaluate each seed separately to reset LSTM.
        episodes = []
        panel = rotation % max(1, spec.evaluation_panels)
        panel_seed = eval_seed + panel * spec.evaluation_episodes
        start_seed = panel_seed if index == stage else eval_seed
        for seed in range(start_seed, start_seed + count):
            state = None
            episode_start[:] = True
            episodes.append(evaluate_policy(env, policy, [seed]))
        raw = [report["episodes"][0] for report in episodes]
        # Reuse the canonical aggregator once more with recorded results avoided by evaluating
        # as one batch for feed-forward; recurrent aggregation is calculated directly below.
        numeric = ("survival_time", "wave", "reward", "asteroid_reward", "shots_fired",
                   "asteroids_destroyed", "accuracy", "resolved_accuracy", "waves_cleared",
                   "mean_wave_clear_time", "large_destroyed", "medium_destroyed",
                   "small_destroyed")
        import statistics
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
                        "seed_start": start_seed, **aggregate})
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


def set_effective_learning_rate(model, rate: float) -> None:
    """Set every SB3 source of truth for a constant learning rate."""
    model.learning_rate = float(rate)
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(rate)


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
        promotion_history=saved_state.get("promotion_history"))
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
                  f"accuracy {current['mean_accuracy']:.3f}, {rate:.0f} decisions/s" +
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
            record = {
                "episode": self.episode, "environment_steps": int(self.num_timesteps),
                "effective_learning_rate": float(
                    self.model.policy.optimizer.param_groups[0]["lr"]),
                "wall_seconds": time.monotonic() - started,
            }
            record.update({name.removeprefix("train/"): float(values[name])
                           for name in wanted if name in values})
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
