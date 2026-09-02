"""Centralized two-ship PPO with simultaneous joint actions and team-only success."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from ..controllers import PilotController
from ..math2d import Vec2, wrapped_delta
from .curriculum import CurriculumStage, load_curriculum
from .environment import (ASTEROID_FEATURES, MOBILE_ACTIONS, MOBILE_SHIP_FEATURES,
                          PROJECTILE_FEATURES, TEAMMATE_FEATURES, RewardConfig,
                          global_feature_count)
from .multiagent import MultiAgentAsteroidsEnv
from .ppo_support import training_seed
from .training import prune_checkpoints


TEAM_CURRICULUM_VERSION = 8
TEAM_SHIPS = 2
MAX_ASTEROIDS = 26
EVAL_SEED = 10_000
TEST_SEED = 10_256


@dataclass(slots=True)
class TeamPPOSettings:
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.02
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 1024
    batch_size: int = 1024
    n_epochs: int = 4
    challenger_patience: int = 8


TEAM_REWARD = RewardConfig(
    # Preserve a dense aiming signal without letting fragment farming dominate survival.
    # In v4, ~112 hits paid ~224 while dying cost only 20, making a stationary gun turret
    # rational at round 29. Size-aware values keep a whole large-asteroid family bounded.
    large=0.60, medium=0.30, small=0.15,
    survival_bonus=0.10, round_clear=30.0,
    death_penalty=30.0, timeout_penalty=8.0,
    miss_penalty=0.01, collision_penalty=10.0,
    friendly_fire_penalty=20.0, friendly_fire_dealt_penalty=20.0,
    time_scaled_survival=True, active_time_penalty=0.0, safety_progress=0.0,
)


# Each manoeuvre has a firing counterpart at the same offset.  Factoring the action this
# way lets the policy learn each ship's movement once instead of relearning it across a
# 16x16 table of opaque joint categories.
MANEUVER_ACTIONS = tuple(MOBILE_ACTIONS[index] for index in range(8))
FIRING_ACTIONS = tuple(MOBILE_ACTIONS[index + 8] for index in range(8))


def team_curriculum() -> tuple[CurriculumStage, ...]:
    """Forced-shooting lessons, survival warmups, and a gradual production ladder."""
    solo = load_curriculum("configs/rl-survival-v2.toml")
    base = solo.stages[0]
    waves = (
        replace(base, name="team-wave-1", survival=False, ships=2,
                composition=(1, 1), target_waves=1, max_seconds=20.0,
                min_speed=18.0, max_speed=26.0, promotion_clear_rate=0.85,
                promotion_accuracy=0.0),
        replace(base, name="team-wave-2", survival=False, ships=2,
                composition=(1, 1, 1, 1), target_waves=1, max_seconds=25.0,
                min_speed=22.0, max_speed=32.0, promotion_clear_rate=0.85,
                promotion_accuracy=0.0),
        replace(base, name="team-wave-3", survival=False, ships=2,
                composition=(1, 1, 1, 1), target_waves=1, max_seconds=25.0,
                min_speed=22.0, max_speed=32.0, promotion_clear_rate=0.85,
                promotion_accuracy=0.0),
        replace(base, name="team-wave-4", survival=False, ships=2,
                composition=(2, 2, 1, 1), target_waves=1, max_seconds=30.0,
                min_speed=26.0, max_speed=38.0, promotion_clear_rate=0.85,
                promotion_accuracy=0.0),
    )
    bridges = tuple(
        replace(base, name=f"team-survival-warmup-{index}", survival=True, ships=2,
                initial_asteroids=count, spawn_interval=interval,
                max_seconds=30.0, no_hit_seconds=0.0,
                promotion_clear_rate=0.75, promotion_accuracy=0.0)
        for index, (count, interval) in enumerate(((2, 6.0), (4, 3.5), (6, 2.75)), 1)
    )
    ladder = tuple(
        replace(stage, name=f"team-{stage.name}", ships=2,
                promotion_clear_rate=0.75, promotion_accuracy=0.0)
        for stage in solo.stages)
    # Solo rounds 28 -> 29 jump from a 50/50 medium/large mixture to all-large rocks.
    # The team champion measured 76.6% on the former, 70.3% at 56.25% large, 67.2% at
    # 68.75% large, and about 53-61% on the latter.  Insert one-composition-only rung per
    # sixteenth while holding round-28 physics fixed.  Old round-29 checkpoints were at
    # index 35; the first new bridge deliberately occupies that same index.
    round_28 = ladder[27]
    large_bridges = tuple(
        replace(
            round_28,
            name=f"team-large-bridge-{large}-of-16",
            asteroid_size=[2] * (16 - large) + [3] * large,
        )
        for large in range(9, 17)
    )
    return waves + bridges + ladder[:28] + large_bridges + ladder[28:]


def team_stage_config(stage: CurriculumStage):
    solo = load_curriculum("configs/rl-survival-v2.toml")
    config = stage.game_config(solo.base)
    # The legacy 80px radius starts the ships only 160px apart; measured friendly-fire kills
    # land around 108px, so the first shot can decide the episode before either policy has
    # acted meaningfully. A 250px radius leaves room for actual deconfliction while keeping
    # the same 900px toroidal arena and full collision rules.
    config.ship.spawn_radius = 250.0
    # Add collision modes one at a time, but never distort projectile lifetime.  The old
    # 0.5-second stage cut effective range by 66%, creating a shooting cliff that did not
    # exist in the final game and teaching the policy to spam unreachable targets.
    if stage.name == "team-wave-1":
        config.ship.friendly_collisions = "off"
    elif stage.name == "team-wave-2":
        config.ship.friendly_collisions = "ships"
    return config


def require_team_ppo():
    try:
        import torch
        from gymnasium import Env, spaces
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("team PPO requires pip install -e '.[ppo]'") from exc
    return torch, Env, spaces, MaskablePPO, BaseCallback, DummyVecEnv, SubprocVecEnv


class CentralizedTeamEnv(Env):
    """Gym-compatible adapter: two local views in, two simultaneous actions out."""

    metadata = {"render_modes": []}

    def __init__(self, *, rank: int = 0, num_envs: int = 1, seed: int = 0,
                 stage: int = 0, force_stage: int | None = None,
                 episode_offset: int = 0, evasion_shield: bool = False,
                 fire_assist: bool = False):
        super().__init__()
        self.rank = int(rank)
        self.num_envs = max(1, int(num_envs))
        self.base_seed = int(seed)
        self.episode_offset = max(0, int(episode_offset))
        self.current_stage = int(stage)
        self.force_stage = force_stage
        self.evasion_shield = bool(evasion_shield)
        self.fire_assist = bool(fire_assist)
        self._fire_pilot = PilotController()
        self.local_episode = 0
        self.rng = random.Random(seed + rank * 100_003)
        self.stages = team_curriculum()
        self.stage_index = int(force_stage if force_stage is not None else stage)
        self.native = self._make_native(self.stage_index)
        self.local_width = self.native.observation_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.local_width,), dtype=np.float32)
        # Two manoeuvres plus one central fire assignment (none / first / second) use 19
        # logits and make simultaneous fire impossible by construction.
        self.action_space = spaces.MultiDiscrete((8, 8, 3))
        self._order = ("ship1", "ship2")
        self._observations: dict[str, np.ndarray] = {}
        self._movement_lesson = False

    def _make_native(self, index: int) -> MultiAgentAsteroidsEnv:
        stage = self.stages[index]
        config = team_stage_config(stage)
        return MultiAgentAsteroidsEnv(
            config, max_ships=2, max_asteroids=MAX_ASTEROIDS,
            max_decisions=stage.max_decisions, history_frames=8,
            history_long_frames=8, history_long_stride=8,
            reward_config=TEAM_REWARD, completion=stage.completion,
            terminate_on_team_death=True, observation_version=7,
            mask_unsafe_fire=config.ship.friendly_collisions == "full")

    def set_stage(self, stage: int) -> None:
        self.current_stage = max(0, min(int(stage), len(self.stages) - 1))

    def _sample_stage(self) -> int:
        if self.force_stage is not None or self.current_stage == 0:
            return int(self.force_stage if self.force_stage is not None else self.current_stage)
        draw = self.rng.random()
        if draw < 0.90:
            return self.current_stage
        # Retain the adjacent skill, but stop replaying hazard-free lessons after the
        # policy has reached full collision and friendly-fire physics.
        return self.current_stage - 1

    def _joint_observation(self) -> np.ndarray:
        return np.concatenate([self._observations[key] for key in self._order]).astype(
            np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.base_seed = int(seed)
            self.local_episode = 0
            self.rng.seed(seed + self.rank * 100_003)
        self.stage_index = self._sample_stage()
        self.native = self._make_native(self.stage_index)
        serial = self.episode_offset + self.rank + self.local_episode * self.num_envs
        episode_seed = (int(seed) if self.force_stage is not None and seed is not None else
                        training_seed(self.base_seed, serial, EVAL_SEED, 384))
        self.local_episode += 1
        self._observations, _ = self.native.reset(episode_seed)
        self._movement_lesson = False
        self._order = (("ship2", "ship1") if self.rng.random() < 0.5
                       else ("ship1", "ship2"))
        return self._joint_observation(), {
            "curriculum_stage": self.stage_index, "controller_order": self._order,
            "movement_lesson": self._movement_lesson}

    def action_masks(self) -> np.ndarray:
        masks = self.native.action_masks()
        first, second = (masks[key] for key in self._order)
        can_fire = (bool(np.any(first[8:])), bool(np.any(second[8:])))
        if self._movement_lesson:
            can_fire = (False, False)
        # MaskablePPO represents MultiDiscrete masks as the concatenation of each branch.
        return np.concatenate((first[:8], second[:8], (True, *can_fire)))

    def _evasion_override(self, ship_id: str) -> int | None:
        """Analytical emergency manoeuvre, or None when no imminent collision exists."""
        ship = next(item for item in self.native.state.ships if item.id == ship_id)
        if not ship.alive:
            return None
        origin = Vec2(ship.x, ship.y)
        threats = []
        for rock in self.native.state.asteroids:
            delta = wrapped_delta(origin, Vec2(rock.x, rock.y),
                                  self.native.state.width, self.native.state.height)
            relative = Vec2(rock.vx - ship.vx, rock.vy - ship.vy)
            speed2 = relative.dot(relative)
            if speed2 <= 1e-9:
                continue
            ttc = -delta.dot(relative) / speed2
            if not 0.0 < ttc < 1.5:
                continue
            miss = Vec2(delta.x + relative.x * ttc,
                        delta.y + relative.y * ttc).length()
            if miss <= rock.radius + self.native.config.ship.radius + 36.0:
                threats.append((ttc, delta))
        if not threats:
            return None
        escape = Vec2(0.0, 0.0)
        for ttc, delta in threats:
            scale = 1.0 / ((ttc + 0.20) * max(delta.length(), 1e-6))
            escape = Vec2(escape.x - delta.x * scale, escape.y - delta.y * scale)
        desired = math.atan2(escape.y, escape.x)
        error = (desired - ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = -1 if error < -0.12 else (1 if error > 0.12 else 0)
        thrust = abs(error) < 0.65
        return {
            (0, False): 0, (-1, False): 1, (1, False): 2,
            (0, True): 5, (-1, True): 6, (1, True): 7,
        }[(turn, thrust)]

    def step(self, actions):
        choice = np.asarray(actions, dtype=np.int64).reshape(-1)
        if choice.size != 3:
            raise ValueError("team action must contain two manoeuvres and one shooter")
        first, second, shooter = (int(value) for value in choice)
        manoeuvres = [first, second]
        if self.evasion_shield:
            for slot, ship_id in enumerate(self._order):
                override = self._evasion_override(ship_id)
                if override is not None:
                    manoeuvres[slot] = override
        first, second = manoeuvres
        mapped = {self._order[0]: first, self._order[1]: second}
        if shooter in (1, 2):
            slot = shooter - 1
            firing = int(FIRING_ACTIONS[manoeuvres[slot]])
            ship_id = self._order[slot]
            # A rejected shot becomes the identical manoeuvre without firing.  Full
            # friendly-fire physics remains enabled, including hits caused by later motion;
            # this merely supplies the basic trajectory interlock a real two-ship team has.
            if self.native.action_masks()[ship_id][firing]:
                mapped[ship_id] = firing
        if self.fire_assist:
            # Preserve learned manoeuvres, but let either ship take an independently safe,
            # analytically aligned shot. This tests whether the one-shooter action branch is
            # the round-29 capacity bottleneck without changing navigation.
            masks = self.native.action_masks()
            for slot, ship_id in enumerate(self._order):
                firing = int(FIRING_ACTIONS[manoeuvres[slot]])
                if self._fire_pilot.action(self.native.state, ship_id).fire \
                        and masks[ship_id][firing]:
                    mapped[ship_id] = firing
        self._observations, reward, terminated, truncated, info = self.native.step(mapped)
        if terminated or truncated:
            info["curriculum_stage"] = self.stage_index
            info["controller_order"] = self._order
            info["movement_lesson"] = self._movement_lesson
        return self._joint_observation(), float(reward), terminated, truncated, info


def _policy_kwargs(local_width: int) -> dict[str, Any]:
    from .networks import TeamSetFeaturesExtractor
    return {
        "features_extractor_class": TeamSetFeaturesExtractor,
        "features_extractor_kwargs": {
            "local_width": local_width,
            "ship_features": MOBILE_SHIP_FEATURES,
            "asteroid_slots": MAX_ASTEROIDS,
            "asteroid_features": ASTEROID_FEATURES + 32,
            "projectile_slots": 16,
            "projectile_features": PROJECTILE_FEATURES,
            "teammate_slots": 1,
            "teammate_features": TEAMMATE_FEATURES,
            "global_features": global_feature_count(7) + 8,
        },
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }


def evaluate_centralized_team(checkpoint_or_model, *, stage: int, episodes: int = 128,
                              seed: int = TEST_SEED) -> dict:
    _, _, _, MaskablePPO, _, _, _ = require_team_ppo()
    if hasattr(checkpoint_or_model, "predict"):
        model = checkpoint_or_model
    else:
        checkpoint = Path(checkpoint_or_model)
        model = MaskablePPO.load(checkpoint / "model.zip", device="cpu")
    env = CentralizedTeamEnv(seed=seed, stage=stage, force_stage=stage)
    records = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action, _ = model.predict(
                observation, deterministic=True, action_masks=env.action_masks())
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        records.append(info["episode_metrics"])
    numeric = ("alive_ship_time_fraction", "final_alive_fraction", "asteroids_destroyed",
               "shots_fired", "shots_missed", "ship_collisions", "friendly_fire",
               "reward")
    return {
        "stage": stage, "name": env.stages[stage].name, "episodes": episodes,
        "success_rate": sum(bool(row["is_success"]) for row in records) / episodes,
        **{f"mean_{key}": sum(float(row.get(key, 0.0)) for row in records) / episodes
           for key in numeric},
    }


def _checkpoint_metadata(model, stage: int, settings: TeamPPOSettings,
                         started: float, seed: int) -> dict:
    return {
        "algorithm": "centralized_team_ppo", "curriculum_version": TEAM_CURRICULUM_VERSION,
        "environment_steps": int(model.num_timesteps), "stage": int(stage), "ships": 2,
        "seed": int(seed), "settings": asdict(settings),
        "wall_seconds": time.monotonic() - started,
    }


def _save_team_checkpoint(model, destination: Path, stage: int,
                          settings: TeamPPOSettings, started: float, seed: int) -> Path:
    checkpoint = destination / f"checkpoint_{int(model.num_timesteps):09d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint / "model.zip")
    metadata = _checkpoint_metadata(model, stage, settings, started, seed)
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return checkpoint


def _set_team_learning_rate(model, learning_rate: float) -> None:
    """Set both SB3's schedule and the live optimizer after a rejected candidate."""
    rate = float(learning_rate)
    model.learning_rate = rate
    model.lr_schedule = lambda _: rate
    for group in model.policy.optimizer.param_groups:
        group["lr"] = rate


def _next_team_best_score(previous: float, candidate: float, *, accepted: bool,
                          promoted: bool, mastered: bool) -> float:
    """Return the score represented by the champion that remains on disk."""
    if promoted:
        return float(candidate) if mastered else 0.0
    if accepted:
        return max(float(previous), float(candidate))
    return float(previous)


def _team_rejection_outcome(consecutive: int, patience: int) -> tuple[str, int]:
    """Let a challenger cross temporary regressions before restoring the champion."""
    count = int(consecutive) + 1
    if count >= max(1, int(patience)):
        return "rolled_back", 0
    return "continued", count


def _restore_team_state(resume: Path, metadata: dict[str, Any],
                        state: dict[str, Any]) -> None:
    """Restore curriculum state, including legacy champions that only saved metadata."""
    candidate = resume / "team_curriculum_state.json"
    if not candidate.is_file():
        candidate = resume.parent / "team_curriculum_state.json"
    if candidate.is_file():
        state.update(json.loads(candidate.read_text(encoding="utf-8")))
        return

    # v4 champions predate the run-level state file.  Their metadata identifies the
    # evaluated checkpoint, while the matching evaluation record tells us whether that
    # checkpoint had just promoted and preserves its held-out score.  Falling back to stage
    # zero here silently turned a round-29 rescue into a fresh warm-up run.
    state["stage"] = int(metadata.get("stage", 0))
    evaluation_log = resume.parent / "evaluation.jsonl"
    checkpoint_steps = int(metadata.get("environment_steps", -1))
    if not evaluation_log.is_file() or checkpoint_steps < 0:
        return
    for line in evaluation_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("environment_steps", -2)) != checkpoint_steps:
            continue
        state["stage"] = int(record.get("next_training_stage", state["stage"]))
        state["best_success_rate"] = (
            0.0 if record.get("promoted")
            else float(record.get("current", {}).get("success_rate", 0.0)))
        state["last_evaluated_stage"] = int(
            record.get("training_stage", metadata.get("stage", state["stage"])))
        state["last_success_rate"] = float(
            record.get("current", {}).get("success_rate", 0.0))
        state["retention_ok"] = bool(record.get("retention_ok", True))


def train_centralized_team(output_dir: str | Path, *, steps: int, seed: int = 0,
                           parallel_envs: int = 8, eval_every: int = 250_000,
                           eval_episodes: int = 128, device: str = "cpu",
                           resume: str | Path | None = None,
                           settings: TeamPPOSettings | None = None,
                           stop_when_mastered: bool = False,
                           keep_checkpoints: int = 3) -> Path:
    torch, _, _, MaskablePPO, BaseCallback, DummyVecEnv, SubprocVecEnv = require_team_ppo()
    del torch
    settings = settings or TeamPPOSettings()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "team_curriculum_state.json"
    state = {"stage": 0, "best_success_rate": 0.0, "mastered": False,
             "rejected_candidates": 0, "accepted_candidates": 0,
             "consecutive_rejections": 0}
    if resume:
        resume = Path(resume)
        metadata = json.loads((resume / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") != "centralized_team_ppo":
            raise ValueError("legacy MAPPO/snapshot checkpoints cannot resume joint team PPO")
        # v5 changes rewards and round-29 sampling but retains v4 observations/actions, so a
        # pre-cliff v4 checkpoint is the intended initialization for the movement rescue.
        if int(metadata.get("curriculum_version", -1)) not in {
                4, 5, 6, 7, TEAM_CURRICULUM_VERSION}:
            raise ValueError("team checkpoint curriculum version does not match")
        _restore_team_state(resume, metadata, state)
        # A checkpoint from another run is an initialization, not a continuation. Preserve
        # its learned stage, but do not inherit a decayed learning rate, rejection counters,
        # or an old-stage score whose meaning changed with this curriculum.
        if destination.resolve() != resume.parent.resolve():
            restored_stage = int(state["stage"])
            state = {"stage": restored_stage, "best_success_rate": 0.0,
                     "mastered": False, "rejected_candidates": 0,
                     "accepted_candidates": 0, "consecutive_rejections": 0}
    stage = int(state["stage"])
    n_envs = max(1, int(parallel_envs))
    training_log = destination / "training.jsonl"
    episode_offset = 0
    if resume:
        source_log = training_log if training_log.is_file() else Path(resume).parent / "training.jsonl"
        if source_log.is_file():
            episode_offset = sum(1 for line in source_log.read_text(
                encoding="utf-8").splitlines() if line.strip())
    factories = [lambda rank=rank: CentralizedTeamEnv(
        rank=rank, num_envs=n_envs, seed=seed, stage=stage,
        episode_offset=episode_offset) for rank in range(n_envs)]
    vec_env = (DummyVecEnv(factories) if n_envs == 1 else
               SubprocVecEnv(factories, start_method="spawn"))
    prototype = CentralizedTeamEnv(seed=seed, stage=stage, force_stage=stage)
    if resume:
        model = MaskablePPO.load(Path(resume) / "model.zip", env=vec_env, device=device)
        # SB3 restores optimization hyperparameters from the source archive. Explicitly use
        # the rescue run's exploration coefficient when migrating a compatible v4 policy.
        model.ent_coef = settings.ent_coef
        if destination.resolve() != Path(resume).parent.resolve():
            model.policy.optimizer.state.clear()
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env, learning_rate=settings.learning_rate,
            n_steps=settings.n_steps, batch_size=settings.batch_size,
            n_epochs=settings.n_epochs, gamma=settings.gamma,
            gae_lambda=settings.gae_lambda, clip_range=settings.clip_range,
            target_kl=settings.target_kl,
            ent_coef=settings.ent_coef, vf_coef=settings.vf_coef,
            policy_kwargs=_policy_kwargs(prototype.local_width), seed=seed,
            device=device, verbose=0)
    started = time.monotonic()
    evaluation_log = destination / "evaluation.jsonl"
    effective_lr = float(state.get("effective_learning_rate", settings.learning_rate))
    _set_team_learning_rate(model, effective_lr)
    champion = destination / "champion"
    fresh_destination = not (champion / "model.zip").is_file()
    if fresh_destination:
        if resume:
            baseline = evaluate_centralized_team(
                model, stage=stage, episodes=eval_episodes, seed=EVAL_SEED)
            state.update({"best_success_rate": baseline["success_rate"],
                          "last_evaluated_stage": stage,
                          "last_success_rate": baseline["success_rate"],
                          "retention_ok": True,
                          "effective_learning_rate": effective_lr,
                          "last_candidate_action": "bootstrap"})
            print(f"team bootstrap: {baseline['name']} success "
                  f"{baseline['success_rate']:.1%}", flush=True)
        champion.mkdir(exist_ok=True)
        model.save(champion / "model.zip")
        (champion / "metadata.json").write_text(
            json.dumps(_checkpoint_metadata(model, stage, settings, started, seed),
                       indent=2) + "\n", encoding="utf-8")
        if not resume:
            state["best_success_rate"] = 0.0
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    class EpisodeLogger(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                metrics = info.get("episode_metrics")
                if metrics:
                    record = {"environment_steps": int(self.num_timesteps),
                              "curriculum_stage": info.get("curriculum_stage"),
                              "movement_lesson": info.get("movement_lesson", False),
                              **metrics}
                    with training_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
            return True

    target_steps = int(model.num_timesteps) + max(0, int(steps))
    checkpoint = destination
    while model.num_timesteps < target_steps and not state.get("mastered"):
        chunk = min(max(1, int(eval_every)), target_steps - int(model.num_timesteps))
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    callback=EpisodeLogger(), progress_bar=False)
        current = evaluate_centralized_team(
            model, stage=stage, episodes=eval_episodes, seed=EVAL_SEED)
        retained = []
        if stage:
            for prior in range(max(0, stage - 2), stage):
                retained.append(evaluate_centralized_team(
                    model, stage=prior, episodes=16, seed=EVAL_SEED + 512 + prior * 16))
        retention_ok = (not retained or (
            sum(item["success_rate"] * item["episodes"] for item in retained)
            / sum(item["episodes"] for item in retained) >= 0.70
            and min(item["success_rate"] for item in retained) >= 0.50))
        target = float(team_curriculum()[stage].promotion_clear_rate)
        promoted = current["success_rate"] >= target and retention_ok
        evaluated_stage = stage
        previous_best = float(state.get("best_success_rate", 0.0))
        accepted = bool(retention_ok and current["success_rate"] > previous_best)
        if promoted:
            if stage < len(team_curriculum()) - 1:
                stage += 1
                vec_env.env_method("set_stage", stage)
            else:
                state["mastered"] = True
        state.update({"stage": stage, "last_evaluated_stage": evaluated_stage,
                      "last_success_rate": current["success_rate"],
                      # This must describe the champion actually left on disk. A rejected
                      # candidate can outscore it while failing retention; banking that
                      # unrepresented score makes all later candidates impossible to accept.
                      "best_success_rate": _next_team_best_score(
                          previous_best, current["success_rate"], accepted=accepted,
                          promoted=promoted, mastered=bool(state.get("mastered"))),
                      "retention_ok": retention_ok})
        action = "promoted" if promoted else "accepted" if accepted else "continued"
        if accepted or promoted:
            state["consecutive_rejections"] = 0
            state["accepted_candidates"] = int(state.get("accepted_candidates", 0)) + 1
            champion.mkdir(exist_ok=True)
            model.save(champion / "model.zip")
            (champion / "metadata.json").write_text(
                json.dumps(_checkpoint_metadata(
                    model, evaluated_stage, settings, started, seed), indent=2) + "\n",
                encoding="utf-8")
        else:
            _set_team_learning_rate(model, effective_lr)
            state["rejected_candidates"] = int(state.get("rejected_candidates", 0)) + 1
            action, consecutive = _team_rejection_outcome(
                int(state.get("consecutive_rejections", 0)),
                settings.challenger_patience,
            )
            state["consecutive_rejections"] = consecutive
            if action == "rolled_back":
                # PPO can need to cross a temporary held-out regression to improve a harder
                # stage. Preserve those intermediate policies for several evaluations, but
                # restore the durable champion after a sustained failure and make the next
                # attempt more conservative.
                model.set_parameters(champion / "model.zip", exact_match=True, device=device)
                effective_lr = max(1e-5, effective_lr * 0.80)
                _set_team_learning_rate(model, effective_lr)
        state["effective_learning_rate"] = effective_lr
        state["last_candidate_action"] = action
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        checkpoint = _save_team_checkpoint(model, destination, stage, settings, started, seed)
        (checkpoint / "team_curriculum_state.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8")
        removed = prune_checkpoints(destination, keep=max(0, int(keep_checkpoints)))
        if removed:
            print(f"  checkpoint retention: removed {len(removed)} old checkpoint(s)",
                  flush=True)
        record = {"environment_steps": int(model.num_timesteps),
                  "training_stage": evaluated_stage, "next_training_stage": stage,
                  "target": target, "promoted": promoted, "retention_ok": retention_ok,
                  "candidate_action": action, "effective_learning_rate": effective_lr,
                  "best_success_rate": state["best_success_rate"],
                  "current": current, "retention": retained}
        with evaluation_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"team eval @ {model.num_timesteps}: {current['name']} success "
              f"{current['success_rate']:.1%} of {target:.0%}" +
              (f" - PROMOTED to {team_curriculum()[stage].name}" if promoted and not state.get("mastered")
               else " - MASTERED" if state.get("mastered")
               else f" - {action.upper()} (best {state['best_success_rate']:.1%}, "
                    f"lr {effective_lr:.2g})"), flush=True)
        if stop_when_mastered and state.get("mastered"):
            break
    vec_env.close()
    return checkpoint


def play_centralized_team(checkpoint: str | Path, *, stage: int | None = None,
                          seed: int = 7) -> int:
    import pygame
    from ..renderer import Renderer
    _, _, _, MaskablePPO, _, _, _ = require_team_ppo()
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    stage = int(metadata.get("stage", 0) if stage is None else stage)
    model = MaskablePPO.load(checkpoint / "model.zip", device="cpu")
    env = CentralizedTeamEnv(seed=seed, stage=stage, force_stage=stage)
    observation, _ = env.reset(seed=seed)
    pygame.init()
    renderer = Renderer(pygame, env.native.config.arena.width, env.native.config.arena.height)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                              and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        if not (env.native.state.terminated or env.native.state.truncated):
            action, _ = model.predict(
                observation, deterministic=True, action_masks=env.action_masks())
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + 1)
                seed += 1
        renderer.draw(env.native.state)
        clock.tick(15)
    pygame.quit()
    return 0
