"""Centralized two-ship PPO with simultaneous joint actions and team-only success."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from .curriculum import CurriculumStage, load_curriculum
from .environment import (ASTEROID_FEATURES, MOBILE_ACTIONS, MOBILE_SHIP_FEATURES,
                          PROJECTILE_FEATURES, TEAMMATE_FEATURES, RewardConfig,
                          global_feature_count)
from .multiagent import MultiAgentAsteroidsEnv
from .ppo_support import training_seed


TEAM_CURRICULUM_VERSION = 3
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
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 1024
    batch_size: int = 1024
    n_epochs: int = 4


TEAM_REWARD = RewardConfig(
    large=0.25, medium=0.25, small=0.25,
    survival_bonus=0.02, round_clear=30.0,
    death_penalty=20.0, timeout_penalty=10.0,
    # Finite wave warmups make shooting mandatory, so this can teach deliberate fire without
    # recreating the old "never shoot" optimum that an empty survival warmup produced.
    miss_penalty=0.10, collision_penalty=5.0,
    friendly_fire_penalty=5.0, friendly_fire_dealt_penalty=5.0,
    time_scaled_survival=True, active_time_penalty=0.0,
)


def team_curriculum() -> tuple[CurriculumStage, ...]:
    """Three forced-shooting lessons, three survival bridges, then the real ladder."""
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
                promotion_clear_rate=0.85, promotion_accuracy=0.0)
        for index, (count, interval) in enumerate(((2, 6.0), (4, 3.5), (6, 2.75)), 1)
    )
    ladder = tuple(
        replace(stage, name=f"team-{stage.name}", ships=2,
                promotion_clear_rate=0.75, promotion_accuracy=0.0)
        for stage in solo.stages)
    return waves + bridges + ladder


def team_stage_config(stage: CurriculumStage):
    solo = load_curriculum("configs/rl-survival-v2.toml")
    config = stage.game_config(solo.base)
    # The legacy 80px radius starts the ships only 160px apart; measured friendly-fire kills
    # land around 108px, so the first shot can decide the episode before either policy has
    # acted meaningfully. A 250px radius leaves room for actual deconfliction while keeping
    # the same 900px toroidal arena and full collision rules.
    config.ship.spawn_radius = 250.0
    # Do not ask an untrained policy to solve aiming, collision avoidance, and long-lived
    # friendly fire in the same first episode. Every wave still requires shooting; hazards
    # are added one at a time, then remain fully enabled throughout survival training.
    if stage.name == "team-wave-1":
        config.ship.friendly_collisions = "off"
    elif stage.name == "team-wave-2":
        config.ship.friendly_collisions = "ships"
    elif stage.name == "team-wave-3":
        config.projectile.lifetime = 0.5
    elif stage.name == "team-wave-4":
        config.projectile.lifetime = 0.75
    elif stage.name == "team-survival-warmup-1":
        config.projectile.lifetime = 1.0
    elif stage.name == "team-survival-warmup-2":
        config.projectile.lifetime = 1.2
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
                 episode_offset: int = 0):
        super().__init__()
        self.rank = int(rank)
        self.num_envs = max(1, int(num_envs))
        self.base_seed = int(seed)
        self.episode_offset = max(0, int(episode_offset))
        self.current_stage = int(stage)
        self.force_stage = force_stage
        self.local_episode = 0
        self.rng = random.Random(seed + rank * 100_003)
        self.stages = team_curriculum()
        self.stage_index = int(force_stage if force_stage is not None else stage)
        self.native = self._make_native(self.stage_index)
        self.local_width = self.native.observation_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.local_width,), dtype=np.float32)
        # A single categorical distribution is intentional. MultiDiscrete makes the two
        # actions conditionally independent, so it cannot learn "ship 1 fires OR ship 2
        # fires" without repeatedly sampling the dangerous "both fire" combination.
        # The 16x16 joint action retains every pair while allowing correlated decisions.
        self.action_space = spaces.Discrete(16 * 16)
        self._order = ("ship1", "ship2")
        self._observations: dict[str, np.ndarray] = {}

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
        if draw < 0.80:
            return self.current_stage
        foundations = list(range(min(4, self.current_stage)))
        if draw < 0.90 and foundations:
            return self.rng.choice(foundations)
        return self.rng.randrange(self.current_stage)

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
        self._order = (("ship2", "ship1") if self.rng.random() < 0.5
                       else ("ship1", "ship2"))
        return self._joint_observation(), {
            "curriculum_stage": self.stage_index, "controller_order": self._order}

    def action_masks(self) -> np.ndarray:
        masks = self.native.action_masks()
        first, second = (masks[key] for key in self._order)
        joint = np.logical_and(first[:, None], second[None, :])
        # One active shooter per four-frame decision is ample at 15 decisions/second and
        # makes deconfliction learnable. Friendly fire is still enabled: a badly aimed solo
        # shot can kill the teammate and receives the full terminal penalty.
        firing = np.asarray([action.fire for action in MOBILE_ACTIONS], dtype=bool)
        joint[np.logical_and(firing[:, None], firing[None, :])] = False
        return joint.reshape(-1)

    def step(self, actions):
        joint_action = int(np.asarray(actions).reshape(-1)[0])
        first, second = divmod(joint_action, len(MOBILE_ACTIONS))
        mapped = {self._order[0]: first, self._order[1]: second}
        self._observations, reward, terminated, truncated, info = self.native.step(mapped)
        if terminated or truncated:
            info["curriculum_stage"] = self.stage_index
            info["controller_order"] = self._order
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


def train_centralized_team(output_dir: str | Path, *, steps: int, seed: int = 0,
                           parallel_envs: int = 8, eval_every: int = 250_000,
                           eval_episodes: int = 128, device: str = "cpu",
                           resume: str | Path | None = None,
                           settings: TeamPPOSettings | None = None,
                           stop_when_mastered: bool = False) -> Path:
    torch, _, _, MaskablePPO, BaseCallback, DummyVecEnv, SubprocVecEnv = require_team_ppo()
    del torch
    settings = settings or TeamPPOSettings()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "team_curriculum_state.json"
    state = {"stage": 0, "best_success_rate": 0.0, "mastered": False}
    if resume:
        resume = Path(resume)
        metadata = json.loads((resume / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") != "centralized_team_ppo":
            raise ValueError("legacy MAPPO/snapshot checkpoints cannot resume joint team PPO")
        if int(metadata.get("curriculum_version", -1)) != TEAM_CURRICULUM_VERSION:
            raise ValueError("team checkpoint curriculum version does not match")
        candidate = resume / "team_curriculum_state.json"
        if not candidate.is_file():
            candidate = resume.parent / "team_curriculum_state.json"
        if candidate.is_file():
            state.update(json.loads(candidate.read_text(encoding="utf-8")))
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
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env, learning_rate=settings.learning_rate,
            n_steps=settings.n_steps, batch_size=settings.batch_size,
            n_epochs=settings.n_epochs, gamma=settings.gamma,
            gae_lambda=settings.gae_lambda, clip_range=settings.clip_range,
            ent_coef=settings.ent_coef, vf_coef=settings.vf_coef,
            policy_kwargs=_policy_kwargs(prototype.local_width), seed=seed,
            device=device, verbose=0)
    started = time.monotonic()
    evaluation_log = destination / "evaluation.jsonl"

    class EpisodeLogger(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                metrics = info.get("episode_metrics")
                if metrics:
                    record = {"environment_steps": int(self.num_timesteps),
                              "curriculum_stage": info.get("curriculum_stage"), **metrics}
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
        target = 0.85 if stage < 7 else 0.75
        promoted = current["success_rate"] >= target and retention_ok
        evaluated_stage = stage
        previous_best = float(state.get("best_success_rate", 0.0))
        is_best = current["success_rate"] >= previous_best
        if promoted:
            if stage < len(team_curriculum()) - 1:
                stage += 1
                vec_env.env_method("set_stage", stage)
            else:
                state["mastered"] = True
        state.update({"stage": stage, "last_evaluated_stage": evaluated_stage,
                      "last_success_rate": current["success_rate"],
                      # A promoted stage starts its own best-score history. Keeping the
                      # easier stage's score here would prevent a new-stage champion from
                      # being recorded until it beat an incomparable earlier result.
                      "best_success_rate": (0.0 if promoted and not state.get("mastered")
                                            else max(previous_best,
                                                     current["success_rate"])),
                      "retention_ok": retention_ok})
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        checkpoint = _save_team_checkpoint(model, destination, stage, settings, started, seed)
        (checkpoint / "team_curriculum_state.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8")
        record = {"environment_steps": int(model.num_timesteps),
                  "training_stage": evaluated_stage, "next_training_stage": stage,
                  "target": target, "promoted": promoted, "retention_ok": retention_ok,
                  "current": current, "retention": retained}
        with evaluation_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if is_best or promoted:
            champion = destination / "champion"
            champion.mkdir(exist_ok=True)
            model.save(champion / "model.zip")
            (champion / "metadata.json").write_text(
                json.dumps(_checkpoint_metadata(
                    model, evaluated_stage, settings, started, seed), indent=2) + "\n",
                encoding="utf-8")
        print(f"team eval @ {model.num_timesteps}: {current['name']} success "
              f"{current['success_rate']:.1%} of {target:.0%}" +
              (f" - PROMOTED to {team_curriculum()[stage].name}" if promoted and not state.get("mastered")
               else " - MASTERED" if state.get("mastered") else ""), flush=True)
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
