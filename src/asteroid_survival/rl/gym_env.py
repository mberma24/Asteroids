"""Gymnasium adapters for the deterministic Asteroids environment.

This module deliberately keeps Gymnasium optional: the game and MuZero trainer can be used
without installing the much larger PyTorch/SB3 dependency group.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise ImportError(
        "PPO dependencies are not installed. Run: .venv/bin/pip install -e '.[ppo]'"
    ) from exc

from .curriculum import CurriculumManager, load_curriculum
from .environment import AsteroidsRLEnv
from .ppo_support import SnapshotPolicy, training_seed


class GymAsteroidsEnv(gym.Env[np.ndarray, int]):
    """A standards-compliant wrapper around one fixed :class:`AsteroidsRLEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, env: AsteroidsRLEnv):
        super().__init__()
        self.env = env
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(env.num_actions)
        self._seed = 0

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._seed = int(seed)
        observation, info = self.env.reset(self._seed)
        self._seed += 1
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        return observation, float(reward), terminated, truncated, info


class CurriculumGymEnv(gym.Env[np.ndarray, int]):
    """One vector-worker that selects a curriculum lesson at every episode reset."""

    metadata = {"render_modes": []}

    def __init__(self, curriculum_path: str | Path, *, rank: int, num_envs: int,
                 seed: int, episode_offset: int = 0, stage: int = 0,
                 history_frames: int = 8, history_long_frames: int = 8,
                 history_long_stride: int = 8, eval_seed: int = 10_000,
                 companion_snapshot: str | Path | None = None):
        super().__init__()
        self.curriculum_path = str(curriculum_path)
        self.spec = load_curriculum(curriculum_path)
        self.rank = int(rank)
        self.num_envs = max(1, int(num_envs))
        self.base_seed = int(seed)
        self.episode_offset = int(episode_offset)
        self.eval_seed = int(eval_seed)
        self.history_frames = int(history_frames)
        self.history_long_frames = int(history_long_frames)
        self.history_long_stride = int(history_long_stride)
        self.manager = CurriculumManager(self.spec, seed + rank * 100_003, stage=stage)
        self.recovery_stage: int | None = None
        self.local_episode = 0
        self.stage_index = stage
        # One shared snapshot per run, reloaded by each worker when the trainer rewrites it.
        self.companion_policy = (
            SnapshotPolicy(companion_snapshot) if companion_snapshot
            and self.spec.max_teammates else None)
        self.env = self._make_env(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.env.num_actions)

    def _make_env(self, stage_index: int) -> AsteroidsRLEnv:
        stage = self.spec.stages[stage_index]
        reward = (self.spec.reward if stage.miss_penalty is None else
                  replace(self.spec.reward, miss_penalty=stage.miss_penalty))
        return AsteroidsRLEnv(
            stage.game_config(self.spec.base), frame_skip=4,
            max_decisions=stage.max_decisions, no_hit_seconds=stage.no_hit_seconds,
            history_frames=self.history_frames,
            history_long_frames=self.history_long_frames,
            history_long_stride=self.history_long_stride, reward_config=reward,
            completion=stage.completion, max_teammates=self.spec.max_teammates,
            companion_policy=self.companion_policy,
            observation_version=self.spec.observation_version)

    def set_curriculum_state(self, stage: int,
                             recovery_stage: int | None = None) -> None:
        self.manager.stage = max(0, min(int(stage), len(self.spec.stages) - 1))
        self.recovery_stage = (None if recovery_stage is None else int(recovery_stage))

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self.base_seed = int(seed)
            self.local_episode = 0
        self.stage_index = self.manager.sample_stage(focus_stage=self.recovery_stage)
        self.env = self._make_env(self.stage_index)
        serial = self.episode_offset + self.rank + self.local_episode * self.num_envs
        reserved = (self.spec.evaluation_panels * self.spec.evaluation_episodes
                    + self.spec.test_evaluation_episodes)
        episode_seed = training_seed(self.base_seed, serial, self.eval_seed, reserved)
        self.local_episode += 1
        observation, info = self.env.reset(episode_seed)
        return observation, {**info, "curriculum_stage": self.stage_index}

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        if terminated or truncated:
            info["curriculum_stage"] = self.stage_index
        return observation, float(reward), terminated, truncated, info
