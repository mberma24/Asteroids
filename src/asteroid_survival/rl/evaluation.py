from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

import numpy as np

from .environment import AsteroidsRLEnv


Policy = Callable[[np.ndarray], int]


def evaluate_policy(env: AsteroidsRLEnv, policy: Policy, seeds: list[int]) -> dict:
    episodes = []
    for seed in seeds:
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        observation, _ = env.reset(seed)
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(policy(observation))
            done = terminated or truncated
        episodes.append(info["episode_metrics"])
    numeric = ("survival_time", "wave", "reward", "asteroid_reward", "shots_fired",
               "asteroids_destroyed", "accuracy",
               "resolved_accuracy", "waves_cleared", "mean_wave_clear_time",
               "large_destroyed", "medium_destroyed", "small_destroyed")
    clear_rate = sum(bool(e.get("completed_stage")) for e in episodes) / len(episodes)
    limit = max(1e-9, env.max_decisions * env.frame_skip / env.config.arena.fps)
    survival_fraction = sum(min(1.0, float(e["survival_time"]) / limit)
                            for e in episodes) / len(episodes)
    aggregate = {
        "episodes": len(episodes),
        "survival_rate_to_limit": sum(e["survived_to_limit"] for e in episodes) / len(episodes),
        "clear_rate": clear_rate,
        "survival_fraction": survival_fraction,
        "completion_rate": (survival_fraction if env.completion == "survival" else clear_rate),
    }
    for name in numeric:
        values = [float(episode[name]) for episode in episodes]
        aggregate[f"mean_{name}"] = statistics.fmean(values)
        aggregate[f"median_{name}"] = statistics.median(values)
        aggregate[f"min_{name}"] = min(values)
        aggregate[f"max_{name}"] = max(values)
    return {"aggregate": aggregate, "episodes": episodes}


def save_evaluation(report: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
