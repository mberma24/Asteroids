#!/usr/bin/env python3
"""Evaluate a team checkpoint on one temporary medium/large composition."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sb3_contrib import MaskablePPO

import asteroid_survival.rl.team_ppo as team_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-stage", type=int, default=34)
    parser.add_argument("--large", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.large <= args.total:
        parser.error("--large must be between zero and --total")

    stages = team_ppo.team_curriculum()
    sizes = [2] * (args.total - args.large) + [3] * args.large
    calibrated = replace(
        stages[args.base_stage],
        name=f"calibration-{args.large}-of-{args.total}-large",
        asteroid_size=sizes,
    )
    team_ppo.team_curriculum = lambda: (
        stages[:args.base_stage] + (calibrated,) + stages[args.base_stage + 1:]
    )
    model = MaskablePPO.load(args.checkpoint / "model.zip", device="cpu")
    result = team_ppo.evaluate_centralized_team(
        model,
        stage=args.base_stage,
        episodes=args.episodes,
        seed=team_ppo.EVAL_SEED,
    )
    print(
        f"{args.large}/{args.total} large: "
        f"success={result['success_rate']:.1%} "
        f"alive={result['mean_alive_ship_time_fraction']:.3f} "
        f"kills={result['mean_asteroids_destroyed']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
