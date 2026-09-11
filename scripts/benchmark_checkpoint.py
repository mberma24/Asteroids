"""Score a PPO checkpoint on the fixed round-29 benchmarks, with any observation version.

The benchmarks are the ones `orbit_experiment.py evaluate` defined for v18 -- `current` is
round 29 as shipped at 05f4e57 (orbit ramped in at 2%), `full` is round 29 at 5e31a99 (orbit
at its full share), `retentionN` is round N of the current ladder -- so results line up seed
for seed with every earlier report that used them. That evaluator lives in an archived
checkout whose `src` predates observation v11 and cannot load a v21 checkpoint; this one runs
on the live `src` and builds the same environments from the same config snapshots.

Usage: PYTHONPATH=src python scripts/benchmark_checkpoint.py CHECKPOINT OUTPUT
           [--names current,full] [--seed 1000000832] [--count 256] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import PPOController, _stage_env

SNAPSHOTS = Path("/home/ubuntu/Asteroids/experiments/v18/snapshots")
_WORKER: dict = {}


def _init(checkpoint: str, snapshots: str) -> None:
    import torch
    torch.set_num_threads(1)
    controller = PPOController(checkpoint, device="cpu")
    _WORKER.update(controller=controller, layout=controller.metadata["observation_layout"],
                   specs={name: load_curriculum(Path(snapshots) / name / "rl-survival-v3.toml")
                          for name in ("current", "full")})


def _episode(job: tuple[str, int]) -> dict:
    name, seed = job
    spec = _WORKER["specs"]["full" if name == "full" else "current"]
    index = int(name.removeprefix("retention")) - 1 if name.startswith("retention") else 28
    env = _stage_env(spec, index, _WORKER["layout"])
    controller = _WORKER["controller"]
    controller.reset()
    observation, _ = env.reset(seed)
    done = False
    while not done:
        observation, _, terminated, truncated, info = env.step(controller(observation))
        done = terminated or truncated
    row = dict(info["episode_metrics"])
    row["seed"] = seed
    return row


def aggregate(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "clear_rate": float(np.mean([r["completed_stage"] for r in rows])),
        "completion_rate": float(np.mean([min(1., r["survival_time"] / 30.) for r in rows])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "mean_survival_time": float(np.mean([r["survival_time"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--names", default="current,full")
    parser.add_argument("--seed", type=int, default=1_000_000_832)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--snapshots", default=str(SNAPSHOTS),
                        help="directory holding current/ and full/ config snapshots")
    args = parser.parse_args()

    started = time.time()
    names = args.names.split(",")
    jobs = [(name, seed) for name in names
            for seed in range(args.seed, args.seed + args.count)]
    results: dict[str, list] = {name: [] for name in names}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.checkpoint, args.snapshots)) as pool:
        for done, (job, row) in enumerate(zip(jobs, pool.map(_episode, jobs)), start=1):
            results[job[0]].append(row)
            if done % 64 == 0:
                print(f"  {done}/{len(jobs)} episodes", flush=True)
    report = {"checkpoint": args.checkpoint, "seconds": time.time() - started,
              "benchmarks": {name: {"aggregate": aggregate(rows), "episodes": rows}
                             for name, rows in results.items()}}
    Path(args.output).write_text(json.dumps(report) + "\n", encoding="utf-8")
    for name, value in report["benchmarks"].items():
        print(f"{name}: {json.dumps(value['aggregate'])}", flush=True)


if __name__ == "__main__":
    main()
