"""Would declining the shots the predictor calls fatal actually save the policy?

Zero training. The champion plays each seed twice -- once normally, once with every firing
action the v11 warning block flags replaced by the same action minus the shot -- and the two
are compared on identical seeds. Nothing is learned, so there is no risk of teaching the
policy not to fire; this only asks whether those shots are what kills it.

The question is live because the information alone has not helped. Observations v8 and v11
both handed the policy this prediction and both measured null, and when v13 got the v8 block
the share of its fatal shots that were flagged fell from 93% to 62% -- it stopped taking the
close fatal shots and started dying to distant ones the predictor cannot see (it catches
every hit inside 0.4s, 36% at 0.6-1.0s, 1.4% beyond). If deaths simply move, masking will not
move the clear rate, and this is an hour spent instead of a training run.

The flag is the audit's, so the numbers line up with `scripts/fragment_warning_audit.py`:
the shot hits something, that something splits, and the worst fragment clearance is a
collision (`worst_clearance <= 0`, which the observation carries as clearance/150).

Usage: PYTHONPATH=src python scripts/masked_shot_diagnostic.py CHECKPOINT OUTPUT
           [--names retention31] [--seed 1000004000] [--count 64] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asteroid_survival.actions import Action
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.environment import FIRING_ACTIONS
from asteroid_survival.rl.ppo import PPOController, _stage_env

SNAPSHOTS = Path("/home/ubuntu/Asteroids/experiments/v18/snapshots")
WARNING_FEATURES = 4
_WORKER: dict = {}

ACTIONS = list(Action)
# Every firing action is some movement plus the shot; the counterpart is that movement alone,
# so masking changes what the ship fires and nothing about where it goes.
UNFIRED = {}
for _index, _action in enumerate(ACTIONS):
    if not _action.fire:
        continue
    _match = next(i for i, other in enumerate(ACTIONS)
                  if not other.fire and other.turn == _action.turn
                  and other.thrust == _action.thrust)
    UNFIRED[_index] = _match
assert len(UNFIRED) == len(FIRING_ACTIONS), "every firing action needs a non-firing twin"
FIRING_INDEX = {ACTIONS.index(action): slot for slot, action in enumerate(FIRING_ACTIONS)}


def flagged(observation: np.ndarray, action: int) -> bool:
    """Does the v11 block call this firing action's shot fatal?"""
    slot = FIRING_INDEX.get(action)
    if slot is None:
        return False
    block = observation[-WARNING_FEATURES * len(FIRING_ACTIONS):]
    hit, splits, clearance, _ = block[slot * WARNING_FEATURES:(slot + 1) * WARNING_FEATURES]
    return bool(hit >= 0.5 and splits >= 0.5 and clearance <= 0.0)


def _init(checkpoint: str, snapshots: str) -> None:
    import torch
    torch.set_num_threads(1)
    controller = PPOController(checkpoint, device="cpu")
    layout = controller.metadata["observation_layout"]
    if int(layout.get("version", 0)) < 11:
        raise SystemExit("the masked diagnostic needs an observation v11 checkpoint")
    _WORKER.update(controller=controller, layout=layout,
                   specs={name: load_curriculum(Path(snapshots) / name / "rl-survival-v3.toml")
                          for name in ("current", "full")})


def _episode(job: tuple[str, int, bool]) -> dict:
    name, seed, mask = job
    spec = _WORKER["specs"]["full" if name == "full" else "current"]
    index = int(name.removeprefix("retention")) - 1 if name.startswith("retention") else 28
    env = _stage_env(spec, index, _WORKER["layout"])
    controller = _WORKER["controller"]
    controller.reset()
    observation, _ = env.reset(seed)
    masked = warned = 0
    done = False
    while not done:
        action = controller(observation)
        if flagged(observation, action):
            warned += 1
            if mask:
                action = UNFIRED[action]
                masked += 1
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    row = dict(info["episode_metrics"])
    row.update(seed=seed, masked_shots=masked, flagged_choices=warned, benchmark=name)
    return row


def aggregate(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "clear_rate": float(np.mean([r["completed_stage"] for r in rows])),
        "completion_rate": float(np.mean([min(1., r["survival_time"] / 30.) for r in rows])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "mean_shots_fired": float(np.mean([r["shots_fired"] for r in rows])),
        "mean_asteroids_destroyed": float(np.mean([r["asteroids_destroyed"] for r in rows])),
        "mean_flagged_choices": float(np.mean([r["flagged_choices"] for r in rows])),
        "mean_masked_shots": float(np.mean([r["masked_shots"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--names", default="retention31")
    parser.add_argument("--seed", type=int, default=1_000_004_000)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--snapshots", default=str(SNAPSHOTS))
    args = parser.parse_args()

    started = time.time()
    names = args.names.split(",")
    seeds = range(args.seed, args.seed + args.count)
    jobs = [(name, seed, mask) for name in names for seed in seeds for mask in (False, True)]
    results: dict[tuple[str, bool], list] = {(n, m): [] for n in names for m in (False, True)}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.checkpoint, args.snapshots)) as pool:
        for done, (job, row) in enumerate(zip(jobs, pool.map(_episode, jobs)), start=1):
            results[(job[0], job[2])].append(row)
            if done % 32 == 0:
                print(f"  {done}/{len(jobs)} episodes", flush=True)

    rng = np.random.default_rng(1847)
    report = {"checkpoint": args.checkpoint, "seconds": time.time() - started,
              "benchmarks": {}}
    for name in names:
        normal, masked = results[(name, False)], results[(name, True)]
        by_seed = {r["seed"]: r for r in normal}
        difference = np.array([int(r["completed_stage"])
                               - int(by_seed[r["seed"]]["completed_stage"]) for r in masked],
                              dtype=float)
        draws = difference[rng.integers(0, len(difference),
                                        (10_000, len(difference)))].mean(axis=1)
        low, high = np.quantile(draws, [.025, .975])
        report["benchmarks"][name] = {
            "normal": aggregate(normal), "masked": aggregate(masked),
            "paired_difference": {"clear": float(difference.mean()),
                                  "ci95": [float(low), float(high)],
                                  "saved": int((difference > 0).sum()),
                                  "cost": int((difference < 0).sum())},
            "episodes": {"normal": normal, "masked": masked},
        }
        summary = report["benchmarks"][name]
        print(f"\n{name}: clear {summary['normal']['clear_rate']:.3f} -> "
              f"{summary['masked']['clear_rate']:.3f}  "
              f"({summary['paired_difference']['clear']:+.3f} "
              f"[{low:+.3f}, {high:+.3f}], saved {summary['paired_difference']['saved']}, "
              f"cost {summary['paired_difference']['cost']})", flush=True)
        print(f"  flagged choices/episode {summary['normal']['mean_flagged_choices']:.1f}, "
              f"masked {summary['masked']['mean_masked_shots']:.1f}; shots "
              f"{summary['normal']['mean_shots_fired']:.1f} -> "
              f"{summary['masked']['mean_shots_fired']:.1f}; kills "
              f"{summary['normal']['mean_asteroids_destroyed']:.1f} -> "
              f"{summary['masked']['mean_asteroids_destroyed']:.1f}", flush=True)
    Path(args.output).write_text(json.dumps(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
