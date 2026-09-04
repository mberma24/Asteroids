"""Is the policy actually using the newest observation block, or ignoring it?

Weight magnitude is a proxy and a poor one -- influence is weight times input. This
measures influence directly: the same checkpoint over the same seeds, once normally and once
with the trailing block replaced by the value the encoder writes when it has nothing to
report. If the two agree episode for episode, the block is not reaching the policy's
decisions yet, whatever its weights look like.

Usage: python scripts/ablate_block.py CHECKPOINT SEEDS WORKERS CURRICULUM [WIDTH] [STAGE]
"""
import json, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import _stage_env, PPOController

CKPT = sys.argv[1]
CURRICULUM = sys.argv[4]
WIDTH = int(sys.argv[5]) if len(sys.argv) > 5 else 10
STAGE = int(sys.argv[6]) if len(sys.argv) > 6 else 26
_META = json.load(open(os.path.join(CKPT, "metadata.json")))
LAYOUT = dict(_META.get("observation_layout") or {})
for _key, _default in (("history_frames", 8), ("history_long_frames", 8),
                       ("history_long_stride", 8), ("max_projectiles", 8), ("version", 7)):
    LAYOUT.setdefault(_key, _default)
# What `encode_observation` writes for "no shot is possible or nothing would be hit".
BLANK = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)


def run(job):
    seed, ablate = job
    spec = load_curriculum(CURRICULUM)
    env = _stage_env(spec, STAGE, LAYOUT)
    controller = PPOController(CKPT)
    observation, _ = env.reset(seed)
    done = False
    actions = []
    while not done:
        if ablate:
            observation = observation.copy()
            observation[-WIDTH:] = BLANK[:WIDTH]
        action = controller(observation)
        actions.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    metrics = info["episode_metrics"]
    return {"seed": seed, "ablated": ablate, "cleared": bool(metrics["completed_stage"]),
            "survival": float(metrics["survival_time"]), "actions": actions}


if __name__ == "__main__":
    seeds = list(range(10000, 10000 + int(sys.argv[2])))
    jobs = [(s, a) for a in (False, True) for s in seeds]
    with ProcessPoolExecutor(max_workers=int(sys.argv[3])) as pool:
        rows = list(pool.map(run, jobs))
    normal = {r["seed"]: r for r in rows if not r["ablated"]}
    ablated = {r["seed"]: r for r in rows if r["ablated"]}
    same = sum(normal[s]["actions"] == ablated[s]["actions"] for s in seeds)
    diverged = [s for s in seeds if normal[s]["actions"] != ablated[s]["actions"]]
    first = [next(i for i, (a, b) in enumerate(zip(normal[s]["actions"], ablated[s]["actions"]))
                  if a != b) for s in diverged] if diverged else []
    print(f"clear   normal {statistics.fmean(normal[s]['cleared'] for s in seeds):.3f}"
          f"   ablated {statistics.fmean(ablated[s]['cleared'] for s in seeds):.3f}")
    print(f"survival normal {statistics.fmean(normal[s]['survival'] for s in seeds):.2f}s"
          f"  ablated {statistics.fmean(ablated[s]['survival'] for s in seeds):.2f}s")
    print(f"episodes with an identical action sequence: {same}/{len(seeds)}")
    if first:
        print(f"where they differ, first differing decision: median {statistics.median(first):.0f}"
              f" of a 450-decision round")
