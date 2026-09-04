"""Why did round 29 collapse? Two things change there at once, so separate them.

Round 28 is 50% large rocks; round 29 is 100% large. That composition step is a known cliff.
Round 29 is also where `orbit` joins the pattern pool. This scores one checkpoint on the same
held-out seeds with each change applied alone, so the drop can be attributed.

Usage: python scripts/round29_ablation.py CHECKPOINT [SEEDS] [WORKERS]
"""
import json, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.evaluation import evaluate_policy
from asteroid_survival.rl.ppo import _stage_env, PPOController

CKPT = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
CURRICULUM = "configs/rl-survival-v3.toml"
_META = json.load(open(os.path.join(CKPT, "metadata.json")))
LAYOUT = dict(_META.get("observation_layout") or {})
for key, default in (("history_frames", 8), ("history_long_frames", 8),
                     ("history_long_stride", 8), ("max_projectiles", 8), ("version", 9)):
    LAYOUT.setdefault(key, default)

ELEVEN = ["sine", "arc", "s_curve", "zigzag", "sawtooth", "lane_change",
          "corkscrew", "figure_eight", "spiral", "serpentine", "brownian"]

VARIANTS = {
    "round 28 (reference)":        dict(stage=27),
    "round 29 as shipped":         dict(stage=28),
    "round 29 without orbit":      dict(stage=28, pool=ELEVEN),
    "round 29, round-28 rock mix": dict(stage=28, sizes=[2, 2, 3, 3]),
    "round 29, neither change":    dict(stage=28, pool=ELEVEN, sizes=[2, 2, 3, 3]),
}


def run(job):
    label, seed = job
    spec = load_curriculum(CURRICULUM)
    plan = VARIANTS[label]
    env = _stage_env(spec, plan["stage"], LAYOUT)
    rock = env.config.asteroid
    if "pool" in plan:
        rock.pattern_pool = list(plan["pool"])
        rock.linear_probability = 1 / 12
    if "sizes" in plan:
        rock.spawn_size = list(plan["sizes"])
    env.config.validate()
    controller = PPOController(CKPT)
    return label, evaluate_policy(env, controller, [seed])["episodes"][0]


if __name__ == "__main__":
    jobs = [(label, 10_000 + i) for label in VARIANTS for i in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        results = list(pool.map(run, jobs))
    print(f"{CKPT}, {SEEDS} held-out seeds each\n")
    print(f"{'variant':<30} {'clear':>7} {'survived':>10} {'destroyed':>10} {'accuracy':>9}")
    for label in VARIANTS:
        rows = [row for name, row in results if name == label]
        print(f"{label:<30} {statistics.fmean(r['completed_stage'] for r in rows):>7.3f}"
              f" {statistics.fmean(r['survival_time'] for r in rows):>9.1f}s"
              f" {statistics.fmean(r['asteroids_destroyed'] for r in rows):>10.1f}"
              f" {statistics.fmean(r['accuracy'] for r in rows):>9.3f}")
