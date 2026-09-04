"""How rare does `orbit` have to be for round 29 to be passable while still being taught?

Removing a pattern outright means the policy never learns it. This project already has a rule
about that from the co-op work: soften how much of a challenge there is, never whether the
skill is needed at all. A rung that can be cleared by not having the skill will teach the
policy not to have it.

So the question is not "in or out" but "at what share". The pool is drawn uniformly, so
repeating the other patterns thins orbit without removing it.

Usage: python scripts/orbit_share.py CHECKPOINT [SEEDS] [WORKERS]
"""
import json, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.evaluation import evaluate_policy
from asteroid_survival.rl.ppo import _stage_env, PPOController

CKPT = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ELEVEN = ["sine", "arc", "s_curve", "zigzag", "sawtooth", "lane_change",
          "corkscrew", "figure_eight", "spiral", "serpentine", "brownian"]
REPEATS = [0, 1, 2, 4, 8]          # 0 means orbit is absent entirely
_META = json.load(open(os.path.join(CKPT, "metadata.json")))
LAYOUT = dict(_META.get("observation_layout") or {})
for key, default in (("history_frames", 8), ("history_long_frames", 8),
                     ("history_long_stride", 8), ("max_projectiles", 8), ("version", 10)):
    LAYOUT.setdefault(key, default)


def pool_for(repeat):
    return ELEVEN * max(1, repeat) + (["orbit"] if repeat else [])


def run(job):
    repeat, seed = job
    spec = load_curriculum("configs/rl-survival-v3.toml")
    env = _stage_env(spec, 28, LAYOUT)
    pool = pool_for(repeat)
    env.config.asteroid.pattern_pool = list(pool)
    env.config.asteroid.linear_probability = 1 / (len(set(pool)) + 1)
    env.config.validate()
    episode = evaluate_policy(env, PPOController(CKPT), [seed])["episodes"][0]
    orbits = sum(1 for _ in ()) if repeat == 0 else None
    return repeat, episode


if __name__ == "__main__":
    jobs = [(r, 10_000 + i) for r in REPEATS for i in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        results = list(pool.map(run, jobs))
    print(f"round 29, {SEEDS} held-out seeds each. Round 28 is 0.734 for reference.\n")
    print(f"{'orbit share':>12} {'clear':>8} {'survived':>10} {'orbit rocks/episode':>21}")
    for repeat in REPEATS:
        rows = [row for r, row in results if r == repeat]
        share = 0.0 if repeat == 0 else 1 / (11 * repeat + 1)
        # ~20 rocks are on the field over a round; the share times that is what the policy meets.
        print(f"{share:>11.1%} {statistics.fmean(r['completed_stage'] for r in rows):>8.3f}"
              f" {statistics.fmean(r['survival_time'] for r in rows):>9.1f}s"
              f" {share * statistics.fmean(r['asteroids_destroyed'] for r in rows):>20.1f}")
