"""How hard is `orbit` as a function of its rate and reach, measured rather than reasoned.

Slowing it from 2 turns per period to 1 was expected to help, on the theory that its
tangential speed was the problem. It measured 8 points worse. This scores the same checkpoint
on the same held-out seeds across the grid, so the setting is chosen from data.

Usage: python scripts/orbit_sweep.py CHECKPOINT [SEEDS] [WORKERS]
"""
import json, math, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src")

CKPT = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
GRID = [(2.0, 2.2), (2.0, 1.5), (1.5, 2.2), (1.0, 2.2), (1.0, 1.5), (3.0, 2.2)]


def run(job):
    rate, reach, seed = job
    import asteroid_survival.patterns as patterns
    from asteroid_survival import config as config_module
    patterns.ORBIT_RATE = rate
    config_module.PATTERN_REACH["orbit"] = reach
    import asteroid_survival.simulation as simulation
    simulation.PATTERN_REACH["orbit"] = reach
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.evaluation import evaluate_policy
    from asteroid_survival.rl.ppo import _stage_env, PPOController

    meta = json.load(open(os.path.join(CKPT, "metadata.json")))
    layout = dict(meta.get("observation_layout") or {})
    for key, default in (("history_frames", 8), ("history_long_frames", 8),
                         ("history_long_stride", 8), ("max_projectiles", 8), ("version", 9)):
        layout.setdefault(key, default)
    spec = load_curriculum("configs/rl-survival-v3.toml")
    env = _stage_env(spec, 28, layout)
    return (rate, reach), evaluate_policy(env, PPOController(CKPT), [seed])["episodes"][0]


if __name__ == "__main__":
    jobs = [(rate, reach, 10_000 + i) for rate, reach in GRID for i in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        results = list(pool.map(run, jobs))
    print(f"round 29, {SEEDS} held-out seeds each. Round 28 reference is 0.734,\n"
          f"and the same round with orbit removed is 0.719.\n")
    print(f"{'rate':>5} {'reach':>6} {'clear':>7} {'survived':>10} {'tangential':>12}")
    spec_rock = None
    for key in GRID:
        rows = [row for name, row in results if name == key]
        rate, reach = key
        import asteroid_survival.patterns as patterns  # noqa: F401
        from asteroid_survival.rl.curriculum import load_curriculum
        rock = load_curriculum("configs/rl-survival-v3.toml").stages[28].game_config(
            load_curriculum("configs/rl-survival-v3.toml").base).asteroid
        amplitude = min((rock.amplitude_min + rock.amplitude_max) / 2 * reach, 270)
        period = (rock.wavelength_min + rock.wavelength_max) / 2
        tangential = amplitude * rate * (2 * math.pi / period)
        print(f"{rate:>5.1f} {reach:>6.1f} "
              f"{statistics.fmean(r['completed_stage'] for r in rows):>7.3f} "
              f"{statistics.fmean(r['survival_time'] for r in rows):>9.1f}s "
              f"{tangential:>10.0f}px/s")
