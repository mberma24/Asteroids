"""How expressed is a round's asteroid motion, and how hard is it to extrapolate?

Two numbers per round, both measured on the simulator rather than derived from the config:

  swing        the mean amplitude a rock is dealt, i.e. how big the shape is
  0.5s error   how far a rock lands from where its current velocity says it will be half a
               second later, which is the horizon the agent actually acts on

A ladder that raises difficulty by making rocks *faster* moves the second number and barely
touches the first. A ladder that raises it by expressing the *pattern* moves both.

Usage: python scripts/pattern_expression.py CURRICULUM [ROUNDS] [SEEDS]
       python scripts/pattern_expression.py configs/rl-survival-v3.toml 28,40,52,64,76,88,96
"""
import statistics
import sys

sys.path.insert(0, "src")
from asteroid_survival.actions import Action
from asteroid_survival.math2d import Vec2, wrapped_delta
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.simulation import Simulation

HORIZON_FRAMES = 30          # 0.5s at 60fps


def measure(spec, round_number: int, seeds: range) -> dict:
    config = spec.stages[round_number - 1].game_config(spec.base)
    width, height = config.arena.width, config.arena.height
    errors, swings = [], []
    for seed in seeds:
        sim = Simulation(config)
        sim.reset(seed)
        tracks: dict[int, list] = {}
        while not (sim.terminated or sim.truncated) and sim.step_count < 1800:
            for rock in sim._asteroids:
                tracks.setdefault(rock.id, []).append(
                    (Vec2(rock.pos.x, rock.pos.y), Vec2(rock.vel.x, rock.vel.y)))
                swings.append(rock.amplitude)
            sim.step({sim._ships[0].id: Action.NOOP})
        for track in tracks.values():
            for index in range(len(track) - HORIZON_FRAMES):
                position, velocity = track[index]
                later, _ = track[index + HORIZON_FRAMES]
                errors.append(wrapped_delta(
                    position + velocity * (HORIZON_FRAMES / 60.0), later, width, height
                ).length())
    errors.sort()
    return {"round": round_number, "swing": statistics.fmean(swings),
            "median": statistics.median(errors), "n": len(errors),
            "p90": errors[int(0.90 * len(errors))], "p99": errors[int(0.99 * len(errors))]}


def main() -> None:
    spec = load_curriculum(sys.argv[1])
    rounds = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [1, 28, 52, 96])
    seeds = range(30, 30 + (int(sys.argv[3]) if len(sys.argv) > 3 else 10))
    print(f"{sys.argv[1]}   ({len(seeds)} episodes per round)")
    print(f"{'round':>6} {'mean swing':>11} {'0.5s error: median':>20} {'p90':>8} {'p99':>8}")
    for number in rounds:
        row = measure(spec, number, seeds)
        print(f"{row['round']:>6} {row['swing']:>9.0f} px {row['median']:>17.1f} px "
              f"{row['p90']:>6.1f} px {row['p99']:>6.1f} px")


if __name__ == "__main__":
    main()
