"""Trace actual projectile collisions independently of the warning being evaluated.

Read-only policy rollout: no action masking, rewards, or simulation rules are changed.
Each fatal child is joined to its real creating projectile, including missed predictions.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, "src")
import torch
from asteroid_survival.math2d import wrapped_distance
from asteroid_survival.simulation import ASTEROID_RADII
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import _stage_env, PPOController


def run_episode(ctl, spec, layout, stage, seed):
    env = _stage_env(spec, stage, layout)
    obs, _ = env.reset(seed)
    sim = env.simulation
    shots, children = {}, {}
    pending = {}
    matches = {}
    fatal = None
    real_collisions = sim._collisions
    real_step = sim.step

    def collisions(events):
        # Identical ordering/geometry to the real projectile collision loop, at the
        # actual physics frame, before it removes parents and appends children.
        removed = set()
        matches.clear()
        for projectile in sim._projectiles:
            for rock in sim._asteroids:
                if rock.id in removed:
                    continue
                if wrapped_distance(projectile.pos, rock.pos, env.config.arena.width,
                                    env.config.arena.height) > (
                        env.config.projectile.radius + ASTEROID_RADII[rock.size]):
                    continue
                removed.add(rock.id)
                matches[rock.id] = projectile.id
                break
        real_collisions(events)

    def step(actions):
        nonlocal fatal
        result = real_step(actions)
        for event in result.events:
            if event.kind == "projectile_fired" and event.detail == env.agent_id:
                shots[int(event.entity_id)] = dict(pending, fired_frame=sim.step_count)
            elif event.kind == "asteroid_shot":
                parent = int(event.entity_id)
                assert parent in matches, "trace disagrees with simulator collision"
                projectile = matches[parent]
                if projectile in shots:
                    shots[projectile]["actual_target"] = parent
                    shots[projectile]["hit_frame"] = sim.step_count
            elif event.kind == "asteroid_split":
                parent, child = int(event.detail), int(event.entity_id)
                children[child] = matches[parent]
            elif event.kind == "ship_destroyed" and event.entity_id == env.agent_id:
                killer = int(event.detail)
                shot = shots.get(children.get(killer))
                fatal = {"killer": killer, "fragment": killer in children,
                         "shot": shot, "death_frame": sim.step_count}
        return result

    sim._collisions = collisions
    sim.step = step
    done = False
    while not done:
        index = ctl(obs)
        action = env.actions[index]
        pending = {}
        if action.fire:
            for name, corrected, actual in (
                    ("legacy_observed", False, False),
                    ("corrected_observed", True, False),
                    ("legacy_action", False, True),
                    ("corrected_action", True, True)):
                prediction = sim.fire_consequence(
                    env.agent_id, within_frames=env.frame_skip, corrected=corrected,
                    turn=action.turn if actual else 0.0,
                    thrust=action.thrust if actual else False)
                pending[name] = asdict(prediction) if prediction else None
        obs, _, terminated, truncated, info = env.step(index)
        done = terminated or truncated
    counts = {}
    for name in ("legacy_observed", "corrected_observed", "legacy_action", "corrected_action"):
        counts[name] = {
            "predicted": sum(s.get(name) is not None for s in shots.values()),
            "flagged": sum(s.get(name) is not None and s[name]["splits"]
                           and s[name]["worst_clearance"] <= 0 for s in shots.values()),
            "correct_target": sum(s.get(name) is not None and
                                  s[name]["target_id"] == s.get("actual_target")
                                  for s in shots.values())}
    return {"seed": seed, "cleared": bool(info["episode_metrics"]["completed_stage"]),
            "shots": len(shots), "hits": sum("actual_target" in s for s in shots.values()),
            "counts": counts, "fatal": fatal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--curriculum", default="configs/rl-survival-v3.toml")
    parser.add_argument("--round", type=int, default=29)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=1000001200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    layout = json.loads((Path(args.checkpoint) / "metadata.json").read_text())["observation_layout"]
    ctl = PPOController(args.checkpoint)
    spec = load_curriculum(args.curriculum)
    started = time.monotonic()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rows.append(run_episode(ctl, spec, layout, args.round - 1, seed))
        if len(rows) % 8 == 0:
            print(f"{len(rows)}/{args.seeds} seeds, {time.monotonic()-started:.1f}s", flush=True)
    fatal_shots = [r["fatal"]["shot"] for r in rows
                   if r["fatal"] and r["fatal"]["shot"] is not None]
    summary = {"episodes": len(rows), "clears": sum(r["cleared"] for r in rows),
               "deaths": sum(r["fatal"] is not None for r in rows),
               "fragment_deaths": sum(bool(r["fatal"] and r["fatal"]["fragment"]) for r in rows),
               "traced_fatal_shots": len(fatal_shots), "warnings": {}}
    for name in ("legacy_observed", "corrected_observed", "legacy_action", "corrected_action"):
        summary["warnings"][name] = {
            "fatal_predicted": sum(s.get(name) is not None for s in fatal_shots),
            "fatal_target_correct": sum(s.get(name) is not None and
                                       s[name]["target_id"] == s["actual_target"] for s in fatal_shots),
            "fatal_flagged": sum(s.get(name) is not None and s[name]["splits"] and
                                 s[name]["worst_clearance"] <= 0 for s in fatal_shots),
            "fatal_target_correct_and_flagged": sum(
                s.get(name) is not None and s[name]["target_id"] == s["actual_target"]
                and s[name]["splits"] and s[name]["worst_clearance"] <= 0 for s in fatal_shots),
            "all_shots": sum(r["shots"] for r in rows),
            "all_flagged": sum(r["counts"][name]["flagged"] for r in rows)}
    report = {"checkpoint": args.checkpoint, "round": args.round, "summary": summary,
              "elapsed_seconds": time.monotonic()-started, "rows": rows}
    with open(args.output, "x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
