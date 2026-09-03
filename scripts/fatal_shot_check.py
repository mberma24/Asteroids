"""Would the fire-consequence prediction have flagged the shots that kill the policy?

The measurement the observation block is worth building for. For every episode it replays a
checkpoint, records what `Simulation.fire_consequence` said at each decision the policy
fired, attributes each fragment to the shot that created it, and reports the predicted
worst-case fragment clearance of the shot whose piece went on to kill the ship -- against
the same prediction for every other shot taken.

Usage: python scripts/fatal_shot_check.py CHECKPOINT SEEDS WORKERS [CURRICULUM]
"""
import json, math, statistics, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import _stage_env, PPOController

LAYOUT = {"history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
          "max_projectiles": 8, "version": 7}
CKPT = sys.argv[1]
CURRICULUM = sys.argv[4] if len(sys.argv) > 4 else "configs/rl-survival-v2-detfrag.toml"
STAGE = 25


def run(seed):
    spec = load_curriculum(CURRICULUM)
    env = _stage_env(spec, STAGE, LAYOUT)
    ctl = PPOController(CKPT)
    obs, _ = env.reset(seed)
    fps = env.config.arena.fps
    shots = []          # (predicted hit time in seconds, prediction) for each shot fired
    fragment_shot = {}  # fragment id -> the prediction of the shot that created it
    events = []
    real_step = env.simulation.step

    def wrapped(actions):
        result = real_step(actions)
        events.extend((env.simulation.step_count, e) for e in result.events)
        return result

    env.simulation.step = wrapped
    done = False
    while not done:
        before = len(events)
        elapsed = env.state.elapsed
        action_index = ctl(obs)
        action = env.actions[action_index]
        # The simulator rotates before it fires, so ask about the heading the shot actually
        # leaves along, not the one the ship is sitting at when the decision is made.
        prediction = (env.simulation.fire_consequence(
            env.agent_id, turn=action.turn, thrust=action.thrust,
            within_frames=env.frame_skip) if action.fire else None)
        obs, _, terminated, truncated, info = env.step(action_index)
        fired = any(e.kind == "projectile_fired" and e.detail == env.agent_id
                    for _, e in events[before:])
        if fired and prediction is not None:
            shots.append((elapsed + prediction.time_to_hit, prediction))
        for _, e in events[before:]:
            if e.kind != "asteroid_split":
                continue
            parent, child = int(e.detail), int(e.entity_id)
            when = env.state.elapsed
            match = [(abs(t - when), p) for t, p in shots
                     if p.target_id == parent and abs(t - when) < 0.35]
            if match:
                fragment_shot[child] = min(match)[1]
        done = terminated or truncated
    metrics = info["episode_metrics"]
    out = {"seed": seed, "cleared": bool(metrics["completed_stage"]),
           "shots_predicted": len(shots),
           "all_clearances": [p.worst_clearance for _, p in shots if p.splits]}
    if metrics["completed_stage"]:
        return out
    kill = [e for _, e in events if e.kind == "ship_destroyed" and e.entity_id == env.agent_id]
    if not kill:
        return out
    killer = int(kill[0].detail)
    out["killer_attributed"] = killer in fragment_shot
    if killer in fragment_shot:
        prediction = fragment_shot[killer]
        out["fatal_clearance"] = prediction.worst_clearance
        out["fatal_time_to_hit"] = prediction.time_to_hit
        out["fatal_distance"] = prediction.distance
        out["fatal_size"] = prediction.size
    return out


if __name__ == "__main__":
    seeds = list(range(10000, 10000 + int(sys.argv[2])))
    with ProcessPoolExecutor(max_workers=int(sys.argv[3])) as pool:
        rows = list(pool.map(run, seeds))
    deaths = [r for r in rows if not r["cleared"]]
    attributed = [r for r in deaths if r.get("killer_attributed")]
    every = [c for r in rows for c in r["all_clearances"]]
    fatal = [r["fatal_clearance"] for r in attributed]
    print(f"clear {statistics.fmean(r['cleared'] for r in rows):.3f} over {len(rows)}; "
          f"deaths {len(deaths)}, killer traced to a predicted shot in {len(attributed)}")
    if every:
        print(f"predicted worst fragment clearance, every splitting shot (n={len(every)}): "
              f"median {statistics.median(every):.0f}px, "
              f"{sum(c <= 0 for c in every) / len(every):.1%} predicted to hit the ship")
    if fatal:
        print(f"                            the shots that killed it (n={len(fatal)}): "
              f"median {statistics.median(fatal):.0f}px, "
              f"{sum(c <= 0 for c in fatal) / len(fatal):.1%} predicted to hit the ship")
        print("fatal shot clearances:", [round(c) for c in sorted(fatal)])
        print("fatal shot range (px):", [round(r["fatal_distance"]) for r in attributed])
        print("fatal target size:", Counter(r["fatal_size"] for r in attributed))
    json.dump(rows, open("metrics/fatal-shot-check.json", "w"), indent=1)
