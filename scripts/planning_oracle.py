"""Upper bound on round difficulty: a receding-horizon planner with perfect information.

The scripted `pilot` is a heuristic with no lookahead, so its clear rate measures the
pilot, not the task. This measures the task. At every decision it forks the *true*
simulation -- `Simulation` holds a single seeded `random.Random`, so `deepcopy` reproduces
future spawns exactly -- rolls K candidate plans forward H decisions, and commits the first
action of the best one.

Candidates are closed-loop perturbations of the pilot: at each decision a candidate either
takes the pilot's action or, with probability `epsilon`, a uniformly random one. Candidate 0
is the unperturbed pilot, so the planner can never score below the pilot except by sampling
noise. That matters: a weak oracle that fails to clear a bar proves nothing.

Usage:  python3 scripts/planning_oracle.py --stages 25 --seeds 32 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asteroid_survival.controllers import PilotController          # noqa: E402
from asteroid_survival.rl.curriculum import load_curriculum         # noqa: E402
from asteroid_survival.rl.ppo import _stage_env                     # noqa: E402

LAYOUT = {"history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
          "max_projectiles": 8, "version": 7}
SURVIVED_BONUS = 10_000.0
HIT_WEIGHT = 20.0
CLEARANCE_WEIGHT = 0.5


def _clearance(snapshot, agent_id: str, width: float, height: float) -> float:
    """Distance from the ship to the nearest rock, respecting screen wrap."""
    ship = next((s for s in snapshot.ships if s.id == agent_id), None)
    if ship is None or not snapshot.asteroids:
        return 0.0
    best = math.inf
    for a in snapshot.asteroids:
        dx = abs(a.x - ship.x) % width
        dy = abs(a.y - ship.y) % height
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        best = min(best, math.hypot(dx, dy) - a.radius)
    return max(0.0, best)


class PlanningOracle:
    def __init__(self, env, *, candidates: int, horizon: int, epsilon: float, seed: int,
                 blind: bool = False):
        self.env = env
        # `blind` reseeds the forked simulator's RNG for every rollout, so the planner keeps
        # its lookahead over the rocks already on the field (their paths are deterministic)
        # but cannot foresee where new spawns land or what speed, pattern and phase each
        # fragment of a rock it shoots will be dealt. That is the information a reactive
        # policy with perfect memory could in principle have; the default is an upper bound
        # that includes clairvoyance.
        self.blind = blind
        self.candidates = candidates
        self.horizon = horizon
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.pilot = PilotController()
        self.agent_id = env.agent_id
        self.frame_skip = env.frame_skip
        self.actions = env.actions
        self.width = env.config.arena.width
        self.height = env.config.arena.height

    def _pilot_index(self, state) -> int:
        return self.actions.index(self.pilot.action(state, self.agent_id))

    def _rollout(self, sim, perturbations: list[int | None]) -> float:
        hits = 0
        frames = 0
        snapshot = None
        for step in range(self.horizon):
            forced = perturbations[step]
            if forced is None:
                action = self.actions[self._pilot_index(sim.snapshot())]
            else:
                action = self.actions[forced]
            for _ in range(self.frame_skip):
                result = sim.step({self.agent_id: action})
                snapshot = result.snapshot
                frames += 1
                for event in result.events:
                    if event.kind == "asteroid_shot" and event.detail == self.agent_id:
                        hits += 1
                    elif (event.kind == "ship_destroyed"
                          and event.entity_id == self.agent_id):
                        return float(frames)          # died: score is how long it lasted
                if result.terminated or result.truncated:
                    return (SURVIVED_BONUS + HIT_WEIGHT * hits
                            + CLEARANCE_WEIGHT * _clearance(snapshot, self.agent_id,
                                                            self.width, self.height))
        return (SURVIVED_BONUS + HIT_WEIGHT * hits
                + CLEARANCE_WEIGHT * _clearance(snapshot, self.agent_id,
                                                self.width, self.height))

    def act(self) -> int:
        base = self._pilot_index(self.env.state)
        best_score, best_action = -math.inf, base
        for candidate in range(self.candidates):
            if candidate == 0:
                plan = [None] * self.horizon          # the unperturbed pilot
            else:
                plan = [self.rng.randrange(len(self.actions))
                        if self.rng.random() < self.epsilon else None
                        for _ in range(self.horizon)]
            first = base if plan[0] is None else plan[0]
            fork = copy.deepcopy(self.env.simulation)
            if self.blind:
                fork._rng = random.Random(self.rng.getrandbits(64))
            score = self._rollout(fork, plan)
            if score > best_score:
                best_score, best_action = score, first
        return best_action


def _write_trace(path: str, chunks_obs: list, chunks_act: list) -> None:
    """Write the dataset so far. Called periodically so a long run is crash-tolerant."""
    if not chunks_act:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary,
                        observations=np.concatenate(chunks_obs),
                        actions=np.concatenate(chunks_act))
    temporary.replace(destination)


def run_seed(args) -> dict:
    stage_index, seed, candidates, horizon, epsilon, curriculum, record, blind = args
    spec = load_curriculum(curriculum)
    env = _stage_env(spec, stage_index, LAYOUT)
    observation, _ = env.reset(seed)
    oracle = PlanningOracle(env, candidates=candidates, horizon=horizon,
                            epsilon=epsilon, seed=seed, blind=blind)
    done = False
    info: dict = {}
    trace = []
    while not done:
        action = oracle.act()
        if record:
            # The pair a behavioural-cloning run needs: what the policy sees, and what a
            # searcher with two seconds of verified lookahead does about it.
            trace.append((observation.astype(np.float32), int(action)))
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    metrics = info["episode_metrics"]
    limit = spec.stages[stage_index].max_seconds
    # Pack before returning. A list of 1265 Python floats per state costs ~32 bytes each;
    # as float32 it is 4, which is the difference between ~9 GB in the parent and ~0.5 GB.
    packed = ((np.stack([o for o, _ in trace]),
               np.asarray([a for _, a in trace], dtype=np.int64))
              if trace else (np.zeros((0, 0), np.float32), np.zeros((0,), np.int64)))
    return {"seed": seed, "trace_obs": packed[0], "trace_act": packed[1],
            "cleared": bool(metrics.get("completed_stage")),
            "survival_time": float(metrics["survival_time"]),
            "completion": min(1.0, float(metrics["survival_time"]) / limit),
            "destroyed": int(metrics["asteroids_destroyed"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="25", help="comma-separated zero-based indices")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=30, help="decisions of lookahead")
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--curriculum", default="configs/rl-survival-v2.toml")
    parser.add_argument("--output", default="metrics/planning-oracle.json")
    parser.add_argument("--blind", action="store_true",
                        help="reseed the forked RNG per rollout: lookahead without "
                             "clairvoyance about spawns and fragments")
    parser.add_argument("--record", metavar="PATH",
                        help="also write (observation, oracle action) pairs here as npz, "
                             "for behavioural cloning or agreement analysis")
    args = parser.parse_args()

    report = {"candidates": args.candidates, "horizon": args.horizon,
              "epsilon": args.epsilon, "seeds": args.seeds, "blind": bool(args.blind),
              "stages": {}}
    for stage_index in [int(x) for x in args.stages.split(",")]:
        jobs = [(stage_index, s, args.candidates, args.horizon, args.epsilon,
                 args.curriculum, bool(args.record), bool(args.blind))
                for s in range(args.seed_start, args.seed_start + args.seeds)]
        results = []
        chunks_obs, chunks_act = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, result in enumerate(pool.map(run_seed, jobs), start=1):
                if args.record:
                    chunks_obs.append(result.pop("trace_obs"))
                    chunks_act.append(result.pop("trace_act"))
                else:
                    result.pop("trace_obs", None)
                    result.pop("trace_act", None)
                results.append(result)
                if args.record and done % 20 == 0:
                    _write_trace(args.record, chunks_obs, chunks_act)
                    print(f"  {done}/{len(jobs)} episodes, "
                          f"{sum(len(c) for c in chunks_act):,} pairs checkpointed",
                          flush=True)
                elif done % 20 == 0:
                    print(f"  {done}/{len(jobs)} episodes", flush=True)
        clear = statistics.fmean(r["cleared"] for r in results)
        completion = statistics.fmean(r["completion"] for r in results)
        name = load_curriculum(args.curriculum).stages[stage_index].name
        if args.record:
            _write_trace(args.record, chunks_obs, chunks_act)
            print(f"  recorded {sum(len(c) for c in chunks_act):,} "
                  f"(observation, action) pairs -> {args.record}", flush=True)
        report["stages"][str(stage_index)] = {
            "name": name, "clear_rate": clear, "completion": completion,
            "mean_survival": statistics.fmean(r["survival_time"] for r in results),
            "episodes": results}
        print(f"{name} (index {stage_index}): oracle clear {clear:.3f}  "
              f"completion {completion:.3f}  "
              f"mean_survival {statistics.fmean(r['survival_time'] for r in results):.1f}s",
              flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
