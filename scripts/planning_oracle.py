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
        return self.decide()[0]

    def decide(self) -> tuple[int, int]:
        """The chosen action, and a bitmask of every first action that survived the horizon.

        The argmax is one of up to sixteen near-equivalent plans, so as a label it carries
        noise: two plans that both survive differ mostly in which random perturbation was
        drawn. The mask records every first action the search found survivable, which a
        cloner can score as a set instead of guessing which of them the oracle happened to
        pick. It is only as complete as the sample -- an action no candidate started with is
        absent, not unsafe.
        """
        base = self._pilot_index(self.env.state)
        best_score, best_action = -math.inf, base
        safe = 0
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
            if score >= SURVIVED_BONUS:
                safe |= 1 << first
            if score > best_score:
                best_score, best_action = score, first
        return best_action, safe


def _write_trace(path: str, chunks: dict[str, list]) -> None:
    """Write the dataset so far. Called periodically so a long run is crash-tolerant."""
    if not chunks["actions"]:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **{k: np.concatenate(v) for k, v in chunks.items()})
    temporary.replace(destination)


_POPCOUNT = sum(((np.arange(1 << 16, dtype=np.uint32) >> bit) & 1).astype(np.uint8)
                for bit in range(16))


def label_noise(actions: np.ndarray, safe: np.ndarray, num_actions: int) -> dict:
    """How reproducible the oracle's own label is, from repeated decisions on one state.

    `actions` and `safe` are (states, repeats). The search draws its perturbations fresh
    each time, so its argmax is a sample from a distribution over actions rather than a
    function of the state. `self_agreement` estimates the collision probability of that
    distribution, sum_a p_a^2, which is a *lower* bound on the top-1 accuracy a perfect
    learner could reach against a fresh label (that ceiling is max_a p_a >= sum_a p_a^2).
    So a cloner scoring above self-agreement is not thereby beating the oracle.
    """
    states, repeats = actions.shape
    if states == 0 or repeats < 2:
        return {}
    agree, in_other, jaccard = [], [], []
    for i in range(repeats):
        for j in range(repeats):
            if i == j:
                continue
            in_other.append((safe[:, j].astype(np.uint32) >> actions[:, i]) & 1)
            if i < j:
                agree.append(actions[:, i] == actions[:, j])
                union = _POPCOUNT[safe[:, i] | safe[:, j]].astype(np.float64)
                inter = _POPCOUNT[safe[:, i] & safe[:, j]].astype(np.float64)
                jaccard.append(np.divide(inter, union, out=np.ones_like(union),
                                         where=union > 0))
    onehot = actions[:, :, None] == np.arange(num_actions)[None, None, :]
    return {
        "states": int(states), "repeats": int(repeats),
        "self_agreement": float(np.mean(np.concatenate(agree))),
        "chosen_in_other_safe_set": float(np.mean(np.concatenate(in_other))),
        "safe_set_jaccard": float(np.mean(np.concatenate(jaccard))),
        # Biased upward at small `repeats` -- the most frequent of N draws overstates the
        # true mode -- so read it as an optimistic reading of the ceiling, not an estimate.
        "plurality_share": float(np.mean(onehot.sum(axis=1).max(axis=1) / repeats)),
        "mean_safe_set_size": float(np.mean(_POPCOUNT[safe])),
    }


def layout_from(checkpoint: str | None) -> dict:
    """The observation layout a recording must use to be fed back into `checkpoint`."""
    if not checkpoint:
        return LAYOUT
    metadata = json.loads((Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
    layout = dict(metadata["observation_layout"])
    for key, default in LAYOUT.items():
        layout.setdefault(key, default)
    return layout


def run_seed(job: dict) -> dict:
    stage_index, layout = job["stage"], job["layout"]
    spec = load_curriculum(job["curriculum"])
    env = _stage_env(spec, stage_index, layout)
    observation, _ = env.reset(job["seed"])
    oracle = PlanningOracle(env, candidates=job["candidates"], horizon=job["horizon"],
                            epsilon=job["epsilon"], seed=job["seed"], blind=job["blind"])
    repeats = max(1, int(job.get("repeat_labels", 1)))
    record = job["record"]
    policy = None
    if job.get("driver"):
        # DAgger: the policy under training steers, so the states visited are the ones it
        # actually reaches, and the oracle only supplies the label for each of them.
        import torch
        from asteroid_survival.rl.ppo import PPOController
        torch.set_num_threads(1)          # one worker per core; torch must not fan out
        policy = PPOController(job["driver"], device="cpu")
        policy.reset()
    done = False
    info: dict = {}
    trace = []
    while not done:
        # Repeated decisions on the *same* state, each with fresh perturbation draws. The
        # search is stochastic, so its argmax is a sample rather than a function of the
        # state; how often two draws agree is the label noise a cloner cannot train away.
        labels = [oracle.decide() for _ in range(repeats)]
        action, safe = labels[0]
        if record:
            # The pair a behavioural-cloning run needs: what the policy sees, and what a
            # searcher with two seconds of verified lookahead does about it.
            trace.append((observation.astype(np.float32), int(action), int(safe),
                          [a for a, _ in labels], [s for _, s in labels]))
        if policy is not None:
            action = policy(observation)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    metrics = info["episode_metrics"]
    limit = spec.stages[stage_index].max_seconds
    # Pack before returning. A list of 1265 Python floats per state costs ~32 bytes each;
    # as float32 it is 4, which is the difference between ~9 GB in the parent and ~0.5 GB.
    if trace:
        packed = {"observations": np.stack([row[0] for row in trace]),
                  "actions": np.asarray([row[1] for row in trace], dtype=np.int64),
                  "safe": np.asarray([row[2] for row in trace], dtype=np.uint16)}
        if repeats > 1:
            packed["repeat_actions"] = np.asarray([row[3] for row in trace], dtype=np.int64)
            packed["repeat_safe"] = np.asarray([row[4] for row in trace], dtype=np.uint16)
    else:
        packed = {"observations": np.zeros((0, 0), np.float32),
                  "actions": np.zeros((0,), np.int64), "safe": np.zeros((0,), np.uint16)}
        if repeats > 1:
            packed["repeat_actions"] = np.zeros((0, repeats), np.int64)
            packed["repeat_safe"] = np.zeros((0, repeats), np.uint16)
    return {"seed": job["seed"], "trace": packed,
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
    parser.add_argument("--layout-from", metavar="CHECKPOINT",
                        help="record observations in this checkpoint's layout, so the "
                             "pairs can be fed back into it (default: the v7 layout)")
    parser.add_argument("--driver", metavar="CHECKPOINT",
                        help="let this PPO checkpoint steer while the oracle only labels "
                             "(DAgger); the oracle's clear rate is then the driver's")
    parser.add_argument("--repeat-labels", type=int, default=1, metavar="N",
                        help="decide N times per state with fresh draws, to measure how "
                             "much of the label is noise; costs N times the rollouts")
    args = parser.parse_args()
    if args.repeat_labels > 1 and not args.record:
        parser.error("--repeat-labels needs --record: the repeated labels are written "
                     "beside the observations they belong to")

    layout = layout_from(args.layout_from)
    report = {"candidates": args.candidates, "horizon": args.horizon,
              "epsilon": args.epsilon, "seeds": args.seeds, "blind": bool(args.blind),
              "layout": layout, "driver": args.driver,
              "repeat_labels": args.repeat_labels, "stages": {}}
    for stage_index in [int(x) for x in args.stages.split(",")]:
        jobs = [{"stage": stage_index, "seed": s, "candidates": args.candidates,
                 "horizon": args.horizon, "epsilon": args.epsilon,
                 "curriculum": args.curriculum, "record": bool(args.record),
                 "blind": bool(args.blind), "layout": layout, "driver": args.driver,
                 "repeat_labels": args.repeat_labels}
                for s in range(args.seed_start, args.seed_start + args.seeds)]
        results = []
        chunks: dict[str, list] = {"observations": [], "actions": [], "safe": [],
                                   "episodes": []}
        if args.repeat_labels > 1:
            chunks.update(repeat_actions=[], repeat_safe=[])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, result in enumerate(pool.map(run_seed, jobs), start=1):
                trace = result.pop("trace")
                if args.record:
                    for key, value in trace.items():
                        chunks[key].append(value)
                    # Episode index per pair, so a cloner can hold out whole episodes:
                    # neighbouring frames are near-duplicates and a random split leaks.
                    chunks["episodes"].append(
                        np.full(len(trace["actions"]), done - 1, dtype=np.int32))
                results.append(result)
                if args.record and done % 20 == 0:
                    _write_trace(args.record, chunks)
                    print(f"  {done}/{len(jobs)} episodes, "
                          f"{sum(len(c) for c in chunks['actions']):,} pairs checkpointed",
                          flush=True)
                elif done % 20 == 0:
                    print(f"  {done}/{len(jobs)} episodes", flush=True)
        clear = statistics.fmean(r["cleared"] for r in results)
        completion = statistics.fmean(r["completion"] for r in results)
        name = load_curriculum(args.curriculum).stages[stage_index].name
        if args.record:
            _write_trace(args.record, chunks)
            print(f"  recorded {sum(len(c) for c in chunks['actions']):,} "
                  f"(observation, action) pairs -> {args.record}", flush=True)
        if args.repeat_labels > 1 and chunks["repeat_actions"]:
            noise = label_noise(np.concatenate(chunks["repeat_actions"]),
                                np.concatenate(chunks["repeat_safe"]),
                                len(_stage_env(load_curriculum(args.curriculum),
                                               stage_index, layout).actions))
            report.setdefault("label_noise", {})[str(stage_index)] = noise
            print(f"  label noise: the oracle repeats its own choice "
                  f"{noise['self_agreement']:.3f} of the time over {noise['states']:,} "
                  f"states; its pick is in another draw's safe set "
                  f"{noise['chosen_in_other_safe_set']:.3f}, safe-set Jaccard "
                  f"{noise['safe_set_jaccard']:.3f}", flush=True)
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
