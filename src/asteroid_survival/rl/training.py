from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..config import load_config
from ..controllers import ClosestAsteroidController
from .environment import (ASTEROID_FEATURES, PROJECTILE_FEATURES, AsteroidsRLEnv,
                          RewardConfig, ship_feature_count)
from .evaluation import evaluate_policy, save_evaluation
from .muzero import MuZeroAgent, MuZeroSettings, ReplayBuffer, Transition, finish_episode


SUMMARY_FIELDS = ("survival_time", "wave", "waves_cleared", "asteroids_destroyed",
                  "accuracy", "reward", "shots_fired")


def _resume_artifact(destination: Path, resume: str | Path, filename: str) -> Path:
    """Use local run state first, then the run that owns the resume checkpoint.

    A checkpoint only contains the neural network.  Curriculum position and replay live next
    to it, so resuming into a newly named directory must explicitly recover those artifacts.
    """
    local = destination / filename
    if local.exists():
        return local
    checkpoint_copy = Path(resume) / filename
    if checkpoint_copy.exists():
        return checkpoint_copy
    return Path(resume).parent / filename


def _checkpoint_score(record: dict, completion_target: float = 0.80,
                      accuracy_target: float = 0.20) -> tuple[int, int, float, float, float, float]:
    """Rank a checkpoint by progress toward the same two gates used for promotion.

    Completion-first ordering rejected policies that passed both gates if an older model
    happened to complete two more of the 32 validation seeds.  The bottleneck ratio instead
    rewards whichever requirement is furthest behind, and a gate-passing policy always beats
    one that has not passed.
    """
    stages = record.get("stages") or []
    if not stages:
        return (-1, 0, float(record.get("mean_survival_time", 0.0)),
                float(record.get("mean_accuracy", 0.0)), 0.0, 0.0)
    index = min(int(record.get("training_stage", 0)), len(stages) - 1)
    stage = stages[index]
    completion = float(stage.get("completion_rate", 0.0))
    accuracy = float(stage.get("mean_accuracy", 0.0))
    completion_target = max(float(completion_target), 1e-9)
    accuracy_target = max(float(accuracy_target), 1e-9)
    completion_ratio = completion / completion_target
    accuracy_ratio = accuracy / accuracy_target
    passed = int(completion_ratio >= 1.0 and accuracy_ratio >= 1.0)
    return (index, passed, min(completion_ratio, accuracy_ratio),
            completion_ratio + accuracy_ratio, completion, accuracy)


class ChampionTracker:
    """Persist the best held-out policy without mutating the active learner."""

    def __init__(self, run: str | Path, retention_completion: float, patience: int = 4,
                 retention_floor: float = 0.50,
                 initial_learning_rate: float = 1e-3,
                 rollback_lr_factor: float = 0.5,
                 minimum_learning_rate: float = 1.25e-4,
                 promotion_completion: float = 0.80,
                 accuracy_targets: tuple[float, ...] = ()):
        self.run = Path(run)
        self.path = self.run / "champion"
        self.state_path = self.run / "champion_state.json"
        self.retention_completion = retention_completion
        self.retention_floor = float(retention_floor)
        self.patience = patience
        self.initial_learning_rate = float(initial_learning_rate)
        self.rollback_lr_factor = float(rollback_lr_factor)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.promotion_completion = float(promotion_completion)
        self.accuracy_targets = tuple(float(value) for value in accuracy_targets)
        self.state: dict = {}

    def _score(self, record: dict) -> tuple[int, int, float, float, float, float]:
        stages = record.get("stages") or []
        index = min(int(record.get("training_stage", 0)), max(0, len(stages) - 1))
        accuracy_target = (self.accuracy_targets[index]
                           if index < len(self.accuracy_targets) else 0.20)
        return _checkpoint_score(
            record, completion_target=self.promotion_completion,
            accuracy_target=accuracy_target)

    def _learning_rate_after(self, rollbacks: int) -> float:
        return max(self.minimum_learning_rate,
                   self.initial_learning_rate * self.rollback_lr_factor ** rollbacks)

    def _eligible(self, record: dict) -> bool:
        from .curriculum import retention_holds

        stages = record.get("stages") or []
        if not stages:
            return True
        current = min(int(record.get("training_stage", 0)), len(stages) - 1)
        return retention_holds(stages[:current],
                               retention_completion=self.retention_completion,
                               retention_floor=self.retention_floor)

    def _install(self, checkpoint: Path, record: dict, *, rollbacks: int | None = None) -> None:
        temporary = self.run / ".champion-new"
        previous = self.run / ".champion-old"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        shutil.copytree(checkpoint, temporary)
        if self.path.exists():
            self.path.replace(previous)
        temporary.replace(self.path)
        shutil.rmtree(previous, ignore_errors=True)
        self.state = {
            "episode": int(record["episode"]),
            "training_stage": int(record.get("training_stage", 0)),
            "score": list(self._score(record)),
            "completion_rate": float((record.get("stages") or [{}])[
                min(int(record.get("training_stage", 0)),
                    len(record.get("stages") or [{}]) - 1)
            ].get("completion_rate", 0.0)),
            "accuracy": float((record.get("stages") or [{}])[
                min(int(record.get("training_stage", 0)),
                    len(record.get("stages") or [{}]) - 1)
            ].get("mean_accuracy", 0.0)),
            "evaluations_since_improvement": 0,
            "retention_failures": 0,
            "recoveries": int(self.state.get("recoveries", 0)),
            "restorations": int(self.state.get("restorations", 0)),
            "rollbacks": int(self.state.get("rollbacks", 0) if rollbacks is None else rollbacks),
            "patience": self.patience,
            "initial_learning_rate": self.initial_learning_rate,
            "learning_rate": float(self.state.get(
                "learning_rate", self._learning_rate_after(int(
                    self.state.get("rollbacks", 0) if rollbacks is None else rollbacks)))),
            "rollback_lr_factor": self.rollback_lr_factor,
            "minimum_learning_rate": self.minimum_learning_rate,
        }
        self.save()

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def bootstrap(self, evaluation_log: Path) -> bool:
        """Load persisted state, or discover the best eligible retained checkpoint."""
        if self.state_path.is_file() and self.path.is_dir():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state["patience"] = self.patience
            self.state["initial_learning_rate"] = self.initial_learning_rate
            self.state["rollback_lr_factor"] = self.rollback_lr_factor
            self.state["minimum_learning_rate"] = self.minimum_learning_rate
            self.state.setdefault(
                "learning_rate",
                self._learning_rate_after(int(self.state.get("rollbacks", 0))))
            self.state.setdefault("retention_failures", 0)
            self.state.setdefault("recoveries", 0)
            self.state.setdefault("restorations", 0)
            # Migrate completion-first scores written by the destructive rollback version.
            if evaluation_log.is_file():
                for line in evaluation_log.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if int(record.get("episode", -1)) == int(self.state.get("episode", -2)):
                        self.state["score"] = list(self._score(record))
                        break
            self.save()
            return False
        records = []
        if evaluation_log.is_file():
            for line in evaluation_log.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    checkpoint = self.run / f"checkpoint_{int(record['episode']):06d}"
                except (ValueError, KeyError, TypeError):
                    continue
                if checkpoint.is_dir() and self._eligible(record):
                    records.append((record, checkpoint))
        if not records:
            return False
        record, checkpoint = max(records, key=lambda item: self._score(item[0]))
        self._install(checkpoint, record, rollbacks=0)
        return True

    def forgotten_stage(self, record: dict) -> int | None:
        stages = record.get("stages") or []
        current = min(int(record.get("training_stage", 0)), max(0, len(stages) - 1))
        failures = [
            (float(stage.get("completion_rate", 0.0)) / self.retention_completion, index)
            for index, stage in enumerate(stages[:current])
            if int(stage.get("episodes", 1) or 0) > 0
            and float(stage.get("completion_rate", 0.0)) < self.retention_completion
        ]
        return min(failures)[1] if failures else None

    def consider(self, record: dict, checkpoint: Path, *, allow_recovery: bool = True) -> str:
        """Return ``improved``, ``continue``, ``restore``, or ``recover``.

        A plateau on the new lesson is not forgetting. Only repeated failure on a mastered
        lesson starts mixed recovery. A same-stage plateau restores champion weights while
        replay remains intact, so good policy parameters are not slowly trained away.
        """
        if not self.state or (
                self._eligible(record)
                and self._score(record) > tuple(self.state.get("score", (-1, 0, 0)))):
            self._install(checkpoint, record)
            return "improved"
        self.state["evaluations_since_improvement"] = int(
            self.state.get("evaluations_since_improvement", 0)) + 1
        if self._eligible(record) or not allow_recovery:
            self.state["retention_failures"] = 0
            if (allow_recovery and self.state["evaluations_since_improvement"]
                    >= self.patience):
                self.state["evaluations_since_improvement"] = 0
                self.state["restorations"] = int(self.state.get("restorations", 0)) + 1
                self.save()
                return "restore"
            self.save()
            return "continue"
        self.state["retention_failures"] = int(self.state.get("retention_failures", 0)) + 1
        if self.state["retention_failures"] >= self.patience:
            self.state["retention_failures"] = 0
            self.state["recoveries"] = int(self.state.get("recoveries", 0)) + 1
            self.state["evaluations_since_improvement"] = 0
            self.save()
            return "recover"
        self.save()
        return "continue"


def restore_agent_from_champion(agent: MuZeroAgent, checkpoint: str | Path,
                                learning_rate: float, seed: int) -> None:
    """Restore policy/model weights but deliberately retain replay and episode counters."""
    champion = MuZeroAgent.load(checkpoint, seed=seed)
    if (champion.observation_size, champion.num_actions) != (
            agent.observation_size, agent.num_actions):
        raise ValueError("champion observation/action shapes do not match the challenger")
    agent.params = champion.params
    agent.reset_optimizer(learning_rate)


def prune_checkpoints(run: str | Path, keep: int = 3) -> list[Path]:
    """Keep the best evaluated checkpoint plus the newest checkpoints in a run."""
    directory = Path(run)
    checkpoints = sorted(path for path in directory.glob("checkpoint_*") if path.is_dir())
    has_champion = (directory / "champion").is_dir()
    target = max(1, keep - 1) if has_champion else keep
    if keep <= 0 or len(checkpoints) <= target:
        return []
    records: dict[int, dict] = {}
    evaluation_log = directory / "evaluation.jsonl"
    if evaluation_log.is_file():
        for line in evaluation_log.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                records[int(record["episode"])] = record
            except (ValueError, KeyError, TypeError):
                continue
    evaluated = [path for path in checkpoints
                 if int(path.name.removeprefix("checkpoint_")) in records]
    protected = set(checkpoints[-max(1, target if has_champion else keep - 1):])
    # The dedicated champion directory supersedes the historical best checkpoint copy.
    # Runs without champion support retain the old best-plus-newest behavior.
    if evaluated and keep > 1 and not has_champion:
        protected.add(max(
            evaluated,
            key=lambda path: _checkpoint_score(
                records[int(path.name.removeprefix("checkpoint_"))])))
    for path in reversed(checkpoints):
        if len(protected) >= target:
            break
        protected.add(path)
    removed = []
    for path in checkpoints:
        if path not in protected:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def training_seed(base: int, index: int, reserved_start: int, reserved_count: int) -> int:
    """Seed for the ``index``-th training episode, skipping the held-out evaluation band.

    Training seeds advance by one per episode, so a long enough run eventually walks onto the
    frozen evaluation seeds and starts training on the very levels it is scored against.
    Reserving that band keeps the held-out numbers honest no matter how long a run goes.
    """
    seed = base + index
    if reserved_count > 0 and seed >= reserved_start:
        seed += reserved_count
    return seed


def truncate_log_after_checkpoint(path: str | Path, episode: int) -> int:
    """Discard unsaved tail records when resuming from an earlier durable checkpoint."""
    source = Path(path)
    if not source.is_file():
        return 0
    lines = source.read_text(encoding="utf-8").splitlines()
    kept = []
    for line in lines:
        try:
            record_episode = int(json.loads(line)["episode"])
        except (ValueError, KeyError, TypeError):
            continue
        if record_episode <= episode:
            kept.append(line)
    removed = len(lines) - len(kept)
    if removed:
        source.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return removed


def observation_layout(env: AsteroidsRLEnv) -> dict:
    return {
        "version": 4,
        "ship_features": ship_feature_count(env.config),
        "asteroid_features": ASTEROID_FEATURES,
        "max_asteroids": env.max_asteroids,
        "projectile_features": PROJECTILE_FEATURES,
        "max_projectiles": env.max_projectiles,
        "history_frames": env.history_frames,
        "history_long_frames": env.history_long_frames,
        "history_long_stride": env.history_long_stride,
        "history_offsets": env.history_slots,
        "mobile": env.config.ship.mobile,
        "actions": [action.name for action in env.actions],
    }


def initialize_agent_from_policy(checkpoint: str | Path, observation_size: int,
                                 num_actions: int, settings: MuZeroSettings,
                                 seed: int) -> MuZeroAgent:
    """Transfer perception, dynamics, and aiming while resetting task-specific learning."""
    source = MuZeroAgent.load(checkpoint, seed=seed)
    if source.num_actions != num_actions or source.observation_size > observation_size:
        raise ValueError("initial checkpoint observation/action shapes cannot be transferred")
    agent = MuZeroAgent(observation_size, num_actions, settings, seed=seed)
    for name in ("dynamics_net", "policy_hidden", "policy_output"):
        agent.params["params"][name] = source.params["params"][name]
    source_representation = source.params["params"]["representation_net"]
    target_representation = agent.params["params"]["representation_net"]
    expanded_input = False
    for layer_name, source_layer in source_representation.items():
        target_layer = target_representation[layer_name]
        for parameter_name, source_value in source_layer.items():
            target_value = target_layer[parameter_name]
            if source_value.shape == target_value.shape:
                target_layer[parameter_name] = source_value
            elif (parameter_name == "kernel"
                  and source_value.ndim == target_value.ndim == 2
                  and source_value.shape[0] == source.observation_size
                  and target_value.shape[0] == observation_size
                  and source_value.shape[1] == target_value.shape[1]):
                # Observation v4 appends projectile slots. The old rows retain their exact
                # semantics and weights; only appended rows keep the fresh initializer.
                target_layer[parameter_name] = target_value.at[
                    :source.observation_size, :].set(source_value)
                expanded_input = True
            else:
                raise ValueError(
                    f"cannot transfer representation parameter {layer_name}/{parameter_name}: "
                    f"{source_value.shape} -> {target_value.shape}")
    if source.observation_size != observation_size and not expanded_input:
        raise ValueError("could not locate the representation input layer to expand")
    # The fresh agent retains new value/reward/continuation heads, optimizer moments,
    # episode count, and training-step count. Old replay is intentionally not loaded.
    return agent


def _initial_stage_from_checkpoint(checkpoint: str | Path) -> int:
    """Continue a transferred champion at the lesson it proved it could train on."""
    source = Path(checkpoint)
    state_path = source / "curriculum_state.json"
    if state_path.is_file():
        try:
            return int(json.loads(state_path.read_text(encoding="utf-8")).get("stage", 0))
        except (OSError, ValueError, TypeError):
            return 0
    champion_state = source.parent / "champion_state.json"
    if source.name == "champion" and champion_state.is_file():
        try:
            state = json.loads(champion_state.read_text(encoding="utf-8"))
            return int(state.get("training_stage", 0))
        except (OSError, ValueError, TypeError):
            return 0
    return 0


def summarize_episodes(records: list[dict]) -> dict:
    """Collapse a block of episodes into best/worst/average stats.

    Single-episode numbers are too noisy to read as progress, so the run reports blocks.
    """
    summary: dict = {
        "episodes": f"{records[0]['episode']}-{records[-1]['episode']}",
        "count": len(records),
    }
    for field in SUMMARY_FIELDS:
        values = [float(record[field]) for record in records if field in record]
        if not values:
            continue
        summary[field] = {
            "avg": round(statistics.fmean(values), 3),
            "best": round(max(values), 3),
            "worst": round(min(values), 3),
        }
    completed = [bool(record["completed_stage"]) for record in records
                 if "completed_stage" in record]
    if completed:
        summary["completion_rate"] = round(statistics.fmean(completed), 3)
    losses = [float(record["loss"]) for record in records if "loss" in record]
    if losses:
        summary["loss_avg"] = round(statistics.fmean(losses), 4)
    return summary


def format_summary(summary: dict) -> str:
    def block(field: str, unit: str = "") -> str:
        stats = summary.get(field)
        if not stats:
            return ""
        return (f" | {field.replace('_', ' ')}: avg {stats['avg']}{unit} "
                f"best {stats['best']}{unit} worst {stats['worst']}{unit}")

    line = f"episodes {summary['episodes']} ({summary['count']})"
    line += block("survival_time", "s")
    if "completion_rate" in summary:
        line += f" | completion {summary['completion_rate']:.1%}"
        line += block("waves_cleared")
    elif summary.get("wave", {}).get("best"):
        line += block("wave")
    line += block("asteroids_destroyed") + block("accuracy") + block("reward")
    if "loss_avg" in summary:
        line += f" | loss {summary['loss_avg']}"
    return line


def train(config_path: str | Path, output_dir: str | Path, *, episodes: int, seed: int,
          simulations: int, max_decisions: int, checkpoint_every: int,
          updates_per_episode: int = 32,
          resume: str | Path | None = None, asteroid_reward: float = 0.1,
          eval_every: int = 0, eval_episodes: int = 20, eval_seed: int = 10_000,
          log_every: int = 10, parallel_envs: int = 1, shot_penalty: float = 0.0,
          history_frames: int = 0, history_long_frames: int = 0,
          history_long_stride: int = 8) -> Path:
    config = load_config(config_path)
    env = AsteroidsRLEnv(
        config, frame_skip=4, max_decisions=max_decisions, asteroid_reward=asteroid_reward,
        shot_penalty=shot_penalty, history_frames=history_frames,
        history_long_frames=history_long_frames, history_long_stride=history_long_stride)
    settings = MuZeroSettings(num_simulations=simulations,
                              updates_per_episode=updates_per_episode)
    agent = (MuZeroAgent.load(resume, seed=seed) if resume else
             MuZeroAgent(env.observation_size, env.num_actions, settings, seed=seed))
    if (agent.observation_size, agent.num_actions) != (env.observation_size, env.num_actions):
        raise ValueError("checkpoint observation/action shapes do not match this configuration")
    if resume:
        agent.settings.num_simulations = simulations
        agent.settings.updates_per_episode = updates_per_episode
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_log = destination / "training.jsonl"
    evaluation_log = destination / "evaluation.jsonl"
    if resume:
        removed = (truncate_log_after_checkpoint(training_log, agent.episodes)
                   + truncate_log_after_checkpoint(evaluation_log, agent.episodes))
        if removed:
            print(json.dumps({"discarded_unsaved_log_records": removed}), flush=True)
    replay_path = destination / "replay.npz"
    settings = agent.settings
    replay = ReplayBuffer(settings.replay_capacity, seed)
    if resume:
        restored = replay.load(
            replay_path, observation_size=env.observation_size, num_actions=env.num_actions)
        print(json.dumps({"resumed_replay_transitions": restored}))
    # A fixed seed set held out from training, so checkpoints are comparable to each other.
    evaluation_seeds = list(range(eval_seed, eval_seed + eval_episodes))
    evaluation_env = AsteroidsRLEnv(
        config, frame_skip=4, max_decisions=max_decisions, asteroid_reward=asteroid_reward,
        shot_penalty=shot_penalty, history_frames=history_frames,
        history_long_frames=history_long_frames, history_long_stride=history_long_stride)

    layout = observation_layout(env)
    final_episode = agent.episodes + episodes
    pending: list[dict] = []
    # Self-play environments stepped in lockstep so one batched search serves all of them.
    workers = [
        AsteroidsRLEnv(config, frame_skip=4, max_decisions=max_decisions,
                       asteroid_reward=asteroid_reward, shot_penalty=shot_penalty,
                       history_frames=history_frames,
                       history_long_frames=history_long_frames,
                       history_long_stride=history_long_stride)
        for _ in range(max(1, parallel_envs))
    ]
    reserved = eval_episodes if eval_every else 0
    observations = [worker.reset(training_seed(
        seed, agent.episodes + index, eval_seed, reserved))[0]
                    for index, worker in enumerate(workers)]
    trajectories: list[list[Transition]] = [[] for _ in workers]
    # None marks a worker whose episode is surplus to --episodes; it keeps stepping only so
    # the batch shape stays constant, and its trajectory is discarded.
    active = [True] * len(workers)
    assigned = agent.episodes + len(workers)
    episode = agent.episodes

    while episode < final_episode:
        actions, policies, search_values = agent.search_batch(
            np.stack(observations), explore=True)
        for index, worker in enumerate(workers):
            action = int(actions[index])
            next_observation, reward, terminated, truncated, info = worker.step(action)
            done = terminated or truncated
            trajectories[index].append(Transition(
                observations[index], action, reward, policies[index], next_observation, done,
                search_value=float(search_values[index]),
            ))
            observations[index] = next_observation
            if not done:
                continue

            trajectory = trajectories[index]
            trajectories[index] = []
            observations[index] = worker.reset(
                training_seed(seed, assigned, eval_seed, reserved))[0]
            assigned += 1
            if not active[index]:
                continue
            episode += 1
            finish_episode(trajectory, settings.discount, n_step=settings.n_step,
                           episode_id=episode,
                           successful=bool(info["episode_metrics"].get("completed_stage")))
            replay.extend(trajectory)
            losses = {}
            for _ in range(settings.updates_per_episode):
                if len(replay) >= settings.batch_size:
                    losses = agent.train_batch(
                        replay.sample(settings.batch_size, settings.unroll_steps))
            record = {"episode": episode, **info["episode_metrics"], **losses}
            with training_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            pending.append(record)
            if log_every and (len(pending) >= log_every or episode == final_episode):
                print(format_summary(summarize_episodes(pending)), flush=True)
                pending = []
            if episode % checkpoint_every == 0 or episode == final_episode:
                agent.save(destination / f"checkpoint_{episode:06d}", episodes=episode,
                           observation_layout=layout)
                replay.save(replay_path)
            if episode + sum(active) > final_episode:
                active[index] = False  # enough episodes are already in flight
            _maybe_evaluate(episode, final_episode, eval_every, agent, evaluation_env,
                            evaluation_seeds, evaluation_log)
    return destination / f"checkpoint_{final_episode:06d}"


def train_curriculum(curriculum_path: str | Path, output_dir: str | Path, *, episodes: int,
                     seed: int, simulations: int, checkpoint_every: int = 250,
                     updates_per_episode: int = 32,
                     eval_every: int = 250, parallel_envs: int = 16,
                     history_frames: int = 8, history_long_frames: int = 8,
                     history_long_stride: int = 8, log_every: int = 250,
                     resume: str | Path | None = None,
                     initialize_from: str | Path | None = None, eval_seed: int = 10_000,
                     keep_checkpoints: int = 3, stop_when_mastered: bool = False,
                     champion_patience: int = 4, rollback_lr_factor: float = 0.5,
                     minimum_learning_rate: float = 1.25e-4,
                     learning_rate: float = 1e-3,
                     resume_learning_rate: float | None = None,
                     start_stage: int | None = None) -> Path:
    """Train on mastery-gated wave tasks while evaluating every frozen stage separately."""
    from .curriculum import (CurriculumManager, load_curriculum, reward_matches,
                             task_hash, task_hash_matches)

    spec = load_curriculum(curriculum_path)
    if resume and initialize_from:
        raise ValueError("use either resume or initialize_from, not both")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "curriculum_state.json"
    saved_state = {}
    if resume:
        resume_state = _resume_artifact(destination, resume, "curriculum_state.json")
        if resume_state.exists():
            saved_state = json.loads(resume_state.read_text(encoding="utf-8"))
    elif initialize_from:
        saved_state = {"stage": _initial_stage_from_checkpoint(initialize_from)}
    if start_stage is not None:
        if not 1 <= start_stage <= len(spec.stages):
            raise ValueError(f"start_stage must be between 1 and {len(spec.stages)}")
        saved_state.update({"stage": start_stage - 1, "streak": 0,
                            "promotion_history": [], "mastered": False,
                            "recovery_stage": None, "recovery_until": 0})
    manager = CurriculumManager(
        spec, seed, stage=int(saved_state.get("stage", 0)),
        streak=int(saved_state.get("streak", 0)),
        mastered=bool(saved_state.get("mastered", False)),
        promotion_history=saved_state.get("promotion_history"))
    recovery_stage = saved_state.get("recovery_stage")
    recovery_stage = int(recovery_stage) if recovery_stage is not None else None
    recovery_until = int(saved_state.get("recovery_until", 0))

    stage_configs = [stage.game_config(spec.base) for stage in spec.stages]

    def make_env(stage_index: int) -> AsteroidsRLEnv:
        stage = spec.stages[stage_index]
        stage_reward = (spec.reward if stage.miss_penalty is None else
                        replace(spec.reward, miss_penalty=stage.miss_penalty))
        return AsteroidsRLEnv(
            stage_configs[stage_index], frame_skip=4, max_decisions=stage.max_decisions,
            no_hit_seconds=stage.no_hit_seconds,
            history_frames=history_frames, history_long_frames=history_long_frames,
            history_long_stride=history_long_stride, reward_config=stage_reward,
            completion=stage.completion)

    prototype = make_env(0)
    expected_layout = observation_layout(prototype)
    expected_layout["task_hash"] = task_hash(spec, stage_configs)
    expected_layout["reward"] = asdict(spec.reward)
    expected_layout["curriculum"] = str(curriculum_path)
    settings = MuZeroSettings(num_simulations=simulations, learning_rate=learning_rate,
                              updates_per_episode=updates_per_episode)
    if resume:
        agent = MuZeroAgent.load(resume, seed=seed)
    elif initialize_from:
        agent = initialize_agent_from_policy(
            initialize_from, prototype.observation_size, prototype.num_actions, settings, seed)
        print(json.dumps({
            "initialized_from": str(initialize_from),
            "initialized_stage": manager.stage + 1,
            "old_replay_loaded": False,
            "learning_rate": agent.settings.learning_rate,
            "observation_size": prototype.observation_size,
        }), flush=True)
    else:
        agent = MuZeroAgent(
            prototype.observation_size, prototype.num_actions, settings, seed=seed)
    if (agent.observation_size, agent.num_actions) != (
            prototype.observation_size, prototype.num_actions):
        raise ValueError("checkpoint observation/action shapes do not match the curriculum")
    if resume:
        metadata = json.loads((Path(resume) / "metadata.json").read_text(encoding="utf-8"))
        stored_layout = metadata.get("observation_layout") or {}
        compared = ("version", "ship_features", "asteroid_features", "max_asteroids",
                    "projectile_features", "max_projectiles",
                    "history_frames", "history_long_frames", "history_long_stride",
                    "history_offsets", "mobile", "actions")
        if (any(stored_layout.get(key) != expected_layout.get(key) for key in compared)
                or not task_hash_matches(stored_layout.get("task_hash"), spec, stage_configs)
                or not reward_matches(stored_layout.get("reward"), spec.reward)):
            raise ValueError("checkpoint observation manifest does not match this curriculum")
        agent.settings.num_simulations = simulations
        agent.settings.updates_per_episode = updates_per_episode
        if resume_learning_rate is not None:
            agent.reset_optimizer(resume_learning_rate)

    training_log = destination / "training.jsonl"
    evaluation_log = destination / "evaluation.jsonl"
    if resume:
        removed = (truncate_log_after_checkpoint(training_log, agent.episodes)
                   + truncate_log_after_checkpoint(evaluation_log, agent.episodes))
        if removed:
            print(json.dumps({"discarded_unsaved_log_records": removed}), flush=True)
    # A new run resumed from a dedicated champion starts with that policy protected before
    # the first challenger evaluation. Its replay still comes from the source run below.
    if resume and Path(resume).name == "champion" and not (
            destination / "champion").exists():
        source_state = Path(resume).parent / "champion_state.json"
        if source_state.is_file():
            shutil.copytree(resume, destination / "champion")
            shutil.copy2(source_state, destination / "champion_state.json")
    champion = ChampionTracker(
        destination, spec.retention_completion, patience=champion_patience,
        retention_floor=spec.retention_floor,
        initial_learning_rate=agent.settings.learning_rate,
        rollback_lr_factor=rollback_lr_factor,
        minimum_learning_rate=minimum_learning_rate,
        promotion_completion=spec.promotion_completion,
        accuracy_targets=tuple(
            spec.promotion_accuracy if stage.promotion_accuracy is None
            else stage.promotion_accuracy
            for stage in spec.stages))
    discovered_champion = champion.bootstrap(evaluation_log)
    if resume:
        if resume_learning_rate is not None:
            champion.state["learning_rate"] = agent.settings.learning_rate
            champion.save()
        print(json.dumps({
            "resumed_latest_episode": agent.episodes,
            "champion_episode": champion.state.get("episode"),
            "discovered_champion": discovered_champion,
            "optimizer_preserved": resume_learning_rate is None,
            "replay_preserved": True,
            "learning_rate": agent.settings.learning_rate,
        }), flush=True)

    replay = ReplayBuffer(agent.settings.replay_capacity, seed)
    replay_path = destination / "replay.npz"
    if resume:
        replay_source = _resume_artifact(destination, resume, "replay.npz")
        episode_stages = {}
        if training_log.is_file():
            for line in training_log.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    episode_stages[int(item["episode"])] = int(item.get("curriculum_stage", 0))
                except (ValueError, KeyError, TypeError):
                    continue
        print(json.dumps({"resumed_replay_transitions": replay.load(
            replay_source, observation_size=prototype.observation_size,
            num_actions=prototype.num_actions, episode_stages=episode_stages)}))
    final_episode = agent.episodes + episodes
    episode = agent.episodes
    pending: list[dict] = []

    def sample_training_stage() -> int:
        focus = recovery_stage if recovery_stage is not None and episode < recovery_until else None
        return manager.sample_stage(focus_stage=focus)

    worker_stages = [sample_training_stage() for _ in range(max(1, parallel_envs))]
    workers = [make_env(index) for index in worker_stages]
    reserved = max(spec.evaluation_episodes, spec.retention_evaluation_episodes)
    observations = [worker.reset(training_seed(
        seed, episode + i, eval_seed, reserved))[0]
                    for i, worker in enumerate(workers)]
    trajectories: list[list[Transition]] = [[] for _ in workers]
    active = [True] * len(workers)
    assigned = episode + len(workers)
    layout = expected_layout

    def evaluate_stages() -> list[dict]:
        """Evaluate only learned stages; scoring unseen future mechanics wastes most runtime."""
        results = []
        for index, stage in enumerate(spec.stages[:manager.stage + 1]):
            count = (spec.evaluation_episodes if index == manager.stage
                     else spec.retention_evaluation_episodes)
            seeds = list(range(eval_seed, eval_seed + count))
            env = make_env(index)
            report = evaluate_policy(
                env,
                lambda obs, e=env: agent.search(
                    obs, explore=False, invalid_actions=e.inert_actions)[0],
                seeds)
            results.append({"stage": index, "name": stage.name, **report["aggregate"]})
        return results

    while episode < final_episode:
        shared_mask = tuple(all(flags) for flags in zip(*(w.inert_actions for w in workers)))
        actions, policies, search_values = agent.search_batch(
            np.stack(observations), explore=True, invalid_actions=shared_mask)
        for index, worker in enumerate(workers):
            action = int(actions[index])
            next_observation, reward, terminated, truncated, info = worker.step(action)
            done = terminated or truncated
            trajectories[index].append(Transition(
                observations[index], action, reward, policies[index], next_observation, done,
                search_value=float(search_values[index])))
            observations[index] = next_observation
            if not done:
                continue

            trajectory, trajectories[index] = trajectories[index], []
            completed_stage = worker_stages[index]
            next_stage = sample_training_stage()
            worker_stages[index] = next_stage
            workers[index] = make_env(next_stage)
            observations[index] = workers[index].reset(
                training_seed(seed, assigned, eval_seed, reserved))[0]
            assigned += 1
            if not active[index]:
                continue
            episode += 1
            finish_episode(trajectory, agent.settings.discount, n_step=agent.settings.n_step,
                           episode_id=episode,
                           successful=bool(info["episode_metrics"].get("completed_stage")),
                           stage=completed_stage)
            replay.extend(trajectory)
            losses = {}
            for _ in range(agent.settings.updates_per_episode):
                if len(replay) >= agent.settings.batch_size:
                    losses = agent.train_batch(
                        replay.sample(
                            agent.settings.batch_size, agent.settings.unroll_steps,
                            current_stage=completed_stage))
            record = {"episode": episode, "curriculum_stage": completed_stage,
                      "stage_name": spec.stages[completed_stage].name,
                      **info["episode_metrics"], **losses}
            with training_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            pending.append(record)
            if log_every and (len(pending) >= log_every or episode == final_episode):
                print(format_summary(summarize_episodes(pending)), flush=True)
                pending = []

            should_eval = eval_every and (episode % eval_every == 0 or episode == final_episode)
            evaluation = None
            if should_eval:
                evaluated_stage = manager.stage
                results = evaluate_stages()
                promoted = manager.consider_promotion(results)
                evaluation = {"episode": episode, "training_stage": evaluated_stage,
                              "next_training_stage": manager.stage,
                              "promotion_streak": manager.streak, "promoted": promoted,
                              "promotion_completion_target": spec.promotion_completion,
                              "stages": results}
                current = results[evaluated_stage]
                stage_spec = spec.stages[evaluated_stage]
                accuracy_target = (spec.promotion_accuracy
                                   if stage_spec.promotion_accuracy is None
                                   else stage_spec.promotion_accuracy)
                evaluation["promotion_accuracy_target"] = accuracy_target
                print(f"  curriculum eval @ {episode}: stage {evaluated_stage + 1} "
                      f"({stage_spec.name}) "
                      f"completion {current['completion_rate']:.1%}, "
                      f"accuracy {current['mean_accuracy']:.3f} "
                      f"(target {accuracy_target:.3f})" +
                      (f" - PROMOTED to stage {manager.stage + 1}" if promoted else ""),
                      flush=True)

            should_checkpoint = (episode % checkpoint_every == 0 or episode == final_episode
                                 or should_eval)
            checkpoint_path = destination / f"checkpoint_{episode:06d}"
            if should_checkpoint:
                agent.save(checkpoint_path, episodes=episode,
                           observation_layout=layout)
                replay.save(replay_path)
                champion_action = "none"
                if evaluation is not None:
                    champion_action = champion.consider(
                        evaluation, checkpoint_path, allow_recovery=evaluated_stage > 0)
                    forgotten = champion.forgotten_stage(evaluation)
                    if forgotten is None:
                        recovery_stage = None
                        recovery_until = 0
                    elif champion_action == "recover":
                        recovery_stage = forgotten
                        recovery_until = episode + max(eval_every, 1) * champion.patience
                    if champion_action in {"restore", "recover"}:
                        restore_agent_from_champion(
                            agent, champion.path,
                            float(champion.state.get(
                                "learning_rate", agent.settings.learning_rate)), seed)
                    evaluation.update({
                        "champion_action": champion_action,
                        "champion_episode": champion.state.get("episode"),
                        "evaluations_since_improvement": champion.state.get(
                            "evaluations_since_improvement", 0),
                        "rollback_count": champion.state.get("rollbacks", 0),
                        "recovery_count": champion.state.get("recoveries", 0),
                        "restoration_count": champion.state.get("restorations", 0),
                        "retention_failures": champion.state.get("retention_failures", 0),
                        "recovery_stage": recovery_stage,
                        "recovery_until": recovery_until,
                        "learning_rate": champion.state.get("learning_rate"),
                    })
                    if champion_action == "recover":
                        print(f"  champion restored; mixed recovery for stage "
                              f"{recovery_stage + 1} through episode {recovery_until}; "
                              f"replay preserved", flush=True)
                    elif champion_action == "restore":
                        print("  challenger plateau: champion weights restored; replay "
                              "preserved", flush=True)
                    with evaluation_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(evaluation) + "\n")
                state_text = json.dumps({"stage": manager.stage,
                                         "streak": manager.streak,
                                         "promotion_history": manager.promotion_history,
                                         "mastered": manager.mastered,
                                         "recovery_stage": recovery_stage,
                                         "recovery_until": recovery_until}, indent=2) + "\n"
                state_path.write_text(state_text, encoding="utf-8")
                # Unlike replay (hundreds of MB), this tiny snapshot belongs in each
                # checkpoint so an explicitly selected older checkpoint restores its stage.
                (checkpoint_path / "curriculum_state.json").write_text(
                    state_text, encoding="utf-8")
                removed = prune_checkpoints(destination, keep_checkpoints)
                if removed:
                    print(f"  checkpoint retention: removed {len(removed)} old checkpoint(s)",
                          flush=True)
                if should_eval:
                    print(f"  champion: episode {champion.state.get('episode')} | "
                          f"action {champion_action} | "
                          f"retention failures {champion.state.get('retention_failures', 0)}"
                          f"/{champion.patience} | latest continues",
                          flush=True)
                    print(f"  watch champion: ./run.sh preview {destination}", flush=True)
                if stop_when_mastered and manager.mastered:
                    print("  curriculum mastered: every stage passed its held-out gate",
                          flush=True)
                    return checkpoint_path
            if episode + sum(active) > final_episode:
                active[index] = False
    return destination / f"checkpoint_{final_episode:06d}"


def _maybe_evaluate(episode: int, final_episode: int, eval_every: int, agent,
                    evaluation_env, evaluation_seeds: list[int], evaluation_log: Path) -> None:
    """Score the current parameters on the held-out seed set with exploration disabled."""
    if not eval_every:
        return
    if episode % eval_every and episode != final_episode:
        return
    report = evaluate_policy(
        evaluation_env,
        lambda observation: agent.search(observation, explore=False)[0],
        evaluation_seeds,
    )
    aggregate = report["aggregate"]
    with evaluation_log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"episode": episode, **aggregate}) + "\n")
    print(
        f"  eval @ {episode} on {len(evaluation_seeds)} held-out seeds: "
        f"survival avg {aggregate['mean_survival_time']:.2f}s "
        f"best {aggregate['max_survival_time']:.2f}s "
        f"worst {aggregate['min_survival_time']:.2f}s | "
        f"kills avg {aggregate['mean_asteroids_destroyed']:.2f} "
        f"best {aggregate['max_asteroids_destroyed']:.0f} | "
        f"accuracy avg {aggregate['mean_accuracy']:.3f}",
        flush=True,
    )


def evaluate(checkpoint: str | Path, config_path: str | Path, output_path: str | Path, *,
             episodes: int, seed: int, max_decisions: int, asteroid_reward: float = 0.1,
             shot_penalty: float = 0.0, history_frames: int = 0,
             history_long_frames: int = 0, history_long_stride: int = 8) -> dict:
    config = load_config(config_path)
    env = AsteroidsRLEnv(
        config, frame_skip=4, max_decisions=max_decisions, asteroid_reward=asteroid_reward,
        shot_penalty=shot_penalty, history_frames=history_frames,
        history_long_frames=history_long_frames, history_long_stride=history_long_stride)
    agent = MuZeroAgent.load(checkpoint, seed=seed)
    if (agent.observation_size, agent.num_actions) != (env.observation_size, env.num_actions):
        raise ValueError("checkpoint observation/action shapes do not match this configuration")
    report = evaluate_policy(
        env, lambda observation: agent.search(observation, explore=False)[0],
        list(range(seed, seed + episodes)),
    )
    save_evaluation(report, output_path)
    return report


def evaluate_baseline(config_path: str | Path, output_path: str | Path, *, episodes: int,
                      seed: int, max_decisions: int, asteroid_reward: float = 0.1,
                      shot_penalty: float = 0.0, history_frames: int = 0,
             history_long_frames: int = 0, history_long_stride: int = 8) -> dict:
    config = load_config(config_path)
    env = AsteroidsRLEnv(
        config, frame_skip=4, max_decisions=max_decisions, asteroid_reward=asteroid_reward,
        shot_penalty=shot_penalty, history_frames=history_frames,
        history_long_frames=history_long_frames, history_long_stride=history_long_stride)
    controller = ClosestAsteroidController()

    def policy(observation) -> int:
        del observation
        assert env.state is not None
        action = controller.action(env.state, env.agent_id)
        return env.actions.index(action)

    report = evaluate_policy(env, policy, list(range(seed, seed + episodes)))
    report["controller"] = "closest_asteroid_no_prediction"
    save_evaluation(report, output_path)
    return report
