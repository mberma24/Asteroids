"""Small PPO research utilities kept independent of the optional MuZero/JAX stack."""
from __future__ import annotations

import json
import shutil
import statistics

import numpy as np
from pathlib import Path

from .environment import (ASTEROID_FEATURES, GLOBAL_FEATURES, PROJECTILE_FEATURES,
                          AsteroidsRLEnv, ship_feature_count)


def training_seed(base: int, index: int, reserved_start: int, reserved_count: int) -> int:
    seed = base + index
    return seed + reserved_count if reserved_count > 0 and seed >= reserved_start else seed


def truncate_log(path: str | Path, episode: int) -> int:
    source = Path(path)
    if not source.is_file():
        return 0
    lines = source.read_text(encoding="utf-8").splitlines()
    kept = []
    for line in lines:
        try:
            if int(json.loads(line)["episode"]) <= episode:
                kept.append(line)
        except (ValueError, KeyError, TypeError):
            continue
    removed = len(lines) - len(kept)
    if removed:
        source.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return removed


def observation_layout(env: AsteroidsRLEnv) -> dict:
    return {
        "version": 5 if env.global_features else 4,
        "ship_features": ship_feature_count(env.config),
        "asteroid_features": ASTEROID_FEATURES, "max_asteroids": env.max_asteroids,
        "projectile_features": PROJECTILE_FEATURES,
        "max_projectiles": env.max_projectiles,
        "history_frames": env.history_frames,
        "history_long_frames": env.history_long_frames,
        "history_long_stride": env.history_long_stride,
        "history_offsets": env.history_slots, "mobile": env.config.ship.mobile,
        "max_teammates": env.max_teammates,
        "global_features": GLOBAL_FEATURES if env.global_features else 0,
        "reveal_progress": env.reveal_progress,
        "actions": [action.name for action in env.actions],
    }


class SnapshotPolicy:
    """The learner's own policy, as seen from inside an environment worker.

    Companion ships fly the same model the learner does, but environments run in separate
    processes, so the live network cannot be shared with them. The trainer drops a snapshot
    beside the run and this reloads it whenever the file changes, so a companion is a
    slightly stale copy of the learner rather than a scripted bystander -- ordinary
    self-play practice, and far cheaper than keeping a policy in every worker.

    Before the first snapshot exists the companions simply hold station, which is a
    harmless opening rather than a special case to handle.
    """

    def __init__(self, path: str | Path, recurrent: bool = False):
        self.path = Path(path)
        self.recurrent = recurrent
        self._model = None
        self._stamp: float | None = None

    def __call__(self, observation) -> int:
        model = self._current()
        if model is None:
            return 0
        action, _ = model.predict(observation, deterministic=False)
        return int(np.asarray(action).item())

    def _current(self):
        if not self.path.is_file():
            return None
        stamp = self.path.stat().st_mtime
        if self._model is None or stamp != self._stamp:
            if self.recurrent:
                from sb3_contrib import RecurrentPPO as algorithm
            else:
                from stable_baselines3 import PPO as algorithm
            try:
                self._model = algorithm.load(self.path, device="cpu")
            except (OSError, ValueError, EOFError):
                return self._model      # a snapshot caught mid-write; use the last good one
            self._stamp = stamp
        return self._model


def format_episode_block(records: list[dict]) -> str:
    def mean(name: str) -> float:
        values = [float(record[name]) for record in records if name in record]
        return statistics.fmean(values) if values else 0.0

    completion = statistics.fmean(bool(record.get("completed_stage")) for record in records)
    return (f"episodes {records[0]['episode']}-{records[-1]['episode']} ({len(records)})"
            f" | survival avg {mean('survival_time'):.3f}s"
            f" | completion {completion:.1%}"
            f" | asteroids destroyed avg {mean('asteroids_destroyed'):.3f}"
            f" | accuracy avg {mean('accuracy'):.3f}"
            f" | reward avg {mean('reward'):.3f}")


def _score(record: dict, completion_target: float,
           accuracy_targets: tuple[float, ...], clear_target: float = 0.0) -> tuple:
    stages = record.get("stages") or []
    if not stages:
        return (-1, 0, 0.0, 0.0)
    index = min(int(record.get("training_stage", 0)), len(stages) - 1)
    stage = stages[index]
    completion = float(stage.get("completion_rate", 0.0))
    accuracy = float(stage.get("mean_accuracy", 0.0))
    accuracy_target = accuracy_targets[index] if index < len(accuracy_targets) else 0.05
    completion_ratio = completion / max(completion_target, 1e-9)
    accuracy_ratio = accuracy / max(accuracy_target, 1e-9)
    if clear_target > 0:
        clear_ratio = float(stage.get("clear_rate", 0.0)) / clear_target
        return (index, int(completion_ratio >= 1 and clear_ratio >= 1
                           and accuracy_ratio >= 1),
                min(completion_ratio, clear_ratio, accuracy_ratio),
                completion_ratio + clear_ratio + accuracy_ratio)
    return (index, int(completion_ratio >= 1 and accuracy_ratio >= 1),
            min(completion_ratio, accuracy_ratio), completion_ratio + accuracy_ratio)


SMOOTHING_WINDOW = 3
"""Evaluations averaged before the champion is allowed to change.

A single held-out evaluation is a small sample: at 32 episodes and a true completion rate
near 60%, its standard error is about 8 points, so readings swing +-17 points on noise
alone. Crowning a champion on the best single reading is therefore max-selection bias, and
it is not hypothetical -- a champion was once installed on a reading of 84.4% whose true
level, re-measured on 96 episodes, was 57.3%. Nothing could beat that phantom, so the
plateau logic fired forever, the learning rate collapsed to its floor, and the run was
repeatedly restored to a snapshot no better than what it already had.

Averaging a few consecutive evaluations makes a fluke worth only 1/N of the signal.
"""


class PPOChampionTracker:
    """Protect the best held-out PPO without importing or coupling to MuZero."""

    def __init__(self, run: str | Path, retention_completion: float, *, patience: int,
                 retention_floor: float = 0.50,
                 learning_rate: float, minimum_learning_rate: float,
                 promotion_completion: float, accuracy_targets: tuple[float, ...],
                 clear_target: float = 0.0):
        self.run = Path(run)
        self.path = self.run / "champion"
        self.state_path = self.run / "champion_state.json"
        self.retention_completion = float(retention_completion)
        self.retention_floor = float(retention_floor)
        self.patience = int(patience)
        self.initial_learning_rate = float(learning_rate)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.promotion_completion = float(promotion_completion)
        self.accuracy_targets = accuracy_targets
        self.clear_target = float(clear_target)
        self.state: dict = {}

    def _eligible(self, record: dict) -> bool:
        from .curriculum import retention_holds

        stages = record.get("stages") or []
        current = min(int(record.get("training_stage", 0)), max(0, len(stages) - 1))
        return retention_holds(stages[:current],
                               retention_completion=self.retention_completion,
                               retention_floor=self.retention_floor)

    def _record_score(self, record: dict) -> tuple:
        return _score(record, self.promotion_completion, self.accuracy_targets,
                      self.clear_target)

    def _stage_of(self, record: dict) -> int:
        stages = record.get("stages") or []
        return min(int(record.get("training_stage", 0)), max(0, len(stages) - 1))

    def _observe(self, record: dict) -> float:
        """Add this evaluation to the rolling window and return the smoothed completion.

        The window resets on promotion: completion on a new stage says nothing about the
        previous one, and carrying it over would let an easy stage prop up a hard one.
        """
        stage = self._stage_of(record)
        window = self.state.get("window") or []
        clears = self.state.get("clear_window") or []
        if int(self.state.get("window_stage", -1)) != stage:
            window, clears = [], []
        stages = record.get("stages") or [{}]
        window.append(float(stages[stage].get("completion_rate", 0.0)))
        # Clear rate is smoothed over the same window as completion. It used to be read from
        # the single latest evaluation, which is the max-selection bias this window exists to
        # prevent -- applied to the one metric it was not protecting.
        clears.append(float(stages[stage].get("clear_rate", 0.0)))
        window, clears = window[-SMOOTHING_WINDOW:], clears[-SMOOTHING_WINDOW:]
        self.state["window"] = window
        self.state["clear_window"] = clears
        self.state["window_stage"] = stage
        return statistics.fmean(window)

    def _smoothed_clear(self) -> float:
        clears = self.state.get("clear_window") or []
        return statistics.fmean(clears) if clears else 0.0

    def _smoothed_better(self, record: dict, smoothed: float) -> bool:
        """Whether this really is a better policy, rather than a luckier evaluation."""
        stage = self._stage_of(record)
        champion_stage = int(self.state.get("training_stage", -1))
        if stage != champion_stage:
            return stage > champion_stage        # a promotion always installs
        if len(self.state.get("window") or []) < SMOOTHING_WINDOW:
            return False                          # not enough evidence yet
        # Beat the incumbent, not the promotion bar. Requiring `promotion_clear_rate` here
        # froze the champion on any stage the run stalled below it: the policy installed on
        # arrival -- the weakest one that stage will ever see -- stayed champion forever,
        # which is exactly when tracking the best one matters most. Promotion already owns
        # the absolute bar, and `_eligible` separately guards the earlier stages.
        if self._smoothed_clear() < float(self.state.get("clear_estimate", 0.0)):
            return False
        return smoothed > float(self.state.get("completion_estimate", 0.0))

    def _install(self, checkpoint: Path, record: dict) -> None:
        temporary = self.run / ".champion-new"
        previous = self.run / ".champion-old"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        shutil.copytree(checkpoint, temporary)
        if self.path.exists():
            self.path.replace(previous)
        temporary.replace(self.path)
        shutil.rmtree(previous, ignore_errors=True)
        index = min(int(record.get("training_stage", 0)), len(record["stages"]) - 1)
        stage = record["stages"][index]
        old = self.state
        self.state = {
            "episode": int(record["episode"]), "training_stage": index,
            "score": list(self._record_score(record)),
            "completion_rate": float(stage.get("completion_rate", 0.0)),
            "clear_rate": float(stage.get("clear_rate", 0.0)),
            "accuracy": float(stage.get("mean_accuracy", 0.0)),
            "evaluations_since_improvement": 0, "retention_failures": 0,
            "recoveries": int(old.get("recoveries", 0)),
            "restorations": int(old.get("restorations", 0)),
            "rollbacks": 0, "patience": self.patience,
            "learning_rate": float(old.get("learning_rate", self.initial_learning_rate)),
            "initial_learning_rate": self.initial_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
        }
        self.save()

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def bootstrap(self, evaluation_log: Path) -> bool:
        if self.path.is_dir() and self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return False
        candidates = []
        if evaluation_log.is_file():
            for line in evaluation_log.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    checkpoint = self.run / f"checkpoint_{int(record['episode']):06d}"
                except (ValueError, KeyError, TypeError):
                    continue
                if checkpoint.is_dir() and self._eligible(record):
                    candidates.append((record, checkpoint))
        if not candidates:
            return False
        record, checkpoint = max(candidates, key=lambda item: self._record_score(item[0]))
        self._install(checkpoint, record)
        return True

    def forgotten_stage(self, record: dict) -> int | None:
        stages = record.get("stages") or []
        current = min(int(record.get("training_stage", 0)), max(0, len(stages) - 1))
        failures = [(float(stage.get("completion_rate", 0.0)), index)
                    for index, stage in enumerate(stages[:current])
                    if int(stage.get("episodes", 1) or 0) > 0
                    and float(stage.get("completion_rate", 0.0)) < self.retention_completion]
        return min(failures)[1] if failures else None

    def consider(self, record: dict, checkpoint: Path, *, allow_recovery: bool) -> str:
        smoothed = self._observe(record) if self.state else 0.0
        if not self.state:
            self._install(checkpoint, record)
            self._observe(record)
            current = (record.get("stages") or [{}])[self._stage_of(record)]
            self.state["completion_estimate"] = float(current.get("completion_rate", 0.0))
            # Seed the clear bar here too. Left unset it defaults to zero, and the very gate
            # that keeps a champion from being replaced by a worse one never engages.
            self.state["clear_estimate"] = float(current.get("clear_rate", 0.0))
            self.save()
            return "improved"
        if self._eligible(record) and self._smoothed_better(record, smoothed):
            window, window_stage = self.state.get("window"), self.state.get("window_stage")
            clears = self.state.get("clear_window")
            self._install(checkpoint, record)
            self.state["window"], self.state["window_stage"] = window, window_stage
            self.state["clear_window"] = clears
            self.state["completion_estimate"] = smoothed
            self.state["clear_estimate"] = self._smoothed_clear()
            self.save()
            return "improved"
        self.state["evaluations_since_improvement"] = int(
            self.state.get("evaluations_since_improvement", 0)) + 1
        forgotten = self.forgotten_stage(record)
        if forgotten is not None and allow_recovery:
            self.state["retention_failures"] = int(
                self.state.get("retention_failures", 0)) + 1
            if self.state["retention_failures"] >= self.patience:
                self.state["retention_failures"] = 0
                self.state["recoveries"] = int(self.state.get("recoveries", 0)) + 1
                self.state["evaluations_since_improvement"] = 0
                self.save()
                return "recover"
        else:
            self.state["retention_failures"] = 0
        if (allow_recovery and self.state["evaluations_since_improvement"] >= self.patience):
            self.state["evaluations_since_improvement"] = 0
            # PPO is on-policy. Replacing weights from inside an evaluation callback leaves
            # a rollout buffer produced by a different policy and invalidates the update.
            # The protected champion remains a serving artifact; rollback is an explicit
            # new run initialized from champion, never an in-place training action.
            # The estimate stays the champion's own level. Raising it to the best reading
            # ever seen built a bar the champion itself could not clear, so a long plateau
            # made installing a *better* policy progressively harder -- the phantom-champion
            # failure this smoothing was added to prevent, arriving by the opposite route.
        self.save()
        return "continue"


def prune_ppo_checkpoints(run: str | Path, keep: int) -> list[Path]:
    checkpoints = sorted(Path(run).glob("checkpoint_*"))
    target = max(1, keep - 1) if (Path(run) / "champion").is_dir() else keep
    if keep <= 0 or len(checkpoints) <= target:
        return []
    removed = checkpoints[:-target]
    for checkpoint in removed:
        shutil.rmtree(checkpoint)
    return removed
