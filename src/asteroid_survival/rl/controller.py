from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from ..actions import Action
from ..config import GameConfig
from ..state import WorldSnapshot
from .environment import (ASTEROID_FEATURES, GLOBAL_FEATURES, MAX_PROJECTILES,
                          MOBILE_ACTIONS, PROJECTILE_FEATURES, SHIP_FEATURES,
                          STATIONARY_ACTIONS, TEAMMATE_FEATURES,
                          encode_observation, history_offsets, ship_feature_count)
from .muzero import MuZeroAgent


class MuZeroController:
    """Run a saved MuZero checkpoint at the same four-frame cadence used in training."""

    def __init__(self, config: GameConfig, checkpoint: str | Path, *, frame_skip: int = 4,
                 max_decisions: int = 900, seed: int = 0, history_long_frames: int = 0,
                 history_long_stride: int = 8):
        self.config = config
        self.frame_skip = frame_skip
        self.max_decisions = max_decisions
        self.max_asteroids = config.asteroid.active_cap
        self.actions = MOBILE_ACTIONS if config.ship.mobile else STATIONARY_ACTIONS
        self.agent = MuZeroAgent.load(checkpoint, seed=seed)
        layout = self._stored_layout(checkpoint)
        history_long_frames = int(layout.get("history_long_frames", history_long_frames))
        history_long_stride = int(layout.get("history_long_stride", history_long_stride))
        stored_dense = layout.get("history_frames")
        self.max_projectiles = int(layout.get("max_projectiles", 0))
        self.max_teammates = int(layout.get("max_teammates", 0))
        self.global_features = int(layout.get("global_features", 0)) > 0
        if layout.get("version", 1) >= 2:
            if int(layout.get("max_asteroids", -1)) != self.max_asteroids:
                raise ValueError("checkpoint asteroid slot count does not match this game")
            if bool(layout.get("mobile")) != config.ship.mobile:
                raise ValueError("checkpoint movement setting does not match this game")
            if layout.get("actions") != [action.name for action in self.actions]:
                raise ValueError("checkpoint action mapping does not match this game")
            self.history_frames = int(stored_dense or 0)
            expected = ship_feature_count(config) + self.max_asteroids * (
                ASTEROID_FEATURES + 2 * (self.history_frames + history_long_frames)) + (
                    self.max_projectiles * PROJECTILE_FEATURES)
            if expected != self.agent.observation_size:
                raise ValueError("checkpoint observation manifest does not match its parameters")
        else:
        # Derive the history length the checkpoint was trained with from its observation
        # size, so a history-trained model plays without having to be told how it was built.
            inferred = None
            # Manifest-less checkpoints created by current code contain the standard
            # projectile block; truly legacy checkpoints contain none. Try both layouts.
            for projectile_slots in (MAX_PROJECTILES, 0):
                asteroid_total = (self.agent.observation_size - SHIP_FEATURES
                                  - projectile_slots * PROJECTILE_FEATURES)
                width, remainder = divmod(asteroid_total, self.max_asteroids)
                total_slots, odd = divmod(width - ASTEROID_FEATURES, 2)
                if not remainder and not odd and total_slots >= history_long_frames:
                    inferred = (projectile_slots, total_slots)
                    break
            if inferred is None:
                raise ValueError(
                    f"checkpoint observation size {self.agent.observation_size} does not fit "
                    "this configuration; it was trained on a different observation layout")
            self.max_projectiles, total_slots = inferred
            self.history_frames = total_slots - history_long_frames
        self.history_slots = history_offsets(
            self.history_frames, history_long_frames, history_long_stride)
        self._history_depth = (max(self.history_slots) + 1) if self.history_slots else 0
        self._history: dict[int, deque] = {}
        if self.agent.num_actions != len(self.actions):
            raise ValueError("checkpoint action shape does not match the play configuration")
        self._action = Action.NOOP
        self._next_decision_step = 0

    @staticmethod
    def _stored_layout(checkpoint: str | Path) -> dict:
        """Read the history layout the checkpoint was trained with, if it recorded one."""
        try:
            metadata = json.loads(
                (Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        layout = metadata.get("observation_layout") or {}
        return layout if isinstance(layout, dict) else {}

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        ship = next(ship for ship in snapshot.ships if ship.id == ship_id)
        if not ship.alive:
            return Action.NOOP
        if snapshot.step >= self._next_decision_step:
            observation = encode_observation(
                snapshot, ship_id, self.config, self.max_decisions, self.max_asteroids,
                self.frame_skip, self._history, self.history_frames, self.history_slots,
                self.max_projectiles)
            self._record_history(snapshot)
            action_index, _, _ = self.agent.search(observation, explore=False)
            self._action = self.actions[action_index]
            self._next_decision_step = snapshot.step + self.frame_skip
        return self._action

    def _record_history(self, snapshot: WorldSnapshot) -> None:
        """Track past positions at the decision cadence used during training."""
        if not self._history_depth:
            return
        live = set()
        for asteroid in snapshot.asteroids:
            live.add(asteroid.id)
            track = self._history.get(asteroid.id)
            if track is None:
                track = deque(maxlen=self._history_depth)
                self._history[asteroid.id] = track
            track.appendleft((asteroid.x, asteroid.y))
        for asteroid_id in [key for key in self._history if key not in live]:
            del self._history[asteroid_id]


class PPOPlayController:
    """Run a PPO checkpoint against a live shared-arena snapshot."""

    def __init__(self, config: GameConfig, checkpoint: str | Path, *, frame_skip: int = 4,
                 max_decisions: int = 900, device: str = "auto"):
        from .ppo import PPOController

        self.config = config
        self.frame_skip = frame_skip
        self.max_decisions = max_decisions
        self.max_asteroids = config.asteroid.active_cap
        self.actions = MOBILE_ACTIONS if config.ship.mobile else STATIONARY_ACTIONS
        self.policy = PPOController(checkpoint, device=device)
        metadata = self.policy.metadata
        layout = metadata.get("observation_layout") or {}
        if int(layout.get("version", 1)) < 2:
            raise ValueError("live showdown requires a PPO checkpoint with an observation manifest")
        if int(layout.get("max_asteroids", -1)) != self.max_asteroids:
            raise ValueError("checkpoint asteroid slot count does not match this game")
        if bool(layout.get("mobile")) != config.ship.mobile:
            raise ValueError("checkpoint movement setting does not match this game")
        if layout.get("actions") != [action.name for action in self.actions]:
            raise ValueError("checkpoint action mapping does not match this game")

        self.history_frames = int(layout.get("history_frames", 0))
        history_long_frames = int(layout.get("history_long_frames", 0))
        history_long_stride = int(layout.get("history_long_stride", 8))
        self.max_projectiles = int(layout.get("max_projectiles", 0))
        # Both of these are read below and in `action`, but were never assigned here, so a
        # live showdown against any checkpoint carrying them raised AttributeError before it
        # could draw a frame. Every v5 checkpoint carries the global block.
        self.max_teammates = int(layout.get("max_teammates", 0))
        self.global_features = int(layout.get("global_features", 0)) > 0
        self.history_slots = history_offsets(
            self.history_frames, history_long_frames, history_long_stride)
        self._history_depth = (max(self.history_slots) + 1) if self.history_slots else 0
        expected = ship_feature_count(config) + self.max_asteroids * (
            ASTEROID_FEATURES + 2 * len(self.history_slots)) + (
                self.max_projectiles * PROJECTILE_FEATURES) + (
                self.max_teammates * TEAMMATE_FEATURES) + (
                GLOBAL_FEATURES if self.global_features else 0)
        if expected != int(metadata.get("observation_size", -1)):
            raise ValueError("checkpoint observation manifest does not match its parameters")
        if int(metadata.get("num_actions", -1)) != len(self.actions):
            raise ValueError("checkpoint action shape does not match the play configuration")
        self._history: dict[int, deque] = {}
        self._action = Action.NOOP
        self._next_decision_step = 0

    def reset(self) -> None:
        self.policy.reset()
        self._history.clear()
        self._action = Action.NOOP
        self._next_decision_step = 0

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        ship = next(ship for ship in snapshot.ships if ship.id == ship_id)
        if not ship.alive:
            return Action.NOOP
        if snapshot.step >= self._next_decision_step:
            observation = encode_observation(
                snapshot, ship_id, self.config, self.max_decisions, self.max_asteroids,
                self.frame_skip, self._history, self.history_frames, self.history_slots,
                self.max_projectiles, self.max_teammates, False,
                global_features=self.global_features)
            self._record_history(snapshot)
            self._action = self.actions[self.policy(observation)]
            self._next_decision_step = snapshot.step + self.frame_skip
        return self._action

    def _record_history(self, snapshot: WorldSnapshot) -> None:
        if not self._history_depth:
            return
        live = set()
        for asteroid in snapshot.asteroids:
            live.add(asteroid.id)
            track = self._history.get(asteroid.id)
            if track is None:
                track = deque(maxlen=self._history_depth)
                self._history[asteroid.id] = track
            track.appendleft((asteroid.x, asteroid.y))
        for asteroid_id in [key for key in self._history if key not in live]:
            del self._history[asteroid_id]
