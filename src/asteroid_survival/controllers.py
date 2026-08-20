from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from .actions import Action
from .math2d import Vec2, wrapped_delta
from .state import ShipSnapshot, WorldSnapshot


class Controller(Protocol):
    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action: ...


class RandomController:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        return Action(self._rng.randrange(len(Action)))


class HeuristicController:
    """Small deterministic threat-aware baseline, intentionally not optimal."""

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        ship = next(s for s in snapshot.ships if s.id == ship_id)
        if not ship.alive or not snapshot.asteroids:
            return Action.NOOP
        target = min(snapshot.asteroids, key=lambda a: wrapped_delta(
            Vec2(ship.x, ship.y), Vec2(a.x, a.y), snapshot.width, snapshot.height).length())
        delta = wrapped_delta(Vec2(ship.x, ship.y), Vec2(target.x, target.y), snapshot.width, snapshot.height)
        desired = math.atan2(delta.y, delta.x)
        error = (desired - ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = -1 if error < -0.10 else (1 if error > 0.10 else 0)
        closing = Vec2(target.vx - ship.vx, target.vy - ship.vy).dot(delta.normalized()) < 0
        danger = delta.length() < 175 and closing
        fire = abs(error) < 0.28
        thrust = danger or delta.length() > 280
        return action_from_components(turn, thrust, fire)


class ClosestAsteroidController:
    """Aim at the nearest asteroid's current position, without prediction."""

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        ship = next(s for s in snapshot.ships if s.id == ship_id)
        if not ship.alive or not snapshot.asteroids:
            return Action.NOOP
        origin = Vec2(ship.x, ship.y)
        target = min(
            snapshot.asteroids,
            key=lambda asteroid: wrapped_delta(
                origin, Vec2(asteroid.x, asteroid.y), snapshot.width, snapshot.height).length(),
        )
        delta = wrapped_delta(
            origin, Vec2(target.x, target.y), snapshot.width, snapshot.height)
        desired = math.atan2(delta.y, delta.x)
        error = (desired - ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = -1 if error < -0.025 else (1 if error > 0.025 else 0)
        fire = abs(error) < 0.04
        if turn and abs(error) < 0.25:
            if turn < 0:
                return Action.FINE_LEFT_FIRE if fire else Action.FINE_LEFT
            return Action.FINE_RIGHT_FIRE if fire else Action.FINE_RIGHT
        return action_from_components(turn, False, fire)


def action_from_components(turn: int, thrust: bool, fire: bool) -> Action:
    table = {
        (0, False, False): Action.NOOP, (-1, False, False): Action.LEFT,
        (1, False, False): Action.RIGHT, (0, True, False): Action.THRUST,
        (-1, True, False): Action.LEFT_THRUST, (1, True, False): Action.RIGHT_THRUST,
        (0, False, True): Action.FIRE, (-1, False, True): Action.LEFT_FIRE,
        (1, False, True): Action.RIGHT_FIRE, (0, True, True): Action.THRUST_FIRE,
        (-1, True, True): Action.LEFT_THRUST_FIRE, (1, True, True): Action.RIGHT_THRUST_FIRE,
    }
    return table[(turn, thrust, fire)]


@dataclass(frozen=True)
class KeyProfile:
    left: int
    right: int
    thrust: int
    fire: int
    alternate_left: int | None = None
    alternate_right: int | None = None
    alternate_thrust: int | None = None


def human_action(keys, profile: KeyProfile) -> Action:
    def pressed(primary: int, alternate: int | None = None) -> bool:
        return bool(keys[primary]) or (alternate is not None and bool(keys[alternate]))

    left = pressed(profile.left, profile.alternate_left)
    right = pressed(profile.right, profile.alternate_right)
    thrust = pressed(profile.thrust, profile.alternate_thrust)
    turn = int(right) - int(left)
    return action_from_components(turn, thrust, bool(keys[profile.fire]))


def gamepad_action(joystick) -> Action:
    axis = joystick.get_axis(0)
    turn = -1 if axis < -0.35 else (1 if axis > 0.35 else 0)
    thrust = joystick.get_button(0) or joystick.get_axis(5) > 0.35
    fire = joystick.get_button(1) or joystick.get_button(2)
    return action_from_components(turn, bool(thrust), bool(fire))
