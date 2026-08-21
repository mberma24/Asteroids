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


class PilotController:
    """Scripted pilot: leads its shots, and flies away from what is about to hit it.

    A stronger reference than `ClosestAsteroidController`, which never thrusts and aims at
    where a rock *is* rather than where it will be. Both failings matter more the faster the
    ladder gets, so greedy stops being a meaningful yardstick exactly where the interesting
    comparisons start. Nothing here is learned: the same snapshot always yields the same
    action.

    Three ideas, in priority order:

    1. **Threat first.** For every rock, the time of closest approach and the miss distance
       at that moment are closed-form. A rock that will pass within its own radius plus a
       margin, soon, is a threat; the ship steers to the escape heading and burns.
    2. **Lead the target otherwise.** Solving |d + v t| = bullet_speed * t gives the time a
       shot needs, and therefore where to aim. Firing at the current position misses any
       crossing rock, which is most of them once patterns turn on.
    3. **Only shoot what is worth shooting**: inside bullet range, and roughly lined up.
    """

    def __init__(self, *, bullet_speed: float = 540.0, bullet_lifetime: float = 1.45,
                 danger_seconds: float = 1.0, margin: float = 40.0):
        self.bullet_speed = bullet_speed
        self.bullet_range = bullet_speed * bullet_lifetime
        self.danger_seconds = danger_seconds
        self.margin = margin

    def _threat(self, delta: Vec2, relative: Vec2, radius: float,
                ship_radius: float) -> tuple[float, float] | None:
        """Time to closest approach and miss distance, or None if it never closes."""
        speed_squared = relative.dot(relative)
        if speed_squared <= 1e-9:
            return None
        time = -delta.dot(relative) / speed_squared
        if time < 0:                                  # already past its nearest point
            return None
        miss = Vec2(delta.x + relative.x * time, delta.y + relative.y * time).length()
        if miss > radius + ship_radius + self.margin:
            return None
        return time, miss

    def _lead(self, delta: Vec2, relative: Vec2) -> float | None:
        """Bearing that intercepts, from the positive root of |d + v t| = speed * t."""
        a = relative.dot(relative) - self.bullet_speed ** 2
        b = 2 * delta.dot(relative)
        c = delta.dot(delta)
        if abs(a) < 1e-6:
            time = -c / b if abs(b) > 1e-9 else None
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return None
            root = math.sqrt(discriminant)
            times = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
            time = min(times) if times else None
        if time is None or time <= 0:
            return None
        aim = Vec2(delta.x + relative.x * time, delta.y + relative.y * time)
        return math.atan2(aim.y, aim.x)

    def action(self, snapshot: WorldSnapshot, ship_id: str) -> Action:
        ship = next(s for s in snapshot.ships if s.id == ship_id)
        if not ship.alive or not snapshot.asteroids:
            return Action.NOOP
        origin = Vec2(ship.x, ship.y)
        ship_radius = getattr(ship, "radius", 12.0)

        threats: list[tuple[float, Vec2]] = []
        shots: list[tuple[float, float, float]] = []   # (priority, bearing, distance)
        for asteroid in snapshot.asteroids:
            delta = wrapped_delta(origin, Vec2(asteroid.x, asteroid.y),
                                  snapshot.width, snapshot.height)
            relative = Vec2(asteroid.vx - ship.vx, asteroid.vy - ship.vy)
            found = self._threat(delta, relative, asteroid.radius, ship_radius)
            if found is not None and found[0] < self.danger_seconds:
                threats.append((found[0], delta))
            distance = delta.length()
            if distance <= self.bullet_range:
                bearing = self._lead(delta, Vec2(asteroid.vx, asteroid.vy))
                if bearing is not None:
                    # Prefer whatever will reach the ship soonest, not whatever is nearest:
                    # clearing an inbound rock early is worth more than picking off a rock
                    # that is drifting away, and it is what stops the field filling up.
                    priority = found[0] if found is not None else 4.0 + distance / 400.0
                    shots.append((priority, bearing, distance))
        shots.sort()
        best_bearing = shots[0][1] if shots else None
        best_distance = shots[0][2] if shots else math.inf

        def aligned_shot() -> bool:
            """Fire if anything in range is under the crosshair, whatever we are steering for.

            Waiting for the chosen target to line up wastes every rotation spent escaping;
            firing opportunistically keeps the kill rate up while dodging, which is what the
            greedy baseline gets for free by never manoeuvring at all.
            """
            for _, bearing, distance in shots:
                error = (bearing - ship.angle + math.pi) % (2 * math.pi) - math.pi
                if abs(error) < max(0.05, math.atan2(18.0, max(40.0, distance))):
                    return True
            return False

        if threats:
            # Escape along the sum of the incoming bearings, weighted by urgency, so the
            # ship runs from the crowd rather than from whichever rock happens to be first.
            escape = Vec2(0.0, 0.0)
            for time, delta in threats:
                weight = 1.0 / (time + 0.25)
                length = max(1e-6, delta.length())
                escape = Vec2(escape.x - delta.x / length * weight,
                              escape.y - delta.y / length * weight)
            desired = math.atan2(escape.y, escape.x)
            error = (desired - ship.angle + math.pi) % (2 * math.pi) - math.pi
            turn = -1 if error < -0.12 else (1 if error > 0.12 else 0)
            # Only burn when roughly pointed at the escape heading; thrusting sideways
            # into a converging rock is worse than holding still.
            thrust = abs(error) < 0.6
            return action_from_components(turn, thrust, aligned_shot())

        if best_bearing is None:
            return Action.NOOP
        error = (best_bearing - ship.angle + math.pi) % (2 * math.pi) - math.pi
        turn = -1 if error < -0.05 else (1 if error > 0.05 else 0)
        return action_from_components(turn, False, aligned_shot())


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
