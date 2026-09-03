from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .actions import Action
from .config import AsteroidConfig, GameConfig
from .math2d import Vec2, from_angle, wrap, wrapped_delta, wrapped_distance
from .patterns import trajectory
from .state import (AsteroidSnapshot, GameEvent, ObjectiveSnapshot, ProjectileSnapshot,
                    ShipSnapshot, StepResult, TerminalReason, WorldSnapshot)


@dataclass(slots=True)
class _Ship:
    id: str
    pos: Vec2
    vel: Vec2
    angle: float
    cooldown: float = 0.0
    alive: bool = True


@dataclass(slots=True)
class _Asteroid:
    id: int
    origin: Vec2
    forward: Vec2
    speed: float
    pattern: str
    amplitude: float
    frequency: float
    phase: float
    age: float
    size: int
    pos: Vec2
    vel: Vec2


@dataclass(slots=True)
class _Projectile:
    id: int
    owner_id: str
    pos: Vec2
    vel: Vec2
    age: float = 0.0


ASTEROID_RADII = {1: 13.0, 2: 24.0, 3: 39.0}


@dataclass(frozen=True, slots=True)
class FireConsequence:
    """What a shot taken right now would hit, and what its fragments would then do."""

    time_to_hit: float
    target_id: int
    distance: float
    size: int
    splits: bool
    bearing: float
    worst_clearance: float
    """Closest the worse of the two fragments comes to the ship; negative means a hit."""
    worst_at: float
    """Seconds from now at which that closest approach happens."""
    closing: float
    """Speed the fragment is closing on the ship at that moment."""


class Simulation:
    """Deterministic fixed-step game simulation with no rendering dependencies."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.config.validate()
        self.dt = 1.0 / self.config.arena.fps
        self._rng = random.Random()
        self._seed = 0
        self._ships: list[_Ship] = []
        self._asteroids: list[_Asteroid] = []
        self._projectiles: list[_Projectile] = []
        self._spawn_clock = 0.0
        self._wave = 0
        self._wave_pending = 0
        self._wave_pending_sizes: list[int] = []
        self._wave_timer = 0.0
        self._wave_clear_recorded = True
        self._next_id = 1
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.terminal_reason: TerminalReason | None = None
        self.object_health = self.config.objective.object_health

    def reset(self, seed: int | None = None) -> WorldSnapshot:
        self._seed = 0 if seed is None else seed
        self._rng.seed(self._seed)
        self._asteroids.clear()
        self._projectiles.clear()
        self._ships.clear()
        self._next_id = 1
        self._spawn_clock = 0.0
        self._wave = 0
        self._wave_pending = 0
        self._wave_pending_sizes.clear()
        self._wave_timer = self.config.asteroid.initial_wave_delay
        self._wave_clear_recorded = True
        self.step_count = 0
        self.terminated = self.truncated = False
        self.terminal_reason = None
        self.object_health = self.config.objective.object_health
        center = Vec2(self.config.arena.width / 2, self.config.arena.height / 2)
        count = len(self.config.ships)
        ring = self.config.ship.spawn_radius if count > 1 or self.config.objective.protect else 0.0
        for i, spec in enumerate(self.config.ships):
            angle = -math.pi / 2 + i * 2 * math.pi / count
            self._ships.append(_Ship(spec.id, center + from_angle(angle) * ring, Vec2(), angle))
        self._populate_field()
        return self.snapshot()

    @property
    def spawn_phase(self) -> float:
        """Fraction of the interval-spawn clock that has elapsed.

        This intentionally exposes no mutable simulation state.  RL observations need to
        distinguish "a spawn just happened" from "another rock is imminent", while the
        renderer and the stable :class:`WorldSnapshot` schema do not need that detail.
        """
        interval = max(float(self.config.asteroid.spawn_interval), 1e-9)
        return max(0.0, min(1.0, float(self._spawn_clock) / interval))

    def _populate_field(self) -> None:
        """Place the opening asteroids, keeping every one clear of the ships.

        Positions are drawn from the whole arena rather than the edges, because these are
        meant to be mid-flight rather than newly arrived. A pattern's lateral offset at
        age zero can be as large as its amplitude, so clearance is checked against the
        trajectory's actual starting point rather than the requested origin.
        """
        cfg = self.config.asteroid
        width, height = self.config.arena.width, self.config.arena.height
        wanted = min(cfg.initial_asteroids, cfg.active_cap)
        for _ in range(wanted):
            for _attempt in range(64):
                origin = Vec2(self._rng.uniform(0, width), self._rng.uniform(0, height))
                heading = from_angle(self._rng.uniform(0.0, 2 * math.pi))
                asteroid = self._spawn_asteroid(pos=origin, direction=heading)
                start, velocity = self._start_of(asteroid)
                if self._clear_of_ships(asteroid, start):
                    asteroid.pos, asteroid.vel = start, velocity
                    self._asteroids.append(asteroid)
                    break

    def step(self, actions: Mapping[str, Action | int]) -> StepResult:
        if self.terminated or self.truncated:
            raise RuntimeError("episode is finished; call reset() before stepping again")
        known = {s.id for s in self._ships}
        unknown = set(actions) - known
        if unknown:
            raise ValueError(f"actions contain unknown ship ids: {sorted(unknown)}")
        events: list[GameEvent] = []
        self.step_count += 1
        self._update_ships(actions, events)
        self._update_projectiles(events)
        self._update_asteroids()
        if self.config.asteroid.spawn_mode == "wave":
            self._advance_waves(events)
        else:
            self._advance_interval_spawning(events)
        self._collisions(events)
        self._record_wave_clear(events)
        if self.config.objective.protect and self.object_health <= 0:
            self.terminated = True
            self.terminal_reason = TerminalReason.OBJECT_DESTROYED
        elif not any(s.alive for s in self._ships):
            self.terminated = True
            self.terminal_reason = TerminalReason.ALL_SHIPS_DESTROYED
        elif self.config.objective.max_steps is not None and self.step_count >= self.config.objective.max_steps:
            self.truncated = True
            self.terminal_reason = TerminalReason.STEP_LIMIT
        snap = self.snapshot()
        return StepResult(snap, tuple(events), self.terminated, self.truncated, self.terminal_reason)

    def _update_ships(self, actions: Mapping[str, Action | int], events: list[GameEvent]) -> None:
        c = self.config.ship
        for ship in self._ships:
            if not ship.alive:
                continue
            action = Action(actions.get(ship.id, Action.NOOP))
            ship.angle = (ship.angle + action.turn * c.turn_speed * self.dt) % (2 * math.pi)
            if action.thrust and c.mobile:
                ship.vel = ship.vel + from_angle(ship.angle) * (c.acceleration * self.dt)
            if c.mobile:
                ship.vel = (ship.vel * max(0.0, 1.0 - c.drag * self.dt)).limited(c.max_speed)
                ship.pos = wrap(ship.pos + ship.vel * self.dt, self.config.arena.width, self.config.arena.height)
            ship.cooldown = max(0.0, ship.cooldown - self.dt)
            if action.fire and ship.cooldown <= 0:
                direction = from_angle(ship.angle)
                pos = wrap(ship.pos + direction * (c.radius + 5), self.config.arena.width, self.config.arena.height)
                velocity = ship.vel + direction * self.config.projectile.speed
                projectile = _Projectile(self._new_id(), ship.id, pos, velocity)
                self._projectiles.append(projectile)
                ship.cooldown = c.fire_cooldown
                events.append(GameEvent("projectile_fired", self.step_count, str(projectile.id), ship.id))

    def fire_consequence(self, ship_id: str, *, horizon: float = 1.0
                        ) -> "FireConsequence | None":
        """If this ship fired right now, what would the shot hit and where would the pieces go?

        The policy's dominant cause of death is a fragment of a rock it has just shot: 18 of
        the v12 champion's 21 deaths over 96 seeds, most of them under a second old. The
        decision that kills it is the shot, taken half a second earlier, and at 180 px/s^2
        the ship displaces about 7px in the quarter-second it has once the piece is on its
        way -- so there is nothing to learn about dodging, and everything to learn about not
        taking the shot. Nothing in the observation says what a shot would produce.

        With ``fragment_motion = "inherit"`` that answer is a closed form rather than a
        prediction: a fragment keeps its parent's pattern, swing and phase, its heading is
        the parent's velocity rotated by +-0.45 rad, and its speed is the parent's rescaled by
        size. This walks the shot to its first hit and the two resulting pieces out to
        ``horizon``, against a ship that holds its current velocity.

        Returns ``None`` when no shot is possible (the weapon is on cooldown or the ship is
        dead) or when nothing is hit inside the projectile's lifetime.
        """
        ship = next((s for s in self._ships if s.id == ship_id), None)
        if ship is None or not ship.alive or ship.cooldown > 0.0:
            return None
        c, w, h = self.config.ship, self.config.arena.width, self.config.arena.height
        pc = self.config.projectile
        direction = from_angle(ship.angle)
        origin = wrap(ship.pos + direction * (c.radius + 5), w, h)
        velocity = ship.vel + direction * pc.speed

        # Which rock the shot reaches first. Linear over the projectile's 1.45s lifetime is
        # accurate to ~2px at 0.5s and ~7px at 1.0s on this curriculum (measured 2026-08-26),
        # well inside the 13-39px radii being tested against.
        best_time, best_rock = math.inf, None
        for a in self._asteroids:
            delta = wrapped_delta(origin, a.pos, w, h)
            rvx, rvy = a.vel.x - velocity.x, a.vel.y - velocity.y
            reach = pc.radius + ASTEROID_RADII[a.size]
            speed2 = rvx * rvx + rvy * rvy
            if speed2 < 1e-9:
                continue
            b = delta.x * rvx + delta.y * rvy
            discriminant = b * b - speed2 * (delta.x * delta.x + delta.y * delta.y
                                             - reach * reach)
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            for candidate in sorted(((-b - root) / speed2, (-b + root) / speed2)):
                if 0.0 <= candidate <= pc.lifetime and candidate < best_time:
                    best_time, best_rock = candidate, a
                    break
        if best_rock is None:
            return None

        # Where the parent actually is when the shot lands, on its true curved path.
        hit_pos, hit_vel = trajectory(
            best_rock.origin, best_rock.forward, best_rock.speed, best_rock.pattern,
            best_rock.age + best_time, best_rock.amplitude, best_rock.frequency,
            best_rock.phase)
        hit_pos = wrap(hit_pos, w, h)
        splits = best_rock.size > 1
        # A split that would exceed the active cap is suppressed by the simulator, so a full
        # field makes shooting paradoxically safe. Read the cap the same way `_collisions`
        # does rather than assuming it.
        if len(self._asteroids) >= self._difficulty().active_cap:
            splits = False

        worst_clearance, worst_at, worst_closing = math.inf, horizon, 0.0
        if splits:
            child_size = best_rock.size - 1
            cfg = self.config.asteroid
            child_speed = (best_rock.speed / self._size_multiplier(cfg, best_rock.size)
                           * self._size_multiplier(cfg, child_size))
            base_angle = math.atan2(hit_vel.y, hit_vel.x)
            steps = max(1, int(round(horizon * self.config.arena.fps / 2)))
            interval = horizon / steps
            for sign in (-1, 1):
                forward = from_angle(base_angle + sign * 0.45)
                # A fragment inherits everything but its heading and scale; on a `random`
                # ladder the speed and shape are redrawn instead and this is only indicative.
                for index in range(steps + 1):
                    tau = index * interval
                    child_pos, child_vel = trajectory(
                        hit_pos, forward, child_speed, best_rock.pattern, tau,
                        best_rock.amplitude, best_rock.frequency, best_rock.phase)
                    # The ship holds its velocity, decayed by drag, which is the neutral
                    # assumption: what happens if it does nothing about the piece.
                    future = ship.vel * ((max(0.0, 1.0 - c.drag * self.dt))
                                         ** (tau * self.config.arena.fps))
                    ship_pos = ship.pos + ship.vel * tau if not c.mobile else (
                        ship.pos + (ship.vel + future) * (0.5 * tau))
                    gap = wrapped_delta(wrap(ship_pos, w, h), wrap(child_pos, w, h), w, h)
                    clearance = gap.length() - ASTEROID_RADII[child_size] - c.radius
                    if clearance < worst_clearance:
                        distance = max(gap.length(), 1e-9)
                        worst_clearance, worst_at = clearance, best_time + tau
                        worst_closing = -((child_vel.x - future.x) * gap.x / distance
                                          + (child_vel.y - future.y) * gap.y / distance)
        if worst_clearance is math.inf:
            worst_clearance = math.hypot(w / 2, h / 2)
        offset = wrapped_delta(ship.pos, hit_pos, w, h)
        bearing = math.atan2(offset.y, offset.x) - ship.angle
        return FireConsequence(
            time_to_hit=best_time, target_id=best_rock.id, distance=offset.length(),
            size=best_rock.size, splits=splits, bearing=bearing,
            worst_clearance=worst_clearance, worst_at=worst_at, closing=worst_closing)

    def _update_projectiles(self, events: list[GameEvent]) -> None:
        pc = self.config.projectile
        kept = []
        for p in self._projectiles:
            p.age += self.dt
            p.pos = wrap(p.pos + p.vel * self.dt, self.config.arena.width, self.config.arena.height)
            if p.age < pc.lifetime:
                kept.append(p)
            else:
                events.append(GameEvent("projectile_expired", self.step_count, str(p.id), p.owner_id))
        self._projectiles = kept

    def _update_asteroids(self) -> None:
        for a in self._asteroids:
            a.age += self.dt
            pos, vel = trajectory(a.origin, a.forward, a.speed, a.pattern, a.age,
                                  a.amplitude, a.frequency, a.phase)
            a.pos = wrap(pos, self.config.arena.width, self.config.arena.height)
            a.vel = vel

    def _pattern(self) -> str:
        cfg = self.config.asteroid
        if cfg.motion_mode == "linear":
            return "linear"
        if cfg.motion_mode == "specific":
            return cfg.specific_pattern
        if cfg.linear_probability and self._rng.random() < cfg.linear_probability:
            return "linear"
        return self._rng.choice(cfg.pattern_pool)

    def _difficulty(self):
        return self.config.asteroid.difficulty_at(self.step_count * self.dt)

    def _start_of(self, asteroid: _Asteroid) -> tuple[Vec2, Vec2]:
        """Where an asteroid actually is at age zero, and how fast.

        Not the same as its origin: a pattern's lateral offset at t=0 can be as large as its
        amplitude, so clearance has to be judged on the trajectory's real starting point.
        """
        position, velocity = trajectory(
            asteroid.origin, asteroid.forward, asteroid.speed, asteroid.pattern, 0.0,
            asteroid.amplitude, asteroid.frequency, asteroid.phase)
        return wrap(position, self.config.arena.width, self.config.arena.height), velocity

    def _clear_of_ships(self, asteroid: _Asteroid, position: Vec2) -> bool:
        cfg = self.config.asteroid
        clearance = (ASTEROID_RADII[asteroid.size] + self.config.ship.radius
                     + cfg.spawn_safe_radius + cfg.spawn_safe_seconds * asteroid.speed)
        width, height = self.config.arena.width, self.config.arena.height
        return all(wrapped_distance(position, ship.pos, width, height) > clearance
                   for ship in self._ships if ship.alive)

    def _emit_asteroid(self, events: list[GameEvent], size: int | None = None) -> None:
        cfg = self.config.asteroid
        guarded = cfg.spawn_safe_radius > 0 or cfg.spawn_safe_seconds > 0
        asteroid = self._spawn_asteroid(size=size)
        if guarded:
            # Edge spawns are only safe while the ship is away from the edges, and the arena
            # wraps, so a ship can sit exactly where the next rock appears. Left unguarded
            # this materialises asteroids on top of the ship -- measured overlapping it by
            # 22px -- which is an unavoidable death rather than a missed dodge.
            for _ in range(32):
                position, velocity = self._start_of(asteroid)
                if self._clear_of_ships(asteroid, position):
                    asteroid.pos, asteroid.vel = position, velocity
                    break
                asteroid = self._spawn_asteroid(size=size)
            else:
                # Boxed in on every side: spawn anyway rather than quietly thinning the
                # field, which would make a crowded round easier exactly when it is hardest.
                asteroid.pos, asteroid.vel = self._start_of(asteroid)
        self._asteroids.append(asteroid)
        events.append(
            GameEvent("asteroid_spawned", self.step_count, str(asteroid.id), asteroid.pattern))

    def _advance_interval_spawning(self, events: list[GameEvent]) -> None:
        self._spawn_clock += self.dt
        difficulty = self._difficulty()
        while (self._spawn_clock >= difficulty.spawn_interval
               and len(self._asteroids) < difficulty.active_cap):
            self._spawn_clock -= difficulty.spawn_interval
            self._emit_asteroid(events)

    def _advance_waves(self, events: list[GameEvent]) -> None:
        """Release asteroids in waves once the field has been worn down.

        A wave arrives only when at most ``wave_threshold`` asteroids remain, and its
        members trickle in one at a time so a wave never materialises on top of the ship.
        """
        cfg = self.config.asteroid
        difficulty = self._difficulty()
        self._wave_timer = max(0.0, self._wave_timer - self.dt)

        if self._wave_pending > 0:
            if self._wave_timer <= 0.0 and len(self._asteroids) < difficulty.active_cap:
                size = self._wave_pending_sizes.pop(0) if self._wave_pending_sizes else None
                self._emit_asteroid(events, size=size)
                self._wave_pending -= 1
                self._wave_timer = cfg.wave_spawn_interval
            return

        if len(self._asteroids) > difficulty.wave_threshold:
            self._wave_timer = max(self._wave_timer, cfg.wave_delay)
            return
        if self._wave_timer > 0.0:
            return  # the pause between waves is still running
        self._wave += 1
        self._wave_clear_recorded = False
        if cfg.wave_composition:
            self._wave_pending_sizes = list(cfg.wave_composition)
            self._wave_pending = len(self._wave_pending_sizes)
        else:
            self._wave_pending_sizes = []
            self._wave_pending = cfg.wave_size_for(self._wave)
        events.append(GameEvent("wave_started", self.step_count, str(self._wave)))

    def _record_wave_clear(self, events: list[GameEvent]) -> None:
        """Emit one completion event after the last fragment of an active wave is gone."""
        if (self.config.asteroid.spawn_mode != "wave" or self._wave <= 0
                or self._wave_pending or self._asteroids or self._wave_clear_recorded):
            return
        if any(event.kind in {"ship_destroyed", "object_damaged"} for event in events):
            return  # crashing into the final rock is not a successful clear
        self._wave_clear_recorded = True
        self._wave_timer = max(self._wave_timer, self.config.asteroid.wave_delay)
        events.append(GameEvent("wave_cleared", self.step_count, str(self._wave)))
        target = self.config.objective.max_waves
        if target is not None and self._wave >= target:
            self.truncated = True
            self.terminal_reason = TerminalReason.WAVES_CLEARED

    @staticmethod
    def _size_multiplier(cfg: AsteroidConfig, size: int) -> float:
        return (cfg.small_speed_multiplier if size == 1 else
                cfg.medium_speed_multiplier if size == 2 else 1.0)

    def _spawn_asteroid(self, *, pos: Vec2 | None = None, size: int | None = None,
                        direction: Vec2 | None = None,
                        parent: _Asteroid | None = None) -> _Asteroid:
        cfg, width, height = self.config.asteroid, self.config.arena.width, self.config.arena.height
        fresh_spawn = pos is None
        if pos is None:
            side = self._rng.randrange(4)
            if side == 0:
                pos, target = Vec2(self._rng.uniform(0, width), 0), Vec2(width / 2, height * 0.65)
            elif side == 1:
                pos, target = Vec2(self._rng.uniform(0, width), height), Vec2(width / 2, height * 0.35)
            elif side == 2:
                pos, target = Vec2(0, self._rng.uniform(0, height)), Vec2(width * 0.65, height / 2)
            else:
                pos, target = Vec2(width, self._rng.uniform(0, height)), Vec2(width * 0.35, height / 2)
            direction = (target - pos).normalized()
        direction = direction or Vec2(1, 0)
        # Speed and waviness are read at spawn time, so asteroids created later in an
        # episode are faster and swing wider than the ones that opened it.
        difficulty = self._difficulty()
        if fresh_spawn and cfg.heading_mode == "random":
            direction = from_angle(self._rng.uniform(0.0, 2 * math.pi))
        elif fresh_spawn and (cfg.heading_mode == "spread" or difficulty.spawn_spread > 0.0):
            # Scatter the heading so asteroids do not all converge on the arena centre.
            # Split fragments keep their own fixed offsets, so they are left alone.
            spread = math.radians(difficulty.spawn_spread) / 2.0
            angle = math.atan2(direction.y, direction.x) + self._rng.uniform(-spread, spread)
            direction = from_angle(angle)
        if size is not None:
            spawn_size = size
        elif isinstance(cfg.spawn_size, list):
            # A weighted mixture: repeat a size in the list to make it more likely, which is
            # how a bridge round is "mostly small with the occasional medium".
            spawn_size = self._rng.choice(cfg.spawn_size)
        else:
            spawn_size = cfg.spawn_size or self._rng.choice((1, 2, 3))
        multiplier = self._size_multiplier(cfg, spawn_size)
        if parent is not None and cfg.fragment_motion == "inherit":
            # A fragment's whole motion follows from its parent's, which the player has
            # been watching: same pattern, swing and phase, and the parent's speed rescaled
            # by the size multipliers. Nothing about it is drawn at the moment of the shot.
            speed = parent.speed / self._size_multiplier(cfg, parent.size) * multiplier
            return _Asteroid(self._new_id(), pos, direction.normalized(), speed,
                             parent.pattern, parent.amplitude, parent.frequency,
                             parent.phase, 0.0, spawn_size, pos,
                             direction.normalized() * speed)
        speed = self._rng.uniform(difficulty.min_speed, difficulty.max_speed) * multiplier
        pattern = self._pattern()
        period = self._rng.uniform(difficulty.wavelength_min,
                                   max(difficulty.wavelength_min, difficulty.wavelength_max))
        amplitude_max = max(cfg.amplitude_min, difficulty.amplitude_max)
        amplitude = self._rng.uniform(cfg.amplitude_min, amplitude_max)
        # One scale for all three, so a varied rock reads as coherently sluggish -- slower,
        # swinging less far, and taking longer to do it -- rather than as an incoherent mix
        # of fast travel with a lazy wobble.
        if cfg.variety_probability > 0 and self._rng.random() < cfg.variety_probability:
            scale = self._rng.uniform(min(1.0, cfg.variety_scale_min), 1.0)
            speed *= scale
            amplitude *= scale
            period /= max(1e-6, scale)
        frequency = 2 * math.pi / period
        asteroid = _Asteroid(self._new_id(), pos, direction.normalized(), speed, pattern,
                             amplitude, frequency,
                             self._rng.uniform(0, 2 * math.pi), 0.0, spawn_size, pos,
                             direction.normalized() * speed)
        return asteroid

    def _collisions(self, events: list[GameEvent]) -> None:
        w, h = self.config.arena.width, self.config.arena.height
        removed_a: set[int] = set()
        removed_p: set[int] = set()
        children: list[_Asteroid] = []
        for p in self._projectiles:
            for a in self._asteroids:
                if (a.id in removed_a or wrapped_distance(p.pos, a.pos, w, h)
                        > self.config.projectile.radius + ASTEROID_RADII[a.size]):
                    continue
                removed_p.add(p.id)
                removed_a.add(a.id)
                events.append(GameEvent("asteroid_shot", self.step_count, str(a.id), p.owner_id))
                if a.size > 1:
                    for sign in (-1, 1):
                        if (len(self._asteroids) - len(removed_a) + len(children)
                                >= self._difficulty().active_cap):
                            break
                        base_angle = math.atan2(a.vel.y, a.vel.x) + sign * 0.45
                        child = self._spawn_asteroid(pos=a.pos, size=a.size - 1,
                                                     direction=from_angle(base_angle), parent=a)
                        children.append(child)
                        events.append(GameEvent("asteroid_split", self.step_count, str(child.id), str(a.id)))
                break
        for a in self._asteroids:
            if a.id in removed_a:
                continue
            radius = ASTEROID_RADII[a.size]
            hit = False
            for ship in self._ships:
                if self.config.ship.invulnerable:
                    continue
                if ship.alive and wrapped_distance(a.pos, ship.pos, w, h) <= radius + self.config.ship.radius:
                    ship.alive = False
                    removed_a.add(a.id)
                    hit = True
                    events.append(GameEvent("ship_destroyed", self.step_count, ship.id, str(a.id)))
                    break
            if hit:
                continue
            if self.config.objective.protect:
                center = Vec2(w / 2, h / 2)
                if wrapped_distance(a.pos, center, w, h) <= radius + self.config.objective.object_radius:
                    self.object_health -= a.size
                    removed_a.add(a.id)
                    events.append(GameEvent("object_damaged", self.step_count, "objective", str(a.size)))
        # Optional teammate collision rules. "full" currently adds projectile friendly fire too.
        if self.config.ship.friendly_collisions in {"ships", "full"}:
            for i, first in enumerate(self._ships):
                for second in self._ships[i + 1:]:
                    if first.alive and second.alive and wrapped_distance(first.pos, second.pos, w, h) <= 2 * self.config.ship.radius:
                        first.alive = second.alive = False
                        events.append(GameEvent("ship_collision", self.step_count, first.id, second.id))
        if self.config.ship.friendly_collisions == "full":
            for p in self._projectiles:
                if p.id in removed_p:
                    continue
                for ship in self._ships:
                    if (ship.alive and ship.id != p.owner_id and wrapped_distance(p.pos, ship.pos, w, h)
                            <= self.config.projectile.radius + self.config.ship.radius):
                        ship.alive = False
                        removed_p.add(p.id)
                        events.append(GameEvent("friendly_fire", self.step_count, ship.id, p.owner_id))
                        break
                if self.config.objective.protect and p.id not in removed_p:
                    center = Vec2(w / 2, h / 2)
                    if (wrapped_distance(p.pos, center, w, h)
                            <= self.config.projectile.radius + self.config.objective.object_radius):
                        self.object_health -= 1
                        removed_p.add(p.id)
                        events.append(GameEvent("object_damaged", self.step_count, "objective", "friendly_fire"))
        self._asteroids = [a for a in self._asteroids if a.id not in removed_a] + children
        self._projectiles = [p for p in self._projectiles if p.id not in removed_p]

    def _new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def snapshot(self) -> WorldSnapshot:
        c = self.config
        return WorldSnapshot(
            self.step_count, self.step_count * self.dt, c.arena.width, c.arena.height,
            tuple(ShipSnapshot(s.id, s.pos.x, s.pos.y, s.vel.x, s.vel.y, s.angle, s.alive,
                               c.ship.radius, s.cooldown)
                  for s in self._ships),
            tuple(AsteroidSnapshot(a.id, a.pos.x, a.pos.y, a.vel.x, a.vel.y, a.size,
                                   ASTEROID_RADII[a.size], a.pattern) for a in self._asteroids),
            tuple(ProjectileSnapshot(p.id, p.owner_id, p.pos.x, p.pos.y, p.vel.x, p.vel.y,
                                     c.projectile.radius, p.age) for p in self._projectiles),
            ObjectiveSnapshot(c.objective.protect, c.arena.width / 2, c.arena.height / 2,
                              c.objective.object_radius, self.object_health),
            self.terminated, self.truncated, self.terminal_reason, self._wave,
            self._difficulty() if c.asteroid.is_ramped else None,
        )
