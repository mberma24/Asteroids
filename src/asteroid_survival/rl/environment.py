from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from ..actions import Action
from ..config import GameConfig
from ..math2d import Vec2, wrapped_delta
from ..patterns import PEAK_SPEED_FACTOR
from ..simulation import ASTEROID_RADII, Simulation
from ..state import GameEvent, WorldSnapshot


STATIONARY_ACTIONS = (
    Action.NOOP, Action.LEFT, Action.RIGHT, Action.FINE_LEFT, Action.FINE_RIGHT,
    Action.FIRE, Action.LEFT_FIRE, Action.RIGHT_FIRE,
    Action.FINE_LEFT_FIRE, Action.FINE_RIGHT_FIRE,
)
MOBILE_ACTIONS = tuple(Action)
ASTEROID_FEATURES = 12
PROJECTILE_FEATURES = 10
MAX_PROJECTILES = 8
TEAMMATE_FEATURES = 8
GLOBAL_FEATURES = 16
SHIP_FEATURES = 7
MOBILE_SHIP_FEATURES = 11


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Arcade clearing reward. ``None`` keeps the legacy survival reward."""

    large: float = 0.2
    medium: float = 0.5
    small: float = 1.0
    wave_clear: float = 5.0
    fast_clear: float = 3.0
    accuracy: float = 2.0
    miss_penalty: float = 0.02
    timeout_penalty: float = 1.0
    death_penalty: float = 5.0
    active_time_penalty: float = 0.02
    par_time_per_wave: float = 45.0
    # Survival rounds. There is nothing to clear, so staying alive has to be paid for
    # directly; `active_time_penalty` and `timeout_penalty` are its exact opposites and a
    # survival curriculum sets both to zero.
    survival_bonus: float = 0.0
    """Reward per decision survived; the core term for a survival round.

    Paid per decision rather than per simulated second so that its total over a round is
    read directly off the decision limit: 0.02 over a 900-decision round is worth 18.
    """
    round_clear: float = 0.0
    """Paid once for reaching the decision limit alive, the survival analogue of a clear."""
    # Signed potential shaping: turning toward the current target earns a small amount and
    # turning away loses the same amount. Keeping it signed prevents oscillation farming.
    aim_progress: float = 0.0
    time_scaled_survival: bool = False
    """Prorate survival shaping when an episode ends inside a frame-skipped decision."""
    safety_progress: float = 0.0
    """Optional signed change in predicted safety; zero in the baseline curriculum."""


def ship_feature_count(config: GameConfig) -> int:
    return MOBILE_SHIP_FEATURES if config.ship.mobile else SHIP_FEATURES


def encode_observation(state: WorldSnapshot, agent_id: str, config: GameConfig,
                       max_decisions: int, max_asteroids: int, frame_skip: int = 4,
                       history: dict[int, deque] | None = None,
                       history_frames: int = 0, offsets: list[int] | None = None,
                       max_projectiles: int = MAX_PROJECTILES,
                       max_teammates: int = 0,
                       reveal_progress: bool = True, *, global_features: bool = False,
                       spawn_phase: float = 0.0) -> np.ndarray:
    slots = history_offsets(history_frames) if offsets is None else offsets
    ship = next(ship for ship in state.ships if ship.id == agent_id)
    # Without the cooldown the agent cannot tell whether FIRE will do anything: the weapon is
    # unavailable for roughly fire_cooldown / (frame_skip / fps) consecutive decisions.
    cooldown = (ship.cooldown / config.ship.fire_cooldown) if config.ship.fire_cooldown else 0.0
    values = [
        math.sin(ship.angle), math.cos(ship.angle), float(ship.alive),
        len(state.asteroids) / max_asteroids,
        (min(1.0, state.elapsed / (max_decisions * frame_skip / config.arena.fps))
         if reveal_progress else 0.0),
        min(1.0, max(0.0, cooldown)),
        float(ship.cooldown <= 0.0),
    ]
    if config.ship.mobile:
        scale = max(config.ship.max_speed, 1.0)
        values.extend((
            ship.vx / scale,
            ship.vy / scale,
            float(config.ship.acceleration > 0.0),
            float(config.ship.invulnerable),
        ))
    origin = Vec2(ship.x, ship.y)
    speed_scale = max(
        1.0,
        config.asteroid.max_speed
        + PEAK_SPEED_FACTOR * config.asteroid.amplitude_max * 2 * math.pi
        / config.asteroid.wavelength_min,
    )
    nearest = sorted(
        state.asteroids,
        key=lambda asteroid: wrapped_delta(
            origin, Vec2(asteroid.x, asteroid.y), state.width, state.height).length(),
    )[:max_asteroids]
    diagonal = math.hypot(state.width / 2, state.height / 2)
    for asteroid in nearest:
        delta = wrapped_delta(origin, Vec2(asteroid.x, asteroid.y), state.width, state.height)
        distance = delta.length()
        # Aiming depends on the bearing from the ship's heading to the asteroid, so supply it
        # directly. Recovering it from world-frame x/y is hard, and dividing x and y by
        # different constants would skew every angle in the process.
        bearing = math.atan2(delta.y, delta.x) - ship.angle
        if distance > 1e-6:
            unit_x, unit_y = delta.x / distance, delta.y / distance
        else:
            unit_x, unit_y = 0.0, 0.0
        closing = -(asteroid.vx * unit_x + asteroid.vy * unit_y)
        tangential = asteroid.vx * -unit_y + asteroid.vy * unit_x
        values.extend((
            1.0,
            delta.x / diagonal,
            delta.y / diagonal,
            asteroid.vx / speed_scale,
            asteroid.vy / speed_scale,
            asteroid.radius / ASTEROID_RADII[3],
            asteroid.size / 3,
            distance / diagonal,
            math.sin(bearing),
            math.cos(bearing),
            closing / speed_scale,
            tangential / speed_scale,
        ))
        # Past offsets for this specific asteroid, keyed by id: the slots are re-sorted by
        # distance every step, so a naive stack of past observations would misalign them.
        # Asteroid paths are deterministic but curved, and their curvature is not otherwise
        # observable, so this is what makes the future recoverable at all.
        if slots:
            track = history.get(asteroid.id) if history is not None else None
            for index in slots:
                if track is not None and index < len(track):
                    past_x, past_y = track[index]
                    # Wrap-aware, like every other position here: raw subtraction turns an
                    # asteroid crossing the screen edge into a full-arena jump, which is
                    # physically impossible over one decision and swamps the real signal.
                    past = wrapped_delta(Vec2(asteroid.x, asteroid.y), Vec2(past_x, past_y),
                                         state.width, state.height)
                    values.extend((past.x / diagonal, past.y / diagonal))
                else:
                    values.extend((0.0, 0.0))  # newly spawned: no history that far back
    values.extend([0.0] * ((max_asteroids - len(nearest)) * (ASTEROID_FEATURES + 2 * len(slots))))
    # Active shots are part of the physical state. Without them, two visually identical
    # observations can have very different futures (a hit is already on the way versus no
    # shot exists), so the reward/value model is forced to guess. Keep these slots after the
    # legacy asteroid block so an older representation can be expanded without moving any
    # of its learned input weights.
    projectile_speed = max(config.projectile.speed, 1.0)
    lifetime = max(config.projectile.lifetime, 1e-6)
    projectiles = sorted(
        state.projectiles,
        key=lambda projectile: wrapped_delta(
            origin, Vec2(projectile.x, projectile.y), state.width, state.height).length(),
    )[:max_projectiles]
    for projectile in projectiles:
        delta = wrapped_delta(
            origin, Vec2(projectile.x, projectile.y), state.width, state.height)
        distance = delta.length()
        bearing = math.atan2(delta.y, delta.x) - ship.angle
        values.extend((
            1.0,
            delta.x / diagonal,
            delta.y / diagonal,
            projectile.vx / projectile_speed,
            projectile.vy / projectile_speed,
            distance / diagonal,
            math.sin(bearing),
            math.cos(bearing),
            min(1.0, max(0.0, (lifetime - projectile.age) / lifetime)),
            float(projectile.owner_id == agent_id),
        ))
    values.extend([0.0] * ((max_projectiles - len(projectiles)) * PROJECTILE_FEATURES))
    # Teammates, last of all, so adding them leaves every earlier input weight in place.
    # Without them a policy cannot see the ships it is required not to collide with, and
    # co-operative rounds would be unlearnable rather than merely hard.
    if max_teammates:
        others = sorted(
            (other for other in state.ships if other.id != agent_id and other.alive),
            key=lambda other: wrapped_delta(
                origin, Vec2(other.x, other.y), state.width, state.height).length(),
        )[:max_teammates]
        for other in others:
            delta = wrapped_delta(origin, Vec2(other.x, other.y), state.width, state.height)
            distance = delta.length()
            bearing = math.atan2(delta.y, delta.x) - ship.angle
            values.extend((
                1.0,
                delta.x / diagonal,
                delta.y / diagonal,
                other.vx / speed_scale,
                other.vy / speed_scale,
                distance / diagonal,
                math.sin(bearing),
                math.cos(bearing),
            ))
        values.extend([0.0] * ((max_teammates - len(others)) * TEAMMATE_FEATURES))
    if global_features:
        difficulty = config.asteroid.difficulty_at(state.elapsed)
        # A compact absolute-difficulty block fixes an ambiguity in the legacy encoding:
        # asteroid velocities were normalized by the stage maximum, making slow and fast
        # rounds look alike.  The old 1,235 inputs stay byte-for-byte in front of this block.
        values.extend((
            min(1.0, max(0.0, spawn_phase)),
            min(1.0, difficulty.spawn_interval / 5.0),
            min(1.0, difficulty.min_speed / 250.0),
            min(1.0, difficulty.max_speed / 250.0),
            min(1.0, difficulty.amplitude_max / 200.0),
            min(1.0, difficulty.wavelength_min / 6.0),
            min(1.0, difficulty.wavelength_max / 6.0),
            min(1.0, difficulty.spawn_spread / 180.0),
        ))
        threats = []
        for asteroid in state.asteroids:
            delta = wrapped_delta(origin, Vec2(asteroid.x, asteroid.y), state.width, state.height)
            rvx, rvy = asteroid.vx - ship.vx, asteroid.vy - ship.vy
            speed2 = rvx * rvx + rvy * rvy
            ttc = max(0.0, -(delta.x * rvx + delta.y * rvy) / speed2) if speed2 > 1e-9 else 5.0
            ttc = min(5.0, ttc)
            miss = max(0.0, math.hypot(delta.x + rvx * ttc, delta.y + rvy * ttc)
                       - asteroid.radius - config.ship.radius)
            threats.append((ttc, miss, asteroid, delta, rvx, rvy))
        if threats:
            ttc, miss, threat, delta, rvx, rvy = min(threats, key=lambda x: (x[0], x[1]))
            distance = max(delta.length(), 1e-9)
            ux, uy = delta.x / distance, delta.y / distance
            bearing = math.atan2(delta.y, delta.x) - ship.angle
            closing = -(rvx * ux + rvy * uy)
            tangential = rvx * -uy + rvy * ux
            signed = lambda value: math.copysign(math.log1p(abs(value)) / math.log1p(2500.0), value)
            values.extend((1.0, ttc / 5.0, min(1.0, miss / diagonal),
                           math.sin(bearing), math.cos(bearing),
                           signed(closing), signed(tangential),
                           threat.radius / ASTEROID_RADII[3]))
        else:
            values.extend((0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    return np.asarray(values, dtype=np.float32)


def history_offsets(history_frames: int, long_frames: int = 0, long_stride: int = 1) -> list[int]:
    """Which past decisions to store for each asteroid.

    The recent ones are kept densely, for the fine detail that aiming needs. The older ones
    are sampled every ``long_stride`` decisions, so a handful of slots can still reach back
    across a whole oscillation. Asteroid paths repeat on a 1.7-4.5s period, and a dense
    buffer long enough to cover that would be enormous.
    """
    offsets = list(range(history_frames))
    if long_frames and history_frames:
        start = history_frames - 1
        offsets += [start + long_stride * (i + 1) for i in range(long_frames)]
    return offsets


def feature_width(history_frames: int, long_frames: int = 0) -> int:
    return ASTEROID_FEATURES + 2 * (history_frames + long_frames)


@dataclass(slots=True)
class EpisodeMetrics:
    seed: int
    survival_time: float = 0.0
    wave: int = 0
    """Waves released; 0 outside wave spawning. A more legible score than survival time."""
    decisions: int = 0
    simulation_steps: int = 0
    reward: float = 0.0
    asteroid_reward: float = 0.0
    shot_penalty: float = 0.0
    shots_fired: int = 0
    shots_missed: int = 0
    shots_resolved: int = 0
    asteroids_destroyed: int = 0
    large_destroyed: int = 0
    medium_destroyed: int = 0
    small_destroyed: int = 0
    waves_cleared: int = 0
    wave_clear_times: list[float] = field(default_factory=list)
    hit_reward: float = 0.0
    survival_reward: float = 0.0
    round_clear_reward: float = 0.0
    wave_clear_reward: float = 0.0
    fast_clear_reward: float = 0.0
    accuracy_reward: float = 0.0
    aim_progress_reward: float = 0.0
    safety_progress_reward: float = 0.0
    miss_penalty: float = 0.0
    timeout_penalty: float = 0.0
    death_penalty: float = 0.0
    time_penalty: float = 0.0
    survived_to_limit: bool = False
    stalled_out: bool = False
    """Ended early after going no_hit_seconds without destroying anything."""
    last_hit_time: float = 0.0
    completed_stage: bool = False
    terminal_reason: str | None = None
    action_counts: list[int] = field(default_factory=lambda: [0] * len(MOBILE_ACTIONS))
    mean_ship_speed: float = 0.0
    mean_asteroid_count: float = 0.0
    minimum_clearance: float | None = None

    @property
    def accuracy(self) -> float:
        return self.asteroids_destroyed / self.shots_fired if self.shots_fired else 0.0

    @property
    def resolved_accuracy(self) -> float:
        return self.asteroids_destroyed / self.shots_resolved if self.shots_resolved else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["accuracy"] = self.accuracy
        result["resolved_accuracy"] = self.resolved_accuracy
        result["mean_wave_clear_time"] = (
            sum(self.wave_clear_times) / len(self.wave_clear_times)
            if self.wave_clear_times else 0.0)
        return result


class AsteroidsRLEnv:
    """Small Gym-like adapter with fixed observations and reproducible episode metrics.

    Reward is elapsed survival time plus a configurable bonus for asteroid hits.
    """

    def __init__(self, config: GameConfig, agent_id: str | None = None, *, frame_skip: int = 4,
                 max_decisions: int = 1800, max_asteroids: int | None = None,
                 max_projectiles: int = MAX_PROJECTILES,
                 asteroid_reward: float = 0.1, shot_penalty: float = 0.0,
                 history_frames: int = 0, history_long_frames: int = 0,
                 history_long_stride: int = 8, reward_config: RewardConfig | None = None,
                 no_hit_seconds: float = 0.0, completion: str = "waves",
                 max_teammates: int = 0,
                 companion_policy: Callable[[np.ndarray], int] | None = None,
                 reveal_progress: bool | None = None, global_features: bool = False):
        if completion not in {"waves", "survival"}:
            raise ValueError("completion must be waves or survival")
        self.completion = completion
        if frame_skip < 1 or max_decisions < 1:
            raise ValueError("frame_skip and max_decisions must be positive")
        if asteroid_reward < 0:
            raise ValueError("asteroid_reward cannot be negative")
        if shot_penalty < 0:
            raise ValueError("shot_penalty cannot be negative")
        if no_hit_seconds < 0:
            raise ValueError("no_hit_seconds cannot be negative")
        self.no_hit_seconds = no_hit_seconds
        self.config = config
        self.agent_id = agent_id or config.ships[0].id
        if self.agent_id not in {ship.id for ship in config.ships}:
            raise ValueError(f"unknown agent ship: {self.agent_id}")
        self.frame_skip = frame_skip
        self.max_decisions = max_decisions
        self.max_asteroids = max_asteroids or config.asteroid.active_cap
        if max_projectiles < 0:
            raise ValueError("max_projectiles cannot be negative")
        self.max_projectiles = max_projectiles
        self.max_teammates = max_teammates
        # Co-operative rounds put more than one ship on the field, every one driven by the
        # same policy. Left unset the companions simply hold station, which is what keeps
        # single-ship behaviour identical.
        self.companion_policy = companion_policy
        self.global_features = bool(global_features)
        self.companion_ids = [spec.id for spec in config.ships if spec.id != self.agent_id]
        # A survival round is scored on staying alive, and its decision limit is only there
        # to bound episode cost. Telling the policy how close that limit is invites it to
        # learn behaviour keyed to the deadline -- coasting into it, or giving up once past
        # it -- which is exactly wrong for a task whose point is to survive indefinitely.
        # The slot stays, always zero, so the observation layout does not change.
        self.reveal_progress = (completion != "survival" if reveal_progress is None
                                else bool(reveal_progress))
        self.asteroid_reward = asteroid_reward
        self.shot_penalty = shot_penalty
        self.reward_config = reward_config
        if history_frames < 0 or history_long_frames < 0:
            raise ValueError("history frame counts cannot be negative")
        if history_long_stride < 1:
            raise ValueError("history_long_stride must be at least 1")
        if history_long_frames and not history_frames:
            raise ValueError("history_long_frames requires history_frames")
        self.history_frames = history_frames
        self.history_long_frames = history_long_frames
        self.history_long_stride = history_long_stride
        self.history_slots = history_offsets(
            history_frames, history_long_frames, history_long_stride)
        # The buffer has to reach as far back as the oldest slot samples.
        self._history_depth = (max(self.history_slots) + 1) if self.history_slots else 0
        self._history: dict[int, deque] = {}
        self.actions = MOBILE_ACTIONS if config.ship.mobile else STATIONARY_ACTIONS
        self.simulation = Simulation(config)
        self.state: WorldSnapshot | None = None
        self.metrics: EpisodeMetrics | None = None
        self._wave_started_at = 0.0
        self._wave_shots = 0
        self._wave_hits = 0

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def inert_actions(self) -> tuple[bool, ...]:
        """Which actions are indistinguishable from a cheaper one under this config.

        With ``acceleration = 0`` thrust does nothing, so every thrusting action duplicates its
        non-thrusting twin. Search that considers them anyway spends a large share of its budget
        exploring identical options, and the policy splits probability across them.
        """
        thrust_is_dead = not self.config.ship.mobile or self.config.ship.acceleration <= 0.0
        return tuple(bool(action.thrust) and thrust_is_dead for action in self.actions)

    @property
    def observation_size(self) -> int:
        return (ship_feature_count(self.config) + self.max_asteroids * (
            ASTEROID_FEATURES + 2 * len(self.history_slots)) + (
                self.max_projectiles * PROJECTILE_FEATURES) + (
                    self.max_teammates * TEAMMATE_FEATURES)
                + (GLOBAL_FEATURES if self.global_features else 0))

    def reset(self, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        self.state = self.simulation.reset(seed)
        self.metrics = EpisodeMetrics(seed=seed)
        self._history.clear()
        self._wave_started_at = 0.0
        self._wave_shots = self._wave_hits = 0
        return self._observation(self.state), {"seed": seed}

    def _record_history(self) -> None:
        """Append this decision's asteroid positions, dropping any that no longer exist."""
        if not self._history_depth or self.state is None:
            return
        live = set()
        for asteroid in self.state.asteroids:
            live.add(asteroid.id)
            track = self._history.get(asteroid.id)
            if track is None:
                track = deque(maxlen=self._history_depth)
                self._history[asteroid.id] = track
            track.appendleft((asteroid.x, asteroid.y))
        for asteroid_id in [key for key in self._history if key not in live]:
            del self._history[asteroid_id]

    def _nearest_aim_error(self, asteroid_id: int | None = None) -> tuple[int, float] | None:
        """Return target id and absolute wrapped bearing error for reward shaping."""
        if self.state is None or not self.state.asteroids:
            return None
        ship = next(ship for ship in self.state.ships if ship.id == self.agent_id)
        origin = Vec2(ship.x, ship.y)
        candidates = self.state.asteroids
        if asteroid_id is not None:
            candidates = tuple(rock for rock in candidates if rock.id == asteroid_id)
            if not candidates:
                return None
        rock = min(candidates, key=lambda item: wrapped_delta(
            origin, Vec2(item.x, item.y), self.state.width, self.state.height).length())
        delta = wrapped_delta(
            origin, Vec2(rock.x, rock.y), self.state.width, self.state.height)
        bearing = math.atan2(delta.y, delta.x) - ship.angle
        wrapped = (bearing + math.pi) % (2 * math.pi) - math.pi
        return rock.id, abs(wrapped)

    def _safety_potential(self) -> float:
        """Bounded predicted clearance potential used only when explicitly configured."""
        if self.state is None or not self.state.asteroids:
            return 1.0
        ship = next(item for item in self.state.ships if item.id == self.agent_id)
        origin = Vec2(ship.x, ship.y)
        candidates = []
        for rock in self.state.asteroids:
            delta = wrapped_delta(origin, Vec2(rock.x, rock.y),
                                  self.state.width, self.state.height)
            vx, vy = rock.vx - ship.vx, rock.vy - ship.vy
            speed2 = vx * vx + vy * vy
            ttc = max(0.0, -(delta.x * vx + delta.y * vy) / speed2) if speed2 > 1e-9 else 5.0
            ttc = min(5.0, ttc)
            clearance = max(0.0, math.hypot(delta.x + vx * ttc, delta.y + vy * ttc)
                            - rock.radius - self.config.ship.radius)
            candidates.append(0.5 * min(1.0, ttc / 5.0)
                              + 0.5 * min(1.0, clearance / 150.0))
        return min(candidates)

    def step(self, action_index: int, *, on_frame: Callable[[WorldSnapshot], None] | None = None
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision. ``on_frame`` sees every simulated frame, not just the last,
        so an interactive viewer can render smoothly while still deciding at the agent's rate.
        """
        if self.state is None or self.metrics is None:
            raise RuntimeError("call reset() before step()")
        if not 0 <= int(action_index) < self.num_actions:
            raise ValueError(f"action index must be in [0, {self.num_actions})")
        action = self.actions[int(action_index)]
        self.metrics.action_counts[int(action_index)] += 1
        aim_before = self._nearest_aim_error()
        safety_before = self._safety_potential()
        start_elapsed = self.state.elapsed
        shots_before = self.metrics.shots_fired
        asteroid_hits = 0
        arcade_reward = 0.0
        terminated = truncated = False
        companions = self._companion_actions()
        for _ in range(self.frame_skip):
            sizes = {asteroid.id: asteroid.size for asteroid in self.state.asteroids}
            result = self.simulation.step({self.agent_id: action, **companions})
            self.state = result.snapshot
            hits, event_reward = self._record_events(result.events, sizes)
            asteroid_hits += hits
            arcade_reward += event_reward
            terminated = result.terminated or not self._agent_alive()
            truncated = result.truncated
            if on_frame is not None:
                on_frame(self.state)
            if terminated or truncated:
                break
        hit_reward = asteroid_hits * self.asteroid_reward
        # Firing is otherwise free, so nothing pushes the agent toward trigger discipline;
        # a wasted shot only costs it a cooldown several decisions later.
        shot_cost = (self.metrics.shots_fired - shots_before) * self.shot_penalty
        elapsed = max(0.0, self.state.elapsed - start_elapsed)
        if self.reward_config is None:
            reward = elapsed + hit_reward - shot_cost
        else:
            normal_elapsed = self.frame_skip / self.config.arena.fps
            fraction = (elapsed / normal_elapsed if self.reward_config.time_scaled_survival
                        else 1.0)
            survival_reward = self.reward_config.survival_bonus * fraction
            self.metrics.survival_reward += survival_reward
            arcade_reward += survival_reward
            time_cost = elapsed * self.reward_config.active_time_penalty
            self.metrics.time_penalty += time_cost
            reward = arcade_reward - time_cost
            if aim_before is not None and self.reward_config.aim_progress:
                aim_after = self._nearest_aim_error(aim_before[0])
                if aim_after is not None:
                    aim_reward = self.reward_config.aim_progress * (
                        aim_before[1] - aim_after[1])
                    reward += aim_reward
                    self.metrics.aim_progress_reward += aim_reward
            if self.reward_config.safety_progress:
                safety_reward = self.reward_config.safety_progress * (
                    self._safety_potential() - safety_before)
                reward += safety_reward
                self.metrics.safety_progress_reward += safety_reward
        self.metrics.decisions += 1
        self.metrics.simulation_steps = self.state.step
        self.metrics.survival_time = self.state.elapsed
        self.metrics.wave = self.state.wave
        ship_now = next(ship for ship in self.state.ships if ship.id == self.agent_id)
        count = self.metrics.decisions
        speed = math.hypot(ship_now.vx, ship_now.vy)
        self.metrics.mean_ship_speed += (speed - self.metrics.mean_ship_speed) / count
        self.metrics.mean_asteroid_count += (
            len(self.state.asteroids) - self.metrics.mean_asteroid_count) / count
        origin_now = Vec2(ship_now.x, ship_now.y)
        if self.state.asteroids:
            clearance = min(
                wrapped_delta(origin_now, Vec2(rock.x, rock.y),
                              self.state.width, self.state.height).length()
                - ship_now.radius - rock.radius for rock in self.state.asteroids)
            self.metrics.minimum_clearance = (clearance if self.metrics.minimum_clearance is None
                                              else min(self.metrics.minimum_clearance, clearance))
        if self.reward_config is None:
            self.metrics.asteroid_reward += hit_reward
        else:
            self.metrics.asteroid_reward = self.metrics.hit_reward
        self.metrics.shot_penalty += shot_cost
        # A dry spell is the signature of firing at nothing: competent play never goes more
        # than ~2s between hits, while a stuck policy can burn the entire episode. Ending
        # early puts the penalty near the behaviour that earned it instead of a minute later,
        # and stops unproductive episodes from dominating the compute budget.
        if asteroid_hits:
            self.metrics.last_hit_time = self.state.elapsed
        dry_spell = self.state.elapsed - self.metrics.last_hit_time
        stalled = (
            self.no_hit_seconds > 0.0
            and bool(self.state.asteroids)  # nothing to shoot at is not the agent's fault
            and dry_spell >= self.no_hit_seconds
            and not terminated and not truncated
        )
        timed_out = self.metrics.decisions >= self.max_decisions and not terminated and not truncated
        if timed_out or stalled:
            truncated = True
            self.metrics.survived_to_limit = not stalled
            self.metrics.stalled_out = stalled
            if self.reward_config is not None:
                if self.completion == "survival" and self.metrics.survived_to_limit:
                    # Lasting the whole round is the objective; charging the wave-mode
                    # timeout penalty here would punish the agent for succeeding.
                    reward += self.reward_config.round_clear
                    self.metrics.round_clear_reward += self.reward_config.round_clear
                else:
                    reward -= self.reward_config.timeout_penalty
                    self.metrics.timeout_penalty += self.reward_config.timeout_penalty
        # Once an episode ends, every projectile still in flight has lost its chance to hit.
        # Count and charge those shots now; otherwise spraying immediately before a clear
        # escapes the miss penalty entirely.
        if terminated or truncated:
            unresolved = max(0, self.metrics.shots_fired - self.metrics.shots_resolved)
            if unresolved:
                self.metrics.shots_missed += unresolved
                self.metrics.shots_resolved += unresolved
                if self.reward_config is not None:
                    terminal_miss_cost = unresolved * self.reward_config.miss_penalty
                    reward -= terminal_miss_cost
                    self.metrics.miss_penalty += terminal_miss_cost
        self.metrics.reward += reward
        if terminated or truncated:
            self.metrics.completed_stage = (
                self.metrics.survived_to_limit if self.completion == "survival" else
                (self.state.terminal_reason is not None
                 and self.state.terminal_reason.value == "waves_cleared"))
            self.metrics.terminal_reason = (
                self.state.terminal_reason.value if self.state.terminal_reason else
                "agent_destroyed" if not self._agent_alive() else
                "no_hit_timeout" if self.metrics.stalled_out else "evaluation_limit"
            )
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode_metrics"] = self.metrics.to_dict()
        return self._observation(self.state), reward, terminated, truncated, info

    def _agent_alive(self) -> bool:
        assert self.state is not None
        return next(ship.alive for ship in self.state.ships if ship.id == self.agent_id)

    def _record_events(self, events: tuple[GameEvent, ...],
                       asteroid_sizes: dict[int, int]) -> tuple[int, float]:
        assert self.metrics is not None
        asteroid_hits = 0
        reward = 0.0
        for event in events:
            if event.kind == "projectile_fired" and event.detail == self.agent_id:
                self.metrics.shots_fired += 1
                self._wave_shots += 1
            elif event.kind == "asteroid_shot" and event.detail == self.agent_id:
                asteroid_hits += 1
                self._wave_hits += 1
                self.metrics.asteroids_destroyed += 1
                size = asteroid_sizes.get(int(event.entity_id))
                if size == 3:
                    self.metrics.large_destroyed += 1
                elif size == 2:
                    self.metrics.medium_destroyed += 1
                elif size == 1:
                    self.metrics.small_destroyed += 1
                if self.reward_config is not None:
                    points = {3: self.reward_config.large, 2: self.reward_config.medium,
                              1: self.reward_config.small}.get(size, 0.0)
                    reward += points
                    self.metrics.hit_reward += points
            elif event.kind == "projectile_expired" and event.detail == self.agent_id:
                self.metrics.shots_missed += 1
                self.metrics.shots_resolved += 1
                if self.reward_config is not None:
                    reward -= self.reward_config.miss_penalty
                    self.metrics.miss_penalty += self.reward_config.miss_penalty
            elif event.kind == "wave_started":
                self._wave_started_at = self.state.elapsed if self.state else 0.0
                self._wave_shots = self._wave_hits = 0
            elif event.kind == "wave_cleared":
                clear_time = max(0.0, (self.state.elapsed if self.state else 0.0)
                                 - self._wave_started_at)
                self.metrics.waves_cleared += 1
                self.metrics.wave_clear_times.append(clear_time)
                if self.reward_config is not None:
                    accuracy = self._wave_hits / self._wave_shots if self._wave_shots else 0.0
                    fast = self.reward_config.fast_clear * max(
                        0.0, 1.0 - clear_time / self.reward_config.par_time_per_wave)
                    accurate = self.reward_config.accuracy * accuracy
                    reward += self.reward_config.wave_clear + fast + accurate
                    self.metrics.wave_clear_reward += self.reward_config.wave_clear
                    self.metrics.fast_clear_reward += fast
                    self.metrics.accuracy_reward += accurate
            elif event.kind == "ship_destroyed" and event.entity_id == self.agent_id:
                if self.reward_config is not None:
                    reward -= self.reward_config.death_penalty
                    self.metrics.death_penalty += self.reward_config.death_penalty
        # Every hit resolves the projectile immediately.
        self.metrics.shots_resolved += asteroid_hits
        return asteroid_hits, reward

    def _observation(self, state: WorldSnapshot) -> np.ndarray:
        # Encode against strictly past positions, then fold this decision into the history,
        # so an asteroid's own current position never appears as its own history.
        observation = self._observe_as(state, self.agent_id)
        self._record_history()
        return observation

    def _observe_as(self, state: WorldSnapshot, ship_id: str) -> np.ndarray:
        """The same encoding, from another ship's point of view.

        Companions run the same policy on their own view of the world, so a co-operative
        round really is one model flying every ship rather than a policy plus scripted
        bystanders. The asteroid history is shared because it is keyed by asteroid, not by
        observer.
        """
        return encode_observation(
            state, ship_id, self.config, self.max_decisions, self.max_asteroids,
            self.frame_skip, self._history, self.history_frames, self.history_slots,
            self.max_projectiles, self.max_teammates, self.reveal_progress,
            global_features=self.global_features,
            spawn_phase=self.simulation.spawn_phase)

    def _companion_actions(self) -> dict[str, Action]:
        """One action per companion, held for the whole frame-skip like the learner's."""
        assert self.state is not None
        if not self.companion_ids or self.companion_policy is None:
            return {}
        actions = {}
        for ship_id in self.companion_ids:
            ship = next((s for s in self.state.ships if s.id == ship_id), None)
            if ship is None or not ship.alive:
                continue
            index = int(self.companion_policy(self._observe_as(self.state, ship_id)))
            actions[ship_id] = self.actions[max(0, min(index, len(self.actions) - 1))]
        return actions
