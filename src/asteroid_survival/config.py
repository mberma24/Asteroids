from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PATTERN_NAMES = (
    "sine", "zigzag", "sawtooth", "arc", "s_curve", "lane_change",
    "serpentine", "corkscrew", "figure_eight", "spiral", "brownian", "gust",
)


@dataclass(slots=True)
class ArenaConfig:
    width: int = 900
    height: int = 900
    fps: int = 60


@dataclass(slots=True)
class ShipSpec:
    id: str
    controller: str = "heuristic"
    input_profile: str = "keyboard_1"


@dataclass(slots=True)
class ShipConfig:
    mobile: bool = True
    invulnerable: bool = False
    """Training aid: asteroids pass through ships without destroying either one."""
    friendly_collisions: str = "off"
    spawn_radius: float = 80.0
    """Initial distance from arena centre for multi-ship/objective layouts."""
    radius: float = 14.0
    acceleration: float = 220.0
    drag: float = 0.35
    max_speed: float = 320.0
    turn_speed: float = 5.0
    fire_cooldown: float = 0.24


@dataclass(frozen=True, slots=True)
class Difficulty:
    """Asteroid settings in force at one moment of an episode."""
    spawn_interval: float
    active_cap: int
    min_speed: float
    max_speed: float
    amplitude_max: float
    wave_threshold: int = 0
    spawn_spread: float = 0.0
    wavelength_min: float = 1.7
    wavelength_max: float = 4.5
    tier: int | None = None
    """One-based difficulty step now in force; ``None`` when the ramp is continuous."""


@dataclass(slots=True)
class AsteroidConfig:
    spawn_mode: str = "interval"
    """``interval`` drips asteroids in on a clock; ``wave`` refills in arcade-style waves."""
    spawn_interval: float = 1.25
    active_cap: int = 32
    # Wave mode. A wave is released once the field has been worn down to wave_threshold
    # asteroids, then its members arrive one at a time rather than all at once. The
    # original arcade game used a threshold of zero (clear the screen to advance) and grew
    # each wave by two rocks; raising the threshold over time is what makes later waves
    # overlap and crowd the field.
    wave_threshold: int = 0
    wave_threshold_start: int | None = None
    wave_size: int = 4
    """Asteroids released in the first wave."""
    wave_composition: list[int] = field(default_factory=list)
    """Optional exact sizes released each wave (1=small, 2=medium, 3=large)."""
    wave_growth: int = 2
    """Extra asteroids added to each subsequent wave."""
    wave_size_max: int = 11
    wave_spawn_interval: float = 0.4
    """Gap between the individual asteroids of one wave."""
    wave_delay: float = 1.6
    """Breathing room after the threshold is met, before the next wave starts arriving."""
    initial_wave_delay: float = 0.0
    """Grace period before the first wave starts arriving."""
    spawn_size: int | list[int] | None = None
    """Size every asteroid spawns at; a list draws uniformly from it, ``None`` from all three."""
    initial_asteroids: int = 0
    """Asteroids already in flight when the episode starts.

    Interval spawning otherwise opens on an empty arena, so the first seconds of a round are
    free time that does not reflect its difficulty. Pre-placing the field makes a round
    representative from the first decision.
    """
    spawn_safe_radius: float = 0.0
    """Fixed clearance kept between a pre-placed asteroid and every ship, beyond their radii."""
    spawn_safe_seconds: float = 0.0
    """Extra clearance worth this many seconds of travel at the round's top speed.

    A fixed radius is a shrinking reaction window as asteroids get faster: 180px is over
    four seconds at 42 px/s but under a second and a half at 125 px/s. Scaling with speed
    keeps the opening fair at every rung of a difficulty ladder.
    """
    min_speed: float = 75.0
    max_speed: float = 145.0
    motion_mode: str = "pool"
    linear_probability: float = 0.0
    """Chance that a pool-mode spawn uses arcade-style linear motion."""
    specific_pattern: str = "sine"
    pattern_pool: list[str] = field(default_factory=lambda: list(PATTERN_NAMES))
    amplitude_min: float = 28.0
    amplitude_max: float = 85.0
    wavelength_min: float = 1.7
    wavelength_max: float = 4.5
    heading_mode: str = "aimed"
    """Fresh-spawn heading: ``aimed``, ``spread``, or uniformly ``random``."""
    medium_speed_multiplier: float = 1.0
    small_speed_multiplier: float = 1.2
    fragment_motion: str = "random"
    """How a split fragment moves: ``random`` draws speed, pattern, waviness and phase
    afresh from the RNG at the moment of the shot; ``inherit`` keeps the parent's, with
    speed rescaled by the size multipliers, as the arcade original does.

    Measured 2026-09-02 on round 26 of the survival-v2 ladder: 19 of the champion's 23
    deaths were fragments of a rock it had itself just shot, most under 0.42s old. With
    ``random`` a point-blank shot at a medium or large rock is a gamble on a hidden draw the
    ship has no time to react to; a planner that could peek at that draw cleared 0.969 where
    one that could not cleared 0.812. ``inherit`` makes every fragment's path a function of
    what was already on screen, so there are no hidden-information deaths to learn around.
    """
    # An in-episode difficulty ramp. Every episode starts at the *_start values and reaches
    # the configured values after ramp_seconds. Any start left unset holds that field
    # constant, so a config with no starts behaves exactly as it did before.
    spawn_interval_start: float | None = None
    active_cap_start: int | None = None
    min_speed_start: float | None = None
    max_speed_start: float | None = None
    amplitude_max_start: float | None = None
    wavelength_min_start: float | None = None
    wavelength_max_start: float | None = None
    spawn_spread: float = 0.0
    """Degrees of random scatter added to a spawn's heading.

    Zero aims every asteroid straight at the arena centre, which is punishing for a
    stationary ship because nothing ever drifts harmlessly past. The arcade game scattered
    them; widening this makes the field survivable, so narrowing it over a run is itself a
    difficulty ramp.
    """
    spawn_spread_start: float | None = None
    ramp_seconds: float = 60.0
    ramp_step_seconds: float = 0.0
    """Hold difficulty constant for this many seconds, then jump to the next step.

    Zero ramps continuously. Stepping makes the ramp legible while playing and keeps the
    observation distribution stationary within a step, which is easier for a learner than
    difficulty that drifts under it on every frame.
    """
    variety_probability: float = 0.0
    """Chance that a spawning asteroid is drawn slow instead of at the round's difficulty.

    A round whose every rock is fast, wide-swinging, and short-period is a narrow
    distribution, and a policy can pass it by assuming those properties rather than by
    reading each rock. When this fires, the asteroid's speed and amplitude are scaled down
    and its period stretched by the same factor, so it is coherently sluggish rather than
    randomly inconsistent. Zero keeps every rock at the round's difficulty.
    """
    variety_scale_min: float = 0.25
    """Slowest draw a varied asteroid can take, as a fraction of the round's difficulty."""

    endless_pressure_per_minute: float = 0.0
    """Growth that continues after the ramp finishes, so no run is survivable forever.

    Once ``ramp_seconds`` has elapsed every further minute adds this fraction to a pressure
    multiplier: speeds are multiplied by it and the spawn interval divided by it. Amplitude,
    active cap, and spread stop at their ramp targets, because those are bounded by the
    arena and by the fixed observation layout. Zero keeps the historical behaviour, where
    difficulty freezes once the ramp completes.
    """

    def tier_elapsed(self, elapsed: float) -> float:
        """Elapsed time snapped back to the start of the difficulty step now in force."""
        if self.ramp_step_seconds <= 0:
            return max(0.0, elapsed)
        return (self.tier_at(elapsed) - 1) * self.ramp_step_seconds

    def tier_at(self, elapsed: float) -> int | None:
        """One-based difficulty step in force; ``None`` when the ramp is continuous."""
        if self.ramp_step_seconds <= 0:
            return None
        return 1 + int(max(0.0, elapsed) // self.ramp_step_seconds)

    def ramp_progress(self, elapsed: float) -> float:
        """0 at the start of an episode, 1 once the ramp has fully played out."""
        if self.ramp_seconds <= 0:
            return 1.0
        return min(1.0, max(0.0, self.tier_elapsed(elapsed) / self.ramp_seconds))

    def endless_pressure(self, elapsed: float) -> float:
        """Difficulty multiplier for the open-ended phase after the ramp has finished."""
        if self.endless_pressure_per_minute <= 0:
            return 1.0
        return 1.0 + self.endless_pressure_per_minute * max(
            0.0, self.tier_elapsed(elapsed) - self.ramp_seconds) / 60.0

    @property
    def is_ramped(self) -> bool:
        """Whether difficulty moves during an episode at all."""
        return self.endless_pressure_per_minute > 0 or self.ramp_step_seconds > 0 or any(
            getattr(self, f"{name}_start") is not None
            for name in ("spawn_interval", "active_cap", "min_speed", "max_speed",
                         "amplitude_max", "wavelength_min", "wavelength_max", "spawn_spread"))

    def difficulty_at(self, elapsed: float) -> "Difficulty":
        """Every ramped knob, interpolated for a point in the episode."""
        progress = self.ramp_progress(elapsed)
        pressure = self.endless_pressure(elapsed)

        def value(name: str) -> float:
            end = getattr(self, name)
            start = getattr(self, f"{name}_start")
            return end if start is None else start + (end - start) * progress

        return Difficulty(
            # A floor on the interval keeps the spawner from looping forever on one frame.
            spawn_interval=max(1.0 / 60.0, value("spawn_interval") / pressure),
            active_cap=int(round(value("active_cap"))),
            min_speed=value("min_speed") * pressure,
            max_speed=value("max_speed") * pressure,
            amplitude_max=value("amplitude_max"),
            wave_threshold=int(round(value("wave_threshold"))),
            spawn_spread=value("spawn_spread"),
            wavelength_min=value("wavelength_min"),
            wavelength_max=value("wavelength_max"),
            tier=self.tier_at(elapsed),
        )

    def wave_size_for(self, wave: int) -> int:
        """Arcade growth: the first wave is ``wave_size``, each later one adds ``wave_growth``."""
        return min(self.wave_size_max, self.wave_size + self.wave_growth * max(0, wave - 1))


@dataclass(slots=True)
class ObjectiveConfig:
    protect: bool = False
    object_health: int = 10
    object_radius: float = 28.0
    max_steps: int | None = None
    max_waves: int | None = None


@dataclass(slots=True)
class ProjectileConfig:
    speed: float = 540.0
    lifetime: float = 1.45
    radius: float = 3.0


@dataclass(slots=True)
class GameConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    ship: ShipConfig = field(default_factory=ShipConfig)
    asteroid: AsteroidConfig = field(default_factory=AsteroidConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    projectile: ProjectileConfig = field(default_factory=ProjectileConfig)
    ships: list[ShipSpec] = field(default_factory=lambda: [
        ShipSpec("alpha", "human", "keyboard_1"), ShipSpec("beta", "heuristic")
    ])

    def validate(self) -> None:
        if not self.ships:
            raise ValueError("configuration must contain at least one ship")
        ids = [s.id for s in self.ships]
        if len(ids) != len(set(ids)):
            raise ValueError("ship ids must be unique")
        if self.ship.friendly_collisions not in {"off", "ships", "full"}:
            raise ValueError("friendly_collisions must be off, ships, or full")
        if self.ship.spawn_radius < 0:
            raise ValueError("ship spawn_radius cannot be negative")
        if self.asteroid.fragment_motion not in {"random", "inherit"}:
            raise ValueError("fragment_motion must be random or inherit")
        if self.asteroid.spawn_mode not in {"interval", "wave"}:
            raise ValueError("spawn_mode must be interval or wave")
        if self.asteroid.wave_size < 1 or self.asteroid.wave_size_max < 1:
            raise ValueError("wave sizes must be positive")
        if any(size not in {1, 2, 3} for size in self.asteroid.wave_composition):
            raise ValueError("wave_composition entries must be 1, 2, or 3")
        if self.asteroid.wave_spawn_interval < 0 or self.asteroid.wave_delay < 0:
            raise ValueError("wave timings cannot be negative")
        if self.asteroid.initial_wave_delay < 0:
            raise ValueError("initial_wave_delay cannot be negative")
        if self.asteroid.heading_mode not in {"aimed", "spread", "random"}:
            raise ValueError("heading_mode must be aimed, spread, or random")
        if not 0.0 <= self.asteroid.linear_probability <= 1.0:
            raise ValueError("linear_probability must be between 0 and 1")
        if self.asteroid.medium_speed_multiplier <= 0 or self.asteroid.small_speed_multiplier <= 0:
            raise ValueError("asteroid size speed multipliers must be positive")
        if not 0.0 <= self.asteroid.spawn_spread <= 360.0:
            raise ValueError("spawn_spread must be between 0 and 360 degrees")
        if self.asteroid.motion_mode not in {"linear", "specific", "pool"}:
            raise ValueError("motion_mode must be linear, specific, or pool")
        names = ([self.asteroid.specific_pattern] if self.asteroid.motion_mode == "specific"
                 else self.asteroid.pattern_pool if self.asteroid.motion_mode == "pool" else [])
        unknown = set(names) - set(PATTERN_NAMES)
        if unknown:
            raise ValueError(f"unknown asteroid patterns: {sorted(unknown)}")
        if self.asteroid.motion_mode == "pool" and not names:
            raise ValueError("pattern_pool cannot be empty")
        if self.asteroid.active_cap < 1 or self.arena.fps < 1:
            raise ValueError("active_cap and fps must be positive")
        if self.asteroid.spawn_interval <= 0:
            raise ValueError("spawn_interval must be positive")
        sizes = self.asteroid.spawn_size
        if isinstance(sizes, list):
            if not sizes or any(size not in {1, 2, 3} for size in sizes):
                raise ValueError("spawn_size list entries must be 1, 2, or 3")
        elif sizes is not None and sizes not in {1, 2, 3}:
            raise ValueError("spawn_size must be 1, 2, or 3")
        if self.asteroid.initial_asteroids < 0:
            raise ValueError("initial_asteroids cannot be negative")
        if self.asteroid.initial_asteroids > self.asteroid.active_cap:
            raise ValueError("initial_asteroids cannot exceed active_cap")
        if self.asteroid.spawn_safe_radius < 0 or self.asteroid.spawn_safe_seconds < 0:
            raise ValueError("spawn clearance cannot be negative")
        if self.asteroid.min_speed < 0 or self.asteroid.max_speed < self.asteroid.min_speed:
            raise ValueError("asteroid speed range is invalid")
        if self.asteroid.wavelength_min <= 0 or self.asteroid.wavelength_max < self.asteroid.wavelength_min:
            raise ValueError("asteroid wavelength range is invalid")
        if (self.asteroid.wavelength_min_start is not None
                and self.asteroid.wavelength_min_start <= 0):
            raise ValueError("wavelength_min_start must be positive")
        if self.asteroid.endless_pressure_per_minute < 0:
            raise ValueError("endless_pressure_per_minute cannot be negative")
        if self.asteroid.ramp_step_seconds < 0:
            raise ValueError("ramp_step_seconds cannot be negative")
        if self.objective.max_waves is not None and self.objective.max_waves < 1:
            raise ValueError("max_waves must be positive")
        for spec in self.ships:
            if spec.controller not in {
                    "human", "random", "heuristic", "closest", "pilot", "muzero", "ppo"}:
                raise ValueError(f"unknown controller for {spec.id}: {spec.controller}")


def _construct(cls: type, raw: dict[str, Any]):
    allowed = cls.__dataclass_fields__.keys()
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> GameConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    known = {"arena", "ship", "asteroid", "objective", "projectile", "ships"}
    if set(raw) - known:
        raise ValueError(f"unknown top-level fields: {sorted(set(raw) - known)}")
    cfg = GameConfig(
        arena=_construct(ArenaConfig, raw.get("arena", {})),
        ship=_construct(ShipConfig, raw.get("ship", {})),
        asteroid=_construct(AsteroidConfig, raw.get("asteroid", {})),
        objective=_construct(ObjectiveConfig, raw.get("objective", {})),
        projectile=_construct(ProjectileConfig, raw.get("projectile", {})),
        ships=[_construct(ShipSpec, x) for x in raw.get("ships", [])] or GameConfig().ships,
    )
    cfg.validate()
    return cfg
