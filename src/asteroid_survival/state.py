from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .config import Difficulty


class TerminalReason(StrEnum):
    ALL_SHIPS_DESTROYED = "all_ships_destroyed"
    OBJECT_DESTROYED = "object_destroyed"
    STEP_LIMIT = "step_limit"
    WAVES_CLEARED = "waves_cleared"


@dataclass(frozen=True, slots=True)
class ShipSnapshot:
    id: str
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    alive: bool
    radius: float
    cooldown: float = 0.0
    """Seconds remaining before this ship can fire again."""


@dataclass(frozen=True, slots=True)
class AsteroidSnapshot:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    size: int
    radius: float
    pattern: str


@dataclass(frozen=True, slots=True)
class ProjectileSnapshot:
    id: int
    owner_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    age: float = 0.0
    """Seconds since this projectile was fired."""


@dataclass(frozen=True, slots=True)
class ObjectiveSnapshot:
    enabled: bool
    x: float
    y: float
    radius: float
    health: int


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    step: int
    elapsed: float
    width: int
    height: int
    ships: tuple[ShipSnapshot, ...]
    asteroids: tuple[AsteroidSnapshot, ...]
    projectiles: tuple[ProjectileSnapshot, ...]
    objective: ObjectiveSnapshot
    terminated: bool
    truncated: bool
    terminal_reason: TerminalReason | None
    wave: int = 0
    """Waves released so far; 0 when the run does not use wave spawning."""
    difficulty: Difficulty | None = None
    """Asteroid settings in force right now; ``None`` when difficulty is fixed for the run."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GameEvent:
    kind: str
    step: int
    entity_id: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class StepResult:
    snapshot: WorldSnapshot
    events: tuple[GameEvent, ...]
    terminated: bool
    truncated: bool
    terminal_reason: TerminalReason | None
