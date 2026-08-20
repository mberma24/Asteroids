from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        n = self.length()
        return self * (1.0 / n) if n else Vec2()

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def limited(self, maximum: float) -> "Vec2":
        n = self.length()
        return self * (maximum / n) if n > maximum else self


def from_angle(angle: float) -> Vec2:
    return Vec2(math.cos(angle), math.sin(angle))


def wrap(v: Vec2, width: float, height: float) -> Vec2:
    return Vec2(v.x % width, v.y % height)


def wrapped_delta(a: Vec2, b: Vec2, width: float, height: float) -> Vec2:
    dx = (b.x - a.x + width / 2) % width - width / 2
    dy = (b.y - a.y + height / 2) % height - height / 2
    return Vec2(dx, dy)


def wrapped_distance(a: Vec2, b: Vec2, width: float, height: float) -> float:
    return wrapped_delta(a, b, width, height).length()

