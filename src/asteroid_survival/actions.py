from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FINE_LEFT = 3
    FINE_RIGHT = 4
    THRUST = 5
    LEFT_THRUST = 6
    RIGHT_THRUST = 7
    FIRE = 8
    LEFT_FIRE = 9
    RIGHT_FIRE = 10
    FINE_LEFT_FIRE = 11
    FINE_RIGHT_FIRE = 12
    THRUST_FIRE = 13
    LEFT_THRUST_FIRE = 14
    RIGHT_THRUST_FIRE = 15

    @property
    def turn(self) -> float:
        if self in (self.FINE_LEFT, self.FINE_LEFT_FIRE):
            return -0.15
        if self in (self.FINE_RIGHT, self.FINE_RIGHT_FIRE):
            return 0.15
        if self in (self.LEFT, self.LEFT_THRUST, self.LEFT_FIRE, self.LEFT_THRUST_FIRE):
            return -1.0
        if self in (self.RIGHT, self.RIGHT_THRUST, self.RIGHT_FIRE, self.RIGHT_THRUST_FIRE):
            return 1.0
        return 0.0

    @property
    def thrust(self) -> bool:
        return self in (self.THRUST, self.LEFT_THRUST, self.RIGHT_THRUST, self.THRUST_FIRE,
                        self.LEFT_THRUST_FIRE, self.RIGHT_THRUST_FIRE)

    @property
    def fire(self) -> bool:
        return self in (
            self.FIRE, self.LEFT_FIRE, self.RIGHT_FIRE,
            self.FINE_LEFT_FIRE, self.FINE_RIGHT_FIRE,
            self.THRUST_FIRE, self.LEFT_THRUST_FIRE, self.RIGHT_THRUST_FIRE,
        )
