"""Asteroid trajectory shapes.

Each named pattern displaces an asteroid from the straight line it would otherwise fly.
The displacement has two components: ``lateral``, perpendicular to the drift direction, and
``along``, parallel to it. The along component is what makes genuinely different shapes
possible -- a lateral-only offset can only ever produce "constant forward speed plus a
wiggle", so circles, loops, and figure eights are unreachable without it.

The ten are deliberately spread across distinct families rather than being variations on a
sine:

    sine          smooth, even oscillation
    zigzag        constant lateral speed, instant reversals
    sawtooth      long drift across, fast run back, at irregular intervals
    brownian      an aimless wander with no shape and no period
    s_curve       one-time swing into a new lane, then straight
    lane_change   square steps that dwell in each lane
    serpentine    erratic full-width sweeps, jagged, never repeats -- the hardest
    arc           one wide, continuous, sweeping turn
    corkscrew     tight repeated loops
    figure_eight  a true figure eight, crossing its own path
    spiral        loops that widen from nothing to full amplitude

Every shape stays inside ``amplitude`` so a curriculum's amplitude setting remains a real
bound on how far an asteroid strays.

Two of them are deliberately unpredictable rather than merely fast. They are built from
frequencies with no common multiple, so the path never repeats however long it is watched,
and from triangle waves rather than sinusoids, so direction changes are abrupt corners that
cannot be extrapolated through. Everything here is still a pure function of time, amplitude,
frequency, and phase: erratic to watch, but exactly reproducible from a seed.
"""
from __future__ import annotations

import math

from .math2d import Vec2


GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0
"""Rate spacing for the erratic patterns.

The ratio has to be genuinely irrational, not a rounded decimal. Spacing the rates 0.80,
2.09, 3.38, 5.47 looked golden but every ratio was a exact fraction, so the combined path
quietly repeated every 48.8s. Powers of the golden ratio are the hardest numbers to
approximate with small fractions, which is exactly the property wanted here.
"""

_BROWNIAN_TERMS = 6
_BROWNIAN_BASE = 0.55
_BROWNIAN_ALONG = 0.45
"""How much of the wander runs along the direction of travel rather than across it.

Kept well under one. Wandering along the drift axis cancels forward progress, and a rock
that never closes is a rock that cannot threaten: an even split made this the easiest
pattern of the eleven by a wide margin. This keeps the drift visibly two-dimensional while
most of the movement still sweeps across the arena.
"""
_BROWNIAN_SCALE = (1.0 / math.sqrt(1.0 + _BROWNIAN_ALONG ** 2)) / sum(
    1.0 / (_BROWNIAN_BASE * GOLDEN ** index) for index in range(_BROWNIAN_TERMS))
"""1/f component weights, normalised so along and lateral together stay inside amplitude."""

PEAK_SPEED_FACTOR = 3.0
"""Largest displacement speed any pattern reaches, in units of ``amplitude * frequency``.

A plain sine peaks at exactly ``amplitude * frequency``; the sharper shapes reach more than
that (``serpentine`` and ``lane_change`` are the fastest, at about 2.6x). Observation
normalisation needs this, otherwise the fastest patterns saturate the velocity feature.
Measured by ``test_pattern_peak_speeds_stay_within_the_declared_factor``.
"""


def _triangle(x: float) -> tuple[float, float]:
    phase = (x / (2 * math.pi)) % 1.0
    value = 4 * abs(phase - 0.5) - 1
    derivative = (-2 / math.pi) if phase < 0.5 else (2 / math.pi)
    return value, derivative


def _ramp(x: float, rise: float = 0.82) -> tuple[float, float]:
    """An asymmetric triangle: a long ramp one way, a quick run back.

    Deliberately not a true sawtooth. A sawtooth's position jumps from one extreme to the
    other between one frame and the next, which on screen is a rock teleporting across the
    arena rather than flying. This keeps the lopsided ramp-and-return feel while staying
    continuous, so the return is fast but flyable.
    """
    phase = (x / (2 * math.pi)) % 1.0
    if phase < rise:
        return 2 * phase / rise - 1, 2 / (rise * 2 * math.pi)
    return 1 - 2 * (phase - rise) / (1 - rise), -2 / ((1 - rise) * 2 * math.pi)


def _loop(x: float, radius: float, rate: float, frequency: float
          ) -> tuple[float, float, float, float]:
    """A circle of the given radius, traced at ``rate`` turns per pattern period.

    Returned as an along/lateral offset pair so the asteroid genuinely curves round rather
    than sliding sideways. ``1 - cos`` keeps the whole loop on one side of the drift line,
    so the excursion is bounded by twice the radius.
    """
    angle = rate * x
    spin = rate * frequency
    return (radius * math.sin(angle), radius * math.cos(angle) * spin,
            radius * (1 - math.cos(angle)), radius * math.sin(angle) * spin)


def pattern_offset(name: str, t: float, amplitude: float, frequency: float, phase: float
                   ) -> tuple[float, float, float, float]:
    """Along offset, along velocity, lateral offset, lateral velocity for one pattern."""
    x = frequency * t + phase
    a, w = amplitude, frequency

    if name == "sine":
        return 0.0, 0.0, a * math.sin(x), a * w * math.cos(x)

    if name == "zigzag":
        value, slope = _triangle(x)
        return 0.0, 0.0, a * value, a * slope * w

    if name == "sawtooth":
        # A long drift across and a fast run back, with the teeth stretched and squeezed by
        # a slow incommensurate wobble so the returns never fall on a beat.
        rate = 0.7
        wobble = rate / GOLDEN
        warp = 0.38 * math.sin(wobble * x)
        dwarp = 0.38 * wobble * math.cos(wobble * x)
        value, slope = _ramp(rate * x + warp)
        return 0.0, 0.0, a * value, a * slope * (rate + dwarp) * w

    if name == "brownian":
        # An aimless wander: no shape to recognise, no period, and never the same twice,
        # because the component phases are derived from this asteroid's own phase.
        #
        # Built by spectral synthesis rather than by accumulating random steps, because a
        # pattern is a pure function of time -- there is nowhere to keep a walker's state,
        # and it has to return an exact velocity as well as a position. Amplitudes fall as
        # 1/f, which is what gives Brownian motion its character: mostly slow, large,
        # directionless drift with finer detail riding on top.
        along = along_velocity = lateral = lateral_velocity = 0.0
        for index in range(_BROWNIAN_TERMS):
            rate = _BROWNIAN_BASE * GOLDEN ** index
            weight = _BROWNIAN_SCALE / rate
            lateral_angle = rate * x + phase * (index + 1) * 1.37 + index * 2.11
            along_angle = rate * x + phase * (index + 1) * 0.83 + index * 4.19
            lateral += weight * math.sin(lateral_angle)
            lateral_velocity += weight * rate * w * math.cos(lateral_angle)
            along += _BROWNIAN_ALONG * weight * math.sin(along_angle)
            along_velocity += _BROWNIAN_ALONG * weight * rate * w * math.cos(along_angle)
        return a * along, a * along_velocity, a * lateral, a * lateral_velocity

    if name == "s_curve":
        # Long, rounded sweeps with a pause at each extreme: a lazy S rather than a wave.
        # It has to keep returning across the field -- a one-shot version let rocks settle
        # onto a straight line and drift away, which stalled even the greedy baseline.
        rate, softness = 0.8, 1.5
        inner = softness * math.sin(rate * x)
        tanh = math.tanh(inner)
        return (0.0, 0.0, a * tanh,
                a * softness * rate * w * math.cos(rate * x) * (1 - tanh * tanh))

    if name == "lane_change":
        # Near-square with a long dwell: it parks in a lane, then snaps across. Run at half
        # rate so its fundamental differs from `sine` and the two do not read alike.
        rate, sharp = 0.5, 5.0
        inner = sharp * math.sin(rate * x)
        tanh = math.tanh(inner)
        return (0.0, 0.0, a * tanh,
                a * sharp * rate * w * math.cos(rate * x) * (1 - tanh * tanh))

    if name == "serpentine":
        # The hardest of the ten, and the only genuinely unpredictable one.
        #
        # A triangle wave whose rate is itself modulated, at an irrational ratio to its own
        # frequency. Full-amplitude sweeps like a sine, sharp corners like a zigzag, and no
        # period to learn: each swing is a different length from the last, forever.
        #
        # Two earlier designs were measured and rejected. High-frequency jitter, and jitter
        # over a slow sweep, both made this one of the *easiest* patterns to survive (67%
        # completion against the greedy baseline, versus 21% for a plain sine): small rapid
        # shakes cover little ground, and oscillating along the direction of travel cancels
        # forward progress. Difficulty here comes from large coherent sweeps, so that is
        # what this keeps -- it only makes their timing unguessable.
        rate, warp = 1.7, 1.6
        u = rate * x + warp * math.sin(rate * x / GOLDEN)
        du = rate * w * (1 + warp * math.cos(rate * x / GOLDEN) / GOLDEN)
        value, slope = _triangle(u)
        return 0.0, 0.0, a * value, a * slope * du

    if name == "arc":
        # A single wide sweeping turn: a slow circle, so within one crossing of the arena
        # it reads as a long curve rather than a loop.
        return _loop(x, a * 0.5, 0.45, w)

    if name == "corkscrew":
        # Tight, fast loops that visibly curl back on themselves.
        return _loop(x, a * 0.34, 1.6, w)

    if name == "figure_eight":
        # A real figure eight, laid out along the direction of travel: the long axis runs
        # forward and back while the short axis crosses at twice the rate, so the path
        # crosses itself. Previously this was sin(x)cos(x) laterally, which is exactly a
        # half-amplitude sine at double frequency -- indistinguishable from `sine`.
        return (a * math.sin(x), a * w * math.cos(x),
                a * 0.5 * math.sin(2 * x), a * w * math.cos(2 * x))

    if name == "spiral":
        # Loops that widen from nothing to the configured amplitude and then hold. The cap
        # keeps amplitude a real bound; without it difficulty drifted inside an episode.
        growth = min(1.0, t / 6.0)
        radius = a * 0.5 * growth
        rate = 1.0
        angle = rate * x
        spin = rate * w
        dradius = 0.0 if growth >= 1.0 else a * 0.5 / 6.0
        return (radius * math.sin(angle),
                dradius * math.sin(angle) + radius * math.cos(angle) * spin,
                radius * (1 - math.cos(angle)),
                dradius * (1 - math.cos(angle)) + radius * math.sin(angle) * spin)

    return 0.0, 0.0, 0.0, 0.0


def lateral_motion(name: str, t: float, amplitude: float, frequency: float, phase: float
                   ) -> tuple[float, float]:
    """The lateral component only, kept for callers that just want the sideways swing."""
    _, _, lateral, lateral_velocity = pattern_offset(name, t, amplitude, frequency, phase)
    return lateral, lateral_velocity


def trajectory(origin: Vec2, forward: Vec2, speed: float, name: str, t: float,
               amplitude: float, frequency: float, phase: float) -> tuple[Vec2, Vec2]:
    lateral = Vec2(-forward.y, forward.x)
    along, along_velocity, offset, offset_velocity = pattern_offset(
        name, t, amplitude, frequency, phase)
    return (origin + forward * (speed * t + along) + lateral * offset,
            forward * (speed + along_velocity) + lateral * offset_velocity)
