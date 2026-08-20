from __future__ import annotations

from dataclasses import replace

import pytest

from asteroid_survival import Action, GameConfig, Simulation
from asteroid_survival.config import AsteroidConfig, ObjectiveConfig, ShipConfig, ShipSpec
from asteroid_survival.math2d import Vec2
from asteroid_survival.patterns import lateral_motion, trajectory
from asteroid_survival.simulation import _Projectile


def agent_config(**overrides) -> GameConfig:
    cfg = GameConfig(ships=[ShipSpec("one", "heuristic"), ShipSpec("two", "random")])
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def test_seed_and_actions_are_deterministic():
    cfg = agent_config(asteroid=AsteroidConfig(spawn_interval=0.05, active_cap=8))
    first, second = Simulation(cfg), Simulation(cfg)
    assert first.reset(42) == second.reset(42)
    for _ in range(100):
        actions = {"one": Action.THRUST_FIRE, "two": Action.LEFT}
        assert first.step(actions) == second.step(actions)


def test_different_seeds_change_spawns():
    cfg = agent_config(asteroid=AsteroidConfig(spawn_interval=0.01))
    a, b = Simulation(cfg), Simulation(cfg)
    a.reset(1)
    b.reset(2)
    assert a.step({}).snapshot.asteroids != b.step({}).snapshot.asteroids


def test_unknown_ship_action_and_finished_episode_errors():
    cfg = GameConfig(ships=[ShipSpec("one")], objective=ObjectiveConfig(max_steps=1))
    sim = Simulation(cfg)
    sim.reset(0)
    with pytest.raises(ValueError):
        sim.step({"ghost": Action.NOOP})
    result = sim.step({})
    assert result.truncated and result.terminal_reason.value == "step_limit"
    with pytest.raises(RuntimeError):
        sim.step({})


def test_stationary_ship_rotates_and_fires_but_does_not_translate():
    cfg = GameConfig(ships=[ShipSpec("one")], ship=ShipConfig(mobile=False))
    sim = Simulation(cfg)
    before = sim.reset(0).ships[0]
    after = sim.step({"one": Action.RIGHT_THRUST_FIRE}).snapshot.ships[0]
    assert (after.x, after.y) == (before.x, before.y)
    assert after.angle != before.angle
    assert len(sim.snapshot().projectiles) == 1


@pytest.mark.parametrize("pattern", [
    "sine", "zigzag", "sawtooth", "arc", "s_curve", "lane_change",
    "serpentine", "corkscrew", "figure_eight", "spiral",
])
def test_patterns_are_finite_and_deterministic(pattern):
    args = (Vec2(10, 20), Vec2(1, 0), 100, pattern, 1.4, 50, 2.2, 0.3)
    first = trajectory(*args)
    assert first == trajectory(*args)
    assert all(abs(value) < 100_000 for vector in first for value in (vector.x, vector.y))


def test_spiral_respects_configured_amplitude_at_long_survival_times():
    for elapsed in (10.0, 100.0, 10_000.0):
        offset, _ = lateral_motion("spiral", elapsed, 50.0, 2.0, 0.3)
        assert abs(offset) <= 50.0


def test_motion_modes_select_expected_patterns():
    for mode, expected in (("linear", {"linear"}), ("specific", {"arc"})):
        asteroid = AsteroidConfig(spawn_interval=0.01, motion_mode=mode, specific_pattern="arc")
        cfg = agent_config(asteroid=asteroid)
        sim = Simulation(cfg)
        sim.reset(4)
        patterns = {a.pattern for a in sim.step({}).snapshot.asteroids}
        assert patterns == expected
    asteroid = AsteroidConfig(spawn_interval=0.001, motion_mode="pool", pattern_pool=["sine", "zigzag"])
    sim = Simulation(agent_config(asteroid=asteroid))
    sim.reset(4)
    patterns = {a.pattern for a in sim.step({}).snapshot.asteroids}
    assert patterns <= {"sine", "zigzag"}


def test_asteroid_population_never_exceeds_cap():
    cfg = agent_config(asteroid=AsteroidConfig(spawn_interval=0.001, active_cap=3))
    sim = Simulation(cfg)
    sim.reset(7)
    for _ in range(50):
        state = sim.step({}).snapshot
        assert len(state.asteroids) <= 3


def test_configured_asteroid_spawn_size_is_used():
    cfg = agent_config(asteroid=AsteroidConfig(spawn_interval=0.001, spawn_size=3))
    sim = Simulation(cfg)
    sim.reset(4)

    state = sim.step({}).snapshot

    assert state.asteroids
    assert {asteroid.size for asteroid in state.asteroids} == {3}


def test_missing_actions_mean_noop():
    cfg = GameConfig(ships=[ShipSpec("one")], asteroid=AsteroidConfig(spawn_interval=999))
    first, second = Simulation(cfg), Simulation(cfg)
    first.reset(3); second.reset(3)
    assert first.step({}).snapshot == second.step({"one": Action.NOOP}).snapshot


def test_all_actions_are_accepted():
    for action in Action:
        cfg = GameConfig(ships=[ShipSpec("one")], asteroid=AsteroidConfig(spawn_interval=999))
        sim = Simulation(cfg)
        sim.reset(0)
        assert sim.step({"one": action}).snapshot.step == 1


def test_invulnerable_ship_and_asteroid_pass_through_each_other():
    cfg = GameConfig(
        ship=ShipConfig(invulnerable=True), ships=[ShipSpec("one")],
        asteroid=AsteroidConfig(spawn_interval=999, min_speed=0.0, max_speed=0.0,
                                motion_mode="linear"))
    sim = Simulation(cfg)
    state = sim.reset(0)
    ship = state.ships[0]
    asteroid = sim._spawn_asteroid(
        pos=Vec2(ship.x, ship.y), size=1, direction=Vec2(1, 0))
    sim._asteroids = [asteroid]

    state = sim.step({"one": Action.NOOP}).snapshot

    assert state.ships[0].alive
    assert len(state.asteroids) == 1


def test_projectile_splits_large_asteroid_within_cap():
    cfg = GameConfig(ships=[ShipSpec("one")], asteroid=AsteroidConfig(spawn_interval=999, active_cap=2))
    sim = Simulation(cfg)
    sim.reset(0)
    asteroid = sim._spawn_asteroid(pos=Vec2(100, 100), size=3, direction=Vec2(1, 0))
    sim._asteroids.append(asteroid)
    sim._projectiles.append(_Projectile(999, "one", Vec2(100, 100), Vec2()))
    result = sim.step({})
    assert len(result.snapshot.asteroids) == 2
    assert {a.size for a in result.snapshot.asteroids} == {2}
    assert sum(e.kind == "asteroid_split" for e in result.events) == 2


def test_asteroid_collision_destroys_ship_in_one_hit():
    cfg = GameConfig(ships=[ShipSpec("one")], asteroid=AsteroidConfig(spawn_interval=999))
    sim = Simulation(cfg)
    state = sim.reset(0)
    pos = Vec2(state.ships[0].x, state.ships[0].y)
    sim._asteroids.append(sim._spawn_asteroid(pos=pos, size=1, direction=Vec2(1, 0)))
    result = sim.step({})
    assert result.terminated
    assert not result.snapshot.ships[0].alive


def test_protected_object_takes_size_based_damage():
    cfg = GameConfig(ships=[ShipSpec("one")], asteroid=AsteroidConfig(spawn_interval=999),
                     objective=ObjectiveConfig(protect=True, object_health=10))
    sim = Simulation(cfg)
    sim.reset(0)
    center = Vec2(cfg.arena.width / 2, cfg.arena.height / 2)
    sim._asteroids.append(sim._spawn_asteroid(pos=center, size=3, direction=Vec2(1, 0)))
    assert sim.step({}).snapshot.objective.health == 7


def test_friendly_ship_collisions_are_configurable():
    for mode, should_die in (("off", False), ("ships", True), ("full", True)):
        cfg = GameConfig(ships=[ShipSpec("one"), ShipSpec("two")],
                         ship=ShipConfig(friendly_collisions=mode),
                         asteroid=AsteroidConfig(spawn_interval=999))
        sim = Simulation(cfg)
        sim.reset(0)
        sim._ships[1].pos = sim._ships[0].pos
        result = sim.step({})
        assert (not result.snapshot.ships[0].alive) is should_die


def test_random_heading_mode_is_not_aimed_at_the_centre():
    import math
    import statistics

    asteroid = AsteroidConfig(spawn_interval=999, heading_mode="random")
    sim = Simulation(GameConfig(ships=[ShipSpec("one")], asteroid=asteroid))
    sim.reset(8)
    centre = Vec2(450, 450)
    errors = []
    for _ in range(300):
        rock = sim._spawn_asteroid(size=3)
        aimed = math.atan2(centre.y - rock.pos.y, centre.x - rock.pos.x)
        actual = math.atan2(rock.vel.y, rock.vel.x)
        errors.append(abs((actual - aimed + math.pi) % (2 * math.pi) - math.pi))
    assert statistics.fmean(errors) > 1.3


def test_crashing_into_final_rock_does_not_clear_wave():
    asteroid = AsteroidConfig(spawn_mode="wave", spawn_interval=999, wave_size=1,
                              wave_size_max=1, spawn_size=1)
    objective = ObjectiveConfig(max_waves=1)
    sim = Simulation(GameConfig(ships=[ShipSpec("one")], asteroid=asteroid,
                                objective=objective))
    state = sim.reset(2)
    sim._wave = 1
    sim._wave_pending = 0
    sim._wave_clear_recorded = False
    position = Vec2(state.ships[0].x, state.ships[0].y)
    rock = sim._spawn_asteroid(pos=position, size=1, direction=Vec2(1, 0))
    rock.pattern = "linear"
    sim._asteroids = [rock]
    result = sim.step({"one": Action.NOOP})
    assert result.snapshot.terminal_reason.value == "all_ships_destroyed"
    assert not any(event.kind == "wave_cleared" for event in result.events)


def test_no_two_patterns_are_near_duplicates():
    """Every pattern should be its own shape, not a variation on a sine.

    Before the along-track rewrite, `figure_eight` was sin(x)cos(x) laterally -- exactly a
    half-amplitude sine at double frequency, correlating 1.000 with `sine` -- and eight
    pairs sat above 0.96. A lateral-only offset cannot express a loop, so every pattern was
    forced into the same family.
    """
    import math

    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.patterns import pattern_offset

    frequency = 2 * math.pi / 3.0
    samples = [i * 12.0 / 2000 for i in range(2000)]
    shapes = {}
    for name in PATTERN_NAMES:
        points = [pattern_offset(name, t, 1.0, frequency, 0.0) for t in samples]
        flat = [value for along, _, lateral, _ in points for value in (along, lateral)]
        mean = sum(flat) / len(flat)
        centred = [value - mean for value in flat]
        norm = math.sqrt(sum(v * v for v in centred)) or 1.0
        shapes[name] = [v / norm for v in centred]

    worst, pair = 0.0, ()
    for i, first in enumerate(PATTERN_NAMES):
        for second in PATTERN_NAMES[i + 1:]:
            similarity = abs(sum(a * b for a, b in zip(shapes[first], shapes[second])))
            if similarity > worst:
                worst, pair = similarity, (first, second)
    assert worst < 0.75, f"{pair[0]} and {pair[1]} are near-duplicates ({worst:.3f})"


def test_every_pattern_stays_inside_its_amplitude():
    """Amplitude has to stay a real bound, or a curriculum cannot control difficulty."""
    import math

    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.patterns import pattern_offset

    for name in PATTERN_NAMES:
        for phase in (0.0, 1.3, 4.1):
            excursions = [
                math.hypot(offset[0], offset[2])
                for offset in (pattern_offset(name, i * 0.02, 50.0, 2.0, phase)
                               for i in range(3000))
            ]
            assert max(excursions) <= 50.0 * 1.02, f"{name} strays beyond its amplitude"


def test_pattern_peak_speeds_stay_within_the_declared_factor():
    """PEAK_SPEED_FACTOR normalises the observation; it must remain an upper bound."""
    import math

    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.patterns import PEAK_SPEED_FACTOR, pattern_offset

    amplitude, frequency = 50.0, 2.0
    for name in PATTERN_NAMES:
        peak = max(
            math.hypot(offset[1], offset[3])
            for offset in (pattern_offset(name, i * 0.005, amplitude, frequency, 0.0)
                           for i in range(6000))
        )
        assert peak <= PEAK_SPEED_FACTOR * amplitude * frequency, name


def test_patterns_that_loop_move_along_the_drift_axis():
    """Loops, arcs, and figure eights need along-track motion; wiggles must not have it."""
    from asteroid_survival.patterns import pattern_offset

    def along_range(name):
        values = [pattern_offset(name, i * 0.02, 50.0, 2.0, 0.0)[0] for i in range(3000)]
        return max(values) - min(values)

    for name in ("arc", "corkscrew", "figure_eight", "spiral"):
        assert along_range(name) > 1.0, f"{name} should curve, not just slide sideways"
    for name in ("sine", "zigzag", "sawtooth", "s_curve", "lane_change", "serpentine"):
        assert along_range(name) == 0.0, f"{name} should be a pure lateral offset"


def test_the_erratic_patterns_sweep_wide_but_never_settle_into_a_period():
    """`serpentine` is the hardest of the ten, by measurement rather than by intent.

    What makes a pattern dangerous here is large coherent sweeps, not agitation. Two
    earlier designs -- high-frequency jitter, and jitter riding on a slow sweep -- were the
    *easiest* patterns to survive (67% completion for the greedy baseline against 21% for a
    plain sine), because small rapid shakes cover little ground. This pins the properties
    that made the accepted design hardest at 12%: it sweeps the full amplitude like a sine,
    turns in corners like a zigzag, and never repeats.
    """
    import math

    from asteroid_survival.patterns import GOLDEN, pattern_offset

    assert abs(GOLDEN - (1 + 5 ** 0.5) / 2) < 1e-12

    def samples(name, count=4000, dt=1 / 240):
        return [pattern_offset(name, i * dt, 1.0, 2 * math.pi / 3.0, 0.0)
                for i in range(count)]

    lateral = [point[2] for point in samples("serpentine")]

    # Full-width sweeps, like a sine and unlike a jitter.
    assert max(lateral) > 0.95 and min(lateral) < -0.95

    # Corners, not curves. Measured on the signed velocity, not its magnitude: at a
    # triangle's corner the direction reverses while the speed is unchanged, so comparing
    # speeds alone would report a sharp reversal as perfectly smooth.
    def sharpest_turn(name):
        velocities = [(p[1], p[3]) for p in samples(name)]
        return max(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(velocities, velocities[1:]))

    assert sharpest_turn("serpentine") > 4 * sharpest_turn("sine")

    # No period to learn: consecutive sweeps differ in length.
    turns = [i for i in range(1, len(lateral) - 1)
             if (lateral[i] - lateral[i - 1]) * (lateral[i + 1] - lateral[i]) < 0]
    gaps = [b - a for a, b in zip(turns, turns[1:])]
    assert len(gaps) > 6
    assert max(gaps) > 1.5 * min(gaps), "sweep lengths should vary, not repeat on a beat"

    # `sawtooth` is lopsided rather than jagged: a long drift one way and a fast run back.
    # It used to be a true sawtooth, which snapped back discontinuously -- a rock teleporting
    # across the arena. It is continuous now, so what is checked is the asymmetry.
    speeds = sorted(math.hypot(p[1], p[3]) for p in samples("sawtooth"))
    assert speeds[-1] > 3 * speeds[len(speeds) // 4], "the return should outpace the drift"


def test_no_pattern_ever_teleports():
    """Position must be continuous at frame rate: a rock flies, it does not jump.

    `sawtooth` used to be a true sawtooth, whose position snaps from one extreme to the
    other between consecutive frames. On screen that is a rock vanishing and reappearing
    across the arena, which no amount of tuning elsewhere can compensate for.
    """
    import math

    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.patterns import pattern_offset

    dt = 1 / 60
    for name in PATTERN_NAMES:
        for phase in (0.0, 1.1, 2.7, 5.3):
            points = [pattern_offset(name, i * dt, 100.0, 2 * math.pi / 3.0, phase)
                      for i in range(2400)]
            jump = max(math.hypot(b[0] - a[0], b[2] - a[2])
                       for a, b in zip(points, points[1:]))
            assert jump < 12.0, f"{name} moves {jump:.1f}px in one frame: that is a teleport"


def test_reported_velocity_matches_how_the_position_actually_changes():
    """Every pattern's derivative must be right, or observed velocity is a lie.

    The RL observation feeds asteroid velocity to the policy, and the greedy controller
    leads its shots with it. A pattern whose stated velocity disagrees with its motion
    silently poisons both.
    """
    import math

    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.patterns import pattern_offset

    dt = 1e-4
    for name in PATTERN_NAMES:
        for phase in (0.0, 2.2, 4.8):
            errors, scale = [], 1e-9
            for step in range(400):
                t = 0.05 + step * 0.017
                before = pattern_offset(name, t - dt, 50.0, 2.0, phase)
                after = pattern_offset(name, t + dt, 50.0, 2.0, phase)
                stated = pattern_offset(name, t, 50.0, 2.0, phase)
                measured = ((after[0] - before[0]) / (2 * dt),
                            (after[2] - before[2]) / (2 * dt))
                errors.append(math.hypot(measured[0] - stated[1], measured[1] - stated[3]))
                scale = max(scale, math.hypot(stated[1], stated[3]))
            # Piecewise shapes have corners and kinks where no single slope is correct, so
            # judge the bulk of the samples rather than the worst one.
            errors.sort()
            typical = errors[int(0.95 * len(errors))]
            assert typical < 0.02 * scale, (
                f"{name} reports a velocity its motion does not follow "
                f"({typical:.2f} against a peak speed of {scale:.0f})")


def test_brownian_wanders_without_a_shape_or_a_period():
    """It should read as aimless drift, not as any recognisable curve."""
    import math

    from asteroid_survival.patterns import pattern_offset

    def path(phase, count=6000):
        return [pattern_offset("brownian", i / 60, 1.0, 2 * math.pi / 3.0, phase)
                for i in range(count)]

    points = path(1.1)

    # Direction changes constantly and in both axes: it is a wander, not a wave.
    assert max(abs(p[0]) for p in points) > 0.1, "brownian should drift along its path too"
    lateral = [p[2] for p in points]
    turns = sum(1 for a, b, c in zip(lateral, lateral[1:], lateral[2:])
                if (b - a) * (c - b) < 0)
    assert turns > 20

    # No two asteroids wander alike: the phase seeds every component.
    other = path(4.7)
    overlap = sum(abs(a[2] - b[2]) for a, b in zip(points, other)) / len(points)
    assert overlap > 0.1, "different phases should produce visibly different wanders"
