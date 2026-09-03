"""The survival v3 ladder: rounds 1-27 unchanged, expressive patterns from round 28 on."""
import math

from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.environment import DIFFICULTY_SCALE_FEATURES
from asteroid_survival.rl.ppo import _stage_env

LAYOUT = {"history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
          "max_projectiles": 8}


def _v3():
    return load_curriculum("configs/rl-survival-v3.toml")


def test_v3_rounds_1_to_27_are_the_detfrag_ladder_unchanged():
    """The load-bearing one: a champion trained on rounds 1-27 must transfer untouched.

    Compared against rl-survival-v2-detfrag.toml, not rl-survival-v2.toml, because v3 keeps
    deterministic fragments -- that is the only intended difference from plain v2.
    """
    v3, detfrag = _v3(), load_curriculum("configs/rl-survival-v2-detfrag.toml")
    for index in range(27):
        assert (v3.stages[index].game_config(v3.base)
                == detfrag.stages[index].game_config(detfrag.base)), f"round {index + 1}"


def test_v3_is_96_rounds_with_continuous_names():
    v3 = _v3()
    assert len(v3.stages) == 96
    assert [stage.name for stage in v3.stages] == [
        f"survival-v3-round-{n}" for n in range(1, 97)]


def test_the_seam_at_round_28_matches_the_v2_ladder_exactly():
    """The new tier starts where the old one was, so the ladder has no step at round 28."""
    v3, v2 = _v3(), load_curriculum("configs/rl-survival-v2.toml")
    new, old = v3.stages[27], v2.stages[27]
    for field in ("min_speed", "max_speed", "amplitude_min", "amplitude_max",
                  "wavelength_min", "wavelength_max", "spawn_interval", "spawn_spread",
                  "initial_asteroids"):
        assert math.isclose(getattr(new, field), getattr(old, field), abs_tol=1e-4), field
    assert new.asteroid_size == old.asteroid_size


def test_the_composition_boundaries_are_where_v2_puts_them():
    v3 = _v3()
    sizes = {n: v3.stages[n - 1].asteroid_size for n in (28, 29, 52, 53, 82, 83, 96)}
    assert sizes[28] == [2, 2, 3, 3]
    assert sizes[29] == sizes[52] == 3
    assert sizes[53] == sizes[82] == [3, 3, 3, 2]
    assert sizes[83] is sizes[96] is None            # mixed


def test_the_new_tier_grows_the_pattern_and_reverses_the_period():
    """Amplitude rises, the period rises with it, and speed rises at half v2's rate."""
    v3, v2 = _v3(), load_curriculum("configs/rl-survival-v2.toml")
    tier = v3.stages[27:]
    for a, b in zip(tier, tier[1:]):
        assert b.amplitude_max > a.amplitude_max
        assert b.wavelength_min >= a.wavelength_min and b.wavelength_max >= a.wavelength_max
        assert b.min_speed >= a.min_speed and b.max_speed >= a.max_speed
        assert b.spawn_interval <= a.spawn_interval
        assert b.wavelength_max >= b.wavelength_min          # config.validate() would reject
    assert v3.stages[95].amplitude_max > 1.7 * v2.stages[95].amplitude_max
    assert v3.stages[95].wavelength_min > v3.stages[27].wavelength_min
    assert v3.stages[95].max_speed < 0.75 * v2.stages[95].max_speed


def test_the_swing_stays_inside_the_arena():
    """Amplitude is a sideways swing either way, so it has to stay well under the arena."""
    v3 = _v3()
    assert max(stage.amplitude_max for stage in v3.stages) <= 300.0
    assert v3.base.arena.width == v3.base.arena.height == 900


def test_the_spawn_guard_survives_the_extends_boundary():
    """Both guard knobs silently default to zero, which would disable spawn clearance."""
    v3 = _v3()
    for stage in v3.stages:
        assert stage.spawn_safe_radius == 60.0 and stage.spawn_safe_seconds == 1.8


def test_v9_appends_two_inputs_and_moves_nothing():
    v3 = _v3()
    eight = _stage_env(v3, 95, {**LAYOUT, "version": 8})
    nine = _stage_env(v3, 95, {**LAYOUT, "version": 9})
    assert nine.observation_size - eight.observation_size == DIFFICULTY_SCALE_FEATURES
    old, new = eight.reset(10000)[0], nine.reset(10000)[0]
    assert (old == new[:len(old)]).all(), "v9 moved an existing input"


def test_the_v9_inputs_never_saturate_on_this_ladder():
    """The point of the block: the v5 copies clamp from round 71 and stop distinguishing."""
    v3 = _v3()
    env = _stage_env(v3, 95, {**LAYOUT, "version": 9})
    observation, _ = env.reset(10000)
    assert 0.0 < observation[-2] < 1.0 and 0.0 < observation[-1] < 1.0
    top = v3.stages[95]
    assert top.amplitude_max / 300.0 < 1.0 and top.wavelength_max / 8.0 < 1.0
    # ...and the v5 originals really do clamp up there, which is why this block exists.
    assert top.amplitude_max / 200.0 > 1.0 and top.wavelength_max / 6.0 > 1.0


def test_tumble_is_playable_but_out_of_the_default_pool():
    """A new pattern must not restate the task of every ladder that names no patterns.

    `pattern_pool` is part of the task hash and defaults to `PATTERN_NAMES`, so appending to
    that tuple silently changes the task of every curriculum relying on the default and
    strands its checkpoints. `tumble` therefore sits outside it until a ladder opts in.
    """
    from asteroid_survival.config import (EXPERIMENTAL_PATTERNS, KNOWN_PATTERNS,
                                          PATTERN_NAMES, GameConfig)

    assert "tumble" in EXPERIMENTAL_PATTERNS and "tumble" in KNOWN_PATTERNS
    assert "tumble" not in PATTERN_NAMES
    assert list(GameConfig().asteroid.pattern_pool) == list(PATTERN_NAMES)
    # It is still a legal thing to ask a config for.
    config = GameConfig()
    config.asteroid.motion_mode = "pool"
    config.asteroid.pattern_pool = ["tumble", "sine"]
    config.validate()


def test_tumble_covers_ground_instead_of_shivering():
    """Agitation that covers nothing was measured to be the easiest thing in the pool."""
    import math
    import statistics

    from asteroid_survival.patterns import pattern_offset

    amplitude, frequency = 50.0, 2 * math.pi / 4.0
    offsets, turns, previous = [], 0, None
    for step in range(30000):
        _, _, lateral, lateral_velocity = pattern_offset(
            "tumble", step * 0.002, amplitude, frequency, 1.1)
        offsets.append(abs(lateral))
        rising = lateral_velocity > 0
        turns += previous is not None and rising != previous
        previous = rising
    # A sine sits 64% off the centreline and turns 0.5 times a second. This has to hold its
    # own on displacement while changing direction more often.
    assert statistics.fmean(offsets) / amplitude > 0.45
    assert turns / 60 > 0.7


def test_tumble_never_repeats():
    import math
    import statistics

    from asteroid_survival.patterns import pattern_offset

    def correlation(name):
        track = [pattern_offset(name, i * 0.01, 50.0, 2 * math.pi / 4.0, 1.1)[2]
                 for i in range(60000)]
        half = len(track) // 2
        first, second = track[:half], track[half:2 * half]
        mean_a, mean_b = statistics.fmean(first), statistics.fmean(second)
        top = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second))
        bottom = math.sqrt(sum((a - mean_a) ** 2 for a in first)
                           * sum((b - mean_b) ** 2 for b in second)) or 1.0
        return top / bottom

    assert correlation("sine") > 0.99          # the check detects a period when there is one
    assert abs(correlation("tumble")) < 0.25


def test_tumble_is_the_same_episode_on_every_machine():
    """Value noise from an integer hash, not from sin() of a large argument."""
    from asteroid_survival.patterns import _hash_unit, pattern_offset

    assert _hash_unit(12345, 7) == _hash_unit(12345, 7)
    assert _hash_unit(12345, 7) != _hash_unit(12346, 7)
    assert -1.0 <= _hash_unit(999, 3) <= 1.0
    first = pattern_offset("tumble", 3.25, 50.0, 2.0, 1.1)
    assert pattern_offset("tumble", 3.25, 50.0, 2.0, 1.1) == first


def test_tumble_is_dealt_more_room_than_the_patterns_meant_to_be_waves():
    """Amplitude is one budget for the whole round, and chaos needs more of it than a wave.

    At round 29 the round's amplitude tops out near 48px against a 900px arena, where every
    shape reads as a gentle wave. `PATTERN_REACH` boosts `tumble` at spawn, outside
    `pattern_offset`, so amplitude stays an exact bound on the pattern function itself.
    """
    import statistics

    from asteroid_survival.actions import Action
    from asteroid_survival.config import PATTERN_REACH, PATTERN_REACH_CAP
    from asteroid_survival.simulation import Simulation

    spec = _v3()
    dealt = {}
    for name in ("sine", "tumble"):
        config = spec.stages[28].game_config(spec.base)
        config.asteroid.motion_mode = "specific"
        config.asteroid.specific_pattern = name
        config.ship.invulnerable = True
        sim = Simulation(config)
        sim.reset(4)
        amplitudes = []
        for _ in range(900):
            sim.step({ship.id: Action.NOOP for ship in config.ships})
            amplitudes += [rock.amplitude for rock in sim._asteroids]
        dealt[name] = statistics.fmean(amplitudes)
    assert dealt["tumble"] > 2.5 * dealt["sine"]
    assert PATTERN_REACH["tumble"] == 3.0

    # The boost is capped, or a late round would sweep most of the arena.
    top = spec.stages[95].game_config(spec.base)
    top.asteroid.motion_mode = "specific"
    top.asteroid.specific_pattern = "tumble"
    top.ship.invulnerable = True
    sim = Simulation(top)
    sim.reset(4)
    for _ in range(600):
        sim.step({ship.id: Action.NOOP for ship in top.ships})
    ceiling = PATTERN_REACH_CAP * top.arena.width
    assert max(rock.amplitude for rock in sim._asteroids) <= ceiling + 1e-6


def test_the_reach_boost_leaves_every_other_pattern_alone():
    from asteroid_survival.actions import Action
    from asteroid_survival.simulation import Simulation

    spec = _v3()
    config = spec.stages[28].game_config(spec.base)
    config.ship.invulnerable = True
    sim = Simulation(config)
    sim.reset(9)
    for _ in range(900):
        sim.step({ship.id: Action.NOOP for ship in config.ships})
    # No stage names `tumble` yet, so nothing on the field may exceed the round's amplitude.
    assert all(rock.amplitude <= config.asteroid.amplitude_max + 1e-6
               for rock in sim._asteroids)
