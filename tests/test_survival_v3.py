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
