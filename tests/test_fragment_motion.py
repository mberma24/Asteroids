"""Deterministic fragments: a split rock's pieces follow from the parent, not the RNG."""
import copy
import math
import random

from asteroid_survival.config import GameConfig
from asteroid_survival.math2d import Vec2, from_angle
from asteroid_survival.rl.curriculum import load_curriculum, task_hash
from asteroid_survival.simulation import Simulation, _Asteroid


def _parent(sim: Simulation) -> _Asteroid:
    return _Asteroid(99, Vec2(450.0, 450.0), Vec2(1.0, 0.0), 80.0, "sine", 40.0, 2.0, 1.25,
                     3.0, 3, Vec2(450.0, 450.0), Vec2(80.0, 0.0))


def _fragment_after_reseed(config: GameConfig, seed: int) -> _Asteroid:
    sim = Simulation(config)
    sim.reset(0)
    sim._rng = random.Random(seed)
    return sim._spawn_asteroid(pos=Vec2(450.0, 450.0), size=2,
                               direction=from_angle(0.45), parent=_parent(sim))


def test_inherited_fragments_do_not_depend_on_the_rng():
    config = GameConfig()
    config.asteroid.fragment_motion = "inherit"
    config.asteroid.medium_speed_multiplier = 1.15
    first, second = _fragment_after_reseed(config, 1), _fragment_after_reseed(config, 2)
    for child in (first, second):
        assert child.pattern == "sine"
        assert child.amplitude == 40.0 and child.frequency == 2.0 and child.phase == 1.25
        assert math.isclose(child.speed, 80.0 * 1.15)
        assert child.size == 2 and child.age == 0.0
    assert (first.speed, first.pattern, first.phase) == (second.speed, second.pattern, second.phase)


def test_random_fragments_still_depend_on_the_rng():
    config = GameConfig()
    first, second = _fragment_after_reseed(config, 1), _fragment_after_reseed(config, 2)
    assert (first.speed, first.pattern, first.phase) != (second.speed, second.pattern, second.phase)


def test_fragment_motion_is_validated():
    config = GameConfig()
    config.asteroid.fragment_motion = "sideways"
    try:
        config.validate()
    except ValueError as exc:
        assert "fragment_motion" in str(exc)
    else:
        raise AssertionError("an unknown fragment_motion must be rejected")


def test_the_detfrag_ladder_is_survival_v2_with_inherited_fragments():
    base = load_curriculum("configs/rl-survival-v2.toml")
    detfrag = load_curriculum("configs/rl-survival-v2-detfrag.toml")
    assert len(detfrag.stages) == len(base.stages) == 96
    assert detfrag.promotion_clear_rate == base.promotion_clear_rate
    assert detfrag.reward == base.reward
    for index in (0, 25, 95):
        ours = detfrag.stages[index].game_config(detfrag.base)
        theirs = base.stages[index].game_config(base.base)
        assert ours.asteroid.fragment_motion == "inherit"
        assert theirs.asteroid.fragment_motion == "random"
        ours.asteroid.fragment_motion = "random"
        assert ours == theirs
    # A different task, so a survival-v2 checkpoint must fork in rather than resume.
    assert task_hash(detfrag) != task_hash(base)


def test_adding_the_knob_left_every_existing_task_hash_alone():
    # The default is stripped from the hash payload, so checkpoints recorded before the
    # field existed still match their curriculum.
    spec = load_curriculum("configs/rl-survival-v2.toml")
    config = spec.stages[25].game_config(spec.base)
    from asteroid_survival.rl.curriculum import _specified
    assert "fragment_motion" not in _specified(config)["asteroid"]


def _round26_env():
    spec = load_curriculum("configs/rl-survival-v2-detfrag.toml")
    return spec.stages[25].game_config(spec.base)


def test_fire_consequence_matches_actually_firing():
    """Checked against the simulator, not against itself.

    The intersection solve extrapolates the rock linearly, so it is exact for a point-blank
    shot and degrades as the flight lengthens and curvature accumulates. Measured over 813
    predictions on round 26: 100% correct under 0.2s, 94-97% out to 0.6s. The regime that
    matters is the short one -- the shots whose fragments kill the ship land in well under
    half a second -- so that is what this pins.
    """
    from asteroid_survival.actions import Action

    config = _round26_env()
    checked = correct = close_ok = close_total = 0
    for seed in (10000, 10001, 10002):
        sim = Simulation(config)
        sim.reset(seed)
        for _ in range(240):
            if sim.terminated or sim.truncated:
                break
            ship = next(s for s in sim._ships if s.id == "agent")
            prediction = sim.fire_consequence("agent") if ship.cooldown <= 0 else None
            if prediction is not None:
                fork = copy.deepcopy(sim)
                struck = None
                for frame in range(int(round(prediction.time_to_hit * 60)) + 3):
                    if fork.terminated or fork.truncated:
                        break
                    result = fork.step({"agent": Action.FIRE if frame == 0 else Action.NOOP})
                    struck = struck or next(
                        (int(event.entity_id) for event in result.events
                         if event.kind == "asteroid_shot" and event.detail == "agent"), None)
                checked += 1
                correct += struck == prediction.target_id
                if prediction.time_to_hit <= 0.2:
                    close_total += 1
                    close_ok += struck == prediction.target_id
            sim.step({"agent": Action.NOOP})
    assert checked > 100, f"only {checked} predictions exercised"
    assert close_total > 20, f"only {close_total} point-blank predictions exercised"
    assert close_ok == close_total, f"point-blank agreement {close_ok}/{close_total}"
    assert correct / checked > 0.90, f"overall agreement {correct}/{checked}"


def test_a_shot_is_not_predicted_while_the_weapon_is_cooling():
    config = _round26_env()
    sim = Simulation(config)
    sim.reset(10000)
    next(s for s in sim._ships if s.id == "agent").cooldown = 0.2
    assert sim.fire_consequence("agent") is None
