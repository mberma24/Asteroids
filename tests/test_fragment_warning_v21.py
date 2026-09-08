"""Regression checks for opt-in firing warnings; legacy checkpoints remain unchanged."""
import copy
import math

from asteroid_survival.actions import Action
from asteroid_survival.config import GameConfig
from asteroid_survival.math2d import Vec2, wrapped_distance
from asteroid_survival.simulation import ASTEROID_RADII, Simulation, _Asteroid, _Projectile


def scene(*, mobile=True, cap=32):
    config = GameConfig()
    config.ship.mobile = mobile
    config.asteroid.fragment_motion = "inherit"
    config.asteroid.initial_asteroids = 0
    config.asteroid.active_cap = cap
    config.asteroid.spawn_interval = 1000.0
    sim = Simulation(config)
    sim.reset(1)
    ship = sim._ships[0]
    ship.pos, ship.vel, ship.angle = Vec2(200.0, 250.0), Vec2(200.0, 0.0), 0.0
    rock = _Asteroid(99, Vec2(360.0, 250.0), Vec2(1.0, 0.0), 0.0,
                     "linear", 0.0, 1.0, 0.0, 0.0, 3,
                     Vec2(360.0, 250.0), Vec2(0.0, 0.0))
    sim._asteroids = [rock]
    return sim


def test_corrected_warning_accounts_for_ship_travel_before_split():
    sim = scene()
    ship = sim._ships[0]
    old = sim.fire_consequence(ship.id, horizon=0)
    fixed = sim.fire_consequence(ship.id, horizon=0, corrected=True)
    assert old is not None and fixed is not None
    fork = copy.deepcopy(sim)
    actual = None
    for frame in range(120):
        result = fork.step({ship.id: Action.FIRE if frame == 0 else Action.NOOP})
        ids = {int(e.entity_id) for e in result.events if e.kind == "asteroid_split"}
        if ids:
            child = next(a for a in fork._asteroids if a.id in ids)
            actual = wrapped_distance(fork._ships[0].pos, child.pos,
                                      sim.config.arena.width, sim.config.arena.height)
            actual -= ASTEROID_RADII[child.size] + sim.config.ship.radius
            break
    assert actual is not None
    # Linear continuous hit time differs from the simulator's discrete collision frame.
    assert abs(fixed.worst_clearance - actual) < 2 * ship.vel.length() * sim.dt
    assert abs(old.worst_clearance - actual) > 15.0


def test_full_field_really_produces_one_child_and_warning_reports_it():
    sim = scene(cap=1)
    ship = sim._ships[0]
    old = sim.fire_consequence(ship.id)
    fixed = sim.fire_consequence(ship.id, corrected=True)
    assert old is not None and not old.splits  # legacy input semantics preserved
    assert fixed is not None and fixed.splits
    sim._projectiles = [_Projectile(100, ship.id, sim._asteroids[0].pos, Vec2(0.0, 0.0))]
    events = []
    sim._collisions(events)
    assert sum(e.kind == "asteroid_split" for e in events) == 1
    assert len(sim._asteroids) == 1 and sim._asteroids[0].size == 2


def test_prediction_is_read_only_and_legacy_is_default():
    sim = scene()
    before = copy.deepcopy(sim.__dict__)
    prediction = sim.fire_consequence(sim._ships[0].id)
    assert prediction == sim.fire_consequence(sim._ships[0].id, corrected=False)
    sim.fire_consequence(sim._ships[0].id, corrected=True)
    assert sim.snapshot() == SimulationSnapshot(before, sim)
    assert sim._rng.getstate() == before["_rng"].getstate()
    assert sim._asteroids == before["_asteroids"]
    assert sim._ships == before["_ships"]


def SimulationSnapshot(state, template):
    original = copy.copy(template)
    original.__dict__ = state
    return original.snapshot()



def test_v11_appends_action_warnings_without_changing_v10_or_task():
    from asteroid_survival.rl.curriculum import load_curriculum, task_hash
    from asteroid_survival.rl.environment import FIRING_ACTIONS, ACTION_FIRE_CONSEQUENCE_FEATURES
    from asteroid_survival.rl.ppo import _stage_env
    base = load_curriculum("configs/rl-survival-v3.toml")
    spec = load_curriculum("configs/rl-survival-v3-action-fire.toml")
    assert spec.observation_version == 11
    assert spec.stages == base.stages and spec.reward == base.reward
    assert task_hash(spec) == task_hash(base)
    layout = {"history_frames": 8, "history_long_frames": 8,
              "history_long_stride": 8, "max_projectiles": 8}
    older = _stage_env(base, 28, {**layout, "version": 10})
    newer = _stage_env(spec, 28, {**layout, "version": 11})
    old_obs, _ = older.reset(1000001400)
    new_obs, _ = newer.reset(1000001400)
    assert len(new_obs) - len(old_obs) == ACTION_FIRE_CONSEQUENCE_FEATURES == 32
    saw_prediction = False
    for index in range(80):
        assert (new_obs[:len(old_obs)] == old_obs).all()
        block = new_obs[-32:].reshape(8, 4)
        for row, action in zip(block, FIRING_ACTIONS):
            prediction = newer.simulation.fire_consequence(
                newer.agent_id, within_frames=newer.frame_skip,
                turn=action.turn, thrust=action.thrust, corrected=True)
            assert row[0] == float(prediction is not None)
            if prediction is not None:
                saw_prediction = True
                assert row[1] == float(prediction.splits)
                assert math.isclose(row[2], max(-1, min(1, prediction.worst_clearance / 150)),
                                    abs_tol=1e-6)
        a = index % len(newer.actions)
        old_obs, _, done, truncated, _ = older.step(a)
        new_obs, _, new_done, new_truncated, _ = newer.step(a)
        assert (done, truncated) == (new_done, new_truncated)
        if done or truncated:
            break
    assert saw_prediction

