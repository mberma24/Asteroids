from __future__ import annotations

import math
import statistics
import json
from collections import deque
from pathlib import Path

import pytest

from asteroid_survival.actions import Action
from asteroid_survival.controllers import ClosestAsteroidController
from asteroid_survival.config import (PATTERN_NAMES, AsteroidConfig, GameConfig,
                                      ShipConfig, ShipSpec)
from asteroid_survival.math2d import Vec2
from asteroid_survival.rl.environment import AsteroidsRLEnv, RewardConfig, encode_observation
from asteroid_survival.rl.evaluation import evaluate_policy
from asteroid_survival.simulation import Simulation, _Projectile


def test_arcade_curriculum_has_consistent_mobile_observations():
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.environment import RewardConfig

    spec = load_curriculum("configs/rl-curriculum.toml")
    shapes = set()
    for stage in spec.stages:
        env = AsteroidsRLEnv(
            stage.game_config(spec.base), max_decisions=stage.max_decisions,
            history_frames=8, history_long_frames=8, history_long_stride=8,
            reward_config=spec.reward)
        observation, _ = env.reset(3)
        shapes.add(observation.shape)
        assert env.num_actions == 16
        assert env.history_slots[-1] == 71
    assert shapes == {(1235,)}


def test_advanced_nonlinear_curriculum_is_progressive_and_learnable():
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.config import PATTERN_NAMES

    spec = load_curriculum("configs/rl-nonlinear.toml")
    foundation = load_curriculum("configs/rl-curriculum.toml")
    assert spec.stages[:28] == foundation.stages
    assert len(spec.stages) == 48
    added = spec.stages[28:]
    assert all(stage.movement_enabled for stage in added)
    assert all(stage.linear_probability == 0.0 for stage in added)
    assert all(stage.patterns == PATTERN_NAMES for stage in added)
    assert [stage.min_speed for stage in added] == [30.0 + i for i in range(20)]
    assert [stage.max_speed for stage in added] == [45.0 + i for i in range(20)]
    assert [stage.amplitude_min for stage in added] == [25.0 + 3 * i for i in range(20)]
    assert [stage.amplitude_max for stage in added] == [50.0 + 5 * i for i in range(20)]
    assert [stage.wavelength_min for stage in added] == pytest.approx(
        [3.0 - 0.02 * i for i in range(20)])
    assert [stage.wavelength_max for stage in added] == pytest.approx(
        [4.5 - 0.04 * i for i in range(20)])
    assert [len(stage.composition) for stage in added] == [
        3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11, 11]

    cooldown = spec.base.ship.fire_cooldown
    for stage in added:
        bodies = sum({1: 1, 2: 3, 3: 7}[size] for size in stage.composition)
        assert stage.max_seconds >= bodies / 0.10 * cooldown
        assert stage.wavelength_min > 0


def test_curriculum_bridges_target_count_before_new_mechanics():
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    triplet = [(3,), (3, 3), (3, 3, 3)]
    ladder = [(1,), (1, 1), (2,), (2, 1), (2, 2), (2, 2, 1),
              (3,), (3, 1), (3, 2), (3, 3)]
    assert [stage.composition for stage in spec.stages[:10]] == ladder
    # Windows are derived from what the greedy baseline measurably needs, not written by
    # hand, so assert the property rather than pinning values that are meant to be re-derived.
    assert all(stage.no_hit_seconds >= 20.0 for stage in spec.stages)
    assert [stage.miss_penalty for stage in spec.stages[:10]] == [0.03] * 10
    assert all((stage.promotion_accuracy or spec.promotion_accuracy) == 0.05
               for stage in spec.stages)
    assert spec.reward.fast_clear == 6.0
    assert spec.reward.accuracy == 2.0
    assert [stage.composition for stage in spec.stages[10:16]] == triplet * 2
    for stage in spec.stages[:10]:
        config = stage.game_config(spec.base)
        assert config.asteroid.min_speed == config.asteroid.max_speed == 0.0
        assert config.ship.mobile and config.ship.acceleration == 0.0
    assert all(stage.ship_invulnerable for stage in spec.stages[10:13])
    assert all(not stage.ship_invulnerable for stage in spec.stages[13:])
    assert all(not stage.movement_enabled for stage in spec.stages[:19])
    assert all(stage.movement_enabled for stage in spec.stages[19:])

    safe_env = AsteroidsRLEnv(spec.stages[10].game_config(spec.base))
    safe_observation, _ = safe_env.reset(1)
    assert safe_observation[9] == 0.0 and safe_observation[10] == 1.0
    mobile_env = AsteroidsRLEnv(spec.stages[19].game_config(spec.base))
    mobile_observation, _ = mobile_env.reset(1)
    assert mobile_observation[9] == 1.0 and mobile_observation[10] == 0.0


def test_wave_composition_spawns_the_configured_sizes_in_order():
    from asteroid_survival.simulation import Simulation

    config = wave_config(wave_composition=[2, 1], wave_size=2, wave_size_max=2,
                         wave_growth=0, min_speed=0.0, max_speed=0.0,
                         wave_spawn_interval=0.0, motion_mode="linear")
    simulation = Simulation(config)
    simulation.reset(4)
    for _ in range(5):
        state = simulation.step({"agent": Action.NOOP}).snapshot
    assert [asteroid.size for asteroid in state.asteroids] == [2, 1]
    assert all(asteroid.vx == asteroid.vy == 0.0 for asteroid in state.asteroids)


def test_training_seeds_never_touch_the_held_out_evaluation_band():
    from asteroid_survival.rl.training import training_seed

    eval_start, eval_count = 10_000, 32
    reserved = set(range(eval_start, eval_start + eval_count))
    # Far more episodes than the evaluation band's offset, which is what made the naive
    # "seed + episode" scheme walk straight onto the held-out seeds mid-run.
    seeds = [training_seed(0, index, eval_start, eval_count) for index in range(40_000)]

    assert not reserved & set(seeds)
    assert len(set(seeds)) == len(seeds)  # every episode still gets its own level
    assert seeds[:3] == [0, 1, 2]
    assert seeds[eval_start] == eval_start + eval_count  # jumps the band exactly
    assert seeds == sorted(seeds)


def test_training_seeds_are_unchanged_when_no_band_is_reserved():
    from asteroid_survival.rl.training import training_seed

    assert [training_seed(7, i, 10_000, 0) for i in range(4)] == [7, 8, 9, 10]


def test_resume_truncates_only_records_after_the_durable_checkpoint(tmp_path):
    import json
    from asteroid_survival.rl.training import truncate_log_after_checkpoint

    log = tmp_path / "training.jsonl"
    log.write_text("".join(json.dumps({"episode": episode}) + "\n"
                           for episode in (1, 2, 3, 4)))

    assert truncate_log_after_checkpoint(log, 2) == 2
    assert [json.loads(line)["episode"] for line in log.read_text().splitlines()] == [1, 2]


def test_every_stage_is_winnable_from_a_cold_start():
    """Each round must be clearable at a beginner's accuracy, not just at the baseline's.

    Sizing limits against the greedy controller (which shoots at ~0.69) once made stage 2
    impossible below 0.134 accuracy, so the wave-clear reward became unreachable and training
    stalled for thousands of episodes.
    """
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    cooldown = spec.base.ship.fire_cooldown
    for stage in spec.stages:
        bodies = sum(1 if size == 1 else (3 if size == 2 else 7) for size in stage.composition)
        needed = bodies / 0.10 * cooldown
        assert stage.max_seconds >= needed, (
            f"{stage.name}: {bodies} bodies needs {needed:.0f}s at 0.10 accuracy "
            f"but the limit is {stage.max_seconds:.0f}s")


def test_no_hit_timeout_ends_a_dry_spell_but_not_a_productive_run():
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    stage = spec.stages[0]
    env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=4,
                         max_decisions=stage.max_decisions, reward_config=spec.reward,
                         no_hit_seconds=stage.no_hit_seconds)
    env.reset(7)
    noop = env.actions.index(Action.NOOP)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(noop)
        done = terminated or truncated

    metrics = info["episode_metrics"]
    assert metrics["terminal_reason"] == "no_hit_timeout"
    assert metrics["stalled_out"]
    # Cut off by the dry-spell window, well before the episode limit.
    assert metrics["survival_time"] >= stage.no_hit_seconds
    assert metrics["survival_time"] < stage.no_hit_seconds + 5.0
    assert metrics["survival_time"] < stage.max_seconds


def test_no_hit_window_never_cuts_off_the_baseline():
    """Each stage's window must clear the baseline's worst legitimate dry spell.

    A 6s window measured on stationary rocks cut greedy off on 12 of 21 stages, because a
    curving target genuinely takes longer between hits.
    """
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    controller = ClosestAsteroidController()
    for stage in spec.stages:
        env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=4,
                             max_decisions=stage.max_decisions, reward_config=spec.reward,
                             no_hit_seconds=stage.no_hit_seconds)
        for seed in range(20000, 20006):
            env.reset(seed)
            done = False
            while not done:
                action = env.actions.index(controller.action(env.state, env.agent_id))
                _, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            assert info["episode_metrics"]["terminal_reason"] != "no_hit_timeout", (
                f"{stage.name}: window of {stage.no_hit_seconds}s cuts off competent play")


def test_thrust_actions_are_inert_until_movement_is_enabled():
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    for stage in spec.stages:
        env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=4)
        inert = env.inert_actions
        expected = [action.thrust for action in env.actions]
        if stage.movement_enabled:
            assert not any(inert), f"{stage.name}: nothing should be masked once thrust works"
        else:
            assert list(inert) == expected, f"{stage.name}: every thrust action is a duplicate"
            assert 0 < sum(inert) < len(inert)  # never mask everything


def test_masked_actions_are_never_searched_or_selected():
    muzero = muzero_module()
    import numpy as np

    env = AsteroidsRLEnv(rl_config(), frame_skip=4, max_asteroids=4)
    mask = tuple(action.thrust for action in env.actions)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=32), seed=0)
    observation, _ = env.reset(1)
    batch = np.stack([observation] * 6)

    actions, weights, _ = agent.search_batch(batch, explore=True, invalid_actions=mask)

    masked = [i for i, m in enumerate(mask) if m]
    assert not set(int(a) for a in actions) & set(masked)
    assert weights[:, masked].sum() == pytest.approx(0.0, abs=1e-6)
    # The remaining options still receive the whole budget.
    assert weights.sum(axis=1) == pytest.approx(np.ones(len(batch)), abs=1e-4)


def test_search_refuses_to_mask_every_action():
    muzero = muzero_module()

    env = AsteroidsRLEnv(rl_config(), frame_skip=4, max_asteroids=4)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)
    observation, _ = env.reset(1)

    with pytest.raises(ValueError, match="mask every action"):
        agent.search(observation, invalid_actions=(True,) * env.num_actions)


def test_mastery_gate_requires_two_good_fixed_evaluations():
    from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum

    manager = CurriculumManager(load_curriculum("configs/rl-curriculum.toml"), seed=1)
    good = [{"completion_rate": 0.9, "mean_accuracy": 0.6}] * 5
    assert not manager.consider_promotion(good)
    assert manager.stage == 0
    assert manager.consider_promotion(good)
    assert manager.stage == 1


def test_mastery_gate_accepts_two_passes_among_last_four_evaluations():
    from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum

    manager = CurriculumManager(load_curriculum("configs/rl-curriculum.toml"), seed=1)
    good = [{"completion_rate": 0.9, "mean_accuracy": 0.6}] * 5
    bad = [{"completion_rate": 0.2, "mean_accuracy": 0.0}] * 5
    assert not manager.consider_promotion(good)
    assert not manager.consider_promotion(bad)
    assert manager.consider_promotion(good)
    assert manager.stage == 1


def test_recovery_sampling_mixes_weak_current_and_other_stages():
    from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum

    manager = CurriculumManager(load_curriculum("configs/rl-curriculum.toml"), seed=7, stage=3)
    samples = [manager.sample_stage(focus_stage=0) for _ in range(10_000)]
    proportions = {stage: samples.count(stage) / len(samples) for stage in range(4)}
    assert proportions[0] == pytest.approx(0.40, abs=0.03)
    assert proportions[3] == pytest.approx(0.40, abs=0.03)
    assert proportions[1] + proportions[2] == pytest.approx(0.20, abs=0.03)


def test_final_stage_can_be_marked_mastered():
    from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    manager = CurriculumManager(spec, seed=1, stage=len(spec.stages) - 1)
    good = [{"completion_rate": 0.9, "mean_accuracy": 0.6}] * len(spec.stages)

    assert not manager.consider_promotion(good)
    assert not manager.mastered
    assert not manager.consider_promotion(good)  # no next stage, but training is complete
    assert manager.mastered


def test_curriculum_progress_graph_is_dependency_free(tmp_path):
    import json

    from asteroid_survival.rl.plotting import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    records = []
    for episode, completion in ((250, 0.2), (500, 0.7)):
        records.append({
            "episode": episode, "training_stage": 0,
            "stages": [{"name": "linear", "completion_rate": completion,
                        "mean_wave": 1.0, "mean_accuracy": 0.5,
                        "mean_mean_wave_clear_time": 20.0}],
        })
    (run / "evaluation.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = plot_progress(run, tmp_path / "progress.svg")
    assert output.read_text(encoding="utf-8").startswith("<svg")
    assert "linear" in output.read_text(encoding="utf-8")


def test_curriculum_progress_graph_adds_stages_after_promotion(tmp_path):
    import json

    from asteroid_survival.rl.plotting import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    first = {"name": "stationary", "completion_rate": 0.9, "mean_wave": 1.0,
             "mean_accuracy": 0.4, "mean_mean_wave_clear_time": 8.0}
    second = {"name": "linear", "completion_rate": 0.5, "mean_wave": 1.0,
              "mean_accuracy": 0.3, "mean_mean_wave_clear_time": 15.0}
    records = [
        {"episode": 250, "training_stage": 0, "stages": [first]},
        {"episode": 500, "training_stage": 1, "stages": [first, second]},
    ]
    (run / "evaluation.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    output = plot_progress(run, tmp_path / "progress.svg")
    svg = output.read_text(encoding="utf-8")
    assert "stationary" in svg and "linear" in svg


def test_mobile_observation_contains_ship_velocity():
    config = rl_config()
    config.ship.mobile = True
    env = AsteroidsRLEnv(config, frame_skip=1, max_decisions=50, max_asteroids=4)
    observation, _ = env.reset(7)
    assert observation.shape == (11 + 4 * 12 + 8 * 10,)
    assert observation[7] == observation[8] == 0.0
    assert observation[9] == 1.0  # movement enabled
    assert observation[10] == 0.0  # vulnerable
    observation, *_ = env.step(env.actions.index(Action.THRUST))
    assert abs(float(observation[7])) + abs(float(observation[8])) > 0.0


def test_arcade_reward_marks_a_shot_clear_as_success():
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    stage = spec.stages[0]
    env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=1,
                         max_decisions=stage.max_decisions, reward_config=spec.reward)
    env.reset(4)
    sim = env.simulation
    sim._wave = 1
    sim._wave_pending = 0
    sim._wave_clear_recorded = False
    rock = sim._spawn_asteroid(pos=Vec2(100, 100), size=1, direction=Vec2(1, 0))
    sim._asteroids = [rock]
    sim._projectiles = [_Projectile(999, "agent", Vec2(100, 100), Vec2())]
    env.state = sim.snapshot()

    _, reward, _, truncated, info = env.step(env.actions.index(Action.NOOP))

    assert truncated and reward > spec.reward.wave_clear
    assert info["episode_metrics"]["completed_stage"]
    assert info["episode_metrics"]["waves_cleared"] == 1


def test_terminal_projectiles_are_resolved_and_penalized_as_misses():
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    stage = spec.stages[0]
    env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=1,
                         max_decisions=stage.max_decisions, reward_config=spec.reward)
    env.reset(4)
    sim = env.simulation
    sim._wave = 1
    sim._wave_pending = 0
    sim._wave_clear_recorded = False
    rock = sim._spawn_asteroid(pos=Vec2(100, 100), size=1, direction=Vec2(1, 0))
    sim._asteroids = [rock]
    sim._projectiles = [
        _Projectile(998, "agent", Vec2(100, 100), Vec2()),
        _Projectile(999, "agent", Vec2(300, 300), Vec2()),
    ]
    assert env.metrics is not None
    env.metrics.shots_fired = 2
    env.state = sim.snapshot()

    _, _, terminated, truncated, info = env.step(env.actions.index(Action.NOOP))

    metrics = info["episode_metrics"]
    assert terminated or truncated
    assert metrics["shots_resolved"] == 2
    assert metrics["shots_missed"] == 1
    assert metrics["miss_penalty"] == pytest.approx(spec.reward.miss_penalty)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_arcade_miss_penalty_is_small_and_recorded():
    reward_config = RewardConfig(active_time_penalty=0.0)
    env = AsteroidsRLEnv(rl_config(), frame_skip=1, max_decisions=10,
                         reward_config=reward_config)
    env.reset(4)
    sim = env.simulation
    sim._projectiles = [
        _Projectile(999, "agent", Vec2(100, 100), Vec2(), sim.config.projectile.lifetime)
    ]
    env.state = sim.snapshot()

    _, reward, _, _, _ = env.step(env.actions.index(Action.NOOP))

    assert reward == pytest.approx(-0.02)
    assert env.metrics is not None
    assert env.metrics.shots_missed == 1
    assert env.metrics.miss_penalty == pytest.approx(0.02)


def test_aim_progress_reward_is_signed_and_cannot_be_farmed_by_oscillation():
    def turn_reward(action):
        config = rl_config()
        config.asteroid.min_speed = config.asteroid.max_speed = 0.0
        env = AsteroidsRLEnv(
            config, frame_skip=1, max_decisions=20,
            reward_config=RewardConfig(
                active_time_penalty=0.0, aim_progress=0.15))
        env.reset(4)
        assert env.state is not None
        ship = next(ship for ship in env.state.ships if ship.id == env.agent_id)
        target_angle = ship.angle + 0.5
        rock = env.simulation._spawn_asteroid(
            pos=Vec2(ship.x + 100 * math.cos(target_angle),
                     ship.y + 100 * math.sin(target_angle)),
            size=1, direction=Vec2())
        env.simulation._asteroids = [rock]
        env.state = env.simulation.snapshot()
        _, reward, _, _, _ = env.step(env.actions.index(action))
        assert env.metrics is not None
        return reward, env.metrics.aim_progress_reward

    left = turn_reward(Action.LEFT)
    right = turn_reward(Action.RIGHT)
    assert max(left[0], right[0]) > 0.0
    assert min(left[0], right[0]) < 0.0
    assert left[1] + right[1] == pytest.approx(0.0, abs=1e-6)


def test_arcade_timeout_has_lenient_distinct_penalty():
    reward_config = RewardConfig(active_time_penalty=0.0)
    env = AsteroidsRLEnv(rl_config(), frame_skip=1, max_decisions=1,
                         reward_config=reward_config)
    env.reset(4)

    _, reward, terminated, truncated, info = env.step(env.actions.index(Action.NOOP))

    assert not terminated and truncated
    assert reward == pytest.approx(-1.0)
    assert info["episode_metrics"]["timeout_penalty"] == pytest.approx(1.0)
    assert info["episode_metrics"]["terminal_reason"] == "evaluation_limit"


def rl_config() -> GameConfig:
    return GameConfig(
        ships=[ShipSpec("agent", "random")],
        ship=ShipConfig(mobile=False),
        asteroid=AsteroidConfig(spawn_interval=999),
    )


def test_rl_environment_has_fixed_observation_and_stationary_actions():
    env = AsteroidsRLEnv(rl_config(), frame_skip=2, max_decisions=3, max_asteroids=4)
    observation, _ = env.reset(7)
    assert observation.shape == (7 + 4 * 12 + 8 * 10,)
    assert env.num_actions == 10
    observation, reward, terminated, truncated, _ = env.step(0)
    assert observation.shape == (135,)
    assert reward > 0
    assert not terminated and not truncated


def test_observation_exposes_weapon_cooldown():
    config = rl_config()
    env = AsteroidsRLEnv(config, frame_skip=1, max_decisions=50, max_asteroids=4)
    observation, _ = env.reset(7)
    assert observation[5] == pytest.approx(0.0)  # ready to fire
    assert observation[6] == pytest.approx(1.0)

    fire = env.actions.index(Action.FIRE)
    observation, *_ = env.step(fire)

    assert observation[5] > 0.0  # cooldown running, so FIRE is a no-op
    assert observation[6] == pytest.approx(0.0)

    steps = math.ceil(config.ship.fire_cooldown * config.arena.fps)
    for _ in range(steps):
        observation, *_ = env.step(env.actions.index(Action.NOOP))
    assert observation[6] == pytest.approx(1.0)  # ready again


def test_bearing_features_are_relative_to_ship_heading():
    from asteroid_survival.rl.environment import SHIP_FEATURES, encode_observation

    config = rl_config()
    env = AsteroidsRLEnv(config, frame_skip=1, max_decisions=50, max_asteroids=4)
    env.reset(1)
    ship = next(s for s in env.simulation._ships if s.id == "agent")
    # Place an asteroid directly along +x, then point the ship straight at it.
    position = Vec2(ship.pos.x + 200.0, ship.pos.y)
    asteroid = env.simulation._spawn_asteroid(pos=position, size=1, direction=Vec2(1, 0))
    asteroid.pattern = "linear"
    env.simulation._asteroids.append(asteroid)

    def bearing_of(angle: float) -> tuple[float, float]:
        ship.angle = angle
        observation = encode_observation(
            env.simulation.snapshot(), "agent", config, 50, 4, 1)
        return float(observation[SHIP_FEATURES + 8]), float(observation[SHIP_FEATURES + 9])

    aligned_sin, aligned_cos = bearing_of(0.0)
    assert aligned_sin == pytest.approx(0.0, abs=1e-5)
    assert aligned_cos == pytest.approx(1.0, abs=1e-5)  # dead ahead

    # Rotating the ship must change the bearing even though the asteroid has not moved.
    turned_sin, turned_cos = bearing_of(math.pi / 2)
    assert turned_sin == pytest.approx(-1.0, abs=1e-5)
    assert turned_cos == pytest.approx(0.0, abs=1e-5)

    behind_sin, behind_cos = bearing_of(math.pi)
    assert behind_cos == pytest.approx(-1.0, abs=1e-5)


def test_observation_includes_active_projectile_state():
    from asteroid_survival.rl.environment import (ASTEROID_FEATURES, PROJECTILE_FEATURES,
                                                   SHIP_FEATURES)

    config = rl_config()
    env = AsteroidsRLEnv(
        config, frame_skip=1, max_decisions=50, max_asteroids=4, max_projectiles=2)
    observation, _ = env.reset(7)
    projectile_start = SHIP_FEATURES + 4 * ASTEROID_FEATURES
    assert observation.shape == (projectile_start + 2 * PROJECTILE_FEATURES,)
    assert observation[projectile_start] == pytest.approx(0.0)

    observation, *_ = env.step(env.actions.index(Action.FIRE))

    assert observation[projectile_start] == pytest.approx(1.0)  # occupied slot
    assert observation[projectile_start + 3] != 0.0  # projectile velocity
    assert 0.0 < observation[projectile_start + 8] < 1.0  # lifetime remaining
    assert observation[projectile_start + 9] == pytest.approx(1.0)  # owned by agent


def test_shot_penalty_is_subtracted_per_projectile():
    without = AsteroidsRLEnv(rl_config(), frame_skip=1, asteroid_reward=0.0)
    with_penalty = AsteroidsRLEnv(
        rl_config(), frame_skip=1, asteroid_reward=0.0, shot_penalty=0.25)
    fire = without.actions.index(Action.FIRE)
    without.reset(1)
    with_penalty.reset(1)

    _, plain_reward, *_ = without.step(fire)
    _, penalized_reward, *_ = with_penalty.step(fire)

    assert with_penalty.metrics is not None
    assert with_penalty.metrics.shots_fired == 1
    assert penalized_reward == pytest.approx(plain_reward - 0.25)
    assert with_penalty.metrics.shot_penalty == pytest.approx(0.25)


def test_shot_penalty_rejects_negative_values():
    with pytest.raises(ValueError):
        AsteroidsRLEnv(rl_config(), shot_penalty=-0.1)


def test_difficulty_ramp_interpolates_and_is_off_by_default():
    from asteroid_survival.config import AsteroidConfig

    steady = AsteroidConfig(spawn_interval=1.25, active_cap=32, min_speed=75.0, max_speed=145.0)
    for elapsed in (0.0, 999.0):  # unset starts mean constant, as before
        difficulty = steady.difficulty_at(elapsed)
        assert (difficulty.spawn_interval, difficulty.active_cap) == (1.25, 32)
        assert (difficulty.min_speed, difficulty.max_speed) == (75.0, 145.0)

    ramped = AsteroidConfig(
        spawn_interval=1.25, active_cap=32, min_speed=75.0, max_speed=145.0,
        amplitude_max=150.0, spawn_interval_start=4.0, active_cap_start=6,
        min_speed_start=40.0, max_speed_start=70.0, amplitude_max_start=60.0,
        ramp_seconds=60.0)

    start = ramped.difficulty_at(0.0)
    assert (start.spawn_interval, start.active_cap) == (4.0, 6)
    assert (start.min_speed, start.max_speed, start.amplitude_max) == (40.0, 70.0, 60.0)

    midpoint = ramped.difficulty_at(30.0)
    assert midpoint.spawn_interval == pytest.approx(2.625)
    assert midpoint.active_cap == 19
    assert midpoint.min_speed == pytest.approx(57.5)
    assert midpoint.max_speed == pytest.approx(107.5)
    assert midpoint.amplitude_max == pytest.approx(105.0)

    for elapsed in (60.0, 120.0):  # reaches the configured values, then clamps
        end = ramped.difficulty_at(elapsed)
        assert (end.spawn_interval, end.active_cap) == (1.25, 32)
        assert (end.min_speed, end.max_speed, end.amplitude_max) == (75.0, 145.0, 150.0)


def test_endless_pressure_keeps_growing_after_the_ramp_finishes():
    from asteroid_survival.config import AsteroidConfig

    endless = AsteroidConfig(
        spawn_interval=1.0, min_speed=100.0, max_speed=150.0, amplitude_max=150.0,
        active_cap=26, min_speed_start=30.0, max_speed_start=45.0,
        ramp_seconds=60.0, endless_pressure_per_minute=0.5)

    at_ramp_end = endless.difficulty_at(60.0)
    assert (at_ramp_end.min_speed, at_ramp_end.max_speed) == (100.0, 150.0)
    assert at_ramp_end.spawn_interval == pytest.approx(1.0)

    later = endless.difficulty_at(180.0)  # two extra minutes at +50% each
    assert later.min_speed == pytest.approx(200.0)
    assert later.max_speed == pytest.approx(300.0)
    assert later.spawn_interval == pytest.approx(0.5)
    # Bounded knobs stop at their targets: the arena and observation layout depend on them.
    assert later.active_cap == 26
    assert later.amplitude_max == pytest.approx(150.0)

    # The spawn interval never collapses below one frame, whatever the elapsed time.
    assert endless.difficulty_at(100_000.0).spawn_interval >= 1.0 / 60.0


def test_endless_pressure_is_off_by_default():
    from asteroid_survival.config import AsteroidConfig

    steady = AsteroidConfig(spawn_interval=1.25, min_speed=75.0, max_speed=145.0)
    assert steady.is_ramped is False
    assert steady.difficulty_at(10_000.0).spawn_interval == 1.25
    assert steady.difficulty_at(10_000.0).min_speed == 75.0


def test_wavelength_window_ramps_toward_faster_oscillation():
    from asteroid_survival.config import AsteroidConfig

    ramped = AsteroidConfig(
        wavelength_min=1.6, wavelength_max=2.6,
        wavelength_min_start=3.4, wavelength_max_start=5.2, ramp_seconds=60.0)
    assert ramped.is_ramped is True
    start = ramped.difficulty_at(0.0)
    assert (start.wavelength_min, start.wavelength_max) == (3.4, 5.2)
    end = ramped.difficulty_at(60.0)
    assert (end.wavelength_min, end.wavelength_max) == (1.6, 2.6)


def test_difficulty_steps_hold_still_between_tiers():
    from asteroid_survival.config import AsteroidConfig

    stepped = AsteroidConfig(
        min_speed=100.0, max_speed=150.0, min_speed_start=30.0, max_speed_start=45.0,
        ramp_seconds=120.0, ramp_step_seconds=20.0, endless_pressure_per_minute=0.35)

    # Difficulty is constant inside a tier and jumps at the boundary.
    assert stepped.difficulty_at(0.0) == stepped.difficulty_at(19.99)
    assert stepped.difficulty_at(20.0).min_speed > stepped.difficulty_at(19.99).min_speed
    assert [stepped.difficulty_at(t).tier for t in (0.0, 19.99, 20.0, 60.0, 120.0)] == [
        1, 1, 2, 4, 7]

    # Steps keep arriving after the ramp finishes, driven by endless pressure.
    assert stepped.difficulty_at(139.99) == stepped.difficulty_at(120.0)
    assert stepped.difficulty_at(140.0).min_speed > stepped.difficulty_at(120.0).min_speed

    # A continuous ramp is unchanged and reports no tier.
    continuous = AsteroidConfig(min_speed=100.0, min_speed_start=30.0, ramp_seconds=120.0)
    assert continuous.difficulty_at(10.0).tier is None
    assert continuous.difficulty_at(10.0) != continuous.difficulty_at(11.0)


def test_endless_config_starts_gently_and_never_stops_getting_harder():
    from asteroid_survival.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "endless.toml")
    assert config.asteroid.spawn_mode == "interval"
    assert set(config.asteroid.pattern_pool) == set(PATTERN_NAMES)
    assert config.objective.max_steps is None  # nothing truncates an endless run

    assert config.asteroid.ramp_step_seconds == 20.0  # difficulty arrives in visible steps
    opening = config.asteroid.difficulty_at(0.0)
    late = config.asteroid.difficulty_at(60.0)  # the ramp targets, reached at tier 4
    assert (opening.tier, late.tier) == (1, 4)
    assert late.amplitude_max == config.asteroid.amplitude_max
    assert late.spawn_spread == config.asteroid.spawn_spread
    assert opening.spawn_interval > late.spawn_interval
    assert opening.active_cap < late.active_cap
    assert opening.max_speed < late.max_speed
    assert opening.amplitude_max < late.amplitude_max
    assert opening.wavelength_min > late.wavelength_min
    assert opening.spawn_spread > late.spawn_spread  # wide scatter narrows onto the ship

    much_later = config.asteroid.difficulty_at(180.0)
    assert much_later.max_speed > late.max_speed
    assert much_later.spawn_interval < late.spawn_interval


def test_endless_simulation_spawns_faster_rocks_as_the_run_goes_on():
    from asteroid_survival.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "endless.toml")
    config.ship.invulnerable = True  # measure the ramp, not how long a still ship lives
    sim = Simulation(config)
    sim.reset(3)

    def mean_speed_over(seconds):
        speeds = []
        for _ in range(int(seconds * config.arena.fps)):
            state = sim.step({config.ships[0].id: Action.NOOP}).snapshot
            speeds.extend(math.hypot(a.vx, a.vy) for a in state.asteroids)
        return sum(speeds) / len(speeds)

    early = mean_speed_over(20.0)
    for _ in range(int(40.0 * config.arena.fps)):  # skip ahead two tiers
        sim.step({config.ships[0].id: Action.NOOP})
    late = mean_speed_over(20.0)
    assert late > early * 1.2


def test_ramp_delays_asteroid_pressure():
    from asteroid_survival.config import AsteroidConfig

    def count_after(config_asteroid, seconds):
        config = rl_config()
        config.asteroid = config_asteroid
        env = AsteroidsRLEnv(config, frame_skip=4, max_decisions=900)
        env.reset(5)
        peak = 0
        for _ in range(int(seconds * config.arena.fps / 4)):
            _, _, terminated, truncated, _ = env.step(0)
            assert env.state is not None
            peak = max(peak, len(env.state.asteroids))
            if terminated or truncated:
                break
        return peak

    steady = AsteroidConfig(spawn_interval=1.25, active_cap=32)
    ramped = AsteroidConfig(spawn_interval=1.25, active_cap=32, spawn_interval_start=4.0,
                            active_cap_start=6, ramp_seconds=60.0)
    assert count_after(ramped, 8.0) < count_after(steady, 8.0)


def wave_config(**overrides):
    from asteroid_survival.config import AsteroidConfig

    config = rl_config()
    # Wide scatter keeps the stationary test ship alive long enough to observe spawning.
    settings = dict(spawn_mode="wave", wave_size=4, wave_growth=2, wave_size_max=11,
                    wave_spawn_interval=0.4, wave_delay=1.0, wave_threshold=0,
                    active_cap=40, spawn_size=1, min_speed=60.0, max_speed=90.0,
                    spawn_spread=175.0)
    settings.update(overrides)
    config.asteroid = AsteroidConfig(**settings)
    return config


def test_wave_sizes_grow_like_the_arcade_game():
    asteroid = wave_config().asteroid
    assert [asteroid.wave_size_for(w) for w in range(1, 8)] == [4, 6, 8, 10, 11, 11, 11]


def test_a_wave_arrives_gradually_rather_than_all_at_once():
    from asteroid_survival.simulation import Simulation

    config = wave_config()
    simulation = Simulation(config)
    state = simulation.reset(4)
    assert state.wave == 0 and not state.asteroids

    counts = []
    for _ in range(int(1.5 * config.arena.fps)):
        state = simulation.step({"agent": Action.NOOP}).snapshot
        counts.append(len(state.asteroids))

    assert state.wave == 1
    # Four asteroids, arriving one at a time rather than appearing together.
    assert max(counts) == 4
    assert sorted(set(counts)) == [0, 1, 2, 3, 4]
    # The fourth cannot land before three spawn gaps have elapsed.
    first_full = counts.index(4)
    assert first_full >= 3 * config.asteroid.wave_spawn_interval * config.arena.fps


def test_no_new_wave_until_the_field_drops_to_the_threshold():
    from asteroid_survival.simulation import Simulation

    config = wave_config(wave_threshold=0)
    simulation = Simulation(config)
    simulation.reset(4)
    for _ in range(int(4 * config.arena.fps)):
        if simulation.terminated or simulation.truncated:
            break
        simulation.step({"agent": Action.NOOP})
    assert simulation._wave == 1
    assert simulation._wave_pending == 0

    # Asteroids are still up, so no second wave no matter how long we wait.
    for _ in range(int(20 * config.arena.fps)):
        if simulation.terminated or simulation.truncated:
            break
        simulation.step({"agent": Action.NOOP})
    assert simulation._wave == 1

    # Clear the field and the next wave follows, larger than the first.
    simulation._asteroids.clear()
    simulation.terminated = simulation.truncated = False
    for _ in range(int((config.asteroid.wave_delay + 1.0) * config.arena.fps)):
        simulation.step({"agent": Action.NOOP})
    assert simulation._wave == 2


def test_a_higher_threshold_releases_waves_while_asteroids_remain():
    from asteroid_survival.simulation import Simulation

    config = wave_config(wave_threshold=6)
    simulation = Simulation(config)
    simulation.reset(4)
    for _ in range(int(6 * config.arena.fps)):
        if simulation.terminated or simulation.truncated:
            break
        simulation.step({"agent": Action.NOOP})

    # Four asteroids is already at or below a threshold of six, so wave 2 comes anyway.
    assert simulation._wave >= 2


def test_wave_state_does_not_leak_across_resets():
    from asteroid_survival.simulation import Simulation

    config = wave_config()
    simulation = Simulation(config)
    simulation.reset(4)
    for _ in range(int(3 * config.arena.fps)):
        simulation.step({"agent": Action.NOOP})
    assert simulation._wave == 1

    state = simulation.reset(4)
    assert state.wave == 0
    assert simulation._wave_pending == 0 and simulation._wave_timer == 0.0


def test_spawn_spread_scatters_headings_away_from_the_centre():
    import math
    from asteroid_survival.simulation import Simulation

    def headings(spread):
        config = wave_config(spawn_spread=spread)
        simulation = Simulation(config)
        simulation.reset(1)
        centre = Vec2(config.arena.width / 2, config.arena.height / 2)
        errors = []
        for _ in range(200):
            asteroid = simulation._spawn_asteroid(size=3)
            to_centre = math.atan2(centre.y - asteroid.pos.y, centre.x - asteroid.pos.x)
            heading = math.atan2(asteroid.vel.y, asteroid.vel.x)
            errors.append(abs((heading - to_centre + math.pi) % (2 * math.pi) - math.pi))
        return errors

    aimed = headings(0.0)
    scattered = headings(170.0)
    assert max(aimed) < math.radians(20)  # every rock homes on the centre
    assert max(scattered) > math.radians(60)  # most drift wide of it
    assert statistics.fmean(scattered) > statistics.fmean(aimed)


def test_ramped_asteroids_spawn_slower_early_than_late():
    from asteroid_survival.config import AsteroidConfig
    from asteroid_survival.simulation import Simulation

    config = rl_config()
    config.asteroid = AsteroidConfig(
        spawn_interval=0.4, active_cap=40, min_speed=75.0, max_speed=145.0,
        min_speed_start=40.0, max_speed_start=70.0, ramp_seconds=60.0)
    simulation = Simulation(config)
    simulation.reset(9)

    def spawn_speeds(elapsed_seconds):
        simulation.step_count = int(elapsed_seconds * config.arena.fps)
        return [simulation._spawn_asteroid(size=3).vel.length() for _ in range(200)]

    early = spawn_speeds(0.0)
    late = spawn_speeds(60.0)

    # Ranges are sampled, so compare the bounds the ramp actually moves.
    assert max(early) == pytest.approx(70.0, abs=2.0)
    assert max(late) == pytest.approx(145.0, abs=3.0)
    assert min(early) == pytest.approx(40.0, abs=2.0)
    assert min(late) == pytest.approx(75.0, abs=3.0)


def test_history_tracks_each_asteroid_by_id():
    from asteroid_survival.rl.environment import (ASTEROID_FEATURES, SHIP_FEATURES,
                                                  feature_width)

    env = AsteroidsRLEnv(rl_config(), frame_skip=4, max_decisions=900, history_frames=3)
    plain = AsteroidsRLEnv(rl_config(), frame_skip=4, max_decisions=900)
    assert env.observation_size == plain.observation_size + env.max_asteroids * 3 * 2

    observation, _ = env.reset(3)
    width = feature_width(3)
    # Nothing has been seen yet, so every history slot starts empty rather than echoing "now".
    start = SHIP_FEATURES + ASTEROID_FEATURES
    assert list(observation[start:SHIP_FEATURES + width]) == [0.0] * 6

    for _ in range(30):
        observation, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert env.state is not None
    # Dead asteroids must not accumulate in the buffer.
    assert set(env._history) == {asteroid.id for asteroid in env.state.asteroids}


def test_long_term_history_reaches_further_back_for_the_same_slot_count():
    from asteroid_survival.rl.environment import history_offsets

    dense = history_offsets(16)
    tiered = history_offsets(8, 8, 8)
    assert len(dense) == len(tiered) == 16
    # Same budget, far deeper reach: 15 decisions back versus 71.
    assert max(dense) == 15
    assert max(tiered) == 71
    # The recent end stays dense, so fine detail for aiming is not traded away.
    assert tiered[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert sorted(tiered) == tiered  # strictly ordered oldest-last


def test_long_history_buffer_is_deep_enough_for_its_oldest_slot():
    env = AsteroidsRLEnv(rl_config(), frame_skip=4, history_frames=8,
                         history_long_frames=8, history_long_stride=8)
    plain = AsteroidsRLEnv(rl_config(), frame_skip=4)
    assert env.observation_size == plain.observation_size + env.max_asteroids * 16 * 2
    assert env._history_depth == 72  # must retain the oldest sampled decision

    env.reset(3)
    for _ in range(90):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert env.state is not None
    for track in env._history.values():
        assert len(track) <= 72


def test_long_history_slots_fill_in_over_time():
    from asteroid_survival.rl.environment import SHIP_FEATURES

    config = rl_config()
    # The default test config barely spawns, so give it asteroids to actually track.
    config.asteroid = AsteroidConfig(spawn_interval=0.3, active_cap=8, spawn_size=1,
                                     min_speed=40.0, max_speed=60.0, spawn_spread=175.0)
    env = AsteroidsRLEnv(config, frame_skip=4, history_frames=2,
                         history_long_frames=2, history_long_stride=10)
    # Slots are 0, 1, 11, 21 decisions back.
    assert env.history_slots == [0, 1, 11, 21]

    width = ASTEROID_FEATURES_COUNT + 2 * len(env.history_slots)

    def oldest_slots(observation):
        """The 21-decisions-back entry for every tracked asteroid."""
        for slot in range(env.max_asteroids):
            base = SHIP_FEATURES + slot * width + ASTEROID_FEATURES_COUNT + 2 * 3
            yield observation[base], observation[base + 1]

    observation, _ = env.reset(3)
    assert all(x == 0.0 and y == 0.0 for x, y in oldest_slots(observation))

    seen_far_history = False
    for _ in range(40):
        observation, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
        # The nearest asteroid is often newly spawned, so check every tracked slot.
        if any(x != 0.0 or y != 0.0 for x, y in oldest_slots(observation)):
            seen_far_history = True
    assert seen_far_history, "the 21-decision-old slot never populated"
    assert max(len(track) for track in env._history.values()) == env._history_depth


ASTEROID_FEATURES_COUNT = 12


def test_history_rejects_long_frames_without_short_ones():
    with pytest.raises(ValueError, match="requires history_frames"):
        AsteroidsRLEnv(rl_config(), history_frames=0, history_long_frames=4)
    with pytest.raises(ValueError):
        AsteroidsRLEnv(rl_config(), history_frames=4, history_long_stride=0)


def test_history_deltas_wrap_around_the_arena_edge():
    from asteroid_survival.rl.environment import ASTEROID_FEATURES, SHIP_FEATURES

    config = rl_config()
    env = AsteroidsRLEnv(config, frame_skip=1, max_asteroids=1, history_frames=1)
    env.reset(1)
    width, height = config.arena.width, config.arena.height
    # An asteroid that just crossed the right edge: a few pixels of travel, not a full arena.
    asteroid = env.simulation._spawn_asteroid(
        pos=Vec2(4.0, height / 2), size=1, direction=Vec2(1, 0))
    asteroid.pattern = "linear"
    env.simulation._asteroids.append(asteroid)
    env._history[asteroid.id] = deque([(width - 6.0, height / 2)], maxlen=1)

    observation = encode_observation(
        env.simulation.snapshot(), "agent", config, 900, 1, 1, env._history, 1)

    diagonal = math.hypot(width / 2, height / 2)
    dx = observation[SHIP_FEATURES + ASTEROID_FEATURES] * diagonal
    assert dx == pytest.approx(-10.0, abs=1e-3)  # ten pixels back, not ~900 forward
    assert abs(dx) < 50.0


def test_history_is_zero_when_disabled():
    env = AsteroidsRLEnv(rl_config(), frame_skip=4, history_frames=0)
    env.reset(1)
    for _ in range(5):
        env.step(0)
    assert env._history == {}


def test_history_frames_rejects_negative():
    with pytest.raises(ValueError):
        AsteroidsRLEnv(rl_config(), history_frames=-1)


def test_evaluation_reports_aggregate_metrics():
    env = AsteroidsRLEnv(rl_config(), frame_skip=1, max_decisions=2)
    report = evaluate_policy(env, lambda observation: 0, [3, 4])
    assert report["aggregate"]["episodes"] == 2
    assert report["aggregate"]["mean_survival_time"] > 0
    assert len(report["episodes"]) == 2


def test_asteroid_hit_adds_configured_reward():
    env = AsteroidsRLEnv(rl_config(), frame_skip=1, asteroid_reward=0.25)
    env.reset(1)
    position = Vec2(100, 100)
    asteroid = env.simulation._spawn_asteroid(pos=position, size=1, direction=Vec2(1, 0))
    asteroid.pattern = "linear"
    env.simulation._asteroids.append(asteroid)
    env.simulation._projectiles.append(_Projectile(999, "agent", position, asteroid.vel))

    _, reward, _, _, _ = env.step(0)

    assert env.metrics is not None
    assert env.metrics.asteroids_destroyed == 1
    assert env.metrics.asteroid_reward == 0.25
    assert reward == pytest.approx(1 / 60 + 0.25)


def muzero_module():
    return pytest.importorskip(
        "asteroid_survival.rl.muzero", reason="requires the optional rl extra")


def test_search_returns_valid_action_and_policy_over_repeated_calls():
    muzero = muzero_module()
    import numpy as np

    env = AsteroidsRLEnv(rl_config(), frame_skip=2, max_decisions=3, max_asteroids=4)
    observation, _ = env.reset(7)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)

    for _ in range(3):  # the compiled search is reused; it must stay correct across calls
        action, policy, value = agent.search(observation, explore=True)
        assert 0 <= action < env.num_actions
        assert policy.shape == (env.num_actions,)
        assert np.isfinite(policy).all()
        assert np.isfinite(value)


def test_greedy_search_is_deterministic_for_a_fixed_observation():
    muzero = muzero_module()

    env = AsteroidsRLEnv(rl_config(), frame_skip=2, max_decisions=3, max_asteroids=4)
    observation, _ = env.reset(7)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)

    actions = {agent.search(observation, explore=False)[0] for _ in range(4)}
    assert len(actions) == 1


def test_train_batch_reports_finite_losses_and_changes_parameters():
    muzero = muzero_module()
    import jax
    import numpy as np

    env = AsteroidsRLEnv(rl_config(), frame_skip=2, max_decisions=3, max_asteroids=4)
    observation, _ = env.reset(7)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)
    uniform = np.full(env.num_actions, 1 / env.num_actions, dtype=np.float32)
    sequence = [
        muzero.Transition(observation, 0, 1.0, uniform, observation, False, 1.0, uniform)
        for _ in range(agent.settings.unroll_steps + 1)
    ]
    batch = [list(sequence) for _ in range(8)]

    before = jax.tree.leaves(agent.params)[0].copy()
    losses = agent.train_batch(batch)
    after = jax.tree.leaves(agent.params)[0]

    assert set(losses) >= {"loss", "policy_loss", "value_loss", "reward_loss"}
    assert all(np.isfinite(value) for value in losses.values())
    assert not np.allclose(before, after)


def test_train_batch_handles_sequences_that_end_early():
    muzero = muzero_module()
    import numpy as np

    env = AsteroidsRLEnv(rl_config(), frame_skip=2, max_decisions=3, max_asteroids=4)
    observation, _ = env.reset(7)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)
    uniform = np.full(env.num_actions, 1 / env.num_actions, dtype=np.float32)

    def make(length):
        return [muzero.Transition(observation, 0, 1.0, uniform, observation, False, 1.0, uniform)
                for _ in range(length)]

    # A one-step sequence must not blow up: the missing unroll steps are masked out.
    losses = agent.train_batch([make(1), make(3), make(agent.settings.unroll_steps + 1)])

    assert all(np.isfinite(value) for value in losses.values())


def test_scalar_transform_round_trips():
    muzero = muzero_module()
    import numpy as np

    values = np.array([-50.0, -1.0, 0.0, 0.5, 7.0, 300.0], dtype=np.float32)
    restored = muzero.scalar_inverse(muzero.scalar_transform(values))
    np.testing.assert_allclose(np.asarray(restored), values, rtol=1e-3, atol=1e-3)


def test_n_step_targets_bootstrap_instead_of_summing_whole_episode():
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    observation = np.zeros(4, dtype=np.float32)
    transitions = [
        muzero.Transition(observation, 0, 1.0, uniform, observation, index == 19,
                          search_value=5.0)
        for index in range(20)
    ]

    muzero.finish_episode(transitions, discount=1.0, n_step=2, episode_id=7)

    # Two rewards of 1.0, then bootstrap off the stored search value, rather than
    # accumulating all 20 remaining rewards.
    assert transitions[0].value_target == pytest.approx(1.0 + 1.0 + 5.0)
    assert all(t.episode_id == 7 for t in transitions)
    # The tail has no step to bootstrap from, so it falls back to the remaining rewards.
    assert transitions[19].value_target == pytest.approx(1.0)


def test_sample_does_not_unroll_across_episode_boundaries():
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    observation = np.zeros(4, dtype=np.float32)
    buffer = muzero.ReplayBuffer(capacity=100, seed=0)
    for episode in (1, 2):
        buffer.extend([
            muzero.Transition(observation, 0, 1.0, uniform, observation, False,
                              episode_id=episode)
            for _ in range(5)
        ])

    for sequence in buffer.sample(40, unroll_steps=5):
        assert len({transition.episode_id for transition in sequence}) == 1


def test_replay_sampling_reserves_space_for_successful_episodes():
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    observation = np.zeros(4, dtype=np.float32)
    buffer = muzero.ReplayBuffer(capacity=200, seed=0)
    for index in range(100):
        buffer.extend([muzero.Transition(
            observation, 0, 0.0, uniform, observation, True,
            episode_id=index, successful=index < 5)])

    samples = buffer.sample(64)

    # All five rare successes are included; uniform replay would normally yield only ~3.
    assert sum(sequence[0].successful for sequence in samples) >= 5


def test_curriculum_replay_is_stage_balanced_for_storage_and_sampling():
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    observation = np.zeros(4, dtype=np.float32)
    buffer = muzero.ReplayBuffer(capacity=40, seed=0)
    for stage in (0, 1):
        buffer.extend([muzero.Transition(
            observation, 0, 0.0, uniform, observation, index == 29,
            episode_id=stage + 1, stage=stage)
            for index in range(30)])

    assert len(buffer) == 40
    assert {stage: sum(t.stage == stage for t in buffer.items) for stage in (0, 1)} == {
        0: 20, 1: 20}
    samples = buffer.sample(20, current_stage=1, current_fraction=0.60)
    assert sum(sequence[0].stage == 1 for sequence in samples) == 12
    assert sum(sequence[0].stage == 0 for sequence in samples) == 8


def test_replay_buffer_rejects_a_buffer_with_a_stale_observation_size(tmp_path):
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    buffer = muzero.ReplayBuffer(capacity=10, seed=0)
    buffer.extend([muzero.Transition(
        np.zeros(4, dtype=np.float32), 0, 1.0, uniform,
        np.zeros(4, dtype=np.float32), False)])
    path = buffer.save(tmp_path / "replay.npz")

    restored = muzero.ReplayBuffer(capacity=10, seed=0)
    assert restored.load(path, observation_size=99) == 0
    assert restored.load(path, observation_size=4) == 1


@pytest.mark.parametrize("history_frames", [0, 3, 8])
def test_play_controller_infers_history_length_from_the_checkpoint(tmp_path, history_frames):
    muzero = muzero_module()
    from asteroid_survival.rl.controller import MuZeroController

    config = rl_config()
    env = AsteroidsRLEnv(config, frame_skip=4, history_frames=history_frames)
    agent = muzero.MuZeroAgent(
        env.observation_size, env.num_actions,
        muzero.MuZeroSettings(num_simulations=8), seed=0)
    checkpoint = agent.save(tmp_path / "checkpoint", episodes=1)

    # Nothing tells the controller how the model was built; it reads it off the shapes.
    controller = MuZeroController(config, checkpoint, seed=0)
    assert controller.history_frames == history_frames

    snapshot = env.simulation.reset(4)
    for _ in range(12):
        action = controller.action(snapshot, "agent")
        assert action in controller.actions
        snapshot = env.simulation.step({"agent": action}).snapshot


def test_play_controller_rejects_an_incompatible_checkpoint(tmp_path):
    muzero = muzero_module()
    from asteroid_survival.rl.controller import MuZeroController

    config = rl_config()
    agent = muzero.MuZeroAgent(37, 6, muzero.MuZeroSettings(num_simulations=8), seed=0)
    checkpoint = agent.save(tmp_path / "checkpoint", episodes=1)

    with pytest.raises(ValueError, match="different observation layout"):
        MuZeroController(config, checkpoint, seed=0)


def test_replay_buffer_round_trips_through_disk(tmp_path):
    muzero = muzero_module()
    import numpy as np

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    observation = np.arange(4, dtype=np.float32)
    original = muzero.ReplayBuffer(capacity=10, seed=0)
    original.extend([
        muzero.Transition(observation, 2, 0.5, uniform, observation + 1, True, 1.5, uniform,
                          successful=True, stage=2),
        muzero.Transition(observation, 1, 0.25, uniform, observation + 2, False, 0.75, uniform,
                          stage=1),
    ])
    path = original.save(tmp_path / "replay.npz")

    restored = muzero.ReplayBuffer(capacity=10, seed=0)
    assert restored.load(path) == 2
    for source, target in zip(original.items, restored.items):
        assert source.action == target.action
        assert source.done == target.done
        assert source.reward == pytest.approx(target.reward)
        assert source.value_target == pytest.approx(target.value_target)
        assert source.successful == target.successful
        assert source.stage == target.stage
        np.testing.assert_allclose(source.observation, target.observation)
        np.testing.assert_allclose(source.next_observation, target.next_observation)


def test_replay_buffer_load_is_a_no_op_when_no_file_exists(tmp_path):
    muzero = muzero_module()
    buffer = muzero.ReplayBuffer(capacity=10, seed=0)
    assert buffer.load(tmp_path / "missing.npz") == 0
    assert buffer.items == []


def test_legacy_replay_recovers_stage_from_episode_log(tmp_path):
    muzero = muzero_module()
    import numpy as np

    path = tmp_path / "legacy.npz"
    observations = np.zeros((2, 4), dtype=np.float32)
    policies = np.full((2, 3), 1 / 3, dtype=np.float32)
    np.savez(
        path, observation=observations, action=np.zeros(2, dtype=np.int32),
        reward=np.zeros(2, dtype=np.float32), policy=policies,
        next_observation=observations, done=np.ones(2, dtype=bool),
        value_target=np.zeros(2, dtype=np.float32), next_policy=policies,
        search_value=np.zeros(2, dtype=np.float32), episode_id=np.asarray([10, 20]),
        successful=np.ones(2, dtype=bool))

    restored = muzero.ReplayBuffer(capacity=10, seed=0)
    assert restored.load(path, episode_stages={10: 1, 20: 3}) == 2
    assert [transition.stage for transition in restored.items] == [1, 3]


def test_resume_artifact_falls_back_to_checkpoint_run(tmp_path):
    from asteroid_survival.rl.training import _resume_artifact

    old_run = tmp_path / "old"
    checkpoint = old_run / "checkpoint_000250"
    checkpoint.mkdir(parents=True)
    (old_run / "replay.npz").write_bytes(b"replay")
    new_run = tmp_path / "new"
    new_run.mkdir()

    assert _resume_artifact(new_run, checkpoint, "replay.npz") == old_run / "replay.npz"

    (checkpoint / "curriculum_state.json").write_text("{}")
    assert (_resume_artifact(new_run, checkpoint, "curriculum_state.json")
            == checkpoint / "curriculum_state.json")


def test_checkpoint_retention_keeps_best_and_newest(tmp_path):
    import json
    from asteroid_survival.rl.training import prune_checkpoints

    for episode in (250, 500, 750, 1000, 1250):
        (tmp_path / f"checkpoint_{episode:06d}").mkdir()
    records = [
        {"episode": 250, "training_stage": 0,
         "stages": [{"completion_rate": 0.9, "mean_accuracy": 0.4}]},
        {"episode": 500, "training_stage": 1,
         "stages": [{}, {"completion_rate": 0.8, "mean_accuracy": 0.3}]},
        {"episode": 750, "training_stage": 1,
         "stages": [{}, {"completion_rate": 0.2, "mean_accuracy": 0.1}]},
    ]
    (tmp_path / "evaluation.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))

    removed = prune_checkpoints(tmp_path, keep=3)

    assert len(removed) == 2
    assert (tmp_path / "checkpoint_000500").is_dir()  # best at the furthest stage
    assert (tmp_path / "checkpoint_001000").is_dir()
    assert (tmp_path / "checkpoint_001250").is_dir()


def test_checkpoint_retention_does_not_duplicate_dedicated_champion(tmp_path):
    from asteroid_survival.rl.training import prune_checkpoints

    (tmp_path / "champion").mkdir()
    for episode in (500, 750, 1000):
        (tmp_path / f"checkpoint_{episode:06d}").mkdir()

    prune_checkpoints(tmp_path, keep=3)

    assert sorted(path.name for path in tmp_path.glob("checkpoint_*")) == [
        "checkpoint_000750", "checkpoint_001000"]


def test_champion_tracker_discovers_best_eligible_model_without_rolling_back_plateaus(tmp_path):
    import json
    from asteroid_survival.rl.training import ChampionTracker

    def record(episode, completion, accuracy, retention=0.8):
        return {
            "episode": episode, "training_stage": 1,
            "stages": [
                {"completion_rate": retention, "mean_accuracy": 0.4},
                {"completion_rate": completion, "mean_accuracy": accuracy},
            ],
        }

    records = [record(250, 0.4, 0.2), record(500, 0.6, 0.25),
               record(750, 0.9, 0.5, retention=0.5)]  # ineligible: forgot stage 1
    for item in records:
        checkpoint = tmp_path / f"checkpoint_{item['episode']:06d}"
        checkpoint.mkdir()
        (checkpoint / "marker").write_text(str(item["episode"]))
    log = tmp_path / "evaluation.jsonl"
    log.write_text("".join(json.dumps(item) + "\n" for item in records))

    tracker = ChampionTracker(tmp_path, retention_completion=0.75, patience=2,
                              initial_learning_rate=0.001,
                              minimum_learning_rate=0.000125)
    assert tracker.bootstrap(log)
    assert tracker.state["episode"] == 500
    assert (tmp_path / "champion" / "marker").read_text() == "500"

    worse = record(1000, 0.5, 0.3)
    worse_checkpoint = tmp_path / "checkpoint_001000"
    worse_checkpoint.mkdir()
    (worse_checkpoint / "marker").write_text("1000")
    assert tracker.consider(worse, worse_checkpoint) == "continue"
    assert tracker.consider(worse, worse_checkpoint) == "restore"
    assert tracker.state["episode"] == 500
    assert tracker.state["rollbacks"] == 0
    assert tracker.state["restorations"] == 1
    assert tracker.state["learning_rate"] == 0.001

    forgot = record(1100, 0.7, 0.3, retention=0.5)
    assert tracker.consider(forgot, worse_checkpoint) == "continue"
    assert tracker.consider(forgot, worse_checkpoint) == "recover"
    assert tracker.state["recoveries"] == 1

    better = record(1250, 0.7, 0.2)
    better_checkpoint = tmp_path / "checkpoint_001250"
    better_checkpoint.mkdir()
    (better_checkpoint / "marker").write_text("1250")
    assert tracker.consider(better, better_checkpoint) == "improved"
    assert tracker.state["episode"] == 1250
    assert tracker.state["evaluations_since_improvement"] == 0
    assert tracker.state["learning_rate"] == 0.001


def test_gate_passing_challenger_beats_completion_only_champion(tmp_path):
    from asteroid_survival.rl.training import ChampionTracker

    def record(episode, completion, accuracy):
        return {"episode": episode, "training_stage": 1, "stages": [
            {"completion_rate": 1.0, "mean_accuracy": 0.2},
            {"completion_rate": completion, "mean_accuracy": accuracy},
        ]}

    tracker = ChampionTracker(
        tmp_path, retention_completion=0.75, promotion_completion=0.80,
        accuracy_targets=(0.10, 0.14))
    champion = tmp_path / "checkpoint_000250"
    champion.mkdir()
    tracker._install(champion, record(250, 0.84375, 0.128))

    challenger = tmp_path / "checkpoint_000500"
    challenger.mkdir()
    assert tracker.consider(record(500, 0.8125, 0.144), challenger) == "improved"
    assert tracker.state["episode"] == 500


def test_champion_tracker_does_not_rollback_cold_start_stage(tmp_path):
    import json
    from asteroid_survival.rl.training import ChampionTracker

    checkpoint = tmp_path / "checkpoint_000250"
    checkpoint.mkdir()
    record = {"episode": 250, "training_stage": 0, "stages": [
        {"completion_rate": 0.5, "mean_accuracy": 0.1}]}
    log = tmp_path / "evaluation.jsonl"
    log.write_text(json.dumps(record) + "\n")
    tracker = ChampionTracker(tmp_path, retention_completion=0.75, patience=2)
    assert tracker.bootstrap(log)

    worse = {"episode": 500, "training_stage": 0, "stages": [
        {"completion_rate": 0.4, "mean_accuracy": 0.1}]}
    for _ in range(5):
        assert tracker.consider(worse, checkpoint, allow_recovery=False) == "continue"
    assert tracker.state["rollbacks"] == 0
    assert tracker.state["evaluations_since_improvement"] == 5
    assert tracker.state["learning_rate"] == 0.001


def test_champion_tracker_migrates_rollback_learning_rate(tmp_path):
    import json
    from asteroid_survival.rl.training import ChampionTracker

    (tmp_path / "champion").mkdir()
    (tmp_path / "champion_state.json").write_text(json.dumps({
        "episode": 100, "training_stage": 0, "score": [0, 1.0, 0.2],
        "completion_rate": 1.0, "accuracy": 0.2,
        "evaluations_since_improvement": 0, "rollbacks": 8,
    }))
    tracker = ChampionTracker(tmp_path, retention_completion=0.75,
                              initial_learning_rate=0.001,
                              minimum_learning_rate=0.000125)
    assert not tracker.bootstrap(tmp_path / "missing.jsonl")
    assert tracker.state["learning_rate"] == 0.000125


def test_preview_resolves_run_champion_before_other_checkpoints(tmp_path):
    import json
    from asteroid_survival.rl.preview import resolve_checkpoint

    champion = tmp_path / "champion"
    champion.mkdir()
    (champion / "metadata.json").write_text("{}")
    (tmp_path / "checkpoint_999999").mkdir()
    (tmp_path / "champion_state.json").write_text(json.dumps({"episode": 4750}))

    checkpoint, _ = resolve_checkpoint(tmp_path)

    assert checkpoint == champion


def test_policy_initialization_transfers_aiming_but_resets_task_heads(tmp_path):
    muzero = muzero_module()
    import jax
    import numpy as np
    from asteroid_survival.rl.training import initialize_agent_from_policy

    settings = muzero.MuZeroSettings(num_simulations=8)
    source = muzero.MuZeroAgent(12, 4, settings, seed=7)
    source.episodes = 99
    checkpoint = source.save(tmp_path / "source", episodes=99)
    transferred = initialize_agent_from_policy(checkpoint, 12, 4, settings, seed=3)
    fresh = muzero.MuZeroAgent(12, 4, settings, seed=3)

    def equal(left, right):
        return all(np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))

    assert equal(transferred.params["params"]["representation_net"],
                 source.params["params"]["representation_net"])
    assert equal(transferred.params["params"]["policy_output"],
                 source.params["params"]["policy_output"])
    assert equal(transferred.params["params"]["reward_output"],
                 fresh.params["params"]["reward_output"])
    assert equal(transferred.params["params"]["value_output"],
                 fresh.params["params"]["value_output"])
    assert transferred.episodes == 0 and transferred.training_steps == 0


def test_policy_initialization_expands_only_representation_input_rows(tmp_path):
    muzero = muzero_module()
    import numpy as np
    from asteroid_survival.rl.training import initialize_agent_from_policy

    settings = muzero.MuZeroSettings(num_simulations=8)
    source = muzero.MuZeroAgent(12, 4, settings, seed=7)
    checkpoint = source.save(tmp_path / "source", episodes=99)
    transferred = initialize_agent_from_policy(checkpoint, 22, 4, settings, seed=3)
    fresh = muzero.MuZeroAgent(22, 4, settings, seed=3)

    source_kernel = source.params["params"]["representation_net"]["layers_0"]["kernel"]
    transferred_kernel = transferred.params[
        "params"]["representation_net"]["layers_0"]["kernel"]
    fresh_kernel = fresh.params["params"]["representation_net"]["layers_0"]["kernel"]
    assert np.array_equal(transferred_kernel[:12], source_kernel)
    assert np.array_equal(transferred_kernel[12:], fresh_kernel[12:])


def test_task_hash_ignores_config_fields_left_at_their_defaults():
    """Adding an unused option to the game config must not invalidate old checkpoints.

    Hashing every field once meant that four new endless-mode knobs changed the hash of
    every stage, so previewing and resuming any existing run failed with a curriculum
    mismatch it had nothing to do with.
    """
    from asteroid_survival.rl.curriculum import load_curriculum, task_hash, task_hash_matches

    spec = load_curriculum("configs/rl-nonlinear.toml")
    baseline = task_hash(spec)

    # Writing a default back over a default is not a change to the task.
    spec.base.asteroid.endless_pressure_per_minute = 0.0
    spec.base.asteroid.ramp_step_seconds = 0.0
    assert task_hash(spec) == baseline
    assert task_hash_matches(baseline, spec)

    # A field the curriculum actually sets still changes it.
    spec.base.asteroid.endless_pressure_per_minute = 0.35
    assert task_hash(spec) != baseline
    assert not task_hash_matches(baseline, spec)


def _frozen_spec():
    """A curriculum built inline so its task digest can be pinned forever.

    Deliberately not loaded from a config file. An earlier version of this test pinned the
    digest of `configs/rl-nonlinear.toml`, which meant any legitimate change to that
    curriculum -- adding a trajectory pattern, say -- broke the test even though the hashing
    logic was fine. Freezing the output requires freezing the input.
    """
    from asteroid_survival.config import GameConfig
    from asteroid_survival.rl.curriculum import CurriculumSpec, CurriculumStage
    from asteroid_survival.rl.environment import RewardConfig

    stage = CurriculumStage(
        name="frozen-1", composition=(3, 3), target_waves=1, min_speed=30.0,
        max_speed=45.0, linear_probability=0.0, patterns=("sine", "zigzag"),
        amplitude_min=10.0, amplitude_max=40.0, wavelength_min=3.0, wavelength_max=4.5,
        movement_enabled=True, ship_invulnerable=False, max_seconds=60.0,
        no_hit_seconds=8.0, promotion_accuracy=0.05, miss_penalty=0.02)
    return CurriculumSpec(base=GameConfig(), reward=RewardConfig(), stages=(stage,))


def test_legacy_task_hash_is_reproducible_for_a_fixed_curriculum():
    """The pre-`_specified` digest must stay byte-stable, or old checkpoints stop verifying.

    A frozen value, not a reimplementation: earlier attempts enumerated the fields added
    since, and had to be edited on every schema change -- the maintenance trap that broke
    the guard twice to begin with.

    The digest covers what a stage actually configures, so a genuine change to the task --
    fragment speed multipliers, say -- moves it, and the constant below is updated on
    purpose. An *unexpected* failure here means real checkpoints have stopped verifying,
    which is exactly the alarm wanted.
    """
    from asteroid_survival.rl.curriculum import _legacy_task_hash, task_hash_matches

    spec = _frozen_spec()
    recorded = "331bc8bdd9e7371c32dcdb2a0a97d1741caec4ce99ae7819c31fc9308cad3eb5"

    assert _legacy_task_hash(spec) == recorded
    assert task_hash_matches(recorded, spec)
    assert task_hash_matches(None, spec)  # checkpoints predating hashing at all


def _endless_spec():
    from asteroid_survival.rl.curriculum import load_curriculum
    return load_curriculum("configs/rl-endless.toml")


def _play_stage(spec, index, seed, controller=None):
    from asteroid_survival.controllers import ClosestAsteroidController

    stage = spec.stages[index]
    config = stage.game_config(spec.base)
    config.ships = [ShipSpec("agent", "closest")]
    env = AsteroidsRLEnv(config, "agent", frame_skip=4, max_decisions=stage.max_decisions,
                         no_hit_seconds=stage.no_hit_seconds, reward_config=spec.reward,
                         completion=stage.completion)
    env.reset(seed)
    controller = controller or ClosestAsteroidController()
    while True:
        action = controller.action(env.state, "agent")
        index_of = env.actions.index(action) if action in env.actions else 0
        _, _, terminated, truncated, info = env.step(index_of)
        if terminated or truncated:
            return info["episode_metrics"]


def test_survival_round_is_cleared_by_lasting_to_the_decision_limit():
    """A survival round has no wave to clear, so reaching the limit alive is the clear."""
    spec = _endless_spec()
    metrics = _play_stage(spec, 0, 500)

    assert metrics["survived_to_limit"] is True
    assert metrics["completed_stage"] is True
    assert metrics["terminal_reason"] == "evaluation_limit"
    # Clearing must not be charged the wave-mode timeout penalty, which would punish success.
    assert metrics["timeout_penalty"] == 0.0
    assert metrics["round_clear_reward"] == pytest.approx(spec.reward.round_clear)
    assert metrics["survival_reward"] == pytest.approx(
        spec.reward.survival_bonus * spec.stages[0].max_decisions)


def test_dying_in_a_survival_round_is_not_a_clear():
    spec = _endless_spec()
    metrics = _play_stage(spec, 75, 500)  # far above where the greedy baseline survives

    assert metrics["completed_stage"] is False
    assert metrics["survived_to_limit"] is False
    assert metrics["round_clear_reward"] == 0.0
    assert metrics["death_penalty"] == pytest.approx(spec.reward.death_penalty)


def test_survival_stages_spawn_on_a_clock_with_no_wave_target():
    spec = _endless_spec()
    config = spec.stages[0].game_config(spec.base)

    assert spec.stages[0].survival is True
    assert spec.stages[0].completion == "survival"
    assert config.asteroid.spawn_mode == "interval"
    assert config.objective.max_waves is None
    assert config.objective.max_steps is None
    assert config.asteroid.heading_mode == "aimed"
    assert set(config.asteroid.pattern_pool) == set(PATTERN_NAMES)
    # Difficulty must not ramp inside a round: that is the whole point of the ladder.
    assert config.asteroid.is_ramped is False


def test_wave_stages_still_complete_by_clearing_waves():
    """Regression: adding survival rounds must not change the wave curriculum."""
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    stage = spec.stages[0]
    config = stage.game_config(spec.base)

    assert stage.survival is False
    assert stage.completion == "waves"
    assert config.asteroid.spawn_mode == "wave"
    assert config.objective.max_waves == stage.target_waves
    assert config.asteroid.heading_mode == "random"


def test_endless_ladder_holds_episode_cost_flat_while_difficulty_climbs():
    spec = _endless_spec()
    stages = spec.stages

    assert len(stages) == 96
    assert all(stage.survival for stage in stages)
    # Constant survival target is what bounds training cost at every rung of the ladder.
    assert {stage.max_decisions for stage in stages} == {450}
    # The observation layout is sized from active_cap, so it may not vary across rounds.
    assert len({stage.game_config(spec.base).asteroid.active_cap for stage in stages}) == 1

    # Within the long full-pattern tier every knob tightens round on round.
    for earlier, later in zip(stages[52:82], stages[53:82]):
        assert later.max_speed > earlier.max_speed
        assert later.amplitude_max > earlier.amplitude_max
        assert later.wavelength_min < earlier.wavelength_min
        assert later.spawn_interval < earlier.spawn_interval
        assert later.spawn_spread < earlier.spawn_spread
    assert stages[-1].spawn_interval > 0


def test_endless_reward_does_not_charge_for_staying_alive():
    spec = _endless_spec()

    assert spec.reward.survival_bonus > 0
    assert spec.reward.round_clear > 0
    assert spec.reward.active_time_penalty == 0.0  # would charge per second survived
    assert spec.reward.timeout_penalty == 0.0      # would charge for clearing the round


def test_greedy_baseline_clears_early_endless_rounds_and_fails_late_ones():
    """Anchors the ladder: the ladder is useless if nothing separates rungs."""
    spec = _endless_spec()
    early = [_play_stage(spec, 0, seed)["completed_stage"] for seed in range(500, 506)]
    late = [_play_stage(spec, 75, seed)["completed_stage"] for seed in range(500, 506)]

    assert all(early)
    assert not any(late)


def test_legacy_task_hash_survives_future_fields_being_added():
    """The legacy digest is an allowlist snapshot, so new fields cannot disturb it.

    Adding fields to AsteroidConfig broke every checkpoint once; fixing that with an
    exclusion list broke them again as soon as fields landed on CurriculumStage and
    RewardConfig instead. This pins the allowlist behaviour that replaced it.
    """
    from dataclasses import replace

    from asteroid_survival.rl.curriculum import _legacy_task_hash, load_curriculum

    spec = load_curriculum("configs/rl-nonlinear.toml")
    baseline = _legacy_task_hash(spec)

    # Fields added after the legacy hash was recorded are outside the snapshot entirely,
    # so changing them cannot move the digest, whatever value they take.
    spec.base.asteroid.endless_pressure_per_minute = 9.9
    spec.base.asteroid.ramp_step_seconds = 7.0
    drifted = replace(
        spec, reward=replace(spec.reward, survival_bonus=3.0, round_clear=99.0),
        stages=tuple(replace(stage, spawn_interval=0.1, spawn_spread=5.0)
                     for stage in spec.stages))
    assert _legacy_task_hash(drifted) == baseline

    # A field the snapshot does cover still changes it. It has to be one `game_config`
    # does not overwrite per stage -- base asteroid speeds, for instance, are replaced by
    # the stage's own values and so are invisible here.
    spec.base.projectile.speed = spec.base.projectile.speed + 1.0
    assert _legacy_task_hash(spec) != baseline


def test_preview_warns_instead_of_refusing_when_a_curriculum_has_drifted(tmp_path, capsys):
    """Preview must never be blocked by a hash: it is how you look at a trained model."""
    from asteroid_survival.rl.preview import preview_checkpoint

    checkpoint = tmp_path / "checkpoint_000100"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({
        "algorithm": "muzero", "observation_size": 8, "num_actions": 4,
        "observation_layout": {"curriculum": "configs/rl-nonlinear.toml",
                               "task_hash": "0" * 64, "history_frames": 0},
    }), encoding="utf-8")

    with pytest.raises(BaseException) as caught:
        preview_checkpoint(checkpoint)

    # It got past the hash gate and failed on something real instead.
    assert "different version" not in str(caught.value)
    assert "has changed since this checkpoint was trained" in capsys.readouterr().out


def test_survival_rounds_open_with_the_field_already_occupied():
    """An empty opening hands out free seconds that do not reflect the round's difficulty."""
    spec = _endless_spec()
    for index in (0, 14, 29):
        stage = spec.stages[index]
        config = stage.game_config(spec.base)
        assert config.asteroid.initial_asteroids > 0
        sim = Simulation(config)
        state = sim.reset(1234 + index)
        assert len(state.asteroids) == config.asteroid.initial_asteroids

    # Later rounds start more crowded than earlier ones.
    counts = [stage.game_config(spec.base).asteroid.initial_asteroids
              for stage in spec.stages]
    assert counts[0] < counts[-1]
    # `initial_asteroids` is deliberately non-monotone: it is the anti-idle floor, spent
    # where speeds are lowest and bought back at the top, so the upper tiers ramp it while
    # the foundation eases it down as the spawn clock tightens.
    assert counts[52:] == sorted(counts[52:])


def test_opening_asteroids_keep_clear_of_the_ship_including_across_the_wrap():
    from asteroid_survival.math2d import Vec2, wrapped_distance

    spec = _endless_spec()
    for index in (0, 45, 95):
        stage = spec.stages[index]
        config = stage.game_config(spec.base)
        sim = Simulation(config)
        for seed in range(300, 320):
            state = sim.reset(seed)
            ship = state.ships[0]
            for asteroid in state.asteroids:
                gap = wrapped_distance(Vec2(asteroid.x, asteroid.y), Vec2(ship.x, ship.y),
                                       state.width, state.height)
                # Clearance is time-based, so check the guaranteed fixed floor.
                assert gap > (asteroid.radius + ship.radius
                              + config.asteroid.spawn_safe_radius)


def test_opening_clearance_grows_with_asteroid_speed():
    """A fixed radius shrinks the reaction window as a ladder speeds up; this must not."""
    from asteroid_survival.math2d import Vec2, wrapped_distance

    spec = _endless_spec()

    def closest_gap(index):
        config = spec.stages[index].game_config(spec.base)
        sim = Simulation(config)
        gaps = []
        for seed in range(300, 330):
            state = sim.reset(seed)
            ship = state.ships[0]
            gaps.extend(
                wrapped_distance(Vec2(a.x, a.y), Vec2(ship.x, ship.y),
                                 state.width, state.height) - a.radius - ship.radius
                for a in state.asteroids)
        return min(gaps)

    assert spec.stages[95].max_speed > spec.stages[0].max_speed
    assert closest_gap(95) > closest_gap(0)


def test_wave_stages_still_open_on_an_empty_arena():
    """Regression: pre-population is a survival-round feature only."""
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-curriculum.toml")
    config = spec.stages[0].game_config(spec.base)
    assert config.asteroid.initial_asteroids == 0
    assert config.asteroid.spawn_safe_radius == 0.0
    assert config.asteroid.spawn_safe_seconds == 0.0
    assert len(Simulation(config).reset(7).asteroids) == 0


def test_initial_asteroids_cannot_exceed_the_observation_slots():
    from asteroid_survival.config import GameConfig

    config = GameConfig()
    config.asteroid.active_cap = 8
    config.asteroid.initial_asteroids = 9
    with pytest.raises(ValueError):
        config.validate()


def test_retention_pools_the_sample_instead_of_trusting_the_worst_draw():
    """One unlucky small-sample read must not block promotion.

    The nonlinear run stalled at round 38 doing exactly this: it cleared the current
    round's gate repeatedly while 8-episode reads on long-mastered rounds dipped below the
    floor by chance.
    """
    from asteroid_survival.rl.curriculum import retention_holds

    healthy = [{"completion_rate": 1.0, "episodes": 8} for _ in range(36)]
    unlucky = healthy + [{"completion_rate": 0.625, "episodes": 8}]

    assert retention_holds(unlucky, retention_completion=0.75, retention_floor=0.50)
    # The old rule was a conjunction, and this is the case it got wrong.
    assert not all(s["completion_rate"] >= 0.75 for s in unlucky)


def test_retention_still_catches_a_single_collapsed_stage():
    from asteroid_survival.rl.curriculum import retention_holds

    collapsed = ([{"completion_rate": 1.0, "episodes": 8} for _ in range(36)]
                 + [{"completion_rate": 0.0, "episodes": 8}])
    assert not retention_holds(collapsed, retention_completion=0.75, retention_floor=0.50)


def test_retention_catches_broad_decay_even_with_no_stage_collapsing():
    """Every stage above the floor, but the policy has plainly decayed overall."""
    from asteroid_survival.rl.curriculum import retention_holds

    decayed = [{"completion_rate": 0.55, "episodes": 8} for _ in range(37)]
    assert min(s["completion_rate"] for s in decayed) > 0.50   # no single collapse
    assert not retention_holds(decayed, retention_completion=0.75, retention_floor=0.50)


def test_retention_weights_by_sample_size_and_skips_unmeasured_stages():
    from asteroid_survival.rl.curriculum import retention_holds

    assert retention_holds([], retention_completion=0.75, retention_floor=0.50)
    # A stage with no episodes this round carries no evidence either way.
    mixed = [{"completion_rate": 0.0, "episodes": 0},
             {"completion_rate": 1.0, "episodes": 8}]
    assert retention_holds(mixed, retention_completion=0.75, retention_floor=0.50)
    # A large weak sample outweighs a tiny strong one.
    lopsided = [{"completion_rate": 0.60, "episodes": 64},
                {"completion_rate": 1.00, "episodes": 4}]
    assert not retention_holds(lopsided, retention_completion=0.75, retention_floor=0.50)


def test_curriculum_promotion_uses_the_pooled_retention_rule():
    from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum

    spec = load_curriculum("configs/rl-nonlinear.toml")
    manager = CurriculumManager(spec)
    manager.stage = 37

    def results(current_completion, worst_prior):
        rows = [{"completion_rate": 1.0, "mean_accuracy": 0.5, "episodes": 8}
                for _ in range(37)]
        rows[5] = {"completion_rate": worst_prior, "mean_accuracy": 0.5, "episodes": 8}
        rows.append({"completion_rate": current_completion, "mean_accuracy": 0.5,
                     "episodes": 32})
        return rows

    # An unlucky prior read no longer blocks a passing current stage.
    manager.consider_promotion(results(0.94, 0.625))
    assert manager.promotion_history[-1] is True
    # A collapsed prior stage still does.
    manager.consider_promotion(results(0.94, 0.0))
    assert manager.promotion_history[-1] is False


def test_versus_labels_are_short_and_stay_unique():
    from asteroid_survival.rl.comparison import contender_label

    taken = set()
    assert contender_label("models/ppo-nonlinear-v2-0818-1152/champion", taken) == "nonlinear-v2"
    assert contender_label("models/ppo-nonlinear-0817-2340/champion", taken) == "nonlinear"
    assert contender_label("models/muzero-arcade-0101-0101/champion", taken) == "arcade"
    # Two runs of the same family keep the timestamp that distinguishes them.
    assert contender_label("models/ppo-nonlinear-0819-0900/champion", taken) == "nonlinear-0819-0900"


def test_versus_ranks_contenders_by_survival():
    from asteroid_survival.rl.comparison import format_table

    table = format_table({
        "slow": {"survival_time": {"avg": 10.0, "best": 12.0, "worst": 8.0},
                 "wave": {"avg": 1.0}, "asteroids_destroyed": {"avg": 5.0},
                 "accuracy": {"avg": 0.4}},
        "fast": {"survival_time": {"avg": 40.0, "best": 60.0, "worst": 20.0},
                 "wave": {"avg": 2.0}, "asteroids_destroyed": {"avg": 50.0},
                 "accuracy": {"avg": 0.5}},
    })
    rows = [line.split()[0] for line in table.splitlines()[2:]]
    assert rows == ["fast", "slow"], "the best survivor should be listed first"


def test_versus_scores_each_contender_in_its_own_game():
    """Unlike showdown, nobody clears anyone else's asteroids."""
    from asteroid_survival.rl.comparison import compare
    from asteroid_survival.modes import build

    config, _ = build("survival", 1, controllers=["closest"])
    report = compare(config, Path("/dev/null") if False else
                     Path(__file__).parent / "_versus.json",
                     checkpoints=[], episodes=2, seed=4242, max_decisions=120,
                     include_human=False)
    # Both scripted baselines run by default; the pilot is the harder of the two.
    assert set(report["summary"]) == {"greedy", "pilot"}
    assert report["summary"]["greedy"]["episodes"] == 2
    assert report["summary"]["pilot"]["episodes"] == 2
    (Path(__file__).parent / "_versus.json").unlink(missing_ok=True)


def test_teammates_are_visible_and_only_when_the_round_has_them():
    """A policy cannot avoid ships it cannot see, so co-operative rounds need the slots.

    They are appended last so that adding them leaves every earlier input weight in place,
    and a round without teammates encodes exactly what it always did.
    """
    from asteroid_survival.config import GameConfig
    from asteroid_survival.rl.environment import TEAMMATE_FEATURES

    config = GameConfig()
    config.ship.friendly_collisions = "full"
    config.ships = [ShipSpec("alpha", "ppo"), ShipSpec("beta", "ppo")]

    solo = AsteroidsRLEnv(config, "alpha", max_decisions=100)
    coop = AsteroidsRLEnv(config, "alpha", max_decisions=100, max_teammates=1)
    assert coop.observation_size == solo.observation_size + TEAMMATE_FEATURES

    for env in (solo, coop):
        observation, _ = env.reset(1)
        assert len(observation) == env.observation_size
    # The single-ship encoding is untouched by the feature existing.
    assert list(coop.reset(1)[0][:solo.observation_size]) == list(solo.reset(1)[0])

    teammate = coop.reset(1)[0][-TEAMMATE_FEATURES:]
    assert teammate[0] == 1.0, "a live teammate should register as present"
    assert any(abs(float(value)) > 0 for value in teammate[1:]), "and carry its position"


def test_companions_fly_the_same_policy_and_can_kill_the_learner():
    """Co-operative rounds are one model flying every ship, not a policy plus bystanders."""
    from asteroid_survival.actions import Action
    from asteroid_survival.config import GameConfig

    config = GameConfig()
    config.ship.friendly_collisions = "full"
    config.asteroid.spawn_interval = 999.0        # isolate the ships from the asteroids
    config.ships = [ShipSpec("alpha", "ppo"), ShipSpec("beta", "ppo")]

    seen: list[int] = []
    thrust_index: list[int] = []

    def policy(observation):
        seen.append(len(observation))
        return thrust_index[0]

    env = AsteroidsRLEnv(config, "alpha", max_decisions=400, max_teammates=1,
                         companion_policy=policy)
    thrust_index.append(env.actions.index(Action.THRUST))
    env.reset(1)
    assert env.companion_ids == ["beta"]

    ended = False
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(thrust_index[0])
        if terminated or truncated:
            ended = terminated
            break

    assert seen, "the companion must actually be asked for actions"
    assert all(size == env.observation_size for size in seen), (
        "the companion sees the same observation shape the learner does")
    assert ended, "two ships thrusting into each other should end the episode"
    assert not any(ship.alive for ship in env.state.ships), "both ships die in a collision"


def test_survival_rounds_mix_linear_motion_back_in():
    spec = _endless_spec()
    for stage in (spec.stages[52], spec.stages[70], spec.stages[95]):
        assert abs(stage.linear_probability - 1 / 12) < 0.001
        assert len(stage.patterns) == len(PATTERN_NAMES)
    # The wave curricula are unaffected.
    from asteroid_survival.rl.curriculum import load_curriculum
    assert load_curriculum("configs/rl-nonlinear.toml").stages[40].linear_probability == 0.0


def test_no_asteroid_ever_spawns_on_top_of_a_ship():
    """The arena wraps, so a ship can sit exactly where the next edge spawn appears.

    Guarding only the opening field left mid-episode spawns unguarded, and they were
    measured materialising 22px *inside* the ship -- an unavoidable death rather than a
    missed dodge.
    """
    from asteroid_survival.actions import Action
    from asteroid_survival.math2d import Vec2, wrapped_distance

    spec = _endless_spec()
    for index in (0, 45, 95):
        stage = spec.stages[index]
        config = stage.game_config(spec.base)
        config.ships = [ShipSpec("pilot", "human")]
        config.ship.invulnerable = True          # fly into the corners on purpose
        sim = Simulation(config)
        worst = float("inf")
        for seed in range(12):
            state = sim.reset(seed)
            known = {asteroid.id for asteroid in state.asteroids}
            for _ in range(600):
                state = sim.step({"pilot": Action.THRUST}).snapshot
                ship = state.ships[0]
                for asteroid in state.asteroids:
                    if asteroid.id in known:
                        continue
                    known.add(asteroid.id)
                    worst = min(worst, wrapped_distance(
                        Vec2(asteroid.x, asteroid.y), Vec2(ship.x, ship.y),
                        state.width, state.height) - asteroid.radius - ship.radius)
        assert worst > config.asteroid.spawn_safe_radius, (
            f"round {index + 1}: a spawn came within {worst:.0f}px of the ship")


def test_wave_rounds_keep_spawning_exactly_as_they_did():
    """The spawn guard is config-gated, so curricula that do not set it are untouched."""
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-nonlinear.toml")
    config = spec.stages[40].game_config(spec.base)
    assert config.asteroid.spawn_safe_radius == 0.0
    assert config.asteroid.spawn_safe_seconds == 0.0

    counts = []
    for seed in (1, 2, 3):
        sim = Simulation(config)
        state = sim.reset(seed)
        for _ in range(400):
            state = sim.step({config.ships[0].id: 0}).snapshot
            if state.terminated or state.truncated:
                break
        counts.append(len(state.asteroids))
    assert any(counts), "wave spawning still fills the field"


def test_survival_rounds_hide_the_deadline_from_the_policy():
    """A survival round's decision limit only bounds cost; it must not shape behaviour.

    Revealing how close the limit is invites a policy to key on the deadline -- coasting
    into it, or giving up past it -- which is the opposite of surviving indefinitely.
    """
    spec = _endless_spec()
    stage = spec.stages[9]
    config = stage.game_config(spec.base)
    config.ships = [ShipSpec("agent", "closest")]

    survival = AsteroidsRLEnv(config, "agent", max_decisions=stage.max_decisions,
                              reward_config=spec.reward, completion="survival")
    assert survival.reveal_progress is False
    observation, _ = survival.reset(3)
    for _ in range(150):
        observation, _, terminated, truncated, _ = survival.step(0)
        if terminated or truncated:
            break
        assert observation[4] == 0.0, "elapsed time must stay hidden all episode"

    # Wave rounds still see it: clearing quickly is rewarded there, so the clock is relevant.
    waves = AsteroidsRLEnv(config, "agent", max_decisions=stage.max_decisions,
                           reward_config=spec.reward, completion="waves")
    assert waves.reveal_progress is True
    waves.reset(3)
    for _ in range(150):
        observation, _, terminated, truncated, _ = waves.step(0)
        if terminated or truncated:
            break
    assert observation[4] > 0.0

    # The slot is kept either way, so the observation layout is unchanged.
    assert survival.observation_size == waves.observation_size


def test_the_survival_ladder_climbs_from_straight_lines_to_every_curve():
    """Six tiers, joined so difficulty never steps backwards between them.

    One new thing at a time, the way the arcade curriculum does it: asteroid size first
    (a small does not split, so an early round never becomes a crowd), then curvature, then
    the full pattern set, then mixed sizes.
    """
    spec = _endless_spec()
    assert len(spec.stages) == 96

    small, bridge = spec.stages[:10], spec.stages[10:16]
    medium, large = spec.stages[16:26], spec.stages[26:38]
    curves, full, mixed = spec.stages[38:52], spec.stages[52:82], spec.stages[82:]

    for tier, size in ((small, 1), (medium, 2), (large, 3), (curves, 3), (full, 3)):
        assert {stage.asteroid_size for stage in tier} == {size}
    # The bridge introduces splitting one rock in four rather than all at once.
    assert {tuple(stage.asteroid_size) for stage in bridge} == {(1, 1, 1, 2)}
    assert {stage.asteroid_size for stage in mixed} == {None}, "the top tier mixes sizes"

    for tier in (small, bridge, medium, large):
        assert all(stage.patterns == () for stage in tier), "the size tiers fly straight"
    # Straight is one option among equals, not a special case with its own share: the
    # probability is 1/(patterns + 1) so every trajectory is equally likely.
    assert all(len(stage.patterns) == 3 for stage in curves)
    assert all(abs(stage.linear_probability - 1 / 4) < 0.001 for stage in curves)
    for tier in (full, mixed):
        assert all(len(stage.patterns) == len(PATTERN_NAMES) for stage in tier)
        assert all(abs(stage.linear_probability - 1 / 12) < 0.001 for stage in tier)

    # Every round is a survival round on the same thirty-second budget.
    assert all(stage.survival for stage in spec.stages)
    assert {stage.max_decisions for stage in spec.stages} == {450}

    # Speed, spread and oscillation period never step backwards anywhere on the ladder.
    # The tolerance absorbs rounding in per-round steps -- a tier's last round lands on
    # 44.99 where the next begins at 45.0 -- not a real backwards step.
    slack = 0.05
    for earlier, later in zip(spec.stages, spec.stages[1:]):
        assert later.max_speed >= earlier.max_speed - slack
        assert later.spawn_spread <= earlier.spawn_spread + slack
        assert later.wavelength_min <= earlier.wavelength_min + slack

    # Density is the exception, and deliberately so: where a tier introduces a new mechanic
    # the spawn clock resets to something looser, the way the arcade curriculum drops back
    # to the easiest composition before re-climbing. Round 11 introduces splitting and eases
    # from 2.60s to 4.20s. Within a tier it is monotone, and each tier must climb back past
    # where the previous one ended.
    for tier in (small, bridge, medium, large, curves, full, mixed):
        for earlier, later in zip(tier, tier[1:]):
            assert later.spawn_interval <= earlier.spawn_interval + slack
    assert bridge[-1].spawn_interval < small[-1].spawn_interval
    assert large[-1].spawn_interval < medium[-1].spawn_interval


def test_no_survival_round_can_be_passed_by_doing_nothing():
    """A survival round a motionless policy coasts through is a passivity trap.

    Stationary asteroids were rejected as a foundation for exactly this reason: doing
    literally nothing cleared such a round 24 times out of 24. With a survival-dominant
    reward that would be worth more than playing.
    """
    spec = _endless_spec()
    for index in (0, 4, 9, 12, 20, 30, 45, 70, 90):
        stage = spec.stages[index]
        config = stage.game_config(spec.base)
        config.ships = [ShipSpec("idle", "closest")]
        env = AsteroidsRLEnv(config, "idle", frame_skip=4,
                             max_decisions=stage.max_decisions,
                             reward_config=spec.reward, completion=stage.completion)
        cleared = 0
        for seed in range(300, 324):
            env.reset(seed)
            while True:
                _, _, terminated, truncated, info = env.step(0)   # never act
                if terminated or truncated:
                    break
            cleared += bool(info["episode_metrics"]["completed_stage"])
        # Tightened from 0.5: at thirty seconds the old bar was not a real guard, because
        # the field has half as long to fill.
        assert cleared / 24 < 0.30, (
            f"round {index + 1}: doing nothing clears {cleared}/24, which beats playing")


def test_survival_reward_puts_staying_alive_first():
    spec = _endless_spec()
    reward = spec.reward
    full_round = reward.survival_bonus * spec.stages[0].max_decisions

    assert full_round > 4 * reward.small * 10, "surviving should outweigh a good kill streak"
    assert reward.small > 0 and reward.large > 0, "hits still earn something"
    assert reward.active_time_penalty == 0.0   # would charge for staying alive
    assert reward.timeout_penalty == 0.0       # would charge for clearing the round
    assert reward.death_penalty > 0


def test_asteroid_size_reaches_the_simulation_and_leaves_wave_rounds_alone():
    """`asteroid_size` is survival-only, and defaults so no existing task hash moves.

    It is deliberately not called `spawn_size`: that name already means something else in a
    hand-written stage table, where it is the legacy per-wave fallback that desugars into a
    composition. Reusing it would collide silently.
    """
    from asteroid_survival.rl.curriculum import _asteroid_size, load_curriculum

    spec = _endless_spec()
    sizes = {stage.name.split("-")[0]: stage.game_config(spec.base).asteroid.spawn_size
             for stage in spec.stages}
    assert sizes["small"] == 1 and sizes["medium"] == 2 and sizes["large"] == 3
    assert sizes["bridge"] == [1, 1, 1, 2], "the bridge tier spawns a weighted mixture"
    assert sizes["curve"] == 3 and sizes["full"] == 3
    assert sizes["mixed"] is None, "None lets the simulation roll a size per spawn"

    # Wave curricula are untouched: they still pass None and pick sizes from composition.
    wave = load_curriculum("configs/rl-curriculum.toml")
    assert wave.stages[0].game_config(wave.base).asteroid.spawn_size is None

    assert [_asteroid_size(v) for v in ("small", "medium", "large", "mixed", None, 2)] == [
        1, 2, 3, None, None, 2]
    for bad in ("huge", 0, 4):
        with pytest.raises(ValueError):
            _asteroid_size(bad)


def test_the_wave_curricula_task_hashes_are_frozen():
    """Adding survival-only fields must never invalidate a wave-trained checkpoint.

    These digests were recorded before `asteroid_size` existed. If one moves, some change
    has leaked out of the survival ladder into the curricula real checkpoints were trained
    on, and every one of them has silently stopped verifying.
    """
    from asteroid_survival.rl.curriculum import load_curriculum, task_hash

    assert task_hash(load_curriculum("configs/rl-curriculum.toml")) == (
        "d0feb6200a907e1177c6d97b1fb691db6461f732041cec13a5268fcae082005f")
    assert task_hash(load_curriculum("configs/rl-nonlinear.toml")) == (
        "5c9d9d02f54c69541ae4deb857d9663816f58ab21f74d3594fa676a718cc76ec")


def test_the_survival_gate_accounts_for_the_shorter_round():
    """Thirty-second rounds are markedly easier to survive, so the gate had to rise.

    Measured on the greedy baseline, the same round reads 9-19 points higher at thirty
    seconds than at sixty in the region where the gate bites, and up to 47 points higher on
    the hardest rounds. An 80% gate at thirty seconds is a much weaker bar than the 80% it
    replaced.
    """
    spec = _endless_spec()
    assert spec.promotion_completion == 0.90
    assert spec.retention_completion == 0.75, "hysteresis: promote high, retain lower"
    assert spec.promotion_completion > spec.retention_completion

    # Noise must not be the binding constraint at this gate.
    from math import comb

    episodes = spec.evaluation_episodes
    needed = int(spec.promotion_completion * episodes + 0.999)

    def reads_at_or_above(true_rate):
        return sum(comb(episodes, k) * true_rate ** k * (1 - true_rate) ** (episodes - k)
                   for k in range(needed, episodes + 1))

    # A policy comfortably past the gate should clear a single evaluation almost always.
    assert reads_at_or_above(0.95) > 0.95
    # And one well below it should almost never sneak through.
    assert reads_at_or_above(0.80) < 0.10


def test_breaking_a_rock_is_never_worth_more_than_the_rock():
    """Hit rewards must be split-neutral, or the policy is paid to manufacture threats.

    The ordering was once inverted -- large 0.1, medium 0.25, small 0.5 -- inherited from the
    wave curriculum, where clearing a wave means destroying everything and the fiddly smalls
    deserve the most. In a survival round it meant shooting a medium paid 0.25 and produced
    two smalls worth 1.0. The policy learned exactly that: on the first round with splitting
    it made 43.7 kills at 0.50 accuracy against 18.7 at 0.19 on the round below, and died
    twice as often, surrounded by fragments of its own making.
    """
    spec = _endless_spec()
    reward = spec.reward

    assert reward.large == pytest.approx(2 * reward.medium)
    assert reward.medium == pytest.approx(2 * reward.small)
    assert reward.large > reward.medium > reward.small, "ordered by danger, not by fiddliness"


def test_survival_outweighs_shooting_in_the_reward():
    """Staying alive is the objective; hits are shaping and must stay shaping."""
    spec = _endless_spec()
    reward = spec.reward
    decisions = spec.stages[0].max_decisions

    surviving = reward.survival_bonus * decisions + reward.round_clear
    # The largest plausible haul: a full field of larges, each cleared to nothing.
    chain = reward.large + 2 * reward.medium + 4 * reward.small
    best_case_hits = chain * spec.stages[-1].game_config(spec.base).asteroid.active_cap

    assert surviving > 0
    assert reward.survival_bonus * decisions > 2 * reward.round_clear
    # Measured in play, survival is 70-79% of what the policy actually earns; this pins the
    # weaker structural property that one full round outweighs a large share of the field.
    assert surviving > best_case_hits / 3


def test_survival_rounds_are_scored_on_time_survived_not_on_clearing():
    """The objective is survival time, so that is what promotion measures.

    A binary "did it reach the limit" throws away everything about *how* long a failed
    episode lasted, and it is far noisier. On round 11 it read 69.5% where the fraction of
    time actually survived was 83.0% -- a 14-point gap -- and every promotion, retention
    check and champion comparison was being driven by the 69.5%.
    """
    from asteroid_survival.rl.evaluation import evaluate_policy

    spec = _endless_spec()
    stage = spec.stages[0]
    config = stage.game_config(spec.base)
    config.ships = [ShipSpec("agent", "closest")]
    env = AsteroidsRLEnv(config, "agent", frame_skip=4, max_decisions=stage.max_decisions,
                         reward_config=spec.reward, completion="survival")

    report = evaluate_policy(env, lambda observation: 0, list(range(200, 212)))
    aggregate = report["aggregate"]

    assert "survival_fraction" in aggregate and "clear_rate" in aggregate
    assert aggregate["completion_rate"] == pytest.approx(aggregate["survival_fraction"])
    # Partial credit is the point: an idle policy that lives most of the round scores for it.
    assert aggregate["survival_fraction"] > aggregate["clear_rate"]

    # Wave rounds keep the binary, where clearing really is all-or-nothing.
    waves = AsteroidsRLEnv(config, "agent", frame_skip=4,
                           max_decisions=stage.max_decisions,
                           reward_config=spec.reward, completion="waves")
    wave_report = evaluate_policy(waves, lambda observation: 0, list(range(200, 206)))
    assert wave_report["aggregate"]["completion_rate"] == pytest.approx(
        wave_report["aggregate"]["clear_rate"])


def test_idling_cannot_reach_the_gate_under_the_survival_metric():
    """Partial credit makes idling look better than it did; it must still fail the gate.

    Scored on clears a do-nothing policy reads about 30%; scored on time survived it reads
    about 75%, because it does live most of a short round before something finds it. That is
    a much smaller margin to the 90% gate than the binary gave, so it is worth pinning.
    """
    spec = _endless_spec()
    for index in (0, 12, 30):
        stage = spec.stages[index]
        config = stage.game_config(spec.base)
        config.ships = [ShipSpec("idle", "closest")]
        env = AsteroidsRLEnv(config, "idle", frame_skip=4,
                             max_decisions=stage.max_decisions,
                             reward_config=spec.reward, completion=stage.completion)
        lived = 0.0
        for seed in range(400, 432):
            env.reset(seed)
            while True:
                _, _, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break
            lived += min(1.0, info["episode_metrics"]["survival_time"] / stage.max_seconds)
        fraction = lived / 32
        assert fraction < spec.promotion_completion - 0.10, (
            f"round {index + 1}: idling scores {fraction:.0%} against a "
            f"{spec.promotion_completion:.0%} gate")


def test_scoring_a_survival_round_can_report_a_clear():
    """`compare` must score a survival round the way training does.

    The environment defaults to wave completion, and `modes.build` caps a survival round
    with an objective step limit so a human stops where training scores. Together those made
    `completed_stage` structurally impossible in `compare`, `watch`, and `versus`: the game
    ended one moment before the decision-limit truncation that sets `survived_to_limit`, so
    every survival round reported a zero clear rate however long a contender lasted.
    """
    from asteroid_survival.rl.comparison import compare
    from asteroid_survival.modes import build, round_env_settings

    settings = round_env_settings("survival-v2", 1)
    assert settings["completion"] == "survival"

    config, _ = build("survival-v2", 1, controllers=["closest"], scoring=True)
    assert config.objective.max_steps is None, "an objective cap preempts survived_to_limit"

    output = Path(__file__).parent / "_survival_clear.json"
    # `max_decisions` here is the generic default a caller would pass; the round's own
    # budget in `settings` has to win, or the truncation that sets the flag never fires.
    report = compare(config, output, checkpoints=[], episodes=2, seed=4242,
                     max_decisions=900, include_human=False, include_pilot=False,
                     env_settings=settings)
    episodes = report["episodes"]["greedy"]
    assert all(e["completed_stage"] for e in episodes), (
        "round 1 is trivial for greedy; a zero clear rate here means the scoring "
        "environment is not the one training used")
    output.unlink(missing_ok=True)
