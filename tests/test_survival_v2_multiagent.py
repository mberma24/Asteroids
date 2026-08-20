import numpy as np
import pytest

from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum
from asteroid_survival.rl.multiagent import MultiAgentAsteroidsEnv, team_config


def test_survival_v2_has_96_rounds_and_uses_every_pattern_after_28():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    assert len(spec.stages) == 96
    assert spec.observation_version == 5
    assert spec.promotion_completion == pytest.approx(0.90)
    assert spec.promotion_clear_rate == pytest.approx(0.80)
    # Every curve is in the pool from round 29 on. Straight stays in the mix at an
    # equal share rather than being removed -- see the equal-share test below.
    assert all(len(stage.patterns) == 11 for stage in spec.stages[28:])
    assert all(spec.stages[i].min_speed <= spec.stages[i + 1].min_speed
               and spec.stages[i].max_speed <= spec.stages[i + 1].max_speed
               and spec.stages[i].amplitude_max <= spec.stages[i + 1].amplitude_max
               and spec.stages[i].spawn_interval >= spec.stages[i + 1].spawn_interval
               for i in range(95))


def test_v5_appends_sixteen_global_threat_features():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[0]
    from asteroid_survival.rl.environment import AsteroidsRLEnv
    old = AsteroidsRLEnv(stage.game_config(spec.base), max_decisions=stage.max_decisions,
                         history_frames=8, history_long_frames=8)
    new = AsteroidsRLEnv(stage.game_config(spec.base), max_decisions=stage.max_decisions,
                         history_frames=8, history_long_frames=8, global_features=True)
    old_obs, _ = old.reset(11)
    new_obs, _ = new.reset(11)
    assert new.observation_size == old.observation_size + 16
    np.testing.assert_allclose(new_obs[:old.observation_size], old_obs)
    assert np.isfinite(new_obs).all()


def test_v2_mastery_requires_both_survival_fraction_and_full_round_clear_rate():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    manager = CurriculumManager(spec)

    def result(survival, clear):
        return [{"completion_rate": survival, "clear_rate": clear,
                 "mean_accuracy": 0.05, "episodes": 64}]

    assert not manager.consider_promotion(result(0.91, 0.79))
    assert not manager.consider_promotion(result(0.89, 0.90))
    assert not manager.consider_promotion(result(0.91, 0.81))
    assert manager.consider_promotion(result(0.91, 0.81))


def test_multiagent_observations_are_fixed_and_cooldown_masks_fire_actions():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    config = team_config(spec.stages[28].game_config(spec.base), 3, level=9)
    env = MultiAgentAsteroidsEnv(config, max_ships=8, max_asteroids=74,
                                 history_frames=2, history_long_frames=0)
    observations, _ = env.reset(9)
    assert len(observations) == 3
    assert {len(value) for value in observations.values()} == {env.observation_size}
    # Force cooldown through the mutable simulation entity; snapshots then reflect it.
    env.simulation._ships[0].cooldown = 0.2
    env.state = env.simulation.snapshot()
    mask = env.action_masks()["ship1"]
    assert mask[:8].all()
    assert not mask[8:].any()


def test_multiagent_episode_does_not_end_when_only_one_ship_dies():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    config = team_config(spec.stages[28].game_config(spec.base), 3, level=1)
    env = MultiAgentAsteroidsEnv(config, max_ships=8, max_asteroids=74,
                                 history_frames=0, history_long_frames=0)
    env.reset(3)
    env.simulation._ships[0].alive = False
    env.state = env.simulation.snapshot()
    _, _, terminated, _, _ = env.step({"ship2": 0, "ship3": 0})
    assert not terminated
    assert len(env.alive_ids) == 2


def test_straight_line_is_one_trajectory_among_equals():
    """Straight is a pattern, not a privileged share and not an omission.

    A field of nothing but curves is its own narrow distribution -- a policy learns to
    always expect a turn -- and over-weighting straight teaches the opposite. Either way
    the training distribution stops matching the game, so every phase carries exactly
    1/(len(patterns) + 1).
    """
    spec = load_curriculum("configs/rl-survival-v2.toml")
    for index, stage in enumerate(spec.stages, start=1):
        expected = 1.0 / (len(stage.patterns) + 1)
        assert stage.linear_probability == pytest.approx(expected, abs=1e-4), (
            f"round {index}: straight is {stage.linear_probability:.3f} of spawns against "
            f"{expected:.3f} for each of its {len(stage.patterns)} curves")


def test_varied_rounds_extend_v2_without_moving_its_task_hash():
    """The overfitting rounds must not strand checkpoints trained on the base ladder.

    They live in their own file for exactly this reason: appending rounds to
    rl-survival-v2.toml changes what `task_hash` digests, and a run resuming against it
    would refuse to continue.
    """
    from asteroid_survival.rl.curriculum import task_hash

    base = load_curriculum("configs/rl-survival-v2.toml")
    varied = load_curriculum("configs/rl-survival-v2-varied.toml")

    assert len(base.stages) == 96 and len(varied.stages) == 100
    assert task_hash(load_curriculum("configs/rl-survival-v2.toml")) == task_hash(base)
    # Rounds 1-96 are inherited byte for byte, so retention still covers the whole ladder.
    assert [stage.name for stage in varied.stages[:96]] == [stage.name for stage in base.stages]
    assert all(stage.variety_probability == 0.0 for stage in varied.stages[:96])
    assert [round(stage.variety_probability, 2) for stage in varied.stages[96:]] == [
        0.25, 0.30, 0.35, 0.40]


def test_varied_rounds_keep_most_rocks_at_full_difficulty():
    """A minority of slow rocks, not a slower round.

    The check only means something if the round is still round 96 for most of what spawns;
    otherwise it measures whether the policy can handle an easier game, which is a different
    question.
    """
    import math

    from asteroid_survival.rl.environment import AsteroidsRLEnv

    spec = load_curriculum("configs/rl-survival-v2-varied.toml")
    stage = spec.stages[96]
    config = stage.game_config(spec.base)
    config.ship.invulnerable = True
    env = AsteroidsRLEnv(config, "agent", frame_skip=4, max_decisions=stage.max_decisions,
                         reward_config=spec.reward, completion=stage.completion)
    speeds, periods = [], []
    for seed in range(20):
        env.reset(seed)
        for _ in range(60):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        for asteroid in env.simulation._asteroids:
            speeds.append(asteroid.speed)
            if asteroid.frequency:
                periods.append(2 * math.pi / asteroid.frequency)

    slow = [speed for speed in speeds if speed < stage.min_speed * 0.95]
    assert 0.10 < len(slow) / len(speeds) < 0.45, (
        f"{len(slow) / len(speeds):.0%} of rocks are slow; wanted a minority")
    assert min(speeds) < stage.min_speed * 0.6, "no genuinely slow rock ever appeared"
    assert max(periods) > stage.wavelength_max * 2, "no genuinely slow oscillation appeared"
