import numpy as np
import pytest

from asteroid_survival.rl.curriculum import CurriculumManager, load_curriculum
from asteroid_survival.rl.multiagent import MultiAgentAsteroidsEnv, team_config


def test_survival_v2_has_96_rounds_and_uses_every_pattern_after_28():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    assert len(spec.stages) == 96
    assert spec.observation_version == 7
    assert spec.promotion_completion == pytest.approx(0.90)
    # 0.75 since 2026-08-26: the planning oracle clears round 26 at 0.969 but does it with
    # the simulator in hand, and cloning it yields a policy clearing 0.078 -- so it never
    # established 0.80 was reachable from the observation. At a true ~0.73, pooled promotion
    # at 0.80 needs ~183 evaluations; at 0.75 it needs 4.
    assert spec.promotion_clear_rate == pytest.approx(0.75)
    assert spec.promotion_pool
    # Every curve is in the pool from round 29 on. Straight stays in the mix at an
    # equal share rather than being removed -- see the equal-share test below.
    assert all(len(stage.patterns) == 11 for stage in spec.stages[28:])
    assert all(spec.stages[i].min_speed <= spec.stages[i + 1].min_speed
               and spec.stages[i].max_speed <= spec.stages[i + 1].max_speed
               and spec.stages[i].amplitude_max <= spec.stages[i + 1].amplitude_max
               and spec.stages[i].spawn_interval >= spec.stages[i + 1].spawn_interval
               for i in range(95))


def test_round_23_bridge_separates_new_patterns_from_large_asteroids():
    bridge = load_curriculum("configs/rl-survival-v2-round23-bridge.toml")
    target = load_curriculum("configs/rl-survival-v2.toml").stages[22]

    assert len(bridge.stages) == 4
    assert bridge.observation_version == 5
    assert bridge.promotion_completion == pytest.approx(0.90)
    # The bridge curricula set 0.80 explicitly rather than inheriting, and are historical
    # experiments -- the round-26 bridge stalled at 0.706 and was abandoned -- so they keep
    # the original bar even though the live v2 ladder moved to 0.75.
    assert bridge.promotion_clear_rate == pytest.approx(0.80)
    assert [len(stage.patterns) for stage in bridge.stages] == [8, 11, 11, 11]
    assert bridge.stages[0].asteroid_size == 2
    assert bridge.stages[1].asteroid_size == 2
    assert bridge.stages[2].asteroid_size == [2, 2, 2, 2, 2, 2, 2, 3]
    assert bridge.stages[3].asteroid_size == [2, 2, 2, 3]

    # The last lesson is exactly round 23 on every physical and motion-distribution knob.
    final = bridge.stages[-1]
    for name in ("min_speed", "max_speed", "amplitude_min", "amplitude_max",
                 "wavelength_min", "wavelength_max", "spawn_interval", "spawn_spread",
                 "initial_asteroids", "linear_probability"):
        assert getattr(final, name) == pytest.approx(getattr(target, name))
    assert final.asteroid_size == target.asteroid_size
    assert final.patterns == target.patterns


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


def test_v6_appends_wrap_safe_absolute_ship_position():
    from asteroid_survival.math2d import Vec2
    from asteroid_survival.rl.environment import AsteroidsRLEnv

    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[25]
    config = stage.game_config(spec.base)
    v5 = AsteroidsRLEnv(config, max_decisions=stage.max_decisions,
                        history_frames=8, history_long_frames=8,
                        observation_version=5)
    v6 = AsteroidsRLEnv(config, max_decisions=stage.max_decisions,
                        history_frames=8, history_long_frames=8,
                        observation_version=6)
    old, _ = v5.reset(11)
    new, _ = v6.reset(11)

    assert v6.observation_size == v5.observation_size + 4
    np.testing.assert_allclose(new[:v5.observation_size], old)

    # The extra block is periodic, so the two representations of each wrap seam agree.
    v6.simulation._ships[0].pos = Vec2(0.0, 0.0)
    at_zero = v6._observe_as(v6.simulation.snapshot(), v6.agent_id)[-4:]
    v6.simulation._ships[0].pos = Vec2(config.arena.width, config.arena.height)
    at_wrap = v6._observe_as(v6.simulation.snapshot(), v6.agent_id)[-4:]
    np.testing.assert_allclose(at_zero, at_wrap, atol=1e-6)
    np.testing.assert_allclose(at_zero, [0.0, 1.0, 0.0, 1.0], atol=1e-6)


def test_v7_appends_corrected_collision_threat_and_fragment_context():
    from asteroid_survival.math2d import Vec2
    from asteroid_survival.rl.environment import AsteroidsRLEnv, collision_prediction

    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[25]
    config = stage.game_config(spec.base)
    v6 = AsteroidsRLEnv(config, max_decisions=stage.max_decisions,
                        observation_version=6)
    v7 = AsteroidsRLEnv(config, max_decisions=stage.max_decisions,
                        observation_version=7)
    old, _ = v6.reset(17)
    new, _ = v7.reset(17)

    assert v7.observation_size == v6.observation_size + 10
    np.testing.assert_allclose(new[:v6.observation_size], old)

    # A rock closing from 100px away is urgent. The same rock moving away used to be
    # clamped to TTC zero and incorrectly outrank the real collision course.
    approaching = collision_prediction(Vec2(100.0, 0.0), -20.0, 0.0, 30.0)
    receding = collision_prediction(Vec2(100.0, 0.0), 20.0, 0.0, 30.0)
    assert approaching[0] == pytest.approx(5.0)
    assert approaching[1] == pytest.approx(0.0)
    assert receding[0] == pytest.approx(5.0)
    assert receding[1] > 150.0


def test_v7_encodes_split_parent_and_age_for_new_fragments():
    from asteroid_survival.state import GameEvent
    from asteroid_survival.rl.environment import AsteroidsRLEnv

    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[25]
    env = AsteroidsRLEnv(stage.game_config(spec.base), frame_skip=1,
                         max_decisions=stage.max_decisions, observation_version=7)
    env.reset(23)
    env._record_asteroid_events((GameEvent("asteroid_split", 1, "999", "123"),),
                                {123: 3})
    assert env._asteroid_context[999] == pytest.approx((env.state.elapsed, 3))
    # Directly exercise the appended context encoding with the currently selected threat.
    env._asteroid_context = {
        rock.id: (env.state.elapsed - 1.0, 3) for rock in env.state.asteroids}
    observation = env._observe_as(env.state, env.agent_id)
    assert observation[-2] == pytest.approx(0.2)
    assert observation[-1] == pytest.approx(1.0)


def test_v2_mastery_requires_both_survival_fraction_and_full_round_clear_rate():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    manager = CurriculumManager(spec)

    def result(survival, clear):
        return [{"completion_rate": survival, "clear_rate": clear,
                 "mean_accuracy": 0.05, "episodes": 64}]

    assert not manager.consider_promotion(result(0.91, 0.79))
    assert not manager.consider_promotion(result(0.89, 0.90))
    assert not manager.consider_promotion(result(0.91, 0.81))
    # The four-panel pool is 90.5% completion and exactly 80% clears.
    assert manager.consider_promotion(result(0.93, 0.90))


def test_v2_promotion_pools_all_four_disjoint_panels():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    manager = CurriculumManager(spec)

    def result(clear):
        return [{"completion_rate": 0.95, "clear_rate": clear,
                 "mean_accuracy": 0.06, "episodes": 64}]

    # The old two-passes-among-four rule would promote after the first two lucky panels.
    assert not manager.consider_promotion(result(0.90))
    assert not manager.consider_promotion(result(0.90))
    assert not manager.consider_promotion(result(0.55))
    assert not manager.consider_promotion(result(0.55))
    assert manager.stage == 0
    assert manager.promotion_pool["episodes"] == 256
    assert manager.promotion_pool["completion_rate"] == pytest.approx(0.95)
    assert manager.promotion_pool["clear_rate"] == pytest.approx(0.725)


def test_v2_promotion_pool_survives_resume():
    spec = load_curriculum("configs/rl-survival-v2.toml")
    first = CurriculumManager(spec)
    result = [{"completion_rate": 0.95, "clear_rate": 0.85,
               "mean_accuracy": 0.06, "episodes": 64}]
    assert not first.consider_promotion(result)
    assert not first.consider_promotion(result)

    resumed = CurriculumManager(
        spec, stage=first.stage, streak=first.streak,
        promotion_history=first.promotion_history,
        promotion_samples=first.promotion_samples)
    assert not resumed.consider_promotion(result)
    assert resumed.consider_promotion(result)
    assert resumed.stage == 1
    assert resumed.promotion_pool["episodes"] == 256


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


def test_the_round26_bridge_lands_exactly_on_round_26():
    """The bridge's last lesson must be round 26, or it bridges to somewhere else.

    Every physical knob is pinned by hand, so a later edit to survival-v2 would silently
    leave the bridge ramping toward a round that no longer exists as written. Only the large
    asteroid fraction is meant to differ across the lessons.
    """
    from asteroid_survival.rl.curriculum import load_curriculum

    bridge = load_curriculum("configs/rl-survival-v2-round26-bridge.toml")
    target = load_curriculum("configs/rl-survival-v2.toml").stages[25]
    final = bridge.stages[-1]

    assert list(final.asteroid_size) == list(target.asteroid_size)
    assert set(final.patterns) == set(target.patterns)
    for field in ("min_speed", "max_speed", "amplitude_max", "wavelength_min",
                  "wavelength_max", "spawn_interval", "spawn_spread", "max_seconds"):
        assert getattr(final, field) == pytest.approx(getattr(target, field)), field
    assert final.initial_asteroids == target.initial_asteroids

    # A ramp, and only in the one dimension under test.
    fractions = [sum(1 for s in st.asteroid_size if s == 3) / len(st.asteroid_size)
                 for st in bridge.stages]
    assert fractions == sorted(fractions)
    assert fractions[0] > 0.25 and fractions[-1] == 0.5


def test_coop_tier_kills_ships_that_collide_or_shoot_each_other():
    """The two-ship tier must enforce both halves of the rule, from its own config.

    `rl-endless-coop.toml` claims these rules in its header and never sets them, inheriting
    "off" from its base. This tier gets them because `ships > 1` forces
    `friendly_collisions = "full"` in `CurriculumStage.game_config`, so the guarantee is
    structural rather than a comment -- and this test is what says so.
    """
    from asteroid_survival.actions import Action
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.simulation import Simulation

    spec = load_curriculum("configs/rl-survival-v2-coop.toml")
    stage = spec.stages[96]                     # the first co-operative round
    assert stage.ships == 2
    config = stage.game_config(spec.base)
    assert config.ship.friendly_collisions == "full"
    assert [ship.id for ship in config.ships] == ["ship1", "ship2"]

    # Bumping into a teammate kills both ships.
    config.asteroid.spawn_interval = 999.0
    config.asteroid.initial_asteroids = 0
    simulation = Simulation(config)
    simulation.reset(0)
    simulation._ships[1].pos = simulation._ships[0].pos
    ships = simulation.step({}).snapshot.ships
    assert not ships[0].alive and not ships[1].alive, "a collision must kill both ships"

    # A shot that lands on a teammate kills it.
    simulation = Simulation(config)
    simulation.reset(0)
    simulation.step({"ship1": Action.FIRE})
    assert simulation._projectiles, "ship1 should have fired"
    for _ in range(6):                          # let the shot clear ship1's own hull
        simulation.step({})
    projectile = simulation._projectiles[0]
    assert projectile.owner_id == "ship1"
    simulation._ships[1].pos = projectile.pos
    result = simulation.step({})
    hit = [event for event in result.events if event.kind == "friendly_fire"]
    assert hit, "a projectile landing on a teammate must raise friendly_fire"
    assert not result.snapshot.ships[1].alive, "and must kill it"


def test_coop_retention_never_samples_a_round_the_fork_never_mastered():
    """A tier extending a long parent must not be judged on rounds it never learned.

    `retention_holds` enforces a per-stage completion floor, so one sampled round the policy
    cannot play fails retention outright and blocks promotion forever. The co-operative tier
    inherits all 96 solo rounds as prior but forks from a policy that reached round 26.
    """
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.ppo import retention_stages

    spec = load_curriculum("configs/rl-survival-v2-coop.toml")
    assert spec.retention_stage_limit == 26
    sampled = set()
    for rotation in range(40):
        sampled |= retention_stages(spec, 96, rotation)
    assert sampled, "retention should still sample something"
    assert max(sampled) < 26, f"sampled an unmastered round: {sorted(sampled)[-3:]}"

    # Without a limit the same tier reaches deep into rounds it never learned.
    solo = load_curriculum("configs/rl-survival-v2.toml")
    assert solo.retention_stage_limit == 0
    unlimited = set()
    for rotation in range(40):
        unlimited |= retention_stages(solo, 90, rotation)
    assert max(unlimited) > 26


def test_coop_tier_mirrors_every_solo_round_and_scales_past_two_ships():
    """Co-operative round N must be solo round N's field, with the ship count the only change.

    If the fields drift, a drop in score cannot be attributed to co-operation, which is the
    entire point of the tier.
    """
    from pathlib import Path

    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.environment import TEAMMATE_FEATURES
    from asteroid_survival.rl.ppo import _stage_env

    solo = load_curriculum("configs/rl-survival-v2.toml")
    coop = load_curriculum("configs/rl-survival-v2-coop.toml")
    assert len(coop.stages) == 2 * len(solo.stages)
    assert coop.stages[96].name == "coop2-round-1"
    assert coop.stages[-1].name == "coop2-round-96"

    fields = ("min_speed", "max_speed", "spawn_interval", "spawn_spread",
              "initial_asteroids", "amplitude_max", "wavelength_min")
    for index in range(len(solo.stages)):
        theirs = solo.stages[index].game_config(solo.base).asteroid
        ours = coop.stages[96 + index].game_config(coop.base).asteroid
        for field in fields:
            assert getattr(theirs, field) == pytest.approx(getattr(ours, field)), (
                f"round {index + 1} differs on {field}")
        assert solo.stages[index].composition == coop.stages[96 + index].composition
        assert coop.stages[96 + index].ships == 2

    # Raising `ships` scales the arena and the observation, one slot per other ship.
    layout = {"history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
              "max_projectiles": 8, "version": 7}
    solo_size = _stage_env(solo, 0, layout).observation_size
    template = Path("configs/rl-survival-v2-coop.toml").read_text()
    for count in (3, 5):
        scratch = Path(f"configs/_tmp-coop-{count}.toml")
        scratch.write_text(template.replace("ships = 2", f"ships = {count}"))
        try:
            spec = load_curriculum(scratch)
            config = spec.stages[96].game_config(spec.base)
            assert len(config.ships) == count
            assert config.ship.friendly_collisions == "full"
            assert spec.max_teammates == count - 1
            env = _stage_env(spec, 96, {**layout, "max_teammates": spec.max_teammates})
            assert env.observation_size == solo_size + (count - 1) * TEAMMATE_FEATURES
        finally:
            scratch.unlink()


def test_teammate_slot_carries_heading_and_cooldown():
    """A policy must be able to tell a teammate about to shoot through it from a nearby one."""
    from asteroid_survival.config import GameConfig, ShipSpec
    from asteroid_survival.rl.environment import TEAMMATE_FEATURES, AsteroidsRLEnv

    config = GameConfig()
    config.ship.friendly_collisions = "full"
    config.ships = [ShipSpec("alpha", "ppo"), ShipSpec("beta", "ppo")]
    env = AsteroidsRLEnv(config, "alpha", max_decisions=50, max_teammates=1)
    env.reset(3)
    assert env.simulation is not None

    def teammate_slot():
        # `env.state` is a frozen snapshot, so mutating the simulation's ship does not reach
        # it until a fresh one is taken.
        return env._observation(env.simulation.snapshot())[-TEAMMATE_FEATURES:]

    beta = next(ship for ship in env.simulation._ships if ship.id == "beta")
    beta.angle = 0.0
    facing_east = teammate_slot()
    beta.angle = 3.14159 / 2
    facing_north = teammate_slot()
    assert list(facing_east[8:10]) != list(facing_north[8:10]), (
        "heading must reach the observation")

    beta.cooldown = 0.0
    ready = teammate_slot()[10]
    beta.cooldown = config.ship.fire_cooldown
    reloading = teammate_slot()[10]
    assert ready == pytest.approx(0.0)
    assert reloading > ready, "cooldown must reach the observation"


def _coop_env(**kwargs):
    """A two-ship env carrying the co-operative penalties, isolated from asteroids."""
    from dataclasses import replace

    from asteroid_survival.config import GameConfig, ShipSpec
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.environment import AsteroidsRLEnv

    reward = load_curriculum("configs/rl-coop2-scratch.toml").reward
    config = GameConfig()
    config.ship.friendly_collisions = "full"
    config.ships = [ShipSpec("ship1", "ppo"), ShipSpec("ship2", "ppo")]
    config.asteroid.spawn_interval = 999.0
    config.asteroid.initial_asteroids = 0
    # `completion="survival"` matters: the env defaults to "waves", where lasting the round
    # is not what clears it. Every co-operative stage sets `survival = true`.
    kwargs.setdefault("completion", "survival")
    return AsteroidsRLEnv(config, "ship1", max_decisions=100, max_teammates=1,
                          reward_config=reward, **kwargs), reward


def test_dying_to_a_teammate_costs_something():
    """Collision and friendly-fire deaths were free: neither raises `ship_destroyed`.

    Measured on models/coop2-scratch before this fix, `death_penalty` averaged 0.10 while 56%
    of episodes ended in a death -- the two failure modes a co-operative round exists to teach
    carried no signal at all.
    """
    from asteroid_survival.math2d import Vec2

    # Killed by a teammate's hull.
    env, reward = _coop_env()
    env.reset(0)
    env.simulation._ships[1].pos = env.simulation._ships[0].pos
    _, value, terminated, _, info = env.step(0)
    assert terminated
    metrics = info["episode_metrics"]
    assert metrics["ship_collisions"] == 1
    assert metrics["collision_penalty"] == pytest.approx(reward.collision_penalty)
    assert value < 0, "a collision death must cost more than the decision paid"

    # Killed by a teammate's shot.
    from asteroid_survival.simulation import _Projectile
    env, reward = _coop_env()
    env.reset(0)
    ship = env.simulation._ships[0]
    env.simulation._projectiles.append(
        _Projectile(9001, "ship2", Vec2(ship.pos.x, ship.pos.y), Vec2(0.0, 0.0)))
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["episode_metrics"]["friendly_fire_taken"] == 1
    assert info["episode_metrics"]["friendly_fire_penalty"] == pytest.approx(
        reward.friendly_fire_penalty)


def test_killing_the_teammate_is_punished_harder_than_dying():
    """Otherwise it pays: the shooter lost nothing and inherited an emptier arena."""
    from asteroid_survival.math2d import Vec2
    from asteroid_survival.simulation import _Projectile

    env, reward = _coop_env()
    env.reset(0)
    other = env.simulation._ships[1]
    env.simulation._projectiles.append(
        _Projectile(9002, "ship1", Vec2(other.pos.x, other.pos.y), Vec2(0.0, 0.0)))
    _, value, terminated, truncated, _ = env.step(0)
    # The episode does NOT end -- the agent is alive and now has the arena to itself, which
    # is exactly why shooting the partner paid before this penalty existed.
    assert not terminated and not truncated
    assert env.metrics.friendly_fire_dealt == 1
    assert env.metrics.friendly_fire_dealt_penalty == pytest.approx(
        reward.friendly_fire_dealt_penalty)
    assert value < 0
    # Co-operating only wins while this exceeds roughly 45 * p(accident); at 12.0 it lost
    # above a 25% accident rate, and the from-scratch run died in 56% of episodes.
    assert reward.friendly_fire_dealt_penalty > reward.death_penalty
    assert reward.friendly_fire_dealt_penalty >= 27.0


def test_a_teammates_projectile_is_never_crowded_out_by_the_agents_own():
    """Slots were filled by distance, and the agent's own shots start at distance zero.

    Lifetime 1.45s against a 0.24s cooldown leaves about six of its own projectiles alive, so
    an 8-slot list hid exactly the incoming fire the agent has to dodge.
    """
    from asteroid_survival.math2d import Vec2
    from asteroid_survival.simulation import _Projectile

    env, _ = _coop_env(max_projectiles=8)
    env.reset(0)
    ship = env.simulation._ships[0]
    for index in range(8):                       # fill every slot with the agent's own shots
        env.simulation._projectiles.append(_Projectile(
            9100 + index, "ship1",
            Vec2(ship.pos.x + 4.0 * index, ship.pos.y), Vec2(0.0, -540.0)))
    own_only = env._observation(env.simulation.snapshot()).copy()

    # One teammate shot, deliberately the most distant projectile on the field.
    env.simulation._projectiles.append(_Projectile(
        9200, "ship2", Vec2(ship.pos.x + 300.0, ship.pos.y + 300.0), Vec2(0.0, -540.0)))
    with_teammate = env._observation(env.simulation.snapshot())
    assert list(own_only) != list(with_teammate), (
        "the teammate's projectile must displace one of the agent's own")


def test_warmup_rounds_ramp_difficulty_without_ever_removing_the_need_to_shoot():
    """Softening the rock COUNT is safe; softening it to zero is not.

    An empty arena's optimal policy is `never fire`, and coop2-v3 learned exactly that over
    7,000 episodes before promoting onto a round where firing is mandatory -- accuracy then
    collapsed to 0.05 and it never recovered. Every rung must need both aiming and spacing.
    """
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-coop2-scratch.toml")
    assert [stage.name for stage in spec.stages[:3]] == [
        "coop2-warmup-1", "coop2-warmup-2", "coop2-warmup-3"]
    assert spec.stages[3].name == "coop2-round-1"

    first = spec.stages[0]
    config = first.game_config(spec.base)
    assert first.ships == 2
    assert config.ship.friendly_collisions == "full"
    assert first.promotion_clear_rate == pytest.approx(0.85)

    counts = []
    for stage in spec.stages[:4]:
        stage_config = stage.game_config(spec.base)
        assert stage_config.asteroid.initial_asteroids > 0, (
            f"{stage.name} has nothing to shoot, so its optimal policy is to never fire")
        assert stage_config.asteroid.spawn_interval < 100.0, (
            f"{stage.name} never spawns, so its optimal policy is to never fire")
        counts.append(stage_config.asteroid.initial_asteroids)
    assert counts == sorted(counts), f"warm-up must ramp, got {counts}"
    assert counts[0] < counts[-1], "the ramp must actually go somewhere"

    # Defaults to 6.0, which would end a sparse round's episodes early as a stall.
    assert spec.stages[0].no_hit_seconds == 0.0


def test_solo_curricula_keep_the_cooperative_penalties_switched_off():
    """The new reward terms must be inert everywhere they are not configured."""
    from asteroid_survival.rl.curriculum import load_curriculum

    for name in ("configs/rl-survival-v2.toml", "configs/rl-curriculum.toml"):
        reward = load_curriculum(name).reward
        assert reward.collision_penalty == 0.0
        assert reward.friendly_fire_penalty == 0.0
        assert reward.friendly_fire_dealt_penalty == 0.0


def test_safety_potential_sees_a_teammate_not_just_asteroids():
    """Nothing rewarded the separation that avoids a point-blank shot.

    Collisions need 28px of closing, but measured teammate kills land at a median of 108px,
    so a policy that learns "do not touch" still drifts well inside shooting range -- which is
    what happened: collisions fell to 1.8% of episodes while friendly fire stayed at 63%.
    """
    from asteroid_survival.math2d import Vec2

    env, _ = _coop_env()
    env.reset(0)
    ship, other = env.simulation._ships[0], env.simulation._ships[1]

    other.pos = Vec2(ship.pos.x + 400.0, ship.pos.y)
    env.state = env.simulation.snapshot()
    far = env._safety_potential()

    other.pos = Vec2(ship.pos.x + 40.0, ship.pos.y)
    env.state = env.simulation.snapshot()
    near = env._safety_potential()

    assert near < far, "closing on a teammate must reduce the safety potential"
    assert far == pytest.approx(1.0, abs=0.35), "a distant teammate should read as safe"


def test_teammate_separation_is_recorded():
    env, _ = _coop_env()
    env.reset(0)
    for _ in range(5):
        env.step(0)
    assert env.metrics.mean_teammate_distance > 0.0
    assert env.metrics.minimum_teammate_distance is not None


def test_a_cooperative_round_is_cleared_by_the_team_not_the_survivor():
    """Scoring the learner alone made shooting the teammate a winning move.

    Measured on models/coop2-v2 checkpoint_013500: the policy cleared warm-up 1 in 30 of 30
    episodes while killing its partner in all 30 -- and that 0.92 clear rate is what promoted
    it. No penalty outweighs an objective that rewards the behaviour.
    """
    from asteroid_survival.math2d import Vec2
    from asteroid_survival.simulation import _Projectile

    env, _ = _coop_env()
    env.max_decisions = 3
    env.reset(0)
    other = env.simulation._ships[1]
    env.simulation._projectiles.append(
        _Projectile(9500, "ship1", Vec2(other.pos.x, other.pos.y), Vec2(0.0, 0.0)))
    info = {}
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    assert metrics["friendly_fire_dealt"] == 1
    assert metrics["survived_to_limit"], "the agent itself did last the round"
    assert not metrics["team_survived_to_limit"], "but its teammate did not"
    assert not metrics["completed_stage"], (
        "outliving the teammate must not count as clearing a co-operative round")
    assert metrics["round_clear_reward"] == 0.0

    # Both alive to the limit still clears.
    env, _ = _coop_env()
    env.max_decisions = 3
    env.reset(0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    assert metrics["team_survived_to_limit"]
    assert metrics["completed_stage"]
    assert metrics["round_clear_reward"] > 0.0


def test_solo_rounds_still_clear_on_the_agent_alone():
    """The team rule must apply only where there is a team."""
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.ppo import _stage_env

    spec = load_curriculum("configs/rl-survival-v2.toml")
    layout = {"history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
              "max_projectiles": 8, "version": 7}
    env = _stage_env(spec, 0, layout)
    assert env.companion_ids == []
    env.max_decisions = 3
    env.reset(0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    assert metrics["completed_stage"] == metrics["survived_to_limit"]


def test_losing_a_teammate_costs_even_when_nobody_is_to_blame():
    """Otherwise the agent has no reason to protect its partner, only to avoid shooting it.

    A teammate killed by an asteroid was free; the sole consequence was the forfeited
    round_clear, which is sparse and arrives only at the end.
    """
    from asteroid_survival.math2d import Vec2

    env, reward = _coop_env()
    assert reward.teammate_death_penalty > 0.0
    env.reset(0)
    # Drop a rock onto the teammate; the agent is nowhere near it.
    other = env.simulation._ships[1]
    env.simulation._asteroids.append(
        env.simulation._spawn_asteroid(pos=Vec2(other.pos.x, other.pos.y), size=1,
                                       direction=Vec2(0.0, 1.0)))
    _, value, _, _, _ = env.step(0)
    assert env.metrics.teammate_deaths == 1
    assert env.metrics.teammate_death_penalty == pytest.approx(reward.teammate_death_penalty)
    assert env.metrics.friendly_fire_dealt == 0, "the agent did not cause this one"
    assert value < 0


def test_shooting_a_teammate_costs_more_than_merely_losing_one():
    """Losing a partner is bad; killing one yourself has to be worse."""
    from asteroid_survival.rl.curriculum import load_curriculum

    reward = load_curriculum("configs/rl-coop2-scratch.toml").reward
    lost = reward.teammate_death_penalty
    shot = reward.teammate_death_penalty + reward.friendly_fire_dealt_penalty
    assert shot > lost
    # And both have to stay below what co-operating pays, or the ordering inverts.
    assert shot < reward.survival_bonus * 450 + reward.round_clear
