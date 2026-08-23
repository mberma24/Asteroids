from __future__ import annotations

import json

import numpy as np
import pytest


gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")


def make_native_env():
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.environment import AsteroidsRLEnv

    spec = load_curriculum("configs/rl-curriculum.toml")
    stage = spec.stages[0]
    return AsteroidsRLEnv(
        stage.game_config(spec.base), frame_skip=4, max_decisions=stage.max_decisions,
        no_hit_seconds=stage.no_hit_seconds, history_frames=8,
        history_long_frames=8, history_long_stride=8, reward_config=spec.reward)


def test_gym_wrapper_matches_native_seeded_transitions():
    from asteroid_survival.rl.gym_env import GymAsteroidsEnv

    native = make_native_env()
    wrapped_native = make_native_env()
    wrapped = GymAsteroidsEnv(wrapped_native)
    expected, _ = native.reset(123)
    actual, _ = wrapped.reset(seed=123)
    np.testing.assert_array_equal(actual, expected)
    assert wrapped.observation_space.shape == (1235,)
    assert wrapped.action_space.n == 16

    for action in (0, 1, 8, 2, 8):
        expected_step = native.step(action)
        actual_step = wrapped.step(action)
        np.testing.assert_array_equal(actual_step[0], expected_step[0])
        assert actual_step[1:] == expected_step[1:]


def test_curriculum_gym_env_exposes_stage_and_fixed_spaces():
    from stable_baselines3.common.env_checker import check_env
    from asteroid_survival.rl.gym_env import CurriculumGymEnv

    env = CurriculumGymEnv(
        "configs/rl-curriculum.toml", rank=0, num_envs=1, seed=7,
        history_frames=8, history_long_frames=8, history_long_stride=8)
    check_env(env)
    env.set_curriculum_state(3)
    observation, info = env.reset()
    assert observation.shape == (1235,)
    assert info["curriculum_stage"] in range(4)


def test_curriculum_env_runs_in_spawned_vector_workers():
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from asteroid_survival.rl.gym_env import CurriculumGymEnv

    factories = [
        lambda rank=rank: CurriculumGymEnv(
            "configs/rl-curriculum.toml", rank=rank, num_envs=2, seed=9,
            history_frames=8, history_long_frames=8, history_long_stride=8)
        for rank in range(2)
    ]
    vec = SubprocVecEnv(factories, start_method="spawn")
    try:
        observations = vec.reset()
        assert observations.shape == (2, 1235)
        observations, rewards, dones, infos = vec.step(np.zeros(2, dtype=np.int64))
        assert observations.shape == (2, 1235)
        assert rewards.shape == dones.shape == (2,)
    finally:
        vec.close()


@pytest.mark.parametrize("recurrent", [False, True])
def test_ppo_checkpoint_save_load_and_deterministic_controller(tmp_path, recurrent):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from asteroid_survival.rl.gym_env import GymAsteroidsEnv
    from asteroid_survival.rl.ppo import PPOController

    vec = DummyVecEnv([lambda: GymAsteroidsEnv(make_native_env())])
    observation = vec.reset()[0]
    if recurrent:
        model = RecurrentPPO(
            "MlpLstmPolicy", vec, n_steps=8, batch_size=8, n_epochs=1,
            policy_kwargs={"lstm_hidden_size": 16,
                           "net_arch": {"pi": [16], "vf": [16]}}, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1,
            policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}}, device="cpu")
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    model.save(checkpoint / "model.zip")
    (checkpoint / "metadata.json").write_text(json.dumps({
        "algorithm": "recurrent_ppo" if recurrent else "ppo",
        "recurrent": recurrent, "observation_size": 1235, "num_actions": 16,
        "observation_layout": {},
    }))
    controller = PPOController(checkpoint, device="cpu")
    first = controller(observation)
    controller.reset()
    assert controller(observation) == first
    if recurrent:
        assert controller.state is not None
        controller.reset()
        assert controller.state is None
    vec.close()


def _tracker(tmp_path, patience=2):
    from asteroid_survival.rl.ppo_support import PPOChampionTracker

    return PPOChampionTracker(
        tmp_path, retention_completion=0.75, patience=patience, learning_rate=3e-4,
        minimum_learning_rate=1e-4, promotion_completion=0.80,
        accuracy_targets=(0.05,) * 60)


def _record(episode, stage, completion, accuracy=0.5, prior=1.0, clear=1.0):
    stages = [{"completion_rate": prior, "clear_rate": clear,
               "mean_accuracy": accuracy, "episodes": 8}
              for _ in range(stage)]
    stages.append({"completion_rate": completion, "clear_rate": clear,
                   "mean_accuracy": accuracy,
                   "episodes": 96})
    return {"episode": episode, "training_stage": stage, "stages": stages}


def _checkpoint(tmp_path, episode):
    path = tmp_path / f"checkpoint_{episode:06d}"
    path.mkdir(exist_ok=True)
    (path / "model.zip").write_text("x", encoding="utf-8")
    return path


def test_one_lucky_evaluation_does_not_crown_a_champion(tmp_path):
    """Max-selection bias pinned a real run: an 84.4% reading whose true level was 57.3%.

    Nothing could beat the phantom, so the plateau logic fired forever and the learning
    rate collapsed. A champion now has to hold up across several evaluations.
    """
    tracker = _tracker(tmp_path)
    tracker.consider(_record(250, 40, 0.60), _checkpoint(tmp_path, 250), allow_recovery=True)
    assert tracker.state["clear_rate"] == pytest.approx(1.0)
    baseline = tracker.state["completion_estimate"]

    # One outlier, then the true level reasserts itself.
    tracker.consider(_record(500, 40, 0.844), _checkpoint(tmp_path, 500), allow_recovery=True)
    assert tracker.state["episode"] == 250, "a single spike must not install a champion"
    tracker.consider(_record(750, 40, 0.58), _checkpoint(tmp_path, 750), allow_recovery=True)
    tracker.consider(_record(1000, 40, 0.60), _checkpoint(tmp_path, 1000), allow_recovery=True)
    assert tracker.state["completion_estimate"] <= baseline + 0.15, (
        "the champion's recorded level should track the smoothed rate, not the spike")


def test_a_sustained_improvement_still_installs(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.consider(_record(250, 40, 0.55), _checkpoint(tmp_path, 250), allow_recovery=True)
    for episode, completion in ((500, 0.70), (750, 0.72), (1000, 0.74)):
        tracker.consider(_record(episode, 40, completion),
                         _checkpoint(tmp_path, episode), allow_recovery=True)
    assert tracker.state["episode"] == 1000
    assert tracker.state["completion_estimate"] > 0.65


def test_promotion_to_a_new_stage_always_installs(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.consider(_record(250, 40, 0.85), _checkpoint(tmp_path, 250), allow_recovery=True)
    action = tracker.consider(_record(500, 41, 0.40),
                              _checkpoint(tmp_path, 500), allow_recovery=True)
    assert action == "improved" and tracker.state["training_stage"] == 41


def test_ppo_never_restores_weights_from_inside_an_on_policy_rollout(tmp_path):
    """Champion is a serving artifact; rollback starts a new run explicitly."""
    tracker = _tracker(tmp_path, patience=2)
    tracker.consider(_record(250, 40, 0.60), _checkpoint(tmp_path, 250), allow_recovery=True)

    actions = [tracker.consider(_record(episode, 40, 0.62),
                                _checkpoint(tmp_path, episode), allow_recovery=True)
               for episode in (500, 750, 1000, 1250)]
    assert "restore" not in actions, "training is level with the champion, not behind it"

    # Even a real collapse cannot swap policies underneath an already-collected rollout.
    collapsed = [tracker.consider(_record(episode, 40, 0.10),
                                  _checkpoint(tmp_path, episode), allow_recovery=True)
                 for episode in (1500, 1750, 2000, 2250)]
    assert "restore" not in collapsed


def test_retention_rotates_over_prior_stages_instead_of_scoring_them_all():
    """Re-scoring every prior stage cost more than the training it interleaved with.

    At curriculum stage 41 the prior-stage sweep was two thirds of every evaluation and
    half of the run's total decisions. Retention is judged on the pooled sample, so a
    rotating subset answers the same question far more cheaply.
    """
    from asteroid_survival.rl.curriculum import load_curriculum
    from asteroid_survival.rl.ppo import retention_stages

    spec = load_curriculum("configs/rl-nonlinear.toml")
    assert spec.retention_sample == 10

    covered = set()
    for rotation in range(4):
        picked = retention_stages(spec, 40, rotation)
        assert len(picked) == 10
        covered |= picked
    assert covered == set(range(40)), "every prior stage must be revisited regularly"

    # Fewer prior stages than the sample size: everything is scored.
    assert retention_stages(spec, 6, 0) == set(range(6))

    # Disabled entirely, which is the default and what every curriculum did before.
    from dataclasses import replace

    assert retention_stages(replace(spec, retention_sample=0), 12, 3) == set(range(12))


def test_unscored_stages_are_neutral_not_forgotten(tmp_path):
    """A stage skipped by rotation carries no evidence, so it must not read as a failure."""
    from asteroid_survival.rl.curriculum import retention_holds

    stages = [{"completion_rate": 1.0, "episodes": 8},
              {"completion_rate": 0.0, "episodes": 0},   # skipped this round
              {"completion_rate": 0.9, "episodes": 8}]
    assert retention_holds(stages, retention_completion=0.75, retention_floor=0.50)

    tracker = _tracker(tmp_path)
    record = {"episode": 500, "training_stage": 3,
              "stages": stages + [{"completion_rate": 0.6, "mean_accuracy": 0.5,
                                   "episodes": 96}]}
    assert tracker.forgotten_stage(record) is None


def test_set_encoder_is_blind_to_slot_order_and_to_padding():
    """The asteroid block is a set, and the network must treat it as one.

    Measured on a trained policy at round 17: 19.8% of occupied asteroid slots change
    contents between consecutive decisions, fifteen times a second, and 51% of the block is
    zero-padding. A plain MLP has to learn a function stable under constant permutation of
    its own inputs, 26 times over.
    """
    import numpy as np
    import torch
    from gymnasium import spaces

    from asteroid_survival.rl.networks import SetFeaturesExtractor

    space = spaces.Box(-np.inf, np.inf, (1235,), np.float32)
    encoder = SetFeaturesExtractor(
        space, ship_features=11, asteroid_slots=26, asteroid_features=44,
        projectile_slots=8, projectile_features=10)
    assert encoder.features_dim == encoder(torch.zeros(2, 1235)).shape[1]

    # The same asteroid in a different slot must encode identically.
    body = torch.arange(44, dtype=torch.float32) / 44
    body[0] = 1.0                                    # presence flag
    first, eighth = torch.zeros(1, 1235), torch.zeros(1, 1235)
    first[0, 11:55] = body
    eighth[0, 11 + 44 * 7:11 + 44 * 8] = body
    assert torch.allclose(encoder(first), encoder(eighth), atol=1e-6)

    # Empty slots must not drag the pooled mean: one asteroid is one asteroid, not 1/26th.
    assert not torch.allclose(encoder(first), encoder(torch.zeros(1, 1235)), atol=1e-6)


def test_set_encoder_refuses_a_layout_it_does_not_match():
    """A silent mismatch would reshape the observation into nonsense."""
    import numpy as np
    import pytest as _pytest
    from gymnasium import spaces

    from asteroid_survival.rl.networks import SetFeaturesExtractor

    with _pytest.raises(ValueError, match="drifted apart"):
        SetFeaturesExtractor(
            spaces.Box(-np.inf, np.inf, (1235,), np.float32), ship_features=11,
            asteroid_slots=25, asteroid_features=44,
            projectile_slots=8, projectile_features=10)


def test_a_stalled_stage_still_advances_its_champion(tmp_path):
    """A run stuck below the promotion bar must still track its best policy.

    The same-stage gate used to require `promotion_clear_rate`. On a stage the run stalled
    below -- exactly where the champion matters, because it is the fork point and the
    artifact `versus` and `play` resolve to -- no evaluation could ever pass it, so the
    policy installed on arrival stayed champion forever. A real run sat 60,000 episodes on
    a champion whose clear rate was 0.55 while the live policy was reaching 0.73.
    """
    from asteroid_survival.rl.ppo_support import PPOChampionTracker

    tracker = PPOChampionTracker(
        tmp_path, retention_completion=0.75, patience=2, learning_rate=3e-4,
        minimum_learning_rate=1e-4, promotion_completion=0.90,
        accuracy_targets=(0.05,) * 60, clear_target=0.80)

    tracker.consider(_record(40000, 25, 0.78, clear=0.55),
                     _checkpoint(tmp_path, 40000), allow_recovery=True)
    assert tracker.state["episode"] == 40000

    # Sustained, and better than the incumbent -- but never at the 0.80 promotion bar.
    for episode in (40500, 41000, 41500, 42000):
        tracker.consider(_record(episode, 25, 0.86, clear=0.73),
                         _checkpoint(tmp_path, episode), allow_recovery=True)

    assert tracker.state["episode"] > 40000, (
        "a stage stalled under the promotion bar froze its champion permanently")
    assert tracker.state["clear_rate"] == pytest.approx(0.73)


def test_a_plateau_does_not_raise_the_bar_above_the_champion(tmp_path):
    """The estimate is the champion's own level, not the best reading ever seen.

    Ratcheting it upward on every plateau built a bar the champion itself could not clear,
    so the longer a run stalled the harder installing a *better* policy became.
    """
    from asteroid_survival.rl.ppo_support import PPOChampionTracker

    tracker = PPOChampionTracker(
        tmp_path, retention_completion=0.75, patience=2, learning_rate=3e-4,
        minimum_learning_rate=1e-4, promotion_completion=0.90,
        accuracy_targets=(0.05,) * 60, clear_target=0.80)
    tracker.consider(_record(250, 40, 0.78, clear=0.75),
                     _checkpoint(tmp_path, 250), allow_recovery=True)
    installed = tracker.state["completion_estimate"]

    # Clear rate collapses, so no install is allowed; retention on earlier stages still
    # holds, so the plateau counter runs to patience. It was then raising the completion bar
    # on readings it had just refused to install.
    for episode in (500, 750, 1000, 1250):
        tracker.consider(_record(episode, 40, 0.95, clear=0.40),
                         _checkpoint(tmp_path, episode), allow_recovery=True)

    assert tracker.state["episode"] == 250, "a collapsed clear rate must not install"
    assert tracker.state["completion_estimate"] <= installed + 1e-9, (
        "a reading the tracker refused to install must not become the standing bar")


def test_live_play_controller_accepts_a_v5_checkpoint(monkeypatch):
    """`showdown` must be able to drive a checkpoint carrying the v5 global block.

    `PPOPlayController` read `max_teammates` and `global_features` in its manifest check and
    again in `action`, but never assigned them, so a live showdown against any v5 checkpoint
    raised AttributeError before drawing a frame. Every current checkpoint is v5.
    """
    from asteroid_survival.rl.environment import MOBILE_ACTIONS
    from asteroid_survival.modes import build
    from asteroid_survival.rl import ppo as ppo_module
    from asteroid_survival.rl.controller import PPOPlayController

    config, _ = build("survival-v2", 26, controllers=["human", "closest", "ppo"])
    layout = {
        "version": 5, "max_asteroids": config.asteroid.active_cap, "mobile": True,
        "history_frames": 8, "history_long_frames": 8, "history_long_stride": 8,
        "max_projectiles": 8, "max_teammates": 0, "global_features": 16,
        "actions": [action.name for action in MOBILE_ACTIONS],
    }

    class _Stub:
        def __init__(self, *args, **kwargs):
            self.metadata = {"observation_layout": layout, "observation_size": 1251,
                             "num_actions": len(MOBILE_ACTIONS), "recurrent": False}

    monkeypatch.setattr(ppo_module, "PPOController", _Stub)
    controller = PPOPlayController(config, "unused")
    assert controller.global_features is True
    assert controller.max_teammates == 0
    assert len(controller.history_slots) == 16


def _floor(base=0.0025, floor=0.8, **kwargs):
    from asteroid_survival.rl.ppo import EntropyFloorController

    return EntropyFloorController(base=base, floor=floor, **kwargs)


def test_entropy_floor_raises_the_coefficient_below_target():
    """The measured collapse: entropy sat near 0.55 nats while learning was dead."""
    controller = _floor()
    seen = [controller.update(0.55) for _ in range(60)]
    assert seen == sorted(seen), "the coefficient must not wander while entropy is low"
    assert controller.coefficient > 4 * controller.base
    assert controller.coefficient <= controller.ceiling


def test_entropy_floor_returns_to_base_but_never_below():
    """Enabling a floor must never leave a run less explored than the same run without one."""
    controller = _floor()
    for _ in range(60):
        controller.update(0.55)
    raised = controller.coefficient
    seen = [controller.update(1.2) for _ in range(400)]
    assert seen == sorted(seen, reverse=True)
    assert raised > controller.base
    assert controller.coefficient == pytest.approx(controller.base)


def test_entropy_floor_step_is_bounded():
    """The anti-oscillation contract: no single update may move the coefficient far."""
    import math

    controller = _floor()
    for _ in range(controller.WARMUP_UPDATES + 1):
        controller.update(0.8)
    before = controller.coefficient
    controller.update(0.0)                     # maximal error
    assert controller.coefficient / before <= math.exp(controller.MAX_LOG_STEP) + 1e-12


def test_entropy_floor_is_quiet_inside_the_deadband():
    controller = _floor()
    for _ in range(10):
        controller.update(0.8)
    settled = controller.coefficient
    for index in range(100):
        controller.update(0.79 if index % 2 else 0.82)
    assert controller.coefficient == pytest.approx(settled)


def test_entropy_floor_stays_subordinate_to_the_kl_cap():
    """Buying exploration by driving SB3 to truncate its epoch loop is the stall, not the fix."""
    controller = _floor()
    for _ in range(10):
        controller.update(0.8)
    settled = controller.coefficient

    for _ in range(20):
        controller.update(0.5, approx_kl=0.03, target_kl=0.02)
    assert controller.coefficient == pytest.approx(settled), "must not climb while over target"

    controller.update(0.5, approx_kl=0.010, target_kl=0.02)
    assert controller.coefficient > settled, "must climb once KL is back under target"

    raised = controller.coefficient
    for _ in range(60):                        # lift the average clear of the floor first
        controller.update(1.5, approx_kl=0.03, target_kl=0.02)
    assert controller.coefficient < raised, "coming back down is always allowed"


def test_entropy_floor_converges_without_oscillating():
    """Closed loop against a toy monotone plant, seeded at the observed dead level."""
    import math

    controller = _floor()
    entropy, history = 0.55, []
    for _ in range(1500):
        coefficient = controller.update(entropy)
        entropy += 0.02 * (10.0 * math.sqrt(coefficient) - entropy)
        history.append(coefficient)
    assert controller.smoothed == pytest.approx(controller.floor, rel=0.10)
    tail = history[-100:]
    assert (max(tail) - min(tail)) / max(tail) < 0.05, "the coefficient must settle, not ring"


def test_update_record_flips_the_sb3_entropy_sign():
    """SB3 logs entropy_loss as negative mean entropy; unflipped it pins the controller."""
    from asteroid_survival.rl.ppo import _update_record

    controller = _floor()
    values = {"train/entropy_loss": -0.62, "train/approx_kl": 0.016}
    record = _update_record(values, episode=100, steps=2048, wall_seconds=1.0,
                            learning_rate=5e-5, ent_coef=0.0025, controller=controller,
                            target_kl=0.02)
    assert record["entropy"] == pytest.approx(0.62)
    assert record["ent_coef"] == pytest.approx(controller.coefficient)
    assert record["entropy_floor"] == pytest.approx(0.8)
    assert "smoothed_entropy" in record and "entropy_coefficient_base" in record


def test_update_record_still_logs_the_coefficient_without_a_floor():
    """Constant-coefficient runs need the same columns, or the two cannot be compared."""
    from asteroid_survival.rl.ppo import _update_record

    record = _update_record({"train/entropy_loss": -0.62}, episode=1, steps=8,
                            wall_seconds=0.5, learning_rate=5e-5, ent_coef=0.0025)
    assert record["entropy"] == pytest.approx(0.62)
    assert record["ent_coef"] == pytest.approx(0.0025)
    assert "entropy_floor" not in record


def test_entropy_floor_is_opt_in():
    from asteroid_survival.rl.ppo import entropy_floor_controller

    assert entropy_floor_controller(base=0.0025, floor=None) is None
    assert entropy_floor_controller(base=0.0025, floor=0.0) is None
    assert entropy_floor_controller(base=0.0025, floor=0.8).coefficient == pytest.approx(0.0025)


def test_entropy_floor_restores_only_its_own_state():
    from asteroid_survival.rl.ppo import entropy_floor_controller

    stored = {"floor": 0.8, "base": 0.0025, "coefficient": 0.012,
              "smoothed_entropy": 0.61, "updates": 400}
    resumed = entropy_floor_controller(base=0.0025, floor=0.8, restored=stored)
    assert resumed.coefficient == pytest.approx(0.012)
    assert resumed.smoothed == pytest.approx(0.61)

    # A changed flag means the stored coefficient describes a different controller.
    moved = entropy_floor_controller(base=0.0025, floor=1.2, restored=stored)
    assert moved.coefficient == pytest.approx(0.0025)
    assert entropy_floor_controller(base=0.0025, floor=0.8,
                                    restored={"floor": "bad"}).coefficient == pytest.approx(0.0025)


def test_entropy_floor_round_trips_through_settings():
    from dataclasses import asdict

    from asteroid_survival.rl.ppo import PPOTrainSettings

    stored = asdict(PPOTrainSettings(entropy_floor=0.8))
    assert PPOTrainSettings(**stored).entropy_floor == pytest.approx(0.8)
    old = {k: v for k, v in asdict(PPOTrainSettings()).items() if k != "entropy_floor"}
    assert PPOTrainSettings(**old).entropy_floor is None
