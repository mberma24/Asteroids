"""The oracle-cloning pipeline: recorded labels, the cloner's loss, and the checkpoint it
writes. Slow parts (a real PPO checkpoint) are skipped unless stable-baselines3 is present."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import planning_oracle  # noqa: E402
import clone_oracle  # noqa: E402

from asteroid_survival.rl.curriculum import load_curriculum  # noqa: E402
from asteroid_survival.rl.ppo import _stage_env  # noqa: E402

CURRICULUM = str(ROOT / "configs/rl-survival-v3-action-fire.toml")


def job(seed: int, layout: dict, **overrides) -> dict:
    return {"stage": 28, "seed": seed, "candidates": 2, "horizon": 2, "epsilon": 0.5,
            "curriculum": CURRICULUM, "record": True, "blind": True, "layout": layout,
            "driver": None, **overrides}


def test_decide_marks_the_chosen_action_safe_when_it_survives():
    spec = load_curriculum(CURRICULUM)
    env = _stage_env(spec, 28, planning_oracle.LAYOUT)
    env.reset(5_000_000)
    oracle = planning_oracle.PlanningOracle(env, candidates=4, horizon=3, epsilon=0.5,
                                            seed=1, blind=True)
    action, safe = oracle.decide()
    assert 0 <= action < len(env.actions)
    assert 0 <= safe < 1 << len(env.actions)
    # Three decisions into a fresh round nothing has reached the ship: every candidate
    # survives, so the chosen action must be in the set and the set is non-empty.
    assert (safe >> action) & 1 == 1


def test_recording_uses_the_requested_layout_and_labels(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    layout = {**planning_oracle.LAYOUT, "version": 11}
    (checkpoint / "metadata.json").write_text(json.dumps({"observation_layout": layout}))
    assert planning_oracle.layout_from(str(checkpoint))["version"] == 11
    assert planning_oracle.layout_from(None)["version"] == planning_oracle.LAYOUT["version"]
    result = planning_oracle.run_seed(job(5_000_001, layout))
    trace = result["trace"]
    spec = load_curriculum(CURRICULUM)
    width = _stage_env(spec, 28, layout).observation_size
    assert trace["observations"].shape == (len(trace["actions"]), width)
    assert trace["safe"].dtype == np.uint16
    assert len(trace["safe"]) == len(trace["actions"]) > 0


def test_repeat_labels_record_one_column_per_draw():
    layout = planning_oracle.LAYOUT
    result = planning_oracle.run_seed(job(5_000_004, layout, repeat_labels=3))
    trace = result["trace"]
    assert trace["repeat_actions"].shape == (len(trace["actions"]), 3)
    assert trace["repeat_safe"].shape == (len(trace["actions"]), 3)
    # The recorded single label is the first draw, and the action taken is that one.
    assert (trace["repeat_actions"][:, 0] == trace["actions"]).all()
    assert (trace["repeat_safe"][:, 0] == trace["safe"]).all()


def test_label_noise_bounds_a_cloner_against_a_stochastic_label():
    identical = np.array([[3, 3, 3], [7, 7, 7]])
    safe = np.array([[0b1000, 0b1000, 0b1000], [0b10000000] * 3], dtype=np.uint16)
    same = planning_oracle.label_noise(identical, safe, 16)
    assert same["self_agreement"] == pytest.approx(1.0)
    assert same["safe_set_jaccard"] == pytest.approx(1.0)
    assert same["chosen_in_other_safe_set"] == pytest.approx(1.0)
    assert same["plurality_share"] == pytest.approx(1.0)

    # Every draw differs: the label is pure noise, and no learner can do better than the
    # most common action -- self-agreement is 0 even though each pick was survivable.
    disjoint = np.array([[0, 1, 2]])
    sets = np.array([[0b0111, 0b0111, 0b0111]], dtype=np.uint16)
    noisy = planning_oracle.label_noise(disjoint, sets, 16)
    assert noisy["self_agreement"] == pytest.approx(0.0)
    assert noisy["chosen_in_other_safe_set"] == pytest.approx(1.0)
    assert noisy["mean_safe_set_size"] == pytest.approx(3.0)
    assert planning_oracle.label_noise(np.zeros((0, 2), int), np.zeros((0, 2), np.uint16),
                                       16) == {}


def test_set_loss_rewards_any_safe_action_and_ignores_empty_sets():
    logits = torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    actions = torch.tensor([0, 3])
    # Row 0: safe = {1, 2}; row 1: no safe set recorded.
    safe = torch.tensor([0b0110, 0])
    total, parts = clone_oracle.losses(logits, actions, safe, set_weight=1.0)
    assert parts["set"] == pytest.approx(-float(torch.log_softmax(logits[0], -1)[1:3]
                                                .exp().sum().log()), abs=1e-6)
    assert total.item() == pytest.approx(parts["argmax"] + parts["set"])
    only_argmax, _ = clone_oracle.losses(logits, actions, safe, set_weight=0.0)
    assert only_argmax.item() == pytest.approx(parts["argmax"])


def test_episode_split_holds_out_whole_episodes():
    episodes = np.repeat(np.arange(10), 3)
    train = clone_oracle.episode_split(episodes, 0.2)
    assert train.sum() == 24 and (~train).sum() == 6
    assert set(episodes[~train]) == {8, 9}


def test_load_traces_offsets_episode_ids(tmp_path):
    for index in range(2):
        np.savez(tmp_path / f"t{index}.npz", observations=np.zeros((4, 3), np.float32),
                 actions=np.zeros(4, np.int64), safe=np.zeros(4, np.uint16),
                 episodes=np.array([0, 0, 1, 1], np.int32))
    data = clone_oracle.load_traces([str(tmp_path / "t0.npz"), str(tmp_path / "t1.npz")])
    assert data["episodes"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None,
                    reason="needs stable-baselines3")
@pytest.mark.parametrize("net_arch", [None, "16,16"])
def test_clone_writes_a_checkpoint_initialize_from_accepts(tmp_path, net_arch):
    import subprocess
    from stable_baselines3 import PPO
    from asteroid_survival.rl.gym_env import GymAsteroidsEnv
    from asteroid_survival.rl.ppo import PPOController, _task_layout

    spec = load_curriculum(CURRICULUM)
    layout = {**planning_oracle.LAYOUT, "version": 11}
    prototype = _stage_env(spec, 28, layout)
    model = PPO("MlpPolicy", GymAsteroidsEnv(prototype), device="cpu", n_steps=8,
                batch_size=8, policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, seed=0)
    source = tmp_path / "source"
    source.mkdir()
    model.save(source / "model.zip")
    (source / "metadata.json").write_text(json.dumps({
        "algorithm": "ppo", "recurrent": False, "episodes": 5, "environment_steps": 50,
        "observation_size": prototype.observation_size, "num_actions": prototype.num_actions,
        "observation_layout": _task_layout(spec, prototype, CURRICULUM),
        "settings": {"n_steps": 8, "batch_size": 8}}))
    results = [planning_oracle.run_seed(job(seed, layout))["trace"]
               for seed in (5_000_002, 5_000_003)]
    trace = tmp_path / "trace.npz"
    np.savez(trace, **{k: np.concatenate([r[k] for r in results]) for k in results[0]},
             episodes=np.concatenate([np.full(len(r["actions"]), i, np.int32)
                                      for i, r in enumerate(results)]))
    output = tmp_path / "clone"
    command = [sys.executable, str(ROOT / "scripts/clone_oracle.py"), "--source",
               str(source), "--output", str(output), "--traces", str(trace),
               "--epochs", "2", "--batch", "64", "--threads", "1", "--validation", "0.5"]
    if net_arch:
        command += ["--net-arch", net_arch]
    subprocess.run(command, check=True, cwd=ROOT,
                   env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    report = json.loads((output / "clone-report.json").read_text())
    assert report["epochs"][-1]["train"]["argmax"] < report["epochs"][0]["train"]["argmax"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["episodes"] == 0 and metadata["parent_checkpoint"] == str(source)
    controller = PPOController(output, device="cpu")
    observation, _ = prototype.reset(5_000_004)
    assert 0 <= controller(observation) < prototype.num_actions
    # The rebuilt width has to survive the save/load round trip, or PPO would silently
    # fine-tune a differently shaped network than the one that was cloned.
    width = controller.model.policy.mlp_extractor.policy_net[0].out_features
    assert width == (16 if net_arch else 8)
    if net_arch:
        assert report["rebuilt"]["net_arch"] == {"pi": [16, 16], "vf": [8]}
        # The critic came across; only the actor stack and head are new.
        assert 0 < report["rebuilt"]["parameters_transferred"] < \
            report["rebuilt"]["parameters_total"]
        value = controller.model.policy.mlp_extractor.value_net[0]
        original = PPO.load(source / "model.zip", device="cpu")
        assert torch.equal(value.weight, original.policy.mlp_extractor.value_net[0].weight)
