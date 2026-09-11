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
    result = planning_oracle.run_seed(
        (28, 5_000_001, 2, 2, 0.5, CURRICULUM, True, True, layout, None))
    trace = result["trace"]
    spec = load_curriculum(CURRICULUM)
    width = _stage_env(spec, 28, layout).observation_size
    assert trace["observations"].shape == (len(trace["actions"]), width)
    assert trace["safe"].dtype == np.uint16
    assert len(trace["safe"]) == len(trace["actions"]) > 0


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
def test_clone_writes_a_checkpoint_initialize_from_accepts(tmp_path):
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
    results = [planning_oracle.run_seed(
        (28, seed, 2, 2, 0.5, CURRICULUM, True, True, layout, None))["trace"]
        for seed in (5_000_002, 5_000_003)]
    trace = tmp_path / "trace.npz"
    np.savez(trace, **{k: np.concatenate([r[k] for r in results]) for k in results[0]},
             episodes=np.concatenate([np.full(len(r["actions"]), i, np.int32)
                                      for i, r in enumerate(results)]))
    output = tmp_path / "clone"
    subprocess.run([sys.executable, str(ROOT / "scripts/clone_oracle.py"), "--source",
                    str(source), "--output", str(output), "--traces", str(trace),
                    "--epochs", "2", "--batch", "64", "--threads", "1", "--validation",
                    "0.5"], check=True, cwd=ROOT,
                   env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    report = json.loads((output / "clone-report.json").read_text())
    assert report["epochs"][-1]["train"]["argmax"] < report["epochs"][0]["train"]["argmax"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["episodes"] == 0 and metadata["parent_checkpoint"] == str(source)
    controller = PPOController(output, device="cpu")
    observation, _ = prototype.reset(5_000_003)
    assert 0 <= controller(observation) < prototype.num_actions
