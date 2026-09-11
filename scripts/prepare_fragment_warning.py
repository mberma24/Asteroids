"""Prepare an action-warning checkpoint, proving behavior preservation before use."""
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "src")
import numpy as np
import torch
from stable_baselines3 import PPO
from asteroid_survival.rl.curriculum import load_curriculum, task_hash
from asteroid_survival.rl.gym_env import GymAsteroidsEnv
from asteroid_survival.rl.ppo import _stage_env, _task_layout, widen_policy

ROOT = Path("/home/ubuntu/Asteroids/experiments/fragment-warning-v21")
CONFIG = "configs/rl-survival-v3-action-fire.toml"


def main():
    torch.set_num_threads(1)
    metadata = json.loads((ROOT / "source/metadata.json").read_text())
    old_spec = load_curriculum("configs/rl-survival-v3.toml")
    spec = load_curriculum(CONFIG)
    assert task_hash(spec) == task_hash(old_spec)
    assert spec.stages == old_spec.stages and spec.reward == old_spec.reward
    layout = metadata["observation_layout"]
    new_layout = {**layout, "version": 11}
    prototype = _stage_env(spec, 28, new_layout)
    source = PPO.load(ROOT / "source/model.zip", device="cpu")
    settings = metadata["settings"]
    common = {key: settings[key] for key in (
        "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
        "vf_coef", "max_grad_norm")}
    target = PPO("MlpPolicy", GymAsteroidsEnv(prototype), device="cpu",
                 learning_rate=5e-5 / 3, ent_coef=0.0025, target_kl=0.02,
                 policy_kwargs=source.policy_kwargs, seed=2108, **common)
    for key, value in common.items():
        actual = getattr(target, key)
        assert (actual(1.0) if callable(actual) else actual) == value
    assert widen_policy(target, source, torch) == 2
    samples = []
    outcomes = {}
    elapsed = {}
    for label, version, model in (("source", 10, source), ("widened", 11, target)):
        started = time.monotonic()
        rows = []
        for seed in range(1000001300, 1000001316):
            env = _stage_env(spec, 28, {**layout, "version": version})
            obs, _ = env.reset(seed)
            actions = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                actions.append(int(action))
                if label == "widened" and len(samples) < 1024:
                    samples.append(obs.copy())
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            rows.append({"seed": seed, "actions": actions,
                         "cleared": bool(info["episode_metrics"]["completed_stage"])})
        outcomes[label] = rows
        elapsed[label] = time.monotonic() - started
    assert outcomes["source"] == outcomes["widened"], "zero-filled transfer changed decisions"
    with torch.no_grad():
        observation = torch.as_tensor(np.stack(samples))
        old_dist = source.policy.get_distribution(observation[:, :metadata["observation_size"]])
        new_dist = target.policy.get_distribution(observation)
        difference = float((old_dist.distribution.probs-new_dist.distribution.probs).abs().max())
        assert difference < 1e-5
    destination = ROOT / "widened-source"
    destination.mkdir(exist_ok=False)
    target.save(destination / "model.zip")
    metadata = copy.deepcopy(metadata)
    metadata.update(observation_size=prototype.observation_size,
                    observation_layout=_task_layout(spec, prototype, CONFIG),
                    parent_checkpoint=str(ROOT / "source"), episodes=0, environment_steps=0,
                    configured_learning_rate=5e-5/3, effective_learning_rate=5e-5/3)
    metadata["settings"]["learning_rate"] = 5e-5/3
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    report = {"identical_episode_trajectories": 16,
              "decisions": sum(len(r["actions"]) for r in outcomes["source"]),
              "max_action_probability_difference": difference,
              "observation_size": prototype.observation_size,
              "elapsed_seconds": elapsed,
              "throughput_ratio": elapsed["source"]/elapsed["widened"],
              "task_hash_unchanged": True}
    (ROOT / "transfer-check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
