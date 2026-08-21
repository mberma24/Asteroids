"""Head-to-head comparison of a human, the greedy baseline, and any number of checkpoints.

Every contender plays alone, on its own run of the same seeds, through the same
:class:`AsteroidsRLEnv`, so nobody clears anyone else's asteroids and the reported metrics
are computed identically. That is the difference from `showdown`, which puts everyone in one
shared arena where one player's kills help the others.

The human is sampled once per decision rather than once per frame, matching the agents'
control rate instead of giving the human four times the reactions.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

from ..config import GameConfig, load_config
from ..controllers import ClosestAsteroidController, KeyProfile, action_from_components
from .environment import AsteroidsRLEnv

CONTENDERS = ("human", "greedy", "muzero", "ppo", "lstm_ppo")
REPORTED = ("survival_time", "wave", "asteroids_destroyed", "accuracy", "shots_fired")


def _make_env(config: GameConfig, max_decisions: int, asteroid_reward: float,
              shot_penalty: float, history_frames: int = 0, history_long_frames: int = 0,
              history_long_stride: int = 8, max_projectiles: int = 8,
              global_features: bool = False) -> AsteroidsRLEnv:
    return AsteroidsRLEnv(config, frame_skip=4, max_decisions=max_decisions,
                          asteroid_reward=asteroid_reward, shot_penalty=shot_penalty,
                          history_frames=history_frames,
                          history_long_frames=history_long_frames,
                          history_long_stride=history_long_stride,
                          max_projectiles=max_projectiles,
                          global_features=global_features)


def run_policy(env: AsteroidsRLEnv, policy: Callable[[Any], int], seeds: list[int]) -> list[dict]:
    episodes = []
    for seed in seeds:
        observation, _ = env.reset(seed)
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(policy(observation))
            done = terminated or truncated
        episodes.append(info["episode_metrics"])
    return episodes


def pilot_episodes(config: GameConfig, seeds: list[int], **kwargs) -> list[dict]:
    """The scripted pilot: leads its shots and dodges. A harder yardstick than greedy."""
    from ..controllers import PilotController

    env = _make_env(config, **kwargs)
    controller = PilotController()

    def policy(observation) -> int:
        del observation
        assert env.state is not None
        return env.actions.index(controller.action(env.state, env.agent_id))

    return run_policy(env, policy, seeds)


def greedy_episodes(config: GameConfig, seeds: list[int], **kwargs) -> list[dict]:
    env = _make_env(config, **kwargs)
    controller = ClosestAsteroidController()

    def policy(observation) -> int:
        del observation
        assert env.state is not None
        return env.actions.index(controller.action(env.state, env.agent_id))

    return run_policy(env, policy, seeds)


def muzero_episodes(config: GameConfig, checkpoint: str | Path, seeds: list[int],
                    *, seed: int = 0, **kwargs) -> list[dict]:
    from .muzero import MuZeroAgent

    metadata = json.loads((Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
    layout = metadata.get("observation_layout") or {}
    env = _make_env(
        config, max_projectiles=int(layout.get("max_projectiles", 0)), **kwargs)
    agent = MuZeroAgent.load(checkpoint, seed=seed)
    if (agent.observation_size, agent.num_actions) != (env.observation_size, env.num_actions):
        raise SystemExit(
            f"checkpoint {checkpoint} expects observation size {agent.observation_size} but this "
            f"config produces {env.observation_size}; it was trained on a different observation")
    return run_policy(env, lambda observation: agent.search(observation, explore=False)[0], seeds)


def ppo_episodes(config: GameConfig, checkpoint: str | Path, seeds: list[int],
                 **kwargs) -> list[dict]:
    from .ppo import PPOController

    metadata = json.loads((Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
    layout = metadata.get("observation_layout") or {}
    # Observation v5 appends global threat features. Without this the scorer hands a v4
    # observation to a v5 policy and stable-baselines3 rejects the shape outright, so
    # `watch`/`compare`/`versus` could not score any v5 checkpoint at all.
    env = _make_env(config, max_projectiles=int(layout.get("max_projectiles", 0)),
                    global_features=int(layout.get("version", 4)) >= 5, **kwargs)
    controller = PPOController(checkpoint)
    episodes = []
    for episode_seed in seeds:
        controller.reset()
        observation, _ = env.reset(episode_seed)
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(controller(observation))
            done = terminated or truncated
        episodes.append(info["episode_metrics"])
    return episodes


def human_episodes(config: GameConfig, seeds: list[int], **kwargs) -> list[dict]:
    """Play the given seeds interactively, one decision per ``frame_skip`` frames."""
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on the optional display extra
        raise SystemExit("Pygame is required to play. Run: pip install -e .") from exc
    from ..renderer import Renderer

    env = _make_env(config, **kwargs)
    pygame.init()
    renderer = Renderer(pygame, config.arena.width, config.arena.height)
    clock = pygame.time.Clock()
    profile = KeyProfile(
        pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_SPACE,
        pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP,
    )
    episodes: list[dict] = []
    try:
        for index, seed in enumerate(seeds, start=1):
            print(f"  seed {seed} ({index}/{len(seeds)}) - A/D or arrows to turn, "
                  "space to fire, Escape to give up on this run")
            env.reset(seed)
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit("comparison aborted")
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        done = True
                    if event.type == pygame.VIDEORESIZE:
                        renderer.resize(event.w, event.h)
                if done:
                    break
                keys = pygame.key.get_pressed()

                def held(primary: int, alternate: int | None = None) -> bool:
                    return bool(keys[primary]) or (alternate is not None and bool(keys[alternate]))

                action = action_from_components(
                    int(held(profile.right, profile.alternate_right))
                    - int(held(profile.left, profile.alternate_left)),
                    config.ship.mobile and held(profile.thrust, profile.alternate_thrust),
                    held(profile.fire),
                )

                def render(snapshot) -> None:
                    renderer.draw(snapshot, False)
                    clock.tick(config.arena.fps)

                _, _, terminated, truncated, info = env.step(
                    env.actions.index(action), on_frame=render)
                done = terminated or truncated
            if "episode_metrics" in info:
                episodes.append(info["episode_metrics"])
                print(f"    survived {info['episode_metrics']['survival_time']:.2f}s, "
                      f"{info['episode_metrics']['asteroids_destroyed']} destroyed")
    finally:
        pygame.quit()
    return episodes


def summarize(episodes: list[dict]) -> dict:
    summary: dict = {"episodes": len(episodes)}
    for field in REPORTED:
        values = [float(episode[field]) for episode in episodes]
        summary[field] = {
            "avg": statistics.fmean(values),
            "best": max(values),
            "worst": min(values),
        }
    return summary


def contender_label(checkpoint: str | Path, taken: set[str]) -> str:
    """A short, unique name for a checkpoint, taken from the run it belongs to.

    Run directories carry a launch timestamp that is noise in a results table, so it is
    dropped -- and put back only if two runs would otherwise collide.
    """
    path = Path(checkpoint)
    stem = path.parent.name or path.name
    for prefix in ("ppo-", "muzero-"):
        stem = stem.removeprefix(prefix)
    short = re.sub(r"-\d{4}-\d{4}$", "", stem)
    for candidate in (short, stem, f"{stem}/{path.name}"):
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    taken.add(stem)
    return stem


def format_table(results: dict[str, dict]) -> str:
    width = max([10] + [len(name) + 2 for name in results])
    header = (f"{'':{width}}{'survival avg':>14}{'best':>9}{'worst':>9}"
              f"{'wave avg':>10}{'kills avg':>11}{'accuracy':>10}")
    lines = [header, "-" * len(header)]
    # Best survival first, so the ranking is the first thing read.
    ordered = sorted(results.items(),
                     key=lambda item: -item[1]["survival_time"]["avg"])
    for name, summary in ordered:
        if not summary:
            continue
        lines.append(
            f"{name:{width}}"
            f"{summary['survival_time']['avg']:>13.2f}s"
            f"{summary['survival_time']['best']:>8.2f}s"
            f"{summary['survival_time']['worst']:>8.2f}s"
            f"{summary['wave']['avg']:>10.1f}"
            f"{summary['asteroids_destroyed']['avg']:>11.2f}"
            f"{summary['accuracy']['avg']:>10.3f}"
        )
    return "\n".join(lines)


def compare(config_path: str | Path | GameConfig, output_path: str | Path, *,
            checkpoint: str | Path | None = None,
            checkpoints: list[str | Path] | None = None,
            episodes: int, seed: int, max_decisions: int, asteroid_reward: float = 0.1,
            shot_penalty: float = 0.0, history_frames: int = 0,
            history_long_frames: int = 0, history_long_stride: int = 8,
            include_human: bool = True, include_greedy: bool = True,
            include_pilot: bool = True) -> dict:
    config = config_path if isinstance(config_path, GameConfig) else load_config(config_path)
    seeds = list(range(seed, seed + episodes))
    env_kwargs = {"max_decisions": max_decisions, "asteroid_reward": asteroid_reward,
                  "shot_penalty": shot_penalty, "history_frames": history_frames,
                  "history_long_frames": history_long_frames,
                  "history_long_stride": history_long_stride}
    played: dict[str, list[dict]] = {}

    entries = list(checkpoints or [])
    if checkpoint and checkpoint not in entries:
        entries.insert(0, checkpoint)

    if include_human:
        print(f"Your turn: {episodes} runs on seeds {seeds[0]}-{seeds[-1]}.")
        played["human"] = human_episodes(config, seeds, **env_kwargs)
    if include_greedy:
        print("Running greedy baseline...")
        played["greedy"] = greedy_episodes(config, seeds, **env_kwargs)
    if include_pilot:
        print("Running pilot baseline...")
        played["pilot"] = pilot_episodes(config, seeds, **env_kwargs)

    taken = set(played)
    lineup: dict[str, str] = {}
    for entry in entries:
        metadata = json.loads(
            (Path(entry) / "metadata.json").read_text(encoding="utf-8"))
        algorithm = metadata.get("algorithm", "muzero")
        label = contender_label(entry, taken)
        lineup[label] = str(entry)
        print(f"Running {algorithm} {entry} as '{label}'...")
        if algorithm in {"ppo", "recurrent_ppo"}:
            played[label] = ppo_episodes(config, entry, seeds, **env_kwargs)
        else:
            played[label] = muzero_episodes(config, entry, seeds, **env_kwargs)

    results = {name: summarize(runs) for name, runs in played.items() if runs}
    report = {
        "seeds": seeds,
        "checkpoint": str(entries[0]) if entries else None,
        "lineup": lineup,
        "summary": results,
        "episodes": played,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
