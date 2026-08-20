from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .curriculum import load_curriculum, task_hash_matches
from .environment import AsteroidsRLEnv


def _score(record: dict) -> float:
    stages = record.get("stages") or []
    if not stages:
        return float("-inf")
    index = min(int(record.get("training_stage", 0)), len(stages) - 1)
    stage = stages[index]
    return (1_000_000.0 * index
            + 1000.0 * float(stage.get("completion_rate", 0.0))
            + 10.0 * float(stage.get("mean_wave", 0.0))
            + float(stage.get("mean_accuracy", 0.0)))


def resolve_checkpoint(target: str | Path) -> tuple[Path, dict | None]:
    """Resolve a checkpoint directly, or the best held-out checkpoint inside a run."""
    target = Path(target)
    if (target / "metadata.json").is_file():
        checkpoint = target
    elif (target / "champion" / "metadata.json").is_file():
        checkpoint = target / "champion"
    else:
        log = target / "evaluation.jsonl"
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if log.is_file() else []
        eligible = [record for record in records
                    if (target / f"checkpoint_{int(record['episode']):06d}").is_dir()]
        if eligible:
            checkpoint = target / f"checkpoint_{int(max(eligible, key=_score)['episode']):06d}"
        else:
            checkpoints = sorted(target.glob("checkpoint_*"))
            if not checkpoints:
                raise SystemExit(f"no checkpoint found in {target}")
            checkpoint = checkpoints[-1]

    if checkpoint.name == "champion":
        state = checkpoint.parent / "champion_state.json"
        episode = (int(json.loads(state.read_text(encoding="utf-8")).get("episode", 0))
                   if state.is_file() else 0)
    else:
        episode = int(checkpoint.name.removeprefix("checkpoint_") or 0)
    log = checkpoint.parent / "evaluation.jsonl"
    record = None
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            candidate = json.loads(line)
            if int(candidate.get("episode", -1)) == episode:
                record = candidate
                break
    return checkpoint, record


def preview_checkpoint(target: str | Path, *, seed: int = 10_000,
                       curriculum_path: str | Path | None = None,
                       stage_index: int | None = None) -> None:
    """Play a deterministic evaluation episode for a saved curriculum checkpoint."""
    checkpoint, evaluation = resolve_checkpoint(target)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    layout = metadata.get("observation_layout") or {}
    curriculum_path = curriculum_path or layout.get("curriculum")
    if not curriculum_path:
        raise SystemExit("checkpoint does not record a curriculum; pass --curriculum")
    spec = load_curriculum(curriculum_path)
    if not task_hash_matches(layout.get("task_hash"), spec):
        # A warning, not a hard stop. Preview is for looking at a model, and refusing to
        # show it is worse than showing it against a curriculum that has since drifted --
        # especially since a code change alone has caused this false alarm before.
        print(f"warning: {curriculum_path} has changed since this checkpoint was trained, "
              "so the round below may not be the one it learned", flush=True)
    if stage_index is None:
        stage_index = int((evaluation or {}).get("training_stage", 0))
    if not 0 <= stage_index < len(spec.stages):
        raise SystemExit(f"stage must be between 1 and {len(spec.stages)}")
    stage = spec.stages[stage_index]
    config = stage.game_config(spec.base)
    stage_reward = (spec.reward if stage.miss_penalty is None else
                    replace(spec.reward, miss_penalty=stage.miss_penalty))
    env = AsteroidsRLEnv(
        config, frame_skip=4, max_decisions=stage.max_decisions,
        history_frames=int(layout.get("history_frames", 0)),
        history_long_frames=int(layout.get("history_long_frames", 0)),
        history_long_stride=int(layout.get("history_long_stride", 8)),
        max_projectiles=int(layout.get("max_projectiles", 0)),
        reward_config=stage_reward, completion=stage.completion)
    algorithm = metadata.get("algorithm", "muzero")
    if algorithm in {"ppo", "recurrent_ppo"}:
        from .ppo import PPOController
        controller = PPOController(checkpoint)
        observation_size = int(metadata["observation_size"])
        num_actions = int(metadata["num_actions"])
        label = "LSTM-PPO" if metadata.get("recurrent") else "PPO"
    else:
        from .muzero import MuZeroAgent
        agent = MuZeroAgent.load(checkpoint, seed=seed)
        agent.settings.num_simulations = max(agent.num_actions, agent.settings.num_simulations)
        controller = None
        observation_size, num_actions = agent.observation_size, agent.num_actions
        label = "MuZero"
    if (observation_size, num_actions) != (env.observation_size, env.num_actions):
        raise SystemExit("checkpoint observation/action shape does not match this curriculum")

    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on graphical installation
        raise SystemExit("Pygame is required to preview a checkpoint") from exc
    from ..renderer import Renderer

    pygame.init()
    renderer = Renderer(pygame, config.arena.width, config.arena.height)
    pygame.display.set_caption(
        f"{label} preview — {checkpoint.name} — stage {stage_index + 1}: {stage.name}")
    clock = pygame.time.Clock()
    current_seed = seed
    observation, _ = env.reset(current_seed)
    done = False
    running = True
    print(f"previewing {checkpoint} | stage {stage_index + 1}: {stage.name} | seed {current_seed}")
    print("R repeats, N uses the next seed, Escape closes the preview")
    try:
        while running:
            restart = False
            next_seed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    renderer.resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    restart = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    next_seed = True
            if not running:
                break
            if restart or next_seed:
                current_seed += int(next_seed)
                observation, _ = env.reset(current_seed)
                if controller is not None:
                    controller.reset()
                done = False
            if done:
                assert env.state is not None
                renderer.draw(env.state, False)
                clock.tick(30)
                continue

            action = (controller(observation) if controller is not None else
                      agent.search(observation, explore=False)[0])

            def render(snapshot) -> None:
                renderer.draw(snapshot, False)
                clock.tick(config.arena.fps)

            observation, _, terminated, truncated, info = env.step(action, on_frame=render)
            done = terminated or truncated
            if done:
                metrics = info["episode_metrics"]
                print(f"seed {current_seed}: completion {bool(metrics['completed_stage'])}, "
                      f"kills {metrics['asteroids_destroyed']}, "
                      f"accuracy {metrics['accuracy']:.3f}, reward {metrics['reward']:.3f}")
    finally:
        pygame.quit()
