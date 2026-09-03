from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from .actions import Action
from .config import GameConfig, ShipSpec, load_config
from .modes import MODES, build as build_mode
from .controllers import (ClosestAsteroidController, HeuristicController, KeyProfile,
                          PilotController, RandomController, gamepad_action, human_action)
from .simulation import Simulation


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.toml"


def _checkpoint_fits(checkpoint: Path, config: GameConfig) -> bool:
    """Whether a checkpoint's observation layout matches this configuration."""
    from .rl.environment import (ASTEROID_FEATURES, MOBILE_ACTIONS, PROJECTILE_FEATURES,
                                 SHIP_FEATURES,
                                 STATIONARY_ACTIONS, ship_feature_count)

    try:
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        size = int(metadata["observation_size"])
    except (OSError, ValueError, KeyError):
        return False
    layout = metadata.get("observation_layout") or {}
    if layout.get("version", 1) >= 2:
        return (
            layout.get("mobile") == config.ship.mobile
            and int(layout.get("max_asteroids", -1)) == config.asteroid.active_cap
            and layout.get("actions") == [action.name for action in (
                MOBILE_ACTIONS if config.ship.mobile else STATIONARY_ACTIONS)]
            and size == ship_feature_count(config) + config.asteroid.active_cap * (
                ASTEROID_FEATURES + 2 * (int(layout.get("history_frames", 0))
                                         + int(layout.get("history_long_frames", 0)))) + (
                                             int(layout.get("max_projectiles", 0))
                                             * PROJECTILE_FEATURES) + (
                                             int(layout.get("max_teammates", 0)) * 8) + (
                                             int(layout.get("global_features", 0)))
        )
    width, remainder = divmod(size - SHIP_FEATURES, config.asteroid.active_cap)
    slots, odd = divmod(width - ASTEROID_FEATURES, 2)
    return remainder == 0 and odd == 0 and slots >= 0


def _evaluation_scores() -> dict[Path, float]:
    """Held-out survival for every checkpoint whose run recorded an evaluation."""
    models = Path(__file__).resolve().parents[2] / "models"
    scores: dict[Path, float] = {}
    for log in models.glob("*/evaluation.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                episode = int(record["episode"])
            except (ValueError, KeyError):
                continue
            if record.get("stages"):
                index = min(int(record.get("training_stage", 0)), len(record["stages"]) - 1)
                stage = record["stages"][index]
                survival = (1_000_000.0 * index
                            + 1000.0 * float(stage["completion_rate"])
                            + 10.0 * float(stage["mean_wave"])
                            + float(stage["mean_accuracy"]))
            else:
                try:
                    survival = float(record["mean_survival_time"])
                except (ValueError, KeyError):
                    continue
            checkpoint = log.parent / f"checkpoint_{episode:06d}"
            if checkpoint.is_dir():
                scores[checkpoint] = survival
        champion_state = log.parent / "champion_state.json"
        champion = log.parent / "champion"
        if champion.is_dir() and champion_state.is_file():
            try:
                state = json.loads(champion_state.read_text(encoding="utf-8"))
                score = state.get("score", [-1, 0.0, 0.0])
                scores[champion] = (1_000_000.0 * float(score[0])
                                    + 1000.0 * float(score[1]) + float(score[2]))
            except (OSError, ValueError, TypeError, IndexError):
                pass
    return scores


def _latest_checkpoint(config: GameConfig | None = None, *, best: bool = True,
                       algorithms: set[str] | None = None,
                       all_runs: bool = False) -> Path:
    """Pick a checkpoint this config can load, from the run you are currently training.

    Selection is two-stage, and the order matters. Across runs it takes the **most recent**
    one; inside that run it takes the **best** held-out checkpoint, because training is noisy
    and the final checkpoint is regularly worse than an earlier one.

    Ranking by score across every run instead looks reasonable and is quietly wrong: an
    abandoned run that happened to reach a later curriculum stage outranks the run you are
    actually training, so `showdown` and `preview` keep showing a stale model and have to be
    overridden by hand every time. Set ``all_runs`` to search everything by score anyway.
    """
    models = Path(__file__).resolve().parents[2] / "models"
    checkpoints = [
        path for path in models.glob("*/checkpoint_*")
        if path.is_dir() and (path / "metadata.json").is_file()
    ]
    checkpoints.extend(
        path for path in models.glob("*/champion")
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    if algorithms is not None:
        checkpoints = [path for path in checkpoints if json.loads(
            (path / "metadata.json").read_text(encoding="utf-8")).get(
                "algorithm", "muzero") in algorithms]
    if not checkpoints:
        raise SystemExit(
            "no compatible checkpoint found under models/; pass --checkpoint or train one first")
    checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if best:
        scored = _evaluation_scores()
        checkpoints.sort(key=lambda path: scored.get(path, float("-inf")), reverse=True)
    usable = [path for path in checkpoints
              if config is None or _checkpoint_fits(path, config)]
    if usable and not all_runs:
        newest_run = max(usable, key=lambda path: path.stat().st_mtime).parent
        usable = [path for path in usable if path.parent == newest_run]
    if not usable:
        listing = "\n".join(f"  {path}" for path in checkpoints[:8])
        raise SystemExit(
            "no checkpoint matches this configuration's observation layout; every model "
            f"below was trained on a different one:\n{listing}\n"
            "Train a new model, or pass --checkpoint with a matching config.")
    return usable[0]


def _controllers(config: GameConfig, seed: int):
    result = {}
    for i, spec in enumerate(config.ships):
        if spec.controller == "random":
            result[spec.id] = RandomController(seed + 1009 * (i + 1))
        elif spec.controller == "heuristic":
            result[spec.id] = HeuristicController()
        elif spec.controller == "closest":
            result[spec.id] = ClosestAsteroidController()
        elif spec.controller == "pilot":
            result[spec.id] = PilotController()
    return result


def _curriculum_round_config(curriculum_path: str | Path, round_number: int) -> tuple[GameConfig, str]:
    """Build one exact curriculum round for a human-controlled play session."""
    from .rl.curriculum import load_curriculum

    spec = load_curriculum(curriculum_path)
    if not 1 <= round_number <= len(spec.stages):
        raise ValueError(f"round must be between 1 and {len(spec.stages)}")
    stage = spec.stages[round_number - 1]
    config = stage.game_config(spec.base)
    config.ships = [ShipSpec("you", "human", "keyboard_1")]
    if stage.survival:
        # A survival round is cleared by lasting max_seconds, so stop the human there too
        # rather than letting the round run on past the point training scores.
        config.objective.max_steps = stage.max_decisions * 4
    config.validate()
    return config, stage.name


def simulate(config: GameConfig, seed: int, max_steps: int | None) -> int:
    humans = [s.id for s in config.ships if s.controller == "human"]
    if humans:
        raise SystemExit(f"headless mode cannot use human controllers: {', '.join(humans)}")
    sim = Simulation(config)
    state = sim.reset(seed)
    controllers = _controllers(config, seed)
    limit = max_steps or config.objective.max_steps or 100_000
    while not state.terminated and not state.truncated and state.step < limit:
        actions = {ship_id: controller.action(state, ship_id) for ship_id, controller in controllers.items()}
        state = sim.step(actions).snapshot
    print(json.dumps({
        "seed": seed, "steps": state.step, "elapsed": state.elapsed,
        "living_ships": sum(s.alive for s in state.ships), "asteroids": len(state.asteroids),
        "terminated": state.terminated, "truncated": state.truncated,
        "terminal_reason": state.terminal_reason.value if state.terminal_reason else "cli_step_limit",
        "object_health": state.objective.health if state.objective.enabled else None,
    }, indent=2))
    return 0


def play(config: GameConfig, seed: int, checkpoint: Path | None = None, *,
         trails: bool = False, all_runs: bool = False) -> int:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Pygame is required for play mode. Run: pip install -e .") from exc
    from .renderer import Renderer
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    profiles = {
        "keyboard_1": KeyProfile(
            pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_SPACE,
            pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP,
        ),
        "keyboard_2": KeyProfile(pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_RCTRL),
    }
    sim = Simulation(config)
    state = sim.reset(seed)
    controllers = _controllers(config, seed)
    muzero_specs = [spec for spec in config.ships if spec.controller == "muzero"]
    ppo_specs = [spec for spec in config.ships if spec.controller == "ppo"]
    if muzero_specs and ppo_specs:
        raise SystemExit("play mode supports one learned checkpoint algorithm at a time")
    if muzero_specs:
        from .rl.controller import MuZeroController
        selected_checkpoint = checkpoint or _latest_checkpoint(
            config, algorithms={"muzero"}, all_runs=all_runs)
        selected_metadata = json.loads(
            (selected_checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if selected_metadata.get("algorithm", "muzero") != "muzero":
            raise SystemExit("showdown currently requires a MuZero checkpoint")
        print(f"using MuZero checkpoint: {selected_checkpoint}")
        for index, spec in enumerate(muzero_specs):
            controllers[spec.id] = MuZeroController(
                config, selected_checkpoint, seed=seed + index)
    elif ppo_specs:
        from .rl.controller import PPOPlayController
        selected_checkpoint = checkpoint or _latest_checkpoint(
            config, algorithms={"ppo", "recurrent_ppo"}, all_runs=all_runs)
        selected_metadata = json.loads(
            (selected_checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if selected_metadata.get("algorithm") not in {"ppo", "recurrent_ppo"}:
            raise SystemExit("PPO showdown requires a PPO or recurrent PPO checkpoint")
        label = "LSTM-PPO" if selected_metadata.get("recurrent") else "PPO"
        print(f"using {label} checkpoint: {selected_checkpoint}")
        for spec in ppo_specs:
            controllers[spec.id] = PPOPlayController(config, selected_checkpoint)
    renderer = Renderer(pygame, config.arena.width, config.arena.height, trails=trails)
    clock, paused, running = pygame.time.Clock(), False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                renderer.resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = sim.reset(seed)
                controllers = _controllers(config, seed)
                if muzero_specs:
                    for index, spec in enumerate(muzero_specs):
                        controllers[spec.id] = MuZeroController(
                            config, selected_checkpoint, seed=seed + index)
                elif ppo_specs:
                    for spec in ppo_specs:
                        controllers[spec.id] = PPOPlayController(config, selected_checkpoint)
                paused = False
        if not paused and not state.terminated and not state.truncated:
            keys = pygame.key.get_pressed()
            actions: dict[str, Action] = {}
            gamepad_index = 0
            for spec in config.ships:
                if spec.controller != "human":
                    actions[spec.id] = controllers[spec.id].action(state, spec.id)
                elif spec.input_profile.startswith("gamepad"):
                    try:
                        index = int(spec.input_profile.removeprefix("gamepad_")) - 1
                    except ValueError:
                        index = gamepad_index
                    actions[spec.id] = gamepad_action(joysticks[index]) if index < len(joysticks) else Action.NOOP
                    gamepad_index += 1
                else:
                    profile = profiles.get(spec.input_profile, profiles["keyboard_1"])
                    actions[spec.id] = human_action(keys, profile)
            state = sim.step(actions).snapshot
        renderer.draw(state, paused)
        clock.tick(config.arena.fps)
    pygame.quit()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asteroid-survival")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("play", "simulate"):
        child = sub.add_parser(name)
        child.add_argument("--config", type=Path, default=_default_config())
        child.add_argument("--seed", type=int, default=0)
        if name == "play":
            child.add_argument("--checkpoint", type=Path)
        if name == "simulate":
            child.add_argument("--max-steps", type=int)
    arena_parser = sub.add_parser(
        "arena", help="play any mode, alone or against greedy and a trained model")
    arena_parser.add_argument("mode", nargs="?", default="arcade",
                              choices=sorted(MODES), help="which mode to play")
    arena_parser.add_argument("round", nargs="?", type=int,
                              help="round number for round-based modes")
    arena_parser.add_argument(
        "--with", dest="opponents", default="",
        help="comma-separated lineup to add, e.g. closest,ppo")
    arena_parser.add_argument("--seed", type=int, default=7)
    arena_parser.add_argument("--checkpoint", type=Path)
    arena_parser.add_argument(
        "--any-run", action="store_true",
        help="pick the best-scoring model from any run, not just the newest")
    patterns_parser = sub.add_parser(
        "patterns", help="watch trajectory shapes with motion trails and labels")
    patterns_parser.add_argument(
        "pattern", nargs="?", help="one pattern to watch; omit to see them all mixed")
    patterns_parser.add_argument("--seed", type=int, default=3)
    best_parser = sub.add_parser(
        "best-checkpoint", help="print the best checkpoint a mode can load")
    best_parser.add_argument("mode", nargs="?", default="arcade", choices=sorted(MODES))
    best_parser.add_argument("round", nargs="?", type=int)
    best_parser.add_argument("--algorithms", default="ppo,recurrent_ppo")
    best_parser.add_argument("--any-run", action="store_true")
    play_round_parser = sub.add_parser(
        "play-round", help="personally play one exact curriculum round")
    play_round_parser.add_argument("round", type=int)
    play_round_parser.add_argument(
        "--curriculum", type=Path, default=Path("configs/rl-nonlinear.toml"))
    play_round_parser.add_argument("--seed", type=int, default=7)
    train_parser = sub.add_parser("train", help="train and checkpoint a MuZero agent")
    train_parser.add_argument("--config", type=Path, default=Path("configs/rl.toml"))
    train_parser.add_argument(
        "--curriculum", type=Path,
        help="mastery-gated curriculum TOML; replaces --config and legacy reward flags")
    train_parser.add_argument("--output", type=Path, default=Path("models/muzero"))
    train_parser.add_argument("--episodes", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--simulations", type=int, default=16)
    train_parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Adam learning rate for a fresh or policy-initialized run")
    train_parser.add_argument(
        "--updates-per-episode", type=int, default=32,
        help="gradient batches per completed episode; lower values train faster")
    train_parser.add_argument("--max-decisions", type=int, default=900)
    train_parser.add_argument("--checkpoint-every", type=int, default=10)
    train_parser.add_argument(
        "--keep-checkpoints", type=int, default=3,
        help="retain the best evaluated checkpoint and newest checkpoints (0 keeps all)")
    train_parser.add_argument(
        "--stop-when-mastered", action="store_true",
        help="stop early after the final curriculum stage passes its mastery gate")
    train_parser.add_argument(
        "--champion-patience", type=int, default=4,
        help="evaluations without improvement before restoring the best model")
    train_parser.add_argument(
        "--rollback-lr-factor", type=float, default=0.5,
        help="multiply the learning rate by this after each champion rollback")
    train_parser.add_argument(
        "--minimum-learning-rate", type=float, default=1.25e-4,
        help="lowest automatic learning rate used after champion rollbacks")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--resume-learning-rate", type=float,
        help="reset Adam at this rate while retaining resumed weights and replay")
    train_parser.add_argument(
        "--start-stage", type=int,
        help="one-based curriculum stage override for a transferred/resumed champion")
    train_parser.add_argument(
        "--initialize-from", type=Path,
        help="transfer perception/dynamics/policy weights but reset task-specific training")
    train_parser.add_argument("--asteroid-reward", type=float, default=0.1)
    train_parser.add_argument(
        "--shot-penalty", type=float, default=0.0,
        help="reward subtracted per projectile fired, encouraging trigger discipline")
    train_parser.add_argument(
        "--history-frames", type=int, default=0,
        help="past positions kept per asteroid so curvature is observable")
    train_parser.add_argument(
        "--history-long-frames", type=int, default=0,
        help="additional older positions, sampled sparsely, to span a full oscillation")
    train_parser.add_argument("--history-long-stride", type=int, default=8)
    train_parser.add_argument(
        "--eval-every", type=int, default=0,
        help="evaluate on a fixed held-out seed set every N episodes (0 disables)")
    train_parser.add_argument("--eval-episodes", type=int, default=20)
    train_parser.add_argument("--eval-seed", type=int, default=10_000)
    train_parser.add_argument(
        "--log-every", type=int, default=10,
        help="print averaged stats every N episodes instead of one line per episode")
    train_parser.add_argument(
        "--parallel-envs", type=int, default=1,
        help="self-play environments stepped together in one batched search")
    ppo_parser = sub.add_parser(
        "train-ppo", help="train feed-forward or recurrent PPO on the mastery curriculum")
    ppo_parser.add_argument("--curriculum", type=Path,
                            default=Path("configs/rl-curriculum.toml"))
    ppo_parser.add_argument("--output", type=Path, default=Path("models/ppo"))
    ppo_parser.add_argument("--episodes", type=int, default=10_000)
    ppo_parser.add_argument("--seed", type=int, default=0)
    ppo_parser.add_argument("--recurrent", action="store_true")
    ppo_parser.add_argument("--parallel-envs", type=int, default=8)
    ppo_parser.add_argument(
        "--learning-rate", type=float,
        help="override the Adam rate, keeping every other resumed setting")
    ppo_parser.add_argument("--gamma", type=float)
    ppo_parser.add_argument(
        "--ent-coef", type=float,
        help="override the entropy bonus, keeping every other resumed setting")
    ppo_parser.add_argument(
        "--vf-coef", type=float,
        help="override the value-loss coefficient, keeping every other resumed setting")
    ppo_parser.add_argument(
        "--entropy-floor", type=float,
        help="hold policy entropy near NATS by adapting the entropy bonus between "
             "updates; 0 clears a floor restored from a checkpoint")
    ppo_parser.add_argument("--target-kl", type=float)
    ppo_parser.add_argument("--n-epochs", type=int)
    ppo_parser.add_argument(
        "--encoder", choices=("mlp", "set"), default="mlp",
        help="'set' pools the asteroid/projectile slots permutation-invariantly")
    ppo_parser.add_argument(
        "--stop-when-mastered", action="store_true",
        help="stop as soon as the final curriculum stage passes its gate")
    ppo_parser.add_argument("--history-frames", type=int, default=8)
    ppo_parser.add_argument("--history-long-frames", type=int, default=8)
    ppo_parser.add_argument("--history-long-stride", type=int, default=8)
    ppo_parser.add_argument("--eval-every", type=int, default=250)
    ppo_parser.add_argument("--eval-seed", type=int, default=10_000)
    ppo_parser.add_argument("--keep-checkpoints", type=int, default=3)
    ppo_parser.add_argument("--champion-patience", type=int, default=4)
    ppo_parser.add_argument("--resume", type=Path)
    ppo_parser.add_argument(
        "--initialize-from", type=Path,
        help="transfer PPO policy weights into a new curriculum and reset optimizer/progress")
    ppo_parser.add_argument(
        "--start-stage", type=int,
        help="one-based starting stage for a policy-initialized curriculum")
    ppo_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    mappo_parser = sub.add_parser(
        "train-mappo", help="train one shared policy across a 1-8 ship team")
    mappo_parser.add_argument("--output", type=Path, default=Path("models/mappo-team"))
    mappo_parser.add_argument("--episodes", type=int, default=1000)
    mappo_parser.add_argument("--max-ships", type=int, default=8)
    mappo_parser.add_argument("--protect", action="store_true")
    mappo_parser.add_argument("--initialize-from", type=Path)
    mappo_parser.add_argument("--resume", type=Path)
    mappo_parser.add_argument("--seed", type=int, default=0)
    mappo_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    team_train = sub.add_parser(
        "train-team", help="train centralized PPO controlling both ships jointly")
    team_train.add_argument("--output", type=Path, default=Path("models/team-ppo"))
    team_train.add_argument("--steps", type=int, default=1_000_000)
    team_train.add_argument("--parallel-envs", type=int, default=8)
    team_train.add_argument("--eval-every", type=int, default=250_000)
    team_train.add_argument("--eval-episodes", type=int, default=128)
    team_train.add_argument("--keep-checkpoints", type=int, default=3)
    team_train.add_argument("--resume", type=Path)
    team_train.add_argument("--seed", type=int, default=0)
    team_train.add_argument("--stop-when-mastered", action="store_true")
    team_train.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    team_eval = sub.add_parser("evaluate-team", help="score a shared MAPPO checkpoint")
    team_eval.add_argument("--checkpoint", type=Path, required=True)
    team_eval.add_argument("--episodes", type=int, default=64)
    team_eval.add_argument("--ships", type=int, default=8)
    team_eval.add_argument("--level", type=int, default=12)
    team_eval.add_argument("--protect", action="store_true")
    team_eval.add_argument("--seed", type=int, default=20_000)
    team_eval.add_argument("--stage", type=int,
                           help="one-based centralized team curriculum stage")
    team_play = sub.add_parser("play-team", help="watch 1-8 copies of a shared actor")
    team_play.add_argument("--checkpoint", type=Path, required=True)
    team_play.add_argument("--ships", type=int, default=8)
    team_play.add_argument("--level", type=int, default=12)
    team_play.add_argument("--protect", action="store_true")
    team_play.add_argument("--seed", type=int, default=7)
    team_play.add_argument("--stage", type=int,
                           help="one-based centralized team curriculum stage")
    graph_parser = sub.add_parser("graph", help="graph held-out progress for a model run")
    graph_parser.add_argument("--run", type=Path, required=True)
    graph_parser.add_argument(
        "--view", choices=("completion", "survival", "both"), default="both",
        help="rate lines to show in the terminal")
    graph_parser.add_argument(
        "--height", type=int, default=20,
        help="chart rows (8-40; default 20)")
    preview_parser = sub.add_parser(
        "preview", help="watch the best evaluated checkpoint from a curriculum run")
    preview_parser.add_argument("target", type=Path, help="run directory or checkpoint")
    preview_parser.add_argument("--seed", type=int, default=10_000)
    preview_parser.add_argument("--curriculum", type=Path)
    preview_parser.add_argument("--stage", type=int, help="one-based curriculum stage")
    evaluate_parser = sub.add_parser("evaluate", help="evaluate a saved RL checkpoint")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--config", type=Path, default=Path("configs/rl.toml"))
    evaluate_parser.add_argument("--output", type=Path, default=Path("metrics/evaluation.json"))
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=10_000)
    evaluate_parser.add_argument("--stage", type=int, help="one-based PPO curriculum stage")
    evaluate_parser.add_argument("--max-decisions", type=int, default=900)
    evaluate_parser.add_argument("--asteroid-reward", type=float, default=0.1)
    evaluate_parser.add_argument(
        "--shot-penalty", type=float, default=0.0,
        help="reward subtracted per projectile fired, encouraging trigger discipline")
    evaluate_parser.add_argument(
        "--history-frames", type=int, default=0,
        help="past positions kept per asteroid so curvature is observable")
    evaluate_parser.add_argument(
        "--history-long-frames", type=int, default=0,
        help="additional older positions, sampled sparsely, to span a full oscillation")
    evaluate_parser.add_argument("--history-long-stride", type=int, default=8)
    baseline_parser = sub.add_parser(
        "evaluate-baseline", help="evaluate the non-learning closest-asteroid controller")
    baseline_parser.add_argument("--config", type=Path, default=Path("configs/rl.toml"))
    baseline_parser.add_argument("--output", type=Path, default=Path("metrics/closest-baseline.json"))
    baseline_parser.add_argument("--episodes", type=int, default=100)
    baseline_parser.add_argument("--seed", type=int, default=10_000)
    baseline_parser.add_argument("--max-decisions", type=int, default=900)
    baseline_parser.add_argument("--asteroid-reward", type=float, default=0.1)
    baseline_parser.add_argument(
        "--shot-penalty", type=float, default=0.0,
        help="reward subtracted per projectile fired, encouraging trigger discipline")
    baseline_parser.add_argument(
        "--history-frames", type=int, default=0,
        help="past positions kept per asteroid so curvature is observable")
    baseline_parser.add_argument(
        "--history-long-frames", type=int, default=0,
        help="additional older positions, sampled sparsely, to span a full oscillation")
    baseline_parser.add_argument("--history-long-stride", type=int, default=8)
    compare_parser = sub.add_parser(
        "compare", help="play the same seeds as the greedy baseline and a MuZero checkpoint")
    compare_parser.add_argument("--config", type=Path, default=Path("configs/rl.toml"))
    compare_parser.add_argument("--checkpoint", type=Path)
    compare_parser.add_argument("--output", type=Path, default=Path("metrics/comparison.json"))
    compare_parser.add_argument("--episodes", type=int, default=5)
    compare_parser.add_argument("--seed", type=int, default=50_000)
    compare_parser.add_argument("--max-decisions", type=int, default=900)
    compare_parser.add_argument("--asteroid-reward", type=float, default=0.1)
    compare_parser.add_argument("--shot-penalty", type=float, default=0.0)
    compare_parser.add_argument(
        "--history-frames", type=int, default=0,
        help="past positions kept per asteroid so curvature is observable")
    compare_parser.add_argument(
        "--history-long-frames", type=int, default=0,
        help="additional older positions, sampled sparsely, to span a full oscillation")
    compare_parser.add_argument("--history-long-stride", type=int, default=8)
    compare_parser.add_argument(
        "--no-human", action="store_true", help="score the agents without playing yourself")
    compare_parser.add_argument(
        "--mode", choices=sorted(MODES), help="score inside a named mode instead of --config")
    compare_parser.add_argument("--round", type=int, help="round number for round-based modes")
    compare_parser.add_argument(
        "--model", dest="models", action="append", default=[], type=Path,
        help="a checkpoint to include; repeat to compare several models")
    compare_parser.add_argument(
        "--no-greedy", action="store_true", help="leave the greedy baseline out")
    compare_parser.add_argument(
        "--no-pilot", action="store_true", help="leave the scripted pilot baseline out")
    args = parser.parse_args(argv)
    if args.command == "graph":
        from .rl.plotting import format_progress
        width = shutil.get_terminal_size(fallback=(100, 24)).columns
        print(format_progress(args.run, view=args.view, width=width, height=args.height,
                              color=sys.stdout.isatty() and "NO_COLOR" not in os.environ))
        return 0
    if args.command == "arena":
        controllers = ["human"] + [name for name in args.opponents.split(",") if name]
        try:
            config, label = build_mode(args.mode, args.round, controllers=controllers)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"playing {label} | seed {args.seed}")
        return play(config, args.seed, args.checkpoint, all_runs=args.any_run)
    if args.command == "patterns":
        from .modes import pattern_showcase

        config, label = pattern_showcase(args.pattern)
        print(f"watching {label} | seed {args.seed}")
        print("the ship cannot be destroyed here; fly around and watch the paths")
        return play(config, args.seed, trails=True)
    if args.command == "best-checkpoint":
        config, _ = build_mode(args.mode, args.round)
        algorithms = {name for name in args.algorithms.split(",") if name}
        try:
            print(_latest_checkpoint(config, algorithms=algorithms or None,
                                     all_runs=args.any_run))
        except SystemExit:
            return 1
        return 0
    if args.command == "play-round":
        try:
            config, name = _curriculum_round_config(args.curriculum, args.round)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"playing round {args.round}: {name} | seed {args.seed}")
        return play(config, args.seed)
    if args.command == "preview":
        from .rl.preview import preview_checkpoint
        preview_checkpoint(
            args.target, seed=args.seed, curriculum_path=args.curriculum,
            stage_index=(args.stage - 1) if args.stage is not None else None)
        return 0
    if args.command == "compare":
        from .rl.comparison import compare, format_table
        target = args.config
        env_settings: dict = {}
        if args.mode:
            from .modes import round_env_settings

            target, label = build_mode(args.mode, args.round, controllers=["closest"],
                                       scoring=True)
            env_settings = round_env_settings(args.mode, args.round)
            print(f"scoring {label}")
        report = compare(
            target, args.output, checkpoint=args.checkpoint, checkpoints=args.models,
            env_settings=env_settings,
            include_greedy=not args.no_greedy, include_pilot=not args.no_pilot,
            episodes=args.episodes,
            seed=args.seed, max_decisions=args.max_decisions,
            asteroid_reward=args.asteroid_reward, shot_penalty=args.shot_penalty,
            history_frames=args.history_frames,
            history_long_frames=args.history_long_frames,
            history_long_stride=args.history_long_stride, include_human=not args.no_human,
        )
        print()
        print(format_table(report["summary"]))
        print(f"\nsaved comparison: {args.output}")
        return 0
    if args.command == "train-ppo":
        from .rl.ppo import PPOTrainSettings, train_ppo_curriculum
        ppo_settings = None
        if args.gamma is not None or args.target_kl is not None or args.n_epochs is not None:
            ppo_settings = PPOTrainSettings()
            if args.gamma is not None:
                ppo_settings.gamma = args.gamma
            if args.target_kl is not None:
                ppo_settings.target_kl = args.target_kl
            if args.n_epochs is not None:
                ppo_settings.n_epochs = args.n_epochs
        checkpoint = train_ppo_curriculum(
            args.curriculum, args.output, episodes=args.episodes,
            recurrent=args.recurrent, seed=args.seed,
            parallel_envs=args.parallel_envs, history_frames=args.history_frames,
            history_long_frames=args.history_long_frames,
            history_long_stride=args.history_long_stride,
            eval_every=args.eval_every, eval_seed=args.eval_seed,
            keep_checkpoints=args.keep_checkpoints,
            champion_patience=args.champion_patience,
            resume=args.resume, initialize_from=args.initialize_from,
            start_stage=(args.start_stage - 1) if args.start_stage is not None else 0,
            device=args.device, learning_rate=args.learning_rate,
            ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            entropy_floor=args.entropy_floor,
            settings=ppo_settings, stop_when_mastered=args.stop_when_mastered,
            encoder=args.encoder)
        print(f"saved checkpoint: {checkpoint}")
        return 0
    if args.command == "train-mappo":
        from .rl.multiagent import train_shared_mappo
        checkpoint = train_shared_mappo(
            args.output, episodes=args.episodes, max_ships=args.max_ships,
            protect=args.protect, seed=args.seed, device=args.device,
            initialize_from=args.initialize_from, resume=args.resume)
        print(f"saved checkpoint: {checkpoint}")
        return 0
    if args.command == "train-team":
        from .rl.team_ppo import train_centralized_team
        checkpoint = train_centralized_team(
            args.output, steps=args.steps, seed=args.seed,
            parallel_envs=args.parallel_envs, eval_every=args.eval_every,
            eval_episodes=args.eval_episodes, device=args.device,
            resume=args.resume, stop_when_mastered=args.stop_when_mastered,
            keep_checkpoints=args.keep_checkpoints)
        print(f"saved checkpoint: {checkpoint}")
        return 0
    if args.command == "evaluate-team":
        metadata = json.loads(
            (args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") == "centralized_team_ppo":
            from .rl.team_ppo import evaluate_centralized_team
            stage = ((args.stage - 1) if args.stage is not None else
                     int(metadata.get("stage", 0)))
            report = evaluate_centralized_team(
                args.checkpoint, stage=stage, episodes=args.episodes, seed=args.seed)
        else:
            from .rl.multiagent import evaluate_shared_mappo
            report = evaluate_shared_mappo(
                args.checkpoint, episodes=args.episodes, ships=args.ships,
                level=args.level, protect=args.protect, seed=args.seed)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "play-team":
        metadata = json.loads(
            (args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") == "centralized_team_ppo":
            from .rl.team_ppo import play_centralized_team
            return play_centralized_team(
                args.checkpoint, stage=(args.stage - 1) if args.stage else None,
                seed=args.seed)
        from .rl.multiagent import play_shared_mappo
        return play_shared_mappo(
            args.checkpoint, ships=args.ships, level=args.level,
            protect=args.protect, seed=args.seed)
    if args.command == "train":
        from .rl.training import train, train_curriculum
        if args.curriculum:
            checkpoint = train_curriculum(
                args.curriculum, args.output, episodes=args.episodes, seed=args.seed,
                simulations=args.simulations, checkpoint_every=args.checkpoint_every,
                updates_per_episode=args.updates_per_episode,
                eval_every=args.eval_every or 250, parallel_envs=args.parallel_envs,
                history_frames=args.history_frames,
                history_long_frames=args.history_long_frames,
                history_long_stride=args.history_long_stride, log_every=args.log_every,
                resume=args.resume, eval_seed=args.eval_seed,
                initialize_from=args.initialize_from,
                keep_checkpoints=args.keep_checkpoints,
                stop_when_mastered=args.stop_when_mastered,
                champion_patience=args.champion_patience,
                rollback_lr_factor=args.rollback_lr_factor,
                minimum_learning_rate=args.minimum_learning_rate,
                learning_rate=args.learning_rate,
                resume_learning_rate=args.resume_learning_rate,
                start_stage=args.start_stage)
            print(f"saved checkpoint: {checkpoint}")
            return 0
        checkpoint = train(
            args.config, args.output, episodes=args.episodes, seed=args.seed,
            simulations=args.simulations, max_decisions=args.max_decisions,
            checkpoint_every=args.checkpoint_every, resume=args.resume,
            updates_per_episode=args.updates_per_episode,
            asteroid_reward=args.asteroid_reward, eval_every=args.eval_every,
            eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
            log_every=args.log_every, parallel_envs=args.parallel_envs,
            shot_penalty=args.shot_penalty, history_frames=args.history_frames,
            history_long_frames=args.history_long_frames,
            history_long_stride=args.history_long_stride,
        )
        print(f"saved checkpoint: {checkpoint}")
        return 0
    if args.command == "evaluate":
        metadata = json.loads(
            (args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") in {"ppo", "recurrent_ppo"}:
            from .rl.ppo import evaluate_ppo_checkpoint
            report = evaluate_ppo_checkpoint(
                args.checkpoint, args.output, episodes=args.episodes, seed=args.seed,
                stage_index=(args.stage - 1) if args.stage is not None else None)
        else:
            from .rl.training import evaluate
            report = evaluate(
                args.checkpoint, args.config, args.output, episodes=args.episodes,
                seed=args.seed, max_decisions=args.max_decisions,
                asteroid_reward=args.asteroid_reward, shot_penalty=args.shot_penalty,
                history_frames=args.history_frames,
                history_long_frames=args.history_long_frames,
                history_long_stride=args.history_long_stride,
            )
        print(json.dumps(report["aggregate"], indent=2))
        print(f"saved evaluation: {args.output}")
        return 0
    if args.command == "evaluate-baseline":
        from .rl.training import evaluate_baseline
        report = evaluate_baseline(
            args.config, args.output, episodes=args.episodes, seed=args.seed,
            max_decisions=args.max_decisions, asteroid_reward=args.asteroid_reward,
            shot_penalty=args.shot_penalty, history_frames=args.history_frames,
            history_long_frames=args.history_long_frames,
            history_long_stride=args.history_long_stride,
        )
        print(json.dumps(report["aggregate"], indent=2))
        print(f"saved evaluation: {args.output}")
        return 0
    config = load_config(args.config)
    return play(config, args.seed, args.checkpoint) if args.command == "play" else simulate(config, args.seed, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
