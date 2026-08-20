from __future__ import annotations

from dataclasses import replace

import pytest

from asteroid_survival.actions import Action
from asteroid_survival.config import GameConfig, ShipSpec, load_config
from asteroid_survival.controllers import (ClosestAsteroidController, HeuristicController,
                                           KeyProfile, RandomController, human_action)
from asteroid_survival.simulation import Simulation
from asteroid_survival.state import AsteroidSnapshot


def test_presets_load():
    for name in ("default", "protection", "stationary_protection", "headless",
                 "solo", "endless", "rl-arcade"):
        cfg = load_config(f"configs/{name}.toml")
        assert cfg.ships


def test_every_mode_builds_a_showdown_lineup():
    """One builder replaces the hand-written showdown-*.toml files.

    Those files had to be kept in step with the curricula they mirrored by hand, and had
    already drifted: the nonlinear one duplicated round 48's numbers as literals.
    """
    from asteroid_survival.modes import MODES, build

    for name in MODES:
        number = 1 if MODES[name].is_round else None
        cfg, label = build(name, number, controllers=["human", "closest", "ppo"])
        assert [(ship.id, ship.controller) for ship in cfg.ships] == [
            ("you", "human"), ("greedy", "closest"), ("model", "ppo"),
        ]
        assert label
        # A shared observation layout is what lets one checkpoint play every mode.
        assert cfg.asteroid.active_cap == 26
        assert cfg.ship.mobile is True


def test_round_modes_read_difficulty_from_the_curriculum_not_a_copy():
    from asteroid_survival.modes import build
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-nonlinear.toml")
    stage = spec.stages[47]
    cfg, label = build("round", 48)

    assert "48" in label and stage.name in label
    assert (cfg.asteroid.min_speed, cfg.asteroid.max_speed) == (stage.min_speed,
                                                                stage.max_speed)
    assert (cfg.asteroid.amplitude_min, cfg.asteroid.amplitude_max) == (
        stage.amplitude_min, stage.amplitude_max)
    assert set(cfg.asteroid.pattern_pool) == set(stage.patterns)


def test_mode_errors_are_actionable():
    from asteroid_survival.modes import build

    with pytest.raises(SystemExit, match="unknown mode"):
        build("nonsense")
    with pytest.raises(SystemExit, match="between 1 and 96"):
        build("survival", 99)
    with pytest.raises(SystemExit, match="needs a round"):
        build("survival")
    with pytest.raises(SystemExit, match="does not take a round"):
        build("arcade", 3)
    with pytest.raises(ValueError, match="unknown controllers"):
        build("arcade", controllers=["human", "wizard"])


def test_survival_rounds_stop_a_human_where_training_scores_them():
    from asteroid_survival.modes import build
    from asteroid_survival.rl.curriculum import load_curriculum

    spec = load_curriculum("configs/rl-endless.toml")
    cfg, _ = build("survival", 12)
    assert cfg.objective.max_steps == spec.stages[11].max_decisions * 4

    arcade, _ = build("arcade")
    assert arcade.objective.max_steps is None


def test_personal_round_uses_exact_curriculum_difficulty_with_human_ship():
    from asteroid_survival.cli import _curriculum_round_config

    cfg, name = _curriculum_round_config("configs/rl-nonlinear.toml", 37)
    assert name == "nonlinear-round-37"
    assert [(ship.id, ship.controller) for ship in cfg.ships] == [("you", "human")]
    assert len(cfg.asteroid.wave_composition) == 7
    assert (cfg.asteroid.min_speed, cfg.asteroid.max_speed) == (38.0, 53.0)
    assert cfg.asteroid.motion_mode == "pool"


def test_default_play_lineup_is_baseline_human_and_muzero():
    cfg = load_config("configs/default.toml")
    assert [(ship.id, ship.controller) for ship in cfg.ships] == [
        ("greedy", "closest"), ("human", "human"), ("muzero", "muzero"),
    ]


def test_duplicate_ship_ids_rejected():
    cfg = GameConfig(ships=[ShipSpec("same"), ShipSpec("same")])
    with pytest.raises(ValueError):
        cfg.validate()


def test_random_controller_is_seeded():
    snapshot = Simulation(GameConfig(ships=[ShipSpec("one")])).reset(0)
    a, b = RandomController(18), RandomController(18)
    assert [a.action(snapshot, "one") for _ in range(20)] == [b.action(snapshot, "one") for _ in range(20)]


def test_heuristic_returns_valid_action():
    cfg = GameConfig(ships=[ShipSpec("one")])
    sim = Simulation(cfg)
    state = sim.reset(0)
    state = sim.step({}).snapshot
    assert isinstance(HeuristicController().action(state, "one"), Action)


def test_human_action_accepts_alternate_movement_keys():
    profile = KeyProfile(0, 1, 2, 3, alternate_left=4, alternate_right=5, alternate_thrust=6)
    keys = [False] * 7
    keys[5] = True
    keys[6] = True

    assert human_action(keys, profile) == Action.RIGHT_THRUST


def test_closest_controller_aims_without_thrust():
    cfg = GameConfig(ships=[ShipSpec("one")])
    sim = Simulation(cfg)
    state = sim.reset(0)
    ship = state.ships[0]
    state = replace(state, asteroids=(
        AsteroidSnapshot(1, ship.x + 100, ship.y, 0, 0, 3, 39, "linear"),
        AsteroidSnapshot(2, ship.x - 300, ship.y, 0, 0, 3, 39, "linear"),
    ))
    action = ClosestAsteroidController().action(state, "one")
    assert action == Action.RIGHT


def test_pattern_showcase_is_safe_and_shows_unsplit_paths():
    from asteroid_survival.config import PATTERN_NAMES
    from asteroid_survival.modes import pattern_showcase

    config, label = pattern_showcase()
    assert config.ship.invulnerable is True          # it is for watching, not playing
    assert config.asteroid.spawn_size == 1           # small rocks never split into clutter
    assert config.asteroid.motion_mode == "pool"
    assert set(config.asteroid.pattern_pool) == set(PATTERN_NAMES)
    assert str(len(PATTERN_NAMES)) in label

    single, label = pattern_showcase("corkscrew")
    assert single.asteroid.motion_mode == "specific"
    assert single.asteroid.specific_pattern == "corkscrew"
    assert "corkscrew" in label

    with pytest.raises(SystemExit, match="unknown pattern"):
        pattern_showcase("banana")


def test_model_selection_follows_the_newest_run_not_the_highest_score(tmp_path, monkeypatch):
    """An abandoned run that reached a later stage must not shadow the live one.

    Ranking by score across every run meant `showdown` and `preview` kept resolving to a
    stale model -- one trained on trajectory dynamics that no longer existed -- and had to
    be overridden by hand on every invocation.
    """
    import json
    import os
    import time

    import asteroid_survival.cli as cli

    models = tmp_path / "models"

    def write_run(name, episode, stage, when):
        checkpoint = models / name / f"checkpoint_{episode:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps({"algorithm": "ppo"}), encoding="utf-8")
        os.utime(checkpoint, (when, when))
        return checkpoint

    stale = write_run("abandoned", 20000, 39, time.time() - 10_000)
    live = write_run("live", 500, 5, time.time())

    # `models/` is found relative to the package, so point the package at the fixture.
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src" / "pkg" / "cli.py"))
    monkeypatch.setattr(cli.Path, "resolve", lambda self: self)
    monkeypatch.setattr(cli, "_checkpoint_fits", lambda path, config: True)
    # The abandoned run scores far higher: it reached curriculum stage 39 against stage 5.
    monkeypatch.setattr(cli, "_evaluation_scores",
                        lambda: {stale: 39_000_000.0, live: 5_000_000.0})

    assert cli._latest_checkpoint(None, algorithms={"ppo"}).parent.name == "live"
    assert cli._latest_checkpoint(
        None, algorithms={"ppo"}, all_runs=True).parent.name == "abandoned"


def test_every_dispatched_command_actually_exists():
    """A refactor once deleted two command bodies while leaving their dispatch entries.

    `./run.sh train-ppo-endless` then failed with "command not found" at the moment it was
    needed. Nothing else checks the shell script's internal consistency.
    """
    import re
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
    defined = set(re.findall(r"^(cmd_[a-z_]+)\(\) \{", script, re.MULTILINE))
    referenced = set(re.findall(r"\b(cmd_[a-z_]+)\b", script))
    assert referenced <= defined, f"dispatched but never defined: {sorted(referenced - defined)}"
