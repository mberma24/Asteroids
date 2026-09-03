"""The playable modes, and how to build a configuration for any of them.

Every way of playing this game is one of four modes. Keeping them in one table means `play`,
`showdown`, `watch`, and `preview` all accept the same vocabulary instead of each growing its
own flags and its own hand-written showdown config file.

    arcade          clear-the-wave arcade play
    endless         one run whose difficulty ramps forever
    round N         nonlinear curriculum round, 1-48
    survival N      survival ladder round (straight lines, then curves)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import GameConfig, ShipSpec, load_config


@dataclass(frozen=True, slots=True)
class Mode:
    name: str
    summary: str
    config: str | None = None
    curriculum: str | None = None
    """Set when the mode is one round of a curriculum rather than a standalone config."""

    @property
    def is_round(self) -> bool:
        return self.curriculum is not None


MODES: dict[str, Mode] = {
    "arcade": Mode("arcade", "clear-the-wave arcade play", config="configs/solo.toml"),
    "endless": Mode("endless", "one run, difficulty ramps forever",
                    config="configs/endless.toml"),
    "round": Mode("round", "nonlinear curriculum round 1-48",
                  curriculum="configs/rl-nonlinear.toml"),
    "survival": Mode("survival", "survival ladder round",
                     curriculum="configs/rl-endless.toml"),
    "survival-v2": Mode("survival-v2", "nonlinear-first survival v2 round",
                        curriculum="configs/rl-survival-v2.toml"),
    # From round 28 the pattern, not the speed, is what grows: amplitude roughly doubles
    # against v2 and the period lengthens instead of shortening, so a rock traces one big arc
    # rather than vibrating across the arena. Rounds 1-27 are v2's, unchanged.
    "survival-v3": Mode("survival-v3", "survival v3, expressive patterns from round 28",
                        curriculum="configs/rl-survival-v3.toml"),
    # Rounds 97-100 mix slow rocks back into the late-game distribution, so this mode is
    # how the overfitting check is played, watched, or scored.
    "varied": Mode("varied", "survival v2 plus the varied overfitting rounds",
                   curriculum="configs/rl-survival-v2-varied.toml"),
}

# Controller name -> the ship id it plays under. Ids are what the scoreboard shows.
# `_controllers` in the CLI can already build every one of these; leaving a name out here
# only blocks it from a lineup. `heuristic` is the sole scripted controller that manoeuvres,
# which makes it the one to watch when the question is what movement looks like -- it is not
# a yardstick, and its own docstring says so.
ROSTER_IDS = {"human": "you", "closest": "greedy", "pilot": "pilot",
              "heuristic": "heuristic", "random": "random",
              "ppo": "model", "muzero": "model"}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def roster(controllers: list[str]) -> list[ShipSpec]:
    """Ship specs for a lineup like ``["human", "closest", "ppo"]``."""
    unknown = [name for name in controllers if name not in ROSTER_IDS]
    if unknown:
        raise ValueError(f"unknown controllers: {unknown}; "
                         f"choose from {sorted(ROSTER_IDS)}")
    return [ShipSpec(ROSTER_IDS[name], name,
                     "keyboard_1" if name == "human" else "keyboard_1")
            for name in controllers]


def round_count(mode_name: str) -> int:
    """How many rounds a round-based mode has; zero for standalone modes."""
    mode = resolve(mode_name)
    if not mode.is_round:
        return 0
    from .rl.curriculum import load_curriculum
    return len(load_curriculum(_project_path(mode.curriculum)).stages)


def pattern_showcase(pattern: str | None = None) -> tuple[GameConfig, str]:
    """A slow, safe arena for watching trajectory shapes rather than playing against them.

    The ship is invulnerable and asteroids are small so they never split: the point is to
    watch paths, and a field of fragments obscures them. Amplitude is large and speed low so
    a full period of each shape is visible while it crosses the arena.
    """
    from .config import KNOWN_PATTERNS

    if pattern is not None and pattern not in KNOWN_PATTERNS:
        raise SystemExit(
            f"unknown pattern: {pattern}; choose from {', '.join(KNOWN_PATTERNS)}")
    config = load_config(_project_path("configs/solo.toml"))
    config.ship.invulnerable = True
    asteroid = config.asteroid
    asteroid.spawn_mode = "interval"
    asteroid.spawn_interval = 1.6
    asteroid.active_cap = 12
    asteroid.spawn_size = 1
    asteroid.heading_mode = "random"
    asteroid.min_speed, asteroid.max_speed = 26.0, 34.0
    asteroid.small_speed_multiplier = 1.0
    asteroid.amplitude_min, asteroid.amplitude_max = 90.0, 130.0
    asteroid.wavelength_min, asteroid.wavelength_max = 3.2, 4.2
    if pattern is None:
        asteroid.motion_mode = "pool"
        # The showcase exists to look at shapes, so it shows the experimental ones too.
        asteroid.pattern_pool = list(KNOWN_PATTERNS)
        label = f"all {len(KNOWN_PATTERNS)} patterns"
    else:
        asteroid.motion_mode = "specific"
        asteroid.specific_pattern = pattern
        label = f"pattern: {pattern}"
    asteroid.linear_probability = 0.0
    config.ships = roster(["human"])
    config.validate()
    return config, label


def resolve(mode_name: str) -> Mode:
    try:
        return MODES[mode_name]
    except KeyError:
        raise SystemExit(
            f"unknown mode: {mode_name}; choose from {', '.join(MODES)}") from None


def round_env_settings(mode_name: str, round_number: int | None) -> dict:
    """Environment settings a curriculum round is scored under, for `compare` and friends.

    `build` returns only a `GameConfig`, which cannot express how the round *ends*. Scoring a
    survival round under the environment's `completion="waves"` default makes
    `completed_stage` structurally false, because it is read from `survived_to_limit`, which
    in turn needs `decisions >= max_decisions`. Both values live on the stage, so both have
    to travel with the config or comparisons silently report a zero clear rate.
    """
    mode = MODES.get(mode_name) if mode_name else None
    if mode is None or not mode.is_round or round_number is None:
        return {}
    from .rl.curriculum import load_curriculum

    spec = load_curriculum(_project_path(mode.curriculum))
    if not 1 <= round_number <= len(spec.stages):
        return {}
    stage = spec.stages[round_number - 1]
    return {"completion": stage.completion, "max_decisions": stage.max_decisions,
            "no_hit_seconds": stage.no_hit_seconds}


def build(mode_name: str, round_number: int | None = None, *,
          controllers: list[str] | None = None,
          scoring: bool = False) -> tuple[GameConfig, str]:
    """A configuration for one mode, and a label describing what it is.

    The same builder serves solo play and showdowns; only the lineup differs. That is why
    there are no longer separate `showdown-*.toml` files to keep in step with the curricula
    they were supposed to mirror.
    """
    mode = resolve(mode_name)
    controllers = list(controllers or ["human"])
    if mode.is_round:
        from .rl.curriculum import load_curriculum

        spec = load_curriculum(_project_path(mode.curriculum))
        if round_number is None:
            raise SystemExit(f"mode '{mode.name}' needs a round: 1-{len(spec.stages)}")
        if not 1 <= round_number <= len(spec.stages):
            raise SystemExit(
                f"round for '{mode.name}' must be between 1 and {len(spec.stages)}")
        stage = spec.stages[round_number - 1]
        config = stage.game_config(spec.base)
        if stage.survival and not scoring:
            # A survival round is cleared by lasting max_seconds, so stop a human there too
            # rather than letting it run past the point training scores.
            #
            # Not when scoring, though: the RL environment marks `survived_to_limit` only on
            # its own decision-limit truncation, and an objective step limit terminates the
            # game one moment sooner, so the flag -- and with it `completed_stage` -- would
            # never be set. Training leaves `max_steps` at None for exactly this reason.
            config.objective.max_steps = stage.max_decisions * 4
        label = f"{mode.name} {round_number}: {stage.name}"
    else:
        if round_number is not None:
            raise SystemExit(f"mode '{mode.name}' does not take a round number")
        config = load_config(_project_path(mode.config))
        label = f"{mode.name}: {mode.summary}"
    config.ships = roster(controllers)
    config.validate()
    return config, label
