from __future__ import annotations

import copy
import hashlib
import json
import random
import tomllib
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path

from ..config import GameConfig, ShipSpec, load_config
from .environment import RewardConfig


# The exact fields the original task hash covered, frozen as a snapshot so checkpoints
# trained before `task_hash` changed still verify.
#
# This is an allowlist, not an exclusion list, and that distinction is the whole point: a
# field added to any of these dataclasses in the future is simply absent here, so it cannot
# change the legacy digest. An earlier attempt used an exclusion list and broke a second
# time the moment fields were added to `CurriculumStage` and `RewardConfig` instead of
# `AsteroidConfig`. Do not add new field names to this table.
_LEGACY_FIELDS: dict[str, tuple[str, ...]] = {
    "CurriculumStage": (
        "name", "composition", "target_waves", "min_speed", "max_speed",
        "linear_probability", "patterns", "amplitude_min", "amplitude_max",
        "wavelength_min", "wavelength_max", "movement_enabled", "ship_invulnerable",
        "max_seconds", "no_hit_seconds", "promotion_accuracy", "miss_penalty"),
    "arena": ("width", "height", "fps"),
    "ship": ("mobile", "invulnerable", "friendly_collisions", "radius", "acceleration",
             "drag", "max_speed", "turn_speed", "fire_cooldown"),
    "asteroid": (
        "spawn_mode", "spawn_interval", "active_cap", "wave_threshold",
        "wave_threshold_start", "wave_size", "wave_composition", "wave_growth",
        "wave_size_max", "wave_spawn_interval", "wave_delay", "initial_wave_delay",
        "spawn_size", "min_speed", "max_speed", "motion_mode", "linear_probability",
        "specific_pattern", "pattern_pool", "amplitude_min", "amplitude_max",
        "wavelength_min", "wavelength_max", "heading_mode", "medium_speed_multiplier",
        "small_speed_multiplier", "spawn_interval_start", "active_cap_start",
        "min_speed_start", "max_speed_start", "amplitude_max_start", "spawn_spread",
        "spawn_spread_start", "ramp_seconds"),
    "objective": ("protect", "object_health", "object_radius", "max_steps", "max_waves"),
    "projectile": ("speed", "lifetime", "radius"),
    "ships": ("id", "controller", "input_profile"),
    "RewardConfig": ("large", "medium", "small", "wave_clear", "fast_clear", "accuracy",
                     "miss_penalty", "timeout_penalty", "death_penalty",
                     "active_time_penalty", "par_time_per_wave", "aim_progress"),
}


def _specified(value):
    """Strip dataclass fields still at their default, recursively.

    A curriculum's identity is what it actually specifies. Hashing every field means that
    adding an unrelated option to a config dataclass invalidates every checkpoint ever
    trained, even though no curriculum uses that option.
    """
    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for field in fields(value):
            current = getattr(value, field.name)
            if field.default is not MISSING:
                default = field.default
            elif field.default_factory is not MISSING:
                default = field.default_factory()
            else:
                result[field.name] = _specified(current)  # required: always significant
                continue
            if current != default:
                result[field.name] = _specified(current)
        return result
    if isinstance(value, (list, tuple)):
        return [_specified(item) for item in value]
    return value


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def task_hash(spec, stage_configs: list[GameConfig] | None = None) -> str:
    """Identity of the task a checkpoint was trained on: stages, configs, and reward."""
    configs = stage_configs or [stage.game_config(spec.base) for stage in spec.stages]
    return _digest({"stage_specs": [_specified(stage) for stage in spec.stages],
                    "stages": [_specified(config) for config in configs],
                    "reward": _specified(spec.reward)})


def _only(payload: dict, key: str) -> dict:
    """Keep just the fields the legacy schema recorded for this dataclass."""
    allowed = _LEGACY_FIELDS[key]
    return {name: payload[name] for name in allowed if name in payload}


def _legacy_task_hash(spec, stage_configs: list[GameConfig] | None = None) -> str:
    """The pre-`_specified` hash, for checkpoints trained before this module changed."""
    configs = stage_configs or [stage.game_config(spec.base) for stage in spec.stages]

    def game_payload(config: GameConfig) -> dict:
        payload = asdict(config)
        result = {section: _only(payload[section], section)
                  for section in ("arena", "ship", "asteroid", "objective", "projectile")}
        result["ships"] = [_only(ship, "ships") for ship in payload["ships"]]
        return result

    return _digest(
        {"stage_specs": [_only(asdict(stage), "CurriculumStage") for stage in spec.stages],
         "stages": [game_payload(config) for config in configs],
         "reward": _only(asdict(spec.reward), "RewardConfig")})


def reward_matches(stored: dict | None, reward: RewardConfig) -> bool:
    """Whether a checkpoint's recorded reward still describes this one.

    Compared field by field rather than as whole dicts. A reward term added after the
    checkpoint was written is absent from its record, and comparing the dicts directly then
    reports a mismatch even when every value the checkpoint knew about is identical -- which
    is what blocked resuming the nonlinear run once survival rewards were added. A new term
    is only inert if it is at its default, so that is exactly what is required of it.
    """
    if not stored:
        return True
    current = asdict(reward)
    if any(stored[name] != current.get(name) for name in stored):
        return False
    defaults = asdict(RewardConfig())
    return all(current[name] == defaults[name] for name in current if name not in stored)


def task_hash_matches(stored: str | None, spec,
                      stage_configs: list[GameConfig] | None = None) -> bool:
    """Whether a checkpoint's recorded task hash still describes this curriculum."""
    if not stored:
        return True  # older checkpoints recorded no hash at all
    return stored in {task_hash(spec, stage_configs),
                      _legacy_task_hash(spec, stage_configs)}


_SIZE_NAMES = {"small": 1, "medium": 2, "large": 3, "mixed": None, "random": None}


def _asteroid_size(value) -> int | list[int] | None:
    """A size name, a 1-3 integer, `mixed`/`None`, or a list forming a weighted mixture.

    A list is drawn from uniformly, so repeating an entry weights it: `["small", "small",
    "small", "medium"]` is one medium in four. That is what lets a tier introduce splitting
    gradually instead of switching every rock over at once.
    """
    if value is None:
        return None
    if isinstance(value, list):
        sizes = [_asteroid_size(item) for item in value]
        if not sizes or any(size is None for size in sizes):
            raise ValueError("an asteroid_size list must contain only sizes")
        return sizes
    if isinstance(value, str):
        if value not in _SIZE_NAMES:
            raise ValueError(
                f"asteroid_size must be one of {sorted(_SIZE_NAMES)}, 1-3, or omitted")
        return _SIZE_NAMES[value]
    size = int(value)
    if size not in {1, 2, 3}:
        raise ValueError("asteroid_size must be 1, 2, or 3")
    return size


def retention_holds(prior: list[dict], *, retention_completion: float,
                    retention_floor: float) -> bool:
    """Whether earlier stages are still held, judged on the pooled sample.

    Requiring every prior stage to clear the floor independently is a conjunction of as many
    noisy tests as there are stages, and it grows stricter with every promotion. Each stage
    is scored on only a handful of episodes, so a stage genuinely retained at 85% reads below
    a 75% floor about a tenth of the time; across 37 prior stages the conjunction then passes
    roughly 1.6% of the time on sampling noise alone. That is what stalled the nonlinear run
    at round 38: it was clearing the current round's gate and being blocked by 8-episode
    reads on rounds it had long since mastered.

    Pooling those same episodes asks the question the gate actually cares about -- has the
    policy forgotten what it learned -- using the whole sample rather than the worst draw. A
    per-stage floor is still enforced, but low enough to catch genuine collapse instead of
    an unlucky sample. Stages with no episodes this round are skipped.
    """
    scored = [(float(stage.get("completion_rate", 0.0)),
               max(0, int(stage.get("episodes", 1) or 0))) for stage in prior]
    scored = [(rate, weight) for rate, weight in scored if weight > 0]
    if not scored:
        return True
    total = sum(weight for _, weight in scored)
    pooled = sum(rate * weight for rate, weight in scored) / total
    return pooled >= retention_completion and min(rate for rate, _ in scored) >= retention_floor


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    composition: tuple[int, ...]
    target_waves: int
    min_speed: float
    max_speed: float
    linear_probability: float
    patterns: tuple[str, ...]
    amplitude_min: float
    amplitude_max: float
    wavelength_min: float
    wavelength_max: float
    movement_enabled: bool
    ship_invulnerable: bool
    max_seconds: float
    no_hit_seconds: float
    promotion_accuracy: float | None
    miss_penalty: float | None
    survival: bool = False
    """Endless round: clearing it means lasting ``max_seconds`` instead of clearing waves."""
    spawn_interval: float = 1.25
    """Survival rounds only. Asteroids never leave the arena, so this is the difficulty."""
    spawn_spread: float = 0.0
    """Survival rounds only. Degrees of scatter away from the centre; narrower is harder."""
    initial_asteroids: int = 0
    """Survival rounds only. Asteroids already in flight when the round begins."""
    spawn_safe_radius: float = 0.0
    """Survival rounds only. Fixed clearance kept between those asteroids and the ship."""
    spawn_safe_seconds: float = 0.0
    """Survival rounds only. Clearance worth this long at the asteroid's own speed."""
    asteroid_size: int | list[int] | None = 3
    """Survival rounds only: the size every asteroid spawns at, or ``None`` to mix them.

    Named `asteroid_size` rather than `spawn_size` on purpose -- `spawn_size` already means
    something else in a hand-written stage table, where it is the legacy per-wave fallback
    that desugars into a composition. Defaults to 3 so that every existing stage strips it
    from the task hash and no trained checkpoint is invalidated by the field existing.
    """
    variety_probability: float = 0.0
    """Chance a rock is drawn slow rather than at this round's difficulty.

    An overfitting check. Late rounds are uniformly fast, wide, and short-period, which a
    policy can exploit by assuming those properties instead of reading each rock. Defaults
    to 0.0 so every existing stage strips it from the task hash.
    """
    ships: int = 1
    """How many ships fly this round, all of them driven by the same policy.

    Above one, ship-to-ship collisions and friendly fire are lethal, so the policy has to
    learn to share the arena with copies of itself rather than only to dodge asteroids.
    """

    @property
    def max_decisions(self) -> int:
        return max(1, int(self.max_seconds * 15))

    @property
    def completion(self) -> str:
        return "survival" if self.survival else "waves"

    @property
    def teammates(self) -> int:
        return max(0, self.ships - 1)

    def game_config(self, base: GameConfig) -> GameConfig:
        config = copy.deepcopy(base)
        # Keep the mobile observation and action vocabulary in every stage so one
        # network can move through the whole curriculum. Early thrust actions are harmless
        # aliases until movement is enabled.
        config.ship.mobile = True
        config.ship.invulnerable = self.ship_invulnerable
        config.ship.acceleration = 180.0 if self.movement_enabled else 0.0
        config.ship.drag = 0.15
        config.ship.max_speed = 240.0
        asteroid = config.asteroid
        asteroid.spawn_mode = "interval" if self.survival else "wave"
        asteroid.wave_threshold = 0
        asteroid.wave_threshold_start = None
        asteroid.wave_composition = list(self.composition)
        asteroid.wave_size = len(self.composition)
        asteroid.wave_growth = 0
        asteroid.wave_size_max = len(self.composition)
        asteroid.active_cap = 26
        asteroid.active_cap_start = None
        asteroid.initial_wave_delay = 0.5
        asteroid.wave_delay = 0.75
        asteroid.wave_spawn_interval = 0.20
        asteroid.spawn_size = self.asteroid_size if self.survival else None
        asteroid.heading_mode = "aimed" if self.survival else "random"
        asteroid.spawn_spread_start = None
        if self.survival:
            asteroid.spawn_interval = self.spawn_interval
            asteroid.spawn_interval_start = None
            asteroid.spawn_spread = self.spawn_spread
            asteroid.initial_asteroids = min(self.initial_asteroids, asteroid.active_cap)
            asteroid.spawn_safe_radius = self.spawn_safe_radius
            asteroid.spawn_safe_seconds = self.spawn_safe_seconds
            asteroid.variety_probability = self.variety_probability
        else:
            # Leave every wave-stage knob exactly as it was, so the config a wave stage
            # produces -- and therefore its task hash -- is unchanged by survival support.
            asteroid.spawn_spread = 0.0
        asteroid.min_speed = self.min_speed
        asteroid.max_speed = self.max_speed
        asteroid.min_speed_start = asteroid.max_speed_start = None
        # Fragments outrun their parent, in every mode. The wave curriculum used to leave
        # these at 1.0, so a policy trained on it had never seen debris move faster than the
        # rock it came from -- and then died to its own splits in arcade play, where the
        # multipliers are 1.15 and 1.35. It shot well and mispredicted the pieces.
        asteroid.medium_speed_multiplier = 1.15
        asteroid.small_speed_multiplier = 1.35
        asteroid.amplitude_min = self.amplitude_min
        asteroid.amplitude_max = self.amplitude_max
        asteroid.amplitude_max_start = None
        asteroid.wavelength_min = self.wavelength_min
        asteroid.wavelength_max = self.wavelength_max
        asteroid.linear_probability = self.linear_probability
        if self.patterns:
            asteroid.motion_mode = "pool"
            asteroid.pattern_pool = list(self.patterns)
        else:
            asteroid.motion_mode = "linear"
        if self.ships > 1:
            config.ships = [ShipSpec(f"ship{index + 1}", "ppo")
                            for index in range(self.ships)]
            # Both halves of the rule the round is teaching: bumping kills both ships, and
            # a shot that lands on a teammate kills it.
            config.ship.friendly_collisions = "full"
        # A survival round has no wave to finish; the decision limit ends it instead.
        config.objective.max_waves = None if self.survival else self.target_waves
        config.objective.max_steps = None
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class CurriculumSpec:
    base: GameConfig
    reward: RewardConfig
    stages: tuple[CurriculumStage, ...]
    evaluation_episodes: int = 64
    retention_evaluation_episodes: int = 16
    retention_sample: int = 0
    """Prior stages scored per evaluation; 0 scores every one of them.

    Re-scoring every prior stage costs more than the training it interleaves with -- at
    curriculum stage 41 it was two thirds of all evaluation decisions and half of the run's
    total. Retention is judged on the pooled sample now, so a rotating subset answers the
    same question: each stage is still revisited regularly, and pooling across evaluations
    recovers the sample size that a single sweep would have given.
    """
    promotion_completion: float = 0.80
    retention_completion: float = 0.75
    retention_floor: float = 0.50
    """A single prior stage this far below is real forgetting, not an unlucky sample."""
    promotion_accuracy: float = 0.45
    promotion_streak: int = 2
    promotion_window: int = 4
    current_stage_probability: float = 0.75
    promotion_clear_rate: float = 0.0
    observation_version: int = 4
    evaluation_panels: int = 1
    test_evaluation_episodes: int = 128

    @property
    def max_teammates(self) -> int:
        """Teammate slots the observation needs: sized for the busiest round, always.

        Sizing per stage would change the observation shape part-way through a run, which
        no single network can follow.
        """
        return max((stage.teammates for stage in self.stages), default=0)


def load_curriculum(path: str | Path) -> CurriculumSpec:
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    settings = raw.get("curriculum", {})

    def relative_path(value: str | Path) -> Path:
        result = Path(value)
        if result.is_absolute():
            return result
        candidate = source.parent / result
        return candidate if candidate.exists() else result

    parent = (load_curriculum(relative_path(settings["extends"]))
              if settings.get("extends") else None)
    if settings.get("base_config"):
        base = load_config(relative_path(settings["base_config"]))
    elif parent is not None:
        base = parent.base
    else:
        base = load_config(relative_path("configs/rl-arcade.toml"))
    reward_values = asdict(parent.reward) if parent is not None else {}
    reward_values.update(raw.get("reward", {}))
    reward = RewardConfig(**reward_values)
    size_names = {"small": 1, "medium": 2, "large": 3}

    def composition(item: dict) -> tuple[int, ...]:
        raw_sizes = item.get("composition")
        if raw_sizes is None:
            # Backward compatibility with the original all-large wave curriculum.
            raw_sizes = [item.get("spawn_size", 3)] * int(item.get("wave_size", 4))
        try:
            sizes = tuple(size_names.get(value, value) for value in raw_sizes)
        except TypeError as exc:
            raise ValueError("stage composition must be a list of asteroid sizes") from exc
        if not sizes or any(size not in {1, 2, 3} for size in sizes):
            raise ValueError("stage composition entries must be small, medium, large, or 1-3")
        return sizes

    stage_items = list(raw.get("stages", []))
    progression = raw.get("progression")
    if progression:
        count = int(progression["rounds"])
        survival = bool(progression.get("survival", False))
        # Survival rounds spawn on a clock, so they have no wave composition to describe.
        compositions = progression.get("compositions") or ([["large"]] if survival else None)
        composition_every = int(progression.get("composition_every", 1))
        if count < 1 or composition_every < 1 or not compositions:
            raise ValueError("progression rounds, composition_every, and compositions must be positive")

        def stepped(name: str, index: int, default: float | None = None) -> float:
            start = progression.get(f"{name}_start")
            if start is None:
                if default is None:
                    raise ValueError(f"progression is missing {name}_start")
                return default
            return float(start) + index * float(progression.get(f"{name}_step", 0.0))

        parent_count = len(parent.stages) if parent is not None else 0
        phases = list(progression.get("phases", []))
        for index in range(count):
            tier = min(index // composition_every, len(compositions) - 1)
            phase = next((item for item in phases
                          if index + 1 <= int(item["until"])), {})
            stage_items.append({
                "name": f"{progression.get('name_prefix', 'round')}-{parent_count + index + 1}",
                "composition": compositions[tier],
                "target_waves": int(progression.get("target_waves", 1)),
                "min_speed": stepped("min_speed", index),
                "max_speed": stepped("max_speed", index),
                "linear_probability": float(phase.get(
                    "linear_probability", stepped(
                        "linear_probability", index,
                        float(progression.get("linear_probability", 0.0))))),
                "patterns": phase.get("patterns", progression.get("patterns", [])),
                "amplitude_min": stepped("amplitude_min", index),
                "amplitude_max": stepped("amplitude_max", index),
                "wavelength_min": stepped("wavelength_min", index),
                "wavelength_max": stepped("wavelength_max", index),
                "movement_enabled": bool(progression.get("movement_enabled", True)),
                "ship_invulnerable": bool(progression.get("ship_invulnerable", False)),
                "max_seconds": stepped("max_seconds", index),
                "no_hit_seconds": stepped("no_hit_seconds", index),
                "survival": survival,
                "spawn_interval": stepped("spawn_interval", index, 1.25),
                "spawn_spread": stepped("spawn_spread", index, 0.0),
                "ships": int(progression.get("ships", 1)),
                "asteroid_size": phase.get(
                    "asteroid_size", progression.get("asteroid_size", 3)),
                "variety_probability": float(phase.get(
                    "variety_probability", progression.get("variety_probability", 0.0))),
                "initial_asteroids": stepped("initial_asteroids", index, 0.0),
                "spawn_safe_radius": float(progression.get("spawn_safe_radius", 0.0)),
                "spawn_safe_seconds": float(progression.get("spawn_safe_seconds", 0.0)),
                "promotion_accuracy": float(progression.get("promotion_accuracy", 0.05)),
                "miss_penalty": float(progression.get("miss_penalty", reward.miss_penalty)),
            })

    added_stages = tuple(CurriculumStage(
        name=item["name"], composition=composition(item),
        target_waves=int(item.get("target_waves", 1)),
        min_speed=float(item["min_speed"]), max_speed=float(item["max_speed"]),
        linear_probability=float(item.get("linear_probability", 1.0)),
        patterns=tuple(item.get("patterns", [])),
        amplitude_min=float(item.get("amplitude_min", 0.0)),
        amplitude_max=float(item.get("amplitude_max", 0.0)),
        wavelength_min=float(item.get("wavelength_min", 3.0)),
        wavelength_max=float(item.get("wavelength_max", 5.0)),
        movement_enabled=bool(item.get("movement_enabled", False)),
        ship_invulnerable=bool(item.get("ship_invulnerable", False)),
        max_seconds=float(item.get("max_seconds", 60.0)),
        no_hit_seconds=float(item.get("no_hit_seconds", 6.0)),
        promotion_accuracy=(float(item["promotion_accuracy"])
                            if "promotion_accuracy" in item else None),
        miss_penalty=(float(item["miss_penalty"])
                      if "miss_penalty" in item else None),
        survival=bool(item.get("survival", False)),
        spawn_interval=float(item.get("spawn_interval", 1.25)),
        spawn_spread=float(item.get("spawn_spread", 0.0)),
        initial_asteroids=int(round(float(item.get("initial_asteroids", 0)))),
        spawn_safe_radius=float(item.get("spawn_safe_radius", 0.0)),
        spawn_safe_seconds=float(item.get("spawn_safe_seconds", 0.0)),
        ships=int(item.get("ships", 1)),
        asteroid_size=_asteroid_size(item.get("asteroid_size", 3)),
        variety_probability=float(item.get("variety_probability", 0.0)),
    ) for item in stage_items)
    stages = (parent.stages if parent is not None else ()) + added_stages
    if not stages:
        raise ValueError("curriculum must contain at least one stage")
    def inherited(name: str, default):
        return settings.get(name, getattr(parent, name) if parent is not None else default)

    return CurriculumSpec(
        base=base, reward=reward, stages=stages,
        evaluation_episodes=int(inherited("evaluation_episodes", 64)),
        retention_evaluation_episodes=int(
            inherited("retention_evaluation_episodes", 16)),
        retention_sample=int(inherited("retention_sample", 0)),
        promotion_completion=float(inherited("promotion_completion", 0.80)),
        retention_completion=float(inherited("retention_completion", 0.75)),
        retention_floor=float(inherited("retention_floor", 0.50)),
        promotion_accuracy=float(inherited("promotion_accuracy", 0.45)),
        promotion_streak=int(inherited("promotion_streak", 2)),
        promotion_window=int(inherited("promotion_window", 4)),
        current_stage_probability=float(inherited("current_stage_probability", 0.75)),
        promotion_clear_rate=float(inherited("promotion_clear_rate", 0.0)),
        observation_version=int(inherited("observation_version", 4)),
        evaluation_panels=int(inherited("evaluation_panels", 1)),
        test_evaluation_episodes=int(inherited("test_evaluation_episodes", 128)),
    )


class CurriculumManager:
    def __init__(self, spec: CurriculumSpec, seed: int = 0, *, stage: int = 0, streak: int = 0,
                 mastered: bool = False, promotion_history: list[bool] | None = None):
        self.spec = spec
        self.stage = max(0, min(stage, len(spec.stages) - 1))
        self.streak = max(0, streak)
        self.mastered = bool(mastered)
        inherited_history = ([True] * self.streak
                             if promotion_history is None else promotion_history)
        self.promotion_history = list(inherited_history)[-spec.promotion_window:]
        self._rng = random.Random(seed)

    def sample_stage(self, *, focus_stage: int | None = None,
                     focus_probability: float = 0.40,
                     recovery_current_probability: float = 0.40) -> int:
        """Choose a lesson, mixing recovery with current and other retained lessons."""
        if focus_stage is not None and 0 <= focus_stage < self.stage:
            draw = self._rng.random()
            if draw < focus_probability:
                return focus_stage
            if draw < focus_probability + recovery_current_probability:
                return self.stage
            others = [index for index in range(self.stage) if index != focus_stage]
            return self._rng.choice(others) if others else self.stage
        if self.stage == 0 or self._rng.random() < self.spec.current_stage_probability:
            return self.stage
        return self._rng.randrange(self.stage)

    def consider_promotion(self, results: list[dict]) -> bool:
        current = results[self.stage]
        stage = self.spec.stages[self.stage]
        accuracy_threshold = (self.spec.promotion_accuracy
                              if stage.promotion_accuracy is None
                              else stage.promotion_accuracy)
        current_ok = (current["completion_rate"] >= self.spec.promotion_completion
                      and current.get("clear_rate", 0.0) >= self.spec.promotion_clear_rate
                      and current["mean_accuracy"] >= accuracy_threshold)
        retained = retention_holds(
            results[:self.stage],
            retention_completion=self.spec.retention_completion,
            retention_floor=self.spec.retention_floor)
        passed = bool(current_ok and retained)
        self.promotion_history.append(passed)
        self.promotion_history = self.promotion_history[-self.spec.promotion_window:]
        self.streak = sum(self.promotion_history)
        if self.streak < self.spec.promotion_streak:
            return False
        if self.stage < len(self.spec.stages) - 1:
            self.stage += 1
            self.streak = 0
            self.promotion_history = []
            return True
        self.mastered = True
        return False
