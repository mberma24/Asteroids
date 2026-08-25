"""Shared-policy multi-agent environment and a compact MAPPO trainer.

The existing co-op curriculum used one PPO learner plus frozen companion snapshots.  That
is useful self-play exposure, but it is not multi-agent learning: companion actions never
enter the rollout buffer and the reward ends when the designated learner dies.  This module
advances all ships together, emits one local observation/action sample per living ship, and
uses a centralized team value during training while exporting a decentralized shared actor.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..actions import Action
from ..config import GameConfig, ShipSpec
from ..math2d import Vec2, wrapped_delta
from ..simulation import ASTEROID_RADII, Simulation
from .curriculum import load_curriculum
from .environment import (ASTEROID_FEATURES, GLOBAL_FEATURES, MOBILE_ACTIONS,
                          MOBILE_SHIP_FEATURES, PROJECTILE_FEATURES, TEAMMATE_FEATURES,
                          RewardConfig, encode_observation, global_feature_count,
                          history_offsets)

OBJECTIVE_FEATURES = 8
TEAM_LEVEL_ROUNDS = (29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 96)


def team_config(base: GameConfig, ships: int, *, level: int, protect: bool = False) -> GameConfig:
    """Scale one solo-v2 stage into a bounded 1-8 ship cooperative arena."""
    if not 1 <= ships <= 8:
        raise ValueError("ships must be between 1 and 8")
    if not 1 <= level <= len(TEAM_LEVEL_ROUNDS):
        raise ValueError(f"level must be between 1 and {len(TEAM_LEVEL_ROUNDS)}")
    config = copy.deepcopy(base)
    scale = ships ** 0.25
    pressure = math.sqrt(ships)
    config.arena.width = int(round(config.arena.width * scale))
    config.arena.height = int(round(config.arena.height * scale))
    config.ships = [ShipSpec(f"ship{index + 1}", "ppo") for index in range(ships)]
    config.asteroid.active_cap = max(config.asteroid.active_cap,
                                     int(round(config.asteroid.active_cap * pressure)))
    config.asteroid.initial_asteroids = min(
        config.asteroid.active_cap,
        int(round(config.asteroid.initial_asteroids * pressure)))
    config.asteroid.spawn_interval /= pressure
    config.ship.friendly_collisions = (
        "off" if level <= 4 else "ships" if level <= 8 else "full")
    config.objective.protect = bool(protect)
    if protect:
        # Protection difficulty is deliberately monotone: target bias is represented by
        # narrowing spread, while health falls from 20 to 8 across the twelve levels.
        config.objective.object_health = max(8, 21 - level)
        config.asteroid.spawn_spread = min(config.asteroid.spawn_spread,
                                           150.0 - 10.0 * level)
    config.validate()
    return config


class MultiAgentAsteroidsEnv:
    """Parallel shared-team environment with fixed-width local observations."""

    def __init__(self, config: GameConfig, *, max_ships: int = 8,
                 max_asteroids: int | None = None,
                 max_decisions: int = 450, frame_skip: int = 4,
                 history_frames: int = 8, history_long_frames: int = 8,
                 history_long_stride: int = 8,
                 reward_config: RewardConfig | None = None,
                 completion: str = "survival",
                 terminate_on_team_death: bool = False,
                 observation_version: int = 7,
                 mask_unsafe_fire: bool = False):
        if len(config.ships) > max_ships:
            raise ValueError("config has more ships than max_ships")
        self.config = config
        self.max_ships = max_ships
        self.max_teammates = max_ships - 1
        self.max_decisions = max_decisions
        self.frame_skip = frame_skip
        self.max_asteroids = max_asteroids or config.asteroid.active_cap
        if self.max_asteroids < config.asteroid.active_cap:
            raise ValueError("max_asteroids cannot be smaller than the active cap")
        self.max_projectiles = 8 * max_ships
        self.history_slots = history_offsets(
            history_frames, history_long_frames, history_long_stride)
        self._history_depth = max(self.history_slots) + 1 if self.history_slots else 0
        self._history: dict[int, deque] = {}
        self.simulation = Simulation(config)
        self.state = None
        self.decisions = 0
        self.initial_health = max(1, config.objective.object_health)
        self.initial_ship_count = len(config.ships)
        self.reward_config = reward_config
        self.completion = completion
        self.terminate_on_team_death = bool(terminate_on_team_death)
        self.observation_version = int(observation_version)
        self.mask_unsafe_fire = bool(mask_unsafe_fire)
        self.metrics: dict = {}

    @property
    def observation_size(self) -> int:
        return (MOBILE_SHIP_FEATURES
                + self.max_asteroids * (ASTEROID_FEATURES + 2 * len(self.history_slots))
                + self.max_projectiles * PROJECTILE_FEATURES
                + self.max_teammates * TEAMMATE_FEATURES
                + global_feature_count(self.observation_version) + OBJECTIVE_FEATURES)

    def reset(self, seed: int = 0) -> tuple[dict[str, np.ndarray], dict]:
        self.state = self.simulation.reset(seed)
        self.decisions = 0
        self._history.clear()
        self.metrics = {
            "seed": seed, "alive_ship_time": 0.0, "ship_deaths": 0,
            "ship_collisions": 0, "friendly_fire": 0, "object_damage": 0,
            "asteroids_destroyed": 0, "reward": 0.0,
            "shots_fired": 0, "shots_missed": 0, "shots_resolved": 0,
            "is_success": False,
        }
        observations = self._observations()
        self._record_history()
        return observations, {"alive": self.alive_ids}

    @property
    def alive_ids(self) -> list[str]:
        return [ship.id for ship in self.state.ships if ship.alive]

    def _objective(self, ship_id: str) -> np.ndarray:
        ship = next(item for item in self.state.ships if item.id == ship_id)
        objective = self.state.objective
        if not objective.enabled:
            return np.zeros(OBJECTIVE_FEATURES, dtype=np.float32)
        delta = wrapped_delta(Vec2(ship.x, ship.y), Vec2(objective.x, objective.y),
                              self.state.width, self.state.height)
        diagonal = math.hypot(self.state.width / 2, self.state.height / 2)
        bearing = math.atan2(delta.y, delta.x) - ship.angle
        return np.asarray((1.0, delta.x / diagonal, delta.y / diagonal,
                           delta.length() / diagonal, math.sin(bearing), math.cos(bearing),
                           max(0.0, objective.health / self.initial_health),
                           objective.radius / ASTEROID_RADII[3]), dtype=np.float32)

    def _observations(self) -> dict[str, np.ndarray]:
        result = {}
        for ship in self.state.ships:
            if not ship.alive:
                result[ship.id] = np.zeros(self.observation_size, dtype=np.float32)
                continue
            local = encode_observation(
                self.state, ship.id, self.config, self.max_decisions, self.max_asteroids,
                self.frame_skip, self._history, len(self.history_slots), self.history_slots,
                self.max_projectiles, self.max_teammates, False,
                observation_version=self.observation_version,
                spawn_phase=self.simulation.spawn_phase)
            result[ship.id] = np.concatenate((local, self._objective(ship.id)))
        return result

    def _record_history(self) -> None:
        if not self._history_depth:
            return
        live = set()
        for asteroid in self.state.asteroids:
            live.add(asteroid.id)
            self._history.setdefault(
                asteroid.id, deque(maxlen=self._history_depth)).appendleft(
                    (asteroid.x, asteroid.y))
        for key in tuple(self._history):
            if key not in live:
                del self._history[key]

    def action_masks(self) -> dict[str, np.ndarray]:
        masks = {}
        for ship in self.state.ships:
            mask = np.ones(len(MOBILE_ACTIONS), dtype=bool)
            if not ship.alive:
                mask[:] = False
                mask[Action.NOOP] = True
            elif ship.cooldown > 0:
                for action in MOBILE_ACTIONS:
                    if action.fire:
                        mask[int(action)] = False
            if ship.alive and self.mask_unsafe_fire:
                others = [other for other in self.state.ships
                          if other.id != ship.id and other.alive]
                for action in MOBILE_ACTIONS:
                    if action.fire and self._unsafe_fire(ship, action, others):
                        mask[int(action)] = False
            masks[ship.id] = mask
        return masks

    def _unsafe_fire(self, ship, action: Action, others: list) -> bool:
        """Whether firing on the action's first frame crosses a teammate's hit circle."""
        angle = ship.angle + action.turn * self.config.ship.turn_speed / self.config.arena.fps
        dx, dy = math.cos(angle), math.sin(angle)
        lifetime = self.config.projectile.lifetime
        bullet_vx = self.config.projectile.speed * dx
        bullet_vy = self.config.projectile.speed * dy
        margin = self.config.ship.radius + self.config.projectile.radius + 16.0
        for other in others:
            # Test periodic images explicitly: long-lived bullets can cross a seam and hit a
            # teammate whose shortest wrapped displacement lies behind the shooter.
            for x_wrap in (-self.state.width, 0, self.state.width):
                for y_wrap in (-self.state.height, 0, self.state.height):
                    rx = other.x + x_wrap - ship.x
                    ry = other.y + y_wrap - ship.y
                    rvx, rvy = other.vx - bullet_vx, other.vy - bullet_vy
                    speed2 = rvx * rvx + rvy * rvy
                    closest = (max(0.0, min(lifetime, -(rx * rvx + ry * rvy) / speed2))
                               if speed2 > 1e-9 else 0.0)
                    gap = math.hypot(rx + rvx * closest, ry + rvy * closest)
                    if closest > 0.0 and gap <= margin:
                        return True
        return False

    def step(self, actions: dict[str, int]):
        if self.state is None:
            raise RuntimeError("call reset before step")
        before_alive = len(self.alive_ids)
        before_health = self.state.objective.health
        elapsed_start = self.state.elapsed
        events = []
        hit_value = 0.0
        shots_fired = shots_expired = 0
        for _ in range(self.frame_skip):
            sizes = {rock.id: rock.size for rock in self.state.asteroids}
            mapped = {ship_id: MOBILE_ACTIONS[int(index)]
                      for ship_id, index in actions.items() if ship_id in self.alive_ids}
            result = self.simulation.step(mapped)
            self.state = result.snapshot
            events.extend(result.events)
            for event in result.events:
                if event.kind == "asteroid_shot":
                    size = sizes.get(int(event.entity_id), 0)
                    weights = ({3: self.reward_config.large, 2: self.reward_config.medium,
                                1: self.reward_config.small}
                               if self.reward_config is not None else
                               {3: 0.6, 2: 0.3, 1: 0.15})
                    hit_value += weights.get(size, 0.0)
                elif event.kind == "projectile_fired":
                    shots_fired += 1
                elif event.kind == "projectile_expired":
                    shots_expired += 1
            if (self.terminate_on_team_death
                    and len(self.alive_ids) < self.initial_ship_count):
                break
            if result.terminated or result.truncated:
                break
        self.decisions += 1
        elapsed = self.state.elapsed - elapsed_start
        alive = len(self.alive_ids)
        newly_dead = max(0, before_alive - alive)
        collisions = sum(event.kind == "ship_collision" for event in events)
        friendly = sum(event.kind == "friendly_fire" for event in events)
        damage = max(0, before_health - self.state.objective.health)
        normal = self.frame_skip / self.config.arena.fps
        if self.reward_config is None:
            reward = 0.10 * (alive / self.initial_ship_count) * elapsed / normal
            reward += hit_value / self.initial_ship_count
            reward -= 5.0 * newly_dead / self.initial_ship_count
            reward -= 2.0 * (collisions + friendly) / self.initial_ship_count
            if self.config.objective.protect:
                reward -= 10.0 * damage / self.initial_health
        else:
            intact = alive == self.initial_ship_count
            fraction = elapsed / normal if self.reward_config.time_scaled_survival else 1.0
            reward = self.reward_config.survival_bonus * fraction if intact else 0.0
            reward += hit_value
            if newly_dead:
                reward -= self.reward_config.death_penalty
            reward -= self.reward_config.collision_penalty * collisions
            reward -= self.reward_config.friendly_fire_dealt_penalty * friendly
            reward -= self.reward_config.miss_penalty * shots_expired
            if self.config.objective.protect:
                reward -= 10.0 * damage / self.initial_health
        timed_out = self.decisions >= self.max_decisions and not self.state.terminated
        team_failed = self.terminate_on_team_death and alive < self.initial_ship_count
        terminated = bool(self.state.terminated or team_failed)
        truncated = bool(self.state.truncated or timed_out)
        wave_cleared = (self.state.terminal_reason is not None
                        and self.state.terminal_reason.value == "waves_cleared")
        success = bool(alive == self.initial_ship_count and (
            timed_out if self.completion == "survival" else wave_cleared))
        if self.reward_config is None and timed_out:
            reward += 10.0 * alive / self.initial_ship_count
            if self.config.objective.protect:
                reward += 10.0 * max(0, self.state.objective.health) / self.initial_health
        elif self.reward_config is not None:
            if success:
                reward += self.reward_config.round_clear
            elif (timed_out or wave_cleared) and self.completion == "waves":
                reward -= self.reward_config.timeout_penalty
        if terminated and self.state.objective.enabled and self.state.objective.health <= 0:
            reward -= 10.0
        self.metrics["alive_ship_time"] += elapsed * alive / self.initial_ship_count
        self.metrics["ship_deaths"] += newly_dead
        self.metrics["ship_collisions"] += collisions
        self.metrics["friendly_fire"] += friendly
        self.metrics["object_damage"] += damage
        self.metrics["asteroids_destroyed"] += sum(
            event.kind == "asteroid_shot" for event in events)
        self.metrics["shots_fired"] += shots_fired
        self.metrics["shots_missed"] += shots_expired
        self.metrics["shots_resolved"] += shots_expired + sum(
            event.kind == "asteroid_shot" for event in events)
        if (terminated or truncated) and self.reward_config is not None:
            unresolved = max(0, self.metrics["shots_fired"] - self.metrics["shots_resolved"])
            reward -= self.reward_config.miss_penalty * unresolved
            self.metrics["shots_missed"] += unresolved
            self.metrics["shots_resolved"] += unresolved
        self.metrics["is_success"] = success
        self.metrics["reward"] += reward
        observations = self._observations()
        self._record_history()
        info = {"alive": self.alive_ids}
        if terminated or truncated:
            duration = self.max_decisions * normal
            info["episode_metrics"] = {
                **self.metrics,
                "alive_ship_time_fraction": self.metrics["alive_ship_time"] /
                    max(duration, 1e-9),
                "all_ships_survived": alive == self.initial_ship_count,
                "completed_stage": success,
                "final_alive_fraction": alive / self.initial_ship_count,
                "object_survived": not self.state.objective.enabled
                    or self.state.objective.health > 0,
                "final_object_health_fraction": (1.0 if not self.state.objective.enabled else
                    max(0.0, self.state.objective.health / self.initial_health)),
            }
        return observations, reward, terminated, truncated, info


@dataclass(slots=True)
class MAPPOSettings:
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 6
    rollout_steps: int = 256


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("MAPPO requires the PPO dependencies: pip install -e '.[ppo]'") from exc
    return torch, nn


def build_shared_actor_critic(width: int, torch, nn, *, max_asteroids: int,
                              max_ships: int, history_slots: int = 16):
    """Construct the serializable shared actor/centralized critic network."""
    class SharedActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.asteroid_width = ASTEROID_FEATURES + 2 * history_slots
            self.max_asteroids = max_asteroids
            self.max_projectiles = 8 * max_ships
            self.max_teammates = max_ships - 1
            def entity(entity_width):
                return nn.Sequential(nn.Linear(entity_width, 96), nn.ReLU(),
                                     nn.Linear(96, 96), nn.ReLU())
            self.asteroid_encoder = entity(self.asteroid_width)
            self.projectile_encoder = entity(PROJECTILE_FEATURES)
            self.teammate_encoder = entity(TEAMMATE_FEATURES)
            direct = MOBILE_SHIP_FEATURES + GLOBAL_FEATURES + OBJECTIVE_FEATURES
            self.fusion = nn.Sequential(nn.Linear(direct + 6 * 96, 512), nn.ReLU(),
                                        nn.Linear(512, 256), nn.ReLU())
            self.actor = nn.Linear(256, len(MOBILE_ACTIONS))
            self.critic = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

        @staticmethod
        def pool(encoded, present):
            mask = present.unsqueeze(-1)
            mean = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            largest = encoded.masked_fill(mask == 0, float("-inf")).max(1).values
            return torch.cat((mean, torch.nan_to_num(largest, neginf=0.0)), -1)

        def encode(self, observation):
            at = 0
            ship = observation[:, :MOBILE_SHIP_FEATURES]; at += MOBILE_SHIP_FEATURES
            size = self.max_asteroids * self.asteroid_width
            asteroids = observation[:, at:at + size].reshape(
                -1, self.max_asteroids, self.asteroid_width); at += size
            size = self.max_projectiles * PROJECTILE_FEATURES
            projectiles = observation[:, at:at + size].reshape(
                -1, self.max_projectiles, PROJECTILE_FEATURES); at += size
            size = self.max_teammates * TEAMMATE_FEATURES
            teammates = observation[:, at:at + size].reshape(
                -1, self.max_teammates, TEAMMATE_FEATURES); at += size
            globals_and_objective = observation[:, at:]
            combined = torch.cat((
                ship, globals_and_objective,
                self.pool(self.asteroid_encoder(asteroids), asteroids[..., 0]),
                self.pool(self.projectile_encoder(projectiles), projectiles[..., 0]),
                self.pool(self.teammate_encoder(teammates), teammates[..., 0]),
            ), -1)
            return self.fusion(combined)

        def value(self, latents):
            central = torch.cat((latents.mean(0), latents.max(0).values), dim=-1)
            return self.critic(central).squeeze(-1)
    return SharedActorCritic()


def evaluate_shared_mappo(checkpoint: str | Path, *, episodes: int = 64,
                          ships: int = 8, level: int = 12,
                          protect: bool | None = None, seed: int = 20_000) -> dict:
    torch, nn = require_torch()
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if protect is None:
        protect = bool(metadata.get("protect", False))
    max_ships = int(metadata.get("max_ships", 8))
    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[TEAM_LEVEL_ROUNDS[level - 1] - 1]
    config = team_config(stage.game_config(spec.base), ships, level=level, protect=protect)
    env = MultiAgentAsteroidsEnv(
        config, max_ships=max_ships,
        max_asteroids=int(metadata.get("max_asteroids", config.asteroid.active_cap)))
    model = build_shared_actor_critic(
        env.observation_size, torch, nn, max_asteroids=env.max_asteroids,
        max_ships=max_ships, history_slots=len(env.history_slots))
    payload = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    records = []
    for episode in range(episodes):
        observations, _ = env.reset(seed + episode)
        done = False
        while not done:
            alive = env.alive_ids
            batch = torch.as_tensor(np.stack([observations[key] for key in alive]),
                                    dtype=torch.float32)
            with torch.no_grad():
                logits = model.actor(model.encode(batch))
            masks = torch.as_tensor(np.stack([env.action_masks()[key] for key in alive]))
            actions = logits.masked_fill(~masks, -1e9).argmax(-1).tolist()
            observations, _, terminated, truncated, info = env.step(dict(zip(alive, actions)))
            done = terminated or truncated
        records.append(info["episode_metrics"])
    keys = ("alive_ship_time_fraction", "all_ships_survived", "final_alive_fraction",
            "ship_collisions", "friendly_fire", "object_survived",
            "final_object_health_fraction", "reward")
    return {"checkpoint": str(checkpoint), "episodes": episodes, "ships": ships,
            "level": level, "protect": protect,
            **{key: sum(float(row[key]) for row in records) / len(records) for key in keys}}


def play_shared_mappo(checkpoint: str | Path, *, ships: int = 8, level: int = 12,
                      protect: bool | None = None, seed: int = 7) -> int:
    """Render any number of decentralized copies of a saved shared actor."""
    import pygame
    from ..renderer import Renderer

    torch, nn = require_torch()
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if protect is None:
        protect = bool(metadata.get("protect", False))
    max_ships = int(metadata.get("max_ships", 8))
    if ships > max_ships:
        raise ValueError(f"checkpoint supports at most {max_ships} ships")
    spec = load_curriculum("configs/rl-survival-v2.toml")
    stage = spec.stages[TEAM_LEVEL_ROUNDS[level - 1] - 1]
    config = team_config(stage.game_config(spec.base), ships, level=level, protect=protect)
    env = MultiAgentAsteroidsEnv(
        config, max_ships=max_ships,
        max_asteroids=int(metadata.get("max_asteroids", config.asteroid.active_cap)))
    model = build_shared_actor_critic(
        env.observation_size, torch, nn, max_asteroids=env.max_asteroids,
        max_ships=max_ships, history_slots=len(env.history_slots))
    payload = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    pygame.init()
    renderer = Renderer(pygame, config.arena.width, config.arena.height)
    clock = pygame.time.Clock()
    observations, _ = env.reset(seed)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                              and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                seed += 1
                observations, _ = env.reset(seed)
        if not running:
            break
        if not (env.state.terminated or env.state.truncated
                or env.decisions >= env.max_decisions):
            alive = env.alive_ids
            batch = torch.as_tensor(np.stack([observations[key] for key in alive]),
                                    dtype=torch.float32)
            with torch.no_grad():
                logits = model.actor(model.encode(batch))
            masks = torch.as_tensor(np.stack([env.action_masks()[key] for key in alive]))
            choices = logits.masked_fill(~masks, -1e9).argmax(-1).tolist()
            observations, _, _, _, _ = env.step(dict(zip(alive, choices)))
        renderer.draw(env.state)
        clock.tick(15)
    pygame.quit()
    return 0


def train_shared_mappo(output_dir: str | Path, *, episodes: int, max_ships: int = 8,
                       protect: bool = False, seed: int = 0, device: str = "cpu",
                       settings: MAPPOSettings | None = None,
                       initialize_from: str | Path | None = None,
                       resume: str | Path | None = None) -> Path:
    """Train a shared decentralized actor with a centralized team critic.

    This deliberately uses a small in-project loop: SB3's PPO rollout API assumes one action
    and one reward per environment and cannot correctly represent simultaneous ship actions.
    """
    torch, nn = require_torch()
    settings = settings or MAPPOSettings()
    if initialize_from and resume:
        raise ValueError("use either initialize_from or resume")
    if protect and not (initialize_from or resume):
        raise ValueError("protection must initialize from a mastered team-survival checkpoint")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    spec = load_curriculum("configs/rl-survival-v2.toml")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    first_stage = spec.stages[TEAM_LEVEL_ROUNDS[0] - 1]
    first_config = team_config(first_stage.game_config(spec.base), 1, level=1, protect=protect)
    # Curriculum stages normalize their active cap to 26.
    max_asteroids = int(math.ceil(26 * math.sqrt(max_ships)))
    prototype = MultiAgentAsteroidsEnv(
        first_config, max_ships=max_ships, max_asteroids=max_asteroids)
    model = build_shared_actor_critic(
        prototype.observation_size, torch, nn, max_asteroids=max_asteroids,
        max_ships=max_ships, history_slots=len(prototype.history_slots)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    start_episode = 0
    if initialize_from or resume:
        source = Path(initialize_from or resume)
        source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if int(source_metadata.get("max_ships", -1)) != max_ships:
            raise ValueError("team checkpoint max_ships does not match")
        payload = torch.load(source / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        if resume:
            optimizer.load_state_dict(payload["optimizer"])
            start_episode = int(source_metadata.get("episodes", 0))
    log_path = destination / "training.jsonl"
    global_steps = 0
    started = time.monotonic()

    for episode in range(start_episode + 1, start_episode + episodes + 1):
        # Half the lessons use the frontier count; the remainder rehearse every smaller N.
        ship_count = max_ships if rng.random() < 0.5 else rng.randint(1, max_ships)
        level = rng.randint(1, len(TEAM_LEVEL_ROUNDS))
        stage = spec.stages[TEAM_LEVEL_ROUNDS[level - 1] - 1]
        episode_protect = protect and rng.random() < 0.70
        config = team_config(stage.game_config(spec.base), ship_count,
                             level=level, protect=episode_protect)
        env = MultiAgentAsteroidsEnv(
            config, max_ships=max_ships, max_asteroids=max_asteroids)
        observations, _ = env.reset(seed + episode)
        trajectory = []
        done = False
        while not done:
            alive = env.alive_ids
            batch = torch.as_tensor(np.stack([observations[key] for key in alive]),
                                    dtype=torch.float32, device=device)
            latents = model.encode(batch)
            logits = model.actor(latents)
            masks = torch.as_tensor(np.stack([env.action_masks()[key] for key in alive]),
                                    dtype=torch.bool, device=device)
            logits = logits.masked_fill(~masks, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = distribution.sample()
            value = model.value(latents)
            actions = {key: int(action) for key, action in zip(alive, sampled.tolist())}
            next_observations, reward, terminated, truncated, info = env.step(actions)
            trajectory.append((batch.detach(), masks.detach(), sampled.detach(),
                               distribution.log_prob(sampled).detach(), value.detach(), reward))
            observations = next_observations
            done = terminated or truncated
            global_steps += 1

        rewards = [item[5] for item in trajectory]
        values = [float(item[4].item()) for item in trajectory]
        advantages = [0.0] * len(trajectory)
        gae = 0.0
        next_value = 0.0
        for index in reversed(range(len(trajectory))):
            delta = rewards[index] + settings.gamma * next_value - values[index]
            gae = delta + settings.gamma * settings.gae_lambda * gae
            advantages[index] = gae
            next_value = values[index]
        returns = [advantages[i] + values[i] for i in range(len(values))]
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8)

        losses = {}
        for _ in range(settings.epochs):
            policy_terms, entropy_terms, value_terms = [], [], []
            for index, (batch, masks, actions, old_log, _old_value, _reward) in enumerate(trajectory):
                latents = model.encode(batch)
                logits = model.actor(latents).masked_fill(~masks, -1e9)
                distribution = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(distribution.log_prob(actions) - old_log)
                advantage = advantage_tensor[index]
                policy_terms.append(-torch.min(
                    ratio * advantage,
                    ratio.clamp(1 - settings.clip_range, 1 + settings.clip_range) * advantage
                ).mean())
                entropy_terms.append(distribution.entropy().mean())
                target = torch.as_tensor(returns[index], dtype=torch.float32, device=device)
                value_terms.append((model.value(latents) - target).pow(2))
            policy_loss = torch.stack(policy_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            loss = policy_loss + settings.value_coef * value_loss - settings.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses = {"policy_loss": float(policy_loss.detach()),
                      "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                      "gradient_norm": float(gradient_norm), "loss": float(loss.detach())}

        record = {"episode": episode, "environment_steps": global_steps,
                  "ships": ship_count, "level": level, "protect": episode_protect,
                  **info.get("episode_metrics", {}), **losses}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        final_episode = start_episode + episodes
        if episode % 250 == 0 or episode == final_episode:
            checkpoint = destination / f"checkpoint_{episode:06d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                       checkpoint / "model.pt")
            metadata = {"algorithm": "shared_mappo", "episodes": episode,
                        "environment_steps": global_steps, "max_ships": max_ships,
                        "protect": protect, "observation_size": prototype.observation_size,
                        "max_asteroids": max_asteroids,
                        "parent_checkpoint": str(initialize_from) if initialize_from else None,
                        "settings": asdict(settings), "wall_seconds": time.monotonic() - started}
            (checkpoint / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination / f"checkpoint_{start_episode + episodes:06d}"
