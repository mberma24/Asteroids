from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax


@dataclass(slots=True)
class MuZeroSettings:
    latent_size: int = 128
    hidden_size: int = 256
    discount: float = 0.997
    learning_rate: float = 0.001
    num_simulations: int = 16
    batch_size: int = 64
    replay_capacity: int = 50_000
    updates_per_episode: int = 32
    unroll_steps: int = 5
    """Dynamics steps trained per sample. Search is only as good as the model is deep."""
    n_step: int = 10
    """Rewards summed before bootstrapping off a stored search value."""
    value_loss_weight: float = 0.25


VALUE_TRANSFORM_EPSILON = 1e-3


def scalar_transform(value):
    """Compress unbounded returns so value loss cannot drown out the policy loss."""
    return (jnp.sign(value) * (jnp.sqrt(jnp.abs(value) + 1.0) - 1.0)
            + VALUE_TRANSFORM_EPSILON * value)


def scalar_inverse(value):
    """Invert :func:`scalar_transform`, mapping network output back to real returns."""
    epsilon = VALUE_TRANSFORM_EPSILON
    numerator = jnp.sqrt(1.0 + 4.0 * epsilon * (jnp.abs(value) + 1.0 + epsilon)) - 1.0
    return jnp.sign(value) * (jnp.square(numerator / (2.0 * epsilon)) - 1.0)


def scale_gradient(value, scale: float):
    return value * scale + jax.lax.stop_gradient(value) * (1.0 - scale)


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    policy: np.ndarray
    next_observation: np.ndarray
    done: bool
    value_target: float = 0.0
    next_policy: np.ndarray | None = None
    search_value: float = 0.0
    """Root value from the tree search, used to bootstrap n-step targets."""
    episode_id: int = 0
    """Sequence sampling must not unroll across an episode boundary."""
    successful: bool = False
    """Completed episodes are deliberately replayed more often than abundant failures."""
    stage: int = 0
    """Curriculum stage, used to keep short mastered lessons represented in replay."""


class MuZeroNetwork(nn.Module):
    num_actions: int
    latent_size: int
    hidden_size: int
    predict_continuation: bool = True

    def setup(self) -> None:
        self.representation_net = nn.Sequential((
            nn.Dense(self.hidden_size), nn.relu,
            nn.Dense(self.hidden_size), nn.relu,
            nn.Dense(self.latent_size), nn.tanh,
        ))
        self.dynamics_net = nn.Sequential((
            nn.Dense(self.hidden_size), nn.relu,
            nn.Dense(self.latent_size), nn.tanh,
        ))
        self.policy_hidden = nn.Dense(self.hidden_size)
        self.policy_output = nn.Dense(self.num_actions)
        self.value_hidden = nn.Dense(self.hidden_size)
        self.value_output = nn.Dense(1)
        self.reward_hidden = nn.Dense(self.hidden_size)
        self.reward_output = nn.Dense(1)
        if self.predict_continuation:
            self.continuation_output = nn.Dense(
                1, bias_init=nn.initializers.constant(4.0))

    def representation(self, observation: jax.Array) -> jax.Array:
        return self.representation_net(observation)

    def prediction(self, latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        policy = self.policy_output(nn.relu(self.policy_hidden(latent)))
        value = self.value_output(nn.relu(self.value_hidden(latent))).squeeze(-1)
        return policy, value

    def dynamics(self, latent: jax.Array,
                 action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        one_hot = jax.nn.one_hot(action, self.num_actions)
        next_latent = self.dynamics_net(jnp.concatenate((latent, one_hot), axis=-1))
        reward = self.reward_output(nn.relu(self.reward_hidden(next_latent))).squeeze(-1)
        continuation = (self.continuation_output(next_latent).squeeze(-1)
                        if self.predict_continuation else
                        jnp.full(next_latent.shape[:-1], 20.0, dtype=next_latent.dtype))
        return next_latent, reward, continuation

    def __call__(self, observation: jax.Array, action: jax.Array):
        latent = self.representation(observation)
        policy, value = self.prediction(latent)
        next_latent, reward, continuation = self.dynamics(latent, action)
        next_policy, next_value = self.prediction(next_latent)
        return policy, value, reward, continuation, next_policy, next_value


class MuZeroAgent:
    """Compact MuZero learner using DeepMind Mctx for Gumbel MuZero search."""

    def __init__(self, observation_size: int, num_actions: int, settings: MuZeroSettings | None = None,
                 *, seed: int = 0, predict_continuation: bool = True):
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.settings = settings or MuZeroSettings()
        self.predict_continuation = predict_continuation
        self.network = MuZeroNetwork(
            num_actions, self.settings.latent_size, self.settings.hidden_size,
            predict_continuation)
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)
        dummy_observation = jnp.zeros((1, observation_size), dtype=jnp.float32)
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)
        self.params = self.network.init(init_key, dummy_observation, dummy_action)
        self.optimizer = optax.adam(self.settings.learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)
        self.training_steps = 0
        self.episodes = 0
        self._search_cache: dict[int, Any] = {}

    def reset_optimizer(self, learning_rate: float | None = None) -> None:
        """Start fresh Adam momentum, optionally at a new learning rate."""
        if learning_rate is not None:
            self.settings.learning_rate = float(learning_rate)
        self.optimizer = optax.adam(self.settings.learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)

    def _recurrent_fn(self, params, rng_key, action, latent):
        del rng_key
        next_latent, reward, continuation = self.network.apply(
            params, latent, action, method=self.network.dynamics)
        logits, next_value = self.network.apply(
            params, next_latent, method=self.network.prediction)
        # The network learns transformed targets; search must back up real returns.
        output = mctx.RecurrentFnOutput(
            reward=scalar_inverse(reward),
            discount=jnp.asarray(self.settings.discount, dtype=reward.dtype)
            * jax.nn.sigmoid(continuation),
            prior_logits=logits,
            value=scalar_inverse(next_value),
        )
        return output, next_latent

    def _compiled_search(self, simulations: int, invalid: tuple[bool, ...] = ()):
        """Compile the Gumbel MuZero search once per simulation budget and action mask.

        Tracing ``mctx.gumbel_muzero_policy`` on every decision is what made
        search dominate training time, so the traced program is cached here and
        only rebuilt when the budget or the mask changes.
        """
        key = (simulations, invalid)
        if key not in self._search_cache:
            usable = max(1, self.num_actions - sum(invalid))
            mask = jnp.asarray(invalid, dtype=bool) if invalid else None

            @partial(jax.jit, static_argnums=())
            def run(params, search_key, batch, gumbel_scale):
                embedding = self.network.apply(
                    params, batch, method=self.network.representation)
                prior_logits, value = self.network.apply(
                    params, embedding, method=self.network.prediction)
                root = mctx.RootFnOutput(
                    prior_logits=prior_logits, value=scalar_inverse(value), embedding=embedding)
                # Masked actions are excluded from the tree entirely, so the whole simulation
                # budget goes to options that actually differ from one another.
                invalid_actions = (jnp.broadcast_to(mask, prior_logits.shape).astype(
                    prior_logits.dtype) if mask is not None else None)
                result = mctx.gumbel_muzero_policy(
                    params,
                    search_key,
                    root,
                    self._recurrent_fn,
                    num_simulations=simulations,
                    max_num_considered_actions=min(usable, simulations),
                    gumbel_scale=gumbel_scale,
                    invalid_actions=invalid_actions,
                )
                return (result.action, result.action_weights,
                        result.search_tree.summary().value)

            self._search_cache[key] = run
        return self._search_cache[key]

    def search_batch(self, observations: np.ndarray, *, explore: bool = True,
                     invalid_actions: tuple[bool, ...] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search a whole batch of roots in one compiled call.

        Searching many environments together costs barely more than searching one, so
        parallel self-play is where the remaining throughput is. ``invalid_actions`` drops
        actions that cannot do anything under the current configuration.
        """
        self.key, search_key = jax.random.split(self.key)
        simulations = max(self.settings.num_simulations, self.num_actions)
        invalid = tuple(bool(x) for x in invalid_actions) if invalid_actions else ()
        if invalid and all(invalid):
            raise ValueError("cannot mask every action")
        run = self._compiled_search(simulations, invalid)
        actions, weights, root_values = run(
            self.params,
            search_key,
            jnp.asarray(observations, dtype=jnp.float32),
            jnp.float32(1.0 if explore else 0.0),
        )
        return (np.asarray(actions, dtype=np.int32),
                np.asarray(weights, dtype=np.float32),
                np.asarray(root_values, dtype=np.float32))

    def search(self, observation: np.ndarray, *, explore: bool = True,
               invalid_actions: tuple[bool, ...] | None = None) -> tuple[int, np.ndarray, float]:
        """Return the chosen action, the visit-count policy, and the root search value."""
        actions, weights, root_values = self.search_batch(
            np.asarray(observation)[None, :], explore=explore, invalid_actions=invalid_actions)
        return int(actions[0]), weights[0], float(root_values[0])

    @partial(jax.jit, static_argnums=(0,))
    def _update(self, params, optimizer_state, observations, actions, rewards, dones, policies,
                next_observations, values, mask):
        """One optimizer step over a batch of ``unroll_steps``-long sequences.

        ``observations`` is [batch, observation]; every other array carries a leading
        [batch, steps] so the dynamics network is trained to stay accurate several steps
        out rather than only one.
        """
        steps = actions.shape[1]

        def loss_fn(params):
            latent = self.network.apply(params, observations, method=self.network.representation)
            policy_logits, value_prediction = self.network.apply(
                params, latent, method=self.network.prediction)
            policy_loss = -jnp.mean(
                jnp.sum(policies[:, 0] * jax.nn.log_softmax(policy_logits), axis=-1))
            value_loss = jnp.mean((value_prediction - scalar_transform(values[:, 0])) ** 2)
            reward_loss = jnp.zeros(())
            consistency_loss = jnp.zeros(())
            continuation_loss = jnp.zeros(())

            for step in range(steps):
                latent, reward_prediction, continuation_prediction = self.network.apply(
                    params, latent, actions[:, step], method=self.network.dynamics)
                # Halve the gradient flowing into the recurrent path, as in MuZero.
                latent = scale_gradient(latent, 0.5)
                policy_logits, value_prediction = self.network.apply(
                    params, latent, method=self.network.prediction)
                target_latent = jax.lax.stop_gradient(self.network.apply(
                    params, next_observations[:, step], method=self.network.representation))
                weights = mask[:, step]
                total_weight = jnp.maximum(jnp.sum(weights), 1.0)
                future_weights = weights * (1.0 - dones[:, step])
                future_total = jnp.maximum(jnp.sum(future_weights), 1.0)
                reward_loss += jnp.sum(
                    weights * (reward_prediction - scalar_transform(rewards[:, step])) ** 2
                ) / total_weight
                continuation_loss += jnp.sum(
                    weights * optax.sigmoid_binary_cross_entropy(
                        continuation_prediction, 1.0 - dones[:, step])
                ) / total_weight
                # Steps past the end of an episode are masked out rather than trained on.
                policy_loss += jnp.sum(
                    future_weights * -jnp.sum(
                        policies[:, step + 1] * jax.nn.log_softmax(policy_logits), axis=-1)
                ) / future_total
                value_loss += jnp.sum(
                    future_weights
                    * (value_prediction - scalar_transform(values[:, step + 1])) ** 2
                ) / future_total
                consistency_loss += jnp.sum(
                    weights * jnp.mean((latent - target_latent) ** 2, axis=-1)
                ) / total_weight

            scale = 1.0 / steps
            policy_loss *= scale
            value_loss *= scale
            reward_loss *= scale
            consistency_loss *= scale
            continuation_loss *= scale
            total = (policy_loss + self.settings.value_loss_weight * value_loss
                     + reward_loss + 0.25 * consistency_loss + continuation_loss)
            return total, (policy_loss, value_loss, reward_loss, consistency_loss,
                           continuation_loss)

        (loss, parts), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss, parts

    def train_batch(self, sequences: list[list[Transition]]) -> dict[str, float]:
        """Train on sequences produced by :meth:`ReplayBuffer.sample`."""
        steps = self.settings.unroll_steps
        uniform = np.full(self.num_actions, 1 / self.num_actions, dtype=np.float32)
        batch = len(sequences)
        observations = np.stack([sequence[0].observation for sequence in sequences])
        actions = np.zeros((batch, steps), dtype=np.int32)
        rewards = np.zeros((batch, steps), dtype=np.float32)
        dones = np.zeros((batch, steps), dtype=np.float32)
        values = np.zeros((batch, steps + 1), dtype=np.float32)
        policies = np.tile(uniform, (batch, steps + 1, 1))
        next_observations = np.zeros(
            (batch, steps, self.observation_size), dtype=np.float32)
        mask = np.zeros((batch, steps), dtype=np.float32)

        for row, sequence in enumerate(sequences):
            values[row, 0] = sequence[0].value_target
            policies[row, 0] = sequence[0].policy
            for step in range(steps):
                if step >= len(sequence):
                    break  # ran off the end of the episode; stays masked out
                transition = sequence[step]
                actions[row, step] = transition.action
                rewards[row, step] = transition.reward
                dones[row, step] = float(transition.done)
                next_observations[row, step] = transition.next_observation
                mask[row, step] = 1.0
                following = sequence[step + 1] if step + 1 < len(sequence) else None
                if following is not None:
                    values[row, step + 1] = following.value_target
                    policies[row, step + 1] = following.policy

        self.params, self.optimizer_state, loss, parts = self._update(
            self.params, self.optimizer_state,
            jnp.asarray(observations), jnp.asarray(actions), jnp.asarray(rewards),
            jnp.asarray(dones),
            jnp.asarray(policies), jnp.asarray(next_observations), jnp.asarray(values),
            jnp.asarray(mask))
        self.training_steps += 1
        names = ("policy_loss", "value_loss", "reward_loss", "consistency_loss",
                 "continuation_loss")
        result = {"loss": float(loss)}
        result.update({name: float(value) for name, value in zip(names, parts)})
        return result

    def save(self, directory: str | Path, *, episodes: int = 0,
             observation_layout: dict | None = None) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "params.msgpack").write_bytes(serialization.to_bytes(self.params))
        (destination / "optimizer.msgpack").write_bytes(serialization.to_bytes(self.optimizer_state))
        metadata = {
            "format_version": 2,
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            "training_steps": self.training_steps,
            "episodes": episodes,
            "settings": asdict(self.settings),
            "continuation_head": self.predict_continuation,
            # Dense and strided history can produce the same observation size, so the split
            # is recorded rather than guessed from the shape.
            "observation_layout": observation_layout or {},
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, directory: str | Path, *, seed: int = 0) -> "MuZeroAgent":
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        agent = cls(
            metadata["observation_size"], metadata["num_actions"],
            MuZeroSettings(**metadata["settings"]), seed=seed,
            predict_continuation=bool(metadata.get(
                "continuation_head", metadata.get("format_version", 1) >= 2)),
        )
        agent.params = serialization.from_bytes(agent.params, (source / "params.msgpack").read_bytes())
        agent.optimizer_state = agent.optimizer.init(agent.params)
        optimizer_path = source / "optimizer.msgpack"
        if optimizer_path.exists():
            agent.optimizer_state = serialization.from_bytes(
                agent.optimizer_state, optimizer_path.read_bytes())
        agent.training_steps = metadata["training_steps"]
        agent.episodes = metadata.get("episodes", 0)
        return agent


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self._by_stage: dict[int, list[Transition]] = {}
        self.rng = random.Random(seed)

    @property
    def items(self) -> list[Transition]:
        """Flattened compatibility view; training uses the stage-partitioned storage."""
        return [transition for stage in sorted(self._by_stage)
                for transition in self._by_stage[stage]]

    def __len__(self) -> int:
        return sum(len(items) for items in self._by_stage.values())

    def clear(self) -> None:
        self._by_stage.clear()

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self._by_stage.setdefault(int(transition.stage), []).append(transition)
        self._trim()

    def _trim(self) -> None:
        """Reserve half of replay for mastered stages and half for the newest stage.

        Global FIFO made the current, longest lesson occupy more than 80% of replay even
        though 25% of episodes rehearsed older lessons.  Partitioning also makes trimming
        cheap: old entries are removed from one stage list instead of rebuilding 50k items.
        """
        if len(self) <= self.capacity:
            return
        stages = sorted(self._by_stage)
        current = stages[-1]
        previous = stages[:-1]
        if not previous:
            del self._by_stage[current][:-self.capacity]
            return
        prior_budget = self.capacity // 2
        targets = {current: self.capacity - prior_budget}
        for index, stage in enumerate(previous):
            targets[stage] = prior_budget // len(previous) + (
                index < prior_budget % len(previous))

        # A stage with little data gives its unused reservation to stages that can use it.
        unused = sum(max(0, targets[stage] - len(self._by_stage[stage])) for stage in stages)
        for stage in [current, *reversed(previous)]:
            if not unused:
                break
            extra = max(0, len(self._by_stage[stage]) - targets[stage])
            granted = min(unused, extra)
            targets[stage] += granted
            unused -= granted
        for stage in stages:
            excess = len(self._by_stage[stage]) - targets[stage]
            if excess > 0:
                del self._by_stage[stage][:excess]
        self._by_stage = {stage: items for stage, items in self._by_stage.items() if items}

    def sample(self, size: int, unroll_steps: int = 1,
               successful_fraction: float = 0.25, current_stage: int | None = None,
               current_fraction: float = 0.60) -> list[list[Transition]]:
        """Sample sequences that never run across an episode boundary.

        Sequences near the end of an episode come back short; the trainer masks the
        missing steps instead of unrolling into an unrelated episode.
        """
        if not self._by_stage:
            return []
        count = min(size, len(self))
        available_stages = sorted(self._by_stage)
        current = (max(available_stages) if current_stage is None else current_stage)
        if current not in self._by_stage:
            current = max(available_stages)
        previous = [stage for stage in available_stages if stage != current]
        quotas: dict[int, int] = {}
        if previous:
            quotas[current] = min(len(self._by_stage[current]), round(count * current_fraction))
            remaining = count - quotas[current]
            for index, stage in enumerate(previous):
                slots = remaining // len(previous) + (index < remaining % len(previous))
                quotas[stage] = min(len(self._by_stage[stage]), slots)
        else:
            quotas[current] = count

        # If a young stage cannot fill its quota, use any available stage for the remainder.
        assigned = sum(quotas.values())
        cursor = 0
        while assigned < count:
            candidates = [stage for stage in available_stages
                          if quotas.get(stage, 0) < len(self._by_stage[stage])]
            if not candidates:
                break
            stage = candidates[cursor % len(candidates)]
            quotas[stage] = quotas.get(stage, 0) + 1
            assigned += 1
            cursor += 1

        starts: list[tuple[int, int]] = []
        for stage, wanted in quotas.items():
            items = self._by_stage[stage]
            successful = [index for index, transition in enumerate(items)
                          if transition.successful]
            wanted_successes = min(len(successful), round(wanted * successful_fraction))
            indices = self.rng.sample(successful, wanted_successes)
            indices.extend(self.rng.randrange(len(items))
                           for _ in range(wanted - wanted_successes))
            starts.extend((stage, index) for index in indices)
        self.rng.shuffle(starts)
        sequences = []
        for stage, start in starts:
            items = self._by_stage[stage]
            episode = items[start].episode_id
            end = start
            limit = min(start + unroll_steps + 1, len(items))
            while end < limit and items[end].episode_id == episode:
                end += 1
            sequences.append(items[start:end])
        return sequences

    def save(self, path: str | Path) -> Path:
        """Persist the buffer so ``--resume`` continues instead of cold-starting."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        items = self.items
        if not items:
            return destination
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as stream:  # a file object avoids np.savez appending ".npz"
            np.savez(
                stream,
                observation=np.stack([t.observation for t in items]),
                action=np.asarray([t.action for t in items], dtype=np.int32),
                reward=np.asarray([t.reward for t in items], dtype=np.float32),
                policy=np.stack([t.policy for t in items]),
                next_observation=np.stack([t.next_observation for t in items]),
                done=np.asarray([t.done for t in items], dtype=bool),
                value_target=np.asarray(
                    [t.value_target for t in items], dtype=np.float32),
                next_policy=np.stack([
                    t.next_policy if t.next_policy is not None else t.policy
                    for t in items]),
                search_value=np.asarray(
                    [t.search_value for t in items], dtype=np.float32),
                episode_id=np.asarray([t.episode_id for t in items], dtype=np.int64),
                successful=np.asarray([t.successful for t in items], dtype=bool),
                stage=np.asarray([t.stage for t in items], dtype=np.int16),
            )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path, *, observation_size: int | None = None,
             num_actions: int | None = None,
             episode_stages: dict[int, int] | None = None) -> int:
        """Restore a saved buffer, ignoring one whose shapes no longer match the config."""
        source = Path(path)
        if not source.exists():
            return 0
        with np.load(source) as data:
            observations = data["observation"]
            actions = data["action"]
            rewards = data["reward"]
            policies = data["policy"]
            next_observations = data["next_observation"]
            dones = data["done"]
            value_targets = data["value_target"]
            next_policies = data["next_policy"]
            if observation_size is not None and observations.shape[1] != observation_size:
                return 0
            if num_actions is not None and policies.shape[1] != num_actions:
                return 0
            count = len(actions)
            # Buffers written before search values and episode ids were stored still load;
            # each transition then forms its own sequence, which the trainer masks correctly.
            search_values = data["search_value"] if "search_value" in data else np.zeros(count)
            episode_ids = (data["episode_id"] if "episode_id" in data
                           else np.arange(count, dtype=np.int64))
            successes = data["successful"] if "successful" in data else np.zeros(count, bool)
            stages = (data["stage"] if "stage" in data else np.asarray([
                (episode_stages or {}).get(int(episode_id), 0) for episode_id in episode_ids
            ], dtype=np.int16))
            restored = [
                Transition(
                    observations[index], int(actions[index]),
                    float(rewards[index]), policies[index],
                    next_observations[index], bool(dones[index]),
                    float(value_targets[index]), next_policies[index],
                    float(search_values[index]), int(episode_ids[index]),
                    bool(successes[index]), int(stages[index]),
                )
                for index in range(count)
            ]
        self._by_stage.clear()
        self.extend(restored)
        return len(self)


def finish_episode(transitions: list[Transition], discount: float, *, n_step: int = 10,
                   episode_id: int = 0, successful: bool = False, stage: int = 0) -> None:
    """Assign n-step value targets bootstrapped off stored search values.

    Discounting the whole remaining episode instead (a full Monte-Carlo return) makes the
    targets enormously high variance over long episodes, which shows up as a value loss
    that grows while the policy loss stalls.
    """
    count = len(transitions)
    for index, transition in enumerate(transitions):
        bootstrap = min(index + n_step, count)
        value = 0.0
        for offset in range(index, bootstrap):
            value += transitions[offset].reward * discount ** (offset - index)
        if bootstrap < count:
            # Episode continues past the horizon, so bootstrap from that step's search value.
            value += transitions[bootstrap].search_value * discount ** (bootstrap - index)
        transition.value_target = value
        transition.episode_id = episode_id
        transition.successful = successful
        transition.stage = stage
        transition.next_policy = (
            transitions[index + 1].policy if index + 1 < count else
            np.full_like(transition.policy, 1 / len(transition.policy))
        )
