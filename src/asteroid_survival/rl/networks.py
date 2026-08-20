"""A permutation-invariant encoder for the asteroid and projectile blocks.

The observation is a flat vector, but two of its three blocks are *sets*: 26 asteroid slots
and 8 projectile slots, each sorted by distance and re-sorted on every decision. A plain MLP
over that layout has to learn a function that is stable under constant permutation of its own
inputs, and it has to learn it 26 times over, once per slot.

Measured on a trained policy at round 17: **19.8% of occupied asteroid slots change contents
between consecutive decisions**, fifteen times a second, and **51% of the block is
zero-padding** because only 12.7 of 26 slots are typically filled. Half the network's input
capacity is spent on nothing, and the half that is not keeps being permuted underneath it.

This encoder runs one small shared network over each entity and pools the results, so slot
order cannot matter by construction, padding costs nothing, and one set of weights is learned
for "what an asteroid means" rather than 26 near-duplicates. Mean and max pooling together
because they answer different questions: the mean carries how crowded the field is, the max
carries how bad the worst threat is.
"""
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SetFeaturesExtractor(BaseFeaturesExtractor):
    """Encode ship features directly, and the asteroid/projectile sets by pooling."""

    def __init__(self, observation_space: spaces.Box, *, ship_features: int,
                 asteroid_slots: int, asteroid_features: int,
                 projectile_slots: int, projectile_features: int,
                 teammate_slots: int = 0, teammate_features: int = 0,
                 global_features: int = 0,
                 entity_width: int = 96, embedding: int = 128):
        expected = (ship_features + asteroid_slots * asteroid_features
                    + projectile_slots * projectile_features
                    + teammate_slots * teammate_features + global_features)
        if observation_space.shape[0] != expected:
            raise ValueError(
                f"observation is {observation_space.shape[0]} wide but the block layout "
                f"sums to {expected}; the extractor and the encoder have drifted apart")
        # Each pooled set contributes mean and max, so 2 * embedding apiece.
        pooled_sets = 2 + (1 if teammate_slots else 0)
        super().__init__(observation_space,
                         features_dim=ship_features + global_features
                         + pooled_sets * 2 * embedding)
        self.ship_features = ship_features
        self.asteroid_slots, self.asteroid_features = asteroid_slots, asteroid_features
        self.projectile_slots, self.projectile_features = projectile_slots, projectile_features
        self.teammate_slots, self.teammate_features = teammate_slots, teammate_features
        self.global_features = global_features

        def entity_net(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, entity_width), nn.ReLU(),
                nn.Linear(entity_width, embedding), nn.ReLU())

        self.asteroid_net = entity_net(asteroid_features)
        self.projectile_net = entity_net(projectile_features)
        self.teammate_net = entity_net(teammate_features) if teammate_slots else None

    @staticmethod
    def _pool(encoded: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Mean and max over the present entities, with empty slots excluded.

        Empty slots are all-zero in the observation, so they would drag the mean toward zero
        and make "few asteroids" look like "many faint ones". Masking keeps the pooled value
        a statement about the entities that exist.
        """
        mask = present.unsqueeze(-1)
        counted = mask.sum(dim=1).clamp(min=1.0)
        mean = (encoded * mask).sum(dim=1) / counted
        largest = encoded.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        return torch.cat([mean, torch.nan_to_num(largest, neginf=0.0)], dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        at = 0
        ship = observations[:, at:at + self.ship_features]; at += self.ship_features

        width = self.asteroid_slots * self.asteroid_features
        asteroids = observations[:, at:at + width].view(
            -1, self.asteroid_slots, self.asteroid_features); at += width
        # The first feature of every entity slot is its presence flag.
        pooled = [self._pool(self.asteroid_net(asteroids), asteroids[..., 0])]

        width = self.projectile_slots * self.projectile_features
        projectiles = observations[:, at:at + width].view(
            -1, self.projectile_slots, self.projectile_features); at += width
        pooled.append(self._pool(self.projectile_net(projectiles), projectiles[..., 0]))

        if self.teammate_net is not None:
            width = self.teammate_slots * self.teammate_features
            teammates = observations[:, at:at + width].view(
                -1, self.teammate_slots, self.teammate_features)
            at += width
            pooled.append(self._pool(self.teammate_net(teammates), teammates[..., 0]))

        globals_ = observations[:, at:at + self.global_features]
        return torch.cat([ship, globals_, *pooled], dim=-1)
