"""Corrupt-Robust CUCB (He et al., 2022 style)."""
from __future__ import annotations

import math
import numpy as np
from .base import Agent


class CorruptRobustCUCBAgent(Agent):
    """Adversarially robust CUCB using median-of-means estimator.

    Designed for reward corruption robustness (Lykouris et al., 2018 / He et al., 2022).
    Included as a baseline to test whether reward-corruption robustness
    helps in the action-corruption setting.
    """

    name = "corrupt_robust_cucb"

    def __init__(self, d: int, m: int, n_buckets: int = 8, **kwargs):
        super().__init__(d, m)
        self.n_buckets = n_buckets
        # Store raw reward sequences for median-of-means
        self._reward_history: list[list[float]] = [[] for _ in range(d)]

    def _median_of_means(self, arm: int) -> float:
        """Compute median-of-means estimator for an arm."""
        rewards = self._reward_history[arm]
        n = len(rewards)
        if n == 0:
            return 0.5  # Uninformative prior
        if n < self.n_buckets:
            return np.mean(rewards)

        bucket_size = n // self.n_buckets
        bucket_means = []
        for b in range(self.n_buckets):
            start = b * bucket_size
            end = start + bucket_size
            bucket_means.append(np.mean(rewards[start:end]))
        return float(np.median(bucket_means))

    def select_arms(self) -> list[int]:
        # Round-robin initialization
        if self.t < self.d:
            start = (self.t * self.m) % self.d
            arms = [(start + i) % self.d for i in range(self.m)]
            arms = list(dict.fromkeys(arms))
            while len(arms) < self.m:
                for a in range(self.d):
                    if a not in arms:
                        arms.append(a)
                        break
            return arms[:self.m]

        # UCB with median-of-means estimates
        scores = np.array([
            self._median_of_means(i) + math.sqrt(2 * math.log(self.t) / max(len(self._reward_history[i]), 1))
            for i in range(self.d)
        ])
        return list(np.argsort(scores)[-self.m:])

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)
        for arm_id, reward in rewards.items():
            self._reward_history[arm_id].append(reward)

    def reset(self):
        super().reset()
        self._reward_history = [[] for _ in range(self.d)]
