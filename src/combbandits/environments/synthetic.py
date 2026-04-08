"""Synthetic Bernoulli combinatorial semi-bandit environments."""
from __future__ import annotations

import numpy as np
from .base import CombBanditEnv


class SyntheticBernoulliEnv(CombBanditEnv):
    """Stochastic combinatorial semi-bandit with Bernoulli arms.

    Arms have means drawn from a specified structure. Supports multiple
    gap structures for validating theoretical predictions.
    """

    def __init__(
        self,
        d: int = 100,
        m: int = 10,
        gap_type: str = "uniform",
        delta_min: float = 0.05,
        seed: int = 0,
    ):
        super().__init__(d=d, m=m, seed=seed)
        self.gap_type = gap_type
        self.delta_min = delta_min

    def reset(self) -> None:
        if self.gap_type == "uniform":
            # Optimal arms have mean 0.5 + delta_min, rest uniform in [0.1, 0.5]
            self._means = self.rng.uniform(0.1, 0.5, size=self.d)
            top_arms = self.rng.choice(self.d, size=self.m, replace=False)
            self._means[top_arms] = 0.5 + self.delta_min
        elif self.gap_type == "graded":
            # Arms have linearly spaced means; gap varies per arm
            self._means = np.linspace(0.1, 0.5 + self.delta_min, self.d)
            self.rng.shuffle(self._means)
        elif self.gap_type == "clustered":
            # Two clusters: optimal (mean ~0.7) and suboptimal (mean ~0.3)
            self._means = np.full(self.d, 0.3)
            top_arms = self.rng.choice(self.d, size=self.m, replace=False)
            self._means[top_arms] = 0.7
            # Add noise within clusters
            self._means += self.rng.normal(0, 0.02, size=self.d)
            self._means = np.clip(self._means, 0.01, 0.99)
        elif self.gap_type == "hard":
            # All arms close to each other; small gap
            self._means = np.full(self.d, 0.5)
            top_arms = self.rng.choice(self.d, size=self.m, replace=False)
            self._means[top_arms] = 0.5 + self.delta_min
        else:
            raise ValueError(f"Unknown gap_type: {self.gap_type}")

        self._optimal_set = None  # Force recomputation

    def _sample_reward(self, arm: int) -> float:
        return float(self.rng.binomial(1, self.means[arm]))

    def get_arm_metadata(self) -> list[dict]:
        return [
            {"arm_id": i, "label": f"arm_{i}", "category": f"group_{i % 5}"}
            for i in range(self.d)
        ]
