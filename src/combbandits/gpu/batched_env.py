"""GPU-batched combinatorial semi-bandit environments.

All operations are vectorized over n_seeds. Rewards are sampled for all
seeds simultaneously via torch.bernoulli.
"""
from __future__ import annotations

import torch
import numpy as np


class BatchedSyntheticBernoulliEnv:
    """Batched Bernoulli semi-bandit: all n_seeds share the same arm means."""

    def __init__(self, d: int, m: int, n_seeds: int, gap_type: str = "uniform",
                 delta_min: float = 0.05, base_seed: int = 0, device: torch.device | None = None):
        self.d = d
        self.m = m
        self.n_seeds = n_seeds
        self.gap_type = gap_type
        self.delta_min = delta_min
        self.base_seed = base_seed
        self.device = device or torch.device("cpu")

        self.means: torch.Tensor | None = None       # (d,)
        self.optimal_set: torch.Tensor | None = None  # (m,) indices
        self.optimal_reward: float = 0.0

    def reset(self) -> None:
        rng = np.random.RandomState(self.base_seed)

        if self.gap_type == "uniform":
            means_np = rng.uniform(0.1, 0.5, size=self.d)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.5 + self.delta_min
        elif self.gap_type == "hard":
            means_np = np.full(self.d, 0.5)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.5 + self.delta_min
        elif self.gap_type == "graded":
            means_np = np.linspace(0.1, 0.5 + self.delta_min, self.d)
            rng.shuffle(means_np)
        elif self.gap_type == "clustered":
            means_np = np.full(self.d, 0.3)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.7
            means_np += rng.normal(0, 0.02, size=self.d)
            means_np = np.clip(means_np, 0.01, 0.99)
        else:
            raise ValueError(f"Unknown gap_type: {self.gap_type}")

        self.means = torch.tensor(means_np, dtype=torch.float32, device=self.device)
        opt_idx = torch.argsort(self.means, descending=True)[:self.m]
        self.optimal_set = opt_idx
        self.optimal_reward = self.means[opt_idx].sum().item()

    def pull_batched(self, selected: torch.Tensor) -> torch.Tensor:
        """Sample rewards for all seeds simultaneously.

        Args:
            selected: (n_seeds, m) arm indices

        Returns:
            rewards: (n_seeds, m) Bernoulli samples
        """
        # Gather means for selected arms: (n_seeds, m)
        selected_means = self.means[selected]
        return torch.bernoulli(selected_means)

    def instantaneous_regret_batched(self, selected: torch.Tensor) -> torch.Tensor:
        """Compute regret for all seeds: (n_seeds,)."""
        selected_reward = self.means[selected].sum(dim=1)  # (n_seeds,)
        return self.optimal_reward - selected_reward

    def get_arm_metadata(self) -> list[dict]:
        return [{"arm_id": i, "label": f"arm_{i}", "category": f"group_{i % 5}"}
                for i in range(self.d)]
