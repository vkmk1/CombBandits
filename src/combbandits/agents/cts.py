"""CTS: Combinatorial Thompson Sampling (Wang & Chen, 2018)."""
from __future__ import annotations

import numpy as np
from .base import Agent


class CTSAgent(Agent):
    """Combinatorial Thompson Sampling with Beta priors for Bernoulli arms."""

    name = "cts"

    def __init__(self, d: int, m: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, **kwargs):
        super().__init__(d, m)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._alphas = np.full(d, prior_alpha)
        self._betas = np.full(d, prior_beta)

    def select_arms(self) -> list[int]:
        # Sample from Beta posterior for each arm
        samples = np.array([
            np.random.beta(self._alphas[i], self._betas[i])
            for i in range(self.d)
        ])
        return list(np.argsort(samples)[-self.m:])

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)
        for arm_id, reward in rewards.items():
            if reward > 0.5:  # Binary reward
                self._alphas[arm_id] += 1
            else:
                self._betas[arm_id] += 1

    def reset(self):
        super().reset()
        self._alphas = np.full(self.d, self.prior_alpha)
        self._betas = np.full(self.d, self.prior_beta)
