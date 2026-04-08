"""Warm-Start CTS: Thompson Sampling initialized with LLM-derived priors."""
from __future__ import annotations

import numpy as np
from .base import Agent
from ..oracle.base import CLOBase


class WarmStartCTSAgent(Agent):
    """Combinatorial Thompson Sampling with LLM-initialized Beta priors.

    Queries the LLM once at initialization to score each arm (0-10),
    then maps scores to Beta priors: Beta(score, 10-score).
    """

    name = "warm_start_cts"

    def __init__(
        self,
        d: int,
        m: int,
        oracle: CLOBase,
        prior_strength: float = 5.0,
        arm_metadata: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(d, m)
        self.oracle = oracle
        self.prior_strength = prior_strength
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]
        self._alphas = np.ones(d)
        self._betas = np.ones(d)
        self._initialized_prior = False

    def _init_prior_from_llm(self):
        """Query LLM once to get arm scores and set priors."""
        context = {
            "round": 0,
            "task_description": "Select the best items.",
            "history_summary": "",
        }
        response = self.oracle.query(context, self.arm_metadata, np.zeros(self.d))

        # Arms in the suggested set get high prior, others get low
        for arm in response.suggested_set:
            self._alphas[arm] = self.prior_strength
            self._betas[arm] = 1.0

        # Non-suggested arms get neutral-to-low prior
        for arm in range(self.d):
            if arm not in response.suggested_set:
                self._alphas[arm] = 1.0
                self._betas[arm] = self.prior_strength * 0.5

        self._initialized_prior = True

    def select_arms(self) -> list[int]:
        if not self._initialized_prior:
            self._init_prior_from_llm()

        samples = np.array([
            np.random.beta(self._alphas[i], self._betas[i])
            for i in range(self.d)
        ])
        return list(np.argsort(samples)[-self.m:])

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)
        for arm_id, reward in rewards.items():
            if reward > 0.5:
                self._alphas[arm_id] += 1
            else:
                self._betas[arm_id] += 1

    def reset(self):
        super().reset()
        self._alphas = np.ones(self.d)
        self._betas = np.ones(self.d)
        self._initialized_prior = False
