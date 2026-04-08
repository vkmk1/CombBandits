"""EXP4: Adversarial bandit with LLM as expert (Auer et al., 2002)."""
from __future__ import annotations

import math
import numpy as np
from .base import Agent
from ..oracle.base import CLOBase


class EXP4Agent(Agent):
    """EXP4-style adversarial algorithm treating LLM as one expert.

    Two experts: (1) LLM oracle, (2) uniform exploration.
    Maintains exponential weights over experts.
    """

    name = "exp4"

    def __init__(
        self,
        d: int,
        m: int,
        oracle: CLOBase,
        eta: float | None = None,
        arm_metadata: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(d, m)
        self.oracle = oracle
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]
        self.n_experts = 2  # LLM + uniform
        self.eta = eta  # Learning rate; set adaptively if None
        self._weights = np.ones(self.n_experts)
        self._expert_cum_reward = np.zeros(self.n_experts)

    def select_arms(self) -> list[int]:
        # Compute expert probabilities
        probs = self._weights / self._weights.sum()

        # Expert 0: LLM suggestion
        context = {
            "round": self.t,
            "task_description": "Select the best items.",
            "history_summary": f"Round {self.t}",
        }
        response = self.oracle.query(context, self.arm_metadata, self.mu_hat)
        llm_set = response.suggested_set

        # Expert 1: UCB-based (proxy for uniform exploration in combinatorial setting)
        ucb_set = self.top_m_by_ucb()

        # Mix: with prob p[0] play LLM, with prob p[1] play UCB
        if np.random.random() < probs[0]:
            self._last_expert = 0
            return llm_set
        else:
            self._last_expert = 1
            return ucb_set

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)
        total_reward = sum(rewards.values())

        # Update expert weights using EXP3 update
        eta = self.eta or math.sqrt(math.log(self.n_experts) / max(self.t, 1))
        probs = self._weights / self._weights.sum()

        # Importance-weighted reward estimate for the chosen expert
        estimated_reward = total_reward / max(probs[self._last_expert], 1e-8)
        self._weights[self._last_expert] *= math.exp(eta * estimated_reward / self.m)

        # Normalize to prevent overflow
        self._weights /= self._weights.max()

    def reset(self):
        super().reset()
        self._weights = np.ones(self.n_experts)
        self._expert_cum_reward = np.zeros(self.n_experts)
        self._last_expert = 0
