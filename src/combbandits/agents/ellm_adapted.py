"""ELLM-Adapted: LLM exploration bonus adapted to semi-bandit (Du et al., 2023)."""
from __future__ import annotations

import math
import numpy as np
from .base import Agent
from ..oracle.base import CLOBase


class ELLMAdaptedAgent(Agent):
    """ELLM-style exploration bonus for combinatorial semi-bandits.

    Uses LLM suggestions to define per-arm exploration bonuses.
    Arms suggested by the LLM get a bonus added to their UCB index.
    """

    name = "ellm_adapted"

    def __init__(
        self,
        d: int,
        m: int,
        oracle: CLOBase,
        bonus_scale: float = 0.5,
        bonus_decay: float = 0.99,
        arm_metadata: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(d, m)
        self.oracle = oracle
        self.bonus_scale = bonus_scale
        self.bonus_decay = bonus_decay
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]
        self._llm_bonus = np.zeros(d)

    def select_arms(self) -> list[int]:
        # Query LLM
        context = {
            "round": self.t,
            "task_description": "Select the best items.",
            "history_summary": f"Round {self.t}",
        }
        response = self.oracle.query(context, self.arm_metadata, self.mu_hat)

        # Decay existing bonuses and add new ones for suggested arms
        self._llm_bonus *= self.bonus_decay
        for arm in response.suggested_set:
            self._llm_bonus[arm] += self.bonus_scale

        # Select by UCB + LLM bonus
        scores = self.ucb_indices + self._llm_bonus
        return list(np.argsort(scores)[-self.m:])

    def reset(self):
        super().reset()
        self._llm_bonus = np.zeros(self.d)
