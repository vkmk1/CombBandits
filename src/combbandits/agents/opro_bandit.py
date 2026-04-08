"""OPRO-Bandit: Iterative LLM prompt optimization (Yang et al., 2024)."""
from __future__ import annotations

import numpy as np
from .base import Agent
from ..oracle.base import CLOBase


class OPROBanditAgent(Agent):
    """OPRO-style iterative optimization for combinatorial bandits.

    Feeds running reward estimates back to the LLM in an optimization loop.
    The LLM iteratively refines its suggestion based on past performance.
    """

    name = "opro_bandit"

    def __init__(
        self,
        d: int,
        m: int,
        oracle: CLOBase,
        history_length: int = 10,
        arm_metadata: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(d, m)
        self.oracle = oracle
        self.history_length = history_length
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]
        self._past_selections: list[tuple[list[int], float]] = []

    def select_arms(self) -> list[int]:
        # Build optimization history for OPRO-style prompt
        recent = self._past_selections[-self.history_length:]
        history_lines = []
        for sel, reward in recent:
            history_lines.append(f"  Selected {sel} -> reward {reward:.3f}")

        context = {
            "round": self.t,
            "task_description": (
                "You are optimizing a selection. Each round you pick items and observe reward. "
                "Based on past results, pick the best set."
            ),
            "history_summary": (
                "Past selections and rewards:\n" + "\n".join(history_lines)
                if history_lines else "No history yet."
            ),
        }
        response = self.oracle.query(context, self.arm_metadata, self.mu_hat)
        return response.suggested_set

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)
        total = sum(rewards.values())
        self._past_selections.append((selected, total))

    def reset(self):
        super().reset()
        self._past_selections.clear()
