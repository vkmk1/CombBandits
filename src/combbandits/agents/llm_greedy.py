"""LLM-Greedy: Always play the LLM suggestion with no UCB correction."""
from __future__ import annotations

from .base import Agent
from ..oracle.base import CLOBase


class LLMGreedyAgent(Agent):
    """Baseline that always plays the LLM's suggestion without exploration."""

    name = "llm_greedy"

    def __init__(self, d: int, m: int, oracle: CLOBase, arm_metadata: list[dict] | None = None, **kwargs):
        super().__init__(d, m)
        self.oracle = oracle
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]

    def select_arms(self) -> list[int]:
        context = {
            "round": self.t,
            "task_description": "Select the best items.",
            "history_summary": "",
        }
        response = self.oracle.query(context, self.arm_metadata, self.mu_hat)
        return response.suggested_set
