"""CUCB: Standard Combinatorial UCB (Chen et al., 2013)."""
from __future__ import annotations

import numpy as np
from .base import Agent


class CUCBAgent(Agent):
    """Standard CUCB on the full ground set [d].

    Baseline with no LLM guidance. Achieves O~(√(mdT)) regret.
    """

    name = "cucb"

    def __init__(self, d: int, m: int, **kwargs):
        super().__init__(d, m)

    def select_arms(self) -> list[int]:
        # During first d/m rounds, do round-robin to initialize
        if self.t < self.d:
            start = (self.t * self.m) % self.d
            arms = [(start + i) % self.d for i in range(self.m)]
            # Deduplicate
            arms = list(dict.fromkeys(arms))
            while len(arms) < self.m:
                for a in range(self.d):
                    if a not in arms:
                        arms.append(a)
                        break
            return arms[:self.m]

        return self.top_m_by_ucb()
