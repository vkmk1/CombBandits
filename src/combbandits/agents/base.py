"""Abstract base agent for combinatorial semi-bandits."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from ..types import ArmStats


class Agent(ABC):
    """Base class for combinatorial semi-bandit agents.

    All agents select a set of m arms from [d] at each round,
    observe per-arm rewards, and update internal state.
    """

    name: str = "base"

    def __init__(self, d: int, m: int, **kwargs):
        self.d = d
        self.m = m
        self.t = 0  # Current round
        self.arm_stats = [ArmStats(arm_id=i) for i in range(d)]

    @abstractmethod
    def select_arms(self) -> list[int]:
        """Select m arms to play this round. Returns list of arm indices."""

    def update(self, selected: list[int], rewards: dict[int, float]):
        """Update agent state after observing rewards."""
        self.t += 1
        for arm_id, reward in rewards.items():
            self.arm_stats[arm_id].update(reward, self.t)

    def reset(self):
        """Reset agent for a new trial."""
        self.t = 0
        self.arm_stats = [ArmStats(arm_id=i) for i in range(self.d)]

    @property
    def mu_hat(self) -> np.ndarray:
        """Current empirical means for all arms."""
        return np.array([s.mean for s in self.arm_stats])

    @property
    def n_pulls(self) -> np.ndarray:
        """Number of times each arm has been pulled."""
        return np.array([s.n_pulls for s in self.arm_stats])

    @property
    def ucb_indices(self) -> np.ndarray:
        """UCB indices for all arms."""
        return np.array([s.ucb_index for s in self.arm_stats])

    def top_m_by_ucb(self, candidates: list[int] | None = None) -> list[int]:
        """Select top m arms by UCB index from candidates."""
        if candidates is None:
            candidates = list(range(self.d))
        ucb = self.ucb_indices
        sorted_cands = sorted(candidates, key=lambda i: ucb[i], reverse=True)
        return sorted_cands[:self.m]
