"""Base environment for combinatorial semi-bandits."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class CombBanditEnv(ABC):
    """Abstract base for combinatorial semi-bandit environments.

    Ground set [d], superarm size m. At each round the learner selects S ⊆ [d]
    with |S| = m, observes per-arm rewards X_{i,t} ~ ν_i for i in S.
    """

    def __init__(self, d: int, m: int, seed: int = 0):
        self.d = d
        self.m = m
        self.rng = np.random.RandomState(seed)
        self._means: np.ndarray | None = None
        self._optimal_set: list[int] | None = None

    @property
    def means(self) -> np.ndarray:
        if self._means is None:
            raise RuntimeError("Environment not initialized. Call reset().")
        return self._means

    @property
    def optimal_set(self) -> list[int]:
        if self._optimal_set is None:
            self._optimal_set = list(np.argsort(self.means)[-self.m:])
        return self._optimal_set

    @property
    def optimal_reward(self) -> float:
        return float(self.means[self.optimal_set].sum())

    @abstractmethod
    def reset(self) -> None:
        """Initialize or reset the environment."""

    def pull(self, selected_set: list[int]) -> dict[int, float]:
        """Pull arms in selected_set, return per-arm rewards."""
        assert len(selected_set) == self.m, f"Must select exactly {self.m} arms"
        assert all(0 <= a < self.d for a in selected_set), "Invalid arm index"
        rewards = {}
        for arm in selected_set:
            rewards[arm] = self._sample_reward(arm)
        return rewards

    @abstractmethod
    def _sample_reward(self, arm: int) -> float:
        """Sample a single reward from arm's distribution."""

    def get_arm_metadata(self) -> list[dict]:
        """Return metadata for each arm (used in LLM prompts)."""
        return [{"arm_id": i} for i in range(self.d)]

    def instantaneous_regret(self, selected_set: list[int]) -> float:
        return self.optimal_reward - float(self.means[selected_set].sum())
