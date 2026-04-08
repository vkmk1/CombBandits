"""Simulated CLO for synthetic experiments with controllable corruption."""
from __future__ import annotations

import numpy as np
from .base import CLOBase
from ..types import OracleResponse


class SimulatedCLO(CLOBase):
    """Simulated oracle with configurable corruption model.

    Supports 4 corruption types matching the paper's experimental design:
    1. uniform: returns S* w.p. 1-ε, random otherwise
    2. adversarial: returns fixed S_bad w.p. ε, S* otherwise
    3. partial_overlap: returns set overlapping S* on (1-ε)m arms
    4. consistent_wrong: deterministically returns a fixed plausible suboptimal set
    """

    def __init__(
        self,
        d: int,
        m: int,
        optimal_set: list[int],
        arm_means: np.ndarray,
        corruption_type: str = "uniform",
        epsilon: float = 0.0,
        K: int = 3,
        seed: int = 0,
    ):
        super().__init__(d=d, m=m, K=K)
        self.optimal_set = list(optimal_set)
        self.arm_means = arm_means.copy()
        self.corruption_type = corruption_type
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)

        # Pre-compute adversarial / consistent-wrong sets
        suboptimal = [i for i in range(d) if i not in self.optimal_set]
        sorted_suboptimal = sorted(suboptimal, key=lambda i: arm_means[i], reverse=True)

        if corruption_type == "adversarial":
            # Worst m arms
            worst = sorted(suboptimal, key=lambda i: arm_means[i])[:m]
            self._bad_set = worst
        elif corruption_type == "consistent_wrong":
            # Best m suboptimal arms (plausible but wrong)
            self._bad_set = sorted_suboptimal[:m]
        else:
            self._bad_set = sorted_suboptimal[:m]

    def _generate_one_set(self) -> list[int]:
        """Generate a single oracle response."""
        if self.corruption_type == "uniform":
            if self.rng.random() < self.epsilon:
                # Random feasible set
                return list(self.rng.choice(self.d, size=self.m, replace=False))
            return list(self.optimal_set)

        elif self.corruption_type == "adversarial":
            if self.rng.random() < self.epsilon:
                return list(self._bad_set)
            return list(self.optimal_set)

        elif self.corruption_type == "partial_overlap":
            n_correct = max(0, int((1 - self.epsilon) * self.m))
            correct_arms = list(self.rng.choice(self.optimal_set, size=n_correct, replace=False))
            remaining_pool = [i for i in range(self.d) if i not in correct_arms]
            wrong_arms = list(
                self.rng.choice(remaining_pool, size=self.m - n_correct, replace=False)
            )
            return correct_arms + wrong_arms

        elif self.corruption_type == "consistent_wrong":
            # Always returns the same plausible suboptimal set
            return list(self._bad_set)

        raise ValueError(f"Unknown corruption_type: {self.corruption_type}")

    def query(
        self,
        context: dict,
        arm_metadata: list[dict],
        mu_hat: np.ndarray,
    ) -> OracleResponse:
        self.total_queries += self.K

        # Generate K sets
        all_sets = [self._generate_one_set() for _ in range(self.K)]
        primary = all_sets[0]
        kappa = self.compute_consistency(all_sets)

        return OracleResponse(
            suggested_set=primary,
            re_query_sets=all_sets,
            consistency_score=kappa,
            tokens_used=0,
            cached=False,
        )
