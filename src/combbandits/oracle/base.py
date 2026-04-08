"""Base Combinatorial LLM Oracle (CLO) interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from ..types import OracleResponse


class CLOBase(ABC):
    """Abstract base for Combinatorial LLM Oracles.

    Maps context to a distribution over feasible sets of size m from [d].
    Supports K independent re-queries for consistency estimation.
    """

    def __init__(self, d: int, m: int, K: int = 3):
        self.d = d
        self.m = m
        self.K = K
        self.total_queries = 0
        self.total_tokens = 0

    @abstractmethod
    def query(
        self,
        context: dict,
        arm_metadata: list[dict],
        mu_hat: np.ndarray,
    ) -> OracleResponse:
        """Query the oracle for a suggested set.

        Args:
            context: Round context (reward history, round number, etc.)
            arm_metadata: Per-arm metadata for prompt construction.
            mu_hat: Current empirical mean estimates for all arms.

        Returns:
            OracleResponse with primary suggestion and K re-query sets.
        """

    def compute_consistency(self, re_query_sets: list[list[int]]) -> float:
        """Compute κ_t = |intersection of all K sets| / m."""
        if len(re_query_sets) < 2:
            return 1.0
        common = set(re_query_sets[0])
        for s in re_query_sets[1:]:
            common = common.intersection(s)
        return len(common) / self.m
