"""Core data structures for combinatorial semi-bandits."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ArmObservation:
    """Single arm observation at a round."""
    arm_id: int
    reward: float
    round_t: int


@dataclass
class RoundResult:
    """Result of a single round."""
    round_t: int
    selected_set: list[int]
    rewards: dict[int, float]  # arm_id -> reward
    total_reward: float
    instantaneous_regret: float
    cumulative_regret: float
    trust_score: Optional[float] = None
    kappa_t: Optional[float] = None
    rho_t: Optional[float] = None
    hedge_size: int = 0
    reduced_set_size: int = 0
    is_fallback: bool = False
    llm_suggestion: Optional[list[int]] = None


@dataclass
class TrialResult:
    """Result of a full trial (T rounds)."""
    agent_name: str
    env_name: str
    seed: int
    d: int
    m: int
    T: int
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def cumulative_regret(self) -> float:
        return self.rounds[-1].cumulative_regret if self.rounds else 0.0

    @property
    def regret_curve(self) -> np.ndarray:
        return np.array([r.cumulative_regret for r in self.rounds])


@dataclass
class ArmStats:
    """Running statistics for a single arm."""
    arm_id: int
    total_reward: float = 0.0
    n_pulls: int = 0
    mean: float = 0.0
    ucb_index: float = float('inf')

    def update(self, reward: float, t: int):
        self.total_reward += reward
        self.n_pulls += 1
        self.mean = self.total_reward / self.n_pulls
        if self.n_pulls > 0:
            self.ucb_index = self.mean + np.sqrt(2 * np.log(t) / self.n_pulls)
        else:
            self.ucb_index = float('inf')


@dataclass
class OracleResponse:
    """Response from the CLO."""
    suggested_set: list[int]
    re_query_sets: list[list[int]]
    consistency_score: float
    raw_response: Optional[str] = None
    tokens_used: int = 0
    cached: bool = False
