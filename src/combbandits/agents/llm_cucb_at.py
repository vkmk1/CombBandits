"""LLM-CUCB-AT: The main algorithm from the paper.

Combinatorial UCB with Adaptive LLM Trust. Uses composite trust score
(consistency + posterior validation) to adaptively hedge against unreliable
LLM action oracles.
"""
from __future__ import annotations

import math
from collections import deque
import numpy as np
from .base import Agent
from ..oracle.base import CLOBase
from ..types import OracleResponse


class LLMCUCBATAgent(Agent):
    """LLM-CUCB with Adaptive Trust (Algorithm 1 from the paper)."""

    name = "llm_cucb_at"

    def __init__(
        self,
        d: int,
        m: int,
        oracle: CLOBase,
        h_max: int | None = None,
        T_0: int | None = None,
        K: int = 3,
        fallback_window: int | None = None,
        rho_min: float = 0.5,
        arm_metadata: list[dict] | None = None,
        context_builder: callable | None = None,
        **kwargs,
    ):
        super().__init__(d, m)
        self.oracle = oracle
        self.h_max = h_max if h_max is not None else int(math.ceil(math.sqrt(d)))
        self.T_0 = T_0 if T_0 is not None else int(math.ceil(d * math.log(d) / m))
        self.K = K
        self.rho_min = rho_min
        self.arm_metadata = arm_metadata or [{"arm_id": i} for i in range(d)]
        self.context_builder = context_builder

        # State
        self._init_round = 0
        self._regret_tracker: deque[float] = deque(maxlen=200)
        self._force_fallback = False

        # Diagnostics
        self.trust_history: list[dict] = []
        self._last_oracle_response: OracleResponse | None = None

    def _in_init_phase(self) -> bool:
        return self.t < self.T_0

    def _build_context(self) -> dict:
        if self.context_builder is not None:
            return self.context_builder(self)
        top_5 = list(np.argsort(self.mu_hat)[-5:])
        top_5_means = [f"arm {a}: {self.mu_hat[a]:.3f}" for a in top_5]
        return {
            "round": self.t,
            "task_description": (
                "You are helping optimize a sequential item selection task. "
                "Each round, you select a subset of items and observe their rewards. "
                "Your goal is to find the highest-reward subset as quickly as possible."
            ),
            "history_summary": (
                f"Round {self.t}/{self.d * 5}. "
                f"Best items so far (by observed avg reward): {', '.join(top_5_means)}. "
                f"Total arms explored: {int((self.n_pulls > 0).sum())}/{self.d}."
            ),
        }

    def _compute_posterior_validation(self, suggested_set: list[int]) -> float:
        """Compute ρ_t = Σ_{i∈S^L} μ̂_i / max_S Σ_{i∈S} μ̂_i."""
        mu = self.mu_hat
        suggested_reward = sum(mu[i] for i in suggested_set)
        # Best set by empirical means
        best_arms = list(np.argsort(mu)[-self.m:])
        best_reward = sum(mu[i] for i in best_arms)
        if best_reward <= 0:
            return 1.0  # No information yet
        return suggested_reward / best_reward

    def _compute_hedge_size(self, tau: float) -> int:
        return int(math.ceil(self.h_max * (1 - tau)))

    def _build_reduced_set(self, suggested_set: list[int], h_t: int) -> list[int]:
        """Build D_t = S^L ∪ H^ucb_t."""
        if h_t == 0:
            return list(suggested_set)

        # UCB-ranked hedge: top h_t arms NOT in suggested set
        excluded = set(suggested_set)
        candidates = [i for i in range(self.d) if i not in excluded]
        ucb = self.ucb_indices
        sorted_cands = sorted(candidates, key=lambda i: ucb[i], reverse=True)
        hedge = sorted_cands[:h_t]

        return list(suggested_set) + hedge

    def _check_fallback(self) -> bool:
        """Check if corruption detection triggers a fallback."""
        if len(self._regret_tracker) < 5:
            return False
        window = min(int(math.ceil(math.sqrt(self.t))), len(self._regret_tracker))
        recent = list(self._regret_tracker)[-window:]
        avg_regret = np.mean(recent)
        threshold = math.sqrt(self.m * math.log(max(self.t, 2)) / window)
        return avg_regret > threshold

    def select_arms(self) -> list[int]:
        # Phase 1: Initialization (round-robin)
        if self._in_init_phase():
            start = (self.t * self.m) % self.d
            arms = [(start + i) % self.d for i in range(self.m)]
            arms = list(dict.fromkeys(arms))
            while len(arms) < self.m:
                for a in range(self.d):
                    if a not in arms:
                        arms.append(a)
                        break
            return arms[:self.m]

        # Phase 2: Forced fallback — play full CUCB on [d]
        if self._force_fallback:
            self._force_fallback = False
            return self.top_m_by_ucb()

        # Step 1: Query oracle
        context = self._build_context()
        response = self.oracle.query(context, self.arm_metadata, self.mu_hat)
        self._last_oracle_response = response

        # Step 2: Compute composite trust
        kappa_t = response.consistency_score
        rho_t = self._compute_posterior_validation(response.suggested_set)
        tau_t = min(kappa_t, rho_t)

        # Step 3: Build reduced set with hedge
        h_t = self._compute_hedge_size(tau_t)
        reduced_set = self._build_reduced_set(response.suggested_set, h_t)

        self.trust_history.append({
            "round": self.t,
            "kappa": kappa_t,
            "rho": rho_t,
            "tau": tau_t,
            "hedge_size": h_t,
            "reduced_set_size": len(reduced_set),
        })

        # Step 4: UCB play on reduced set
        return self.top_m_by_ucb(candidates=reduced_set)

    def update(self, selected: list[int], rewards: dict[int, float]):
        super().update(selected, rewards)

        if not self._in_init_phase():
            # Update regret tracker (empirical)
            selected_reward = sum(rewards.values())
            best_reward = sum(sorted(self.mu_hat, reverse=True)[:self.m])
            emp_regret = best_reward - selected_reward
            self._regret_tracker.append(emp_regret)

            # Check fallback trigger
            if self._check_fallback():
                self._force_fallback = True

    def reset(self):
        super().reset()
        self._init_round = 0
        self._regret_tracker.clear()
        self._force_fallback = False
        self.trust_history.clear()
        self._last_oracle_response = None

    def get_diagnostics(self) -> dict:
        return {
            "trust_history": self.trust_history,
            "total_oracle_queries": self.oracle.total_queries,
            "total_tokens": self.oracle.total_tokens,
        }
