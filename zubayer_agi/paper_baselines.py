"""Implementations of recent LLM-bandit algorithms from NeurIPS/ICML 2024-2025 papers.

These are the baselines our winners must beat to be publishable.

Implemented:
- TS-LLM (Sun et al. 2025, "Multi-Armed Bandits Meet LLMs"):
  LLM direct selection with temperature decay
- LLM-Jump-Start (Austin et al. 2024):
  LLM generates synthetic pulls as warm-start data
- LLM-CUCB-AT (Kakaria, this repo's paper):
  Adaptive trust combining consistency κ and posterior validation ρ
- Calibration-Gated Pseudo-Obs (2024):
  Like our B1 but only injects when LLM is verified calibrated
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


# ─── TS-LLM (Sun et al. 2025, ICML/NeurIPS) ───────────────────────────────
class TSLLMBaseline(CTSBase):
    """Thompson Sampling with LLM-guided exploration (temperature decay).

    At each round: with probability p_LLM(t) = 1/(1+t/tau), use LLM; else CTS.
    LLM chooses m arms directly; CTS samples from posterior.
    Exploration via LLM decays over time as data accumulates.
    """
    name = "PAPER_ts_llm"

    def __init__(self, d, m, oracle: GPTOracle, tau: float = 100.0,
                 query_interval: int = 100, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.tau = tau
        self.query_interval = query_interval
        self._last_query_t = -1
        self._llm_picks: list[int] = []

    def select_arms(self):
        # Probability of using LLM decays with time
        p_llm = 1.0 / (1.0 + self.t / self.tau)

        # Refresh LLM picks periodically
        if self.t - self._last_query_t >= self.query_interval and self.t > 10:
            self._llm_picks = self.oracle.query_top_m(self.mu_hat.tolist())
            self._last_query_t = self.t
            # Also inject LLM picks as weak prior
            for aid in self._llm_picks:
                self.alphas[aid] += 1.0

        # Stochastic mixing
        if self.rng.random() < p_llm and self._llm_picks:
            # Use LLM's picks (with slight randomization)
            samples = self._sample()
            # Boost LLM picks
            mask = np.zeros(self.d)
            for aid in self._llm_picks[:self.m]:
                mask[aid] = 1.0
            combined = samples + 0.5 * mask
            return list(np.argsort(combined)[::-1][:self.m])
        else:
            return super().select_arms()


# ─── LLM-Jump-Start (Austin et al. 2024) ──────────────────────────────────
class LLMJumpStartBaseline(CTSBase):
    """Jump Starting Bandits with LLM-Generated Prior Knowledge.

    At t=0: LLM predicts mean reward per arm. These are treated as synthetic
    pulls — added as pseudo-observations with fixed strength. Then run CTS.
    """
    name = "PAPER_llm_jump_start"

    def __init__(self, d, m, oracle: GPTOracle, n_pseudo_pulls: int = 5, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.n_pseudo_pulls = n_pseudo_pulls
        self._done = False

    def select_arms(self):
        if not self._done:
            self._done = True
            scores = self.oracle.query_per_arm_scores([0.0] * self.d)
            for aid, s in scores.items():
                mean = s["mean"]
                # Treat as n_pseudo_pulls of synthetic pulls
                self.alphas[aid] += self.n_pseudo_pulls * mean
                self.betas[aid] += self.n_pseudo_pulls * (1 - mean)
        return super().select_arms()


# ─── LLM-CUCB-AT (Kakaria, this repo's main algorithm) ────────────────────
class LLMCUCBATBaseline(CTSBase):
    """Adaptive trust with consistency κ and posterior validation ρ.

    Phase 1: round-robin init for T_0 rounds.
    Phase 2: K re-queries per round to compute κ; trust τ=min(κ,ρ);
             reduced set = LLM picks + (hedge arms) by UCB;
             play top-m within reduced set.

    Simplified variant: we use CTS as the base (not UCB), but adaptive trust
    logic matches the paper.
    """
    name = "PAPER_llm_cucb_at"

    def __init__(self, d, m, oracle: GPTOracle, T_init: int = 30, K: int = 2,
                 h_max: int = 6, query_cooldown: int = 100, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_init = T_init
        self.K = K
        self.h_max = h_max
        self.query_cooldown = query_cooldown
        self._last_query_t = -query_cooldown
        self._cached_primary: list[int] = []
        self._cached_tau: float = 0.0

    def select_arms(self):
        # Phase 1: round-robin
        if self.t < self.T_init:
            start = (self.t * self.m) % self.d
            return [(start + i) % self.d for i in range(self.m)]

        # Phase 2: re-query LLM periodically (not every round — too expensive).
        # The paper's theory allows O(sqrt(T)) queries; cooldown matches that.
        should_requery = (self.t - self._last_query_t) >= self.query_cooldown
        if not should_requery and self._cached_primary:
            # Reuse cached LLM result
            primary = self._cached_primary
            tau = self._cached_tau
        else:
            mu = self.mu_hat.tolist()
            picks_sets = []
            for _ in range(self.K):
                picks = self.oracle.query_top_m(mu)
                picks_sets.append(set(picks[:self.m]))

            # Consistency κ = |intersection| / m
            if picks_sets:
                intersect = picks_sets[0].copy()
                for s in picks_sets[1:]:
                    intersect &= s
                kappa = len(intersect) / self.m
            else:
                kappa = 0.0

            primary = list(picks_sets[0]) if picks_sets else []

            # Posterior validation ρ = sum(mu[S_llm]) / max_possible
            max_sum = sum(sorted(mu, reverse=True)[:self.m])
            if max_sum > 0 and primary:
                llm_sum = sum(mu[a] for a in primary)
                rho = llm_sum / max_sum
            else:
                rho = 0.5

            tau = min(kappa, rho)
            self._cached_primary = primary
            self._cached_tau = tau
            self._last_query_t = self.t

        # Hedge: add top-h UCB arms to reduced set
        h = int(self.h_max * (1 - tau))
        cb = np.sqrt(2.0 * np.log(max(self.t, 1) + 1) / np.maximum(self.n_pulls, 1))
        ucb = self.mu_hat + cb
        hedge = [int(a) for a in np.argsort(ucb)[::-1] if int(a) not in primary][:h]

        reduced_set = list(primary) + hedge
        if not reduced_set or len(reduced_set) < self.m:
            # Fallback: CTS
            return super().select_arms()

        # CTS within reduced set
        samples = self._sample()
        mask = np.full(self.d, -np.inf)
        for a in reduced_set:
            mask[a] = samples[a]
        return list(np.argsort(mask)[::-1][:self.m])


# ─── Calibration-Gated Pseudo-Observations (2024) ─────────────────────────
class CalibrationGatedBaseline(CTSBase):
    """Pseudo-observations from LLM, but only injected when calibration verified.

    Hold out 3 'canary' arms with known stats. Verify LLM agrees on their
    ordering before trusting its outputs for real arms.
    """
    name = "PAPER_calibration_gated"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 40,
                 query_interval: int = 200, obs_weight: float = 5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval
        self.obs_weight = obs_weight

    def select_arms(self):
        if (self.t == self.T_warmup or
            (self.t > self.T_warmup and (self.t - self.T_warmup) % self.query_interval == 0)):
            self._gated_inject()
        return super().select_arms()

    def _gated_inject(self):
        mu = self.mu_hat.tolist()
        # Find 3 arms with highest n_pulls — we know their stats best
        canary_idx = list(np.argsort(self.n_pulls)[::-1][:3])
        canary_mu_order = sorted(canary_idx, key=lambda a: mu[a], reverse=True)

        # Get LLM's per-arm scores
        scores = self.oracle.query_per_arm_scores(mu)
        if not scores:
            return

        # Check LLM agreement on canary ordering
        llm_canary_order = sorted(canary_idx, key=lambda a: scores.get(a, {}).get("mean", 0.5), reverse=True)
        agrees = canary_mu_order == llm_canary_order

        if not agrees:
            # LLM is miscalibrated — skip injection
            return

        # Trust LLM: inject pseudo-observations
        decay = 1.0 / (1 + self.t / 300)
        weight = self.obs_weight * decay
        for aid, s in scores.items():
            self.alphas[aid] += weight * s["mean"]
            self.betas[aid] += weight * (1 - s["mean"])


PAPER_BASELINES = {
    "PAPER_ts_llm": TSLLMBaseline,
    "PAPER_llm_jump_start": LLMJumpStartBaseline,
    "PAPER_llm_cucb_at": LLMCUCBATBaseline,
    "PAPER_calibration_gated": CalibrationGatedBaseline,
}
