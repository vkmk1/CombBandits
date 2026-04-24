"""The post-analysis masterpieces: B2+ (engineering) + CORR-CTS (novel math).

Design principles from the 5-config × 5-seed rigorous experiment:
- B2 wins (p<0.0001) BECAUSE periodic queries + decaying weight compound good info
- B2 fails when LLM per-arm predictions are wrong (67.5% of p<0.5 → pseudo-failure storm)

M1. B2+ (safe-B2):
   Keep B2's winning framework. Remove the failure mode by:
   - Only inject POSITIVE signals (pseudo-successes only)
   - Gate injection: only arms where LLM confidence > 0.55
   - Cap per-arm total injection at sqrt(T)/m

M2. CORR-CTS (correlated sampling with LLM clusters):
   Exploit the Mismatched Sampling Paradox (Atsidakou et al. 2024).
   Sample correlated posterior: arms in same cluster co-sampled.
   Probability all m optimal arms simultaneously optimistic: exp(-Ω(m)) → O(1).
   LLM only affects SAMPLING structure, never updates rewards directly.

M3. B2-CORR (synthesis): B2's periodic injection + CORR sampling.
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


# ─── M1. B2+ (Safe-B2) ────────────────────────────────────────────────────
class B2PlusSafe(CTSBase):
    """B2 with tail-risk removed: positive-only, confidence-gated, capped.

    Keeps B2's winning framework (periodic counterfactual queries with decay)
    but only ever adds pseudo-SUCCESSES, and only for arms LLM is confident
    about. Total injection capped per-arm.
    """
    name = "M1_b2_plus"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_interval: int = 150, base_weight: float = 6.0,
                 conf_threshold: float = 0.55, T_horizon: int = 1500, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval
        self.base_weight = base_weight
        self.conf_threshold = conf_threshold
        # Per-arm injection cap to preserve CTS regret rate
        self.per_arm_cap = max(3.0, math.sqrt(T_horizon) / m)
        self.injected_per_arm = np.zeros(d)

    def select_arms(self):
        if self.t == self.T_warmup:
            self._inject()
        elif (self.t > self.T_warmup and
              (self.t - self.T_warmup) % self.query_interval == 0):
            self._inject()
        return super().select_arms()

    def _inject(self):
        decay = 1.0 / (1 + self.t / 300)
        weight = self.base_weight * decay
        summary = f"t={self.t}, n_pulls_total={int(self.n_pulls.sum())}"
        preds = self.oracle.query_counterfactual(self.mu_hat.tolist(), summary)
        for aid, p in preds.items():
            # GATE 1: only inject positive signals
            if p < self.conf_threshold:
                continue
            # GATE 2: respect per-arm cap
            headroom = self.per_arm_cap - self.injected_per_arm[aid]
            if headroom <= 0:
                continue
            # POSITIVE ONLY: add to alphas only, never betas
            effective = min(weight * p, headroom)
            self.alphas[aid] += effective
            self.injected_per_arm[aid] += effective


# ─── M2. CORR-CTS (Correlated-Sampling CTS) ──────────────────────────────
class CorrelatedCTS(CTSBase):
    """LLM-informed correlated posterior sampling.

    Replaces CTS's independent Beta samples with correlated Gaussian
    approximation. LLM provides cluster structure → within-cluster samples
    are positively correlated. Exploits combinatorial structure via
    exponential variance reduction (Atsidakou et al. 2024).

    CRUCIAL: LLM only affects SAMPLING, never the Beta posterior itself.
    Beta updates remain from real rewards only → preserves CTS properties
    when LLM is wrong (clusters are noise, correlation averages out).
    """
    name = "M2_correlated_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 rho: float = 0.6, n_clusters: int = 8, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.rho = rho  # within-cluster correlation
        self.n_clusters = n_clusters
        self.cluster_of = np.zeros(d, dtype=int)
        self._clusters_built = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._clusters_built:
            self._clusters_built = True
            clusters = self.oracle.query_clusters(self.mu_hat.tolist(), self.n_clusters)
            for cidx, cluster in enumerate(clusters):
                for aid in cluster:
                    self.cluster_of[aid] = cidx

        if not self._clusters_built:
            return super().select_arms()

        return self._correlated_sample()

    def _correlated_sample(self):
        """Sample arms using correlated Gaussian approximation of Beta posterior.

        For each cluster k, draw a shared factor z_k ~ N(0,1).
        For each arm i in cluster k:
          arm_sample_i = mean_i + sigma_i * (sqrt(rho) * z_k + sqrt(1-rho) * eps_i)
        where eps_i ~ N(0,1) independent.

        This gives Corr(arm_i, arm_j) = rho if same cluster, 0 otherwise.
        """
        # Beta(α, β) mean & variance
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)

        # Draw cluster factors
        n_cluster_ids = int(self.cluster_of.max()) + 1
        z = self.np_rng.standard_normal(n_cluster_ids)
        eps = self.np_rng.standard_normal(self.d)

        # Correlated samples
        samples = means + sigmas * (
            math.sqrt(self.rho) * z[self.cluster_of] +
            math.sqrt(1 - self.rho) * eps
        )
        return list(np.argsort(samples)[::-1][:self.m])


# ─── M3. B2-CORR (Synthesis) ─────────────────────────────────────────────
class B2Correlated(CTSBase):
    """B2's winning periodic injection + CORR's correlated sampling.

    The full masterpiece:
    - Correlated sampling (M2) for combinatorial variance reduction
    - Positive-only periodic injection (M1) for fast convergence to top arms
    - Both use the SAME cluster structure → consistent signal from LLM

    If LLM clusters are accurate: exponential variance reduction + targeted boost.
    If LLM clusters are wrong: correlated noise averages out, injections capped.
    """
    name = "M3_b2_correlated"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_interval: int = 150, base_weight: float = 6.0,
                 conf_threshold: float = 0.55, rho: float = 0.6,
                 n_clusters: int = 8, T_horizon: int = 1500, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval
        self.base_weight = base_weight
        self.conf_threshold = conf_threshold
        self.rho = rho
        self.n_clusters = n_clusters
        self.per_arm_cap = max(3.0, math.sqrt(T_horizon) / m)
        self.injected_per_arm = np.zeros(d)
        self.cluster_of = np.zeros(d, dtype=int)
        self._clusters_built = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._clusters_built:
            self._clusters_built = True
            mu = self.mu_hat.tolist()
            clusters = self.oracle.query_clusters(mu, self.n_clusters)
            for cidx, cluster in enumerate(clusters):
                for aid in cluster:
                    self.cluster_of[aid] = cidx
            self._inject()
        elif (self.t > self.T_warmup and self._clusters_built and
              (self.t - self.T_warmup) % self.query_interval == 0):
            self._inject()

        if not self._clusters_built:
            return super().select_arms()
        return self._correlated_sample()

    def _inject(self):
        decay = 1.0 / (1 + self.t / 300)
        weight = self.base_weight * decay
        summary = f"t={self.t}, n_pulls_total={int(self.n_pulls.sum())}"
        preds = self.oracle.query_counterfactual(self.mu_hat.tolist(), summary)
        for aid, p in preds.items():
            if p < self.conf_threshold:
                continue
            headroom = self.per_arm_cap - self.injected_per_arm[aid]
            if headroom <= 0:
                continue
            effective = min(weight * p, headroom)
            self.alphas[aid] += effective
            self.injected_per_arm[aid] += effective

    def _correlated_sample(self):
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        n_cluster_ids = int(self.cluster_of.max()) + 1
        z = self.np_rng.standard_normal(n_cluster_ids)
        eps = self.np_rng.standard_normal(self.d)
        samples = means + sigmas * (
            math.sqrt(self.rho) * z[self.cluster_of] +
            math.sqrt(1 - self.rho) * eps
        )
        return list(np.argsort(samples)[::-1][:self.m])


# ─── M4. B2-PATCH (minimal) — just the positive-only fix on B2 ───────────
class B2Patched(CTSBase):
    """Exact B2 but inject only positive signals.

    This is the surgical minimum-change variant of B2 — we're identifying
    whether the 'positive-only' change alone explains the gains.
    """
    name = "M4_b2_patched"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_interval: int = 200, obs_weight: float = 6.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval
        self.obs_weight = obs_weight

    def select_arms(self):
        if (self.t == self.T_warmup or
            (self.t > self.T_warmup and (self.t - self.T_warmup) % self.query_interval == 0)):
            self._inject()
        return super().select_arms()

    def _inject(self):
        decay = 1.0 / (1 + self.t / 300)
        weight = self.obs_weight * decay
        summary = f"t={self.t}, n_pulls_total={int(self.n_pulls.sum())}"
        preds = self.oracle.query_counterfactual(self.mu_hat.tolist(), summary)
        for aid, p in preds.items():
            if p > 0.5:
                # Positive only — add to alpha, never to beta
                self.alphas[aid] += weight * (p - 0.5) * 2  # scale 0-1


MASTERPIECE_ALGOS = {
    "M1_b2_plus": B2PlusSafe,
    "M2_correlated_cts": CorrelatedCTS,
    "M3_b2_correlated": B2Correlated,
    "M4_b2_patched": B2Patched,
}
