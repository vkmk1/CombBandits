"""Final algorithm suite: bug fixes + synthesis + critical ablations.

Three classes of algorithms:

1. **N3_info_min (FIXED)**: Previous version never triggered its LLM query.
   Fixed to ALWAYS trigger at least one query at t=T_warmup, then gate
   additional queries by actual uncertainty.

2. **N5_corr_full_robust (SYNTHESIS)**: N1 (full kernel covariance) +
   N4 (credibility gating). Combines the two winning approaches.

3. **RandomCorr_CTS (MANDATORY ABLATION)**: Same as our winning
   correlated-sampling algorithms, but with RANDOM clusters instead of
   LLM-derived clusters. If this matches N4/N1, LLM is not contributing.
   **Without this, our paper claim "LLM helps" is not defensible.**
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


# ─── N3 FIXED — always trigger at least once ────────────────────────────
class InfoMinCTSFixed(CTSBase):
    """Fixed version: always fires at t=T_warmup, then info-gates re-queries."""
    name = "N3_info_min_fixed"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 info_threshold_init: float = 0.02, query_cooldown: int = 200,
                 **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.info_threshold_init = info_threshold_init
        self.query_cooldown = query_cooldown
        self._first_fire = False
        self._last_query_t = -1

    def select_arms(self):
        # ALWAYS fire once at T_warmup (so algorithm actually uses LLM)
        if self.t == self.T_warmup and not self._first_fire:
            self._first_fire = True
            self._query_and_inject()
        # Then, periodic gated re-queries
        elif (self.t > self.T_warmup and
              self.t - self._last_query_t >= self.query_cooldown and
              self._first_fire):
            total = self.alphas + self.betas
            variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
            top_indices = np.argsort(self.mu_hat)[::-1][:min(2 * self.m, self.d)]
            avg_unc = variances[top_indices].mean()
            threshold = self.info_threshold_init / math.sqrt(self.t + 1)
            if avg_unc > threshold:
                self._query_and_inject()
        return super().select_arms()

    def _query_and_inject(self):
        picks = self.oracle.query_top_m(self.mu_hat.tolist())
        for aid in picks:
            self.alphas[aid] += 2.0
        self._last_query_t = self.t


# ─── N5 — CORR-Full + Robust credibility (SYNTHESIS) ────────────────────
class CorrFullRobust(CTSBase):
    """Synthesis: full kernel-derived covariance + credibility-gated interpolation.

    Combines:
    - N1's full pairwise kernel covariance (Matérn/RBF on cluster-rank space)
    - N4's credibility weight w_t that interpolates correlated vs independent sampling

    If LLM structure is good → full correlated sampling + exponential variance reduction.
    If LLM structure is adversarial → credibility decays → fallback to independent CTS.
    """
    name = "N5_corr_full_robust"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 n_clusters: int = 8, check_interval: int = 100, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.n_clusters = n_clusters
        self.check_interval = check_interval
        self.Sigma_corr = np.eye(d)
        self._cholesky = None
        self._built = False
        self._llm_suggested_arms = set()
        self.w_credibility = 1.0

    def _build_covariance(self):
        mu_list = self.mu_hat.tolist()
        clusters = self.oracle.query_clusters(mu_list, n_clusters=self.n_clusters)
        cluster_of = np.zeros(self.d, dtype=int)
        for cidx, cluster in enumerate(clusters):
            for aid in cluster:
                if 0 <= aid < self.d:
                    cluster_of[aid] = cidx
        n_cl = int(cluster_of.max()) + 1

        # Cluster rank based on mean mu_hat within cluster
        cluster_mu = np.zeros(n_cl)
        counts = np.zeros(n_cl)
        for aid in range(self.d):
            cluster_mu[cluster_of[aid]] += self.mu_hat[aid]
            counts[cluster_of[aid]] += 1
        cluster_mu /= np.maximum(counts, 1)
        # Sort clusters by mean; rank 0 = best
        cluster_order = np.argsort(cluster_mu)[::-1]
        rank_of = np.zeros(n_cl, dtype=int)
        for r, c in enumerate(cluster_order):
            rank_of[c] = r
        arm_rank = rank_of[cluster_of] / max(1, n_cl - 1)  # normalized 0-1

        # RBF kernel on rank space
        diff = arm_rank[:, None] - arm_rank[None, :]
        dist2 = diff ** 2
        Sigma = self.rho_max * np.exp(-dist2 / (2 * self.kernel_scale ** 2))
        np.fill_diagonal(Sigma, 1.0)
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, 1e-4, None)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.Sigma_corr = Sigma
        try:
            self._cholesky = np.linalg.cholesky(Sigma + 1e-3 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self._cholesky = np.eye(self.d)

        # Remember LLM's top cluster for credibility checking
        best_cluster_id = int(cluster_order[0])
        self._llm_suggested_arms = {a for a in range(self.d) if cluster_of[a] == best_cluster_id}

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            self._build_covariance()

        # Periodic credibility check
        if (self._built and self.t > self.T_warmup and
            self.t % self.check_interval == 0 and self._llm_suggested_arms):
            emp_top = set(int(a) for a in np.argsort(self.mu_hat)[::-1][:self.m])
            overlap = len(emp_top & self._llm_suggested_arms) / self.m
            self.w_credibility = 0.7 * self.w_credibility + 0.3 * overlap
            self.w_credibility = float(np.clip(self.w_credibility, 0.0, 1.0))

        if not self._built:
            return super().select_arms()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)

        # Independent Gaussian sample
        eps = self.np_rng.standard_normal(self.d)
        cts_samples = means + sigmas * eps

        # Full-kernel correlated sample
        z = self.np_rng.standard_normal(self.d)
        correlated_noise = self._cholesky @ z
        corr_samples = means + sigmas * correlated_noise

        # Interpolate by credibility
        w = self.w_credibility
        samples = w * corr_samples + (1 - w) * cts_samples
        return list(np.argsort(samples)[::-1][:self.m])


# ─── RandomCorr_CTS — MANDATORY ABLATION ─────────────────────────────────
class RandomCorrCTS(CTSBase):
    """CRITICAL ABLATION: same algorithm as M2/N1/N4 but with RANDOM clusters.

    If RandomCorr beats CTS but by less than N4, LLM is contributing.
    If RandomCorr matches N4, LLM is irrelevant — gains come from correlated
    sampling alone.

    This is the ablation any reviewer will demand before accepting our
    'LLM-guided' claim.
    """
    name = "ABLATION_random_corr"

    def __init__(self, d, m, T_warmup: int = 30, rho: float = 0.6,
                 n_clusters: int = 8, **kw):
        # NOTE: no oracle! This algorithm uses NO LLM.
        super().__init__(d, m, **kw)
        self.T_warmup = T_warmup
        self.rho = rho
        self.n_clusters = n_clusters
        self.cluster_of = np.zeros(d, dtype=int)
        self._built = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            # RANDOM cluster assignment
            random_assign = self.np_rng.randint(0, self.n_clusters, size=self.d)
            self.cluster_of = random_assign

        if not self._built:
            return super().select_arms()
        return self._correlated_sample()

    def _correlated_sample(self):
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        n_cl = int(self.cluster_of.max()) + 1
        z = self.np_rng.standard_normal(n_cl)
        eps = self.np_rng.standard_normal(self.d)
        samples = means + sigmas * (
            math.sqrt(self.rho) * z[self.cluster_of] +
            math.sqrt(1 - self.rho) * eps
        )
        return list(np.argsort(samples)[::-1][:self.m])


FINAL_ALGOS = {
    "N3_info_min_fixed": InfoMinCTSFixed,
    "N5_corr_full_robust": CorrFullRobust,
    "ABLATION_random_corr": RandomCorrCTS,
}
