"""Long-horizon variants of CorrCTS to fix the advantage-decay observed in Tier 5.

Problem: CorrCTS-Full builds a static RBF kernel at t=T_warmup=30 from noisy
empirical means. At T=25,000 the advantage over CTS peaks at t≈5000 then decays
because stale correlations force co-pulling of miscorrelated arms.

This file implements 5 variants exploring different fixes:

  V1_decay_kernel       : static LLM kernel, but rho_max decays over time
                          (cheapest — tests "fade correlation" hypothesis)
  V2_requery_logspaced  : re-query LLM at t ∈ {30, 300, 3000} — 3 calls/trial
                          (log-spaced schedule, matches LLM-BO intuition)
  V3_blend_llm_data     : re-query at {30, 500, 2500, 10000}; blend LLM kernel
                          with data-empirical kernel, weight shifts to data
  V4_refine_topk        : 2 calls — at t=30 (full) and t=5000 (top-50 only);
                          replace top-K block of kernel, keep rest
  V5_ensemble_kernels   : 3 LLM calls at t=30 with different prompts; maintain
                          Bayesian posterior over kernels, sample kernel then TS

Reference: _build_covariance in breakthrough_algorithms.py:49
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


def _build_rbf_from_clusters(clusters: list[list[int]], d: int,
                              kernel_scale: float = 0.5,
                              rho_max: float = 0.7) -> np.ndarray:
    """Shared kernel construction: cluster membership → RBF correlation."""
    cluster_of = np.zeros(d, dtype=int)
    for cidx, cluster in enumerate(clusters):
        for aid in cluster:
            if 0 <= aid < d:
                cluster_of[aid] = cidx
    n_clusters = int(cluster_of.max()) + 1

    cluster_rank = np.arange(n_clusters) / max(1, n_clusters - 1)
    arm_rank = cluster_rank[cluster_of]
    diff = arm_rank[:, None] - arm_rank[None, :]
    Sigma = rho_max * np.exp(-(diff ** 2) / (2 * kernel_scale ** 2))
    np.fill_diagonal(Sigma, 1.0)
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, 1e-4, None)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return Sigma


def _cholesky_safe(Sigma: np.ndarray, d: int) -> np.ndarray:
    try:
        return np.linalg.cholesky(Sigma + 1e-3 * np.eye(d))
    except np.linalg.LinAlgError:
        return np.eye(d)


# ─── V1: Decay-only (no re-query) ─────────────────────────────────────────
class V1DecayKernel(CTSBase):
    """Same LLM kernel as N1, but rho_max decays as 1/(1 + t/tau).

    At t=0: full LLM correlation.
    At t=tau: half strength.
    At t=10*tau: ~10% strength → essentially CTS.
    """
    name = "V1_decay_kernel"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 tau: float = 5000.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max_base = rho_max
        self.tau = tau
        self._built = False
        self.Sigma_base = np.eye(d)

    def _build(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma_base = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max_base
        )
        self._built = True

    def _current_kernel(self):
        # Convex combination of two PSD matrices is PSD (no eigh needed).
        decay = 1.0 / (1.0 + self.t / self.tau)
        K = decay * self.Sigma_base + (1.0 - decay) * np.eye(self.d)
        return K

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build()

        if not self._built:
            return super().select_arms()

        Sigma = self._current_kernel()
        L = _cholesky_safe(Sigma, self.d)

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (L @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V2: Log-spaced re-query ──────────────────────────────────────────────
class V2RequeryLogspaced(CTSBase):
    """Re-query LLM at t ∈ {30, 300, 3000}. Replace kernel each time.

    LLM sees increasingly accurate mu_hat, so later clusters should be better.
    """
    name = "V2_requery_logspaced"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 query_times=(30, 300, 3000), **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.query_times = set(query_times)
        self.Sigma = np.eye(d)
        self._cholesky = np.eye(d)
        self._built = False

    def _rebuild(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._cholesky = _cholesky_safe(self.Sigma, self.d)
        self._built = True

    def select_arms(self):
        if self.t in self.query_times:
            self._rebuild()

        if not self._built:
            return super().select_arms()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (self._cholesky @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V3: Calibration-gated blend (LLM + data) ────────────────────────────
class V3BlendLLMData(CTSBase):
    """Re-query LLM at {30, 500, 2500, 10000}; blend LLM kernel with
    data-empirical kernel. Weight shifts to data as t grows.

    K_t = w(t) * K_LLM + (1 - w(t)) * K_data
    w(t) = 1 / (1 + t / tau)
    """
    name = "V3_blend_llm_data"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 query_times=(30, 500, 2500, 10000), tau: float = 3000.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.query_times = set(query_times)
        self.tau = tau
        self.K_llm = np.eye(d)
        self._built = False
        # Track mu_hat history for data-kernel estimation
        self._mu_history: list[np.ndarray] = []
        self._history_interval = 100

    def _rebuild_llm(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.K_llm = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._built = True

    def _data_kernel(self) -> np.ndarray:
        """Empirical correlation of arm mu_hat trajectories. Falls back to I."""
        if len(self._mu_history) < 3:
            return np.eye(self.d)
        X = np.array(self._mu_history)  # (history, d)
        X = X - X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        Xn = X / std
        C = (Xn.T @ Xn) / len(Xn)
        # Scale to [0, rho_max] range, positive part only
        C = 0.5 * (C + C.T)
        C = np.clip(C, -self.rho_max, self.rho_max)
        np.fill_diagonal(C, 1.0)
        # Force PSD
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-4, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C

    def select_arms(self):
        if self.t in self.query_times:
            self._rebuild_llm()
        if self.t > 0 and self.t % self._history_interval == 0:
            self._mu_history.append(self.mu_hat.copy())

        if not self._built:
            return super().select_arms()

        w = 1.0 / (1.0 + self.t / self.tau)
        K = w * self.K_llm + (1.0 - w) * self._data_kernel()
        L = _cholesky_safe(K, self.d)

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (L @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V4: Refine top-K at t=5000 ──────────────────────────────────────────
class V4RefineTopK(CTSBase):
    """Two queries: full cluster at t=30, top-K re-cluster at t=5000.

    At t=5000, empirical means are well-estimated. Ask LLM to re-cluster only
    the top K=min(m*5, d) arms by current UCB. Replace only the top-K block of
    the kernel; keep the rest of the kernel fixed (those arms are clearly bad).
    """
    name = "V4_refine_topk"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 refine_t: int = 5000, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.refine_t = refine_t
        self.Sigma = np.eye(d)
        self._cholesky = np.eye(d)
        self._built = False
        self._refined = False
        self.K = max(self.m * 5, 10)

    def _build_initial(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._cholesky = _cholesky_safe(self.Sigma, self.d)
        self._built = True

    def _refine_topk(self):
        """Re-query LLM only on top-K arms, replace that block."""
        # Find top-K by empirical mean
        top_idx = np.argsort(self.mu_hat)[::-1][:self.K].tolist()
        # Sub-problem: mu_hat restricted to these arms
        sub_mu = self.mu_hat[top_idx].tolist()
        # Build a temporary top-K oracle view — just query clusters on subset
        # We use the full oracle's cluster query but re-map the arms
        sub_oracle = _TopKOracleProxy(self.oracle, top_idx, self.d)
        sub_clusters = sub_oracle.query_clusters(sub_mu, n_clusters=5)
        # Build sub-kernel in top-K space
        K_new = _build_rbf_from_clusters(
            sub_clusters, self.K, self.kernel_scale, self.rho_max
        )
        # Splice into full Sigma at top_idx positions
        for ii, i in enumerate(top_idx):
            for jj, j in enumerate(top_idx):
                self.Sigma[i, j] = K_new[ii, jj]
        # Re-ensure PSD
        eigvals, eigvecs = np.linalg.eigh(self.Sigma)
        eigvals = np.clip(eigvals, 1e-4, None)
        self.Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self._cholesky = _cholesky_safe(self.Sigma, self.d)
        self._refined = True

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build_initial()
        if self.t == self.refine_t and not self._refined and self._built:
            self._refine_topk()

        if not self._built:
            return super().select_arms()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (self._cholesky @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


class _TopKOracleProxy:
    """Tiny adapter that lets us query the oracle on a subset of arms."""
    def __init__(self, oracle: GPTOracle, original_ids: list[int], d: int):
        self.oracle = oracle
        self.original_ids = original_ids
        self.d = d

    def query_clusters(self, sub_mu: list[float], n_clusters: int):
        """Ask oracle to cluster the subset; returns clusters in sub-index space."""
        k = len(sub_mu)
        # Hack: temporarily override oracle's d for the query
        saved_d = self.oracle.d
        self.oracle.d = k
        try:
            clusters = self.oracle.query_clusters(sub_mu, n_clusters=n_clusters)
        finally:
            self.oracle.d = saved_d
        # Filter to valid sub-indices
        filtered = [[a for a in c if 0 <= a < k] for c in clusters]
        return [c for c in filtered if c]


# ─── V5: Ensemble of LLM kernels ─────────────────────────────────────────
class V5EnsembleKernels(CTSBase):
    """3 LLM calls at t=30 with different prompt seeds → 3 candidate kernels.

    Maintain log marginal likelihood p(data | K_k) for each. At each round,
    draw K ~ posterior over kernels, then TS. Robust if any LLM response is
    misleading.
    """
    name = "V5_ensemble_kernels"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 n_kernels: int = 3, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.n_kernels = n_kernels
        self.kernels = []
        self.cholesky_k = []
        self.log_marg = np.zeros(n_kernels)  # prior uniform
        self._built = False

    def _build(self):
        """3 queries with different temperatures/prompts (via oracle re-queries).

        We vary n_clusters to get diverse partitions.
        """
        for k in range(self.n_kernels):
            # Vary n_clusters 6, 8, 10 to get diverse partitions
            n_cl = 6 + 2 * k
            try:
                clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=n_cl)
            except Exception:
                clusters = [[i] for i in range(self.d)]
            Sigma = _build_rbf_from_clusters(
                clusters, self.d, self.kernel_scale, self.rho_max
            )
            self.kernels.append(Sigma)
            self.cholesky_k.append(_cholesky_safe(Sigma, self.d))
        self._built = True

    def _update_log_marg(self, selected: list[int], rewards: list[float]):
        """Rough log marginal likelihood update: for each kernel, score how
        well it predicts the observed correlation in rewards.

        Simple heuristic: if rewards among selected arms show positive correlation,
        up-weight kernels that have high off-diagonal entries among selected pairs.
        """
        if len(selected) < 2:
            return
        r_arr = np.array(rewards)
        r_diff = r_arr - r_arr.mean()
        for k, K in enumerate(self.kernels):
            # Score: sum_{i,j in selected, i!=j} K[i,j] * sign(r_i * r_j diff-mean)
            score = 0.0
            for ii in range(len(selected)):
                for jj in range(ii + 1, len(selected)):
                    i_arm, j_arm = selected[ii], selected[jj]
                    score += K[i_arm, j_arm] * r_diff[ii] * r_diff[jj]
            self.log_marg[k] += 0.1 * score  # small step

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build()

        if not self._built:
            return super().select_arms()

        # Posterior over kernels
        logp = self.log_marg - self.log_marg.max()
        probs = np.exp(logp)
        probs = probs / probs.sum()
        k = self.np_rng.choice(self.n_kernels, p=probs)
        L = self.cholesky_k[k]

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (L @ eps)
        return list(np.argsort(theta)[::-1][:self.m])

    def update(self, selected: list[int], rewards: list[float]):
        if self._built:
            self._update_log_marg(selected, rewards)
        super().update(selected, rewards)


# ─── V6: CLUB-style edge pruning ─────────────────────────────────────────
class V6EdgePruning(CTSBase):
    """Start with LLM kernel; delete edges (i,j) when |mu_i - mu_j| exceeds
    a concentration threshold. Clusters 'melt' as data resolves arms.

    Reference: Gentile et al. CLUB (2014), UniCLUB (ICLR 2025) — edges are
    kept only while confidence intervals overlap. At t -> inf, every edge is
    pruned and we recover pure CTS.

    No new LLM calls; only 1 at t=30.
    """
    name = "V6_edge_pruning"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 beta: float = 2.0, prune_interval: int = 500, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.beta = beta
        self.prune_interval = prune_interval
        self.Sigma = np.eye(d)
        self._cholesky = np.eye(d)
        self._built = False

    def _build(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._cholesky = _cholesky_safe(self.Sigma, self.d)
        self._built = True

    def _prune_edges(self):
        """For each pair, if |mu_i - mu_j| > beta * sqrt(1/n_i + 1/n_j),
        zero out Sigma[i,j]. The edge is considered 'resolved different'."""
        mu = self.mu_hat
        n = np.maximum(self.n_pulls, 1)
        # Vectorized pair-wise gap check
        gap = np.abs(mu[:, None] - mu[None, :])
        ci = self.beta * np.sqrt(1.0 / n[:, None] + 1.0 / n[None, :])
        prune_mask = gap > ci
        # Don't prune the diagonal
        np.fill_diagonal(prune_mask, False)
        # Shrink pruned edges toward zero (don't fully zero to keep PSD smooth)
        Sigma_new = self.Sigma.copy()
        Sigma_new[prune_mask] *= 0.1
        # Re-ensure PSD
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma_new)
        eigvals = np.clip(eigvals, 1e-4, None)
        self.Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self._cholesky = _cholesky_safe(self.Sigma, self.d)

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build()
        if self._built and self.t > self.T_warmup and self.t % self.prune_interval == 0:
            self._prune_edges()

        if not self._built:
            return super().select_arms()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (self._cholesky @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V7: Per-arm correlation dampening ────────────────────────────────────
class V7PerArmDamping(CTSBase):
    """Off-diagonal entries of Sigma scaled by sqrt(f(n_i) * f(n_j)) where
    f(n) = 1/(1 + n/n_0). Well-pulled arms decouple from their cluster;
    under-pulled arms still benefit from kernel-based information sharing.

    Only 1 LLM call at t=30. Cheapest variant.
    """
    name = "V7_per_arm_damping"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 n_0: float = 100.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.n_0 = n_0
        self.Sigma_base = np.eye(d)
        self._built = False

    def _build(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma_base = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._built = True

    def _current_kernel(self):
        # Hadamard product of two PSD matrices is PSD (Schur product theorem):
        # damp = sqrt(f) * sqrt(f).T is rank-1 PSD; Sigma_base is PSD;
        # therefore Sigma_base ⊙ damp_outer (diag forced to 1) is PSD.
        # No eigh needed. ~50× faster.
        f = 1.0 / (1.0 + self.n_pulls / self.n_0)
        sqrt_f = np.sqrt(f)
        damp = np.outer(sqrt_f, sqrt_f)
        Sigma = self.Sigma_base * damp
        np.fill_diagonal(Sigma, 1.0)
        return Sigma

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build()

        if not self._built:
            return super().select_arms()

        Sigma = self._current_kernel()
        L = _cholesky_safe(Sigma, self.d)

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (L @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


LONGHORIZON_ALGOS = {
    "V1_decay_kernel": V1DecayKernel,
    "V2_requery_logspaced": V2RequeryLogspaced,
    "V3_blend_llm_data": V3BlendLLMData,
    "V4_refine_topk": V4RefineTopK,
    "V5_ensemble_kernels": V5EnsembleKernels,
    "V6_edge_pruning": V6EdgePruning,
    "V7_per_arm_damping": V7PerArmDamping,
}
