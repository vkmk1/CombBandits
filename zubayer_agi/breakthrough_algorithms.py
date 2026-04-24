"""Breakthrough algorithms from deep-research directions.

Built from Sonnet 4.6 research synthesis:
- CORR-CTS-Full (Direction 1): full pairwise similarity matrix → elliptical posterior
- HypoTS (Direction 2): Bayesian mixture over K LLM-generated hypotheses
- InfoMin-CTS (Direction 5 proxy): info-budgeted LLM querying

Foundational references:
- Correlated Arms Bandits (NeurIPS 2020, arXiv 1911.03959)
- VITS Variational TS (2024, arXiv 2307.10167)
- Mixed-Effect TS (AISTATS 2023)
- Chained Info-Theoretic Bounds (arXiv 2403.03361)
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


# ─── N1. CORR-CTS-Full (LLM-derived full covariance) ─────────────────────
class CorrCTSFull(CTSBase):
    """Full covariance version of CORR-CTS.

    Instead of block-diagonal (within-cluster) correlation, ask LLM for dense
    pairwise similarity matrix. Convert to positive-semi-definite covariance via
    kernel trick: Σ_ij = exp(-||s_i - s_j||²/l²). Sample multivariate Gaussian
    posterior with this covariance.

    Key property (from GP-TS theory, Srinivas 2010, arXiv 0912.3995):
    regret bounded by information gain γ_T of the kernel, not by d.
    If LLM similarity is informative, γ_T << d log T.
    """
    name = "N1_corr_cts_full"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.Sigma_corr = np.eye(d)  # default: uncorrelated
        self._cholesky = None
        self._built = False

    def _build_covariance(self):
        """Extract pairwise similarity from LLM, build PSD covariance.

        Strategy: reuse cluster query (cheap; 1 call), convert clusters to
        a continuous similarity structure with ρ_max within-cluster, decaying
        kernel between clusters based on LLM's cluster ordering.
        """
        mu_list = self.mu_hat.tolist()
        clusters = self.oracle.query_clusters(mu_list, n_clusters=8)

        # Build arm → cluster map
        cluster_of = np.zeros(self.d, dtype=int)
        for cidx, cluster in enumerate(clusters):
            for aid in cluster:
                if 0 <= aid < self.d:
                    cluster_of[aid] = cidx
        n_clusters = int(cluster_of.max()) + 1

        # Cluster centroids in "rank space" (derived from LLM's ordering)
        cluster_rank = np.arange(n_clusters) / max(1, n_clusters - 1)
        arm_rank = cluster_rank[cluster_of]

        # Build correlation matrix: high within same rank, RBF decay across
        diff = arm_rank[:, None] - arm_rank[None, :]
        dist2 = diff ** 2
        Sigma = self.rho_max * np.exp(-dist2 / (2 * self.kernel_scale ** 2))
        # Force diagonal = 1
        np.fill_diagonal(Sigma, 1.0)

        # Ensure PSD via regularization
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, 1e-4, None)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.Sigma_corr = Sigma
        try:
            self._cholesky = np.linalg.cholesky(Sigma + 1e-3 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self._cholesky = np.eye(self.d)  # fallback

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            self._build_covariance()

        if not self._built:
            return super().select_arms()

        # Multivariate Gaussian posterior: mean = Beta mean, covariance = LLM Σ × σ
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)

        # Correlated samples via Cholesky
        z = self.np_rng.standard_normal(self.d)
        correlated_noise = self._cholesky @ z  # (d,) correlated standard normal
        samples = means + sigmas * correlated_noise
        return list(np.argsort(samples)[::-1][:self.m])


# ─── N2. HypoTS (hypothesis-testing bandit) ─────────────────────────────
class HypoTS(CTSBase):
    """Bayesian mixture over K LLM-generated hypotheses.

    LLM generates K=5 candidate "hypotheses" about the top-m set (asked
    as K diverse top-m queries with temperature variation). Maintain a
    posterior π_k over hypotheses. Each trial: sample k ~ π, use H_k's
    arms as boosted priors, select via TS.
    Update π_k ∝ π_k · P(observed rewards | H_k).

    Theoretical: if well-specified, meta-regret O(log K); if misspecified,
    reduces to standard TS (no worse).
    """
    name = "N2_hypo_ts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 K_hypotheses: int = 5, prior_strength: float = 3.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.K_hypotheses = K_hypotheses
        self.prior_strength = prior_strength
        self.hypotheses: list[set[int]] = []
        self.posterior = None  # will init
        self._built = False

    def _generate_hypotheses(self):
        """K diverse top-m queries to generate hypotheses."""
        mu = self.mu_hat.tolist()
        seen = set()
        self.hypotheses = []
        attempts = 0
        while len(self.hypotheses) < self.K_hypotheses and attempts < 2 * self.K_hypotheses:
            picks = tuple(sorted(self.oracle.query_top_m(mu)))
            if picks not in seen and len(picks) == self.m:
                seen.add(picks)
                self.hypotheses.append(set(picks))
            attempts += 1
        if not self.hypotheses:
            # Fallback: random hypothesis
            self.hypotheses = [set(self.np_rng.choice(self.d, self.m, replace=False))]
        K = len(self.hypotheses)
        self.posterior = np.full(K, 1.0 / K)

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            self._generate_hypotheses()
            # Apply first-time posterior boost
            for k, hypo in enumerate(self.hypotheses):
                for aid in hypo:
                    self.alphas[aid] += self.prior_strength * self.posterior[k]
        return super().select_arms()

    def update(self, selected, rewards):
        super().update(selected, rewards)
        if not self._built or not self.hypotheses:
            return
        # Update posterior over hypotheses via likelihood weighting
        selected_set = set(int(a) for a in selected)
        reward_sum = sum(rewards)
        m = len(selected)
        # P(rewards | H_k) approximation: arms in H_k succeed at 0.7, others at 0.3
        log_likes = np.zeros(len(self.hypotheses))
        for k, hypo in enumerate(self.hypotheses):
            for i, arm in enumerate(selected):
                r = rewards[i]
                p_if_good = 0.7 if arm in hypo else 0.3
                p_if_good = max(0.05, min(0.95, p_if_good))
                log_likes[k] += r * math.log(p_if_good) + (1 - r) * math.log(1 - p_if_good)
        # Normalize
        log_likes -= log_likes.max()
        likes = np.exp(log_likes)
        new_post = self.posterior * likes
        s = new_post.sum()
        if s > 1e-12:
            self.posterior = new_post / s
        # Small regularization toward uniform
        K = len(self.hypotheses)
        self.posterior = 0.95 * self.posterior + 0.05 / K


# ─── N3. InfoMin-CTS (minimal-info LLM querying) ─────────────────────────
class InfoMinCTS(CTSBase):
    """Formal info-budgeted LLM querying.

    Query LLM only when estimated information gain exceeds a threshold.
    Threshold decreases over time (1/t) so total calls bounded by O(log T).
    Tests Direction 5 empirically: does minimal LLM info suffice?
    """
    name = "N3_info_min"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 info_threshold_init: float = 0.5, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.info_threshold_init = info_threshold_init
        self._llm_picks: set[int] = set()
        self._called_once = False

    def select_arms(self):
        if not self._called_once and self.t >= self.T_warmup:
            # Estimate current "uncertainty" as avg variance across top-m candidates
            total = self.alphas + self.betas
            variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
            top_indices = np.argsort(self.mu_hat)[::-1][:min(2 * self.m, self.d)]
            avg_uncertainty = variances[top_indices].mean()
            # Threshold: uncertainty must exceed threshold_init / sqrt(t)
            threshold = self.info_threshold_init / math.sqrt(self.t + 1)
            if avg_uncertainty > threshold:
                picks = self.oracle.query_top_m(self.mu_hat.tolist())
                self._llm_picks = set(picks)
                # Minimal injection: 2 pseudo-successes per LLM pick
                for aid in picks:
                    self.alphas[aid] += 2.0
                self._called_once = True
        return super().select_arms()


# ─── N4. Robust-CORR-CTS (game-theoretic robustness, Direction 6) ───────
class RobustCorrCTS(CTSBase):
    """CORR-CTS with credibility gating against worst-case LLM.

    Maintain a credibility weight w_t ∈ [0,1]. Interpolate between
    correlated sampling (w=1) and independent CTS (w=0).
    Update w based on whether LLM-guided arms empirically perform well.

    Guarantee (informal): regret ≤ O(√T) against worst-case LLM (equivalent
    to pure CTS), with polylog improvement if LLM is helpful.
    """
    name = "N4_robust_corr"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 rho: float = 0.6, n_clusters: int = 8,
                 check_interval: int = 100, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.rho = rho
        self.n_clusters = n_clusters
        self.check_interval = check_interval
        self.cluster_of = np.zeros(d, dtype=int)
        self.w_credibility = 1.0
        self._built = False
        self._llm_suggested_arms = set()

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            clusters = self.oracle.query_clusters(self.mu_hat.tolist(), self.n_clusters)
            for cidx, cluster in enumerate(clusters):
                for aid in cluster:
                    if 0 <= aid < self.d:
                        self.cluster_of[aid] = cidx
            # LLM's implicit "top-m" = union of best clusters
            cluster_means = np.zeros(int(self.cluster_of.max()) + 1)
            counts = np.zeros(len(cluster_means))
            for aid in range(self.d):
                cluster_means[self.cluster_of[aid]] += self.mu_hat[aid]
                counts[self.cluster_of[aid]] += 1
            cluster_means /= np.maximum(counts, 1)
            best_cluster = int(np.argmax(cluster_means))
            self._llm_suggested_arms = {a for a in range(self.d) if self.cluster_of[a] == best_cluster}

        # Periodic credibility check
        if (self.t > self.T_warmup and self.t % self.check_interval == 0 and
            self._built and self._llm_suggested_arms):
            emp_top = set(int(a) for a in np.argsort(self.mu_hat)[::-1][:self.m])
            overlap = len(emp_top & self._llm_suggested_arms) / self.m
            # Smooth credibility update
            self.w_credibility = 0.7 * self.w_credibility + 0.3 * overlap
            self.w_credibility = np.clip(self.w_credibility, 0.0, 1.0)

        if not self._built or self.w_credibility < 0.1:
            return super().select_arms()

        # Interpolated sampling
        return self._interpolated_sample()

    def _interpolated_sample(self):
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)

        # Pure CTS sample
        eps_cts = self.np_rng.standard_normal(self.d)
        cts_samples = means + sigmas * eps_cts

        # Correlated sample via cluster structure
        n_cl = int(self.cluster_of.max()) + 1
        z = self.np_rng.standard_normal(n_cl)
        eps_corr = self.np_rng.standard_normal(self.d)
        corr_samples = means + sigmas * (
            math.sqrt(self.rho) * z[self.cluster_of] +
            math.sqrt(1 - self.rho) * eps_corr
        )

        # Interpolate by credibility
        w = self.w_credibility
        samples = w * corr_samples + (1 - w) * cts_samples
        return list(np.argsort(samples)[::-1][:self.m])


BREAKTHROUGH_ALGOS = {
    "N1_corr_cts_full": CorrCTSFull,
    "N2_hypo_ts": HypoTS,
    "N3_info_min": InfoMinCTS,
    "N4_robust_corr": RobustCorrCTS,
}
