"""Long-horizon variants — second wave (V11–V15+).

These address the late-horizon decay more fundamentally than V1/V6/V7:

  V11_quality_gated     : Online predictive-LL monitor + adaptive LLM re-query
                          (matches user intuition: "re-cluster when divergent")
  V13_kmeans_refine     : Self-distillation — k-means on mu_hat at t∈{2k,10k},
                          blend with LLM partition (no new LLM calls)
  V15_mixed_effect_ts   : Hierarchical Bayes (Kveton AISTATS 2023) —
                          per-cluster pooled Beta + per-arm deviation;
                          hyperposterior auto-refines over time
  V8_bci                : Bayesian Cluster Inference — partition is latent,
                          CRP prior with LLM as base, posterior updates via
                          collapsed Gibbs (most principled, slowest)

Compares against V6_edge_pruning (current Tier 7 winner).
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle
from longhorizon_variants import _build_rbf_from_clusters, _cholesky_safe


# ─── V11: Quality-gated re-query ──────────────────────────────────────────
class V11QualityGated(CTSBase):
    """Track online predictive log-likelihood of the kernel; re-query LLM
    when quality degrades below threshold.

    Mechanism:
    - Build kernel at t=30 (LLM call #1).
    - Each round, after observing rewards r_t for selected arms S_t,
      compute a 'kernel quality' score: how well does Sigma predict the
      observed within-batch reward correlation?
    - Maintain EMA of quality. If quality_emA < threshold, re-query LLM
      (with fresh mu_hat) and rebuild kernel. Cap at K=5 re-queries per trial.
    """
    name = "V11_quality_gated"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 quality_ema: float = 0.95, quality_threshold: float = -0.5,
                 max_requeries: int = 4, min_requery_gap: int = 1000, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.quality_ema = quality_ema
        self.quality_threshold = quality_threshold
        self.max_requeries = max_requeries
        self.min_requery_gap = min_requery_gap
        self.Sigma = np.eye(d)
        self._cholesky = np.eye(d)
        self._built = False
        self._quality = 0.0
        self._n_requeries = 0
        self._last_query_t = 0

    def _rebuild(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        self.Sigma = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._cholesky = _cholesky_safe(self.Sigma, self.d)
        self._built = True
        self._last_query_t = self.t
        self._quality = 0.0  # reset quality after rebuild

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._rebuild()
            return super().select_arms() if not self._built else self._sample_corr()

        # Trigger: quality below threshold AND enough rounds since last query
        if (self._built and
            self._n_requeries < self.max_requeries and
            self.t - self._last_query_t >= self.min_requery_gap and
            self._quality < self.quality_threshold):
            self._rebuild()
            self._n_requeries += 1

        if not self._built:
            return super().select_arms()
        return self._sample_corr()

    def _sample_corr(self):
        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (self._cholesky @ eps)
        return list(np.argsort(theta)[::-1][:self.m])

    def update(self, selected: list[int], rewards: list[float]):
        # Quality score: sum_{i,j in selected, i<j} K[i,j] * (r_i - bar)(r_j - bar)
        # Positive when kernel agrees with observed within-batch covariance,
        # negative when disagrees. Normalize and EMA.
        if self._built and len(selected) >= 2:
            r = np.array(rewards)
            r_centered = r - r.mean()
            score = 0.0
            n_pairs = 0
            for ii in range(len(selected)):
                for jj in range(ii + 1, len(selected)):
                    i, j = selected[ii], selected[jj]
                    score += self.Sigma[i, j] * r_centered[ii] * r_centered[jj]
                    n_pairs += 1
            if n_pairs > 0:
                score /= n_pairs
                self._quality = self.quality_ema * self._quality + (1 - self.quality_ema) * score
        super().update(selected, rewards)


# ─── V13: K-means refinement (self-distillation) ─────────────────────────
class V13KMeansRefine(CTSBase):
    """At t=30: query LLM, build initial kernel.
    At t∈{2000, 10000}: run weighted k-means on mu_hat (weights = sqrt(n_pulls))
    to get a data-driven partition. Blend LLM kernel with k-means kernel.

    No additional LLM calls. ~2 expensive operations over the entire trial.
    """
    name = "V13_kmeans_refine"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 refine_times=(2000, 10000), n_clusters: int = 8, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.refine_times = set(refine_times)
        self.n_clusters = n_clusters
        self.K_llm = np.eye(d)
        self.K_data = np.eye(d)
        self._has_data_kernel = False
        self._built = False
        self._cholesky = np.eye(d)

    def _build_llm(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=self.n_clusters)
        self.K_llm = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        self._built = True
        self._update_cholesky()

    def _kmeans_partition(self):
        """Weighted k-means on mu_hat (1-D), return cluster assignment list-of-lists."""
        mu = self.mu_hat
        weights = np.sqrt(np.maximum(self.n_pulls, 1))
        # 1-D k-means via sorting + equal-mass quantile binning weighted
        # Simpler: sort by mu, divide into n_clusters quantile bins by weight
        order = np.argsort(mu)
        sorted_w = weights[order]
        cum_w = np.cumsum(sorted_w)
        total_w = cum_w[-1]
        boundaries = np.searchsorted(cum_w, np.linspace(0, total_w, self.n_clusters + 1))
        clusters = []
        for k in range(self.n_clusters):
            lo, hi = boundaries[k], boundaries[k + 1]
            if hi > lo:
                clusters.append(order[lo:hi].tolist())
        return [c for c in clusters if c]

    def _refine(self):
        clusters = self._kmeans_partition()
        K_data = _build_rbf_from_clusters(
            clusters, self.d, self.kernel_scale, self.rho_max
        )
        if not self._has_data_kernel:
            self.K_data = K_data
        else:
            # EMA blend with previous
            self.K_data = 0.5 * self.K_data + 0.5 * K_data
        self._has_data_kernel = True
        self._update_cholesky()

    def _update_cholesky(self):
        # Convex combination — both PSD → result is PSD
        if self._has_data_kernel:
            # Decay LLM weight: starts at 1, drops to 0.2 by t=10k
            w_llm = max(0.2, 1.0 - self.t / 12000.0)
            K = w_llm * self.K_llm + (1 - w_llm) * self.K_data
        else:
            K = self.K_llm
        self._cholesky = _cholesky_safe(K, self.d)

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build_llm()
        if self.t in self.refine_times and self._built:
            self._refine()

        if not self._built:
            return super().select_arms()

        # Update Cholesky periodically to reflect time-varying weight
        if self._built and self.t > self.T_warmup and self.t % 500 == 0:
            self._update_cholesky()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)
        eps = self.np_rng.standard_normal(self.d)
        theta = means + sigmas * (self._cholesky @ eps)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V15: Mixed-Effect Thompson Sampling (Kveton 2023) ───────────────────
class V15MixedEffectTS(CTSBase):
    """Hierarchical Beta-Binomial:
      - Each cluster c has hyperparameters (a_c, b_c) acting as 'cluster mean'
      - Each arm i in cluster c has Beta(a_c + α_i, b_c + β_i) marginal
      - Sample: first sample cluster mean μ_c ~ Beta(a_c, b_c),
                then sample arm θ_i ~ Beta(α_i + a_c·κ, β_i + b_c·κ) (shrinkage to cluster)

    LLM provides cluster assignment ONCE at t=30.
    Hyperparameters (a_c, b_c) update from cluster's pooled data.

    This is fundamentally different from kernel approaches: it's a true
    hierarchical Bayesian model, not a sampling-time correction.
    """
    name = "V15_mixed_effect_ts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 n_clusters: int = 8, shrinkage_kappa: float = 5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.n_clusters = n_clusters
        self.kappa = shrinkage_kappa
        self.cluster_of = np.zeros(d, dtype=int)
        self._built = False

    def _build(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=self.n_clusters)
        for cidx, cluster in enumerate(clusters):
            for aid in cluster:
                if 0 <= aid < self.d:
                    self.cluster_of[aid] = cidx
        self._built = True

    def _cluster_hyperposterior(self):
        """For each cluster c, compute pooled (a_c, b_c) from member arms."""
        n_cl = int(self.cluster_of.max()) + 1
        a_c = np.zeros(n_cl)
        b_c = np.zeros(n_cl)
        for c in range(n_cl):
            members = self.cluster_of == c
            a_c[c] = self.alphas[members].sum() - members.sum()  # subtract Beta(1,1) prior count
            b_c[c] = self.betas[members].sum() - members.sum()
        # Re-add a small prior to avoid degenerate
        a_c = np.maximum(a_c, 0) + 1.0
        b_c = np.maximum(b_c, 0) + 1.0
        return a_c, b_c

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build()

        if not self._built:
            return super().select_arms()

        a_c, b_c = self._cluster_hyperposterior()
        # Sample cluster means (one per cluster)
        cluster_means = self.np_rng.beta(a_c, b_c)
        # For each arm, sample with shrinkage toward cluster mean:
        # Effective Beta(α_i + κ·μ_c, β_i + κ·(1-μ_c))
        c = self.cluster_of
        kappa = self.kappa
        a_eff = self.alphas + kappa * cluster_means[c]
        b_eff = self.betas + kappa * (1 - cluster_means[c])
        theta = self.np_rng.beta(a_eff, b_eff)
        return list(np.argsort(theta)[::-1][:self.m])


# ─── V8: Bayesian Cluster Inference (BCI) ─────────────────────────────────
class V8BCI(CTSBase):
    """Bayesian Cluster Inference: partition is latent.

    Uses a Chinese Restaurant Process prior with the LLM partition as a
    'recommended seating' (high prior probability). Posterior updates via
    a collapsed Gibbs sampler over arm cluster assignments, where each
    cluster's marginal likelihood is Beta-Bernoulli on its pooled rewards.

    Per round:
      1. (Periodically) run K Gibbs sweeps on partition
      2. Sample partition from posterior → build kernel → sample θ → pick top-m
    """
    name = "V8_bci"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 alpha_crp: float = 2.0, llm_trust_lambda: float = 3.0,
                 lambda_half_life: float = 6000.0,
                 gibbs_sweeps_per_round: int = 1,
                 gibbs_refresh_interval: int = 100,
                 gibbs_refresh_sweeps: int = 10,
                 shrinkage_kappa: float = 5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.alpha_crp = alpha_crp
        self.llm_trust_lambda = llm_trust_lambda
        self.lambda_half_life = lambda_half_life
        self.gibbs_sweeps_per_round = gibbs_sweeps_per_round
        self.gibbs_refresh_interval = gibbs_refresh_interval
        self.gibbs_refresh_sweeps = gibbs_refresh_sweeps
        self.kappa = shrinkage_kappa
        # cluster assignment as int[d]
        self.z = np.zeros(d, dtype=int)
        self.llm_z = np.zeros(d, dtype=int)
        # Empirical-Bayes Beta base measure (set at warmup)
        self.a0 = 1.0
        self.b0 = 1.0
        self._built = False

    def _build_initial(self):
        clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=8)
        for cidx, cluster in enumerate(clusters):
            for aid in cluster:
                if 0 <= aid < self.d:
                    self.z[aid] = cidx
        self.llm_z = self.z.copy()
        # Empirical-Bayes base measure: weak Beta from method-of-moments on mu_hat
        mu_bar = float(np.clip(self.mu_hat.mean(), 0.05, 0.95))
        var = float(max(self.mu_hat.var(), 1e-3))
        nu = mu_bar * (1 - mu_bar) / var - 1.0
        nu = max(nu, 1.0)
        self.a0 = max(1.0, 0.1 * mu_bar * nu)
        self.b0 = max(1.0, 0.1 * (1 - mu_bar) * nu)
        # Initial burn-in
        for _ in range(self.gibbs_refresh_sweeps):
            self._gibbs_sweep()
        self._built = True

    def _current_lambda(self):
        """Anneal λ over time: λ(t) = λ * exp(-t / T_half). Bayesian annealing."""
        return self.llm_trust_lambda * math.exp(-self.t / self.lambda_half_life)

    def _gibbs_sweep(self):
        """Collapsed Gibbs over partition. Same-table-as-LLM bonus uses
        annealed λ. Beta-Binomial marginals via lgamma."""
        from math import lgamma
        lam = self._current_lambda()
        idx_perm = np.argsort(self.np_rng.standard_normal(self.d))  # cheap permutation
        for i in idx_perm:
            i = int(i)
            # Stats per cluster excluding arm i
            mask_excl = np.ones(self.d, dtype=bool)
            mask_excl[i] = False
            z_others = self.z[mask_excl]
            a_others = self.alphas[mask_excl] - 1.0  # raw success counts
            b_others = self.betas[mask_excl] - 1.0   # raw failure counts
            a_i = self.alphas[i] - 1.0
            b_i = self.betas[i] - 1.0

            unique_cs = np.unique(z_others)
            log_probs = []
            cluster_ids = []
            for c in unique_cs:
                mem = z_others == c
                size = int(mem.sum())
                a_c = float(a_others[mem].sum())
                b_c = float(b_others[mem].sum())
                # CRP prior: log size
                log_p = math.log(size)
                # LLM bonus: weight by fraction of cluster c arms that the LLM put with arm i
                llm_match = float(np.sum(
                    (z_others[mem] == c) &
                    (self.llm_z[mask_excl][mem] == self.llm_z[i])
                )) / max(size, 1)
                log_p += lam * llm_match
                # Beta-Binomial marginal: log[B(a0+a_c+a_i, b0+b_c+b_i) / B(a0+a_c, b0+b_c)]
                log_p += (
                    lgamma(self.a0 + a_c + a_i) + lgamma(self.b0 + b_c + b_i)
                    - lgamma(self.a0 + a_c + a_i + self.b0 + b_c + b_i)
                    - lgamma(self.a0 + a_c) - lgamma(self.b0 + b_c)
                    + lgamma(self.a0 + a_c + self.b0 + b_c)
                )
                log_probs.append(log_p)
                cluster_ids.append(int(c))
            # New table option
            log_p_new = math.log(self.alpha_crp) + (
                lgamma(self.a0 + a_i) + lgamma(self.b0 + b_i)
                - lgamma(self.a0 + a_i + self.b0 + b_i)
                - lgamma(self.a0) - lgamma(self.b0) + lgamma(self.a0 + self.b0)
            )
            log_probs.append(log_p_new)
            new_id = int(unique_cs.max()) + 1 if len(unique_cs) > 0 else 0
            cluster_ids.append(new_id)

            log_probs = np.array(log_probs)
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            choice = int(self.np_rng.choice(len(probs), p=probs))
            self.z[i] = cluster_ids[choice]

        # Compact labels
        unique = np.unique(self.z)
        remap = {int(old): new for new, old in enumerate(unique)}
        self.z = np.array([remap[int(v)] for v in self.z])

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._build_initial()

        if self._built and self.t > self.T_warmup:
            # Light per-round update (1 sweep) + periodic deep refresh
            if self.t % self.gibbs_refresh_interval == 0:
                for _ in range(self.gibbs_refresh_sweeps):
                    self._gibbs_sweep()
            elif self.gibbs_sweeps_per_round > 0 and self.t % 5 == 0:
                # Sweep every 5 rounds (cheap amortization)
                for _ in range(self.gibbs_sweeps_per_round):
                    self._gibbs_sweep()

        if not self._built:
            return super().select_arms()

        # Hierarchical TS: sample cluster mean, then arm θ with shrinkage to it
        n_cl = int(self.z.max()) + 1
        a_c = np.zeros(n_cl)
        b_c = np.zeros(n_cl)
        for c in range(n_cl):
            members = self.z == c
            a_c[c] = self.alphas[members].sum() - members.sum() + self.a0
            b_c[c] = self.betas[members].sum() - members.sum() + self.b0
        cluster_mean = self.np_rng.beta(np.maximum(a_c, 0.1), np.maximum(b_c, 0.1))
        # Per-arm shrinkage that decays with n_pulls (κ_i = κ / sqrt(1 + n_i))
        shrink = self.kappa / np.sqrt(1.0 + self.n_pulls)
        mu_c = cluster_mean[self.z]
        a_eff = self.alphas + shrink * mu_c
        b_eff = self.betas + shrink * (1.0 - mu_c)
        theta = self.np_rng.beta(np.maximum(a_eff, 0.1), np.maximum(b_eff, 0.1))
        return list(np.argsort(theta)[::-1][:self.m])


LONGHORIZON_V2_ALGOS = {
    "V8_bci": V8BCI,
    "V11_quality_gated": V11QualityGated,
    "V13_kmeans_refine": V13KMeansRefine,
    "V15_mixed_effect_ts": V15MixedEffectTS,
}
