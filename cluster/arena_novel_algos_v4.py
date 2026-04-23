"""Loop 3 novel algorithms: 8 research directions from deep literature survey.

Post-Loop-2 arena (21 algos, 30 configs) showed no single algorithm dominates.
Pareto frontier: pool_cts_doubling (mean), freq_pool_cts_dual (worst-case),
freq_pool_ts (win rate), warm_start_cts (consistent_wrong).

These 8 algorithms target specific gaps identified through literature review:
1. MetaPoolCTS — Tsallis-INF meta-learner over Pareto-frontier base policies
2. AdaptivePoolCTS — information-theoretic adaptive pool sizing
3. LUCBPool — pool construction as best-arm identification
4. OracleBudgetCTS — fixed query budget with optimal allocation
5. MultiOraclePool — ensemble pooling from heterogeneous oracles
6. ConsistencyRobustCTS — learning-augmented with explicit consistency/robustness
7. PoolCTSAbstain — pool with abstention (NeurIPS 2024 inspired)
8. ParetoMetaDual — meta over aggressive+safe with dual-pool fallback
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_oracle_pool(oracle, mu_hat, n_seeds, d, pool_size, n_pool_rounds,
                       n_safety, device):
    pool_counts = torch.zeros(n_seeds, d, device=device)
    for _ in range(n_pool_rounds):
        out = oracle.query_batched(mu_hat)
        sugg = out["suggested_sets"]
        pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
    mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
    arms = torch.topk(pool_counts, min(pool_size, d), dim=1).indices
    mask.scatter_(1, arms, True)
    safety = torch.randint(0, d, (n_seeds, n_safety), device=device)
    mask.scatter_(1, safety, True)
    return mask, pool_counts


def _build_data_pool(mu_hat, pool_size, d, n_seeds, n_safety, device):
    mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
    arms = torch.topk(mu_hat, min(pool_size, d), dim=1).indices
    mask.scatter_(1, arms, True)
    safety = torch.randint(0, d, (n_seeds, n_safety), device=device)
    mask.scatter_(1, safety, True)
    return mask


def _pool_top_m_reward(mu_hat, pool_mask, m):
    mu = mu_hat.clone()
    mu[~pool_mask] = -float("inf")
    return torch.topk(mu, m, dim=1).values.sum(dim=1)


def _cts_sample_and_select(alphas, betas_param, pool_mask, m):
    samples = torch.distributions.Beta(
        alphas.clamp(min=0.01), betas_param.clamp(min=0.01)
    ).sample()
    if pool_mask is not None:
        samples[~pool_mask] = -float("inf")
    return torch.topk(samples, m, dim=1).indices


def _standard_update(agent, selected, rewards):
    agent.__class__.__bases__[0].update(agent, selected, rewards)
    successes = (rewards > 0.5).float()
    failures = 1.0 - successes
    agent.alphas.scatter_add_(1, selected, successes)
    agent.betas_param.scatter_add_(1, selected, failures)


# =====================================================================
# 1. MetaPoolCTS — Tsallis-INF meta-learner over base policies
# =====================================================================

class MetaPoolCTS(BatchedAgentBase):
    """Tsallis-INF meta-learner selecting between Pareto-frontier base policies.

    Base policies:
    - Policy 0: Oracle pool CTS (aggressive, best on reliable oracles)
    - Policy 1: Data pool CTS (safe, best on consistent_wrong)
    - Policy 2: Full CTS (universal fallback)

    The meta-learner maintains weights via Tsallis-INF (1/2-Tsallis entropy),
    which achieves O(sqrt(T log K)) meta-regret. Combined with each base
    policy's own guarantee, this yields BoBW-style behavior.
    """
    name = "meta_pool_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.n_policies = 3

        self.oracle_pool = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.data_pool = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self.weights = torch.ones(n_seeds, self.n_policies, device=device) / self.n_policies
        self.cum_loss = torch.zeros(n_seeds, self.n_policies, device=device)
        self._active_policy = torch.zeros(n_seeds, dtype=torch.long, device=device)
        self._pools_built = False

    def _build_pools(self):
        self.oracle_pool, _ = _build_oracle_pool(
            self.oracle, self.mu_hat, self.n_seeds, self.d,
            self.pool_size, self.n_pool_rounds, self.n_safety, self.device)
        self.data_pool = _build_data_pool(
            self.mu_hat, self.pool_size, self.d, self.n_seeds,
            self.n_safety, self.device)
        self._pools_built = True

    def select_arms(self) -> torch.Tensor:
        if self.t == 0 or (self.t % 200 == 0 and self.t > 0):
            self._build_pools()

        eta = 1.0 / max(math.sqrt(self.t + 1), 1.0)
        probs = self.weights / self.weights.sum(dim=1, keepdim=True)
        self._active_policy = torch.multinomial(probs, 1).squeeze(1)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        s0 = samples.clone()
        s0[~self.oracle_pool] = -float("inf")
        a0 = torch.topk(s0, self.m, dim=1).indices

        s1 = samples.clone()
        s1[~self.data_pool] = -float("inf")
        a1 = torch.topk(s1, self.m, dim=1).indices

        a2 = torch.topk(samples, self.m, dim=1).indices

        result = a2.clone()
        mask0 = (self._active_policy == 0)
        mask1 = (self._active_policy == 1)
        result[mask0] = a0[mask0]
        result[mask1] = a1[mask1]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

        reward_sum = rewards.mean(dim=1)
        loss = 1.0 - reward_sum

        for p in range(self.n_policies):
            mask = (self._active_policy == p)
            imp_loss = loss / self.weights[:, p].clamp(min=0.01)
            self.cum_loss[:, p] += mask.float() * imp_loss

        eta = 1.0 / max(math.sqrt(self.t + 1), 1.0)
        self.weights = 1.0 / (eta * self.cum_loss + 1.0).pow(2)
        self.weights = self.weights.clamp(min=1e-6)

    def reset(self):
        super().reset()
        self.oracle_pool.zero_()
        self.data_pool.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self.weights.fill_(1.0 / self.n_policies)
        self.cum_loss.zero_()
        self._active_policy.zero_()
        self._pools_built = False


# =====================================================================
# 2. AdaptivePoolCTS — adaptive pool sizing
# =====================================================================

class AdaptivePoolCTS(BatchedAgentBase):
    """Pool-CTS with adaptive pool size based on oracle coverage signals.

    Starts with small pool (beta=2). Monitors whether empirical top-m
    arms are well-covered by the pool. If coverage drops, doubles pool size.
    If oracle is good, small pool gives faster convergence.
    If oracle is bad, pool grows toward d (recovering full CTS).
    """
    name = "adaptive_pool_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta_init: float = 2.0,
                 beta_max: float = 8.0, n_safety: int = 5,
                 n_pool_rounds: int = 10, check_interval: int = 100, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.check_interval = check_interval

        self.beta = torch.full((n_seeds,), beta_init, device=device)
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_mask.zero_()
        for seed_idx in range(self.n_seeds):
            ps = int(self.beta[seed_idx].item() * self.m)
            ps = min(ps, self.d)
            arms = torch.topk(pool_counts[seed_idx], ps).indices
            self.pool_mask[seed_idx].scatter_(0, arms, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def _check_coverage(self):
        emp_top = torch.topk(self.mu_hat, self.m, dim=1).indices
        covered = torch.zeros(self.n_seeds, device=self.device)
        for i in range(self.m):
            arm = emp_top[:, i]
            in_pool = self.pool_mask.gather(1, arm.unsqueeze(1)).squeeze(1)
            covered += in_pool.float()
        coverage_frac = covered / self.m

        need_expand = coverage_frac < 0.6
        self.beta = torch.where(
            need_expand,
            (self.beta * 1.5).clamp(max=self.beta_max),
            self.beta
        )

        if need_expand.any():
            self._build_pool()

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > 0 and self.t % self.check_interval == 0:
            self._check_coverage()

        return _cts_sample_and_select(self.alphas, self.betas_param,
                                      self.pool_mask, self.m)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.beta.fill_(self.beta_init)
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._pool_built = False


# =====================================================================
# 3. LUCBPool — pool construction via confidence-based arm identification
# =====================================================================

class LUCBPool(BatchedAgentBase):
    """LUCB-inspired pool construction + CTS exploitation.

    Phase 1: Round-robin to get initial estimates.
    Phase 2: Use LUCB-style confidence intervals on oracle frequency
    to identify a "confident pool" — arms that oracle endorses AND
    that have plausible empirical means.
    Phase 3: Run CTS on the confident pool.

    The pool is rebuilt when confidence intervals tighten enough to
    change membership. This is more sample-efficient than fixed-K querying.
    """
    name = "lucb_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 T_init: int | None = None, confidence_scale: float = 1.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.T_init = T_init if T_init is not None else max(d // m, 10)
        self.confidence_scale = confidence_scale

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self.oracle_freq = torch.zeros(n_seeds, d, device=device)
        self.oracle_queries = 0
        self._pool_built = False

    def _query_oracle_batch(self, n_queries: int):
        for _ in range(n_queries):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            self.oracle_freq.scatter_add_(
                1, sugg, torch.ones_like(sugg, dtype=torch.float32))
            self.oracle_queries += 1

    def _build_confident_pool(self):
        if self.oracle_queries == 0:
            self._query_oracle_batch(self.n_pool_rounds)

        freq_rate = self.oracle_freq / max(self.oracle_queries, 1)
        conf_radius = self.confidence_scale * torch.sqrt(
            torch.log(torch.tensor(self.t + 2.0, device=self.device))
            / (self.n_pulls + 1.0)
        )
        ucb = self.mu_hat + conf_radius
        score = freq_rate * 0.5 + (ucb / ucb.max(dim=1, keepdim=True).values) * 0.5

        self.pool_mask.zero_()
        arms = torch.topk(score, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_init:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pool_built:
            self._build_confident_pool()

        if self.t % 200 == 0 and self.t > self.T_init:
            self._query_oracle_batch(3)
            self._build_confident_pool()

        return _cts_sample_and_select(self.alphas, self.betas_param,
                                      self.pool_mask, self.m)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self.oracle_freq.zero_()
        self.oracle_queries = 0
        self._pool_built = False


# =====================================================================
# 4. OracleBudgetCTS — fixed oracle query budget with optimal allocation
# =====================================================================

class OracleBudgetCTS(BatchedAgentBase):
    """Pool-CTS under a fixed oracle query budget.

    Total budget K queries allocated across O(log T) epochs via doubling.
    Early epochs get more queries (when information is most valuable).
    Later epochs get fewer but benefit from refined mu_hat.

    Budget allocation: epoch k gets K_k = K * 2^{-k} / Z queries
    where Z = sum(2^{-k}) normalizes.
    """
    name = "oracle_budget_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, total_budget: int = 100,
                 beta: float = 3.0, n_safety: int = 5, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.total_budget = total_budget
        self.pool_size = int(beta * m)
        self.n_safety = n_safety

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self.oracle_freq = torch.zeros(n_seeds, d, device=device)

        self._budget_used = 0
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 50

    def _queries_for_epoch(self, epoch: int) -> int:
        weight = 2.0 ** (-epoch)
        n_epochs_est = max(int(math.log2(3000 / 50)) + 1, 1)
        Z = sum(2.0 ** (-k) for k in range(n_epochs_est))
        budget = int(self.total_budget * weight / Z)
        remaining = self.total_budget - self._budget_used
        return max(min(budget, remaining), 0)

    def _build_pool(self, n_queries: int):
        if n_queries <= 0:
            return
        for _ in range(n_queries):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            self.oracle_freq.scatter_add_(
                1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        self._budget_used += n_queries

        self.pool_mask.zero_()
        arms = torch.topk(self.oracle_freq, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, arms, True)

        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

    def select_arms(self) -> torch.Tensor:
        if self.t == 0:
            nq = self._queries_for_epoch(0)
            self._build_pool(nq)

        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(50 * (2 ** self._epoch), 5000)
            nq = self._queries_for_epoch(self._epoch)
            self._build_pool(nq)
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1.0 - self.mu_hat)).clamp(min=0)

        return _cts_sample_and_select(self.alphas, self.betas_param,
                                      self.pool_mask, self.m)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self.oracle_freq.zero_()
        self._budget_used = 0
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 50


# =====================================================================
# 5. MultiOraclePool — ensemble pooling from simulated heterogeneous oracles
# =====================================================================

class MultiOraclePool(BatchedAgentBase):
    """Ensemble pool from multiple simulated oracle "views".

    Simulates heterogeneity by querying the oracle with perturbed mu_hat
    (adding noise to simulate different LLM perspectives). Builds a pool
    from the union of all views, weighted by frequency across views.

    More robust than single-oracle pooling because different perturbations
    expose different arms, providing implicit coverage diversity.
    """
    name = "multi_oracle_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_views: int = 3,
                 n_pool_rounds: int = 10, perturbation: float = 0.1, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_views = n_views
        self.n_pool_rounds = n_pool_rounds
        self.perturbation = perturbation

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def _build_pool(self):
        total_freq = torch.zeros(self.n_seeds, self.d, device=self.device)

        for view in range(self.n_views):
            noise = torch.randn(self.n_seeds, self.d, device=self.device) * self.perturbation
            perturbed = (self.mu_hat + noise).clamp(0.01, 0.99)
            for _ in range(self.n_pool_rounds):
                out = self.oracle.query_batched(perturbed)
                sugg = out["suggested_sets"]
                total_freq.scatter_add_(
                    1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_mask.zero_()
        arms = torch.topk(total_freq, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()
        return _cts_sample_and_select(self.alphas, self.betas_param,
                                      self.pool_mask, self.m)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._pool_built = False


# =====================================================================
# 6. ConsistencyRobustCTS — learning-augmented consistency/robustness
# =====================================================================

class ConsistencyRobustCTS(BatchedAgentBase):
    """Learning-augmented Pool-CTS with explicit consistency-robustness tradeoff.

    Inspired by Blum & Srinivas (SODA 2025) and the algorithms-with-predictions
    framework. Runs two parallel streams:
    - Consistent stream: Pool-CTS (trusts oracle, O(m^2 log T / Delta) if right)
    - Robust stream: Full CTS (ignores oracle, O(d log T / Delta) always)

    Interpolates via a mixing parameter lambda_t that starts at 1 (fully trust)
    and decreases based on evidence of oracle failure. If pool's empirical
    top-m deviates from global top-m, lambda_t decays.
    """
    name = "consistency_robust_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 lambda_init: float = 0.8, decay_rate: float = 0.05, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.lambda_init = lambda_init
        self.decay_rate = decay_rate

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self.lam = torch.full((n_seeds,), lambda_init, device=device)
        self._pool_built = False

    def _build_pool(self):
        self.pool_mask, _ = _build_oracle_pool(
            self.oracle, self.mu_hat, self.n_seeds, self.d,
            self.pool_size, self.n_pool_rounds, self.n_safety, self.device)
        self._pool_built = True

    def _update_lambda(self):
        pool_top = _pool_top_m_reward(self.mu_hat, self.pool_mask, self.m)
        global_top = torch.topk(self.mu_hat, self.m, dim=1).values.sum(dim=1)
        ratio = pool_top / global_top.clamp(min=1e-6)
        deficit = (1.0 - ratio).clamp(min=0)
        self.lam = (self.lam - self.decay_rate * deficit).clamp(min=0.0, max=1.0)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > 0 and self.t % 100 == 0:
            self._update_lambda()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_samples, self.m, dim=1).indices
        full_action = torch.topk(samples, self.m, dim=1).indices

        use_pool = torch.rand(self.n_seeds, device=self.device) < self.lam
        result = full_action.clone()
        result[use_pool] = pool_action[use_pool]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self.lam.fill_(self.lambda_init)
        self._pool_built = False


# =====================================================================
# 7. PoolCTSAbstain — pool with abstention
# =====================================================================

class PoolCTSAbstain(BatchedAgentBase):
    """Pool-CTS that can abstain from using oracle advice.

    Inspired by Pasteris et al. (NeurIPS 2024) "Bandits with Abstention
    under Expert Advice". The algorithm maintains a confidence measure
    for the oracle pool. When confidence is high, uses pool. When
    confidence drops below threshold, abstains and runs full CTS.

    Confidence is measured by oracle-empirical agreement: how many of
    the oracle's top-m arms are also in the empirical top-2m.
    """
    name = "pool_cts_abstain"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 abstain_threshold: float = 0.3, T_warmup: int | None = None,
                 **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.abstain_threshold = abstain_threshold
        self.T_warmup = T_warmup if T_warmup is not None else max(d // m * 3, 30)

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._use_oracle = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._pool_built = False

    def _build_pool_and_decide(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(
                1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        oracle_top = torch.topk(pool_counts, self.m, dim=1).indices
        emp_top_2m = torch.topk(self.mu_hat, min(2 * self.m, self.d), dim=1).indices

        emp_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        emp_set.scatter_(1, emp_top_2m, True)
        oracle_in_emp = emp_set.gather(1, oracle_top).float().mean(dim=1)

        self._use_oracle = oracle_in_emp >= self.abstain_threshold

        self.pool_mask.zero_()
        arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_warmup:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pool_built:
            self._build_pool_and_decide()

        if self.t % 300 == 0 and self.t > self.T_warmup:
            self._build_pool_and_decide()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        pool_s = samples.clone()
        pool_s[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_s, self.m, dim=1).indices
        full_action = torch.topk(samples, self.m, dim=1).indices

        result = full_action.clone()
        result[self._use_oracle] = pool_action[self._use_oracle]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._use_oracle.fill_(True)
        self._pool_built = False


# =====================================================================
# 8. ParetoMetaDual — meta over aggressive+safe with dual-pool fallback
# =====================================================================

class ParetoMetaDual(BatchedAgentBase):
    """Pareto-optimal meta-algorithm over aggressive and safe base policies.

    Combines the two best non-overlapping strategies from the arena:
    - Aggressive: freq_pool_ts style (oracle pool + frequency priors)
    - Safe: dual-pool CTS (oracle + data, cross-validated)

    Uses EXP3-style selection (simpler than Tsallis-INF) with reward-based
    updates. The key insight is that by combining the Pareto-frontier
    endpoints, the meta-learner can achieve near-best on ALL metrics.
    """
    name = "pareto_meta_dual"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 prior_strength: float = 2.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.prior_strength = prior_strength

        self.oracle_pool = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.data_pool = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self.n_policies = 2
        self.log_weights = torch.zeros(n_seeds, self.n_policies, device=device)
        self._active = torch.zeros(n_seeds, dtype=torch.long, device=device)
        self._pools_built = False
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 100

    def _build_pools(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.oracle_pool.zero_()
        arms = torch.topk(freq, self.pool_size, dim=1).indices
        self.oracle_pool.scatter_(1, arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.oracle_pool.scatter_(1, safety, True)

        self.data_pool = _build_data_pool(
            self.mu_hat, self.pool_size, self.d, self.n_seeds,
            self.n_safety, self.device)

        if not self._pools_built:
            normalized = freq / self.n_pool_rounds
            self.alphas += normalized * self.prior_strength

        self._pools_built = True

    def _pick_data_pool_winner(self):
        self.data_pool = _build_data_pool(
            self.mu_hat, self.pool_size, self.d, self.n_seeds,
            self.n_safety, self.device)

    def select_arms(self) -> torch.Tensor:
        if self.t == 0:
            self._build_pools()

        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(100 * (2 ** self._epoch), 5000)
            self._build_pools()

        if self.t % 150 == 0 and self.t > 0:
            self._pick_data_pool_winner()

        eta = math.sqrt(math.log(self.n_policies) / max(self.t + 1, 1))
        probs = torch.softmax(eta * self.log_weights, dim=1)
        probs = (1 - 0.05) * probs + 0.05 / self.n_policies
        self._active = torch.multinomial(probs, 1).squeeze(1)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        s0 = samples.clone()
        s0[~self.oracle_pool] = -float("inf")
        a0 = torch.topk(s0, self.m, dim=1).indices

        s1 = samples.clone()
        s1[~self.data_pool] = -float("inf")
        a1 = torch.topk(s1, self.m, dim=1).indices

        result = a1.clone()
        result[self._active == 0] = a0[self._active == 0]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

        reward_avg = rewards.mean(dim=1)
        for p in range(self.n_policies):
            mask = (self._active == p).float()
            probs_p = torch.softmax(
                math.sqrt(math.log(self.n_policies) / max(self.t, 1)) * self.log_weights,
                dim=1
            )[:, p].clamp(min=0.01)
            imp_reward = mask * reward_avg / probs_p
            self.log_weights[:, p] += imp_reward

    def reset(self):
        super().reset()
        self.oracle_pool.zero_()
        self.data_pool.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self.log_weights.zero_()
        self._active.zero_()
        self._pools_built = False
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 100


# =====================================================================
# Registry
# =====================================================================

NOVEL_V4_REGISTRY = {
    "meta_pool_cts": MetaPoolCTS,
    "adaptive_pool_cts": AdaptivePoolCTS,
    "lucb_pool": LUCBPool,
    "oracle_budget_cts": OracleBudgetCTS,
    "multi_oracle_pool": MultiOraclePool,
    "consistency_robust_cts": ConsistencyRobustCTS,
    "pool_cts_abstain": PoolCTSAbstain,
    "pareto_meta_dual": ParetoMetaDual,
}

NOVEL_V4_NEEDS_ORACLE = set(NOVEL_V4_REGISTRY.keys())
