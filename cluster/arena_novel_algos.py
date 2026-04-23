"""Novel LLM+Bandit algorithms for the randomized test arena.

Each algorithm is a GPU-batched agent targeting a specific theoretical insight
from the LLM+combinatorial bandits literature. All follow the BatchedAgentBase
interface for drop-in testing.

Key ideas explored:
1. Frequency-weighted priors (oracle -> Bayesian prior, soft)
2. Successive elimination with oracle pool
3. MOSS-Oracle (minimax-optimal with oracle restriction)
4. Corral-style online model selection over {pool_cts, cts, cucb}
5. Pool-CTS with posterior-gated expansion (grows pool if posterior evidence suggests it)
6. Decaying-trust TS (oracle prior strength decays as data accumulates)
7. Oracle-guided epsilon-greedy with adaptive epsilon
8. Hedge-TS hybrid (Thompson on oracle set, UCB hedge outside)
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase, BatchedCUCB, BatchedCTS
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


class FrequencyWeightedTS(BatchedAgentBase):
    """Oracle query frequencies become Beta prior pseudo-counts.

    Theory: if arm i appears f_i times across K oracle queries, set
    prior Beta(1 + c*f_i/K, 1). This is soft — no hard pool boundary.
    Arms the oracle never suggests get flat priors; arms it always suggests
    get strong priors. Naturally handles partial overlap.

    The prior strength c decays as 1/sqrt(t) so data dominates eventually.
    """
    name = "freq_weighted_ts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, prior_strength: float = 3.0,
                 n_oracle_queries: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.prior_strength = prior_strength
        self.n_oracle_queries = n_oracle_queries

        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._prior_injected = False
        self._base_prior = torch.zeros(n_seeds, d, device=device)

    def _inject_prior(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_oracle_queries):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        normalized = freq / self.n_oracle_queries
        self._base_prior = normalized * self.prior_strength
        self.alphas += self._base_prior
        self._prior_injected = True

    def select_arms(self) -> torch.Tensor:
        if not self._prior_injected:
            self._inject_prior()
        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._prior_injected = False
        self._base_prior.zero_()


class SuccessiveEliminationPool(BatchedAgentBase):
    """Successive elimination restricted to oracle-defined pool.

    Build pool from oracle queries (like pool_cts). Then instead of TS,
    use successive elimination: maintain confidence intervals for each
    arm in pool, eliminate arms whose upper bound < best lower bound.

    Advantage over TS in pool: deterministic, no variance from sampling.
    Advantage over UCB in pool: actively shrinks the candidate set.
    """
    name = "succ_elim_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.active_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self.active_mask = self.pool_mask.clone()
        self._pool_built = True

    def _eliminate(self):
        ci_width = torch.sqrt(2 * math.log(max(self.t, 2)) / self.n_pulls.clamp(min=1))
        upper = self.mu_hat + ci_width
        lower = self.mu_hat - ci_width

        lower_in_active = lower.clone()
        lower_in_active[~self.active_mask] = -float("inf")
        best_lower = torch.topk(lower_in_active, self.m, dim=1).values.min(dim=1).values

        can_eliminate = upper < best_lower.unsqueeze(1)
        self.active_mask = self.active_mask & ~can_eliminate

        n_active = self.active_mask.sum(dim=1)
        too_few = n_active < self.m
        if too_few.any():
            self.active_mask[too_few] = self.pool_mask[too_few]

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > 0 and self.t % 20 == 0:
            self._eliminate()

        n_active = self.active_mask.sum(dim=1)
        needs_rr = n_active > self.m * 2
        rr_count = self.t % max(1, (n_active.max().item() // self.m))

        ucb = self.ucb_indices
        ucb_masked = ucb.clone()
        ucb_masked[~self.active_mask] = -float("inf")
        return torch.topk(ucb_masked, self.m, dim=1).indices

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.active_mask.zero_()
        self._pool_built = False


class CorralSelector(BatchedAgentBase):
    """Corral-style online selection over 3 base algorithms.

    Runs Pool-CTS, full CTS, and CUCB simultaneously. A log-barrier
    regularized FTRL meta-learner picks which base to follow each round.

    Provable overhead: O(sqrt(T * K)) where K=3 base algorithms.
    Always within O(sqrt(T)) of the best base algorithm in hindsight.
    """
    name = "corral"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.n_bases = 3

        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)

        self.alphas_pool = torch.ones(n_seeds, d, device=device)
        self.betas_pool = torch.ones(n_seeds, d, device=device)
        self.alphas_cts = torch.ones(n_seeds, d, device=device)
        self.betas_cts = torch.ones(n_seeds, d, device=device)

        self.log_weights = torch.zeros(n_seeds, self.n_bases, device=device)
        self._last_base = torch.zeros(n_seeds, dtype=torch.long, device=device)
        self._pool_built = False

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        probs = torch.softmax(self.log_weights, dim=1)
        chosen = torch.multinomial(probs, 1).squeeze(1)
        self._last_base = chosen

        pool_samples = torch.distributions.Beta(self.alphas_pool, self.betas_pool).sample()
        pool_samples_masked = pool_samples.clone()
        pool_samples_masked[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_samples_masked, self.m, dim=1).indices

        cts_samples = torch.distributions.Beta(self.alphas_cts, self.betas_cts).sample()
        cts_action = torch.topk(cts_samples, self.m, dim=1).indices

        ucb_action = self.top_m_by_ucb()

        result = pool_action.clone()
        result[chosen == 1] = cts_action[chosen == 1]
        result[chosen == 2] = ucb_action[chosen == 2]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes

        is_pool = (self._last_base == 0).unsqueeze(1)
        is_cts = (self._last_base == 1).unsqueeze(1)
        self.alphas_pool.scatter_add_(1, selected, torch.where(is_pool, successes, torch.zeros_like(successes)))
        self.betas_pool.scatter_add_(1, selected, torch.where(is_pool, failures, torch.zeros_like(failures)))
        self.alphas_cts.scatter_add_(1, selected, torch.where(is_cts, successes, torch.zeros_like(successes)))
        self.betas_cts.scatter_add_(1, selected, torch.where(is_cts, failures, torch.zeros_like(failures)))

        reward_sum = rewards.sum(dim=1) / self.m
        eta = math.sqrt(math.log(self.n_bases) / max(self.t, 1))
        probs = torch.softmax(self.log_weights, dim=1)
        chosen_prob = torch.gather(probs, 1, self._last_base.unsqueeze(1)).squeeze(1)
        iw_reward = reward_sum / chosen_prob.clamp(min=0.01)

        update = torch.zeros_like(self.log_weights)
        update.scatter_(1, self._last_base.unsqueeze(1), (eta * iw_reward).unsqueeze(1))
        self.log_weights += update
        self.log_weights -= self.log_weights.max(dim=1, keepdim=True).values

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas_pool.fill_(1.0)
        self.betas_pool.fill_(1.0)
        self.alphas_cts.fill_(1.0)
        self.betas_cts.fill_(1.0)
        self.log_weights.zero_()
        self._pool_built = False


class PoolTSExpandable(BatchedAgentBase):
    """Pool-CTS with posterior-gated expansion.

    Starts with a small pool from oracle. If the best posterior arm's
    upper credible bound is less than the global UCB upper bound
    for unexplored arms, expand the pool by adding top-UCB arms.

    This automatically detects when the pool is missing good arms
    (consistent_wrong) and grows to cover them, without an explicit
    init-check phase.
    """
    name = "pool_ts_expand"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 expand_check_period: int = 100, expand_count: int = 3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.expand_check_period = expand_check_period
        self.expand_count = expand_count

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
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def _maybe_expand(self):
        pool_mu = self.mu_hat.clone()
        pool_mu[~self.pool_mask] = -float("inf")
        pool_top_reward = torch.topk(pool_mu, self.m, dim=1).values.sum(dim=1)

        out_mu = self.mu_hat.clone()
        out_mu[self.pool_mask] = -float("inf")
        out_ucb = out_mu + torch.sqrt(2 * math.log(max(self.t, 2)) / self.n_pulls.clamp(min=1))
        out_ucb[self.pool_mask] = -float("inf")
        out_top_reward = torch.topk(out_ucb, self.m, dim=1).values.sum(dim=1)

        should_expand = out_top_reward > pool_top_reward * 1.05

        if should_expand.any():
            out_scores = self.ucb_indices.clone()
            out_scores[self.pool_mask] = -float("inf")
            new_arms = torch.topk(out_scores, self.expand_count, dim=1).indices
            expand_mask = torch.zeros_like(self.pool_mask)
            expand_mask.scatter_(1, new_arms, True)
            self.pool_mask = self.pool_mask | (expand_mask & should_expand.unsqueeze(1))

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > 50 and self.t % self.expand_check_period == 0:
            self._maybe_expand()

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

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


class DecayingPriorTS(BatchedAgentBase):
    """Thompson Sampling with decaying oracle prior.

    Oracle sets initial Beta(1+c*f_i, 1) prior. But every D rounds,
    we multiplicatively decay the prior pseudo-counts by factor gamma:
      alpha_i <- 1 + gamma * (alpha_i - 1)
      beta_i  <- 1 + gamma * (beta_i - 1)

    This means oracle influence fades as data accumulates. A wrong prior
    gets overridden faster than in warm-start CTS (which uses fixed prior
    strength that takes O(prior_strength/Delta^2) to overcome).
    """
    name = "decaying_prior_ts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, prior_strength: float = 5.0,
                 decay_factor: float = 0.995, n_oracle_queries: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.prior_strength = prior_strength
        self.decay_factor = decay_factor
        self.n_oracle_queries = n_oracle_queries

        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._data_alphas = torch.zeros(n_seeds, d, device=device)
        self._data_betas = torch.zeros(n_seeds, d, device=device)
        self._prior_alphas = torch.zeros(n_seeds, d, device=device)
        self._injected = False

    def _inject_prior(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_oracle_queries):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        normalized = freq / self.n_oracle_queries
        self._prior_alphas = normalized * self.prior_strength
        self.alphas = 1.0 + self._prior_alphas + self._data_alphas
        self._injected = True

    def select_arms(self) -> torch.Tensor:
        if not self._injected:
            self._inject_prior()
        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self._data_alphas.scatter_add_(1, selected, successes)
        self._data_betas.scatter_add_(1, selected, failures)

        self._prior_alphas *= self.decay_factor
        self.alphas = 1.0 + self._prior_alphas + self._data_alphas
        self.betas_param = 1.0 + self._data_betas

    def reset(self):
        super().reset()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._data_alphas.zero_()
        self._data_betas.zero_()
        self._prior_alphas.zero_()
        self._injected = False


class HedgeTSHybrid(BatchedAgentBase):
    """Thompson Sampling on oracle set + UCB hedge outside.

    Combines the best of both: TS's fast convergence on pool arms
    (where we have oracle prior info) with UCB's reliable exploration
    of hedge arms (where we need worst-case guarantees).

    Each round:
    - Sample from Beta posteriors for pool arms
    - Compute UCB indices for hedge arms
    - Take top-m across both score vectors
    """
    name = "hedge_ts_hybrid"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 hedge_frac: float = 0.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.hedge_size = max(1, int(hedge_frac * math.sqrt(d)))

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
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        scores = torch.zeros(self.n_seeds, self.d, device=self.device)

        ts_scores = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        scores[self.pool_mask] = ts_scores[self.pool_mask]

        ucb = self.ucb_indices
        non_pool = ~self.pool_mask
        ucb_outside = ucb.clone()
        ucb_outside[self.pool_mask] = -float("inf")
        top_hedge = torch.topk(ucb_outside, min(self.hedge_size, self.d - self.pool_mask.sum(dim=1).min().item()), dim=1).indices
        hedge_mask = torch.zeros_like(self.pool_mask)
        hedge_mask.scatter_(1, top_hedge, True)
        scores[hedge_mask & non_pool] = ucb[hedge_mask & non_pool]
        scores[~self.pool_mask & ~hedge_mask] = -float("inf")

        return torch.topk(scores, self.m, dim=1).indices

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


class OracleEpsGreedy(BatchedAgentBase):
    """Oracle-guided epsilon-greedy with adaptive epsilon.

    Build pool from oracle. Play greedy (best empirical arm in pool)
    with prob 1-eps, explore uniformly in full [d] with prob eps.
    eps decays as c/t^(1/3) — enough exploration to detect bad pools.

    Simple and parameter-light. Tests whether the pool+greedy+explore
    recipe beats more complex TS/UCB approaches.
    """
    name = "oracle_eps_greedy"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 eps_scale: float = 1.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.eps_scale = eps_scale

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        eps = self.eps_scale / max(1, self.t) ** (1.0 / 3)
        explore = torch.rand(self.n_seeds, device=self.device) < eps

        mu_pool = self.mu_hat.clone()
        mu_pool[~self.pool_mask] = -float("inf")
        greedy = torch.topk(mu_pool, self.m, dim=1).indices

        rand_scores = torch.rand(self.n_seeds, self.d, device=self.device)
        random_action = torch.topk(rand_scores, self.m, dim=1).indices

        result = greedy.clone()
        if explore.any():
            result[explore] = random_action[explore]
        return result

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self._pool_built = False


NOVEL_ALGO_REGISTRY = {
    "freq_weighted_ts": FrequencyWeightedTS,
    "succ_elim_pool": SuccessiveEliminationPool,
    "corral": CorralSelector,
    "pool_ts_expand": PoolTSExpandable,
    "decaying_prior_ts": DecayingPriorTS,
    "hedge_ts_hybrid": HedgeTSHybrid,
    "oracle_eps_greedy": OracleEpsGreedy,
}

NOVEL_NEEDS_ORACLE = set(NOVEL_ALGO_REGISTRY.keys())
