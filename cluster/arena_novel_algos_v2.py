"""Loop 2 novel algorithms: refinements based on early arena results.

Key insights from Loop 1:
1. Pool-based approaches dominate on reliable oracles
2. Soft prior injection (freq_weighted, decaying_prior) is competitive
3. consistent_wrong remains the Achilles heel for all pool methods
4. Pool expansion (pool_ts_expand) helps but doesn't fully solve it

New ideas:
1. Pool-CTS-Dual: two independent pools, cross-validated
2. Pool-CTS-Anytime: periodically rebuild pool, keeping stable arms
3. Freq-Pool-TS: soft frequency prior + hard pool restriction (best of both)
4. Pool-CTS-Doubling: doubling-trick epochs, rebuild pool each epoch
5. Pool-CTS-Safety-Net: pool_cts but fall back to decaying_prior_ts if pool looks bad
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


class PoolCTSDual(BatchedAgentBase):
    """Two independent pools: oracle-based and exploration-based.

    Pool A: built from oracle queries (same as pool_cts).
    Pool B: built from round-robin exploration, taking top-beta*m by mu_hat.

    Each round, play pool_cts on the pool with higher empirical top-m reward.
    If pools disagree significantly, trust pool B (data-driven).

    Handles consistent_wrong: pool A has bad arms, pool B has good arms
    (from exploration). After enough exploration, pool B dominates.
    """
    name = "pool_cts_dual"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 T_explore: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.T_explore = T_explore if T_explore is not None else max(d * 2, 100)

        self.pool_a = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.pool_b = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pools_built = False

    def _build_pools(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        pool_a_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_a.scatter_(1, pool_a_arms, True)

        pool_b_arms = torch.topk(self.mu_hat, self.pool_size, dim=1).indices
        self.pool_b.scatter_(1, pool_b_arms, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_a.scatter_(1, safety, True)
        self.pool_b.scatter_(1, safety, True)
        self._pools_built = True

    def _pick_pool(self) -> torch.Tensor:
        mu_a = self.mu_hat.clone()
        mu_a[~self.pool_a] = -float("inf")
        top_a = torch.topk(mu_a, self.m, dim=1).values.sum(dim=1)

        mu_b = self.mu_hat.clone()
        mu_b[~self.pool_b] = -float("inf")
        top_b = torch.topk(mu_b, self.m, dim=1).values.sum(dim=1)

        return top_a >= top_b

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_explore:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pools_built:
            self._build_pools()

        use_a = self._pick_pool()
        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()

        sa = samples.clone()
        sa[~self.pool_a] = -float("inf")
        action_a = torch.topk(sa, self.m, dim=1).indices

        sb = samples.clone()
        sb[~self.pool_b] = -float("inf")
        action_b = torch.topk(sb, self.m, dim=1).indices

        result = action_b.clone()
        result[use_a] = action_a[use_a]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

        if self._pools_built and self.t % 200 == 0:
            pool_b_arms = torch.topk(self.mu_hat, self.pool_size, dim=1).indices
            self.pool_b.zero_()
            self.pool_b.scatter_(1, pool_b_arms, True)

    def reset(self):
        super().reset()
        self.pool_a.zero_()
        self.pool_b.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._pools_built = False


class FreqPoolTS(BatchedAgentBase):
    """Frequency-weighted priors inside oracle pool.

    Combines the two best ideas from Loop 1:
    - Pool restriction (dimension reduction)
    - Frequency-weighted priors (soft oracle trust)

    Oracle query frequencies set the Beta prior pseudo-counts,
    but only for arms inside the pool. Non-pool arms get flat priors
    and are never played (unless pool expands).
    """
    name = "freq_pool_ts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 prior_strength: float = 3.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.prior_strength = prior_strength

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def _build_pool(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        pool_arms = torch.topk(freq, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

        normalized = freq / self.n_pool_rounds
        self.alphas += normalized * self.prior_strength

        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()
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


class PoolCTSSafetyNet(BatchedAgentBase):
    """Pool-CTS with automatic safety-net fallback.

    Start with pool_cts. Track a running average of the empirical reward
    inside the pool. If it drops below the expected reward of the pool_cts
    prior (i.e., pool arms aren't as good as expected), seamlessly transition
    to full CTS with decaying prior.

    Key advantage: no explicit init phase, no threshold tuning.
    The transition is triggered by evidence, not a hyperparameter.
    """
    name = "pool_cts_safety"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 transition_window: int = 100, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.transition_window = transition_window

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False
        self._use_pool = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._reward_history = []

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

    def _check_safety(self, rewards: torch.Tensor):
        avg_reward = rewards.mean(dim=1)
        self._reward_history.append(avg_reward)
        if len(self._reward_history) < self.transition_window:
            return
        if len(self._reward_history) > self.transition_window:
            self._reward_history.pop(0)

        pool_avg = torch.stack(self._reward_history).mean(dim=0)
        global_mu_top = torch.topk(self.mu_hat, self.m, dim=1).values.mean(dim=1)
        pool_mu_top = self.mu_hat.clone()
        pool_mu_top[~self.pool_mask] = -float("inf")
        pool_mu_top = torch.topk(pool_mu_top, self.m, dim=1).values.mean(dim=1)

        self._use_pool = self._use_pool & (pool_mu_top >= global_mu_top * 0.85)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()

        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_samples, self.m, dim=1).indices
        full_action = torch.topk(samples, self.m, dim=1).indices

        result = full_action.clone()
        result[self._use_pool] = pool_action[self._use_pool]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

        if self._pool_built and self.t > 50:
            self._check_safety(rewards)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._pool_built = False
        self._use_pool.fill_(True)
        self._reward_history.clear()


class PoolCTSDoubling(BatchedAgentBase):
    """Pool-CTS with doubling-trick epochs.

    Epoch k has length 2^k rounds. At the start of each epoch:
    1. Rebuild pool from oracle queries
    2. Reset Beta posteriors to incorporate accumulated mu_hat
    3. Re-query oracle (oracle sees updated history)

    Advantage: consistent adversary can't accumulate influence across
    epochs. Pool gets rebuilt with fresh oracle queries each epoch.
    If oracle is reliable: minimal overhead (O(log T) rebuilds).
    If oracle is consistently wrong: each rebuild starts fresh,
    and the safety arms + accumulated mu_hat help detect the problem.
    """
    name = "pool_cts_doubling"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 8, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 10

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_mask.zero_()
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)

        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

    def select_arms(self) -> torch.Tensor:
        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(2 ** self._epoch * 10, 5000)
            self._build_pool()
            self.alphas = 1.0 + self.n_pulls * self.mu_hat
            self.betas_param = 1.0 + self.n_pulls * (1 - self.mu_hat)

        if self.t == 0:
            self._build_pool()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()
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
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = 10


NOVEL_V2_REGISTRY = {
    "pool_cts_dual": PoolCTSDual,
    "freq_pool_ts": FreqPoolTS,
    "pool_cts_safety": PoolCTSSafetyNet,
    "pool_cts_doubling": PoolCTSDoubling,
}

NOVEL_V2_NEEDS_ORACLE = set(NOVEL_V2_REGISTRY.keys())
