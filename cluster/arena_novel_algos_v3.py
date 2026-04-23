"""Loop 2 hybrid algorithms: combining the best of Loop 1.

Key insight from Loop 1 arena (30 configs, T=3000):
- pool_cts_doubling: best overall mean (595), principled epoch robustness
- pool_cts_dual: best on consistent_wrong (541), cross-validates oracle vs data
- freq_pool_ts: best win rate (27%), soft priors + hard pool
- pool_cts_safety: highest win rate (33%), auto-detects bad pools

Loop 2 hybrids combine these strengths:
1. Pool-CTS-Dual-Doubling: dual pools + epoch resets
2. Freq-Pool-CTS-Dual: frequency priors + dual pools
3. Pool-CTS-Adaptive-Doubling: doubling epochs + safety-net fallback
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


class PoolCTSDualDoubling(BatchedAgentBase):
    """THE HERO CANDIDATE: Dual pools with doubling-trick epochs.

    Combines the two strongest approaches from Loop 1:
    - Dual pools (oracle-based Pool A + data-based Pool B) for robust
      cross-validation — catches consistent_wrong
    - Doubling epochs that rebuild both pools periodically — prevents
      accumulation of bad oracle influence

    Each epoch k (length 2^k * base):
    1. Rebuild Pool A from fresh oracle queries
    2. Rebuild Pool B from top arms by empirical mean
    3. Pick the pool with higher empirical top-m reward
    4. Run CTS on the winning pool

    Theoretical sketch:
    - Reliable oracle: Pool A contains S*, CTS converges in O(m^2 log T / Delta)
    - Consistent_wrong: Pool B eventually contains S* after O(d/Delta^2) samples,
      then CTS converges. Epoch rebuild ensures Pool B stays fresh.
    - O(log T) epoch rebuilds total, each costs O(K) oracle queries.
    """
    name = "pool_cts_dual_doubling"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 epoch_base: int = 50, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.epoch_base = epoch_base

        self.pool_a = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.pool_b = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.active_pool = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = epoch_base
        self._using_a = torch.ones(n_seeds, dtype=torch.bool, device=device)

    def _build_pools(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_a.zero_()
        pool_a_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_a.scatter_(1, pool_a_arms, True)

        self.pool_b.zero_()
        pool_b_arms = torch.topk(self.mu_hat, self.pool_size, dim=1).indices
        self.pool_b.scatter_(1, pool_b_arms, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_a.scatter_(1, safety, True)
        self.pool_b.scatter_(1, safety, True)

        self._pick_pool()

    def _pick_pool(self):
        mu_a = self.mu_hat.clone()
        mu_a[~self.pool_a] = -float("inf")
        top_a = torch.topk(mu_a, self.m, dim=1).values.sum(dim=1)

        mu_b = self.mu_hat.clone()
        mu_b[~self.pool_b] = -float("inf")
        top_b = torch.topk(mu_b, self.m, dim=1).values.sum(dim=1)

        self._using_a = top_a >= top_b
        self.active_pool = torch.where(
            self._using_a.unsqueeze(1).expand_as(self.pool_a),
            self.pool_a, self.pool_b
        )

    def _maybe_new_epoch(self):
        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(self.epoch_base * (2 ** self._epoch), 10000)
            self._build_pools()
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1.0 - self.mu_hat)).clamp(min=0)

    def select_arms(self) -> torch.Tensor:
        if self.t == 0:
            self._build_pools()

        self._maybe_new_epoch()

        if self.t % 100 == 0 and self.t > 0:
            self._pick_pool()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()
        samples[~self.active_pool] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_a.zero_()
        self.pool_b.zero_()
        self.active_pool.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = self.epoch_base
        self._using_a.fill_(True)


class FreqPoolCTSDual(BatchedAgentBase):
    """Frequency-weighted priors inside dual pools.

    Combines freq_pool_ts (best win rate) with pool_cts_dual (best consistent_wrong):
    - Build oracle pool with frequency-weighted priors
    - Build data pool from exploration
    - Cross-validate and pick the better pool
    - Frequency priors boost convergence on the winning pool
    """
    name = "freq_pool_cts_dual"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 prior_strength: float = 3.0, T_explore: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.prior_strength = prior_strength
        self.T_explore = T_explore if T_explore is not None else max(d * 2, 100)

        self.pool_a = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.pool_b = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pools_built = False
        self._using_a = torch.ones(n_seeds, dtype=torch.bool, device=device)

    def _build_pools(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        pool_a_arms = torch.topk(freq, self.pool_size, dim=1).indices
        self.pool_a.scatter_(1, pool_a_arms, True)

        normalized = freq / self.n_pool_rounds
        self.alphas += normalized * self.prior_strength

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

        self._using_a = top_a >= top_b
        return self._using_a

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_explore:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pools_built:
            self._build_pools()

        if self.t % 100 == 0:
            self._pick_pool()

        if self.t % 200 == 0 and self._pools_built:
            pool_b_arms = torch.topk(self.mu_hat, self.pool_size, dim=1).indices
            self.pool_b.zero_()
            self.pool_b.scatter_(1, pool_b_arms, True)

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()

        sa = samples.clone()
        sa[~self.pool_a] = -float("inf")
        action_a = torch.topk(sa, self.m, dim=1).indices

        sb = samples.clone()
        sb[~self.pool_b] = -float("inf")
        action_b = torch.topk(sb, self.m, dim=1).indices

        result = action_b.clone()
        result[self._using_a] = action_a[self._using_a]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_a.zero_()
        self.pool_b.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._pools_built = False
        self._using_a.fill_(True)


class PoolCTSAdaptiveDoubling(BatchedAgentBase):
    """Doubling epochs + safety-net auto-fallback.

    Within each epoch, run pool_cts on oracle pool. Monitor empirical reward.
    If pool top-m empirical reward drops below global top-m, switch to full CTS
    for the rest of the epoch. At epoch boundary, try oracle pool again.

    This gives the oracle a fresh chance each epoch (handles non-stationary),
    while the safety-net catches bad pools within each epoch.
    """
    name = "pool_cts_adapt_doubling"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 epoch_base: int = 50, safety_threshold: float = 0.85, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.epoch_base = epoch_base
        self.safety_threshold = safety_threshold

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = epoch_base
        self._use_pool = torch.ones(n_seeds, dtype=torch.bool, device=device)

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_mask.zero_()
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

        self._use_pool.fill_(True)

    def _maybe_new_epoch(self):
        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(self.epoch_base * (2 ** self._epoch), 10000)
            self._build_pool()
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1.0 - self.mu_hat)).clamp(min=0)

    def _check_safety(self):
        pool_mu = self.mu_hat.clone()
        pool_mu[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_mu, self.m, dim=1).values.sum(dim=1)
        global_top = torch.topk(self.mu_hat, self.m, dim=1).values.sum(dim=1)
        self._use_pool = self._use_pool & (pool_top >= global_top * self.safety_threshold)

    def select_arms(self) -> torch.Tensor:
        if self.t == 0:
            self._build_pool()

        self._maybe_new_epoch()

        if self.t % 50 == 0 and self.t > 20:
            self._check_safety()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

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

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = self.epoch_base
        self._use_pool.fill_(True)


NOVEL_V3_REGISTRY = {
    "pool_cts_dual_doubling": PoolCTSDualDoubling,
    "freq_pool_cts_dual": FreqPoolCTSDual,
    "pool_cts_adapt_doubling": PoolCTSAdaptiveDoubling,
}

NOVEL_V3_NEEDS_ORACLE = set(NOVEL_V3_REGISTRY.keys())
