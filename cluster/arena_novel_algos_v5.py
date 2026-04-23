"""Loop 3 hybrids: combining Loop 3 winners.

Key finding from Loop 3 arena (29 algos, 30 configs):
- adaptive_pool_cts: 67% win rate, #1 on adversarial/partial/uniform, LAST on consistent_wrong
- freq_pool_cts_dual: best consistent_wrong (546), best worst-case (1647)
- pool_cts_abstain: #4 overall, decent consistent_wrong (935)
- oracle_budget_cts: #3 overall, efficient query usage

The gap: adaptive_pool_cts needs consistent_wrong protection.
Hybrids combine adaptive sizing with data-pool fallback.
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


class AdaptivePoolDual(BatchedAgentBase):
    """THE HYBRID HERO: Adaptive pool sizing + dual-pool cross-validation.

    Combines adaptive_pool_cts (67% win rate) with pool_cts_dual
    (best consistent_wrong). Maintains two pools:
    - Pool A: oracle-based with adaptive sizing (starts small, grows if needed)
    - Pool B: data-based from empirical top arms (always correct eventually)

    Cross-validates: if Pool B's top-m reward exceeds Pool A's, switch to B.
    This catches consistent_wrong (where Pool A grows toward bad arms)
    while preserving adaptive_pool_cts's dominance elsewhere.
    """
    name = "adaptive_pool_dual"

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
        self.pool_a = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.pool_b = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._using_a = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._pool_built = False

    def _build_pools(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_a.zero_()
        for seed_idx in range(self.n_seeds):
            ps = int(self.beta[seed_idx].item() * self.m)
            ps = min(ps, self.d)
            arms = torch.topk(pool_counts[seed_idx], ps).indices
            self.pool_a[seed_idx].scatter_(0, arms, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_a.scatter_(1, safety, True)

        pool_size_b = int(self.beta_init * self.m)
        self.pool_b.zero_()
        b_arms = torch.topk(self.mu_hat, min(pool_size_b, self.d), dim=1).indices
        self.pool_b.scatter_(1, b_arms, True)
        self.pool_b.scatter_(1, safety, True)

        self._pool_built = True

    def _check_coverage_and_cross_validate(self):
        emp_top = torch.topk(self.mu_hat, self.m, dim=1).indices
        covered = torch.zeros(self.n_seeds, device=self.device)
        for i in range(self.m):
            arm = emp_top[:, i]
            in_pool = self.pool_a.gather(1, arm.unsqueeze(1)).squeeze(1)
            covered += in_pool.float()
        coverage_frac = covered / self.m

        need_expand = coverage_frac < 0.6
        self.beta = torch.where(
            need_expand,
            (self.beta * 1.5).clamp(max=self.beta_max),
            self.beta
        )

        mu_a = self.mu_hat.clone()
        mu_a[~self.pool_a] = -float("inf")
        top_a = torch.topk(mu_a, self.m, dim=1).values.sum(dim=1)

        pool_size_b = int(self.beta_init * self.m)
        self.pool_b.zero_()
        b_arms = torch.topk(self.mu_hat, min(pool_size_b, self.d), dim=1).indices
        self.pool_b.scatter_(1, b_arms, True)

        mu_b = self.mu_hat.clone()
        mu_b[~self.pool_b] = -float("inf")
        top_b = torch.topk(mu_b, self.m, dim=1).values.sum(dim=1)

        self._using_a = top_a >= top_b * 0.95

        if need_expand.any():
            self._build_pools()

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pools()

        if self.t > 0 and self.t % self.check_interval == 0:
            self._check_coverage_and_cross_validate()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

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
        self.beta.fill_(self.beta_init)
        self.pool_a.zero_()
        self.pool_b.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._using_a.fill_(True)
        self._pool_built = False


class AdaptivePoolAbstain(BatchedAgentBase):
    """Adaptive pool sizing + abstention mechanism.

    Combines adaptive_pool_cts (67% win rate) with pool_cts_abstain:
    - Adaptive sizing for optimal pool when oracle is helpful
    - Abstention: if oracle-empirical agreement is low after warmup,
      fall back to full CTS (no pool restriction)

    Catches consistent_wrong via the agreement test while preserving
    adaptive sizing's advantage on all other corruption types.
    """
    name = "adaptive_pool_abstain"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta_init: float = 2.0,
                 beta_max: float = 8.0, n_safety: int = 5,
                 n_pool_rounds: int = 10, T_warmup: int | None = None,
                 abstain_threshold: float = 0.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.T_warmup = T_warmup if T_warmup is not None else max(d // m * 3, 30)
        self.abstain_threshold = abstain_threshold

        self.beta = torch.full((n_seeds,), beta_init, device=device)
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
            self._build_pool_and_decide()

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_warmup:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pool_built:
            self._build_pool_and_decide()

        if self.t > 0 and self.t % 150 == 0:
            self._check_coverage()

        if self.t > 0 and self.t % 300 == 0:
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
        self.beta.fill_(self.beta_init)
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._use_oracle.fill_(True)
        self._pool_built = False


class AdaptivePoolDoubling(BatchedAgentBase):
    """Adaptive pool sizing + doubling epochs.

    Combines adaptive_pool_cts with pool_cts_doubling:
    - Adaptive pool size that grows when coverage is low
    - Doubling epochs that rebuild the pool periodically
    - Each epoch: re-query oracle, re-assess pool size, rebuild

    The doubling prevents long-run damage from a bad pool,
    while adaptive sizing keeps the pool tight when oracle is good.
    """
    name = "adaptive_pool_doubling"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta_init: float = 2.0,
                 beta_max: float = 6.0, n_safety: int = 5,
                 n_pool_rounds: int = 10, epoch_base: int = 50, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.epoch_base = epoch_base

        self.beta = torch.full((n_seeds,), beta_init, device=device)
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)

        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = epoch_base

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

        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

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

    def select_arms(self) -> torch.Tensor:
        if self.t == 0:
            self._build_pool()

        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(self.epoch_base * (2 ** self._epoch), 5000)
            self._check_coverage()
            self._build_pool()
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1.0 - self.mu_hat)).clamp(min=0)

        if self.t > 0 and self.t % 100 == 0:
            self._check_coverage()

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
        self.beta.fill_(self.beta_init)
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = self.epoch_base


class AdaptiveFreqDual(BatchedAgentBase):
    """Adaptive sizing + frequency priors + dual pools.

    The ultimate combination: takes the three strongest mechanisms and
    merges them:
    - Adaptive pool sizing from adaptive_pool_cts (67% win rate)
    - Frequency-weighted priors from freq_pool_ts (37% win rate in Loop 2)
    - Dual pools from pool_cts_dual (best consistent_wrong)

    Pool A: oracle-based, adaptive size, frequency priors
    Pool B: data-based, fixed size
    Cross-validate and pick winner.
    """
    name = "adaptive_freq_dual"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta_init: float = 2.0,
                 beta_max: float = 8.0, n_safety: int = 5,
                 n_pool_rounds: int = 10, prior_strength: float = 2.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.prior_strength = prior_strength

        self.beta = torch.full((n_seeds,), beta_init, device=device)
        self.pool_a = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.pool_b = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._using_a = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._pool_built = False

    def _build_pools(self):
        freq = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            freq.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        self.pool_a.zero_()
        for seed_idx in range(self.n_seeds):
            ps = int(self.beta[seed_idx].item() * self.m)
            ps = min(ps, self.d)
            arms = torch.topk(freq[seed_idx], ps).indices
            self.pool_a[seed_idx].scatter_(0, arms, True)

        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_a.scatter_(1, safety, True)

        if not self._pool_built:
            normalized = freq / self.n_pool_rounds
            self.alphas += normalized * self.prior_strength

        pool_size_b = int(3.0 * self.m)
        self.pool_b.zero_()
        b_arms = torch.topk(self.mu_hat, min(pool_size_b, self.d), dim=1).indices
        self.pool_b.scatter_(1, b_arms, True)
        self.pool_b.scatter_(1, safety, True)

        self._pool_built = True

    def _cross_validate(self):
        mu_a = self.mu_hat.clone()
        mu_a[~self.pool_a] = -float("inf")
        top_a = torch.topk(mu_a, self.m, dim=1).values.sum(dim=1)

        pool_size_b = int(3.0 * self.m)
        self.pool_b.zero_()
        b_arms = torch.topk(self.mu_hat, min(pool_size_b, self.d), dim=1).indices
        self.pool_b.scatter_(1, b_arms, True)

        mu_b = self.mu_hat.clone()
        mu_b[~self.pool_b] = -float("inf")
        top_b = torch.topk(mu_b, self.m, dim=1).values.sum(dim=1)

        self._using_a = top_a >= top_b * 0.95

    def _check_coverage(self):
        emp_top = torch.topk(self.mu_hat, self.m, dim=1).indices
        covered = torch.zeros(self.n_seeds, device=self.device)
        for i in range(self.m):
            arm = emp_top[:, i]
            in_pool = self.pool_a.gather(1, arm.unsqueeze(1)).squeeze(1)
            covered += in_pool.float()
        coverage_frac = covered / self.m
        need_expand = coverage_frac < 0.6
        self.beta = torch.where(
            need_expand,
            (self.beta * 1.5).clamp(max=self.beta_max),
            self.beta
        )
        if need_expand.any():
            self._build_pools()

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pools()

        if self.t > 0 and self.t % 100 == 0:
            self._check_coverage()
            self._cross_validate()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

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
        self.beta.fill_(self.beta_init)
        self.pool_a.zero_()
        self.pool_b.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._using_a.fill_(True)
        self._pool_built = False


NOVEL_V5_REGISTRY = {
    "adaptive_pool_dual": AdaptivePoolDual,
    "adaptive_pool_abstain": AdaptivePoolAbstain,
    "adaptive_pool_doubling": AdaptivePoolDoubling,
    "adaptive_freq_dual": AdaptiveFreqDual,
}

NOVEL_V5_NEEDS_ORACLE = set(NOVEL_V5_REGISTRY.keys())
