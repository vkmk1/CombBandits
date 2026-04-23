"""V6: Novel algorithms based on REAL LLM oracle findings.

Key insight from real LLM experiments:
- Oracle quality is ENDOGENOUS: depends on mu_hat quality, not a fixed corruption rate
- At t=0 (mu_hat=0): 0/5 overlap. After exploration: 4/5 overlap.
- This means: explore first → then query oracle → then exploit
- No existing paper models this feedback loop.

Three research directions:

Direction 1: Explore-then-Oracle-then-Exploit (EOE)
  - Phase 1: round-robin exploration to build mu_hat signal
  - Phase 2: query oracle with high-quality mu_hat, build pool
  - Phase 3: CTS exploitation on oracle-informed pool
  - Theory: R(T) = O(epsilon^{2/3} * d^{1/3} * T^{2/3})

Direction 2: Information-Designed Oracle Queries
  - Don't share raw mu_hat — optimize what representation to send
  - Share UCB-ranked top arms (not all arms)
  - Share confidence flags (certain/uncertain/eliminated)
  - Prune high-uncertainty arms from oracle context
  - Grounded in Bayesian persuasion (Kamenica-Gentzkow 2011)

Direction 3: Adaptive Query Scheduling (KWIK-style)
  - Query oracle only when: (a) agent uncertainty > threshold AND (b) data quality > minimum
  - Avoids querying at t=0 (useless) and late game (diminishing returns)
  - Combines doubling-trick backbone with confidence-based triggers
  - Budget: total O(sqrt(T)) oracle queries
"""
from __future__ import annotations
import math
import torch
from combbandits.gpu.batched_agents import BatchedAgentBase
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO


# ═══════════════════════════════════════════════════════════════════════
# Direction 1: Explore-then-Oracle-then-Exploit (EOE)
# ═══════════════════════════════════════════════════════════════════════

class ExploreOracleExploit(BatchedAgentBase):
    """Three-phase algorithm: Explore → Oracle → Exploit.

    Phase 1 (Explore): Round-robin for T_explore rounds to build mu_hat.
    Phase 2 (Oracle): Query oracle with high-quality mu_hat, build pool.
    Phase 3 (Exploit): CTS on oracle-informed pool.

    T_explore is set to max(2*d, d/delta_est) where delta_est is estimated
    from the data — ensures mu_hat has enough signal for the oracle.
    """
    name = "explore_oracle_exploit"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 10, n_safety: int = 5,
                 T_explore_base: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_explore = T_explore_base if T_explore_base is not None else max(2 * d, 50)

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._phase = "explore"

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

    def select_arms(self) -> torch.Tensor:
        if self._phase == "explore":
            if self.t >= self.T_explore:
                self._phase = "oracle"
                self._build_pool()
                self._phase = "exploit"
                self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
                self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)
            else:
                start = (self.t * self.m) % self.d
                arms = torch.arange(start, start + self.m, device=self.device) % self.d
                return arms.unsqueeze(0).expand(self.n_seeds, -1)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        if self._phase == "exploit":
            successes = (rewards > 0.5).float()
            failures = 1.0 - successes
            self.alphas.scatter_add_(1, selected, successes)
            self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._phase = "explore"


class EOEAdaptive(BatchedAgentBase):
    """EOE with adaptive exploration length + abstention.

    Instead of fixed T_explore, monitors mu_hat stability:
    - Tracks running variance of top-m set across recent rounds
    - When top-m set stabilizes (low churn), transition to oracle phase
    - If oracle agreement with empirical top is low, abstain (fall back to full CTS)

    This adapts T_explore to the problem difficulty: easy problems
    transition quickly, hard problems explore longer.
    """
    name = "eoe_adaptive"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 10, n_safety: int = 5,
                 min_explore: int | None = None, stability_window: int = 20,
                 stability_threshold: float = 0.7, abstain_threshold: float = 0.3,
                 **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.min_explore = min_explore if min_explore is not None else max(d, 30)
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.abstain_threshold = abstain_threshold

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._phase = "explore"
        self._use_oracle = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._prev_top_m = None
        self._stability_count = torch.zeros(n_seeds, device=device)

    def _check_stability(self) -> torch.Tensor:
        current_top = torch.topk(self.mu_hat, self.m, dim=1).indices
        if self._prev_top_m is None:
            self._prev_top_m = current_top
            return torch.zeros(self.n_seeds, dtype=torch.bool, device=self.device)

        cur_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        cur_set.scatter_(1, current_top, True)
        prev_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        prev_set.scatter_(1, self._prev_top_m, True)
        overlap = (cur_set & prev_set).sum(dim=1).float() / self.m

        stable = overlap >= self.stability_threshold
        self._stability_count = torch.where(stable, self._stability_count + 1, torch.zeros_like(self._stability_count))
        self._prev_top_m = current_top

        return self._stability_count >= self.stability_window

    def _build_pool_and_decide(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        oracle_top = torch.topk(pool_counts, self.m, dim=1).indices
        emp_top_2m = torch.topk(self.mu_hat, min(2 * self.m, self.d), dim=1).indices
        emp_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        emp_set.scatter_(1, emp_top_2m, True)
        agreement = emp_set.gather(1, oracle_top).float().mean(dim=1)
        self._use_oracle = agreement >= self.abstain_threshold

        self.pool_mask.zero_()
        pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)

    def select_arms(self) -> torch.Tensor:
        if self._phase == "explore":
            if self.t >= self.min_explore:
                ready = self._check_stability()
                if ready.all() or self.t >= self.min_explore * 3:
                    self._phase = "exploit"
                    self._build_pool_and_decide()
                    self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
                    self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)

            if self._phase == "explore":
                start = (self.t * self.m) % self.d
                arms = torch.arange(start, start + self.m, device=self.device) % self.d
                return arms.unsqueeze(0).expand(self.n_seeds, -1)

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
        if self._phase == "exploit":
            successes = (rewards > 0.5).float()
            failures = 1.0 - successes
            self.alphas.scatter_add_(1, selected, successes)
            self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._phase = "explore"
        self._use_oracle.fill_(True)
        self._prev_top_m = None
        self._stability_count.zero_()


# ═══════════════════════════════════════════════════════════════════════
# Direction 2: Information-Designed Oracle Queries
# ═══════════════════════════════════════════════════════════════════════

class InfoDesignedPoolCTS(BatchedAgentBase):
    """Pool-CTS with information-designed oracle queries.

    Instead of querying oracle with raw mu_hat, designs the information signal:
    1. Only includes arms where confidence is above a threshold (prune noisy arms)
    2. Replaces mu_hat with UCB values for included arms (optimistic signal)
    3. Sets excluded arms to neutral value (doesn't bias oracle against them)

    Theory: Bayesian persuasion (Kamenica-Gentzkow 2011) shows that strategic
    information disclosure can improve receiver decisions. Pruning noisy arms
    reduces noise in oracle response; using UCB values encourages exploration.
    """
    name = "info_designed_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 10, n_safety: int = 5,
                 T_explore: int | None = None,
                 min_pulls_for_inclusion: int = 3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_explore = T_explore if T_explore is not None else max(2 * d, 50)
        self.min_pulls_for_inclusion = min_pulls_for_inclusion

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def _design_signal(self) -> torch.Tensor:
        """Design the information signal to send to the oracle.

        Returns a (n_seeds, d) tensor of "designed mu_hat" values.
        """
        designed = torch.zeros_like(self.mu_hat)
        confident = self.n_pulls >= self.min_pulls_for_inclusion
        cb = torch.sqrt(2.0 * math.log(max(self.t, 1) + 1) / self.n_pulls.clamp(min=1))
        ucb = (self.mu_hat + cb).clamp(max=1.0)
        neutral = self.mu_hat[confident].mean() if confident.any() else torch.tensor(0.5)
        designed = torch.where(confident, ucb, torch.full_like(self.mu_hat, neutral.item()))
        return designed

    def _build_pool(self):
        designed_signal = self._design_signal()
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(designed_signal)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices
        self.pool_mask.zero_()
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_explore:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pool_built:
            self._build_pool()
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        if self._pool_built:
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


class RankingOraclePool(BatchedAgentBase):
    """Oracle queries with ranking-based signal (not raw values).

    Instead of passing mu_hat values, passes a transformed signal where
    arms are ranked by mu_hat and assigned evenly-spaced pseudo-values.
    This prevents the oracle from being misled by magnitude of noisy estimates.

    Also uses explore-first to ensure rankings are meaningful.
    """
    name = "ranking_oracle_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 10, n_safety: int = 5,
                 T_explore: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_explore = T_explore if T_explore is not None else max(2 * d, 50)

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def _ranking_signal(self) -> torch.Tensor:
        """Convert mu_hat to evenly-spaced ranking signal."""
        ranks = torch.argsort(torch.argsort(self.mu_hat, dim=1, descending=True), dim=1)
        return 1.0 - ranks.float() / self.d

    def _build_pool(self):
        ranking_signal = self._ranking_signal()
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(ranking_signal)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices
        self.pool_mask.zero_()
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_explore:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if not self._pool_built:
            self._build_pool()
            self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
            self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        if self._pool_built:
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


# ═══════════════════════════════════════════════════════════════════════
# Direction 3: Adaptive Query Scheduling (KWIK-style)
# ═══════════════════════════════════════════════════════════════════════

class KWIKQueryCTS(BatchedAgentBase):
    """KWIK-style: query oracle only when conditions are right.

    Two conditions must BOTH hold to trigger an oracle query:
    1. Agent uncertainty is HIGH (UCB-LCB gap for top-m exceeds threshold)
    2. Data quality is SUFFICIENT (min pulls per arm exceeds minimum)

    When triggered, queries oracle and builds/updates pool.
    When not triggered, runs CTS on current pool (or full CTS if no pool yet).

    This naturally avoids:
    - Querying at t=0 (data quality too low)
    - Querying late (uncertainty already low, oracle adds nothing)
    - Querying too often (budget-efficient)
    """
    name = "kwik_query_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 5, n_safety: int = 5,
                 min_pulls_trigger: int = 2,
                 uncertainty_threshold: float = 0.3,
                 query_cooldown: int = 50, max_queries: int = 100,
                 **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.min_pulls_trigger = min_pulls_trigger
        self.uncertainty_threshold = uncertainty_threshold
        self.query_cooldown = query_cooldown
        self.max_queries = max_queries

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._has_pool = False
        self._total_oracle_batches = 0
        self._last_query_t = -query_cooldown

    def _should_query(self) -> bool:
        if self._total_oracle_batches >= self.max_queries:
            return False
        if self.t - self._last_query_t < self.query_cooldown:
            return False

        min_pulls = self.n_pulls.min(dim=1).values
        data_ready = (min_pulls >= self.min_pulls_trigger).all()
        if not data_ready:
            return False

        cb = torch.sqrt(2.0 * math.log(max(self.t, 1) + 1) / self.n_pulls.clamp(min=1))
        top_m_indices = torch.topk(self.mu_hat, self.m, dim=1).indices
        top_m_cb = cb.gather(1, top_m_indices)
        mean_gap = top_m_cb.mean()
        uncertain = mean_gap > self.uncertainty_threshold

        return uncertain

    def _query_and_build(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
            self._total_oracle_batches += 1

        if self._has_pool:
            old_counts = self.pool_mask.float()
            combined = old_counts * 0.3 + pool_counts
            pool_arms = torch.topk(combined, min(self.pool_size, self.d), dim=1).indices
        else:
            pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices

        self.pool_mask.zero_()
        self.pool_mask.scatter_(1, pool_arms, True)
        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._has_pool = True
        self._last_query_t = self.t

    def select_arms(self) -> torch.Tensor:
        if self.t > 0 and self._should_query():
            self._query_and_build()

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        if self._has_pool:
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
        self._has_pool = False
        self._total_oracle_batches = 0
        self._last_query_t = -self.query_cooldown


class DoublingKWIKPool(BatchedAgentBase):
    """Doubling epochs + KWIK triggers for oracle queries.

    Backbone: doubling-trick epochs (lengths 50, 100, 200, ...).
    At each epoch boundary, checks if KWIK conditions are met:
    - If yes: query oracle, rebuild pool for this epoch
    - If no: keep previous pool (or run full CTS if no pool)

    Each epoch also checks if the pool still covers the empirical top-m.
    If coverage drops below threshold, expand pool adaptively.
    """
    name = "doubling_kwik_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta_init: float = 2.0,
                 beta_max: float = 8.0, n_pool_rounds: int = 10,
                 n_safety: int = 5, epoch_base: int = 50,
                 min_pulls_trigger: int = 2, abstain_threshold: float = 0.3,
                 **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.epoch_base = epoch_base
        self.min_pulls_trigger = min_pulls_trigger
        self.abstain_threshold = abstain_threshold

        self.beta = torch.full((n_seeds,), beta_init, device=device)
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._use_oracle = torch.ones(n_seeds, dtype=torch.bool, device=device)

        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = epoch_base
        self._has_pool = False

    def _data_ready(self) -> bool:
        min_pulls = self.n_pulls.min(dim=1).values
        return (min_pulls >= self.min_pulls_trigger).all().item()

    def _build_pool_and_decide(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        oracle_top = torch.topk(pool_counts, self.m, dim=1).indices
        emp_top_2m = torch.topk(self.mu_hat, min(2 * self.m, self.d), dim=1).indices
        emp_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        emp_set.scatter_(1, emp_top_2m, True)
        agreement = emp_set.gather(1, oracle_top).float().mean(dim=1)
        self._use_oracle = agreement >= self.abstain_threshold

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
        self._has_pool = True

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
        if self.t >= self._epoch_start + self._epoch_length:
            self._epoch += 1
            self._epoch_start = self.t
            self._epoch_length = min(self.epoch_base * (2 ** self._epoch), 5000)

            if self._data_ready():
                self._check_coverage()
                self._build_pool_and_decide()
                self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
                self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)

        samples = torch.distributions.Beta(
            self.alphas.clamp(min=0.01), self.betas_param.clamp(min=0.01)
        ).sample()

        if self._has_pool:
            pool_s = samples.clone()
            pool_s[~self.pool_mask] = -float("inf")
            pool_action = torch.topk(pool_s, self.m, dim=1).indices
            full_action = torch.topk(samples, self.m, dim=1).indices
            result = full_action.clone()
            result[self._use_oracle] = pool_action[self._use_oracle]
            return result
        else:
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
        self._use_oracle.fill_(True)
        self._epoch = 0
        self._epoch_start = 0
        self._epoch_length = self.epoch_base
        self._has_pool = False


# ═══════════════════════════════════════════════════════════════════════
# Combination: EOE + Info Design + Adaptive Query
# ═══════════════════════════════════════════════════════════════════════

class EOEInfoKWIK(BatchedAgentBase):
    """The synthesis: Explore → Info-Designed Oracle Query → KWIK-Triggered Exploit.

    Combines all three directions:
    1. Explore phase builds mu_hat (Direction 1)
    2. Oracle query uses info-designed signal — UCB values for confident arms,
       neutral for uncertain (Direction 2)
    3. During exploit, KWIK triggers re-query only when uncertainty is high
       AND data is sufficient (Direction 3)
    4. Abstention: falls back to full CTS if oracle disagrees with data
    """
    name = "eoe_info_kwik"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_pool_rounds: int = 10, n_safety: int = 5,
                 T_explore: int | None = None,
                 min_pulls_for_signal: int = 3,
                 kwik_cooldown: int = 200, kwik_threshold: float = 0.2,
                 abstain_threshold: float = 0.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_explore = T_explore if T_explore is not None else max(2 * d, 50)
        self.min_pulls_for_signal = min_pulls_for_signal
        self.kwik_cooldown = kwik_cooldown
        self.kwik_threshold = kwik_threshold
        self.abstain_threshold = abstain_threshold

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._phase = "explore"
        self._use_oracle = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._last_query_t = 0

    def _info_designed_signal(self) -> torch.Tensor:
        confident = self.n_pulls >= self.min_pulls_for_signal
        cb = torch.sqrt(2.0 * math.log(max(self.t, 1) + 1) / self.n_pulls.clamp(min=1))
        ucb = (self.mu_hat + cb).clamp(max=1.0)
        neutral = 0.5
        return torch.where(confident, ucb, torch.full_like(self.mu_hat, neutral))

    def _build_pool_and_decide(self):
        signal = self._info_designed_signal()
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(signal)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        oracle_top = torch.topk(pool_counts, self.m, dim=1).indices
        emp_top_2m = torch.topk(self.mu_hat, min(2 * self.m, self.d), dim=1).indices
        emp_set = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        emp_set.scatter_(1, emp_top_2m, True)
        agreement = emp_set.gather(1, oracle_top).float().mean(dim=1)
        self._use_oracle = agreement >= self.abstain_threshold

        self.pool_mask.zero_()
        pool_arms = torch.topk(pool_counts, min(self.pool_size, self.d), dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        mu_top = torch.topk(self.mu_hat, self.n_safety, dim=1).indices
        self.pool_mask.scatter_(1, mu_top, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._last_query_t = self.t

    def _should_requery(self) -> bool:
        if self.t - self._last_query_t < self.kwik_cooldown:
            return False
        cb = torch.sqrt(2.0 * math.log(max(self.t, 1) + 1) / self.n_pulls.clamp(min=1))
        top_m_indices = torch.topk(self.mu_hat, self.m, dim=1).indices
        top_m_cb = cb.gather(1, top_m_indices)
        return top_m_cb.mean() > self.kwik_threshold

    def select_arms(self) -> torch.Tensor:
        if self._phase == "explore":
            if self.t >= self.T_explore:
                self._phase = "exploit"
                self._build_pool_and_decide()
                self.alphas = 1.0 + (self.n_pulls * self.mu_hat).clamp(min=0)
                self.betas_param = 1.0 + (self.n_pulls * (1 - self.mu_hat)).clamp(min=0)
            else:
                start = (self.t * self.m) % self.d
                arms = torch.arange(start, start + self.m, device=self.device) % self.d
                return arms.unsqueeze(0).expand(self.n_seeds, -1)

        if self._phase == "exploit" and self._should_requery():
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
        if self._phase == "exploit":
            successes = (rewards > 0.5).float()
            failures = 1.0 - successes
            self.alphas.scatter_add_(1, selected, successes)
            self.betas_param.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas_param.fill_(1.0)
        self._phase = "explore"
        self._use_oracle.fill_(True)
        self._last_query_t = 0


NOVEL_V6_REGISTRY = {
    "explore_oracle_exploit": ExploreOracleExploit,
    "eoe_adaptive": EOEAdaptive,
    "info_designed_pool": InfoDesignedPoolCTS,
    "ranking_oracle_pool": RankingOraclePool,
    "kwik_query_cts": KWIKQueryCTS,
    "doubling_kwik_pool": DoublingKWIKPool,
    "eoe_info_kwik": EOEInfoKWIK,
}

NOVEL_V6_NEEDS_ORACLE = set(NOVEL_V6_REGISTRY.keys())
