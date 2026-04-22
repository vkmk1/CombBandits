"""GPU-batched algorithmic variants for Round 2 testing.

Each variant is a concrete modification to LLM-CUCB-AT targeting specific failures
identified in the Round 1 validation experiment. See research_synthesis.md.

All variants share the BatchedAgentBase interface for drop-in testing.
"""
from __future__ import annotations

import math
import torch

from .batched_agents import BatchedAgentBase, BatchedCUCB, BatchedLLMCUCBAT
from .batched_oracle import BatchedSimulatedCLO


class BatchedMetaBoBW(BatchedAgentBase):
    """V1: Meta-BoBW (Tsallis-INF over {vanilla CUCB, LLM-CUCB-AT}).

    Top-level meta-learner selects a policy per seed per round via Tsallis-1/2 FTRL.
    Tracks importance-weighted reward for each policy.
    Provably: Regret <= Regret(better policy) + O(sqrt(T) log 2).
    """
    name = "meta_bobw"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.policy_cucb = BatchedCUCB(d, m, n_seeds, device)
        self.policy_llm = BatchedLLMCUCBAT(d, m, n_seeds, device, oracle)

        self.log_weights = torch.zeros(n_seeds, 2, device=device)
        self._last_policy = torch.zeros(n_seeds, dtype=torch.long, device=device)
        self._last_prob = torch.ones(n_seeds, device=device) * 0.5

    def _sample_policy(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.log_weights, dim=1)
        choose_llm = torch.rand(self.n_seeds, device=self.device) < w[:, 1]
        prob_chosen = torch.where(choose_llm, w[:, 1], w[:, 0]).clamp(min=1e-4)
        return choose_llm, prob_chosen

    def select_arms(self) -> torch.Tensor:
        choose_llm, prob_chosen = self._sample_policy()
        self._last_policy = choose_llm.long()
        self._last_prob = prob_chosen

        cucb_action = self.policy_cucb.select_arms()
        llm_action = self.policy_llm.select_arms()

        result = cucb_action.clone()
        result[choose_llm] = llm_action[choose_llm]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        self.policy_cucb.update(selected, rewards)
        self.policy_llm.update(selected, rewards)

        total_reward = rewards.sum(dim=1) / self.m
        eta = 1.0 / math.sqrt(max(self.t, 1))
        loss = 1.0 - total_reward
        importance_loss = loss / self._last_prob.clamp(min=1e-4)

        policy_idx = self._last_policy.unsqueeze(1)
        update_vec = torch.zeros_like(self.log_weights)
        update_vec.scatter_(1, policy_idx, -eta * importance_loss.unsqueeze(1))
        self.log_weights += update_vec
        self.log_weights -= self.log_weights.max(dim=1, keepdim=True).values

    def reset(self):
        super().reset()
        self.policy_cucb.reset()
        self.policy_llm.reset()
        self.log_weights.zero_()


class BatchedExplorationFloor(BatchedAgentBase):
    """V3: LLM-CUCB-AT with forced uniform exploration floor.

    With probability epsilon_t = t^(-1/3), pull a uniformly random super-arm
    instead of following the oracle/hedge. Provably breaks sublinear attacks.
    """
    name = "explore_floor"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, alpha: float = 1.0/3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.alpha = alpha
        self.inner = BatchedLLMCUCBAT(d, m, n_seeds, device, oracle)

    def select_arms(self) -> torch.Tensor:
        base = self.inner.select_arms()
        eps_t = 1.0 / max(self.t, 1) ** self.alpha
        explore_mask = torch.rand(self.n_seeds, device=self.device) < eps_t
        if explore_mask.any():
            scores = torch.rand(int(explore_mask.sum().item()), self.d, device=self.device)
            random_sets = torch.topk(scores, self.m, dim=1).indices
            base = base.clone()
            base[explore_mask] = random_sets
        return base

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        self.inner.update(selected, rewards)

    def reset(self):
        super().reset()
        self.inner.reset()


class BatchedPoolRestriction(BatchedAgentBase):
    """V5: Oracle-Guided Pool Restriction.

    LLM's suggested set defines a pool P of size beta*m (accumulated across
    early rounds). Run vanilla CUCB restricted to P union (random safety arms).
    Sidesteps per-round trust entirely.
    """
    name = "pool_restrict"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._safety_arms: torch.Tensor | None = None
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

        if self.t < self.d // self.m:
            scores = torch.rand(self.n_seeds, self.d, device=self.device)
            scores[~self.pool_mask] = -float("inf")
            return torch.topk(scores, self.m, dim=1).indices

        ucb = self.ucb_indices
        ucb = ucb.clone()
        ucb[~self.pool_mask] = -float("inf")
        return torch.topk(ucb, self.m, dim=1).indices

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self._pool_built = False


class BatchedDivergenceTrust(BatchedAgentBase):
    """V2: LLM-CUCB-AT with divergence-based trust (replaces consistency).

    Replaces kappa (self-consistency) with a divergence score that *penalizes*
    confident agreement. The key insight: a deterministic adversary achieves
    perfect consistency but should be distrusted.

    tau = min(1 - entropy_deficit, rho)
    where entropy_deficit measures how far the oracle's suggestion distribution
    is from a "reasonable uncertainty" level (uniform over top candidates).
    """
    name = "div_trust"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, h_max: int | None = None,
                 T_0: int | None = None, K: int = 3,
                 min_entropy_target: float = 0.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.h_max = h_max if h_max is not None else int(math.ceil(math.sqrt(d)))
        self.T_0 = T_0 if T_0 is not None else int(math.ceil(d * math.log(d) / m))
        self.K = K
        self.min_entropy_target = min_entropy_target

        self.kappa_history: list[float] = []
        self.rho_history: list[float] = []
        self.tau_history: list[float] = []
        self.hedge_history: list[float] = []

        self._oracle_endorsement = torch.zeros(n_seeds, d, device=device)
        self._force_fallback = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_0:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        oracle_out = self.oracle.query_batched(self.mu_hat)
        suggested = oracle_out["suggested_sets"]
        consistency = oracle_out["consistency"]

        self._oracle_endorsement *= 0.99
        self._oracle_endorsement.scatter_add_(
            1, suggested, torch.ones_like(suggested, dtype=torch.float32))

        # Disagreement-based trust: if oracle always picks same set (high endorsement
        # concentration on few arms), we TRUST LESS, not more.
        top_endorsed = torch.topk(self._oracle_endorsement, self.m, dim=1).values.sum(dim=1)
        total_endorsed = self._oracle_endorsement.sum(dim=1).clamp(min=1e-3)
        concentration = top_endorsed / total_endorsed  # in [m/d, 1]
        divergence_trust = 1.0 - torch.clamp(
            (concentration - self.min_entropy_target) /
            (1.0 - self.min_entropy_target), min=0.0, max=1.0)

        suggested_reward = torch.gather(self.mu_hat, 1, suggested).sum(dim=1)
        best_arms = torch.topk(self.mu_hat, self.m, dim=1).indices
        best_reward = torch.gather(self.mu_hat, 1, best_arms).sum(dim=1)
        rho = suggested_reward / best_reward.clamp(min=1e-8)

        tau = torch.min(divergence_trust, rho)

        self.kappa_history.append(divergence_trust.mean().item())
        self.rho_history.append(rho.mean().item())
        self.tau_history.append(tau.mean().item())

        h = torch.ceil(self.h_max * (1.0 - tau)).long()
        self.hedge_history.append(h.float().mean().item())

        ucb = self.ucb_indices
        sugg_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        sugg_mask.scatter_(1, suggested, True)

        ucb_for_hedge = ucb.clone()
        ucb_for_hedge[sugg_mask] = -float("inf")

        max_h = h.max().item()
        cand_mask = sugg_mask.clone()
        if max_h > 0 and max_h <= self.d - self.m:
            hedge_arms = torch.topk(ucb_for_hedge, min(max_h, self.d - self.m), dim=1).indices
            col_idx = torch.arange(hedge_arms.shape[1], device=self.device).unsqueeze(0)
            include_mask = col_idx < h.unsqueeze(1)
            hedge_mask = torch.zeros_like(cand_mask)
            hedge_mask.scatter_(1, hedge_arms, include_mask)
            cand_mask = cand_mask | hedge_mask

        cand_mask[self._force_fallback] = True
        self._force_fallback.zero_()

        return self.top_m_by_ucb(mask=cand_mask)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)

    def reset(self):
        super().reset()
        self._oracle_endorsement.zero_()
        self._force_fallback.zero_()


class BatchedEpochRobust(BatchedAgentBase):
    """V4: BARBAT-style epoch-wrapper around LLM-CUCB-AT.

    Divide T into static exponentially-growing epochs [2^k, 2^(k+1)).
    Each epoch uses increasingly tight confidence radii.
    Crucially, trust is RESET each epoch — consistent-wrong adversary cannot
    accumulate trust across epoch boundaries.
    """
    name = "epoch_robust"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, epoch_base: float = 2.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.epoch_base = epoch_base
        self.inner = BatchedLLMCUCBAT(d, m, n_seeds, device, oracle)
        self.epoch_idx = 0
        self.epoch_end = 10

    def _check_epoch_transition(self):
        if self.t >= self.epoch_end:
            self.epoch_idx += 1
            self.epoch_end = int(10 * (self.epoch_base ** self.epoch_idx))
            # Reset trust state (the regret window and fallback flags)
            self.inner._regret_window.clear()
            self.inner._force_fallback.zero_()
            # Inflate the hedge ceiling slightly per epoch
            self.inner.h_max = min(
                self.d - self.m,
                int(math.ceil(math.sqrt(self.d) * (1.0 + 0.2 * self.epoch_idx)))
            )

    def select_arms(self) -> torch.Tensor:
        self._check_epoch_transition()
        return self.inner.select_arms()

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        self.inner.update(selected, rewards)

    def reset(self):
        super().reset()
        self.inner.reset()
        self.epoch_idx = 0
        self.epoch_end = 10


class BatchedCombined(BatchedAgentBase):
    """V7: Combined — BoBW + Divergence-Trust + Exploration Floor.

    The "hero" algorithm combining the three top-ranked modifications.
    """
    name = "combined"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, alpha: float = 1.0/3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.alpha = alpha
        self.policy_cucb = BatchedCUCB(d, m, n_seeds, device)
        self.policy_llm = BatchedDivergenceTrust(d, m, n_seeds, device, oracle)

        self.log_weights = torch.zeros(n_seeds, 2, device=device)
        self._last_policy = torch.zeros(n_seeds, dtype=torch.long, device=device)
        self._last_prob = torch.ones(n_seeds, device=device) * 0.5

    def _sample_policy(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.log_weights, dim=1)
        choose_llm = torch.rand(self.n_seeds, device=self.device) < w[:, 1]
        prob_chosen = torch.where(choose_llm, w[:, 1], w[:, 0]).clamp(min=1e-4)
        return choose_llm, prob_chosen

    def select_arms(self) -> torch.Tensor:
        choose_llm, prob_chosen = self._sample_policy()
        self._last_policy = choose_llm.long()
        self._last_prob = prob_chosen

        cucb_action = self.policy_cucb.select_arms()
        llm_action = self.policy_llm.select_arms()

        result = cucb_action.clone()
        result[choose_llm] = llm_action[choose_llm]

        eps_t = 1.0 / max(self.t, 1) ** self.alpha
        explore_mask = torch.rand(self.n_seeds, device=self.device) < eps_t
        if explore_mask.any():
            scores = torch.rand(int(explore_mask.sum().item()), self.d, device=self.device)
            random_sets = torch.topk(scores, self.m, dim=1).indices
            result[explore_mask] = random_sets

        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        self.policy_cucb.update(selected, rewards)
        self.policy_llm.update(selected, rewards)

        total_reward = rewards.sum(dim=1) / self.m
        eta = 1.0 / math.sqrt(max(self.t, 1))
        loss = 1.0 - total_reward
        importance_loss = loss / self._last_prob.clamp(min=1e-4)

        policy_idx = self._last_policy.unsqueeze(1)
        update_vec = torch.zeros_like(self.log_weights)
        update_vec.scatter_(1, policy_idx, -eta * importance_loss.unsqueeze(1))
        self.log_weights += update_vec
        self.log_weights -= self.log_weights.max(dim=1, keepdim=True).values

    def reset(self):
        super().reset()
        self.policy_cucb.reset()
        self.policy_llm.reset()
        self.log_weights.zero_()


class BatchedDivergenceTrustV2(BatchedAgentBase):
    """V2b: Conditional divergence trust — penalize concentration ONLY when
    oracle's top arms disagree with empirical top arms.

    Fixes the design flaw in V2 that over-penalized correct concentration.
    Trust is now: min(agreement(oracle, empirical), rho).
    """
    name = "div_trust_v2"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, h_max: int | None = None,
                 T_0: int | None = None, K: int = 3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.h_max = h_max if h_max is not None else int(math.ceil(math.sqrt(d)))
        self.T_0 = T_0 if T_0 is not None else int(math.ceil(d * math.log(d) / m))
        self.K = K

        self.kappa_history: list[float] = []
        self.rho_history: list[float] = []
        self.tau_history: list[float] = []
        self.hedge_history: list[float] = []

        self._oracle_endorsement = torch.zeros(n_seeds, d, device=device)
        self._force_fallback = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def select_arms(self) -> torch.Tensor:
        if self.t < self.T_0:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        oracle_out = self.oracle.query_batched(self.mu_hat)
        suggested = oracle_out["suggested_sets"]

        self._oracle_endorsement *= 0.99
        self._oracle_endorsement.scatter_add_(
            1, suggested, torch.ones_like(suggested, dtype=torch.float32))

        oracle_top = torch.topk(self._oracle_endorsement, self.m, dim=1).indices
        mu_top = torch.topk(self.mu_hat, self.m, dim=1).indices

        oracle_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        oracle_mask.scatter_(1, oracle_top, True)
        mu_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        mu_mask.scatter_(1, mu_top, True)

        agreement = (oracle_mask & mu_mask).sum(dim=1).float() / self.m

        suggested_reward = torch.gather(self.mu_hat, 1, suggested).sum(dim=1)
        best_reward = torch.gather(self.mu_hat, 1, mu_top).sum(dim=1)
        rho = suggested_reward / best_reward.clamp(min=1e-8)

        tau = torch.min(agreement, rho)

        self.kappa_history.append(agreement.mean().item())
        self.rho_history.append(rho.mean().item())
        self.tau_history.append(tau.mean().item())

        h = torch.ceil(self.h_max * (1.0 - tau)).long()
        self.hedge_history.append(h.float().mean().item())

        ucb = self.ucb_indices
        sugg_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        sugg_mask.scatter_(1, suggested, True)

        ucb_for_hedge = ucb.clone()
        ucb_for_hedge[sugg_mask] = -float("inf")

        max_h = h.max().item()
        cand_mask = sugg_mask.clone()
        if max_h > 0 and max_h <= self.d - self.m:
            hedge_arms = torch.topk(ucb_for_hedge, min(max_h, self.d - self.m), dim=1).indices
            col_idx = torch.arange(hedge_arms.shape[1], device=self.device).unsqueeze(0)
            include_mask = col_idx < h.unsqueeze(1)
            hedge_mask = torch.zeros_like(cand_mask)
            hedge_mask.scatter_(1, hedge_arms, include_mask)
            cand_mask = cand_mask | hedge_mask

        cand_mask[self._force_fallback] = True
        self._force_fallback.zero_()

        return self.top_m_by_ucb(mask=cand_mask)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)

    def reset(self):
        super().reset()
        self._oracle_endorsement.zero_()
        self._force_fallback.zero_()


class BatchedPoolWithTrust(BatchedAgentBase):
    """V5b: Pool restriction with trust monitor.

    Starts with pool_restrict. Continues to probe oracle each round; if oracle's
    current suggestion disagrees substantially with empirical top-m inside pool,
    expand the pool (add top-k UCB arms from outside pool).
    """
    name = "pool_with_trust"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 expand_threshold: float = 0.5, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.expand_threshold = expand_threshold
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False
        self._expansion_events = 0

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

        if self.t < self.d // self.m:
            scores = torch.rand(self.n_seeds, self.d, device=self.device)
            scores[~self.pool_mask] = -float("inf")
            return torch.topk(scores, self.m, dim=1).indices

        if self.t % 50 == 0 and self.t > self.T_check():
            self._check_expansion()

        ucb = self.ucb_indices
        ucb = ucb.clone()
        ucb[~self.pool_mask] = -float("inf")
        return torch.topk(ucb, self.m, dim=1).indices

    def T_check(self) -> int:
        return max(self.d // self.m, 100)

    def _check_expansion(self):
        """Monitor: if in-pool top-m is dominated by out-of-pool UCB, expand."""
        ucb = self.ucb_indices
        in_pool_ucb = ucb.clone()
        in_pool_ucb[~self.pool_mask] = -float("inf")
        in_pool_top = torch.topk(in_pool_ucb, self.m, dim=1).values.sum(dim=1)

        out_pool_ucb = ucb.clone()
        out_pool_ucb[self.pool_mask] = -float("inf")
        out_pool_top = torch.topk(out_pool_ucb, self.m, dim=1).values.sum(dim=1)

        expand = out_pool_top > in_pool_top * 1.1
        if expand.any():
            n_add = 3
            for _ in range(n_add):
                expand_scores = out_pool_ucb.clone()
                expand_scores[self.pool_mask] = -float("inf")
                to_add = torch.topk(expand_scores, 1, dim=1).indices
                add_mask = torch.zeros_like(self.pool_mask)
                add_mask.scatter_(1, to_add, True)
                self.pool_mask = self.pool_mask | (add_mask & expand.unsqueeze(1))
                out_pool_ucb[add_mask] = -float("inf")
            self._expansion_events += int(expand.sum().item())

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self._pool_built = False
        self._expansion_events = 0


class BatchedPoolCTS(BatchedAgentBase):
    """V8: Pool restriction + Thompson sampling inside pool.

    Combines Round 2's winning pool_restrict structure with CTS's Beta posterior
    (which had the strongest perfect-oracle regret).
    """
    name = "pool_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
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

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()
        samples = samples.clone()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False


class BatchedMetaBoBWWarm(BatchedAgentBase):
    """V1b: Meta-BoBW with LLM-warm-start — start believing LLM policy, only
    shift mass to CUCB when LLM policy underperforms.

    Avoids the meta-learning tax in the good case by initializing log_weights
    = [-3, 0] (97% weight on LLM policy initially).
    """
    name = "meta_bobw_warm"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, warm_bias: float = 3.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.policy_cucb = BatchedCUCB(d, m, n_seeds, device)
        self.policy_llm = BatchedLLMCUCBAT(d, m, n_seeds, device, oracle)

        self.log_weights = torch.zeros(n_seeds, 2, device=device)
        self.log_weights[:, 1] = warm_bias
        self._last_policy = torch.ones(n_seeds, dtype=torch.long, device=device)
        self._last_prob = torch.ones(n_seeds, device=device) * 0.9

    def _sample_policy(self):
        w = torch.softmax(self.log_weights, dim=1)
        choose_llm = torch.rand(self.n_seeds, device=self.device) < w[:, 1]
        prob_chosen = torch.where(choose_llm, w[:, 1], w[:, 0]).clamp(min=1e-4)
        return choose_llm, prob_chosen

    def select_arms(self) -> torch.Tensor:
        choose_llm, prob_chosen = self._sample_policy()
        self._last_policy = choose_llm.long()
        self._last_prob = prob_chosen

        cucb_action = self.policy_cucb.select_arms()
        llm_action = self.policy_llm.select_arms()

        result = cucb_action.clone()
        result[choose_llm] = llm_action[choose_llm]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        self.policy_cucb.update(selected, rewards)
        self.policy_llm.update(selected, rewards)

        total_reward = rewards.sum(dim=1) / self.m
        eta = 1.0 / math.sqrt(max(self.t, 1))
        loss = 1.0 - total_reward
        importance_loss = loss / self._last_prob.clamp(min=1e-4)

        policy_idx = self._last_policy.unsqueeze(1)
        update_vec = torch.zeros_like(self.log_weights)
        update_vec.scatter_(1, policy_idx, -eta * importance_loss.unsqueeze(1))
        self.log_weights += update_vec
        self.log_weights -= self.log_weights.max(dim=1, keepdim=True).values

    def reset(self):
        super().reset()
        self.policy_cucb.reset()
        self.policy_llm.reset()
        self.log_weights.zero_()
        self.log_weights[:, 1] = 3.0


class BatchedAdaptivePool(BatchedAgentBase):
    """V9: Adaptive pool — pool_restrict + periodic probe-based corruption detection.

    Periodically (every probe_period rounds) pulls a random out-of-pool super-arm
    and compares its empirical reward to the in-pool best empirical reward.
    If random arm consistently outperforms → corrupted pool detected → abandon
    pool, fall back to full-ground-set CUCB.

    This catches consistent_wrong: the bad pool's arms have means ~0.4, while
    random in-pool arms have mean ~0.3, but random OUT-of-pool arms include
    the true optimal arms with mean ~0.55.
    """
    name = "adaptive_pool"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 probe_period: int = 20, probe_threshold: float = 0.05,
                 probe_window: int = 50, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.probe_period = probe_period
        self.probe_threshold = probe_threshold
        self.probe_window = probe_window

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False
        self._abandon_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)
        self._probe_history: list[tuple[torch.Tensor, torch.Tensor]] = []  # (probe_reward, pool_reward)

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

    def _check_corruption(self):
        """If probe-reward > pool-reward consistently over window, mark pool bad."""
        if len(self._probe_history) < self.probe_window // self.probe_period:
            return
        recent = self._probe_history[-(self.probe_window // self.probe_period):]
        probe_avg = torch.stack([p for p, _ in recent]).mean(dim=0)
        pool_avg = torch.stack([q for _, q in recent]).mean(dim=0)
        diff = probe_avg - pool_avg
        self._abandon_pool = self._abandon_pool | (diff > self.probe_threshold)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t < self.d // self.m:
            scores = torch.rand(self.n_seeds, self.d, device=self.device)
            scores[~self.pool_mask] = -float("inf")
            return torch.topk(scores, self.m, dim=1).indices

        is_probe = (self.t % self.probe_period == 0) and (self.t > self.n_pool_rounds + 50)

        if is_probe:
            scores = torch.rand(self.n_seeds, self.d, device=self.device)
            scores[self.pool_mask] = -float("inf")
            probe_set = torch.topk(scores, self.m, dim=1).indices

            ucb = self.ucb_indices
            ucb_pool = ucb.clone()
            ucb_pool[~self.pool_mask] = -float("inf")
            pool_set = torch.topk(ucb_pool, self.m, dim=1).indices

            result = pool_set.clone()
            # For seeds that have abandoned the pool, use full UCB
            if self._abandon_pool.any():
                full_ucb_top = torch.topk(ucb, self.m, dim=1).indices
                result[self._abandon_pool] = full_ucb_top[self._abandon_pool]
            # For non-abandoned seeds, use probe set on alternating rounds
            use_probe = ~self._abandon_pool
            result[use_probe] = probe_set[use_probe]
            self._last_was_probe = True
            self._last_probe_set = probe_set
            self._last_pool_set = pool_set
            return result

        ucb = self.ucb_indices
        if self._abandon_pool.all():
            return torch.topk(ucb, self.m, dim=1).indices
        ucb_masked = ucb.clone()
        ucb_masked[~self.pool_mask] = -float("inf")
        pool_result = torch.topk(ucb_masked, self.m, dim=1).indices
        if self._abandon_pool.any():
            full_result = torch.topk(ucb, self.m, dim=1).indices
            pool_result = pool_result.clone()
            pool_result[self._abandon_pool] = full_result[self._abandon_pool]
        self._last_was_probe = False
        return pool_result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        if getattr(self, "_last_was_probe", False):
            probe_rew = rewards.sum(dim=1)
            # Compute what pool would have yielded (estimated from mu_hat)
            pool_rew_est = torch.gather(self.mu_hat, 1, self._last_pool_set).sum(dim=1) * self.m
            self._probe_history.append((probe_rew, pool_rew_est))
            if len(self._probe_history) > 100:
                self._probe_history.pop(0)
            self._check_corruption()

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self._pool_built = False
        self._abandon_pool.zero_()
        self._probe_history.clear()


class BatchedPoolCTSAdaptive(BatchedAgentBase):
    """R4-V1: pool_cts + corruption-probe detection.

    Combines Round 3's champion (pool_cts with Thompson posterior) with an
    explicit corruption detector. Periodically samples a random out-of-pool
    super-arm; if its observed mean beats the pool top-m by a margin over a
    window, mark pool as corrupted and switch to full-arm CTS.
    """
    name = "pool_cts_adaptive"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 probe_period: int = 30, probe_threshold: float = 0.03,
                 probe_window: int = 40, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.probe_period = probe_period
        self.probe_threshold = probe_threshold
        self.probe_window = probe_window

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._pool_built = False
        self._abandon_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)

        self._probe_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._last_was_probe = False
        self._last_probe_set: torch.Tensor | None = None
        self._last_pool_top: torch.Tensor | None = None

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

    def _check_corruption(self):
        if len(self._probe_history) < 5:
            return
        window = min(self.probe_window // self.probe_period, len(self._probe_history))
        recent = self._probe_history[-window:]
        probe_avg = torch.stack([p for p, _ in recent]).mean(dim=0)
        pool_avg = torch.stack([q for _, q in recent]).mean(dim=0)
        diff = probe_avg - pool_avg
        self._abandon_pool = self._abandon_pool | (diff > self.probe_threshold)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()

        is_probe = (self.t > self.n_pool_rounds + 50 and
                    self.t % self.probe_period == 0 and
                    not self._abandon_pool.all())

        self._last_was_probe = is_probe

        if is_probe:
            active = ~self._abandon_pool
            rand_scores = torch.rand(self.n_seeds, self.d, device=self.device)
            rand_scores[self.pool_mask] = -float("inf")
            probe_set = torch.topk(rand_scores, self.m, dim=1).indices

            pool_samples = samples.clone()
            pool_samples[~self.pool_mask] = -float("inf")
            pool_top = torch.topk(pool_samples, self.m, dim=1).indices
            self._last_probe_set = probe_set
            self._last_pool_top = pool_top

            result = pool_top.clone()
            result[active] = probe_set[active]
            if self._abandon_pool.any():
                result[self._abandon_pool] = torch.topk(samples, self.m, dim=1).indices[self._abandon_pool]
            return result

        if self._abandon_pool.all():
            return torch.topk(samples, self.m, dim=1).indices
        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_samples, self.m, dim=1).indices

        if self._abandon_pool.any():
            full_top = torch.topk(samples, self.m, dim=1).indices
            pool_top = pool_top.clone()
            pool_top[self._abandon_pool] = full_top[self._abandon_pool]
        return pool_top

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

        if self._last_was_probe and self._last_pool_top is not None:
            probe_reward = rewards.mean(dim=1)
            pool_mean = torch.gather(self.alphas / (self.alphas + self.betas),
                                      1, self._last_pool_top).mean(dim=1)
            self._probe_history.append((probe_reward, pool_mean))
            if len(self._probe_history) > 100:
                self._probe_history.pop(0)
            self._check_corruption()

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False
        self._abandon_pool.zero_()
        self._probe_history.clear()


class BatchedPoolDivTrust(BatchedAgentBase):
    """R4-V2: pool_restrict + div_trust_v2 agreement check.

    Uses the pool for selection (R3 winner), but monitors via the div_trust_v2
    agreement check (pool-top vs empirical-top). If agreement stays low for
    a window, mark pool as corrupted.
    """
    name = "pool_div_trust"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 agreement_threshold: float = 0.3,
                 check_start: int = 200, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.agreement_threshold = agreement_threshold
        self.check_start = check_start

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False
        self._abandon_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)

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

    def _check_agreement(self):
        """If pool-top mu_hat agreement with global-top mu_hat is low, abandon."""
        pool_mu = self.mu_hat.clone()
        pool_mu[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_mu, self.m, dim=1).indices
        global_top = torch.topk(self.mu_hat, self.m, dim=1).indices

        pool_top_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        pool_top_mask.scatter_(1, pool_top, True)
        global_top_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        global_top_mask.scatter_(1, global_top, True)

        agreement = (pool_top_mask & global_top_mask).sum(dim=1).float() / self.m
        self._abandon_pool = self._abandon_pool | (agreement < self.agreement_threshold)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > self.check_start and self.t % 50 == 0:
            self._check_agreement()

        if self.t < self.d // self.m:
            scores = torch.rand(self.n_seeds, self.d, device=self.device)
            scores[~self.pool_mask] = -float("inf")
            return torch.topk(scores, self.m, dim=1).indices

        ucb = self.ucb_indices
        if self._abandon_pool.all():
            return torch.topk(ucb, self.m, dim=1).indices

        ucb_masked = ucb.clone()
        ucb_masked[~self.pool_mask] = -float("inf")
        result = torch.topk(ucb_masked, self.m, dim=1).indices
        if self._abandon_pool.any():
            full_top = torch.topk(ucb, self.m, dim=1).indices
            result = result.clone()
            result[self._abandon_pool] = full_top[self._abandon_pool]
        return result

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self._pool_built = False
        self._abandon_pool.zero_()


class BatchedPoolCTSCG(BatchedAgentBase):
    """HERO ALGORITHM: Pool-CTS with Soft Calibration Gate.

    Key insight: prior pool-CTS variants used HARD threshold-based switching,
    which causes bimodal performance under noisy detection. We use a SOFT mix:
    each round, choose between pool-CTS and full-CTS with a probability driven
    by the empirical evidence of pool quality.

    Algorithm:
      Phase 1 (t < T_init): round-robin all d arms (gives baseline mu_hat for each arm)
      Phase 2: build pool P from K oracle queries
      Phase 3 (t >= T_init): each round, compute
          gate = sigmoid((sum mu_hat[full_top_m] - sum mu_hat[pool_top_m]) / sigma)
        With probability `gate`, use FULL-CTS (Beta sampling on all d arms)
        Else, use POOL-CTS (Beta sampling restricted to pool)
        Recompute gate every recompute_period rounds.

    Theoretical properties:
      - When pool contains optimum: gate -> 0, regret = O(m^2 log T / Delta_min) on pool arms.
      - When pool excludes optimum (consistent_wrong): gate -> 1, regret = O(m^2 log T / Delta_min) on full arms.
      - Smooth interpolation in between based on calibration evidence.
    """
    name = "pool_cts_cg"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 T_init: int | None = None, sigma: float = 0.5,
                 recompute_period: int = 50, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.T_init = T_init if T_init is not None else max(d // m, 10)
        self.sigma = sigma
        self.recompute_period = recompute_period

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

        # Cached gate per seed (probability of using full CTS)
        self._gate = torch.full((n_seeds,), 0.5, device=device)

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

    def _compute_gate(self):
        """Gate = sigmoid((full_top_m_sum - pool_top_m_sum) / sigma).
        High gate => pool is missing better arms, prefer full CTS.
        """
        # Full top-m by empirical mean
        full_top_sum = torch.topk(self.mu_hat, self.m, dim=1).values.sum(dim=1)
        # Pool top-m
        pool_mu = self.mu_hat.clone()
        pool_mu[~self.pool_mask] = -float("inf")
        pool_top_sum = torch.topk(pool_mu, self.m, dim=1).values.sum(dim=1)
        # Gate
        diff = (full_top_sum - pool_top_sum) / self.sigma
        self._gate = torch.sigmoid(diff)

    def select_arms(self) -> torch.Tensor:
        # Phase 1: round-robin init across all d arms
        if self.t < self.T_init:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        # Build pool once after init
        if not self._pool_built:
            self._build_pool()
            self._compute_gate()

        # Recompute gate periodically as mu_hat improves
        if (self.t - self.T_init) % self.recompute_period == 0:
            self._compute_gate()

        # Sample policy per seed: probability gate => use full-CTS, else pool-CTS
        use_full = torch.rand(self.n_seeds, device=self.device) < self._gate

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()

        # Pool-CTS choice
        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_samples, self.m, dim=1).indices

        # Full-CTS choice
        full_top = torch.topk(samples, self.m, dim=1).indices

        result = pool_top.clone()
        result[use_full] = full_top[use_full]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False
        self._gate.fill_(0.5)


class BatchedPoolCTSETC(BatchedAgentBase):
    """R5-V1: Explore-then-commit over {pool_cts, cts}.

    Runs both pool_cts and cts in parallel for T_explore rounds (half each),
    observes empirical reward per policy, then commits to the higher-reward one.

    Provable guarantee: Regret <= min(Regret(pool_cts), Regret(cts)) + O(T_explore)
    where T_explore = c * sqrt(T * log T).

    No hyperparameter tuning needed (T_explore is determined by T). No detection
    logic — just cross-validation via observed rewards.
    """
    name = "pool_cts_etc"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 T_total: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds

        # Explore for sqrt(T log T) rounds (split between two policies)
        T_total = T_total if T_total is not None else 30000
        self.T_explore = int(math.ceil(math.sqrt(T_total * math.log(T_total)) / 2)) * 2

        # Pool policy state
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas_pool = torch.ones(n_seeds, d, device=device)
        self.betas_pool = torch.ones(n_seeds, d, device=device)

        # CTS policy state
        self.alphas_cts = torch.ones(n_seeds, d, device=device)
        self.betas_cts = torch.ones(n_seeds, d, device=device)

        # Track rewards per policy
        self._pool_rewards = torch.zeros(n_seeds, device=device)
        self._cts_rewards = torch.zeros(n_seeds, device=device)
        self._pool_pulls = torch.zeros(n_seeds, device=device)
        self._cts_pulls = torch.zeros(n_seeds, device=device)

        self._pool_built = False
        self._committed_policy = torch.full((n_seeds,), -1, dtype=torch.long, device=device)
        self._last_policy = torch.zeros(n_seeds, dtype=torch.long, device=device)  # 0=pool, 1=cts

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

    def _commit(self):
        pool_mean = self._pool_rewards / self._pool_pulls.clamp(min=1)
        cts_mean = self._cts_rewards / self._cts_pulls.clamp(min=1)
        self._committed_policy = (cts_mean > pool_mean).long()  # 0 if pool better, 1 if cts

    def _pool_select(self, alphas, betas) -> torch.Tensor:
        samples = torch.distributions.Beta(alphas, betas).sample()
        samples = samples.clone()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def _cts_select(self, alphas, betas) -> torch.Tensor:
        samples = torch.distributions.Beta(alphas, betas).sample()
        return torch.topk(samples, self.m, dim=1).indices

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t < self.T_explore:
            # Alternate: even rounds → pool, odd rounds → cts
            use_cts = (self.t % 2 == 1)
            self._last_policy = torch.full((self.n_seeds,), int(use_cts),
                                            dtype=torch.long, device=self.device)
            if use_cts:
                return self._cts_select(self.alphas_cts, self.betas_cts)
            else:
                return self._pool_select(self.alphas_pool, self.betas_pool)

        if self.t == self.T_explore:
            self._commit()

        # Commit phase
        use_cts_mask = (self._committed_policy == 1)
        self._last_policy = self._committed_policy

        pool_action = self._pool_select(self.alphas_pool, self.betas_pool)
        cts_action = self._cts_select(self.alphas_cts, self.betas_cts)

        result = pool_action.clone()
        result[use_cts_mask] = cts_action[use_cts_mask]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes

        # Only update the policy that was used per seed
        used_cts = (self._last_policy == 1).unsqueeze(1)
        used_pool = ~used_cts

        alpha_add_pool = torch.where(used_pool, successes, torch.zeros_like(successes))
        beta_add_pool = torch.where(used_pool, failures, torch.zeros_like(failures))
        self.alphas_pool.scatter_add_(1, selected, alpha_add_pool)
        self.betas_pool.scatter_add_(1, selected, beta_add_pool)

        alpha_add_cts = torch.where(used_cts, successes, torch.zeros_like(successes))
        beta_add_cts = torch.where(used_cts, failures, torch.zeros_like(failures))
        self.alphas_cts.scatter_add_(1, selected, alpha_add_cts)
        self.betas_cts.scatter_add_(1, selected, beta_add_cts)

        # Track per-policy rewards (only during explore phase)
        if self.t <= self.T_explore:
            total_reward = rewards.sum(dim=1)
            pool_seeds = (self._last_policy == 0)
            cts_seeds = ~pool_seeds
            self._pool_rewards = self._pool_rewards + total_reward * pool_seeds.float()
            self._cts_rewards = self._cts_rewards + total_reward * cts_seeds.float()
            self._pool_pulls = self._pool_pulls + pool_seeds.float()
            self._cts_pulls = self._cts_pulls + cts_seeds.float()

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas_pool.fill_(1.0)
        self.betas_pool.fill_(1.0)
        self.alphas_cts.fill_(1.0)
        self.betas_cts.fill_(1.0)
        self._pool_rewards.zero_()
        self._cts_rewards.zero_()
        self._pool_pulls.zero_()
        self._cts_pulls.zero_()
        self._pool_built = False
        self._committed_policy.fill_(-1)


class BatchedPoolCTSInitCheck(BatchedAgentBase):
    """R4-V5: pool_cts with initial round-robin + post-init agreement check.

    THE INSIGHT: pure oracle-side diversity can't distinguish deterministic-correct
    from deterministic-wrong. We need independent reward feedback.

    Design:
      1. t in [0, T_init): round-robin all d arms (CUCB-style init, gives baseline mu_hat)
      2. At t=T_init: check if oracle's suggested set agrees with empirical top-m.
         If disagreement > threshold, don't build pool, use full CTS.
         If agreement is high, build pool from oracle queries.
      3. t >= T_init: pool_cts on good side, full CTS on bad side.

    This correctly distinguishes:
      - perfect: init gives correct top-m, oracle agrees → use pool
      - consistent_wrong: init gives correct top-m, oracle suggests wrong → reject pool, full CTS
      - uniform/adversarial: oracle suggestions are mostly correct, maybe partial agreement → use pool
    """
    name = "pool_cts_ic"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 agreement_threshold: float = 0.5, T_init: int | None = None, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.agreement_threshold = agreement_threshold
        self.T_init = T_init if T_init is not None else max(d // m, 10)

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._pool_built = False
        self._use_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def _build_pool_and_check(self):
        """After init, query oracle and compare to empirical top-m."""
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        oracle_top = torch.topk(pool_counts, self.m, dim=1).indices
        mu_top = torch.topk(self.mu_hat, self.m, dim=1).indices

        oracle_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        oracle_mask.scatter_(1, oracle_top, True)
        mu_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        mu_mask.scatter_(1, mu_top, True)

        agreement = (oracle_mask & mu_mask).sum(dim=1).float() / self.m
        self._use_pool = agreement >= self.agreement_threshold

        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        # Phase 1: round-robin init
        if self.t < self.T_init:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        # Phase 2: build pool once with check
        if not self._pool_built:
            self._build_pool_and_check()

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()

        if self._use_pool.all():
            samples_masked = samples.clone()
            samples_masked[~self.pool_mask] = -float("inf")
            return torch.topk(samples_masked, self.m, dim=1).indices

        if not self._use_pool.any():
            return torch.topk(samples, self.m, dim=1).indices

        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_samples, self.m, dim=1).indices
        full_top = torch.topk(samples, self.m, dim=1).indices

        result = full_top.clone()
        result[self._use_pool] = pool_top[self._use_pool]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False
        self._use_pool.zero_()


class BatchedPoolCTSBuildCheck(BatchedAgentBase):
    """R4-V3: pool_cts with build-time diversity check.

    KEY INSIGHT from Round 3: pool_cts dominates on reliable oracles (357-566
    regret) but fails catastrophically on consistent_wrong (15,403). A consistent
    adversary returns the SAME set for every pool-query. This is detectable at
    build time — just check the number of unique arms across N pool queries.

    If unique arms across N queries < diversity_threshold * m, declare the oracle
    adversarial and fall back to CTS on the full ground set.

    Simple, cheap, one-time — no ongoing probe tax.
    """
    name = "pool_cts_bc"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 diversity_threshold: float = 1.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.diversity_threshold = diversity_threshold
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._pool_built = False
        self._use_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        # Diversity check: number of unique arms across queries
        unique_arms = (pool_counts > 0).sum(dim=1).float()
        self._use_pool = unique_arms >= self.diversity_threshold * self.m

        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()

        if self._use_pool.all():
            samples_masked = samples.clone()
            samples_masked[~self.pool_mask] = -float("inf")
            return torch.topk(samples_masked, self.m, dim=1).indices

        if not self._use_pool.any():
            return torch.topk(samples, self.m, dim=1).indices

        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_samples, self.m, dim=1).indices
        full_top = torch.topk(samples, self.m, dim=1).indices

        result = full_top.clone()
        result[self._use_pool] = pool_top[self._use_pool]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False
        self._use_pool.zero_()


class BatchedPoolCTSDivTrust(BatchedAgentBase):
    """R4-V4: pool_cts + ongoing div_trust_v2 agreement check.

    Combines build-time + runtime protection. If pool-top mu_hat disagrees
    with global-top mu_hat by too much, abandon pool and switch to full CTS.
    """
    name = "pool_cts_dt"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, beta: float = 3.0,
                 n_safety: int = 5, n_pool_rounds: int = 10,
                 agreement_threshold: float = 0.3, check_start: int = 300,
                 diversity_threshold: float = 1.3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.pool_size = int(beta * m)
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.agreement_threshold = agreement_threshold
        self.check_start = check_start
        self.diversity_threshold = diversity_threshold

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._pool_built = False
        self._use_pool = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

        unique_arms = (pool_counts > 0).sum(dim=1).float()
        self._use_pool = unique_arms >= self.diversity_threshold * self.m

        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        safety = torch.randint(0, self.d, (self.n_seeds, self.n_safety), device=self.device)
        self.pool_mask.scatter_(1, safety, True)
        self._pool_built = True

    def _check_runtime(self):
        """Extra runtime check: if empirical pool-top disagrees with global-top, abandon."""
        pool_mu = self.mu_hat.clone()
        pool_mu[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_mu, self.m, dim=1).indices
        global_top = torch.topk(self.mu_hat, self.m, dim=1).indices

        pool_top_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        pool_top_mask.scatter_(1, pool_top, True)
        global_top_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        global_top_mask.scatter_(1, global_top, True)

        agreement = (pool_top_mask & global_top_mask).sum(dim=1).float() / self.m
        self._use_pool = self._use_pool & (agreement >= self.agreement_threshold)

    def select_arms(self) -> torch.Tensor:
        if not self._pool_built:
            self._build_pool()

        if self.t > self.check_start and self.t % 100 == 0:
            self._check_runtime()

        samples = torch.distributions.Beta(self.alphas, self.betas).sample()

        if self._use_pool.all():
            samples_masked = samples.clone()
            samples_masked[~self.pool_mask] = -float("inf")
            return torch.topk(samples_masked, self.m, dim=1).indices

        if not self._use_pool.any():
            return torch.topk(samples, self.m, dim=1).indices

        pool_samples = samples.clone()
        pool_samples[~self.pool_mask] = -float("inf")
        pool_top = torch.topk(pool_samples, self.m, dim=1).indices
        full_top = torch.topk(samples, self.m, dim=1).indices

        result = full_top.clone()
        result[self._use_pool] = pool_top[self._use_pool]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.pool_mask.zero_()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._pool_built = False
        self._use_pool.zero_()


VARIANT_REGISTRY = {
    "meta_bobw": BatchedMetaBoBW,
    "explore_floor": BatchedExplorationFloor,
    "pool_restrict": BatchedPoolRestriction,
    "div_trust": BatchedDivergenceTrust,
    "epoch_robust": BatchedEpochRobust,
    "combined": BatchedCombined,
    "div_trust_v2": BatchedDivergenceTrustV2,
    "pool_with_trust": BatchedPoolWithTrust,
    "pool_cts": BatchedPoolCTS,
    "meta_bobw_warm": BatchedMetaBoBWWarm,
    "adaptive_pool": BatchedAdaptivePool,
    "pool_cts_adaptive": BatchedPoolCTSAdaptive,
    "pool_div_trust": BatchedPoolDivTrust,
    "pool_cts_bc": BatchedPoolCTSBuildCheck,
    "pool_cts_dt": BatchedPoolCTSDivTrust,
    "pool_cts_ic": BatchedPoolCTSInitCheck,
    "pool_cts_ic50": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=50, agreement_threshold=0.5, **kw),
    "pool_cts_ic100": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=100, agreement_threshold=0.3, **kw),
    "pool_cts_ic200": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=200, agreement_threshold=0.3, **kw),
    "pool_cts_ic100_40": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=100, agreement_threshold=0.4, **kw),
    "pool_cts_ic50_40": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=50, agreement_threshold=0.4, **kw),
    "pool_cts_ic150_40": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=150, agreement_threshold=0.4, **kw),
    "pool_cts_ic1000": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=1000, agreement_threshold=0.3, **kw),
    "pool_cts_ic1000_50": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=1000, agreement_threshold=0.5, **kw),
    "pool_cts_ic1600": lambda *a, **kw: BatchedPoolCTSInitCheck(*a, T_init=1600, agreement_threshold=0.5, **kw),
    "pool_cts_etc": BatchedPoolCTSETC,
    "pool_cts_cg": BatchedPoolCTSCG,
    "pool_cts_cg_t100": lambda *a, **kw: BatchedPoolCTSCG(*a, T_init=100, **kw),
    "pool_cts_cg_t200": lambda *a, **kw: BatchedPoolCTSCG(*a, T_init=200, **kw),
    "pool_cts_cg_t500": lambda *a, **kw: BatchedPoolCTSCG(*a, T_init=500, **kw),
    "pool_cts_cg_sigma01": lambda *a, **kw: BatchedPoolCTSCG(*a, T_init=200, sigma=0.1, **kw),
    "pool_cts_cg_sigma02": lambda *a, **kw: BatchedPoolCTSCG(*a, T_init=200, sigma=0.2, **kw),
}

VARIANT_NEEDS_ORACLE = set(VARIANT_REGISTRY.keys())
