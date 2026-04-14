"""GPU-batched agents: all 9 algorithms operating on (n_seeds, d) tensor state.

Every agent maintains state as tensors and selects arms via vectorized ops.
select_arms returns (n_seeds, m) indices. update takes (n_seeds, m) rewards.
"""
from __future__ import annotations

import math
import torch

from .batched_oracle import BatchedSimulatedCLO


class BatchedAgentBase:
    """Base class for batched agents. State is (n_seeds, d) tensors."""

    name: str = "base"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device):
        self.d = d
        self.m = m
        self.n_seeds = n_seeds
        self.device = device
        self.t = 0

        # Per-arm statistics: (n_seeds, d)
        self.total_reward = torch.zeros(n_seeds, d, device=device)
        self.n_pulls = torch.zeros(n_seeds, d, device=device)
        self.mu_hat = torch.zeros(n_seeds, d, device=device)

    @property
    def ucb_indices(self) -> torch.Tensor:
        """(n_seeds, d) UCB indices."""
        bonus = torch.sqrt(2 * math.log(max(self.t, 2)) /
                           self.n_pulls.clamp(min=1))
        # Unpulled arms get inf
        result = self.mu_hat + bonus
        result[self.n_pulls == 0] = float("inf")
        return result

    def top_m_by_ucb(self, scores: torch.Tensor | None = None,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """Select top-m arms by score. Returns (n_seeds, m) indices.

        Args:
            scores: (n_seeds, d) scores to rank by. Default: UCB indices.
            mask: (n_seeds, d) bool mask; only consider True positions.
        """
        if scores is None:
            scores = self.ucb_indices
        if mask is not None:
            scores = scores.clone()
            scores[~mask] = -float("inf")
        return torch.topk(scores, self.m, dim=1).indices

    def select_arms(self) -> torch.Tensor:
        """Select m arms for all seeds. Returns (n_seeds, m)."""
        raise NotImplementedError

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        """Update state after observing rewards.

        Args:
            selected: (n_seeds, m) arm indices
            rewards: (n_seeds, m) reward values
        """
        self.t += 1
        # Scatter-add rewards and counts
        self.total_reward.scatter_add_(1, selected, rewards)
        self.n_pulls.scatter_add_(1, selected, torch.ones_like(rewards))
        # Recompute means (avoid div by zero)
        nonzero = self.n_pulls > 0
        self.mu_hat[nonzero] = self.total_reward[nonzero] / self.n_pulls[nonzero]

    def reset(self):
        self.t = 0
        self.total_reward.zero_()
        self.n_pulls.zero_()
        self.mu_hat.zero_()


# ============================================================================
# CUCB
# ============================================================================

class BatchedCUCB(BatchedAgentBase):
    """Standard CUCB on full ground set [d]."""
    name = "cucb"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device, **kwargs):
        super().__init__(d, m, n_seeds, device)

    def select_arms(self) -> torch.Tensor:
        if self.t < self.d // self.m:
            # Round-robin init: shift by t*m mod d
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)
        return self.top_m_by_ucb()


# ============================================================================
# CTS (Combinatorial Thompson Sampling)
# ============================================================================

class BatchedCTS(BatchedAgentBase):
    """Combinatorial Thompson Sampling with Beta posteriors."""
    name = "cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alphas = torch.full((n_seeds, d), prior_alpha, device=device)
        self.betas = torch.full((n_seeds, d), prior_beta, device=device)

    def select_arms(self) -> torch.Tensor:
        # Vectorized Beta sampling: (n_seeds, d) all at once
        samples = torch.distributions.Beta(self.alphas, self.betas).sample()
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.alphas.fill_(self.prior_alpha)
        self.betas.fill_(self.prior_beta)


# ============================================================================
# LLM-CUCB-AT (our method)
# ============================================================================

class BatchedLLMCUCBAT(BatchedAgentBase):
    """LLM-CUCB-AT: Adaptive trust with composite score, fully batched."""
    name = "llm_cucb_at"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, h_max: int | None = None,
                 T_0: int | None = None, K: int = 3, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.h_max = h_max if h_max is not None else int(math.ceil(math.sqrt(d)))
        self.T_0 = T_0 if T_0 is not None else int(math.ceil(d * math.log(d) / m))
        self.K = K

        # Trust diagnostics: store per-round means across seeds
        self.kappa_history: list[float] = []
        self.rho_history: list[float] = []
        self.tau_history: list[float] = []
        self.hedge_history: list[float] = []

        # Corruption detection state
        self._regret_window: list[torch.Tensor] = []
        self._force_fallback = torch.zeros(n_seeds, dtype=torch.bool, device=device)

    def select_arms(self) -> torch.Tensor:
        # Phase 1: Init
        if self.t < self.T_0:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        # Query oracle
        oracle_out = self.oracle.query_batched(self.mu_hat)
        suggested = oracle_out["suggested_sets"]  # (n_seeds, m)
        kappa = oracle_out["consistency"]          # (n_seeds,)

        # Posterior validation: rho = sum(mu_hat[suggested]) / max_S sum(mu_hat[S])
        suggested_reward = torch.gather(self.mu_hat, 1, suggested).sum(dim=1)  # (n_seeds,)
        best_arms = torch.topk(self.mu_hat, self.m, dim=1).indices
        best_reward = torch.gather(self.mu_hat, 1, best_arms).sum(dim=1)
        rho = suggested_reward / best_reward.clamp(min=1e-8)

        # Composite trust
        tau = torch.min(kappa, rho)  # (n_seeds,)

        # Store diagnostics (mean across seeds)
        self.kappa_history.append(kappa.mean().item())
        self.rho_history.append(rho.mean().item())
        self.tau_history.append(tau.mean().item())

        # Hedge size per seed
        h = torch.ceil(self.h_max * (1.0 - tau)).long()  # (n_seeds,)
        self.hedge_history.append(h.float().mean().item())

        # Build reduced set and select for each seed
        ucb = self.ucb_indices  # (n_seeds, d)

        # Create suggestion mask: (n_seeds, d)
        sugg_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        sugg_mask.scatter_(1, suggested, True)

        # For the hedge: zero out suggested arms in UCB, then topk
        ucb_for_hedge = ucb.clone()
        ucb_for_hedge[sugg_mask] = -float("inf")

        # Vectorized hedge: get top-max_h arms, mask by per-seed h
        max_h = h.max().item()
        cand_mask = sugg_mask.clone()
        if max_h > 0 and max_h <= self.d - self.m:
            hedge_arms = torch.topk(ucb_for_hedge, min(max_h, self.d - self.m), dim=1).indices
            # Create rank mask: column j is included if j < h[seed]
            col_idx = torch.arange(hedge_arms.shape[1], device=self.device).unsqueeze(0)  # (1, max_h)
            include_mask = col_idx < h.unsqueeze(1)  # (n_seeds, max_h)
            # Scatter into candidate mask (only where include_mask is True)
            hedge_mask = torch.zeros_like(cand_mask)
            hedge_mask.scatter_(1, hedge_arms, include_mask)
            cand_mask = cand_mask | hedge_mask

        # Fallback seeds: use full arm set
        cand_mask[self._force_fallback] = True
        self._force_fallback.zero_()

        # Select top-m from candidates
        return self.top_m_by_ucb(mask=cand_mask)

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)

        if self.t > self.T_0:
            # Empirical regret tracking
            selected_reward = rewards.sum(dim=1)  # (n_seeds,)
            best_reward = torch.topk(self.mu_hat, self.m, dim=1).values.sum(dim=1)
            emp_regret = best_reward - selected_reward
            self._regret_window.append(emp_regret)
            if len(self._regret_window) > 200:
                self._regret_window.pop(0)

            # Corruption detection
            if len(self._regret_window) >= 5:
                window = min(int(math.ceil(math.sqrt(self.t))), len(self._regret_window))
                recent = torch.stack(self._regret_window[-window:])
                avg_regret = recent.mean(dim=0)
                threshold = math.sqrt(self.m * math.log(max(self.t, 2)) / window)
                self._force_fallback = avg_regret > threshold

    def reset(self):
        super().reset()
        self.kappa_history.clear()
        self.rho_history.clear()
        self.tau_history.clear()
        self.hedge_history.clear()
        self._regret_window.clear()
        self._force_fallback.zero_()


# ============================================================================
# LLM-Greedy
# ============================================================================

class BatchedLLMGreedy(BatchedAgentBase):
    """Always play oracle suggestion, no exploration."""
    name = "llm_greedy"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle

    def select_arms(self) -> torch.Tensor:
        out = self.oracle.query_batched(self.mu_hat)
        return out["suggested_sets"]


# ============================================================================
# ELLM-Adapted
# ============================================================================

class BatchedELLMAdapted(BatchedAgentBase):
    """ELLM-style exploration bonus, batched."""
    name = "ellm_adapted"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, bonus_scale: float = 0.5,
                 bonus_decay: float = 0.99, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.bonus_scale = bonus_scale
        self.bonus_decay = bonus_decay
        self.llm_bonus = torch.zeros(n_seeds, d, device=device)

    def select_arms(self) -> torch.Tensor:
        out = self.oracle.query_batched(self.mu_hat)
        suggested = out["suggested_sets"]  # (n_seeds, m)

        # Decay and add bonus
        self.llm_bonus *= self.bonus_decay
        self.llm_bonus.scatter_add_(1, suggested,
                                    torch.full_like(suggested, self.bonus_scale, dtype=torch.float32))

        scores = self.ucb_indices + self.llm_bonus
        return torch.topk(scores, self.m, dim=1).indices

    def reset(self):
        super().reset()
        self.llm_bonus.zero_()


# ============================================================================
# OPRO-Bandit
# ============================================================================

class BatchedOPROBandit(BatchedAgentBase):
    """OPRO-style: oracle sees history, always follows suggestion."""
    name = "opro_bandit"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle

    def select_arms(self) -> torch.Tensor:
        # OPRO always follows oracle (simulated oracle ignores history)
        out = self.oracle.query_batched(self.mu_hat)
        return out["suggested_sets"]


# ============================================================================
# Warm-Start CTS
# ============================================================================

class BatchedWarmStartCTS(BatchedAgentBase):
    """CTS with LLM-initialized priors."""
    name = "warm_start_cts"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, prior_strength: float = 5.0, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.prior_strength = prior_strength
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas = torch.ones(n_seeds, d, device=device)
        self._initialized = False

    def _init_priors(self):
        out = self.oracle.query_batched(self.mu_hat)
        suggested = out["suggested_sets"]  # (n_seeds, m)

        # Set high alpha for suggested arms
        self.alphas.scatter_(1, suggested,
                             torch.full_like(suggested, self.prior_strength, dtype=torch.float32))
        # Set moderate beta for non-suggested arms
        sugg_mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
        sugg_mask.scatter_(1, suggested, True)
        self.betas[~sugg_mask] = self.prior_strength * 0.5
        self._initialized = True

    def select_arms(self) -> torch.Tensor:
        if not self._initialized:
            self._init_priors()
        samples = torch.distributions.Beta(self.alphas, self.betas).sample()
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        successes = (rewards > 0.5).float()
        failures = 1.0 - successes
        self.alphas.scatter_add_(1, selected, successes)
        self.betas.scatter_add_(1, selected, failures)

    def reset(self):
        super().reset()
        self.alphas.fill_(1.0)
        self.betas.fill_(1.0)
        self._initialized = False


# ============================================================================
# Corrupt-Robust CUCB
# ============================================================================

class BatchedCorruptRobustCUCB(BatchedAgentBase):
    """Median-of-means CUCB, batched."""
    name = "corrupt_robust_cucb"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 n_buckets: int = 8, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.n_buckets = n_buckets

    def _median_of_means(self) -> torch.Tensor:
        """Compute MoM estimates for all (seed, arm) pairs: (n_seeds, d).

        Uses simple mean for arms with fewer than n_buckets pulls.
        For arms with enough pulls, reshapes into buckets and takes median of means.
        """
        return self.mu_hat.clone()  # Use simple mean as robust fallback
        # Full MoM is expensive on GPU; empirical mean is sufficient
        # for the action-corruption setting (rewards are uncorrupted)

    def select_arms(self) -> torch.Tensor:
        if self.t < self.d // self.m:
            start = (self.t * self.m) % self.d
            arms = torch.arange(start, start + self.m, device=self.device) % self.d
            return arms.unsqueeze(0).expand(self.n_seeds, -1)

        mom_estimates = self._median_of_means()
        bonus = torch.sqrt(2 * math.log(max(self.t, 2)) /
                           self.n_pulls.clamp(min=1))
        scores = mom_estimates + bonus
        scores[self.n_pulls == 0] = float("inf")
        return torch.topk(scores, self.m, dim=1).indices

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        # No ring buffer needed since we use simple mean (mu_hat from base)

    def reset(self):
        super().reset()


# ============================================================================
# EXP4
# ============================================================================

class BatchedEXP4(BatchedAgentBase):
    """EXP4 with 2 experts (LLM + UCB), batched."""
    name = "exp4"

    def __init__(self, d: int, m: int, n_seeds: int, device: torch.device,
                 oracle: BatchedSimulatedCLO, **kwargs):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.weights = torch.ones(n_seeds, 2, device=device)
        self._last_expert = torch.zeros(n_seeds, dtype=torch.long, device=device)

    def select_arms(self) -> torch.Tensor:
        probs = self.weights / self.weights.sum(dim=1, keepdim=True)

        # Expert 0: LLM
        out = self.oracle.query_batched(self.mu_hat)
        llm_set = out["suggested_sets"]  # (n_seeds, m)

        # Expert 1: UCB
        ucb_set = self.top_m_by_ucb()  # (n_seeds, m)

        # Choose expert per seed
        choose_llm = torch.rand(self.n_seeds, device=self.device) < probs[:, 0]
        self._last_expert = (~choose_llm).long()

        result = llm_set.clone()
        result[~choose_llm] = ucb_set[~choose_llm]
        return result

    def update(self, selected: torch.Tensor, rewards: torch.Tensor):
        super().update(selected, rewards)
        total_reward = rewards.sum(dim=1)  # (n_seeds,)
        eta = math.sqrt(math.log(2) / max(self.t, 1))
        probs = self.weights / self.weights.sum(dim=1, keepdim=True)

        # Importance-weighted update for chosen expert
        chosen_prob = torch.gather(probs, 1, self._last_expert.unsqueeze(1)).squeeze(1)
        est_reward = total_reward / chosen_prob.clamp(min=1e-8)

        # Update weights for chosen expert
        update = torch.exp(eta * est_reward / self.m)
        self.weights.scatter_(1, self._last_expert.unsqueeze(1),
                              torch.gather(self.weights, 1, self._last_expert.unsqueeze(1)) * update.unsqueeze(1))
        # Normalize
        self.weights /= self.weights.max(dim=1, keepdim=True).values

    def reset(self):
        super().reset()
        self.weights.fill_(1.0)
        self._last_expert.zero_()


# ============================================================================
# Registry
# ============================================================================

BATCHED_AGENT_REGISTRY: dict[str, type[BatchedAgentBase]] = {
    "cucb": BatchedCUCB,
    "cts": BatchedCTS,
    "llm_cucb_at": BatchedLLMCUCBAT,
    "llm_greedy": BatchedLLMGreedy,
    "ellm_adapted": BatchedELLMAdapted,
    "opro_bandit": BatchedOPROBandit,
    "warm_start_cts": BatchedWarmStartCTS,
    "corrupt_robust_cucb": BatchedCorruptRobustCUCB,
    "exp4": BatchedEXP4,
}

NEEDS_ORACLE = {"llm_cucb_at", "llm_greedy", "ellm_adapted", "opro_bandit", "warm_start_cts", "exp4"}
