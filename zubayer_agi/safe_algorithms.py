"""Safe LLM-CTS algorithms — designed from bulletproof diagnosis.

Core principles from the failure analysis:
1. LLM mean-estimates are SYSTEMATICALLY biased: 67.5% of arms predicted p<0.5
   with high confidence → over-injected pseudo-failures poison good arms
2. Repeated injections compound the bias (B2's failure mode)
3. Single strong prior on one arm over-suppresses others (A1's failure mode)

Design principles (from algorithms-with-predictions framework):
1. ASYMMETRIC injection: only POSITIVE signals (boost arms), never negative
2. BOUNDED budget: total LLM weight ≤ c·sqrt(T) → recovers CTS regret bound
3. REVERSIBLE: track contributions, un-inject if validation fails
4. GATED: only inject when LLM's earlier picks are empirically validated
5. SELECTIVE: only under-explored arms get LLM priors (avoid over-nudging arms with data)

This is the "α-consistency / β-robustness" Pareto design from learning-augmented
algorithms theory.
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase
from oracle import GPTOracle


# ─── S1. Optimistic-Only CTS (OO-CTS) ─────────────────────────────────────
class OptimisticOnlyCTS(CTSBase):
    """LLM can only BOOST arms, never suppress them.

    Injects pseudo-successes on arms the LLM picks; does NOT inject
    pseudo-failures on arms LLM thinks are bad. Inspired by the ML-predictions
    framework: optimism under uncertainty with a prediction hint.
    """
    name = "SAFE_S1_optimistic_only"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 boost_strength: float = 5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.boost_strength = boost_strength
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            picks = self.oracle.query_top_m(self.mu_hat.tolist())
            # ONLY add pseudo-successes. Never add failures.
            for aid in picks:
                self.alphas[aid] += self.boost_strength
            # betas UNCHANGED → no arm gets suppressed
        return super().select_arms()


# ─── S2. Bounded-Budget CTS (BB-CTS) ──────────────────────────────────────
class BoundedBudgetCTS(CTSBase):
    """Total LLM pseudo-obs bounded by c·sqrt(T). Preserves CTS regret rate.

    Queries at doubling-epoch boundaries. Each query contributes at most
    B/n_epochs pseudo-obs. Asymptotically the LLM contribution is lower-order.
    """
    name = "SAFE_S2_bounded_budget"

    def __init__(self, d, m, oracle: GPTOracle, T_horizon: int = 1500,
                 budget_const: float = 2.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.total_budget = budget_const * math.sqrt(T_horizon)
        self.used_budget = 0.0
        self._epoch_starts = [int(30 * (1.5 ** i)) for i in range(10)]
        self._epoch_starts = [e for e in self._epoch_starts if e < T_horizon]
        self._epoch_idx = 0

    def select_arms(self):
        if (self._epoch_idx < len(self._epoch_starts) and
            self.t >= self._epoch_starts[self._epoch_idx]):
            self._inject()
            self._epoch_idx += 1
        return super().select_arms()

    def _inject(self):
        remaining = self.total_budget - self.used_budget
        if remaining <= 0:
            return
        # Each epoch gets 1/(remaining epochs + 1) of budget
        epochs_left = len(self._epoch_starts) - self._epoch_idx
        epoch_budget = remaining / max(1, epochs_left)
        per_arm_strength = epoch_budget / self.m  # distributed over top-m arms

        picks = self.oracle.query_top_m(self.mu_hat.tolist())
        for aid in picks[:self.m]:
            self.alphas[aid] += per_arm_strength
            self.used_budget += per_arm_strength


# ─── S3. Reversible-Injection CTS (RI-CTS) ────────────────────────────────
class ReversibleInjectionCTS(CTSBase):
    """LLM contributions are tracked and REVERSED if validation fails.

    After each injection, wait 30 rounds, then check: are LLM's picks
    outperforming empirical top? If not, subtract the contribution.
    """
    name = "SAFE_S3_reversible"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 boost_strength: float = 5.0, validation_delay: int = 30,
                 **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.boost_strength = boost_strength
        self.validation_delay = validation_delay
        self._contribution_log: list[tuple[int, int, float]] = []  # (t_added, arm, weight)
        self._llm_picks: set[int] = set()
        self._injected_at: int = -1
        self._validated = False

    def select_arms(self):
        # Initial injection
        if self.t == self.T_warmup and self._injected_at < 0:
            picks = self.oracle.query_top_m(self.mu_hat.tolist())
            self._llm_picks = set(picks)
            for aid in picks:
                self.alphas[aid] += self.boost_strength
                self._contribution_log.append((self.t, aid, self.boost_strength))
            self._injected_at = self.t

        # Validate: after validation_delay, check if LLM picks empirically good
        if (self._injected_at > 0 and not self._validated and
            self.t >= self._injected_at + self.validation_delay):
            self._validated = True
            mu = self.mu_hat
            emp_top = set(int(a) for a in np.argsort(mu)[::-1][:self.m])
            agreement = len(self._llm_picks & emp_top) / max(1, self.m)

            if agreement < 0.4:  # less than 2/5 agreement → revoke
                for t_added, arm, weight in self._contribution_log:
                    self.alphas[arm] -= weight  # REVERSE
                self._contribution_log.clear()

        return super().select_arms()


# ─── S4. Gated-Elimination CTS (GE-CTS) ───────────────────────────────────
class GatedEliminationCTS(CTSBase):
    """Elimination, but ONLY for arms with low empirical mu_hat AND LLM agrees.

    Safer than F1 because we require DUAL confirmation: data says arm looks bad
    AND LLM says arm looks bad. Eliminates only the intersection.
    """
    name = "SAFE_S4_gated_elim"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 50, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._mask = np.ones(d, dtype=bool)
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.copy()
            # Empirical bottom 1/3
            emp_bottom = set(int(a) for a in np.argsort(mu)[:self.d // 3])
            # LLM eliminations
            llm_elim = set(self.oracle.query_elimination(mu.tolist()))
            # Dual: only eliminate arms BOTH data and LLM say are bad
            to_eliminate = emp_bottom & llm_elim
            # Safety: never eliminate more than d/2 arms, never eliminate arms
            # with above-median mu_hat
            med = np.median(mu)
            to_eliminate = {a for a in to_eliminate if mu[a] < med}
            to_eliminate = set(list(to_eliminate)[:self.d // 2])
            for a in to_eliminate:
                self._mask[a] = False

        samples = self._sample()
        samples = np.where(self._mask, samples, -np.inf)
        return list(np.argsort(samples)[::-1][:self.m])


# ─── S5. Mix-in CTS (MI-CTS) — TS-LLM with safer mixing ───────────────────
class MixInCTS(CTSBase):
    """Mix between CTS and LLM with decaying LLM weight.

    At round t, with probability 1/sqrt(t) use LLM; else CTS.
    As t grows, LLM influence fades. Preserves CTS asymptotic regret.
    Unlike TS-LLM, the LLM is only queried when we're about to use it.
    """
    name = "SAFE_S5_mixin"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_cooldown: int = 80, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_cooldown = query_cooldown
        self._cached_picks: list[int] = []
        self._last_query: int = -query_cooldown

    def select_arms(self):
        if self.t < self.T_warmup:
            return super().select_arms()

        p_llm = 1.0 / math.sqrt(max(self.t - self.T_warmup + 1, 1))
        if np.random.random() < p_llm:
            if self.t - self._last_query >= self.query_cooldown:
                self._cached_picks = self.oracle.query_top_m(self.mu_hat.tolist())
                self._last_query = self.t
            if self._cached_picks:
                # Mix: LLM's picks + top UCB arms, sample from this reduced set
                cb = np.sqrt(2.0 * np.log(max(self.t, 1) + 1) / np.maximum(self.n_pulls, 1))
                ucb = self.mu_hat + cb
                hedge = [a for a in np.argsort(ucb)[::-1] if a not in self._cached_picks][:self.m]
                candidates = list(set(self._cached_picks) | set(hedge))
                samples = self._sample()
                masked = np.full(self.d, -np.inf)
                for a in candidates:
                    masked[a] = samples[a]
                return list(np.argsort(masked)[::-1][:self.m])
        return super().select_arms()


# ─── S6. SAFE-CTS (the synthesis) ─────────────────────────────────────────
class SafeCTS(CTSBase):
    """The masterpiece: combines all safe-design principles.

    - Optimistic-only injection (no failures added)
    - Bounded budget O(sqrt(T))
    - Reversible via validation gate
    - Only injects on under-pulled arms
    - Graceful fallback to CTS on detected LLM error
    """
    name = "SAFE_S6_masterpiece"

    def __init__(self, d, m, oracle: GPTOracle, T_horizon: int = 1500,
                 T_warmup: int = 30, budget_const: float = 3.0,
                 validation_delay: int = 50, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.total_budget = budget_const * math.sqrt(T_horizon)
        self.used_budget = 0.0
        self.validation_delay = validation_delay
        self._contribution_log: list[tuple[int, int, float]] = []
        self._last_llm_picks: set[int] = set()
        self._last_inject_t: int = -1
        self._llm_disabled: bool = False
        # Query schedule: doubling from T_warmup
        self._schedule = []
        t = T_warmup
        while t < T_horizon:
            self._schedule.append(t)
            t = int(t * 1.6)
        self._schedule_idx = 0

    def select_arms(self):
        # Validate previous injection before next one
        if (self._last_inject_t > 0 and self._last_llm_picks and
            self.t >= self._last_inject_t + self.validation_delay and
            not self._llm_disabled):
            emp_top = set(int(a) for a in np.argsort(self.mu_hat)[::-1][:self.m])
            agreement = len(self._last_llm_picks & emp_top) / max(1, self.m)
            if agreement == 0.0:
                # Strong signal that LLM is wrong: reverse and disable
                for _, arm, w in self._contribution_log:
                    self.alphas[arm] -= w
                self._contribution_log.clear()
                self._llm_disabled = True
            elif agreement < 0.4:
                # Partial disagreement: reverse but keep LLM enabled
                for _, arm, w in self._contribution_log:
                    self.alphas[arm] -= w * 0.5  # half-reverse
                self._contribution_log.clear()

        # Inject if scheduled and budget remains
        if (self._schedule_idx < len(self._schedule) and
            self.t >= self._schedule[self._schedule_idx] and
            not self._llm_disabled and
            self.used_budget < self.total_budget):
            self._inject()
            self._schedule_idx += 1

        return super().select_arms()

    def _inject(self):
        remaining = self.total_budget - self.used_budget
        epochs_left = len(self._schedule) - self._schedule_idx
        per_epoch = remaining / max(1, epochs_left)
        per_arm = per_epoch / self.m

        picks = self.oracle.query_top_m(self.mu_hat.tolist())
        self._last_llm_picks = set(picks)
        self._last_inject_t = self.t

        for aid in picks[:self.m]:
            # Selective: only inject on under-pulled arms to avoid redundancy
            if self.n_pulls[aid] < math.sqrt(self.t + 1):
                self.alphas[aid] += per_arm
                self._contribution_log.append((self.t, aid, per_arm))
                self.used_budget += per_arm


SAFE_ALGORITHMS = {
    "SAFE_S1_optimistic_only": OptimisticOnlyCTS,
    "SAFE_S2_bounded_budget": BoundedBudgetCTS,
    "SAFE_S3_reversible": ReversibleInjectionCTS,
    "SAFE_S4_gated_elim": GatedEliminationCTS,
    "SAFE_S5_mixin": MixInCTS,
    "SAFE_S6_masterpiece": SafeCTS,
}
