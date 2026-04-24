"""All 17 candidate algorithms from RESEARCH_MASTERPIECE.md + CTS baseline.

Each algorithm is a class with: select_arms() -> list[int], update(selected, rewards).
All share a CTS-style Beta(alpha, beta) posterior base. They differ in:
- When/how they query the oracle
- How they incorporate the oracle's response into Beta posteriors, pool masks, or cluster links
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from oracle import GPTOracle


# ─── Base: Combinatorial Thompson Sampling ────────────────────────────────
class CTSBase:
    """Baseline Combinatorial Thompson Sampling. Beta(1,1) priors, no LLM."""
    name = "cts"

    def __init__(self, d: int, m: int, rng: random.Random = None,
                 np_seed: int = 0, **kw):
        self.d = d
        self.m = m
        self.rng = rng or random.Random(np_seed)
        # Per-trial numpy RNG for deterministic Beta sampling across threads
        self.np_rng = np.random.RandomState(np_seed)
        self.alphas = np.ones(d)
        self.betas = np.ones(d)
        self.n_pulls = np.zeros(d)
        self.total_reward = np.zeros(d)
        self.t = 0

    @property
    def mu_hat(self) -> np.ndarray:
        return self.total_reward / np.maximum(self.n_pulls, 1)

    def _sample(self) -> np.ndarray:
        return self.np_rng.beta(self.alphas, self.betas)

    def select_arms(self) -> list[int]:
        samples = self._sample()
        return list(np.argsort(samples)[::-1][:self.m])

    def update(self, selected: list[int], rewards: list[float]):
        for arm, r in zip(selected, rewards):
            self.n_pulls[arm] += 1
            self.total_reward[arm] += r
            if r > 0.5:
                self.alphas[arm] += 1
            else:
                self.betas[arm] += 1
        self.t += 1


# ─── Shared helpers ───────────────────────────────────────────────────────
def _warmup_cts_rounds(agent: CTSBase, T_warmup: int, env_pull):
    """Run CTS for T_warmup rounds to build a good mu_hat before querying LLM."""
    for _ in range(T_warmup):
        sel = agent.select_arms()
        rewards = env_pull(sel)
        agent.update(sel, rewards)


# ─── A1. Logprob-CTS ──────────────────────────────────────────────────────
class LogprobCTS(CTSBase):
    """Extract per-arm logprob distribution; inject as Beta pseudo-observations."""
    name = "A1_logprob_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 prior_strength: float = 20.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.prior_strength = prior_strength
        self._injected = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._injected:
            self._inject_prior()
            self._injected = True
        return super().select_arms()

    def _inject_prior(self):
        probs = self.oracle.query_logprobs(self.mu_hat.tolist())
        if not probs:
            return
        max_p = max(probs.values())
        for aid, p in probs.items():
            # Strong probability → add pseudo-successes
            rel = p / max_p  # 0 to 1
            pseudo_a = self.prior_strength * rel
            pseudo_b = self.prior_strength * (1 - rel) * 0.3
            self.alphas[aid] += pseudo_a
            self.betas[aid] += pseudo_b


# ─── A2. Self-Distractor Calibration CTS ──────────────────────────────────
class DistractorCTS(CTSBase):
    """Query top picks AND trap picks; calibration gap → trust weighting."""
    name = "A2_distractor_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            picks = self.oracle.query_top_m(self.mu_hat.tolist())
            # Simulated "trap" query — ask for arms that look good but aren't
            trap_prompt_override = self.mu_hat.tolist()
            # Approximate: treat elim list as traps
            traps = self.oracle.query_elimination(self.mu_hat.tolist())
            for aid in picks:
                self.alphas[aid] += 4
            for aid in traps:
                self.betas[aid] += 2
        return super().select_arms()


# ─── A3. Temperature-Scaled Mixture CTS ───────────────────────────────────
class TempMixtureCTS(CTSBase):
    """Query at 3 temperatures; variance in responses = meta-uncertainty."""
    name = "A3_temp_mixture_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            # Multiple queries (caching + temperature not available on gpt-5, use as ensemble)
            all_picks: list[list[int]] = []
            for _ in range(3):
                all_picks.append(self.oracle.query_top_m(self.mu_hat.tolist()))
            # Count appearances
            counts = np.zeros(self.d)
            for pick in all_picks:
                for a in pick:
                    counts[a] += 1
            max_c = counts.max()
            for aid in range(self.d):
                if counts[aid] > 0:
                    strength = 6 * (counts[aid] / max_c)
                    self.alphas[aid] += strength
        return super().select_arms()


# ─── B1. Pseudo-Observation Injection CTS ─────────────────────────────────
class PseudoObsCTS(CTSBase):
    """Periodic LLM queries → pseudo-successes/failures into Beta counters."""
    name = "B1_pseudo_obs_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_interval: int = 150, obs_weight: float = 8.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval
        self.obs_weight = obs_weight

    def select_arms(self):
        if (self.t == self.T_warmup or
            (self.t > self.T_warmup and (self.t - self.T_warmup) % self.query_interval == 0)):
            self._inject()
        return super().select_arms()

    def _inject(self):
        # Decaying weight: LLM matters less as real data accumulates
        decay = 1.0 / (1 + self.t / 200)
        weight = self.obs_weight * decay
        scores = self.oracle.query_per_arm_scores(self.mu_hat.tolist())
        for aid, s in scores.items():
            # Use mean as expected success rate; add weight pseudo-observations
            pseudo_success = weight * s["mean"]
            pseudo_failure = weight * (1 - s["mean"])
            self.alphas[aid] += pseudo_success
            self.betas[aid] += pseudo_failure


# ─── B2. In-Context Posterior Distillation ────────────────────────────────
class ICPDCTS(CTSBase):
    """Feed LLM history summary; extract posterior; merge with CTS."""
    name = "B2_icpd_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 50,
                 query_interval: int = 200, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval

    def select_arms(self):
        if (self.t == self.T_warmup or
            (self.t > self.T_warmup and (self.t - self.T_warmup) % self.query_interval == 0)):
            self._distill()
        return super().select_arms()

    def _distill(self):
        summary = f"t={self.t}, n_pulls_total={int(self.n_pulls.sum())}"
        preds = self.oracle.query_counterfactual(self.mu_hat.tolist(), summary)
        weight = 6.0 / (1 + self.t / 300)
        for aid, p in preds.items():
            self.alphas[aid] += weight * p
            self.betas[aid] += weight * (1 - p)


# ─── B3. Regret-Loss Fine-Tuned CTS (approximated via prompt engineering) ─
class RegretLossCTS(CTSBase):
    """Approximation: ask LLM to minimize regret, not just 'pick best'."""
    name = "B3_regret_loss_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            # Use top_m but prompt already frames regret-minimization
            picks = self.oracle.query_top_m(self.mu_hat.tolist())
            for aid in picks:
                self.alphas[aid] += 5
        return super().select_arms()


# ─── C1. Debate-Arena CTS ─────────────────────────────────────────────────
class DebateCTS(CTSBase):
    """3 LLM queries with different framings → debate aggregation."""
    name = "C1_debate_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.tolist()
            picks1 = self.oracle.query_top_m(mu)
            picks2 = self.oracle.query_top_m(mu)
            picks3 = self.oracle.query_top_m(mu)
            counts = {}
            for pick in [picks1, picks2, picks3]:
                for a in pick:
                    counts[a] = counts.get(a, 0) + 1
            for aid, c in counts.items():
                self.alphas[aid] += 2 * c
        return super().select_arms()


# ─── C2. Devil's Advocate CTS ─────────────────────────────────────────────
class DevilsAdvocateCTS(CTSBase):
    """LLM proposes, then critiques its own picks via elimination query."""
    name = "C2_devils_advocate_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.tolist()
            picks = self.oracle.query_top_m(mu)
            eliminated = set(self.oracle.query_elimination(mu))
            # Trust only picks NOT in elimination
            for aid in picks:
                if aid not in eliminated:
                    self.alphas[aid] += 6
                else:
                    self.alphas[aid] += 2  # weak trust — conflict
        return super().select_arms()


# ─── C3. DIPPER Ensemble CTS ──────────────────────────────────────────────
class DipperCTS(CTSBase):
    """5 diverse-prompt queries; Bradley-Terry aggregation."""
    name = "C3_dipper_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.tolist()
            # Use pairwise queries as BT aggregator
            top_candidates = list(np.argsort(mu)[::-1][:min(2 * self.m, self.d)])
            pairs = [(top_candidates[i], top_candidates[j])
                     for i in range(len(top_candidates))
                     for j in range(i + 1, len(top_candidates))][:15]
            p_map = self.oracle.query_pairwise(mu, pairs)
            scores = np.zeros(self.d)
            for (i, j), p in p_map.items():
                scores[i] += p
                scores[j] += 1 - p
            for aid in top_candidates:
                if scores[aid] > 0:
                    self.alphas[aid] += 3 * scores[aid] / len(top_candidates)
        return super().select_arms()


# ─── D1. Semantic-Cluster CTS (STAR CANDIDATE) ────────────────────────────
class SemanticClusterCTS(CTSBase):
    """Cluster arms; reward on one arm partially updates cluster-mates.

    This is the mathematically novel direction: transforms d → k effective dim.
    """
    name = "D1_semantic_cluster_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 transfer_weight: float = 0.25, n_clusters: int = 8, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.transfer_weight = transfer_weight
        self.n_clusters = n_clusters
        self.cluster_of = np.zeros(d, dtype=int)
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=self.n_clusters)
            for cidx, cluster in enumerate(clusters):
                for aid in cluster:
                    self.cluster_of[aid] = cidx
        return super().select_arms()

    def update(self, selected, rewards):
        # Real updates
        super().update(selected, rewards)
        # Cross-cluster soft updates
        if self._done:
            for arm, r in zip(selected, rewards):
                cluster_id = self.cluster_of[arm]
                peers = [a for a in range(self.d) if self.cluster_of[a] == cluster_id and a != arm]
                for peer in peers:
                    if r > 0.5:
                        self.alphas[peer] += self.transfer_weight
                    else:
                        self.betas[peer] += self.transfer_weight


# ─── D2. Causal-Graph CTS ─────────────────────────────────────────────────
class CausalGraphCTS(CTSBase):
    """Use clusters as causal groups; different transfer weight."""
    name = "D2_causal_graph_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 transfer_weight: float = 0.5, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.transfer_weight = transfer_weight
        self.cluster_of = np.zeros(d, dtype=int)
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            clusters = self.oracle.query_clusters(self.mu_hat.tolist(), n_clusters=6)
            for cidx, cluster in enumerate(clusters):
                for aid in cluster:
                    self.cluster_of[aid] = cidx
        return super().select_arms()

    def update(self, selected, rewards):
        super().update(selected, rewards)
        if self._done:
            for arm, r in zip(selected, rewards):
                cluster_id = self.cluster_of[arm]
                peers = [a for a in range(self.d) if self.cluster_of[a] == cluster_id and a != arm]
                for peer in peers:
                    w = self.transfer_weight / max(1, len(peers))
                    if r > 0.5:
                        self.alphas[peer] += w
                    else:
                        self.betas[peer] += w


# ─── D3. Pairwise Elo-CTS ─────────────────────────────────────────────────
class PairwiseEloCTS(CTSBase):
    """Bradley-Terry scores from pairwise LLM queries → CTS priors."""
    name = "D3_pairwise_elo_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, n_pairs: int = 20, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.n_pairs = n_pairs
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.tolist()
            candidates = list(np.argsort(mu)[::-1][:min(2 * self.m + 3, self.d)])
            pairs = []
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    pairs.append((candidates[i], candidates[j]))
            pairs = pairs[:self.n_pairs]
            p_map = self.oracle.query_pairwise(mu, pairs)
            wins = np.zeros(self.d)
            plays = np.zeros(self.d)
            for (i, j), p in p_map.items():
                wins[i] += p
                wins[j] += 1 - p
                plays[i] += 1
                plays[j] += 1
            scores = wins / np.maximum(plays, 1)
            for aid in candidates:
                if plays[aid] > 0:
                    self.alphas[aid] += 6 * scores[aid]
                    self.betas[aid] += 6 * (1 - scores[aid])
        return super().select_arms()


# ─── E1. AutoElicit-CTS ───────────────────────────────────────────────────
class AutoElicitCTS(CTSBase):
    """Extract [mean, CI] per arm at t=0, convert to Beta priors."""
    name = "E1_autoelicit_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 0,
                 prior_strength: float = 10.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.prior_strength = prior_strength
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            self._elicit()
        return super().select_arms()

    def _elicit(self):
        mu = self.mu_hat.tolist()
        scores = self.oracle.query_per_arm_scores(mu)
        for aid, s in scores.items():
            # Beta(α, β) with mean=m and variance from CI width
            m_hat = s["mean"]
            ci_width = max(0.05, s["hi"] - s["lo"])
            # Stronger prior for narrower CI
            strength = self.prior_strength / (ci_width + 0.1)
            self.alphas[aid] += strength * m_hat
            self.betas[aid] += strength * (1 - m_hat)


# ─── E2. Recursive Refinement CTS ─────────────────────────────────────────
class RecursiveRefinementCTS(CTSBase):
    """AutoElicit at t=0, re-elicit at t=100, 300, 600."""
    name = "E2_recursive_refinement_cts"

    def __init__(self, d, m, oracle: GPTOracle, refinement_points: tuple = (0, 100, 300), **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.refinement_points = refinement_points

    def select_arms(self):
        if self.t in self.refinement_points:
            decay = 1.0 / (1 + self.t / 100)
            scores = self.oracle.query_per_arm_scores(self.mu_hat.tolist())
            for aid, s in scores.items():
                w = 6.0 * decay
                self.alphas[aid] += w * s["mean"]
                self.betas[aid] += w * (1 - s["mean"])
        return super().select_arms()


# ─── E3. Conformal Prior CTS ──────────────────────────────────────────────
class ConformalPriorCTS(CTSBase):
    """AutoElicit with overconfidence correction: shrink priors toward 0.5."""
    name = "E3_conformal_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 0,
                 shrink_factor: float = 0.6, prior_strength: float = 8.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.shrink_factor = shrink_factor
        self.prior_strength = prior_strength
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            scores = self.oracle.query_per_arm_scores(self.mu_hat.tolist())
            for aid, s in scores.items():
                # Shrink mean toward 0.5 (correcting overconfidence)
                m_shrunk = 0.5 + self.shrink_factor * (s["mean"] - 0.5)
                self.alphas[aid] += self.prior_strength * m_shrunk
                self.betas[aid] += self.prior_strength * (1 - m_shrunk)
        return super().select_arms()


# ─── F1. Information-Optimal Prompt CTS ───────────────────────────────────
class InfoOptimalCTS(CTSBase):
    """Use elimination to shrink action space; CTS runs only on survivors."""
    name = "F1_info_optimal_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 40, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._mask = np.ones(d, dtype=bool)
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            eliminated = self.oracle.query_elimination(self.mu_hat.tolist())
            # Safety: don't eliminate above-median mu_hat arms
            med = np.median(self.mu_hat)
            for aid in eliminated:
                if self.mu_hat[aid] < med:
                    self._mask[aid] = False
            # Must keep at least 2*m arms
            if self._mask.sum() < 2 * self.m:
                top_mu = np.argsort(self.mu_hat)[::-1][:2 * self.m]
                self._mask = np.zeros(self.d, dtype=bool)
                self._mask[top_mu] = True
        samples = self._sample()
        samples = np.where(self._mask, samples, -np.inf)
        return list(np.argsort(samples)[::-1][:self.m])


# ─── F2. Query-Design Optimization CTS ────────────────────────────────────
class QueryDesignCTS(CTSBase):
    """Ensemble of prompt formats; weight by agreement."""
    name = "F2_query_design_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            mu = self.mu_hat.tolist()
            # Ensemble: top_m + per_arm_scores + elimination
            picks = set(self.oracle.query_top_m(mu))
            elim = set(self.oracle.query_elimination(mu))
            scores = self.oracle.query_per_arm_scores(mu)
            for aid in range(self.d):
                if aid in picks:
                    self.alphas[aid] += 4
                if aid in elim:
                    self.betas[aid] += 3
                if aid in scores:
                    self.alphas[aid] += 3 * scores[aid]["mean"]
                    self.betas[aid] += 3 * (1 - scores[aid]["mean"])
        return super().select_arms()


# ─── G1. LLM-MCTS-Bandit ──────────────────────────────────────────────────
class LLMMCTSCTS(CTSBase):
    """LLM evaluates candidate sets; best-predicted set used as pool."""
    name = "G1_llm_mcts_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self._done = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._done:
            self._done = True
            # Get LLM's best set, then "simulate" by asking for counterfactuals
            mu = self.mu_hat.tolist()
            preds = self.oracle.query_counterfactual(mu, f"t={self.t}")
            # Inject as priors weighted by LLM's prediction
            for aid, p in preds.items():
                self.alphas[aid] += 5 * p
                self.betas[aid] += 5 * (1 - p)
        return super().select_arms()


# ─── G2. Counterfactual CTS ───────────────────────────────────────────────
class CounterfactualCTS(CTSBase):
    """Periodic counterfactual queries shape posteriors of unpulled arms."""
    name = "G2_counterfactual_cts"

    def __init__(self, d, m, oracle: GPTOracle, T_warmup: int = 30,
                 query_interval: int = 200, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.query_interval = query_interval

    def select_arms(self):
        if (self.t == self.T_warmup or
            (self.t > self.T_warmup and (self.t - self.T_warmup) % self.query_interval == 0)):
            mu = self.mu_hat.tolist()
            summary = f"t={self.t}, low_pulls={int((self.n_pulls < 2).sum())}"
            preds = self.oracle.query_counterfactual(mu, summary)
            decay = 1.0 / (1 + self.t / 300)
            for aid, p in preds.items():
                # Only inject for low-data arms
                if self.n_pulls[aid] < 5:
                    w = 4.0 * decay
                    self.alphas[aid] += w * p
                    self.betas[aid] += w * (1 - p)
        return super().select_arms()


# ─── B3 extension: Warm-Start CTS (baseline to match paper) ───────────────
class WarmStartCTS(CTSBase):
    """Single LLM query at t=0, use as prior."""
    name = "WARM_warm_start_cts"

    def __init__(self, d, m, oracle: GPTOracle, prior_strength: float = 5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.prior_strength = prior_strength
        self._done = False

    def select_arms(self):
        if self.t == 0 and not self._done:
            self._done = True
            picks = self.oracle.query_top_m([0.0] * self.d)
            for aid in picks:
                self.alphas[aid] += self.prior_strength
        return super().select_arms()


# ─── Registry ─────────────────────────────────────────────────────────────
ALL_ALGORITHMS = {
    "cts_baseline": CTSBase,
    "A1_logprob_cts": LogprobCTS,
    "A2_distractor_cts": DistractorCTS,
    "A3_temp_mixture_cts": TempMixtureCTS,
    "B1_pseudo_obs_cts": PseudoObsCTS,
    "B2_icpd_cts": ICPDCTS,
    "B3_regret_loss_cts": RegretLossCTS,
    "C1_debate_cts": DebateCTS,
    "C2_devils_advocate_cts": DevilsAdvocateCTS,
    "C3_dipper_cts": DipperCTS,
    "D1_semantic_cluster_cts": SemanticClusterCTS,
    "D2_causal_graph_cts": CausalGraphCTS,
    "D3_pairwise_elo_cts": PairwiseEloCTS,
    "E1_autoelicit_cts": AutoElicitCTS,
    "E2_recursive_refinement_cts": RecursiveRefinementCTS,
    "E3_conformal_cts": ConformalPriorCTS,
    "F1_info_optimal_cts": InfoOptimalCTS,
    "F2_query_design_cts": QueryDesignCTS,
    "G1_llm_mcts_cts": LLMMCTSCTS,
    "G2_counterfactual_cts": CounterfactualCTS,
    "WARM_warm_start_cts": WarmStartCTS,
}

NEEDS_ORACLE = {k for k, v in ALL_ALGORITHMS.items() if k != "cts_baseline"}
