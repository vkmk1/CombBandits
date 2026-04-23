#!/usr/bin/env python3
"""V2 Real LLM oracle + CTS-warmup algorithms.

Key fix: all oracle algorithms now use CTS warmup instead of round-robin.

Finding from diagnostic: round-robin warmup gives 0/5 oracle overlap at d=30, m=5
because each arm only gets 1 binary pull. CTS warmup naturally concentrates pulls
on promising arms, giving 4/5 overlap at the same budget (30 rounds).

This is the paper's central contribution: with endogenous oracle quality,
the warmup strategy determines oracle effectiveness.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path

import boto3
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combbandits.gpu.batched_agents import BatchedAgentBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class RealLLMBatchedOracle:
    """Adapter: wraps a real Bedrock LLM to match the query_batched interface."""

    def __init__(self, d: int, m: int, K: int = 1,
                 model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                 region: str = "us-east-1",
                 temperature: float = 0.7,
                 arm_means: torch.Tensor = None,
                 optimal_set: torch.Tensor = None):
        self.d = d
        self.m = m
        self.K = K
        self.model_id = model_id
        self.temperature = temperature
        self.total_queries = 0
        self.total_tokens = 0
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._arm_means = arm_means
        self._optimal_set = optimal_set
        self._call_log = []

    def _build_prompt(self, mu_hat_np: np.ndarray, variant: int = 0) -> str:
        d, m = self.d, self.m
        ranked = sorted(range(d), key=lambda i: mu_hat_np[i], reverse=True)

        arm_lines = []
        for i in ranked[:min(40, d)]:
            arm_lines.append(f"Arm {i}: avg_reward={mu_hat_np[i]:.3f}")
        arm_block = "\n".join(arm_lines)

        variants = [
            (f"You are an expert advisor for a multi-armed bandit problem.\n\n"
             f"Select exactly {m} arms from {d} candidates to maximize total reward.\n"
             f"Here are the arms ranked by current average reward:\n\n{arm_block}\n\n"
             f"Pick the {m} arms most likely to have the highest TRUE reward.\n"
             f"Consider that average rewards are noisy estimates.\n\n"
             f"Return ONLY a JSON list of exactly {m} integer IDs. Example: [3, 7, 12, 1, 9]"),
            (f"Sequential optimization: pick {m} of {d} options to maximize reward.\n\n"
             f"Current estimates:\n{arm_block}\n\n"
             f"Which {m} items are best? Consider noise in the estimates.\n"
             f"Output: JSON array of {m} integer IDs."),
            (f"Help select the best subset. Choose {m} from {d}.\n\n"
             f"Performance data:\n{arm_block}\n\n"
             f"Return a JSON list of {m} arm IDs."),
        ]
        return variants[variant % len(variants)]

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        resp = self._client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 128, "temperature": self.temperature},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        tokens = usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
        return text, tokens

    def _parse_response(self, text: str) -> list[int]:
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list):
                ids = [int(x) for x in parsed if 0 <= int(x) < self.d]
                if len(ids) >= self.m:
                    return ids[:self.m]
        except (json.JSONDecodeError, ValueError):
            pass
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', text) if 0 <= int(x) < self.d]
        if len(numbers) >= self.m:
            return numbers[:self.m]
        while len(numbers) < self.m:
            for i in range(self.d):
                if i not in numbers:
                    numbers.append(i)
                    if len(numbers) == self.m:
                        break
        return numbers[:self.m]

    def query_batched(self, mu_hat: torch.Tensor) -> dict:
        n_seeds = mu_hat.shape[0]
        assert n_seeds == 1, "Real LLM oracle only supports n_seeds=1"
        mu_hat_np = mu_hat[0].cpu().numpy()

        all_sets = []
        total_tok = 0
        for k in range(self.K):
            prompt = self._build_prompt(mu_hat_np, variant=k)
            text, tok = self._call_llm(prompt)
            total_tok += tok
            suggested = self._parse_response(text)
            all_sets.append(suggested)

            overlap = len(set(suggested) & set(self._optimal_set.cpu().numpy()))
            self._call_log.append({
                "query_k": k, "suggested": suggested,
                "overlap_with_optimal": overlap,
                "tokens": tok,
            })

        self.total_queries += self.K
        self.total_tokens += total_tok

        primary = torch.tensor([all_sets[0]], dtype=torch.long, device=mu_hat.device)

        masks = []
        for s in all_sets:
            mask = torch.zeros(1, self.d, dtype=torch.bool, device=mu_hat.device)
            mask[0, s] = True
            masks.append(mask)
        intersection = masks[0]
        for mask in masks[1:]:
            intersection = intersection & mask
        kappa = intersection.sum(dim=1).float() / self.m

        return {"suggested_sets": primary, "consistency": kappa}

    def get_diagnostics(self) -> dict:
        if not self._call_log:
            return {}
        overlaps = [c["overlap_with_optimal"] for c in self._call_log]
        return {
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "mean_overlap": np.mean(overlaps),
            "overlap_rate": np.mean([o == self.m for o in overlaps]),
            "n_calls": len(self._call_log),
        }


# ═══════════════════════════════════════════════════════════════════════
# CTS-Warmup Pool-CTS (replaces round-robin WarmPoolCTS)
# ═══════════════════════════════════════════════════════════════════════

class CTSWarmPoolCTS(BatchedAgentBase):
    """Pool-CTS with CTS warmup instead of round-robin.

    Phase 1: CTS exploration (concentrates pulls on promising arms)
    Phase 2: Query oracle with high-quality mu_hat, build pool
    Phase 3: CTS exploitation on oracle pool

    CTS warmup gives 4/5 oracle overlap at 30 rounds (d=30, m=5) vs
    0/5 for round-robin at the same budget.
    """
    name = "cts_warm_pool"

    def __init__(self, d, m, n_seeds, device, oracle, beta=3,
                 n_pool_rounds=10, T_warmup=None):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.T_warmup = T_warmup if T_warmup is not None else max(d, 30)

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._pool_built = False

    def select_arms(self):
        if self.t < self.T_warmup:
            samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
            return torch.topk(samples, self.m, dim=1).indices

        if not self._pool_built:
            self._build_pool()

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def _build_pool(self):
        pool_counts = torch.zeros(self.n_seeds, self.d, device=self.device)
        for _ in range(self.n_pool_rounds):
            out = self.oracle.query_batched(self.mu_hat)
            sugg = out["suggested_sets"]
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))
        pool_arms = torch.topk(pool_counts, self.pool_size, dim=1).indices
        self.pool_mask.scatter_(1, pool_arms, True)
        self._pool_built = True
        logger.info(f"  CTSWarmPool: pool built after {self.T_warmup} CTS warmup rounds")

    def update(self, selected, rewards):
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


# ═══════════════════════════════════════════════════════════════════════
# Adaptive Pool + CTS Warmup + Abstention
# ═══════════════════════════════════════════════════════════════════════

class CTSWarmAdaptiveAbstain(BatchedAgentBase):
    """AdaptivePoolAbstain with CTS warmup (our best algo + best warmup).

    Same as AdaptivePoolAbstain but replaces round-robin warmup with CTS.
    Also re-queries oracle periodically as more data accumulates.
    """
    name = "cts_warm_adaptive_abstain"

    def __init__(self, d, m, n_seeds, device, oracle, beta_init=2.0,
                 beta_max=8.0, n_safety=5, n_pool_rounds=10,
                 T_warmup=None, abstain_threshold=0.3):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.n_safety = n_safety
        self.n_pool_rounds = n_pool_rounds
        self.T_warmup = T_warmup if T_warmup is not None else max(d, 30)
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
            pool_counts.scatter_add_(1, sugg, torch.ones_like(sugg, dtype=torch.float32))

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

    def select_arms(self):
        if self.t < self.T_warmup:
            samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
            return torch.topk(samples, self.m, dim=1).indices

        if not self._pool_built:
            self._build_pool_and_decide()

        if self.t > 0 and self.t % 150 == 0:
            self._check_coverage()

        if self.t > 0 and self.t % 300 == 0:
            self._build_pool_and_decide()

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        pool_s = samples.clone()
        pool_s[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_s, self.m, dim=1).indices
        full_action = torch.topk(samples, self.m, dim=1).indices

        result = full_action.clone()
        result[self._use_oracle] = pool_action[self._use_oracle]
        return result

    def update(self, selected, rewards):
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


# ═══════════════════════════════════════════════════════════════════════
# KWIK Query CTS with CTS Warmup (guarantees data quality before queries)
# ═══════════════════════════════════════════════════════════════════════

class CTSWarmKWIKQuery(BatchedAgentBase):
    """KWIK-style oracle querying with CTS warmup.

    Phase 1: CTS exploration (no oracle queries)
    Phase 2+: KWIK-triggered oracle queries when uncertainty high + data sufficient
    Re-queries periodically as mu_hat improves.
    """
    name = "cts_warm_kwik"

    def __init__(self, d, m, n_seeds, device, oracle, beta=3.0,
                 n_pool_rounds=5, n_safety=5,
                 T_warmup=None, min_pulls_trigger=2,
                 uncertainty_threshold=0.3, query_cooldown=50,
                 max_queries=100):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_warmup = T_warmup if T_warmup is not None else max(d, 30)
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

    def _should_query(self):
        if self._total_oracle_batches >= self.max_queries:
            return False
        if self.t - self._last_query_t < self.query_cooldown:
            return False
        cb = torch.sqrt(2.0 * math.log(max(self.t, 1) + 1) / self.n_pulls.clamp(min=1))
        top_m_indices = torch.topk(self.mu_hat, self.m, dim=1).indices
        top_m_cb = cb.gather(1, top_m_indices)
        return top_m_cb.mean() > self.uncertainty_threshold

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

    def select_arms(self):
        if self.t < self.T_warmup:
            samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
            return torch.topk(samples, self.m, dim=1).indices

        if self.t == self.T_warmup or (self.t > self.T_warmup and self._should_query()):
            self._query_and_build()

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        if self._has_pool:
            samples[~self.pool_mask] = -float("inf")
        return torch.topk(samples, self.m, dim=1).indices

    def update(self, selected, rewards):
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


# ═══════════════════════════════════════════════════════════════════════
# EOE with CTS warmup (instead of round-robin explore phase)
# ═══════════════════════════════════════════════════════════════════════

class CTSWarmEOE(BatchedAgentBase):
    """Explore-Oracle-Exploit with CTS warmup.

    Same three phases but uses CTS (not round-robin) for exploration,
    dramatically improving oracle overlap in the oracle phase.
    """
    name = "cts_warm_eoe"

    def __init__(self, d, m, n_seeds, device, oracle, beta=3.0,
                 n_pool_rounds=10, n_safety=5,
                 T_warmup=None, abstain_threshold=0.3,
                 stability_window=20, stability_threshold=0.7):
        super().__init__(d, m, n_seeds, device)
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.n_safety = n_safety
        self.T_warmup = T_warmup if T_warmup is not None else max(d, 30)
        self.abstain_threshold = abstain_threshold
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self.alphas = torch.ones(n_seeds, d, device=device)
        self.betas_param = torch.ones(n_seeds, d, device=device)
        self._phase = "explore"
        self._use_oracle = torch.ones(n_seeds, dtype=torch.bool, device=device)
        self._prev_top_m = None
        self._stability_count = torch.zeros(n_seeds, device=device)

    def _check_stability(self):
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

    def select_arms(self):
        if self._phase == "explore":
            if self.t >= self.T_warmup:
                ready = self._check_stability()
                if ready.all() or self.t >= self.T_warmup * 3:
                    self._phase = "exploit"
                    self._build_pool_and_decide()

            if self._phase == "explore":
                samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
                return torch.topk(samples, self.m, dim=1).indices

        samples = torch.distributions.Beta(self.alphas, self.betas_param).sample()
        pool_s = samples.clone()
        pool_s[~self.pool_mask] = -float("inf")
        pool_action = torch.topk(pool_s, self.m, dim=1).indices
        full_action = torch.topk(samples, self.m, dim=1).indices

        result = full_action.clone()
        result[self._use_oracle] = pool_action[self._use_oracle]
        return result

    def update(self, selected, rewards):
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
        self._phase = "explore"
        self._use_oracle.fill_(True)
        self._prev_top_m = None
        self._stability_count.zero_()
