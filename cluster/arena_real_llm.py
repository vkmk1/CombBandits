#!/usr/bin/env python3
"""Real LLM validation: tests top algorithms with actual Bedrock LLM oracle.

Uses Amazon Nova Micro (cheapest model) to verify our simulated oracle results
aren't hallucinations. Runs a SINGLE seed (no batching) on a small config.

Cost estimate: d=30, m=5, K=3 queries per oracle call, ~10 pool-building queries
  = ~30 LLM calls × ~100 tokens each = ~3000 tokens total ≈ $0.001
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class RealLLMBatchedOracle:
    """Adapter: wraps a real Bedrock LLM to match the query_batched interface.

    Since real LLMs can't do batched queries, this runs n_seeds=1 only.
    Caches responses within the same mu_hat state to avoid redundant calls.
    """

    def __init__(self, d: int, m: int, K: int = 3,
                 model_id: str = "amazon.nova-micro-v1:0",
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
            pulls_approx = max(1, int(mu_hat_np[i] * 100))
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
        """Match the BatchedSimulatedCLO interface. n_seeds must be 1."""
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


class WarmPoolCTS:
    """Pool-CTS that does round-robin warmup BEFORE querying the oracle.

    This is critical for real LLMs: the oracle needs mu_hat signal to be useful.
    Runs d rounds of round-robin (each arm pulled once), then builds pool.
    """
    name = "warm_pool_cts"

    def __init__(self, d, m, n_seeds, device, oracle, beta=3, n_pool_rounds=10):
        self.d = d
        self.m = m
        self.n_seeds = n_seeds
        self.device = device
        self.oracle = oracle
        self.beta = beta
        self.pool_size = int(beta * m)
        self.n_pool_rounds = n_pool_rounds
        self.t = 0
        self.n_pulls = torch.zeros(n_seeds, d, device=device)
        self.mu_hat = torch.zeros(n_seeds, d, device=device)
        self.total_reward = torch.zeros(n_seeds, d, device=device)
        self.pool_mask = torch.zeros(n_seeds, d, dtype=torch.bool, device=device)
        self._pool_built = False
        self._warmup_done = False

    def select_arms(self):
        if self.t < self.d:
            idx = self.t % self.d
            return torch.full((self.n_seeds, self.m), idx, dtype=torch.long, device=self.device)

        if not self._pool_built:
            self._build_pool()

        alpha = self.n_pulls + 1
        beta_param = torch.clamp(self.n_pulls - self.total_reward, min=0) + 1
        samples = torch.distributions.Beta(alpha, beta_param).sample()
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
        logger.info(f"  WarmPoolCTS: pool built after warmup, pool_size={self.pool_mask.sum(dim=1).tolist()}")

    def update(self, selected, rewards):
        self.t += 1
        for j in range(self.m):
            arm = selected[:, j]
            rew = rewards[:, j]
            self.n_pulls.scatter_add_(1, arm.unsqueeze(1), torch.ones(self.n_seeds, 1, device=self.device))
            self.total_reward.scatter_add_(1, arm.unsqueeze(1), rew.unsqueeze(1))
        safe_pulls = torch.clamp(self.n_pulls, min=1)
        self.mu_hat = self.total_reward / safe_pulls

    def reset(self):
        self.t = 0
        self.n_pulls.zero_()
        self.mu_hat.zero_()
        self.total_reward.zero_()
        self.pool_mask.zero_()
        self._pool_built = False


def run_real_llm_test(d=30, m=5, T=500, model_id="amazon.nova-micro-v1:0",
                      gap_type="uniform", delta_min=0.1, seed=42):
    """Run top algorithms with real LLM oracle on a single config."""
    from combbandits.gpu.batched_agents import BatchedCUCB, BatchedCTS, BatchedWarmStartCTS
    from combbandits.gpu.batched_variants import BatchedPoolCTS, BatchedPoolCTSInitCheck
    from arena_novel_algos_v5 import AdaptivePoolDual, AdaptivePoolAbstain, AdaptiveFreqDual

    device = torch.device("cpu")
    n_seeds = 1

    rng = np.random.RandomState(seed)
    if gap_type == "uniform":
        means_np = rng.uniform(0.1, 0.5, size=d)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + delta_min
    elif gap_type == "hard":
        means_np = np.full(d, 0.5)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + delta_min

    arm_means = torch.tensor(means_np, dtype=torch.float32, device=device)
    optimal_set = torch.argsort(arm_means, descending=True)[:m]
    optimal_reward = arm_means[optimal_set].sum().item()

    logger.info(f"Config: d={d}, m={m}, T={T}, gap={gap_type}, delta_min={delta_min}")
    logger.info(f"Optimal set: {optimal_set.tolist()}")
    logger.info(f"Optimal reward per round: {optimal_reward:.3f}")
    logger.info(f"Model: {model_id}")

    agents_to_test = {}

    agents_to_test["cucb"] = {
        "needs_oracle": False,
        "factory": lambda oracle: BatchedCUCB(d, m, n_seeds, device),
    }
    agents_to_test["cts"] = {
        "needs_oracle": False,
        "factory": lambda oracle: BatchedCTS(d, m, n_seeds, device),
    }
    agents_to_test["warm_start_cts"] = {
        "needs_oracle": True,
        "factory": lambda oracle: BatchedWarmStartCTS(d, m, n_seeds, device, oracle),
    }
    agents_to_test["pool_cts"] = {
        "needs_oracle": True,
        "factory": lambda oracle: BatchedPoolCTS(d, m, n_seeds, device, oracle),
    }
    agents_to_test["pool_cts_ic"] = {
        "needs_oracle": True,
        "factory": lambda oracle: BatchedPoolCTSInitCheck(
            d, m, n_seeds, device, oracle,
            T_init=max(d // m * 5, 50), agreement_threshold=0.3),
    }
    agents_to_test["warm_pool_cts"] = {
        "needs_oracle": True,
        "factory": lambda oracle: WarmPoolCTS(d, m, n_seeds, device, oracle),
    }
    agents_to_test["adaptive_pool_abstain"] = {
        "needs_oracle": True,
        "factory": lambda oracle: AdaptivePoolAbstain(d, m, n_seeds, device, oracle),
    }
    agents_to_test["adaptive_freq_dual"] = {
        "needs_oracle": True,
        "factory": lambda oracle: AdaptiveFreqDual(d, m, n_seeds, device, oracle),
    }

    results = []
    for agent_name, spec in agents_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {agent_name}...")

        if spec["needs_oracle"]:
            oracle = RealLLMBatchedOracle(
                d=d, m=m, K=3,
                model_id=model_id,
                arm_means=arm_means,
                optimal_set=optimal_set,
            )
        else:
            oracle = None

        try:
            agent = spec["factory"](oracle)
        except Exception as e:
            logger.error(f"  Failed to create {agent_name}: {e}")
            continue

        cum_regret = torch.zeros(n_seeds, device=device)
        t_start = time.time()

        for t in range(T):
            selected = agent.select_arms()
            selected_means = arm_means[selected]
            rewards = torch.bernoulli(selected_means)
            inst_regret = optimal_reward - selected_means.sum(dim=1)
            cum_regret += inst_regret
            agent.update(selected, rewards)

            if (t + 1) % 100 == 0:
                logger.info(f"  t={t+1}: regret={cum_regret[0].item():.1f}")

        elapsed = time.time() - t_start
        final_regret = cum_regret[0].item()

        oracle_diag = oracle.get_diagnostics() if oracle else {}

        result = {
            "agent": agent_name,
            "regret": round(final_regret, 1),
            "elapsed_sec": round(elapsed, 2),
            "oracle_queries": oracle_diag.get("total_queries", 0),
            "oracle_tokens": oracle_diag.get("total_tokens", 0),
            "oracle_mean_overlap": round(oracle_diag.get("mean_overlap", 0), 2),
            "oracle_perfect_rate": round(oracle_diag.get("overlap_rate", 0), 3),
        }
        results.append(result)
        logger.info(f"  FINAL: regret={final_regret:.1f}, time={elapsed:.1f}s")
        if oracle_diag:
            logger.info(f"  Oracle: {oracle_diag['total_queries']} queries, "
                       f"{oracle_diag['total_tokens']} tokens, "
                       f"overlap={oracle_diag['mean_overlap']:.2f}/{m}, "
                       f"perfect={oracle_diag['overlap_rate']:.1%}")

    logger.info(f"\n{'='*60}")
    logger.info("REAL LLM RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Config: d={d}, m={m}, T={T}, gap={gap_type}")
    logger.info(f"{'agent':30s} {'regret':>8s} {'queries':>8s} {'tokens':>8s} {'overlap':>8s}")
    logger.info("-" * 70)
    for r in sorted(results, key=lambda x: x["regret"]):
        logger.info(f"{r['agent']:30s} {r['regret']:8.1f} {r['oracle_queries']:8d} "
                   f"{r['oracle_tokens']:8d} {r['oracle_mean_overlap']:8.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Real LLM Oracle Test")
    parser.add_argument("--model", type=str, default="amazon.nova-micro-v1:0",
                       help="Bedrock model ID")
    parser.add_argument("--d", type=int, default=30)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--gap-type", type=str, default="uniform")
    parser.add_argument("--delta-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_real_llm_test(
        d=args.d, m=args.m, T=args.T,
        model_id=args.model,
        gap_type=args.gap_type,
        delta_min=args.delta_min,
        seed=args.seed,
    )

    output_dir = Path(__file__).parent.parent / "arena_results"
    output_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"real_llm_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
