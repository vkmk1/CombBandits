#!/usr/bin/env python3
"""Diagnostic: what warmup length gives good LLM oracle overlap for d=30, m=5?

Tests both round-robin and CTS warmup strategies at various lengths,
measuring oracle overlap quality at each point.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arena_real_llm import RealLLMBatchedOracle

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"


def run_warmup_test(d=30, m=5, seed=42, warmup_lengths=None):
    device = torch.device("cpu")
    rng = np.random.RandomState(seed)

    means_np = rng.uniform(0.1, 0.5, size=d)
    top_arms = rng.choice(d, size=m, replace=False)
    means_np[top_arms] = 0.6
    arm_means = torch.tensor(means_np, dtype=torch.float32, device=device)
    optimal_set = torch.argsort(arm_means, descending=True)[:m]
    optimal_reward = arm_means[optimal_set].sum().item()

    print(f"Config: d={d}, m={m}, optimal={optimal_set.tolist()}")
    print(f"Optimal means: {arm_means[optimal_set].tolist()}")
    print()

    if warmup_lengths is None:
        warmup_lengths = [0, 30, 60, 100, 150, 200, 300]

    # --- Test 1: Round-robin warmup ---
    print("=== ROUND-ROBIN WARMUP ===")
    for wl in warmup_lengths:
        n_pulls = torch.zeros(1, d, device=device)
        total_reward = torch.zeros(1, d, device=device)

        for t in range(wl):
            arm_idx = t % d
            reward = torch.bernoulli(arm_means[arm_idx].unsqueeze(0))
            n_pulls[0, arm_idx] += 1
            total_reward[0, arm_idx] += reward.item()

        mu_hat = total_reward / n_pulls.clamp(min=1)

        oracle = RealLLMBatchedOracle(
            d=d, m=m, K=1, model_id=MODEL, region=REGION,
            arm_means=arm_means, optimal_set=optimal_set,
        )
        out = oracle.query_batched(mu_hat)
        suggested = out["suggested_sets"][0].tolist()
        overlap = len(set(suggested) & set(optimal_set.tolist()))

        mu_top = torch.topk(mu_hat[0], m).indices.tolist()
        emp_overlap = len(set(mu_top) & set(optimal_set.tolist()))

        pulls_per_arm = wl / d if d > 0 else 0
        print(f"  warmup={wl:4d} ({pulls_per_arm:.1f} pulls/arm): "
              f"oracle_overlap={overlap}/{m}, emp_overlap={emp_overlap}/{m}, "
              f"suggested={suggested}")

    # --- Test 2: CTS warmup (smarter exploration) ---
    print()
    print("=== CTS WARMUP ===")
    for wl in warmup_lengths:
        n_pulls = torch.zeros(1, d, device=device)
        total_reward = torch.zeros(1, d, device=device)
        alphas = torch.ones(1, d, device=device)
        betas = torch.ones(1, d, device=device)

        for t in range(wl):
            samples = torch.distributions.Beta(alphas, betas).sample()
            selected = torch.topk(samples, m, dim=1).indices

            selected_means = arm_means[selected]
            rewards = torch.bernoulli(selected_means)

            for j in range(m):
                arm = selected[0, j].item()
                rew = rewards[0, j].item()
                n_pulls[0, arm] += 1
                total_reward[0, arm] += rew
                if rew > 0.5:
                    alphas[0, arm] += 1
                else:
                    betas[0, arm] += 1

        mu_hat = total_reward / n_pulls.clamp(min=1)

        oracle = RealLLMBatchedOracle(
            d=d, m=m, K=1, model_id=MODEL, region=REGION,
            arm_means=arm_means, optimal_set=optimal_set,
        )
        out = oracle.query_batched(mu_hat)
        suggested = out["suggested_sets"][0].tolist()
        overlap = len(set(suggested) & set(optimal_set.tolist()))

        mu_top = torch.topk(mu_hat[0], m).indices.tolist()
        emp_overlap = len(set(mu_top) & set(optimal_set.tolist()))
        pulled = (n_pulls[0] > 0).sum().item()

        print(f"  warmup={wl:4d} ({pulled}/{d} arms pulled): "
              f"oracle_overlap={overlap}/{m}, emp_overlap={emp_overlap}/{m}, "
              f"suggested={suggested}")

    # --- Test 3: Hybrid - round-robin d rounds then CTS ---
    print()
    print("=== HYBRID: d rounds round-robin + CTS ===")
    for extra_cts in [0, 30, 60, 100, 150]:
        total_warmup = d + extra_cts
        n_pulls = torch.zeros(1, d, device=device)
        total_reward = torch.zeros(1, d, device=device)
        alphas = torch.ones(1, d, device=device)
        betas = torch.ones(1, d, device=device)

        for t in range(d):
            arm_idx = t % d
            reward = torch.bernoulli(arm_means[arm_idx].unsqueeze(0))
            n_pulls[0, arm_idx] += 1
            total_reward[0, arm_idx] += reward.item()
            if reward.item() > 0.5:
                alphas[0, arm_idx] += 1
            else:
                betas[0, arm_idx] += 1

        for t in range(extra_cts):
            samples = torch.distributions.Beta(alphas, betas).sample()
            selected = torch.topk(samples, m, dim=1).indices
            selected_means = arm_means[selected]
            rewards = torch.bernoulli(selected_means)
            for j in range(m):
                arm = selected[0, j].item()
                rew = rewards[0, j].item()
                n_pulls[0, arm] += 1
                total_reward[0, arm] += rew
                if rew > 0.5:
                    alphas[0, arm] += 1
                else:
                    betas[0, arm] += 1

        mu_hat = total_reward / n_pulls.clamp(min=1)

        oracle = RealLLMBatchedOracle(
            d=d, m=m, K=1, model_id=MODEL, region=REGION,
            arm_means=arm_means, optimal_set=optimal_set,
        )
        out = oracle.query_batched(mu_hat)
        suggested = out["suggested_sets"][0].tolist()
        overlap = len(set(suggested) & set(optimal_set.tolist()))

        mu_top = torch.topk(mu_hat[0], m).indices.tolist()
        emp_overlap = len(set(mu_top) & set(optimal_set.tolist()))

        print(f"  warmup={total_warmup:4d} (d+{extra_cts} CTS): "
              f"oracle_overlap={overlap}/{m}, emp_overlap={emp_overlap}/{m}, "
              f"suggested={suggested}")


if __name__ == "__main__":
    run_warmup_test()
