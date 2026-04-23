#!/usr/bin/env python3
"""Held-out evaluation: tests top algorithms on UNSEEN configs.

Uses master_seed=2024 (development was on seed=42) to generate fresh configs.
Also tests at longer horizon T=5000 and larger d values (200,300).
This validates that arena results are not overfitted to the development configs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combbandits.gpu.batched_agents import BatchedCUCB, BatchedCTS, BatchedWarmStartCTS
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO
from combbandits.gpu.batched_variants import (
    BatchedPoolCTS, BatchedPoolCTSInitCheck,
)
from arena_novel_algos_v2 import PoolCTSDual, FreqPoolTS, PoolCTSDoubling
from arena_novel_algos_v3 import FreqPoolCTSDual
from arena_novel_algos_v4 import AdaptivePoolCTS, PoolCTSAbstain, OracleBudgetCTS
from arena_novel_algos_v5 import AdaptivePoolDual, AdaptivePoolAbstain, AdaptiveFreqDual

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_heldout_configs(n_configs: int, master_seed: int = 2024,
                             include_large_d: bool = True) -> list[dict]:
    """Generate held-out configs with DIFFERENT seed than development."""
    rng = np.random.RandomState(master_seed)
    configs = []

    d_choices = [30, 50, 100, 150]
    if include_large_d:
        d_choices += [200, 300]
    m_choices_by_d = {
        30: [3, 5], 50: [5, 8], 100: [5, 10],
        150: [10, 15], 200: [10, 15], 300: [10, 20],
    }
    gap_types = ["uniform", "hard", "graded", "clustered"]
    corruption_types = ["uniform", "adversarial", "partial_overlap", "consistent_wrong"]

    for i in range(n_configs):
        d = rng.choice(d_choices)
        m = rng.choice(m_choices_by_d[d])
        gap_type = rng.choice(gap_types)
        delta_min = rng.uniform(0.03, 0.15)
        corruption_type = rng.choice(corruption_types)

        if corruption_type == "consistent_wrong":
            epsilon = 1.0
        elif corruption_type == "uniform":
            epsilon = rng.uniform(0.0, 0.5)
        elif corruption_type == "adversarial":
            epsilon = rng.uniform(0.05, 0.4)
        else:
            epsilon = rng.uniform(0.1, 0.5)

        configs.append({
            "config_id": i,
            "env": {
                "type": "synthetic_bernoulli",
                "d": int(d),
                "m": int(m),
                "gap_type": gap_type,
                "delta_min": float(delta_min),
                "seed": int(rng.randint(0, 100000)),
            },
            "oracle": {
                "corruption_type": corruption_type,
                "epsilon": float(epsilon),
                "K": 3,
            },
        })

    return configs


def build_top_factories() -> dict[str, callable]:
    """Only the top contenders + baselines."""
    factories = {}
    factories["cucb"] = lambda d, m, n_seeds, device, oracle: BatchedCUCB(d, m, n_seeds, device)
    factories["cts"] = lambda d, m, n_seeds, device, oracle: BatchedCTS(d, m, n_seeds, device)
    factories["warm_start_cts"] = lambda d, m, n_seeds, device, oracle: BatchedWarmStartCTS(d, m, n_seeds, device, oracle)
    factories["pool_cts"] = lambda d, m, n_seeds, device, oracle: BatchedPoolCTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_ic"] = lambda d, m, n_seeds, device, oracle: BatchedPoolCTSInitCheck(d, m, n_seeds, device, oracle, T_init=max(d // m * 5, 50), agreement_threshold=0.3)
    factories["pool_cts_dual"] = lambda d, m, n_seeds, device, oracle: PoolCTSDual(d, m, n_seeds, device, oracle)
    factories["freq_pool_ts"] = lambda d, m, n_seeds, device, oracle: FreqPoolTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_doubling"] = lambda d, m, n_seeds, device, oracle: PoolCTSDoubling(d, m, n_seeds, device, oracle)
    factories["freq_pool_cts_dual"] = lambda d, m, n_seeds, device, oracle: FreqPoolCTSDual(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_cts"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolCTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_abstain"] = lambda d, m, n_seeds, device, oracle: PoolCTSAbstain(d, m, n_seeds, device, oracle)
    factories["oracle_budget_cts"] = lambda d, m, n_seeds, device, oracle: OracleBudgetCTS(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_dual"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolDual(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_abstain"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolAbstain(d, m, n_seeds, device, oracle)
    factories["adaptive_freq_dual"] = lambda d, m, n_seeds, device, oracle: AdaptiveFreqDual(d, m, n_seeds, device, oracle)
    return factories


def run_single_config(config, agent_factories, T, n_seeds, device):
    """Run all agents on one config. Same as arena_runner but simplified."""
    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    d, m = env_cfg["d"], env_cfg["m"]

    rng = np.random.RandomState(env_cfg["seed"])
    gap_type = env_cfg["gap_type"]
    delta_min = env_cfg["delta_min"]

    if gap_type == "uniform":
        means_np = rng.uniform(0.1, 0.5, size=d)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + delta_min
    elif gap_type == "hard":
        means_np = np.full(d, 0.5)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + delta_min
    elif gap_type == "graded":
        means_np = np.linspace(0.1, 0.5 + delta_min, d)
        rng.shuffle(means_np)
    elif gap_type == "clustered":
        means_np = np.full(d, 0.3)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.7
        means_np += rng.normal(0, 0.02, size=d)
        means_np = np.clip(means_np, 0.01, 0.99)

    arm_means = torch.tensor(means_np, dtype=torch.float32, device=device)
    optimal_set = torch.argsort(arm_means, descending=True)[:m]
    optimal_reward = arm_means[optimal_set].sum().item()

    oracle = BatchedSimulatedCLO(
        d=d, m=m, n_seeds=n_seeds,
        optimal_set=optimal_set, arm_means=arm_means,
        corruption_type=oracle_cfg["corruption_type"],
        epsilon=oracle_cfg["epsilon"],
        K=oracle_cfg["K"], device=device,
    )

    results = []
    for agent_name, factory in agent_factories.items():
        oracle.total_queries = 0
        try:
            agent = factory(d=d, m=m, n_seeds=n_seeds, device=device, oracle=oracle)
        except Exception as e:
            logger.warning(f"  Skipping {agent_name}: {e}")
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

        elapsed = time.time() - t_start
        final_regret = cum_regret.cpu().numpy()
        results.append({
            "config_id": config["config_id"],
            "agent": agent_name,
            "d": d, "m": m,
            "gap_type": gap_type, "delta_min": delta_min,
            "corruption_type": oracle_cfg["corruption_type"],
            "epsilon": oracle_cfg["epsilon"],
            "regret_mean": float(final_regret.mean()),
            "regret_std": float(final_regret.std()),
            "regret_median": float(np.median(final_regret)),
            "elapsed_sec": elapsed,
            "T": T, "n_seeds": n_seeds,
        })
        agent.reset()

    return results


def analyze_results(results):
    import pandas as pd
    df = pd.DataFrame(results)
    lines = []
    lines.append("=" * 80)
    lines.append("HELD-OUT EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append(f"Total configs: {df['config_id'].nunique()}")
    lines.append(f"Total agent-config pairs: {len(df)}")
    lines.append("")

    overall = df.groupby("agent")["regret_mean"].agg(["mean", "median", "std"]).round(1)
    overall = overall.sort_values("mean")
    lines.append("--- OVERALL RANKING (lower = better) ---")
    lines.append(overall.to_string())
    lines.append("")

    for ct in sorted(df["corruption_type"].unique()):
        sub = df[df["corruption_type"] == ct]
        ranking = sub.groupby("agent")["regret_mean"].agg(["mean", "median"]).round(1)
        ranking = ranking.sort_values("mean")
        lines.append(f"--- {ct.upper()} (n={sub['config_id'].nunique()} configs) ---")
        lines.append(ranking.to_string())
        lines.append("")

    wins = {}
    for cid, group in df.groupby("config_id"):
        best = group.loc[group["regret_mean"].idxmin(), "agent"]
        wins[best] = wins.get(best, 0) + 1
    total = df["config_id"].nunique()
    lines.append("--- WIN RATE ---")
    for agent, count in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {agent:25s}: {count:3d}/{total} ({100*count/total:.1f}%)")
    lines.append("")

    worst = df.groupby("agent")["regret_mean"].max().sort_values()
    lines.append("--- WORST CASE ---")
    lines.append(worst.round(1).to_string())

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Held-out Evaluation")
    parser.add_argument("--n-configs", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--T", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--master-seed", type=int, default=2024)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Held-out eval: {args.n_configs} configs × {args.n_seeds} seeds × T={args.T} (seed={args.master_seed})")

    configs = generate_heldout_configs(args.n_configs, args.master_seed)
    factories = build_top_factories()
    logger.info(f"Testing {len(factories)} top algorithms")

    all_results = []
    for i, config in enumerate(configs):
        cfg_desc = (f"d={config['env']['d']} m={config['env']['m']} "
                    f"{config['env']['gap_type']} {config['oracle']['corruption_type']} "
                    f"eps={config['oracle']['epsilon']:.2f}")
        logger.info(f"Config {i+1}/{args.n_configs}: {cfg_desc}")
        results = run_single_config(config, factories, args.T, args.n_seeds, device)
        all_results.extend(results)

    report = analyze_results(all_results)
    print("\n" + report)

    output_dir = Path(__file__).parent.parent / "arena_results"
    output_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"heldout_{ts}_report.txt"
    results_path = output_dir / f"heldout_{ts}.json"

    with open(report_path, "w") as f:
        f.write(report)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
