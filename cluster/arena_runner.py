#!/usr/bin/env python3
"""Randomized Test Arena for LLM+Combinatorial Bandit Algorithms.

Generates random (env, oracle) configurations to prevent overfitting.
Runs all algorithms through the same random configs and reports
aggregate performance metrics.

Usage:
    python cluster/arena_runner.py [--n-configs 50] [--n-seeds 10] [--T 5000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combbandits.gpu.batched_agents import (
    BatchedAgentBase, BatchedCUCB, BatchedCTS, BatchedLLMCUCBAT,
    BatchedWarmStartCTS, BatchedEXP4,
)
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO
from combbandits.gpu.batched_variants import (
    BatchedPoolCTS, BatchedPoolCTSCG, BatchedPoolCTSInitCheck,
    BatchedPoolCTSETC,
)
from arena_novel_algos import (
    FrequencyWeightedTS, SuccessiveEliminationPool, CorralSelector,
    PoolTSExpandable, DecayingPriorTS, HedgeTSHybrid, OracleEpsGreedy,
)
from arena_novel_algos_v2 import (
    PoolCTSDual, FreqPoolTS, PoolCTSSafetyNet, PoolCTSDoubling,
)
from arena_novel_algos_v3 import (
    PoolCTSDualDoubling, FreqPoolCTSDual, PoolCTSAdaptiveDoubling,
)
from arena_novel_algos_v4 import (
    MetaPoolCTS, AdaptivePoolCTS, LUCBPool, OracleBudgetCTS,
    MultiOraclePool, ConsistencyRobustCTS, PoolCTSAbstain, ParetoMetaDual,
)
from arena_novel_algos_v5 import (
    AdaptivePoolDual, AdaptivePoolAbstain, AdaptivePoolDoubling, AdaptiveFreqDual,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ==========================================================================
# Random config generation
# ==========================================================================

def generate_random_configs(n_configs: int, master_seed: int = 42) -> list[dict]:
    """Generate n_configs random (env, oracle) configurations.

    Randomizes: d, m, gap_type, delta_min, corruption_type, epsilon.
    Each config gets a unique base_seed so environments differ.
    """
    rng = np.random.RandomState(master_seed)
    configs = []

    d_choices = [30, 50, 100, 150]
    m_choices_by_d = {30: [3, 5], 50: [5, 8], 100: [5, 10], 150: [10, 15]}
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


# ==========================================================================
# Single trial runner
# ==========================================================================

def run_single_config(
    config: dict,
    agent_factories: dict[str, callable],
    T: int,
    n_seeds: int,
    device: torch.device,
) -> list[dict]:
    """Run all agents on one random config. Returns list of result dicts."""
    env_cfg = config["env"]
    oracle_cfg = config["oracle"]
    d, m = env_cfg["d"], env_cfg["m"]

    # Build environment
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

    # Build oracle
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
            "regret_max": float(final_regret.max()),
            "regret_min": float(final_regret.min()),
            "elapsed_sec": elapsed,
            "T": T,
            "n_seeds": n_seeds,
        })

        agent.reset()

    return results


# ==========================================================================
# Agent factories
# ==========================================================================

def build_agent_factories() -> dict[str, callable]:
    """Build factory functions for all algorithms to test."""
    factories = {}

    # --- Baselines ---
    factories["cucb"] = lambda d, m, n_seeds, device, oracle: BatchedCUCB(d, m, n_seeds, device)
    factories["cts"] = lambda d, m, n_seeds, device, oracle: BatchedCTS(d, m, n_seeds, device)
    factories["llm_cucb_at"] = lambda d, m, n_seeds, device, oracle: BatchedLLMCUCBAT(d, m, n_seeds, device, oracle)
    factories["warm_start_cts"] = lambda d, m, n_seeds, device, oracle: BatchedWarmStartCTS(d, m, n_seeds, device, oracle)

    # --- Existing champions ---
    factories["pool_cts"] = lambda d, m, n_seeds, device, oracle: BatchedPoolCTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_cg"] = lambda d, m, n_seeds, device, oracle: BatchedPoolCTSCG(
        d, m, n_seeds, device, oracle, T_init=max(d // m, 10), sigma=0.3)
    factories["pool_cts_ic"] = lambda d, m, n_seeds, device, oracle: BatchedPoolCTSInitCheck(
        d, m, n_seeds, device, oracle, T_init=max(d // m * 5, 50), agreement_threshold=0.3)

    # --- Novel algorithms (Loop 1) ---
    factories["freq_weighted_ts"] = lambda d, m, n_seeds, device, oracle: FrequencyWeightedTS(d, m, n_seeds, device, oracle)
    factories["succ_elim_pool"] = lambda d, m, n_seeds, device, oracle: SuccessiveEliminationPool(d, m, n_seeds, device, oracle)
    factories["corral"] = lambda d, m, n_seeds, device, oracle: CorralSelector(d, m, n_seeds, device, oracle)
    factories["pool_ts_expand"] = lambda d, m, n_seeds, device, oracle: PoolTSExpandable(d, m, n_seeds, device, oracle)
    factories["decaying_prior_ts"] = lambda d, m, n_seeds, device, oracle: DecayingPriorTS(d, m, n_seeds, device, oracle)
    factories["hedge_ts_hybrid"] = lambda d, m, n_seeds, device, oracle: HedgeTSHybrid(d, m, n_seeds, device, oracle)
    factories["oracle_eps_greedy"] = lambda d, m, n_seeds, device, oracle: OracleEpsGreedy(d, m, n_seeds, device, oracle)

    # --- Novel algorithms (Loop 2) ---
    factories["pool_cts_dual"] = lambda d, m, n_seeds, device, oracle: PoolCTSDual(d, m, n_seeds, device, oracle)
    factories["freq_pool_ts"] = lambda d, m, n_seeds, device, oracle: FreqPoolTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_safety"] = lambda d, m, n_seeds, device, oracle: PoolCTSSafetyNet(d, m, n_seeds, device, oracle)
    factories["pool_cts_doubling"] = lambda d, m, n_seeds, device, oracle: PoolCTSDoubling(d, m, n_seeds, device, oracle)

    # --- Novel algorithms (Loop 2 hybrids) ---
    factories["pool_cts_dual_doubling"] = lambda d, m, n_seeds, device, oracle: PoolCTSDualDoubling(d, m, n_seeds, device, oracle)
    factories["freq_pool_cts_dual"] = lambda d, m, n_seeds, device, oracle: FreqPoolCTSDual(d, m, n_seeds, device, oracle)
    factories["pool_cts_adapt_doubling"] = lambda d, m, n_seeds, device, oracle: PoolCTSAdaptiveDoubling(d, m, n_seeds, device, oracle)

    # --- Novel algorithms (Loop 3 — research directions) ---
    factories["meta_pool_cts"] = lambda d, m, n_seeds, device, oracle: MetaPoolCTS(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_cts"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolCTS(d, m, n_seeds, device, oracle)
    factories["lucb_pool"] = lambda d, m, n_seeds, device, oracle: LUCBPool(d, m, n_seeds, device, oracle)
    factories["oracle_budget_cts"] = lambda d, m, n_seeds, device, oracle: OracleBudgetCTS(d, m, n_seeds, device, oracle)
    factories["multi_oracle_pool"] = lambda d, m, n_seeds, device, oracle: MultiOraclePool(d, m, n_seeds, device, oracle)
    factories["consistency_robust_cts"] = lambda d, m, n_seeds, device, oracle: ConsistencyRobustCTS(d, m, n_seeds, device, oracle)
    factories["pool_cts_abstain"] = lambda d, m, n_seeds, device, oracle: PoolCTSAbstain(d, m, n_seeds, device, oracle)
    factories["pareto_meta_dual"] = lambda d, m, n_seeds, device, oracle: ParetoMetaDual(d, m, n_seeds, device, oracle)

    # --- Novel algorithms (Loop 3 hybrids — adaptive winners) ---
    factories["adaptive_pool_dual"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolDual(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_abstain"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolAbstain(d, m, n_seeds, device, oracle)
    factories["adaptive_pool_doubling"] = lambda d, m, n_seeds, device, oracle: AdaptivePoolDoubling(d, m, n_seeds, device, oracle)
    factories["adaptive_freq_dual"] = lambda d, m, n_seeds, device, oracle: AdaptiveFreqDual(d, m, n_seeds, device, oracle)

    return factories


# ==========================================================================
# Analysis
# ==========================================================================

def analyze_results(results: list[dict]) -> str:
    """Compute aggregate statistics and rankings."""
    import pandas as pd

    df = pd.DataFrame(results)

    lines = []
    lines.append("=" * 80)
    lines.append("RANDOMIZED ARENA RESULTS")
    lines.append("=" * 80)
    lines.append(f"Total configs: {df['config_id'].nunique()}")
    lines.append(f"Total agent-config pairs: {len(df)}")
    lines.append("")

    # Overall ranking by mean regret
    overall = df.groupby("agent")["regret_mean"].agg(["mean", "median", "std"]).round(1)
    overall = overall.sort_values("mean")
    lines.append("--- OVERALL RANKING (lower = better) ---")
    lines.append(overall.to_string())
    lines.append("")

    # Ranking by corruption type
    for ct in df["corruption_type"].unique():
        sub = df[df["corruption_type"] == ct]
        ranking = sub.groupby("agent")["regret_mean"].agg(["mean", "median"]).round(1)
        ranking = ranking.sort_values("mean")
        lines.append(f"--- {ct.upper()} (n={sub['config_id'].nunique()} configs) ---")
        lines.append(ranking.to_string())
        lines.append("")

    # Win rate: how often each agent is #1 on a config
    wins = {}
    for cid, group in df.groupby("config_id"):
        best = group.loc[group["regret_mean"].idxmin(), "agent"]
        wins[best] = wins.get(best, 0) + 1
    total_configs = df["config_id"].nunique()
    lines.append("--- WIN RATE (fraction of configs where agent has lowest regret) ---")
    for agent, count in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {agent:25s}: {count:3d}/{total_configs} ({100*count/total_configs:.1f}%)")
    lines.append("")

    # Normalized regret: for each config, divide by CUCB's regret
    if "cucb" in df["agent"].values:
        cucb_regret = df[df["agent"] == "cucb"].set_index("config_id")["regret_mean"]
        df_norm = df.copy()
        df_norm["norm_regret"] = df_norm.apply(
            lambda r: r["regret_mean"] / max(cucb_regret.get(r["config_id"], 1), 1), axis=1)
        norm_ranking = df_norm.groupby("agent")["norm_regret"].agg(["mean", "median"]).round(3)
        norm_ranking = norm_ranking.sort_values("mean")
        lines.append("--- NORMALIZED REGRET (relative to CUCB, lower = better) ---")
        lines.append(norm_ranking.to_string())
        lines.append("")

    # Worst-case analysis: max regret across configs
    worst = df.groupby("agent")["regret_mean"].max().sort_values()
    lines.append("--- WORST CASE (max regret across all configs) ---")
    lines.append(worst.to_string())
    lines.append("")

    # Speed
    speed = df.groupby("agent")["elapsed_sec"].mean().sort_values()
    lines.append("--- AVERAGE RUNTIME PER CONFIG (seconds) ---")
    lines.append(speed.round(2).to_string())

    return "\n".join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Randomized Test Arena")
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Arena: {args.n_configs} configs × {args.n_seeds} seeds × T={args.T} on {device}")

    configs = generate_random_configs(args.n_configs, args.master_seed)
    factories = build_agent_factories()
    n_agents = len(factories)

    logger.info(f"Testing {n_agents} algorithms: {list(factories.keys())}")

    all_results = []
    total_start = time.time()

    for i, config in enumerate(configs):
        cfg_desc = (f"d={config['env']['d']} m={config['env']['m']} "
                    f"{config['env']['gap_type']} {config['oracle']['corruption_type']} "
                    f"eps={config['oracle']['epsilon']:.2f}")
        logger.info(f"Config {i+1}/{args.n_configs}: {cfg_desc}")

        results = run_single_config(config, factories, args.T, args.n_seeds, device)
        all_results.extend(results)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start
            eta = elapsed / (i + 1) * (args.n_configs - i - 1)
            logger.info(f"  Progress: {i+1}/{args.n_configs} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    total_elapsed = time.time() - total_start
    logger.info(f"Arena complete in {total_elapsed:.1f}s")

    # Analyze
    report = analyze_results(all_results)
    print("\n" + report)

    # Save
    output_dir = Path(__file__).parent.parent / "arena_results"
    output_dir.mkdir(exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"arena_{ts}.json"
    report_path = output_dir / f"arena_{ts}_report.txt"

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Report saved to {report_path}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
