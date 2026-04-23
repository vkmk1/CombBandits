#!/usr/bin/env python3
"""Production real-LLM experiment: parallel multi-seed, multi-config runner.

Designed for ICML/NeurIPS-quality experiments with:
- T=2000, 20 configs, 20 seeds (matches EVOLvE/Harris gold standard)
- 10 algorithms: 2 baselines + 8 oracle-guided
- Per-round regret curves with 95% CI
- Parallel workers (16+) for speed
- Haiku 4.5 on Bedrock

Usage:
    python cluster/production_real_llm.py --workers 16 --T 2000 --n-configs 20 --n-seeds 20
    python cluster/production_real_llm.py --smoke  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Config generation (deterministic from master_seed)
# ═══════════════════════════════════════════════════════════════════════

def generate_configs(n_configs: int, master_seed: int = 2024) -> list[dict]:
    rng = np.random.RandomState(master_seed)
    configs = []

    d_choices = [20, 30, 50]
    m_choices_by_d = {20: [3, 5], 30: [5], 50: [5, 8]}
    gap_types = ["uniform", "hard"]
    delta_mins = [0.05, 0.1, 0.15, 0.2]

    for i in range(n_configs):
        d = int(rng.choice(d_choices))
        m = int(rng.choice(m_choices_by_d[d]))
        gap_type = str(rng.choice(gap_types))
        delta_min = float(rng.choice(delta_mins))

        configs.append({
            "config_id": i,
            "d": d, "m": m,
            "gap_type": gap_type,
            "delta_min": delta_min,
            "env_seed": int(rng.randint(0, 100000)),
        })

    return configs


# ═══════════════════════════════════════════════════════════════════════
# Single (config, seed) run — this is the unit of parallelism
# ═══════════════════════════════════════════════════════════════════════

def run_single_trial(args: dict) -> list[dict]:
    """Run all algorithms on one (config, seed) pair. Returns list of results."""
    import boto3

    config = args["config"]
    seed = args["seed"]
    T = args["T"]
    model_id = args["model_id"]
    region = args["region"]

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from arena_real_llm import RealLLMBatchedOracle, WarmPoolCTS
    from combbandits.gpu.batched_agents import BatchedCUCB, BatchedCTS, BatchedWarmStartCTS
    from combbandits.gpu.batched_variants import BatchedPoolCTS, BatchedPoolCTSInitCheck

    d, m = config["d"], config["m"]
    device = torch.device("cpu")
    n_seeds = 1

    # Build environment (deterministic from config seed + trial seed)
    combined_seed = config["env_seed"] * 10000 + seed
    rng = np.random.RandomState(combined_seed)

    if config["gap_type"] == "uniform":
        means_np = rng.uniform(0.1, 0.5, size=d)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + config["delta_min"]
    elif config["gap_type"] == "hard":
        means_np = np.full(d, 0.5)
        top_arms = rng.choice(d, size=m, replace=False)
        means_np[top_arms] = 0.5 + config["delta_min"]

    arm_means = torch.tensor(means_np, dtype=torch.float32, device=device)
    optimal_set = torch.argsort(arm_means, descending=True)[:m]
    optimal_reward = arm_means[optimal_set].sum().item()

    # Reward RNG (seeded per trial)
    reward_rng = torch.Generator(device=device)
    reward_rng.manual_seed(combined_seed + 999999)

    # Algorithm definitions — trimmed to publishable set
    # Baselines: CUCB (no oracle), CTS (no oracle)
    # Oracle-guided: warm_start_cts, pool_cts, pool_cts_ic, warm_pool_cts
    # Novel V5: adaptive_pool_abstain (best robust algo)
    # Novel V6: kwik_query_cts (best endogenous-aware), eoe_adaptive
    algo_defs = [
        ("cucb", False, lambda o: BatchedCUCB(d, m, n_seeds, device)),
        ("cts", False, lambda o: BatchedCTS(d, m, n_seeds, device)),
        ("warm_start_cts", True, lambda o: BatchedWarmStartCTS(d, m, n_seeds, device, o)),
        ("pool_cts", True, lambda o: BatchedPoolCTS(d, m, n_seeds, device, o)),
        ("pool_cts_ic", True, lambda o: BatchedPoolCTSInitCheck(
            d, m, n_seeds, device, o,
            T_init=max(d // m * 5, 50), agreement_threshold=0.3)),
        ("warm_pool_cts", True, lambda o: WarmPoolCTS(d, m, n_seeds, device, o)),
    ]

    try:
        from arena_novel_algos_v5 import AdaptivePoolAbstain
        algo_defs.append(("adaptive_pool_abstain", True, lambda o: AdaptivePoolAbstain(d, m, n_seeds, device, o)))
    except ImportError:
        pass

    try:
        from arena_novel_algos_v6 import KWIKQueryCTS, EOEAdaptive
        algo_defs.append(("kwik_query_cts", True, lambda o: KWIKQueryCTS(d, m, n_seeds, device, o)))
        algo_defs.append(("eoe_adaptive", True, lambda o: EOEAdaptive(d, m, n_seeds, device, o)))
    except ImportError:
        pass

    results = []

    for algo_name, needs_oracle, factory in algo_defs:
        if needs_oracle:
            oracle = RealLLMBatchedOracle(
                d=d, m=m, K=1,
                model_id=model_id,
                region=region,
                arm_means=arm_means,
                optimal_set=optimal_set,
            )
        else:
            oracle = None

        try:
            agent = factory(oracle)
        except Exception as e:
            logger.warning(f"  config={config['config_id']} seed={seed} {algo_name}: SKIP ({e})")
            continue

        cum_regret = 0.0
        regret_curve = []
        t_start = time.time()

        for t in range(T):
            selected = agent.select_arms()
            selected_means = arm_means[selected]
            rewards = torch.bernoulli(selected_means, generator=reward_rng)
            inst_regret = optimal_reward - selected_means.sum(dim=1).item()
            cum_regret += inst_regret
            agent.update(selected, rewards)

            if (t + 1) % 100 == 0:
                regret_curve.append(round(cum_regret, 2))

        elapsed = time.time() - t_start
        oracle_diag = oracle.get_diagnostics() if oracle else {}

        results.append({
            "config_id": config["config_id"],
            "seed": seed,
            "agent": algo_name,
            "d": d, "m": m,
            "gap_type": config["gap_type"],
            "delta_min": config["delta_min"],
            "T": T,
            "final_regret": round(cum_regret, 2),
            "regret_curve": regret_curve,
            "elapsed_sec": round(elapsed, 2),
            "oracle_queries": oracle_diag.get("total_queries", 0),
            "oracle_tokens": oracle_diag.get("total_tokens", 0),
            "oracle_mean_overlap": round(oracle_diag.get("mean_overlap", 0), 2),
            "oracle_perfect_rate": round(oracle_diag.get("overlap_rate", 0), 3),
        })

    cid = config["config_id"]
    logger.info(f"  config={cid} seed={seed} done ({len(results)} algos, {elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis and reporting
# ═══════════════════════════════════════════════════════════════════════

def analyze(all_results: list[dict], output_dir: Path):
    import pandas as pd

    df = pd.DataFrame(all_results)
    n_configs = df["config_id"].nunique()
    n_seeds = df["seed"].nunique()
    n_agents = df["agent"].nunique()

    lines = []
    lines.append("=" * 80)
    lines.append("PRODUCTION REAL-LLM EXPERIMENT RESULTS")
    lines.append("=" * 80)
    lines.append(f"Configs: {n_configs} | Seeds: {n_seeds} | Agents: {n_agents} | T: {df['T'].iloc[0]}")
    lines.append(f"Model: {all_results[0].get('model_id', 'unknown')}")
    lines.append("")

    # Per-agent stats aggregated across configs and seeds
    stats = df.groupby("agent")["final_regret"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["stderr"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["stderr"]
    stats = stats.sort_values("mean").round(1)

    lines.append("--- OVERALL RANKING (mean ± stderr, 95% CI) ---")
    for agent, row in stats.iterrows():
        lines.append(f"  {agent:28s}  {row['mean']:8.1f} ± {row['stderr']:5.1f}  "
                     f"[{row['ci95_lo']:7.1f}, {row['ci95_hi']:7.1f}]  "
                     f"median={row['median']:.1f}")
    lines.append("")

    # Per gap_type breakdown
    for gt in sorted(df["gap_type"].unique()):
        sub = df[df["gap_type"] == gt]
        ranking = sub.groupby("agent")["final_regret"].agg(["mean", "std", "count"])
        ranking["stderr"] = ranking["std"] / np.sqrt(ranking["count"])
        ranking = ranking.sort_values("mean").round(1)
        lines.append(f"--- {gt.upper()} (n={sub['config_id'].nunique()} configs) ---")
        for agent, row in ranking.iterrows():
            lines.append(f"  {agent:28s}  {row['mean']:8.1f} ± {row['stderr']:5.1f}")
        lines.append("")

    # Win rate
    wins = {}
    for (cid, seed), group in df.groupby(["config_id", "seed"]):
        best = group.loc[group["final_regret"].idxmin(), "agent"]
        wins[best] = wins.get(best, 0) + 1
    total_trials = n_configs * n_seeds
    lines.append("--- WIN RATE ---")
    for agent, count in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {agent:28s}: {count:4d}/{total_trials} ({100*count/total_trials:.1f}%)")
    lines.append("")

    # Oracle efficiency
    oracle_stats = df[df["oracle_queries"] > 0].groupby("agent").agg({
        "oracle_queries": "mean",
        "oracle_tokens": "mean",
        "oracle_mean_overlap": "mean",
        "oracle_perfect_rate": "mean",
    }).round(2)
    lines.append("--- ORACLE EFFICIENCY ---")
    lines.append(f"  {'agent':28s} {'queries':>8s} {'tokens':>8s} {'overlap':>8s} {'perfect':>8s}")
    for agent, row in oracle_stats.iterrows():
        lines.append(f"  {agent:28s} {row['oracle_queries']:8.0f} {row['oracle_tokens']:8.0f} "
                     f"{row['oracle_mean_overlap']:8.2f} {row['oracle_perfect_rate']:8.3f}")
    lines.append("")

    # Worst case
    worst = df.groupby("agent")["final_regret"].max().sort_values()
    lines.append("--- WORST CASE ---")
    for agent, val in worst.items():
        lines.append(f"  {agent:28s}  {val:.1f}")

    report = "\n".join(lines)
    print("\n" + report)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"production_real_llm_{ts}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save raw results
    results_path = output_dir / f"production_real_llm_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f)

    # Save regret curves as separate file (for plotting)
    curves = {}
    for r in all_results:
        key = r["agent"]
        if key not in curves:
            curves[key] = []
        curves[key].append(r["regret_curve"])
    curves_path = output_dir / f"production_real_llm_{ts}_curves.json"
    with open(curves_path, "w") as f:
        json.dump(curves, f)

    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Curves: {curves_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Production Real-LLM Experiment")
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", type=str, default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--master-seed", type=int, default=2024)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test: 2 configs, 2 seeds, T=200")
    args = parser.parse_args()

    if args.smoke:
        args.T = 200
        args.n_configs = 2
        args.n_seeds = 2
        args.workers = 2

    configs = generate_configs(args.n_configs, args.master_seed)

    # Build all (config, seed) pairs
    tasks = []
    for config in configs:
        for seed in range(args.n_seeds):
            tasks.append({
                "config": config,
                "seed": seed,
                "T": args.T,
                "model_id": args.model,
                "region": args.region,
            })

    total_tasks = len(tasks)
    logger.info(f"Production experiment: {args.n_configs} configs × {args.n_seeds} seeds = "
                f"{total_tasks} trials, T={args.T}, workers={args.workers}")
    logger.info(f"Model: {args.model}")

    # Estimate
    est_calls_per_trial = 250  # ~8 oracle algos × ~30 calls each
    est_total_calls = total_tasks * est_calls_per_trial
    est_tokens = est_total_calls * 550
    est_cost = est_tokens / 1e6 * 3  # ~$1 input + $5 output averaged to ~$3/M
    est_time_serial = est_total_calls  # 1s per call
    est_time_parallel = est_time_serial / args.workers
    logger.info(f"Estimate: ~{est_total_calls:,} LLM calls, ~{est_tokens/1e6:.0f}M tokens, "
                f"~${est_cost:.0f}, ~{est_time_parallel/3600:.1f}h with {args.workers} workers")

    all_results = []
    t_start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                task = futures[future]
                logger.error(f"  FAILED config={task['config']['config_id']} "
                             f"seed={task['seed']}: {e}")

            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{total_tasks} "
                            f"({100*completed/total_tasks:.0f}%) "
                            f"ETA: {eta/60:.0f}min")

    total_elapsed = time.time() - t_start
    logger.info(f"Done: {total_tasks} trials in {total_elapsed/60:.1f}min")

    # Add model_id to results for report
    for r in all_results:
        r["model_id"] = args.model

    output_dir = Path(__file__).parent.parent / "arena_results"
    analyze(all_results, output_dir)


if __name__ == "__main__":
    main()
