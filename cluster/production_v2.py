#!/usr/bin/env python3
"""Production V2: CTS-warmup vs round-robin warmup comparison.

The experiment that proves the paper's thesis: with endogenous oracle quality,
warmup strategy determines algorithm performance.

Compares:
- Baselines: CUCB, CTS (no oracle)
- Round-robin warmup: pool_cts, warm_pool_cts, adaptive_pool_abstain (old)
- CTS warmup: cts_warm_pool, cts_warm_adaptive_abstain, cts_warm_kwik, cts_warm_eoe (new)
- Warm-start CTS (queries at t=0, no warmup — worst case)

All configs use d=30, m=5 as requested.

Usage:
    python3 cluster/production_v2.py --workers 8 --T 2000 --n-configs 20 --n-seeds 20
    python3 cluster/production_v2.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import torch

S3_BUCKET = "combbandits-results-099841456154"
S3_KEY = "live/results.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_configs(n_configs: int, master_seed: int = 2024) -> list[dict]:
    rng = np.random.RandomState(master_seed)
    configs = []

    gap_types = ["uniform", "hard"]
    delta_mins = [0.05, 0.1, 0.15, 0.2]

    for i in range(n_configs):
        gap_type = str(rng.choice(gap_types))
        delta_min = float(rng.choice(delta_mins))

        configs.append({
            "config_id": i,
            "d": 30, "m": 5,
            "gap_type": gap_type,
            "delta_min": delta_min,
            "env_seed": int(rng.randint(0, 100000)),
        })

    return configs


def run_single_trial(args: dict) -> list[dict]:
    """Run all algorithms on one (config, seed) pair."""
    config = args["config"]
    seed = args["seed"]
    T = args["T"]
    model_id = args["model_id"]
    region = args["region"]

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from arena_real_llm import RealLLMBatchedOracle, WarmPoolCTS
    from arena_real_llm_v2 import (
        RealLLMBatchedOracle as V2Oracle,
        CTSWarmPoolCTS, CTSWarmAdaptiveAbstain,
        CTSWarmKWIKQuery, CTSWarmEOE,
    )
    from combbandits.gpu.batched_agents import BatchedCUCB, BatchedCTS, BatchedWarmStartCTS
    from combbandits.gpu.batched_variants import BatchedPoolCTS

    d, m = config["d"], config["m"]
    device = torch.device("cpu")
    n_seeds = 1

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

    reward_rng = torch.Generator(device=device)
    reward_rng.manual_seed(combined_seed + 999999)

    def make_oracle():
        return V2Oracle(
            d=d, m=m, K=1,
            model_id=model_id, region=region,
            arm_means=arm_means, optimal_set=optimal_set,
        )

    try:
        from arena_novel_algos_v5 import AdaptivePoolAbstain
        has_v5 = True
    except ImportError:
        has_v5 = False

    try:
        from arena_novel_algos_v6 import KWIKQueryCTS, EOEAdaptive
        has_v6 = True
    except ImportError:
        has_v6 = False

    algo_defs = [
        # Baselines (no oracle)
        ("cucb", False, lambda: BatchedCUCB(d, m, n_seeds, device)),
        ("cts", False, lambda: BatchedCTS(d, m, n_seeds, device)),
        # Round-robin warmup (old)
        ("warm_start_cts", True, lambda: BatchedWarmStartCTS(d, m, n_seeds, device, make_oracle())),
        ("pool_cts_rr", True, lambda: BatchedPoolCTS(d, m, n_seeds, device, make_oracle())),
        ("warm_pool_rr", True, lambda: WarmPoolCTS(d, m, n_seeds, device, make_oracle())),
        # CTS warmup (new)
        ("cts_warm_pool", True, lambda: CTSWarmPoolCTS(d, m, n_seeds, device, make_oracle())),
        ("cts_warm_adaptive", True, lambda: CTSWarmAdaptiveAbstain(d, m, n_seeds, device, make_oracle())),
        ("cts_warm_kwik", True, lambda: CTSWarmKWIKQuery(d, m, n_seeds, device, make_oracle())),
        ("cts_warm_eoe", True, lambda: CTSWarmEOE(d, m, n_seeds, device, make_oracle())),
    ]

    if has_v5:
        algo_defs.append(("adaptive_abstain_rr", True,
                         lambda: AdaptivePoolAbstain(d, m, n_seeds, device, make_oracle())))
    if has_v6:
        algo_defs.append(("kwik_query_rr", True,
                         lambda: KWIKQueryCTS(d, m, n_seeds, device, make_oracle())))
        algo_defs.append(("eoe_adaptive_rr", True,
                         lambda: EOEAdaptive(d, m, n_seeds, device, make_oracle())))

    results = []

    for algo_name, needs_oracle, factory in algo_defs:
        try:
            agent = factory()
        except Exception as e:
            logger.warning(f"  config={config['config_id']} seed={seed} {algo_name}: SKIP ({e})")
            continue

        # Get oracle ref for diagnostics
        oracle_ref = None
        if needs_oracle:
            for attr in ['oracle', '_oracle']:
                if hasattr(agent, attr):
                    oracle_ref = getattr(agent, attr)
                    break

        cum_regret = 0.0
        regret_curve = []

        # Reset reward RNG for fair comparison
        reward_rng.manual_seed(combined_seed + 999999)
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
        oracle_diag = oracle_ref.get_diagnostics() if oracle_ref else {}

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


def analyze(all_results: list[dict], output_dir: Path):
    import pandas as pd

    df = pd.DataFrame(all_results)
    n_configs = df["config_id"].nunique()
    n_seeds = df["seed"].nunique()
    n_agents = df["agent"].nunique()

    lines = []
    lines.append("=" * 80)
    lines.append("PRODUCTION V2: CTS-WARMUP vs ROUND-ROBIN COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Configs: {n_configs} | Seeds: {n_seeds} | Agents: {n_agents} | T: {df['T'].iloc[0]}")
    lines.append(f"All configs: d=30, m=5")
    lines.append(f"Model: {all_results[0].get('model_id', 'unknown')}")
    lines.append("")

    stats = df.groupby("agent")["final_regret"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["stderr"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["stderr"]
    stats = stats.sort_values("mean").round(1)

    lines.append("--- OVERALL RANKING (mean ± stderr, 95% CI) ---")
    for agent, row in stats.iterrows():
        tag = ""
        if "cts_warm" in agent:
            tag = " [CTS-WARM]"
        elif "_rr" in agent or agent in ("warm_start_cts",):
            tag = " [RR-WARM]"
        elif agent in ("cucb", "cts"):
            tag = " [BASELINE]"
        lines.append(f"  {agent:30s}  {row['mean']:8.1f} ± {row['stderr']:5.1f}  "
                     f"[{row['ci95_lo']:7.1f}, {row['ci95_hi']:7.1f}]  "
                     f"median={row['median']:.1f}{tag}")
    lines.append("")

    for gt in sorted(df["gap_type"].unique()):
        sub = df[df["gap_type"] == gt]
        ranking = sub.groupby("agent")["final_regret"].agg(["mean", "std", "count"])
        ranking["stderr"] = ranking["std"] / np.sqrt(ranking["count"])
        ranking = ranking.sort_values("mean").round(1)
        lines.append(f"--- {gt.upper()} (n={sub['config_id'].nunique()} configs) ---")
        for agent, row in ranking.iterrows():
            lines.append(f"  {agent:30s}  {row['mean']:8.1f} ± {row['stderr']:5.1f}")
        lines.append("")

    # Win rate
    wins = {}
    for (cid, seed), group in df.groupby(["config_id", "seed"]):
        best = group.loc[group["final_regret"].idxmin(), "agent"]
        wins[best] = wins.get(best, 0) + 1
    total_trials = n_configs * n_seeds
    lines.append("--- WIN RATE ---")
    for agent, count in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {agent:30s}: {count:4d}/{total_trials} ({100*count/total_trials:.1f}%)")
    lines.append("")

    # Oracle efficiency
    oracle_stats = df[df["oracle_queries"] > 0].groupby("agent").agg({
        "oracle_queries": "mean",
        "oracle_tokens": "mean",
        "oracle_mean_overlap": "mean",
        "oracle_perfect_rate": "mean",
    }).round(2)
    lines.append("--- ORACLE EFFICIENCY ---")
    lines.append(f"  {'agent':30s} {'queries':>8s} {'tokens':>8s} {'overlap':>8s} {'perfect':>8s}")
    for agent, row in oracle_stats.sort_values("oracle_mean_overlap", ascending=False).iterrows():
        tag = " *" if "cts_warm" in agent else ""
        lines.append(f"  {agent:30s} {row['oracle_queries']:8.0f} {row['oracle_tokens']:8.0f} "
                     f"{row['oracle_mean_overlap']:8.2f} {row['oracle_perfect_rate']:8.3f}{tag}")
    lines.append("")

    # Worst case
    worst = df.groupby("agent")["final_regret"].max().sort_values()
    lines.append("--- WORST CASE ---")
    for agent, val in worst.items():
        lines.append(f"  {agent:30s}  {val:.1f}")

    # CTS-warmup vs round-robin head-to-head
    lines.append("")
    lines.append("--- HEAD-TO-HEAD: CTS-WARMUP vs ROUND-ROBIN ---")
    pairs = [
        ("cts_warm_pool", "warm_pool_rr", "Pool-CTS"),
        ("cts_warm_pool", "pool_cts_rr", "Pool-CTS (no warmup)"),
        ("cts_warm_adaptive", "adaptive_abstain_rr", "Adaptive+Abstain"),
        ("cts_warm_kwik", "kwik_query_rr", "KWIK Query"),
        ("cts_warm_eoe", "eoe_adaptive_rr", "EOE Adaptive"),
    ]
    for new_name, old_name, label in pairs:
        if new_name in stats.index and old_name in stats.index:
            new_val = stats.loc[new_name, "mean"]
            old_val = stats.loc[old_name, "mean"]
            pct = (old_val - new_val) / old_val * 100
            lines.append(f"  {label:25s}  CTS={new_val:.1f}  RR={old_val:.1f}  "
                        f"improvement={pct:.1f}%")

    report = "\n".join(lines)
    print("\n" + report)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"production_v2_{ts}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    results_path = output_dir / f"production_v2_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f)

    curves = {}
    for r in all_results:
        key = r["agent"]
        if key not in curves:
            curves[key] = []
        curves[key].append(r["regret_curve"])
    curves_path = output_dir / f"production_v2_{ts}_curves.json"
    with open(curves_path, "w") as f:
        json.dump(curves, f)

    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Curves: {curves_path}")


def main():
    parser = argparse.ArgumentParser(description="Production V2: CTS-Warmup Experiment")
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=str, default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--master-seed", type=int, default=2024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.T = 200
        args.n_configs = 2
        args.n_seeds = 2
        args.workers = 2

    configs = generate_configs(args.n_configs, args.master_seed)

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
    logger.info(f"Production V2: {args.n_configs} configs × {args.n_seeds} seeds = "
                f"{total_tasks} trials, T={args.T}, workers={args.workers}")
    logger.info(f"All configs: d=30, m=5 | Model: {args.model}")

    # ~13 algos × ~30 oracle calls each × T/100 re-queries
    est_calls_per_trial = 350
    est_total_calls = total_tasks * est_calls_per_trial
    est_tokens = est_total_calls * 550
    est_cost = est_tokens / 1e6 * 3
    logger.info(f"Estimate: ~{est_total_calls:,} LLM calls, ~${est_cost:.0f}")

    all_results = []
    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed = 0
    s3_client = boto3.client("s3", region_name="us-east-1")
    _s3_lock = threading.Lock()

    def push_live(results_so_far, n_completed):
        payload = {
            "experiment": {
                "total_trials": total_tasks,
                "completed": n_completed,
                "T": args.T,
                "n_configs": args.n_configs,
                "n_seeds": args.n_seeds,
                "d": 30, "m": 5,
                "model": "claude-haiku-4.5",
                "start_time": start_iso,
                "est_cost_usd": 77,
            },
            "results": results_so_far,
        }
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET, Key=S3_KEY,
                Body=json.dumps(payload),
                ContentType="application/json",
            )
        except Exception as e:
            logger.warning(f"S3 push failed: {e}")

    push_live([], 0)

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

            if completed % 2 == 0 or completed == total_tasks:
                threading.Thread(
                    target=push_live,
                    args=(list(all_results), completed),
                    daemon=True,
                ).start()

            if completed % 10 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{total_tasks} "
                            f"({100*completed/total_tasks:.0f}%) "
                            f"ETA: {eta/60:.0f}min")

    total_elapsed = time.time() - t_start
    logger.info(f"Done: {total_tasks} trials in {total_elapsed/60:.1f}min")

    for r in all_results:
        r["model_id"] = args.model

    output_dir = Path(__file__).parent.parent / "arena_results"
    analyze(all_results, output_dir)


if __name__ == "__main__":
    main()
