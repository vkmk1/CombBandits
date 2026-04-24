"""Tier 6: Screen 7 long-horizon variants against N1 baseline.

Tests V1-V7 plus CTS and N1 (original CorrCTS-Full) baselines.
Uses the same 16 configs as Tier 5.

Phase 1 (screening): T=25000, n_seeds=10, all configs
Phase 2 (full): T=25000, n_seeds=20, all configs (top-3 winners only)

The variants:
  V1_decay_kernel       : 1 LLM call + decay schedule
  V2_requery_logspaced  : 3 LLM calls at {30, 300, 3000}
  V3_blend_llm_data     : 4 LLM calls + data-kernel blending
  V4_refine_topk        : 2 LLM calls (full + top-K refinement at t=5000)
  V5_ensemble_kernels   : 3 LLM calls at t=30 (diverse partitions)
  V6_edge_pruning       : 1 LLM call + CLUB-style edge pruning
  V7_per_arm_damping    : 1 LLM call + per-arm correlation damping
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS, CTSBase
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from longhorizon_variants import LONGHORIZON_ALGOS
from oracle_instrumented import InstrumentedOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-5.4"

TIER6_ALGOS = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    **LONGHORIZON_ALGOS,
}
NEEDS_ORACLE = {
    "N1_corr_full", "V1_decay_kernel", "V2_requery_logspaced",
    "V3_blend_llm_data", "V4_refine_topk", "V5_ensemble_kernels",
    "V6_edge_pruning", "V7_per_arm_damping",
}


def generate_configs():
    configs = []
    cid = 0
    combos = [
        ("uniform", 0.20, 100),
        ("uniform", 0.10, 200),
        ("hard", 0.05, 300),
        ("staggered", 0.02, 400),
    ]
    for d in [25, 50]:
        for m in [3, 5]:
            for gap_type, delta, seed_offset in combos:
                configs.append({
                    "config_id": cid,
                    "d": d, "m": m,
                    "gap_type": gap_type,
                    "delta_min": delta,
                    "env_seed": seed_offset + cid,
                })
                cid += 1
    return configs


def build_env(config, seed):
    d, m = config["d"], config["m"]
    rng = np.random.RandomState(config["env_seed"] * 1000 + seed)
    gap_type = config["gap_type"]
    delta = config["delta_min"]

    if gap_type == "uniform":
        means = np.full(d, 0.7 - delta)
        top_idx = rng.choice(d, m, replace=False)
        means[top_idx] = 0.7
    elif gap_type == "hard":
        means = np.full(d, 0.7 - 0.2)
        top_idx = rng.choice(d, m, replace=False)
        means[top_idx] = 0.7
        remaining = [i for i in range(d) if i not in top_idx]
        near_idx = rng.choice(remaining, min(m, len(remaining)), replace=False)
        means[near_idx] = 0.7 - delta
    elif gap_type == "staggered":
        means = np.linspace(0.7, 0.3, d)
        rng.shuffle(means)
    else:
        raise ValueError(f"Unknown gap type: {gap_type}")

    optimal = np.argsort(means)[::-1][:m]
    optimal_reward = means[optimal].sum()
    return means, optimal, optimal_reward


def run_single_trial(algo_name, config, seed, T, out_dir):
    trial_id = f"c{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER6_ALGOS[algo_name]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": trial_np_seed}

    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=algo_name,
        )
        kwargs["oracle"] = oracle

    agent = AlgoClass(**kwargs)

    cum_regret = 0.0
    regret_curve = []
    t_start = time.time()

    for t in range(T):
        if oracle is not None:
            oracle.current_t = t
        selected = list(agent.select_arms())
        if len(selected) < config["m"]:
            used = set(selected)
            remaining = [a for a in range(config["d"]) if a not in used]
            selected.extend(remaining[:config["m"] - len(selected)])
        selected = selected[:config["m"]]
        selected_means = means[selected]
        rewards = (reward_rng.uniform(size=config["m"]) < selected_means).astype(float)
        cum_regret += optimal_reward - selected_means.sum()
        agent.update(selected, rewards.tolist())
        if (t + 1) % 50 == 0:
            regret_curve.append(round(float(cum_regret), 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}
    return {
        "trial_id": trial_id, "algo": algo_name, "model": MODEL,
        "config_id": config["config_id"], "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"], "delta_min": config["delta_min"],
        "env_seed": config["env_seed"],
        "T": T, "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve, "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
    }


def analyze(all_results, out_dir):
    import pandas as pd
    from scipy.stats import wilcoxon

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 100)
    lines.append(f"TIER 6 VARIANTS — {len(valid)} trials | model={MODEL}")
    lines.append("=" * 100)
    lines.append(f"Algos: {df['algo'].nunique()} | Configs: {df['config_id'].nunique()} | "
                 f"Seeds: {df['seed'].nunique()} | T: {df['T'].iloc[0]}")
    lines.append("")
    lines.append("--- GLOBAL RANKING ---")
    agg = df.groupby("algo")["final_regret"].agg(["mean", "std", "count"]).reset_index()
    agg["stderr"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.sort_values("mean")
    cts_mean = agg[agg["algo"] == "CTS"]["mean"].iloc[0] if "CTS" in agg["algo"].values else float("nan")
    n1_mean = agg[agg["algo"] == "N1_corr_full"]["mean"].iloc[0] if "N1_corr_full" in agg["algo"].values else float("nan")
    for _, row in agg.iterrows():
        vs_cts = f"+{100*(cts_mean - row['mean'])/cts_mean:.1f}%" if cts_mean == cts_mean else ""
        vs_n1 = f"{100*(n1_mean - row['mean'])/n1_mean:+.1f}%" if n1_mean == n1_mean else ""
        lines.append(f"  {row['algo']:30s} mean={row['mean']:7.1f}  se={row['stderr']:6.2f}  "
                     f"vs_CTS={vs_cts:>8s}  vs_N1={vs_n1:>8s}  n={int(row['count'])}")
    lines.append("")

    # Paired tests vs CTS and vs N1
    for baseline in ["CTS", "N1_corr_full"]:
        lines.append(f"--- PAIRED TESTS vs {baseline} ---")
        pivot = df.pivot_table(index=["config_id", "seed"], columns="algo", values="final_regret")
        if baseline not in pivot.columns:
            continue
        base_vals = pivot[baseline]
        for algo in pivot.columns:
            if algo == baseline:
                continue
            pair = pivot[[baseline, algo]].dropna()
            if len(pair) < 5:
                continue
            diffs = pair[baseline] - pair[algo]
            wins = int((diffs > 0).sum())
            losses = int((diffs < 0).sum())
            try:
                _, p = wilcoxon(pair[baseline], pair[algo]) if len(pair) > 0 else (0, 1.0)
            except Exception:
                p = 1.0
            lines.append(f"  {algo:30s}  W/L={wins}/{losses}  mean_adv_over_{baseline}={diffs.mean():+7.2f}  wilcoxon_p={p:.4f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=25000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--algos", type=str, default="",
                        help="Comma-separated list of algos to run (default: all)")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated list of config_ids (default: all)")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier6_variants_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    log_file = out_dir / "llm_calls.jsonl"
    InstrumentedOracle.set_log_file(log_file)

    configs = generate_configs()
    if args.configs:
        keep = set(int(x) for x in args.configs.split(","))
        configs = [c for c in configs if c["config_id"] in keep]

    if args.algos:
        algos = [a for a in args.algos.split(",") if a.strip()]
    else:
        algos = list(TIER6_ALGOS.keys())

    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    import random
    random.seed(42)
    random.shuffle(tasks)

    logger.info(f"Running {len(algos)} algos x {len(configs)} configs x {args.n_seeds} seeds "
                f"= {len(tasks)} trials (T={args.T}, model={MODEL})")

    all_results = []
    t_start = time.time()
    completed = 0

    raw_path = out_dir / "raw_trials.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, a, c, s, args.T, out_dir): (a, c, s)
                   for a, c, s in tasks}
        with open(raw_path, "w") as f:
            for fut in as_completed(futures):
                a, c, s = futures[fut]
                try:
                    result = fut.result()
                    all_results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 25 == 0:
                        elapsed = time.time() - t_start
                        rate = completed / elapsed
                        eta = (len(tasks) - completed) / rate / 60
                        logger.info(f"[{completed}/{len(tasks)}] {a:30s} c{c['config_id']} s{s} "
                                    f"r={result['final_regret']:.1f} ETA={eta:.1f}min")
                except Exception as e:
                    logger.exception(f"Trial {a} c{c['config_id']} s{s} failed: {e}")

    # Summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": ts, "T": args.T, "n_seeds": args.n_seeds,
            "n_configs": len(configs), "n_algos": len(algos),
            "total_trials": completed, "model": MODEL,
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)

    analyze(all_results, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
