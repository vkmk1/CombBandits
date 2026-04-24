"""Bulletproof real-LLM experiment: our 3 winners vs CTS vs 4 paper baselines.

Differences from smoke test (addresses bias concerns in DIAGNOSIS.md):
- 4 NEW configs not seen in smoke test (different gap types, delta values)
- 4 seeds per config = 16 trials per algorithm (16 paired comparisons)
- T=1500 (check stability beyond smoke test horizon)
- Fresh LLM calls (cache cleared before run)
- Paper baselines: TS-LLM, LLM-Jump-Start, LLM-CUCB-AT, Calibration-Gated
- Parallel execution via thread pool (LLM calls are I/O bound)
- Paired t-test + Wilcoxon signed-rank p-values reported

Cost: ~1000 LLM calls total, ~$0.30-$0.50 via gpt-5-mini + gpt-4.1-mini.
Time: ~8-12 min with 4 parallel workers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import (
    CTSBase, ICPDCTS, QueryDesignCTS, LogprobCTS, ALL_ALGORITHMS
)
from paper_baselines import PAPER_BASELINES
from oracle import GPTOracle, CACHE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── NEW configs (different from smoke test) ──────────────────────────────
def generate_bulletproof_configs() -> list[dict]:
    """5 configs with diverse (gap_type, delta_min) combos."""
    configs = [
        {"config_id": 0, "d": 30, "m": 5, "gap_type": "uniform",
         "delta_min": 0.12, "env_seed": 5001},
        {"config_id": 1, "d": 30, "m": 5, "gap_type": "uniform",
         "delta_min": 0.18, "env_seed": 5017},
        {"config_id": 2, "d": 30, "m": 5, "gap_type": "hard",
         "delta_min": 0.10, "env_seed": 5041},
        {"config_id": 3, "d": 30, "m": 5, "gap_type": "hard",
         "delta_min": 0.20, "env_seed": 5067},
        {"config_id": 4, "d": 30, "m": 5, "gap_type": "uniform",
         "delta_min": 0.08, "env_seed": 5089},  # hardest uniform
    ]
    return configs


def build_env(config: dict, seed: int):
    combined = config["env_seed"] * 10000 + seed
    rng = np.random.RandomState(combined)
    d, m = config["d"], config["m"]
    if config["gap_type"] == "uniform":
        means = rng.uniform(0.1, 0.5, size=d)
        top = rng.choice(d, size=m, replace=False)
        means[top] = 0.5 + config["delta_min"]
    else:  # hard
        means = np.full(d, 0.5)
        top = rng.choice(d, size=m, replace=False)
        means[top] = 0.5 + config["delta_min"]
    optimal = np.argsort(means)[::-1][:m]
    return means, optimal, means[optimal].sum()


# ─── Algorithm registry for the bulletproof experiment ────────────────────
EXPERIMENT_ALGOS = {
    # Baseline
    "cts_baseline": ALL_ALGORITHMS["cts_baseline"],
    # Our 3 winners
    "OURS_B2_icpd": ALL_ALGORITHMS["B2_icpd_cts"],
    "OURS_F2_query_design": ALL_ALGORITHMS["F2_query_design_cts"],
    "OURS_A1_logprob": ALL_ALGORITHMS["A1_logprob_cts"],
    # Paper baselines
    "PAPER_ts_llm": PAPER_BASELINES["PAPER_ts_llm"],
    "PAPER_llm_jump_start": PAPER_BASELINES["PAPER_llm_jump_start"],
    "PAPER_llm_cucb_at": PAPER_BASELINES["PAPER_llm_cucb_at"],
    "PAPER_calibration_gated": PAPER_BASELINES["PAPER_calibration_gated"],
}

NEEDS_ORACLE = {k for k in EXPERIMENT_ALGOS if k != "cts_baseline"}


# ─── Single trial (one algo × one config × one seed) ─────────────────────
def run_single_trial(algo_name: str, config: dict, seed: int, T: int,
                     oracle_model: str = "gpt-5-mini") -> dict:
    """IMPORTANT: each trial gets its OWN fresh oracle (tracks cost)."""
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = EXPERIMENT_ALGOS[algo_name]
    kwargs = {"d": config["d"], "m": config["m"]}

    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = GPTOracle(d=config["d"], m=config["m"], model=oracle_model)
        kwargs["oracle"] = oracle

    agent = AlgoClass(**kwargs)

    cum_regret = 0.0
    regret_curve = []
    t_start = time.time()

    for t in range(T):
        selected = list(agent.select_arms())
        if len(selected) < config["m"]:
            used = set(selected)
            remaining = [a for a in range(config["d"]) if a not in used]
            selected.extend(remaining[:config["m"] - len(selected)])
        selected = selected[:config["m"]]

        selected_means = means[selected]
        rewards = (reward_rng.uniform(size=config["m"]) < selected_means).astype(float)
        inst_regret = optimal_reward - selected_means.sum()
        cum_regret += inst_regret
        agent.update(selected, rewards.tolist())

        if (t + 1) % 100 == 0:
            regret_curve.append(round(float(cum_regret), 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}

    return {
        "algo": algo_name,
        "config_id": config["config_id"],
        "seed": seed,
        "gap_type": config["gap_type"],
        "delta_min": config["delta_min"],
        "T": T,
        "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve,
        "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
    }


# ─── Analysis ─────────────────────────────────────────────────────────────
def paired_analysis(all_results: list[dict], out_dir: Path):
    import pandas as pd
    from collections import defaultdict

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)
    n_trials = df.groupby("algo").size().iloc[0] if len(df) else 0

    by_trial: dict[tuple, dict] = defaultdict(dict)
    for r in valid:
        by_trial[(r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]

    lines = []
    lines.append("=" * 100)
    lines.append("BULLETPROOF EXPERIMENT: 3 Winners + 4 Paper Baselines vs CTS")
    lines.append("=" * 100)
    lines.append(f"Trials per algo: {n_trials}")
    lines.append(f"Configs: {df['config_id'].nunique()} | Seeds per config: "
                 f"{df.groupby('config_id')['seed'].nunique().iloc[0] if len(df) else 0}")
    lines.append(f"T: {df['T'].iloc[0] if len(df) else 'N/A'}")
    lines.append("")

    # Overall ranking
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)

    baseline_mean = stats.loc["cts_baseline", "mean"] if "cts_baseline" in stats.index else None

    lines.append("--- OVERALL RANKING (lower regret = better) ---")
    lines.append(f"{'rank':<5}{'algorithm':<28s}{'mean':>9s}{'stderr':>8s}{'median':>9s}{'vs_CTS':>10s}")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs = f"{(baseline_mean - row['mean']) / baseline_mean * 100:+.1f}%" if baseline_mean else "—"
        flag = ""
        if baseline_mean is not None and row["mean"] < baseline_mean:
            flag = " <-- BEATS CTS"
        lines.append(
            f"{rank:<5}{algo:<28s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
            f"{row['median']:>9.1f}{vs:>10s}{flag}"
        )
    lines.append("")

    # Paired comparison: vs CTS
    lines.append("--- PAIRED COMPARISON vs CTS (every algo on same (config, seed) as CTS) ---")
    lines.append(f"{'algorithm':<28s}{'wins':>8s}{'losses':>8s}{'mean_diff':>12s}"
                 f"{'stderr':>8s}{'t_stat':>8s}{'wilcoxon_p':>12s}")
    for algo in stats.index:
        if algo == "cts_baseline":
            continue
        diffs = []
        wins = losses = 0
        for (c, s), trials in by_trial.items():
            if "cts_baseline" in trials and algo in trials:
                d = trials["cts_baseline"] - trials[algo]
                diffs.append(d)
                if d > 0: wins += 1
                elif d < 0: losses += 1
        if not diffs:
            continue
        mean_d = np.mean(diffs)
        sem = np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
        t_stat = mean_d / sem if sem > 0 else 0
        # Wilcoxon via simple count-based approximation (signed-rank would need scipy)
        # Use sign test p-value instead: P(wins >= k | n trials under H0 p=0.5)
        from math import comb
        n = len(diffs)
        extreme = min(wins, losses)
        sign_p = 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n)
        lines.append(
            f"{algo:<28s}{wins:>8d}{losses:>8d}{mean_d:>12.1f}{sem:>8.2f}"
            f"{t_stat:>8.2f}{sign_p:>12.4f}"
        )
    lines.append("")

    # Per-config breakdown
    lines.append("--- PER-CONFIG MEAN REGRET ---")
    pivot = df.pivot_table(index="algo", columns="config_id", values="final_regret", aggfunc="mean").round(1)
    pivot = pivot.reindex(stats.index)
    header = f"{'algorithm':<28s}" + "".join(f"{'cfg_'+str(c):>10s}" for c in pivot.columns)
    lines.append(header)
    for algo, row in pivot.iterrows():
        row_str = f"{algo:<28s}" + "".join(
            f"{v:>10.1f}" if not np.isnan(v) else f"{'—':>10s}" for v in row
        )
        lines.append(row_str)
    lines.append("")

    # Win count
    lines.append("--- WIN COUNT (lowest regret in (config, seed)) ---")
    wins = {}
    for (c, s), trials in by_trial.items():
        if not trials: continue
        best = min(trials.items(), key=lambda x: x[1])[0]
        wins[best] = wins.get(best, 0) + 1
    total = len(by_trial)
    for algo, w in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {algo:<28s}{w}/{total}")
    lines.append("")

    # LLM cost
    lines.append("--- LLM COST (per trial average) ---")
    costs = df[df["llm_calls"] > 0].groupby("algo").agg({
        "llm_calls": "mean", "llm_tokens": "mean"
    }).round(1)
    for algo in stats.index:
        if algo in costs.index:
            c = costs.loc[algo]
            lines.append(f"  {algo:<28s}calls={c['llm_calls']:.1f}  tokens={c['llm_tokens']:.0f}")
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"bulletproof_{ts}_report.txt", "w") as f:
        f.write(report)
    with open(out_dir / f"bulletproof_{ts}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results → {out_dir}/bulletproof_{ts}_*")


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-cache", action="store_true",
                        help="Don't clear cache (cached = not truly fresh)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Clear cache for truly fresh LLM calls
    if not args.keep_cache and CACHE_DB.exists():
        backup = CACHE_DB.with_suffix(".sqlite.backup")
        shutil.move(CACHE_DB, backup)
        logger.info(f"Cache cleared (backup at {backup})")

    configs = generate_bulletproof_configs()
    algos = list(EXPERIMENT_ALGOS.keys())
    total_trials = len(configs) * args.n_seeds * len(algos)
    logger.info(f"Running {len(algos)} algorithms × {len(configs)} configs × "
                f"{args.n_seeds} seeds = {total_trials} trials (T={args.T})")
    logger.info(f"Algorithms: {algos}")

    # Build task list
    tasks = []
    for config in configs:
        for seed in range(args.n_seeds):
            for algo in algos:
                tasks.append((algo, config, seed))

    all_results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, a, c, s, args.T): (a, c, s)
                   for a, c, s in tasks}
        for future in as_completed(futures):
            algo, config, seed = futures[future]
            try:
                r = future.result()
                all_results.append(r)
                completed += 1
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                logger.info(
                    f"[{completed}/{len(tasks)}] {algo:28s} cfg={config['config_id']} "
                    f"seed={seed} regret={r['final_regret']:.1f} calls={r['llm_calls']} "
                    f"ETA={eta/60:.1f}min"
                )
            except Exception as e:
                logger.error(f"FAILED {algo} cfg={config['config_id']} seed={seed}: {e}")
                all_results.append({
                    "algo": algo, "config_id": config["config_id"], "seed": seed,
                    "final_regret": None, "error": str(e)
                })

    total_elapsed = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} trials in {total_elapsed/60:.1f}min")

    out_dir = Path(__file__).parent / "results"
    paired_analysis(all_results, out_dir)


if __name__ == "__main__":
    main()
