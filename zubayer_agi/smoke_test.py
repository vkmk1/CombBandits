"""Smoke test: run all 17 novel LLM-bandit algorithms + CTS baseline.

Configs: d=30, m=5, T=800, 3 gap types × 3 seeds = 9 trials per algorithm.
All algorithms share a single GPTOracle with SQLite cache, so repeated prompts
(which happen across seeds with same initial state) are free.

Output: comparison table, per-algo regret, LLM call counts, cache hit rate.
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

from algorithms import ALL_ALGORITHMS, NEEDS_ORACLE
from oracle import GPTOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Config generation ────────────────────────────────────────────────────
def generate_configs() -> list[dict]:
    """3 representative configs matching production settings."""
    configs = []
    for i, (gap_type, delta_min) in enumerate([
        ("uniform", 0.1),
        ("uniform", 0.2),
        ("hard", 0.15),
    ]):
        configs.append({
            "config_id": i,
            "d": 30, "m": 5,
            "gap_type": gap_type,
            "delta_min": delta_min,
            "env_seed": 1000 + i * 7,
        })
    return configs


def build_env(config: dict, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
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
    optimal_reward = means[optimal].sum()
    return means, optimal, optimal_reward


# ─── Trial runner ─────────────────────────────────────────────────────────
def run_algo_trial(algo_name: str, config: dict, seed: int, T: int, oracle: GPTOracle):
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = ALL_ALGORITHMS[algo_name]
    kwargs = {"d": config["d"], "m": config["m"]}
    if algo_name in NEEDS_ORACLE:
        kwargs["oracle"] = oracle

    agent = AlgoClass(**kwargs)

    cum_regret = 0.0
    regret_curve = []
    t_start = time.time()

    for t in range(T):
        selected = agent.select_arms()
        if len(selected) < config["m"]:
            # Fallback: pad with random
            used = set(selected)
            remaining = [a for a in range(config["d"]) if a not in used]
            selected.extend(remaining[:config["m"] - len(selected)])

        selected_means = means[selected[:config["m"]]]
        rewards = (reward_rng.uniform(size=config["m"]) < selected_means).astype(float)
        inst_regret = optimal_reward - selected_means.sum()
        cum_regret += inst_regret
        agent.update(selected[:config["m"]], rewards.tolist())

        if (t + 1) % 100 == 0:
            regret_curve.append(round(cum_regret, 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if algo_name in NEEDS_ORACLE else {}

    return {
        "algo": algo_name,
        "config_id": config["config_id"],
        "seed": seed,
        "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve,
        "elapsed_sec": round(elapsed, 2),
        "llm_calls_so_far": diag.get("total_calls", 0),
        "llm_tokens_so_far": diag.get("total_tokens", 0),
        "cache_hits_so_far": diag.get("cache_hits", 0),
    }


def run_all_for_config_seed(config: dict, seed: int, T: int, algos_to_run: list[str],
                            shared_oracle: GPTOracle):
    """Run all algorithms for one (config, seed) pair."""
    results = []
    for algo_name in algos_to_run:
        try:
            r = run_algo_trial(algo_name, config, seed, T, shared_oracle)
            results.append(r)
            logger.info(f"  {algo_name:32s} config={config['config_id']} seed={seed} "
                        f"regret={r['final_regret']:.1f} elapsed={r['elapsed_sec']:.1f}s")
        except Exception as e:
            logger.error(f"  FAILED {algo_name} config={config['config_id']} seed={seed}: {e}")
            results.append({
                "algo": algo_name, "config_id": config["config_id"], "seed": seed,
                "final_regret": None, "error": str(e),
            })
    return results


# ─── Analysis ─────────────────────────────────────────────────────────────
def analyze(all_results: list[dict], out_dir: Path):
    import pandas as pd
    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 90)
    lines.append("SMOKE TEST: 17 LLM-BANDIT ALGORITHMS vs CTS BASELINE")
    lines.append("=" * 90)
    lines.append(f"Trials per algo: {df.groupby('algo').size().iloc[0] if len(df) > 0 else 0}")
    lines.append("")

    # Overall ranking
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "min", "max", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)

    baseline_mean = stats.loc["cts_baseline", "mean"] if "cts_baseline" in stats.index else None

    lines.append("--- OVERALL RANKING (lower regret = better) ---")
    lines.append(f"{'rank':<5} {'algorithm':<32s} {'mean_regret':>12s} {'stderr':>8s} {'vs_CTS':>10s}")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        if baseline_mean is not None:
            pct = (baseline_mean - row["mean"]) / baseline_mean * 100
            vs = f"{pct:+.1f}%"
        else:
            vs = "—"
        flag = ""
        if baseline_mean is not None and row["mean"] < baseline_mean:
            flag = " <-- beats CTS"
        lines.append(f"{rank:<5} {algo:<32s} {row['mean']:>12.1f} {row['stderr']:>8.2f} {vs:>10s}{flag}")
    lines.append("")

    # Per-config breakdown
    lines.append("--- PER-CONFIG MEAN REGRET ---")
    pivot = df.pivot_table(index="algo", columns="config_id", values="final_regret", aggfunc="mean").round(1)
    pivot = pivot.reindex(stats.index)
    header = f"{'algorithm':<32s}" + "".join(f"{'cfg_'+str(c):>10s}" for c in pivot.columns)
    lines.append(header)
    for algo, row in pivot.iterrows():
        row_str = f"{algo:<32s}" + "".join(f"{v:>10.1f}" if not np.isnan(v) else f"{'—':>10s}" for v in row)
        lines.append(row_str)
    lines.append("")

    # Win rate
    lines.append("--- WIN COUNT (how often each algo had lowest regret in (config, seed)) ---")
    wins = {}
    for (cid, sd), grp in df.groupby(["config_id", "seed"]):
        best = grp.loc[grp["final_regret"].idxmin(), "algo"]
        wins[best] = wins.get(best, 0) + 1
    total = df.groupby(["config_id", "seed"]).ngroups
    for algo, w in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {algo:<32s} {w}/{total}")
    lines.append("")

    # LLM efficiency
    if "llm_calls_so_far" in df.columns:
        lines.append("--- LLM COST (max calls per trial, cumulative across configs/seeds) ---")
        # Use max per algo (approximate since shared oracle makes this tricky)
        calls = df.groupby("algo")["llm_calls_so_far"].max().round(0)
        for algo in stats.index:
            if algo in calls.index:
                lines.append(f"  {algo:<32s} max_calls_seen={int(calls[algo])}")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"smoke_{ts}_report.txt", "w") as f:
        f.write(report)
    with open(out_dir / f"smoke_{ts}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results → {out_dir}/smoke_{ts}_*")


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--algos", type=str, default="all")
    parser.add_argument("--skip", type=str, default="", help="comma-separated algo names to skip")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    configs = generate_configs()

    if args.algos == "all":
        algos_to_run = list(ALL_ALGORITHMS.keys())
    else:
        algos_to_run = args.algos.split(",")

    if args.skip:
        skip = set(args.skip.split(","))
        algos_to_run = [a for a in algos_to_run if a not in skip]

    logger.info(f"Running {len(algos_to_run)} algorithms × {len(configs)} configs × {args.n_seeds} seeds = "
                f"{len(algos_to_run) * len(configs) * args.n_seeds} trials")
    logger.info(f"T={args.T}, d=30, m=5")

    # Shared oracle so all algos benefit from cache
    shared_oracle = GPTOracle(d=30, m=5)

    all_results = []
    for config in configs:
        for seed in range(args.n_seeds):
            logger.info(f"=== config {config['config_id']} (gap={config['gap_type']}, "
                        f"delta={config['delta_min']}) seed {seed} ===")
            results = run_all_for_config_seed(config, seed, args.T, algos_to_run, shared_oracle)
            all_results.extend(results)

    logger.info(f"Oracle final diagnostics: {shared_oracle.diagnostics()}")

    out_dir = Path(__file__).parent / "results"
    analyze(all_results, out_dir)


if __name__ == "__main__":
    main()
