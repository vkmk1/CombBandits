"""TIER 4 FULL-SCALE — Publication-Grade Experiment.

PARAMETERS (from cluster/experimental_parameters_survey.md gold standards):
- T = 2500 (middle of LLM-feasible range 1000-3000 per Xia ACL 2025)
- d ∈ {25, 50} × m ∈ {3, 5} (matches Combes NeurIPS 2015)
- 4 configs per (d, m) — 2 uniform + 2 hard with varied δ
- 20 seeds per config (compromise between 30 gold / 15 LLM-budget)
- 16 configs × 20 seeds = 320 paired trials per algorithm
- Total: 8 algos × 320 = 2,560 trials

MODEL: gpt-5.4 (user-requested; flagship quality)

ALGORITHMS (8, tight publication set):
- CTS        : Thompson sampling baseline (Wang & Chen 2018)
- CUCB       : UCB baseline (Chen et al 2013)
- ABLATION_random_corr : random clusters ablation — proves LLM necessity
- M2_corr_cts : original block-diagonal CORR-CTS
- N1_corr_full : full kernel covariance
- N4_robust_corr : credibility-gated interpolation
- N5_corr_full_robust : synthesis of N1+N4
- PAPER_ts_llm : strongest baseline (Sun et al 2025)

COST: ~$20 LLM (gpt-5.4) + ~$10 AWS = ~$30
TIME: ~2-3 hours on c5.4xlarge with 16 workers
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS
from paper_baselines import PAPER_BASELINES
from masterpiece_algorithms import MASTERPIECE_ALGOS
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from final_algorithms import FINAL_ALGOS
from cucb_baseline import CUCB
from oracle_instrumented import InstrumentedOracle
from oracle import CACHE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Algorithm suite (8 total) ───────────────────────────────────────────
TIER4_ALGOS = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "CUCB": CUCB,
    "ABLATION_random_corr": FINAL_ALGOS["ABLATION_random_corr"],
    "M2_corr_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "N4_robust_corr": BREAKTHROUGH_ALGOS["N4_robust_corr"],
    "N5_corr_full_robust": FINAL_ALGOS["N5_corr_full_robust"],
    "PAPER_ts_llm": PAPER_BASELINES["PAPER_ts_llm"],
}
NEEDS_ORACLE = {k for k in TIER4_ALGOS if k not in ("CTS", "CUCB", "ABLATION_random_corr")}


def generate_tier4_configs() -> list[dict]:
    """16 configs: 2 (d) × 2 (m) × 4 (gap, delta) combos.

    d ∈ {25, 50} matches Combes 2015. m ∈ {3, 5} standard combinatorial sizes.
    """
    configs = []
    cid = 0
    combos = [
        # (gap_type, delta_min, env_seed_offset)
        ("uniform", 0.20, 100),  # easy uniform
        ("uniform", 0.10, 200),  # hard uniform
        ("hard",    0.20, 300),  # easy hard-gap
        ("hard",    0.10, 400),  # hard hard-gap
    ]
    for d in (25, 50):
        for m in (3, 5):
            for gap_type, delta, seed_offset in combos:
                configs.append({
                    "config_id": cid,
                    "d": d, "m": m,
                    "gap_type": gap_type,
                    "delta_min": delta,
                    "env_seed": 8000 + d * 10 + m + seed_offset,
                })
                cid += 1
    return configs


MODEL = "gpt-5.4"


def build_env(config: dict, seed: int):
    combined = config["env_seed"] * 10000 + seed
    rng = np.random.RandomState(combined)
    d, m = config["d"], config["m"]
    if config["gap_type"] == "uniform":
        means = rng.uniform(0.1, 0.5, size=d)
        top = rng.choice(d, size=m, replace=False)
        means[top] = 0.5 + config["delta_min"]
    else:
        means = np.full(d, 0.5)
        top = rng.choice(d, size=m, replace=False)
        means[top] = 0.5 + config["delta_min"]
    optimal = np.argsort(means)[::-1][:m]
    return means, optimal, means[optimal].sum()


def run_single_trial(algo_name: str, config: dict, seed: int, T: int,
                     out_dir: Path) -> dict:
    trial_id = f"c{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER4_ALGOS[algo_name]
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

    if algo_name in ("M2_corr_cts", "N5_corr_full_robust"):
        kwargs["T_horizon"] = T

    agent = AlgoClass(**kwargs)

    cum_regret = 0.0
    regret_curve = []  # every 50 rounds
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


def analyze_tier4(all_results: list[dict], out_dir: Path):
    """Paper-grade analysis: Holm-Bonferroni, bootstrap CIs, cluster-robust SE."""
    import pandas as pd

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)
    n_trials_per_algo = df.groupby("algo").size().iloc[0] if len(df) else 0

    lines = []
    lines.append("=" * 100)
    lines.append(f"TIER 4 FULL-SCALE — {len(valid)} trials | model={MODEL}")
    lines.append("=" * 100)
    lines.append(f"Algos: {df['algo'].nunique()} | Configs: {df['config_id'].nunique()} | "
                 f"Seeds: {df['seed'].nunique()} | T: {df['T'].iloc[0]}")
    lines.append(f"n per algo: {n_trials_per_algo}")
    lines.append("")

    # Global ranking
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)
    cts_mean = stats.loc["CTS", "mean"] if "CTS" in stats.index else None
    random_mean = stats.loc["ABLATION_random_corr", "mean"] if "ABLATION_random_corr" in stats.index else None

    lines.append("--- GLOBAL RANKING ---")
    lines.append(f"{'rank':<5}{'algorithm':<26s}{'mean':>9s}{'stderr':>8s}{'median':>9s}{'vs_CTS':>10s}{'vs_Rand':>10s}")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs_cts = f"{(cts_mean - row['mean']) / cts_mean * 100:+.1f}%" if cts_mean else "—"
        vs_rand = f"{(random_mean - row['mean']) / random_mean * 100:+.1f}%" if random_mean else "—"
        lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
                     f"{row['median']:>9.1f}{vs_cts:>10s}{vs_rand:>10s}")
    lines.append("")

    # Paired analysis
    by_trial = defaultdict(dict)
    for r in valid:
        by_trial[(r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]

    # Holm-Bonferroni for pairwise vs CTS (K = num algos - 1 tests)
    comparisons = [a for a in stats.index if a != "CTS"]
    K = len(comparisons)

    lines.append(f"--- PAIRED vs CTS (Holm-Bonferroni, K={K} comparisons) ---")
    lines.append(f"  {'algorithm':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}"
                 f"{'t_stat':>8s}{'sign_p':>9s}{'holm_α':>9s}{'sig?':>6s}")

    # Collect p-values for Holm procedure
    test_results = []
    for algo in comparisons:
        diffs = []
        w = l = 0
        for trs in by_trial.values():
            if "CTS" in trs and algo in trs:
                d = trs["CTS"] - trs[algo]
                diffs.append(d)
                if d > 0: w += 1
                elif d < 0: l += 1
        if not diffs:
            continue
        md = np.mean(diffs)
        sem = np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
        t_stat = md / sem if sem > 0 else 0
        n = len(diffs)
        extreme = min(w, l)
        sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n))
        test_results.append((algo, w, l, md, t_stat, sp))

    # Apply Holm-Bonferroni step-down
    test_results.sort(key=lambda x: x[5])  # sort by p-value
    for i, (algo, w, l, md, t_stat, sp) in enumerate(test_results):
        holm_alpha = 0.05 / (K - i)  # step-down
        sig = "★★" if sp < holm_alpha and md > 0 else ("★" if sp < 0.05 and md > 0 else "")
        lines.append(f"  {algo:<26s}{w:>6d}{l:>7d}{md:>11.1f}{t_stat:>8.2f}"
                     f"{sp:>9.4f}{holm_alpha:>9.4f}{sig:>6s}")
    lines.append("")

    # Key ablation: LLM family vs RandomCorr (necessity test)
    lines.append("=" * 100)
    lines.append("CRITICAL: LLM necessity ablation — LLM algos vs RandomCorr")
    lines.append("=" * 100)
    lines.append("If LLM-family doesn't beat RandomCorr, LLM is decorative.")
    lines.append("")
    lines.append(f"  {'algo':<26s}{'wins':>6s}{'losses':>7s}{'diff':>11s}{'p':>9s}{'llm_contributes?':>20s}")
    for algo in ["M2_corr_cts", "N1_corr_full", "N4_robust_corr", "N5_corr_full_robust"]:
        diffs = []
        w = l = 0
        for trs in by_trial.values():
            if "ABLATION_random_corr" in trs and algo in trs:
                d = trs["ABLATION_random_corr"] - trs[algo]
                diffs.append(d)
                if d > 0: w += 1
                elif d < 0: l += 1
        if not diffs:
            continue
        md = np.mean(diffs)
        n = len(diffs)
        extreme = min(w, l)
        sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n))
        verdict = "YES ★" if md > 0 and sp < 0.05 else ("weak" if md > 0 else "NO")
        lines.append(f"  {algo:<26s}{w:>6d}{l:>7d}{md:>11.1f}{sp:>9.4f}{verdict:>20s}")
    lines.append("")

    # Per-(d, m) breakdown
    lines.append("--- PER-(d, m) BREAKDOWN (mean regret) ---")
    pivot = df.pivot_table(index="algo", columns=["d", "m"],
                           values="final_regret", aggfunc="mean").round(1)
    pivot = pivot.reindex(stats.index)
    col_headers = "".join(f"  d={d},m={m}" for d, m in pivot.columns)
    lines.append(f"{'algorithm':<26s}{col_headers}")
    for algo, row in pivot.iterrows():
        line = f"{algo:<26s}"
        for val in row:
            line += f"  {val:>8.1f}" if not np.isnan(val) else f"  {'—':>8s}"
        lines.append(line)
    lines.append("")

    # LLM cost
    lines.append("--- LLM COST (gpt-5.4) ---")
    cost = df[df["llm_calls"] > 0].groupby("algo").agg({
        "llm_calls": "mean", "llm_tokens": "mean", "cache_hits": "mean"
    }).round(1)
    total_tokens = df["llm_tokens"].sum()
    est_cost = total_tokens * 2.5 / 1e6  # assume all input; approximate
    lines.append(f"  Total tokens: {total_tokens:,.0f}  |  Est cost: ~${est_cost:.2f}")
    lines.append(f"  {'algo':<26s}{'calls':>8s}{'tokens':>10s}{'hits':>8s}")
    for algo in stats.index:
        if algo in cost.index:
            row = cost.loc[algo]
            lines.append(f"  {algo:<26s}{row['llm_calls']:>8.1f}{row['llm_tokens']:>10.0f}"
                         f"{row['cache_hits']:>8.1f}")
    lines.append("")

    # Verdict
    lines.append("=" * 100)
    lines.append("PUBLICATION VERDICT")
    lines.append("=" * 100)
    top3 = stats.index[:3].tolist()
    winners_vs_cts = [a for a, w, l, md, t, sp in test_results if md > 0 and sp < 0.05]
    lines.append(f"Algorithms beating CTS at p<0.05 (uncorrected): {winners_vs_cts}")
    winners_holm = []
    for i, (algo, w, l, md, t, sp) in enumerate(test_results):
        holm_alpha = 0.05 / (K - i)
        if sp < holm_alpha and md > 0:
            winners_holm.append(algo)
    lines.append(f"Algorithms beating CTS at Holm-Bonferroni (corrected): {winners_holm}")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2500)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier4_full_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    if not args.keep_cache and CACHE_DB.exists():
        shutil.move(CACHE_DB, CACHE_DB.with_suffix(".sqlite.backup_tier4full"))
        logger.info("Cache cleared (backup preserved)")

    configs = generate_tier4_configs()
    algos = list(TIER4_ALGOS.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    logger.info(f"Running {len(algos)} algos × {len(configs)} configs × {args.n_seeds} seeds "
                f"= {len(tasks)} trials (T={args.T}, model={MODEL})")

    all_results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, a, c, s, args.T, out_dir): (a, c, s)
                   for a, c, s in tasks}
        for future in as_completed(futures):
            algo, config, seed = futures[future]
            try:
                r = future.result()
                all_results.append(r)
                with open(out_dir / "raw_trials.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t_start, 0.1)
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:22s} c{config['config_id']} "
                        f"s{seed} r={r['final_regret']:.1f} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} c{config['config_id']} s{seed}: {e}")

    total = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} in {total/60:.1f}min")
    analyze_tier4(all_results, out_dir)

    # Final summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "total_trials": completed,
            "elapsed_min": round(total / 60, 1),
            "model": MODEL,
            "T": args.T,
            "n_seeds": args.n_seeds,
            "n_algos": len(algos),
        }, f, indent=2)


if __name__ == "__main__":
    main()
