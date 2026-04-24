"""Tier 3: FULL CROSSED experiment — every (algo, config, model, seed).

Unlike Tier 2 where each config was tied to a single model, here EVERY model
evaluates EVERY config. This isolates algorithm effects from config effects.

Design:
- 10 algorithms (baselines + our earlier + new breakthroughs)
- 12 configs (same as Tier 2)
- 3 models (gpt-4.1-mini, gpt-5-mini, gpt-5.4)
- 3 seeds per (model, config)
Total: 10 × 12 × 3 × 3 = 1080 trials

Same elaborate logging as Tier 2.
Allows DIRECT comparison:
- Holds (config, seed) fixed → measure pure model effect
- Holds (model, seed) fixed → measure pure config effect
- Holds everything but algo fixed → pure algorithm effect (paired)
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
from oracle_instrumented import InstrumentedOracle
from oracle import CACHE_DB
from tier2_experiment import generate_tier2_configs, build_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


TIER3_ALGOS = {
    # Baselines
    "cts_baseline": ALL_ALGORITHMS["cts_baseline"],
    # Earlier promising (user requested re-test)
    "OURS_B2_icpd": ALL_ALGORITHMS["B2_icpd_cts"],
    "OURS_F2_query_design": ALL_ALGORITHMS["F2_query_design_cts"],
    # Current champion
    "CHAMP_M2_corr_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    # Best paper baselines
    "PAPER_ts_llm": PAPER_BASELINES["PAPER_ts_llm"],
    "PAPER_cal_gated": PAPER_BASELINES["PAPER_calibration_gated"],
    # New breakthrough directions
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "N2_hypo_ts": BREAKTHROUGH_ALGOS["N2_hypo_ts"],
    "N3_info_min": BREAKTHROUGH_ALGOS["N3_info_min"],
    "N4_robust_corr": BREAKTHROUGH_ALGOS["N4_robust_corr"],
}
NEEDS_ORACLE = {k for k in TIER3_ALGOS if k != "cts_baseline"}

MODELS = ["gpt-4.1-mini", "gpt-5-mini", "gpt-5.4"]


def run_single_trial(algo_name: str, config: dict, seed: int, model: str,
                     T: int, out_dir: Path) -> dict:
    trial_id = f"{model[:8]}_cfg{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER3_ALGOS[algo_name]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": trial_np_seed}

    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=model,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=algo_name,
        )
        kwargs["oracle"] = oracle

    if algo_name in ("CHAMP_M2_corr_cts",):
        kwargs["T_horizon"] = T

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
        if (t + 1) % 100 == 0:
            regret_curve.append(round(float(cum_regret), 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}
    return {
        "trial_id": trial_id, "algo": algo_name, "model": model,
        "config_id": config["config_id"], "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"], "delta_min": config["delta_min"],
        "T": T, "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve, "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
    }


def analyze_crossed(all_results: list[dict], out_dir: Path):
    """Crossed design analysis: isolate algo / model / config effects."""
    import pandas as pd

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 100)
    lines.append("TIER 3 CROSSED EXPERIMENT — Every algo × config × model × seed")
    lines.append("=" * 100)
    lines.append(f"Total trials: {len(valid)}")
    lines.append(f"Algos: {df['algo'].nunique()}, Configs: {df['config_id'].nunique()}, "
                 f"Models: {df['model'].nunique()}, Seeds: {df['seed'].nunique()}")
    lines.append("")

    # Global ranking (all combined)
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)
    cts_mean = stats.loc["cts_baseline", "mean"] if "cts_baseline" in stats.index else None

    lines.append("--- GLOBAL RANKING (n trials each) ---")
    lines.append(f"{'rank':<5}{'algorithm':<26s}{'mean':>9s}{'stderr':>8s}{'median':>9s}{'vs_CTS':>10s}")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs = f"{(cts_mean - row['mean']) / cts_mean * 100:+.1f}%" if cts_mean else "—"
        flag = " <-- BEATS CTS" if cts_mean and row["mean"] < cts_mean else ""
        lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
                     f"{row['median']:>9.1f}{vs:>10s}{flag}")
    lines.append("")

    # Per-model ranking
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        s = sub.groupby("algo")["final_regret"].agg(["mean", "std", "count"])
        s["stderr"] = s["std"] / np.sqrt(s["count"])
        s = s.sort_values("mean").round(2)
        cm = s.loc["cts_baseline", "mean"] if "cts_baseline" in s.index else None
        lines.append(f"--- MODEL: {model} ---")
        lines.append(f"{'rank':<5}{'algorithm':<26s}{'mean':>9s}{'stderr':>8s}{'vs_CTS':>10s}")
        for rank, (algo, row) in enumerate(s.iterrows(), 1):
            vs = f"{(cm - row['mean']) / cm * 100:+.1f}%" if cm else "—"
            flag = " <-- BEATS" if cm and row["mean"] < cm else ""
            lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}{vs:>10s}{flag}")
        lines.append("")

    # Paired comparison vs CTS — GLOBAL (same config+seed+model)
    by_trial = defaultdict(dict)
    for r in valid:
        by_trial[(r["model"], r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]

    lines.append("--- PAIRED vs CTS (each trial: same (model, config, seed)) ---")
    lines.append(f"  {'algorithm':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}"
                 f"{'stderr':>8s}{'t_stat':>8s}{'sign_p':>9s}")
    for algo in [a for a in stats.index if a != "cts_baseline"]:
        diffs = []
        w = l = 0
        for trs in by_trial.values():
            if "cts_baseline" in trs and algo in trs:
                d = trs["cts_baseline"] - trs[algo]
                diffs.append(d)
                if d > 0: w += 1
                elif d < 0: l += 1
        if not diffs: continue
        md = np.mean(diffs)
        sem = np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
        t_stat = md / sem if sem > 0 else 0
        n = len(diffs)
        extreme = min(w, l)
        sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n)) if n > 0 else 1.0
        flag = " ★★" if md > 0 and sp < 0.01 else " ★" if md > 0 and sp < 0.05 else ""
        lines.append(f"  {algo:<26s}{w:>6d}{l:>7d}{md:>11.1f}{sem:>8.2f}"
                     f"{t_stat:>8.2f}{sp:>9.4f}{flag}")
    lines.append("")

    # PAIRED vs CTS per model
    lines.append("--- PAIRED vs CTS (per model breakdown) ---")
    for model in sorted(df["model"].unique()):
        lines.append(f"\n  {model}:")
        lines.append(f"  {'algorithm':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}"
                     f"{'sign_p':>9s}")
        for algo in [a for a in stats.index if a != "cts_baseline"]:
            diffs = []
            w = l = 0
            for (m, c, s), trs in by_trial.items():
                if m != model: continue
                if "cts_baseline" in trs and algo in trs:
                    d = trs["cts_baseline"] - trs[algo]
                    diffs.append(d)
                    if d > 0: w += 1
                    elif d < 0: l += 1
            if not diffs: continue
            md = np.mean(diffs)
            n = len(diffs)
            extreme = min(w, l)
            sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n))
            flag = " ★★" if md > 0 and sp < 0.01 else " ★" if md > 0 and sp < 0.05 else ""
            lines.append(f"  {algo:<26s}{w:>6d}{l:>7d}{md:>11.1f}{sp:>9.4f}{flag}")
    lines.append("")

    # HEAD-TO-HEAD: CHAMP_M2 vs N1/N2/N3/N4 (is any breakthrough better?)
    lines.append("--- HEAD-TO-HEAD: CORR-CTS vs Breakthroughs (same (model,config,seed)) ---")
    lines.append(f"  {'challenger':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}{'sign_p':>9s}")
    for challenger in ["N1_corr_full", "N2_hypo_ts", "N3_info_min", "N4_robust_corr"]:
        diffs = []
        w = l = 0
        for trs in by_trial.values():
            if "CHAMP_M2_corr_cts" in trs and challenger in trs:
                d = trs["CHAMP_M2_corr_cts"] - trs[challenger]
                diffs.append(d)
                if d > 0: w += 1
                elif d < 0: l += 1
        if not diffs: continue
        md = np.mean(diffs)
        n = len(diffs)
        extreme = min(w, l)
        sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n))
        flag = " ★★ BEATS CORR" if md > 0 and sp < 0.01 else " ★ BEATS CORR" if md > 0 and sp < 0.05 else ""
        lines.append(f"  {challenger:<26s}{w:>6d}{l:>7d}{md:>11.1f}{sp:>9.4f}{flag}")
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier3_crossed_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    if not args.keep_cache and CACHE_DB.exists():
        shutil.move(CACHE_DB, CACHE_DB.with_suffix(".sqlite.backup_tier3"))
        logger.info("Cache cleared")

    configs = generate_tier2_configs()
    algos = list(TIER3_ALGOS.keys())
    tasks = [(a, c, s, m) for a in algos for c in configs for s in range(args.n_seeds) for m in MODELS]
    logger.info(f"Running {len(algos)} algos × {len(configs)} configs × {args.n_seeds} seeds × "
                f"{len(MODELS)} models = {len(tasks)} trials (T={args.T})")

    all_results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, a, c, s, m, args.T, out_dir): (a, c, s, m)
                   for a, c, s, m in tasks}
        for future in as_completed(futures):
            algo, config, seed, model = futures[future]
            try:
                r = future.result()
                all_results.append(r)
                with open(out_dir / "raw_trials.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t_start, 0.1)
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:22s} cfg={config['config_id']} "
                        f"seed={seed} {model} regret={r['final_regret']:.1f} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} cfg={config['config_id']} seed={seed} {model}: {e}")

    logger.info(f"Done: {completed}/{len(tasks)} in {(time.time()-t_start)/60:.1f}min")
    analyze_crossed(all_results, out_dir)


if __name__ == "__main__":
    main()
