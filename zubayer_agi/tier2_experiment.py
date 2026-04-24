"""Tier 2 experiment: 3 models × 4 configs each × 10 seeds × 8 algos = 960 trials.

ELABORATE LOGGING to results/tier2_{timestamp}/:
- raw_trials.jsonl     — one line per trial (full regret curve, state summary)
- llm_calls.jsonl      — every LLM call (prompts, responses, tokens, logprobs)
- algo_states.jsonl    — per-trial algorithm state snapshots (every 100 rounds)
- report.txt           — human-readable summary
- report.json          — machine-readable aggregate stats

Model split (non-overlapping):
- gpt-4.1-mini: configs 0-3, seeds 0-9 (40 trials/algo)
- gpt-5-mini:   configs 4-7, seeds 0-9 (40 trials/algo)
- gpt-5.4:      configs 8-11, seeds 0-9 (40 trials/algo)

Total: 12 unique configs × 10 seeds = 120 trials per algo, 8 algos = 960 trials.
Pre-registered statistical tests at the bottom.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS
from paper_baselines import PAPER_BASELINES
from masterpiece_algorithms import MASTERPIECE_ALGOS
from oracle_instrumented import InstrumentedOracle
from oracle import CACHE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Tier 2 algorithm roster ──────────────────────────────────────────────
TIER2_ALGOS = {
    "cts_baseline": ALL_ALGORITHMS["cts_baseline"],
    "OURS_B2_icpd": ALL_ALGORITHMS["B2_icpd_cts"],
    "OURS_M2_corr_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    "PAPER_ts_llm": PAPER_BASELINES["PAPER_ts_llm"],
    "PAPER_jump_start": PAPER_BASELINES["PAPER_llm_jump_start"],
    "PAPER_cal_gated": PAPER_BASELINES["PAPER_calibration_gated"],
    "PAPER_llm_cucb_at": PAPER_BASELINES["PAPER_llm_cucb_at"],
    "weak_oracle_topm": ALL_ALGORITHMS["WARM_warm_start_cts"],  # stand-in for direct-LLM-greedy
}
NEEDS_ORACLE = {k for k in TIER2_ALGOS if k != "cts_baseline"}


def generate_tier2_configs() -> list[dict]:
    """12 configs spanning d∈{30,50}, mix of uniform/hard, varied δ_min."""
    configs = []
    cid = 0
    for d, delta, gap_type, seed in [
        # d=30 batch (4 configs for gpt-4.1-mini)
        (30, 0.12, "uniform", 6001),
        (30, 0.18, "uniform", 6013),
        (30, 0.10, "hard",    6029),
        (30, 0.20, "hard",    6047),
        # d=50 batch (4 configs for gpt-5-mini)
        (50, 0.10, "uniform", 6061),
        (50, 0.15, "uniform", 6079),
        (50, 0.08, "hard",    6091),
        (50, 0.18, "hard",    6113),
        # d=30/50 mix (4 configs for gpt-5.4) — OOD variety
        (30, 0.08, "uniform", 6131),  # hardest uniform
        (30, 0.15, "hard",    6149),
        (50, 0.12, "uniform", 6167),
        (50, 0.10, "hard",    6173),
    ]:
        configs.append({
            "config_id": cid, "d": d, "m": 5,
            "gap_type": gap_type, "delta_min": delta, "env_seed": seed,
        })
        cid += 1
    return configs


# Model assignment by config range
def config_to_model(config_id: int) -> str:
    if config_id < 4:
        return "gpt-4.1-mini"
    elif config_id < 8:
        return "gpt-5-mini"
    else:
        return "gpt-5.4"


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


# ─── Trial runner with full state logging ─────────────────────────────────
def run_single_trial(algo_name: str, config: dict, seed: int, T: int,
                     out_dir: Path) -> dict:
    model = config_to_model(config["config_id"])
    trial_id = f"cfg{config['config_id']}_s{seed}_{algo_name}"

    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER2_ALGOS[algo_name]
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

    if algo_name in ("M1_b2_plus", "M3_b2_correlated", "OURS_M2_corr_cts"):
        kwargs["T_horizon"] = T

    agent = AlgoClass(**kwargs)

    cum_regret = 0.0
    regret_curve = []           # every 100 rounds
    state_snapshots = []        # every 100 rounds
    selections_log = []         # every round, compact
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
        inst_regret = optimal_reward - selected_means.sum()
        cum_regret += inst_regret
        agent.update(selected, rewards.tolist())

        # Compact per-round selections log
        if t < 100 or t % 50 == 0:
            selections_log.append({
                "t": t, "arms": [int(a) for a in selected],
                "rewards": [int(r) for r in rewards],
                "inst_regret": round(float(inst_regret), 4),
            })

        if (t + 1) % 100 == 0:
            regret_curve.append(round(cum_regret, 2))
            # Snapshot state
            snap = {
                "trial_id": trial_id, "t": t + 1,
                "cum_regret": round(float(cum_regret), 2),
                "n_pulls_optimal": int(sum(agent.n_pulls[o] for o in optimal.tolist())),
                "n_pulls_total": int(agent.n_pulls.sum()),
                "mu_hat_optimal_avg": round(float(np.mean(agent.mu_hat[optimal])), 4),
                "top5_by_mu_hat": [int(a) for a in np.argsort(agent.mu_hat)[::-1][:5]],
                "agreement_with_optimal": int(len(
                    set(np.argsort(agent.mu_hat)[::-1][:5].tolist()) &
                    set(optimal.tolist())
                )),
            }
            state_snapshots.append(snap)

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}

    # Write state snapshots to JSONL
    state_path = out_dir / "algo_states.jsonl"
    with open(state_path, "a") as f:
        for snap in state_snapshots:
            f.write(json.dumps(snap) + "\n")

    trial_result = {
        "trial_id": trial_id,
        "algo": algo_name,
        "model": model,
        "config_id": config["config_id"],
        "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"],
        "delta_min": config["delta_min"],
        "env_seed": config["env_seed"],
        "T": T,
        "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve,
        "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
        "optimal_reward_per_round": round(float(optimal_reward), 4),
        "early_selections": selections_log[:100],  # first 100 rounds in detail
    }
    return trial_result


# ─── Aggregate analysis ───────────────────────────────────────────────────
def analyze_and_report(all_results: list[dict], out_dir: Path):
    import pandas as pd
    from math import comb
    from collections import defaultdict

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 100)
    lines.append("TIER 2 EXPERIMENT: Multi-Model Evaluation (gpt-4.1-mini + gpt-5-mini + gpt-5.4)")
    lines.append("=" * 100)
    lines.append(f"Total trials: {len(valid)}")
    lines.append(f"Unique configs: {df['config_id'].nunique()}")
    lines.append(f"Algorithms: {df['algo'].nunique()}")
    lines.append(f"Seeds per config: {df.groupby('config_id')['seed'].nunique().iloc[0] if len(df) else 0}")
    lines.append(f"T: {df['T'].iloc[0] if len(df) else 'N/A'}")
    lines.append("")

    # Per-model ranking (key insight — does B2 win on all 3 models?)
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        n_trials = sub.groupby("algo").size().iloc[0]
        stats = sub.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
        stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
        stats = stats.sort_values("mean").round(2)
        baseline = stats.loc["cts_baseline", "mean"] if "cts_baseline" in stats.index else None

        lines.append(f"--- MODEL: {model} ({n_trials} trials/algo) ---")
        lines.append(f"{'rank':<5}{'algorithm':<26s}{'mean':>9s}{'stderr':>8s}{'median':>9s}{'vs_CTS':>10s}")
        for rank, (algo, row) in enumerate(stats.iterrows(), 1):
            vs = f"{(baseline - row['mean']) / baseline * 100:+.1f}%" if baseline else "—"
            flag = " <-- BEATS CTS" if baseline and row["mean"] < baseline else ""
            lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
                         f"{row['median']:>9.1f}{vs:>10s}{flag}")
        lines.append("")

    # Global ranking (all models combined)
    lines.append("--- GLOBAL RANKING (all 3 models aggregated) ---")
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)
    baseline = stats.loc["cts_baseline", "mean"] if "cts_baseline" in stats.index else None
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs = f"{(baseline - row['mean']) / baseline * 100:+.1f}%" if baseline else "—"
        flag = " <-- BEATS CTS" if baseline and row["mean"] < baseline else ""
        lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
                     f"{row['median']:>9.1f}{vs:>10s}{flag}")
    lines.append("")

    # Paired comparison vs CTS (per model + global)
    by_trial = defaultdict(dict)
    for r in valid:
        by_trial[(r["model"], r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]

    lines.append("--- PAIRED COMPARISON vs CTS (per model) ---")
    for model in sorted(df["model"].unique()):
        lines.append(f"\n  {model}:")
        lines.append(f"  {'algorithm':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}"
                     f"{'stderr':>8s}{'t_stat':>8s}{'sign_p':>9s}")
        for algo in [a for a in stats.index if a != "cts_baseline"]:
            diffs = []
            wins = losses = 0
            for (m, c, s), trials in by_trial.items():
                if m != model: continue
                if "cts_baseline" in trials and algo in trials:
                    d = trials["cts_baseline"] - trials[algo]
                    diffs.append(d)
                    if d > 0: wins += 1
                    elif d < 0: losses += 1
            if not diffs: continue
            mean_d = np.mean(diffs)
            sem = np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
            t_stat = mean_d / sem if sem > 0 else 0
            n = len(diffs)
            extreme = min(wins, losses)
            sign_p = 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n) if n > 0 else 1.0
            sign_p = min(1.0, sign_p)
            lines.append(f"  {algo:<26s}{wins:>6d}{losses:>7d}{mean_d:>11.1f}{sem:>8.2f}"
                         f"{t_stat:>8.2f}{sign_p:>9.4f}")
    lines.append("")

    # Per-config breakdown
    lines.append("--- PER-CONFIG MEAN REGRET (all models) ---")
    pivot = df.pivot_table(index="algo", columns="config_id",
                           values="final_regret", aggfunc="mean").round(1)
    pivot = pivot.reindex(stats.index)
    header = f"{'algorithm':<26s}" + "".join(f"{'c'+str(c):>7s}" for c in pivot.columns)
    lines.append(header)
    for algo, row in pivot.iterrows():
        row_str = f"{algo:<26s}" + "".join(
            f"{v:>7.0f}" if not np.isnan(v) else f"{'—':>7s}" for v in row
        )
        lines.append(row_str)
    lines.append("")

    # LLM cost per model × algo
    lines.append("--- LLM COST (calls/trial, tokens/trial) ---")
    cost = df[df["llm_calls"] > 0].groupby(["model", "algo"]).agg({
        "llm_calls": "mean", "llm_tokens": "mean"
    }).round(1)
    lines.append(f"  {'model':<18s}{'algorithm':<26s}{'calls':>8s}{'tokens':>8s}")
    for (m, a), row in cost.iterrows():
        lines.append(f"  {m:<18s}{a:<26s}{row['llm_calls']:>8.1f}{row['llm_tokens']:>8.0f}")
    lines.append("")

    # Statistical summary
    lines.append("=" * 100)
    lines.append("STATISTICAL SIGNIFICANCE (pre-registered tests)")
    lines.append("=" * 100)
    # Global paired comparison
    for algo in ["OURS_B2_icpd", "OURS_M2_corr_cts"]:
        diffs = []
        for (m, c, s), trials in by_trial.items():
            if "cts_baseline" in trials and algo in trials:
                diffs.append(trials["cts_baseline"] - trials[algo])
        if not diffs: continue
        n = len(diffs)
        mean_d = np.mean(diffs)
        sem = np.std(diffs, ddof=1) / np.sqrt(n)
        t_stat = mean_d / sem if sem > 0 else 0
        wins = sum(1 for d in diffs if d > 0)
        extreme = min(wins, n - wins)
        sign_p = 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n)
        lines.append(f"  {algo}: n={n}, wins={wins}/{n}, mean_reduction={mean_d:.1f}, "
                     f"t={t_stat:.2f}, sign_p={min(1.0, sign_p):.4f}")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Fresh output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    # Fresh cache unless --keep-cache
    if not args.keep_cache and CACHE_DB.exists():
        backup = CACHE_DB.with_suffix(".sqlite.backup_tier2")
        shutil.move(CACHE_DB, backup)
        logger.info(f"Cache cleared → {backup}")

    # Set instrumented log file
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")

    configs = generate_tier2_configs()
    algos = list(TIER2_ALGOS.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]

    logger.info(f"Running {len(algos)} algos × {len(configs)} configs × {args.n_seeds} seeds "
                f"= {len(tasks)} trials (T={args.T})")
    logger.info(f"Models: gpt-4.1-mini (configs 0-3), gpt-5-mini (4-7), gpt-5.4 (8-11)")

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
                # Write to raw_trials.jsonl incrementally
                with open(out_dir / "raw_trials.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                completed += 1
                rate = completed / max(time.time() - t_start, 0.1)
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                if completed % 10 == 0 or completed == len(tasks):
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:22s} cfg={config['config_id']} "
                        f"seed={seed} model={config_to_model(config['config_id'])} "
                        f"regret={r['final_regret']:.1f} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} cfg={config['config_id']} seed={seed}: {e}")
                all_results.append({
                    "algo": algo, "config_id": config["config_id"], "seed": seed,
                    "final_regret": None, "error": str(e),
                })

    total = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} in {total/60:.1f}min")

    analyze_and_report(all_results, out_dir)

    # Machine-readable summary
    with open(out_dir / "report.json", "w") as f:
        json.dump({
            "total_trials": len(all_results),
            "results": all_results,
            "elapsed_min": round(total / 60, 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
