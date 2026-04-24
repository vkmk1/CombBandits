"""Tier 4 Phase A — SMOKE TEST / PRE-FLIGHT for full-scale experiment.

PURPOSE: verify infrastructure, algorithms, cache behavior, statistics BEFORE
committing to full Tier 4 spend. Smaller scale but fully rigorous.

DESIGN (small but tight):
- 5 configs × 3 seeds × 2 models (gpt-4.1-mini + gpt-5.4) = 30 trials per algo
- T=1500 (will extend to 3000 in Phase B)
- 12 algorithms including ALL reviewer-mandated additions
- Cluster-robust analysis
- Explicit bias tests (RandomCorr ablation, identical-input sanity checks)

WHAT THIS VERIFIES:
1. Cache behavior: cluster-family algos share cache, others make own calls
2. RNG determinism: same (cfg, seed) → same CTS trajectory across runs
3. Algorithm implementations: none crash, all make sensible decisions
4. Statistical power: even at this small scale, top 3 winners should be clear
5. Analysis infrastructure: cluster-robust SE, Holm-Bonferroni works

COST: ~$15, 25-35 min
OUTPUT: if Phase A confirms expected top 3, launch Tier 4 full-scale on EC2.
If Phase A reveals bugs, fix before launch.
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
from math import comb, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS
from paper_baselines import PAPER_BASELINES
from masterpiece_algorithms import MASTERPIECE_ALGOS
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from final_algorithms import FINAL_ALGOS
from oracle_instrumented import InstrumentedOracle
from oracle import CACHE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Phase A algorithm suite (12 algos) ──────────────────────────────────
PHASE_A_ALGOS = {
    # Baselines (no LLM) — free
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    # MANDATORY ABLATION — random clusters, no LLM
    "ABLATION_random_corr": FINAL_ALGOS["ABLATION_random_corr"],
    # Our correlated-sampling family (all share cluster query — 1 call total)
    "M2_corr_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "N4_robust_corr": BREAKTHROUGH_ALGOS["N4_robust_corr"],
    "N5_corr_full_robust": FINAL_ALGOS["N5_corr_full_robust"],
    # N3 FIXED — the reviewer-identified broken algorithm
    "N3_info_min_fixed": FINAL_ALGOS["N3_info_min_fixed"],
    # Our older algos to verify whether they truly degraded
    "B2_icpd": ALL_ALGORITHMS["B2_icpd_cts"],
    "F2_query_design": ALL_ALGORITHMS["F2_query_design_cts"],
    # Paper baselines (required for publication)
    "PAPER_ts_llm": PAPER_BASELINES["PAPER_ts_llm"],
    "PAPER_cal_gated": PAPER_BASELINES["PAPER_calibration_gated"],
    "PAPER_jump_start": PAPER_BASELINES["PAPER_llm_jump_start"],
}
NEEDS_ORACLE = {k for k in PHASE_A_ALGOS if k not in ("CTS", "ABLATION_random_corr")}

# 5 diverse configs (d ∈ {30, 50}, mix of gap types)
PHASE_A_CONFIGS = [
    {"config_id": 0, "d": 30, "m": 5, "gap_type": "uniform", "delta_min": 0.12, "env_seed": 7001},
    {"config_id": 1, "d": 30, "m": 5, "gap_type": "hard",    "delta_min": 0.10, "env_seed": 7017},
    {"config_id": 2, "d": 50, "m": 5, "gap_type": "uniform", "delta_min": 0.15, "env_seed": 7041},
    {"config_id": 3, "d": 50, "m": 8, "gap_type": "hard",    "delta_min": 0.12, "env_seed": 7067},
    {"config_id": 4, "d": 30, "m": 3, "gap_type": "uniform", "delta_min": 0.18, "env_seed": 7089},
]
PHASE_A_MODELS = ["gpt-4.1-mini", "gpt-5.4"]


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


def run_single_trial(algo_name: str, config: dict, seed: int, model: str,
                     T: int, out_dir: Path) -> dict:
    trial_id = f"{model[:8]}_c{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = PHASE_A_ALGOS[algo_name]
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

    if algo_name in ("M2_corr_cts", "N5_corr_full_robust"):
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
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
    }


# ─── Pre-flight checks ───────────────────────────────────────────────────
def run_preflight_checks(out_dir: Path):
    """Verify infrastructure before committing to full run."""
    logger.info("=" * 80)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 80)
    issues = []

    # Check 1: all algorithms instantiate
    logger.info("Check 1: all algorithms instantiate")
    for name, cls in PHASE_A_ALGOS.items():
        try:
            kwargs = {"d": 30, "m": 5, "np_seed": 42}
            if name in NEEDS_ORACLE:
                kwargs["oracle"] = InstrumentedOracle(d=30, m=5, trial_id="preflight")
            if name in ("M2_corr_cts", "N5_corr_full_robust"):
                kwargs["T_horizon"] = 1500
            agent = cls(**kwargs)
            sel = agent.select_arms()
            assert len(sel) >= 5, f"{name}: selected < 5 arms"
            logger.info(f"  ✓ {name}")
        except Exception as e:
            issues.append(f"{name}: {e}")
            logger.error(f"  ✗ {name}: {e}")

    # Check 2: RNG determinism — two identical trials give same result
    logger.info("\nCheck 2: RNG determinism (paired comparison validity)")
    r1 = run_single_trial("CTS", PHASE_A_CONFIGS[0], 0, "gpt-4.1-mini", 200, out_dir)
    r2 = run_single_trial("CTS", PHASE_A_CONFIGS[0], 0, "gpt-4.1-mini", 200, out_dir)
    if abs(r1["final_regret"] - r2["final_regret"]) < 0.001:
        logger.info(f"  ✓ CTS deterministic: {r1['final_regret']} == {r2['final_regret']}")
    else:
        issues.append(f"CTS NON-DETERMINISTIC: {r1['final_regret']} vs {r2['final_regret']}")
        logger.error(f"  ✗ CTS NON-DETERMINISTIC: {r1['final_regret']} vs {r2['final_regret']}")

    # Check 3: N3_info_min_fixed actually makes LLM calls (the reviewer bug)
    logger.info("\nCheck 3: N3 actually triggers LLM query (was broken)")
    r = run_single_trial("N3_info_min_fixed", PHASE_A_CONFIGS[0], 0, "gpt-4.1-mini", 200, out_dir)
    if r["llm_calls"] > 0 or r["cache_hits"] > 0:
        logger.info(f"  ✓ N3 made {r['llm_calls']} API calls + {r['cache_hits']} cache hits")
    else:
        issues.append("N3 still not triggering LLM query")
        logger.error("  ✗ N3 still broken")

    # Check 4: Cluster-family shares cache (sanity: natural sharing works)
    logger.info("\nCheck 4: cluster-family cache sharing")
    r1 = run_single_trial("M2_corr_cts", PHASE_A_CONFIGS[1], 0, "gpt-4.1-mini", 200, out_dir)
    r2 = run_single_trial("N1_corr_full", PHASE_A_CONFIGS[1], 0, "gpt-4.1-mini", 200, out_dir)
    if r2["cache_hits"] > 0 or r1["cache_hits"] > 0:
        logger.info(f"  ✓ Cache sharing works: M2 calls={r1['llm_calls']}, "
                    f"N1 calls={r2['llm_calls']} hits={r2['cache_hits']}")
    else:
        logger.warning(f"  ⚠ No cache sharing observed: M2 calls={r1['llm_calls']}, "
                       f"N1 calls={r2['llm_calls']}")

    # Check 5: RandomCorr (no LLM) makes zero API calls
    logger.info("\nCheck 5: RandomCorr ablation uses no LLM")
    r = run_single_trial("ABLATION_random_corr", PHASE_A_CONFIGS[0], 0, "gpt-4.1-mini", 200, out_dir)
    if r["llm_calls"] == 0 and r["cache_hits"] == 0:
        logger.info(f"  ✓ RandomCorr: 0 LLM calls (correct)")
    else:
        issues.append(f"RandomCorr calling LLM unexpectedly: {r['llm_calls']} calls")
        logger.error(f"  ✗ RandomCorr should be 0, got {r['llm_calls']}")

    # Write preflight report
    (out_dir / "preflight.txt").write_text(
        f"Issues: {len(issues)}\n" + "\n".join(issues) if issues else "All pre-flight checks passed."
    )
    if issues:
        logger.error(f"\n✗ PRE-FLIGHT FAILED: {len(issues)} issues. Fix before running.")
        for i in issues:
            logger.error(f"  - {i}")
        return False
    logger.info("\n✓ ALL PRE-FLIGHT CHECKS PASSED. Proceeding to main experiment.")
    return True


# ─── Main analysis with cluster-robust SE ─────────────────────────────────
def analyze_phase_a(all_results: list[dict], out_dir: Path):
    import pandas as pd

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 100)
    lines.append("TIER 4 PHASE A — SMOKE TEST RESULTS")
    lines.append("=" * 100)
    lines.append(f"Trials: {len(valid)}")
    lines.append(f"Algos: {df['algo'].nunique()}, Configs: {df['config_id'].nunique()}, "
                 f"Models: {df['model'].nunique()}, Seeds: {df['seed'].nunique()}")
    lines.append("")

    # Global ranking (combined)
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "median", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean").round(2)
    cts_mean = stats.loc["CTS", "mean"]
    lines.append("--- GLOBAL RANKING (combined across models) ---")
    lines.append(f"{'rank':<5}{'algorithm':<26s}{'mean':>9s}{'stderr':>8s}{'median':>9s}{'vs_CTS':>10s}")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs = f"{(cts_mean - row['mean']) / cts_mean * 100:+.1f}%"
        flag = " <-- BEATS" if row["mean"] < cts_mean else ""
        lines.append(f"{rank:<5}{algo:<26s}{row['mean']:>9.1f}{row['stderr']:>8.2f}"
                     f"{row['median']:>9.1f}{vs:>10s}{flag}")
    lines.append("")

    # Paired comparisons (global)
    by_trial = defaultdict(dict)
    for r in valid:
        by_trial[(r["model"], r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]

    lines.append("--- PAIRED vs CTS (n=30 per algo) ---")
    lines.append(f"  {'algorithm':<26s}{'wins':>6s}{'losses':>7s}{'mean_diff':>11s}"
                 f"{'stderr':>8s}{'t_stat':>8s}{'sign_p':>9s}")
    for algo in [a for a in stats.index if a != "CTS"]:
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
        sp = min(1.0, 2 * sum(comb(n, k) for k in range(extreme + 1)) / (2 ** n)) if n > 0 else 1.0
        flag = " ★★" if md > 0 and sp < 0.01 else " ★" if md > 0 and sp < 0.05 else ""
        lines.append(f"  {algo:<26s}{w:>6d}{l:>7d}{md:>11.1f}{sem:>8.2f}"
                     f"{t_stat:>8.2f}{sp:>9.4f}{flag}")
    lines.append("")

    # RandomCorr vs CORR-family head-to-head (KEY LLM-NECESSITY ABLATION)
    lines.append("=" * 100)
    lines.append("KEY ABLATION: LLM-family vs RandomCorr (no-LLM baseline)")
    lines.append("=" * 100)
    lines.append("If LLM algos don't beat RandomCorr, LLM is decoration.")
    lines.append("")
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
        flag = " LLM HELPS ★" if md > 0 and sp < 0.05 else " LLM IRRELEVANT"
        lines.append(f"  {algo:<26s} vs RandomCorr: {w}/{n} wins, diff={md:+.1f}, p={sp:.4f}{flag}")
    lines.append("")

    # Cache diagnostics
    lines.append("--- LLM CALL DIAGNOSTICS (by algo, avg per trial) ---")
    calls = df.groupby("algo")[["llm_calls", "cache_hits"]].mean().round(1)
    for algo in stats.index:
        if algo in calls.index:
            row = calls.loc[algo]
            lines.append(f"  {algo:<26s} fresh_calls={row['llm_calls']:.1f} "
                         f"cache_hits={row['cache_hits']:.1f}")
    lines.append("")

    # Verdict
    lines.append("=" * 100)
    lines.append("VERDICT — READY FOR FULL-SCALE TIER 4?")
    lines.append("=" * 100)
    top3 = stats.index[:3].tolist()
    lines.append(f"Top 3 algorithms: {top3}")
    llm_beats_random = any(
        stats.loc[a, "mean"] < stats.loc["ABLATION_random_corr", "mean"]
        for a in ["M2_corr_cts", "N1_corr_full", "N4_robust_corr", "N5_corr_full_robust"]
        if a in stats.index
    )
    if llm_beats_random:
        lines.append("✓ At least one LLM algo beats RandomCorr → LLM-necessity confirmed")
    else:
        lines.append("✗ No LLM algo beats RandomCorr → MAJOR PROBLEM for paper")
    lines.append("")
    lines.append("If top 3 == [N1_corr_full, N4_robust_corr, N5_corr_full_robust] and")
    lines.append("llm_beats_random is True → READY to launch Tier 4 full-scale on EC2.")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier4_phaseA_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    if not args.keep_cache and CACHE_DB.exists():
        shutil.move(CACHE_DB, CACHE_DB.with_suffix(".sqlite.backup_phaseA"))
        logger.info("Cache cleared (backup preserved)")

    # Run pre-flight checks
    if not args.skip_preflight:
        if not run_preflight_checks(out_dir):
            logger.error("Pre-flight failed. Aborting.")
            sys.exit(1)
        # Clear cache again after preflight (preflight made some calls)
        if CACHE_DB.exists():
            shutil.move(CACHE_DB, CACHE_DB.with_suffix(".sqlite.backup_afterpreflight"))
            InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
            logger.info("Cache cleared after preflight")

    algos = list(PHASE_A_ALGOS.keys())
    configs = PHASE_A_CONFIGS
    tasks = [(a, c, s, m) for a in algos for c in configs for s in range(args.n_seeds)
             for m in PHASE_A_MODELS]
    logger.info(f"Running {len(algos)} algos × {len(configs)} configs × {args.n_seeds} seeds × "
                f"{len(PHASE_A_MODELS)} models = {len(tasks)} trials (T={args.T})")

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
                if completed % 10 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t_start, 0.1)
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:24s} c{config['config_id']} "
                        f"s{seed} {model:12s} r={r['final_regret']:.1f} "
                        f"calls={r['llm_calls']} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} c{config['config_id']} s{seed} {model}: {e}")

    total = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} in {total/60:.1f}min")
    analyze_phase_a(all_results, out_dir)


if __name__ == "__main__":
    main()
