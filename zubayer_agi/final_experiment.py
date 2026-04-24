"""Final experiment: masterpieces (M1-M4) vs best candidates (B2, F2, Cal-Gated) vs CTS.

Uses deterministic per-trial np_seed for true paired comparison.
5 configs × 5 seeds × 8 algorithms × T=1500.
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

from algorithms import ALL_ALGORITHMS
from paper_baselines import PAPER_BASELINES
from safe_algorithms import SAFE_ALGORITHMS
from masterpiece_algorithms import MASTERPIECE_ALGOS
from oracle import GPTOracle
from bulletproof_experiment import generate_bulletproof_configs, build_env, paired_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ALL = {
    # Baseline
    "cts_baseline": ALL_ALGORITHMS["cts_baseline"],
    # Current champion + strong baselines
    "CHAMP_B2_icpd": ALL_ALGORITHMS["B2_icpd_cts"],
    "OURS_F2_query_design": ALL_ALGORITHMS["F2_query_design_cts"],
    "PAPER_calibration_gated": PAPER_BASELINES["PAPER_calibration_gated"],
    # Masterpieces
    "M1_b2_plus": MASTERPIECE_ALGOS["M1_b2_plus"],
    "M2_correlated_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    "M3_b2_correlated": MASTERPIECE_ALGOS["M3_b2_correlated"],
    "M4_b2_patched": MASTERPIECE_ALGOS["M4_b2_patched"],
}
NEEDS_ORACLE = {k for k in ALL if k != "cts_baseline"}


def run_single_trial(algo_name: str, config: dict, seed: int, T: int) -> dict:
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = ALL[algo_name]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": trial_np_seed}

    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = GPTOracle(d=config["d"], m=config["m"])
        kwargs["oracle"] = oracle

    # Algos with T_horizon param
    if algo_name in ("M1_b2_plus", "M3_b2_correlated"):
        kwargs["T_horizon"] = T

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    configs = generate_bulletproof_configs()
    algos = list(ALL.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    logger.info(f"Running {len(algos)} algorithms × {len(configs)} configs × "
                f"{args.n_seeds} seeds = {len(tasks)} trials (T={args.T})")

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
                rate = completed / max(time.time() - t_start, 0.1)
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                if completed % 10 == 0 or completed == len(tasks):
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:28s} cfg={config['config_id']} "
                        f"seed={seed} regret={r['final_regret']:.1f} "
                        f"ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} cfg={config['config_id']} seed={seed}: {e}")
                all_results.append({
                    "algo": algo, "config_id": config["config_id"], "seed": seed,
                    "final_regret": None, "error": str(e),
                })

    total = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} in {total/60:.1f}min")

    out_dir = Path(__file__).parent / "results"
    paired_analysis(all_results, out_dir)


if __name__ == "__main__":
    main()
