"""Tier 2.5: Head-to-head — CORR-CTS vs 4 new breakthrough algorithms.

Compares current champion CORR-CTS against the 4 breakthrough directions
from Sonnet 4.6 research:
- N1 CORR-CTS-Full (full covariance, kernel-based)
- N2 HypoTS (K-hypothesis mixture)
- N3 InfoMin-CTS (info-budgeted querying)
- N4 Robust-CORR (credibility-gated)

Uses same 3-model split as Tier 2 for consistency:
- gpt-4.1-mini: configs 0-3
- gpt-5-mini: configs 4-7
- gpt-5.4: configs 8-11

If any N# beats CORR-CTS significantly → new champion → paper's main algorithm.
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
from masterpiece_algorithms import MASTERPIECE_ALGOS
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from oracle_instrumented import InstrumentedOracle
from tier2_experiment import (
    generate_tier2_configs, config_to_model, build_env, analyze_and_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TIER2_5_ALGOS = {
    "cts_baseline": ALL_ALGORITHMS["cts_baseline"],
    "CHAMP_M2_corr_cts": MASTERPIECE_ALGOS["M2_correlated_cts"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "N2_hypo_ts": BREAKTHROUGH_ALGOS["N2_hypo_ts"],
    "N3_info_min": BREAKTHROUGH_ALGOS["N3_info_min"],
    "N4_robust_corr": BREAKTHROUGH_ALGOS["N4_robust_corr"],
}
NEEDS_ORACLE = {k for k in TIER2_5_ALGOS if k != "cts_baseline"}


def run_single_trial(algo_name, config, seed, T, out_dir):
    model = config_to_model(config["config_id"])
    trial_id = f"cfg{config['config_id']}_s{seed}_{algo_name}"

    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER2_5_ALGOS[algo_name]
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

    if algo_name == "CHAMP_M2_corr_cts":
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
        "optimal_reward_per_round": round(float(optimal_reward), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier25_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    configs = generate_tier2_configs()
    algos = list(TIER2_5_ALGOS.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    logger.info(f"Running {len(algos)} algos × {len(configs)} configs × {args.n_seeds} seeds "
                f"= {len(tasks)} trials (T={args.T})")

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
                if completed % 10 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t_start, 0.1)
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:22s} cfg={config['config_id']} "
                        f"seed={seed} model={config_to_model(config['config_id'])} "
                        f"regret={r['final_regret']:.1f} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} cfg={config['config_id']} seed={seed}: {e}")

    logger.info(f"Done: {completed}/{len(tasks)} in {(time.time()-t_start)/60:.1f}min")
    analyze_and_report(all_results, out_dir)


if __name__ == "__main__":
    main()
