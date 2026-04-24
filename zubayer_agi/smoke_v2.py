"""Fast smoke test for V8/V11/V13/V15 against V6 (current winner) + CTS.

T=8000, n_seeds=6, 4 representative configs (one per d×m): ~144 trials.
On 4 workers, ~10-15 minutes wall clock. Cheap LLM calls (~6 per algo per trial).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS
from longhorizon_variants import LONGHORIZON_ALGOS
from longhorizon_v2 import LONGHORIZON_V2_ALGOS
from oracle_instrumented import InstrumentedOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-5.4"

SMOKE_ALGOS = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "V6_edge_pruning": LONGHORIZON_ALGOS["V6_edge_pruning"],
    **LONGHORIZON_V2_ALGOS,
}
NEEDS_ORACLE = {k for k in SMOKE_ALGOS if k != "CTS"}


def representative_configs():
    """One config per (d, m) — 4 total."""
    return [
        {"config_id": 0,  "d": 25, "m": 3, "gap_type": "uniform", "delta_min": 0.10, "env_seed": 200},
        {"config_id": 7,  "d": 25, "m": 5, "gap_type": "staggered", "delta_min": 0.02, "env_seed": 407},
        {"config_id": 8,  "d": 50, "m": 3, "gap_type": "uniform", "delta_min": 0.20, "env_seed": 108},
        {"config_id": 14, "d": 50, "m": 5, "gap_type": "hard", "delta_min": 0.05, "env_seed": 314},
    ]


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
    optimal = np.argsort(means)[::-1][:m]
    return means, optimal, means[optimal].sum()


def run_trial(algo_name, config, seed, T):
    trial_id = f"c{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, opt_r = build_env(config, seed)
    rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)
    AlgoClass = SMOKE_ALGOS[algo_name]
    np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": np_seed}
    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=algo_name,
        )
        kwargs["oracle"] = oracle
    agent = AlgoClass(**kwargs)
    cum = 0.0
    curve = []
    t_start = time.time()
    for t in range(T):
        if oracle is not None:
            oracle.current_t = t
        sel = list(agent.select_arms())
        if len(sel) < config["m"]:
            used = set(sel)
            sel.extend([a for a in range(config["d"]) if a not in used][:config["m"] - len(sel)])
        sel = sel[:config["m"]]
        smeans = means[sel]
        rew = (rng.uniform(size=config["m"]) < smeans).astype(float)
        cum += opt_r - smeans.sum()
        agent.update(sel, rew.tolist())
        if (t + 1) % 100 == 0:
            curve.append(round(float(cum), 2))
    elapsed = time.time() - t_start
    return {
        "trial_id": trial_id, "algo": algo_name,
        "config_id": config["config_id"], "seed": seed,
        "d": config["d"], "m": config["m"],
        "T": T, "final_regret": round(float(cum), 2),
        "regret_curve": curve, "elapsed_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8000)
    parser.add_argument("--n-seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4,
                        help="Keep low to not starve the main Tier-7 run.")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"smoke_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    configs = representative_configs()
    algos = list(SMOKE_ALGOS.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    import random
    random.seed(7); random.shuffle(tasks)

    logger.info(f"{len(algos)} algos x {len(configs)} configs x {args.n_seeds} seeds = "
                f"{len(tasks)} trials at T={args.T}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_trial, a, c, s, args.T): (a, c, s) for a, c, s in tasks}
        with open(out_dir / "raw_trials.jsonl", "w") as f:
            done = 0
            for fut in as_completed(futures):
                a, c, s = futures[fut]
                try:
                    r = fut.result()
                    results.append(r); f.write(json.dumps(r) + "\n"); f.flush()
                    done += 1
                    if done % 10 == 0:
                        logger.info(f"[{done}/{len(tasks)}] {a} c{c['config_id']} s{s} r={r['final_regret']:.1f}")
                except Exception as e:
                    logger.exception(f"Trial {a} c{c['config_id']} s{s} failed: {e}")

    # Quick analysis
    import statistics
    from collections import defaultdict
    by_algo = defaultdict(list)
    times = defaultdict(list)
    for r in results:
        by_algo[r["algo"]].append(r["final_regret"])
        times[r["algo"]].append(r["elapsed_sec"])
    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS (T={}, n_seeds={})".format(args.T, args.n_seeds))
    print("=" * 80)
    print(f"{'algo':30s} {'mean':>8s} {'se':>7s} {'min':>7s} {'max':>7s} {'n':>4s} {'time/trial':>11s}")
    cts_mean = statistics.mean(by_algo["CTS"])
    for algo in sorted(by_algo, key=lambda a: statistics.mean(by_algo[a])):
        vs = by_algo[algo]
        m = statistics.mean(vs)
        se = statistics.stdev(vs)/len(vs)**0.5 if len(vs) > 1 else 0
        delta = (cts_mean - m) / cts_mean * 100
        t_mean = statistics.mean(times[algo])
        print(f"{algo:30s} {m:8.1f} {se:7.2f} {min(vs):7.1f} {max(vs):7.1f} {len(vs):4d} {t_mean:9.1f}s  Δ_CTS={delta:+.1f}%")

    # Paired vs V6
    print(f"\n{'='*80}\nPAIRED vs V6_edge_pruning (matched config,seed):\n{'='*80}")
    by_cs = defaultdict(dict)
    for r in results:
        by_cs[(r["config_id"], r["seed"])][r["algo"]] = r["final_regret"]
    for algo in sorted(by_algo):
        if algo == "V6_edge_pruning":
            continue
        diffs = [by_cs[k]["V6_edge_pruning"] - by_cs[k][algo]
                 for k in by_cs if "V6_edge_pruning" in by_cs[k] and algo in by_cs[k]]
        if not diffs:
            continue
        wins = sum(1 for d in diffs if d > 0)
        print(f"  {algo:30s} W/L={wins}/{len(diffs)-wins}  mean_adv_over_V6={statistics.mean(diffs):+7.1f}")


if __name__ == "__main__":
    main()
