"""Tier 7 ADDON: run new variants on the SAME (config, seed) pairs as the
main Tier-7 validation run, so LLM calls cache-hit from the main run's cache
and cost is ~$0.

Low worker count (default 2) so it doesn't starve the main Tier-7 workers.

Variants run here are cache-friendly: ONE LLM call at t=30 + pure numpy logic.

    python3 tier7_addon.py --variants V13_kmeans_refine,V15_mixed_effect_ts \\
        --T 25000 --n-seeds 15 --workers 2
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

ALL_ALGOS = {**LONGHORIZON_ALGOS, **LONGHORIZON_V2_ALGOS, "CTS": ALL_ALGORITHMS["cts_baseline"]}


def generate_configs_same_as_tier7():
    """Identical to tier7_validation.generate_configs(config_set='both').

    Must match exactly for cache keys to align with the main Tier-7 run.
    """
    configs = []
    cid = 0
    for offsets, split in [
        ([("uniform", 0.20, 100), ("uniform", 0.10, 200),
          ("hard", 0.05, 300), ("staggered", 0.02, 400)], "original"),
        ([("uniform", 0.20, 500), ("uniform", 0.10, 600),
          ("hard", 0.05, 700), ("staggered", 0.02, 800)], "held_out"),
    ]:
        for d in [25, 50]:
            for m in [3, 5]:
                for gap_type, delta, seed_offset in offsets:
                    configs.append({
                        "config_id": cid,
                        "d": d, "m": m,
                        "gap_type": gap_type,
                        "delta_min": delta,
                        "env_seed": seed_offset + cid,
                        "split": split,
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
    optimal = np.argsort(means)[::-1][:m]
    return means, optimal, means[optimal].sum()


def parse_variant(spec):
    if "@" not in spec:
        return spec, spec, {}
    base, hp = spec.split("@", 1)
    kwargs, parts = {}, [base]
    for kv in hp.split(","):
        k, v = kv.split("=")
        try:
            v_p = float(v)
            v_p = int(v_p) if v_p.is_integer() else v_p
        except ValueError:
            v_p = v
        kwargs[k] = v_p
        parts.append(f"{k}{v_p}")
    return base, "_".join(parts), kwargs


def run_trial(spec, config, seed, T):
    base, name, extra = parse_variant(spec)
    trial_id = f"c{config['config_id']}_s{seed}_{name}"
    means, optimal, opt_r = build_env(config, seed)
    rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)
    AlgoClass = ALL_ALGOS[base]
    np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": np_seed, **extra}
    oracle = None
    if base != "CTS":
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=name,
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
        if (t + 1) % 50 == 0:
            curve.append(round(float(cum), 2))
    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}
    return {
        "trial_id": trial_id,
        "algo": name,
        "base_class": base,
        "model": MODEL,
        "config_id": config["config_id"],
        "split": config.get("split", "original"),
        "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"], "delta_min": config["delta_min"],
        "env_seed": config["env_seed"],
        "T": T, "final_regret": round(float(cum), 2),
        "regret_curve": curve, "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
        "hp": extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--T", type=int, default=25000)
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2,
                        help="Keep at 2-4 to not starve main Tier-7 run")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier7_addon_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    specs = [s.strip() for s in args.variants.split(",") if s.strip()]
    configs = generate_configs_same_as_tier7()
    tasks = [(v, c, s) for v in specs for c in configs for s in range(args.n_seeds)]
    import random
    random.seed(42); random.shuffle(tasks)

    # Write meta.json — dashboard consumes this
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "T": args.T, "n_seeds": args.n_seeds, "n_configs": len(configs),
            "model": MODEL, "variants": specs,
            "total_trials": len(tasks), "timestamp": ts,
            "note": "ADDON to main Tier-7 run, reusing its LLM cache.",
        }, f, indent=2)

    logger.info(f"{len(specs)} variants x {len(configs)} configs x {args.n_seeds} seeds = "
                f"{len(tasks)} trials at T={args.T}, workers={args.workers}")

    results = []
    t_start = time.time()
    raw_path = out_dir / "raw_trials.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_trial, v, c, s, args.T): (v, c, s) for v, c, s in tasks}
        with open(raw_path, "w") as f:
            done = 0
            for fut in as_completed(futures):
                v, c, s = futures[fut]
                try:
                    r = fut.result()
                    results.append(r); f.write(json.dumps(r) + "\n"); f.flush()
                    done += 1
                    if done % 25 == 0:
                        elapsed = time.time() - t_start
                        eta = (len(tasks) - done) / (done / elapsed) / 60
                        logger.info(f"[{done}/{len(tasks)}] {v} c{c['config_id']} s{s} "
                                    f"r={r['final_regret']:.1f} ETA={eta:.1f}min")
                except Exception as e:
                    logger.exception(f"Trial {v} c{c['config_id']} s{s} failed: {e}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": ts, "T": args.T, "n_seeds": args.n_seeds,
            "variants": specs, "n_configs": len(configs),
            "total_trials": len(results),
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
