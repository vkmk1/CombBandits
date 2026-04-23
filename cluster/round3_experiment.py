#!/usr/bin/env python3
"""Round 3: Test refined variants based on Round 2 learnings.

Key lessons from Round 2:
- pool_restrict is the strongest winner (2075 @ perfect, 1900 @ uniform_0.2)
- div_trust fails under perfect oracle (6874 — over-conservative)
- combined is worse than CUCB (three taxes compound)
- Meta-BoBW has a small but noticeable overhead

Round 3 tests: fixed divergence-trust, pool+trust hybrid, pool+CTS, warm-started meta-BoBW.
"""
import sys
import os
import json
import time
import logging

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from combbandits.gpu.batched_agents import BATCHED_AGENT_REGISTRY, NEEDS_ORACLE
from combbandits.gpu.batched_variants import VARIANT_REGISTRY

ALL_REGISTRY = {**BATCHED_AGENT_REGISTRY, **VARIANT_REGISTRY}
ALL_NEEDS_ORACLE = NEEDS_ORACLE | set(VARIANT_REGISTRY.keys())

import combbandits.gpu.batched_trial as bt
bt.BATCHED_AGENT_REGISTRY = ALL_REGISTRY
bt.NEEDS_ORACLE = ALL_NEEDS_ORACLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGENTS = [
    "cucb",             # baseline
    "cts",              # strong baseline
    "llm_cucb_at",      # original
    "pool_restrict",    # Round 2 winner (on reliable oracle)
    "div_trust",        # Round 2: works on consistent_wrong only
    "div_trust_v2",     # Round 3: fixed divergence (should work BOTH)
    "pool_with_trust",  # Round 3: pool + expansion monitor
    "pool_cts",         # Round 3: pool + Thompson
    "adaptive_pool",    # Round 3: pool + probe-based corruption detector
    "meta_bobw_warm",   # Round 3: warm-started BoBW
]

CONFIG = {
    "T": 30000,
    "n_seeds": 50,
    "env": {"type": "synthetic_bernoulli", "d": 100, "m": 10,
            "gap_type": "uniform", "delta_min": 0.05},
    "oracles": [
        {"name": "perfect",          "oracle": {"corruption_type": "uniform", "epsilon": 0.0, "K": 3}},
        {"name": "uniform_0.1",      "oracle": {"corruption_type": "uniform", "epsilon": 0.1, "K": 3}},
        {"name": "uniform_0.3",      "oracle": {"corruption_type": "uniform", "epsilon": 0.3, "K": 3}},
        {"name": "uniform_0.5",      "oracle": {"corruption_type": "uniform", "epsilon": 0.5, "K": 3}},
        {"name": "consistent_wrong", "oracle": {"corruption_type": "consistent_wrong", "epsilon": 1.0, "K": 3}},
        {"name": "adversarial_0.3",  "oracle": {"corruption_type": "adversarial", "epsilon": 0.3, "K": 3}},
        {"name": "partial_0.3",      "oracle": {"corruption_type": "partial_overlap", "epsilon": 0.3, "K": 3}},
    ],
}


def main():
    T, n_seeds, env_cfg = CONFIG["T"], CONFIG["n_seeds"], CONFIG["env"]
    logger.info(f"Round 3: T={T}, seeds={n_seeds}, device={DEVICE}")

    results = {}
    start = time.time()

    for scenario in CONFIG["oracles"]:
        sname = scenario["name"]
        oc = scenario["oracle"]
        results[sname] = {}
        logger.info(f"\n=== {sname} ({oc['corruption_type']} eps={oc['epsilon']}) ===")

        for name in AGENTS:
            t0 = time.time()
            try:
                trial = bt.run_batched_trial(
                    agent_name=name, env_cfg=env_cfg, oracle_cfg=oc,
                    T=T, n_seeds=n_seeds, device=DEVICE, log_interval=T,
                )
            except Exception as e:
                logger.error(f"  {name}: ERROR {type(e).__name__}: {e}")
                results[sname][name] = {"error": str(e)}
                continue
            elapsed = time.time() - t0
            regrets = [r["final_regret"] for r in trial]
            results[sname][name] = {
                "mean": float(np.mean(regrets)),
                "std": float(np.std(regrets)),
                "median": float(np.median(regrets)),
                "elapsed_s": round(elapsed, 1),
            }
            logger.info(f"  {name:18s} | regret={np.mean(regrets):8.1f} ± {np.std(regrets):6.1f} | {elapsed:.1f}s")

    total = time.time() - start
    logger.info(f"\nTotal: {total:.1f}s")

    print("\n" + "=" * 120)
    print("ROUND 3 RESULTS")
    print("=" * 120)
    header = f"{'Agent':18s}  " + "  ".join(f"{s['name']:16s}" for s in CONFIG["oracles"])
    print(header)
    print("-" * len(header))
    for name in AGENTS:
        row = f"{name:18s}  "
        for scenario in CONFIG["oracles"]:
            stats = results[scenario["name"]].get(name, {})
            row += f"{stats.get('mean', 0):8.1f}        " if "mean" in stats else f"{'ERROR':16s}  "
        print(row)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'round3')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'round3_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
