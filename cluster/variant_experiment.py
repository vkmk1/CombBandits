#!/usr/bin/env python3
"""Round 2: Test algorithmic variants against baselines + original LLM-CUCB-AT.

Quick-iteration harness: T=30000, 50 seeds, 5 canonical scenarios, 7 agents.
Targets ~15-20 min on CPU, much faster on GPU.
"""
import sys
import os
import json
import time
import logging

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from combbandits.gpu.batched_env import build_batched_env
from combbandits.gpu.batched_oracle import BatchedSimulatedCLO
from combbandits.gpu.batched_agents import BATCHED_AGENT_REGISTRY, NEEDS_ORACLE
from combbandits.gpu.batched_variants import VARIANT_REGISTRY
from combbandits.gpu.batched_trial import run_batched_trial

# Combine registries
ALL_REGISTRY = {**BATCHED_AGENT_REGISTRY, **VARIANT_REGISTRY}
ALL_NEEDS_ORACLE = NEEDS_ORACLE | set(VARIANT_REGISTRY.keys())

# Monkey-patch the registry used by run_batched_trial
import combbandits.gpu.batched_trial as bt
bt.BATCHED_AGENT_REGISTRY = ALL_REGISTRY
bt.NEEDS_ORACLE = ALL_NEEDS_ORACLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baselines + our variants
AGENTS_TO_TEST = [
    "cucb",             # baseline
    "cts",              # strong baseline
    "llm_cucb_at",      # original (broken under consistent_wrong)
    "meta_bobw",        # V1
    "explore_floor",    # V3
    "pool_restrict",    # V5
    "div_trust",        # V2
    "epoch_robust",     # V4
    "combined",         # V7
]

CONFIG = {
    "T": 30000,
    "n_seeds": 50,
    "env": {"type": "synthetic_bernoulli", "d": 100, "m": 10,
            "gap_type": "uniform", "delta_min": 0.05},

    "oracles": [
        {"name": "perfect",          "oracle": {"corruption_type": "uniform", "epsilon": 0.0, "K": 3}},
        {"name": "uniform_0.2",      "oracle": {"corruption_type": "uniform", "epsilon": 0.2, "K": 3}},
        {"name": "uniform_0.5",      "oracle": {"corruption_type": "uniform", "epsilon": 0.5, "K": 3}},
        {"name": "consistent_wrong", "oracle": {"corruption_type": "consistent_wrong", "epsilon": 1.0, "K": 3}},
        {"name": "adversarial_0.3",  "oracle": {"corruption_type": "adversarial", "epsilon": 0.3, "K": 3}},
    ],
}


def main():
    T = CONFIG["T"]
    n_seeds = CONFIG["n_seeds"]
    env_cfg = CONFIG["env"]

    logger.info(f"Round 2 variants: T={T}, seeds={n_seeds}, device={DEVICE}")
    logger.info(f"Env: d={env_cfg['d']}, m={env_cfg['m']}")
    logger.info(f"Agents: {AGENTS_TO_TEST}")

    start = time.time()
    results = {}  # {scenario_name: {agent_name: stats}}

    for scenario in CONFIG["oracles"]:
        sname = scenario["name"]
        oracle_cfg = scenario["oracle"]
        results[sname] = {}
        logger.info(f"\n=== {sname} ({oracle_cfg['corruption_type']} eps={oracle_cfg['epsilon']}) ===")

        for agent_name in AGENTS_TO_TEST:
            t0 = time.time()
            try:
                trial = run_batched_trial(
                    agent_name=agent_name,
                    env_cfg=env_cfg,
                    oracle_cfg=oracle_cfg,
                    T=T,
                    n_seeds=n_seeds,
                    device=DEVICE,
                    log_interval=T,
                )
            except Exception as e:
                logger.error(f"  {agent_name}: FAILED with {type(e).__name__}: {e}")
                results[sname][agent_name] = {"error": str(e)}
                continue

            elapsed = time.time() - t0
            regrets = [r["final_regret"] for r in trial]
            results[sname][agent_name] = {
                "mean": float(np.mean(regrets)),
                "std": float(np.std(regrets)),
                "median": float(np.median(regrets)),
                "elapsed_s": round(elapsed, 1),
            }
            logger.info(f"  {agent_name:18s} | regret={np.mean(regrets):8.1f} ± {np.std(regrets):6.1f} | {elapsed:.1f}s")

    total = time.time() - start
    logger.info(f"\nTotal: {total:.1f}s")

    # Summary table
    print("\n" + "=" * 100)
    print("ROUND 2 RESULTS — Variants vs Baselines (final regret, mean)")
    print("=" * 100)
    header = f"{'Agent':20s}  " + "  ".join(f"{s['name']:18s}" for s in CONFIG["oracles"])
    print(header)
    print("-" * len(header))
    for agent_name in AGENTS_TO_TEST:
        row = f"{agent_name:20s}  "
        for scenario in CONFIG["oracles"]:
            stats = results[scenario["name"]].get(agent_name, {})
            if "error" in stats:
                row += f"{'ERROR':18s}  "
            else:
                row += f"{stats['mean']:8.1f} ± {stats['std']:6.1f}  "
        print(row)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'round2')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'variant_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_dir}/variant_results.json")


if __name__ == "__main__":
    main()
