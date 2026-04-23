#!/usr/bin/env python3
"""Round 5 — local fast version.

Only the agents that actually matter:
  - cucb (baseline)
  - cts (strong baseline)
  - pool_cts (our champion on reliable)
  - pool_cts_ic1600 (statistically-sufficient init for consistent_wrong detection)
  - pool_cts_etc (alternative robust strategy)

Tests all 5 corruption scenarios at T=30,000, 50 seeds. ~10 min on Mac CPU.
"""
import sys, os, json, time, logging
import torch, numpy as np

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
DEVICE = torch.device("cpu")  # forced CPU for stability

AGENTS = ["cucb", "cts", "pool_cts", "pool_cts_ic1600", "pool_cts_etc"]
T = 30000
N_SEEDS = 50
ENV = {"type": "synthetic_bernoulli", "d": 100, "m": 10, "gap_type": "uniform", "delta_min": 0.05}
SCENARIOS = [
    ("perfect",          {"corruption_type": "uniform", "epsilon": 0.0, "K": 3}),
    ("uniform_0.3",      {"corruption_type": "uniform", "epsilon": 0.3, "K": 3}),
    ("uniform_0.5",      {"corruption_type": "uniform", "epsilon": 0.5, "K": 3}),
    ("consistent_wrong", {"corruption_type": "consistent_wrong", "epsilon": 1.0, "K": 3}),
    ("adversarial_0.3",  {"corruption_type": "adversarial", "epsilon": 0.3, "K": 3}),
]


def main():
    logger.info(f"Round 5 local: T={T}, seeds={N_SEEDS}, agents={AGENTS}")
    results = {}
    start = time.time()
    for sname, oc in SCENARIOS:
        results[sname] = {}
        logger.info(f"\n=== {sname} ===")
        for name in AGENTS:
            t0 = time.time()
            kwargs = {"T_total": T} if name == "pool_cts_etc" else {}
            try:
                trial = bt.run_batched_trial(
                    agent_name=name, env_cfg=ENV, oracle_cfg=oc,
                    T=T, n_seeds=N_SEEDS, device=DEVICE, log_interval=T,
                    agent_config=kwargs,
                )
            except Exception as e:
                logger.error(f"  {name}: ERROR {e}")
                continue
            regrets = [r["final_regret"] for r in trial]
            results[sname][name] = {"mean": float(np.mean(regrets)), "std": float(np.std(regrets)),
                                     "elapsed_s": round(time.time()-t0, 1)}
            logger.info(f"  {name:22s} | regret={np.mean(regrets):8.1f} ± {np.std(regrets):6.1f} | {time.time()-t0:.1f}s")

    print("\n" + "=" * 110)
    print("ROUND 5 LOCAL RESULTS — Champion Selection")
    print("=" * 110)
    header = f"{'Agent':22s}  " + "  ".join(f"{s:18s}" for s, _ in SCENARIOS)
    print(header)
    print("-" * len(header))
    for name in AGENTS:
        row = f"{name:22s}  "
        for sname, _ in SCENARIOS:
            stats = results[sname].get(name, {})
            row += f"{stats.get('mean', 0):8.1f} ± {stats.get('std', 0):6.1f}  " if "mean" in stats else f"{'ERROR':18s}  "
        print(row)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'round5_local')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'round5_local_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Total: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
