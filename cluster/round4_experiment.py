#!/usr/bin/env python3
"""Round 4: Finalist candidates — pool_cts vs tuned pool_cts_ic variants.

After Round 3, pool_cts is the clear champion on reliable oracles (14-18x better
than CUCB) but fails catastrophically on consistent_wrong. Round 4 tests tuned
pool_cts_ic variants (init + agreement check) that try to detect consistent_wrong.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGENTS = [
    "cucb",
    "cts",
    "pool_cts",              # Round 3 champion
    "pool_cts_ic50",         # T_init=50, thresh=0.5
    "pool_cts_ic50_40",      # T_init=50, thresh=0.4
    "pool_cts_ic100",        # T_init=100, thresh=0.3
    "pool_cts_ic100_40",     # T_init=100, thresh=0.4 (leading candidate)
    "pool_cts_ic150_40",     # T_init=150, thresh=0.4
    "pool_cts_ic200",        # T_init=200, thresh=0.3
]

CONFIG = {
    "T": 30000,
    "n_seeds": 50,
    "env": {"type": "synthetic_bernoulli", "d": 100, "m": 10,
            "gap_type": "uniform", "delta_min": 0.05},
    "oracles": [
        {"name": "perfect",          "oracle": {"corruption_type": "uniform", "epsilon": 0.0, "K": 3}},
        {"name": "uniform_0.3",      "oracle": {"corruption_type": "uniform", "epsilon": 0.3, "K": 3}},
        {"name": "uniform_0.5",      "oracle": {"corruption_type": "uniform", "epsilon": 0.5, "K": 3}},
        {"name": "consistent_wrong", "oracle": {"corruption_type": "consistent_wrong", "epsilon": 1.0, "K": 3}},
        {"name": "adversarial_0.3",  "oracle": {"corruption_type": "adversarial", "epsilon": 0.3, "K": 3}},
    ],
}


def main():
    T, n_seeds, env_cfg = CONFIG["T"], CONFIG["n_seeds"], CONFIG["env"]
    logger.info(f"Round 4: T={T}, seeds={n_seeds}, device={DEVICE}")
    results = {}
    start = time.time()
    for scenario in CONFIG["oracles"]:
        sname, oc = scenario["name"], scenario["oracle"]
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
            regrets = [r["final_regret"] for r in trial]
            results[sname][name] = {
                "mean": float(np.mean(regrets)), "std": float(np.std(regrets)),
                "median": float(np.median(regrets)), "elapsed_s": round(time.time()-t0, 1),
            }
            logger.info(f"  {name:22s} | regret={np.mean(regrets):8.1f} ± {np.std(regrets):6.1f} | {time.time()-t0:.1f}s")

    logger.info(f"\nTotal: {time.time()-start:.1f}s")

    print("\n" + "=" * 120)
    print("ROUND 4 RESULTS")
    print("=" * 120)
    header = f"{'Agent':22s}  " + "  ".join(f"{s['name']:18s}" for s in CONFIG["oracles"])
    print(header)
    print("-" * len(header))
    for name in AGENTS:
        row = f"{name:22s}  "
        for scenario in CONFIG["oracles"]:
            stats = results[scenario["name"]].get(name, {})
            row += f"{stats.get('mean', 0):8.1f} ± {stats.get('std', 0):6.1f}  " if "mean" in stats else f"{'ERROR':18s}  "
        print(row)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'round4')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'round4_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
