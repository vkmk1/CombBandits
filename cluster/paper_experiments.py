#!/usr/bin/env python3
"""All remaining experiments needed for the ICML workshop paper.

Generates:
  results/benchmark_hero/gap_results.json    -- gap structure generalization
  results/ablation/sigma_sweep.json          -- gate scale ablation
  results/ablation/tinit_sweep.json          -- T_init ablation
  results/lowerbound/regret_curves.json      -- regret-vs-T for lowerbound illustration
"""
import sys, os, json, time, logging
from pathlib import Path
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
DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run_one(agent, env, oc, T, seeds, **agent_cfg):
    trial = bt.run_batched_trial(
        agent_name=agent, env_cfg=env, oracle_cfg=oc,
        T=T, n_seeds=seeds, device=DEVICE, log_interval=T*2,
        agent_config=agent_cfg or {},
    )
    regrets = [r["final_regret"] for r in trial]
    return {"mean": float(np.mean(regrets)), "std": float(np.std(regrets))}


def gap_structure_experiment(T=20000, seeds=30):
    """Table 2: generalization across gap structures."""
    logger.info("=== gap_structure_experiment ===")
    out = {}
    for gap_type, delta in [("uniform", 0.05), ("clustered", 0.05), ("graded", 0.02)]:
        env = {"type":"synthetic_bernoulli","d":100,"m":10,"gap_type":gap_type,"delta_min":delta}
        out[gap_type] = {}
        logger.info(f"  gap_type={gap_type}")
        for agent in ["cts", "pool_cts", "pool_cts_cg_sigma01"]:
            out[gap_type][agent] = {}
            for sname, oc in [("perfect",{"corruption_type":"uniform","epsilon":0.0,"K":3}),
                                ("uniform_0.3",{"corruption_type":"uniform","epsilon":0.3,"K":3}),
                                ("consistent_wrong",{"corruption_type":"consistent_wrong","epsilon":1.0,"K":3}),
                                ("adversarial_0.3",{"corruption_type":"adversarial","epsilon":0.3,"K":3})]:
                t0 = time.time()
                stats = run_one(agent, env, oc, T, seeds)
                stats["elapsed_s"] = round(time.time()-t0, 1)
                out[gap_type][agent][sname] = stats
                logger.info(f"    {gap_type}/{agent}/{sname}: {stats['mean']:.0f}±{stats['std']:.0f}")
    out_dir = RESULTS / "benchmark_hero"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gap_results.json", "w") as f:
        json.dump(out, f, indent=2)


def sigma_ablation(T=20000, seeds=30):
    """Figure: sigma sweep on consistent_wrong."""
    logger.info("=== sigma_ablation ===")
    sigmas = [0.05, 0.1, 0.2, 0.5]
    env = {"type":"synthetic_bernoulli","d":100,"m":10,"gap_type":"uniform","delta_min":0.05}
    oc = {"corruption_type":"consistent_wrong","epsilon":1.0,"K":3}
    means, stds = [], []
    for sig in sigmas:
        # Use the base BatchedPoolCTSCG with custom sigma via agent_config
        t0 = time.time()
        stats = run_one("pool_cts_cg", env, oc, T, seeds, sigma=sig, T_init=200)
        means.append(stats["mean"])
        stds.append(stats["std"])
        logger.info(f"  sigma={sig}: {stats['mean']:.0f}±{stats['std']:.0f}  ({time.time()-t0:.1f}s)")
    out_dir = RESULTS / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sigma_sweep.json", "w") as f:
        json.dump({"sigmas": sigmas, "mean": means, "std": stds}, f, indent=2)


def tinit_ablation(T=20000, seeds=30):
    """Figure: T_init sweep on perfect + consistent_wrong."""
    logger.info("=== tinit_ablation ===")
    T_inits = [50, 100, 200, 500, 1000]
    env = {"type":"synthetic_bernoulli","d":100,"m":10,"gap_type":"uniform","delta_min":0.05}
    perfect_means, perfect_stds = [], []
    cwrong_means, cwrong_stds = [], []
    for ti in T_inits:
        for label, oc, m_list, s_list in [
            ("perfect", {"corruption_type":"uniform","epsilon":0.0,"K":3}, perfect_means, perfect_stds),
            ("consistent_wrong", {"corruption_type":"consistent_wrong","epsilon":1.0,"K":3}, cwrong_means, cwrong_stds),
        ]:
            t0 = time.time()
            stats = run_one("pool_cts_cg", env, oc, T, seeds, T_init=ti, sigma=0.1)
            m_list.append(stats["mean"])
            s_list.append(stats["std"])
            logger.info(f"  T_init={ti}/{label}: {stats['mean']:.0f}±{stats['std']:.0f}  ({time.time()-t0:.1f}s)")
    out_dir = RESULTS / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tinit_sweep.json", "w") as f:
        json.dump({
            "T_inits": T_inits,
            "perfect": perfect_means, "perfect_std": perfect_stds,
            "consistent_wrong": cwrong_means, "consistent_wrong_std": cwrong_stds,
        }, f, indent=2)


def regret_vs_T_curves(seeds=30):
    """Figure: regret-vs-T for lowerbound illustration."""
    logger.info("=== regret_vs_T_curves ===")
    T_values = [2000, 5000, 10000, 20000, 30000]
    env = {"type":"synthetic_bernoulli","d":100,"m":10,"gap_type":"uniform","delta_min":0.05}
    oc = {"corruption_type":"consistent_wrong","epsilon":1.0,"K":3}
    out = {"T": T_values}
    for agent in ["cucb", "llm_cucb_at", "pool_cts_cg_sigma01"]:
        means = []
        for T in T_values:
            t0 = time.time()
            stats = run_one(agent, env, oc, T, seeds)
            means.append(stats["mean"])
            logger.info(f"  {agent}/T={T}: {stats['mean']:.0f}±{stats['std']:.0f}  ({time.time()-t0:.1f}s)")
        out[agent] = means
    out_dir = RESULTS / "lowerbound"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "regret_curves.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    start = time.time()
    gap_structure_experiment()
    sigma_ablation()
    tinit_ablation()
    regret_vs_T_curves()
    logger.info(f"\nALL DONE. Total: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
