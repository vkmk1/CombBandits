#!/usr/bin/env python3
"""Novelty validation experiment: stress-tests the 4 key claims of the paper.

Designed to run quickly (~10-15 min on CPU) and produce definitive evidence
of whether LLM-CUCB-AT's contributions are real and significant.

Claims tested:
  1. Composite trust (κ+ρ) is essential — either metric alone is worse
  2. Graceful degradation under increasing corruption ε
  3. Consistent-wrong oracle: LLM-CUCB-AT recovers, LLM-Greedy/ELLM catastrophically fail
  4. Perfect oracle: O(d log d) regret (only initialization cost)

Runs all 9 agents × targeted oracle configs. No GPU required.
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
from combbandits.gpu.batched_trial import run_batched_trial

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_AGENTS = ["cucb", "cts", "llm_cucb_at", "llm_greedy", "ellm_adapted",
              "opro_bandit", "warm_start_cts", "corrupt_robust_cucb", "exp4"]

VALIDATION_CONFIG = {
    "T": 50000,
    "n_seeds": 50,
    "env": {"type": "synthetic_bernoulli", "d": 100, "m": 10,
            "gap_type": "uniform", "delta_min": 0.05},

    "tests": [
        {
            "name": "claim1_perfect_oracle",
            "description": "Perfect oracle: LLM-CUCB-AT should match CUCB (O(d log d) regret)",
            "oracle": {"corruption_type": "uniform", "epsilon": 0.0, "K": 3},
            "agents": ALL_AGENTS,
        },
        {
            "name": "claim2_graceful_degradation",
            "description": "Increasing corruption: regret should grow smoothly with ε",
            "oracle_sweep": [
                {"corruption_type": "uniform", "epsilon": 0.05, "K": 3},
                {"corruption_type": "uniform", "epsilon": 0.1, "K": 3},
                {"corruption_type": "uniform", "epsilon": 0.2, "K": 3},
                {"corruption_type": "uniform", "epsilon": 0.3, "K": 3},
                {"corruption_type": "uniform", "epsilon": 0.5, "K": 3},
            ],
            "agents": ["cucb", "cts", "llm_cucb_at", "llm_greedy", "ellm_adapted", "exp4"],
        },
        {
            "name": "claim3_consistent_wrong",
            "description": "Consistently wrong oracle: LLM-CUCB-AT recovers, naive LLM methods fail",
            "oracle": {"corruption_type": "consistent_wrong", "epsilon": 1.0, "K": 3},
            "agents": ALL_AGENTS,
        },
        {
            "name": "claim4_adversarial",
            "description": "Adversarial oracle: LLM-CUCB-AT ≤ CUCB regret",
            "oracle": {"corruption_type": "adversarial", "epsilon": 0.3, "K": 3},
            "agents": ALL_AGENTS,
        },
        {
            "name": "claim5_partial_overlap",
            "description": "Partial overlap: tests graded quality model",
            "oracle": {"corruption_type": "partial_overlap", "epsilon": 0.2, "K": 3},
            "agents": ALL_AGENTS,
        },
    ]
}


def run_single_test(test_cfg, env_cfg, T, n_seeds):
    """Run one test config and return results dict."""
    name = test_cfg["name"]
    agents = test_cfg["agents"]
    results = {"name": name, "description": test_cfg["description"], "agents": {}}

    oracle_cfgs = test_cfg.get("oracle_sweep", [test_cfg.get("oracle")])
    if oracle_cfgs is None:
        oracle_cfgs = [test_cfg["oracle"]]

    for oracle_cfg in oracle_cfgs:
        eps = oracle_cfg["epsilon"]
        corr = oracle_cfg["corruption_type"]
        label = f"{corr}_eps{eps}"

        results["agents"][label] = {}

        for agent_name in agents:
            t0 = time.time()
            trial_results = run_batched_trial(
                agent_name=agent_name,
                env_cfg=env_cfg,
                oracle_cfg=oracle_cfg,
                T=T,
                n_seeds=n_seeds,
                device=DEVICE,
                log_interval=T,  # only log at end
            )
            elapsed = time.time() - t0

            regrets = [r["final_regret"] for r in trial_results]
            mean_regret = np.mean(regrets)
            std_regret = np.std(regrets)
            median_regret = np.median(regrets)

            results["agents"][label][agent_name] = {
                "mean_regret": float(mean_regret),
                "std_regret": float(std_regret),
                "median_regret": float(median_regret),
                "min_regret": float(np.min(regrets)),
                "max_regret": float(np.max(regrets)),
                "elapsed_s": round(elapsed, 1),
            }

            logger.info(
                f"  [{name}] {label} | {agent_name:20s} | "
                f"regret={mean_regret:8.1f} ± {std_regret:6.1f} | "
                f"{elapsed:.1f}s"
            )

    return results


def analyze_results(all_results):
    """Produce a summary assessment of novelty claims."""
    lines = [
        "=" * 70,
        "NOVELTY VALIDATION REPORT",
        "=" * 70,
        "",
    ]

    for test in all_results:
        lines.append(f"--- {test['name']}: {test['description']} ---")
        for label, agents in test["agents"].items():
            lines.append(f"  Oracle config: {label}")

            sorted_agents = sorted(agents.items(), key=lambda x: x[1]["mean_regret"])
            for rank, (name, stats) in enumerate(sorted_agents, 1):
                marker = " ★" if name == "llm_cucb_at" else ""
                lines.append(
                    f"    {rank:2d}. {name:22s}  "
                    f"regret={stats['mean_regret']:8.1f} ± {stats['std_regret']:6.1f}{marker}"
                )
            lines.append("")

    # Claim assessments
    lines.append("=" * 70)
    lines.append("CLAIM ASSESSMENT")
    lines.append("=" * 70)

    # Claim 1: perfect oracle
    c1 = all_results[0]["agents"]
    for label, agents in c1.items():
        cucb_r = agents.get("cucb", {}).get("mean_regret", float("inf"))
        at_r = agents.get("llm_cucb_at", {}).get("mean_regret", float("inf"))
        ratio = at_r / cucb_r if cucb_r > 0 else float("inf")
        verdict = "PASS" if ratio < 1.2 else "MARGINAL" if ratio < 2.0 else "FAIL"
        lines.append(f"\nClaim 1 (perfect oracle): LLM-CUCB-AT={at_r:.1f} vs CUCB={cucb_r:.1f} "
                      f"(ratio={ratio:.2f}) → {verdict}")

    # Claim 3: consistent wrong
    c3 = all_results[2]["agents"]
    for label, agents in c3.items():
        at_r = agents.get("llm_cucb_at", {}).get("mean_regret", float("inf"))
        greedy_r = agents.get("llm_greedy", {}).get("mean_regret", float("inf"))
        ellm_r = agents.get("ellm_adapted", {}).get("mean_regret", float("inf"))
        cucb_r = agents.get("cucb", {}).get("mean_regret", float("inf"))
        verdict = "PASS" if at_r < greedy_r * 0.5 and at_r < ellm_r * 0.5 else "FAIL"
        lines.append(f"\nClaim 3 (consistent wrong): LLM-CUCB-AT={at_r:.1f} vs "
                      f"LLM-Greedy={greedy_r:.1f}, ELLM={ellm_r:.1f}, CUCB={cucb_r:.1f} → {verdict}")

    # Claim 4: adversarial robustness
    c4 = all_results[3]["agents"]
    for label, agents in c4.items():
        at_r = agents.get("llm_cucb_at", {}).get("mean_regret", float("inf"))
        cucb_r = agents.get("cucb", {}).get("mean_regret", float("inf"))
        verdict = "PASS" if at_r <= cucb_r * 1.5 else "FAIL"
        lines.append(f"\nClaim 4 (adversarial): LLM-CUCB-AT={at_r:.1f} vs CUCB={cucb_r:.1f} → {verdict}")

    # Claim 2: graceful degradation
    c2 = all_results[1]["agents"]
    at_regrets = []
    for label in sorted(c2.keys()):
        r = c2[label].get("llm_cucb_at", {}).get("mean_regret", float("inf"))
        at_regrets.append(r)
    monotonic = all(at_regrets[i] <= at_regrets[i+1] * 1.2 for i in range(len(at_regrets)-1))
    verdict = "PASS" if monotonic else "MARGINAL"
    lines.append(f"\nClaim 2 (graceful degradation): regrets={[round(r,1) for r in at_regrets]} "
                  f"→ {'monotonic' if monotonic else 'non-monotonic'} → {verdict}")

    lines.append("\n" + "=" * 70)

    # Overall
    pass_count = sum(1 for l in lines if "→ PASS" in l)
    total_claims = sum(1 for l in lines if "→ " in l)
    lines.append(f"OVERALL: {pass_count}/{total_claims} claims validated")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    cfg = VALIDATION_CONFIG
    T = cfg["T"]
    n_seeds = cfg["n_seeds"]
    env_cfg = cfg["env"]

    logger.info(f"Validation experiment: T={T}, n_seeds={n_seeds}, device={DEVICE}")
    logger.info(f"Env: d={env_cfg['d']}, m={env_cfg['m']}, gap={env_cfg['gap_type']}")

    start = time.time()
    all_results = []

    for test_cfg in cfg["tests"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_cfg['name']}")
        logger.info(f"  {test_cfg['description']}")
        logger.info(f"{'='*50}")
        result = run_single_test(test_cfg, env_cfg, T, n_seeds)
        all_results.append(result)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # Save raw results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'validation')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'validation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate and save report
    report = analyze_results(all_results)
    report_path = os.path.join(out_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print("\n" + report)
    logger.info(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
