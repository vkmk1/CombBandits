"""Metrics computation for experiment results with statistical rigor."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> pd.DataFrame:
    """Aggregate trial results into summary statistics with confidence intervals.

    Groups by (agent, env, corruption_type, epsilon) and computes
    mean, std, SE, 95% CI, and median of final regret across seeds.
    """
    rows = []
    for r in results:
        regret_curve = np.array(r["regret_curve"])
        rows.append({
            "agent": r["agent"],
            "env": r["env"],
            "corruption_type": r["corruption_type"],
            "epsilon": r["epsilon"],
            "seed": r["seed"],
            "d": r["d"],
            "m": r["m"],
            "T": r["T"],
            "final_regret": r["final_regret"],
            "regret_at_quarter": regret_curve[len(regret_curve)//4] if len(regret_curve) > 3 else 0,
            "regret_at_half": regret_curve[len(regret_curve)//2] if len(regret_curve) > 1 else 0,
            "oracle_queries": r.get("oracle_queries", 0),
            "oracle_tokens": r.get("oracle_tokens", 0),
        })

    df = pd.DataFrame(rows)

    group_cols = ["agent", "env", "corruption_type", "epsilon", "d", "m", "T"]
    summary = df.groupby(group_cols).agg(
        mean_regret=("final_regret", "mean"),
        median_regret=("final_regret", "median"),
        std_regret=("final_regret", "std"),
        se_regret=("final_regret", lambda x: x.std() / np.sqrt(len(x))),
        ci95_lower=("final_regret", lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
        ci95_upper=("final_regret", lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x))),
        n_seeds=("seed", "count"),
        mean_queries=("oracle_queries", "mean"),
        mean_tokens=("oracle_tokens", "mean"),
    ).reset_index()

    return summary


def pairwise_significance(results: list[dict], agent_a: str, agent_b: str,
                          corruption_type: str | None = None,
                          epsilon: float | None = None) -> dict:
    """Wilcoxon signed-rank test comparing two agents across seeds.

    Returns test statistic, p-value, effect size (Cohen's d), and verdict.
    """
    regrets_a = []
    regrets_b = []

    # Group by seed
    by_seed_a = {}
    by_seed_b = {}
    for r in results:
        if corruption_type and r["corruption_type"] != corruption_type:
            continue
        if epsilon is not None and abs(r["epsilon"] - epsilon) > 1e-6:
            continue
        if r["agent"] == agent_a:
            by_seed_a[r["seed"]] = r["final_regret"]
        elif r["agent"] == agent_b:
            by_seed_b[r["seed"]] = r["final_regret"]

    common_seeds = sorted(set(by_seed_a.keys()) & set(by_seed_b.keys()))
    if len(common_seeds) < 5:
        return {"error": f"Too few common seeds ({len(common_seeds)}). Need >= 5."}

    regrets_a = [by_seed_a[s] for s in common_seeds]
    regrets_b = [by_seed_b[s] for s in common_seeds]

    # Wilcoxon signed-rank test (paired, non-parametric)
    stat, p_value = stats.wilcoxon(regrets_a, regrets_b, alternative="two-sided")

    # Effect size: Cohen's d
    diffs = np.array(regrets_a) - np.array(regrets_b)
    cohens_d = diffs.mean() / (diffs.std() + 1e-10)

    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "n_seeds": len(common_seeds),
        "mean_a": np.mean(regrets_a),
        "mean_b": np.mean(regrets_b),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
        "winner": agent_a if np.mean(regrets_a) < np.mean(regrets_b) else agent_b,
    }


def regret_curves_by_agent(results: list[dict]) -> dict[str, dict]:
    """Extract mean regret curves grouped by agent.

    Returns dict mapping label -> {mean, std, se, ci95_lower, ci95_upper} arrays.
    """
    curves = defaultdict(list)
    for r in results:
        key = (r["agent"], r["corruption_type"], r["epsilon"])
        curves[key].append(np.array(r["regret_curve"]))

    out = {}
    for key, arrs in curves.items():
        min_len = min(len(a) for a in arrs)
        stacked = np.stack([a[:min_len] for a in arrs])
        n = len(arrs)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        se = std / np.sqrt(n)
        label = f"{key[0]}_corr={key[1]}_eps={key[2]}"
        out[label] = {
            "mean": mean,
            "std": std,
            "se": se,
            "ci95_lower": mean - 1.96 * se,
            "ci95_upper": mean + 1.96 * se,
            "n_seeds": n,
        }
    return out


def regret_vs_epsilon(results: list[dict], agent: str,
                      corruption_type: str = "uniform") -> pd.DataFrame:
    """Extract final regret as a function of epsilon for theory validation.

    For verifying that regret scales as O(epsilon * m * T / Delta_min).
    """
    rows = []
    for r in results:
        if r["agent"] != agent or r["corruption_type"] != corruption_type:
            continue
        rows.append({
            "epsilon": r["epsilon"],
            "final_regret": r["final_regret"],
            "seed": r["seed"],
            "d": r["d"],
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = df.groupby(["epsilon", "d"]).agg(
        mean_regret=("final_regret", "mean"),
        se_regret=("final_regret", lambda x: x.std() / np.sqrt(len(x))),
        n_seeds=("seed", "count"),
    ).reset_index()
    return summary


def regret_vs_dimension(results: list[dict], agent: str,
                        corruption_type: str = "uniform",
                        epsilon: float = 0.1) -> pd.DataFrame:
    """Extract final regret as a function of d for dimension reduction validation.

    For verifying that LLM-CUCB-AT scales as O(sqrt(m * (m + h_max) * T))
    rather than O(sqrt(m * d * T)).
    """
    rows = []
    for r in results:
        if r["agent"] != agent or r["corruption_type"] != corruption_type:
            continue
        if abs(r["epsilon"] - epsilon) > 1e-6:
            continue
        rows.append({
            "d": r["d"],
            "final_regret": r["final_regret"],
            "seed": r["seed"],
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = df.groupby("d").agg(
        mean_regret=("final_regret", "mean"),
        se_regret=("final_regret", lambda x: x.std() / np.sqrt(len(x))),
        n_seeds=("seed", "count"),
    ).reset_index()
    return summary


def trust_score_trajectory(results: list[dict],
                           corruption_type: str | None = None,
                           epsilon: float | None = None) -> dict:
    """Extract trust score trajectories for LLM-CUCB-AT.

    Returns dict with kappa, rho, tau arrays (mean across seeds).
    """
    kappas = []
    rhos = []
    taus = []

    for r in results:
        if r["agent"] != "llm_cucb_at":
            continue
        if corruption_type and r["corruption_type"] != corruption_type:
            continue
        if epsilon is not None and abs(r["epsilon"] - epsilon) > 1e-6:
            continue
        if r.get("trust_kappa") and any(x is not None for x in r["trust_kappa"]):
            kappas.append([x if x is not None else float('nan') for x in r["trust_kappa"]])
            rhos.append([x if x is not None else float('nan') for x in r["trust_rho"]])
            taus.append([x if x is not None else float('nan') for x in r["trust_tau"]])

    if not kappas:
        return {}

    min_len = min(len(k) for k in kappas)
    return {
        "kappa_mean": np.nanmean([k[:min_len] for k in kappas], axis=0),
        "rho_mean": np.nanmean([r[:min_len] for r in rhos], axis=0),
        "tau_mean": np.nanmean([t[:min_len] for t in taus], axis=0),
        "n_seeds": len(kappas),
    }
