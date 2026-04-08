"""Metrics computation for experiment results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> pd.DataFrame:
    """Aggregate trial results into summary statistics.

    Groups by (agent, env, corruption_type, epsilon) and computes
    mean/std/se of final regret and other metrics across seeds.
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
            "regret_at_half": regret_curve[len(regret_curve)//2] if len(regret_curve) > 1 else 0,
            "oracle_queries": r.get("oracle_queries", 0),
            "oracle_tokens": r.get("oracle_tokens", 0),
        })

    df = pd.DataFrame(rows)

    # Group statistics
    group_cols = ["agent", "env", "corruption_type", "epsilon", "d", "m", "T"]
    summary = df.groupby(group_cols).agg(
        mean_regret=("final_regret", "mean"),
        std_regret=("final_regret", "std"),
        se_regret=("final_regret", lambda x: x.std() / np.sqrt(len(x))),
        n_seeds=("seed", "count"),
        mean_queries=("oracle_queries", "mean"),
        mean_tokens=("oracle_tokens", "mean"),
    ).reset_index()

    return summary


def regret_curves_by_agent(results: list[dict]) -> dict[str, np.ndarray]:
    """Extract mean regret curves grouped by agent.

    Returns dict mapping agent_name -> (T,) array of mean cumulative regret.
    """
    from collections import defaultdict
    curves = defaultdict(list)
    for r in results:
        key = (r["agent"], r["corruption_type"], r["epsilon"])
        curves[key].append(np.array(r["regret_curve"]))

    out = {}
    for key, arrs in curves.items():
        min_len = min(len(a) for a in arrs)
        stacked = np.stack([a[:min_len] for a in arrs])
        agent_name = f"{key[0]}_corr={key[1]}_eps={key[2]}"
        out[agent_name] = {
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "se": stacked.std(axis=0) / np.sqrt(len(arrs)),
        }
    return out
