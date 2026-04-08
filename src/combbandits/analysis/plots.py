"""Plotting utilities for experiment results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import load_results, regret_curves_by_agent


# NeurIPS-quality defaults
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

AGENT_COLORS = {
    "cucb": "#1f77b4",
    "cts": "#ff7f0e",
    "llm_cucb_at": "#d62728",
    "llm_greedy": "#9467bd",
    "ellm_adapted": "#8c564b",
    "opro_bandit": "#e377c2",
    "corrupt_robust_cucb": "#7f7f7f",
    "warm_start_cts": "#bcbd22",
    "exp4": "#17becf",
}

AGENT_LABELS = {
    "cucb": "CUCB",
    "cts": "CTS",
    "llm_cucb_at": "LLM-CUCB-AT (ours)",
    "llm_greedy": "LLM-Greedy",
    "ellm_adapted": "ELLM-Adapted",
    "opro_bandit": "OPRO-Bandit",
    "corrupt_robust_cucb": "Corrupt-Robust CUCB",
    "warm_start_cts": "Warm-Start CTS",
    "exp4": "EXP4",
}


def plot_regret_curves(
    results_path: str,
    output_path: str = "figures/regret_curves.pdf",
    filter_corruption: str | None = None,
    filter_epsilon: float | None = None,
):
    """Plot cumulative regret curves for all agents."""
    results = load_results(results_path)
    if filter_corruption:
        results = [r for r in results if r["corruption_type"] == filter_corruption]
    if filter_epsilon is not None:
        results = [r for r in results if abs(r["epsilon"] - filter_epsilon) < 1e-6]

    curves = regret_curves_by_agent(results)

    fig, ax = plt.subplots()
    for key, data in sorted(curves.items()):
        agent = key.split("_corr=")[0]
        color = AGENT_COLORS.get(agent, "black")
        label = AGENT_LABELS.get(agent, agent)
        T = len(data["mean"])
        x = np.arange(T)
        ax.plot(x, data["mean"], color=color, label=label)
        ax.fill_between(x, data["mean"] - data["se"], data["mean"] + data["se"],
                        alpha=0.15, color=color)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative Regret $R_t$")
    title = "Cumulative Regret"
    if filter_corruption:
        title += f" ({filter_corruption}, $\\varepsilon={filter_epsilon}$)"
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_trust_diagnostics(
    results_path: str,
    output_path: str = "figures/trust_diagnostics.pdf",
):
    """Plot trust score (kappa, rho, tau) trajectories for LLM-CUCB-AT."""
    results = load_results(results_path)
    llm_results = [r for r in results if r["agent"] == "llm_cucb_at"]
    if not llm_results:
        return

    # Take first seed for clarity
    r = llm_results[0]
    # Trust data would need to be stored in trial results; placeholder
    fig, ax = plt.subplots()
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Trust Score")
    ax.set_title("Trust Score Trajectory (LLM-CUCB-AT)")
    ax.text(0.5, 0.5, "Trust diagnostics require extended result format",
            transform=ax.transAxes, ha="center", fontsize=10, alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_corruption_comparison(
    results_path: str,
    output_path: str = "figures/corruption_comparison.pdf",
):
    """Bar chart comparing final regret across corruption types and epsilon values."""
    results = load_results(results_path)

    from .metrics import compute_metrics
    summary = compute_metrics(results)

    # Filter to key agents
    key_agents = ["cucb", "llm_cucb_at", "llm_greedy", "ellm_adapted", "warm_start_cts"]
    summary = summary[summary["agent"].isin(key_agents)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, corr_type in enumerate(["uniform", "consistent_wrong"]):
        ax = axes[ax_idx]
        sub = summary[summary["corruption_type"] == corr_type]
        if sub.empty:
            ax.set_title(f"No data for {corr_type}")
            continue

        epsilons = sorted(sub["epsilon"].unique())
        n_agents = len(key_agents)
        width = 0.8 / n_agents
        x = np.arange(len(epsilons))

        for i, agent in enumerate(key_agents):
            agent_data = sub[sub["agent"] == agent]
            means = [agent_data[agent_data["epsilon"] == e]["mean_regret"].values[0]
                     if len(agent_data[agent_data["epsilon"] == e]) > 0 else 0
                     for e in epsilons]
            ses = [agent_data[agent_data["epsilon"] == e]["se_regret"].values[0]
                   if len(agent_data[agent_data["epsilon"] == e]) > 0 else 0
                   for e in epsilons]
            color = AGENT_COLORS.get(agent, "gray")
            label = AGENT_LABELS.get(agent, agent)
            ax.bar(x + i * width, means, width, yerr=ses,
                   color=color, label=label, alpha=0.85)

        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Final Cumulative Regret")
        ax.set_title(f"Corruption: {corr_type}")
        ax.set_xticks(x + width * n_agents / 2)
        ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
        if ax_idx == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
