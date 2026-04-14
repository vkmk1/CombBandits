"""Publication-quality plotting utilities for experiment results."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import (
    load_results, regret_curves_by_agent, compute_metrics,
    regret_vs_epsilon, regret_vs_dimension, trust_score_trajectory,
    pairwise_significance,
)


# Publication-quality defaults
plt.rcParams.update({
    "figure.figsize": (5.5, 3.8),
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
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
    "corrupt_robust_cucb": "Robust-CUCB",
    "warm_start_cts": "Warm-Start CTS",
    "exp4": "EXP4",
}

AGENT_LINESTYLES = {
    "cucb": "-",
    "cts": "-",
    "llm_cucb_at": "-",
    "llm_greedy": "--",
    "ellm_adapted": "-.",
    "opro_bandit": ":",
    "corrupt_robust_cucb": "--",
    "warm_start_cts": "-.",
    "exp4": ":",
}


def _savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf" if path.endswith(".pdf") else "png")
    plt.close(fig)


def plot_regret_curves(
    results_path: str,
    output_path: str = "figures/regret_curves.pdf",
    filter_corruption: str | None = None,
    filter_epsilon: float | None = None,
    agents: list[str] | None = None,
):
    """Plot cumulative regret curves with 95% CI shading."""
    results = load_results(results_path)
    if filter_corruption:
        results = [r for r in results if r["corruption_type"] == filter_corruption]
    if filter_epsilon is not None:
        results = [r for r in results if abs(r["epsilon"] - filter_epsilon) < 1e-6]

    curves = regret_curves_by_agent(results)

    fig, ax = plt.subplots()
    for key in sorted(curves.keys()):
        data = curves[key]
        agent = key.split("_corr=")[0]
        if agents and agent not in agents:
            continue
        color = AGENT_COLORS.get(agent, "black")
        label = AGENT_LABELS.get(agent, agent)
        ls = AGENT_LINESTYLES.get(agent, "-")
        T = len(data["mean"])
        x = np.arange(T)
        ax.plot(x, data["mean"], color=color, label=label, linestyle=ls)
        ax.fill_between(x, data["ci95_lower"], data["ci95_upper"],
                        alpha=0.12, color=color)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cumulative Regret $R_t$")
    title = "Cumulative Regret"
    if filter_corruption:
        title += f" ({filter_corruption}"
        if filter_epsilon is not None:
            title += f", $\\varepsilon={filter_epsilon}$"
        title += ")"
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    _savefig(fig, output_path)


def plot_regret_curves_multipanel(
    results_path: str,
    output_dir: str = "figures",
    corruption_types: list[str] | None = None,
    epsilons: list[float] | None = None,
):
    """Multi-panel figure: one subplot per (corruption_type, epsilon) pair."""
    results = load_results(results_path)

    if corruption_types is None:
        corruption_types = sorted(set(r["corruption_type"] for r in results))
    if epsilons is None:
        epsilons = sorted(set(r["epsilon"] for r in results))

    panels = [(ct, e) for ct in corruption_types for e in epsilons
              if any(r["corruption_type"] == ct and abs(r["epsilon"] - e) < 1e-6 for r in results)]

    n_panels = len(panels)
    if n_panels == 0:
        return
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (ct, eps) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = [r for r in results if r["corruption_type"] == ct and abs(r["epsilon"] - eps) < 1e-6]
        curves = regret_curves_by_agent(sub)

        for key in sorted(curves.keys()):
            data = curves[key]
            agent = key.split("_corr=")[0]
            color = AGENT_COLORS.get(agent, "black")
            label = AGENT_LABELS.get(agent, agent)
            ls = AGENT_LINESTYLES.get(agent, "-")
            T = len(data["mean"])
            ax.plot(np.arange(T), data["mean"], color=color, label=label, linestyle=ls)
            ax.fill_between(np.arange(T), data["ci95_lower"], data["ci95_upper"],
                            alpha=0.12, color=color)

        ax.set_title(f"{ct}, $\\varepsilon={eps}$", fontsize=10)
        ax.set_xlabel("Round $t$")
        if col == 0:
            ax.set_ylabel("Cumulative Regret")
        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper left")

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    _savefig(fig, f"{output_dir}/regret_multipanel.pdf")


def plot_trust_diagnostics(
    results_path: str,
    output_path: str = "figures/trust_diagnostics.pdf",
):
    """Plot trust score (kappa, rho, tau) trajectories for LLM-CUCB-AT
    across different corruption types — the headline diagnostic figure."""
    results = load_results(results_path)

    # Get all (corruption_type, epsilon) pairs that have trust data
    configs = set()
    for r in results:
        if r["agent"] == "llm_cucb_at" and r.get("trust_kappa"):
            configs.add((r["corruption_type"], r["epsilon"]))

    if not configs:
        # Fallback: no trust data in results
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trust diagnostics in results.\nRe-run experiments to capture trust scores.",
                transform=ax.transAxes, ha="center", va="center", fontsize=10, alpha=0.5)
        _savefig(fig, output_path)
        return

    configs = sorted(configs)
    ncols = min(3, len(configs))
    nrows = (len(configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

    for idx, (ct, eps) in enumerate(configs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        traj = trust_score_trajectory(results, corruption_type=ct, epsilon=eps)
        if not traj:
            continue

        T = len(traj["kappa_mean"])
        x = np.arange(T)
        ax.plot(x, traj["kappa_mean"], label="$\\kappa_t$ (consistency)", color="#1f77b4", linewidth=1.2)
        ax.plot(x, traj["rho_mean"], label="$\\rho_t$ (validation)", color="#ff7f0e", linewidth=1.2)
        ax.plot(x, traj["tau_mean"], label="$\\tau_t$ (composite)", color="#d62728", linewidth=2)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Trust Score")
        ax.set_title(f"{ct}, $\\varepsilon={eps}$", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(configs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Trust Score Trajectories (LLM-CUCB-AT)", fontsize=12, y=1.02)
    fig.tight_layout()
    _savefig(fig, output_path)


def plot_corruption_comparison(
    results_path: str,
    output_path: str = "figures/corruption_comparison.pdf",
):
    """Bar chart comparing final regret across corruption types with significance markers."""
    results = load_results(results_path)
    summary = compute_metrics(results)

    key_agents = ["cucb", "llm_cucb_at", "llm_greedy", "ellm_adapted", "warm_start_cts"]
    summary = summary[summary["agent"].isin(key_agents)]

    corr_types = sorted(summary["corruption_type"].unique())
    ncols = min(3, len(corr_types))
    nrows = (len(corr_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax_idx, corr_type in enumerate(corr_types):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        sub = summary[summary["corruption_type"] == corr_type]
        if sub.empty:
            continue

        epsilons = sorted(sub["epsilon"].unique())
        present_agents = [a for a in key_agents if a in sub["agent"].values]
        n_agents = len(present_agents)
        if n_agents == 0:
            continue
        width = 0.8 / n_agents
        x = np.arange(len(epsilons))

        for i, agent in enumerate(present_agents):
            agent_data = sub[sub["agent"] == agent]
            means = []
            ses = []
            for e in epsilons:
                match = agent_data[agent_data["epsilon"] == e]
                means.append(float(match["mean_regret"].iloc[0]) if len(match) > 0 else 0)
                ses.append(float(match["se_regret"].iloc[0]) if len(match) > 0 else 0)
            color = AGENT_COLORS.get(agent, "gray")
            label = AGENT_LABELS.get(agent, agent)
            ax.bar(x + i * width, means, width, yerr=ses,
                   color=color, label=label, alpha=0.85, capsize=2)

        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Final Cumulative Regret")
        ax.set_title(f"Corruption: {corr_type}")
        ax.set_xticks(x + width * n_agents / 2)
        ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
        if ax_idx == 0:
            ax.legend(fontsize=6, ncol=2)

    for idx in range(len(corr_types), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    _savefig(fig, output_path)


def plot_regret_vs_epsilon(
    results_path: str,
    output_path: str = "figures/regret_vs_epsilon.pdf",
    agents: list[str] | None = None,
    corruption_type: str = "uniform",
):
    """Theory validation: regret as function of epsilon.

    Overlays theoretical prediction O(epsilon * m * T / Delta_min)
    to verify the corruption cost term.
    """
    results = load_results(results_path)
    if agents is None:
        agents = ["llm_cucb_at", "cucb", "llm_greedy"]

    fig, ax = plt.subplots()

    for agent in agents:
        data = regret_vs_epsilon(results, agent, corruption_type)
        if data.empty:
            continue
        color = AGENT_COLORS.get(agent, "black")
        label = AGENT_LABELS.get(agent, agent)
        for d_val in sorted(data["d"].unique()):
            sub = data[data["d"] == d_val]
            marker = {50: "o", 100: "s", 200: "^"}.get(d_val, "D")
            ax.errorbar(sub["epsilon"], sub["mean_regret"], yerr=sub["se_regret"],
                        color=color, marker=marker, capsize=3,
                        label=f"{label} ($d={d_val}$)", linestyle="-", markersize=5)

    # Theoretical overlay: linear in epsilon
    eps_range = np.linspace(0, 0.5, 50)
    T = results[0]["T"] if results else 10000
    m = results[0]["m"] if results else 10
    # Scale to match data (approximate)
    ax.plot(eps_range, eps_range * m * T * 0.02, "k--", alpha=0.4,
            label="$O(\\varepsilon \\cdot m \\cdot T / \\Delta_{\\min})$")

    ax.set_xlabel("Corruption Rate $\\varepsilon$")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_title("Regret vs. Corruption Rate (Theory Validation)")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, output_path)


def plot_regret_vs_dimension(
    results_path: str,
    output_path: str = "figures/regret_vs_dimension.pdf",
    corruption_type: str = "uniform",
    epsilon: float = 0.1,
):
    """Theory validation: regret as function of d.

    LLM-CUCB-AT should scale as O(sqrt(m * (m + sqrt(d)) * T)),
    significantly flatter than CUCB's O(sqrt(m * d * T)).
    """
    results = load_results(results_path)
    fig, ax = plt.subplots()

    for agent in ["llm_cucb_at", "cucb", "cts"]:
        data = regret_vs_dimension(results, agent, corruption_type, epsilon)
        if data.empty:
            continue
        color = AGENT_COLORS.get(agent, "black")
        label = AGENT_LABELS.get(agent, agent)
        ax.errorbar(data["d"], data["mean_regret"], yerr=data["se_regret"],
                    color=color, marker="o", capsize=3, label=label, linewidth=1.8)

    # Theoretical curves
    d_range = np.linspace(20, 250, 50)
    T = results[0]["T"] if results else 10000
    m = results[0]["m"] if results else 10
    scale = 0.5  # Calibrate to data
    ax.plot(d_range, scale * np.sqrt(m * d_range * T), "b:", alpha=0.4,
            label="$O(\\sqrt{mdT})$ (CUCB theory)")
    ax.plot(d_range, scale * np.sqrt(m * (m + np.sqrt(d_range)) * T), "r:", alpha=0.4,
            label="$O(\\sqrt{m(m+\\sqrt{d})T})$ (ours)")

    ax.set_xlabel("Ground Set Size $d$")
    ax.set_ylabel("Final Cumulative Regret")
    ax.set_title(f"Dimension Reduction ($\\varepsilon={epsilon}$)")
    ax.legend(fontsize=7)
    _savefig(fig, output_path)


def generate_all_figures(results_path: str, output_dir: str = "figures"):
    """Generate the complete figure set for the paper."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Figure 1: Multi-panel regret curves
    plot_regret_curves_multipanel(results_path, output_dir)

    # Figure 2: Trust score trajectories
    plot_trust_diagnostics(results_path, f"{output_dir}/trust_diagnostics.pdf")

    # Figure 3: Corruption comparison bar chart
    plot_corruption_comparison(results_path, f"{output_dir}/corruption_comparison.pdf")

    # Figure 4: Regret vs epsilon (theory validation)
    plot_regret_vs_epsilon(results_path, f"{output_dir}/regret_vs_epsilon.pdf")

    # Figure 5: Regret vs dimension (dimension reduction)
    plot_regret_vs_dimension(results_path, f"{output_dir}/regret_vs_dimension.pdf")

    # Figure 6: Headline — clean vs consistently wrong
    plot_regret_curves(results_path, f"{output_dir}/headline_clean.pdf",
                       filter_corruption="uniform", filter_epsilon=0.0)
    plot_regret_curves(results_path, f"{output_dir}/headline_consistent_wrong.pdf",
                       filter_corruption="consistent_wrong", filter_epsilon=1.0)

    # Statistical significance table
    results = load_results(results_path)
    sig_results = []
    for ct in ["uniform", "consistent_wrong", "adversarial"]:
        for eps in [0.0, 0.1, 0.3, 0.5, 1.0]:
            sig = pairwise_significance(results, "llm_cucb_at", "cucb",
                                        corruption_type=ct, epsilon=eps)
            if "error" not in sig:
                sig["corruption_type"] = ct
                sig["epsilon"] = eps
                sig_results.append(sig)

    if sig_results:
        import pandas as pd
        sig_df = pd.DataFrame(sig_results)
        sig_df.to_csv(f"{output_dir}/significance_tests.csv", index=False)
