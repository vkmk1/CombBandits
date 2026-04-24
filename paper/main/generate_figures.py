"""Generate publication-quality regret curve figures from Tier 4 raw_trials.jsonl."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = "/Users/anishkataria/CombBandits/zubayer_agi/results/tier4_full_20260424_050709/raw_trials.jsonl"
OUT = "/Users/anishkataria/CombBandits/paper/main/figures"

ALGO_DISPLAY = {
    "CTS": "CTS",
    "CUCB": "CUCB",
    "N1_corr_full": "CorrCTS-Full (N1)",
    "N4_robust_corr": "RobustCorrCTS (N4)",
    "N5_corr_full_robust": "N5 (Full+Robust)",
    "M2_corr_cts": "M2 (block-diag)",
    "ABLATION_random_corr": "RandomCorr",
    "PAPER_ts_llm": "TS-LLM",
}

ALGO_ORDER = [
    "N1_corr_full", "N5_corr_full_robust", "N4_robust_corr",
    "PAPER_ts_llm", "ABLATION_random_corr", "M2_corr_cts", "CTS", "CUCB"
]

COLORS = {
    "N1_corr_full": "#1f77b4",
    "N5_corr_full_robust": "#2ca02c",
    "N4_robust_corr": "#9467bd",
    "PAPER_ts_llm": "#ff7f0e",
    "ABLATION_random_corr": "#8c564b",
    "M2_corr_cts": "#e377c2",
    "CTS": "#7f7f7f",
    "CUCB": "#d62728",
}

trials = []
with open(DATA) as f:
    for line in f:
        trials.append(json.loads(line))

settings = [(25, 3), (25, 5), (50, 3), (50, 5)]

for d_val, m_val in settings:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    subset = [t for t in trials if t["d"] == d_val and t["m"] == m_val]

    for algo_key in ALGO_ORDER:
        algo_trials = [t for t in subset if t["algo"] == algo_key]
        if not algo_trials:
            continue

        curves = np.array([t["regret_curve"] for t in algo_trials])
        n_points = curves.shape[1]
        x = np.linspace(50, 2500, n_points)
        mean = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(curves.shape[0])

        label = ALGO_DISPLAY.get(algo_key, algo_key)
        color = COLORS.get(algo_key, "#333333")

        if algo_key == "CUCB":
            ax.plot(x, mean, color=color, label=label, linewidth=1.0, linestyle="--", alpha=0.6)
        else:
            ax.plot(x, mean, color=color, label=label, linewidth=1.3)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12)

    ax.set_xlabel("Round $t$", fontsize=9)
    ax.set_ylabel("Cumulative regret", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if d_val == 25 and m_val == 3:
        ax.legend(fontsize=5.5, loc="upper left", framealpha=0.9, ncol=2)

    plt.tight_layout(pad=0.4)
    fname = f"{OUT}/regret_d{d_val}_m{m_val}.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

print("All figures generated.")
