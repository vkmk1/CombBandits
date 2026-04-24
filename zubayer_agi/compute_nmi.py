"""Compute NMI (Normalized Mutual Information) between LLM clusters and ground truth.

Ground truth partition: {optimal arms} vs {suboptimal arms}.
LLM partition: cluster assignments from oracle cache.

Reads from:
  - raw_trials.jsonl (to get optimal_set per config)
  - llm_calls.jsonl (to get cluster responses per trial)

Outputs NMI statistics for Appendix B of the paper.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent / "results" / "tier4_full_20260424_050709"
)


def parse_clusters(response_text: str) -> list[list[int]] | None:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    start = text.find("[[")
    if start < 0:
        start = text.find("[")
        if start < 0:
            return None
    try:
        parsed = json.loads(text[start:])
        if isinstance(parsed, list) and all(isinstance(c, list) for c in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        bracket_depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '[':
                bracket_depth += 1
            elif text[i] == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    end = i
                    break
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, list) and all(isinstance(c, list) for c in parsed):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute NMI between two label arrays (same length)."""
    n = len(labels_true)
    if n == 0:
        return 0.0

    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    contingency = np.zeros((len(classes_true), len(classes_pred)))
    true_map = {c: i for i, c in enumerate(classes_true)}
    pred_map = {c: i for i, c in enumerate(classes_pred)}
    for i in range(n):
        contingency[true_map[labels_true[i]], pred_map[labels_pred[i]]] += 1

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    mi = 0.0
    for i in range(len(classes_true)):
        for j in range(len(classes_pred)):
            if contingency[i, j] > 0:
                mi += (contingency[i, j] / n) * np.log(
                    (n * contingency[i, j]) / (row_sums[i] * col_sums[j])
                )

    h_true = -np.sum((row_sums / n) * np.log(row_sums / n + 1e-15))
    h_pred = -np.sum((col_sums / n) * np.log(col_sums / n + 1e-15))

    denom = 0.5 * (h_true + h_pred)
    if denom < 1e-15:
        return 0.0
    return mi / denom


def main():
    raw_trials_path = RESULTS_DIR / "raw_trials.jsonl"
    llm_calls_path = RESULTS_DIR / "llm_calls.jsonl"

    if not raw_trials_path.exists():
        print(f"ERROR: {raw_trials_path} not found"); sys.exit(1)
    if not llm_calls_path.exists():
        print(f"ERROR: {llm_calls_path} not found"); sys.exit(1)

    trials = {}
    with open(raw_trials_path) as f:
        for line in f:
            t = json.loads(line)
            trials[t["trial_id"]] = t

    cluster_calls = []
    with open(llm_calls_path) as f:
        for line in f:
            call = json.loads(line)
            if call.get("query_type") == "clusters":
                cluster_calls.append(call)

    print(f"Loaded {len(trials)} trials, {len(cluster_calls)} cluster calls")

    nmi_by_setting = defaultdict(list)
    nmi_by_algo = defaultdict(list)
    all_nmis = []

    for call in cluster_calls:
        tid = call["trial_id"]
        algo = call["algo"]

        parts = tid.split("_", 2)
        config_id = int(parts[0][1:])
        seed = int(parts[1][1:])

        trial_key_n1 = f"c{config_id}_s{seed}_N1_corr_full"
        trial_key_n4 = f"c{config_id}_s{seed}_N4_robust_corr"
        trial_key_cts = f"c{config_id}_s{seed}_CTS"

        ref_trial = None
        for tk in [trial_key_n1, trial_key_n4, trial_key_cts, tid]:
            if tk in trials:
                ref_trial = trials[tk]
                break
        if ref_trial is None:
            continue

        d = ref_trial["d"]
        optimal_set = set(ref_trial["optimal_set"])

        clusters = parse_clusters(call.get("response_text", ""))
        if clusters is None:
            continue

        labels_true = np.array([1 if i in optimal_set else 0 for i in range(d)])

        labels_pred = np.zeros(d, dtype=int)
        for cidx, cluster in enumerate(clusters):
            for arm in cluster:
                if 0 <= arm < d:
                    labels_pred[arm] = cidx

        score = nmi(labels_true, labels_pred)
        setting = (d, ref_trial["m"])
        nmi_by_setting[setting].append(score)
        nmi_by_algo[algo].append(score)
        all_nmis.append(score)

    print(f"\nComputed NMI for {len(all_nmis)} cluster calls")
    print(f"\n{'='*60}")
    print("NMI: LLM Clusters vs True {Optimal, Suboptimal} Partition")
    print(f"{'='*60}")

    print(f"\nOverall: NMI = {np.mean(all_nmis):.3f} +/- {np.std(all_nmis)/np.sqrt(len(all_nmis)):.3f} "
          f"(n={len(all_nmis)})")

    print(f"\n--- By (d, m) setting ---")
    for (d, m) in sorted(nmi_by_setting.keys()):
        vals = nmi_by_setting[(d, m)]
        print(f"  d={d}, m={m}: NMI = {np.mean(vals):.3f} +/- {np.std(vals)/np.sqrt(len(vals)):.3f} "
              f"(n={len(vals)})")

    print(f"\n--- By algorithm ---")
    for algo in sorted(nmi_by_algo.keys()):
        vals = nmi_by_algo[algo]
        print(f"  {algo:30s}: NMI = {np.mean(vals):.3f} +/- {np.std(vals)/np.sqrt(len(vals)):.3f} "
              f"(n={len(vals)})")

    print(f"\n--- For paper (Appendix B table) ---")
    for (d, m) in sorted(nmi_by_setting.keys()):
        vals = nmi_by_setting[(d, m)]
        print(f"  d={d}, m={m} & {np.mean(vals):.2f} $\\pm$ {np.std(vals)/np.sqrt(len(vals)):.2f} \\\\")


if __name__ == "__main__":
    main()
