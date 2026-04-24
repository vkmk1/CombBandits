"""Tier 7: RIGOROUS VALIDATION of top long-horizon variants.

Phase 1 (Tier 6) was screening — 7 variants vs N1 on the 16 configs we observed.
Phase 2 (this file) addresses the overfitting concerns from that run:

1. HELD-OUT CONFIGS: new env_seed offsets (500/600/700/800) produce different
   arm permutations from the 100/200/300/400 offsets used in Tier 4/5/6.
2. HYPERPARAMETER SENSITIVITY: for each winner, run 2-3 HP settings to show
   the result isn't a single fragile point.
3. MORE SEEDS: n_seeds=15 (vs 10 in screening).
4. HOLM-BONFERRONI: adjust p-values for multiple comparisons vs N1.
5. JOINT OUT/IN TEST: run on both original 16 configs AND held-out 16 configs;
   compare performance delta. Overfit variants will show large (original - held) gap.

This should be run AFTER Tier 6 screening picks winners.

USAGE:
    python3 tier7_validation.py --variants V1_decay_kernel,V6_edge_pruning \\
        --n-seeds 15 --T 25000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from longhorizon_variants import LONGHORIZON_ALGOS
from oracle_instrumented import InstrumentedOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-5.4"

BASE_ALGOS = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    **LONGHORIZON_ALGOS,
}


def generate_configs(config_set: str = "both"):
    """Generate configs.

    config_set:
      - 'original': env_seed offsets 100/200/300/400 — same as Tier 4/5/6
      - 'held_out': env_seed offsets 500/600/700/800 — fresh seeds, new arm layouts
      - 'both': union (for joint validation)
    """
    configs = []
    cid = 0

    def emit(offsets, suffix=""):
        nonlocal cid
        for d in [25, 50]:
            for m in [3, 5]:
                for gap_type, delta, seed_offset in offsets:
                    configs.append({
                        "config_id": cid,
                        "d": d, "m": m,
                        "gap_type": gap_type,
                        "delta_min": delta,
                        "env_seed": seed_offset + cid,
                        "split": suffix,
                    })
                    cid += 1

    if config_set in ("original", "both"):
        emit([
            ("uniform", 0.20, 100),
            ("uniform", 0.10, 200),
            ("hard", 0.05, 300),
            ("staggered", 0.02, 400),
        ], suffix="original")

    if config_set in ("held_out", "both"):
        emit([
            ("uniform", 0.20, 500),
            ("uniform", 0.10, 600),
            ("hard", 0.05, 700),
            ("staggered", 0.02, 800),
        ], suffix="held_out")

    return configs


def build_env(config, seed):
    d, m = config["d"], config["m"]
    rng = np.random.RandomState(config["env_seed"] * 1000 + seed)
    gap_type = config["gap_type"]
    delta = config["delta_min"]

    if gap_type == "uniform":
        means = np.full(d, 0.7 - delta)
        top_idx = rng.choice(d, m, replace=False)
        means[top_idx] = 0.7
    elif gap_type == "hard":
        means = np.full(d, 0.7 - 0.2)
        top_idx = rng.choice(d, m, replace=False)
        means[top_idx] = 0.7
        remaining = [i for i in range(d) if i not in top_idx]
        near_idx = rng.choice(remaining, min(m, len(remaining)), replace=False)
        means[near_idx] = 0.7 - delta
    elif gap_type == "staggered":
        means = np.linspace(0.7, 0.3, d)
        rng.shuffle(means)
    else:
        raise ValueError(gap_type)

    optimal = np.argsort(means)[::-1][:m]
    optimal_reward = means[optimal].sum()
    return means, optimal, optimal_reward


def parse_variant_spec(spec: str):
    """Parse 'V1_decay_kernel@tau=3000' or 'V7_per_arm_damping@n_0=500'.

    Returns (class_key, variant_name, kwargs).
    """
    if "@" not in spec:
        return spec, spec, {}
    base, hp = spec.split("@", 1)
    kwargs = {}
    name_parts = [base]
    for kv in hp.split(","):
        k, v = kv.split("=")
        k = k.strip()
        try:
            v_parsed = float(v)
            if v_parsed.is_integer():
                v_parsed = int(v_parsed)
        except ValueError:
            v_parsed = v
        kwargs[k] = v_parsed
        name_parts.append(f"{k}{v_parsed}")
    return base, "_".join(name_parts), kwargs


def run_single_trial(variant_spec, config, seed, T):
    class_key, variant_name, extra_kwargs = parse_variant_spec(variant_spec)
    trial_id = f"c{config['config_id']}_s{seed}_{variant_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = BASE_ALGOS[class_key]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": trial_np_seed, **extra_kwargs}

    oracle = None
    if class_key != "CTS":
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=variant_name,
        )
        kwargs["oracle"] = oracle

    agent = AlgoClass(**kwargs)
    cum_regret = 0.0
    regret_curve = []
    t_start = time.time()

    for t in range(T):
        if oracle is not None:
            oracle.current_t = t
        selected = list(agent.select_arms())
        if len(selected) < config["m"]:
            used = set(selected)
            remaining = [a for a in range(config["d"]) if a not in used]
            selected.extend(remaining[:config["m"] - len(selected)])
        selected = selected[:config["m"]]
        selected_means = means[selected]
        rewards = (reward_rng.uniform(size=config["m"]) < selected_means).astype(float)
        cum_regret += optimal_reward - selected_means.sum()
        agent.update(selected, rewards.tolist())
        if (t + 1) % 50 == 0:
            regret_curve.append(round(float(cum_regret), 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}
    return {
        "trial_id": trial_id,
        "algo": variant_name,
        "base_class": class_key,
        "model": MODEL,
        "config_id": config["config_id"],
        "split": config.get("split", "original"),
        "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"], "delta_min": config["delta_min"],
        "env_seed": config["env_seed"],
        "T": T, "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve, "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
        "hp": extra_kwargs,
    }


def analyze(all_results, out_dir, baseline: str = "N1_corr_full"):
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.DataFrame(all_results)
    lines = []
    lines.append("=" * 100)
    lines.append(f"TIER 7 VALIDATION — {len(df)} trials | baseline={baseline}")
    lines.append("=" * 100)

    # Overall and per-split
    for split in ["original", "held_out", "both"]:
        sub = df if split == "both" else df[df["split"] == split]
        if len(sub) == 0:
            continue
        lines.append(f"\n--- SPLIT: {split} (n_trials={len(sub)}) ---")
        agg = sub.groupby("algo")["final_regret"].agg(["mean", "std", "count"]).reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        agg = agg.sort_values("mean")
        base_mean = agg[agg["algo"] == baseline]["mean"]
        base_mean = base_mean.iloc[0] if len(base_mean) else float("nan")
        for _, row in agg.iterrows():
            vs_base = f"{100*(base_mean - row['mean'])/base_mean:+5.1f}%" if base_mean == base_mean else ""
            lines.append(f"  {row['algo']:35s} mean={row['mean']:7.1f}  se={row['se']:6.2f}  "
                         f"vs_{baseline}={vs_base:>8s}  n={int(row['count'])}")

    # Paired tests with Holm-Bonferroni correction
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"PAIRED WILCOXON vs {baseline} (Holm-Bonferroni corrected)")
    lines.append("=" * 100)
    for split in ["original", "held_out", "both"]:
        sub = df if split == "both" else df[df["split"] == split]
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(index=["config_id", "seed"], columns="algo",
                                values="final_regret")
        if baseline not in pivot.columns:
            continue
        comparisons = [c for c in pivot.columns if c != baseline]
        p_vals = []
        raw_rows = []
        for algo in comparisons:
            pair = pivot[[baseline, algo]].dropna()
            if len(pair) < 5:
                raw_rows.append((algo, 0, 0, 0, 1.0))
                p_vals.append(1.0)
                continue
            diffs = pair[baseline] - pair[algo]
            wins = int((diffs > 0).sum())
            losses = int((diffs < 0).sum())
            try:
                _, p = wilcoxon(pair[baseline], pair[algo])
            except ValueError:
                p = 1.0
            raw_rows.append((algo, wins, losses, diffs.mean(), p))
            p_vals.append(p)

        # Holm-Bonferroni
        K = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        holm_sig = [False] * K
        for rank, idx in enumerate(sorted_idx):
            alpha_k = 0.05 / (K - rank)
            if p_vals[idx] < alpha_k:
                holm_sig[idx] = True
            else:
                break

        lines.append(f"\n--- SPLIT: {split} ---")
        lines.append(f"{'algo':35s}  {'W/L':>9s}  {'mean_adv':>8s}  {'p':>9s}  {'sig':>5s}")
        for (algo, w, l, adv, p), sig in zip(raw_rows, holm_sig):
            lines.append(f"  {algo:35s}  {w:3d}/{l:3d}    {adv:+7.2f}  {p:9.4f}  {'**' if sig else '':>5s}")

    # Generalization gap: original minus held_out advantage
    lines.append("")
    lines.append("=" * 100)
    lines.append("GENERALIZATION GAP (overfit detector)")
    lines.append("=" * 100)
    lines.append("For each variant: advantage vs N1 on original configs - advantage vs N1 on held-out.")
    lines.append("Large positive gap = variant is overfit to original configs.")
    for algo in df["algo"].unique():
        if algo == baseline:
            continue
        orig = df[(df["split"] == "original")]
        held = df[(df["split"] == "held_out")]
        if len(orig) == 0 or len(held) == 0:
            continue
        orig_pivot = orig.pivot_table(index=["config_id", "seed"], columns="algo", values="final_regret")
        held_pivot = held.pivot_table(index=["config_id", "seed"], columns="algo", values="final_regret")
        if algo not in orig_pivot.columns or baseline not in orig_pivot.columns:
            continue
        if algo not in held_pivot.columns or baseline not in held_pivot.columns:
            continue
        orig_adv = (orig_pivot[baseline] - orig_pivot[algo]).mean()
        held_adv = (held_pivot[baseline] - held_pivot[algo]).mean()
        gap = orig_adv - held_adv
        flag = "  ⚠ OVERFIT" if gap > 20 else ""
        lines.append(f"  {algo:35s}  orig_adv={orig_adv:+6.1f}  held_adv={held_adv:+6.1f}  "
                     f"gap={gap:+6.1f}{flag}")

    report = "\n".join(lines)
    print(report)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, required=True,
                        help="Comma list of variant specs. Supports HP overrides: "
                             "'V1_decay_kernel@tau=3000,V1_decay_kernel@tau=10000'")
    parser.add_argument("--T", type=int, default=25000)
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--config-set", choices=["original", "held_out", "both"], default="both")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier7_validation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")

    # Always include baseline CTS and N1
    variant_specs = ["CTS", "N1_corr_full"] + [v.strip() for v in args.variants.split(",") if v.strip()]

    configs = generate_configs(args.config_set)
    tasks = [(v, c, s) for v in variant_specs for c in configs for s in range(args.n_seeds)]

    import random
    random.seed(42)
    random.shuffle(tasks)

    logger.info(f"Running {len(variant_specs)} variants x {len(configs)} configs x "
                f"{args.n_seeds} seeds = {len(tasks)} trials (T={args.T})")

    # Write meta.json at START so dashboard has totals immediately
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "T": args.T,
            "n_seeds": args.n_seeds,
            "n_configs": len(configs),
            "model": MODEL,
            "variants": variant_specs,
            "total_trials": len(tasks),
            "timestamp": ts,
            "config_set": args.config_set,
        }, f, indent=2)

    all_results = []
    t_start = time.time()
    raw_path = out_dir / "raw_trials.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, v, c, s, args.T): (v, c, s)
                   for v, c, s in tasks}
        with open(raw_path, "w") as f:
            for fut in as_completed(futures):
                v, c, s = futures[fut]
                try:
                    result = fut.result()
                    all_results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    if len(all_results) % 25 == 0:
                        elapsed = time.time() - t_start
                        eta = (len(tasks) - len(all_results)) / (len(all_results) / elapsed) / 60
                        logger.info(f"[{len(all_results)}/{len(tasks)}] {v} c{c['config_id']} "
                                    f"s{s} r={result['final_regret']:.1f} ETA={eta:.1f}min")
                except Exception as e:
                    logger.exception(f"Trial {v} c{c['config_id']} s{s} failed: {e}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": ts, "T": args.T, "n_seeds": args.n_seeds,
            "variants": variant_specs, "n_configs": len(configs),
            "total_trials": len(all_results),
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)

    analyze(all_results, out_dir, baseline="N1_corr_full")


if __name__ == "__main__":
    main()
