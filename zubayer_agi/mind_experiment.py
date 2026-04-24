"""MIND Experiment: Test Tier 7 winners on simulated news recommendation.

Tier 7 validated V13/V6/N1 on synthetic Bernoulli configs (d=25-50, m=3-5).
This tests whether the advantage generalizes to MIND-simulated (d=200, m=5),
which has structured rewards from category preferences — a qualitatively
different problem.

Algorithms:
  - CTS (baseline)
  - N1_corr_full (Tier 7 #3)
  - V6_edge_pruning (Tier 7 #2)
  - V13_kmeans_refine (Tier 7 #1)
  - V7_per_arm_damping (Tier 7, significant but small effect)
  - warm_start_cts (exp4 winner at T=2000)

Configs:
  - n_categories ∈ {5, 10, 20}: tests if LLM clustering helps when
    category structure is coarse (5) vs fine-grained (20)
  - Each config uses different env_seeds for held-out validity

USAGE:
    python3 mind_experiment.py --T 10000 --n-seeds 15 --workers 8
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

from algorithms import ALL_ALGORITHMS, CTSBase
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from longhorizon_variants import LONGHORIZON_ALGOS
from longhorizon_v2 import LONGHORIZON_V2_ALGOS
from oracle_instrumented import InstrumentedOracle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-5.4"

ALGO_REGISTRY = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "V6_edge_pruning": LONGHORIZON_ALGOS["V6_edge_pruning"],
    "V13_kmeans_refine": LONGHORIZON_V2_ALGOS["V13_kmeans_refine"],
    "V7_per_arm_damping": LONGHORIZON_ALGOS["V7_per_arm_damping"],
}

# Warm-start CTS needs special handling (it's in the main agents, not zubayer_agi)
# We'll implement a simple version inline since the core logic is: inject LLM prior into Beta params
class WarmStartCTS(CTSBase):
    """CTS with LLM-initialized Beta priors (the exp4_mind winner)."""
    name = "warm_start_cts"

    def __init__(self, d, m, oracle, T_warmup=30, prior_strength=5.0, **kw):
        super().__init__(d, m, **kw)
        self.oracle = oracle
        self.T_warmup = T_warmup
        self.prior_strength = prior_strength
        self._initialized = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._initialized:
            picks = self.oracle.query_top_m(self.mu_hat.tolist())
            for aid in picks:
                if 0 <= aid < self.d:
                    self.alphas[aid] += self.prior_strength
            self._initialized = True
        return super().select_arms()

ALGO_REGISTRY["warm_start_cts"] = WarmStartCTS


def generate_configs():
    """Generate MIND-simulated configs varying category structure."""
    configs = []
    cid = 0
    for n_cats in [5, 10, 20]:
        for env_seed_offset in [100, 200, 300]:
            configs.append({
                "config_id": cid,
                "d": 200,
                "m": 5,
                "n_categories": n_cats,
                "env_seed": env_seed_offset + cid,
                "label": f"cats={n_cats}_seed{env_seed_offset}",
            })
            cid += 1
    return configs


def build_mind_env(config, seed):
    """Build a MINDEnvSimulated-style environment.

    Returns (means, optimal_set, optimal_reward, categories) for a given
    config and trial seed.
    """
    d = config["d"]
    m = config["m"]
    n_cats = config["n_categories"]
    rng = np.random.RandomState(config["env_seed"] * 1000 + seed)

    categories = rng.randint(0, n_cats, size=d)
    user_pref = rng.dirichlet(np.ones(n_cats))
    base_quality = rng.beta(2, 5, size=d)
    category_boost = user_pref[categories]
    means = np.clip(0.3 * base_quality + 0.7 * category_boost, 0.01, 0.99)

    optimal = np.argsort(means)[::-1][:m]
    optimal_reward = means[optimal].sum()
    return means, optimal, optimal_reward, categories


def run_single_trial(variant_name, config, seed, T):
    means, optimal, optimal_reward, categories = build_mind_env(config, seed)
    trial_id = f"c{config['config_id']}_s{seed}_{variant_name}"
    d, m = config["d"], config["m"]
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = ALGO_REGISTRY[variant_name]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": d, "m": m, "np_seed": trial_np_seed}

    oracle = None
    if variant_name != "CTS":
        oracle = InstrumentedOracle(
            d=d, m=m, model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=variant_name,
        )
        kwargs["oracle"] = oracle

    # V7 needs n_0 HP
    if variant_name == "V7_per_arm_damping":
        kwargs["n_0"] = 500

    agent = AlgoClass(**kwargs)
    cum_regret = 0.0
    regret_curve = []
    t_start = time.time()

    for t in range(T):
        if oracle is not None:
            oracle.current_t = t
        selected = list(agent.select_arms())
        if len(selected) < m:
            used = set(selected)
            remaining = [a for a in range(d) if a not in used]
            selected.extend(remaining[:m - len(selected)])
        selected = selected[:m]
        selected_means = means[selected]
        rewards = (reward_rng.uniform(size=m) < selected_means).astype(float)
        cum_regret += optimal_reward - selected_means.sum()
        agent.update(selected, rewards.tolist())
        if (t + 1) % 50 == 0:
            regret_curve.append(round(float(cum_regret), 2))

    elapsed = time.time() - t_start
    diag = oracle.diagnostics() if oracle else {}
    return {
        "trial_id": trial_id,
        "algo": variant_name,
        "model": MODEL,
        "config_id": config["config_id"],
        "n_categories": config["n_categories"],
        "seed": seed,
        "d": d, "m": m,
        "env_seed": config["env_seed"],
        "label": config["label"],
        "T": T,
        "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve,
        "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
    }


def analyze(all_results, out_dir, baseline="CTS"):
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.DataFrame(all_results)
    lines = []
    lines.append("=" * 100)
    lines.append(f"MIND EXPERIMENT — {len(df)} trials | baseline={baseline}")
    lines.append("=" * 100)

    # Overall rankings
    lines.append("\n--- OVERALL ---")
    agg = df.groupby("algo")["final_regret"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.sort_values("mean")
    base_mean = agg[agg["algo"] == baseline]["mean"]
    base_mean = base_mean.iloc[0] if len(base_mean) else float("nan")
    for _, row in agg.iterrows():
        vs = f"{100*(base_mean - row['mean'])/base_mean:+5.1f}%" if base_mean == base_mean else ""
        lines.append(f"  {row['algo']:25s} mean={row['mean']:8.1f}  se={row['se']:6.2f}  "
                     f"vs_{baseline}={vs:>8s}  n={int(row['count'])}")

    # Per n_categories breakdown
    for n_cats in sorted(df["n_categories"].unique()):
        sub = df[df["n_categories"] == n_cats]
        lines.append(f"\n--- n_categories={n_cats} (n={len(sub)}) ---")
        agg = sub.groupby("algo")["final_regret"].agg(["mean", "std", "count"]).reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        agg = agg.sort_values("mean")
        base_mean_sub = agg[agg["algo"] == baseline]["mean"]
        base_mean_sub = base_mean_sub.iloc[0] if len(base_mean_sub) else float("nan")
        for _, row in agg.iterrows():
            vs = f"{100*(base_mean_sub - row['mean'])/base_mean_sub:+5.1f}%" if base_mean_sub == base_mean_sub else ""
            lines.append(f"  {row['algo']:25s} mean={row['mean']:8.1f}  se={row['se']:6.2f}  "
                         f"vs_{baseline}={vs:>8s}  n={int(row['count'])}")

    # Paired Wilcoxon vs CTS (overall)
    lines.append("\n" + "=" * 100)
    lines.append(f"PAIRED WILCOXON vs {baseline} (Holm-Bonferroni)")
    lines.append("=" * 100)
    pivot = df.pivot_table(index=["config_id", "seed"], columns="algo", values="final_regret")
    if baseline in pivot.columns:
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

        K = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        holm_sig = [False] * K
        for rank, idx in enumerate(sorted_idx):
            alpha_k = 0.05 / (K - rank)
            if p_vals[idx] < alpha_k:
                holm_sig[idx] = True
            else:
                break

        lines.append(f"{'algo':25s}  {'W/L':>9s}  {'mean_adv':>8s}  {'p':>9s}  {'sig':>5s}")
        for (algo, w, l, adv, p), sig in zip(raw_rows, holm_sig):
            lines.append(f"  {algo:25s}  {w:3d}/{l:3d}    {adv:+7.2f}  {p:9.6f}  {'**' if sig else '':>5s}")

    report = "\n".join(lines)
    print(report)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"mind_experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")

    variant_specs = list(ALGO_REGISTRY.keys())
    configs = generate_configs()
    tasks = [(v, c, s) for v in variant_specs for c in configs for s in range(args.n_seeds)]

    import random
    random.seed(42)
    random.shuffle(tasks)

    logger.info(f"{len(variant_specs)} algos x {len(configs)} configs x "
                f"{args.n_seeds} seeds = {len(tasks)} trials (T={args.T})")

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "T": args.T,
            "n_seeds": args.n_seeds,
            "n_configs": len(configs),
            "model": MODEL,
            "variants": variant_specs,
            "total_trials": len(tasks),
            "timestamp": ts,
            "experiment": "mind_simulated",
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
                        logger.info(f"[{len(all_results)}/{len(tasks)}] {v} "
                                    f"c{c['config_id']} s{s} r={result['final_regret']:.1f} "
                                    f"ETA={eta:.1f}min")
                except Exception as e:
                    logger.exception(f"Trial {v} c{c['config_id']} s{s} failed: {e}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": ts, "T": args.T, "n_seeds": args.n_seeds,
            "variants": variant_specs, "n_configs": len(configs),
            "total_trials": len(all_results),
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)

    analyze(all_results, out_dir)


if __name__ == "__main__":
    main()
