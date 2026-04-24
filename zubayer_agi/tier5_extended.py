"""TIER 5 EXTENDED — Long-horizon + RandomKernel ablation.

Addresses ICML reviewer concerns:
1. T=25000 (10x original) to prove gains aren't transient
2. RandomKernel ablation: random clusters + full RBF kernel
   (isolates LLM cluster quality from kernel construction)
3. NMI cluster quality computation

ALGORITHMS (7):
- CTS        : Thompson sampling baseline
- CUCB       : UCB baseline
- N1_corr_full : CorrCTS-Full (our champion)
- N4_robust_corr : RobustCorrCTS
- ABLATION_random_corr : RandomCorr (block-diagonal, random clusters)
- ABLATION_random_kernel : RandomKernel (full RBF kernel, random clusters)  [NEW]
- PAPER_ts_llm : TS-LLM baseline

MODEL: gpt-5.4
COST: ~$70 LLM (TS-LLM dominates) + ~$5 EC2 = ~$75
TIME: ~2-3 hours on c5.4xlarge with 16 workers
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ALL_ALGORITHMS, CTSBase
from paper_baselines import PAPER_BASELINES
from breakthrough_algorithms import BREAKTHROUGH_ALGOS
from final_algorithms import FINAL_ALGOS
from cucb_baseline import CUCB
from oracle_instrumented import InstrumentedOracle
from oracle import CACHE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-5.4"


# ─── NEW: RandomKernel ablation ──────────────────────────────────────────
class RandomKernelCTS(CTSBase):
    """Random clusters + full RBF kernel (same construction as CorrCTS-Full).

    This isolates whether improvement comes from:
    (a) the kernel construction itself, or
    (b) the LLM's cluster ordering.
    If CorrCTS-Full >> RandomKernel, the LLM's clusters matter.
    If CorrCTS-Full ≈ RandomKernel, it's just the kernel.
    """
    name = "ABLATION_random_kernel"

    def __init__(self, d, m, T_warmup: int = 30,
                 kernel_scale: float = 0.5, rho_max: float = 0.7,
                 n_clusters: int = 8, **kw):
        super().__init__(d, m, **kw)
        self.T_warmup = T_warmup
        self.kernel_scale = kernel_scale
        self.rho_max = rho_max
        self.n_clusters = n_clusters
        self._cholesky = None
        self._built = False

    def select_arms(self):
        if self.t == self.T_warmup and not self._built:
            self._built = True
            self._build_random_kernel()

        if not self._built:
            return super().select_arms()

        total = self.alphas + self.betas
        means = self.alphas / total
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        sigmas = np.sqrt(variances)

        z = self.np_rng.standard_normal(self.d)
        correlated_noise = self._cholesky @ z
        samples = means + sigmas * correlated_noise
        return list(np.argsort(samples)[::-1][:self.m])

    def _build_random_kernel(self):
        """Same RBF kernel construction as CorrCTS-Full, but random clusters."""
        cluster_of = self.np_rng.randint(0, self.n_clusters, size=self.d)
        n_cl = int(cluster_of.max()) + 1

        cluster_rank = np.arange(n_cl) / max(1, n_cl - 1)
        arm_rank = cluster_rank[cluster_of]

        diff = arm_rank[:, None] - arm_rank[None, :]
        dist2 = diff ** 2
        Sigma = self.rho_max * np.exp(-dist2 / (2 * self.kernel_scale ** 2))
        np.fill_diagonal(Sigma, 1.0)

        Sigma = 0.5 * (Sigma + Sigma.T)
        try:
            self._cholesky = np.linalg.cholesky(Sigma + 1e-3 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self._cholesky = np.eye(self.d)


# ─── Algorithm suite (7 total) ──────────────────────────────────────────
TIER5_ALGOS = {
    "CTS": ALL_ALGORITHMS["cts_baseline"],
    "CUCB": CUCB,
    "N1_corr_full": BREAKTHROUGH_ALGOS["N1_corr_cts_full"],
    "N4_robust_corr": BREAKTHROUGH_ALGOS["N4_robust_corr"],
    "ABLATION_random_corr": FINAL_ALGOS["ABLATION_random_corr"],
    "ABLATION_random_kernel": RandomKernelCTS,
}
NEEDS_ORACLE = {"N1_corr_full", "N4_robust_corr"}


def generate_configs() -> list[dict]:
    """16 configs: same as Tier 4."""
    configs = []
    cid = 0
    combos = [
        ("uniform", 0.20, 100),
        ("uniform", 0.10, 200),
        ("hard", 0.05, 300),
        ("staggered", 0.02, 400),
    ]
    for d in [25, 50]:
        for m in [3, 5]:
            for gap_type, delta, seed_offset in combos:
                configs.append({
                    "config_id": cid,
                    "d": d, "m": m,
                    "gap_type": gap_type,
                    "delta_min": delta,
                    "env_seed": seed_offset + cid,
                })
                cid += 1
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
        top_idx = np.argsort(means)[::-1][:m]
    else:
        raise ValueError(f"Unknown gap type: {gap_type}")

    optimal = np.argsort(means)[::-1][:m]
    optimal_reward = means[optimal].sum()
    return means, optimal, optimal_reward


def run_single_trial(algo_name, config, seed, T, out_dir):
    trial_id = f"c{config['config_id']}_s{seed}_{algo_name}"
    means, optimal, optimal_reward = build_env(config, seed)
    reward_rng = np.random.RandomState(config["env_seed"] * 10000 + seed + 999999)

    AlgoClass = TIER5_ALGOS[algo_name]
    trial_np_seed = (config["env_seed"] * 10_000 + seed * 37) % (2**31)
    kwargs = {"d": config["d"], "m": config["m"], "np_seed": trial_np_seed}

    oracle = None
    if algo_name in NEEDS_ORACLE:
        oracle = InstrumentedOracle(
            d=config["d"], m=config["m"], model=MODEL,
            trial_id=trial_id, config_id=config["config_id"],
            seed=seed, algo_name=algo_name,
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
        "trial_id": trial_id, "algo": algo_name, "model": MODEL,
        "config_id": config["config_id"], "seed": seed,
        "d": config["d"], "m": config["m"],
        "gap_type": config["gap_type"], "delta_min": config["delta_min"],
        "env_seed": config["env_seed"],
        "T": T, "final_regret": round(float(cum_regret), 2),
        "regret_curve": regret_curve, "elapsed_sec": round(elapsed, 2),
        "llm_calls": diag.get("total_calls", 0),
        "llm_tokens": diag.get("total_tokens", 0),
        "cache_hits": diag.get("cache_hits", 0),
        "optimal_set": [int(a) for a in optimal.tolist()],
    }


def analyze(all_results, out_dir):
    """Quick analysis — full analysis done locally."""
    import pandas as pd
    from scipy.stats import wilcoxon

    valid = [r for r in all_results if r.get("final_regret") is not None]
    df = pd.DataFrame(valid)

    lines = []
    lines.append("=" * 100)
    lines.append(f"TIER 5 EXTENDED — {len(valid)} trials | model={MODEL}")
    lines.append("=" * 100)
    lines.append(f"Algos: {df['algo'].nunique()} | Configs: {df['config_id'].nunique()} | "
                 f"Seeds: {df['seed'].nunique()} | T: {df['T'].iloc[0]}")

    # Global ranking
    stats = df.groupby("algo")["final_regret"].agg(["mean", "std", "count"])
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.sort_values("mean")

    cts_mean = stats.loc["CTS", "mean"] if "CTS" in stats.index else None

    lines.append("\n--- GLOBAL RANKING ---")
    for rank, (algo, row) in enumerate(stats.iterrows(), 1):
        vs_cts = f"+{(1 - row['mean']/cts_mean)*100:.1f}%" if cts_mean and algo != "CTS" else "---"
        lines.append(f"  {rank}  {algo:30s}  mean={row['mean']:.1f}  se={row['se']:.2f}  vs_CTS={vs_cts}")

    # Paired tests vs CTS
    lines.append("\n--- PAIRED TESTS vs CTS ---")
    cts_trials = {(r["config_id"], r["seed"]): r["final_regret"]
                  for r in valid if r["algo"] == "CTS"}
    for algo in stats.index:
        if algo == "CTS":
            continue
        algo_trials = {(r["config_id"], r["seed"]): r["final_regret"]
                       for r in valid if r["algo"] == algo}
        keys = sorted(set(cts_trials.keys()) & set(algo_trials.keys()))
        if len(keys) < 10:
            continue
        diffs = [cts_trials[k] - algo_trials[k] for k in keys]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        mean_diff = np.mean(diffs)
        try:
            _, p_wilcoxon = wilcoxon(diffs, alternative="two-sided")
        except Exception:
            p_wilcoxon = 1.0
        lines.append(f"  {algo:30s}  W/L={wins}/{losses}  diff={mean_diff:.1f}  "
                     f"wilcoxon_p={p_wilcoxon:.6f}")

    # Per (d, m) breakdown
    lines.append("\n--- PER-(d,m) BREAKDOWN ---")
    pivot = df.pivot_table(values="final_regret", index="algo",
                           columns=["d", "m"], aggfunc="mean")
    lines.append(pivot.to_string())

    # RandomKernel vs CorrCTS-Full (key new ablation)
    lines.append("\n--- KEY ABLATION: CorrCTS-Full vs RandomKernel ---")
    rk_trials = {(r["config_id"], r["seed"]): r["final_regret"]
                 for r in valid if r["algo"] == "ABLATION_random_kernel"}
    n1_trials = {(r["config_id"], r["seed"]): r["final_regret"]
                 for r in valid if r["algo"] == "N1_corr_full"}
    keys = sorted(set(rk_trials.keys()) & set(n1_trials.keys()))
    if keys:
        diffs = [rk_trials[k] - n1_trials[k] for k in keys]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        lines.append(f"  CorrCTS-Full beats RandomKernel: {wins}/{wins+losses} trials")
        lines.append(f"  Mean advantage: {np.mean(diffs):.1f} regret")
        try:
            _, p = wilcoxon(diffs, alternative="two-sided")
            lines.append(f"  Wilcoxon p = {p:.6f}")
        except Exception:
            pass

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=25000)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-cache", action="store_true", default=True)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"tier5_extended_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    InstrumentedOracle.set_log_file(out_dir / "llm_calls.jsonl")
    logger.info(f"Output: {out_dir}")

    configs = generate_configs()
    algos = list(TIER5_ALGOS.keys())
    tasks = [(a, c, s) for a in algos for c in configs for s in range(args.n_seeds)]
    import random
    random.shuffle(tasks)
    logger.info(f"Running {len(algos)} algos x {len(configs)} configs x {args.n_seeds} seeds "
                f"= {len(tasks)} trials (T={args.T}, model={MODEL})")

    all_results = []
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_trial, a, c, s, args.T, out_dir): (a, c, s)
                   for a, c, s in tasks}
        for future in as_completed(futures):
            algo, config, seed = futures[future]
            try:
                r = future.result()
                all_results.append(r)
                with open(out_dir / "raw_trials.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    rate = completed / max(time.time() - t_start, 0.1)
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(tasks)}] {algo:30s} c{config['config_id']} "
                        f"s{seed} r={r['final_regret']:.1f} ETA={eta/60:.1f}min"
                    )
            except Exception as e:
                logger.error(f"FAILED {algo} c{config['config_id']} s{seed}: {e}")

    total = time.time() - t_start
    logger.info(f"Done: {completed}/{len(tasks)} in {total/60:.1f}min")
    analyze(all_results, out_dir)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "total_trials": completed,
            "elapsed_min": round(total / 60, 1),
            "model": MODEL,
            "T": args.T,
            "n_seeds": args.n_seeds,
            "n_algos": len(algos),
        }, f, indent=2)


if __name__ == "__main__":
    main()
