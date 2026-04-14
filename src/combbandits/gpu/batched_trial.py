"""GPU-batched trial execution: run all seeds simultaneously on one device.

A single call to run_batched_trial executes the full bandit loop for
n_seeds in parallel. Returns results in the same JSON-compatible format
as the CPU runner for seamless integration with the analysis pipeline.
"""
from __future__ import annotations

import logging
import time
import math

import torch
import numpy as np

from .batched_env import BatchedSyntheticBernoulliEnv
from .batched_oracle import BatchedSimulatedCLO
from .batched_agents import (
    BATCHED_AGENT_REGISTRY, NEEDS_ORACLE, BatchedLLMCUCBAT,
)
from .device import get_device

logger = logging.getLogger(__name__)


def run_batched_trial(
    agent_name: str,
    env_cfg: dict,
    oracle_cfg: dict,
    T: int,
    n_seeds: int = 100,
    device: torch.device | None = None,
    log_interval: int = 10000,
    agent_config: dict | None = None,
) -> list[dict]:
    """Run one (agent, env, oracle) config across n_seeds simultaneously on GPU.

    Returns a list of n_seeds result dicts, one per seed, in the same format
    as the CPU runner's _run_single output.
    """
    device = device or get_device()
    agent_config = agent_config or {}

    # --- Build environment ---
    env = BatchedSyntheticBernoulliEnv(
        d=env_cfg.get("d", 100),
        m=env_cfg.get("m", 10),
        n_seeds=n_seeds,
        gap_type=env_cfg.get("gap_type", "uniform"),
        delta_min=env_cfg.get("delta_min", 0.05),
        base_seed=env_cfg.get("seed", 0),
        device=device,
    )
    env.reset()
    d, m = env.d, env.m

    # --- Build oracle ---
    oracle = BatchedSimulatedCLO(
        d=d, m=m, n_seeds=n_seeds,
        optimal_set=env.optimal_set,
        arm_means=env.means,
        corruption_type=oracle_cfg.get("corruption_type", "uniform"),
        epsilon=oracle_cfg.get("epsilon", 0.0),
        K=oracle_cfg.get("K", 3),
        device=device,
    )

    # --- Build agent ---
    agent_cls = BATCHED_AGENT_REGISTRY[agent_name]
    kwargs = {"d": d, "m": m, "n_seeds": n_seeds, "device": device}
    if agent_name in NEEDS_ORACLE:
        kwargs["oracle"] = oracle
    kwargs.update(agent_config)
    agent = agent_cls(**kwargs)

    # --- Run trial ---
    # Regret tracking: (n_seeds, T) would be huge; store subsampled + final
    n_checkpoints = min(T, 500)
    checkpoint_indices = np.linspace(0, T - 1, n_checkpoints, dtype=int)
    checkpoint_set = set(checkpoint_indices.tolist())

    cum_regret = torch.zeros(n_seeds, device=device)
    regret_checkpoints = torch.zeros(n_seeds, n_checkpoints, device=device)
    cp_idx = 0

    start_time = time.time()

    for t in range(T):
        # Select arms: (n_seeds, m)
        selected = agent.select_arms()

        # Sample rewards: (n_seeds, m)
        rewards = env.pull_batched(selected)

        # Compute regret
        inst_regret = env.instantaneous_regret_batched(selected)  # (n_seeds,)
        cum_regret += inst_regret

        # Checkpoint
        if t in checkpoint_set:
            regret_checkpoints[:, cp_idx] = cum_regret
            cp_idx += 1

        # Update agent
        agent.update(selected, rewards)

        if (t + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            mean_regret = cum_regret.mean().item()
            logger.info(
                f"[{agent_name}] Round {t+1}/{T} | "
                f"Mean regret: {mean_regret:.1f} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Device: {device}"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"[{agent_name}] Batched trial done ({n_seeds} seeds). "
        f"Mean final regret: {cum_regret.mean().item():.1f} | "
        f"Time: {elapsed:.1f}s"
    )

    # --- Build result dicts (same format as CPU runner) ---
    regret_np = regret_checkpoints.cpu().numpy()
    cum_regret_np = cum_regret.cpu().numpy()

    # Trust diagnostics for LLM-CUCB-AT
    trust_kappa = None
    trust_rho = None
    trust_tau = None
    hedge_sizes = None
    if isinstance(agent, BatchedLLMCUCBAT):
        trust_kappa = agent.kappa_history
        trust_rho = agent.rho_history
        trust_tau = agent.tau_history
        hedge_sizes = agent.hedge_history

    results = []
    for seed_idx in range(n_seeds):
        results.append({
            "agent": agent_name,
            "env": env_cfg.get("type", "synthetic_bernoulli"),
            "seed": seed_idx,
            "corruption_type": oracle_cfg.get("corruption_type", "none"),
            "epsilon": oracle_cfg.get("epsilon", 0.0),
            "d": d,
            "m": m,
            "T": T,
            "final_regret": float(cum_regret_np[seed_idx]),
            "regret_curve": regret_np[seed_idx].tolist(),
            "oracle_queries": oracle.total_queries // n_seeds,
            "oracle_tokens": 0,
            # Trust diagnostics are per-round means across seeds (not per-seed)
            "trust_kappa": trust_kappa,
            "trust_rho": trust_rho,
            "trust_tau": trust_tau,
            "hedge_sizes": hedge_sizes,
            "oracle_overlap_fractions": None,
            "regret_100": regret_np[seed_idx][
                np.linspace(0, len(regret_np[seed_idx]) - 1, 100, dtype=int)
            ].tolist() if len(regret_np[seed_idx]) > 100 else regret_np[seed_idx].tolist(),
            "gpu_batched": True,
            "device": str(device),
        })

    return results


def run_batched_experiment(config: dict, device: torch.device | None = None) -> list[dict]:
    """Run a full experiment config using the batched GPU runner.

    Same config format as the CPU ExperimentRunner, but runs each
    (agent, env, oracle) group as a single batched trial.

    Returns flat list of per-seed result dicts.
    """
    device = device or get_device()
    logger.info(f"GPU batched experiment on device: {device}")

    agents = config.get("agents", ["cucb"])
    envs = config.get("environments", [{"type": "synthetic_bernoulli", "d": 100, "m": 10}])
    oracles = config.get("oracles", [{"type": "simulated", "epsilon": 0.0}])
    T = config.get("T", 10000)
    n_seeds = config.get("n_seeds", 100)
    log_interval = config.get("log_interval", 10000)
    agent_configs = config.get("agent_configs", {})

    all_results = []
    total_groups = 0

    for agent_name in agents:
        for env_cfg in envs:
            oracle_list = oracles if agent_name in NEEDS_ORACLE else [oracles[0]]
            for oracle_cfg in oracle_list:
                total_groups += 1

    group_idx = 0
    start_all = time.time()

    for agent_name in agents:
        for env_cfg in envs:
            oracle_list = oracles if agent_name in NEEDS_ORACLE else [oracles[0]]
            for oracle_cfg in oracle_list:
                group_idx += 1
                corr = oracle_cfg.get("corruption_type", "none")
                eps = oracle_cfg.get("epsilon", 0.0)
                d_val = env_cfg.get("d", 100)
                logger.info(
                    f"[{group_idx}/{total_groups}] {agent_name} | "
                    f"d={d_val} | {corr} eps={eps} | "
                    f"{n_seeds} seeds on {device}"
                )

                results = run_batched_trial(
                    agent_name=agent_name,
                    env_cfg=env_cfg,
                    oracle_cfg=oracle_cfg,
                    T=T,
                    n_seeds=n_seeds,
                    device=device,
                    log_interval=log_interval,
                    agent_config=agent_configs.get(agent_name, {}),
                )
                all_results.extend(results)

    elapsed = time.time() - start_all
    logger.info(
        f"Experiment complete: {total_groups} groups × {n_seeds} seeds = "
        f"{len(all_results)} trials in {elapsed:.1f}s"
    )
    return all_results
