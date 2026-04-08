"""Experiment runner: orchestrates trials across agents, environments, and seeds."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..agents import AGENT_REGISTRY
from ..environments.synthetic import SyntheticBernoulliEnv
from ..environments.mind import MINDEnvSimulated
from ..environments.influence_max import InfluenceMaxEnvSimulated
from ..oracle.simulated import SimulatedCLO
from ..types import TrialResult
from .trial import run_trial

logger = logging.getLogger(__name__)

ENV_REGISTRY = {
    "synthetic_bernoulli": SyntheticBernoulliEnv,
    "mind_simulated": MINDEnvSimulated,
    "influence_max_simulated": InfluenceMaxEnvSimulated,
}


def _make_oracle(env, oracle_cfg: dict, seed: int):
    """Create a CLO from config."""
    oracle_type = oracle_cfg.get("type", "simulated")
    if oracle_type == "simulated":
        return SimulatedCLO(
            d=env.d,
            m=env.m,
            optimal_set=env.optimal_set,
            arm_means=env.means,
            corruption_type=oracle_cfg.get("corruption_type", "uniform"),
            epsilon=oracle_cfg.get("epsilon", 0.0),
            K=oracle_cfg.get("K", 3),
            seed=seed + 9999,
        )
    elif oracle_type == "llm":
        from ..oracle.llm_oracle import LLMOracle
        return LLMOracle(
            d=env.d,
            m=env.m,
            K=oracle_cfg.get("K", 3),
            primary_model=oracle_cfg.get("primary_model", "gpt-4o"),
            requery_model=oracle_cfg.get("requery_model", "gpt-4o-mini"),
            provider=oracle_cfg.get("provider", "openai"),
            temperature=oracle_cfg.get("temperature", 0.7),
        )
    raise ValueError(f"Unknown oracle type: {oracle_type}")


def _run_single(task: dict) -> dict:
    """Run a single (agent, env, oracle, seed) trial. Called in subprocess."""
    # Reconstruct objects from config
    env_cfg = task["env"]
    env_cls = ENV_REGISTRY[env_cfg["type"]]
    env = env_cls(**{k: v for k, v in env_cfg.items() if k != "type"}, seed=task["seed"])
    env.reset()

    oracle_cfg = task.get("oracle", {"type": "simulated", "epsilon": 0.0})
    oracle = _make_oracle(env, oracle_cfg, task["seed"])

    agent_name = task["agent"]
    agent_cfg = task.get("agent_config", {})
    agent_cls = AGENT_REGISTRY[agent_name]

    # Determine which agents need an oracle
    needs_oracle = agent_name in {"llm_cucb_at", "llm_greedy", "ellm_adapted", "opro_bandit", "warm_start_cts", "exp4"}
    if needs_oracle:
        agent_cfg["oracle"] = oracle
        agent_cfg["arm_metadata"] = env.get_arm_metadata()

    agent = agent_cls(d=env.d, m=env.m, **agent_cfg)

    result = run_trial(agent, env, T=task["T"], seed=task["seed"], log_interval=task.get("log_interval", 5000))

    # Serialize to dict for cross-process return
    return {
        "agent": agent_name,
        "env": env_cfg["type"],
        "seed": task["seed"],
        "corruption_type": oracle_cfg.get("corruption_type", "none"),
        "epsilon": oracle_cfg.get("epsilon", 0.0),
        "d": env.d,
        "m": env.m,
        "T": task["T"],
        "final_regret": result.cumulative_regret,
        "regret_curve": result.regret_curve.tolist(),
        "oracle_queries": oracle.total_queries if hasattr(oracle, "total_queries") else 0,
        "oracle_tokens": oracle.total_tokens if hasattr(oracle, "total_tokens") else 0,
    }


class ExperimentRunner:
    """Orchestrate experiments from a YAML config file."""

    def __init__(self, config_path: str, output_dir: str = "results"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_task_grid(self) -> list[dict]:
        """Expand config into individual trial tasks."""
        tasks = []
        exp = self.config

        agents = exp.get("agents", ["cucb"])
        envs = exp.get("environments", [{"type": "synthetic_bernoulli", "d": 100, "m": 10}])
        seeds = list(range(exp.get("n_seeds", 20)))
        T = exp.get("T", 10000)
        oracles = exp.get("oracles", [{"type": "simulated", "epsilon": 0.0}])
        log_interval = exp.get("log_interval", 5000)

        for agent_name in agents:
            for env_cfg in envs:
                for oracle_cfg in oracles:
                    # Skip oracle config for agents that don't use it
                    needs_oracle = agent_name in {
                        "llm_cucb_at", "llm_greedy", "ellm_adapted",
                        "opro_bandit", "warm_start_cts", "exp4",
                    }
                    if not needs_oracle:
                        # Only run non-oracle agents once (not per oracle config)
                        if oracle_cfg != oracles[0]:
                            continue

                    for seed in seeds:
                        tasks.append({
                            "agent": agent_name,
                            "env": env_cfg,
                            "oracle": oracle_cfg if needs_oracle else {"type": "simulated", "epsilon": 0.0},
                            "seed": seed,
                            "T": T,
                            "agent_config": exp.get("agent_configs", {}).get(agent_name, {}),
                            "log_interval": log_interval,
                        })
        return tasks

    def run(self, max_workers: int | None = None, task_indices: list[int] | None = None):
        """Run all trials, optionally restricted to specific task indices (for SLURM array)."""
        tasks = self.build_task_grid()

        if task_indices is not None:
            tasks = [tasks[i] for i in task_indices if i < len(tasks)]

        logger.info(f"Running {len(tasks)} trials with {max_workers or 'auto'} workers")

        results = []
        if max_workers == 1:
            # Single-process mode (for debugging)
            for task in tasks:
                result = _run_single(task)
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_single, task): i for i, task in enumerate(tasks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(
                            f"[{idx+1}/{len(tasks)}] {result['agent']} | "
                            f"eps={result['epsilon']} | seed={result['seed']} | "
                            f"regret={result['final_regret']:.1f}"
                        )
                    except Exception as e:
                        logger.error(f"Task {idx} failed: {e}")

        # Save results
        exp_name = self.config.get("name", "experiment")
        out_path = self.output_dir / f"{exp_name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")

        return results

    def export_task_list(self, path: str | None = None) -> str:
        """Export task grid as CSV for SLURM array job."""
        tasks = self.build_task_grid()
        path = path or str(self.output_dir / "task_list.csv")
        with open(path, "w") as f:
            f.write("idx,agent,env_type,corruption_type,epsilon,seed,T\n")
            for i, task in enumerate(tasks):
                f.write(
                    f"{i},{task['agent']},{task['env']['type']},"
                    f"{task['oracle'].get('corruption_type','none')},"
                    f"{task['oracle'].get('epsilon',0.0)},"
                    f"{task['seed']},{task['T']}\n"
                )
        logger.info(f"Task list ({len(tasks)} tasks) written to {path}")
        return path
