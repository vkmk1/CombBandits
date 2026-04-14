"""Single trial execution: run one (agent, environment, seed) combination."""
from __future__ import annotations

import logging
import time
import numpy as np

from ..agents.base import Agent
from ..environments.base import CombBanditEnv
from ..types import RoundResult, TrialResult

logger = logging.getLogger(__name__)


def run_trial(
    agent: Agent,
    env: CombBanditEnv,
    T: int,
    seed: int = 0,
    log_interval: int = 1000,
) -> TrialResult:
    """Run a single trial of T rounds.

    Args:
        agent: The bandit agent to evaluate.
        env: The combinatorial semi-bandit environment.
        T: Number of rounds (horizon).
        seed: Random seed for reproducibility.
        log_interval: Log progress every this many rounds.

    Returns:
        TrialResult with per-round data.
    """
    np.random.seed(seed)
    env.reset()
    agent.reset()

    # Pass arm metadata to agent if it supports it
    if hasattr(agent, 'arm_metadata') and agent.arm_metadata is None:
        agent.arm_metadata = env.get_arm_metadata()

    result = TrialResult(
        agent_name=agent.name,
        env_name=type(env).__name__,
        seed=seed,
        d=env.d,
        m=env.m,
        T=T,
    )

    cumulative_regret = 0.0
    start_time = time.time()

    for t in range(T):
        # Agent selects arms
        selected = agent.select_arms()

        # Environment generates rewards
        rewards = env.pull(selected)

        # Compute regret
        inst_regret = env.instantaneous_regret(selected)
        cumulative_regret += inst_regret

        # Build round result
        rr = RoundResult(
            round_t=t,
            selected_set=selected,
            rewards=rewards,
            total_reward=sum(rewards.values()),
            instantaneous_regret=inst_regret,
            cumulative_regret=cumulative_regret,
        )

        # Add diagnostics for LLM-CUCB-AT
        if hasattr(agent, 'trust_history') and agent.trust_history:
            last_trust = agent.trust_history[-1]
            rr.kappa_t = last_trust.get("kappa")
            rr.rho_t = last_trust.get("rho")
            rr.trust_score = last_trust.get("tau")
            rr.hedge_size = last_trust.get("hedge_size", 0)

        if hasattr(agent, '_last_oracle_response') and agent._last_oracle_response is not None:
            rr.llm_suggestion = agent._last_oracle_response.suggested_set

        if hasattr(agent, '_force_fallback'):
            rr.is_fallback = agent._force_fallback

        result.rounds.append(rr)

        # Agent updates
        agent.update(selected, rewards)

        if (t + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"[{agent.name}] Round {t+1}/{T} | "
                f"Cum. regret: {cumulative_regret:.1f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"[{agent.name}] Trial complete. "
        f"Final regret: {cumulative_regret:.1f} | "
        f"Time: {elapsed:.1f}s"
    )

    return result
