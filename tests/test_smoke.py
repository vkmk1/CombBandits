"""Smoke tests to verify basic functionality."""
import numpy as np
import pytest

from combbandits.environments.synthetic import SyntheticBernoulliEnv
from combbandits.oracle.simulated import SimulatedCLO
from combbandits.agents.cucb import CUCBAgent
from combbandits.agents.cts import CTSAgent
from combbandits.agents.llm_cucb_at import LLMCUCBATAgent
from combbandits.agents.llm_greedy import LLMGreedyAgent
from combbandits.engine.trial import run_trial


@pytest.fixture
def env():
    e = SyntheticBernoulliEnv(d=20, m=5, gap_type="uniform", delta_min=0.1, seed=42)
    e.reset()
    return e


@pytest.fixture
def oracle(env):
    return SimulatedCLO(
        d=env.d, m=env.m,
        optimal_set=env.optimal_set,
        arm_means=env.means,
        corruption_type="uniform",
        epsilon=0.1, K=3, seed=42,
    )


def test_env_basic(env):
    assert env.d == 20
    assert env.m == 5
    assert len(env.optimal_set) == 5
    rewards = env.pull(env.optimal_set)
    assert len(rewards) == 5
    assert all(r in (0.0, 1.0) for r in rewards.values())


def test_cucb_runs(env):
    agent = CUCBAgent(d=env.d, m=env.m)
    result = run_trial(agent, env, T=100, seed=42)
    assert len(result.rounds) == 100
    assert result.cumulative_regret >= 0


def test_cts_runs(env):
    agent = CTSAgent(d=env.d, m=env.m)
    result = run_trial(agent, env, T=100, seed=42)
    assert len(result.rounds) == 100


def test_llm_cucb_at_runs(env, oracle):
    agent = LLMCUCBATAgent(
        d=env.d, m=env.m, oracle=oracle,
        arm_metadata=env.get_arm_metadata(),
    )
    result = run_trial(agent, env, T=100, seed=42)
    assert len(result.rounds) == 100
    assert result.cumulative_regret >= 0


def test_llm_greedy_runs(env, oracle):
    agent = LLMGreedyAgent(d=env.d, m=env.m, oracle=oracle)
    result = run_trial(agent, env, T=100, seed=42)
    assert len(result.rounds) == 100


def test_consistent_wrong_oracle(env):
    """LLM-CUCB-AT should not suffer linear regret with consistently wrong oracle."""
    oracle = SimulatedCLO(
        d=env.d, m=env.m,
        optimal_set=env.optimal_set,
        arm_means=env.means,
        corruption_type="consistent_wrong",
        epsilon=1.0, K=3, seed=42,
    )
    agent = LLMCUCBATAgent(
        d=env.d, m=env.m, oracle=oracle,
        arm_metadata=env.get_arm_metadata(),
    )
    result = run_trial(agent, env, T=500, seed=42)
    # Should not be catastrophically bad
    # LLM-Greedy with same oracle would have ~linear regret
    greedy = LLMGreedyAgent(d=env.d, m=env.m, oracle=oracle)
    greedy_result = run_trial(greedy, env, T=500, seed=42)
    # Our method should do better than blind greedy
    assert result.cumulative_regret < greedy_result.cumulative_regret


def test_oracle_consistency():
    """Consistency score computation."""
    from combbandits.oracle.base import CLOBase

    class DummyCLO(CLOBase):
        def query(self, *a, **kw):
            pass

    clo = DummyCLO(d=10, m=3, K=3)
    # Perfect agreement
    assert clo.compute_consistency([[0,1,2], [0,1,2], [0,1,2]]) == 1.0
    # No agreement
    assert clo.compute_consistency([[0,1,2], [3,4,5], [6,7,8]]) == 0.0
    # Partial agreement
    assert clo.compute_consistency([[0,1,2], [0,1,3], [0,1,4]]) == pytest.approx(2/3)
