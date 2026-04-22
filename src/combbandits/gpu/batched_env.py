"""GPU-batched combinatorial semi-bandit environments.

All operations are vectorized over n_seeds. Rewards are sampled for all
seeds simultaneously via torch.bernoulli.
"""
from __future__ import annotations

import torch
import numpy as np


class BatchedEnvBase:
    """Base class for batched environments sharing the pull/regret interface."""

    def __init__(self, d: int, m: int, n_seeds: int, base_seed: int = 0,
                 device: torch.device | None = None):
        self.d = d
        self.m = m
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.device = device or torch.device("cpu")

        self.means: torch.Tensor | None = None
        self.optimal_set: torch.Tensor | None = None
        self.optimal_reward: float = 0.0

    def reset(self) -> None:
        raise NotImplementedError

    def _finalize_means(self, means_np: np.ndarray) -> None:
        self.means = torch.tensor(means_np, dtype=torch.float32, device=self.device)
        opt_idx = torch.argsort(self.means, descending=True)[:self.m]
        self.optimal_set = opt_idx
        self.optimal_reward = self.means[opt_idx].sum().item()

    def pull_batched(self, selected: torch.Tensor) -> torch.Tensor:
        selected_means = self.means[selected]
        return torch.bernoulli(selected_means)

    def instantaneous_regret_batched(self, selected: torch.Tensor) -> torch.Tensor:
        selected_reward = self.means[selected].sum(dim=1)
        return self.optimal_reward - selected_reward


class BatchedSyntheticBernoulliEnv(BatchedEnvBase):
    """Batched Bernoulli semi-bandit: all n_seeds share the same arm means."""

    def __init__(self, d: int, m: int, n_seeds: int, gap_type: str = "uniform",
                 delta_min: float = 0.05, base_seed: int = 0, device: torch.device | None = None):
        super().__init__(d, m, n_seeds, base_seed, device)
        self.gap_type = gap_type
        self.delta_min = delta_min

    def reset(self) -> None:
        rng = np.random.RandomState(self.base_seed)

        if self.gap_type == "uniform":
            means_np = rng.uniform(0.1, 0.5, size=self.d)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.5 + self.delta_min
        elif self.gap_type == "hard":
            means_np = np.full(self.d, 0.5)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.5 + self.delta_min
        elif self.gap_type == "graded":
            means_np = np.linspace(0.1, 0.5 + self.delta_min, self.d)
            rng.shuffle(means_np)
        elif self.gap_type == "clustered":
            means_np = np.full(self.d, 0.3)
            top_arms = rng.choice(self.d, size=self.m, replace=False)
            means_np[top_arms] = 0.7
            means_np += rng.normal(0, 0.02, size=self.d)
            means_np = np.clip(means_np, 0.01, 0.99)
        else:
            raise ValueError(f"Unknown gap_type: {self.gap_type}")

        self._finalize_means(means_np)

    def get_arm_metadata(self) -> list[dict]:
        return [{"arm_id": i, "label": f"arm_{i}", "category": f"group_{i % 5}"}
                for i in range(self.d)]


class BatchedMINDEnvSimulated(BatchedEnvBase):
    """GPU-batched simulated MIND-like environment.

    Reproduces MINDEnvSimulated's reward structure: articles have categories,
    user preferences create structured click probabilities.
    """

    def __init__(self, d: int = 200, m: int = 5, n_seeds: int = 30,
                 n_categories: int = 10, base_seed: int = 0,
                 device: torch.device | None = None):
        super().__init__(d, m, n_seeds, base_seed, device)
        self.n_categories = n_categories

    def reset(self) -> None:
        rng = np.random.RandomState(self.base_seed)
        categories = rng.randint(0, self.n_categories, size=self.d)
        user_pref = rng.dirichlet(np.ones(self.n_categories))
        base_quality = rng.beta(2, 5, size=self.d)
        category_boost = user_pref[categories]
        means_np = np.clip(
            0.3 * base_quality + 0.7 * category_boost + rng.normal(0, 0.05, self.d),
            0.01, 0.99,
        )
        self._finalize_means(means_np)


class BatchedInfluenceMaxEnvSimulated(BatchedEnvBase):
    """GPU-batched simulated influence maximization environment.

    Reproduces InfluenceMaxEnvSimulated's reward structure: nodes in communities
    with degree-based quality.
    """

    def __init__(self, d: int = 200, m: int = 10, n_seeds: int = 30,
                 n_communities: int = 5, base_seed: int = 0,
                 device: torch.device | None = None):
        super().__init__(d, m, n_seeds, base_seed, device)
        self.n_communities = n_communities

    def reset(self) -> None:
        rng = np.random.RandomState(self.base_seed)
        communities = rng.randint(0, self.n_communities, size=self.d)
        degrees = rng.pareto(1.5, size=self.d) + 1
        degrees = degrees / degrees.max()
        community_sizes = np.bincount(communities, minlength=self.n_communities)
        community_quality = community_sizes / community_sizes.max()
        means_np = np.clip(
            0.3 * degrees + 0.5 * community_quality[communities] + rng.normal(0, 0.05, self.d),
            0.01, 0.99,
        )
        self._finalize_means(means_np)


ENV_REGISTRY: dict[str, type[BatchedEnvBase]] = {
    "synthetic_bernoulli": BatchedSyntheticBernoulliEnv,
    "mind_simulated": BatchedMINDEnvSimulated,
    "influence_max_simulated": BatchedInfluenceMaxEnvSimulated,
}


def build_batched_env(env_cfg: dict, n_seeds: int, device: torch.device) -> BatchedEnvBase:
    """Build a batched environment from a config dict."""
    env_type = env_cfg.get("type", "synthetic_bernoulli")
    cls = ENV_REGISTRY.get(env_type)
    if cls is None:
        raise ValueError(f"No GPU-batched env for type '{env_type}'. Available: {list(ENV_REGISTRY)}")
    kwargs = {
        "d": env_cfg.get("d", 100),
        "m": env_cfg.get("m", 10),
        "n_seeds": n_seeds,
        "base_seed": env_cfg.get("seed", 0),
        "device": device,
    }
    if env_type == "synthetic_bernoulli":
        kwargs["gap_type"] = env_cfg.get("gap_type", "uniform")
        kwargs["delta_min"] = env_cfg.get("delta_min", 0.05)
    elif env_type == "mind_simulated":
        kwargs["n_categories"] = env_cfg.get("n_categories", 10)
    elif env_type == "influence_max_simulated":
        kwargs["n_communities"] = env_cfg.get("n_communities", 5)
    return cls(**kwargs)
