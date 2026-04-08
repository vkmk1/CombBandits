"""Influence maximization environment on SNAP social graphs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .base import CombBanditEnv

try:
    import networkx as nx
except ImportError:
    nx = None


class InfluenceMaxEnv(CombBanditEnv):
    """Influence maximization as a combinatorial semi-bandit.

    Arms are candidate seed nodes in a social network. Selecting m seeds and
    running independent cascade yields a reward = spread / |V|.
    """

    def __init__(
        self,
        d: int = 200,
        m: int = 10,
        graph_name: str = "ego-facebook",
        data_dir: str = "data/snap",
        cascade_prob: float = 0.01,
        mc_rollouts: int = 100,
        seed: int = 0,
    ):
        super().__init__(d=d, m=m, seed=seed)
        self.graph_name = graph_name
        self.data_dir = Path(data_dir)
        self.cascade_prob = cascade_prob
        self.mc_rollouts = mc_rollouts
        self._graph: nx.Graph | None = None
        self._candidate_nodes: list[int] = []
        self._node_metadata: list[dict] = []
        self._spread_cache: dict[frozenset, float] = {}

    def _load_graph(self):
        if nx is None:
            raise ImportError("networkx required for InfluenceMaxEnv")

        edge_file = self.data_dir / f"{self.graph_name}.txt"
        if not edge_file.exists():
            raise FileNotFoundError(
                f"Graph not found at {edge_file}. "
                "Download from https://snap.stanford.edu/data/ "
                "See README for instructions."
            )
        self._graph = nx.read_edgelist(str(edge_file), nodetype=int)

    def reset(self) -> None:
        if self._graph is None:
            self._load_graph()

        # Select top-d nodes by degree as candidate seed set
        degree_ranked = sorted(self._graph.degree(), key=lambda x: x[1], reverse=True)
        self._candidate_nodes = [n for n, _ in degree_ranked[:self.d]]

        # Precompute node metadata
        self._node_metadata = []
        for i, node in enumerate(self._candidate_nodes):
            self._node_metadata.append({
                "arm_id": i,
                "node_id": node,
                "degree": self._graph.degree(node),
                "label": f"node_{node}",
            })

        # Estimate mean spread for each individual node (for approximate means)
        self._means = np.zeros(self.d)
        for i, node in enumerate(self._candidate_nodes):
            spreads = [
                self._independent_cascade([node]) for _ in range(min(20, self.mc_rollouts))
            ]
            self._means[i] = np.mean(spreads)

        self._spread_cache.clear()
        self._optimal_set = None

    def _independent_cascade(self, seeds: list[int]) -> float:
        """Run one Monte Carlo rollout of independent cascade."""
        activated = set(seeds)
        frontier = list(seeds)
        while frontier:
            new_frontier = []
            for node in frontier:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in activated:
                        if self.rng.random() < self.cascade_prob:
                            activated.add(neighbor)
                            new_frontier.append(neighbor)
            frontier = new_frontier
        return len(activated) / self._graph.number_of_nodes()

    def pull(self, selected_set: list[int]) -> dict[int, float]:
        """Pull = run cascade with selected seeds, return per-arm marginal contribution."""
        assert len(selected_set) == self.m
        seed_nodes = [self._candidate_nodes[i] for i in selected_set]

        # Run MC rollouts for total spread
        total_spreads = []
        for _ in range(self.mc_rollouts):
            total_spreads.append(self._independent_cascade(seed_nodes))
        total_spread = np.mean(total_spreads)

        # Approximate per-arm reward via leave-one-out marginal contribution
        rewards = {}
        for arm_idx in selected_set:
            other_seeds = [self._candidate_nodes[i] for i in selected_set if i != arm_idx]
            leave_out_spreads = []
            for _ in range(min(10, self.mc_rollouts)):
                leave_out_spreads.append(self._independent_cascade(other_seeds))
            marginal = total_spread - np.mean(leave_out_spreads)
            rewards[arm_idx] = max(0.0, marginal)

        return rewards

    def _sample_reward(self, arm: int) -> float:
        node = self._candidate_nodes[arm]
        return self._independent_cascade([node])

    def get_arm_metadata(self) -> list[dict]:
        return self._node_metadata

    def instantaneous_regret(self, selected_set: list[int]) -> float:
        # For influence max, use approximate regret from cached means
        selected_reward = sum(self._means[i] for i in selected_set)
        return self.optimal_reward - selected_reward


class InfluenceMaxEnvSimulated(CombBanditEnv):
    """Simulated influence maximization when real graph data is unavailable.

    Uses a random graph model with planted community structure.
    Seed quality correlates with degree and community centrality.
    """

    def __init__(self, d: int = 200, m: int = 10, n_communities: int = 5, seed: int = 0):
        super().__init__(d=d, m=m, seed=seed)
        self.n_communities = n_communities

    def reset(self) -> None:
        # Assign nodes to communities
        communities = self.rng.randint(0, self.n_communities, size=self.d)

        # Simulate degree distribution (power law-ish)
        degrees = self.rng.pareto(1.5, size=self.d) + 1
        degrees = degrees / degrees.max()

        # Seed quality: high-degree nodes in large communities are better
        community_sizes = np.bincount(communities, minlength=self.n_communities)
        community_quality = community_sizes / community_sizes.max()

        self._means = np.clip(
            0.3 * degrees + 0.5 * community_quality[communities] + self.rng.normal(0, 0.05, self.d),
            0.01, 0.99,
        )
        self._optimal_set = None

    def _sample_reward(self, arm: int) -> float:
        # Sub-Gaussian noise around mean
        return float(np.clip(self._means[arm] + self.rng.normal(0, 0.1), 0, 1))
