"""GPU-batched simulated CLO.

Generates oracle responses for all n_seeds simultaneously using tensor operations.
Each seed gets independent corruption draws.
"""
from __future__ import annotations

import torch


class BatchedSimulatedCLO:
    """Batched simulated oracle: generates suggestions for all seeds in parallel."""

    def __init__(self, d: int, m: int, n_seeds: int, optimal_set: torch.Tensor,
                 arm_means: torch.Tensor, corruption_type: str = "uniform",
                 epsilon: float = 0.0, K: int = 3, device: torch.device | None = None):
        self.d = d
        self.m = m
        self.n_seeds = n_seeds
        self.K = K
        self.corruption_type = corruption_type
        self.epsilon = epsilon
        self.device = device or torch.device("cpu")
        self.total_queries = 0

        # Store optimal set as tensor: (m,)
        self.optimal_set = optimal_set.to(self.device)

        means = arm_means.to(self.device)
        opt_mask = torch.zeros(d, dtype=torch.bool, device=self.device)
        opt_mask[self.optimal_set] = True

        if corruption_type == "adversarial":
            masked = means.clone()
            masked[opt_mask] = float("inf")
            self._bad_set = torch.argsort(masked)[:m]
        else:
            masked = means.clone()
            masked[opt_mask] = -float("inf")
            self._bad_set = torch.argsort(masked, descending=True)[:m]

    def _generate_sets_batched(self) -> torch.Tensor:
        """Generate one set per seed: (n_seeds, m).

        Returns arm indices for each seed's oracle suggestion.
        """
        if self.corruption_type == "uniform":
            corrupt_mask = torch.rand(self.n_seeds, device=self.device) < self.epsilon
            result = self.optimal_set.unsqueeze(0).expand(self.n_seeds, -1).clone()
            n_corrupt = corrupt_mask.sum().item()
            if n_corrupt > 0:
                # Vectorized random set generation via argsort of random values
                rand_scores = torch.rand(n_corrupt, self.d, device=self.device)
                random_sets = torch.topk(rand_scores, self.m, dim=1).indices
                result[corrupt_mask] = random_sets
            return result

        elif self.corruption_type == "adversarial":
            corrupt_mask = torch.rand(self.n_seeds, device=self.device) < self.epsilon
            result = self.optimal_set.unsqueeze(0).expand(self.n_seeds, -1).clone()
            result[corrupt_mask] = self._bad_set.unsqueeze(0).expand(corrupt_mask.sum().item(), -1)
            return result

        elif self.corruption_type == "partial_overlap":
            n_correct = max(0, int((1 - self.epsilon) * self.m))
            n_wrong = self.m - n_correct
            # Vectorized sampling via argsort of random scores
            rand_opt = torch.rand(self.n_seeds, self.m, device=self.device)
            correct = self.optimal_set[torch.topk(rand_opt, n_correct, dim=1).indices]
            all_arms = torch.arange(self.d, device=self.device)
            opt_mask = torch.zeros(self.d, dtype=torch.bool, device=self.device)
            opt_mask[self.optimal_set] = True
            non_opt = all_arms[~opt_mask]
            rand_non = torch.rand(self.n_seeds, len(non_opt), device=self.device)
            wrong = non_opt[torch.topk(rand_non, n_wrong, dim=1).indices]
            return torch.cat([correct, wrong], dim=1)

        elif self.corruption_type == "consistent_wrong":
            # All seeds get the same bad set
            return self._bad_set.unsqueeze(0).expand(self.n_seeds, -1).clone()

        raise ValueError(f"Unknown corruption_type: {self.corruption_type}")

    def query_batched(self, mu_hat: torch.Tensor) -> dict:
        """Query oracle for all seeds.

        Args:
            mu_hat: (n_seeds, d) current empirical means

        Returns:
            dict with:
                suggested_sets: (n_seeds, m) primary suggestions
                consistency: (n_seeds,) kappa scores
        """
        self.total_queries += self.K * self.n_seeds

        # Generate K sets per seed
        all_sets = [self._generate_sets_batched() for _ in range(self.K)]  # K × (n_seeds, m)
        primary = all_sets[0]  # (n_seeds, m)

        # Compute consistency via boolean masks
        # Convert each set to a (n_seeds, d) binary mask, then intersect
        masks = []
        for sets in all_sets:
            mask = torch.zeros(self.n_seeds, self.d, dtype=torch.bool, device=self.device)
            mask.scatter_(1, sets, True)
            masks.append(mask)

        # Intersection: all K masks must be True
        intersection = masks[0]
        for mask in masks[1:]:
            intersection = intersection & mask
        kappa = intersection.sum(dim=1).float() / self.m  # (n_seeds,)

        return {
            "suggested_sets": primary,
            "consistency": kappa,
        }
