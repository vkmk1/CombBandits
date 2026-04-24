"""Classical CUCB (Combinatorial Upper Confidence Bound) baseline.

Reference: Chen, Wang, Yuan. "Combinatorial Multi-Armed Bandit" (ICML 2013).
Extended JMLR 2016 version: arxiv 1407.8339.

Standard baseline for combinatorial semi-bandits. Must be in any paper
claiming improvement over combinatorial bandit state-of-the-art.
"""
from __future__ import annotations

import math
import numpy as np

from algorithms import CTSBase


class CUCB(CTSBase):
    """Combinatorial UCB.

    For each arm: UCB index = mu_hat + sqrt(3 * log(t) / (2 * N_i))
    (Chen et al 2013 eq 5, with alpha=3 for Bernoulli)
    Play top-m arms by UCB index.
    """
    name = "CUCB"

    def __init__(self, d, m, np_seed: int = 0, **kw):
        super().__init__(d, m, np_seed=np_seed, **kw)

    def select_arms(self):
        # Must pull each arm once before UCB is defined
        n = np.maximum(self.n_pulls, 1)
        t = max(self.t, 1)
        cb = np.sqrt(3 * math.log(t) / (2 * n))
        # Unpulled arms get infinite UCB (force exploration)
        cb = np.where(self.n_pulls == 0, np.inf, cb)
        ucb = self.mu_hat + cb
        return list(np.argsort(ucb)[::-1][:self.m])
