"""MIND news recommendation environment for combinatorial semi-bandits."""
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
from .base import CombBanditEnv


class MINDEnv(CombBanditEnv):
    """MIND dataset environment: select m articles from d candidates per user session.

    Rewards are binary click signals. Each reset() loads a new user session.
    Requires pre-downloaded MIND dataset (see README).
    """

    def __init__(
        self,
        d: int = 200,
        m: int = 5,
        data_dir: str = "data/mind",
        split: str = "train",
        seed: int = 0,
    ):
        super().__init__(d=d, m=m, seed=seed)
        self.data_dir = Path(data_dir)
        self.split = split
        self._sessions: list[dict] | None = None
        self._current_session: dict | None = None
        self._session_idx = 0
        self._article_pool: list[dict] = []
        self._click_probs: np.ndarray | None = None

    def _load_data(self):
        """Load MIND sessions from preprocessed JSON."""
        sessions_path = self.data_dir / self.split / "sessions.json"
        if not sessions_path.exists():
            raise FileNotFoundError(
                f"MIND data not found at {sessions_path}. "
                "Run `python scripts/preprocess_mind.py` first. "
                "See README for download instructions."
            )
        with open(sessions_path) as f:
            self._sessions = json.load(f)
        self.rng.shuffle(self._sessions)

    def reset(self) -> None:
        if self._sessions is None:
            self._load_data()

        # Pick next session
        self._current_session = self._sessions[self._session_idx % len(self._sessions)]
        self._session_idx += 1

        # Subsample d articles from the session's candidate pool
        candidates = self._current_session["candidates"]
        if len(candidates) > self.d:
            idx = self.rng.choice(len(candidates), size=self.d, replace=False)
            self._article_pool = [candidates[i] for i in idx]
        else:
            self._article_pool = candidates[:self.d]
            # Pad if needed
            while len(self._article_pool) < self.d:
                self._article_pool.append({"title": "padding", "click_prob": 0.01})

        self._click_probs = np.array([a.get("click_prob", 0.1) for a in self._article_pool])
        self._means = self._click_probs
        self._optimal_set = None

    def _sample_reward(self, arm: int) -> float:
        return float(self.rng.binomial(1, self._click_probs[arm]))

    def get_arm_metadata(self) -> list[dict]:
        return [
            {
                "arm_id": i,
                "title": self._article_pool[i].get("title", f"article_{i}"),
                "category": self._article_pool[i].get("category", "unknown"),
                "subcategory": self._article_pool[i].get("subcategory", "unknown"),
            }
            for i in range(self.d)
        ]

    @property
    def user_interests(self) -> str:
        if self._current_session is None:
            return ""
        return self._current_session.get("user_interests", "general news")


class MINDEnvSimulated(CombBanditEnv):
    """Simulated MIND-like environment when real data is unavailable.

    Uses synthetic click probabilities with realistic structure:
    articles have categories, and user preferences create structured rewards.
    """

    def __init__(self, d: int = 200, m: int = 5, n_categories: int = 10, seed: int = 0):
        super().__init__(d=d, m=m, seed=seed)
        self.n_categories = n_categories
        self._categories: np.ndarray | None = None
        self._articles: list[dict] = []

    def reset(self) -> None:
        # Assign articles to categories
        self._categories = self.rng.randint(0, self.n_categories, size=self.d)

        # User preference: random preference vector over categories
        user_pref = self.rng.dirichlet(np.ones(self.n_categories))

        # Article quality: base quality + category preference
        base_quality = self.rng.beta(2, 5, size=self.d)
        category_boost = user_pref[self._categories]
        self._means = np.clip(0.3 * base_quality + 0.7 * category_boost, 0.01, 0.99)

        self._articles = [
            {"arm_id": i, "title": f"Article_{i}", "category": f"cat_{self._categories[i]}"}
            for i in range(self.d)
        ]
        self._optimal_set = None

    def _sample_reward(self, arm: int) -> float:
        return float(self.rng.binomial(1, self._means[arm]))

    def get_arm_metadata(self) -> list[dict]:
        return self._articles
