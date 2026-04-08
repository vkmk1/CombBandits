"""Caching wrapper for CLO to implement O(√T) query schedule."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional
import numpy as np

from .base import CLOBase
from ..types import OracleResponse

logger = logging.getLogger(__name__)


class CachedOracle(CLOBase):
    """Wraps a CLO with SQLite disk cache and sublinear query schedule.

    Implements the O(√T) query schedule from Theorem 5 in the paper:
    query at rounds t_1, t_2, ... with inter-query gap Δ_j = ⌈c√j⌉.
    Between queries, reuses the most recent response.
    """

    def __init__(
        self,
        inner_oracle: CLOBase,
        cache_dir: str = "cache/oracle",
        schedule: str = "sqrt",
        schedule_const: float = 1.0,
        enable_disk_cache: bool = True,
    ):
        super().__init__(d=inner_oracle.d, m=inner_oracle.m, K=inner_oracle.K)
        self.inner = inner_oracle
        self.cache_dir = Path(cache_dir)
        self.schedule = schedule
        self.schedule_const = schedule_const
        self.enable_disk_cache = enable_disk_cache

        self._last_response: Optional[OracleResponse] = None
        self._query_count = 0
        self._round_count = 0
        self._next_query_round = 0

        # Disk cache
        self._db: Optional[sqlite3.Connection] = None
        if enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.cache_dir / "oracle_cache.db"))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
            )

    def _should_query(self) -> bool:
        """Determine if we should query the oracle at this round."""
        if self._last_response is None:
            return True
        if self.schedule == "every":
            return True
        if self.schedule == "sqrt":
            return self._round_count >= self._next_query_round
        return True

    def _advance_schedule(self):
        """Compute the next query round."""
        self._query_count += 1
        if self.schedule == "sqrt":
            gap = int(np.ceil(self.schedule_const * np.sqrt(self._query_count)))
            self._next_query_round = self._round_count + gap

    def _cache_key(self, context: dict, mu_hat: np.ndarray) -> str:
        """Hash context + mu_hat for disk cache lookup."""
        # Quantize mu_hat to avoid floating point differences
        quantized = (mu_hat * 100).astype(int).tobytes()
        ctx_str = json.dumps(context, sort_keys=True, default=str)
        raw = ctx_str.encode() + quantized
        return hashlib.sha256(raw).hexdigest()

    def _disk_lookup(self, key: str) -> Optional[OracleResponse]:
        if self._db is None:
            return None
        row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return OracleResponse(
            suggested_set=data["suggested_set"],
            re_query_sets=data["re_query_sets"],
            consistency_score=data["consistency_score"],
            tokens_used=0,
            cached=True,
        )

    def _disk_store(self, key: str, response: OracleResponse):
        if self._db is None:
            return
        data = json.dumps({
            "suggested_set": response.suggested_set,
            "re_query_sets": response.re_query_sets,
            "consistency_score": response.consistency_score,
        })
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", (key, data)
        )
        self._db.commit()

    def query(
        self,
        context: dict,
        arm_metadata: list[dict],
        mu_hat: np.ndarray,
    ) -> OracleResponse:
        self._round_count += 1

        if not self._should_query():
            assert self._last_response is not None
            return OracleResponse(
                suggested_set=self._last_response.suggested_set,
                re_query_sets=self._last_response.re_query_sets,
                consistency_score=self._last_response.consistency_score,
                tokens_used=0,
                cached=True,
            )

        # Check disk cache
        key = self._cache_key(context, mu_hat)
        cached = self._disk_lookup(key)
        if cached is not None:
            self._last_response = cached
            self._advance_schedule()
            return cached

        # Fresh query
        response = self.inner.query(context, arm_metadata, mu_hat)
        self._last_response = response
        self._advance_schedule()
        self.total_queries = self.inner.total_queries
        self.total_tokens = self.inner.total_tokens

        # Store in disk cache
        self._disk_store(key, response)

        return response

    def close(self):
        if self._db:
            self._db.close()
