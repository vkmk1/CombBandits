"""OpenAI-backed oracle with full logprob access, SQLite caching, and multi-modal queries.

Uses gpt-5-mini for all queries. Supports 7 query types matching the 17 algorithms:
1. top_m: classic "pick best m arms"
2. logprobs: force single-token arm ID, extract logprob distribution over all arms
3. per_arm_scores: per-arm [mean, CI] estimates
4. pairwise: is arm i better than arm j?
5. cluster: group similar arms
6. elimination: which arms to remove
7. counterfactual: predict rewards for unpulled arms
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DB = CACHE_DIR / "llm_cache.sqlite"

MODEL = "gpt-5-mini"  # primary model
LOGPROB_MODEL = "gpt-4.1-mini"  # gpt-5-mini blocks logprobs, use 4.1-mini for those queries
FALLBACK_MODEL = "gpt-4o-mini"


# ─── Cache ────────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()


def _init_cache():
    with sqlite3.connect(CACHE_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY,
                model TEXT,
                query_type TEXT,
                response TEXT,
                created_at REAL
            )
        """)


_init_cache()


def _cache_key(query_type: str, prompt: str, model: str, extra: str = "") -> str:
    h = hashlib.sha256(f"{query_type}|{model}|{prompt}|{extra}".encode()).hexdigest()
    return h


def _ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, model TEXT, query_type TEXT,
            response TEXT, created_at REAL
        )
    """)


def _cache_get(key: str) -> dict | None:
    with _cache_lock, sqlite3.connect(CACHE_DB) as con:
        _ensure_table(con)
        row = con.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
    return None


def _cache_put(key: str, model: str, query_type: str, response: dict):
    with _cache_lock, sqlite3.connect(CACHE_DB) as con:
        _ensure_table(con)
        con.execute(
            "INSERT OR REPLACE INTO calls (key, model, query_type, response, created_at) VALUES (?,?,?,?,?)",
            (key, model, query_type, json.dumps(response), time.time()),
        )


# ─── OpenAI client ────────────────────────────────────────────────────────
_client_lock = threading.Lock()
_client = None


def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            _client = OpenAI(api_key=api_key)
    return _client


# ─── Oracle ───────────────────────────────────────────────────────────────
class GPTOracle:
    """All oracle queries go through here. Cached. Counts tokens."""

    def __init__(self, d: int, m: int, model: str = MODEL, temperature: float = 0.3):
        self.d = d
        self.m = m
        self.model = model
        self.temperature = temperature
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cache_hits = 0
        self.call_log: list[dict] = []

    def _format_arm_list(self, mu_hat: list[float], ranked: bool = True) -> str:
        if ranked:
            order = sorted(range(self.d), key=lambda i: mu_hat[i], reverse=True)
        else:
            order = list(range(self.d))
        lines = [f"Arm {i:2d}: avg_reward={mu_hat[i]:.3f}" for i in order]
        return "\n".join(lines)

    def _chat(self, messages: list[dict], logprobs: bool = False, top_logprobs: int = 0,
              max_tokens: int = 256, query_type: str = "chat") -> dict:
        """Raw chat completion. Returns dict with text, tokens, logprobs (if requested)."""
        # Route logprob queries to gpt-4.1-mini (gpt-5-mini blocks them)
        active_model = LOGPROB_MODEL if logprobs else self.model
        # gpt-5 models need more tokens due to reasoning; gpt-4.1 needs less
        is_gpt5 = active_model.startswith("gpt-5")
        effective_max = max_tokens * 4 if is_gpt5 else max_tokens

        prompt_str = json.dumps(messages)
        extra = f"lp={logprobs}|tlp={top_logprobs}|mt={effective_max}|model={active_model}"
        key = _cache_key(query_type, prompt_str, active_model, extra)

        cached = _cache_get(key)
        if cached is not None:
            self.total_cache_hits += 1
            return cached

        client = get_client()
        kwargs: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
        }
        if is_gpt5:
            kwargs["max_completion_tokens"] = effective_max
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["max_tokens"] = effective_max
            kwargs["temperature"] = self.temperature
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs > 0:
                kwargs["top_logprobs"] = top_logprobs

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # Fallback to gpt-4o-mini
            kwargs["model"] = FALLBACK_MODEL
            kwargs.pop("reasoning_effort", None)
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            kwargs["temperature"] = self.temperature
            resp = client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        tokens = (usage.prompt_tokens if usage else 0) + (usage.completion_tokens if usage else 0)

        result: dict[str, Any] = {"text": text, "tokens": tokens}

        if logprobs and choice.logprobs is not None:
            lp_data = []
            for tok in choice.logprobs.content or []:
                entry = {
                    "token": tok.token,
                    "logprob": tok.logprob,
                    "top": [(t.token, t.logprob) for t in (tok.top_logprobs or [])],
                }
                lp_data.append(entry)
            result["logprobs"] = lp_data

        self.total_calls += 1
        self.total_tokens += tokens
        _cache_put(key, self.model, query_type, result)
        return result

    # ─── Query Type 1: Top-m selection (classic) ─────────────────────────
    def query_top_m(self, mu_hat: list[float]) -> list[int]:
        prompt = (
            f"You help a combinatorial bandit learner choose {self.m} arms from {self.d}.\n"
            f"Current avg rewards:\n{self._format_arm_list(mu_hat)}\n\n"
            f"Return exactly {self.m} arm IDs as a JSON array. Example: [3, 7, 12, 1, 9]\n"
            f"Only output the JSON array."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="top_m", max_tokens=128)
        return self._parse_arm_list(out["text"], expected=self.m)

    # ─── Query Type 2: Logprobs extraction (A1, A2, A3) ──────────────────
    def query_logprobs(self, mu_hat: list[float]) -> dict[int, float]:
        """Get per-arm probability from logprobs over next arm-ID token.

        We force the LLM to output a single arm ID, then read the top-20 logprob
        distribution. Returns {arm_id: probability} dict summing to ~1.
        """
        # Build prompt that asks for a single arm ID
        prompt = (
            f"You are ranking arms for a bandit problem. d={self.d}, m={self.m}.\n"
            f"Current avg rewards:\n{self._format_arm_list(mu_hat)}\n\n"
            f"What is the SINGLE best arm? Output only the integer arm ID (0-{self.d-1}), nothing else.\n"
            f"Best arm ID:"
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(
            messages, logprobs=True, top_logprobs=20,
            max_tokens=6, query_type="logprobs",
        )

        probs: dict[int, float] = {}
        if "logprobs" not in out or not out["logprobs"]:
            # fallback: parse text
            try:
                arm = int(out["text"].strip())
                if 0 <= arm < self.d:
                    probs[arm] = 1.0
            except ValueError:
                pass
            return probs

        # Find the first token that is an ASCII digit
        for tok_entry in out["logprobs"]:
            tok = tok_entry["token"].strip()
            if (tok.isascii() and tok.isdigit()) or (
                tok.startswith("-") and tok[1:].isascii() and tok[1:].isdigit()):
                # This is the arm ID token - extract distribution over neighboring digit tokens
                total_prob = 0.0
                for candidate_tok, lp in tok_entry["top"]:
                    ct = candidate_tok.strip()
                    if ct.isascii() and ct.isdigit():
                        arm_id = int(ct)
                        if 0 <= arm_id < self.d:
                            p = math.exp(lp)
                            probs[arm_id] = probs.get(arm_id, 0) + p
                            total_prob += p
                # Normalize
                if total_prob > 0:
                    probs = {k: v / total_prob for k, v in probs.items()}
                break
        return probs

    # ─── Query Type 3: Per-arm [mean, CI] scores (E1, E2, E3, B1) ────────
    def query_per_arm_scores(self, mu_hat: list[float], context: str = "") -> dict[int, dict]:
        """Get per-arm estimate with confidence interval."""
        prompt = (
            f"You are a Bayesian expert. Bandit problem: d={self.d} arms, pick {self.m}.\n"
            f"{context}\n"
            f"Current data:\n{self._format_arm_list(mu_hat)}\n\n"
            f"For EACH arm, output your best guess of its TRUE reward mean (0-1) and "
            f"your 95% confidence interval.\n"
            f"Return JSON: [{{'arm': 0, 'mean': 0.5, 'lo': 0.3, 'hi': 0.7}}, ...] for all {self.d} arms.\n"
            f"Output only the JSON array."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="per_arm_scores", max_tokens=1800)

        result = {}
        try:
            parsed = self._extract_json(out["text"])
            for entry in parsed:
                aid = int(entry["arm"])
                if 0 <= aid < self.d:
                    result[aid] = {
                        "mean": max(0.001, min(0.999, float(entry["mean"]))),
                        "lo": max(0.0, min(1.0, float(entry.get("lo", 0)))),
                        "hi": max(0.0, min(1.0, float(entry.get("hi", 1)))),
                    }
        except Exception:
            pass
        return result

    # ─── Query Type 4: Pairwise comparison (D3) ──────────────────────────
    def query_pairwise(self, mu_hat: list[float], pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """For each (i, j) pair, probability that arm i is better than arm j."""
        pair_str = "\n".join(f"  Pair {k}: arm {i} vs arm {j}" for k, (i, j) in enumerate(pairs))
        prompt = (
            f"Bandit problem: d={self.d}, m={self.m}.\n"
            f"Current data:\n{self._format_arm_list(mu_hat)}\n\n"
            f"For each pair below, estimate the probability (0-1) that the FIRST arm has a higher true reward:\n"
            f"{pair_str}\n\n"
            f"Return JSON array: [{{'pair': 0, 'p_first_better': 0.7}}, ...]\n"
            f"Output only the JSON."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="pairwise", max_tokens=512)

        result = {}
        try:
            parsed = self._extract_json(out["text"])
            for entry in parsed:
                idx = int(entry["pair"])
                if 0 <= idx < len(pairs):
                    p = max(0.01, min(0.99, float(entry["p_first_better"])))
                    result[pairs[idx]] = p
        except Exception:
            pass
        return result

    # ─── Query Type 5: Clustering (D1, D2) ───────────────────────────────
    def query_clusters(self, mu_hat: list[float], n_clusters: int = 8) -> list[list[int]]:
        """Group arms into clusters of similar expected performance."""
        prompt = (
            f"You are grouping {self.d} bandit arms by similarity.\n"
            f"Current data:\n{self._format_arm_list(mu_hat)}\n\n"
            f"Group these {self.d} arms into about {n_clusters} clusters of similar expected reward.\n"
            f"Arms in the same cluster should behave similarly (if one is good, the others likely are too).\n\n"
            f"Return JSON: [[3, 7, 12], [1, 9, 4], ...] where each inner list is a cluster.\n"
            f"Every arm 0-{self.d-1} must appear exactly once. Output only the JSON."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="clusters", max_tokens=1024)

        try:
            clusters = self._extract_json(out["text"])
            clusters = [[int(a) for a in c if 0 <= int(a) < self.d] for c in clusters]
            # Ensure every arm appears
            seen = set()
            for c in clusters:
                seen.update(c)
            missing = [a for a in range(self.d) if a not in seen]
            if missing and clusters:
                clusters[0].extend(missing)
            return [c for c in clusters if c]
        except Exception:
            return [list(range(self.d))]  # fallback: one big cluster

    # ─── Query Type 6: Elimination (F1) ──────────────────────────────────
    def query_elimination(self, mu_hat: list[float]) -> list[int]:
        """Which arms should be eliminated? Returns arm IDs to drop."""
        prompt = (
            f"Bandit problem: d={self.d}, pick {self.m} best.\n"
            f"Current data:\n{self._format_arm_list(mu_hat)}\n\n"
            f"Which arms are almost CERTAINLY NOT in the top {self.m}? "
            f"Be conservative — only eliminate arms you're very confident about.\n"
            f"Return JSON array of arm IDs to eliminate: [2, 5, 11]\n"
            f"Output only the JSON."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="elimination", max_tokens=256)

        try:
            return self._parse_arm_list(out["text"], expected=None)
        except Exception:
            return []

    # ─── Query Type 7: Counterfactual reward prediction (G2) ─────────────
    def query_counterfactual(self, mu_hat: list[float], history_summary: str) -> dict[int, float]:
        """Predict reward probability for each arm given history."""
        prompt = (
            f"Bandit problem: d={self.d}, m={self.m}.\n"
            f"Observed history summary: {history_summary}\n"
            f"Current estimates:\n{self._format_arm_list(mu_hat)}\n\n"
            f"For EACH arm, predict its probability of giving reward=1 on a fresh pull.\n"
            f"Return JSON: [{{'arm': 0, 'p': 0.3}}, ...] for all {self.d} arms.\n"
            f"Output only the JSON."
        )
        messages = [{"role": "user", "content": prompt}]
        out = self._chat(messages, query_type="counterfactual", max_tokens=1200)

        result = {}
        try:
            parsed = self._extract_json(out["text"])
            for entry in parsed:
                aid = int(entry["arm"])
                if 0 <= aid < self.d:
                    result[aid] = max(0.01, min(0.99, float(entry["p"])))
        except Exception:
            pass
        return result

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _parse_arm_list(self, text: str, expected: int | None = None) -> list[int]:
        try:
            parsed = self._extract_json(text)
            if isinstance(parsed, list):
                ids = [int(x) for x in parsed if 0 <= int(x) < self.d]
                if expected is None or len(ids) >= expected:
                    return ids[:expected] if expected else ids
        except Exception:
            pass
        # Regex fallback
        import re
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', text) if 0 <= int(x) < self.d]
        if expected is None:
            return numbers
        return numbers[:expected] if len(numbers) >= expected else numbers

    def _extract_json(self, text: str):
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        # Find first [ or {
        start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
        if start < 0:
            raise ValueError("No JSON found")
        # Find matching close
        depth = 0
        open_char = text[start]
        close_char = "]" if open_char == "[" else "}"
        end = start
        for i in range(start, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        return json.loads(text[start:end + 1])

    def diagnostics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "cache_hits": self.total_cache_hits,
            "cache_hit_rate": self.total_cache_hits / max(1, self.total_calls + self.total_cache_hits),
        }
