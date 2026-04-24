"""Heavily instrumented oracle with per-call JSONL logging.

Wraps the base GPTOracle to log every LLM call with:
- Full prompt + response
- Parsed output
- Token counts, latency
- Cache hit / miss
- trial_id context (config, seed, algo, round)
- Model used

Writes to results/tier2_{timestamp}/llm_calls.jsonl
"""
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any

from oracle import GPTOracle, _cache_key, _cache_get, _cache_put, get_client, LOGPROB_MODEL


_log_lock = threading.Lock()


class InstrumentedOracle(GPTOracle):
    """GPTOracle with elaborate JSONL logging per call."""

    _log_file: Path | None = None  # class-level output file

    def __init__(self, d: int, m: int, model: str = "gpt-5-mini",
                 temperature: float = 0.3,
                 trial_id: str = "unknown",
                 config_id: int = -1, seed: int = -1,
                 algo_name: str = "unknown"):
        super().__init__(d, m, model=model, temperature=temperature)
        self.trial_id = trial_id
        self.config_id = config_id
        self.seed = seed
        self.algo_name = algo_name
        self.current_t = 0

    @classmethod
    def set_log_file(cls, path: Path):
        cls._log_file = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")

    def _log_call(self, entry: dict):
        if self._log_file is None:
            return
        with _log_lock:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _chat(self, messages: list[dict], logprobs: bool = False, top_logprobs: int = 0,
              max_tokens: int = 256, query_type: str = "chat") -> dict:
        """Override to add logging around every call."""
        t_start = time.time()

        # Route logprob queries correctly
        active_model = LOGPROB_MODEL if (logprobs and not self.model.startswith("gpt-4")) else self.model
        is_gpt5_reasoning = active_model == "gpt-5-mini"  # only this one uses reasoning tokens heavily
        effective_max = max_tokens * 4 if is_gpt5_reasoning else max_tokens

        prompt_str = json.dumps(messages)
        extra = f"lp={logprobs}|tlp={top_logprobs}|mt={effective_max}|model={active_model}"
        key = _cache_key(query_type, prompt_str, active_model, extra)

        cached = _cache_get(key)
        if cached is not None:
            self.total_cache_hits += 1
            latency = (time.time() - t_start) * 1000
            self._log_call({
                "t_wall": time.time(),
                "trial_id": self.trial_id,
                "config_id": self.config_id,
                "seed": self.seed,
                "algo": self.algo_name,
                "model": active_model,
                "round_t": self.current_t,
                "query_type": query_type,
                "prompt_summary": messages[0]["content"][:200] if messages else "",
                "response_text": cached.get("text", "")[:500],
                "tokens": cached.get("tokens", 0),
                "cache_hit": True,
                "latency_ms": round(latency, 1),
                "has_logprobs": "logprobs" in cached,
            })
            return cached

        client = get_client()
        kwargs: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
        }
        if is_gpt5_reasoning:
            kwargs["max_completion_tokens"] = effective_max
            kwargs["reasoning_effort"] = "minimal"
        elif active_model.startswith("gpt-5"):
            # gpt-5.4 etc. — no reasoning_effort param
            kwargs["max_completion_tokens"] = effective_max
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
            kwargs["model"] = "gpt-4o-mini"
            kwargs.pop("reasoning_effort", None)
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            if "temperature" not in kwargs:
                kwargs["temperature"] = self.temperature
            resp = client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        result: dict[str, Any] = {"text": text, "tokens": tokens_in + tokens_out}

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
        self.total_tokens += tokens_in + tokens_out
        _cache_put(key, active_model, query_type, result)

        latency = (time.time() - t_start) * 1000
        self._log_call({
            "t_wall": time.time(),
            "trial_id": self.trial_id,
            "config_id": self.config_id,
            "seed": self.seed,
            "algo": self.algo_name,
            "model": active_model,
            "round_t": self.current_t,
            "query_type": query_type,
            "prompt_summary": messages[0]["content"][:300] if messages else "",
            "response_text": text[:500],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_hit": False,
            "latency_ms": round(latency, 1),
            "has_logprobs": "logprobs" in result,
        })

        return result
