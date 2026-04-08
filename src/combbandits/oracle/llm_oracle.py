"""Real LLM-backed CLO using OpenAI/Anthropic APIs."""
from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import Optional
import numpy as np

from .base import CLOBase
from ..types import OracleResponse

logger = logging.getLogger(__name__)


def _build_prompt(
    arm_metadata: list[dict],
    mu_hat: np.ndarray,
    m: int,
    context: dict,
    prompt_variant: int = 0,
) -> str:
    """Build the CLO prompt from arm metadata and context."""
    # Rank arms by empirical mean for context
    ranked = sorted(range(len(mu_hat)), key=lambda i: mu_hat[i], reverse=True)

    # Format arm descriptions
    arm_lines = []
    for i in ranked[:50]:  # Limit context window; top-50 by current estimate
        meta = arm_metadata[i] if i < len(arm_metadata) else {}
        desc_parts = [f"ID={i}"]
        for k, v in meta.items():
            if k != "arm_id":
                desc_parts.append(f"{k}={v}")
        desc_parts.append(f"est_reward={mu_hat[i]:.3f}")
        arm_lines.append(", ".join(desc_parts))

    arm_block = "\n".join(arm_lines)

    task_desc = context.get("task_description", "Select the best items to maximize total reward.")
    history_summary = context.get("history_summary", "")

    # Paraphrase variants for re-query independence
    variants = [
        f"You are an expert advisor. {task_desc}\n\nCandidate items (ranked by current estimate):\n{arm_block}\n\n{history_summary}\n\nSelect exactly {m} item IDs that will maximize total reward. Return ONLY a JSON list of {m} integer IDs.",
        f"As a domain expert, help select the optimal subset. {task_desc}\n\nAvailable options:\n{arm_block}\n\n{history_summary}\n\nChoose the best {m} items. Output a JSON array of {m} IDs only.",
        f"Task: {task_desc}\n\nItems available:\n{arm_block}\n\n{history_summary}\n\nPick {m} items to maximize expected reward. Respond with a JSON list of {m} integer IDs.",
    ]

    return variants[prompt_variant % len(variants)]


def _parse_response(text: str, d: int, m: int) -> list[int]:
    """Parse LLM response into a list of m arm indices."""
    # Try JSON parse first
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            ids = [int(x) for x in parsed if 0 <= int(x) < d]
            if len(ids) >= m:
                return ids[:m]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract integers from text
    numbers = [int(x) for x in re.findall(r'\b(\d+)\b', text) if 0 <= int(x) < d]
    if len(numbers) >= m:
        return numbers[:m]

    # Last resort: return first m valid IDs found
    logger.warning(f"Could not parse {m} valid IDs from LLM response, got {len(numbers)}")
    while len(numbers) < m:
        for i in range(d):
            if i not in numbers:
                numbers.append(i)
                if len(numbers) == m:
                    break
    return numbers[:m]


class LLMOracle(CLOBase):
    """Real LLM oracle using OpenAI or Anthropic API.

    Primary query uses the main model (e.g., GPT-4o).
    Re-queries can use a cheaper model (e.g., GPT-4o-mini or Llama-3-8B via vLLM).
    """

    def __init__(
        self,
        d: int,
        m: int,
        K: int = 3,
        primary_model: str = "gpt-4o",
        requery_model: Optional[str] = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(d=d, m=m, K=K)
        self.primary_model = primary_model
        self.requery_model = requery_model or primary_model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            import openai
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def _call_llm(self, prompt: str, model: str) -> tuple[str, int]:
        """Synchronous LLM call. Returns (response_text, tokens_used)."""
        client = self._get_client()

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return text, tokens

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            text = response.content[0].text if response.content else ""
            tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            return text, tokens

        raise ValueError(f"Unknown provider: {self.provider}")

    def query(
        self,
        context: dict,
        arm_metadata: list[dict],
        mu_hat: np.ndarray,
    ) -> OracleResponse:
        total_tokens = 0
        all_sets = []

        # Primary query
        prompt_0 = _build_prompt(arm_metadata, mu_hat, self.m, context, prompt_variant=0)
        text_0, tok_0 = self._call_llm(prompt_0, self.primary_model)
        total_tokens += tok_0
        primary_set = _parse_response(text_0, self.d, self.m)
        all_sets.append(primary_set)

        # Re-queries (K-1 additional, using cheaper model + prompt paraphrasing)
        for k in range(1, self.K):
            prompt_k = _build_prompt(arm_metadata, mu_hat, self.m, context, prompt_variant=k)
            text_k, tok_k = self._call_llm(prompt_k, self.requery_model)
            total_tokens += tok_k
            requery_set = _parse_response(text_k, self.d, self.m)
            all_sets.append(requery_set)

        self.total_queries += self.K
        self.total_tokens += total_tokens

        kappa = self.compute_consistency(all_sets)

        return OracleResponse(
            suggested_set=primary_set,
            re_query_sets=all_sets,
            consistency_score=kappa,
            raw_response=text_0,
            tokens_used=total_tokens,
        )
