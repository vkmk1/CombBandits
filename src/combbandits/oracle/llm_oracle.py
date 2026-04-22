"""Real LLM-backed CLO using OpenAI/Anthropic/AWS Bedrock APIs."""
from __future__ import annotations

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
    """Build the CLO prompt from arm metadata and context.

    The prompt gives the LLM enough structure to reason about which arms
    to select, without revealing the true reward means. The LLM must use
    its world knowledge (from arm metadata) plus empirical estimates to
    make a combinatorial selection.
    """
    d = len(arm_metadata)
    round_num = context.get("round", 0)

    # Rank arms by empirical mean for context window
    ranked = sorted(range(d), key=lambda i: mu_hat[i], reverse=True)

    # Include top-ranked arms (most promising) plus some exploration candidates
    n_top = min(40, d)
    n_random = min(10, d - n_top) if d > n_top else 0
    included_top = ranked[:n_top]
    included_random = list(np.random.choice(
        [i for i in range(d) if i not in included_top],
        size=n_random, replace=False
    )) if n_random > 0 else []
    included = included_top + included_random

    # Format arm descriptions with rich metadata
    arm_lines = []
    for i in included:
        meta = arm_metadata[i] if i < len(arm_metadata) else {}
        desc_parts = [f"ID={i}"]
        for k, v in meta.items():
            if k != "arm_id":
                desc_parts.append(f"{k}={v}")
        # Show empirical estimate if we have observations
        pulls = context.get("n_pulls", {}).get(i, 0) if isinstance(context.get("n_pulls"), dict) else 0
        if mu_hat[i] > 0 or pulls > 0:
            desc_parts.append(f"avg_reward={mu_hat[i]:.3f}")
        arm_lines.append(", ".join(desc_parts))

    arm_block = "\n".join(arm_lines)

    task_desc = context.get("task_description", "Select the best items to maximize total reward.")
    history_summary = context.get("history_summary", "")
    total_items = d
    showing = len(included)

    # Paraphrase variants for re-query independence (Assumption 2 in the paper)
    variants = [
        (
            f"You are an expert advisor for a sequential decision-making task.\n\n"
            f"TASK: {task_desc}\n\n"
            f"Round {round_num}. You must select exactly {m} items from {total_items} candidates.\n"
            f"Below are {showing} candidates (top-ranked by current performance estimates, "
            f"plus some less-explored options):\n\n{arm_block}\n\n"
            f"{history_summary}\n\n"
            f"Based on the item attributes AND the performance estimates, select the {m} items "
            f"most likely to yield high rewards. Consider both the metadata (what the item IS) "
            f"and the empirical data (how it has performed).\n\n"
            f"Return ONLY a JSON list of exactly {m} integer IDs. Example: [3, 7, 12, 1, 9]"
        ),
        (
            f"As a domain expert, help optimize a combinatorial selection problem.\n\n"
            f"{task_desc}\n\n"
            f"This is round {round_num}. Select {m} of the following {showing} options "
            f"(from {total_items} total) to maximize cumulative reward:\n\n{arm_block}\n\n"
            f"{history_summary}\n\n"
            f"Use your expertise about the item properties to identify the best subset. "
            f"Items with high estimated rewards are good candidates, but also consider "
            f"items whose attributes suggest they should perform well.\n\n"
            f"Output a JSON array of exactly {m} integer IDs."
        ),
        (
            f"Sequential optimization task, round {round_num}.\n\n"
            f"{task_desc}\n\n"
            f"Pick {m} items from these {showing} candidates:\n\n{arm_block}\n\n"
            f"{history_summary}\n\n"
            f"Maximize expected total reward. Consider item attributes and past performance.\n"
            f"Respond with a JSON list of {m} integer IDs."
        ),
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
    """Real LLM oracle using OpenAI, Anthropic, or AWS Bedrock APIs.

    Primary query uses the main model. Re-queries use a cheaper model
    with paraphrased prompts (for consistency estimation independence).

    For provider="bedrock", credentials come from the EC2 instance role
    (no API key needed). Uses boto3 bedrock-runtime.
    """

    def __init__(
        self,
        d: int,
        m: int,
        K: int = 3,
        primary_model: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        requery_model: Optional[str] = None,
        provider: str = "bedrock",
        temperature: float = 0.7,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        region: str = "us-east-1",
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
        self._region = region

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
        elif self.provider == "bedrock":
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
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

        elif self.provider == "bedrock":
            # Bedrock Converse API — works for all Anthropic, Meta, Mistral models
            response = client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            text = response["output"]["message"]["content"][0]["text"]
            usage = response.get("usage", {})
            tokens = usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
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
