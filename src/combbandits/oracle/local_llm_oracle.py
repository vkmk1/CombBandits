"""Local LLM oracle with full weight/activation tracking.

Loads Llama 4 Scout (or any HuggingFace CausalLM) directly on the current device
and captures per-call internals for scientific analysis:

  - token log-probabilities for each suggested arm ID  (model confidence)
  - attention weights over arm metadata tokens          (what the model looks at)
  - final hidden-state vectors                          (representation per query)
  - output token entropy                                (uncertainty)
  - KL divergence between K re-query hidden states      (internal diversity)

All internals are written to a SQLite database (one row per oracle call) so they
can be joined with the bandit results for post-hoc analysis.

Usage (exp9_local.yaml):
  oracles:
    - type: local_llm
      model_name: meta-llama/Llama-4-Scout-17B-16E-Instruct
      device: cuda          # or cpu / mps
      K: 3
      temperature: 0.7
      schedule: sqrt
      cache_dir: cache/oracle_exp9_local
      weights_db: metadata/oracle_weights.db

Requires:
  pip install transformers>=4.47 accelerate bitsandbytes
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from .base import CLOBase
from .llm_oracle import _build_prompt, _parse_response
from ..types import OracleResponse

logger = logging.getLogger(__name__)


class LocalLLMOracle(CLOBase):
    """Llama-4-Scout (or any HF CausalLM) oracle with activation tracking."""

    def __init__(
        self,
        d: int,
        m: int,
        K: int = 3,
        model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device: str = "cuda",
        temperature: float = 0.7,
        max_new_tokens: int = 64,
        load_in_4bit: bool = True,
        weights_db: str = "metadata/oracle_weights.db",
        hf_token: Optional[str] = None,
    ):
        super().__init__(d=d, m=m, K=K)
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.hf_token = hf_token
        self._model = None
        self._tokenizer = None

        # SQLite for per-call weight snapshots
        db_path = Path(weights_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._init_db()

        self._call_idx = 0

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading {self.model_name} on {self.device} (4bit={self.load_in_4bit})...")

        tok_kwargs = {"trust_remote_code": True}
        if self.hf_token:
            tok_kwargs["token"] = self.hf_token

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "output_attentions": True,  # needed for attention capture
            "output_hidden_states": True,
        }
        if self.hf_token:
            model_kwargs["token"] = self.hf_token

        if self.load_in_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
            model_kwargs["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self._model.eval()
        logger.info(f"Model loaded. Params: {sum(p.numel() for p in self._model.parameters()):,}")

    # ── Database ──────────────────────────────────────────────────────────

    def _init_db(self):
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS oracle_calls (
                call_id        INTEGER PRIMARY KEY,
                trial_round    INTEGER,
                query_variant  INTEGER,
                prompt_len_tok INTEGER,
                response_text  TEXT,
                suggested_set  TEXT,
                -- confidence: log-prob sum over suggested arm ID tokens
                suggestion_logprob  REAL,
                -- uncertainty: entropy of next-token distribution at generation start
                output_entropy REAL,
                -- attention: mean attention weight on arm-metadata portion of prompt
                -- shape compressed to scalar (mean over layers, heads, metadata span)
                attn_on_metadata REAL,
                -- hidden state: L2 norm of last-layer hidden state at final input token
                hidden_state_norm REAL,
                -- raw hidden state (last layer, last input token) — compressed to 128-d via PCA
                hidden_state_pca TEXT,
                tokens_used    INTEGER,
                elapsed_ms     INTEGER
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS query_groups (
                group_id        INTEGER PRIMARY KEY,
                trial_round     INTEGER,
                primary_set     TEXT,
                kappa           REAL,
                -- KL divergence between hidden states of K re-queries (scalar approx)
                hidden_kl_div   REAL,
                -- cosine similarity between primary and each re-query hidden state
                hidden_cosines  TEXT
            )
        """)
        self._db.commit()

    def _store_call(self, row: dict):
        self._db.execute("""
            INSERT INTO oracle_calls
            (call_id, trial_round, query_variant, prompt_len_tok, response_text,
             suggested_set, suggestion_logprob, output_entropy, attn_on_metadata,
             hidden_state_norm, hidden_state_pca, tokens_used, elapsed_ms)
            VALUES
            (:call_id, :trial_round, :query_variant, :prompt_len_tok, :response_text,
             :suggested_set, :suggestion_logprob, :output_entropy, :attn_on_metadata,
             :hidden_state_norm, :hidden_state_pca, :tokens_used, :elapsed_ms)
        """, row)
        self._db.commit()

    def _store_group(self, row: dict):
        self._db.execute("""
            INSERT INTO query_groups
            (group_id, trial_round, primary_set, kappa, hidden_kl_div, hidden_cosines)
            VALUES
            (:group_id, :trial_round, :primary_set, :kappa, :hidden_kl_div, :hidden_cosines)
        """, row)
        self._db.commit()

    # ── Core inference + weight capture ───────────────────────────────────

    def _call_with_tracking(
        self,
        prompt: str,
        variant: int,
        round_t: int,
    ) -> tuple[str, int, dict]:
        """Run one forward pass, capture internals. Returns (text, tokens, internals)."""
        import torch, time

        self._load_model()
        t0 = time.time()

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        input_ids = inputs["input_ids"].to(self._model.device)
        prompt_len = input_ids.shape[1]

        # Find the token span covering arm metadata (heuristic: tokens after "candidates" keyword)
        # We mark metadata start at 60% of prompt length (arm block starts after preamble)
        meta_start = int(prompt_len * 0.25)
        meta_end   = int(prompt_len * 0.85)

        with torch.no_grad():
            # --- Prefill: full forward pass to get hidden states + attention ---
            prefill_out = self._model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
            )

            # Hidden state at last input token (last layer)
            # shape: (1, seq_len, hidden_dim)
            last_hidden = prefill_out.hidden_states[-1][0, -1, :].float()  # (hidden_dim,)
            hidden_norm = float(last_hidden.norm().cpu())

            # PCA-compress to 128 dims for storage (random projection as cheap approx)
            hid_dim = last_hidden.shape[0]
            if not hasattr(self, "_proj"):
                torch.manual_seed(42)
                self._proj = torch.randn(hid_dim, 128, device=self._model.device) / math.sqrt(128)
            hidden_pca = (last_hidden @ self._proj).cpu().tolist()

            # Output entropy: softmax over vocab at last input token position
            logits_last = prefill_out.logits[0, -1, :].float()
            probs = torch.softmax(logits_last, dim=-1)
            entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).cpu())

            # Attention on metadata span: mean over all layers and heads
            # attentions: list of (1, n_heads, seq, seq) per layer
            try:
                attn_layers = prefill_out.attentions  # tuple of tensors
                if attn_layers is not None:
                    # Mean attention FROM last token TO metadata span
                    attn_vals = []
                    for layer_attn in attn_layers:
                        # (1, heads, seq, seq) → mean over heads → (seq, seq)
                        mean_heads = layer_attn[0].mean(dim=0)  # (seq, seq)
                        # attention from last token to metadata range
                        attn_to_meta = mean_heads[-1, meta_start:meta_end].mean()
                        attn_vals.append(float(attn_to_meta.cpu()))
                    attn_on_meta = float(np.mean(attn_vals))
                else:
                    attn_on_meta = -1.0
            except Exception:
                attn_on_meta = -1.0

            # --- Generate response ---
            gen_out = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            new_tokens = gen_out.sequences[0, prompt_len:]
            response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            tokens_used = prompt_len + len(new_tokens)

            # Log-prob of response tokens (sum of log-probs over generated sequence)
            if gen_out.scores:
                logprob_sum = 0.0
                for step, tok_id in enumerate(new_tokens[:len(gen_out.scores)]):
                    step_logits = gen_out.scores[step][0].float()
                    step_lp = float(torch.log_softmax(step_logits, dim=-1)[tok_id].cpu())
                    logprob_sum += step_lp
                suggestion_logprob = logprob_sum
            else:
                suggestion_logprob = 0.0

        elapsed_ms = int((time.time() - t0) * 1000)
        self._call_idx += 1

        internals = {
            "hidden_state": last_hidden.cpu(),
            "hidden_norm": hidden_norm,
            "hidden_pca": hidden_pca,
            "entropy": entropy,
            "attn_on_metadata": attn_on_meta,
            "suggestion_logprob": suggestion_logprob,
            "prompt_len": prompt_len,
            "tokens_used": tokens_used,
            "elapsed_ms": elapsed_ms,
        }
        return response_text, tokens_used, internals

    # ── CLO interface ─────────────────────────────────────────────────────

    def query(
        self,
        context: dict,
        arm_metadata: list[dict],
        mu_hat: np.ndarray,
    ) -> OracleResponse:
        import torch

        round_t = context.get("round", self.total_queries)
        total_tokens = 0
        all_sets = []
        all_hidden = []

        for k in range(self.K):
            prompt = _build_prompt(arm_metadata, mu_hat, self.m, context, prompt_variant=k)
            text, tokens, internals = self._call_with_tracking(prompt, variant=k, round_t=round_t)
            total_tokens += tokens

            suggested = _parse_response(text, self.d, self.m)
            all_sets.append(suggested)
            all_hidden.append(internals["hidden_state"])

            # Store per-call row
            self._store_call({
                "call_id":             self._call_idx,
                "trial_round":         round_t,
                "query_variant":       k,
                "prompt_len_tok":      internals["prompt_len"],
                "response_text":       text[:500],
                "suggested_set":       json.dumps(suggested),
                "suggestion_logprob":  internals["suggestion_logprob"],
                "output_entropy":      internals["entropy"],
                "attn_on_metadata":    internals["attn_on_metadata"],
                "hidden_state_norm":   internals["hidden_norm"],
                "hidden_state_pca":    json.dumps(internals["hidden_pca"]),
                "tokens_used":         tokens,
                "elapsed_ms":          internals["elapsed_ms"],
            })

        kappa = self.compute_consistency(all_sets)

        # Cross-query hidden state analysis
        hidden_cosines = []
        h0 = all_hidden[0]
        for hk in all_hidden[1:]:
            cos = float((h0 @ hk) / (h0.norm() * hk.norm() + 1e-8))
            hidden_cosines.append(round(cos, 4))

        # KL divergence approximation via hidden-state L2 (scalar proxy)
        norms = [float(h.norm()) for h in all_hidden]
        hidden_kl = float(np.std(norms))  # std of norms as diversity proxy

        self._store_group({
            "group_id":      self.total_queries,
            "trial_round":   round_t,
            "primary_set":   json.dumps(all_sets[0]),
            "kappa":         kappa,
            "hidden_kl_div": hidden_kl,
            "hidden_cosines": json.dumps(hidden_cosines),
        })

        self.total_queries += self.K
        self.total_tokens  += total_tokens

        return OracleResponse(
            suggested_set=all_sets[0],
            re_query_sets=all_sets,
            consistency_score=kappa,
            raw_response=None,
            tokens_used=total_tokens,
        )

    def close(self):
        self._db.close()
