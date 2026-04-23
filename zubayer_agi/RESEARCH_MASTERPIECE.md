# The Masterpiece Document: Novel LLM Capabilities for Combinatorial Bandits

## Part 1: The Fundamental Reframe

**Every existing LLM-bandit algorithm commits the same cardinal sin**: it reduces the LLM to a **set-valued oracle** — "give me m arms." This is the *lowest-bandwidth* interface possible. An LLM is capable of vastly more nuanced outputs than a discrete set.

Research across 40+ papers reveals LLMs can be reframed as **seven fundamentally different kinds of computational objects**, each opening a different algorithmic direction:

| Frame | LLM as... | Key capability exploited |
|-------|-----------|-------------------------|
| 1 | **Probability generator** | Token-level logprobs give calibrated per-arm scores |
| 2 | **Bayesian reasoner** | In-context learning IS implicit Bayesian inference (provably) |
| 3 | **Debater / ensemble** | Multi-agent debate reduces hallucinations |
| 4 | **Structure discoverer** | Semantic embeddings reveal arm correlations/clusters |
| 5 | **Prior elicitor** | AutoElicit framework extracts calibrated expert priors |
| 6 | **Information designer** | Bayesian persuasion framework: what to send the LLM matters |
| 7 | **World model** | Counterfactual + MCTS simulation of bandit trajectories |

Each frame yields 2-3 novel algorithmic ideas — **17 candidates** to smoke test.

---

## Part 2: The 17 Candidate Algorithms

### Group A — LLM as Probability Generator (exploit logprobs)

**Key insight**: OpenAI/Anthropic APIs return token-level log probabilities. Instead of asking "pick 5 arms" and parsing JSON, we force the LLM to output arm IDs one at a time, and grab the **logprob distribution over every arm at each generation step**. This gives us a full probability distribution over arms, not a discrete set — a 30x bandwidth increase.

**A1. Logprob-CTS (The Probability Extractor)**
- Prompt: "Output the top arm. Top arm: [force this token to be an arm ID]"
- Grab the logprob distribution: get `P(arm=i)` for all 30 arms in one call
- Convert to Beta pseudo-observations: if LLM says `P(arm 5) = 0.7`, add Beta(7, 3) worth of pseudo-observations to arm 5
- **Why it beats CTS**: instead of hard-coding "arm 5 is in the pool" (binary), we softly update CTS's actual Beta posteriors (continuous).

**A2. Self-Distractor Calibration CTS**
- LLM generates its top 5 AND its 5 "trap picks" (arms it thinks look good but probably aren't)
- Use the gap between them as a calibration signal
- **Why**: adaptive trust without needing a held-out validation set

**A3. Temperature-Scaled Mixture CTS**
- Query LLM at 3 different temperatures (0, 0.5, 1.0)
- Variance across temps is a proxy for LLM's true uncertainty
- Weight prior injection by this meta-uncertainty

### Group B — LLM as Bayesian Reasoner

**B1. Pseudo-Observation Injection CTS**
- Periodically query LLM with full history
- LLM outputs per-arm success probability estimates
- Convert to pseudo-successes/failures; add to CTS's Beta counters
- Calibration gate: verify LLM against held-out arms

**B2. In-Context Posterior Distillation (ICPD-CTS)**
- Feed LLM pulls + rewards; ask for posterior over top arms
- LLM internally does Bayesian inference; we extract and merge

**B3. Regret-Loss Fine-Tuned CTS**
- Fine-tune small model on regret-minimization trajectories
- Use fine-tuned output as prior for CTS

### Group C — LLM as Debater / Ensemble

**C1. Debate-Arena CTS** — 3 LLM personalities (optimistic, pessimistic, exploratory) debate

**C2. Devil's Advocate CTS** — one proposes, one critiques

**C3. DIPPER Ensemble CTS** — 5 semantically-diverse prompts, Bradley-Terry aggregation

### Group D — LLM as Structure Discoverer

**D1. Semantic-Cluster CTS** — cluster similar arms; rewards transfer within cluster (shrinks d → k)

**D2. Causal-Graph CTS** — LLM outputs causal graph; reward propagates along edges

**D3. Pairwise Elo-CTS (Bradley-Terry)** — K=10 pairwise queries → BT scores → CTS priors

### Group E — LLM as Prior Elicitor

**E1. AutoElicit-CTS** — per-arm [mean, CI] → matching Beta priors

**E2. Recursive Refinement CTS** — LLM updates its belief iteratively with data

**E3. Conformal Prior CTS** — conformal prediction recalibrates LLM's overconfident priors

### Group F — LLM as Information Designer

**F1. Information-Optimal Prompt CTS** — show top-k UCB arms with rank+confidence flags (not raw mu_hat)

**F2. Query-Design Optimization (QDO-CTS)** — meta-bandit over prompt formats

### Group G — LLM as World Model

**G1. LLM-MCTS-Bandit** — LLM simulates forward, selects minimum-regret trajectory

**G2. Counterfactual CTS** — "what would arm Y have given?" → soft data for unpulled arms

---

## Part 3: Why These Can Actually Beat CTS

CTS's regret is **O(√(d·m·T·log T))**. To beat it, attack one of three levers:

| Lever | How LLM helps | Best candidates |
|-------|--------------|-----------------|
| **Reduce effective d** | Semantic clusters (d → k clusters) | D1, D2, F1 |
| **Better starting priors** | Warm-start Beta posteriors from LLM | A1, B1, E1, E2 |
| **Accelerated exploration** | Don't re-explore LLM-eliminated arms | F1, B1 (via gating) |
| **Correlated updates** | Reward on one arm updates similar arms | D1, D2 |
| **Richer feedback** | Counterfactual + logprob info per query | A1, G2 |

Candidates that hit **multiple levers** are strongest: **D1** reduces d AND enables correlated updates. **A1** gives better priors AND richer feedback. **B1** gives priors AND continuous updates.

---

## Part 4: Ranked Priority for Smoke Testing

**Tier S (highest priority)**:
1. **D1 — Semantic-Cluster CTS**
2. **A1 — Logprob-CTS**
3. **B1 — Pseudo-Observation CTS**
4. **E1 — AutoElicit-CTS**

**Tier A (strong candidates)**:
5. D3 — Pairwise Elo-CTS
6. F1 — Information-Optimal Prompt CTS
7. E2 — Recursive Refinement CTS
8. C3 — DIPPER Ensemble CTS

**Tier B (interesting)**:
9. A2 — Self-Distractor Calibration
10. B2 — In-Context Posterior Distillation
11. D2 — Causal-Graph CTS
12. E3 — Conformal Prior CTS

**Tier C (moonshots)**:
13. C1 — Debate-Arena
14. C2 — Devil's Advocate
15. G1 — LLM-MCTS-Bandit
16. G2 — Counterfactual CTS
17. B3 — Regret-Loss Fine-Tuned (requires training)

---

## Part 5: Smoke Test Infrastructure

**Model**: gpt-5-mini via OpenAI (Bedrock Claude doesn't expose logprobs on foundation models — confirmed via AWS docs).

**Approach**:
1. SQLite cache: keyed by (prompt_hash, model, temperature) → response + logprobs
2. 3 configs (uniform-easy, uniform-hard, hard-gap) × 3 seeds = 9 trials
3. Each trial runs CTS baseline + all 17 novel algorithms
4. T=800 rounds per trial (short enough for speed, long enough to distinguish winners)
5. Total estimated cost: ~$2-5 for all 17 algorithms' full smoke test

**Evaluation metrics**:
- Mean final regret vs CTS baseline
- Win rate across trials
- LLM call efficiency (regret reduction per call)
- Oracle overlap with optimal set

---

## Sources

**LLM Calibration & Logprobs**:
- [Calibrating Verbalized Probabilities for LLMs](https://arxiv.org/html/2410.06707v1)
- [Self-Generated Distractors](https://arxiv.org/html/2509.25532)
- [QA-Calibration (ICLR 2025)](https://assets.amazon.science/6d/70/c50b2eb141d3bcf1565e62b60211/qa-calibration-of-language-model-confidence-scores.pdf)

**In-Context Learning as Bayesian Inference**:
- [In-Context Learning Is Provably Bayesian Inference (2025)](https://arxiv.org/html/2510.10981)
- [Can Transformers Learn Full Bayesian Inference?](https://arxiv.org/pdf/2501.16825)

**LLM Prior Elicitation**:
- [AutoElicit](https://arxiv.org/html/2411.17284v5)
- [LLM-Prior](https://arxiv.org/html/2508.03766v1)
- [Had enough of experts? (ICLR)](https://openreview.net/forum?id=3iDxHRQfVy)

**LLM-Bandit Integration**:
- [Multi-Armed Bandits Meet LLMs](https://arxiv.org/html/2505.13355v1)
- [Calibration-Gated LLM Pseudo-Observations](https://arxiv.org/html/2604.14961)
- [Jump Starting Bandits with LLM Prior](https://arxiv.org/html/2406.19317v1)

**Multi-Agent Debate & Ensembles**:
- [Multiagent Debate (ICML 2024)](https://openreview.net/forum?id=QAwaaLJNCk)
- [DIPPER](https://arxiv.org/html/2412.15238v1)

**Bradley-Terry & Pairwise**:
- [Statistical Framework for Ranking LLMs](https://arxiv.org/html/2412.18407v1)

**Bayesian Persuasion**:
- [Verbalized Bayesian Persuasion (2025)](https://arxiv.org/html/2502.01587)
- [Information Design with LLMs](https://arxiv.org/html/2509.25565)

**LLM-MCTS**:
- [LLM-MCTS (NeurIPS 2023)](https://llm-mcts.github.io/)

**Combinatorial Bandits Theory**:
- [Graph Feedback Bandits (2025)](https://arxiv.org/html/2501.14314)
- [Tight Lower Bounds for Combinatorial MAB](https://arxiv.org/abs/2002.05392)

**Regret & Decision Making**:
- [Do LLM Agents Have Regret?](https://arxiv.org/html/2403.16843v1)
- [Post-Training LLMs as Better Decision-Making Agents (NeurIPS 2025)](https://arxiv.org/abs/2511.04393)
