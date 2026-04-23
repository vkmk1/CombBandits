# Research Synthesis (Round 1)

## Core Diagnosis (consensus across all 4 agents)

**LLM-CUCB-AT's κ (inter-query consistency) is fundamentally backwards.** A deterministic-wrong oracle achieves κ=1 (perfect self-consistency), so consistency *rewards* adversarial determinism. Meanwhile ρ (posterior-validation) fails because the hedge set is constructed from the oracle's suggestion → empirical means self-confirm (keep pulling oracle arms → they look best → keep pulling).

Result: the trust mechanism cannot distinguish reliable-deterministic from adversarial-deterministic oracles. Matches Bayley et al. (2025) breakdown at ≥30% preference flipping.

## Theoretical Hook for NeurIPS

**Theorem (Lower Bound, novel):** Any trust mechanism depending only on oracle self-consistency is Ω(T)-regret-vulnerable to deterministic adversaries.

**Theorem (Upper Bound):** Combining divergence-based trust + epoch-wrapper + exploration floor achieves Õ(√mT + C) matching BARBAT's corruption-robust lower bound.

This turns our current *weakness* (4x worse under consistent_wrong) into our *motivating example*.

## Ranked Algorithmic Variants to Test

### Tier 1 — Core Fixes (high consensus)

**V1: Meta-BoBW** (Agent 3 Candidate A, Agent 1 Direction 3, Agent 2 #5)
- Tsallis-INF over {π₁=vanilla CUCB, π₂=LLM-CUCB-AT}
- Meta-learner picks policy per round, updates based on observed reward
- Provable: Regret ≤ Regret(better policy) + O(√T log 2)
- Difficulty 3

**V2: Divergence-Based Trust** (Agent 2 #2)
- Replace κ with KL(oracle_posterior || uniform)
- Penalize *confident* advice: trust decreases when oracle is too sure
- Difficulty 2

**V3: Exploration Floor** (Agent 2 #3)
- Pull uniformly-random super-arm with prob ε_t = t^(-1/3)
- Provably breaks sublinear attack cost
- Difficulty 1

**V4: BARBAT Epoch-Wrapper** (Agent 2 #1)
- Static exponentially-growing epochs with per-epoch δ_m
- Can't be gamed by consistent adversary
- Regret: Õ(√mT + C)
- Difficulty 3

### Tier 2 — Alternative Framings

**V5: Pool Restriction** (Agent 1, Agent 3 Candidate C)
- LLM shortlists pool P of size βm, CUCB runs on P ∪ (random safety m arms)
- Sidesteps per-round trust decision
- Difficulty 1

**V6: Endorsement-Aware Confidence Inflation** (Agent 3 Candidate B)
- Penalty term γ·f_i·(μ̂_i − μ̂_median)₊ punishes arms the oracle over-endorses AND that look good
- Difficulty 2

### Tier 3 — Bigger Pivots (preserve if needed)

**P1: Coverage-Adaptive Thompson Sampling** (Agent 4 Pivot 1)
- LLM produces prior π₀; TS with π₀; regret scales with coverage C(π₀)
- Interpolates O(log T) (good LLM) to O(√T) (useless LLM)

**P2: Verifier-Augmented Bandits** (Agent 4 Pivot 3)
- LLM proposes action + verifier LLM scores it; costly-feedback bandit
- Rides PRM/o1 test-time-compute wave

**P3: Reasoning-Chain Bandits** (Agent 4 Pivot 2)
- k CoT chains → logprob-weighted distribution over arms as prior
- Self-consistency as uncertainty measure

## Round 2 Experiment Plan

Test V1-V6 on the 5 canonical scenarios (perfect, uniform 0.2, uniform 0.5, consistent_wrong 1.0, adversarial 0.3). Report:
- Final regret mean±std (50 seeds, T=30000)
- Runtime per agent
- Verdict: does it fix consistent_wrong?

Winners advance to Round 3 (larger scale, T=100000, more corruption levels).

## Key Papers To Cite

- **BARBAT**: Fang et al. 2025 (arXiv 2502.07514) — corruption-robust framework
- **Tsallis-INF**: Zimmert-Seldin 2021 (JMLR) — BoBW foundation
- **Bayley et al. 2025** — LLM-warm-start breakdown under corruption (direct prior work)
- **Krishnamurthy et al. 2024** — "Can LLMs Explore In-Context?" (NeurIPS)
- **Park et al. 2025** — "Do LLM Agents Have Regret?" (ICLR)
- **Bogunovic et al. 2022** — corruption-robust linear bandits
- **Lykouris-Vassilvitskii 2018** — learning-augmented algorithm framework
- **Cutkosky-Dann-Das-Zhang 2022** — initial hints for free in linear bandits
- **Calibration-gated LLM pseudo-observations** (arXiv 2604.14961, 2025) — 19% regret reduction vs LinUCB

## What to Skip (flagged by agents)

- MoM estimators inside CUCB — useless against *action* corruption (only helps against *reward* corruption)
- Reward clipping/truncation — same reason as MoM
- κ-only consistency — proven backwards by our lower bound
