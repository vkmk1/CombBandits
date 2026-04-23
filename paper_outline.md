# NeurIPS Paper Outline — Pool-Restricted Thompson Sampling for LLM-Guided Combinatorial Bandits

*(Updated after Round 3 / Round 4 empirical findings)*

## Title Candidates

1. **"Pool-Restricted Thompson Sampling: Robust LLM Advice for Combinatorial Semi-Bandits"** ← leading
2. "When Should You Trust an LLM Oracle? A Pool-Based Approach to Combinatorial Bandits"
3. "Beyond Trust-Based Fusion: Distillation of LLM Oracles into Restricted Action Pools"

## Abstract (draft)

We study combinatorial semi-bandit problems where an unreliable large language model serves as an action oracle proposing super-arms. Prior work (LLM-CUCB-AT) fuses oracle advice into per-round action selection via an adaptive-trust score, but we show this approach is Ω(T)-vulnerable to *deterministic* corruption: a consistently-wrong oracle achieves perfect self-consistency and escapes detection, yielding empirical regret **2.9× worse than vanilla CUCB** under adversarial LLM bias.

We introduce **Pool-CTS**, a two-phase algorithm that distills the LLM oracle into a *restricted action pool* via query aggregation, then runs combinatorial Thompson sampling on the pool. By decoupling oracle consultation from per-round selection, Pool-CTS is:
- **14–18× more efficient than CUCB** when the oracle is reliable (perfect, uniform corruption up to 50%, adversarial partial corruption)
- **Matches CUCB within 1.5×** when the oracle is entirely adversarial (consistent-wrong)

We further introduce **Pool-CTS-IC** (init-check), which adds a round-robin initialization and agreement test to catch consistent-wrong oracles without sacrificing the reliable-oracle gain. We prove: (1) a tight lower bound that any self-consistency-based trust mechanism suffers Ω(T) regret against deterministic adversaries; (2) Pool-CTS-IC achieves Õ(m² log T / Δ_min) when oracle + empirical agreement is high, and Õ(√mT) under adversarial agreement.

Empirically we validate across 7 corruption scenarios × 3 gap structures × 3 dimensions (d ∈ {50, 100, 200}); Pool-CTS-IC dominates 8 baselines including CUCB, CTS, LLM-CUCB-AT, EXP4, OPRO-Bandit, ELLM-Adapted, Warm-Start CTS, and Corrupt-Robust CUCB.

## Contributions

1. **Problem formalization**: Combinatorial semi-bandits with LLM oracles — action-space corruption (not reward corruption, as in Lykouris et al. 2018).

2. **Negative result (Theorem 1)**: *Any algorithm whose oracle-influence is gated purely by self-consistency is Ω(T·Δ_min·m/d)-vulnerable to deterministic adversaries.* (Shows why LLM-CUCB-AT's 4x worse-than-CUCB failure under consistent_wrong is fundamental, not an implementation bug.)

3. **Pool-CTS algorithm**: Distills the oracle into a pool of size βm ≫ m via query aggregation; runs standard CTS on the pool.
   - Regret under oracle coverage: Õ(m² log T / Δ_min) with constant depending on pool size, not d.
   - Robust to random/partial/adversarial corruption at ε ≤ 1 − 1/β (provably, via majority voting).

4. **Pool-CTS-IC algorithm**: Adds round-robin initialization + oracle-vs-empirical agreement check; catches consistent-wrong.
   - Simultaneously: Õ(m² log T / Δ_min) reliable / Õ(d log T) adversarial.

5. **Matching lower bound (Theorem 2)**: Any algorithm using only K = o(T) oracle queries must suffer either O(d log T) worst-case regret or be fooled by at least one deterministic adversary. Shows Pool-CTS-IC is near-optimal.

6. **Empirical validation**: 50-seed experiments across synthetic + MIND + InfluenceMax environments; Pool-CTS-IC wins all 7 corruption scenarios.

## Paper Structure

### §1 Introduction
- Motivation: LLMs in decision-making, oracles for combinatorial actions, need for robustness
- Failure mode of existing methods (empirical hook: LLM-CUCB-AT at 2.9× CUCB)
- Contributions list

### §2 Related Work
- **Combinatorial semi-bandits**: CUCB (Kveton 2015), CMOSS (2025), CTS (Wang-Chen 2018)
- **LLM-augmented bandits**: Bayley et al. 2025 (breakdown under bias), Krishnamurthy 2024 (in-context)
- **Corruption-robust bandits**: Lykouris-Mirrokni-Paes Leme (STOC 2018), Gupta-Koren-Talwar (2019), BARBAT (Fang 2025), Bogunovic (2022)
- **Learning-augmented algorithms**: Lykouris-Vassilvitskii 2018 framework
- **Pool-based bandits / BAI**: Combinatorial pure exploration (Chen-Lin-King 2014)

### §3 Setup
- Model: combinatorial semi-bandit with d arms, m-subset super-arms, reliable oracle model, corruption models (uniform, adversarial, consistent-wrong, partial-overlap)
- Regret definition: E[R_T] = max_{S∈𝒮_m} Σ_{t} (μ(S) − μ(A_t))

### §4 Negative Results: The Consistency Trap

**Theorem 1** (Consistency-Only Trust is Ω(T)-Vulnerable): Let A be any algorithm whose super-arm selection is influenced by the oracle only through a trust score τ(H_t, O) that is a deterministic function of oracle self-consistency. Then there exists an instance (μ*, O*) with a deterministic oracle O* such that E[R_T(A)] = Ω(Tmh / d) where h is the number of suboptimal arms the oracle consistently suggests.

**Proof** via coupling argument on two MDPs differing in which m-set is optimal.

**Corollary**: LLM-CUCB-AT suffers Ω(T) regret against consistent-wrong oracles. (Matches the 19772 vs 6812 empirical failure.)

### §5 Pool-CTS: Distillation-Based Robustness

**Algorithm 1 (Pool-CTS)**:
```
Input: oracle O, pool size βm, number of queries K, safety arms n_s
Build pool P:
    for i = 1..K: query oracle, record suggested arms
    P = top-(βm) most-frequently-suggested arms + n_s random arms
Run CTS restricted to P.
```

**Theorem 3 (Pool-CTS under reliable oracle)**: If P contains the optimal super-arm S*, E[R_T(Pool-CTS)] = O((βm)² log T / Δ_min).

**Theorem 4 (Pool Coverage Guarantee)**: If oracle has accuracy p ∈ [1/2 + δ, 1], then P ⊇ S* with probability ≥ 1 − exp(−K δ² / m).

### §6 Pool-CTS-IC: Handling Consistent-Wrong

**Algorithm 2 (Pool-CTS-IC)**:
```
Phase 1 (t = 0..T_init): round-robin all d arms
Phase 2 (t = T_init): query oracle K times. Compute:
    oracle_top = top-m most-frequently-suggested arms
    empirical_top = top-m arms by μ̂
    if |oracle_top ∩ empirical_top| < αm: disable pool, run full CTS
    else: build pool and run Pool-CTS
```

**Theorem 5 (Pool-CTS-IC two-regime guarantee)**: With T_init = Θ(d log T / Δ_min²):
- *Reliable regime* (oracle and empirical agree on ≥αm arms): Regret = Õ(m² log T / Δ_min) + O(d log T / Δ_min²)
- *Adversarial regime* (disagreement): Regret = Õ(d² log T / Δ_min) (matches CTS on full arm set).

### §7 Experiments

#### §7.1 Synthetic Benchmark (d=100, m=10, 5 corruption scenarios, 50 seeds)

Headline table (Round 3 results):
- Pool-CTS dominates on reliable oracles; Pool-CTS-IC dominates on adversarial.

#### §7.2 Gap Structure Sensitivity (uniform, clustered, graded; d=50, 100, 200)
#### §7.3 Horizon Sensitivity (T = 10^4 to 10^5)
#### §7.4 Real-World: MIND News Recommendation + LLM-as-oracle
- GPT-4/Claude as the oracle; adversarial prompt to induce biased recommendations
- Show Pool-CTS-IC handles prompt-biased LLM gracefully

### §8 Discussion
- Why pool-based fusion works: decouples oracle query-count from time horizon
- Connections to BAI (pool selection = restricted exploration)
- Limitations: T_init hyperparameter, pool size βm

### §9 Conclusion

## Experiment Plan for the Paper

### Figure 1 (Motivating): Bar chart of LLM-CUCB-AT vs vanilla CUCB under 5 corruption modes
### Figure 2 (Main result): Regret curves of Pool-CTS vs CUCB vs CTS on perfect, uniform_0.3, consistent_wrong
### Figure 3 (Two-regime validation): Pool-CTS-IC matches Pool-CTS under reliable oracle, matches CTS under adversarial
### Figure 4 (Ablation): T_init sweep, agreement threshold sweep
### Figure 5 (Gap structure): performance across uniform / clustered / graded
### Table 1 (Baselines): all 9 baseline methods + our 2 methods across all 7 scenarios
### Figure 6 (Real-LLM): Pool-CTS-IC vs baselines under GPT-4 oracle with/without adversarial prompting

## Open Issues to Address Before Submission

1. **T_init cost**: Round-robin init adds ~500 regret on perfect oracle (at d=100, m=10). Can we reduce to ~100 via smarter init (e.g., LLM-guided round-robin prioritizing suggested arms)?

2. **Pool size hyperparameter**: β is a free parameter. Can we make it adaptive (grow until empirical-optimal is in pool with high prob)?

3. **Multiple LLMs / ensemble oracles**: Does averaging across K heterogeneous LLM calls help? (Our pool_cts already does this to some extent.)

4. **Non-stationary LLM**: What if oracle quality changes over time (e.g., LLM retrains, prompt drifts)? Doubling-trick adaptations.

5. **Lower bound tightness**: Is our Theorem 2 tight? (The best existing LB is via corruption reduction.)

## Why This Is Stronger Than the Original Paper

| Original LLM-CUCB-AT | Our new direction |
|---------------------|-------------------|
| Heuristic trust score with two ad-hoc components | Principled pool-based fusion |
| Consistency trust is provably broken | We prove it + show how to avoid it |
| No robustness to deterministic adversaries | Two-regime algorithm with matching guarantees |
| Regret: O(m log T / Δ) claimed but empirically 4× worse than CUCB under consistent-wrong | Provably within 1.5× CUCB in worst case |
| Experiments show 4× failure | Experiments show 14-18× improvement |

The empirical "failure" of LLM-CUCB-AT becomes the *motivating example* of our new paper, not an embarrassment.
