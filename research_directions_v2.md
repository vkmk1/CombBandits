# Novel Research Directions — Post-Loop-2 Synthesis

## Arena Results Summary (21 algorithms, 30 random configs)

| Metric | Winner | Score |
|--------|--------|-------|
| Overall mean regret | pool_cts_doubling | 640 |
| Consistent_wrong | warm_start_cts | 545 |
| Partial_overlap | freq_pool_ts | 229 |
| Adversarial | pool_cts | 290 |
| Win rate | freq_pool_ts | 37% |
| Best worst-case | freq_pool_cts_dual | 1653 |
| Best normalized | warm_start_cts | 0.347 |

**Core tension**: No single algorithm dominates all metrics. The Pareto frontier includes:
- pool_cts_doubling (best overall mean)
- freq_pool_cts_dual (best worst-case + strong consistent_wrong)
- freq_pool_ts (highest win rate)
- warm_start_cts (best consistent_wrong + normalized)

## Literature Landscape (2023-2026)

### Directly Competing Work
1. **LIBRA** (Cao et al., arXiv 2026): Combines suboptimal LLM oracles with bandits for treatment planning. Closest to our setup — but they don't study action-space corruption or pool-based distillation.
2. **LLM-Enhanced MABs** (Sun et al., 2025): LLMs as regression oracles within TS. Single-arm MABs, not combinatorial.
3. **Bouneffouf & Feraud (AAAI 2026)**: Survey of MABs + LLMs. Positions the space but no novel algorithms.
4. **Baheri & Alm (2023)**: LLMs-augmented contextual bandits. First to propose LLM augmentation but no corruption analysis.

### Theoretical Foundations
5. **BARBAT** (Hu & Chen, NeurIPS 2025): Near-optimal corruption-tolerant framework for stochastic bandits. Our epoch-doubling mirrors their approach.
6. **Chen et al. (2026)**: BoBW guarantees for m-set semi-bandits. Directly relevant to our combinatorial setting.
7. **Li et al. (2025)**: Efficient BoBW for contextual combinatorial semi-bandits.
8. **Erez & Koren (2025)**: Lower bounds Ω(sm/ε²) for sparse combinatorial settings.
9. **Wen (2025)**: Adversarial combinatorial semi-bandits with graph feedback.

### Learning-Augmented / Predictions Framework
10. **Blum & Srinivas (SODA 2025)**: Competitive strategies for warm-start with predictions.
11. **Bhaskara et al. (2022)**: Online learning with queried hints.
12. **Lyu & Cheung (ICML 2023)**: Bandits with knapsacks + ML advice.
13. **Li et al. (NeurIPS 2023)**: Beyond black-box advice for MDPs.

### LLMs as Bandit Agents
14. **Krishnamurthy et al. (NeurIPS 2024)**: LLMs fail at in-context exploration. Our Pool-CTS sidesteps this entirely.
15. **Monea et al. (2024)**: LLMs are in-context bandit RL learners.
16. **Zhang et al. (NeurIPS 2023)**: In-context MABs via supervised pretraining.

### Pool-Based / Action Restriction
17. **Harsha et al. (ICML 2025)**: Action pool-based IGW sampling for structured discrete optimization.

---

## Novel Research Directions

### Direction 1: Pool-CTS as a Learning-Augmented Algorithm
**Key insight**: Pool-CTS fits perfectly into the "algorithms with predictions" framework (Lykouris-Vassilvitskii 2018, Blum-Srinivas 2025). The LLM oracle provides a "prediction" — a pool P — and we need:
- **Consistency**: O(m² log T / Δ) when prediction is good (P ⊇ S*)
- **Robustness**: O(d log T / Δ) when prediction is bad (P ∩ S* = ∅)
- **Smoothness**: Graceful interpolation between regimes

This reframing gives us access to the entire learning-augmented algorithm toolkit. **Novel contribution**: First instantiation of the predictions framework for combinatorial semi-bandits with action-space advice.

**Experiment**: Sweep over oracle quality (ε from 0 to 1) and plot regret vs ε. Show the consistency-robustness tradeoff curve. Compare to theoretical bounds.

### Direction 2: Adaptive Pool Size via Information-Theoretic Bounds
**Problem**: We currently fix β=3 (pool size = 3m). But optimal β depends on oracle quality.
- If oracle is perfect: β=1 suffices (just use the oracle suggestion)
- If oracle is garbage: β=d/m (full arm set, i.e., vanilla CTS)

**Idea**: Use the oracle's empirical coverage (what fraction of the top-m arms are in the pool?) to adaptively adjust β. Information-theoretically:
- After K oracle queries, the pool covers S* with prob ≥ 1 − exp(−K·δ²/m) where δ is oracle accuracy margin
- So optimal K ∝ m·log(1/δ_fail) / δ²

**Novel contribution**: First adaptive pool-sizing algorithm with provable coverage guarantees.

**Experiment**: Implement AdaptivePoolCTS that starts with β=2 and doubles pool size whenever empirical evidence suggests the pool doesn't contain S*. Compare to fixed-β variants.

### Direction 3: Multi-Oracle Ensemble Pooling
**Problem**: What if we have access to multiple LLMs with different strengths?

**Idea**: Query K₁ times from Oracle 1 (e.g., GPT-4), K₂ times from Oracle 2 (e.g., Claude), etc. Build a joint frequency map and construct the pool from the ensemble. The ensemble is more robust than any single oracle because:
- Different LLMs have different failure modes
- Majority voting across heterogeneous oracles beats any single oracle

**Novel contribution**: First multi-oracle combinatorial bandit algorithm. Connects to the multi-source advice literature.

**Experiment**: Simulate 2-3 oracles with different corruption types. Show ensemble pooling outperforms any single oracle. Test: (1) all oracles reliable, (2) one adversarial + two reliable, (3) all partially corrupted differently.

### Direction 4: Oracle Query Budget Optimization
**Problem**: Oracle queries cost money/time. How should we allocate a budget of K total queries across time?

**Tradeoff**:
- Front-loaded: build pool once at t=0 (current approach) — wastes nothing on bad pool
- Spread out: query periodically (doubling approach) — adapts to changing oracle quality
- Adaptive: query more when uncertain, less when confident

**Novel contribution**: Oracle-budget-constrained combinatorial bandits. Prove that O(log T) query epochs suffice and derive the optimal epoch schedule.

**Experiment**: Fix total oracle budget K ∈ {10, 50, 200, 1000}. Compare allocation strategies. Show adaptive allocation dominates fixed schedules.

### Direction 5: Pool Construction as Best-Arm Identification
**Key insight**: Building the pool is implicitly solving a BAI (best-arm identification) problem! We're trying to identify the best m arms using noisy oracle signals.

**Connection**: Oracle queries are like noisy comparisons in dueling bandits / BAI. Each query returns a set of m "recommended" arms — a noisy signal about which arms are best.

**Novel contribution**: Reduce pool construction to combinatorial BAI with noisy set-valued feedback. Import algorithms from the BAI literature (LUCB, Track-and-Stop, KL-Racing) for more efficient pool building.

**Experiment**: Implement LUCB-Pool that uses confidence intervals on oracle frequency to decide when the pool is "sufficiently identified." Compare sample efficiency to our current fixed-K approach.

### Direction 6: Contextual Pool-CTS
**Problem**: Oracle reliability may depend on context (e.g., which user, which category).

**Idea**: In contextual combinatorial bandits, build context-dependent pools. The LLM oracle receives context x_t and returns a pool suggestion. Different contexts may have different optimal pools.

**Formulation**: For each context cluster c, maintain a separate pool P_c. Route incoming contexts to their cluster and run Pool-CTS within P_c.

**Novel contribution**: First contextual extension of pool-based LLM-oracle bandits.

### Direction 7: Tighter Lower Bounds for Action-Space Corruption
**Gap**: Our Theorem 1 shows consistency-only trust is Ω(T)-vulnerable. But what's the fundamental lower bound for ANY algorithm facing action-space corruption?

**Conjecture**: Any algorithm using an ε-corrupted action oracle must suffer:
- Ω(ε·m·log(T)/Δ) in the reliable regime (ε < 1/2)
- Ω(d·log(T)/Δ) in the adversarial regime (ε ≥ 1/2)

**Why novel**: Action-space corruption is fundamentally different from:
- Reward corruption (Lykouris et al. 2018): adversary modifies reward, not action
- Expert advice corruption (Agarwal et al.): experts are full policies, not set-valued hints
- Our setting: oracle proposes a set of m arms, some may be suboptimal

**Experiment**: Validate lower bound empirically by plotting regret vs ε for best algorithm. Check if scaling matches Ω(ε·m·log(T)/Δ).

### Direction 8: Pareto-Optimal Algorithm Selection
**Observation from arena**: The Pareto frontier has 4-5 algorithms. No single algorithm dominates.

**Idea**: Rather than searching for ONE best algorithm, design a meta-algorithm that provably sits on the Pareto frontier of (mean regret, worst-case regret, consistent_wrong regret).

**Approach**: Run freq_pool_cts_dual (best worst-case) as the "safe" policy and pool_cts_doubling (best mean) as the "aggressive" policy. Use a Tsallis-INF meta-learner to select between them. The meta-learner automatically routes to the safer policy when it detects adversarial conditions.

**Novel contribution**: First Pareto-optimal meta-algorithm for LLM-oracle combinatorial bandits.

**Experiment**: Implement MetaPoolCTS with 2-3 base policies. Show it achieves near-best performance across ALL metrics simultaneously.

---

## Prioritized Experiment Queue (for Loop 3+)

1. **MetaPoolCTS** (Direction 8) — directly addresses the Pareto gap from arena
2. **AdaptivePoolSize** (Direction 2) — low-hanging fruit, potentially big improvement
3. **LUCB-Pool** (Direction 5) — novel theoretical connection
4. **OracleBudget** (Direction 4) — practical relevance for real LLM calls
5. **MultiOracle** (Direction 3) — testable with simulated oracles now
6. **ConsistencyRobustness curve** (Direction 1) — needed for paper framing
7. **LowerBoundValidation** (Direction 7) — theoretical contribution
8. **ContextualPool** (Direction 6) — extension, lower priority

## Additional Critical Papers (from deep literature survey)

### LLM-as-Oracle (Closest Competitors)
- **LIBRA** (Cao et al., 2026): LLM oracle + bandits for treatment planning
- **Jump Starting Bandits with LLM-Generated Prior Knowledge** (Alamdari, Cao, Wilson, EMNLP 2024): LLMs warm-start TS
- **Jump Start or False Start?** (Bayley et al., 2026): When LLM priors help vs hurt — directly validates our corruption analysis
- **Robustness of LLM-Initialized Bandits Under Noisy Priors** (Bayley et al., 2025): Degradation under misspecification
- **When Do We Need LLMs?** (Berdica et al., 2026): Diagnostic for when LLM reasoning adds value

### Combinatorial Bandits Theory
- **CTS Polynomial Regret** (Zhang & Combes, NeurIPS 2024): First polynomial finite-time CTS bound for linear combinatorial
- **CTS with Approximation Oracles** (Perrault, 2023): CTS under approximation — relevant to LLM-oracle setting
- **Covariance-Adaptive Semi-Bandits** (Zhou et al., NeurIPS 2024): TS-inspired with tight bounds
- **Contextual Semibandits via Supervised Learning Oracles** (Krishnamurthy et al., NeurIPS 2016): Structural ancestor — reduction from combinatorial to oracle

### Corruption & Robustness
- **CRIMeD** (Agrawal et al., 2024): Tight bounds under stochastic corruption — relevant if LLM errors are random not worst-case
- **Robust Contextual CMAB** (Wang et al., IEEE INFOCOM 2025): L1-norm corruption budget for contextual combinatorial
- **Liu et al. (NeurIPS 2024)**: Minimax-optimal corruption-robust linear bandits
- **Bandits with Abstention under Expert Advice** (Pasteris et al., NeurIPS 2024): Algorithms that can abstain — relevant to pool "opt-out"
- **Learning When to Trust in Contextual Bandits** (Ghasemi & Crowley, 2026): Context-dependent corrupt expert advice

### BoBW
- **Tsuchiya, Ito, Honda (AISTATS/ICML 2023)**: Variance-dependent BoBW for combinatorial semi-bandits
- **Dann, Wei, Zimmert (COLT 2023)**: Blackbox BoBW reduction applicable to combinatorial
- **Chen, Lee, Kim, Honda (2026)**: Tightened lower bounds for m-set semi-bandits

### Meta-Learning / In-Context
- **Meta-Learning Adversarial Bandit Algorithms** (Khodak et al., NeurIPS 2023): Meta-learns across stochastic + adversarial
- **In-Context MABs via Supervised Pretraining** (Zhang et al., NeurIPS 2023): Near-optimal through pretraining
- **Foundation Model Exploration** (Sasso et al., 2025): Hybrid model-bandit; finding that LLMs show NO sensitivity to experimental feedback

### Applications
- **Combinatorial Neural Bandits** (Hwang et al., ICML 2023): Neural-network-based combinatorial
- **Cocob** (Yan et al., 2025): Adaptive collaborative combinatorial bandits for recommendation
- **Harsha et al. (ICML 2025)**: Pool-based IGW for structured discrete optimization

---

## Positioning Statement

**The gap our paper fills**: The LLM-oracle niche for combinatorial bandits is genuinely open. Existing work covers:
- (a) LLMs as in-context MAB solvers (Krishnamurthy et al. 2024) — not combinatorial, no formal guarantees
- (b) LLM-generated priors for simple bandits (Alamdari et al. 2024) — not combinatorial, no corruption analysis
- (c) LLMs for combinatorial optimization without bandit loop (Jiang et al. 2025) — offline, not online

Our intersection — using an LLM oracle within a combinatorial bandit algorithm with formal regret guarantees that degrade gracefully with oracle quality — is a **genuine gap**. The "algorithms with predictions" framework (Blum & Srinivas; Drago et al.) gives theoretical scaffolding, and Krishnamurthy's contextual semibandits with oracles (NeurIPS 2016) is our closest structural ancestor.

Key differentiators:
1. **Action-space corruption model** (not reward corruption as in BARBAT)
2. **Pool-based distillation** (not per-round trust as in LLM-CUCB-AT)
3. **Consistency trap lower bound** (first impossibility result for self-consistency trust)
4. **21-algorithm randomized arena** (strongest empirical validation in the LLM+bandits space)
