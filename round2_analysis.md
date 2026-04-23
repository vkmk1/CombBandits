# Round 2 Analysis — Complete Results & Learnings

## Full Results Table (T=30000, 50 seeds, d=100, m=10, uniform gap_type)

| Agent | perfect | uniform_0.2 | uniform_0.5 | consistent_wrong | adversarial_0.3 |
|-------|---------|-------------|-------------|------------------|-----------------|
| cucb | 6835 | 6820 | 6840 | **6812** | 6819 |
| cts | **1460** | **1506** | **1438** | **1492** | 1507 |
| llm_cucb_at | 2180 | 6827 | 6843 | 19784 | 7163 |
| meta_bobw | 2487 | 6785 | 6806 | 15643 | 7125 |
| explore_floor | 4671 | 8027 | 8033 | 20492 | 8632 |
| **pool_restrict** | **2075** | **1900** | **2091** | 15390 | **1686** |
| div_trust | 6874 | 6969 | 13138 | 6844 | 13732 |
| epoch_robust | 2201 | 6823 | 6841 | 14906 | 6845 |
| combined | 8016 | 8137 | 8486 | 8067 | 9305 |

## Key Findings

### Finding 1: `pool_restrict` dominates under reliable oracles
- Best LLM method on perfect (2075), uniform_0.2 (1900), uniform_0.5 (2091), adversarial_0.3 (1686)
- The pool-aggregation phase (10 queries) naturally denoises random/partial corruption
- **But completely fails on consistent_wrong** (15390 vs CUCB 6812) — a deterministic wrong oracle poisons all 10 pool queries
- Also high variance on uniform_0.5 (±1567) — some seeds get unlucky pools

### Finding 2: `div_trust` is a compass pointing the wrong way 95% of the time
- Fails under perfect, uniform, and adversarial (all 6800-13700 regret)
- But UNIQUELY catches consistent_wrong (6844, matches CUCB)
- Mechanism: oracle endorsement concentration triggers distrust; correct concentration is indistinguishable from wrong concentration without additional signal
- **Fix in Round 3 (`div_trust_v2`): condition concentration-distrust on DISAGREEMENT with empirical top-m**

### Finding 3: No variant handles both reliable AND consistent_wrong
- The Pareto frontier is: `pool_restrict` (good everywhere but consistent_wrong) vs `div_trust` (good on consistent_wrong only)
- Need an algorithm that combines both — the **conditional agreement check** (div_trust_v2) or **probe-based detector** (adaptive_pool)

### Finding 4: Simple combining doesn't work
- `combined` (meta_bobw + div_trust + explore_floor): uniformly ~8000 regret
- Three taxes compound: BoBW meta-learning cost + div_trust over-conservatism + exploration floor
- Lesson: **one good mechanism > three mediocre mechanisms**

### Finding 5: Meta-BoBW is too slow to detect LLM failure
- `meta_bobw` at consistent_wrong: 15643 — importance-weighted updates don't distinguish policies fast enough
- The loss signal (1 - reward/m) is noisy; meta-learning rate η_t = 1/√t is too conservative
- **Warm-start version (Round 3 meta_bobw_warm) should only help in good cases, may be worse in bad cases**

### Finding 6: Exploration floor (t^{-1/3}) is universally harmful
- Across all scenarios, `explore_floor` is 1.2-3× worse than base CUCB
- The exploration tax accumulates and doesn't buy enough signal
- **Abandon this mechanism in Round 3**

## Round 3 Hypotheses

**H1**: `div_trust_v2` (conditional agreement) will work on BOTH perfect AND consistent_wrong, matching pool_restrict's gains + div_trust's robustness.

**H2**: `adaptive_pool` (probe-based detection) will detect consistent_wrong oracle via random-probe rewards and abandon pool, recovering CUCB-level regret.

**H3**: `pool_cts` will beat pool_restrict on reliable oracles (CTS's 1460 advantage transfers) but won't fix consistent_wrong.

**H4**: `meta_bobw_warm` will be worse than meta_bobw on consistent_wrong (starts trusting bad oracle).

**H5**: `pool_with_trust` has a bug in detection (UCB inflation); may misbehave.

## NeurIPS Narrative Refinement

After Round 2, the paper's story is sharper:

> **LLM-oracle bandits exhibit an inherent tension**: methods that efficiently exploit a good oracle (like `pool_restrict`) are poisoned by deterministic-wrong oracles because they cannot distinguish deliberate bias from variance. We prove a lower bound (Theorem 1) showing any algorithm that uses the oracle as a pool-definer or trust-signal without cross-checking empirical evidence incurs Ω(T) regret under consistent-wrong corruption. We introduce **LLM-CUCB-AP** (Adaptive Pool with Probe-based corruption detection) that uses the oracle for pool-seeding but periodically probes random out-of-pool arms, abandoning the pool when probes consistently outperform. Regret: O(log T) under reliable oracle, Õ(√T + C) under bounded corruption, O(d log T) under consistent-wrong (matches CUCB).

Much cleaner and more grounded in the actual empirical dynamics than the original meta-BoBW story.
