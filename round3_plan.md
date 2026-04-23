# Round 3 Plan — Refinements Based on Round 2 Emerging Data

## Round 2 Perfect-Oracle Findings (partial, in progress)

| Agent | Regret | Insight |
|-------|--------|---------|
| cts | 1460 | Thompson with Beta gives sharp concentration — possible to borrow |
| pool_restrict | **2075** | **Winner among LLM methods.** Pool framing sidesteps trust entirely |
| llm_cucb_at | 2180 | Original baseline |
| epoch_robust | 2201 | Neutral — epoch resets don't hurt under perfect |
| meta_bobw | 2487 | BoBW overhead visible (~15% cost under perfect) |
| explore_floor | 4670 | Exploration floor too aggressive |
| div_trust | **6874** | **FAILURE** — divergence distrust triggers on correct concentration |
| cucb | 6835 | baseline |

## Critical Design Flaw Identified: Divergence Trust

The current `div_trust` penalizes oracles whose endorsement distribution is concentrated on m arms. But a *correct* deterministic oracle also concentrates on m arms. The divergence measure cannot distinguish correct-concentration from wrong-concentration without reward signal.

### Fix for Round 3: **Conditional-Divergence Trust**

Only penalize concentration *when the concentrated arms don't agree with empirical top-m*. Formally:

```
agreement = |oracle_endorsed_topm ∩ mu_hat_topm| / m
if agreement >= 0.7:   # oracle concentrated on right arms
    trust = high
else:                  # oracle concentrated on wrong arms
    trust = low
```

This turns the divergence-trust into a **cross-check** between oracle concentration and empirical evidence. Should preserve perfect-oracle performance while still catching consistent-wrong.

## Round 3 Variants

### Priority 1: Fixed Divergence Trust
**V2b: `div_trust_v2`** — conditional concentration check (above). Should beat both original `div_trust` (fixes perfect-oracle failure) and `llm_cucb_at` (catches consistent-wrong).

### Priority 2: Hybrid Pool + Trust
**V5b: `pool_with_trust`** — pool_restrict (winner) + divergence-trust monitor. If pool turns out bad (detected via divergence), expand pool or fall back to CUCB.

### Priority 3: Tuned Meta-BoBW
**V1b: `meta_bobw_warm`** — initialize log_weights strongly toward π₂ (LLM policy) to avoid meta-learning tax in the good case. Only shifts to π₁ when needed.

### Priority 4: Pool + Exploration Floor
**V7b: `pool_explore_combo`** — pool_restrict + small exploration floor (ε_t = t^(-1/2) instead of t^(-1/3)). Adds safety without heavy tax.

### Priority 5: CTS-Based Alternative
**V8: `pool_cts`** — pool_restrict with CTS (Thompson) inside the pool instead of UCB. Should leverage CTS's 1460 advantage in perfect-oracle regime.

## Round 3 Experiment Design

- Same 5 scenarios as Round 2
- T=30000, 50 seeds (same as Round 2 for direct comparison)
- Add 2 new scenarios to stress-test robustness:
  - **partial_overlap_0.3**: oracle gets 70% right, 30% wrong (graded quality)
  - **adversarial_0.6**: high-corruption adversarial

## Decision Criteria for Round 3 → Round 4

**Green light** (proceed to real-LLM experiments):
- Any variant beats both `cucb` AND `llm_cucb_at` across *all* 5 scenarios
- Winner's consistent_wrong regret ≤ 1.5× `cucb` regret

**Yellow light** (iterate more):
- Winner works in 4/5 scenarios but fails one specific case — fix that case

**Red light** (pivot):
- No variant consistently beats baselines — explore Pivot 1 (Coverage-Adaptive TS) or Pivot 3 (Verifier-Augmented Bandits)
