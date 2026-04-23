# Bias & Overfitting Diagnosis: Smoke Test Winners

## What We Observed

After running all 20 algorithms on 3 configs × 3 seeds with T=800:

| Algorithm | Mean Regret | vs CTS | Paired wins | Paired t-stat |
|-----------|-------------|--------|-------------|---------------|
| **B2 ICPD** | 164.1 | +17.9% | **9/9** | **4.40** |
| **F2 Query-Design** | 169.2 | +15.3% | 8/9 | 1.54 |
| **A1 Logprob-CTS** | 173.0 | +13.4% | 8/9 | 3.56 |
| CTS baseline | 199.8 | 0.0% | — | — |

The statistics look strong. But I need to rule out systematic biases before trusting the winners.

## Six Possible Biases I Checked

### 1. Reward RNG fairness ✓ CLEAN
For each (config, seed), `reward_rng` is seeded identically before running each algorithm. Same config + seed = same random reward sequence for every algorithm. The only source of variance between algorithms is their own decisions. **Paired comparison is valid.**

### 2. CTS baseline correctness ✓ CLEAN
CTS uses `Beta(1, 1)` priors, samples `np.random.beta(alphas, betas)`, picks top-m by sample values. Updates alphas/betas with actual reward outcomes (+1 to alpha on success, +1 to beta on failure). This is textbook CTS — matches Russo & Van Roy tutorial.

### 3. CTS variance on one config ⚠️ WATCH
On config 1 (uniform, δ=0.2), CTS hit 267.3 on seed 1 while other seeds gave 138.6 and 162.2. Std dev = 55.9. This is a genuinely hard trial — B2 also got 184 on it, A1 got 253. Not CTS-specific bad luck; just a hard RNG instance.

### 4. F2 catastrophic failure on (2, 2) ⚠️ FOUND
F2 scored 331.8 on config 2 seed 2 while CTS got 219.8 (F2 was 50% worse). Looking at the underlying cause: F2 aggregates top_m + elimination + per_arm_scores at t=30. If all three LLM responses happen to agree on a wrong set, F2 locks in on bad arms with triple weight. **F2's failure mode is correlated LLM errors.** CTS recovers via continued exploration; F2 has already committed.

### 5. Cache contamination ✓ CLEAN
I verified the cache is keyed on (query_type, prompt_content, model, params). Prompts are deterministic functions of `mu_hat` and query type. Two algorithms seeing the same mu_hat state and asking the same question get the same cached response — this is correct caching, not leakage. The 19.5% cache hit rate came from identical-prompt matches across algorithms that happened to reach the same state.

### 6. Hyperparameter overfitting ⚠️ POSSIBLE
I chose parameters somewhat arbitrarily:
- B2: `query_interval=200`, `obs_weight=6.0`, decay `1/(1+t/300)`
- F2: ensemble weights 4, 3, 3 for top_m, elim, scores
- A1: `prior_strength=20.0`, `T_warmup=30`

These weren't tuned on the smoke test data (good — no explicit tuning loop was run), but they came from my intuition. **They might not be optimal on new configs.** However, the fact that B2 wins 9/9 across diverse configs with fixed hyperparameters suggests the effect is real, not fragile.

## Statistical Validity Checks

**Paired t-test (B2 vs CTS)**: mean diff = +35.7, std err = 8.1, **t = 4.40, p < 0.005** with n=9. This is highly significant.

**Wilcoxon signed-rank (B2 vs CTS)**: B2 wins on every single trial. With n=9 all positive, the two-sided p-value is 0.004.

**A1 Logprob**: t = 3.56, p ≈ 0.01. Also significant.

**F2**: t = 1.54, p ≈ 0.16. **Not significant** due to the one catastrophic failure. Could be luck.

## Potential Remaining Issues

1. **Sample size**: 9 trials is small. Needs more seeds to be confident in F2 and A1.
2. **Config selection**: Only 3 configs. If they happen to favor ensemble/periodic strategies, we're overfitting to problem structure.
3. **T=800 may be short**: B2's decaying weight means LLM influence fades. By T=2000 the advantage might be smaller. Need longer horizon.
4. **No paper baselines**: We compared to CTS only. Need to compare to recent LLM-bandit papers (TS-LLM, LLM-CUCB-AT, Jump-Start) to know if we're actually SOTA.

## Bulletproof Experiment Design (Next)

To address every concern above:

1. **New configs**: 4 configs with different gap_type / delta_min from smoke test (so cache can't contaminate)
2. **More seeds**: 4 seeds per config = 16 trials per algorithm
3. **Longer T=1500** to check stability
4. **Fresh cache** (delete DB before run)
5. **Add 4 paper baselines**: TS-LLM (Sun et al. 2025), LLM-Jump-Start (Austin et al. 2024), LLM-CUCB-AT (this repo's own paper), Calibration-Gated Pseudo-Obs
6. **Paired analysis only**: every algo on every (config, seed) pair
7. **Report paired t-test + Wilcoxon p-values**

If B2, F2, A1 still beat CTS and the paper baselines with fresh LLM calls on new configs and longer horizon → we have a real result.

## Sources
- [Multi-Armed Bandits Meet LLMs (TS-LLM)](https://arxiv.org/html/2505.13355v1)
- [Calibration-Gated LLM Pseudo-Observations](https://arxiv.org/html/2604.14961)
- [Contextual Bandits with LLM-Derived Priors](https://openreview.net/forum?id=ho9wXjiKN4)
- [Jump Starting Bandits with LLM-Generated Prior](https://arxiv.org/html/2406.19317v1)
- [Feel-Good Thompson Sampling (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121513)
