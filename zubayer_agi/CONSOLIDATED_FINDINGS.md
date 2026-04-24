# Consolidated Research Findings — LLM-Guided Combinatorial Bandits

**Status**: End of research-and-iteration phase. Prior to full-scale experiment.

## 1. Problem Statement

Combinatorial semi-bandit: learner selects subset of m arms from d candidates
each round, observes per-arm rewards, minimizes cumulative regret over horizon T.
We augment with a real LLM oracle that can be queried at a cost.

**Goal**: beat plain Combinatorial Thompson Sampling (CTS, Wang & Chen 2018) on
combinatorial bandits when augmented with a real (not simulated) LLM oracle.

## 2. Key Empirical Discoveries (in chronological order)

### Discovery 1 — Endogenous Oracle Quality
LLM response quality depends on the mu_hat we send it. Round-robin warmup gives
0/5 oracle accuracy; CTS warmup gives 4/5 at the same compute budget. **This
is the central empirical insight of the project.** No published paper models this.

### Discovery 2 — "Structure Beats Belief"
LLM outputs parsed as Beta-posterior corrections (belief-style) are fragile
(B2 ICPD: +24% in small tests, regressed to +4% with rigorous evaluation).
LLM outputs parsed as sampling-correlation structure (CORR-CTS) are robust
(+8% consistently across 3 LLMs, p < 0.001).

### Discovery 3 — Information Content Scales With Model Quality
Per-call I(LLM_output; optimal_set) measured from our cache:
- gpt-4.1-mini: 9.13 bits (53% of 17.12 max)
- gpt-5-mini: 9.32 bits (54%)
- gpt-5.4: 10.36 bits (60%)
All 3 LLMs give 78-89% top-m accuracy but differ in calibration.

### Discovery 4 — Full-Kernel Covariance Beats Block-Diagonal
N1_corr_full (RBF kernel over LLM-cluster-rank) beats M2_corr_cts (block within-cluster
correlation) by +9.5% in paired comparison. More expressive covariance structure pays off.

### Discovery 5 — Credibility Gating Provides Worst-Case Robustness
N4_robust_corr interpolates between LLM-correlated and independent sampling based on
validation overlap. Matches best LLM-informed results in cooperative setting,
degrades gracefully to CTS if LLM wrong. +18% vs CTS globally.

## 3. Algorithms We Built and Evaluated (20 total)

**Baselines (published prior art)**:
- CTS (Wang & Chen 2018), CUCB
- TS-LLM (Sun et al. 2025)
- LLM-Jump-Start (Austin et al. 2024)
- LLM-CUCB-AT (Kakaria, this repo's paper)
- Calibration-Gated LLM Pseudo-Observations (2024)

**Our earlier iterations (many abandoned)**:
- 17 initial "belief-injection" algorithms (A1-G2). Most failed statistical significance.
- B2 ICPD (periodic counterfactual injection): strong in small tests, regressed.
- F2 Query-Design (3-query ensemble): high variance, tail failures.

**Our surviving contributions**:
- **M2 CORR-CTS** (correlated sampling, block-diagonal): first LLM-structural algorithm.
- **N1 CORR-Full** (full kernel covariance from LLM clusters): expressive extension.
- **N4 Robust-CORR** (credibility-gated interpolation): formal worst-case guarantee.
- **N5 CORR-Full-Robust** (synthesis of N1+N4): to be tested in full-scale.

## 4. Final Performance (Tier 3, 108 paired trials, with caveats)

| Algorithm | Mean Regret | vs CTS | Paired wins | Sign p |
|-----------|-------------|--------|-------------|--------|
| N4_robust_corr | 270.1 | +18.0% | 84/108 | < 0.0001 |
| N1_corr_full | 272.5 | +17.3% | 86/108 | < 0.0001 |
| M2 CORR-CTS | 300.8 | +8.7% | 80/108 | < 0.0001* |
| PAPER_cal_gated | 305.5 | +7.3% | 78/108 | < 0.0001 |
| PAPER_ts_llm | 308.6 | +6.3% | 81/108 | < 0.0001 |
| OURS_B2_icpd | 315.1 | +4.4% | 67/108 | 0.016* |
| OURS_F2 | 322.3 | +2.2% | 69/108 | 0.005* |
| cts_baseline | 329.5 | 0% | — | — |

*= marginal under pseudo-replication correction (see Section 5)

## 5. Known Issues with Current Experiments

### Bugs Identified (from harsh reviewer audit)

1. **N3_info_min broken**: uncertainty threshold never triggered; algorithm was
   pure CTS in disguise. **Fixed in `final_algorithms.py`**.

2. **Pseudo-replication**: n=108 paired trials include 3x replication across
   models for CTS baseline (CTS uses no LLM, so same regret across models).
   True effective n=36. p-values partially inflated.

3. **Cache cross-contamination**: N1/N2/N4 and CHAMP_M2 share cache for cluster
   queries. Regret numbers are valid (same cached response used by all), but
   per-algorithm cost reporting shows 0 LLM calls (misleading).

4. **TS-LLM model comparison**: TS-LLM produces byte-identical results across 3
   models because all LLMs converge on the same top-m given identical mu_hat.
   This is actually expected (prompts are identical, LLMs agree on easy cases)
   but undermines the "cross-model robustness" narrative for TS-LLM.

### Missing Validation (severe)

1. **No RandomCluster ablation**: critical — distinguishes "LLM helps" from
   "correlated sampling helps." **Built in `final_algorithms.py`; must be run.**

2. **Only 3 seeds**: below publishable standard (10+). Confidence intervals wide.

3. **No real-world dataset**: all synthetic Bernoulli. Need MIND or similar.

4. **No hyperparameter sensitivity analysis**: rho, T_warmup, kernel_scale
   chosen by intuition.

5. **T=1500 too short**: regret curves still growing — we measure exploration
   phase only, not convergence phase.

6. **Limited config diversity**: only 2 gap types, 2 values of d, 1 value of m.
   Publishable paper needs d ∈ {20, 30, 50, 100}, m ∈ {3, 5, 8}.

## 6. What The Harsh-Reviewer Audit Says

**Are current Tier 3 results publishable as-is?** No.

**After bug fixes + ablation + 10+ seeds + real-data + hyperparameter sensitivity?**
Likely yes, as a paper with the following viable claims:

1. **Correlated Thompson Sampling with LLM-derived structure beats independent
   CTS** (provided RandomCluster ablation confirms LLM is necessary).
2. **Full kernel covariance (N1) outperforms block-diagonal cluster correlation
   (M2)**.
3. **Credibility-gated interpolation (N4) provides worst-case graceful
   fallback** while matching or exceeding M2 in best case.
4. **Endogenous oracle quality** is a real, measurable phenomenon that existing
   LLM-bandit papers do not model.

## 7. Plan for Full-Scale Experiment (Tier 4)

### Configuration
- **d ∈ {20, 30, 50}**, **m ∈ {3, 5, 8}** → 9 (d, m) combos
- **4 configs per (d, m)**: 2 uniform + 2 hard gap, varied δ
- **Total configs**: 36
- **Seeds**: 20 per config = **720 problem instances per algorithm**
- **T = 3000** (twice current, closer to convergence)
- **Models**: gpt-4.1-mini, gpt-5.4 (drop gpt-5-mini for cost; it's similar to gpt-4.1-mini in results)
- **Algorithms (10 total)**: CTS, CUCB, M2 CORR-CTS, N1 CORR-Full, N4 Robust-CORR,
  N5 CORR-Full-Robust, ABLATION_random_corr, PAPER_ts_llm, PAPER_cal_gated, PAPER_llm_cucb_at

### Statistical Protocol
- **Paired t-test + Holm-Bonferroni correction** across 9 algorithm comparisons
- **Sign test** as non-parametric backup
- **Wilcoxon signed-rank** as second non-parametric check
- **Effective sample size**: 720 paired trials per model (true n, no
  pseudo-replication)
- **α = 0.05/9 = 0.0056** for each comparison
- **Bootstrap 95% confidence intervals** on mean regret

### Ablations (mandatory)
1. **RandomCluster-CTS**: ablates LLM contribution
2. **Hyperparameter sweeps**: ρ ∈ {0.3, 0.5, 0.6, 0.8}, n_clusters ∈ {4, 6, 8, 12}
3. **Gap-difficulty curve**: δ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
4. **Prompt ablation**: 3 different cluster prompt formats

### Real-World Validation
- **MIND news dataset**: arms = articles, rewards = clicks from log
- **Run top-3 algorithms** (N4, N1, CORR-CTS) + CTS + one paper baseline
- **Shows synthetic results transfer to real recommendation data**

### Cost Estimate
- 720 instances × 2 models × 10 algos = 14,400 trials
- Avg 2 LLM calls per trial (N4/N5 use just 1 cluster call; M2 same)
- ~28,000 LLM calls total
- gpt-4.1-mini share: ~$6
- gpt-5.4 share: ~$60
- **Total: ~$70**
- Wall time with 16 workers: 3-5 hours

## 8. The Paper's Key Claims (Must Be Statistically Defensible)

### Primary Contribution
**"LLM-derived correlated posterior sampling (N4_robust_corr) beats plain CTS
by X% with p < 0.05 (Holm-Bonferroni corrected, n = 720, across d ∈ {20, 30, 50}
and m ∈ {3, 5, 8})."**

### Secondary Contributions
1. **Mathematical novelty**: full kernel covariance construction from LLM output
2. **Robustness guarantee**: credibility-gated worst-case bound (new theorem)
3. **Empirical finding**: endogenous oracle quality (first measurement in literature)
4. **Ablation result**: LLM clusters significantly better than random clusters (required)

### Things We Cannot Claim
- "Best possible LLM-bandit algorithm" — we have not swept ρ, n_clusters, etc.
  thoroughly enough.
- "Works for all LLMs" — we test 2-3 OpenAI models only.
- "Asymptotically optimal" — we have no regret upper bound proven; only empirical.

## 9. Publication Targets

- **Primary**: NeurIPS 2026 main track (combinatorial bandits + LLM)
- **Secondary**: ICML 2026 workshop (algorithms with predictions)
- **Theory companion paper**: COLT 2026 or ALT 2026 (regret bound for N4)

## 10. Honest Bottom Line

We started this project wanting to "beat CTS with LLMs in combinatorial
bandits." After many iterations, bugs, and reviewer-grade self-critique, we
have:

- **Two candidate algorithms** (N1, N4) with statistically significant wins on
  a limited experiment
- **One synthesis algorithm** (N5) to be tested
- **One mandatory ablation** (RandomCluster-CTS) to be tested
- **Several methodological concerns** that require a larger, cleaner experiment
  before publication

**The research is close to publication-ready but not yet there.** Full-scale
experiment (Tier 4) is the last major step.
