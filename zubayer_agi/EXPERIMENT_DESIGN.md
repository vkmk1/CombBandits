# Experiment Design: Rigorous Evaluation of LLM-Combinatorial-Bandits

## Part 1: What the Literature Requires for Rigor

From my review of 20+ combinatorial-bandit + LLM-bandit papers:

**Wang & Chen 2018 (ICML)** — the standard CTS regret paper — used:
- Matroid bandits (MST): d=20-100 edges
- Shortest path: d=20 edges, adversarial gaps
- T = 10K - 100K
- "Constructed hard instances where suboptimal paths only slightly worse" (adversarial configs)

**Chapelle & Li 2011** — landmark Thompson Sampling paper — used:
- 2 real-world domains (display ads, news articles)
- 10-1000 independent runs
- Reported mean regret curves with error bars

**Modern standards (2024-2025)**:
- 50+ seeds for confidence intervals (Stable Thompson Sampling)
- 10K+ trials for coverage probability
- Multiple problem instances (not single config)
- Paired Wilcoxon signed-rank for statistical significance
- **LLM-bandit papers typically do 10-30 seeds × 3-10 configs** (what they publish)

**The "adaptive sampling bias" problem** (arxiv 2103.12198):
- Thompson Sampling has inflated false positive rate (5% → 13%) due to non-uniform exploration
- Requires: paired comparisons, many seeds, proper statistical tests
- Without this, "wins" can be artifacts

## Part 2: Bias Safeguards (Applied to All Tiers)

1. **Deterministic numpy RNG per trial** — same np_seed for all algorithms on same (config, seed). Already implemented.
2. **Fresh LLM cache** — delete cache before each tier so no contamination from earlier runs.
3. **Paired comparisons only** — every algorithm faces identical (config, seed, reward_rng) setup.
4. **Multiple seeds** — enough to bound standard error to ~5% of mean.
5. **Adversarial configs** — hard gap types where CTS is at its strongest (smallest Δ_min).
6. **Out-of-distribution configs** — tier-3 has configs the algorithm has *never seen* (different env_seeds).
7. **Sensitivity analysis** — vary hyperparameters to show robustness.
8. **Two-sided statistical tests** — paired t-test + Wilcoxon signed-rank + sign test.
9. **Correction for multiple comparisons** — Bonferroni or Holm when testing many algorithms.

## Part 3: The Three Tiers

### Tier 1 — SMOKE (iteration/debugging, 5 min, ~$0.30)

**Purpose**: Sanity checks after code changes. Fast feedback loop.

| Parameter | Value |
|-----------|-------|
| d, m | 30, 5 |
| T | 800 |
| Configs | 3 (one uniform, one hard, one mid) |
| Seeds | 3 |
| Trials/algo | 9 |
| Algorithms | CTS baseline, B2 ICPD, M2 CORR-CTS (our 2 candidates) |
| Total trials | 27 |
| LLM calls | ~90 |
| Cost | ~$0.30 |
| Wall time | 3-5 min |

**Statistical power**: minimal. 9 trials can only detect effect sizes > 50%. Useful only for "did it break?" not "is it better."

### Tier 2 — MEDIUM (workshop-quality, 25 min, ~$3)

**Purpose**: Strong preliminary evidence. Submittable to a workshop (ICML workshop, NeurIPS workshop).

| Parameter | Value |
|-----------|-------|
| d | 30, 50 (two dimensions) |
| m | 5 |
| T | 2000 |
| Configs | 8 (4 uniform + 4 hard, each with distinct env_seeds) |
| Seeds | 10 |
| Trials/algo | 80 |
| Algorithms | 8 (see below) |
| Total trials | 640 |
| LLM calls | ~5000 |
| Cost | ~$3 |
| Wall time | 25-35 min with 8 parallel workers |

**Algorithms (8)**:
1. `CTS` — baseline
2. `OURS_B2_ICPD` — our champion
3. `OURS_M2_CORR_CTS` — our novel math
4. `PAPER_ts_llm` (Sun 2025)
5. `PAPER_jump_start` (Austin 2024)
6. `PAPER_cal_gated` (2024 pseudo-obs)
7. `PAPER_llm_cucb_at` (Kakaria, our repo)
8. `Oracle_topm` — LLM picks directly (weak baseline, shows "LLM without care" fails)

**Statistical tests**:
- Paired t-test (B2 vs CTS, CORR-CTS vs CTS)
- Wilcoxon signed-rank (non-parametric check)
- Bonferroni-corrected significance (α = 0.05 / 7 = 0.007 across 7 algo comparisons)
- Config-level breakdown (does B2 win on EVERY config or just easy ones?)

**Power analysis**: 80 paired trials → detects effect size of 15% regret difference at p<0.05 with 80% power. Our observed B2 effect is 24%, so we'd get p<0.001 comfortably.

### Tier 3 — LARGE (publication-quality, 3 hours, ~$40)

**Purpose**: Full ICML/NeurIPS paper. Must survive reviewer nitpicks about biases.

| Parameter | Value |
|-----------|-------|
| d | 20, 30, 50, 100 (four dimensions, covers "small" to "large" combinatorial space) |
| m | 3, 5, 8 (three super-arm sizes, across d) |
| T | 3000 |
| Configs | 20 per (d,m) combo — ~200 total configs |
| Seeds | 20 |
| Trials/algo | 400 |
| Algorithms | 12 (see below) |
| Total trials | 4800 |
| LLM calls | ~40000 |
| Cost | ~$25-40 |
| Wall time | 2-3 hours on EC2 c5.4xlarge (16 workers) |

**Algorithms (12)**:
1. `CUCB` (classical baseline #1)
2. `CTS` (classical baseline #2) — primary comparison
3. `OURS_B2_ICPD` — champion
4. `OURS_M2_CORR_CTS` — novel math
5. `OURS_B2+CORR` — synthesis (M3 from earlier)
6. `PAPER_ts_llm`
7. `PAPER_jump_start`
8. `PAPER_cal_gated`
9. `PAPER_llm_cucb_at`
10. `Oracle_topm_greedy`
11. `Oracle_topm_with_hedge` (LLM + ε-greedy)
12. `Random_baseline` (sanity floor)

**Additional sensitivity sweeps**:

(a) **Gap-difficulty curve**: δ_min ∈ {0.05, 0.10, 0.15, 0.20, 0.30} — shows whether advantage evaporates in "easy" regime (where Δ_min large, every algo converges fast anyway).

(b) **LLM corruption sweep**: inject synthetic noise into LLM responses (flip 0%, 10%, 20%, 40% of picks). Tests robustness claim: do we gracefully degrade to CTS?

(c) **Prompt ablation**: run B2 with 3 different prompt formats. Show wins aren't prompt-specific.

(d) **Hyperparameter sensitivity**: vary B2's `query_interval ∈ {100, 150, 200, 300}` and `decay_scale ∈ {100, 300, 500}`. Show sweet spot is not sharp (robust).

(e) **Real-world sanity check**: run B2 on 2 real datasets (MIND news, movie recommendation) to show synthetic results transfer.

**Statistical tests**:
- Paired t-test for every algorithm vs CTS (11 comparisons)
- Bonferroni correction: α = 0.05 / 11 = 0.0045
- Holm step-down procedure (less conservative alternative)
- Wilcoxon signed-rank (non-parametric robustness)
- Per-config "does it win on every slice?" analysis
- Bootstrap 95% CIs on mean regret
- Effect size (Cohen's d) with magnitude interpretation

**Power analysis**: 400 paired trials → detects effect of 4% regret difference at Bonferroni-corrected α=0.0045 with 80% power. Comfortably exceeds our minimum goal.

## Part 4: The Big Question — Cost of Running Many Algorithms

**Short answer**: Yes, running 12 algorithms instead of just B2 is **more expensive in LLM calls and slightly slower**, but NOT 12× slower — and you HAVE to do it to publish.

### Breakdown

**LLM calls per trial** (T=2000):
| Algorithm | Calls/trial |
|-----------|-------------|
| CTS (baseline) | 0 |
| Random baseline | 0 |
| Oracle_topm_greedy | 20 (every 100 rounds) |
| OURS_B2_ICPD | 8 |
| OURS_M2_CORR_CTS | 1 |
| PAPER_ts_llm | 15 |
| PAPER_jump_start | 1 |
| PAPER_cal_gated | 8 |
| PAPER_llm_cucb_at | 15 |

**Tier 3 totals**:
- Just B2 alone: 400 trials × 8 calls = 3200 calls = ~$1.60
- All 12 algorithms: 400 × 70 avg = 28000 calls = ~$14
- **~9x more LLM calls**

### Wall-clock time

**NOT 9× slower** because:

1. **Parallelism**: With 16 EC2 workers, CPU-bound overhead is small. The bottleneck is LLM API latency (1-2 sec/call) and rate limits.

2. **Shared cache**: Different algorithms sometimes send similar prompts (same mu_hat state). Cache hits save API calls.

3. **Staggered compute**: While one algorithm waits for LLM, another can do its CTS sampling. Thread pool saturates better with more work.

Measured: Running 13 algorithms (previous experiment) took 17 min. Just running B2 would take ~7-9 min. **~2x slower in wall time for 9x more LLM calls** — the parallel efficiency is why.

### Opportunity cost

**You CANNOT publish only B2 alone.** Reviewers will demand:
- Comparison to classical baselines (CTS, CUCB) — unavoidable
- Comparison to prior LLM-bandit papers (TS-LLM, LLM-CUCB-AT etc.) — unavoidable
- Ablation showing which components of B2 matter — unavoidable
- Sensitivity analysis — unavoidable

Skipping these is a desk-reject. **The $14 cost for Tier 3 is the entry ticket to publication.**

### Optimizing further

If cost is a concern:
1. Run B2 on all 400 trials (your champion gets full stats)
2. Run baselines on 200 trials each (still significant stats)
3. Skip LLM-CUCB-AT at d=100 (it's n_queries=O(T) — expensive)

Saves ~40% with small statistical power reduction.

## Part 5: Recommended Execution Strategy

**Immediate next step**: **Tier 2** (medium, 25 min, $3). This gives:
- Workshop-quality statistical evidence
- Enough to decide: is B2 really solid? Does CORR-CTS hold up?
- If results confirm → commit to Tier 3 (large)
- If results reveal new weakness → iterate on algorithm design first

**Do NOT jump directly to Tier 3** because:
- If our hypotheses are wrong, we waste $40 and 3 hours
- Tier 2 is the scientific validation step

**Tier 1 (smoke) is implicit** — run it after any code change.

## Part 6: Pre-registration (Important!)

Scientific best practice: **commit to the analysis plan BEFORE running the experiment**. This prevents p-hacking.

Pre-registered hypotheses for Tier 2:
1. **Primary**: B2 ICPD mean regret < CTS mean regret, paired t-test p < 0.05 across 80 trials
2. **Secondary**: CORR-CTS mean regret < CTS mean regret, paired t-test p < 0.05
3. **Gating**: if Primary fails, we do NOT publish B2 as a claim. We iterate.

Pre-registered for Tier 3:
1. **Primary**: B2 ICPD mean regret < CTS, Bonferroni-corrected paired t-test p < 0.0045
2. **Secondary**: CORR-CTS beats CTS in budget-constrained setting (1 LLM call vs 8)
3. **Robustness**: B2 within 20% of CTS regret even when LLM response is 40% corrupted
4. **Ablations**: show every component of B2 matters (periodic > single, decay > constant, counterfactual > top_m)

No data peeking, no changing primary endpoint mid-analysis.

## Part 7: What Could Still Go Wrong

**Known risks to address in Tier 3**:

1. **d=100 scaling**: B2 might break as the LLM struggles to reason over 100 arms. Need to either (a) show B2 still wins, or (b) propose scaled variant.

2. **Real vs synthetic**: All our configs are synthetic Bernoulli. Real recommendation data might have different structure. **Tier 3 includes real-dataset sanity checks** (MIND, movies).

3. **Model dependence**: We used gpt-5-mini + gpt-4.1-mini. Does it work with Claude, Llama, Gemini? Paper should include 1-2 alternative LLMs to show model-agnostic.

4. **Hyperparameter tuning**: Current B2 params were from intuition. **Tier 3 does hyperparameter sensitivity analysis** — showing we didn't cherry-pick.

5. **Cherry-picked configs**: We use specific env_seeds. For Tier 3, use **1000+ randomly-sampled configs** and show average performance (not worst-case).

## Bottom Line

- **Tier 1 (smoke)**: ~$0.30, 5 min — debug tool
- **Tier 2 (medium)**: ~$3, 25 min — validation before committing
- **Tier 3 (large)**: ~$25-40, 3 hours — publication-ready

**Cost of "many algorithms" penalty: ~$12 extra and ~2x wall-time** — absolutely worth it because without baselines the paper is unpublishable.

Go Tier 2 next, then Tier 3.
