# Tier 4: Publication-Grade Full-Scale Experiment

**Objective**: produce publishable evidence that N4_robust_corr (or N5 synthesis)
beats plain CTS with real LLMs across a wide problem space, with all reviewer
objections preemptively addressed.

## Part 1 — The Core Experiment

### 1.1 Problem Space (d × m grid)

| d | m | # configs per (d,m) | Rationale |
|---|---|---------------------|-----------|
| 20 | 3 | 4 | Small; classical regime |
| 20 | 5 | 4 | Sanity check |
| 30 | 3 | 4 | Mid-range |
| **30** | **5** | **4** | **Baseline (our prior experiments)** |
| 30 | 8 | 4 | Larger super-arm |
| 50 | 5 | 4 | Matches Sun 2025 experiments |
| 50 | 8 | 4 | More challenging |
| 100 | 5 | 4 | LLM struggles here — stress test |
| 100 | 10 | 4 | Large combinatorial space |

**9 (d, m) combos × 4 configs each = 36 configs total**

### 1.2 Config Generation (per (d, m))

Each config defined by (gap_type, δ_min, env_seed):
- **uniform easy**: δ = 0.20
- **uniform hard**: δ = 0.10
- **hard gap easy**: δ = 0.20 (all suboptimal arms near 0.5)
- **hard gap hard**: δ = 0.08 (all suboptimal very close to optimal)

4 configs × 9 (d,m) = 36 configs.

### 1.3 Seeds

- **20 seeds per config** → 720 (config, seed) pairs per (algo, model)
- Justification: Wang & Chen 2018 use 100; modern papers 30-50. 20 is lower bound
  of publishable; paired-comparison gives us more effective power.

### 1.4 Horizon

- **T = 3000** (was 1500 in Tier 3 — this is 2× longer, closer to convergence)
- Regret curves will be saved every 50 rounds
- Allows reporting both early regret (t ≤ 500) and asymptotic regret (t ≥ 2000)

### 1.5 LLM Models (2 providers for robustness)

| Model | Provider | Input $/M | Output $/M | Rationale |
|-------|----------|-----------|-----------|-----------|
| **gpt-4.1-mini** | OpenAI | $0.40 | $1.60 | Cheapest, has logprobs |
| **gpt-5.4** | OpenAI | $2.50 | $15.00 | Flagship, best accuracy |
| **claude-haiku-4.5** | AWS Bedrock | $0.80 | $4.00 | Different provider — robustness |

Drop gpt-5-mini (too similar to gpt-4.1-mini in our earlier data, saves cost).
Adding Claude proves results aren't OpenAI-specific — addresses "model dependence"
reviewer objection.

### 1.6 Algorithm Suite (10 algorithms)

**Baselines (3)**:
- `CTS` — pure Thompson sampling (Wang & Chen 2018)
- `CUCB` — UCB-based combinatorial bandit (Chen et al. 2013)
- `RandomCorr_CTS` — **CRITICAL ABLATION**: correlated sampling with random clusters (no LLM)

**Our contributions (4)**:
- `M2_corr_cts` — original CORR-CTS (block-diagonal)
- `N1_corr_full` — full kernel covariance
- `N4_robust_corr` — credibility-gated
- `N5_corr_full_robust` — synthesis of N1+N4

**Paper baselines (3)**:
- `TS_LLM` (Sun et al. 2025)
- `LLM_Cal_Gated` (2024 pseudo-observations with calibration gate)
- `LLM_CUCB_AT` (Kakaria, this repo's paper, simplified variant with query_cooldown)

## Part 2 — Ablation Studies (Required for Publication)

### 2.1 Ablation 1: LLM Necessity (built into main experiment)
**`RandomCorr_CTS` vs `N4_robust_corr`**: if LLM-derived clusters don't beat
random clusters, the LLM is decorative and paper is dead.

**Success criterion**: N4 beats RandomCorr with p < 0.01, effect size > 10%.

### 2.2 Ablation 2: Hyperparameter Sensitivity (separate mini-run)

Run **N4_robust_corr** across:
- `ρ ∈ {0.3, 0.5, 0.6, 0.7, 0.8}` — 5 values
- `n_clusters ∈ {4, 6, 8, 12}` — 4 values
- `check_interval ∈ {50, 100, 200}` — 3 values

On 4 representative configs × 10 seeds = 40 trials per (hyperparameter) combo.

**Total**: 5 × 4 × 3 = 60 hyperparameter combinations × 40 = 2,400 trials.
Only on gpt-4.1-mini (cheapest) to save cost.

**Success criterion**: N4's advantage is robust within ±20% across hyperparameter
sweep — no single sharp optimum (which would suggest p-hacking).

### 2.3 Ablation 3: Gap-Difficulty Curve

Sweep δ_min ∈ {0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30} on d=30, m=5.
7 δ values × 2 gap types × 10 seeds = 140 trials per algorithm.

**Success criterion**: N4 beats CTS across all δ, with advantage largest at
medium δ (hardest problems); may converge at δ=0.30 (easy) and δ=0.03 (impossible).

### 2.4 Ablation 4: LLM Corruption Injection

Inject synthetic noise into LLM responses:
- 0% corruption (baseline)
- 20% corruption (flip 20% of LLM-suggested arms to random)
- 50% corruption
- 80% corruption (near-adversarial)

**Success criterion**: N4 degrades gracefully — regret ≤ 1.5× CTS at 80% corruption
(proves the robustness guarantee). This is the theoretical novelty demonstrated empirically.

### 2.5 Ablation 5: Prompt Format

For the cluster query, test 3 prompts:
- "Group these arms into clusters by similarity" (current)
- "Which arms will behave similarly?"
- "If one arm is good, which others are likely good?"

Shows prompt isn't cherry-picked. Only on 4 representative configs × 10 seeds.

### 2.6 Ablation 6 (Optional): Varying Warmup Length

T_warmup ∈ {0, 15, 30, 60, 100}. Shows the 30-round choice isn't arbitrary.

## Part 3 — Real-World Dataset Validation

### 3.1 MIND News Recommendation (Microsoft)
- Public dataset: news articles with real click-through rates
- 30-100 articles (d) from a category; pick 5 for homepage (m)
- Rewards: Bernoulli with observed CTR
- Run top-4 algorithms (CTS, N4, N1, TS-LLM) × 10 seeds × 5 subsamples = 200 trials
- gpt-4.1-mini only to save cost

**Shows synthetic results transfer to real recommendation**

### 3.2 Optional: MovieLens
Smaller-scale second real dataset. Only if time permits.

## Part 4 — Statistical Analysis Protocol (Pre-Registered)

### 4.1 Primary Hypothesis
**H1**: `N4_robust_corr` mean regret < `CTS` mean regret across all 720 paired trials,
after Bonferroni correction for 9 algorithm comparisons.

**Test**: paired t-test + sign test + Wilcoxon signed-rank (all three must agree).

**Adjusted α**: 0.05 / 9 = **0.0056**.

### 4.2 Secondary Hypotheses
- **H2**: `N4_robust_corr` beats `RandomCorr_CTS` at p < 0.01 (proves LLM contribution)
- **H3**: `N1_corr_full` beats `M2_corr_cts` (full kernel > block-diagonal)
- **H4**: `N4_robust_corr` regret ≤ 1.5 × CTS regret at 80% LLM corruption (robustness guarantee)
- **H5**: Our algorithms beat all paper baselines at p < 0.05

### 4.3 Effect Size Requirements
- Cohen's d > 0.3 (medium) for primary claim
- Regret reduction > 10% in practical terms

### 4.4 Reporting
- Bootstrap 95% CIs on mean regret
- Regret-curve plots with shaded stderr bands (matplotlib style like NeurIPS)
- Per-(d,m) breakdown table
- Per-config breakdown table
- Win/loss matrix across all 10 algorithms (10×10 grid)

## Part 5 — AWS Infrastructure

### 5.1 Compute

**Instance**: EC2 c5.4xlarge (16 vCPU, 32GB RAM)
- Cost: $0.68/hour on-demand, $0.20/hr spot
- Our runs are API-latency bound, not CPU-bound → CPU-only is fine
- 16 parallel workers saturate at our query rate

### 5.2 Storage

- **S3 bucket**: `combbandits-results-099841456154` (already exists)
- Results structure:
```
s3://combbandits-results-099841456154/tier4_YYYYMMDD/
  main_experiment/
    raw_trials.jsonl.gz        — one line per trial
    llm_calls.jsonl.gz         — every LLM API call
    algo_states.jsonl.gz       — per-round state snapshots
    report.txt                 — text summary
    report.json                — machine-readable stats
  ablation_hyperparams/
  ablation_gap_difficulty/
  ablation_llm_corruption/
  ablation_prompt_format/
  mind_dataset/
```

### 5.3 Parallelism Strategy

**Main experiment is embarrassingly parallel**: each (algo, config, seed, model) trial
is independent.

With ThreadPoolExecutor(max_workers=16):
- Each thread handles one trial
- LLM calls go through shared SQLite cache (with lock)
- Results streamed to JSONL as they complete

Expected throughput: ~15-20 trials/min. For 720 × 10 × 3 = 21,600 main trials →
**18-24 hours wall time** on one instance.

To cut to ~6 hours: run 4 parallel instances, each handles 1/4 of the trials.

### 5.4 AWS Credentials + Setup

Uses existing EC2 infrastructure from `cluster/launch_production.sh`:
- Role: `princetoncourses-ec2`
- SSH key: `~/.ssh/combbandits-key.pem`
- Security group: `combbandits-sg` (exists)
- AMI: Ubuntu 22.04

Launcher script will:
1. Spawn c5.4xlarge
2. Upload our `zubayer_agi/` code
3. Install deps (openai, pandas, numpy)
4. Store OPENAI_API_KEY via SSM parameter
5. Run experiment with nohup
6. Periodic sync to S3
7. Self-terminate on completion
8. SNS/Slack notification when done

### 5.5 Live Monitoring

Reuse existing dashboard approach:
- Results synced to S3 every 5 min
- Amplify-hosted dashboard polls S3
- URL: https://bandits.easyprincetoncourses.com (already set up)

## Part 6 — Cost Breakdown

### 6.1 Main Experiment
- 10 algos × 36 configs × 20 seeds × 3 models = 21,600 trials
- Avg calls per trial: ~3 (CTS=0, most others 1-8)
- Total LLM calls: ~65,000

Cost by model (assumes each model gets 1/3 = 7200 trials):
- **gpt-4.1-mini** (7200 × 3 calls × 500 tokens = 11M tokens): $4
- **gpt-5.4** (7200 × 3 calls × 500 tokens): $35
- **claude-haiku-4.5** (7200 × 3 × 600 tokens): $14

**Main experiment LLM cost: ~$53**

### 6.2 Ablations
- Hyperparam sensitivity: 2,400 trials × 1 call × 500 tokens × gpt-4.1-mini: $1
- Gap difficulty: 140 × 7 algos × 3 calls × 3 models = $8
- LLM corruption: 140 × 4 algos × 3 calls × 3 models = $5
- Prompt format: 400 × 1 call × 500 tokens = $1
- Real-data (MIND): 200 × 3 calls = $1

**Ablations LLM cost: ~$16**

### 6.3 AWS Compute
- Main experiment: 20 hrs × $0.68 = $14
- Ablations: 4 hrs × $0.68 = $3
- **AWS compute: ~$17**

### 6.4 Total Estimate
- LLM APIs: ~$70
- AWS compute: ~$17
- Data transfer / S3: ~$2
- **TOTAL: ~$90**

With 20% buffer: **$110**.

## Part 7 — Timeline

**Day 1 (setup)**:
- Fix any remaining bugs
- Dry-run on 10 trials locally to verify pipeline
- Launch EC2 instance
- Verify end-to-end

**Day 2-3 (main experiment)**:
- 21,600 main trials run on EC2
- ~18-24 hours wall time
- Automatic S3 sync + dashboard

**Day 3 (ablations)**:
- Run all 6 ablations in parallel (different instances or queued sequentially)
- 4-8 hours

**Day 4 (analysis)**:
- Download results from S3
- Run statistical analysis
- Generate paper-ready tables + figures
- Write up findings

**Total: 4 days**

## Part 8 — Expected Outcomes

### Success Scenario (we get a paper)
- N4 beats CTS by 12-20% across all (d, m) combos, p < 0.0056
- N4 beats RandomCorr by 8-15% (LLM matters)
- N4 beats all 3 paper baselines with p < 0.01
- Regret degrades gracefully under 20-50% LLM corruption
- Robust to hyperparameter choices
- Replicates on MIND real data

**This is a NeurIPS main track paper.**

### Partial Success (workshop paper)
- N4 beats CTS on some (d,m) combos but not all
- LLM contribution smaller than expected (RandomCorr close to N4)
- Mixed results on real data

**Workshop paper at ICML LLM+Bandits workshop.**

### Failure Scenario (we pivot)
- N4 does NOT beat CTS reliably across the space
- RandomCorr matches N4 → LLM decorative
- Paper baselines beat ours

**Pivot: write the experimental-methodology paper (how to correctly evaluate
LLM-bandits) instead of the algorithm paper. The information-bottleneck finding
and endogenous-quality finding are still publishable as observations.**

## Part 9 — What I Will Build Next

To launch Tier 4, I need:

1. **`tier4_main.py`**: the main experiment runner (36 configs × 20 seeds × 3 models × 10 algos)
2. **`tier4_ablations.py`**: the 6 ablation studies
3. **`tier4_mind_real.py`**: MIND dataset loader + runner
4. **`cluster/launch_tier4.sh`**: EC2 spawn + upload + run script
5. **`tier4_analysis.py`**: proper statistical analysis with Holm-Bonferroni, bootstrap CIs, effect sizes
6. **`tier4_figures.py`**: publication-quality matplotlib figures

All should be built locally, smoke-tested on 20 trials, then launched on EC2.

**No execution yet. Awaiting your approval.**

## Part 10 — Potential Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| EC2 instance dies mid-run | Medium | Lose 12 hours | Incremental S3 sync + resume-from-checkpoint script |
| LLM API rate limit | Medium | Slow | Built-in retries; stagger requests |
| Bedrock Claude access blocked | Low | Drop one model | Use only OpenAI models |
| Results disappointing | Medium | No paper | Pivot to methodology paper |
| Cache corruption | Low | Bad data | Fresh cache per run; checksums |
| Bug in one algorithm | Medium | Bad one algo | Parallel smoke-test before launch |

## Bottom Line

**Tier 4 is ~$110, ~4 days, ~21,600 trials + 6 ablations + 1 real-data validation.**

If we launch and it succeeds, we have a NeurIPS 2026 main track submission.
If it fails, we know definitively within 4 days and can pivot.

**Awaiting approval before building + launching.**
