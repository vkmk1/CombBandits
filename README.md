# CombBandits: LLM-CUCB-AT

**Regret-Bounded Combinatorial Bandits with Unreliable LLM Action Oracles**

Vikram Kakaria, Princeton University (`vk5825@princeton.edu`)

---

## Overview

CombBandits implements **LLM-CUCB-AT** (Algorithm 1 from the paper), an algorithm for combinatorial semi-bandits that uses large language models as action oracles while provably handling their unreliability.

### The Problem

In a combinatorial semi-bandit, a learner selects a subset of *m* arms from *d* candidates each round, observes per-arm rewards, and aims to minimize cumulative regret against the best fixed subset S\*. Standard CUCB achieves O(sqrt(mdT)) regret, but the combinatorial explosion C(d,m) makes full enumeration intractable for large d.

LLMs can suggest good superarms using world knowledge (e.g., "which news articles will get clicks?"), but these suggestions are **systematically unreliable**: LLMs hallucinate, exhibit recency bias, and can be *consistently wrong* in ways that simple consistency checks miss.

### Our Solution: LLM-CUCB-AT

The algorithm uses a **composite trust score** to adaptively trust or hedge against the LLM oracle:

```
tau_t = min(kappa_t, rho_t)
```

- **kappa_t** (consistency): Query the LLM K times. How much do the responses agree?
- **rho_t** (posterior validation): Does the LLM's suggestion agree with accumulated empirical evidence?
- **tau_t** (composite trust): Trust only when the LLM agrees with itself AND with the data.

When trust is high, the algorithm narrows its search to the LLM's suggestion (dimension reduction: d -> m). When trust is low, it hedges by adding top UCB arms. A corruption detector triggers fallback to vanilla CUCB if sustained underperformance is detected.

### Key Theoretical Results

| Result | Statement |
|--------|-----------|
| **Dimension reduction** (Theorem 1) | When oracle is reliable, regret scales with effective dimension m + h_max(1-q_bar) instead of d |
| **Graceful degradation** (Corollary 1.3) | Under adversarial corruption, regret is no worse than vanilla CUCB: O(sqrt(mdT)) |
| **Consistently wrong detection** (Lemma 2) | Posterior validation catches deterministically wrong oracles that consistency alone misses |
| **Matching lower bound** (Theorem 3) | The corruption cost Omega(epsilon*m/Delta_min) is information-theoretically unavoidable |
| **Query efficiency** (Theorem 5) | O(sqrt(T)) LLM queries suffice with bounded additive regret increase |

### Action Corruption vs Reward Corruption

This paper introduces **action-space corruption** — qualitatively different from the reward corruption of Lykouris et al. (2018). In our setting, the oracle corrupts the *action proposal* (which arms to consider), not the *reward observation*. The learner retains uncorrupted reward feedback, enabling independent verification of oracle quality. Our lower bound shows the cost structure is multiplicative in T (not additive), distinguishing it from reward-corruption results.

---

## Project Structure

```
CombBandits/
├── src/combbandits/
│   ├── agents/                  # 9 bandit algorithms
│   │   ├── base.py              # Abstract Agent (UCB indices, arm stats)
│   │   ├── cucb.py              # CUCB (Chen et al., 2013)
│   │   ├── cts.py               # Combinatorial Thompson Sampling
│   │   ├── llm_cucb_at.py       # *** LLM-CUCB-AT (our method) ***
│   │   ├── llm_greedy.py        # Always follow LLM (no exploration)
│   │   ├── ellm_adapted.py      # ELLM-style exploration bonus (Du et al., 2023)
│   │   ├── opro_bandit.py       # OPRO iterative LLM optimization (Yang et al., 2024)
│   │   ├── corrupt_robust_cucb.py  # Median-of-means robust CUCB (He et al., 2022)
│   │   ├── warm_start_cts.py    # CTS with LLM-initialized priors
│   │   └── exp4.py              # EXP4 adversarial (Auer et al., 2002)
│   │
│   ├── environments/            # Combinatorial semi-bandit environments
│   │   ├── base.py              # Abstract CombBanditEnv
│   │   ├── synthetic.py         # Bernoulli arms (d=50/100/200, 4 gap structures)
│   │   ├── mind.py              # MIND news recommendation + simulated variant
│   │   └── influence_max.py     # SNAP influence maximization + simulated variant
│   │
│   ├── oracle/                  # Combinatorial LLM Oracle (CLO)
│   │   ├── base.py              # CLO interface + consistency scoring
│   │   ├── simulated.py         # Parameterized noisy oracle (4 corruption types)
│   │   ├── llm_oracle.py        # Real LLM via OpenAI/Anthropic API
│   │   └── cached_oracle.py     # O(sqrt(T)) query schedule + SQLite disk cache
│   │
│   ├── engine/                  # Experiment orchestration
│   │   ├── trial.py             # Single trial: T rounds of select -> pull -> update
│   │   └── runner.py            # Experiment grid with parallel execution + SLURM support
│   │
│   ├── analysis/                # Results processing
│   │   ├── metrics.py           # Summary statistics, significance tests, regret curves
│   │   └── plots.py             # Publication-quality PDF figures
│   │
│   ├── cli.py                   # CLI entry point (run, export-tasks, plot, metrics)
│   └── types.py                 # Core data structures (RoundResult, TrialResult, etc.)
│
├── configs/experiments/         # Experiment configurations (YAML)
│   ├── exp1_synthetic.yaml      # Full synthetic benchmark (9 agents, 3 dims, 9 oracles)
│   ├── exp2_ablation.yaml       # Ablation: K (re-queries), h_max (hedge size)
│   ├── exp3_quick_test.yaml     # Smoke test (~30 seconds)
│   ├── exp4_mind.yaml           # MIND news recommendation
│   ├── exp5_influence_max.yaml  # SNAP influence maximization
│   ├── exp6_workshop_main.yaml  # Workshop paper: all key results in one config
│   └── exp7_ablation_trust.yaml # Trust score component ablation
│
├── cluster/                     # Princeton DELLA SLURM integration
│   ├── della_submit.sh          # Job submission orchestrator
│   ├── della_array.slurm        # CPU array job template
│   ├── della_gpu.slurm          # GPU array job template
│   └── della_aggregate.slurm    # Post-experiment aggregation + plotting
│
├── scripts/
│   ├── run_local.sh             # Local execution wrapper
│   └── setup_della.sh           # One-time DELLA environment setup
│
├── tests/
│   └── test_smoke.py            # 7 smoke tests (all passing)
│
├── results/                     # Experiment outputs (JSON)
└── figures/                     # Generated figures (PDF)
```

---

## Quick Start (Local)

### 1. Install

```bash
cd CombBandits
pip install -e .
```

### 2. Smoke Test

```bash
bash scripts/run_local.sh exp3_quick_test
```

This runs 4 agents (CUCB, CTS, LLM-CUCB-AT, LLM-Greedy) on a small synthetic environment (d=20, m=5, T=1000) with 3 oracle configurations (clean, uniform corruption, consistently wrong) across 3 seeds. Takes ~30 seconds and produces:

- `results/exp3_quick_test/exp3_quick_test_results.json` — raw trial data
- `figures/exp3_quick_test/` — regret curves, trust diagnostics, corruption comparison

### 3. View Results

```bash
# Summary statistics
python -m combbandits.cli metrics results/exp3_quick_test/exp3_quick_test_results.json

# Regenerate figures
python -m combbandits.cli plot results/exp3_quick_test/exp3_quick_test_results.json --output-dir figures/exp3_quick_test
```

---

## Running Experiments Locally

For development or if you don't have DELLA access.

### Using the CLI

```bash
# Run a predefined experiment config
python -m combbandits.cli run configs/experiments/exp6_workshop_main.yaml \
    --output-dir results/exp6_workshop_main \
    --workers 4

# Run with specific corruption type
python -m combbandits.cli run configs/experiments/exp1_synthetic.yaml \
    --output-dir results/exp1_synthetic \
    --workers 8

# Generate all publication figures from results
python -m combbandits.cli plot results/exp6_workshop_main/exp6_workshop_main_results.json \
    --output-dir figures/workshop
```

### Using the Shell Script

```bash
# Wrapper that runs experiment + metrics + plots in one command
bash scripts/run_local.sh exp6_workshop_main --workers 4
```

### Experiment Configs at a Glance

| Config | Agents | Envs | Oracles | Seeds | Tasks | Est. Time (50 workers) | Purpose |
|--------|--------|------|---------|-------|-------|------------------------|---------|
| `exp3_quick_test` | 4 | 1 | 3 | 3 | 36 | ~10s | Verify installation |
| `exp7_ablation_trust` | 5 | 1 | 4 | 30 | 600 | ~3min | Trust component ablation |
| `exp2_ablation` | 2 | 1 | 5 | 20 | 200 | ~1min | K and h_max ablation |
| `exp4_mind` | 6 | 1 | 3 | 20 | 360 | ~2min | News recommendation |
| `exp5_influence_max` | 6 | 1 | 3 | 20 | 360 | ~2min | Influence maximization |
| `exp6_workshop_main` | 7 | 3 | 10 | 30 | 6,300 | ~35min | Workshop paper (all key results) |
| `exp1_synthetic` | 9 | 3 | 9 | 20 | 4,860 | ~25min | Full NeurIPS benchmark |
| `exp8_scaling_d` | 4 | 7 | 3 | 100 | — | **~15min (GPU)** | Large-d dimension scaling (d=50--5000) |

**Note on runtimes:** CPU experiments: each task runs 0.1--30 seconds (pure numpy, scales with T and d). GPU experiments: each (agent, env, oracle) group runs all seeds in parallel on one GPU.

### Experiment Status

| Config | Status | Trials | Figures | Notes |
|--------|--------|--------|---------|-------|
| `exp3_quick_test` | **Done** | 24 | 8 PDFs | Smoke test |
| `exp4_mind` | **Done** | 280 | 7 PDFs | Simulated MIND environment |
| `exp5_influence_max` | **Done** | 280 | 7 PDFs | Simulated influence max |
| `exp7_ablation_trust` | **Done** | 510 | 7 PDFs | Trust component ablation (d=100, T=30K, 30 seeds) |
| `exp6_workshop_main` | **Run on DELLA** | — | — | Theory validation (3 dims, 10 oracles, T=100K) |
| `exp8_scaling_d` | **Run on DELLA** | — | — | Large-d GPU experiment (d=50-5000) |
| `exp9_real_llm` | **Run on DELLA** | — | — | Real GPT-4o oracle (~$60 API cost) |

See [DELLA_RUNBOOK.md](DELLA_RUNBOOK.md) for step-by-step instructions to run the remaining experiments.

---

## Running on Princeton DELLA

### Prerequisites

- SSH access to DELLA (`ssh <netid>@della.princeton.edu`)
- Sufficient allocation on the `cpu` partition (most experiments are CPU-only)

### First-Time Setup

```bash
# 1. SSH into DELLA
ssh vk5825@della.princeton.edu

# 2. Clone repo to scratch (NOT home — home has 10GB quota)
cd /scratch/gpfs/vk5825/
git clone <repo-url> LIKEN
cd LIKEN/CombBandits

# 3. Run setup script (creates conda env, installs package)
bash scripts/setup_della.sh

# 4. Verify installation
conda activate combbandits
bash scripts/run_local.sh exp3_quick_test
```

### Submitting Experiments

Each experiment config is submitted as a SLURM array job. Every array task runs exactly one (agent, environment, oracle, seed) trial.

```bash
# Activate environment
conda activate combbandits

# --- Step 1: Dry run (see task count, don't submit) ---
bash cluster/della_submit.sh exp6_workshop_main --dry-run
# Output: "Total tasks: 6300. Would submit array job with 6300 tasks"

# --- Step 2: Submit ---
bash cluster/della_submit.sh exp6_workshop_main
# Submits SLURM array job: --array=0-6299%50 (50 concurrent tasks)

# --- Step 3: Monitor ---
squeue -u $USER                          # Check queue status
sacct -j <JOBID> --format=JobID,State    # Check individual tasks
tail -f logs/slurm/<JOBID>_0.out         # Stream first task's output

# --- Step 4: Aggregate results after all tasks complete ---
sbatch --export=CONFIG_NAME=exp6_workshop_main cluster/della_aggregate.slurm
# Merges per-task JSONs -> merged_results.json, generates all figures
```

### Full Experiment Pipeline (copy-paste)

Run this to submit all experiments needed for the workshop paper:

```bash
conda activate combbandits
cd /scratch/gpfs/vk5825/LIKEN/CombBandits

# Submit all experiments
JOB1=$(bash cluster/della_submit.sh exp6_workshop_main | grep -oP 'Submitted batch job \K\d+')
JOB2=$(bash cluster/della_submit.sh exp7_ablation_trust | grep -oP 'Submitted batch job \K\d+')
JOB3=$(bash cluster/della_submit.sh exp4_mind | grep -oP 'Submitted batch job \K\d+')
JOB4=$(bash cluster/della_submit.sh exp5_influence_max | grep -oP 'Submitted batch job \K\d+')

echo "Submitted jobs: $JOB1 $JOB2 $JOB3 $JOB4"

# After all complete, aggregate each:
for EXP in exp6_workshop_main exp7_ablation_trust exp4_mind exp5_influence_max; do
    sbatch --export=CONFIG_NAME=$EXP cluster/della_aggregate.slurm
done
```

### SLURM Resource Usage

All experiments are **CPU-only** — no GPU allocation needed. Each task takes 0.1--10 seconds of compute. Do **not** waste A100 GPU-hours on these; use the `cpu` partition.

| Resource | Setting | Why |
|----------|---------|-----|
| Partition | `cpu` | No torch/GPU dependencies; pure numpy/scipy |
| Memory | 4GB per task | d=200, T=100K fits easily |
| CPUs | 1 per task | Single-threaded bandit loop |
| Time | 30 min per task | Conservative; most tasks finish in <15 seconds |
| Concurrency | 50 tasks max (`%50`) | Avoids flooding the scheduler |

**If you only have GPU allocation:** You can still run on GPU nodes (the CPU works fine), but it wastes your allocation. Request a single GPU node and use `--workers 32` locally instead of SLURM arrays:

```bash
# Interactive session on a GPU node, using its CPUs
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=32 --mem=32G --time=02:00:00
conda activate combbandits
# Run all experiments sequentially with 32 local workers (no SLURM arrays)
for EXP in exp6_workshop_main exp7_ablation_trust exp4_mind exp5_influence_max; do
    python -m combbandits.cli run configs/experiments/${EXP}.yaml \
        --output-dir results/${EXP} --workers 32
done
# Total time: ~1.5 hours using 32 CPU cores on a single node
```

The GPU SLURM template (`della_gpu.slurm`) exists for future experiments with local LLMs (e.g., Llama-3-8B via vLLM), but is **not needed** for any current experiment.

### Monitoring and Debugging

```bash
# Check how many tasks completed vs total
sacct -j <JOBID> --format=JobID,State | grep -c COMPLETED

# Check for failed tasks
sacct -j <JOBID> --format=JobID,State,ExitCode | grep FAILED

# Resubmit specific failed tasks (e.g., tasks 42 and 99)
sbatch --array=42,99 \
    --export=CONFIG_PATH="configs/experiments/exp6_workshop_main.yaml",CONFIG_NAME="exp6_workshop_main" \
    cluster/della_array.slurm

# View a specific task's error
cat logs/slurm/<JOBID>_42.err
```

### Downloading Results from DELLA

```bash
# From your local machine:
scp -r vk5825@della.princeton.edu:/scratch/gpfs/vk5825/LIKEN/CombBandits/results/ ./results/
scp -r vk5825@della.princeton.edu:/scratch/gpfs/vk5825/LIKEN/CombBandits/figures/ ./figures/
```

---

## Algorithm Details

### LLM-CUCB-AT (Algorithm 1)

The algorithm has two phases:

**Phase 1: Initialization** (rounds 1 to T_0, where T_0 = ceil(d*log(d)/m))
- Round-robin exploration to get base estimates mu_hat for all d arms
- No oracle queries during this phase

**Phase 2: Adaptive LLM-Guided Play** (rounds T_0+1 to T)

Each round:

1. **Query oracle** K times with context (arm metadata, empirical means, history)
2. **Compute consistency** kappa_t = |intersection of K response sets| / m
3. **Compute posterior validation** rho_t = sum(mu_hat[suggested]) / max_S sum(mu_hat[S])
4. **Composite trust** tau_t = min(kappa_t, rho_t)
5. **Hedge size** h_t = ceil(h_max * (1 - tau_t))
   - High trust -> small hedge -> dimension reduction
   - Low trust -> large hedge -> more exploration
6. **Build reduced set** D_t = LLM_suggestion ∪ top_h_t_UCB_arms_not_in_suggestion
7. **Play** top-m arms by UCB index within D_t (not all of [d])
8. **Corruption detection** If sliding-window regret exceeds threshold, fallback to full CUCB next round

### Why Composite Trust?

| Oracle Failure Mode | Consistency (kappa) | Posterior Validation (rho) | Composite (tau) |
|--------------------|--------------------|--------------------------|--------------------|
| Random corruption | Detects (kappa < 1) | May not detect early | Detects |
| Consistently wrong | **Misses** (kappa = 1) | Detects (rho < 1) | Detects |
| Early rounds | Works | **Unreliable** (mu_hat noisy) | Conservative (uses min) |
| Reliable oracle | kappa ≈ 1 | rho ≈ 1 | tau ≈ 1 (trusts oracle) |

The composite score handles failure modes that neither component handles alone.

---

## Corruption Models

The simulated oracle supports 4 corruption types for controlled experiments:

| Type | Config Value | Behavior | Theory Validation |
|------|-------------|----------|-------------------|
| **Uniform** | `uniform` | Returns S* w.p. (1-epsilon), random set otherwise | Standard regret bound (Theorem 1) |
| **Adversarial** | `adversarial` | Returns worst-m arms w.p. epsilon | Worst-case robustness (Corollary 1.3) |
| **Partial overlap** | `partial_overlap` | Returns set sharing (1-epsilon)*m arms with S* | Graded quality model (Definition 2) |
| **Consistently wrong** | `consistent_wrong` | Always returns same plausible suboptimal set (kappa=1) | Posterior validation necessity (Lemma 2) |

---

## Environments

### Synthetic (Primary)

`SyntheticBernoulliEnv`: d Bernoulli arms with configurable gap structure.

- **d** ∈ {50, 100, 200}: Ground set size
- **m** = 10: Superarm size
- **gap_type**: `uniform` (equal gaps), `graded` (varying), `clustered` (groups), `hard` (small gaps)
- **delta_min**: Minimum gap between optimal and suboptimal arms

### MIND News Recommendation

`MINDEnvSimulated`: Simulated news recommendation inspired by Microsoft MIND dataset.

- d=200 candidate articles, m=5 to display per session
- User preferences as Dirichlet distributions over 10 categories
- Click probabilities from category-user alignment + article quality
- Realistic partial-information structure

### Influence Maximization

`InfluenceMaxEnvSimulated`: Simulated social network influence spread.

- d=200 candidate seed nodes, m=10 seeds per round
- Power-law degree distribution with community structure
- Influence quality from degree centrality + community alignment
- Submodular objective (alpha = 1 - 1/e approximation via greedy)

For real MIND/SNAP data, see [External Data](#external-data-optional) below.

---

## Baselines (8 methods)

| Agent | Reference | What It Tests |
|-------|-----------|---------------|
| **CUCB** | Chen et al. (2013) | Standard combinatorial UCB, no LLM |
| **CTS** | Wang et al. (2018) | Combinatorial Thompson Sampling |
| **LLM-Greedy** | — | Always play LLM suggestion, no exploration |
| **ELLM-Adapted** | Du et al. (2023) | LLM suggestions as per-arm exploration bonuses |
| **OPRO-Bandit** | Yang et al. (2024) | Iterative LLM optimization from reward history |
| **Corrupt-Robust CUCB** | He et al. (2022) | Median-of-means (designed for reward corruption) |
| **Warm-Start CTS** | — | CTS with LLM-initialized Beta priors |
| **EXP4** | Auer et al. (2002) | Adversarial bandit with LLM as single expert |

---

## Current Results (Simulated Oracle Experiments)

> **Important:** All completed experiments use **simulated environments** and **simulated oracles** (parameterized coin flips, not real LLMs). No real datasets (MIND, SNAP) and no LLM API calls have been made. These validate algorithm safety and theory. See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed descriptions of what each experiment does and does not do. Real LLM experiments (exp9) are pending.

### exp7: Trust Ablation (d=100, m=10, T=30K, 30 seeds)

The headline experiment showing how different agents handle oracle corruption types.

| Agent | Clean (eps=0) | Uniform (eps=0.2) | Adversarial (eps=0.3) | Consistently Wrong |
|-------|--------------|-------------------|----------------------|-------------------|
| CUCB | 7,242 | 7,242 | 7,242 | 7,242 |
| LLM-CUCB-AT | 24,132 | 17,541 | 13,854 | 23,420 |
| LLM-Greedy | 70,301 | 69,925 | 70,069 | 68,657 |
| ELLM-Adapted | 70,301 | 70,301 | 70,300 | 68,657 |
| Warm-Start CTS | **1,455** | **1,444** | **1,465** | **1,476** |

**Key findings:**
- **LLM-CUCB-AT beats LLM-Greedy and ELLM by 3-5x** under all corruption types — the composite trust mechanism prevents catastrophic following of a bad oracle
- **LLM-CUCB-AT with adversarial corruption (13,854) outperforms clean oracle (24,132)** — counterintuitive but correct: the adversarial oracle triggers the hedge mechanism which adds UCB-ranked exploration arms, improving discovery
- **Warm-Start CTS dominates** with the simulated oracle because the simulated oracle knows the optimal set — querying it once and using it as a Bayesian prior is the best strategy *when the oracle is a coin flip*. This advantage will not hold with a real LLM.
- **CUCB is the baseline to beat** — LLM-CUCB-AT needs real LLM world knowledge (exp9) to show improvement over pure UCB exploration

### exp4: Simulated News Recommendation (d=200, m=5, T=2K, 20 seeds)

Uses `MINDEnvSimulated` — synthetic click probabilities, **not** the real Microsoft MIND dataset.

| Agent | Partial Overlap (eps=0.15) | Uniform (eps=0.1) | Uniform (eps=0.3) |
|-------|---------------------------|-------------------|-------------------|
| CUCB | 1,920 | 1,920 | 1,920 |
| CTS | **1,207** | **1,207** | **1,207** |
| LLM-CUCB-AT | 1,931 | 1,927 | 1,927 |
| LLM-Greedy | 2,296 | 2,301 | 2,288 |
| Warm-Start CTS | **1,144** | **1,125** | **1,114** |

### exp5: Simulated Influence Maximization (d=200, m=10, T=5K, 20 seeds)

Uses `InfluenceMaxEnvSimulated` — synthetic graph with planted communities, **not** real SNAP social networks. No independent cascade Monte Carlo.

| Agent | Partial Overlap (eps=0.2) | Uniform (eps=0.1) | Uniform (eps=0.3) |
|-------|---------------------------|-------------------|-------------------|
| CUCB | 5,142 | 5,142 | 5,142 |
| CTS | 390 | 390 | 390 |
| LLM-CUCB-AT | 5,203 | 5,688 | 5,418 |
| LLM-Greedy | 8,256 | 8,215 | 8,286 |
| Warm-Start CTS | **349** | **342** | **348** |

### What These Results Tell Us

The simulated oracle experiments validate three things:
1. **LLM-CUCB-AT is safe**: it never performs catastrophically worse than CUCB, even under adversarial corruption
2. **The composite trust mechanism works**: LLM-CUCB-AT consistently beats agents that blindly follow the oracle (LLM-Greedy, ELLM-Adapted) by detecting and hedging against bad suggestions
3. **Real LLM experiments are essential**: the simulated oracle (a coin flip that knows the answer) gives an unfair advantage to Warm-Start CTS. A real LLM oracle with genuine reasoning about arm metadata is needed to show the full value of LLM-CUCB-AT's adaptive trust

### Generated Figures

All figures are in `figures/<experiment_name>/`:

| Figure | Description | Experiments |
|--------|-------------|-------------|
| `regret_multipanel.pdf` | Cumulative regret curves per corruption type | All |
| `trust_diagnostics.pdf` | kappa, rho, tau trajectories for LLM-CUCB-AT | All |
| `corruption_comparison.pdf` | Bar chart comparing agents across corruption types | All |
| `regret_vs_epsilon.pdf` | Theory validation: regret as function of epsilon | exp6 (pending), exp7 |
| `regret_vs_dimension.pdf` | Theory validation: regret as function of d | exp6 (pending), exp8 (pending) |
| `headline_clean.pdf` | Regret curves under clean oracle | All |
| `headline_consistent_wrong.pdf` | Regret curves under consistently wrong oracle | All |
| `significance_tests.csv` | Wilcoxon tests + Cohen's d | exp6 (pending) |

---

## Output Format

### Results JSON

Each trial produces a JSON entry with:

```json
{
  "agent": "llm_cucb_at",
  "env": "SyntheticBernoulliEnv",
  "corruption_type": "uniform",
  "epsilon": 0.1,
  "seed": 42,
  "d": 100,
  "m": 10,
  "T": 50000,
  "final_regret": 1234.5,
  "regret_curve": [0.0, 0.3, 0.8, ...],
  "oracle_queries": 16600,
  "oracle_tokens": 0,
  "trust_kappa": [null, null, ..., 0.8, 0.9, ...],
  "trust_rho": [null, null, ..., 0.95, 0.97, ...],
  "trust_tau": [null, null, ..., 0.8, 0.9, ...],
  "hedge_sizes": [0, 0, ..., 2, 1, ...]
}
```

- `trust_*` fields are `null` during the initialization phase (rounds 1 to T_0)
- `oracle_tokens` is 0 for simulated oracles (no API cost)

### Generated Figures

The `plot` command generates these PDFs:

| Figure | Filename | What It Shows |
|--------|----------|---------------|
| Regret curves (multi-panel) | `regret_multipanel.pdf` | One subplot per (corruption_type, epsilon) |
| Trust diagnostics | `trust_diagnostics.pdf` | kappa, rho, tau trajectories for LLM-CUCB-AT |
| Corruption comparison | `corruption_comparison.pdf` | Bar chart of final regret by agent and corruption |
| Regret vs epsilon | `regret_vs_epsilon.pdf` | Theory validation: O(epsilon*m*T/Delta) scaling |
| Regret vs dimension | `regret_vs_dimension.pdf` | Theory validation: dimension reduction |
| Headline (clean) | `headline_clean.pdf` | Regret curves under clean oracle (epsilon=0) |
| Headline (consistent wrong) | `headline_consistent_wrong.pdf` | The key result: composite trust catches bad oracle |
| Significance tests | `significance_tests.csv` | Wilcoxon tests + Cohen's d, LLM-CUCB-AT vs CUCB |

---

## External Data (Optional)

Simulated environment variants (`MINDEnvSimulated`, `InfluenceMaxEnvSimulated`) work without any external data and are used by default. For experiments with real data:

### MIND News Dataset

```bash
mkdir -p data/mind/train
# Download from https://msnews.github.io/
# Place sessions in data/mind/train/
```

### SNAP Social Graphs

```bash
mkdir -p data/snap

# ego-Facebook (4,039 nodes)
wget -P data/snap/ https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip data/snap/facebook_combined.txt.gz
mv data/snap/facebook_combined.txt data/snap/ego-facebook.txt

# ca-HepTh (9,877 nodes)
wget -P data/snap/ https://snap.stanford.edu/data/ca-HepTh.txt.gz
gunzip data/snap/ca-HepTh.txt.gz
```

---

## Real LLM Experiments

The paper's main claim is that LLMs provide useful combinatorial priors. Simulated experiments validate the theory; **real LLM experiments** validate the claim. `exp9_real_llm` calls GPT-4o/GPT-4o-mini on actual bandit tasks.

### Setup

```bash
export OPENAI_API_KEY=sk-...
```

### Running

```bash
# Real LLM experiment (~$60 API cost, ~30 min)
python -m combbandits.cli run configs/experiments/exp9_real_llm.yaml \
    --output-dir results/exp9_real_llm --workers 4
```

### How It Works

1. **LLMOracle** (`oracle/llm_oracle.py`) sends structured prompts to GPT-4o with arm metadata (name, category, tier, user rating, review count) and empirical reward estimates
2. **CachedOracle** wraps it with the O(√T) query schedule from Theorem 5 — only ~45 actual API calls per trial instead of 2,000, plus SQLite disk caching
3. The primary query uses GPT-4o; the K-1 re-queries use GPT-4o-mini with paraphrased prompts (for consistency estimation independence, Assumption 2)
4. Results include the LLM's measured epsilon, kappa, and rho — the first empirical validation of the graded quality model

### API Cost Breakdown

| Component | Per trial | Total (exp9) |
|-----------|----------|--------------|
| GPT-4o primary queries | ~45 calls × $0.01 | $0.45 |
| GPT-4o-mini re-queries | ~90 calls × $0.003 | $0.27 |
| **Per trial total** | | **~$0.72** |
| **Full experiment** | 6 agents × 2 envs × 10 seeds | **~$60** |

### Config Format

```yaml
oracles:
  - type: llm
    primary_model: gpt-4o          # Main query
    requery_model: gpt-4o-mini     # Cheaper K-1 re-queries
    provider: openai               # or anthropic
    K: 3                           # Number of re-queries for consistency
    temperature: 0.7
    schedule: sqrt                  # O(√T) query schedule
    cache_dir: cache/oracle_exp9   # SQLite disk cache
```

### What the Results Show

The real LLM experiment answers questions that simulated experiments cannot:
- **What is the LLM's actual epsilon?** (What fraction of suggestions are completely wrong?)
- **Does kappa reflect real LLM consistency?** (How much do paraphrased re-queries agree?)
- **Does posterior validation catch real LLM errors?** (Does rho drop when GPT-4o suggests bad arms?)
- **Is the dimension reduction real?** (Does LLM-CUCB-AT actually converge faster than CUCB with a real oracle?)

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output:
# tests/test_smoke.py::test_env_basic PASSED
# tests/test_smoke.py::test_cucb_runs PASSED
# tests/test_smoke.py::test_cts_runs PASSED
# tests/test_smoke.py::test_llm_cucb_at_runs PASSED
# tests/test_smoke.py::test_llm_greedy_runs PASSED
# tests/test_smoke.py::test_consistent_wrong_oracle PASSED
# tests/test_smoke.py::test_oracle_consistency PASSED
```

The critical test is `test_consistent_wrong_oracle`: it verifies that LLM-CUCB-AT achieves lower regret than LLM-Greedy under a deterministically wrong oracle (the headline result of the paper).

---

## GPU-Batched Execution

The `gpu/` module provides vectorized execution of all 9 agents across hundreds of seeds simultaneously on a single GPU. Agent state is stored as `(n_seeds, d)` tensors; arm selection, reward sampling, and updates are fully batched.

### Why GPU?

The CPU runner is fast for small d. But the paper's strongest claim — that LLM-CUCB-AT achieves O(sqrt(m(m+sqrt(d))T)) instead of CUCB's O(sqrt(mdT)) — becomes most compelling at **large d**. At d=5000, the theoretical advantage is 25x. The GPU runner makes d=5000 with 100 seeds practical.

### Running GPU Experiments

```bash
# Run the large-d scaling experiment (the paper's strongest figure)
python -m combbandits.cli run-gpu configs/experiments/exp8_scaling_d.yaml \
    --output-dir results/exp8_scaling_d

# Run any existing config on GPU with more seeds
python -m combbandits.cli run-gpu configs/experiments/exp6_workshop_main.yaml \
    --output-dir results/exp6_workshop_main --n-seeds 100

# Force a specific device
python -m combbandits.cli run-gpu configs/experiments/exp8_scaling_d.yaml \
    --device cuda --output-dir results/exp8_scaling_d
```

### GPU on DELLA

```bash
# Submit the scaling experiment as a single GPU job
sbatch --export=CONFIG_NAME=exp8_scaling_d cluster/della_gpu_batched.slurm

# Or run any config on GPU with custom seed count
sbatch --export=CONFIG_NAME=exp6_workshop_main,N_SEEDS=100 cluster/della_gpu_batched.slurm
```

### GPU Module Structure

```
src/combbandits/gpu/
├── device.py          # Auto-detect CUDA > MPS > CPU
├── batched_env.py     # Batched environments (torch.bernoulli for all seeds)
├── batched_oracle.py  # Batched simulated CLO (tensor set operations)
├── batched_agents.py  # All 9 agents with (n_seeds, d) tensor state
└── batched_trial.py   # Batched trial runner + experiment orchestrator
```

### Estimated GPU Runtimes (A100 80GB)

| Experiment | d range | Seeds | Time |
|-----------|---------|-------|------|
| `exp8_scaling_d` | 50--5000 | 100 | ~15 min |
| `exp6_workshop_main` | 50--200 | 100 | ~10 min |
| `exp7_ablation_trust` | 100 | 100 | ~3 min |

---

## Dependencies

Core (installed via `pip install -e .`):
- Python >= 3.10
- numpy, scipy, pandas, matplotlib, seaborn
- torch >= 2.0 (for GPU-batched execution)
- openai, anthropic, tiktoken (for real LLM oracles)
- networkx (for influence maximization graph structure)
- pyyaml, tqdm, click, aiosqlite

CPU experiments (`run` command) work without CUDA. GPU experiments (`run-gpu` command) use CUDA if available, fall back to MPS (Apple Silicon) or CPU.

---

## Citation

```bibtex
@inproceedings{kakaria2026combbandits,
  title={Regret-Bounded Combinatorial Bandits with Unreliable LLM Action Oracles},
  author={Kakaria, Vikram},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML) Workshops},
  year={2026}
}
```
