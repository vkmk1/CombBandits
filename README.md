# CombBandits: LLM-CUCB-AT

Regret-Bounded Combinatorial Bandits with Unreliable LLM Action Oracles.

NeurIPS 2026 submission codebase. Implements LLM-CUCB-AT (Algorithm 1) with 8 baselines, 3 environment types, 4 corruption models, and full SLURM integration for Princeton DELLA.

## Quick Start

```bash
# 1. Install
cd CombBandits
pip install -e .

# 2. Smoke test (runs in ~30 seconds locally)
bash scripts/run_local.sh exp3_quick_test

# 3. View results
python -m combbandits.cli metrics results/exp3_quick_test/exp3_quick_test_results.json
```

## Project Structure

```
CombBandits/
├── src/combbandits/
│   ├── agents/              # All 9 bandit algorithms
│   │   ├── base.py          # Abstract Agent class
│   │   ├── cucb.py          # CUCB (Chen et al., 2013)
│   │   ├── cts.py           # Combinatorial Thompson Sampling
│   │   ├── llm_cucb_at.py   # *** OUR METHOD ***
│   │   ├── llm_greedy.py    # LLM-Greedy (no exploration)
│   │   ├── ellm_adapted.py  # ELLM-style exploration bonus
│   │   ├── opro_bandit.py   # OPRO-style iterative optimization
│   │   ├── corrupt_robust_cucb.py  # Median-of-means robust CUCB
│   │   ├── warm_start_cts.py       # CTS with LLM-initialized priors
│   │   └── exp4.py          # EXP4 adversarial baseline
│   ├── environments/        # Combinatorial semi-bandit environments
│   │   ├── base.py          # Abstract CombBanditEnv
│   │   ├── synthetic.py     # Bernoulli arms (4 gap structures)
│   │   ├── mind.py          # MIND news recommendation
│   │   └── influence_max.py # SNAP influence maximization
│   ├── oracle/              # Combinatorial LLM Oracle (CLO)
│   │   ├── base.py          # CLO interface
│   │   ├── simulated.py     # Parameterized noisy oracle (4 corruption types)
│   │   ├── llm_oracle.py    # Real LLM via OpenAI/Anthropic API
│   │   └── cached_oracle.py # O(√T) query schedule + disk cache
│   ├── engine/              # Experiment orchestration
│   │   ├── trial.py         # Single trial runner
│   │   └── runner.py        # Experiment grid + parallel execution
│   ├── analysis/            # Results processing
│   │   ├── metrics.py       # Summary statistics
│   │   └── plots.py         # NeurIPS-quality figures
│   ├── cli.py               # CLI entry point
│   └── types.py             # Core data structures
├── configs/experiments/     # YAML experiment configs
│   ├── exp1_synthetic.yaml  # Main synthetic experiments (Table 1)
│   ├── exp2_ablation.yaml   # Ablation studies (Table 2)
│   ├── exp3_quick_test.yaml # Smoke test
│   ├── exp4_mind.yaml       # MIND recommendation
│   └── exp5_influence_max.yaml  # SNAP influence max
├── cluster/                 # DELLA SLURM scripts
│   ├── della_submit.sh      # Job submission orchestrator
│   ├── della_array.slurm    # Array job (CPU)
│   ├── della_gpu.slurm      # Array job (GPU)
│   └── della_aggregate.slurm # Post-experiment aggregation
└── scripts/
    ├── run_local.sh          # Local execution
    └── setup_della.sh        # One-time DELLA setup
```

## Running on Princeton DELLA

### First-time setup

```bash
# SSH into DELLA
ssh <netid>@della.princeton.edu

# Clone and setup
cd /scratch/gpfs/<netid>/
git clone <repo-url> LIKEN
cd LIKEN/CombBandits
bash scripts/setup_della.sh
```

### Running experiments

```bash
# Activate environment
conda activate combbandits

# Run smoke test first
bash scripts/run_local.sh exp3_quick_test

# Submit main synthetic experiments as SLURM array job
bash cluster/della_submit.sh exp1_synthetic

# Check job status
squeue -u $USER

# After all jobs complete, aggregate results
sbatch --export=CONFIG_NAME=exp1_synthetic cluster/della_aggregate.slurm
```

### Experiment configs

Each YAML config defines the full experiment grid. The runner expands it into individual trials:

| Config | Tasks | Time/task | What it tests |
|--------|-------|-----------|---------------|
| `exp3_quick_test` | ~36 | ~10s | Smoke test |
| `exp1_synthetic` | ~4,860 | ~5min | Main results (d=50/100/200, 4 corruption types, 9 agents) |
| `exp2_ablation` | ~200 | ~3min | K, h_max, trust score ablations |
| `exp4_mind` | ~360 | ~2min | MIND recommendation |
| `exp5_influence_max` | ~360 | ~5min | Influence maximization |

### SLURM job management

```bash
# Dry run (see how many tasks without submitting)
bash cluster/della_submit.sh exp1_synthetic --dry-run

# Cancel all your jobs
scancel -u $USER

# Check a specific job's output
cat logs/slurm/<job_id>_<task_id>.out
```

## Corruption Models

The paper defines 4 corruption types for the simulated oracle. Set in YAML configs:

| Type | `corruption_type` | Behavior | Tests |
|------|-------------------|----------|-------|
| Uniform | `uniform` | Returns S* w.p. 1-epsilon, random otherwise | Standard theory |
| Adversarial | `adversarial` | Returns worst set w.p. epsilon | Worst-case robustness |
| Partial overlap | `partial_overlap` | Returns set overlapping S* on (1-epsilon)m arms | Graded quality model |
| Consistently wrong | `consistent_wrong` | Deterministically returns plausible suboptimal set | Posterior validation necessity |

## Using Real LLMs

For experiments with actual LLM oracles (GPT-4o, Claude):

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Edit oracle config in YAML:
# oracles:
#   - type: llm
#     primary_model: gpt-4o
#     requery_model: gpt-4o-mini
#     provider: openai
#     K: 3
#     temperature: 0.7
```

The `CachedOracle` wrapper (enabled by default for LLM oracles) caches responses to SQLite and implements the O(sqrt(T)) query schedule to reduce API costs.

## Downloading External Data

### MIND News Dataset

```bash
mkdir -p data/mind/train
# Download from https://msnews.github.io/
# Place in data/mind/train/
# Then preprocess:
python scripts/preprocess_mind.py  # (to be implemented based on your MIND download)
```

### SNAP Social Graphs

```bash
mkdir -p data/snap
# ego-Facebook:
wget -P data/snap/ https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip data/snap/facebook_combined.txt.gz
mv data/snap/facebook_combined.txt data/snap/ego-facebook.txt

# ca-HepTh:
wget -P data/snap/ https://snap.stanford.edu/data/ca-HepTh.txt.gz
gunzip data/snap/ca-HepTh.txt.gz
```

The simulated environment variants (`MINDEnvSimulated`, `InfluenceMaxEnvSimulated`) work without external data and are used by default.

## Key Design Decisions

**Why composite trust (kappa + rho)?**  
Pure consistency (kappa) fails against deterministically wrong oracles. See `exp1_synthetic` with `consistent_wrong` corruption: agents using kappa-only trust suffer linear regret, while LLM-CUCB-AT with composite trust detects the bad oracle via posterior validation (rho) and recovers.

**Why O(sqrt(d)) hedge?**  
h_max = sqrt(d) balances exploration cost against coverage. Too small: misses optimal arms under corruption. Too large: reduces to vanilla CUCB with no dimension benefit. Ablated in `exp2_ablation`.

**Why process-per-trial parallelism?**  
Each trial is independent and CPU-bound (no GPU needed for synthetic/simulated experiments). SLURM array jobs map naturally to one trial per task. For GPU experiments (real LLM, large influence max), use `della_gpu.slurm`.

## Generating Paper Figures

```bash
# After experiments complete:
python -m combbandits.cli plot results/exp1_synthetic/merged_results.json --output-dir figures/paper

# Specific corruption type:
python -m combbandits.cli plot results/exp1_synthetic/merged_results.json \
    --corruption uniform --epsilon 0.1 --output-dir figures/paper
```

## Citation

```
@inproceedings{kakaria2026combbandits,
  title={Regret-Bounded Combinatorial Bandits with Unreliable LLM Action Oracles},
  author={Kakaria, Vikram},
  booktitle={NeurIPS},
  year={2026}
}
```
