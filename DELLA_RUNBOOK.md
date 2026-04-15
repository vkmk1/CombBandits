# DELLA Runbook — Workshop Paper Experiments

Three experiments remain to be run on DELLA. Copy-paste each section in order.

## Prerequisites

```bash
ssh vk5825@della.princeton.edu
cd /scratch/gpfs/vk5825/LIKEN/CombBandits
conda activate combbandits

# Verify installation
python -m pytest tests/ -v  # all 7 should pass
```

## Experiment 6: Theory Validation (CPU, ~35 min)

The main simulated oracle experiment. Validates regret bounds across 3 dimensions, 10 oracle configs, 7 agents, 30 seeds.

```bash
# Submit as SLURM array job
bash cluster/della_submit.sh exp6_workshop_main

# Monitor
squeue -u $USER
# When done:
sbatch --export=CONFIG_NAME=exp6_workshop_main cluster/della_aggregate.slurm
```

**What it produces:** `figures/exp6_workshop_main/` — regret curves, corruption comparison, regret vs epsilon, regret vs dimension, trust diagnostics. Plus `significance_tests.csv` and LaTeX table via `generate_latex_table.py`.

## Experiment 8: Large-d Scaling (GPU, ~15 min)

The strongest figure in the paper. Tests d from 50 to 5000 with 100 seeds per config on GPU. Validates the O(sqrt(m+sqrt(d))) dimension reduction.

```bash
# Submit as single GPU job (all seeds batched on one A100)
sbatch --export=CONFIG_NAME=exp8_scaling_d cluster/della_gpu_batched.slurm

# Monitor
squeue -u $USER
cat logs/slurm/gpu_batched_*.out
```

**What it produces:** `figures/exp8_scaling_d/` — the dimension scaling plot showing CUCB curving up as sqrt(d) while LLM-CUCB-AT stays flat. This is the paper's strongest result.

## Experiment 9: Real LLM Oracle (CPU + API, ~30 min, ~$60)

The critical experiment for workshop acceptance. Actual GPT-4o API calls, not simulated.

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Run locally on DELLA login node (API-latency bound, not compute bound)
# Or submit as a regular job:
python -m combbandits.cli run configs/experiments/exp9_real_llm.yaml \
    --output-dir results/exp9_real_llm --workers 4

# Generate figures when done
python -m combbandits.cli plot results/exp9_real_llm/exp9_real_llm_results.json \
    --output-dir figures/exp9_real_llm
```

**What it produces:** Real LLM results showing GPT-4o's actual epsilon, kappa, rho on bandit tasks. The first empirical validation that LLMs provide useful combinatorial priors.

**Cost:** ~$60 (135 API calls per trial × 120 trials × $0.003-0.01 per call)

## After All Experiments Complete

```bash
# Generate everything
bash scripts/generate_all_results.sh

# Download to local machine
# (from your laptop, not DELLA):
scp -r vk5825@della.princeton.edu:/scratch/gpfs/vk5825/LIKEN/CombBandits/results/ ./results/
scp -r vk5825@della.princeton.edu:/scratch/gpfs/vk5825/LIKEN/CombBandits/figures/ ./figures/

# Generate LaTeX table for the paper
python scripts/generate_latex_table.py results/exp6_workshop_main/merged_results.json --d 100

# Build the workshop paper
bash paper/build_paper.sh
```

## Troubleshooting

```bash
# Check failed SLURM tasks
sacct -j <JOBID> --format=JobID,State,ExitCode | grep FAILED

# Resubmit specific failed tasks
sbatch --array=42,99 \
    --export=CONFIG_PATH="configs/experiments/exp6_workshop_main.yaml",CONFIG_NAME="exp6_workshop_main" \
    cluster/della_array.slurm

# Check GPU availability
sinfo -p gpu --format="%n %G %t" | head -10

# Check API key is set
echo $OPENAI_API_KEY | head -c 10
```
