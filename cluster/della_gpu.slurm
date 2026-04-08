#!/bin/bash
#SBATCH --job-name=combbandits-gpu
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vk5825@princeton.edu

# ------------------------------------------------------------------
# GPU variant for experiments that need GPU (influence max with large
# Monte Carlo rollouts, or if using local LLM like Llama-3-8B via vLLM)
# ------------------------------------------------------------------

set -euo pipefail

echo "=== Job info ==="
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CONFIG: $CONFIG_PATH"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Start: $(date)"

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.4
conda activate combbandits 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs/slurm

python -m combbandits.cli run \
    "$CONFIG_PATH" \
    --output-dir "results/${CONFIG_NAME}" \
    --workers 1 \
    --task-id "$SLURM_ARRAY_TASK_ID"

echo "=== Done: $(date) ==="
