#!/bin/bash
#SBATCH --job-name=combbandits-gpu
#SBATCH --output=logs/slurm/gpu_batched_%j.out
#SBATCH --error=logs/slurm/gpu_batched_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vk5825@princeton.edu

# ------------------------------------------------------------------
# GPU-batched experiment: runs ALL seeds on a single A100.
# Usage:
#   sbatch --export=CONFIG_NAME=exp8_scaling_d cluster/della_gpu_batched.slurm
#   sbatch --export=CONFIG_NAME=exp6_workshop_main,N_SEEDS=100 cluster/della_gpu_batched.slurm
# ------------------------------------------------------------------

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME via --export}"
N_SEEDS="${N_SEEDS:-}"
CONFIG_PATH="configs/experiments/${CONFIG_NAME}.yaml"

echo "=== GPU Batched Job ==="
echo "Config: $CONFIG_PATH"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Start: $(date)"

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.4
conda activate combbandits 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs/slurm results figures

EXTRA_ARGS=""
if [ -n "$N_SEEDS" ]; then
    EXTRA_ARGS="--n-seeds $N_SEEDS"
fi

python -m combbandits.cli run-gpu \
    "$CONFIG_PATH" \
    --output-dir "results/${CONFIG_NAME}" \
    $EXTRA_ARGS

# Generate figures
python -m combbandits.cli plot \
    "results/${CONFIG_NAME}/${CONFIG_NAME}_results.json" \
    --output-dir "figures/${CONFIG_NAME}"

echo "=== Done: $(date) ==="
