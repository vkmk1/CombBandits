#!/bin/bash
#SBATCH --job-name=combbandits
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vk5825@princeton.edu

# ------------------------------------------------------------------
# DELLA array job: runs a single (agent, env, oracle, seed) trial
# Set CONFIG_PATH and CONFIG_NAME via --export in della_submit.sh
# ------------------------------------------------------------------

set -euo pipefail

echo "=== Job info ==="
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CONFIG: $CONFIG_PATH"
echo "Node: $(hostname)"
echo "Start: $(date)"

# Load modules (adjust for DELLA's module system)
module purge
module load anaconda3/2024.6
conda activate combbandits 2>/dev/null || true

# Navigate to project root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Create log directory
mkdir -p logs/slurm

# Run single task
python -m combbandits.cli run \
    "$CONFIG_PATH" \
    --output-dir "results/${CONFIG_NAME}" \
    --workers 1 \
    --task-id "$SLURM_ARRAY_TASK_ID"

echo "=== Done: $(date) ==="
