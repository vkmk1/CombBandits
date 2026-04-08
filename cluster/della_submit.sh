#!/bin/bash
# Submit experiments to Princeton DELLA cluster
# Usage: bash cluster/della_submit.sh <config_name> [--dry-run]
#
# Examples:
#   bash cluster/della_submit.sh exp1_synthetic
#   bash cluster/della_submit.sh exp3_quick_test --dry-run

set -euo pipefail

CONFIG_NAME="${1:?Usage: della_submit.sh <config_name> [--dry-run]}"
DRY_RUN="${2:-}"
CONFIG_PATH="configs/experiments/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

# Export task list
echo "Exporting task list from $CONFIG_PATH..."
python -m combbandits.cli export-tasks "$CONFIG_PATH" --output "results/${CONFIG_NAME}_tasks.csv"
N_TASKS=$(( $(wc -l < "results/${CONFIG_NAME}_tasks.csv") - 1 ))
echo "Total tasks: $N_TASKS"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "[DRY RUN] Would submit array job with $N_TASKS tasks"
    head -5 "results/${CONFIG_NAME}_tasks.csv"
    exit 0
fi

# Submit SLURM array job
sbatch \
    --array=0-$((N_TASKS - 1))%50 \
    --export=CONFIG_PATH="$CONFIG_PATH",CONFIG_NAME="$CONFIG_NAME" \
    cluster/della_array.slurm

echo "Submitted $N_TASKS tasks as array job"
