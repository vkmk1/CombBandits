#!/bin/bash
#SBATCH --job-name=combbandits-agg
#SBATCH --output=logs/slurm/aggregate_%j.out
#SBATCH --error=logs/slurm/aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu

# ------------------------------------------------------------------
# Post-experiment: aggregate results and generate figures
# Usage: sbatch --export=CONFIG_NAME=exp1_synthetic cluster/della_aggregate.slurm
# ------------------------------------------------------------------

set -euo pipefail

module purge
module load anaconda3/2024.6
conda activate combbandits 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

RESULTS_DIR="results/${CONFIG_NAME}"

echo "Aggregating results from $RESULTS_DIR..."

# Merge per-task JSON results into one file
python -c "
import json, glob, sys
results = []
for f in sorted(glob.glob('${RESULTS_DIR}/*_results.json')):
    with open(f) as fh:
        data = json.load(fh)
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
out = '${RESULTS_DIR}/merged_results.json'
with open(out, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Merged {len(results)} results -> {out}')
"

# Compute metrics
python -m combbandits.cli metrics "${RESULTS_DIR}/merged_results.json"

# Generate plots
python -m combbandits.cli plot "${RESULTS_DIR}/merged_results.json" --output-dir "figures/${CONFIG_NAME}"

echo "Done. Figures in figures/${CONFIG_NAME}/"
