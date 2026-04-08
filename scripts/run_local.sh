#!/bin/bash
# Run experiments locally (for development/debugging)
# Usage: bash scripts/run_local.sh <config_name> [--workers N]
#
# Examples:
#   bash scripts/run_local.sh exp3_quick_test
#   bash scripts/run_local.sh exp3_quick_test --workers 4

set -euo pipefail

CONFIG_NAME="${1:?Usage: run_local.sh <config_name> [--workers N]}"
WORKERS="${3:-1}"

CONFIG_PATH="configs/experiments/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "Running $CONFIG_NAME locally with $WORKERS workers..."

python -m combbandits.cli run \
    "$CONFIG_PATH" \
    --output-dir "results/${CONFIG_NAME}" \
    --workers "$WORKERS"

echo ""
echo "=== Results ==="
python -m combbandits.cli metrics "results/${CONFIG_NAME}/${CONFIG_NAME}_results.json"

echo ""
echo "=== Generating plots ==="
python -m combbandits.cli plot "results/${CONFIG_NAME}/${CONFIG_NAME}_results.json" --output-dir "figures/${CONFIG_NAME}"

echo "Done. Results in results/${CONFIG_NAME}/, figures in figures/${CONFIG_NAME}/"
