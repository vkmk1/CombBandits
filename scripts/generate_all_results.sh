#!/bin/bash
# Generate all figures, tables, and summary from completed experiments.
#
# Run this after all experiments complete:
#   bash scripts/generate_all_results.sh
#
# It checks which result files exist and generates figures for each.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Generating results from completed experiments ==="

EXPERIMENTS=(
    exp3_quick_test
    exp4_mind
    exp5_influence_max
    exp6_workshop_main
    exp7_ablation_trust
    exp8_scaling_d
    exp9_real_llm
)

for EXP in "${EXPERIMENTS[@]}"; do
    RESULTS="results/${EXP}/${EXP}_results.json"
    if [ -f "$RESULTS" ]; then
        echo ""
        echo "--- $EXP ---"
        echo "Metrics:"
        python3 -m combbandits.cli metrics "$RESULTS" 2>/dev/null | head -20
        echo ""
        echo "Generating figures..."
        python3 -m combbandits.cli plot "$RESULTS" --output-dir "figures/${EXP}" 2>/dev/null
        echo "  Figures saved to figures/${EXP}/"
        ls figures/${EXP}/*.pdf 2>/dev/null | sed 's/^/    /'
    else
        echo ""
        echo "--- $EXP --- (NOT RUN: $RESULTS not found)"
    fi
done

# Generate LaTeX table from exp6 if available
EXP6="results/exp6_workshop_main/exp6_workshop_main_results.json"
if [ -f "$EXP6" ]; then
    echo ""
    echo "=== LaTeX Table (d=100) ==="
    python3 scripts/generate_latex_table.py "$EXP6" --d 100
fi

echo ""
echo "=== Summary ==="
echo "Completed experiments:"
for EXP in "${EXPERIMENTS[@]}"; do
    RESULTS="results/${EXP}/${EXP}_results.json"
    if [ -f "$RESULTS" ]; then
        N=$(python3 -c "import json; print(len(json.load(open('$RESULTS'))))" 2>/dev/null)
        echo "  $EXP: $N trials"
    fi
done

echo ""
echo "Missing experiments:"
for EXP in "${EXPERIMENTS[@]}"; do
    RESULTS="results/${EXP}/${EXP}_results.json"
    if [ ! -f "$RESULTS" ]; then
        echo "  $EXP"
    fi
done
