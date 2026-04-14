#!/bin/bash
# Build the workshop paper after experiments are complete.
#
# Usage:
#   bash paper/build_paper.sh
#
# Prerequisites:
#   1. Run experiments: exp6_workshop_main, exp7_ablation_trust, exp4_mind/exp5_influence_max
#   2. Aggregate results (della_aggregate.slurm or run_local.sh)
#   3. This script copies figures and compiles the paper
#
# If running on DELLA, first scp the figures directory to your local machine.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Copying figures from experiment outputs ==="

# Copy from workshop main experiment
for fig in regret_multipanel.pdf trust_diagnostics.pdf corruption_comparison.pdf \
           regret_vs_epsilon.pdf regret_vs_dimension.pdf \
           headline_clean.pdf headline_consistent_wrong.pdf \
           significance_tests.csv; do
    for exp_dir in ../figures/exp6_workshop_main ../figures/exp7_ablation_trust ../figures/exp1_synthetic; do
        if [ -f "$exp_dir/$fig" ]; then
            cp "$exp_dir/$fig" "figures/$fig"
            echo "  Copied $fig from $exp_dir"
            break
        fi
    done
done

echo ""
echo "=== Figures in paper/figures/ ==="
ls -la figures/

echo ""
echo "=== Compiling paper ==="
pdflatex -interaction=nonstopmode workshop_paper
bibtex workshop_paper 2>/dev/null || true
pdflatex -interaction=nonstopmode workshop_paper
pdflatex -interaction=nonstopmode workshop_paper

echo ""
echo "=== Done: workshop_paper.pdf ==="
ls -la workshop_paper.pdf
