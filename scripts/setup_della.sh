#!/bin/bash
# Setup CombBandits environment on Princeton DELLA cluster
# Run this once after cloning the repo to DELLA
#
# Usage: bash scripts/setup_della.sh

set -euo pipefail

echo "=== Setting up CombBandits on DELLA ==="

# Load modules
module purge
module load anaconda3/2024.6

# Create conda environment
conda create -n combbandits python=3.11 -y
conda activate combbandits

# Install package in editable mode
pip install -e ".[dev]"

# Create necessary directories
mkdir -p results logs/slurm figures cache/oracle data

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate combbandits"
echo "Quick test:    bash scripts/run_local.sh exp3_quick_test"
echo "Submit jobs:   bash cluster/della_submit.sh exp1_synthetic"
echo ""
echo "For real LLM experiments, set your API key:"
echo "  export OPENAI_API_KEY=sk-..."
echo ""
echo "For MIND/SNAP data, see README.md for download instructions."
