#!/usr/bin/env bash
# Build the paper PDF.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] Generating figures from results..."
python3 make_figures.py

echo "[2/4] First pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex1.log 2>&1 || \
  { echo "pdflatex pass 1 failed; tail of log:"; tail -40 /tmp/pdflatex1.log; exit 1; }

echo "[3/4] BibTeX pass..."
bibtex main > /tmp/bibtex.log 2>&1 || \
  { echo "bibtex failed; tail of log:"; tail -40 /tmp/bibtex.log; exit 1; }

echo "[4/4] Final pdflatex passes..."
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex2.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/pdflatex3.log 2>&1

if [[ -f main.pdf ]]; then
  echo
  echo "SUCCESS: $(pwd)/main.pdf"
  echo "  pages: $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/ {print $2}')"
else
  echo "FAILED to produce PDF; check /tmp/pdflatex3.log"
  tail -40 /tmp/pdflatex3.log
  exit 1
fi
