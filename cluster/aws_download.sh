#!/usr/bin/env bash
# Download results from S3 after the EC2 instance has finished and self-terminated.
# Usage: bash cluster/aws_download.sh
set -euo pipefail

S3_BUCKET="combbandits-results-099841456154"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Downloading from s3://$S3_BUCKET ..."
aws s3 sync "s3://$S3_BUCKET/results/" "$REPO_DIR/results/"
aws s3 sync "s3://$S3_BUCKET/figures/" "$REPO_DIR/figures/"
aws s3 cp  "s3://$S3_BUCKET/experiment.log" "$REPO_DIR/results/aws_experiment.log" 2>/dev/null || true

echo ""
echo "Done. Results in $REPO_DIR/results/"
echo "Figures in $REPO_DIR/figures/"
