#!/usr/bin/env bash
# Runs on the EC2 instance.
# - Installs deps in a venv
# - Runs all CPU experiments in parallel
# - Collects full metadata (system info, model weights provenance, per-exp logs)
# - Pushes results + metadata to GitHub
# - Self-terminates
set -euo pipefail

S3_BUCKET="combbandits-results-099841456154"
REGION="us-east-1"
LOG="$HOME/experiment.log"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
REPO="$HOME/CombBandits"
VENV="$HOME/venv"
GH_REPO="vkmk1/CombBandits"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

log "=== CombBandits EC2 Runner ==="
log "Instance: $INSTANCE_ID ($INSTANCE_TYPE)  |  CPUs: $(nproc)  |  RAM: $(free -g | awk '/Mem/{print $2}')GB"

cd "$REPO"

# ── Install into venv ──────────────────────────────────────────────────────
log "Creating venv and installing dependencies..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e ".[dev]" 2>&1 | tail -5
log "Install complete."

PYTHON="$VENV/bin/python"

# ── Collect system + environment metadata ─────────────────────────────────
log "Collecting experiment metadata..."
mkdir -p metadata

cat > metadata/run_info.json <<RUNINFO
{
  "run_timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "instance_id": "$INSTANCE_ID",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION",
  "cpus": $(nproc),
  "ram_gb": $(free -g | awk '/Mem/{print $2}'),
  "python_version": "$("$PYTHON" --version 2>&1)",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
  "git_dirty": $(git diff --quiet && echo false || echo true),
  "llm_oracle": {
    "model": "us.meta.llama4-scout-17b-instruct-v1:0",
    "provider": "AWS Bedrock (cross-region inference profile)",
    "model_family": "Llama 4 Scout",
    "active_params_b": 17,
    "total_params_b": 109,
    "architecture": "Mixture-of-Experts (16 experts, top-2 routing)",
    "context_window": 10000000,
    "open_weights": true,
    "weights_url": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "weights_license": "Llama 4 Community License",
    "release_date": "2025-04-05",
    "bedrock_pricing_per_1k_input_tokens": 0.00017,
    "bedrock_pricing_per_1k_output_tokens": 0.00017
  },
  "experiments": ["exp6_workshop_main","exp7_ablation_trust","exp4_mind","exp5_influence_max","exp9_bedrock"]
}
RUNINFO

# Installed package versions for full reproducibility
"$PYTHON" -m pip freeze > metadata/requirements_frozen.txt
log "Metadata written to metadata/"

# ── Run experiments in parallel ────────────────────────────────────────────
# Worker allocation (32 vCPUs total on c7i.8xlarge):
#   exp6: 26 workers (dominant task: 14,040 tasks × T=100K)
#   exp7:  4 workers (510 tasks × T=30K)
#   exp4:  2 workers (280 tasks × T=2K)  — finishes fast, frees cores
#   exp5:  2 workers (280 tasks × T=5K)  — finishes fast, frees cores
#   exp9:  6 workers (180 trials, Bedrock API-latency bound, not CPU bound)
# Total: 40 allocated, but exp4/exp5 finish in <3 min, so steady-state is ~32.

run_exp() {
  local EXP="$1"
  local WORKERS="$2"
  local EXPLOG="$HOME/log_${EXP}.log"
  local START=$(date +%s)
  log "▶  $EXP (workers=$WORKERS)"

  "$PYTHON" -m combbandits.cli run \
    "configs/experiments/${EXP}.yaml" \
    --output-dir "results/$EXP" \
    --workers "$WORKERS" > "$EXPLOG" 2>&1

  "$PYTHON" -m combbandits.cli metrics \
    "results/$EXP/${EXP}_results.json" >> "$EXPLOG" 2>&1 || true

  "$PYTHON" -m combbandits.cli plot \
    "results/$EXP/${EXP}_results.json" \
    --output-dir "figures/$EXP" >> "$EXPLOG" 2>&1 || true

  # Per-experiment metadata: timing, task count, final regret summary
  local END=$(date +%s)
  local ELAPSED=$(( END - START ))
  python3 -c "
import json, sys
with open('results/${EXP}/${EXP}_results.json') as f:
    r = json.load(f)
agents = list({x['agent'] for x in r})
summary = {a: round(sum(x['final_regret'] for x in r if x['agent']==a)/max(1,len([x for x in r if x['agent']==a])),1) for a in agents}
print(json.dumps({'exp': '${EXP}', 'n_trials': len(r), 'wall_time_sec': $ELAPSED, 'mean_final_regret': summary}, indent=2))
" > "metadata/${EXP}_summary.json" 2>/dev/null || true

  log "✓  $EXP done in ${ELAPSED}s"
  cat "$EXPLOG" >> "$LOG"
}

run_exp exp6_workshop_main  26 &  PID_EXP6=$!
run_exp exp7_ablation_trust  4 &  PID_EXP7=$!
run_exp exp4_mind             2 &  PID_EXP4=$!
run_exp exp5_influence_max    2 &  PID_EXP5=$!
run_exp exp9_bedrock          6 &  PID_EXP9=$!

log "All 5 experiments running in parallel..."

FAILED=0
for PAIR in "$PID_EXP6:exp6" "$PID_EXP7:exp7" "$PID_EXP4:exp4" "$PID_EXP5:exp5" "$PID_EXP9:exp9"; do
  PID="${PAIR%%:*}"; NAME="${PAIR##*:}"
  if wait "$PID"; then log "  $NAME: OK"
  else log "  $NAME: FAILED"; FAILED=$(( FAILED + 1 )); fi
done

[[ $FAILED -gt 0 ]] && log "WARNING: $FAILED experiment(s) failed" || log "All experiments complete."

# ── Write combined metadata summary ───────────────────────────────────────
log "Writing combined metadata..."
END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
python3 -c "
import json, glob, os
summaries = {}
for f in glob.glob('metadata/*_summary.json'):
    name = os.path.basename(f).replace('_summary.json','')
    with open(f) as fh:
        summaries[name] = json.load(fh)
with open('metadata/run_info.json') as f:
    info = json.load(f)
info['end_timestamp'] = '$END_TIME'
info['experiment_summaries'] = summaries
info['failed_count'] = $FAILED
with open('metadata/run_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print('metadata/run_info.json updated')
"

cp "$LOG" metadata/full_run.log

# ── Upload to S3 ───────────────────────────────────────────────────────────
log "Uploading to S3..."
aws s3 sync results/  "s3://$S3_BUCKET/results/"  --region "$REGION"
aws s3 sync figures/  "s3://$S3_BUCKET/figures/"  --region "$REGION"
aws s3 sync metadata/ "s3://$S3_BUCKET/metadata/" --region "$REGION"
log "S3 upload done."

# ── Push to GitHub ─────────────────────────────────────────────────────────
log "Pushing results to GitHub..."

GH_TOKEN=$(aws ssm get-parameter \
  --name "/combbandits/github_token" \
  --with-decryption \
  --query "Parameter.Value" \
  --output text 2>/dev/null) || { log "No GitHub token in SSM — skipping push"; GH_TOKEN=""; }

if [[ -n "$GH_TOKEN" ]]; then
  git config user.email "combbandits-runner@aws.ec2"
  git config user.name  "CombBandits EC2 Runner"
  git remote set-url origin "https://${GH_TOKEN}@github.com/${GH_REPO}.git"

  # Stage results, figures, metadata (not venv/cache)
  git add results/ figures/ metadata/ configs/ src/ 2>/dev/null || true
  git add -f results/ figures/ metadata/ 2>/dev/null || true

  COMMIT_MSG="[EC2 runner] Experiment results $(date -u '+%Y-%m-%d %H:%M UTC')

Instance: $INSTANCE_TYPE ($INSTANCE_ID)
Model:    Llama 4 Scout 17B (us.meta.llama4-scout-17b-instruct-v1:0)
Exps:     exp4 exp5 exp6 exp7 exp9_bedrock
Failed:   $FAILED

See metadata/run_info.json for full provenance."

  git commit -m "$COMMIT_MSG" || log "Nothing new to commit"
  git push origin HEAD 2>&1 | tee -a "$LOG" || log "Git push failed — results still in S3"
  log "GitHub push done."
fi

# ── Self-terminate ─────────────────────────────────────────────────────────
log "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
