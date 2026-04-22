#!/usr/bin/env bash
# Runs on g4dn.2xlarge (1x T4 16GB, 8 vCPUs, 32GB RAM, 225GB NVMe).
# Phase 1 (parallel): exp4, exp5, exp6, exp7, exp9_bedrock  — CPU
# Phase 2 (GPU):      exp8_scaling_d                        — GPU batched
# Phase 3 (GPU):      exp9_local (Llama 4 Scout + weights)  — GPU, workers=1
# Uploads to S3 + GitHub, then self-terminates.
set -euo pipefail

S3_BUCKET="combbandits-results-099841456154"
REGION="us-east-1"
LOG="$HOME/experiment.log"
REPO="$HOME/CombBandits"
VENV="$HOME/venv"
GH_REPO="vkmk1/CombBandits"

# IMDSv2 token for metadata queries
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
META_HEADER=(-H "X-aws-ec2-metadata-token: $IMDS_TOKEN")
INSTANCE_ID=$(curl -s "${META_HEADER[@]}" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s "${META_HEADER[@]}" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

log "=== CombBandits GPU Runner ==="
log "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"
log "CPUs: $(nproc)  RAM: $(free -g | awk '/Mem/{print $2}')GB"

cd "$REPO"

# ── GPU check ──────────────────────────────────────────────────────────────
log "GPU status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"

# ── Use NVMe local SSD for scratch (model weights, HF cache) ─────────────
# DLAMI pre-mounts the NVMe instance store at /opt/dlami/nvme
if mountpoint -q /opt/dlami/nvme 2>/dev/null; then
  SCRATCH="/opt/dlami/nvme"
  sudo chmod 777 "$SCRATCH"
  log "NVMe SSD: $(df -h "$SCRATCH" | tail -1)"
else
  SCRATCH="$HOME"
  log "No NVMe mount found, using EBS root."
fi
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"

# ── Install into venv ──────────────────────────────────────────────────────
log "Installing dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv > /dev/null 2>&1
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e ".[dev]"
"$VENV/bin/pip" install -q \
  "transformers>=4.47" accelerate "bitsandbytes>=0.41" \
  huggingface_hub scikit-learn
log "Install complete."

PYTHON="$VENV/bin/python"

# ── Fetch secrets (SSM → local file fallback) ─────────────────────────────
log "Fetching secrets..."
HF_TOKEN=$(aws ssm get-parameter \
  --name "/combbandits/hf_token" --with-decryption \
  --query "Parameter.Value" --output text 2>/dev/null) || HF_TOKEN=""
[[ -z "$HF_TOKEN" && -f "$HOME/hf_token.txt" ]] && HF_TOKEN=$(cat "$HOME/hf_token.txt")
[[ -z "$HF_TOKEN" ]] && log "WARNING: no HF_TOKEN found"
GH_TOKEN=$(aws ssm get-parameter \
  --name "/combbandits/github_token" --with-decryption \
  --query "Parameter.Value" --output text 2>/dev/null) || GH_TOKEN=""
[[ -z "$GH_TOKEN" && -f "$HOME/gh_token.txt" ]] && GH_TOKEN=$(cat "$HOME/gh_token.txt")
[[ -z "$GH_TOKEN" ]] && log "WARNING: no GitHub token found"

export HF_TOKEN

# ── Inject HF token into exp9_local config ────────────────────────────────
if [[ -n "$HF_TOKEN" ]]; then
  "$PYTHON" -c "
import yaml
with open('configs/experiments/exp9_local.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['oracles'][0]['hf_token'] = '$HF_TOKEN'
with open('configs/experiments/exp9_local.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('HF token injected into exp9_local.yaml')
"
fi

# ── Write initial metadata ─────────────────────────────────────────────────
log "Writing metadata..."
mkdir -p metadata
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

cat > metadata/run_info.json << RUNINFO
{
  "run_timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "instance_id": "$INSTANCE_ID",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION",
  "cpus": $(nproc),
  "ram_gb": $(free -g | awk '/Mem/{print $2}'),
  "gpu": "$GPU_NAME",
  "gpu_vram": "$GPU_MEM",
  "python_version": "$("$PYTHON" --version 2>&1)",
  "git_commit": "$(git rev-parse HEAD)",
  "llm_oracle_bedrock": {
    "model": "us.meta.llama4-scout-17b-instruct-v1:0",
    "provider": "AWS Bedrock (cross-region inference profile)",
    "open_weights": true,
    "weights_url": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct"
  },
  "llm_oracle_local": {
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "quantization": "4-bit (bitsandbytes NF4)",
    "device": "cuda (T4 16GB)",
    "weight_tracking": "per-call: logprobs, attention, hidden-state PCA, entropy"
  },
  "experiments": ["exp4_mind","exp5_influence_max","exp6_workshop_main",
                  "exp7_ablation_trust","exp8_scaling_d","exp9_bedrock","exp9_local"]
}
RUNINFO

"$PYTHON" -m pip freeze > metadata/requirements_frozen.txt

# ── Helper: run one experiment, generate figures + summary ─────────────────
FAILED=0

run_exp() {
  local EXP="$1" WORKERS="$2"
  local EXPLOG="$HOME/log_${EXP}.log"
  local START=$(date +%s)
  log "▶  $EXP (workers=$WORKERS)"
  mkdir -p "results/$EXP"

  "$PYTHON" -m combbandits.cli run \
    "configs/experiments/${EXP}.yaml" \
    --output-dir "results/$EXP" \
    --workers "$WORKERS" > "$EXPLOG" 2>&1

  # Figures
  "$PYTHON" -m combbandits.cli plot \
    "results/$EXP/${EXP}_results.json" \
    --output-dir "figures/$EXP" >> "$EXPLOG" 2>&1 || true

  # Per-experiment summary JSON
  local ELAPSED=$(( $(date +%s) - START ))
  "$PYTHON" - << PYEOF >> "$EXPLOG" 2>&1 || true
import json
with open('results/${EXP}/${EXP}_results.json') as f:
    r = json.load(f)
agents = list({x['agent'] for x in r})
summary = {
    'exp': '${EXP}',
    'n_trials': len(r),
    'wall_time_sec': ${ELAPSED},
    'mean_final_regret': {a: round(sum(x['final_regret'] for x in r if x['agent']==a)
                                   /max(1,len([x for x in r if x['agent']==a])),1) for a in agents}
}
with open('metadata/${EXP}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

  log "✓  $EXP done in ${ELAPSED}s"
}

# ── PHASE 1: exp9_bedrock on CPU (API-latency bound, runs during GPU phase) ─
# exp9_bedrock uses real Bedrock API calls — must run on CPU with network I/O.
# All other simulated experiments run on GPU (Phase 2).
log "=== PHASE 1: exp9_bedrock (CPU, background) ==="
run_exp exp9_bedrock 4 &  PID9B=$!

# ── PHASE 2: GPU batched experiments (exp4, exp5, exp6, exp7, exp8) ───────
# All simulated environments (synthetic_bernoulli, mind_simulated,
# influence_max_simulated) and all agents have GPU-batched implementations.
# Each experiment's n_seeds run simultaneously on GPU tensors.
log "=== PHASE 2: GPU batched experiments ==="

GPU_EXPS="exp4_mind exp5_influence_max exp6_workshop_main exp7_ablation_trust exp8_scaling_d"

for EXP in $GPU_EXPS; do
  START=$(date +%s)
  EXPLOG="$HOME/log_${EXP}.log"
  log "▶  $EXP (GPU batched)"
  mkdir -p "results/$EXP"

  "$PYTHON" - "$EXP" << 'PYEOF' > "$EXPLOG" 2>&1
import sys, yaml, json, torch, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
sys.path.insert(0, 'src')
from combbandits.gpu.batched_trial import run_batched_experiment

exp = sys.argv[1]
cfg_path = f'configs/experiments/{exp}.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda')
print(f'Running {exp} on {device} ({torch.cuda.get_device_name(0)})...')
results = run_batched_experiment(cfg, device=device)
print(f'{exp} done: {len(results)} trials')

out_dir = Path(f'results/{exp}')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / f'{exp}_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved results/{exp}/{exp}_results.json')
PYEOF

  if [[ $? -eq 0 ]]; then
    ELAPSED=$(( $(date +%s) - START ))
    log "✓  $EXP (GPU) done in ${ELAPSED}s"
    "$PYTHON" -m combbandits.cli plot \
      "results/$EXP/${EXP}_results.json" \
      --output-dir "figures/$EXP" >> "$EXPLOG" 2>&1 || true

    # Per-experiment summary
    "$PYTHON" - << PYEOF >> "$EXPLOG" 2>&1 || true
import json
with open('results/${EXP}/${EXP}_results.json') as f:
    r = json.load(f)
agents = list({x['agent'] for x in r})
summary = {
    'exp': '${EXP}',
    'n_trials': len(r),
    'wall_time_sec': ${ELAPSED},
    'mean_final_regret': {a: round(sum(x['final_regret'] for x in r if x['agent']==a)
                                   /max(1,len([x for x in r if x['agent']==a])),1) for a in agents}
}
with open('metadata/${EXP}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF
  else
    log "  $EXP: FAILED"; FAILED=$(( FAILED + 1 ))
  fi
done

log "GPU phase complete."

# Wait for exp9_bedrock to finish (may already be done)
log "Waiting for exp9_bedrock..."
if wait "$PID9B"; then log "  exp9_bedrock: OK"
else log "  exp9_bedrock: FAILED"; FAILED=$(( FAILED + 1 )); fi

# ── PHASE 3: exp9_local — Llama 4 Scout + weight tracking ─────────────────
log "=== PHASE 3: exp9_local (Llama 4 Scout, weight tracking) ==="

if [[ -z "$HF_TOKEN" ]]; then
  log "Skipping exp9_local: no HF_TOKEN in SSM."
else
  START=$(date +%s)
  mkdir -p results/exp9_local metadata

  # workers=1: model loaded once, reused across all trials
  "$PYTHON" - << 'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch
sys.path.insert(0, 'src')
from combbandits.engine.runner import ExperimentRunner

vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'VRAM: {vram:.0f} GB')
if vram < 14:
    print(f'WARNING: only {vram:.0f} GB VRAM, 4-bit Llama 4 Scout needs ~14 GB')

runner = ExperimentRunner('configs/experiments/exp9_local.yaml', 'results/exp9_local')
runner.run(max_workers=1)
print('exp9_local complete')
PYEOF

  if [[ $? -eq 0 ]]; then
    ELAPSED=$(( $(date +%s) - START ))
    log "✓  exp9_local done in ${ELAPSED}s"
    "$PYTHON" -m combbandits.cli plot \
      results/exp9_local/exp9_local_results.json \
      --output-dir figures/exp9_local 2>&1 | tee -a "$LOG" || true

    # Weight tracking analysis plots
    "$PYTHON" - << 'PYEOF' 2>&1 | tee -a "$LOG"
import sys, sqlite3, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

db_path = Path('metadata/oracle_weights.db')
if not db_path.exists():
    print('No weights DB found'); sys.exit(0)

db     = sqlite3.connect(str(db_path))
calls  = pd.read_sql('SELECT * FROM oracle_calls  ORDER BY call_id',  db)
groups = pd.read_sql('SELECT * FROM query_groups  ORDER BY group_id', db)
db.close()
print(f'Weight tracking: {len(calls)} oracle calls, {len(groups)} query groups')

Path('figures/exp9_local').mkdir(parents=True, exist_ok=True)
primary = calls[calls['query_variant'] == 0].copy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(primary['trial_round'], primary['output_entropy'],     alpha=.7, color='steelblue')
axes[0,0].set(title='Output entropy over rounds', xlabel='Round', ylabel='Entropy')
axes[0,1].plot(primary['trial_round'], primary['attn_on_metadata'],   alpha=.7, color='crimson')
axes[0,1].set(title='Attention on arm metadata', xlabel='Round', ylabel='Attention')
axes[1,0].plot(primary['trial_round'], primary['suggestion_logprob'], alpha=.7, color='green')
axes[1,0].set(title='Log-prob of suggestion', xlabel='Round', ylabel='Σ log p')
axes[1,1].plot(groups['trial_round'], groups['kappa'],         label='κ (output)', color='orange')
axes[1,1].plot(groups['trial_round'], groups['hidden_kl_div'], label='hidden div', color='purple')
axes[1,1].set(title='Output κ vs internal diversity', xlabel='Round'); axes[1,1].legend()
plt.suptitle('Llama 4 Scout Oracle — Weight Tracking', fontsize=13)
plt.tight_layout()
plt.savefig('figures/exp9_local/weight_tracking.png', dpi=150)
plt.close()
print('Saved figures/exp9_local/weight_tracking.png')

# Hidden-state PCA trajectory
hidden_vecs = np.array([json.loads(r) for r in primary['hidden_state_pca']])
if len(hidden_vecs) >= 3:
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(hidden_vecs)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:,0], coords[:,1],
                    c=primary['trial_round'].values, cmap='viridis', alpha=.7, s=20)
    plt.colorbar(sc, label='Bandit round')
    ax.set(title='Hidden state PCA — representation drift',
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig('figures/exp9_local/hidden_state_pca.png', dpi=150)
    plt.close()
    print('Saved figures/exp9_local/hidden_state_pca.png')
PYEOF
  else
    log "  exp9_local: FAILED"; FAILED=$(( FAILED + 1 ))
  fi
fi

# ── Finalize metadata ──────────────────────────────────────────────────────
log "Finalizing metadata..."
cp "$LOG" metadata/full_run.log
"$PYTHON" - << PYEOF 2>&1 | tee -a "$LOG"
import json, glob, os
from pathlib import Path

summaries = {}
for fp in glob.glob('metadata/*_summary.json'):
    name = os.path.basename(fp).replace('_summary.json', '')
    with open(fp) as f:
        summaries[name] = json.load(f)

with open('metadata/run_info.json') as f:
    info = json.load(f)
info['end_timestamp'] = '$(date -u "+%Y-%m-%dT%H:%M:%SZ")'
info['experiment_summaries'] = summaries
info['failed_count'] = $FAILED

# Add weight tracking DB summary if present
db_path = Path('metadata/oracle_weights.db')
if db_path.exists():
    import sqlite3
    db = sqlite3.connect(str(db_path))
    n_calls = db.execute('SELECT COUNT(*) FROM oracle_calls').fetchone()[0]
    n_groups = db.execute('SELECT COUNT(*) FROM query_groups').fetchone()[0]
    db.close()
    info['weight_tracking'] = {'oracle_calls': n_calls, 'query_groups': n_groups}

with open('metadata/run_info.json', 'w') as f:
    json.dump(info, f, indent=2)
print('metadata/run_info.json finalized')
PYEOF

# ── Upload to S3 ───────────────────────────────────────────────────────────
log "Uploading to S3..."
aws s3 sync results/  "s3://$S3_BUCKET/results/"  --region "$REGION"
aws s3 sync figures/  "s3://$S3_BUCKET/figures/"  --region "$REGION"
aws s3 sync metadata/ "s3://$S3_BUCKET/metadata/" --region "$REGION"
log "S3 upload done."

# ── Push to GitHub ─────────────────────────────────────────────────────────
if [[ -n "$GH_TOKEN" ]]; then
  log "Pushing results to GitHub..."
  git config user.email "combbandits-runner@ec2"
  git config user.name  "CombBandits EC2 Runner"
  git remote set-url origin "https://${GH_TOKEN}@github.com/${GH_REPO}.git"
  git add results/ figures/ metadata/ 2>/dev/null || true
  git commit -m "[EC2 $INSTANCE_TYPE] All experiments complete $(date -u '+%Y-%m-%d %H:%M UTC')

Instance: $INSTANCE_TYPE ($INSTANCE_ID)
GPU:      $GPU_NAME
Exps:     exp4 exp5 exp6 exp7 exp8(GPU) exp9_bedrock exp9_local(GPU+weights)
Failed:   $FAILED

See metadata/run_info.json for full provenance." || log "Nothing new to commit"
  git push origin HEAD 2>&1 | tee -a "$LOG" || log "Git push failed — results still in S3"
  log "GitHub push done."
else
  log "No GitHub token — skipping push. Results are in S3."
fi

# ── Self-terminate ─────────────────────────────────────────────────────────
log "All done. Failed experiments: $FAILED"
log "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
