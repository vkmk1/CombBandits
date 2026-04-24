#!/usr/bin/env bash
# Periodically pull raw_trials.jsonl from EC2, convert to dashboard format, push to S3.
# Usage: bash cluster/sync_tier5_dashboard.sh [IP]
#
# Runs in a loop (every 30s) until the instance terminates or you Ctrl-C.
set -euo pipefail

REGION="us-east-1"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
S3_BUCKET="combbandits-results-099841456154"
POLL_INTERVAL=30

IP="${1:-$(cat /tmp/tier5_ip.txt 2>/dev/null || echo "")}"
if [[ -z "$IP" ]]; then
  echo "Usage: $0 <IP>  (or put IP in /tmp/tier5_ip.txt)"
  exit 1
fi

echo "Syncing dashboard data from ubuntu@$IP every ${POLL_INTERVAL}s"
echo "Dashboard URL: https://${S3_BUCKET}.s3.amazonaws.com/dashboard-live/index.html"
echo "Press Ctrl-C to stop."
echo ""

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

while true; do
  # Try to SCP the raw_trials.jsonl
  scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 -q \
    -i "$KEY_PATH" \
    "ubuntu@$IP:~/zubayer_agi/results/tier5_extended_*/raw_trials.jsonl" \
    "$TMPDIR/raw_trials.jsonl" 2>/dev/null || {
    echo "[$(date '+%H:%M:%S')] Instance unreachable or no data yet. Retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    continue
  }

  LINES=$(wc -l < "$TMPDIR/raw_trials.jsonl" | tr -d ' ')
  echo "[$(date '+%H:%M:%S')] Got $LINES trials. Converting..."

  # Convert to dashboard JSON
  python3 - "$TMPDIR/raw_trials.jsonl" "$TMPDIR/results.json" << 'PYEOF'
import json, sys, time
from collections import defaultdict

raw_path = sys.argv[1]
out_path = sys.argv[2]

trials = []
with open(raw_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        trials.append(json.loads(line))

if not trials:
    sys.exit(0)

t0 = trials[0]
total_algos = 7
total_configs = 16
total_seeds = 20
total_trials = total_algos * total_configs * total_seeds  # 2240

# Estimate cost from LLM tokens
total_tokens = sum(t.get("llm_tokens", 0) for t in trials)
est_cost = total_tokens * 3e-6  # rough $/token for gpt-5.4

# Build results array in dashboard format
results = []
for t in trials:
    results.append({
        "agent": t["algo"],
        "config_id": t["config_id"],
        "gap_type": t.get("gap_type", "unknown"),
        "seed": t["seed"],
        "d": t["d"],
        "m": t["m"],
        "final_regret": t["final_regret"],
        "regret_curve": t.get("regret_curve", []),
        "oracle_mean_overlap": None,
        "oracle_perfect_rate": None,
        "elapsed_sec": t.get("elapsed_sec", 0),
    })

# Find min/max elapsed to estimate start time
elapsed_list = [t.get("elapsed_sec", 0) for t in trials]

data = {
    "experiment": {
        "model": t0.get("model", "gpt-5.4"),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - max(elapsed_list, default=0))),
        "est_cost_usd": round(est_cost, 2),
        "total_trials": total_trials,
        "completed": len(trials),
        "d": "25/50",
        "m": "3/5",
        "T": t0.get("T", 25000),
        "n_configs": total_configs,
        "n_seeds": total_seeds,
    },
    "results": results,
}

with open(out_path, "w") as f:
    json.dump(data, f)

print(f"  {len(trials)}/{total_trials} trials, est cost ${est_cost:.2f}")
PYEOF

  # Upload to S3
  aws s3 cp "$TMPDIR/results.json" "s3://$S3_BUCKET/live/results.json" \
    --region "$REGION" --content-type "application/json" \
    --cache-control "no-cache, no-store, must-revalidate" \
    --quiet 2>/dev/null

  echo "[$(date '+%H:%M:%S')] Uploaded to S3. ${LINES} trials live."
  sleep "$POLL_INTERVAL"
done
