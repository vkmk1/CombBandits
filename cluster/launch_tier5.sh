#!/usr/bin/env bash
# Launch Tier 5 Extended Experiment on EC2 c5.4xlarge.
# T=25000 (10x original), 7 algos incl. RandomKernel ablation.
# CPU-only (API-latency bound). Uses gpt-5.4 via OpenAI.
# Self-terminates when done, syncs results to S3.
#
# Usage: bash cluster/launch_tier5.sh
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
INSTANCE_TYPE="c5.4xlarge"  # 16 vCPUs, 32GB RAM, ~$0.68/hr
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

S3_BUCKET="combbandits-results-099841456154"

# Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
             "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text --region "$REGION")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Tier 5 Extended Experiment Launcher"
echo "  Instance: $INSTANCE_TYPE (16 vCPU, 32GB)"
echo "  Model:    gpt-5.4 via OpenAI"
echo "  Params:   T=25000, 16 configs, 20 seeds, 7 algos, 16 workers"
echo '  Est cost: ~$70 (LLM) + ~$5 (EC2 ~6hr)'
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Security group (reuse existing) ────────────────────────────────────────
SG_NAME="combbandits-sg"
SG_ID=$(aws ec2 describe-security-groups \
  --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")
if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  echo "Creating security group..."
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text --region "$REGION")
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" --description "CombBandits SSH" \
    --vpc-id "$VPC_ID" --query "GroupId" --output text --region "$REGION")
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION"
fi
echo "Security group: $SG_ID"

# ── Launch instance ────────────────────────────────────────────────────────
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-tier5},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

echo "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/tier5_instance.txt

# ── Wait for ready ─────────────────────────────────────────────────────────
echo "Waiting for instance running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Public IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/tier5_ip.txt

echo "Waiting for SSH..."
for i in $(seq 1 60); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  [[ $i -eq 60 ]] && echo "SSH timed out" && exit 1
  sleep 5
done
echo "SSH ready."

# ── Upload zubayer_agi/ ────────────────────────────────────────────────────
echo "Uploading code..."
rsync -az \
  --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='cache/*.sqlite*' --exclude='results/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/zubayer_agi/" "ubuntu@$PUBLIC_IP:~/zubayer_agi/"
echo "Code uploaded."

# ── Read OpenAI key (fallback to env var) ─────────────────────────────────
OPENAI_KEY="${OPENAI_API_KEY:-}"
if [[ -z "$OPENAI_KEY" ]]; then
  OPENAI_KEY=$(aws ssm get-parameter \
    --name "/combbandits/openai_key" --with-decryption \
    --query "Parameter.Value" --output text --region "$REGION" 2>/dev/null || echo "")
fi
if [[ -z "$OPENAI_KEY" ]]; then
  echo "ERROR: OPENAI_API_KEY not set in env and not in SSM. Aborting."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
  exit 1
fi

# ── Write remote script ────────────────────────────────────────────────────
cat > /tmp/tier5_remote.sh << REMOTE
#!/usr/bin/env bash
set -euo pipefail

S3_BUCKET="$S3_BUCKET"
REGION="$REGION"
OPENAI_API_KEY="$OPENAI_KEY"
INSTANCE_ID="$INSTANCE_ID"
export OPENAI_API_KEY

LOG="\$HOME/tier5.log"
cd "\$HOME/zubayer_agi"

log() { echo "[\$(date '+%H:%M:%S')] \$*" | tee -a "\$LOG"; }

log "=== Tier 5 Extended Experiment ==="
log "Instance: \$INSTANCE_ID"
log "T=25000, 7 algos, 16 configs, 20 seeds = 2240 trials"

# Install deps
log "Installing deps..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip > /dev/null 2>&1
python3.11 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q openai numpy pandas scipy
log "Deps installed."

# Run experiment
log "Starting tier5_extended.py (T=25000)..."
venv/bin/python tier5_extended.py --T 25000 --n-seeds 20 --workers 16 --keep-cache 2>&1 | tee -a "\$LOG"

log "Experiment complete."

# Sync results to S3
log "Syncing results to S3..."
RESULTS_DIR=\$(ls -td results/tier5_extended_* | head -1)
aws s3 sync "\$RESULTS_DIR" "s3://\$S3_BUCKET/tier5_extended/\$(basename \$RESULTS_DIR)/" \
  --region "\$REGION" 2>&1 | tee -a "\$LOG"

# Also upload the log
aws s3 cp "\$LOG" "s3://\$S3_BUCKET/tier5_extended/\$(basename \$RESULTS_DIR)/tier5.log" \
  --region "\$REGION" 2>&1 || true

log "S3 sync done."

# Self-terminate
log "Terminating instance..."
aws ec2 terminate-instances --instance-ids "\$INSTANCE_ID" --region "\$REGION"
REMOTE

scp -o StrictHostKeyChecking=no -i "$KEY_PATH" /tmp/tier5_remote.sh "ubuntu@$PUBLIC_IP:~/run_tier5.sh"

# ── Kick off remote run ────────────────────────────────────────────────────
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" \
  "nohup bash ~/run_tier5.sh > ~/tier5.log 2>&1 &"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
echo ""
echo "  Monitor:"
echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'tail -f ~/tier5.log'"
echo ""
echo "  Results will appear at:"
echo "  s3://$S3_BUCKET/tier5_extended/"
echo ""
echo "  Download results:"
echo "  aws s3 sync s3://$S3_BUCKET/tier5_extended/ ~/CombBandits/zubayer_agi/results/tier5/"
echo ""
echo "  Instance SELF-TERMINATES when done (~6 hours)."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
