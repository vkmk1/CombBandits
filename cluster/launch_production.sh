#!/usr/bin/env bash
# Launch production real-LLM experiment on EC2 c5.4xlarge (16 vCPUs, 32GB).
# CPU-only — we're API-latency bound, no GPU needed.
# Self-terminates when done, pushes results to S3 + GitHub.
#
# Usage: bash cluster/launch_production.sh
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
# Ubuntu 22.04 LTS (no GPU needed)
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
             "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text --region "$REGION")
INSTANCE_TYPE="c5.4xlarge"  # 16 vCPUs, 32GB RAM, ~$0.68/hr
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

S3_BUCKET="combbandits-results-099841456154"
GH_REPO="vkmk1/CombBandits"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Production Real-LLM Experiment Launcher"
echo "  Instance: $INSTANCE_TYPE (16 vCPU, 32GB)"
echo "  Model:    Haiku 4.5 on Bedrock"
echo "  Params:   T=2000, 20 configs, 20 seeds, 16 workers"
echo '  Est cost: ~$18 (LLM) + ~$1 (EC2 ~1.5hr)'
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Security group ─────────────────────────────────────────────────────────
SG_NAME="combbandits-sg"
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  echo "Creating security group..."
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text)
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" --description "CombBandits SSH" \
    --vpc-id "$VPC_ID" --query "GroupId" --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
fi
echo "Security group: $SG_ID"

# ── Launch ─────────────────────────────────────────────────────────────────
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-production-llm},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

echo "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/combbandits_prod_instance.txt

# ── Wait for SSH ───────────────────────────────────────────────────────────
echo "Waiting for instance..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Public IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/combbandits_prod_ip.txt

echo "Waiting for SSH..."
for i in $(seq 1 60); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  [[ $i -eq 60 ]] && echo "SSH timed out" && exit 1
  sleep 5
done
echo "SSH ready."

# ── Upload repo ────────────────────────────────────────────────────────────
echo "Uploading repo..."
rsync -az \
  --exclude='.git' --exclude='results/' --exclude='figures/' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  --exclude='cache/' --exclude='metadata/' --exclude='arena_results/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/" "ubuntu@$PUBLIC_IP:~/CombBandits/"
echo "Repo uploaded."

# ── Run experiment (detached) ──────────────────────────────────────────────
GH_TOKEN=$(aws ssm get-parameter \
  --name "/combbandits/github_token" --with-decryption \
  --query "Parameter.Value" --output text 2>/dev/null) || GH_TOKEN=""

# Write remote script with variables baked in
cat > /tmp/combbandits_remote.sh << REMOTE_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

S3_BUCKET="$S3_BUCKET"
REGION="$REGION"
GH_TOKEN="$GH_TOKEN"
GH_REPO="$GH_REPO"
INSTANCE_ID="$INSTANCE_ID"
INSTANCE_TYPE="$INSTANCE_TYPE"

LOG="\$HOME/production.log"
REPO="\$HOME/CombBandits"

log() { echo "[\$(date '+%H:%M:%S')] \$*" | tee -a "\$LOG"; }

log "=== Production Real-LLM Experiment ==="
log "Instance: \$INSTANCE_ID (\$INSTANCE_TYPE)"

cd "\$REPO"

# Install
log "Installing dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip > /dev/null 2>&1
python3 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q -e ".[dev]" 2>/dev/null || ~/venv/bin/pip install -q torch numpy pandas boto3
log "Install complete."

PYTHON="\$HOME/venv/bin/python"

# Run
log "Starting production experiment..."
\$PYTHON cluster/production_real_llm.py \\
  --T 2000 \\
  --n-configs 20 \\
  --n-seeds 20 \\
  --workers 16 \\
  --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
  --region us-east-1 \\
  --master-seed 2024 2>&1 | tee -a "\$LOG"

log "Experiment complete."

# Upload to S3
log "Uploading to S3..."
aws s3 sync arena_results/ "s3://\$S3_BUCKET/arena_results/" --region "\$REGION"
log "S3 upload done."

# Push to GitHub
if [[ -n "\$GH_TOKEN" ]]; then
  log "Pushing to GitHub..."
  git config user.email "combbandits-runner@ec2"
  git config user.name "CombBandits EC2 Runner"
  git remote set-url origin "https://\${GH_TOKEN}@github.com/\${GH_REPO}.git"
  git add arena_results/ cluster/production_real_llm.py cluster/launch_production.sh 2>/dev/null || true
  git commit -m "[EC2 \$INSTANCE_TYPE] Production real-LLM results (Haiku 4.5, T=2000, 20x20)

Instance: \$INSTANCE_TYPE (\$INSTANCE_ID)
Model: Haiku 4.5 on Bedrock
Params: T=2000, 20 configs, 20 seeds, 16 workers" || log "Nothing to commit"
  git push origin HEAD 2>&1 | tee -a "\$LOG" || log "Git push failed — results in S3"
fi

# Self-terminate
log "Terminating instance..."
aws ec2 terminate-instances --instance-ids "\$INSTANCE_ID" --region "\$REGION"
REMOTE_SCRIPT

scp -o StrictHostKeyChecking=no -i "$KEY_PATH" /tmp/combbandits_remote.sh "ubuntu@$PUBLIC_IP:~/run_production.sh"
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" \
  "nohup bash ~/run_production.sh > ~/production.log 2>&1 &"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
echo ""
echo "  Monitor:"
echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'tail -f ~/production.log'"
echo ""
echo "  Instance SELF-TERMINATES when done."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
