#!/usr/bin/env bash
# Launch novelty validation experiment on a cheap CPU instance.
# c5.xlarge: 4 vCPUs, 8GB RAM — CPU-only simulated bandits, no GPU needed.
# Self-terminates when done.
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS x86_64
INSTANCE_TYPE="c5.xlarge"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a /tmp/combbandits_validation_launch.log; }

log "━━━ Novelty Validation Launcher (c5.xlarge) ━━━"

# Security group (reuse existing)
SG_NAME="combbandits-sg"
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
[[ "$SG_ID" == "None" || -z "$SG_ID" ]] && {
  log "ERROR: security group not found. Run aws_launch.sh first."
  exit 1
}

# Launch
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-validation},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

log "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/combbandits_validation_instance_id.txt

# Wait for SSH
log "Waiting for instance..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
log "IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/combbandits_validation_ip.txt

for i in $(seq 1 40); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  [[ $i -eq 40 ]] && log "SSH timed out" && exit 1
  sleep 5
done

# Upload repo
log "Uploading repo..."
rsync -az \
  --exclude='.git' --exclude='results/' --exclude='figures/' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  --exclude='cache/' --exclude='metadata/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/" "ubuntu@$PUBLIC_IP:~/CombBandits/"

# Run experiment
log "Starting validation experiment..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~/CombBandits

# Install Python + deps
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
python3 -m venv .venv
source .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pyyaml

# Run (self-terminate when done)
nohup bash -c '
  cd ~/CombBandits
  source .venv/bin/activate
  python3 cluster/validation_experiment.py 2>&1 | tee ~/validation.log

  # Upload results
  echo "Experiment complete. Results:"
  cat results/validation/validation_report.txt

  # Self-terminate
  TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  INST_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
  echo "Self-terminating $INST_ID..."
  aws ec2 terminate-instances --instance-ids "$INST_ID" --region us-east-1 || true
' > ~/validation.log 2>&1 &

echo "Validation running in background (PID $!)"
REMOTE_SCRIPT

log ""
log "━━━ VALIDATION LAUNCHED ━━━"
log "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
log "  Monitor:  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'tail -f ~/validation.log'"
log "  Instance self-terminates when done."
