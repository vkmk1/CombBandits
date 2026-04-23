#!/usr/bin/env bash
# Launch Round 5 (T=100000, 100 seeds, 8 agents, 5 scenarios) on a c5.2xlarge.
# CPU-only, ~30-60 min runtime expected.
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
AMI_ID="ami-0c7217cdde317cfec"
INSTANCE_TYPE="c5.2xlarge"  # 8 vCPUs, 16 GB RAM
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a /tmp/combbandits_round5_launch.log; }
log "━━━ Round 5 Launcher (c5.2xlarge) ━━━"

SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=combbandits-sg" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
[[ "$SG_ID" == "None" || -z "$SG_ID" ]] && { log "ERROR: SG not found"; exit 1; }

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-round5},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

log "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/combbandits_round5_instance_id.txt

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
log "IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/combbandits_round5_ip.txt

for i in $(seq 1 40); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  [[ $i -eq 40 ]] && log "SSH timed out" && exit 1
  sleep 5
done

log "Uploading repo..."
rsync -az \
  --exclude='.git' --exclude='results/' --exclude='figures/' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  --exclude='cache/' --exclude='metadata/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/" "ubuntu@$PUBLIC_IP:~/CombBandits/"

log "Starting Round 5 experiment..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~/CombBandits
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv awscli > /dev/null 2>&1
python3 -m venv .venv
source .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pyyaml

nohup bash -c '
  cd ~/CombBandits
  source .venv/bin/activate
  python3 cluster/round5_experiment.py 2>&1 | tee ~/round5.log

  echo "=== Round 5 complete. Results: ==="
  cat results/round5/round5_results.json | python3 -m json.tool || true
  echo "=== Self-terminating ==="
  TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  INST_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
  aws ec2 terminate-instances --instance-ids "$INST_ID" --region us-east-1 || true
' > ~/round5.log 2>&1 &

echo "Round 5 running in background"
REMOTE_SCRIPT

log ""
log "━━━ ROUND 5 LAUNCHED ━━━"
log "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
log "  Monitor:  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'tail -f ~/round5.log'"
log "  Self-terminates when done."
