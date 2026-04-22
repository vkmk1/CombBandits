#!/usr/bin/env bash
# Launch CombBandits CPU experiments on EC2.
# Instance self-terminates and uploads results to S3 when done.
#
# Usage: bash cluster/aws_launch.sh
#
# Key:  ~/.ssh/combbandits-key.pem   (created by setup)
# Role: princetoncourses-ec2         (has Bedrock + S3 access)
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
AMI_ID="ami-0c1e21d82fe9c9336"     # AL2023 kernel 6.18 us-east-1
INSTANCE_TYPE="c7i.8xlarge"         # 32 vCPUs, 64 GB — all CPU experiments
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CombBandits AWS Launcher"
echo "  Instance type: $INSTANCE_TYPE"
echo "  Experiments:   exp6 exp7 exp4 exp5 exp9_bedrock"
echo "  GPU (exp8):    run separately on Colab"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Security group ────────────────────────────────────────────────────────────
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

# ── Launch ────────────────────────────────────────────────────────────────────
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-runner},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

echo "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/combbandits_instance_id.txt

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Public IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/combbandits_instance_ip.txt

echo "Waiting for SSH..."
for i in $(seq 1 40); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -i "$KEY_PATH" "ec2-user@$PUBLIC_IP" "echo ok" 2>/dev/null && break
  [[ $i -eq 40 ]] && echo "SSH timed out" && exit 1
  sleep 5
done
echo "SSH ready."

# ── Bootstrap: install python3.11 ────────────────────────────────────────────
echo "Bootstrapping instance..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ec2-user@$PUBLIC_IP" \
  "sudo dnf install -y -q git python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make && \
   sudo ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
   sudo ln -sf /usr/bin/pip3.11 /usr/local/bin/pip3 && \
   pip3 install -q uv && echo 'bootstrap done'"

# ── Upload repo ───────────────────────────────────────────────────────────────
echo "Uploading repo..."
rsync -az \
  --exclude='.git' --exclude='results/' --exclude='figures/' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='cache/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/" "ec2-user@$PUBLIC_IP:~/CombBandits/"
echo "Repo uploaded."

# ── Launch experiments (detached) ────────────────────────────────────────────
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ec2-user@$PUBLIC_IP" \
  "nohup bash ~/CombBandits/cluster/aws_run_experiments.sh > ~/experiment.log 2>&1 &"
echo "Experiments launched in background."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
echo ""
echo "  Monitor live:"
echo "  ssh -i $KEY_PATH ec2-user@$PUBLIC_IP 'tail -f ~/experiment.log'"
echo ""
echo "  When done, pull results:"
echo "  bash cluster/aws_download.sh"
echo ""
echo "  The instance SELF-TERMINATES when all experiments finish."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
