#!/usr/bin/env bash
# Launch CombBandits on EC2 g4dn.2xlarge (1x T4 16GB, 8 vCPUs, 32GB RAM).
# Runs ALL experiments including GPU-batched (exp8) and local Llama 4 Scout (exp9_local).
# Self-terminates and pushes results to S3 + GitHub when done.
#
# Usage: bash cluster/aws_launch.sh
# Prerequisites:
#   ~/.ssh/combbandits-key.pem       — EC2 keypair
#   IAM role princetoncourses-ec2    — Bedrock + S3 + SSM access
#   SSM param /combbandits/github_token  — GitHub PAT
#   SSM param /combbandits/hf_token      — HuggingFace token (for Llama 4 Scout)
#
# Quota required: "Running On-Demand G and VT instances" >= 8 vCPUs
#   Check:   aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA
#   Request: aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8
set -euo pipefail

REGION="us-east-1"
KEY_NAME="combbandits-key"
KEY_PATH="$HOME/.ssh/combbandits-key.pem"
INSTANCE_PROFILE="princetoncourses-ec2"
# Latest Deep Learning Base OSS Nvidia (Ubuntu 22.04) — CUDA 12 + drivers pre-installed
AMI_ID="ami-03d653a715378d2e5"
# g4dn.2xlarge: 8 vCPUs, 32 GB RAM, 1x T4 16 GB, 225 GB NVMe local SSD
INSTANCE_TYPE="g4dn.2xlarge"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CombBandits g4dn.2xlarge Launcher"
echo "  GPU:  NVIDIA T4 16 GB"
echo "  CPUs: 8 vCPUs / 32 GB RAM"
echo "  NVMe: 225 GB local SSD"
echo "  Exps: exp4 exp5 exp6 exp7 exp8(GPU) exp9_bedrock exp9_local(GPU)"
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

# ── Pre-flight: quota check ────────────────────────────────────────────────
GVT_QUOTA=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query "Quota.Value" --output text 2>/dev/null || echo "0")
echo "G/VT on-demand vCPU quota: $GVT_QUOTA (need >= 8)"
if (( $(echo "$GVT_QUOTA < 8" | bc -l) )); then
  echo "ERROR: Insufficient G/VT quota ($GVT_QUOTA vCPUs). Request increase:"
  echo "  aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8"
  exit 1
fi

# ── Launch ─────────────────────────────────────────────────────────────────
# 50 GB gp3 EBS root (repo + venv). Model weights + scratch go on 225 GB NVMe local SSD.
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=combbandits-gpu-runner},{Key=Project,Value=CombBandits}]" \
  --query "Instances[0].InstanceId" --output text)

echo "Launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/combbandits_instance_id.txt

# ── Wait for SSH ───────────────────────────────────────────────────────────
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Public IP: $PUBLIC_IP"
echo "$PUBLIC_IP" > /tmp/combbandits_instance_ip.txt

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
  --exclude='cache/' --exclude='metadata/' \
  -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
  "$REPO_DIR/" "ubuntu@$PUBLIC_IP:~/CombBandits/"
echo "Repo uploaded."

# ── Launch all experiments (detached) ─────────────────────────────────────
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$PUBLIC_IP" \
  "nohup bash ~/CombBandits/cluster/aws_run_experiments.sh > ~/experiment.log 2>&1 &"
echo "Experiments launched in background."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Instance: $INSTANCE_ID  |  IP: $PUBLIC_IP"
echo ""
echo "  Monitor live:"
echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP 'tail -f ~/experiment.log'"
echo ""
echo "  Pull results when done:"
echo "  bash cluster/aws_download.sh"
echo ""
echo "  Instance SELF-TERMINATES when all experiments finish."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
