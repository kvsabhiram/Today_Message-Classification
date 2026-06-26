#!/usr/bin/env bash
# EC2 user-data / one-time bootstrap for a g5.xlarge (Ubuntu 22.04) GPU host.
# Installs NVIDIA driver, Docker, the NVIDIA container toolkit, and the SSM agent
# so the GitHub Actions pipeline can deploy via SSM. Run as root.
set -euxo pipefail

# --- NVIDIA driver (skip if using the AWS Deep Learning AMI, which ships it) ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ubuntu-drivers-common
  ubuntu-drivers install --gpgpu
fi

# --- Docker ---
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
systemctl enable --now docker

# --- NVIDIA Container Toolkit (lets Docker use --gpus all) ---
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# --- AWS CLI (for ECR login inside SSM commands) ---
if ! command -v aws >/dev/null 2>&1; then
  apt-get install -y unzip
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
fi

# --- SSM agent (preinstalled on Ubuntu AWS AMIs; ensure it runs) ---
snap install amazon-ssm-agent --classic 2>/dev/null || true
systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || \
  systemctl enable --now amazon-ssm-agent 2>/dev/null || true

# --- Sanity check ---
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
echo "Bootstrap complete."
