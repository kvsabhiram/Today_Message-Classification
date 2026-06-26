variable "aws_region" {
  description = "AWS region (Mumbai)."
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Name prefix for resources."
  type        = string
  default     = "todozee"
}

variable "ecr_repo_name" {
  description = "ECR repository name (must match the GitHub workflow's ECR_REPO)."
  type        = string
  default     = "todozee-classifier"
}

variable "instance_type" {
  description = "GPU instance type. g5.xlarge = A10G 24GB, native bf16."
  type        = string
  default     = "g5.xlarge"
}

variable "root_volume_gb" {
  description = "Root EBS size. Image bakes in the base model (~6GB) so keep this generous."
  type        = number
  default     = 80
}

variable "ami_id" {
  description = <<-EOT
    Optional AMI override. Leave empty to use the latest Ubuntu 22.04.
    For zero driver hassle, set this to an AWS Deep Learning Base GPU AMI
    (Ubuntu) ID for ap-south-1 — it ships the NVIDIA driver preinstalled, so
    ec2_bootstrap.sh will skip the driver step.
  EOT
  type        = string
  default     = ""
}

variable "app_port" {
  description = "Port the FastAPI service listens on."
  type        = number
  default     = 5011
}

variable "allowed_cidr" {
  description = "CIDR allowed to reach the app port (and SSH if key set). Use your IP/32 to lock down."
  type        = string
  default     = "0.0.0.0/0"
}

variable "ssh_key_name" {
  description = "Optional EC2 key pair name for SSH. Empty = no SSH (use SSM Session Manager)."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "owner/repo that GitHub Actions OIDC is scoped to."
  type        = string
  default     = "kvsabhiram/Today_Message-Classification"
}

variable "github_oidc_branch" {
  description = "Branch ref the OIDC role trusts for deploys."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = <<-EOT
    Create the GitHub OIDC provider. Set to false if the account already has an
    IAM OIDC provider for token.actions.githubusercontent.com (only one allowed).
  EOT
  type        = bool
  default     = true
}
