# Bootstrap for the remote backend (run ONCE, uses LOCAL state).
#
# Chicken-and-egg: the S3 bucket + DynamoDB lock table must exist before the main
# config's `terraform init` can use them. This tiny config creates them with local
# state, then the main config in ../ uses them as its backend.
#
#   cd terraform/bootstrap
#   terraform init && terraform apply
#   # then `cd .. && terraform init` (Terraform will migrate state to S3)

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project   = "todozee-classifier"
      ManagedBy = "terraform-bootstrap"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "state_bucket_name" {
  description = "Globally-unique S3 bucket name for Terraform state."
  type        = string
}

variable "lock_table_name" {
  description = "DynamoDB table name for state locking."
  type        = string
  default     = "todozee-tf-locks"
}

# --- S3 bucket for state ---
resource "aws_s3_bucket" "state" {
  bucket = var.state_bucket_name

  # State is precious — block accidental destroy.
  lifecycle {
    prevent_destroy = true
  }
}

# Keep history of every state write (recover from a bad apply).
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- DynamoDB table for state locking ---
# LockID (String) hash key is the schema Terraform's S3 backend expects.
resource "aws_dynamodb_table" "locks" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

output "state_bucket" {
  value = aws_s3_bucket.state.id
}

output "lock_table" {
  value = aws_dynamodb_table.locks.name
}
