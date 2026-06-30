# Remote state backend — S3 (storage) + DynamoDB (locking & consistency).
#
# The bucket and table are created by ./bootstrap (run that first). Backend blocks
# cannot use variables, so fill these in directly (or pass via `-backend-config`).
#
#   terraform init \
#     -backend-config="bucket=todozee-tfstate-CHANGE-ME-12345" \
#     -backend-config="dynamodb_table=todozee-tf-locks"
#
# DynamoDB gives Terraform a lock (conditional write on LockID) so two applies
# can't corrupt the state file; it also stores a state digest for consistency.
terraform {
  backend "s3" {
    bucket         = "todozee-tfstate-637560253183" # created by ./bootstrap
    key            = "todozee/classifier.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "todozee-tf-locks"
    encrypt        = true
  }
}
