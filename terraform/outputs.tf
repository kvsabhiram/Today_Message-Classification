output "ecr_repository_url" {
  description = "ECR repo URL (the workflow derives this automatically too)."
  value       = aws_ecr_repository.app.repository_url
}

output "instance_id" {
  description = "Set this as the GitHub secret EC2_INSTANCE_ID."
  value       = aws_instance.app.id
}

output "instance_public_ip" {
  description = "Hit the API at http://<this>:<app_port>/health"
  value       = aws_instance.app.public_ip
}

output "app_url" {
  description = "Base URL of the classifier API."
  value       = "http://${aws_instance.app.public_ip}:${var.app_port}"
}

output "github_actions_role_arn" {
  description = "Set this as the GitHub secret AWS_ROLE_ARN (OIDC role to assume)."
  value       = aws_iam_role.github_actions.arn
}

output "github_actions_plan_role_arn" {
  description = "Set this as the GitHub secret AWS_PLAN_ROLE_ARN (read-only PR plan role)."
  value       = aws_iam_role.github_actions_plan.arn
}
