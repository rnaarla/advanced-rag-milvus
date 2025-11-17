output "ecr_repository_url" {
  value       = aws_ecr_repository.rag.repository_url
  description = "ECR repository URL"
}

output "alb_dns_name" {
  value       = aws_lb.rag.dns_name
  description = "Public ALB DNS name for the API"
}


