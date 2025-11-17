variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project/name prefix"
  type        = string
  default     = "advanced-rag"
}

variable "image_tag" {
  description = "Container image tag to deploy (e.g., sha or version)"
  type        = string
}

variable "cpu" {
  description = "Fargate task CPU units"
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory (MB)"
  type        = number
  default     = 2048
}


