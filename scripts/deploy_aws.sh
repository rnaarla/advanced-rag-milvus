#!/usr/bin/env bash
set -euo pipefail

# Prereqs: awscli, terraform, docker logged in with AWS credentials configured
# Usage: scripts/deploy_aws.sh <aws-region> <image-tag>

AWS_REGION="${1:-us-east-1}"
IMAGE_TAG="${2:-latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$ROOT_DIR/infra/terraform/aws"

export AWS_DEFAULT_REGION="$AWS_REGION"

echo "Initializing Terraform..."
pushd "$TF_DIR" >/dev/null
terraform init -upgrade

echo "Creating ECR repository (if not exists)..."
terraform apply -auto-approve -target=aws_ecr_repository.rag -var "aws_region=$AWS_REGION" -var "image_tag=$IMAGE_TAG"
ECR_URL="$(terraform output -raw ecr_repository_url)"
popd >/dev/null

echo "Building Docker image..."
docker build -t "$ECR_URL:$IMAGE_TAG" "$ROOT_DIR"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$(echo "$ECR_URL" | cut -d/ -f1)"

echo "Pushing image to ECR..."
docker push "$ECR_URL:$IMAGE_TAG"

echo "Applying full Terraform with image tag..."
pushd "$TF_DIR" >/dev/null
terraform apply -auto-approve -var "aws_region=$AWS_REGION" -var "image_tag=$IMAGE_TAG"
ALB_DNS="$(terraform output -raw alb_dns_name)"
popd >/dev/null

echo "Deployment successful!"
echo "API Endpoint: http://$ALB_DNS/healthz"
echo "Chat Stream:  http://$ALB_DNS/chat/stream?session_id=<sid>&q=Hello"


