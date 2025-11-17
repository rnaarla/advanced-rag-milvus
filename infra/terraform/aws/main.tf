terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.38"
    }
    random = {
      source = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_ecr_repository" "rag" {
  name                 = "${var.project_name}"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_cloudwatch_log_group" "rag" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "rag" {
  name = "${var.project_name}-cluster"
}

resource "aws_iam_role" "task_execution" {
  name = "${var.project_name}-task-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "task_exec_policy" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task role for app + collector
resource "aws_iam_role" "task_role" {
  name = "${var.project_name}-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "xray_write" {
  role       = aws_iam_role.task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess"
}

resource "aws_iam_role_policy_attachment" "prom_remote_write" {
  role       = aws_iam_role.task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess"
}

resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "ALB SG"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "svc_sg" {
  name        = "${var.project_name}-svc-sg"
  description = "Service SG"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "rag" {
  name               = "${var.project_name}-alb"
  load_balancer_type = "application"
  subnets            = data.aws_subnets.default.ids
  security_groups    = [aws_security_group.alb_sg.id]
}

resource "aws_lb_target_group" "rag" {
  name        = "${var.project_name}-tg"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = data.aws_vpc.default.id
  health_check {
    enabled             = true
    interval            = 30
    path                = "/healthz"
    port                = "8000"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.rag.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.rag.arn
  }
}

locals {
  ecr_url = aws_ecr_repository.rag.repository_url
  image   = "${local.ecr_url}:${var.image_tag}"
}

resource "aws_ecs_task_definition" "rag" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([
    {
      name      = "etcd",
      image     = "quay.io/coreos/etcd:v3.5.5",
      essential = true,
      command   = ["etcd", "-advertise-client-urls=http://127.0.0.1:2379", "-listen-client-urls", "http://0.0.0.0:2379", "--data-dir", "/etcd"],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.rag.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "minio",
      image     = "minio/minio:RELEASE.2023-03-20T20-16-18Z",
      essential = true,
      environment = [
        { name = "MINIO_ACCESS_KEY", value = "minioadmin" },
        { name = "MINIO_SECRET_KEY", value = "minioadmin" }
      ],
      portMappings = [{ containerPort = 9000, hostPort = 9000 }],
      command      = ["server", "/data"],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.rag.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "milvus",
      image     = "milvusdb/milvus:v2.3.3",
      essential = true,
      environment = [
        { name = "ETCD_ENDPOINTS", value = "127.0.0.1:2379" },
        { name = "MINIO_ADDRESS",  value = "127.0.0.1:9000" }
      ],
      command = ["milvus", "run", "standalone"],
      portMappings = [{ containerPort = 19530, hostPort = 19530 }],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.rag.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "api",
      image     = local.image,
      essential = true,
      portMappings = [{ containerPort = 8000, hostPort = 8000 }],
      environment = [
        { name = "MILVUS_HOST", value = "127.0.0.1" },
        { name = "MILVUS_PORT", value = "19530" },
        { name = "API_KEY", value = "change-me" },
        { name = "DATABASE_URL", value = "postgresql://${aws_db_instance.rag.username}:${random_password.db.result}@${aws_db_instance.rag.address}:5432/${aws_db_instance.rag.db_name}" }
      ],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.rag.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "aws-otel-collector",
      image     = "public.ecr.aws/aws-observability/aws-otel-collector:latest",
      essential = true,
      environment = [
        { name = "AWS_REGION", value = var.aws_region }
      ],
      command = ["--config=/etc/ecs/ecs-default-config.yaml"],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.rag.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "rag" {
  name            = "${var.project_name}-svc"
  cluster         = aws_ecs_cluster.rag.id
  task_definition = aws_ecs_task_definition.rag.arn
  launch_type     = "FARGATE"
  desired_count   = 1

  network_configuration {
    subnets         = data.aws_subnets.default.ids
    security_groups = [aws_security_group.svc_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.rag.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# RDS Postgres for chat/feedback persistence
resource "aws_db_subnet_group" "rag" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = data.aws_subnets.default.ids
}

resource "random_password" "db" {
  length  = 16
  special = true
}

resource "aws_security_group" "db_sg" {
  name   = "${var.project_name}-db-sg"
  vpc_id = data.aws_vpc.default.id
  ingress {
    protocol        = "tcp"
    from_port       = 5432
    to_port         = 5432
    security_groups = [aws_security_group.svc_sg.id]
  }
  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "rag" {
  identifier                 = "${var.project_name}-db"
  engine                     = "postgres"
  engine_version             = "15"
  instance_class             = "db.t4g.micro"
  allocated_storage          = 20
  db_name                    = "ragdb"
  username                   = "raguser"
  password                   = random_password.db.result
  db_subnet_group_name       = aws_db_subnet_group.rag.name
  vpc_security_group_ids     = [aws_security_group.db_sg.id]
  publicly_accessible        = false
  skip_final_snapshot        = true
  deletion_protection        = false
  backup_retention_period    = 0
  apply_immediately          = true
}

# AMP workspace (optional for remote write)
resource "aws_prometheus_workspace" "rag" {
  alias = "${var.project_name}-amp"
}


