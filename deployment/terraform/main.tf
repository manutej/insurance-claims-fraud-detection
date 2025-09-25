terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    bucket         = "insurance-claims-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment   = var.environment
      Project       = "insurance-claims"
      ManagedBy     = "terraform"
      Owner         = "insurance-claims-team"
      CostCenter    = "fraud-detection"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name_prefix = "insurance-claims-${var.environment}"

  common_tags = {
    Environment = var.environment
    Project     = "insurance-claims"
    Application = "fraud-detection"
  }

  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  database_subnets = var.database_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "staging" ? true : false
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # Database subnet group
  create_database_subnet_group = true
  database_subnet_group_name   = "${local.name_prefix}-db-subnet-group"

  tags = local.common_tags
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name        = "${local.name_prefix}-eks-cluster"
  description = "Security group for EKS cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "HTTPS traffic from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-eks-cluster-sg"
  })
}

resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
    description     = "PostgreSQL from EKS cluster"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

resource "aws_security_group" "elasticache" {
  name        = "${local.name_prefix}-elasticache"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
    description     = "Redis from EKS cluster"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-elasticache-sg"
  })
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.20"

  cluster_name                   = "${local.name_prefix}-cluster"
  cluster_version               = var.kubernetes_version
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  # Cluster security group
  cluster_additional_security_group_ids = [aws_security_group.eks_cluster.id]

  # Node groups
  eks_managed_node_groups = {
    main = {
      name           = "${local.name_prefix}-main"
      instance_types = var.node_instance_types

      min_size     = var.node_min_size
      max_size     = var.node_max_size
      desired_size = var.node_desired_size

      ami_type               = "AL2_x86_64"
      capacity_type          = "ON_DEMAND"
      disk_size              = 50
      force_update_version   = false

      subnet_ids = module.vpc.private_subnets

      # Launch template
      create_launch_template = true
      launch_template_name   = "${local.name_prefix}-main"
      launch_template_use_name_prefix = true
      launch_template_version = "$Latest"

      remote_access = {
        ec2_ssh_key = var.ec2_key_name
      }

      labels = {
        role = "main"
      }

      taints = {}

      tags = local.common_tags
    }

    spot = {
      name           = "${local.name_prefix}-spot"
      instance_types = var.spot_instance_types

      min_size     = 0
      max_size     = var.spot_max_size
      desired_size = var.spot_desired_size

      ami_type      = "AL2_x86_64"
      capacity_type = "SPOT"
      disk_size     = 50

      subnet_ids = module.vpc.private_subnets

      create_launch_template = true
      launch_template_name   = "${local.name_prefix}-spot"
      launch_template_use_name_prefix = true
      launch_template_version = "$Latest"

      labels = {
        role = "spot"
        "node.kubernetes.io/instance-type" = "spot"
      }

      taints = {
        spot = {
          key    = "spot"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${local.name_prefix}-cluster" = "owned"
      })
    }
  }

  # Cluster access
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "eks-admin"
      groups   = ["system:masters"]
    },
  ]

  aws_auth_users = var.eks_admin_users

  tags = local.common_tags
}

# IAM Role for EKS administration
resource "aws_iam_role" "eks_admin" {
  name = "${local.name_prefix}-eks-admin"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_admin" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_admin.name
}

# RDS PostgreSQL
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_db_instance" "postgresql" {
  identifier = "${local.name_prefix}-postgresql"

  engine         = "postgres"
  engine_version = var.postgresql_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  deletion_protection = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment == "production" ? false : true
  final_snapshot_identifier = var.environment == "production" ? "${local.name_prefix}-final-snapshot" : null

  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-postgresql"
  })
}

resource "aws_iam_role" "rds_monitoring" {
  name = "${local.name_prefix}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name_prefix}-cache-subnet"
  subnet_ids = module.vpc.private_subnets

  tags = local.common_tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for Insurance Claims"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = var.redis_parameter_group

  num_cache_clusters         = var.redis_num_cache_nodes
  automatic_failover_enabled = var.redis_num_cache_nodes > 1
  multi_az_enabled          = var.environment == "production"

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.elasticache.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window         = "03:00-05:00"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis"
  })
}

# S3 Buckets
resource "aws_s3_bucket" "models" {
  bucket = "${local.name_prefix}-models"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-models"
    Purpose = "ML model storage"
  })
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "models" {
  bucket = aws_s3_bucket.models.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "data" {
  bucket = "${local.name_prefix}-data"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-data"
    Purpose = "Application data storage"
  })
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data" {
  bucket = aws_s3_bucket.data.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb"
    enabled = true
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb"
  })
}

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb"
  description = "Security group for ALB"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb-sg"
  })
}

resource "aws_s3_bucket" "alb_logs" {
  bucket = "${local.name_prefix}-alb-logs"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb-logs"
    Purpose = "ALB access logs"
  })
}

resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    id     = "delete_old_logs"
    status = "Enabled"

    expiration {
      days = 90
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${local.name_prefix}/application"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-application-logs"
  })
}

resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${local.name_prefix}/cluster"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cluster-logs"
  })
}