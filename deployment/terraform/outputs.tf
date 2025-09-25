# outputs.tf - Terraform outputs for Insurance Claims Infrastructure

# Network Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "Database subnet IDs"
  value       = module.vpc.database_subnets
}

# EKS Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = module.eks.cluster_primary_security_group_id
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

# Database Outputs
output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.postgresql.id
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.postgresql.arn
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgresql.endpoint
  sensitive   = true
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.postgresql.port
}

output "db_subnet_group_id" {
  description = "Database subnet group ID"
  value       = aws_db_subnet_group.main.id
}

output "db_parameter_group_id" {
  description = "Database parameter group ID"
  value       = aws_db_instance.postgresql.parameter_group_name
}

output "db_security_group_id" {
  description = "Database security group ID"
  value       = aws_security_group.rds.id
}

# Redis Outputs
output "redis_cluster_id" {
  description = "ElastiCache Redis cluster ID"
  value       = aws_elasticache_replication_group.redis.replication_group_id
}

output "redis_cluster_arn" {
  description = "ElastiCache Redis cluster ARN"
  value       = aws_elasticache_replication_group.redis.arn
}

output "redis_primary_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = aws_elasticache_replication_group.redis.configuration_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.elasticache.id
}

# S3 Outputs
output "s3_models_bucket_id" {
  description = "S3 bucket ID for models"
  value       = aws_s3_bucket.models.id
}

output "s3_models_bucket_arn" {
  description = "S3 bucket ARN for models"
  value       = aws_s3_bucket.models.arn
}

output "s3_data_bucket_id" {
  description = "S3 bucket ID for data"
  value       = aws_s3_bucket.data.id
}

output "s3_data_bucket_arn" {
  description = "S3 bucket ARN for data"
  value       = aws_s3_bucket.data.arn
}

output "s3_alb_logs_bucket_id" {
  description = "S3 bucket ID for ALB logs"
  value       = aws_s3_bucket.alb_logs.id
}

# Load Balancer Outputs
output "alb_id" {
  description = "Application Load Balancer ID"
  value       = aws_lb.main.id
}

output "alb_arn" {
  description = "Application Load Balancer ARN"
  value       = aws_lb.main.arn
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Application Load Balancer canonical hosted zone ID"
  value       = aws_lb.main.zone_id
}

output "alb_security_group_id" {
  description = "ALB security group ID"
  value       = aws_security_group.alb.id
}

# CloudWatch Outputs
output "cloudwatch_log_group_application_name" {
  description = "CloudWatch log group name for application logs"
  value       = aws_cloudwatch_log_group.application.name
}

output "cloudwatch_log_group_cluster_name" {
  description = "CloudWatch log group name for cluster logs"
  value       = aws_cloudwatch_log_group.cluster.name
}

# Security Outputs
output "eks_admin_role_arn" {
  description = "EKS admin role ARN"
  value       = aws_iam_role.eks_admin.arn
}

# Connection Information
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_id}"
}

output "database_connection_string" {
  description = "Database connection string (without password)"
  value       = "postgresql://${var.db_username}:PASSWORD@${aws_db_instance.postgresql.endpoint}/${var.db_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${aws_elasticache_replication_group.redis.configuration_endpoint_address}:${aws_elasticache_replication_group.redis.port}"
  sensitive   = true
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "availability_zones" {
  description = "Availability zones used"
  value       = local.azs
}

# Cost Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    eks_cluster    = "~$72/month (cluster) + nodes"
    rds_postgres   = "~$15-200/month depending on instance size"
    elasticache    = "~$15-100/month depending on node type"
    alb           = "~$22/month + data processing"
    nat_gateway   = "~$45/month per AZ"
    data_transfer = "Variable based on usage"
    storage       = "Variable based on usage"
  }
}

# Monitoring URLs (if applicable)
output "monitoring_urls" {
  description = "Monitoring and management URLs"
  value = {
    aws_console = "https://console.aws.amazon.com/"
    eks_console = "https://console.aws.amazon.com/eks/home?region=${var.aws_region}#/clusters/${module.eks.cluster_id}"
    rds_console = "https://console.aws.amazon.com/rds/home?region=${var.aws_region}#database:id=${aws_db_instance.postgresql.id}"
    cloudwatch  = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}"
  }
}

# Secrets Manager Integration
output "secrets_to_create" {
  description = "Secrets that should be created in AWS Secrets Manager or Kubernetes"
  value = {
    database_password = {
      name        = "insurance-claims/${var.environment}/database-password"
      description = "PostgreSQL database password"
      value       = "PLACEHOLDER - Set actual password in Secrets Manager"
    }
    jwt_secret = {
      name        = "insurance-claims/${var.environment}/jwt-secret"
      description = "JWT signing secret"
      value       = "PLACEHOLDER - Generate random JWT secret"
    }
    api_keys = {
      name        = "insurance-claims/${var.environment}/api-keys"
      description = "External API keys"
      value       = "PLACEHOLDER - Set actual API keys"
    }
  }
  sensitive = true
}