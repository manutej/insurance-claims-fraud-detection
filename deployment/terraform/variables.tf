# variables.tf - Terraform variables for Insurance Claims Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "database_subnets" {
  description = "Database subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 10
}

variable "node_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 3
}

variable "spot_instance_types" {
  description = "Instance types for spot node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large", "t3.xlarge"]
}

variable "spot_max_size" {
  description = "Maximum number of spot nodes"
  type        = number
  default     = 20
}

variable "spot_desired_size" {
  description = "Desired number of spot nodes"
  type        = number
  default     = 0
}

variable "ec2_key_name" {
  description = "EC2 Key Pair name for SSH access to nodes"
  type        = string
  default     = null
}

variable "eks_admin_users" {
  description = "List of users with admin access to EKS cluster"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS (GB)"
  type        = number
  default     = 100
}

variable "postgresql_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "claims_db"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "claims_user"
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
  default     = 1
}

variable "redis_parameter_group" {
  description = "Parameter group for Redis"
  type        = string
  default     = "default.redis7"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 14
}

# Security Configuration
variable "enable_waf" {
  description = "Enable AWS WAF for ALB"
  type        = bool
  default     = false
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Backup Configuration
variable "backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for RDS"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "schedule_stop_start" {
  description = "Schedule for stopping/starting resources in non-prod environments"
  type = object({
    start_time = string
    stop_time  = string
    timezone   = string
  })
  default = {
    start_time = "08:00"
    stop_time  = "18:00"
    timezone   = "UTC"
  }
}

# Tagging
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Feature Flags
variable "enable_insights" {
  description = "Enable performance insights and enhanced monitoring"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption at rest for all supported services"
  type        = bool
  default     = true
}

variable "enable_multi_az" {
  description = "Enable Multi-AZ deployment for RDS"
  type        = bool
  default     = false
}

# Application Configuration
variable "app_image_tag" {
  description = "Docker image tag for the application"
  type        = string
  default     = "latest"
}

variable "app_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 2
}

variable "worker_replicas" {
  description = "Number of worker replicas"
  type        = number
  default     = 1
}

# Domain Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backups"
  type        = string
  default     = "us-west-2"
}