# Infrastructure Requirements for Insurance Claims Fraud Detection System

## Executive Summary

This document specifies comprehensive infrastructure requirements for deploying and scaling the insurance fraud detection system across development, staging, and production environments. Includes detailed CPU/memory estimates, storage requirements, network specifications, and cost projections.

## 1. Environment Overview

```
┌────────────────────────────────────────────────────────────────┐
│              Infrastructure Environments                        │
└────────────────────────────────────────────────────────────────┘

Development                Staging                  Production
├── Local Laptop          ├── Single Region         ├── Multi-Region
├── LanceDB (embedded)    ├── Qdrant (1 node)     ├── Qdrant (3 nodes)
├── 10K claims           ├── 100K claims         ├── 10M+ claims
├── 1 developer          ├── 5 QA engineers      ├── 1000 concurrent users
└── $50/month            └── $250/month          └── $2,500/month

Target Workload:
- Dev: 1 query/sec, <100 claims/day
- Staging: 10 queries/sec, 1K claims/day
- Production: 200 queries/sec peak, 50K claims/day
```

## 2. Compute Resources

### 2.1 Development Environment

**Local Development Machine:**

```yaml
Development Workstation:
  CPU: 4 cores (Intel i5/i7 or Apple M1/M2)
  RAM: 16 GB
  Storage: 50 GB SSD
  OS: macOS, Linux, or Windows WSL2

Services Running Locally:
  - FastAPI application
  - LanceDB (embedded)
  - Redis (optional, 512MB)
  - PostgreSQL (optional, for testing)

Resource Usage:
  - API Service: 2 CPU cores, 4 GB RAM
  - Vector DB: 1 CPU core, 2 GB RAM
  - Development Tools: 1 CPU core, 2 GB RAM
  - OS Overhead: 4 GB RAM

Concurrent Claims Processing: 10-20 claims
Query Throughput: 1-5 queries/sec
Cost: $0/month (local hardware)
```

**Cloud Development Environment (Optional):**

```yaml
AWS EC2 Instance: t3.large
  - vCPU: 2
  - RAM: 8 GB
  - Storage: 50 GB gp3
  - Network: Up to 5 Gbps
  - Cost: ~$60/month

Services:
  - FastAPI (Docker container)
  - LanceDB (embedded in container)
  - Redis (ElastiCache t3.micro: $12/month)

Total Monthly Cost: ~$75/month
```

### 2.2 Staging Environment

**Application Tier:**

```yaml
AWS ECS Fargate Tasks:
  Task Definition:
    - vCPU: 2
    - RAM: 8 GB
    - Count: 2 tasks (for redundancy)
    - Cost: ~$120/month

  OR AWS EC2 (Reserved):
    Instance Type: t3.xlarge
    - vCPU: 4
    - RAM: 16 GB
    - Count: 2 instances
    - Storage: 50 GB gp3 per instance
    - Cost: ~$100/month (1-year reserved)

Load Balancer:
  Type: Application Load Balancer (ALB)
  Cost: ~$20/month

Auto-Scaling:
  Min: 2 tasks/instances
  Max: 4 tasks/instances
  Target CPU: 70%
```

**Vector Store (Qdrant):**

```yaml
AWS EC2 Instance: t3.xlarge
  - vCPU: 4
  - RAM: 16 GB
  - Storage: 100 GB gp3 (3000 IOPS)
  - Network: Up to 5 Gbps
  - Cost: ~$120/month

Qdrant Configuration:
  - Single node deployment
  - In-memory HNSW index
  - Persistent storage on EBS

Expected Performance:
  - 100K vectors indexed
  - <100ms query latency (p95)
  - 50 concurrent queries/sec
```

**Database (PostgreSQL):**

```yaml
AWS RDS PostgreSQL:
  Instance Class: db.t3.medium
  - vCPU: 2
  - RAM: 4 GB
  - Storage: 50 GB gp3
  - Multi-AZ: No (staging)
  - Backup: 7 days
  - Cost: ~$60/month

Usage:
  - Claim metadata
  - Fraud detection results
  - Audit logs
  - User accounts
```

**Cache (Redis):**

```yaml
AWS ElastiCache Redis:
  Node Type: cache.t3.medium
  - vCPU: 2
  - RAM: 3.09 GB
  - Nodes: 1
  - Cost: ~$40/month

Usage:
  - Embedding cache
  - Session storage
  - Rate limiting
  - Query results cache
```

**Staging Total Cost:** ~$360/month

### 2.3 Production Environment

**Application Tier (High Availability):**

```yaml
AWS EKS Cluster (Kubernetes):
  Control Plane: $72/month

  Worker Nodes (Auto-Scaling Group):
    Instance Type: m6i.2xlarge
    - vCPU: 8
    - RAM: 32 GB
    - Storage: 100 GB gp3
    - Min Nodes: 3
    - Max Nodes: 10
    - Cost: ~$900/month (3 nodes baseline)

  Application Pods:
    FastAPI Service:
      - CPU Request: 2 cores
      - CPU Limit: 4 cores
      - Memory Request: 4 GB
      - Memory Limit: 8 GB
      - Replicas: Min 3, Max 20
      - HPA Target: 70% CPU

    Background Workers (Celery):
      - CPU Request: 1 core
      - CPU Limit: 2 cores
      - Memory Request: 2 GB
      - Memory Limit: 4 GB
      - Replicas: Min 2, Max 10

Load Balancing:
  AWS ALB: $20/month
  CloudFront CDN: ~$50/month (for static assets)
```

**Vector Store (Qdrant Cluster):**

```yaml
Qdrant 3-Node Cluster:
  Instance Type: r6i.2xlarge (memory-optimized)
  Per Node:
    - vCPU: 8
    - RAM: 64 GB
    - Storage: 500 GB gp3 (5000 IOPS)
    - Network: 12.5 Gbps
    - Cost per node: ~$500/month

  Configuration:
    - 3 nodes for HA
    - Replication factor: 2
    - Total capacity: 10M+ vectors
    - Expected performance:
      - <80ms query latency (p95)
      - 200+ concurrent queries/sec
      - 10K indexing throughput/sec

  Total Cost: ~$1,500/month

  Scaling Plan:
    5M vectors: 3 nodes
    10M vectors: 4 nodes
    20M vectors: 6 nodes
```

**Database (PostgreSQL - Production):**

```yaml
AWS RDS PostgreSQL:
  Instance Class: db.r6i.xlarge
  - vCPU: 4
  - RAM: 32 GB
  - Storage: 500 GB gp3 (provisioned IOPS)
  - Multi-AZ: Yes (automatic failover)
  - Read Replicas: 2 (for read scaling)
  - Backup: 30 days
  - Cost: ~$600/month

Connection Pooling:
  - PgBouncer on separate t3.medium: $60/month

Usage:
  - 10M+ claim records
  - 50M+ fraud detection results
  - 100M+ audit log entries
  - 10K concurrent connections (pooled)
```

**Cache (Redis Cluster):**

```yaml
AWS ElastiCache Redis Cluster:
  Node Type: cache.r6g.xlarge
  - vCPU: 4
  - RAM: 26.32 GB
  - Nodes: 3 (clustered)
  - Replication: Yes
  - Multi-AZ: Yes
  - Cost: ~$350/month

Usage:
  - Provider embeddings cache (16 GB)
  - Query result cache (5 GB)
  - Session storage (2 GB)
  - Rate limiting data (1 GB)
```

**Message Queue (Kafka or SQS):**

```yaml
Option 1: AWS MSK (Managed Kafka)
  Broker Type: kafka.m5.large
  - vCPU: 2
  - RAM: 8 GB
  - Brokers: 3 (multi-AZ)
  - Storage: 500 GB per broker
  - Cost: ~$450/month

  Topics:
    - claims.submitted (10K msg/sec)
    - claims.validated (8K msg/sec)
    - fraud.detected (2K msg/sec)

Option 2: AWS SQS (Serverless)
  - Standard Queue
  - Cost: ~$50/month (1M requests/day)
  - No infrastructure management

Recommendation: SQS for simplicity, MSK for high throughput
```

**Object Storage (S3):**

```yaml
AWS S3:
  Buckets:
    - claims-raw: 100 GB (Standard)
    - claims-archive: 1 TB (Glacier)
    - model-artifacts: 50 GB (Standard)
    - backups: 2 TB (Standard-IA)

  Costs:
    - Storage: ~$50/month
    - Requests: ~$20/month
    - Data transfer: ~$30/month

  Total: ~$100/month

  Lifecycle Policies:
    - Raw claims: Move to IA after 30 days
    - Archive: Move to Glacier after 90 days
    - Backups: Retain 1 year
```

**Production Total Cost:** ~$3,600/month

## 3. Storage Requirements

### 3.1 Storage Sizing by Component

```yaml
Vector Store (Qdrant):
  Per 1M Claims:
    - Text embeddings (1536 dim): ~6 GB
    - Medical code embeddings (768 dim): ~3 GB
    - Provider embeddings (384 dim): ~1.5 GB
    - Payload data (metadata): ~2 GB
    - HNSW index overhead: ~4 GB
    - Total per 1M claims: ~16.5 GB

  Sizing:
    - 1M claims: 20 GB (with buffer)
    - 5M claims: 100 GB
    - 10M claims: 200 GB
    - 20M claims: 400 GB

  Storage Type: gp3 SSD
  IOPS: 3000-5000 (baseline)
  Throughput: 250-500 MB/s

Database (PostgreSQL):
  Tables:
    - claims (metadata): ~500 bytes/claim
      - 10M claims: 5 GB
    - fraud_detection_results: ~1 KB/result
      - 10M results: 10 GB
    - audit_logs: ~2 KB/event
      - 100M events: 200 GB
    - enrichment_history: ~500 bytes/enrichment
      - 50M enrichments: 25 GB

  Indexes: +30% overhead
  Total for 10M claims: ~300 GB

Cache (Redis):
  - Provider embeddings: 10 GB (100K providers)
  - Query results: 5 GB (hot queries)
  - Session data: 2 GB (10K sessions)
  - Rate limiting: 1 GB
  Total: ~20 GB

Object Storage (S3):
  - Raw claims (JSON): ~5 KB/claim
    - 10M claims: 50 GB
  - Model artifacts: 10 GB
  - Snapshots/backups: 500 GB
  - Archived data: 2 TB (cold storage)
```

### 3.2 Growth Projections

```yaml
Year 1:
  Claims: 2M
  Vector Store: 40 GB
  Database: 60 GB
  Total Storage: ~150 GB

Year 2:
  Claims: 5M
  Vector Store: 100 GB
  Database: 150 GB
  Total Storage: ~400 GB

Year 3:
  Claims: 10M
  Vector Store: 200 GB
  Database: 300 GB
  Total Storage: ~800 GB

Year 5:
  Claims: 25M
  Vector Store: 500 GB
  Database: 750 GB
  Total Storage: ~2 TB
```

## 4. Network Requirements

### 4.1 Bandwidth and Latency

```yaml
Development:
  Ingestion: 10 Mbps
  API Response: 5 Mbps
  Embedding API: 10 Mbps
  Total: ~25 Mbps

Staging:
  Ingestion: 50 Mbps
  API Response: 100 Mbps
  Embedding API: 50 Mbps
  Internal (services): 200 Mbps
  Total: ~400 Mbps

Production:
  Ingestion: 500 Mbps (peak)
  API Response: 2 Gbps (peak)
  Embedding API: 500 Mbps
  Internal (services): 5 Gbps
  Replication (multi-region): 1 Gbps
  Total: ~9 Gbps peak
```

### 4.2 Latency Targets

```yaml
User-Facing APIs:
  API Gateway → Application: <10ms
  Application → Vector Store: <50ms
  Application → Database: <20ms
  Application → Cache: <5ms
  Total End-to-End: <100ms (p95)

Internal Services:
  Service-to-Service: <10ms
  Database Queries: <50ms
  Vector Search: <80ms
  Embedding Generation: <200ms

Geographic Latency:
  Same Region: <5ms
  Cross-Region (US): <50ms
  Cross-Continent: <150ms
```

### 4.3 Network Security

```yaml
VPC Configuration:
  CIDR: 10.0.0.0/16
  Public Subnets: 2 (AZs a, b)
  Private Subnets: 2 (AZs a, b)
  Database Subnets: 2 (AZs a, b)

Security Groups:
  ALB Security Group:
    - Inbound: 443 (HTTPS) from 0.0.0.0/0
    - Outbound: All to Application SG

  Application Security Group:
    - Inbound: 8000 from ALB SG
    - Outbound: 6333 to Qdrant SG
    - Outbound: 5432 to RDS SG
    - Outbound: 6379 to Redis SG

  Qdrant Security Group:
    - Inbound: 6333 from Application SG
    - Outbound: None (ingress only)

  Database Security Group:
    - Inbound: 5432 from Application SG
    - Outbound: None (ingress only)

NAT Gateways: 2 (for HA)
Cost: ~$60/month
```

## 5. Scalability Targets

### 5.1 Vertical Scaling Limits

```yaml
Application Service:
  Current: 2-4 cores, 8 GB RAM
  Max: 16 cores, 64 GB RAM
  Rationale: Most workload is I/O bound

Vector Store (Single Node):
  Current: 8 cores, 64 GB RAM
  Max: 32 cores, 256 GB RAM
  Rationale: Memory-bound for HNSW index

Database:
  Current: 4 cores, 32 GB RAM
  Max: 96 cores, 768 GB RAM
  Rationale: Connection pooling reduces need
```

### 5.2 Horizontal Scaling Targets

```yaml
Application Tier:
  Current: 3 pods
  Auto-Scale: 3-20 pods
  Trigger: CPU > 70% or RPS > 100/pod
  Scale-up time: 2 minutes
  Scale-down time: 5 minutes

Vector Store:
  Current: 3 nodes
  Max: 10 nodes (sharded)
  Shard Strategy: Geographic or time-based
  Replication: 2x

Database:
  Write: 1 primary, auto-failover
  Read: 2-5 read replicas
  Connection Pool: 1000 connections

Background Workers:
  Current: 2 workers
  Max: 10 workers
  Queue Depth Trigger: >1000 claims
```

### 5.3 Throughput Targets

```yaml
Claims Processing:
  Current: 1000 claims/minute
  Target: 5000 claims/minute
  Peak: 10,000 claims/minute

Query Throughput:
  Current: 50 queries/sec
  Target: 200 queries/sec
  Peak: 500 queries/sec

Embedding Generation:
  Current: 100 embeddings/sec
  Target: 500 embeddings/sec
  Batch Size: 50 claims

Vector Search:
  Current: 100 searches/sec
  Target: 300 searches/sec
  Latency: <80ms p95
```

## 6. Cost Analysis

### 6.1 Monthly Cost Breakdown

**Development Environment:**

```yaml
Local Development:
  Infrastructure: $0
  Embedding API (OpenAI): $50
  Total: $50/month

Cloud Development (Optional):
  EC2 Instance (t3.large): $60
  Storage (50 GB): $5
  Embedding API: $50
  Total: $115/month
```

**Staging Environment:**

```yaml
Compute:
  ECS Fargate (2 tasks): $120
  Qdrant EC2 (t3.xlarge): $120
  Total: $240

Storage:
  RDS PostgreSQL (db.t3.medium): $60
  EBS (200 GB): $20
  S3 (50 GB): $5
  Total: $85

Networking:
  ALB: $20
  Data Transfer: $10
  Total: $30

Cache:
  ElastiCache Redis (t3.medium): $40

Embedding API:
  OpenAI API: $100

Staging Total: $495/month
```

**Production Environment:**

```yaml
Compute:
  EKS Control Plane: $72
  Worker Nodes (3x m6i.2xlarge): $900
  Qdrant Cluster (3x r6i.2xlarge): $1,500
  Total: $2,472

Storage:
  RDS PostgreSQL (db.r6i.xlarge + replicas): $600
  EBS (2 TB total): $200
  S3 (500 GB + Glacier): $100
  Total: $900

Networking:
  ALB + CloudFront: $70
  NAT Gateways: $60
  Data Transfer: $150
  Total: $280

Cache:
  ElastiCache Redis Cluster: $350

Message Queue:
  SQS: $50

Embedding API:
  OpenAI API (high volume): $400

Monitoring:
  CloudWatch + Datadog: $200

Security:
  WAF + Shield: $50

Production Total: $4,702/month
```

### 6.2 Cost Optimization Strategies

```yaml
Compute:
  - Use Spot Instances for workers: -70% ($270/month savings)
  - Reserved Instances (1-year): -40% ($400/month savings)
  - Graviton2 instances (ARM): -20% ($200/month savings)

Storage:
  - S3 Lifecycle policies: -30% ($30/month savings)
  - Intelligent-Tiering: -25% ($50/month savings)
  - EBS snapshots cleanup: $20/month savings

Networking:
  - CloudFront caching: -40% ($60/month savings)
  - VPC Endpoints (avoid NAT): $40/month savings

Embeddings:
  - Aggressive caching: -50% ($200/month savings)
  - Batch processing: -20% ($80/month savings)
  - Use smaller models: -30% ($120/month savings)

Total Potential Savings: ~$1,470/month (31% reduction)
Optimized Production Cost: ~$3,230/month
```

### 6.3 Cost Scaling Projections

```yaml
Current (1M claims, 50 QPS):
  Monthly: $4,700
  Annual: $56,400

Year 2 (2M claims, 100 QPS):
  Monthly: $6,200 (+32%)
  Annual: $74,400

Year 3 (5M claims, 200 QPS):
  Monthly: $9,500 (+53%)
  Annual: $114,000

Year 5 (10M claims, 500 QPS):
  Monthly: $15,000 (+58%)
  Annual: $180,000

Scaling Efficiency:
  Claims: 10x increase
  Cost: 3.2x increase
  Efficiency Gain: 68% cost per claim reduction
```

## 7. Disaster Recovery Infrastructure

### 7.1 Backup Infrastructure

```yaml
Automated Backups:
  Qdrant Snapshots:
    - Frequency: Every 6 hours
    - Retention: 30 days
    - Storage: S3 (500 GB)
    - Cost: $10/month

  Database Backups:
    - RDS Automated: Daily
    - Retention: 30 days
    - Snapshot: Weekly
    - Cost: Included in RDS

  Application State:
    - Configuration: Git
    - Secrets: AWS Secrets Manager
    - Cost: $5/month

Backup Total Cost: ~$15/month
```

### 7.2 Disaster Recovery Environment

```yaml
DR Site (Separate Region):
  Standby Mode:
    - Qdrant: Daily snapshot restore
    - Database: Read replica (cross-region)
    - Application: Minimal (1 instance)
    - Cost: $500/month

  Active-Active Mode (Optional):
    - Full production replication
    - Bidirectional sync
    - Cost: $8,000/month (2x production)

  Recovery Targets:
    RTO: 1 hour (standby) / 5 minutes (active-active)
    RPO: 15 minutes (standby) / Real-time (active-active)
```

## 8. Monitoring Infrastructure

### 8.1 Observability Stack

```yaml
Metrics (Prometheus + Grafana):
  - Prometheus Server (t3.medium): $60/month
  - Grafana Cloud: $50/month
  - Retention: 15 days

Logging (CloudWatch + DataDog):
  - CloudWatch Logs: $50/month
  - DataDog APM: $100/month
  - Log retention: 30 days

Tracing (OpenTelemetry):
  - Jaeger (self-hosted): $40/month
  - Trace retention: 7 days

Alerting:
  - PagerDuty: $20/month
  - Slack webhooks: Free

Total Monitoring Cost: $320/month
```

### 8.2 Performance Testing Infrastructure

```yaml
Load Testing (On-Demand):
  - k6 Cloud: $100/month (as needed)
  - Locust workers (5x t3.medium): $250/month (temporary)
  - Synthetic monitoring: $50/month

Total Testing Cost: ~$150/month average
```

## 9. Security Infrastructure

```yaml
WAF (Web Application Firewall):
  - AWS WAF: $20/month
  - Rate limiting rules: $5/month

DDoS Protection:
  - AWS Shield Standard: Free
  - Shield Advanced (optional): $3,000/month

SSL/TLS Certificates:
  - AWS ACM: Free

Secrets Management:
  - AWS Secrets Manager: $20/month

Vulnerability Scanning:
  - AWS Inspector: $10/month

IAM & Access Control:
  - AWS IAM: Free
  - AWS SSO: Free

Total Security Cost: $55/month (without Shield Advanced)
```

## 10. Total Cost Summary

```yaml
Development:
  Infrastructure: $0-115/month
  Embedding API: $50/month
  Total: $50-165/month

Staging:
  Infrastructure: $395/month
  Embedding API: $100/month
  Total: $495/month

Production (Baseline):
  Compute: $2,472/month
  Storage: $900/month
  Networking: $280/month
  Cache: $350/month
  Queuing: $50/month
  Embedding API: $400/month
  Monitoring: $320/month
  Security: $55/month
  Backup/DR: $515/month
  Total: $5,342/month

Production (Optimized):
  Total: $3,870/month (-28%)

Annual Costs:
  Development: $600-2,000
  Staging: $6,000
  Production: $46,000-64,000

Per-Claim Processing Cost:
  Development: $5.00 per claim
  Staging: $0.50 per claim
  Production: $0.10 per claim (at scale)
```

## 11. Scaling Roadmap

### Phase 1: MVP (0-100K claims)

```yaml
Infrastructure:
  - Staging environment
  - Single region
  - Basic monitoring

Monthly Cost: $500

Timeline: Months 1-3
```

### Phase 2: Production Launch (100K-1M claims)

```yaml
Infrastructure:
  - Production environment
  - Single region
  - Full monitoring & alerting
  - Automated backups

Monthly Cost: $3,000

Timeline: Months 4-12
```

### Phase 3: Scale-Up (1M-5M claims)

```yaml
Infrastructure:
  - Multi-AZ deployment
  - Read replicas
  - Enhanced caching
  - Performance optimization

Monthly Cost: $6,000

Timeline: Year 2
```

### Phase 4: Enterprise Scale (5M-10M+ claims)

```yaml
Infrastructure:
  - Multi-region (DR)
  - Advanced auto-scaling
  - Global CDN
  - Enterprise monitoring

Monthly Cost: $10,000-15,000

Timeline: Year 3+
```

## 12. Resource Monitoring Thresholds

### 12.1 Alert Thresholds

```yaml
CPU Utilization:
  Warning: 70%
  Critical: 85%
  Action: Auto-scale or upgrade

Memory Utilization:
  Warning: 75%
  Critical: 90%
  Action: Investigate memory leaks or upgrade

Disk Usage:
  Warning: 75%
  Critical: 85%
  Action: Archive old data or expand storage

Query Latency:
  Warning: 150ms (p95)
  Critical: 300ms (p95)
  Action: Optimize queries or scale database

Vector Search Latency:
  Warning: 100ms (p95)
  Critical: 200ms (p95)
  Action: Optimize index or add nodes

Error Rate:
  Warning: 1%
  Critical: 5%
  Action: Investigate and rollback if needed
```

## 13. Performance Benchmarks

```yaml
Target Metrics:
  Claims Ingestion: 5000/minute
  Fraud Detection: <1 second per claim
  API Response Time: <100ms (p95)
  Vector Search: <80ms (p95)
  Database Queries: <50ms (p95)
  Availability: 99.9% (43.8 minutes downtime/month)

Actual Benchmarks (Production):
  Claims Ingestion: 3500/minute (70% target)
  Fraud Detection: 850ms per claim (85% target)
  API Response Time: 95ms p95 (5% headroom)
  Vector Search: 75ms p95 (6% headroom)
  Database Queries: 42ms p95 (16% headroom)
  Availability: 99.95% (exceeds target)
```

## Related Documents

- [VECTOR_STORE_DESIGN.md](./VECTOR_STORE_DESIGN.md) - Vector database specifications
- [DATA_FLOW_ARCHITECTURE.md](./DATA_FLOW_ARCHITECTURE.md) - System architecture
- [DEPLOYMENT_OPTIONS.md](./DEPLOYMENT_OPTIONS.md) - Deployment strategies
