# Vector Store Design for Insurance Fraud Detection RAG System

## Executive Summary

This document outlines the vector database architecture for the insurance fraud detection system's RAG (Retrieval-Augmented Generation) knowledge base. The system requires high-performance vector search for fraud pattern matching, medical coding validation, and historical claim analysis.

## 1. Vector Store Comparison

### 1.1 Evaluated Solutions

| Solution | Type | Strengths | Weaknesses | Best For |
|----------|------|-----------|------------|----------|
| **Pinecone** | Cloud-Native | Managed service, excellent performance, auto-scaling | Vendor lock-in, cost at scale, data residency concerns | Quick deployment, production workloads |
| **Weaviate** | Hybrid | Strong ML integration, GraphQL, multi-tenancy | Complex setup, resource intensive | Complex entity relationships |
| **Qdrant** | Hybrid | Rust-based (fast), excellent filtering, efficient storage | Smaller ecosystem, newer product | High-performance filtering |
| **LanceDB** | Embedded | Zero-copy, serverless, disk-based, cost-effective | Limited cloud features, newer product | Local development, cost optimization |
| **Milvus** | Self-Hosted | Enterprise features, proven scalability, rich ecosystem | Complex operations, resource intensive | Large-scale deployments |

### 1.2 Selection Criteria

**Primary Requirements:**
- Store 1M+ claim embeddings (target: 10M+ for production scale)
- Support multi-field filtering (provider, diagnosis codes, date ranges, fraud types)
- Sub-100ms query latency for 95th percentile
- Support hybrid search (vector + metadata filtering)
- HIPAA compliance capability
- Cost-effective at scale

**Evaluation Metrics:**

```
Performance (30%):
- Query latency: <100ms p95
- Indexing throughput: >10K vectors/sec
- Concurrent query support: >100 queries/sec

Cost (25%):
- Storage cost per million vectors
- Compute cost for queries
- Infrastructure overhead

Operational Complexity (20%):
- Setup time and expertise required
- Maintenance burden
- Monitoring and debugging tools

Compliance (15%):
- HIPAA compatibility
- Data residency controls
- Audit logging

Ecosystem (10%):
- LangChain/LlamaIndex integration
- Community support
- Documentation quality
```

### 1.3 Recommended Solution: **Qdrant**

**Primary Choice: Qdrant**

Rationale:
1. **Performance**: Rust-based implementation provides excellent speed/efficiency balance
2. **Filtering**: Advanced payload filtering crucial for claim attributes (provider NPI, CPT codes, date ranges)
3. **Hybrid Deployment**: Start local, scale to cloud without architecture changes
4. **Cost**: Open-source with efficient resource utilization
5. **Compliance**: Self-hosted option ensures data residency control

**Secondary Choice: LanceDB (Development/Cost-Constrained)**

For development environments or cost-sensitive deployments:
- Zero infrastructure overhead
- Disk-based storage (columnar format)
- Easy local testing
- Simple migration path to Qdrant for production

### 1.4 Architecture Decision

```
Development/Testing:
├── LanceDB (local, embedded)
│   └── Single-file storage
│   └── No server required
│   └── Direct Python API

Staging/Production:
├── Qdrant Cluster (self-hosted or cloud)
│   ├── 3-node cluster (HA)
│   ├── Distributed indexing
│   ├── Replication factor: 2
│   └── Snapshot backups
```

## 2. Qdrant Architecture Design

### 2.1 Collection Schema

**Primary Collection: `fraud_claims_kb`**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

# Collection configuration
collection_config = {
    "name": "fraud_claims_kb",
    "vectors": {
        # Dense embeddings from claim text
        "claim_text": VectorParams(
            size=1536,  # OpenAI ada-002 or similar
            distance=Distance.COSINE
        ),
        # Sparse embeddings for medical codes
        "medical_codes": VectorParams(
            size=768,   # Specialized medical code embeddings
            distance=Distance.COSINE
        )
    },
    # Enable payload indexing for fast filtering
    "payload_schema": {
        "claim_id": PayloadSchemaType.KEYWORD,
        "provider_npi": PayloadSchemaType.KEYWORD,
        "diagnosis_codes": PayloadSchemaType.KEYWORD,
        "procedure_codes": PayloadSchemaType.KEYWORD,
        "fraud_type": PayloadSchemaType.KEYWORD,
        "fraud_indicator": PayloadSchemaType.BOOL,
        "claim_date": PayloadSchemaType.DATETIME,
        "billed_amount": PayloadSchemaType.FLOAT,
        "geographic_region": PayloadSchemaType.KEYWORD,
        "red_flags": PayloadSchemaType.KEYWORD
    }
}
```

**Supporting Collections:**

```
1. provider_patterns_kb
   - Provider behavioral patterns
   - Historical fraud indicators
   - Network relationship embeddings

2. medical_code_kb
   - ICD-10 code descriptions and relationships
   - CPT code descriptions and valid combinations
   - NDC drug information

3. fraud_rules_kb
   - Known fraud patterns
   - Detection rules and thresholds
   - Industry best practices
```

### 2.2 Embedding Strategy

**Multi-Modal Embeddings:**

```python
# Claim text embedding (primary)
claim_text_components = [
    f"Provider: {provider_info}",
    f"Diagnosis: {diagnosis_descriptions}",
    f"Procedures: {procedure_descriptions}",
    f"Patient History: {patient_context}",
    f"Claim Details: {claim_narrative}"
]
claim_text_embedding = embed_model.encode(
    " | ".join(claim_text_components)
)

# Medical code embedding (specialized)
code_vector = medical_code_encoder.encode({
    "icd10": diagnosis_codes,
    "cpt": procedure_codes,
    "ndc": medication_codes
})

# Payload for filtering
payload = {
    "claim_id": claim.claim_id,
    "provider_npi": claim.provider_npi,
    "diagnosis_codes": claim.diagnosis_codes,
    "procedure_codes": claim.procedure_codes,
    "fraud_type": claim.fraud_type if claim.fraud_indicator else None,
    "fraud_indicator": claim.fraud_indicator,
    "claim_date": claim.claim_date.isoformat(),
    "billed_amount": float(claim.billed_amount),
    "geographic_region": claim.geographic_region,
    "red_flags": claim.red_flags
}
```

### 2.3 Indexing Strategy

**HNSW Index Configuration:**

```python
from qdrant_client.models import HnswConfigDiff

index_config = HnswConfigDiff(
    m=16,              # Number of bi-directional links per node
    ef_construct=100,  # Size of dynamic candidate list during construction
    full_scan_threshold=10000,  # Switch to exact search below this
    max_indexing_threads=0  # Use all available threads
)
```

**Index Parameters Rationale:**
- `m=16`: Balance between search speed and memory usage
- `ef_construct=100`: Good recall while maintaining build speed
- `full_scan_threshold=10000`: Optimize small result sets

**Query-Time Configuration:**

```python
# For fraud detection (prioritize recall)
fraud_search_params = SearchParams(
    hnsw_ef=128,  # Larger candidate list for better recall
    exact=False
)

# For exact code matching (prioritize precision)
code_search_params = SearchParams(
    hnsw_ef=64,
    exact=True  # Use brute force for critical matches
)
```

## 3. Deployment Architecture

### 3.1 Local Development

```
┌─────────────────────────────────────────┐
│         Development Environment          │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────┐      ┌───────────────┐  │
│  │  FastAPI   │◄────►│  LanceDB      │  │
│  │  Service   │      │  (embedded)   │  │
│  └────────────┘      └───────────────┘  │
│        │                                 │
│        │ Embedding                       │
│        ▼                                 │
│  ┌────────────┐                         │
│  │ OpenAI API │                         │
│  │ (or local) │                         │
│  └────────────┘                         │
│                                          │
└─────────────────────────────────────────┘

Storage: Local disk (~10GB for 100K claims)
Latency: <50ms for queries
Cost: $0/month (excluding embeddings)
```

### 3.2 Staging Environment

```
┌─────────────────────────────────────────────────────┐
│              Staging Environment (AWS)               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │   ALB        │         │   Qdrant Cluster     │  │
│  │              │         │   (Single Node)      │  │
│  └──────┬───────┘         └──────────────────────┘  │
│         │                           ▲               │
│         ▼                           │               │
│  ┌──────────────┐                  │               │
│  │  FastAPI     │──────────────────┘               │
│  │  Service     │                                   │
│  │  (ECS)       │                                   │
│  └──────────────┘                                   │
│         │                                            │
│         │ Embedding                                  │
│         ▼                                            │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │ Bedrock or   │         │    S3 Backup         │  │
│  │ SageMaker    │         │    (Snapshots)       │  │
│  └──────────────┘         └──────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘

Instance: t3.xlarge (4 vCPU, 16GB RAM)
Storage: 100GB gp3 EBS
Latency: <100ms p95
Cost: ~$150-200/month
```

### 3.3 Production Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   Production Environment (AWS)                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐        ┌──────────────────────────────────┐    │
│  │  CloudFront│        │      Application Layer          │    │
│  │  + WAF     │        │  ┌────────────┐ ┌────────────┐  │    │
│  └─────┬──────┘        │  │  FastAPI   │ │  FastAPI   │  │    │
│        │               │  │  (ECS/EKS) │ │  (ECS/EKS) │  │    │
│        ▼               │  └──────┬─────┘ └──────┬─────┘  │    │
│  ┌────────────┐        └─────────┼──────────────┼────────┘    │
│  │    ALB     │                  │              │              │
│  └─────┬──────┘                  │              │              │
│        │                         ▼              ▼              │
│        │          ┌──────────────────────────────────────┐     │
│        │          │     Qdrant Cluster (3 Nodes)        │     │
│        │          │  ┌─────────┐ ┌─────────┐ ┌────────┐ │     │
│        └─────────►│  │ Node 1  │ │ Node 2  │ │ Node 3 │ │     │
│                   │  │ Primary │ │ Replica │ │ Replica│ │     │
│                   │  └─────────┘ └─────────┘ └────────┘ │     │
│                   └──────────────────────────────────────┘     │
│                                   │                            │
│                                   ▼                            │
│                   ┌──────────────────────────────────────┐     │
│                   │         Storage Layer                │     │
│                   │  ┌──────────┐    ┌──────────────┐   │     │
│                   │  │ EBS gp3  │    │  S3 Backup   │   │     │
│                   │  │ (Primary)│    │  (Snapshots) │   │     │
│                   │  └──────────┘    └──────────────┘   │     │
│                   └──────────────────────────────────────┘     │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Monitoring & Observability                │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │    │
│  │  │CloudWatch│  │ Grafana  │  │  OpenTelemetry   │    │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

Qdrant Nodes: r6i.2xlarge (8 vCPU, 64GB RAM)
Replication: Factor 2
Storage: 500GB gp3 per node
Latency: <80ms p95, <150ms p99
Throughput: >200 queries/sec
Cost: ~$1,500-2,000/month
```

## 4. Scaling Strategy

### 4.1 Vertical Scaling Thresholds

| Metric | Scale Up Trigger | Target Instance |
|--------|------------------|-----------------|
| Memory utilization | >80% for 5 min | Next size up (e.g., 64GB → 128GB) |
| CPU utilization | >70% sustained | More vCPUs (e.g., 8 → 16) |
| Query latency p95 | >150ms | Faster storage (gp3 → io2) |
| Indexing backlog | >1 hour | More compute resources |

### 4.2 Horizontal Scaling Strategy

**Sharding Strategy:**

```python
# Shard by geographic region or time period
shard_config = {
    "collections": [
        {
            "name": "fraud_claims_kb_northeast",
            "filter": {"geographic_region": ["NY", "NJ", "CT", "MA"]},
            "node": "node-1"
        },
        {
            "name": "fraud_claims_kb_southeast",
            "filter": {"geographic_region": ["FL", "GA", "NC", "SC"]},
            "node": "node-2"
        },
        {
            "name": "fraud_claims_kb_west",
            "filter": {"geographic_region": ["CA", "WA", "OR", "NV"]},
            "node": "node-3"
        }
    ]
}
```

**Replication Strategy:**

```
Production Configuration:
- Replication Factor: 2
- Write Consistency: 1 (at least 1 replica confirms)
- Read Consistency: 1 (read from any replica)

High-Availability Configuration:
- Replication Factor: 3
- Write Consistency: 2 (majority writes)
- Read Consistency: 1 (read from nearest replica)
```

### 4.3 Auto-Scaling Policy

```yaml
# Kubernetes HPA for application layer
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
```

### 4.4 Capacity Planning

**Growth Projections:**

```
Year 1 (Current):
├── Claims: 1M embeddings
├── Storage: 50GB
├── QPS: 50 queries/sec
└── Monthly Cost: $200-300

Year 2 (2x Growth):
├── Claims: 2M embeddings
├── Storage: 100GB
├── QPS: 100 queries/sec
└── Monthly Cost: $400-500

Year 3 (5x Growth):
├── Claims: 5M embeddings
├── Storage: 250GB
├── QPS: 250 queries/sec
└── Monthly Cost: $1,000-1,500

Year 5 (10x Growth):
├── Claims: 10M embeddings
├── Storage: 500GB
├── QPS: 500 queries/sec
└── Monthly Cost: $2,000-3,000
```

## 5. Performance Optimization

### 5.1 Query Optimization

**Tiered Search Strategy:**

```python
async def search_similar_fraud_claims(
    query_embedding: List[float],
    filters: Dict[str, Any],
    tier: str = "standard"
) -> List[ScoredPoint]:
    """
    Multi-tier search based on query characteristics.
    """
    if tier == "fast":
        # Quick filtering for high-confidence scenarios
        return await qdrant_client.search(
            collection_name="fraud_claims_kb",
            query_vector=query_embedding,
            query_filter=filters,
            limit=10,
            search_params=SearchParams(hnsw_ef=32, exact=False)
        )
    elif tier == "standard":
        # Balanced search for most queries
        return await qdrant_client.search(
            collection_name="fraud_claims_kb",
            query_vector=query_embedding,
            query_filter=filters,
            limit=20,
            search_params=SearchParams(hnsw_ef=64, exact=False)
        )
    elif tier == "thorough":
        # Deep search for critical fraud detection
        return await qdrant_client.search(
            collection_name="fraud_claims_kb",
            query_vector=query_embedding,
            query_filter=filters,
            limit=50,
            search_params=SearchParams(hnsw_ef=128, exact=False)
        )
```

### 5.2 Caching Strategy

**Multi-Level Caching:**

```
Level 1: Application Cache (Redis)
├── Frequently accessed claims
├── Common query results
├── TTL: 1 hour
└── Size: 2GB

Level 2: Vector Store Cache (Qdrant)
├── HNSW graph in memory
├── Hot payload data
├── Persistent: Yes
└── Size: 16GB RAM

Level 3: Persistent Storage (EBS/S3)
├── Full vector data
├── Snapshots and backups
└── Size: 500GB
```

### 5.3 Batch Processing

```python
async def batch_index_claims(
    claims: List[Claim],
    batch_size: int = 100
) -> None:
    """
    Batch indexing for efficient ingestion.
    """
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i + batch_size]

        # Parallel embedding generation
        embeddings = await asyncio.gather(*[
            generate_embeddings(claim) for claim in batch
        ])

        # Batch upsert to Qdrant
        points = [
            PointStruct(
                id=claim.claim_id,
                vector={
                    "claim_text": emb["text"],
                    "medical_codes": emb["codes"]
                },
                payload=claim.to_payload()
            )
            for claim, emb in zip(batch, embeddings)
        ]

        await qdrant_client.upsert(
            collection_name="fraud_claims_kb",
            points=points,
            wait=False  # Async indexing
        )
```

## 6. Disaster Recovery

### 6.1 Backup Strategy

```yaml
# Automated snapshot schedule
backup_policy:
  frequency: hourly
  retention:
    hourly: 24 snapshots
    daily: 7 snapshots
    weekly: 4 snapshots
    monthly: 12 snapshots
  storage:
    primary: S3 Standard
    archive: S3 Glacier (after 90 days)
  encryption: AES-256
```

### 6.2 Recovery Procedures

**RTO/RPO Targets:**

```
Production:
├── RTO (Recovery Time Objective): 1 hour
├── RPO (Recovery Point Objective): 15 minutes
└── Strategy: Hot standby replica

Staging:
├── RTO: 4 hours
├── RPO: 1 hour
└── Strategy: Snapshot restore
```

**Recovery Runbook:**

```bash
# 1. Identify failed node
kubectl get pods -n fraud-detection

# 2. Trigger failover to replica
kubectl scale deployment qdrant --replicas=2

# 3. Restore from latest snapshot
aws s3 cp s3://fraud-detection-backups/latest.snapshot /tmp/
qdrant-restore --snapshot /tmp/latest.snapshot

# 4. Verify data integrity
curl -X POST http://qdrant:6333/collections/fraud_claims_kb/points/count

# 5. Update DNS/load balancer
aws elbv2 modify-target-group --target-group-arn <arn>
```

## 7. Security Considerations

### 7.1 Encryption

```
At Rest:
├── Vector data: AES-256 (EBS encryption)
├── Backups: AES-256 (S3 SSE-KMS)
└── Snapshots: AES-256

In Transit:
├── Client → API: TLS 1.3
├── API → Qdrant: mTLS
└── Qdrant → Qdrant: mTLS (replication)
```

### 7.2 Access Control

```python
# Role-based access control
class VectorStoreACL:
    def __init__(self):
        self.roles = {
            "fraud_analyst": {
                "collections": ["fraud_claims_kb", "fraud_rules_kb"],
                "operations": ["read", "search"]
            },
            "data_engineer": {
                "collections": ["*"],
                "operations": ["read", "write", "delete"]
            },
            "api_service": {
                "collections": ["fraud_claims_kb"],
                "operations": ["read", "search"]
            }
        }
```

## 8. Monitoring and Observability

### 8.1 Key Metrics

```
Performance Metrics:
├── Query latency (p50, p95, p99)
├── Indexing throughput (vectors/sec)
├── Query throughput (queries/sec)
└── Error rate

Resource Metrics:
├── CPU utilization
├── Memory utilization
├── Disk I/O (IOPS, throughput)
├── Network I/O

Business Metrics:
├── Collection size (vectors)
├── Storage used (GB)
├── Daily query volume
└── Cache hit rate
```

### 8.2 Alerting Thresholds

```yaml
alerts:
  - name: HighQueryLatency
    condition: p95_latency > 200ms for 5m
    severity: warning
    action: notify_oncall

  - name: CriticalQueryLatency
    condition: p95_latency > 500ms for 2m
    severity: critical
    action: page_oncall

  - name: HighMemoryUsage
    condition: memory_utilization > 85% for 10m
    severity: warning
    action: auto_scale

  - name: IndexingFailure
    condition: error_rate > 5% for 5m
    severity: critical
    action: page_oncall
```

## 9. Cost Optimization

### 9.1 Cost Breakdown

```
Development (Local):
├── Infrastructure: $0/month
├── Embeddings: ~$50/month (OpenAI API)
└── Total: $50/month

Staging (AWS):
├── Compute (t3.xlarge): $120/month
├── Storage (100GB gp3): $10/month
├── Data Transfer: $10/month
├── Embeddings: $100/month
└── Total: $240/month

Production (AWS):
├── Compute (3x r6i.2xlarge): $1,200/month
├── Storage (1.5TB gp3): $150/month
├── Data Transfer: $200/month
├── Load Balancer: $30/month
├── Embeddings: $300/month
├── Monitoring: $50/month
└── Total: $1,930/month
```

### 9.2 Optimization Strategies

1. **Use Reserved Instances**: 40% savings on compute
2. **Implement Tiered Storage**: Move cold data to S3 Glacier
3. **Optimize Embeddings**: Cache frequently accessed embeddings
4. **Right-Size Instances**: Use CloudWatch metrics to optimize
5. **Implement Query Caching**: Reduce redundant vector searches

## 10. Migration Path

### 10.1 Development to Staging

```bash
# 1. Export from LanceDB
python scripts/export_lancedb.py --output /tmp/claims_export.parquet

# 2. Transform and load to Qdrant
python scripts/migrate_to_qdrant.py \
  --input /tmp/claims_export.parquet \
  --qdrant-url https://staging-qdrant.example.com \
  --collection fraud_claims_kb

# 3. Verify migration
python scripts/verify_migration.py \
  --source lancedb \
  --target qdrant \
  --sample-size 1000
```

### 10.2 Staging to Production

```bash
# 1. Create snapshot in staging
curl -X POST http://staging-qdrant:6333/collections/fraud_claims_kb/snapshots

# 2. Copy snapshot to production S3
aws s3 cp staging-snapshot.tar.gz s3://prod-backups/

# 3. Restore in production cluster
curl -X PUT http://prod-qdrant:6333/collections/fraud_claims_kb/snapshots/upload \
  --data-binary @prod-snapshot.tar.gz

# 4. Verify and switch traffic
kubectl apply -f k8s/prod-ingress.yaml
```

## Next Steps

1. **Prototype Development**: Implement LanceDB prototype for initial testing
2. **Staging Deployment**: Deploy Qdrant single-node on AWS for integration testing
3. **Performance Benchmarking**: Test query latency and throughput with production-like data
4. **Production Rollout**: Deploy 3-node Qdrant cluster with replication
5. **Monitoring Setup**: Implement comprehensive observability stack

## Related Documents

- [DATA_FLOW_ARCHITECTURE.md](./DATA_FLOW_ARCHITECTURE.md) - Data ingestion and processing pipelines
- [INFRASTRUCTURE_REQUIREMENTS.md](./INFRASTRUCTURE_REQUIREMENTS.md) - Detailed resource sizing and costs
- [DEPLOYMENT_OPTIONS.md](./DEPLOYMENT_OPTIONS.md) - Cloud vs local deployment trade-offs
