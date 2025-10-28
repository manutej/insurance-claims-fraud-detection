# Enrichment Engine Usage Guide

## Overview

The Enrichment Engine is a production-ready system for filling missing insurance claim data using 4 knowledge bases with confidence-scored enrichment decisions.

**Phase**: 2B - Enrichment Engine Core
**Status**: Implemented
**Performance Targets**:
- Single claim enrichment: <500ms
- Batch enrichment: <1s
- Confidence accuracy: >90%
- Cache hit rate: >60%

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Enrichment Engine                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐    ┌──────────────────┐                │
│  │ EnrichmentCache│◄───│  Redis Backend   │                │
│  └────────┬───────┘    └──────────────────┘                │
│           │                                                   │
│  ┌────────▼────────────────────────────────────────┐        │
│  │          EnrichmentEngine Core                   │        │
│  │  - Parallel KB retrieval                        │        │
│  │  - Field enrichment logic                       │        │
│  │  - Batch processing                             │        │
│  └────────┬────────────────────────────────────────┘        │
│           │                                                   │
│  ┌────────▼────────────────────────────────────────┐        │
│  │       ConfidenceScorer (5 factors)              │        │
│  │  - Retrieval quality (40%)                      │        │
│  │  - Source diversity (20%)                       │        │
│  │  - Temporal relevance (15%)                     │        │
│  │  - Cross-validation (15%)                       │        │
│  │  - Regulatory citation (10%)                    │        │
│  └─────────────────────────────────────────────────┘        │
│                                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │        EnrichmentMetricsTracker                 │        │
│  │  - Accuracy per field                           │        │
│  │  - Latency percentiles                          │        │
│  │  - Coverage metrics                             │        │
│  └─────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │     4 Knowledge Bases (Phase 2A)      │
        ├───────────────────────────────────────┤
        │  1. Patient History KB                │
        │  2. Provider Pattern KB               │
        │  3. Medical Coding KB                 │
        │  4. Regulatory KB                     │
        └───────────────────────────────────────┘
```

## Quick Start

### 1. Basic Enrichment

```python
import asyncio
from src.rag.enricher import EnrichmentEngine
from src.rag.schemas import EnrichmentRequest, EnrichmentStrategy

async def enrich_claim():
    # Initialize engine
    engine = EnrichmentEngine(cache_enabled=True)
    await engine.initialize()

    # Create enrichment request
    incomplete_claim = {
        "claim_id": "CLM-2024-001",
        "patient_id": "PAT-10001",
        "provider_npi": "1234567890",
        "procedure_codes": ["99213"],
        # Missing diagnosis_codes, descriptions, etc.
    }

    request = EnrichmentRequest(
        claim_data=incomplete_claim,
        enrichment_strategy=EnrichmentStrategy.ALL,
        confidence_threshold=0.70
    )

    # Enrich
    response = await engine.enrich_claim(request)

    # Use enriched data
    print(f"Enriched Claim: {response.enriched_claim}")
    print(f"Confidence: {response.overall_confidence}")
    print(f"Quality: {response.enrichment_quality_score}")

    # Cleanup
    await engine.close()

asyncio.run(enrich_claim())
```

### 2. Batch Enrichment

```python
from src.rag.schemas import BatchEnrichmentRequest

async def enrich_batch():
    engine = EnrichmentEngine()
    await engine.initialize()

    # Multiple claims to enrich
    claims = [
        {"claim_id": f"CLM-{i}", "procedure_codes": ["99213"]}
        for i in range(100)
    ]

    # Create batch request
    batch_request = BatchEnrichmentRequest(
        requests=[
            EnrichmentRequest(claim_data=claim)
            for claim in claims
        ],
        parallel=True,
        max_concurrency=10
    )

    # Enrich batch
    batch_response = await engine.enrich_batch(batch_request)

    print(f"Total: {batch_response.total_requests}")
    print(f"Successful: {batch_response.successful_count}")
    print(f"Failed: {batch_response.failed_count}")
    print(f"Avg time: {batch_response.average_processing_time_ms}ms")

    await engine.close()

asyncio.run(enrich_batch())
```

### 3. With Metrics Tracking

```python
from src.rag.enrichment_metrics import EnrichmentMetricsTracker

async def enrich_with_metrics():
    engine = EnrichmentEngine()
    await engine.initialize()

    tracker = EnrichmentMetricsTracker()

    # Enrich multiple claims
    for claim in test_claims:
        request = EnrichmentRequest(claim_data=claim)
        response = await engine.enrich_claim(request)

        # Track with ground truth (if available)
        tracker.track_enrichment(request, response, ground_truth_values)

    # Compute metrics
    metrics = tracker.compute_metrics()
    print(f"Average Confidence: {metrics.average_confidence}")
    print(f"P95 Latency: {metrics.p95_latency_ms}ms")
    print(f"Accuracy per field: {metrics.accuracy_per_field}")

    # Generate report
    report = tracker.generate_enrichment_report()

    await engine.close()
```

## Configuration

### Engine Configuration

```python
engine = EnrichmentEngine(
    cache_enabled=True,              # Enable Redis caching
    redis_url="redis://localhost:6379",  # Redis connection
    max_parallel_requests=10         # Max concurrent enrichments
)
```

### Cache Configuration

```python
from src.rag.enrichment_cache import EnrichmentCache

cache = EnrichmentCache(
    redis_url="redis://localhost:6379",
    default_ttl_seconds=86400,  # 24 hours
    key_prefix="enrichment:",
    max_retries=3
)
await cache.initialize()
```

### Confidence Threshold Tuning

```python
request = EnrichmentRequest(
    claim_data=incomplete_claim,
    confidence_threshold=0.80  # Higher threshold = more selective
)

# Quality Tiers:
# - EXCELLENT: confidence >= 0.90
# - GOOD: confidence 0.80-0.89
# - ACCEPTABLE: confidence 0.70-0.79
# - POOR: confidence < 0.70
```

## Enrichment Strategies

### 1. ALL (Default)
Consult all 4 knowledge bases for maximum confidence:

```python
request = EnrichmentRequest(
    claim_data=claim,
    enrichment_strategy=EnrichmentStrategy.ALL
)
```

### 2. PATIENT_HISTORY
Use patient history for diagnosis prediction:

```python
request = EnrichmentRequest(
    claim_data=claim,
    enrichment_strategy=EnrichmentStrategy.PATIENT_HISTORY
)
```

### 3. PROVIDER_PATTERN
Use provider patterns for typical billing:

```python
request = EnrichmentRequest(
    claim_data=claim,
    enrichment_strategy=EnrichmentStrategy.PROVIDER_PATTERN
)
```

### 4. MEDICAL_CODING
Use medical coding standards:

```python
request = EnrichmentRequest(
    claim_data=claim,
    enrichment_strategy=EnrichmentStrategy.MEDICAL_CODING
)
```

### 5. REGULATORY
Validate against regulatory requirements:

```python
request = EnrichmentRequest(
    claim_data=claim,
    enrichment_strategy=EnrichmentStrategy.REGULATORY
)
```

## Cache Management

### Check Cache Statistics

```python
cache = engine.cache
stats = await cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cached enrichments: {stats['cached_enrichments']}")
```

### Invalidate Cache

```python
# Invalidate specific claim
await cache.invalidate(claim_data)

# Invalidate all entries from a KB update
await cache.invalidate_by_kb_update("medical_coding")

# Clear all cache
await cache.clear_all()
```

## Performance Optimization

### 1. Batch Processing
Use batch enrichment for multiple claims:

```python
# Parallel processing (faster)
batch_request = BatchEnrichmentRequest(
    requests=requests,
    parallel=True,
    max_concurrency=20  # Tune based on resources
)
```

### 2. Cache Warming
Pre-populate cache for common claims:

```python
async def warm_cache():
    common_claims = load_common_claims()
    for claim in common_claims:
        request = EnrichmentRequest(claim_data=claim)
        await engine.enrich_claim(request)
```

### 3. Confidence Threshold Tuning
Lower threshold for higher coverage:

```python
# High precision (fewer enrichments, higher confidence)
request = EnrichmentRequest(claim_data=claim, confidence_threshold=0.85)

# High recall (more enrichments, lower confidence)
request = EnrichmentRequest(claim_data=claim, confidence_threshold=0.65)
```

## Error Handling

```python
from src.rag.schemas import EnrichmentResponse

async def safe_enrich(claim):
    try:
        request = EnrichmentRequest(claim_data=claim)
        response = await engine.enrich_claim(request)

        # Check quality
        if response.enrichment_quality_score == "POOR":
            logger.warning(
                "low_quality_enrichment",
                claim_id=claim["claim_id"],
                confidence=response.overall_confidence
            )
            # Consider manual review

        return response

    except Exception as e:
        logger.error("enrichment_failed", error=str(e))
        # Fallback or retry logic
        return None
```

## Monitoring and Observability

### Structured Logging

All components use `structlog` for structured logging:

```python
import structlog

logger = structlog.get_logger(__name__)

# Logs automatically include:
# - timestamp
# - log_level
# - event_type
# - contextual fields
```

### Metrics Dashboard

```python
async def get_metrics_dashboard():
    # Enrichment metrics
    metrics = tracker.compute_metrics()

    # Cache metrics
    cache_stats = await engine.cache.get_stats()

    dashboard = {
        "enrichment": {
            "total": metrics.total_enrichments,
            "avg_confidence": metrics.average_confidence,
            "p95_latency_ms": metrics.p95_latency_ms,
            "coverage": tracker.compute_coverage(),
        },
        "cache": {
            "hit_rate": cache_stats["hit_rate"],
            "total_keys": cache_stats["cached_enrichments"],
        },
        "accuracy": {
            "per_field": metrics.accuracy_per_field,
            "per_kb": metrics.kb_accuracy,
        }
    }

    return dashboard
```

## Best Practices

### 1. Always Initialize and Close
```python
engine = EnrichmentEngine()
await engine.initialize()  # Setup connections
try:
    # Use engine
    pass
finally:
    await engine.close()  # Cleanup
```

### 2. Track Metrics in Production
```python
tracker = EnrichmentMetricsTracker()
# Track all enrichments for analysis
```

### 3. Set Appropriate Thresholds
```python
# Mission-critical: use high threshold
request = EnrichmentRequest(claim_data=claim, confidence_threshold=0.85)

# Data augmentation: use lower threshold
request = EnrichmentRequest(claim_data=claim, confidence_threshold=0.65)
```

### 4. Monitor Cache Performance
```python
stats = await cache.get_stats()
if stats["hit_rate"] < 0.60:
    logger.warning("cache_hit_rate_low", hit_rate=stats["hit_rate"])
    # Consider increasing TTL or warming cache
```

### 5. Handle Low-Confidence Results
```python
if response.overall_confidence < 0.70:
    # Flag for manual review
    # Or use fallback enrichment strategy
    pass
```

## Troubleshooting

### Low Confidence Scores
- **Cause**: Insufficient KB data, poor retrieval quality
- **Solution**: Populate KBs with more data, tune retrieval parameters

### High Latency
- **Cause**: KB queries slow, cache disabled
- **Solution**: Enable caching, optimize KB indices, use batch processing

### Low Cache Hit Rate
- **Cause**: High claim variability, short TTL
- **Solution**: Increase TTL, implement cache warming

### Inaccurate Enrichments
- **Cause**: Poor cross-validation, outdated KB data
- **Solution**: Update KBs regularly, increase confidence threshold

## Next Steps

1. **Integrate with KBs**: Connect enricher to Phase 2A knowledge bases
2. **Production Deployment**: Deploy with monitoring and alerting
3. **Fine-Tuning**: Optimize confidence weights based on production data
4. **Automated Testing**: Run continuous accuracy validation

## API Reference

See detailed API documentation:
- `src/rag/enricher.py` - EnrichmentEngine, FieldEnricher
- `src/rag/confidence_scoring.py` - ConfidenceScorer
- `src/rag/enrichment_cache.py` - EnrichmentCache
- `src/rag/enrichment_metrics.py` - EnrichmentMetricsTracker
- `src/rag/schemas.py` - All Pydantic models
