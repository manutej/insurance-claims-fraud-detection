# Enrichment Engine Quick Reference

## One-Line Summary
**5-factor confidence-scored claim enrichment with Redis caching, <500ms target**

---

## Quick Start

```python
from src.rag.enricher import EnrichmentEngine
from src.rag.schemas import EnrichmentRequest

# Initialize
engine = EnrichmentEngine(cache_enabled=True)
await engine.initialize()

# Enrich
request = EnrichmentRequest(claim_data={"procedure_codes": ["99213"]})
response = await engine.enrich_claim(request)

print(response.enriched_claim)  # Complete claim
print(response.overall_confidence)  # 0.0-1.0
print(response.enrichment_quality_score)  # EXCELLENT/GOOD/ACCEPTABLE/POOR
```

---

## Confidence Formula

```
confidence = 0.40 × retrieval_quality
           + 0.20 × source_diversity
           + 0.15 × temporal_relevance
           + 0.15 × cross_validation
           + 0.10 × regulatory_citation
```

---

## Quality Tiers

| Tier | Range | Action |
|------|-------|--------|
| EXCELLENT | ≥0.90 | Auto-accept |
| GOOD | 0.80-0.89 | Accept with logging |
| ACCEPTABLE | 0.70-0.79 | Accept with review flag |
| POOR | <0.70 | Manual review |

---

## Enrichment Strategies

```python
from src.rag.schemas import EnrichmentStrategy

# Use all 4 KBs
EnrichmentStrategy.ALL

# Specific strategies
EnrichmentStrategy.PATIENT_HISTORY
EnrichmentStrategy.PROVIDER_PATTERN
EnrichmentStrategy.MEDICAL_CODING
EnrichmentStrategy.REGULATORY
```

---

## Batch Processing

```python
from src.rag.schemas import BatchEnrichmentRequest

batch = BatchEnrichmentRequest(
    requests=[EnrichmentRequest(claim_data=c) for c in claims],
    parallel=True,
    max_concurrency=10
)

batch_response = await engine.enrich_batch(batch)
print(f"{batch_response.successful_count}/{batch_response.total_requests}")
```

---

## Cache Operations

```python
# Get stats
stats = await engine.cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Invalidate
await engine.cache.invalidate(claim_data)
await engine.cache.invalidate_by_kb_update("medical_coding")
await engine.cache.clear_all()
```

---

## Metrics Tracking

```python
from src.rag.enrichment_metrics import EnrichmentMetricsTracker

tracker = EnrichmentMetricsTracker()

# Track enrichments
tracker.track_enrichment(request, response, ground_truth)

# Get metrics
metrics = tracker.compute_metrics()
print(f"Avg confidence: {metrics.average_confidence}")
print(f"P95 latency: {metrics.p95_latency_ms}ms")
print(f"Coverage: {tracker.compute_coverage():.2%}")

# Generate report
report = tracker.generate_enrichment_report()
```

---

## Configuration

```python
# Engine
engine = EnrichmentEngine(
    cache_enabled=True,
    redis_url="redis://localhost:6379",
    max_parallel_requests=10
)

# Request
request = EnrichmentRequest(
    claim_data=incomplete_claim,
    enrichment_strategy=EnrichmentStrategy.ALL,
    confidence_threshold=0.70  # Tune for precision/recall
)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/rag/enricher.py` | Main engine |
| `src/rag/confidence_scoring.py` | 5-factor algorithm |
| `src/rag/enrichment_cache.py` | Redis cache |
| `src/rag/enrichment_metrics.py` | Metrics tracking |
| `src/rag/schemas.py` | Pydantic models |

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Single claim | <500ms | Ready for Phase 2C |
| Batch claim | <1s | Ready for Phase 2C |
| Cache hit rate | >60% | Ready for testing |
| Confidence accuracy | >90% | Ready for calibration |
| Test coverage | >90% | ✅ 95% |

---

## Common Patterns

### Error Handling
```python
try:
    response = await engine.enrich_claim(request)
    if response.enrichment_quality_score == "POOR":
        # Flag for review
        pass
except Exception as e:
    logger.error("enrichment_failed", error=str(e))
```

### Cache Warming
```python
async def warm_cache():
    for claim in common_claims:
        await engine.enrich_claim(EnrichmentRequest(claim_data=claim))
```

### Dynamic Thresholding
```python
# High precision
high_precision = EnrichmentRequest(claim_data=claim, confidence_threshold=0.85)

# High recall
high_recall = EnrichmentRequest(claim_data=claim, confidence_threshold=0.65)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low confidence | Populate KBs, increase diversity |
| High latency | Enable caching, use batching |
| Low cache hit rate | Increase TTL, warm cache |
| Inaccurate enrichments | Update KBs, increase threshold |

---

## Test Command

```bash
source venv/bin/activate
pytest tests/unit/rag/test_confidence_scoring.py -v
# Expected: 37 passed, 2 skipped (95% coverage)
```

---

## Documentation

- **Usage Guide**: `docs/ENRICHMENT_ENGINE_GUIDE.md`
- **Algorithm Details**: `docs/CONFIDENCE_SCORING_DETAILED.md`
- **Implementation Summary**: `docs/PHASE_2B_IMPLEMENTATION_SUMMARY.md`
- **This Reference**: `docs/ENRICHMENT_QUICK_REFERENCE.md`

---

**Phase**: 2B Complete ✅ | **Next**: Phase 2C KB Integration
