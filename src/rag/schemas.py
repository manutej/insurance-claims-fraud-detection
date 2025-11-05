"""
Pydantic schemas for enrichment engine components.

Defines data models for enrichment requests, responses, decisions,
and metrics tracking.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class EnrichmentStrategy(str, Enum):
    """Enrichment strategy types."""
    PATIENT_HISTORY = "patient_history"
    PROVIDER_PATTERN = "provider_pattern"
    MEDICAL_CODING = "medical_coding"
    REGULATORY = "regulatory"
    ALL = "all"


class EnrichmentQualityTier(str, Enum):
    """Quality tiers for enrichment results."""
    EXCELLENT = "EXCELLENT"  # confidence >= 0.90
    GOOD = "GOOD"  # confidence 0.80-0.89
    ACCEPTABLE = "ACCEPTABLE"  # confidence 0.70-0.79
    POOR = "POOR"  # confidence < 0.70


class KnowledgeBaseType(str, Enum):
    """Knowledge base types."""
    PATIENT_HISTORY = "patient_history"
    PROVIDER_PATTERN = "provider_pattern"
    MEDICAL_CODING = "medical_coding"
    REGULATORY = "regulatory"


class EnrichmentRequest(BaseModel):
    """Request model for claim enrichment."""

    claim_data: Dict[str, Any] = Field(
        ...,
        description="Partial claim data with missing fields"
    )
    enrichment_strategy: EnrichmentStrategy = Field(
        default=EnrichmentStrategy.ALL,
        description="Strategy to use for enrichment"
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required for enrichment"
    )

    class Config:
        use_enum_values = True


class EnrichmentEvidence(BaseModel):
    """Evidence supporting an enrichment decision."""

    source_kb: KnowledgeBaseType = Field(
        ...,
        description="Knowledge base that provided this evidence"
    )
    document_id: str = Field(
        ...,
        description="ID of the source document"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from retrieval"
    )
    similarity_distance: float = Field(
        ...,
        ge=0.0,
        description="Vector similarity distance"
    )
    content_snippet: Optional[str] = Field(
        None,
        description="Relevant content snippet"
    )

    class Config:
        use_enum_values = True


class EnrichmentDecision(BaseModel):
    """Decision made for enriching a specific field."""

    field_name: str = Field(
        ...,
        description="Name of the field being enriched"
    )
    original_value: Optional[Any] = Field(
        None,
        description="Original value (if any)"
    )
    enriched_value: Any = Field(
        ...,
        description="Enriched/inferred value"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this enrichment"
    )
    source_kb: KnowledgeBaseType = Field(
        ...,
        description="Primary knowledge base used"
    )
    evidence: List[EnrichmentEvidence] = Field(
        default_factory=list,
        description="Supporting evidence from retrieval"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the decision"
    )

    class Config:
        use_enum_values = True


class EnrichmentResponse(BaseModel):
    """Response model for claim enrichment."""

    enriched_claim: Dict[str, Any] = Field(
        ...,
        description="Complete claim with enriched fields"
    )
    enrichment_decisions: Dict[str, EnrichmentDecision] = Field(
        default_factory=dict,
        description="Per-field enrichment details"
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence across all enrichments"
    )
    enrichment_quality_score: EnrichmentQualityTier = Field(
        ...,
        description="Quality tier classification"
    )
    explanation: str = Field(
        ...,
        description="Human-readable summary of enrichment"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of enrichment"
    )

    class Config:
        use_enum_values = True


class ConfidenceFactors(BaseModel):
    """Confidence scoring factors."""

    retrieval_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality of retrieval results"
    )
    source_diversity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Diversity of sources consulted"
    )
    temporal_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Temporal relevance of source data"
    )
    cross_validation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agreement across multiple sources"
    )
    regulatory_citation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Regulatory validation score"
    )

    @validator('*', pre=True)
    def round_to_precision(cls, v):
        """Round all scores to 4 decimal places."""
        if isinstance(v, (int, float)):
            return round(float(v), 4)
        return v


class EnrichmentMetrics(BaseModel):
    """Metrics for enrichment quality and performance."""

    total_enrichments: int = Field(
        default=0,
        ge=0,
        description="Total number of enrichments performed"
    )
    successful_enrichments: int = Field(
        default=0,
        ge=0,
        description="Number of successful enrichments"
    )
    failed_enrichments: int = Field(
        default=0,
        ge=0,
        description="Number of failed enrichments"
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    average_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average latency in milliseconds"
    )
    p50_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="P50 latency percentile"
    )
    p95_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="P95 latency percentile"
    )
    p99_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="P99 latency percentile"
    )
    cache_hit_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Cache hit rate"
    )
    accuracy_per_field: Dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy scores per field"
    )
    kb_accuracy: Dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy per knowledge base"
    )

    class Config:
        validate_assignment = True


class BatchEnrichmentRequest(BaseModel):
    """Request model for batch enrichment."""

    requests: List[EnrichmentRequest] = Field(
        ...,
        min_items=1,
        description="List of enrichment requests"
    )
    parallel: bool = Field(
        default=True,
        description="Execute requests in parallel"
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent requests"
    )


class BatchEnrichmentResponse(BaseModel):
    """Response model for batch enrichment."""

    responses: List[EnrichmentResponse] = Field(
        ...,
        description="List of enrichment responses"
    )
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of requests"
    )
    successful_count: int = Field(
        ...,
        ge=0,
        description="Number of successful enrichments"
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed enrichments"
    )
    total_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds"
    )
    average_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average processing time per request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of batch processing"
    )


class CacheEntry(BaseModel):
    """Model for cached enrichment results."""

    claim_hash: str = Field(
        ...,
        description="Hash of the original claim data"
    )
    enrichment_response: EnrichmentResponse = Field(
        ...,
        description="Cached enrichment response"
    )
    cached_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when cached"
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times accessed"
    )
    ttl_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Time-to-live in seconds"
    )
