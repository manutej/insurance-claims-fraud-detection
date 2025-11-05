"""
Enrichment quality metrics and monitoring.

Tracks enrichment accuracy, latency, coverage, and knowledge base performance.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np
import structlog

from src.rag.schemas import (
    EnrichmentRequest,
    EnrichmentResponse,
    EnrichmentMetrics,
)

logger = structlog.get_logger(__name__)


class EnrichmentMetricsTracker:
    """
    Track and compute enrichment quality metrics.

    Metrics tracked:
    - Accuracy per field (compared to ground truth)
    - Accuracy per knowledge base
    - Coverage (% of fields successfully enriched)
    - Latency percentiles (P50, P95, P99)
    - Cache hit rate

    Usage:
        tracker = EnrichmentMetricsTracker()

        # Track enrichment
        tracker.track_enrichment(request, response, actual_values)

        # Get metrics
        metrics = tracker.compute_metrics()
        print(f"Average confidence: {metrics.average_confidence}")
        print(f"P95 latency: {metrics.p95_latency_ms}ms")
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.enrichments: List[Dict[str, Any]] = []
        self.latencies_ms: List[float] = []
        self.field_accuracy: Dict[str, List[bool]] = defaultdict(list)
        self.kb_accuracy: Dict[str, List[bool]] = defaultdict(list)
        self.confidence_scores: List[float] = []

        logger.info("metrics_tracker_initialized")

    def track_enrichment(
        self,
        request: EnrichmentRequest,
        response: EnrichmentResponse,
        actual_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track an enrichment operation.

        Args:
            request: Original enrichment request
            response: Enrichment response
            actual_values: Ground truth values (if available for accuracy)
        """
        enrichment_record = {
            "timestamp": datetime.utcnow(),
            "request": request,
            "response": response,
            "actual_values": actual_values,
        }
        self.enrichments.append(enrichment_record)

        # Track latency
        if response.processing_time_ms:
            self.latencies_ms.append(response.processing_time_ms)

        # Track confidence
        self.confidence_scores.append(response.overall_confidence)

        # Track accuracy (if ground truth provided)
        if actual_values:
            for field_name, decision in response.enrichment_decisions.items():
                is_correct = actual_values.get(field_name) == decision.enriched_value
                self.field_accuracy[field_name].append(is_correct)
                self.kb_accuracy[decision.source_kb].append(is_correct)

        logger.debug(
            "enrichment_tracked",
            confidence=response.overall_confidence,
            latency_ms=response.processing_time_ms,
        )

    def compute_accuracy_per_field(self) -> Dict[str, float]:
        """
        Compute accuracy for each field.

        Returns:
            Dictionary mapping field names to accuracy [0.0, 1.0]
        """
        accuracy = {}
        for field_name, results in self.field_accuracy.items():
            if results:
                accuracy[field_name] = sum(results) / len(results)
        return accuracy

    def compute_accuracy_per_kb(self) -> Dict[str, float]:
        """
        Compute accuracy for each knowledge base.

        Returns:
            Dictionary mapping KB types to accuracy [0.0, 1.0]
        """
        accuracy = {}
        for kb_type, results in self.kb_accuracy.items():
            if results:
                accuracy[kb_type] = sum(results) / len(results)
        return accuracy

    def compute_coverage(self) -> float:
        """
        Compute enrichment coverage.

        Coverage = % of requested fields successfully enriched above
        confidence threshold.

        Returns:
            Coverage ratio [0.0, 1.0]
        """
        if not self.enrichments:
            return 0.0

        total_fields_requested = 0
        total_fields_enriched = 0

        for record in self.enrichments:
            response = record["response"]
            # Count fields with decisions above ACCEPTABLE quality
            enriched = sum(
                1 for d in response.enrichment_decisions.values() if d.confidence_score >= 0.70
            )
            total_fields_enriched += enriched
            total_fields_requested += len(response.enrichment_decisions)

        if total_fields_requested == 0:
            return 0.0

        return total_fields_enriched / total_fields_requested

    def compute_latency_percentiles(self) -> Dict[str, float]:
        """
        Compute latency percentiles.

        Returns:
            Dictionary with P50, P95, P99 latencies in milliseconds
        """
        if not self.latencies_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = np.array(self.latencies_ms)
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }

    def compute_metrics(self) -> EnrichmentMetrics:
        """
        Compute all enrichment metrics.

        Returns:
            EnrichmentMetrics object with computed statistics
        """
        total = len(self.enrichments)
        # Assume all tracked are successful for now
        successful = total

        avg_confidence = (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores
            else 0.0
        )

        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

        percentiles = self.compute_latency_percentiles()

        metrics = EnrichmentMetrics(
            total_enrichments=total,
            successful_enrichments=successful,
            failed_enrichments=0,
            average_confidence=round(avg_confidence, 4),
            average_latency_ms=round(avg_latency, 2),
            p50_latency_ms=round(percentiles["p50"], 2),
            p95_latency_ms=round(percentiles["p95"], 2),
            p99_latency_ms=round(percentiles["p99"], 2),
            cache_hit_rate=None,  # Set by cache component
            accuracy_per_field=self.compute_accuracy_per_field(),
            kb_accuracy=self.compute_accuracy_per_kb(),
        )

        logger.info(
            "metrics_computed",
            total_enrichments=metrics.total_enrichments,
            average_confidence=metrics.average_confidence,
            p95_latency_ms=metrics.p95_latency_ms,
        )

        return metrics

    def generate_enrichment_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive enrichment report.

        Returns:
            Dictionary with detailed metrics and analysis
        """
        metrics = self.compute_metrics()
        coverage = self.compute_coverage()

        report = {
            "summary": {
                "total_enrichments": metrics.total_enrichments,
                "successful_enrichments": metrics.successful_enrichments,
                "failed_enrichments": metrics.failed_enrichments,
                "average_confidence": metrics.average_confidence,
                "coverage": round(coverage, 4),
            },
            "performance": {
                "average_latency_ms": metrics.average_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
            },
            "accuracy": {
                "per_field": metrics.accuracy_per_field,
                "per_kb": metrics.kb_accuracy,
            },
            "cache": {
                "hit_rate": metrics.cache_hit_rate,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info("enrichment_report_generated")
        return report

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.enrichments.clear()
        self.latencies_ms.clear()
        self.field_accuracy.clear()
        self.kb_accuracy.clear()
        self.confidence_scores.clear()
        logger.info("metrics_reset")
