"""
Fraud Signal Generator for Missing Data Analysis.

This module generates fraud signals based on missing data patterns,
enrichment quality, and submission anomalies. Fraud signals indicate
potential fraudulent activity that should be investigated.

Key Components:
- FraudSignal: Pydantic model for fraud signals
- FraudSignalFromMissingData: Generator for various signal types
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class FraudSignal(BaseModel):
    """
    Pydantic model representing a fraud signal.

    A fraud signal indicates suspicious behavior that may warrant
    further investigation. Signals are scored 0.0-1.0 based on severity.
    """

    signal_type: str = Field(..., description="Unique identifier for signal type")
    signal_name: str = Field(..., description="Human-readable signal description")
    signal_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Signal severity score (0.0-1.0, higher = more suspicious)"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence for this signal"
    )
    recommendation: str = Field(
        ...,
        description="Recommended action based on this signal"
    )
    links_to_kb: List[str] = Field(
        default_factory=list,
        description="Knowledge bases that identified this signal"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="When this signal was generated"
    )

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class FraudSignalFromMissingData:
    """
    Generates fraud signals from missing data and enrichment analysis.

    This class analyzes claims with missing data, enrichment attempts,
    and historical patterns to generate fraud signals indicating
    potential fraudulent activity.
    """

    def signal_provider_submits_incomplete_claims(
        self,
        provider_npi: str,
        provider_pattern: Dict[str, Any],
    ) -> FraudSignal:
        """
        Generate signal when provider frequently submits incomplete claims.

        Args:
            provider_npi: Provider NPI identifier
            provider_pattern: Dict with missing_rate, missing_field_types, etc.

        Returns:
            FraudSignal indicating incomplete submission pattern
        """
        missing_rate = provider_pattern.get("missing_rate", 0.0)
        missing_field_types = provider_pattern.get("missing_field_types", {})
        claim_count = provider_pattern.get("claim_count", 0)

        # Calculate signal strength based on missing rate
        # High missing rate (>50%) = high signal strength
        signal_strength = min(1.0, missing_rate * 1.5)

        # Boost if systematic (same fields always missing)
        if missing_field_types:
            systematic_count = sum(
                1 for count in missing_field_types.values()
                if count / claim_count > 0.50
            )
            if systematic_count > 0:
                signal_strength = min(1.0, signal_strength + (systematic_count * 0.10))

        return FraudSignal(
            signal_type="provider_incomplete_submissions",
            signal_name=f"Provider {provider_npi} frequently submits incomplete claims",
            signal_strength=round(signal_strength, 3),
            evidence={
                "provider_npi": provider_npi,
                "missing_rate": missing_rate,
                "missing_field_types": missing_field_types,
                "claim_count": claim_count,
                "systematic_omission": systematic_count if missing_field_types else 0,
            },
            recommendation="Flag provider for manual review and audit recent claims",
            links_to_kb=["provider_kb"],
        )

    def signal_enrichment_fails(
        self,
        claim: Dict[str, Any],
        enrichment_attempt: Dict[str, Any],
    ) -> FraudSignal:
        """
        Generate signal when claim enrichment fails.

        Args:
            claim: Claim that failed enrichment
            enrichment_attempt: Details of failed enrichment

        Returns:
            FraudSignal indicating enrichment failure
        """
        claim_id = claim.get("claim_id", "unknown")
        reason = enrichment_attempt.get("reason", "Unknown failure")
        confidence = enrichment_attempt.get("confidence", 0.0)

        # Failure to enrich is moderately suspicious
        # Complete failure (confidence 0.0) = higher strength
        signal_strength = 0.60 if confidence == 0.0 else 0.50

        return FraudSignal(
            signal_type="enrichment_failure",
            signal_name=f"Claim {claim_id} could not be enriched",
            signal_strength=signal_strength,
            evidence={
                "claim_id": claim_id,
                "reason": reason,
                "confidence": confidence,
                "status": enrichment_attempt.get("status", "failed"),
            },
            recommendation="Manually review claim for unusual characteristics",
            links_to_kb=["enrichment_engine"],
        )

    def signal_confidence_drops(
        self,
        claim: Dict[str, Any],
        enrichment_response: Dict[str, Any],
    ) -> FraudSignal:
        """
        Generate signal when enrichment confidence is low.

        Args:
            claim: Claim that was enriched
            enrichment_response: Enrichment response with confidence score

        Returns:
            FraudSignal indicating low confidence
        """
        claim_id = claim.get("claim_id", "unknown")
        confidence = enrichment_response.get("confidence", 0.0)

        # Confidence threshold is 0.60
        # Lower confidence = higher signal strength
        confidence_gap = max(0.0, 0.60 - confidence)
        signal_strength = min(1.0, confidence_gap * 2.0)  # Scale gap to 0-1

        return FraudSignal(
            signal_type="low_enrichment_confidence",
            signal_name=f"Low confidence enrichment for claim {claim_id}",
            signal_strength=round(signal_strength, 3),
            evidence={
                "claim_id": claim_id,
                "confidence": confidence,
                "enriched_fields": enrichment_response.get("enriched_fields", []),
                "status": enrichment_response.get("status", "partial"),
            },
            recommendation="Verify enriched data against medical coding standards",
            links_to_kb=["enrichment_engine"],
        )

    def signal_enriched_data_violates_standards(
        self,
        enriched_claim: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> FraudSignal:
        """
        Generate signal when enriched data violates medical coding standards.

        Args:
            enriched_claim: Claim after enrichment
            validation_result: Medical coding validation results

        Returns:
            FraudSignal indicating standard violations
        """
        claim_id = enriched_claim.get("claim_id", "unknown")
        violations = validation_result.get("violations", [])
        severity = validation_result.get("severity", "medium")

        # Map severity to signal strength
        severity_map = {
            "low": 0.40,
            "medium": 0.60,
            "high": 0.80,
            "critical": 0.95,
        }

        signal_strength = severity_map.get(severity, 0.60)

        return FraudSignal(
            signal_type="invalid_medical_combination",
            signal_name=f"Enriched claim {claim_id} violates medical coding standards",
            signal_strength=signal_strength,
            evidence={
                "claim_id": claim_id,
                "violations": violations,
                "severity": severity,
                "diagnosis_codes": enriched_claim.get("diagnosis_codes", []),
                "procedure_codes": enriched_claim.get("procedure_codes", []),
            },
            recommendation="Reject enrichment and request complete claim submission",
            links_to_kb=["medical_coding_kb", "enrichment_engine"],
        )

    def signal_inconsistent_enrichment_pattern(
        self,
        claim: Dict[str, Any],
        enrichment_data: Dict[str, Any],
        historical_enrichments: List[Dict[str, Any]],
    ) -> FraudSignal:
        """
        Generate signal when enrichment doesn't match historical patterns.

        Args:
            claim: Current claim
            enrichment_data: Data used to enrich claim
            historical_enrichments: Historical enrichment patterns

        Returns:
            FraudSignal indicating pattern inconsistency
        """
        claim_id = claim.get("claim_id", "unknown")
        provider_npi = claim.get("provider_npi", "unknown")

        # Analyze deviation from historical patterns
        enriched_procedures = enrichment_data.get("enriched_fields", {}).get("procedure_codes", [])

        # Count how often these procedures appear in history
        if historical_enrichments and enriched_procedures:
            historical_procedure_counts = {}
            for hist in historical_enrichments:
                for proc in hist.get("procedure_codes", []):
                    historical_procedure_counts[proc] = historical_procedure_counts.get(proc, 0) + 1

            # Check if current procedures are rare/unseen
            total_historical = len(historical_enrichments)
            unseen_count = sum(
                1 for proc in enriched_procedures
                if historical_procedure_counts.get(proc, 0) / total_historical < 0.10
            )

            # Signal strength based on how many procedures are unusual
            if enriched_procedures:
                signal_strength = min(1.0, (unseen_count / len(enriched_procedures)) * 0.80)
            else:
                signal_strength = 0.40  # Default moderate suspicion
        else:
            signal_strength = 0.40  # Not enough data

        return FraudSignal(
            signal_type="inconsistent_enrichment",
            signal_name=f"Enrichment for claim {claim_id} deviates from provider's historical pattern",
            signal_strength=round(signal_strength, 3),
            evidence={
                "claim_id": claim_id,
                "provider_npi": provider_npi,
                "enriched_fields": enrichment_data.get("enriched_fields", {}),
                "historical_sample_size": len(historical_enrichments),
            },
            recommendation="Compare enriched data with provider's typical billing patterns",
            links_to_kb=["provider_kb", "enrichment_engine"],
        )

    def signal_unusual_enrichment_source(
        self,
        claim: Dict[str, Any],
        enrichment_sources: Dict[str, Any],
        typical_sources: Dict[str, float],
    ) -> FraudSignal:
        """
        Generate signal when enrichment comes from unusual knowledge base.

        Args:
            claim: Current claim
            enrichment_sources: Sources used for enrichment
            typical_sources: Typical source distribution for this provider

        Returns:
            FraudSignal indicating unusual source
        """
        claim_id = claim.get("claim_id", "unknown")
        final_kb = enrichment_sources.get("final_kb", "unknown")
        fallback_count = enrichment_sources.get("fallback_count", 0)

        # Check how unusual this source is
        typical_rate = typical_sources.get(final_kb, 0.0)

        # If this KB is rarely used (<10%) and required fallbacks, it's suspicious
        if typical_rate < 0.10 and fallback_count > 0:
            signal_strength = min(1.0, 0.50 + (fallback_count * 0.10))
        elif fallback_count >= 3:
            signal_strength = min(1.0, fallback_count * 0.15)
        else:
            signal_strength = 0.30  # Low baseline suspicion

        return FraudSignal(
            signal_type="unusual_enrichment_source",
            signal_name=f"Claim {claim_id} enriched from unusual knowledge base",
            signal_strength=round(signal_strength, 3),
            evidence={
                "claim_id": claim_id,
                "final_kb": final_kb,
                "fallback_count": fallback_count,
                "typical_rate": typical_rate,
            },
            recommendation="Verify enrichment data is appropriate for this provider",
            links_to_kb=[final_kb],
        )

    def signal_enrichment_complexity(
        self,
        claim: Dict[str, Any],
        enrichment_decisions: Dict[str, Any],
    ) -> FraudSignal:
        """
        Generate signal for complex enrichment requiring multiple fallbacks.

        Args:
            claim: Current claim
            enrichment_decisions: Details of enrichment decision process

        Returns:
            FraudSignal indicating high complexity
        """
        claim_id = claim.get("claim_id", "unknown")
        attempts = enrichment_decisions.get("attempts", 1)
        fallbacks = enrichment_decisions.get("fallbacks", [])
        final_strategy = enrichment_decisions.get("final_strategy", "unknown")

        # Many attempts = unusual claim
        signal_strength = min(1.0, (attempts - 1) * 0.15)

        # Boost if had to use default strategy (couldn't find patterns)
        if final_strategy == "default":
            signal_strength = min(1.0, signal_strength + 0.25)

        return FraudSignal(
            signal_type="high_enrichment_complexity",
            signal_name=f"Claim {claim_id} required complex enrichment with {attempts} attempts",
            signal_strength=round(signal_strength, 3),
            evidence={
                "claim_id": claim_id,
                "attempts": attempts,
                "fallbacks": fallbacks,
                "final_strategy": final_strategy,
                "total_time_ms": enrichment_decisions.get("total_time_ms", 0),
            },
            recommendation="Unusual claim structure - verify all data fields",
            links_to_kb=["enrichment_engine"],
        )


def aggregate_fraud_signals(signals: List[FraudSignal]) -> Dict[str, Any]:
    """
    Aggregate multiple fraud signals into a summary.

    Args:
        signals: List of FraudSignal objects

    Returns:
        Dictionary with aggregated metrics
    """
    if not signals:
        return {
            "signal_count": 0,
            "max_strength": 0.0,
            "avg_strength": 0.0,
            "total_strength": 0.0,
            "high_severity_count": 0,
        }

    strengths = [s.signal_strength for s in signals]

    return {
        "signal_count": len(signals),
        "max_strength": max(strengths),
        "avg_strength": sum(strengths) / len(strengths),
        "total_strength": sum(strengths),
        "high_severity_count": sum(1 for s in strengths if s >= 0.70),
        "signal_types": [s.signal_type for s in signals],
    }
