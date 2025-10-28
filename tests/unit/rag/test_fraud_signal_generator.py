"""
Test suite for fraud signal generator module.

Tests cover:
- FraudSignal model validation
- Incomplete submission signal generation
- Enrichment failure signals
- Low confidence signals
- Invalid combination signals
- Inconsistent pattern signals
"""

import pytest
from typing import Dict, List
from decimal import Decimal

# Import the classes we'll be testing (TDD - these don't exist yet)
from src.rag.fraud_signal_generator import (
    FraudSignal,
    FraudSignalFromMissingData,
)


class TestFraudSignalModel:
    """Test suite for FraudSignal Pydantic model."""

    def test_fraud_signal_creation(self):
        """Test creating a valid fraud signal."""
        signal = FraudSignal(
            signal_type="incomplete_submission",
            signal_name="Provider frequently submits incomplete claims",
            signal_strength=0.75,
            evidence={"missing_rate": 0.60, "field_types": ["diagnosis_codes"]},
            recommendation="Flag for manual review",
            links_to_kb=["provider_kb"],
        )

        assert signal.signal_type == "incomplete_submission"
        assert signal.signal_strength == 0.75
        assert "missing_rate" in signal.evidence

    def test_fraud_signal_strength_validation(self):
        """Test that signal strength is validated (0.0-1.0)."""
        # Valid signal
        signal = FraudSignal(
            signal_type="test",
            signal_name="Test signal",
            signal_strength=0.5,
            evidence={},
            recommendation="Test",
        )
        assert signal.signal_strength == 0.5

        # Invalid signal (too high)
        with pytest.raises(ValueError):
            FraudSignal(
                signal_type="test",
                signal_name="Test signal",
                signal_strength=1.5,  # Invalid
                evidence={},
                recommendation="Test",
            )

        # Invalid signal (negative)
        with pytest.raises(ValueError):
            FraudSignal(
                signal_type="test",
                signal_name="Test signal",
                signal_strength=-0.1,  # Invalid
                evidence={},
                recommendation="Test",
            )


class TestFraudSignalFromMissingData:
    """Test suite for FraudSignalFromMissingData class."""

    @pytest.fixture
    def signal_generator(self):
        """Create a FraudSignalFromMissingData instance."""
        return FraudSignalFromMissingData()

    @pytest.fixture
    def provider_with_high_missing_rate(self) -> Dict:
        """Provider pattern with high missing data rate."""
        return {
            "provider_npi": "2234132629",
            "missing_rate": 0.65,
            "missing_field_types": {
                "diagnosis_codes": 15,
                "procedure_codes": 10,
            },
            "claim_count": 20,
        }

    def test_signal_provider_submits_incomplete_claims(
        self, signal_generator, provider_with_high_missing_rate
    ):
        """Test signal generation for incomplete claim submissions."""
        signal = signal_generator.signal_provider_submits_incomplete_claims(
            provider_npi="2234132629",
            provider_pattern=provider_with_high_missing_rate,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "provider_incomplete_submissions"
        assert signal.signal_strength >= 0.50  # High missing rate = high strength
        assert "missing_rate" in signal.evidence
        assert signal.evidence["missing_rate"] == 0.65

    def test_signal_enrichment_fails(self, signal_generator):
        """Test signal generation when enrichment fails."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "diagnosis_codes": None,
            "procedure_codes": None,
        }

        enrichment_attempt = {
            "status": "failed",
            "reason": "No similar patterns found",
            "confidence": 0.0,
        }

        signal = signal_generator.signal_enrichment_fails(
            claim=claim,
            enrichment_attempt=enrichment_attempt,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "enrichment_failure"
        assert signal.signal_strength > 0.0
        assert "reason" in signal.evidence

    def test_signal_confidence_drops(self, signal_generator):
        """Test signal generation for low enrichment confidence."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "diagnosis_codes": ["I10"],
        }

        enrichment_response = {
            "confidence": 0.35,  # Low confidence
            "enriched_fields": ["procedure_codes"],
            "status": "partial",
        }

        signal = signal_generator.signal_confidence_drops(
            claim=claim,
            enrichment_response=enrichment_response,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "low_enrichment_confidence"
        assert signal.signal_strength >= 0.40  # Confidence 0.35 = strength based on gap from 0.60
        assert signal.evidence["confidence"] == 0.35

    def test_signal_enriched_data_violates_standards(self, signal_generator):
        """Test signal for invalid diagnosis-procedure combinations."""
        enriched_claim = {
            "claim_id": "CLM-2024-100001",
            "diagnosis_codes": ["Z00.00"],  # General health exam
            "procedure_codes": ["99285"],  # Emergency dept visit (HIGH complexity)
            "enrichment_source": "similar_pattern",
        }

        validation_result = {
            "is_valid": False,
            "violations": [
                "Procedure 99285 (emergency) incompatible with diagnosis Z00.00 (routine exam)"
            ],
            "severity": "high",
        }

        signal = signal_generator.signal_enriched_data_violates_standards(
            enriched_claim=enriched_claim,
            validation_result=validation_result,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "invalid_medical_combination"
        assert signal.signal_strength >= 0.70  # High severity violation
        assert "violations" in signal.evidence

    def test_signal_inconsistent_enrichment_pattern(self, signal_generator):
        """Test signal for enrichment that doesn't match historical patterns."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "provider_npi": "1234567890",
            "patient_id": "PAT-001",
            "diagnosis_codes": ["I10"],
        }

        enrichment_data = {
            "enriched_fields": {"procedure_codes": ["99285"]},  # Emergency visit
            "confidence": 0.65,
        }

        historical_enrichments = [
            # Historical pattern: this provider/patient always has routine visits
            {
                "procedure_codes": ["99213"],  # Routine office visit
                "diagnosis_codes": ["I10"],
            }
            for _ in range(10)
        ]

        signal = signal_generator.signal_inconsistent_enrichment_pattern(
            claim=claim,
            enrichment_data=enrichment_data,
            historical_enrichments=historical_enrichments,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "inconsistent_enrichment"
        assert signal.signal_strength > 0.0
        assert "enriched_fields" in signal.evidence
        assert "historical_sample_size" in signal.evidence

    def test_signal_unusual_enrichment_source(self, signal_generator):
        """Test signal for enrichment from unusual knowledge base."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "provider_npi": "1234567890",
        }

        enrichment_sources = {
            "primary_kb": "provider_kb",
            "fallback_count": 3,  # Had to try 3 different KBs
            "final_kb": "global_kb",  # Had to fall back to global
        }

        typical_sources = {
            "provider_kb": 0.85,  # This provider usually enriches from provider_kb
            "patient_kb": 0.10,
            "global_kb": 0.05,  # Rarely uses global
        }

        signal = signal_generator.signal_unusual_enrichment_source(
            claim=claim,
            enrichment_sources=enrichment_sources,
            typical_sources=typical_sources,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "unusual_enrichment_source"
        assert signal.signal_strength > 0.0
        assert "fallback_count" in signal.evidence

    def test_signal_enrichment_complexity(self, signal_generator):
        """Test signal for complex enrichment requiring multiple fallbacks."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "provider_npi": "1234567890",
        }

        enrichment_decisions = {
            "attempts": 5,  # Many attempts needed
            "fallbacks": ["provider_kb", "patient_kb", "similar_claims", "global_kb", "default"],
            "final_strategy": "default",  # Had to use default (suspicious)
            "total_time_ms": 2500,
        }

        signal = signal_generator.signal_enrichment_complexity(
            claim=claim,
            enrichment_decisions=enrichment_decisions,
        )

        assert isinstance(signal, FraudSignal)
        assert signal.signal_type == "high_enrichment_complexity"
        assert signal.signal_strength >= 0.50  # Many fallbacks = high complexity
        assert signal.evidence["attempts"] == 5


class TestFraudSignalAggregation:
    """Test aggregating multiple fraud signals."""

    @pytest.fixture
    def signal_generator(self):
        return FraudSignalFromMissingData()

    def test_multiple_signals_for_claim(self, signal_generator):
        """Test generating multiple signals for a single claim."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "provider_npi": "2234132629",
            "diagnosis_codes": None,
            "procedure_codes": None,
        }

        provider_pattern = {
            "missing_rate": 0.70,
            "missing_field_types": {"diagnosis_codes": 18, "procedure_codes": 15},
            "claim_count": 20,
        }

        enrichment_result = {
            "status": "failed",
            "confidence": 0.0,
            "reason": "No patterns found",
        }

        # Generate multiple signals
        signals = []

        signal1 = signal_generator.signal_provider_submits_incomplete_claims(
            provider_npi="2234132629",
            provider_pattern=provider_pattern,
        )
        signals.append(signal1)

        signal2 = signal_generator.signal_enrichment_fails(
            claim=claim,
            enrichment_attempt=enrichment_result,
        )
        signals.append(signal2)

        assert len(signals) == 2
        assert all(isinstance(s, FraudSignal) for s in signals)
        assert signal1.signal_type != signal2.signal_type

    def test_signal_strength_aggregation(self, signal_generator):
        """Test that we can aggregate signal strengths."""
        signals = [
            FraudSignal(
                signal_type=f"test_{i}",
                signal_name=f"Test signal {i}",
                signal_strength=0.50 + (i * 0.1),
                evidence={},
                recommendation="Test",
            )
            for i in range(3)
        ]

        # Calculate aggregate strength (e.g., average or max)
        avg_strength = sum(s.signal_strength for s in signals) / len(signals)
        max_strength = max(s.signal_strength for s in signals)

        assert avg_strength >= 0.50
        assert max_strength >= 0.70
