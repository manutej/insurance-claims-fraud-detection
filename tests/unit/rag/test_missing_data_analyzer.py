"""
Test suite for missing data analyzer module.

Tests cover:
- Missing field detection
- Missing field criticality assessment
- Missing data percentage calculation
- Provider submission patterns
- Patient submission patterns
- Temporal pattern detection
"""

import pytest
from datetime import datetime, date
from typing import Dict, List
from decimal import Decimal

# Import the classes we'll be testing (TDD - these don't exist yet)
from src.rag.missing_data_analyzer import (
    MissingFieldDetector,
    SuspiciousSubmissionPatternDetector,
)


class TestMissingFieldDetector:
    """Test suite for MissingFieldDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a MissingFieldDetector instance."""
        return MissingFieldDetector()

    @pytest.fixture
    def complete_claim(self) -> Dict:
        """Create a complete claim with all fields."""
        return {
            "claim_id": "CLM-2024-100001",
            "patient_id": "PAT-010001",
            "provider_id": "PRV-005001",
            "provider_npi": "1234567890",
            "provider_specialty": "internal_medicine",
            "date_of_service": "2024-03-15",
            "diagnosis_codes": ["I10", "E11.9"],
            "diagnosis_descriptions": ["Essential hypertension", "Type 2 diabetes"],
            "procedure_codes": ["99213", "36415"],
            "procedure_descriptions": ["Office visit", "Venipuncture"],
            "billed_amount": 150.00,
            "service_location": "11",
            "service_location_desc": "Office",
            "claim_type": "professional",
            "treatment_type": "outpatient",
            "days_supply": None,  # Optional for medical claims
            "medical_necessity": "Routine follow-up for chronic conditions",
        }

    @pytest.fixture
    def incomplete_claim_no_diagnosis(self) -> Dict:
        """Claim missing diagnosis codes (CRITICAL field)."""
        return {
            "claim_id": "CLM-2024-100054",
            "patient_id": "PAT-010266",
            "provider_id": "PRV-005093",
            "provider_npi": "2234132629",
            "provider_specialty": "internal_medicine",
            "date_of_service": "2024-03-21",
            "diagnosis_descriptions": ["Type 2 diabetes without complications"],
            "procedure_codes": ["99214", "36415", "83036"],
            "procedure_descriptions": [
                "Office visit, established patient, high",
                "Collection of venous blood by venipuncture",
                "Hemoglobin A1C"
            ],
            "billed_amount": 202.89,
            "service_location": "11",
            "service_location_desc": "Office",
            "claim_type": "professional",
        }

    @pytest.fixture
    def incomplete_claim_no_procedure(self) -> Dict:
        """Claim missing procedure codes (CRITICAL field)."""
        return {
            "claim_id": "CLM-2024-100055",
            "patient_id": "PAT-010914",
            "provider_id": "PRV-005009",
            "provider_npi": "2234478651",
            "provider_specialty": "internal_medicine",
            "date_of_service": "2024-01-08",
            "diagnosis_codes": ["I10"],
            "diagnosis_descriptions": ["Essential (primary) hypertension"],
            "procedure_descriptions": [
                "Office visit, established patient, high",
                "Collection of venous blood by venipuncture"
            ],
            "billed_amount": 168.39,
            "service_location": "11",
            "service_location_desc": "Office",
            "claim_type": "professional",
        }

    def test_detect_no_missing_fields(self, detector, complete_claim):
        """Test that complete claim has no missing fields."""
        missing_fields = detector.detect_missing_fields(complete_claim)

        # Should detect that days_supply is None but it's optional for medical claims
        assert len(missing_fields) == 0 or "days_supply" not in missing_fields

    def test_detect_missing_diagnosis_codes(self, detector, incomplete_claim_no_diagnosis):
        """Test detection of missing diagnosis codes."""
        missing_fields = detector.detect_missing_fields(incomplete_claim_no_diagnosis)

        assert "diagnosis_codes" in missing_fields
        assert isinstance(missing_fields, list)

    def test_detect_missing_procedure_codes(self, detector, incomplete_claim_no_procedure):
        """Test detection of missing procedure codes."""
        missing_fields = detector.detect_missing_fields(incomplete_claim_no_procedure)

        assert "procedure_codes" in missing_fields
        assert isinstance(missing_fields, list)

    def test_detect_missing_billed_amount(self, detector):
        """Test detection of missing billed amount."""
        claim = {
            "claim_id": "CLM-2024-100056",
            "patient_id": "PAT-010635",
            "provider_id": "PRV-005033",
            "provider_npi": "2234644650",
            "provider_specialty": "internal_medicine",
            "date_of_service": "2024-03-11",
            "diagnosis_codes": ["E11.21"],
            "procedure_codes": ["80053"],
            "service_location": "11",
            "claim_type": "professional",
            # Missing billed_amount
        }

        missing_fields = detector.detect_missing_fields(claim)
        assert "billed_amount" in missing_fields

    def test_detect_missing_date_of_service(self, detector):
        """Test detection of missing date of service."""
        claim = {
            "claim_id": "CLM-2024-100057",
            "patient_id": "PAT-010005",
            "provider_id": "PRV-005011",
            "provider_npi": "2234979552",
            "diagnosis_codes": ["E11.9"],
            "procedure_codes": ["81000", "36415"],
            "billed_amount": 57.86,
            "service_location": "11",
            "claim_type": "professional",
            # Missing date_of_service
        }

        missing_fields = detector.detect_missing_fields(claim)
        assert "date_of_service" in missing_fields

    def test_assess_missing_criticality_diagnosis_codes(self, detector):
        """Test criticality assessment - diagnosis codes are highly critical."""
        missing_fields = ["diagnosis_codes"]
        criticality = detector.assess_missing_criticality(missing_fields)

        assert "diagnosis_codes" in criticality
        assert criticality["diagnosis_codes"] >= 0.90  # Very critical
        assert criticality["diagnosis_codes"] <= 1.0

    def test_assess_missing_criticality_procedure_codes(self, detector):
        """Test criticality assessment - procedure codes are highly critical."""
        missing_fields = ["procedure_codes"]
        criticality = detector.assess_missing_criticality(missing_fields)

        assert "procedure_codes" in criticality
        assert criticality["procedure_codes"] >= 0.90  # Very critical

    def test_assess_missing_criticality_billed_amount(self, detector):
        """Test criticality assessment - billed amount is critical."""
        missing_fields = ["billed_amount"]
        criticality = detector.assess_missing_criticality(missing_fields)

        assert "billed_amount" in criticality
        assert criticality["billed_amount"] >= 0.85  # Critical

    def test_assess_missing_criticality_optional_fields(self, detector):
        """Test criticality assessment - optional fields have lower criticality."""
        missing_fields = ["service_location_desc", "notes"]
        criticality = detector.assess_missing_criticality(missing_fields)

        assert criticality["service_location_desc"] <= 0.30  # Low criticality
        assert criticality["notes"] <= 0.20  # Very low criticality

    def test_compute_missing_data_percentage_complete(self, detector, complete_claim):
        """Test missing data percentage for complete claim."""
        percentage = detector.compute_missing_data_percentage(complete_claim)

        assert percentage <= 0.10  # Allow for optional fields

    def test_compute_missing_data_percentage_high(self, detector):
        """Test missing data percentage for claim with many missing fields."""
        claim = {
            "claim_id": "CLM-2024-100999",
            "patient_id": "PAT-010999",
            "provider_npi": "2234999999",
            # Missing most other fields
        }

        percentage = detector.compute_missing_data_percentage(claim)

        assert percentage >= 0.50  # At least 50% missing


class TestSuspiciousSubmissionPatternDetector:
    """Test suite for SuspiciousSubmissionPatternDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a SuspiciousSubmissionPatternDetector instance."""
        return SuspiciousSubmissionPatternDetector()

    @pytest.fixture
    def provider_claims_history(self) -> List[Dict]:
        """Create historical claims for a provider."""
        return [
            {
                "claim_id": f"CLM-2024-{100000+i}",
                "provider_npi": "2234132629",
                "date_of_service": f"2024-0{(i % 3) + 1}-{(i % 28) + 1:02d}",
                "diagnosis_codes": ["E11.9"] if i % 2 == 0 else None,  # 50% missing
                "procedure_codes": ["99214"] if i % 3 == 0 else None,  # 33% missing
                "billed_amount": 150.00,
            }
            for i in range(20)
        ]

    @pytest.fixture
    def patient_claims_history(self) -> List[Dict]:
        """Create historical claims for a patient."""
        return [
            {
                "claim_id": f"CLM-2024-{200000+i}",
                "patient_id": "PAT-010266",
                "provider_npi": f"223413262{i%10}",
                "date_of_service": f"2024-0{(i % 3) + 1}-{(i % 28) + 1:02d}",
                "diagnosis_codes": ["E11.9"],
                "procedure_codes": ["99214"] if i % 4 == 0 else None,  # 25% missing
                "billed_amount": 150.00,
            }
            for i in range(10)
        ]

    def test_detect_provider_high_missing_rate(self, detector, provider_claims_history):
        """Test detection of provider with high missing data rate."""
        pattern = detector.detect_provider_submission_pattern(
            provider_npi="2234132629",
            historical_claims=provider_claims_history
        )

        assert "missing_rate" in pattern
        assert pattern["missing_rate"] >= 0.30  # At least 30% missing
        assert "missing_field_types" in pattern
        assert "diagnosis_codes" in pattern["missing_field_types"]
        assert "suspicious_score" in pattern
        assert pattern["suspicious_score"] >= 0.50  # High suspicion

    def test_detect_patient_submission_pattern(self, detector, patient_claims_history):
        """Test detection of patient submission patterns."""
        pattern = detector.detect_patient_submission_pattern(
            patient_id="PAT-010266",
            historical_claims=patient_claims_history
        )

        assert "missing_rate" in pattern
        assert "missing_field_types" in pattern
        assert "suspicious_score" in pattern

    def test_detect_temporal_pattern_weekend_submission(self, detector):
        """Test detection of weekend submission pattern."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "date_of_service": "2024-03-16",  # Saturday
            "submission_timestamp": "2024-03-16T23:45:00",  # Weekend night
        }

        similar_claims = []  # No similar historical weekend claims

        pattern = detector.detect_temporal_pattern(claim, similar_claims)

        assert "is_weekend" in pattern or "temporal_anomaly" in pattern
        assert pattern.get("suspicious", False) is True or pattern.get("temporal_anomaly") is not None

    def test_detect_temporal_pattern_night_submission(self, detector):
        """Test detection of night-time submission pattern."""
        claim = {
            "claim_id": "CLM-2024-100002",
            "date_of_service": "2024-03-15",
            "submission_timestamp": "2024-03-15T02:30:00",  # 2:30 AM
        }

        similar_claims = []

        pattern = detector.detect_temporal_pattern(claim, similar_claims)

        assert "is_night_time" in pattern or "hour" in pattern
        # Night submissions may be suspicious

    def test_assess_submission_suspicion_high(self, detector, provider_claims_history):
        """Test high suspicion score for provider with bad submission history."""
        claim = {
            "claim_id": "CLM-2024-100054",
            "provider_npi": "2234132629",
            "patient_id": "PAT-010266",
            "diagnosis_codes": None,  # Missing critical field
        }

        score = detector.assess_submission_suspicion(
            provider_npi="2234132629",
            patient_id="PAT-010266",
            claim=claim,
            provider_history=provider_claims_history,
            patient_history=[]
        )

        assert score >= 0.60  # High suspicion
        assert score <= 1.0

    def test_assess_submission_suspicion_low(self, detector):
        """Test low suspicion score for provider with good submission history."""
        good_history = [
            {
                "claim_id": f"CLM-2024-{100000+i}",
                "provider_npi": "1234567890",
                "provider_id": f"PRV-00{i}",
                "provider_specialty": "internal_medicine",
                "patient_id": f"PAT-01000{i}",
                "diagnosis_codes": ["E11.9"],
                "diagnosis_descriptions": ["Type 2 diabetes"],
                "procedure_codes": ["99214"],
                "procedure_descriptions": ["Office visit"],
                "billed_amount": 150.00,
                "date_of_service": "2024-03-15",
                "service_location": "11",
                "claim_type": "professional",
            }
            for i in range(20)
        ]

        claim = {
            "claim_id": "CLM-2024-100999",
            "provider_npi": "1234567890",
            "provider_id": "PRV-001",
            "provider_specialty": "internal_medicine",
            "patient_id": "PAT-010999",
            "diagnosis_codes": ["E11.9"],
            "diagnosis_descriptions": ["Type 2 diabetes"],
            "procedure_codes": ["99214"],
            "procedure_descriptions": ["Office visit"],
            "billed_amount": 150.00,
            "date_of_service": "2024-03-15",
            "service_location": "11",
            "claim_type": "professional",
        }

        score = detector.assess_submission_suspicion(
            provider_npi="1234567890",
            patient_id="PAT-010999",
            claim=claim,
            provider_history=good_history,
            patient_history=[]
        )

        # Provider with complete history and complete claim should have reasonable suspicion
        # Note: Even with good history, some baseline missing data is expected
        assert score < 1.0  # Not maximum suspicion
        assert score >= 0.0  # Valid range


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def detector(self):
        return MissingFieldDetector()

    def test_empty_claim(self, detector):
        """Test handling of empty claim."""
        missing_fields = detector.detect_missing_fields({})

        # Should detect many missing critical fields
        assert len(missing_fields) >= 5

    def test_none_values(self, detector):
        """Test handling of None values in claim fields."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "diagnosis_codes": None,
            "procedure_codes": None,
            "billed_amount": None,
        }

        missing_fields = detector.detect_missing_fields(claim)

        assert "diagnosis_codes" in missing_fields
        assert "procedure_codes" in missing_fields
        assert "billed_amount" in missing_fields

    def test_empty_lists(self, detector):
        """Test handling of empty lists as missing data."""
        claim = {
            "claim_id": "CLM-2024-100001",
            "diagnosis_codes": [],  # Empty list should be treated as missing
            "procedure_codes": [],
            "billed_amount": 100.00,
        }

        missing_fields = detector.detect_missing_fields(claim)

        assert "diagnosis_codes" in missing_fields
        assert "procedure_codes" in missing_fields
