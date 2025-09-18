"""Tests for claim models."""

import pytest
from datetime import date
from decimal import Decimal

from src.models.claim_models import (
    MedicalClaim, PharmacyClaim, NoFaultClaim,
    claim_factory, ClaimBatch, ValidationError
)


class TestMedicalClaim:
    """Test medical claim model."""

    def test_valid_medical_claim(self):
        """Test creating a valid medical claim."""
        claim_data = {
            "claim_id": "CLM-2024-001234",
            "patient_id": "PAT-78901",
            "provider_id": "PRV-45678",
            "provider_npi": "1234567890",
            "date_of_service": "2024-03-15",
            "diagnosis_codes": ["E11.9"],
            "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
            "procedure_codes": ["99213"],
            "procedure_descriptions": ["Office visit"],
            "billed_amount": 125.00,
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": False
        }

        claim = MedicalClaim(**claim_data)

        assert claim.claim_id == "CLM-2024-001234"
        assert claim.billed_amount == Decimal("125.00")
        assert claim.date_of_service == date(2024, 3, 15)
        assert len(claim.diagnosis_codes) == 1
        assert len(claim.procedure_codes) == 1

    def test_invalid_claim_id(self):
        """Test invalid claim ID format."""
        claim_data = {
            "claim_id": "INVALID-ID",
            "patient_id": "PAT-78901",
            "provider_id": "PRV-45678",
            "provider_npi": "1234567890",
            "date_of_service": "2024-03-15",
            "diagnosis_codes": ["E11.9"],
            "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
            "procedure_codes": ["99213"],
            "procedure_descriptions": ["Office visit"],
            "billed_amount": 125.00,
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": False
        }

        with pytest.raises(ValueError):
            MedicalClaim(**claim_data)

    def test_fraud_consistency_validation(self):
        """Test fraud indicator and type consistency."""
        claim_data = {
            "claim_id": "CLM-2024-001234",
            "patient_id": "PAT-78901",
            "provider_id": "PRV-45678",
            "provider_npi": "1234567890",
            "date_of_service": "2024-03-15",
            "diagnosis_codes": ["E11.9"],
            "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
            "procedure_codes": ["99213"],
            "procedure_descriptions": ["Office visit"],
            "billed_amount": 125.00,
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": True,
            # Missing fraud_type
        }

        with pytest.raises(ValueError, match="fraud_type must be specified"):
            MedicalClaim(**claim_data)


class TestClaimFactory:
    """Test claim factory function."""

    def test_medical_claim_creation(self):
        """Test creating medical claim via factory."""
        claim_data = {
            "claim_id": "CLM-2024-001234",
            "patient_id": "PAT-78901",
            "provider_id": "PRV-45678",
            "provider_npi": "1234567890",
            "date_of_service": "2024-03-15",
            "diagnosis_codes": ["E11.9"],
            "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
            "procedure_codes": ["99213"],
            "procedure_descriptions": ["Office visit"],
            "billed_amount": 125.00,
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": False
        }

        claim = claim_factory(claim_data)
        assert isinstance(claim, MedicalClaim)

    def test_pharmacy_claim_creation(self):
        """Test creating pharmacy claim via factory."""
        claim_data = {
            "claim_id": "CLM-2024-001235",
            "patient_id": "PAT-78902",
            "provider_id": "PRV-45679",
            "provider_npi": "1234567891",
            "date_of_service": "2024-03-16",
            "ndc_code": "12345-1234-12",
            "drug_name": "Metformin",
            "strength": "500mg",
            "quantity": 90,
            "days_supply": 30,
            "prescriber_npi": "1234567892",
            "pharmacy_npi": "1234567893",
            "fill_date": "2024-03-16",
            "billed_amount": 25.00,
            "claim_type": "pharmacy",
            "fraud_indicator": False
        }

        claim = claim_factory(claim_data)
        assert isinstance(claim, PharmacyClaim)

    def test_unknown_claim_type(self):
        """Test handling unknown claim type."""
        claim_data = {
            "claim_id": "CLM-2024-001234",
            "patient_id": "PAT-78901",
            "provider_id": "PRV-45678",
            "provider_npi": "1234567890",
            "date_of_service": "2024-03-15",
            "billed_amount": 125.00,
            "claim_type": "unknown_type",
            "fraud_indicator": False
        }

        with pytest.raises(ValueError, match="Cannot determine claim type"):
            claim_factory(claim_data)


class TestClaimBatch:
    """Test claim batch model."""

    def test_batch_creation(self):
        """Test creating a claim batch."""
        claims = [
            MedicalClaim(
                claim_id="CLM-2024-001234",
                patient_id="PAT-78901",
                provider_id="PRV-45678",
                provider_npi="1234567890",
                date_of_service=date(2024, 3, 15),
                diagnosis_codes=["E11.9"],
                diagnosis_descriptions=["Type 2 diabetes mellitus"],
                procedure_codes=["99213"],
                procedure_descriptions=["Office visit"],
                billed_amount=Decimal("125.00"),
                service_location="11",
                claim_type="professional",
                fraud_indicator=False
            )
        ]

        batch = ClaimBatch(claims=claims, batch_id="test-batch")

        assert batch.total_count == 1
        assert batch.batch_id == "test-batch"
        assert batch.processed_at is not None
        assert len(batch.claims) == 1