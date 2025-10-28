"""
Tests for Patient Claim History Knowledge Base.

Tests cover:
- Patient history document creation
- PatientHistoryBuilder (load + embed patient history)
- PatientHistoryRetriever (find similar patient patterns)
- Doctor shopping detection
- Temporal pattern analysis
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from src.rag.knowledge_bases.patient_kb import (
    PatientClaimDocument,
    PatientClaimHistoryKB,
    PatientHistoryBuilder,
    PatientHistoryRetriever,
)


# Test fixtures
@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create in-memory Qdrant client for testing."""
    return QdrantClient(":memory:")


@pytest.fixture
def openai_api_key() -> str:
    """Mock OpenAI API key for testing."""
    return "test-api-key"


@pytest.fixture
def sample_patient_data() -> Dict:
    """Sample patient claim data."""
    return {
        "patient_id": "PAT-001",
        "claim_sequence": [
            {
                "claim_id": "CLM-001",
                "date_of_service": "2024-01-15",
                "provider_npi": "NPI-123",
                "diagnosis_codes": ["E11.9", "I10"],
                "procedure_codes": ["99213"],
                "billed_amount": 125.0,
                "fraud_indicator": False,
            },
            {
                "claim_id": "CLM-002",
                "date_of_service": "2024-02-15",
                "provider_npi": "NPI-123",
                "diagnosis_codes": ["E11.9"],
                "procedure_codes": ["99213", "83036"],
                "billed_amount": 165.0,
                "fraud_indicator": False,
            },
        ],
        "patient_patterns": {
            "provider_count_30d": 1,
            "provider_count_90d": 1,
            "total_claims_30d": 2,
            "total_claims_90d": 2,
            "avg_claim_amount": 145.0,
            "diagnosis_diversity": 0.5,
            "pharmacy_count_30d": 1,
            "controlled_substance_prescriptions": 0,
            "early_refill_count": 0,
        },
        "red_flags": [],
        "temporal_analysis": {
            "date_range_start": "2024-01-15",
            "date_range_end": "2024-02-15",
            "claim_frequency_days": 31.0,
            "geographic_impossibility_count": 0,
        },
    }


@pytest.fixture
def fraud_patient_data() -> Dict:
    """Patient data exhibiting doctor shopping fraud."""
    base_date = datetime(2024, 3, 1)
    claims = []

    # Create claims from 7 different providers in 30 days (doctor shopping)
    for i in range(7):
        claims.append(
            {
                "claim_id": f"CLM-F{i:03d}",
                "date_of_service": (base_date + timedelta(days=i * 4)).strftime(
                    "%Y-%m-%d"
                ),
                "provider_npi": f"NPI-{i+200}",
                "diagnosis_codes": ["M79.3"],  # Chronic pain
                "procedure_codes": ["99213"],
                "billed_amount": 120.0,
                "fraud_indicator": True,
            }
        )

    return {
        "patient_id": "PAT-FRAUD-001",
        "claim_sequence": claims,
        "patient_patterns": {
            "provider_count_30d": 7,
            "provider_count_90d": 7,
            "total_claims_30d": 7,
            "total_claims_90d": 7,
            "avg_claim_amount": 120.0,
            "diagnosis_diversity": 0.1,
            "pharmacy_count_30d": 4,
            "controlled_substance_prescriptions": 7,
            "early_refill_count": 3,
        },
        "red_flags": [
            "Doctor shopping: 7 providers in 30 days",
            "Pharmacy hopping: 4 pharmacies in 30 days",
            "Early refills: 3 instances",
        ],
        "temporal_analysis": {
            "date_range_start": "2024-03-01",
            "date_range_end": "2024-03-28",
            "claim_frequency_days": 4.0,
            "geographic_impossibility_count": 0,
        },
    }


class TestPatientClaimDocument:
    """Test PatientClaimDocument Pydantic model."""

    def test_create_valid_document(self, sample_patient_data):
        """Test creating a valid patient claim document."""
        doc = PatientClaimDocument(
            patient_id=sample_patient_data["patient_id"],
            claim_sequence=sample_patient_data["claim_sequence"],
            patient_patterns=sample_patient_data["patient_patterns"],
            red_flags=sample_patient_data["red_flags"],
            temporal_analysis=sample_patient_data["temporal_analysis"],
        )

        assert doc.patient_id == "PAT-001"
        assert len(doc.claim_sequence) == 2
        assert doc.patient_patterns["provider_count_30d"] == 1
        assert len(doc.red_flags) == 0

    def test_document_validation_fails_invalid_data(self):
        """Test that document validation fails with invalid data."""
        with pytest.raises(ValueError):
            PatientClaimDocument(
                patient_id="",  # Empty patient_id should fail
                claim_sequence=[],
                patient_patterns={},
                red_flags=[],
                temporal_analysis={},
            )

    def test_generate_embedding_text(self, sample_patient_data):
        """Test embedding text generation."""
        doc = PatientClaimDocument(
            patient_id=sample_patient_data["patient_id"],
            claim_sequence=sample_patient_data["claim_sequence"],
            patient_patterns=sample_patient_data["patient_patterns"],
            red_flags=sample_patient_data["red_flags"],
            temporal_analysis=sample_patient_data["temporal_analysis"],
        )

        embedding_text = doc.generate_embedding_text()

        assert "2 claims" in embedding_text
        assert "1 providers" in embedding_text or "1 provider" in embedding_text
        assert "E11.9" in embedding_text
        assert "No red flags" in embedding_text

    def test_detect_doctor_shopping(self, fraud_patient_data):
        """Test doctor shopping detection logic."""
        doc = PatientClaimDocument(
            patient_id=fraud_patient_data["patient_id"],
            claim_sequence=fraud_patient_data["claim_sequence"],
            patient_patterns=fraud_patient_data["patient_patterns"],
            red_flags=fraud_patient_data["red_flags"],
            temporal_analysis=fraud_patient_data["temporal_analysis"],
        )

        # Check if doctor shopping is flagged
        assert any("Doctor shopping" in flag for flag in doc.red_flags)
        assert doc.patient_patterns["provider_count_30d"] >= 5


class TestPatientHistoryBuilder:
    """Test PatientHistoryBuilder for loading and processing patient data."""

    def test_builder_initialization(self):
        """Test builder can be initialized."""
        builder = PatientHistoryBuilder()
        assert builder is not None

    def test_build_from_claims_data(self, sample_patient_data):
        """Test building patient history from claims data."""
        builder = PatientHistoryBuilder()
        doc = builder.build_patient_document(sample_patient_data)

        assert doc.patient_id == "PAT-001"
        assert len(doc.claim_sequence) == 2
        assert doc.patient_patterns["provider_count_90d"] == 1

    def test_calculate_patient_patterns(self, sample_patient_data):
        """Test calculation of patient behavioral patterns."""
        builder = PatientHistoryBuilder()
        patterns = builder.calculate_patient_patterns(
            sample_patient_data["claim_sequence"]
        )

        assert "provider_count_30d" in patterns
        assert "avg_claim_amount" in patterns
        assert patterns["avg_claim_amount"] == 145.0

    def test_temporal_analysis(self, sample_patient_data):
        """Test temporal pattern analysis."""
        builder = PatientHistoryBuilder()
        temporal_data = builder.analyze_temporal_patterns(
            sample_patient_data["claim_sequence"]
        )

        assert "date_range_start" in temporal_data
        assert "date_range_end" in temporal_data
        assert "claim_frequency_days" in temporal_data

    def test_detect_red_flags_normal_patient(self, sample_patient_data):
        """Test that normal patient has no red flags."""
        builder = PatientHistoryBuilder()
        red_flags = builder.detect_red_flags(
            sample_patient_data["patient_patterns"],
            sample_patient_data["temporal_analysis"],
        )

        assert len(red_flags) == 0

    def test_detect_red_flags_fraud_patient(self, fraud_patient_data):
        """Test that fraudulent patient triggers red flags."""
        builder = PatientHistoryBuilder()
        red_flags = builder.detect_red_flags(
            fraud_patient_data["patient_patterns"], fraud_patient_data["temporal_analysis"]
        )

        assert len(red_flags) > 0
        assert any("Doctor shopping" in flag for flag in red_flags)


class TestPatientHistoryRetriever:
    """Test PatientHistoryRetriever for similarity search."""

    def test_retriever_initialization(self, qdrant_client, openai_api_key):
        """Test retriever can be initialized."""
        retriever = PatientHistoryRetriever(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        assert retriever is not None

    @pytest.mark.integration
    def test_find_similar_patients(
        self, qdrant_client, openai_api_key, sample_patient_data, fraud_patient_data
    ):
        """Test finding similar patient patterns."""
        # Skip if no API key
        if openai_api_key == "test-api-key":
            pytest.skip("Requires valid OpenAI API key")

        retriever = PatientHistoryRetriever(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )

        # Build patient documents
        builder = PatientHistoryBuilder()
        normal_doc = builder.build_patient_document(sample_patient_data)
        fraud_doc = builder.build_patient_document(fraud_patient_data)

        # Index documents
        retriever.index_patients([normal_doc, fraud_doc])

        # Search for similar patterns to fraud patient
        results = retriever.find_similar_patterns(fraud_doc, limit=2)

        assert len(results) > 0
        # Fraud patient should match itself with high score
        assert results[0]["score"] > 0.9


class TestPatientClaimHistoryKB:
    """Integration tests for complete Patient Claim History KB."""

    def test_kb_initialization(self, qdrant_client, openai_api_key):
        """Test KB can be initialized and collection created."""
        kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )

        # Create collection
        kb.create_collection()

        assert qdrant_client.collection_exists(kb.collection_name)

    @pytest.mark.integration
    def test_kb_build_from_json(
        self, qdrant_client, openai_api_key, tmp_path, sample_patient_data
    ):
        """Test building KB from JSON file."""
        # Skip if no API key
        if openai_api_key == "test-api-key":
            pytest.skip("Requires valid OpenAI API key")

        # Create temporary JSON file
        json_file = tmp_path / "test_claims.json"
        with open(json_file, "w") as f:
            json.dump([sample_patient_data], f)

        kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        kb.create_collection()

        # Build KB
        kb.build(str(json_file))

        # Verify documents indexed
        stats = kb.get_statistics()
        assert stats.total_documents == 1

    @pytest.mark.integration
    def test_kb_search_doctor_shopping(
        self, qdrant_client, openai_api_key, fraud_patient_data
    ):
        """Test searching for doctor shopping patterns."""
        # Skip if no API key
        if openai_api_key == "test-api-key":
            pytest.skip("Requires valid OpenAI API key")

        kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        kb.create_collection()

        # Build patient document
        builder = PatientHistoryBuilder()
        fraud_doc = builder.build_patient_document(fraud_patient_data)

        # Index
        kb.upsert_documents([fraud_doc])

        # Search for doctor shopping
        results = kb.search(
            query_text="Patient visiting multiple providers for pain medication",
            limit=5,
        )

        assert len(results) > 0

    def test_kb_statistics(self, qdrant_client, openai_api_key):
        """Test KB statistics generation."""
        kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        kb.create_collection()

        stats = kb.get_statistics()

        assert stats.collection_name == "patient_claim_history"
        assert stats.vector_dimensions == 1536
        assert stats.total_documents >= 0

    def test_kb_validate(self, qdrant_client, openai_api_key, sample_patient_data):
        """Test KB validation."""
        kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        kb.create_collection()

        # Build patient document
        builder = PatientHistoryBuilder()
        doc = builder.build_patient_document(sample_patient_data)

        # Index
        kb.upsert_documents([doc])

        # Validate
        is_valid = kb.validate()
        assert is_valid is True
