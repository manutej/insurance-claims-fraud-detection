"""
Test Fixtures Schema for Insurance Fraud Detection System

This module defines Pydantic models for test data generation and validation.
Supports both RAG enrichment testing and fraud detection testing.

Usage:
    from tests.fixtures.TEST_FIXTURES_SCHEMA import (
        CompleteTestClaim,
        IncompleteTestClaim,
        FraudTestCase,
        generate_test_batch
    )
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import random


# ============================================================================
# ENUMERATIONS
# ============================================================================

class FraudType(str, Enum):
    """Types of fraud patterns for testing."""
    UPCODING = "upcoding"
    PHANTOM_BILLING = "phantom_billing"
    UNBUNDLING = "unbundling"
    STAGED_ACCIDENT = "staged_accident"
    PRESCRIPTION_FRAUD = "prescription_fraud"
    KICKBACK_SCHEME = "kickback_scheme"
    NONE = "none"


class ClaimCompleteness(str, Enum):
    """Claim data completeness levels."""
    COMPLETE = "complete"
    MISSING_DIAGNOSIS = "missing_diagnosis"
    MISSING_PROCEDURE = "missing_procedure"
    MISSING_DESCRIPTIONS = "missing_descriptions"
    MISSING_MULTIPLE = "missing_multiple"
    MINIMAL = "minimal"


class EnrichmentConfidenceLevel(str, Enum):
    """Confidence levels for enrichment."""
    HIGH = "high"      # >0.9
    MEDIUM = "medium"  # 0.7-0.9
    LOW = "low"        # 0.5-0.7
    VERY_LOW = "very_low"  # <0.5


class FraudSeverity(str, Enum):
    """Fraud severity levels for test cases."""
    OBVIOUS = "obvious"        # Clear, blatant fraud
    MODERATE = "moderate"      # Suspicious patterns
    SUBTLE = "subtle"          # Borderline cases
    LEGITIMATE = "legitimate"  # Valid claims


# ============================================================================
# BASE TEST MODELS
# ============================================================================

class EnrichmentMetadata(BaseModel):
    """Metadata for RAG enrichment testing."""

    enriched_fields: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    retrieval_sources: Dict[str, str] = Field(default_factory=dict)
    enrichment_timestamp: Optional[datetime] = None
    enrichment_method: Optional[str] = None  # "kb_exact_match", "kb_similarity", "inference"

    @validator('confidence_scores')
    def validate_confidence_scores(cls, v):
        """Ensure confidence scores are in valid range [0,1]."""
        for field, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Confidence score for {field} must be in [0,1], got {score}")
        return v


class FraudIndicators(BaseModel):
    """Fraud indicators for test validation."""

    fraud_type: FraudType
    fraud_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    red_flags: List[str] = Field(default_factory=list)
    triggered_rules: List[str] = Field(default_factory=list)
    contributing_features: Dict[str, float] = Field(default_factory=dict)
    explanation: Optional[str] = None


class ExpectedDetectionResult(BaseModel):
    """Expected fraud detection results for test validation."""

    should_detect_fraud: bool
    expected_fraud_types: List[FraudType] = Field(default_factory=list)
    expected_score_range: tuple[float, float] = Field(default=(0.0, 1.0))
    expected_rules_triggered: List[str] = Field(default_factory=list)
    expected_confidence_level: Optional[str] = None
    notes: Optional[str] = None


# ============================================================================
# CLAIM TEST MODELS
# ============================================================================

class CompleteTestClaim(BaseModel):
    """Complete test claim with all required fields."""

    claim_id: str = Field(..., regex=r"^CLM-TEST-\d+$")
    patient_id: str = Field(..., regex=r"^PAT-\d+$")
    provider_id: str = Field(..., regex=r"^PRV-\d+$")
    provider_npi: str = Field(..., regex=r"^\d{10}$")
    date_of_service: date

    # Medical coding
    diagnosis_codes: List[str] = Field(..., min_items=1)
    diagnosis_descriptions: List[str] = Field(..., min_items=1)
    procedure_codes: List[str] = Field(..., min_items=1)
    procedure_descriptions: List[str] = Field(..., min_items=1)

    # Financial
    billed_amount: Decimal = Field(..., gt=0, decimal_places=2)

    # Metadata
    service_location: str = Field(..., regex=r"^\d{2}$")
    service_location_desc: Optional[str] = None
    claim_type: str = "professional"
    rendering_hours: Optional[Decimal] = Field(None, ge=0, le=24)
    day_of_week: Optional[str] = None

    # Fraud indicators (for testing)
    fraud_indicator: bool = False
    fraud_type: Optional[FraudType] = None
    red_flags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # Test metadata
    test_category: str = "complete"
    expected_result: Optional[ExpectedDetectionResult] = None

    @validator('date_of_service', pre=True)
    def parse_date(cls, v):
        """Parse date string to date object."""
        if isinstance(v, str):
            return datetime.strptime(v, '%Y-%m-%d').date()
        return v

    @validator('diagnosis_codes', 'procedure_codes', each_item=True)
    def validate_code_format(cls, v, field):
        """Validate medical code formats."""
        if field.name == 'diagnosis_codes':
            # ICD-10 format
            import re
            if not re.match(r"^[A-Z]\d{2}(\.\d{1,2})?$", v):
                raise ValueError(f"Invalid ICD-10 code format: {v}")
        elif field.name == 'procedure_codes':
            # CPT format
            if not v.isdigit() or len(v) != 5:
                raise ValueError(f"Invalid CPT code format: {v}")
        return v

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat()
        }


class IncompleteTestClaim(BaseModel):
    """Incomplete test claim for RAG enrichment testing."""

    claim_id: str = Field(..., regex=r"^CLM-TEST-\d+$")
    patient_id: str = Field(..., regex=r"^PAT-\d+$")
    provider_id: str = Field(..., regex=r"^PRV-\d+$")
    provider_npi: str = Field(..., regex=r"^\d{10}$")
    date_of_service: date

    # Optional fields (may be missing for enrichment testing)
    diagnosis_codes: Optional[List[str]] = None
    diagnosis_descriptions: Optional[List[str]] = None
    procedure_codes: Optional[List[str]] = None
    procedure_descriptions: Optional[List[str]] = None

    # Financial (required)
    billed_amount: Decimal = Field(..., gt=0, decimal_places=2)

    # Metadata
    service_location: Optional[str] = None
    service_location_desc: Optional[str] = None
    claim_type: str = "professional"

    # Test metadata
    completeness_level: ClaimCompleteness = ClaimCompleteness.MISSING_MULTIPLE
    ground_truth_claim: Optional[CompleteTestClaim] = None  # For validation
    expected_enrichment_confidence: Optional[EnrichmentConfidenceLevel] = None
    enrichment_metadata: Optional[EnrichmentMetadata] = None

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat()
        }


# ============================================================================
# FRAUD PATTERN TEST CASES
# ============================================================================

class UpcodingTestCase(BaseModel):
    """Test case for upcoding fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Upcoding specific
    actual_service_code: Optional[str] = None  # What should have been billed
    actual_amount: Optional[Decimal] = None
    complexity_mismatch: bool = False
    provider_pattern: Optional[str] = None  # "consistent_upcoding", etc.


class PhantomBillingTestCase(BaseModel):
    """Test case for phantom billing fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Phantom billing specific
    service_actually_rendered: bool = False
    impossible_schedule: bool = False
    ghost_patient: bool = False
    facility_closed: bool = False
    weekend_non_emergency: bool = False


class UnbundlingTestCase(BaseModel):
    """Test case for unbundling fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Unbundling specific
    bundled_procedure_group: Optional[str] = None  # "colonoscopy", "cataract_surgery"
    should_be_single_code: Optional[str] = None
    expected_bundled_amount: Optional[Decimal] = None
    same_day_related_claims: List[str] = Field(default_factory=list)


class StagedAccidentTestCase(BaseModel):
    """Test case for staged accident fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Staged accident specific
    accident_date: date
    accident_location: Optional[str] = None
    attorney_involved: bool = False
    attorney_id: Optional[str] = None
    pre_existing_relationships: List[str] = Field(default_factory=list)
    similar_accidents_count: int = 0
    injury_pattern: Optional[str] = None


class PrescriptionFraudTestCase(BaseModel):
    """Test case for prescription fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Prescription fraud specific
    drug_name: Optional[str] = None
    is_controlled_substance: bool = False
    quantity: Optional[int] = None
    days_supply: Optional[int] = None
    prescriber_npi: Optional[str] = None
    doctor_shopping_indicator: bool = False
    early_refill: bool = False


class KickbackSchemeTestCase(BaseModel):
    """Test case for kickback scheme fraud detection."""

    case_id: str
    case_description: str
    severity: FraudSeverity

    claim: CompleteTestClaim
    expected_result: ExpectedDetectionResult

    # Kickback specific
    referred_to_provider: Optional[str] = None
    referral_concentration: Optional[float] = None
    circular_referral: bool = False
    financial_relationship: Optional[str] = None
    medically_unnecessary: bool = False


# ============================================================================
# TEST BATCH GENERATION
# ============================================================================

class TestBatch(BaseModel):
    """Batch of test claims for testing."""

    batch_id: str
    batch_description: str
    total_claims: int
    fraud_rate: float = Field(ge=0.0, le=1.0)

    complete_claims: List[CompleteTestClaim] = Field(default_factory=list)
    incomplete_claims: List[IncompleteTestClaim] = Field(default_factory=list)

    fraud_distribution: Dict[FraudType, int] = Field(default_factory=dict)
    completeness_distribution: Dict[ClaimCompleteness, int] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('total_claims', always=True)
    def validate_total_claims(cls, v, values):
        """Validate total claims matches actual counts."""
        complete = len(values.get('complete_claims', []))
        incomplete = len(values.get('incomplete_claims', []))
        actual_total = complete + incomplete
        if v != actual_total:
            raise ValueError(f"total_claims ({v}) doesn't match actual ({actual_total})")
        return v


# ============================================================================
# HELPER FACTORIES
# ============================================================================

class TestClaimFactory:
    """Factory for generating test claims."""

    @staticmethod
    def create_valid_claim(
        claim_id: str = "CLM-TEST-001",
        diagnosis: str = "E11.9",
        procedure: str = "99213",
        amount: float = 125.0
    ) -> CompleteTestClaim:
        """Create a valid, legitimate test claim."""
        return CompleteTestClaim(
            claim_id=claim_id,
            patient_id="PAT-10001",
            provider_id="PRV-20001",
            provider_npi="1234567890",
            date_of_service=date.today(),
            diagnosis_codes=[diagnosis],
            diagnosis_descriptions=["Type 2 diabetes mellitus without complications"],
            procedure_codes=[procedure],
            procedure_descriptions=["Office visit, established patient, low complexity"],
            billed_amount=Decimal(str(amount)),
            service_location="11",
            service_location_desc="Office",
            fraud_indicator=False,
            expected_result=ExpectedDetectionResult(
                should_detect_fraud=False,
                expected_score_range=(0.0, 0.3),
                notes="Legitimate claim"
            )
        )

    @staticmethod
    def create_upcoding_claim(
        claim_id: str = "CLM-TEST-FRAUD-001",
        severity: FraudSeverity = FraudSeverity.OBVIOUS
    ) -> CompleteTestClaim:
        """Create an upcoding fraud test claim."""
        return CompleteTestClaim(
            claim_id=claim_id,
            patient_id="PAT-80001",
            provider_id="PRV-FRAUD-001",
            provider_npi="9999999001",
            date_of_service=date.today(),
            diagnosis_codes=["J00"],  # Common cold
            diagnosis_descriptions=["Acute nasopharyngitis (common cold)"],
            procedure_codes=["99215"],  # High complexity visit
            procedure_descriptions=["Office visit, established patient, high complexity"],
            billed_amount=Decimal("325.00"),
            service_location="11",
            fraud_indicator=True,
            fraud_type=FraudType.UPCODING,
            red_flags=[
                "Simple diagnosis billed at highest complexity",
                "Provider bills 90% of visits as 99215"
            ],
            expected_result=ExpectedDetectionResult(
                should_detect_fraud=True,
                expected_fraud_types=[FraudType.UPCODING],
                expected_score_range=(0.7, 1.0),
                expected_rules_triggered=["upcoding_complexity", "amount_anomaly"],
                notes="Clear upcoding: common cold as complex visit"
            )
        )

    @staticmethod
    def create_incomplete_claim(
        claim_id: str = "CLM-TEST-INC-001",
        completeness: ClaimCompleteness = ClaimCompleteness.MISSING_DIAGNOSIS
    ) -> IncompleteTestClaim:
        """Create an incomplete test claim for enrichment testing."""
        base_claim = {
            "claim_id": claim_id,
            "patient_id": "PAT-10001",
            "provider_id": "PRV-20001",
            "provider_npi": "1234567890",
            "date_of_service": date.today(),
            "billed_amount": Decimal("125.00"),
            "completeness_level": completeness
        }

        if completeness == ClaimCompleteness.MISSING_DIAGNOSIS:
            base_claim.update({
                "procedure_codes": ["99213"],
                "procedure_descriptions": ["Office visit, established patient"],
                "service_location": "11"
            })
        elif completeness == ClaimCompleteness.MISSING_PROCEDURE:
            base_claim.update({
                "diagnosis_codes": ["E11.9"],
                "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
                "service_location": "11"
            })
        elif completeness == ClaimCompleteness.MISSING_DESCRIPTIONS:
            base_claim.update({
                "diagnosis_codes": ["E11.9"],
                "procedure_codes": ["99213"],
                "service_location": "11"
            })

        return IncompleteTestClaim(**base_claim)


# ============================================================================
# BATCH GENERATION UTILITIES
# ============================================================================

def generate_test_batch(
    batch_id: str,
    total_claims: int = 100,
    fraud_rate: float = 0.15,
    incomplete_rate: float = 0.3,
    fraud_type_distribution: Optional[Dict[FraudType, float]] = None
) -> TestBatch:
    """
    Generate a batch of test claims with specified characteristics.

    Args:
        batch_id: Unique batch identifier
        total_claims: Total number of claims to generate
        fraud_rate: Proportion of fraudulent claims (0.0-1.0)
        incomplete_rate: Proportion of incomplete claims (0.0-1.0)
        fraud_type_distribution: Distribution of fraud types (normalized)

    Returns:
        TestBatch with generated claims
    """
    if fraud_type_distribution is None:
        fraud_type_distribution = {
            FraudType.UPCODING: 0.35,
            FraudType.PHANTOM_BILLING: 0.20,
            FraudType.UNBUNDLING: 0.20,
            FraudType.STAGED_ACCIDENT: 0.10,
            FraudType.PRESCRIPTION_FRAUD: 0.10,
            FraudType.KICKBACK_SCHEME: 0.05
        }

    factory = TestClaimFactory()
    complete_claims = []
    incomplete_claims = []
    fraud_distribution = {ft: 0 for ft in FraudType}
    completeness_distribution = {cl: 0 for cl in ClaimCompleteness}

    # Generate claims
    num_fraud = int(total_claims * fraud_rate)
    num_legitimate = total_claims - num_fraud
    num_incomplete = int(total_claims * incomplete_rate)

    # Generate legitimate claims
    for i in range(num_legitimate):
        claim_id = f"CLM-TEST-{batch_id}-L{i:04d}"
        if i < num_incomplete:
            # Incomplete legitimate claim
            completeness = random.choice(list(ClaimCompleteness)[1:-1])
            incomplete_claims.append(
                factory.create_incomplete_claim(claim_id, completeness)
            )
            completeness_distribution[completeness] += 1
        else:
            # Complete legitimate claim
            complete_claims.append(factory.create_valid_claim(claim_id))
            completeness_distribution[ClaimCompleteness.COMPLETE] += 1
        fraud_distribution[FraudType.NONE] += 1

    # Generate fraudulent claims
    for i in range(num_fraud):
        claim_id = f"CLM-TEST-{batch_id}-F{i:04d}"
        # Select fraud type based on distribution
        fraud_type = random.choices(
            list(fraud_type_distribution.keys()),
            weights=list(fraud_type_distribution.values())
        )[0]

        # Generate appropriate fraud claim
        if fraud_type == FraudType.UPCODING:
            complete_claims.append(factory.create_upcoding_claim(claim_id))
        # Add other fraud types as needed

        fraud_distribution[fraud_type] += 1
        completeness_distribution[ClaimCompleteness.COMPLETE] += 1

    return TestBatch(
        batch_id=batch_id,
        batch_description=f"Test batch: {total_claims} claims, {fraud_rate*100:.1f}% fraud",
        total_claims=total_claims,
        fraud_rate=fraud_rate,
        complete_claims=complete_claims,
        incomplete_claims=incomplete_claims,
        fraud_distribution=fraud_distribution,
        completeness_distribution=completeness_distribution
    )


# ============================================================================
# FIXTURES FOR PYTEST
# ============================================================================

def pytest_fixtures():
    """
    Returns common pytest fixtures as dictionary.

    Usage in conftest.py:
        from tests.fixtures.TEST_FIXTURES_SCHEMA import pytest_fixtures
        fixtures = pytest_fixtures()

        @pytest.fixture
        def valid_claim():
            return fixtures['valid_claim']
    """
    factory = TestClaimFactory()

    return {
        'valid_claim': factory.create_valid_claim(),
        'upcoding_claim': factory.create_upcoding_claim(),
        'incomplete_claim_missing_diagnosis': factory.create_incomplete_claim(
            completeness=ClaimCompleteness.MISSING_DIAGNOSIS
        ),
        'incomplete_claim_missing_procedure': factory.create_incomplete_claim(
            completeness=ClaimCompleteness.MISSING_PROCEDURE
        ),
        'test_batch_balanced': generate_test_batch(
            batch_id="BAL001",
            total_claims=100,
            fraud_rate=0.5,
            incomplete_rate=0.0
        ),
        'test_batch_realistic': generate_test_batch(
            batch_id="REAL001",
            total_claims=1000,
            fraud_rate=0.12,
            incomplete_rate=0.3
        )
    }


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

class TestResultValidator:
    """Validator for test result assertions."""

    @staticmethod
    def validate_fraud_detection_result(
        actual_fraud_detected: bool,
        actual_fraud_score: float,
        actual_fraud_types: List[FraudType],
        expected: ExpectedDetectionResult
    ) -> tuple[bool, str]:
        """
        Validate fraud detection result against expected outcome.

        Returns:
            (passed, message) tuple
        """
        if actual_fraud_detected != expected.should_detect_fraud:
            return False, f"Expected fraud_detected={expected.should_detect_fraud}, got {actual_fraud_detected}"

        if not (expected.expected_score_range[0] <= actual_fraud_score <= expected.expected_score_range[1]):
            return False, f"Score {actual_fraud_score} outside expected range {expected.expected_score_range}"

        if expected.expected_fraud_types:
            if not set(actual_fraud_types).intersection(set(expected.expected_fraud_types)):
                return False, f"Expected fraud types {expected.expected_fraud_types}, got {actual_fraud_types}"

        return True, "Validation passed"

    @staticmethod
    def validate_enrichment_result(
        enriched_claim: dict,
        ground_truth: CompleteTestClaim,
        expected_confidence: EnrichmentConfidenceLevel
    ) -> tuple[bool, str]:
        """
        Validate enrichment result against ground truth.

        Returns:
            (passed, message) tuple
        """
        # Check if required fields were enriched
        missing_fields = []
        if "diagnosis_codes" not in enriched_claim or not enriched_claim["diagnosis_codes"]:
            if ground_truth.diagnosis_codes:
                missing_fields.append("diagnosis_codes")

        if "procedure_codes" not in enriched_claim or not enriched_claim["procedure_codes"]:
            if ground_truth.procedure_codes:
                missing_fields.append("procedure_codes")

        if missing_fields:
            return False, f"Failed to enrich fields: {missing_fields}"

        # Validate enrichment accuracy
        if "diagnosis_codes" in enriched_claim and ground_truth.diagnosis_codes:
            enriched_diag = set(enriched_claim["diagnosis_codes"])
            ground_truth_diag = set(ground_truth.diagnosis_codes)
            if not enriched_diag.intersection(ground_truth_diag):
                return False, f"Enriched diagnoses {enriched_diag} don't match ground truth {ground_truth_diag}"

        return True, "Enrichment validation passed"
