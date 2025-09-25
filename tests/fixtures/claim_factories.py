"""
Factory classes for generating test insurance claims data.
"""
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from faker import Faker
import factory
from factory import LazyFunction, Sequence, LazyAttribute, SubFactory


fake = Faker()
Faker.seed(42)  # For reproducible test data


class ClaimIdFactory(factory.Factory):
    """Factory for generating claim IDs."""

    class Meta:
        model = str

    claim_id = Sequence(lambda n: f"CLM-2024-{n:06d}")


class PatientIdFactory(factory.Factory):
    """Factory for generating patient IDs."""

    class Meta:
        model = str

    patient_id = Sequence(lambda n: f"PAT-{n:08d}")


class ProviderNPIFactory(factory.Factory):
    """Factory for generating provider NPIs."""

    class Meta:
        model = str

    npi = LazyFunction(lambda: str(random.randint(1000000000, 9999999999)))


class BaseClaim(factory.DictFactory):
    """Base factory for insurance claims."""

    claim_id = Sequence(lambda n: f"CLM-2024-{n:06d}")
    patient_id = Sequence(lambda n: f"PAT-{n:08d}")
    provider_npi = LazyFunction(lambda: str(random.randint(1000000000, 9999999999)))
    service_date = LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
    )
    claim_date = LazyAttribute(
        lambda obj: (
            datetime.fromisoformat(obj.service_date) +
            timedelta(days=random.randint(1, 30))
        ).isoformat()
    )
    billed_amount = LazyFunction(lambda: round(random.uniform(50.0, 5000.0), 2))
    diagnosis_codes = LazyFunction(
        lambda: random.sample([
            'M79.3', 'S13.4', 'M54.2', 'G44.1', 'M25.5',
            'Z51.11', 'M17.0', 'I25.10', 'E11.9', 'F32.9'
        ], k=random.randint(1, 3))
    )
    procedure_codes = LazyFunction(
        lambda: random.sample([
            '99213', '99214', '99215', '73721', '97110',
            '99283', '99284', '99285', '70553', '99281'
        ], k=random.randint(1, 2))
    )
    patient_age = LazyFunction(lambda: random.randint(18, 85))
    provider_specialty = LazyFunction(
        lambda: random.choice([
            'Internal Medicine', 'Emergency Medicine', 'Orthopedic Surgery',
            'Physical Medicine', 'Radiology', 'Neurology', 'Cardiology'
        ])
    )
    service_location = LazyFunction(
        lambda: random.choice([
            'Office', 'Hospital Inpatient', 'Hospital Outpatient',
            'Emergency Room', 'Ambulatory Surgery Center'
        ])
    )


class ValidClaim(BaseClaim):
    """Factory for generating valid (non-fraudulent) claims."""

    fraud_indicator = False
    fraud_type = None
    red_flags = []

    # Valid claims have reasonable amounts
    billed_amount = LazyFunction(lambda: round(random.uniform(50.0, 2000.0), 2))


class UpcodingFraudClaim(BaseClaim):
    """Factory for generating upcoding fraud claims."""

    fraud_indicator = True
    fraud_type = "upcoding"
    red_flags = ["excessive_billing", "complexity_mismatch"]

    # Upcoding has inflated amounts
    billed_amount = LazyFunction(lambda: round(random.uniform(3000.0, 15000.0), 2))

    # Higher complexity procedures
    procedure_codes = LazyFunction(
        lambda: random.sample(['99215', '99285', '70553'], k=1)
    )


class PhantomBillingClaim(BaseClaim):
    """Factory for generating phantom billing fraud claims."""

    fraud_indicator = True
    fraud_type = "phantom_billing"
    red_flags = ["no_patient_contact", "impossible_timeline"]

    # Multiple claims on same day
    service_date = LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
    )


class UnbundlingFraudClaim(BaseClaim):
    """Factory for generating unbundling fraud claims."""

    fraud_indicator = True
    fraud_type = "unbundling"
    red_flags = ["procedure_fragmentation", "excessive_claims"]

    # Multiple related procedures billed separately
    procedure_codes = LazyFunction(
        lambda: random.sample([
            '99213', '99214', '73721', '97110', '97112'
        ], k=random.randint(3, 5))
    )


class StagedAccidentClaim(BaseClaim):
    """Factory for generating staged accident fraud claims."""

    fraud_indicator = True
    fraud_type = "staged_accident"
    red_flags = ["pattern_matching", "suspicious_circumstances"]

    # Accident-related diagnosis codes
    diagnosis_codes = LazyFunction(
        lambda: random.sample(['S13.4', 'M79.3', 'G44.1'], k=2)
    )

    # High billed amounts for accidents
    billed_amount = LazyFunction(lambda: round(random.uniform(5000.0, 25000.0), 2))


class PrescriptionFraudClaim(BaseClaim):
    """Factory for generating prescription fraud claims."""

    fraud_indicator = True
    fraud_type = "prescription_fraud"
    red_flags = ["drug_seeking", "doctor_shopping"]

    # Pain management related codes
    diagnosis_codes = LazyFunction(
        lambda: ['M79.3', 'G44.1']  # Pain-related diagnoses
    )


def generate_mixed_claims_batch(
    total_claims: int = 1000,
    fraud_rate: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Generate a mixed batch of valid and fraudulent claims.

    Args:
        total_claims: Total number of claims to generate
        fraud_rate: Percentage of claims that should be fraudulent (0.0-1.0)

    Returns:
        List of claim dictionaries
    """
    fraud_count = int(total_claims * fraud_rate)
    valid_count = total_claims - fraud_count

    claims = []

    # Generate valid claims
    for _ in range(valid_count):
        claims.append(ValidClaim())

    # Generate fraudulent claims (mix of types)
    fraud_types = [
        UpcodingFraudClaim,
        PhantomBillingClaim,
        UnbundlingFraudClaim,
        StagedAccidentClaim,
        PrescriptionFraudClaim
    ]

    for _ in range(fraud_count):
        fraud_type = random.choice(fraud_types)
        claims.append(fraud_type())

    # Shuffle to mix valid and fraud claims
    random.shuffle(claims)

    return claims


def generate_performance_test_data(size: int = 10000) -> List[Dict[str, Any]]:
    """
    Generate large dataset for performance testing.

    Args:
        size: Number of claims to generate

    Returns:
        List of claim dictionaries
    """
    return generate_mixed_claims_batch(total_claims=size, fraud_rate=0.12)


def generate_accuracy_test_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate balanced dataset for accuracy testing.

    Returns:
        Dictionary with 'valid' and 'fraud' claim lists
    """
    return {
        'valid': [ValidClaim() for _ in range(500)],
        'fraud': [
            UpcodingFraudClaim() for _ in range(100)
        ] + [
            PhantomBillingClaim() for _ in range(100)
        ] + [
            UnbundlingFraudClaim() for _ in range(100)
        ] + [
            StagedAccidentClaim() for _ in range(100)
        ] + [
            PrescriptionFraudClaim() for _ in range(100)
        ]
    }


class MockClaimBatch:
    """Mock claim batch for testing batch processing."""

    def __init__(self, size: int = 1000, fraud_rate: float = 0.15):
        self.claims = generate_mixed_claims_batch(size, fraud_rate)
        self.size = size
        self.fraud_rate = fraud_rate

    def __iter__(self):
        return iter(self.claims)

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, index):
        return self.claims[index]


# Specific test data generators
def create_high_risk_claim() -> Dict[str, Any]:
    """Create a claim with multiple fraud indicators."""
    claim = BaseClaim()
    claim['fraud_indicator'] = True
    claim['fraud_type'] = "multiple_indicators"
    claim['red_flags'] = [
        "excessive_billing",
        "complexity_mismatch",
        "suspicious_timing",
        "pattern_matching"
    ]
    claim['billed_amount'] = 15000.0
    return claim


def create_edge_case_claims() -> List[Dict[str, Any]]:
    """Create claims with edge case values for boundary testing."""
    return [
        # Minimum values
        {
            **BaseClaim(),
            'billed_amount': 0.01,
            'patient_age': 0,
            'diagnosis_codes': [],
            'procedure_codes': []
        },
        # Maximum values
        {
            **BaseClaim(),
            'billed_amount': 999999.99,
            'patient_age': 120,
            'diagnosis_codes': ['M79.3'] * 10,
            'procedure_codes': ['99213'] * 10
        },
        # Null/None values
        {
            **BaseClaim(),
            'billed_amount': None,
            'diagnosis_codes': None,
            'procedure_codes': None
        }
    ]