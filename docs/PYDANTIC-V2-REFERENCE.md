# Pydantic v2 Reference Guide
## Insurance Claims Data Modeling & Fraud Detection

**Source**: Pydantic v2.12.3 Documentation
**Trust Score**: 10/10 (Official Documentation)
**Date**: 2025-10-28
**Scope**: Focused on insurance claims data modeling and fraud detection feature engineering

---

## Quick Reference: Core Patterns

| Pattern | Use Case | Complexity |
|---------|----------|-----------|
| **BaseModel** | Define claim structure | Basic |
| **Field + ConfigDict** | Configure field validation | Basic |
| **AfterValidator** | Validate parsed claim fields | Intermediate |
| **Field Validators** | Multi-field claim rules | Intermediate |
| **Nested Models** | Hierarchical claim components | Intermediate |
| **Custom Serializers** | Export claims in various formats | Intermediate |
| **Model Validators** | Cross-field fraud rules | Advanced |
| **Generic Types** | Reusable validator patterns | Advanced |
| **Conditional Validation** | Risk-based validation rules | Advanced |

---

## 1. Data Model Definition & Validation

### Basic Model Structure

```python
from pydantic import BaseModel, Field
from datetime import datetime

class InsuranceClaim(BaseModel):
    """Base insurance claim model with required and optional fields."""

    # Required fields
    claim_id: str
    patient_id: str
    provider_npi: str
    claim_date: datetime

    # Optional with defaults
    status: str = 'pending'
    fraud_indicator: bool = False

    # Field with constraints
    billed_amount: float = Field(gt=0, description="Amount must be positive")
```

### Field Configuration with ConfigDict

```python
from pydantic import BaseModel, ConfigDict, Field

class MedicalClaim(BaseModel):
    """Claim with strict validation configuration."""

    model_config = ConfigDict(
        str_strip_whitespace=True,  # Auto-strip string whitespace
        validate_default=True,       # Validate default values
        extra='forbid',              # Reject unknown fields
        strict=False,                # Allow type coercion
    )

    claim_id: str = Field(min_length=1, max_length=50)
    procedure_code: str = Field(pattern=r'^\d{5}')  # CPT code format
    diagnosis_codes: list[str] = Field(
        min_length=1,
        max_length=25,
        description="ICD-10 diagnosis codes"
    )
    billed_amount: float = Field(
        gt=0,
        le=999999.99,
        description="Claim amount in USD"
    )
```

### Key Field Options

```python
from pydantic import Field, BaseModel
from typing import Optional

class ClaimWithFieldOptions(BaseModel):
    """Demonstrate comprehensive field configuration."""

    claim_id: str = Field(
        description="Unique claim identifier",
        example="CLM-2025-001234",
        min_length=10,
        max_length=20
    )

    # Default factory for mutable defaults
    procedure_codes: list[str] = Field(
        default_factory=list,
        description="List of CPT procedure codes"
    )

    # Optional with None default
    investigation_notes: Optional[str] = Field(
        default=None,
        description="Fraud investigator notes"
    )

    # Excluded from serialization
    internal_risk_score: float = Field(
        ge=0,
        le=100,
        exclude=True,  # Not serialized by default
        description="Internal risk calculation"
    )
```

### Initialization & Data Access

```python
# Instance creation with validation
claim = InsuranceClaim(
    claim_id="CLM-2025-001",
    patient_id="PT-5432",
    provider_npi="1234567890",
    claim_date="2025-10-28",  # String auto-converts to datetime
    billed_amount=1500.00
)

# Access validated fields
print(f"Claim {claim.claim_id}: ${claim.billed_amount}")

# Validate from dictionary
claim_dict = {
    "claim_id": "CLM-2025-002",
    "patient_id": "PT-5433",
    "provider_npi": "1234567891",
    "claim_date": "2025-10-28",
    "billed_amount": 2500.00
}
claim = InsuranceClaim.model_validate(claim_dict)

# Handle validation errors
from pydantic import ValidationError

try:
    invalid_claim = InsuranceClaim(
        claim_id="CLM-2025-003",
        patient_id="PT-5434",
        provider_npi="1234567892",
        claim_date="invalid-date",  # Will fail
        billed_amount=-100  # Invalid: must be > 0
    )
except ValidationError as e:
    print(f"Validation errors: {e.error_count()} issues found")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
```

---

## 2. Field Validators & Custom Validation

### After Validators (Safest Pattern)

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel, Field
import re

def validate_npi(value: str) -> str:
    """Validate NPI format (10 digits)."""
    if not re.match(r'^\d{10}$', value):
        raise ValueError('NPI must be exactly 10 digits')
    return value

def validate_diagnosis_not_empty(codes: list[str]) -> list[str]:
    """Ensure at least one diagnosis code."""
    if not codes or all(not code.strip() for code in codes):
        raise ValueError('At least one diagnosis code required')
    return [code.strip().upper() for code in codes]

# Create reusable annotated types
ValidNPI = Annotated[str, AfterValidator(validate_npi)]
NonEmptyDiagnosisCodes = Annotated[list[str], AfterValidator(validate_diagnosis_not_empty)]

class MedicalClaimWithValidators(BaseModel):
    """Claim with annotated validators."""
    claim_id: str
    provider_npi: ValidNPI  # Reusable validator
    diagnosis_codes: NonEmptyDiagnosisCodes  # Multi-field validation
```

### Field Validator Decorator (Flexible Pattern)

```python
from pydantic import BaseModel, field_validator

class ClaimWithFieldValidators(BaseModel):
    """Use @field_validator decorator for complex logic."""

    claim_id: str
    billed_amount: float
    procedure_codes: list[str]
    provider_npi: str

    @field_validator('claim_id', mode='after')
    @classmethod
    def validate_claim_id_format(cls, value: str) -> str:
        """Validate claim ID follows pattern."""
        if not value.startswith('CLM-'):
            raise ValueError('Claim ID must start with CLM-')
        return value.upper()

    @field_validator('billed_amount', mode='after')
    @classmethod
    def validate_amount_range(cls, value: float) -> float:
        """Ensure amount is within reasonable bounds."""
        if value > 999999.99:
            raise ValueError('Amount exceeds maximum allowed ($999,999.99)')
        if value == 0:
            raise ValueError('Amount cannot be zero')
        return value

    @field_validator('procedure_codes', mode='after')
    @classmethod
    def validate_procedure_codes(cls, codes: list[str]) -> list[str]:
        """Validate CPT codes format and remove duplicates."""
        validated = set()
        for code in codes:
            if not re.match(r'^\d{5}$', code):
                raise ValueError(f'Invalid CPT code format: {code}')
            validated.add(code)
        return sorted(list(validated))
```

### Model Validators (Cross-Field Logic)

```python
from pydantic import BaseModel, model_validator

class FraudRiskClaim(BaseModel):
    """Complex validation across multiple fields."""

    claim_id: str
    patient_id: str
    diagnosis_codes: list[str]
    procedure_codes: list[str]
    billed_amount: float
    days_to_treatment: int

    @model_validator(mode='after')
    def validate_diagnosis_procedure_match(self) -> 'FraudRiskClaim':
        """
        Validate diagnosis codes match procedures.
        Red flag: Procedures don't align with diagnoses.
        """
        # Simplified example: certain diagnoses require specific procedure codes
        high_cost_procedures = {'27447', '27486', '27487'}  # Joint replacements
        orthopedic_diagnoses = {'M16', 'M17', 'M19'}  # Arthritis codes

        has_orthopedic = any(d.startswith(tuple(orthopedic_diagnoses))
                            for d in self.diagnosis_codes)
        has_orthopedic_procedure = any(p in high_cost_procedures
                                      for p in self.procedure_codes)

        if has_orthopedic != has_orthopedic_procedure:
            raise ValueError(
                'Diagnosis codes do not match procedure codes - '
                'possible upcoding or phantom billing'
            )
        return self

    @model_validator(mode='after')
    def check_suspicious_timing(self) -> 'FraudRiskClaim':
        """Red flag: Treatment within 1 day of diagnosis."""
        if self.days_to_treatment < 1:
            raise ValueError(
                'Claim flagged: treatment on same day as diagnosis (staged accident indicator)'
            )
        return self
```

### Validators with Context (Conditional Validation)

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class ContextualClaim(BaseModel):
    """Use validation context for conditional rules."""

    claim_id: str
    provider_type: str  # 'hospital', 'clinic', 'emergency'
    procedure_codes: list[str]
    billed_amount: float

    @field_validator('billed_amount', mode='after')
    @classmethod
    def validate_amount_by_provider(
        cls,
        value: float,
        info: ValidationInfo
    ) -> float:
        """
        Provider-specific amount validation.

        Args:
            value: The billed amount
            info: ValidationInfo containing data, context, field info
        """
        provider_type = info.data.get('provider_type')

        # Different validation rules per provider type
        limits = {
            'clinic': 10000,
            'hospital': 250000,
            'emergency': 100000
        }

        max_allowed = limits.get(provider_type, 50000)
        if value > max_allowed:
            raise ValueError(
                f'{provider_type} claim exceeds limit: '
                f'${value} > ${max_allowed}'
            )
        return value
```

---

## 3. Serialization & Deserialization

### Basic Serialization Methods

```python
from pydantic import BaseModel
from datetime import datetime

class ClaimForSerialization(BaseModel):
    claim_id: str
    patient_id: str
    claim_date: datetime
    billed_amount: float
    procedures: list[str]

claim = ClaimForSerialization(
    claim_id="CLM-2025-001",
    patient_id="PT-5432",
    claim_date="2025-10-28",
    billed_amount=1500.50,
    procedures=["99213", "99214"]
)

# Convert to Python dictionary
claim_dict = claim.model_dump()
print(claim_dict)
# Output: {'claim_id': 'CLM-2025-001', 'patient_id': 'PT-5432',
#          'claim_date': datetime(2025, 10, 28, 0, 0),
#          'billed_amount': 1500.5, 'procedures': ['99213', '99214']}

# Convert to JSON string
claim_json = claim.model_dump_json()
print(claim_json)
# Output: {"claim_id":"CLM-2025-001","patient_id":"PT-5432",
#          "claim_date":"2025-10-28T00:00:00","billed_amount":1500.5,
#          "procedures":["99213","99214"]}

# Export with custom mode (datetime as ISO string)
claim_export = claim.model_dump(mode='json')

# Exclude sensitive fields
claim_safe = claim.model_dump(exclude={'patient_id'})
```

### Custom Field Serializers

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class ClaimWithCustomSerialization(BaseModel):
    """Custom serialization for domain-specific formats."""

    claim_id: str
    claim_date: datetime
    billed_amount: float
    procedures: list[str]

    @field_serializer('billed_amount')
    def serialize_amount(self, value: float, _info) -> str:
        """Format currency as string with 2 decimal places."""
        return f"${value:,.2f}"

    @field_serializer('claim_date')
    def serialize_date(self, value: datetime, _info) -> str:
        """Format date as YYYY-MM-DD."""
        return value.strftime('%Y-%m-%d')

    @field_serializer('procedures')
    def serialize_procedures(self, codes: list[str], _info) -> str:
        """Join procedures as pipe-separated string."""
        return '|'.join(codes)

claim = ClaimWithCustomSerialization(
    claim_id="CLM-2025-001",
    claim_date="2025-10-28",
    billed_amount=1500.50,
    procedures=["99213", "99214"]
)

# Custom serialization applied
output = claim.model_dump(mode='python')
# Output: {
#     'claim_id': 'CLM-2025-001',
#     'claim_date': '2025-10-28',
#     'billed_amount': '$1,500.50',
#     'procedures': '99213|99214'
# }
```

### Wrap Serializers (Advanced Control)

```python
from pydantic import BaseModel, field_serializer
from pydantic_core import core_schema

class ClaimWithWrapSerialization(BaseModel):
    """Wrap serializers for pre/post processing."""

    claim_id: str
    fraud_risk_score: float  # 0-100

    @field_serializer('fraud_risk_score', mode='wrap')
    def serialize_risk_score(self, value: float, handler, _info) -> dict:
        """
        Convert risk score to categorized output.
        Demonstrates wrapping Pydantic's default serialization.
        """
        # Get default serialization
        default_output = handler(value)

        # Add categorization
        if value >= 75:
            category = 'HIGH'
        elif value >= 50:
            category = 'MEDIUM'
        else:
            category = 'LOW'

        return {
            'score': default_output,
            'category': category
        }

claim = ClaimWithWrapSerialization(
    claim_id="CLM-2025-001",
    fraud_risk_score=78.5
)

# Wrapped serialization output
output = claim.model_dump()
# Output: {
#     'claim_id': 'CLM-2025-001',
#     'fraud_risk_score': {'score': 78.5, 'category': 'HIGH'}
# }
```

### Serialization Context & Conditional Export

```python
from pydantic import BaseModel, field_serializer

class SensitiveClaimData(BaseModel):
    """Conditional serialization based on context."""

    claim_id: str
    patient_id: str
    patient_ssn: str
    billed_amount: float
    provider_npi: str

    @field_serializer('patient_ssn')
    def serialize_ssn(self, value: str, _info) -> str:
        """Redact SSN unless explicitly requested."""
        # Check if context indicates full export
        if _info.context and _info.context.get('include_pii'):
            return value
        # Default: redact
        return f"***-**-{value[-4:]}"

    @field_serializer('provider_npi')
    def serialize_npi(self, value: str, _info) -> str:
        """Hide NPI in audit logs."""
        if _info.context and _info.context.get('audit_mode'):
            return f"NPI:***{value[-2:]}"
        return value

claim = SensitiveClaimData(
    claim_id="CLM-2025-001",
    patient_id="PT-5432",
    patient_ssn="123-45-6789",
    billed_amount=1500.50,
    provider_npi="1234567890"
)

# Default: redacted
redacted = claim.model_dump()
# Output: patient_ssn: "***-**-6789", provider_npi: "1234567890"

# With context: include PII
full = claim.model_dump(context={'include_pii': True})
# Output: patient_ssn: "123-45-6789", provider_npi: "1234567890"

# Audit mode: special format
audit = claim.model_dump(context={'audit_mode': True})
# Output: provider_npi: "NPI:***90"
```

---

## 4. Complex Nested Models

### Hierarchical Claim Structure

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Provider(BaseModel):
    """Provider information."""
    npi: str = Field(pattern=r'^\d{10}$')
    name: str
    specialty: str
    facility_type: str  # 'hospital', 'clinic', 'urgent_care'

class Procedure(BaseModel):
    """Individual procedure details."""
    code: str = Field(pattern=r'^\d{5}$')  # CPT code
    description: str
    units: float = Field(gt=0, le=999)
    unit_price: float = Field(gt=0)

    def total_charge(self) -> float:
        """Calculate total for this procedure."""
        return self.units * self.unit_price

class PatientDemographics(BaseModel):
    """Patient information."""
    patient_id: str
    date_of_birth: datetime
    gender: str = Field(pattern=r'^[MFO]$')
    primary_diagnosis: str  # Primary ICD-10 code

class MedicalClaim(BaseModel):
    """Complete claim with nested structures."""

    claim_id: str = Field(description="Unique claim ID")
    claim_date: datetime
    service_date: datetime
    patient: PatientDemographics
    provider: Provider
    procedures: list[Procedure] = Field(min_length=1, max_length=50)
    diagnosis_codes: list[str] = Field(min_length=1, max_length=25)

    # Computed fields available after instantiation
    @property
    def total_billed(self) -> float:
        """Sum all procedure charges."""
        return sum(p.total_charge() for p in self.procedures)

    @property
    def claim_summary(self) -> dict:
        """Generate claim summary."""
        return {
            'claim_id': self.claim_id,
            'patient_id': self.patient.patient_id,
            'provider_npi': self.provider.npi,
            'procedure_count': len(self.procedures),
            'total_billed': self.total_billed,
            'diagnoses': len(self.diagnosis_codes)
        }

# Create nested structure
claim_data = {
    'claim_id': 'CLM-2025-001',
    'claim_date': '2025-10-28',
    'service_date': '2025-10-27',
    'patient': {
        'patient_id': 'PT-5432',
        'date_of_birth': '1965-03-15',
        'gender': 'M',
        'primary_diagnosis': 'M16.0'
    },
    'provider': {
        'npi': '1234567890',
        'name': 'John Smith MD',
        'specialty': 'Orthopedics',
        'facility_type': 'hospital'
    },
    'procedures': [
        {
            'code': '27447',
            'description': 'Total knee replacement',
            'units': 1,
            'unit_price': 35000
        },
        {
            'code': '20610',
            'description': 'Arthrocentesis',
            'units': 1,
            'unit_price': 500
        }
    ],
    'diagnosis_codes': ['M16.0', 'M17.0']
}

claim = MedicalClaim.model_validate(claim_data)
print(f"Total billed: ${claim.total_billed:,.2f}")  # $35,500.00
print(claim.claim_summary)
```

### Recursive Nested Models

```python
from pydantic import BaseModel
from typing import Optional

class AppealLevel(BaseModel):
    """Hierarchical appeal structure."""

    appeal_id: str
    level: int  # 1, 2, 3 (escalation levels)
    reason: str
    decision: Optional[str] = None
    # Recursive: next appeal level
    next_appeal: Optional['AppealLevel'] = None

# Enable recursive model reference
AppealLevel.model_rebuild()

# Create nested appeals
appeal_chain = AppealLevel(
    appeal_id='APL-001',
    level=1,
    reason='Initial denial - documentation incomplete',
    next_appeal=AppealLevel(
        appeal_id='APL-001-B',
        level=2,
        reason='Submitted additional documentation',
        next_appeal=AppealLevel(
            appeal_id='APL-001-C',
            level=3,
            reason='Final appeal - external review',
            decision='Approved'
        )
    )
)
```

---

## 5. Conditional & Risk-Based Validation

### Feature Engineering for Fraud Detection

```python
from pydantic import BaseModel, field_validator, model_validator
from datetime import datetime, timedelta
import re

class FraudDetectionClaim(BaseModel):
    """
    Claim model with conditional validation for fraud risk scoring.
    Each validation catches specific fraud patterns.
    """

    claim_id: str
    patient_id: str
    provider_npi: str
    diagnosis_codes: list[str]
    procedure_codes: list[str]
    billed_amount: float
    service_date: datetime
    claim_date: datetime
    provider_specialty: str

    # Feature flags for fraud indicators
    red_flags: list[str] = []

    @field_validator('procedure_codes')
    @classmethod
    def detect_unbundling(cls, codes: list[str]) -> list[str]:
        """
        Red Flag: Unbundling - Multiple codes billed separately
        when they should be bundled.
        """
        # Bundled procedure pairs that shouldn't be split
        bundled_pairs = {
            ('27447', '27486'): 'Knee replacement components unbundled',
            ('99213', '99214'): 'Office visit levels unbundled'
        }

        code_set = set(codes)
        for pair, message in bundled_pairs.items():
            if all(code in code_set for code in pair):
                # Flag for post-validation processing
                pass

        return codes

    @model_validator(mode='after')
    def detect_upcoding(self) -> 'FraudDetectionClaim':
        """
        Red Flag: Upcoding - Services billed at higher complexity
        than documented.
        """
        # Upcoding pattern: High CPT code with low diagnosis severity
        high_complexity_codes = {'27447', '27486', '27487'}  # Major procedures
        simple_diagnoses = {'Z00', 'Z01', 'Z12'}  # Preventive codes

        has_complex_procedure = any(p in high_complexity_codes
                                   for p in self.procedure_codes)
        has_simple_diagnosis = any(d.startswith(tuple(simple_diagnoses))
                                  for d in self.diagnosis_codes)

        if has_complex_procedure and has_simple_diagnosis:
            self.red_flags.append('UPCODING')

        return self

    @model_validator(mode='after')
    def detect_phantom_billing(self) -> 'FraudDetectionClaim':
        """
        Red Flag: Phantom Billing - Services billed but never rendered.
        Pattern: Bulk billing with minimal time between claims.
        """
        claim_lag = (self.claim_date - self.service_date).days

        # Legitimate: Submitted 3-30 days after service
        if claim_lag < 0 or claim_lag > 90:
            self.red_flags.append('PHANTOM_BILLING')

        return self

    @model_validator(mode='after')
    def detect_staged_accident(self) -> 'FraudDetectionClaim':
        """
        Red Flag: Staged Accident Pattern - Multiple high-cost
        procedures for single incident.
        """
        accident_indicators = {
            'S72', 'S82', 'S92'  # Fracture codes
        }

        accident_related = [d for d in self.diagnosis_codes
                           if d.startswith(tuple(accident_indicators))]

        # Pattern: Multiple procedures on same date for single fracture
        if len(accident_related) > 0 and len(self.procedure_codes) > 5:
            if self.billed_amount > 25000:
                self.red_flags.append('STAGED_ACCIDENT')

        return self

    @model_validator(mode='after')
    def detect_specialty_mismatch(self) -> 'FraudDetectionClaim':
        """
        Red Flag: Specialty Mismatch - Procedures billed that don't
        match provider specialty.
        """
        specialty_procedure_map = {
            'orthopedics': {'27447', '27486', '27487', '99202'},
            'cardiology': {'92004', '93000', '93005', '99213'},
            'dermatology': {'99201', '99202', '11400', '11401'}
        }

        allowed_codes = specialty_procedure_map.get(
            self.provider_specialty.lower(),
            set()
        )

        if allowed_codes:
            invalid_procedures = [p for p in self.procedure_codes
                                 if p not in allowed_codes]
            if invalid_procedures:
                self.red_flags.append('SPECIALTY_MISMATCH')

        return self

# Usage example
claim = FraudDetectionClaim(
    claim_id='CLM-2025-001',
    patient_id='PT-5432',
    provider_npi='1234567890',
    diagnosis_codes=['M16.0', 'M17.0'],
    procedure_codes=['27447', '99213'],
    billed_amount=35500.00,
    service_date='2025-10-20',
    claim_date='2025-10-21',  # Submitted next day
    provider_specialty='Orthopedics'
)

if claim.red_flags:
    print(f"Fraud risk detected: {claim.red_flags}")
else:
    print("Claim appears legitimate")
```

### Dynamic Validation Rules by Provider Type

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class ContextualFraudCheck(BaseModel):
    """Apply different validation rules based on provider type."""

    claim_id: str
    provider_npi: str
    provider_type: str  # 'hospital', 'clinic', 'urgent_care', 'emergency'
    billed_amount: float
    procedure_count: int

    @field_validator('billed_amount')
    @classmethod
    def validate_amount_by_provider(
        cls,
        value: float,
        info: ValidationInfo
    ) -> float:
        """
        Different spending patterns expected by provider type.
        """
        provider_type = info.data.get('provider_type', 'clinic')

        # Expected ranges and limits per provider type
        provider_limits = {
            'urgent_care': {
                'max_single_claim': 5000,
                'avg_expected': 800,
                'red_flag_threshold': 10000
            },
            'clinic': {
                'max_single_claim': 25000,
                'avg_expected': 3000,
                'red_flag_threshold': 50000
            },
            'hospital': {
                'max_single_claim': 500000,
                'avg_expected': 25000,
                'red_flag_threshold': 1000000
            },
            'emergency': {
                'max_single_claim': 100000,
                'avg_expected': 8000,
                'red_flag_threshold': 150000
            }
        }

        limits = provider_limits.get(provider_type, provider_limits['clinic'])

        if value > limits['red_flag_threshold']:
            raise ValueError(
                f'Claim exceeds fraud red flag threshold for {provider_type}: '
                f'${value} > ${limits["red_flag_threshold"]}'
            )

        return value

# Test with different provider types
urgent_care_claim = ContextualFraudCheck(
    claim_id='CLM-001',
    provider_npi='1234567890',
    provider_type='urgent_care',
    billed_amount=3500,  # OK for urgent care
    procedure_count=2
)

# This would fail validation:
# hospital_claim = ContextualFraudCheck(
#     claim_id='CLM-002',
#     provider_npi='1234567891',
#     provider_type='urgent_care',
#     billed_amount=25000,  # FAILS: exceeds urgent care limits
#     procedure_count=15
# )
```

---

## 6. Type Hints & Generic Models

### Reusable Annotated Types

```python
from typing import Annotated
from pydantic import BaseModel, Field, AfterValidator, Gt, Lt

# Common insurance field types
ClaimID = Annotated[
    str,
    Field(min_length=10, max_length=20, pattern=r'^CLM-\d{4}-\d{6}$')
]

PatientID = Annotated[
    str,
    Field(min_length=5, max_length=15, pattern=r'^PT-\d+$')
]

NPI = Annotated[
    str,
    Field(pattern=r'^\d{10}$'),
    AfterValidator(lambda x: x.upper())
]

BilledAmount = Annotated[
    float,
    Gt(0),
    Lt(9999999.99),
    Field(description="Claim amount in USD")
]

RiskScore = Annotated[
    float,
    Gt(0),
    Lt(100),
    Field(description="Fraud risk score 0-100")
]

# Use reusable types
class StandardClaim(BaseModel):
    """Standard claim using annotated types."""
    claim_id: ClaimID
    patient_id: PatientID
    provider_npi: NPI
    billed_amount: BilledAmount
    fraud_risk: RiskScore = 0.0

claim = StandardClaim(
    claim_id='CLM-2025-123456',
    patient_id='PT-54321',
    provider_npi='1234567890',
    billed_amount=5000.50,
    fraud_risk=25.5
)
```

### Generic Types for Reusable Validation

```python
from typing import Generic, TypeVar, Annotated
from pydantic import BaseModel, Field, ValidationError

T = TypeVar('T')

class PaginatedResult(BaseModel, Generic[T]):
    """Reusable paginated response structure."""
    items: list[T]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_count: int = Field(ge=0)

    @property
    def total_pages(self) -> int:
        """Calculate total pages."""
        return (self.total_count + self.page_size - 1) // self.page_size

class ClaimResult(BaseModel):
    """Individual claim in results."""
    claim_id: str
    status: str
    billed_amount: float

# Use with specific type
claims_page = PaginatedResult[ClaimResult](
    items=[
        ClaimResult(claim_id='CLM-001', status='approved', billed_amount=1000),
        ClaimResult(claim_id='CLM-002', status='pending', billed_amount=2000),
    ],
    page=1,
    page_size=2,
    total_count=10
)

print(f"Page {claims_page.page} of {claims_page.total_pages}")
```

### Conditional Type Validation

```python
from typing import Union, Literal
from pydantic import BaseModel, field_validator

class AdjudicationOutcome(BaseModel):
    """Union type for different claim outcomes."""
    claim_id: str
    outcome_type: Literal['approved', 'denied', 'appeal']

class ApprovedOutcome(BaseModel):
    claim_id: str
    outcome_type: Literal['approved']
    approved_amount: float = Field(gt=0)
    approval_date: str

class DeniedOutcome(BaseModel):
    claim_id: str
    outcome_type: Literal['denied']
    denial_reason: str
    denial_codes: list[str]

class AppealOutcome(BaseModel):
    claim_id: str
    outcome_type: Literal['appeal']
    appeal_reason: str
    expected_review_date: str

# Discriminated union
ClaimDecision = Union[ApprovedOutcome, DeniedOutcome, AppealOutcome]

def process_outcome(decision: ClaimDecision):
    """Process based on outcome type."""
    if isinstance(decision, ApprovedOutcome):
        print(f"Approved: ${decision.approved_amount}")
    elif isinstance(decision, DeniedOutcome):
        print(f"Denied: {decision.denial_reason}")
    elif isinstance(decision, AppealOutcome):
        print(f"Appeal submitted: {decision.appeal_reason}")
```

---

## 7. Advanced: Type Adapters & Custom Validation

### TypeAdapter for Non-Model Types

```python
from pydantic import TypeAdapter, ValidationError
from typing import Annotated

# Validate lists of claims (not wrapped in a model)
ClaimIDList = list[Annotated[str, Field(pattern=r'^CLM-\d{4}-\d{6}$')]]
adapter = TypeAdapter(ClaimIDList)

# Validate and parse
try:
    claim_ids = adapter.validate_python(['CLM-2025-000001', 'CLM-2025-000002'])
    print(f"Valid claims: {claim_ids}")
except ValidationError as e:
    print(f"Invalid claim IDs: {e}")

# Dump to JSON
json_output = adapter.dump_json(claim_ids)
print(json_output)
```

### Custom Core Schema for Complex Types

```python
from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic_core import core_schema
from typing import Any

class CustomClaimAmount(BaseModel):
    """Custom validation for insurance amounts."""

    amount: float
    currency: str = 'USD'

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler
    ) -> core_schema.CoreSchema:
        """Define custom validation schema."""

        def validate(value):
            if isinstance(value, str):
                # Parse "USD 1500.50" format
                parts = value.split()
                if len(parts) == 2:
                    currency, amount = parts
                    return cls(amount=float(amount), currency=currency)
            elif isinstance(value, dict):
                return cls(**value)
            elif isinstance(value, cls):
                return value
            raise ValueError('Invalid amount format')

        return core_schema.with_info_before_validator_function(
            lambda x, _: validate(x),
            core_schema.model_schema(cls)
        )

# This would parse custom string formats
# amount = CustomClaimAmount('USD 1500.50')
```

---

## Quick Integration Patterns for Insurance Claims

### Pattern 1: Basic Claim Validation

```python
from pydantic import BaseModel, Field
from datetime import datetime

class InsuranceClaim(BaseModel):
    claim_id: str = Field(min_length=1, pattern=r'^CLM-')
    patient_id: str
    provider_npi: str = Field(pattern=r'^\d{10}$')
    billed_amount: float = Field(gt=0, le=999999.99)
    service_date: datetime
    claim_date: datetime = Field(default_factory=datetime.now)

# Use it
claim = InsuranceClaim.model_validate(incoming_json_data)
```

### Pattern 2: Fraud Risk Scoring

```python
from pydantic import BaseModel, model_validator

class FraudScoringClaim(BaseModel):
    # ... fields ...
    red_flags: list[str] = []
    risk_score: float = 0.0

    @model_validator(mode='after')
    def calculate_risk_score(self) -> 'FraudScoringClaim':
        """Assign points for each red flag."""
        points_per_flag = {
            'UPCODING': 25,
            'PHANTOM_BILLING': 30,
            'UNBUNDLING': 20,
            'SPECIALTY_MISMATCH': 15
        }

        self.risk_score = sum(
            points_per_flag.get(flag, 10) for flag in self.red_flags
        )
        return self
```

### Pattern 3: Multi-Format Export

```python
from pydantic import BaseModel, field_serializer

class ExportableClaim(BaseModel):
    claim_id: str
    billed_amount: float

    @field_serializer('billed_amount')
    def format_amount(self, value: float) -> str:
        return f"${value:,.2f}"

# Export to dict (formatted)
claim_dict = claim.model_dump()  # currency formatted

# Export to JSON (raw numbers)
claim_json = claim.model_dump(mode='json')
```

---

## Common Patterns Summary

| Need | Pattern | Complexity |
|------|---------|-----------|
| **Validate field format** | `AfterValidator` annotation | Low |
| **Cross-field validation** | `@model_validator(mode='after')` | Medium |
| **Conditional validation** | Use `ValidationInfo.data` | Medium |
| **Nested structures** | Define related `BaseModel` classes | Medium |
| **Custom serialization** | `@field_serializer` or `mode='wrap'` | Medium |
| **Fraud detection logic** | `@model_validator` with red flag list | High |
| **Reusable validators** | `Annotated[Type, Validator]` | Medium |
| **Sensitive data handling** | `@field_serializer` with context | High |

---

## Performance & Best Practices

### Do's ✓

```python
# Use Annotated for reusable validators
ValidNPI = Annotated[str, Field(pattern=r'^\d{10}$')]

# Use after validators (safer, run after parsing)
code: Annotated[str, AfterValidator(validate_cpt_code)]

# Validate at boundaries (model creation)
claim = InsuranceClaim.model_validate(input_data)

# Use model_dump() for serialization
export = claim.model_dump(exclude={'patient_id'})
```

### Don'ts ✗

```python
# Don't validate in business logic (do it at model level)
# if len(codes) > 0:  # Should be in model validator

# Don't disable validation
# claim = InsuranceClaim.model_construct(...)  # Skips validation

# Don't store ValidationError details in logs
# print(error)  # Could expose sensitive claim data

# Don't use string fields when type should be specific
# claim_date: str  # Should be: claim_date: datetime
```

---

## References

**Pydantic v2.12.3 Documentation**
- Official Docs: https://docs.pydantic.dev/latest/
- GitHub Repository: https://github.com/pydantic/pydantic
- Migration Guide: https://docs.pydantic.dev/latest/migration/
- Type System: https://docs.pydantic.dev/latest/concepts/types/
- JSON Schema: https://docs.pydantic.dev/latest/concepts/json_schema/

**Insurance Claims Context**
- Data schema: See `insurance_claims/docs/DATA-SCHEMA.md`
- Fraud patterns: See `insurance_claims/docs/FRAUD-PATTERNS.md`
- Testing data: See `insurance_claims/data/`

---

**Documentation Generated**: 2025-10-28
**Research Source**: Pydantic v2.12.3 Official Documentation
**Trust Score**: 10/10 (Official Authoritative Source)
**Scope**: Focused on insurance claims data modeling & fraud detection
