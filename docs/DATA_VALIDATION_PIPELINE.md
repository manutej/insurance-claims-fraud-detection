# Data Validation Pipeline for Insurance Claims Fraud Detection

## Executive Summary

This document outlines a comprehensive multi-stage data validation pipeline designed to ensure high data quality for the insurance fraud detection system. The pipeline validates schema compliance, medical coding accuracy, business rules, and data completeness before claims enter the fraud detection workflow.

## 1. Validation Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│              Data Validation Pipeline                         │
└──────────────────────────────────────────────────────────────┘

                    Incoming Claim Data
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │         Stage 1: Schema Validation            │
    │  • Field presence and types (Pydantic)        │
    │  • Format validation (regex, ranges)          │
    │  • Required vs optional fields                │
    └───────────────┬───────────────────────────────┘
                    │                    ❌ Schema Invalid
                    │                    └──► [Dead Letter Queue]
                    ▼ ✅ Schema Valid
    ┌───────────────────────────────────────────────┐
    │     Stage 2: Medical Code Validation          │
    │  • ICD-10 code existence                      │
    │  • CPT code validity                          │
    │  • NDC code format                            │
    │  • Code combination rules                     │
    └───────────────┬───────────────────────────────┘
                    │                    ❌ Codes Invalid
                    │                    └──► [Manual Review Queue]
                    ▼ ✅ Codes Valid
    ┌───────────────────────────────────────────────┐
    │      Stage 3: Business Rules Validation       │
    │  • Temporal logic (dates)                     │
    │  • Financial constraints (amounts)            │
    │  • Geographic constraints                     │
    │  • Provider eligibility                       │
    └───────────────┬───────────────────────────────┘
                    │                    ❌ Rules Failed
                    │                    └──► [Exceptions Queue]
                    ▼ ✅ Rules Passed
    ┌───────────────────────────────────────────────┐
    │      Stage 4: Data Quality Assessment         │
    │  • Completeness scoring                       │
    │  • Consistency checks                         │
    │  • Anomaly detection                          │
    │  • Data enrichment validation                 │
    └───────────────┬───────────────────────────────┘
                    │                    ⚠️  Quality Issues
                    │                    └──► [Quality Alerts]
                    ▼ ✅ Quality Acceptable
    ┌───────────────────────────────────────────────┐
    │      Stage 5: Completeness Assessment         │
    │  • Required data present                      │
    │  • Enrichment data availability               │
    │  • Confidence scoring                         │
    │  • Readiness for fraud detection              │
    └───────────────┬───────────────────────────────┘
                    │
                    ▼ ✅ Complete & Valid
            [Fraud Detection Pipeline]
```

## 2. Stage 1: Schema Validation (Pydantic)

### 2.1 Core Claim Schema

```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal
import re

class ClaimSchema(BaseModel):
    """
    Comprehensive Pydantic schema for insurance claims.
    """

    # Required Fields
    claim_id: str = Field(
        ...,
        regex=r'^CLM\d{10}$',
        description="Unique claim identifier (CLM + 10 digits)"
    )

    patient_id: str = Field(
        ...,
        regex=r'^PAT\d{8}$',
        description="Patient identifier (PAT + 8 digits)"
    )

    provider_npi: str = Field(
        ...,
        regex=r'^\d{10}$',
        description="National Provider Identifier (10 digits)"
    )

    # Medical Codes
    diagnosis_codes: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="ICD-10 diagnosis codes"
    )

    procedure_codes: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="CPT procedure codes"
    )

    # Dates
    claim_date: datetime = Field(
        ...,
        description="Date claim submitted"
    )

    service_date_from: date = Field(
        ...,
        description="Start date of service"
    )

    service_date_to: Optional[date] = Field(
        None,
        description="End date of service (for date ranges)"
    )

    # Financial
    billed_amount: Decimal = Field(
        ...,
        ge=0,
        le=1000000,
        decimal_places=2,
        description="Total billed amount"
    )

    allowed_amount: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Approved/allowed amount"
    )

    # Optional Fields
    medication_codes: Optional[List[str]] = Field(
        None,
        description="NDC medication codes"
    )

    modifier_codes: Optional[List[str]] = Field(
        None,
        description="CPT modifier codes"
    )

    place_of_service: Optional[str] = Field(
        None,
        regex=r'^\d{2}$',
        description="Place of service code (2 digits)"
    )

    # Metadata
    submission_source: str = Field(
        ...,
        regex=r'^(api|batch|edi|portal)$',
        description="Source of claim submission"
    )

    # Fraud Indicators (for training data)
    fraud_indicator: Optional[bool] = Field(
        None,
        description="Ground truth fraud indicator (training only)"
    )

    fraud_type: Optional[str] = Field(
        None,
        description="Type of fraud (if fraud_indicator is True)"
    )

    red_flags: Optional[List[str]] = Field(
        None,
        description="Known red flags"
    )

    # Field-Level Validators
    @validator('diagnosis_codes', each_item=True)
    def validate_icd10_format(cls, code: str) -> str:
        """
        Validate ICD-10 code format.
        Format: Letter + 2 digits + optional decimal + 1-4 digits
        Examples: A00, A00.1, Z99.89
        """
        pattern = r'^[A-Z]\d{2}(\.\d{1,4})?$'
        if not re.match(pattern, code):
            raise ValueError(
                f"Invalid ICD-10 format: {code}. "
                f"Expected format: Letter + 2 digits + optional decimal"
            )
        return code.upper()

    @validator('procedure_codes', each_item=True)
    def validate_cpt_format(cls, code: str) -> str:
        """
        Validate CPT code format.
        Format: 5 digits or 4 digits + 1 letter
        Examples: 99213, 0001U
        """
        pattern = r'^(\d{5}|\d{4}[A-Z])$'
        if not re.match(pattern, code):
            raise ValueError(
                f"Invalid CPT format: {code}. "
                f"Expected format: 5 digits or 4 digits + letter"
            )
        return code.upper()

    @validator('medication_codes', each_item=True)
    def validate_ndc_format(cls, code: str) -> str:
        """
        Validate NDC code format.
        Format: 11 digits with hyphens (5-4-2 or 5-3-2)
        Examples: 12345-678-90, 12345-6789-01
        """
        pattern = r'^\d{5}-\d{3,4}-\d{2}$'
        if not re.match(pattern, code):
            raise ValueError(
                f"Invalid NDC format: {code}. "
                f"Expected format: XXXXX-XXX(X)-XX"
            )
        return code

    @validator('provider_npi')
    def validate_npi_checksum(cls, npi: str) -> str:
        """
        Validate NPI using Luhn algorithm.
        """
        if not cls._luhn_checksum(npi):
            raise ValueError(f"Invalid NPI checksum: {npi}")
        return npi

    @staticmethod
    def _luhn_checksum(npi: str) -> bool:
        """
        Luhn algorithm for NPI validation.
        """
        # Add "80840" prefix as per NPI spec
        number = "80840" + npi

        # Luhn algorithm
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))

        return checksum % 10 == 0

    # Cross-Field Validators
    @root_validator
    def validate_date_logic(cls, values):
        """
        Validate temporal relationships between dates.
        """
        service_from = values.get('service_date_from')
        service_to = values.get('service_date_to')
        claim_date = values.get('claim_date')

        if service_from and claim_date:
            # Service date cannot be after claim date
            if service_from > claim_date.date():
                raise ValueError(
                    "service_date_from cannot be after claim_date"
                )

            # Service date cannot be more than 2 years in past
            days_ago = (claim_date.date() - service_from).days
            if days_ago > 730:  # 2 years
                raise ValueError(
                    f"service_date_from is {days_ago} days ago. "
                    f"Maximum allowed: 730 days"
                )

        if service_from and service_to:
            # End date must be after or equal to start date
            if service_to < service_from:
                raise ValueError(
                    "service_date_to must be >= service_date_from"
                )

            # Service range cannot exceed 90 days
            days_diff = (service_to - service_from).days
            if days_diff > 90:
                raise ValueError(
                    f"Service date range ({days_diff} days) exceeds maximum: 90 days"
                )

        return values

    @root_validator
    def validate_fraud_fields(cls, values):
        """
        Validate fraud indicator fields are consistent.
        """
        fraud_indicator = values.get('fraud_indicator')
        fraud_type = values.get('fraud_type')

        if fraud_indicator is True and not fraud_type:
            raise ValueError(
                "fraud_type required when fraud_indicator is True"
            )

        if fraud_indicator is False and fraud_type:
            raise ValueError(
                "fraud_type should not be set when fraud_indicator is False"
            )

        return values

    @root_validator
    def validate_financial_logic(cls, values):
        """
        Validate financial field relationships.
        """
        billed = values.get('billed_amount')
        allowed = values.get('allowed_amount')

        if billed and allowed:
            # Allowed amount should not exceed billed amount
            if allowed > billed:
                raise ValueError(
                    f"allowed_amount (${allowed}) cannot exceed "
                    f"billed_amount (${billed})"
                )

        return values

    class Config:
        # Pydantic configuration
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class ValidationResult(BaseModel):
    """
    Result of validation process.
    """
    valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    severity: str  # 'error', 'warning', 'info'
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SchemaValidator:
    """
    Service for schema validation.
    """
    def __init__(self):
        self.schema = ClaimSchema

    async def validate(self, claim_data: dict) -> ValidationResult:
        """
        Validate claim against schema.
        """
        errors = []
        warnings = []

        try:
            # Attempt to parse with Pydantic
            validated_claim = self.schema(**claim_data)

            # Check for optional fields that are missing
            if not validated_claim.allowed_amount:
                warnings.append({
                    "field": "allowed_amount",
                    "message": "allowed_amount not provided",
                    "impact": "May affect reimbursement calculations"
                })

            return ValidationResult(
                valid=True,
                errors=errors,
                warnings=warnings,
                severity='info' if warnings else 'success'
            )

        except ValidationError as e:
            # Parse Pydantic errors
            for error in e.errors():
                errors.append({
                    "field": ".".join(str(loc) for loc in error['loc']),
                    "message": error['msg'],
                    "type": error['type'],
                    "value": error.get('ctx', {})
                })

            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                severity='error'
            )
```

## 3. Stage 2: Medical Code Validation

### 3.1 ICD-10 Validation Service

```python
from typing import Set, Dict
import aiofiles
import json

class ICD10Validator:
    """
    Validates ICD-10 diagnosis codes against official dataset.
    """
    def __init__(self, icd10_file: str = "resources/icd10_codes.json"):
        self.icd10_codes: Dict[str, Dict] = {}
        self.code_relationships: Dict[str, List[str]] = {}

    async def load_codes(self) -> None:
        """
        Load ICD-10 codes from reference file.
        """
        async with aiofiles.open(self.icd10_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)

            self.icd10_codes = {
                code['code']: {
                    'description': code['description'],
                    'category': code['category'],
                    'billable': code.get('billable', True),
                    'gender_specific': code.get('gender_specific')
                }
                for code in data
            }

    async def validate_code(self, code: str) -> ValidationResult:
        """
        Validate single ICD-10 code.
        """
        errors = []
        warnings = []

        # Check if code exists
        if code not in self.icd10_codes:
            errors.append({
                "code": code,
                "message": f"Unknown ICD-10 code: {code}",
                "severity": "error"
            })
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                severity='error'
            )

        code_info = self.icd10_codes[code]

        # Check if code is billable
        if not code_info['billable']:
            warnings.append({
                "code": code,
                "message": f"Non-billable ICD-10 code: {code}",
                "severity": "warning"
            })

        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            severity='warning' if warnings else 'success'
        )

    async def validate_codes_batch(
        self,
        codes: List[str]
    ) -> ValidationResult:
        """
        Validate multiple ICD-10 codes.
        """
        results = await asyncio.gather(*[
            self.validate_code(code) for code in codes
        ])

        all_errors = []
        all_warnings = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            severity='error' if all_errors else ('warning' if all_warnings else 'success')
        )

    async def validate_code_combinations(
        self,
        diagnosis_codes: List[str],
        patient_gender: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate logical combinations of diagnosis codes.
        """
        warnings = []
        errors = []

        # Check for conflicting diagnoses
        conflicts = self._check_conflicting_diagnoses(diagnosis_codes)
        if conflicts:
            warnings.extend(conflicts)

        # Check gender-specific codes
        if patient_gender:
            gender_issues = self._check_gender_specific_codes(
                diagnosis_codes,
                patient_gender
            )
            if gender_issues:
                errors.extend(gender_issues)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='error' if errors else ('warning' if warnings else 'success')
        )

    def _check_conflicting_diagnoses(
        self,
        codes: List[str]
    ) -> List[Dict]:
        """
        Check for mutually exclusive diagnoses.
        """
        conflicts = []

        # Example: Cannot have both pregnancy and male-specific codes
        pregnancy_codes = [c for c in codes if c.startswith('O')]
        male_specific = [
            c for c in codes
            if self.icd10_codes.get(c, {}).get('gender_specific') == 'male'
        ]

        if pregnancy_codes and male_specific:
            conflicts.append({
                "codes": pregnancy_codes + male_specific,
                "message": "Conflicting gender-specific diagnoses",
                "severity": "warning"
            })

        return conflicts

    def _check_gender_specific_codes(
        self,
        codes: List[str],
        patient_gender: str
    ) -> List[Dict]:
        """
        Validate gender-specific codes match patient gender.
        """
        errors = []

        for code in codes:
            code_info = self.icd10_codes.get(code, {})
            required_gender = code_info.get('gender_specific')

            if required_gender and required_gender != patient_gender.lower():
                errors.append({
                    "code": code,
                    "message": f"Code {code} is {required_gender}-specific, "
                               f"patient is {patient_gender}",
                    "severity": "error"
                })

        return errors
```

### 3.2 CPT Validation Service

```python
class CPTValidator:
    """
    Validates CPT procedure codes.
    """
    def __init__(self, cpt_file: str = "resources/cpt_codes.json"):
        self.cpt_codes: Dict[str, Dict] = {}
        self.bundled_codes: Dict[str, List[str]] = {}
        self.mutually_exclusive: Dict[str, List[str]] = {}

    async def load_codes(self) -> None:
        """
        Load CPT codes and relationships.
        """
        async with aiofiles.open(self.cpt_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)

            self.cpt_codes = {
                code['code']: {
                    'description': code['description'],
                    'category': code['category'],
                    'typical_reimbursement': code.get('typical_reimbursement'),
                    'modifiers_allowed': code.get('modifiers_allowed', [])
                }
                for code in data
            }

            # Load bundled codes (NCCI edits)
            self.bundled_codes = data.get('bundled_codes', {})
            self.mutually_exclusive = data.get('mutually_exclusive', {})

    async def validate_code(self, code: str) -> ValidationResult:
        """
        Validate single CPT code.
        """
        errors = []
        warnings = []

        if code not in self.cpt_codes:
            errors.append({
                "code": code,
                "message": f"Unknown CPT code: {code}",
                "severity": "error"
            })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='error' if errors else 'success'
        )

    async def validate_unbundling(
        self,
        procedure_codes: List[str]
    ) -> ValidationResult:
        """
        Check for unbundling (billing separate codes for bundled procedures).
        """
        warnings = []

        for code in procedure_codes:
            if code in self.bundled_codes:
                bundled_with = self.bundled_codes[code]
                present_bundled = [c for c in bundled_with if c in procedure_codes]

                if present_bundled:
                    warnings.append({
                        "code": code,
                        "bundled_with": present_bundled,
                        "message": f"Potential unbundling: {code} is typically bundled with {present_bundled}",
                        "severity": "warning",
                        "fraud_indicator": "unbundling"
                    })

        return ValidationResult(
            valid=True,  # Warnings, not errors
            errors=[],
            warnings=warnings,
            severity='warning' if warnings else 'success'
        )

    async def validate_mutually_exclusive(
        self,
        procedure_codes: List[str]
    ) -> ValidationResult:
        """
        Check for mutually exclusive procedures billed together.
        """
        errors = []

        for code in procedure_codes:
            if code in self.mutually_exclusive:
                exclusive_with = self.mutually_exclusive[code]
                present_exclusive = [c for c in exclusive_with if c in procedure_codes]

                if present_exclusive:
                    errors.append({
                        "code": code,
                        "exclusive_with": present_exclusive,
                        "message": f"Mutually exclusive codes: {code} and {present_exclusive}",
                        "severity": "error"
                    })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=[],
            severity='error' if errors else 'success'
        )

    async def validate_diagnosis_procedure_alignment(
        self,
        diagnosis_codes: List[str],
        procedure_codes: List[str]
    ) -> ValidationResult:
        """
        Verify procedures are medically necessary for diagnoses.
        """
        warnings = []

        # Load diagnosis-procedure mapping
        valid_combinations = await self._get_valid_combinations()

        for proc_code in procedure_codes:
            valid_diagnoses = valid_combinations.get(proc_code, [])

            if valid_diagnoses:
                # Check if at least one diagnosis is valid for this procedure
                has_valid_diagnosis = any(
                    diag in valid_diagnoses for diag in diagnosis_codes
                )

                if not has_valid_diagnosis:
                    warnings.append({
                        "procedure": proc_code,
                        "diagnoses": diagnosis_codes,
                        "message": f"Procedure {proc_code} may not be medically necessary for diagnoses {diagnosis_codes}",
                        "severity": "warning"
                    })

        return ValidationResult(
            valid=True,  # Warnings, not errors
            errors=[],
            warnings=warnings,
            severity='warning' if warnings else 'success'
        )
```

## 4. Stage 3: Business Rules Validation

```python
class BusinessRulesValidator:
    """
    Validates business logic and operational rules.
    """
    def __init__(self):
        self.rules_engine = RulesEngine()

    async def validate_claim(self, claim: dict) -> ValidationResult:
        """
        Apply all business rules to claim.
        """
        results = await asyncio.gather(
            self.validate_temporal_logic(claim),
            self.validate_financial_constraints(claim),
            self.validate_geographic_constraints(claim),
            self.validate_provider_eligibility(claim),
            self.validate_patient_eligibility(claim),
            self.validate_authorization_requirements(claim)
        )

        # Aggregate results
        all_errors = []
        all_warnings = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            severity='error' if all_errors else ('warning' if all_warnings else 'success')
        )

    async def validate_temporal_logic(self, claim: dict) -> ValidationResult:
        """
        Validate date and time constraints.
        """
        errors = []
        warnings = []

        service_date = claim.get('service_date_from')
        claim_date = claim.get('claim_date')

        # Timely filing rules (claims must be filed within 1 year)
        if service_date and claim_date:
            days_diff = (claim_date.date() - service_date).days

            if days_diff > 365:
                errors.append({
                    "rule": "TIMELY_FILING",
                    "message": f"Claim filed {days_diff} days after service (max: 365)",
                    "severity": "error"
                })

        # Check for future dates
        if service_date and service_date > datetime.now().date():
            errors.append({
                "rule": "FUTURE_SERVICE_DATE",
                "message": "Service date cannot be in the future",
                "severity": "error"
            })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='error' if errors else 'success'
        )

    async def validate_financial_constraints(
        self,
        claim: dict
    ) -> ValidationResult:
        """
        Validate financial amounts and calculations.
        """
        errors = []
        warnings = []

        billed_amount = claim.get('billed_amount', 0)
        procedure_codes = claim.get('procedure_codes', [])

        # Calculate expected amount based on procedures
        expected_amount = await self._calculate_expected_amount(procedure_codes)

        # Check for excessive billing (>300% of expected)
        if expected_amount and billed_amount > expected_amount * 3:
            warnings.append({
                "rule": "EXCESSIVE_BILLING",
                "message": f"Billed ${billed_amount:,.2f} vs expected ${expected_amount:,.2f}",
                "severity": "warning",
                "fraud_indicator": "upcoding"
            })

        # Check for suspicious round numbers
        if billed_amount % 100 == 0 and billed_amount > 1000:
            warnings.append({
                "rule": "ROUND_AMOUNT",
                "message": f"Suspiciously round amount: ${billed_amount:,.2f}",
                "severity": "info"
            })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='warning' if warnings else 'success'
        )

    async def validate_geographic_constraints(
        self,
        claim: dict
    ) -> ValidationResult:
        """
        Validate geographic consistency.
        """
        errors = []
        warnings = []

        patient_location = claim.get('patient_location')
        provider_location = claim.get('provider_location')
        service_date = claim.get('service_date_from')

        if patient_location and provider_location:
            # Calculate distance
            distance_miles = await self._calculate_distance(
                patient_location,
                provider_location
            )

            # Check for implausible distances
            if distance_miles > 500:
                warnings.append({
                    "rule": "GEOGRAPHIC_IMPROBABILITY",
                    "message": f"Patient traveled {distance_miles} miles for service",
                    "severity": "warning"
                })

        # Check for impossible geographic sequences
        if await self._check_temporal_impossibility(claim):
            errors.append({
                "rule": "TEMPORAL_IMPOSSIBILITY",
                "message": "Patient cannot be in multiple locations at same time",
                "severity": "error"
            })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='error' if errors else ('warning' if warnings else 'success')
        )

    async def validate_provider_eligibility(
        self,
        claim: dict
    ) -> ValidationResult:
        """
        Validate provider can perform billed services.
        """
        errors = []
        warnings = []

        provider_npi = claim.get('provider_npi')
        procedure_codes = claim.get('procedure_codes', [])

        # Fetch provider specialty
        provider_info = await self._get_provider_info(provider_npi)

        if provider_info:
            # Check if procedures match provider specialty
            for proc_code in procedure_codes:
                if not await self._is_procedure_in_specialty(
                    proc_code,
                    provider_info['specialty']
                ):
                    warnings.append({
                        "rule": "SPECIALTY_MISMATCH",
                        "message": f"Procedure {proc_code} unusual for {provider_info['specialty']}",
                        "severity": "warning"
                    })

            # Check provider license status
            if provider_info.get('license_status') != 'active':
                errors.append({
                    "rule": "INACTIVE_PROVIDER",
                    "message": f"Provider {provider_npi} has inactive license",
                    "severity": "error"
                })

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity='error' if errors else ('warning' if warnings else 'success')
        )
```

## 5. Stage 4: Data Quality Assessment

```python
class DataQualityAssessor:
    """
    Assesses overall data quality of claims.
    """
    def __init__(self):
        self.quality_metrics = {}

    async def assess_quality(self, claim: dict) -> QualityScore:
        """
        Calculate comprehensive quality score.
        """
        scores = await asyncio.gather(
            self.assess_completeness(claim),
            self.assess_consistency(claim),
            self.assess_accuracy(claim),
            self.assess_timeliness(claim)
        )

        completeness, consistency, accuracy, timeliness = scores

        # Weighted average
        overall_score = (
            completeness * 0.30 +
            consistency * 0.25 +
            accuracy * 0.25 +
            timeliness * 0.20
        )

        return QualityScore(
            overall_score=overall_score,
            completeness_score=completeness,
            consistency_score=consistency,
            accuracy_score=accuracy,
            timeliness_score=timeliness,
            quality_tier=self._determine_quality_tier(overall_score),
            issues=self._identify_issues([completeness, consistency, accuracy, timeliness])
        )

    async def assess_completeness(self, claim: dict) -> float:
        """
        Measure data completeness (0-1).
        """
        required_fields = [
            'claim_id', 'patient_id', 'provider_npi',
            'diagnosis_codes', 'procedure_codes',
            'claim_date', 'service_date_from', 'billed_amount'
        ]

        optional_fields = [
            'service_date_to', 'allowed_amount', 'medication_codes',
            'modifier_codes', 'place_of_service'
        ]

        # Check required fields (weighted heavily)
        required_present = sum(1 for field in required_fields if claim.get(field))
        required_score = required_present / len(required_fields)

        # Check optional fields (weighted lightly)
        optional_present = sum(1 for field in optional_fields if claim.get(field))
        optional_score = optional_present / len(optional_fields)

        # Combined score (required: 80%, optional: 20%)
        return required_score * 0.8 + optional_score * 0.2

    async def assess_consistency(self, claim: dict) -> float:
        """
        Measure internal consistency of data.
        """
        consistency_checks = [
            self._check_date_consistency(claim),
            self._check_amount_consistency(claim),
            self._check_code_consistency(claim),
            self._check_provider_consistency(claim)
        ]

        results = await asyncio.gather(*consistency_checks)
        return sum(results) / len(results)

    async def assess_accuracy(self, claim: dict) -> float:
        """
        Estimate data accuracy based on validation results.
        """
        # Run validators
        schema_result = await schema_validator.validate(claim)
        medical_result = await medical_code_validator.validate(claim)
        business_result = await business_rules_validator.validate(claim)

        # Calculate accuracy score
        total_checks = 3
        passed_checks = sum([
            1 if schema_result.valid else 0,
            1 if medical_result.valid else 0,
            1 if business_result.valid else 0
        ])

        return passed_checks / total_checks

    def _determine_quality_tier(self, overall_score: float) -> str:
        """
        Classify quality into tiers.
        """
        if overall_score >= 0.95:
            return "EXCELLENT"
        elif overall_score >= 0.85:
            return "GOOD"
        elif overall_score >= 0.70:
            return "ACCEPTABLE"
        elif overall_score >= 0.50:
            return "POOR"
        else:
            return "CRITICAL"
```

## 6. Stage 5: Completeness Assessment

```python
class CompletenessAssessor:
    """
    Determines if claim has sufficient data for fraud detection.
    """
    async def assess_readiness(
        self,
        claim: dict,
        quality_score: QualityScore
    ) -> ReadinessResult:
        """
        Determine if claim is ready for fraud detection.
        """
        readiness_checks = {
            "quality_score": quality_score.overall_score >= 0.70,
            "required_fields": await self._check_required_fields(claim),
            "enrichment_available": await self._check_enrichment_data(claim),
            "validation_passed": await self._check_validation_status(claim)
        }

        all_passed = all(readiness_checks.values())
        confidence = sum(1 for v in readiness_checks.values() if v) / len(readiness_checks)

        return ReadinessResult(
            ready=all_passed,
            confidence=confidence,
            readiness_checks=readiness_checks,
            blockers=[k for k, v in readiness_checks.items() if not v],
            recommendation=self._generate_recommendation(readiness_checks)
        )

    async def _check_enrichment_data(self, claim: dict) -> bool:
        """
        Check if enrichment data is available.
        """
        provider_data = await provider_service.get_provider_info(
            claim.get('provider_npi')
        )
        patient_history = await patient_service.get_history(
            claim.get('patient_id')
        )

        return bool(provider_data and patient_history)

    def _generate_recommendation(
        self,
        readiness_checks: Dict[str, bool]
    ) -> str:
        """
        Generate actionable recommendation.
        """
        if all(readiness_checks.values()):
            return "PROCEED_TO_FRAUD_DETECTION"
        elif readiness_checks.get('validation_passed') is False:
            return "REJECT_INVALID_DATA"
        elif readiness_checks.get('enrichment_available') is False:
            return "RETRY_ENRICHMENT"
        elif readiness_checks.get('quality_score') is False:
            return "MANUAL_REVIEW_REQUIRED"
        else:
            return "ADDITIONAL_DATA_NEEDED"
```

## 7. Validation Pipeline Orchestration

```python
class ValidationPipelineOrchestrator:
    """
    Orchestrates the entire validation pipeline.
    """
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.icd10_validator = ICD10Validator()
        self.cpt_validator = CPTValidator()
        self.business_rules_validator = BusinessRulesValidator()
        self.quality_assessor = DataQualityAssessor()
        self.completeness_assessor = CompletenessAssessor()

    async def validate_claim(
        self,
        claim_data: dict
    ) -> PipelineResult:
        """
        Execute full validation pipeline.
        """
        start_time = datetime.utcnow()

        try:
            # Stage 1: Schema Validation
            schema_result = await self.schema_validator.validate(claim_data)
            if not schema_result.valid:
                return PipelineResult(
                    stage_reached="SCHEMA_VALIDATION",
                    status="FAILED",
                    schema_result=schema_result,
                    processing_time=(datetime.utcnow() - start_time).total_seconds()
                )

            # Stage 2: Medical Code Validation
            medical_result = await self._validate_medical_codes(claim_data)
            if not medical_result.valid:
                return PipelineResult(
                    stage_reached="MEDICAL_CODE_VALIDATION",
                    status="FAILED",
                    schema_result=schema_result,
                    medical_result=medical_result,
                    processing_time=(datetime.utcnow() - start_time).total_seconds()
                )

            # Stage 3: Business Rules Validation
            business_result = await self.business_rules_validator.validate_claim(claim_data)

            # Stage 4: Quality Assessment
            quality_score = await self.quality_assessor.assess_quality(claim_data)

            # Stage 5: Completeness Assessment
            readiness = await self.completeness_assessor.assess_readiness(
                claim_data,
                quality_score
            )

            # Determine final status
            if readiness.ready:
                status = "PASSED"
            elif not business_result.valid:
                status = "BUSINESS_RULES_FAILED"
            elif quality_score.quality_tier in ["POOR", "CRITICAL"]:
                status = "QUALITY_INSUFFICIENT"
            else:
                status = "INCOMPLETE"

            return PipelineResult(
                stage_reached="COMPLETENESS_ASSESSMENT",
                status=status,
                schema_result=schema_result,
                medical_result=medical_result,
                business_result=business_result,
                quality_score=quality_score,
                readiness=readiness,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Validation pipeline error: {e}")
            return PipelineResult(
                stage_reached="ERROR",
                status="ERROR",
                error=str(e),
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )

    async def _validate_medical_codes(self, claim: dict) -> ValidationResult:
        """
        Comprehensive medical code validation.
        """
        results = await asyncio.gather(
            self.icd10_validator.validate_codes_batch(claim.get('diagnosis_codes', [])),
            self.cpt_validator.validate_code(claim.get('procedure_codes', [])[0] if claim.get('procedure_codes') else ""),
            self.cpt_validator.validate_unbundling(claim.get('procedure_codes', [])),
            self.cpt_validator.validate_mutually_exclusive(claim.get('procedure_codes', []))
        )

        # Aggregate results
        all_errors = []
        all_warnings = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            severity='error' if all_errors else ('warning' if all_warnings else 'success')
        )
```

## 8. Validation Metrics and Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge

# Define validation metrics
validation_total = Counter(
    'validation_pipeline_total',
    'Total validation attempts',
    ['stage', 'status']
)

validation_duration = Histogram(
    'validation_pipeline_duration_seconds',
    'Validation pipeline duration',
    ['stage']
)

validation_failures = Counter(
    'validation_pipeline_failures_total',
    'Total validation failures',
    ['stage', 'failure_type']
)

data_quality_score = Gauge(
    'data_quality_score',
    'Current data quality score',
    ['quality_tier']
)


# Integration in pipeline
async def validate_with_metrics(claim_data: dict) -> PipelineResult:
    """
    Validation with metrics collection.
    """
    with validation_duration.labels(stage='total').time():
        result = await validation_pipeline.validate_claim(claim_data)

        # Record metrics
        validation_total.labels(
            stage=result.stage_reached,
            status=result.status
        ).inc()

        if result.status in ["FAILED", "ERROR"]:
            validation_failures.labels(
                stage=result.stage_reached,
                failure_type=result.status
            ).inc()

        if result.quality_score:
            data_quality_score.labels(
                quality_tier=result.quality_score.quality_tier
            ).set(result.quality_score.overall_score)

        return result
```

## 9. Performance Targets

```
Stage 1 (Schema): <20ms per claim
Stage 2 (Medical Codes): <50ms per claim
Stage 3 (Business Rules): <100ms per claim
Stage 4 (Quality): <30ms per claim
Stage 5 (Completeness): <20ms per claim

Total Pipeline: <220ms p95 per claim
Batch Processing: >5000 claims/minute
```

## 10. Related Documents

- [DATA_FLOW_ARCHITECTURE.md](./DATA_FLOW_ARCHITECTURE.md) - Overall data flow
- [MEDICAL-CODING-REFERENCE.md](./MEDICAL-CODING-REFERENCE.md) - Medical coding standards
- [DATA_GOVERNANCE.md](./DATA_GOVERNANCE.md) - Data governance and audit
