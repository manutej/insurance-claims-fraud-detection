"""
Data validation module for insurance claims.

Provides comprehensive validation including schema validation, business rules,
data quality checks, and error reporting.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict, Counter
import re
import json

from pydantic import ValidationError as PydanticValidationError
from jsonschema import validate, ValidationError as JsonSchemaError, Draft7Validator

from ..models.claim_models import (
    BaseClaim,
    MedicalClaim,
    PharmacyClaim,
    NoFaultClaim,
    ClaimType,
    FraudType,
    ValidationError,
    ProcessingResult,
    claim_factory,
)

logger = logging.getLogger(__name__)


class ClaimValidator:
    """Comprehensive validator for insurance claims."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Optional validation configuration
        """
        self.config = config or {}
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

        # Business rule thresholds
        self.max_daily_amount = Decimal(self.config.get("max_daily_amount", "50000.00"))
        self.max_procedure_codes = self.config.get("max_procedure_codes", 20)
        self.suspicious_weekend_types = {"professional", "institutional"}
        self.common_fraud_patterns = self._load_fraud_patterns()

    def _load_fraud_patterns(self) -> Dict[str, List[str]]:
        """Load known fraud patterns for validation."""
        return {
            "phantom_billing_indicators": [
                "Service on Sunday when office closed",
                "Sequential SSN pattern",
                "No corresponding appointment records",
                "Patient address doesn't exist",
            ],
            "upcoding_indicators": [
                "Procedure complexity mismatch",
                "Inconsistent diagnosis severity",
                "Billing pattern anomaly",
            ],
            "unbundling_indicators": [
                "Related procedures billed separately",
                "Unusual procedure combinations",
                "Multiple claims same day",
            ],
        }

    def validate_schema(self, claim_data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate claim data against JSON schema.

        Args:
            claim_data: Raw claim data dictionary

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        try:
            # Create appropriate claim model to trigger Pydantic validation
            claim = claim_factory(claim_data)
            return True, errors

        except PydanticValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                validation_error = ValidationError(
                    field_name=field_path,
                    error_message=error["msg"],
                    claim_id=claim_data.get("claim_id"),
                    severity="error",
                )
                errors.append(validation_error)

        except Exception as e:
            validation_error = ValidationError(
                field_name="general",
                error_message=f"Schema validation failed: {str(e)}",
                claim_id=claim_data.get("claim_id"),
                severity="error",
            )
            errors.append(validation_error)

        return False, errors

    def validate_business_rules(self, claim: BaseClaim) -> List[ValidationError]:
        """
        Validate business rules for a claim.

        Args:
            claim: Validated claim object

        Returns:
            List of validation errors and warnings
        """
        errors = []

        # Rule 1: Check for excessive billing amounts
        errors.extend(self._validate_billing_amounts(claim))

        # Rule 2: Check date consistency
        errors.extend(self._validate_dates(claim))

        # Rule 3: Check provider patterns
        errors.extend(self._validate_provider_patterns(claim))

        # Rule 4: Check for suspicious patterns
        errors.extend(self._validate_fraud_patterns(claim))

        # Type-specific validations
        if isinstance(claim, MedicalClaim):
            errors.extend(self._validate_medical_claim_rules(claim))
        elif isinstance(claim, PharmacyClaim):
            errors.extend(self._validate_pharmacy_claim_rules(claim))
        elif isinstance(claim, NoFaultClaim):
            errors.extend(self._validate_no_fault_claim_rules(claim))

        return errors

    def _validate_billing_amounts(self, claim: BaseClaim) -> List[ValidationError]:
        """Validate billing amount business rules."""
        errors = []

        if claim.billed_amount > self.max_daily_amount:
            errors.append(
                ValidationError(
                    field_name="billed_amount",
                    error_message=f"Amount ${claim.billed_amount} exceeds daily maximum ${self.max_daily_amount}",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check for suspicious round numbers
        if claim.billed_amount % 100 == 0 and claim.billed_amount >= 1000:
            errors.append(
                ValidationError(
                    field_name="billed_amount",
                    error_message="Suspicious round number billing amount",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        return errors

    def _validate_dates(self, claim: BaseClaim) -> List[ValidationError]:
        """Validate date-related business rules."""
        errors = []

        # Check if service date is in the future
        if claim.date_of_service > date.today():
            errors.append(
                ValidationError(
                    field_name="date_of_service",
                    error_message="Service date is in the future",
                    claim_id=claim.claim_id,
                    severity="error",
                )
            )

        # Check if service date is too old (over 2 years)
        cutoff_date = date.today() - timedelta(days=730)
        if claim.date_of_service < cutoff_date:
            errors.append(
                ValidationError(
                    field_name="date_of_service",
                    error_message="Service date is over 2 years old",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check weekend services for professional claims
        if (
            claim.claim_type in self.suspicious_weekend_types
            and claim.date_of_service.weekday() == 6
        ):  # Sunday
            errors.append(
                ValidationError(
                    field_name="date_of_service",
                    error_message="Professional service on Sunday",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        return errors

    def _validate_provider_patterns(self, claim: BaseClaim) -> List[ValidationError]:
        """Validate provider-related patterns."""
        errors = []

        # Check for invalid NPI patterns
        if claim.provider_npi.startswith("9999999"):
            errors.append(
                ValidationError(
                    field_name="provider_npi",
                    error_message="Suspicious NPI pattern (starts with 9999999)",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check for test/demo provider IDs
        if "FRAUD" in claim.provider_id or "TEST" in claim.provider_id:
            errors.append(
                ValidationError(
                    field_name="provider_id",
                    error_message="Test or fraud provider ID detected",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        return errors

    def _validate_fraud_patterns(self, claim: BaseClaim) -> List[ValidationError]:
        """Check for known fraud patterns."""
        errors = []

        # Check red flags against known patterns
        for red_flag in claim.red_flags:
            for pattern_type, patterns in self.common_fraud_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in red_flag.lower():
                        errors.append(
                            ValidationError(
                                field_name="red_flags",
                                error_message=f"Known fraud pattern detected: {pattern}",
                                claim_id=claim.claim_id,
                                severity="warning",
                            )
                        )

        return errors

    def _validate_medical_claim_rules(self, claim: MedicalClaim) -> List[ValidationError]:
        """Validate medical claim specific business rules."""
        errors = []

        # Check for excessive procedure codes
        if len(claim.procedure_codes) > self.max_procedure_codes:
            errors.append(
                ValidationError(
                    field_name="procedure_codes",
                    error_message=f"Too many procedure codes: {len(claim.procedure_codes)}",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check for common unbundling patterns
        if len(claim.procedure_codes) > 5:
            errors.append(
                ValidationError(
                    field_name="procedure_codes",
                    error_message="Potential unbundling - many procedure codes",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Validate diagnosis-procedure consistency
        if self._check_diagnosis_procedure_mismatch(claim):
            errors.append(
                ValidationError(
                    field_name="diagnosis_codes",
                    error_message="Diagnosis and procedure codes may be inconsistent",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        return errors

    def _validate_pharmacy_claim_rules(self, claim: PharmacyClaim) -> List[ValidationError]:
        """Validate pharmacy claim specific business rules."""
        errors = []

        # Check for excessive days supply
        if claim.days_supply > 90:
            errors.append(
                ValidationError(
                    field_name="days_supply",
                    error_message=f"Excessive days supply: {claim.days_supply}",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check for suspicious quantities
        if claim.quantity > 1000:
            errors.append(
                ValidationError(
                    field_name="quantity",
                    error_message=f"Suspicious quantity: {claim.quantity}",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        # Check fill date vs service date
        if hasattr(claim, "fill_date") and hasattr(claim, "date_of_service"):
            if claim.fill_date != claim.date_of_service:
                errors.append(
                    ValidationError(
                        field_name="fill_date",
                        error_message="Fill date differs from service date",
                        claim_id=claim.claim_id,
                        severity="warning",
                    )
                )

        return errors

    def _validate_no_fault_claim_rules(self, claim: NoFaultClaim) -> List[ValidationError]:
        """Validate no-fault claim specific business rules."""
        errors = []

        # Check accident date vs service date
        if claim.accident_date > claim.date_of_service:
            errors.append(
                ValidationError(
                    field_name="accident_date",
                    error_message="Accident date is after service date",
                    claim_id=claim.claim_id,
                    severity="error",
                )
            )

        # Check for excessive time between accident and service
        days_diff = (claim.date_of_service - claim.accident_date).days
        if days_diff > 365:
            errors.append(
                ValidationError(
                    field_name="date_of_service",
                    error_message=f"Service date is {days_diff} days after accident",
                    claim_id=claim.claim_id,
                    severity="warning",
                )
            )

        return errors

    def _check_diagnosis_procedure_mismatch(self, claim: MedicalClaim) -> bool:
        """Check for potential diagnosis-procedure mismatches."""
        # Simplified check - in real implementation, would use medical coding databases
        emergency_procedures = ["99281", "99282", "99283", "99284", "99285"]
        routine_diagnoses = ["Z00", "Z01"]  # Routine check-ups

        has_emergency_proc = any(code in emergency_procedures for code in claim.procedure_codes)
        has_routine_diag = any(
            diag.startswith(tuple(routine_diagnoses)) for diag in claim.diagnosis_codes
        )

        return has_emergency_proc and has_routine_diag

    def validate_data_quality(self, claims_data: List[Dict[str, Any]]) -> List[ValidationError]:
        """
        Perform data quality checks across a batch of claims.

        Args:
            claims_data: List of claim dictionaries

        Returns:
            List of data quality validation errors
        """
        errors = []

        # Check for duplicates
        claim_ids = [claim.get("claim_id") for claim in claims_data]
        duplicates = [item for item, count in Counter(claim_ids).items() if count > 1]

        for duplicate_id in duplicates:
            errors.append(
                ValidationError(
                    field_name="claim_id",
                    error_message=f"Duplicate claim ID found: {duplicate_id}",
                    claim_id=duplicate_id,
                    severity="error",
                )
            )

        # Check for suspicious patterns across claims
        errors.extend(self._check_batch_patterns(claims_data))

        return errors

    def _check_batch_patterns(self, claims_data: List[Dict[str, Any]]) -> List[ValidationError]:
        """Check for suspicious patterns across a batch of claims."""
        errors = []

        # Group by provider
        provider_claims = defaultdict(list)
        for claim in claims_data:
            provider_id = claim.get("provider_id")
            if provider_id:
                provider_claims[provider_id].append(claim)

        # Check for excessive claims per provider per day
        for provider_id, claims in provider_claims.items():
            daily_claims = defaultdict(int)
            for claim in claims:
                service_date = claim.get("date_of_service")
                if service_date:
                    daily_claims[service_date] += 1

            for date_str, count in daily_claims.items():
                if count > 50:  # Suspicious threshold
                    errors.append(
                        ValidationError(
                            field_name="provider_activity",
                            error_message=f"Provider {provider_id} has {count} claims on {date_str}",
                            severity="warning",
                        )
                    )

        return errors

    def validate_batch(self, claims_data: List[Dict[str, Any]]) -> ProcessingResult:
        """
        Validate a complete batch of claims.

        Args:
            claims_data: List of claim dictionaries

        Returns:
            ProcessingResult with validation summary
        """
        start_time = datetime.utcnow()
        all_errors = []
        all_warnings = []
        processed_count = 0
        error_count = 0

        # Data quality checks first
        quality_errors = self.validate_data_quality(claims_data)
        all_errors.extend([e for e in quality_errors if e.severity == "error"])
        all_warnings.extend([e for e in quality_errors if e.severity == "warning"])

        # Validate individual claims
        for claim_data in claims_data:
            try:
                # Schema validation
                is_valid, schema_errors = self.validate_schema(claim_data)
                all_errors.extend([e for e in schema_errors if e.severity == "error"])
                all_warnings.extend([e for e in schema_errors if e.severity == "warning"])

                if is_valid:
                    # Create claim object and validate business rules
                    claim = claim_factory(claim_data)
                    business_errors = self.validate_business_rules(claim)
                    all_errors.extend([e for e in business_errors if e.severity == "error"])
                    all_warnings.extend([e for e in business_errors if e.severity == "warning"])

                processed_count += 1

            except Exception as e:
                logger.error(
                    f"Validation failed for claim {claim_data.get('claim_id', 'unknown')}: {e}"
                )
                error_count += 1
                all_errors.append(
                    ValidationError(
                        field_name="general",
                        error_message=f"Validation exception: {str(e)}",
                        claim_id=claim_data.get("claim_id"),
                        severity="error",
                    )
                )

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        return ProcessingResult(
            success=error_count == 0,
            processed_count=processed_count,
            error_count=len(all_errors),
            warnings_count=len(all_warnings),
            errors=all_errors + all_warnings,
            processing_time_seconds=processing_time,
        )


class SchemaManager:
    """Manages JSON schemas for different claim types."""

    def __init__(self):
        """Initialize schema manager with predefined schemas."""
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for validation."""
        return {
            "medical_claim": {
                "type": "object",
                "required": [
                    "claim_id",
                    "patient_id",
                    "provider_id",
                    "provider_npi",
                    "date_of_service",
                    "diagnosis_codes",
                    "procedure_codes",
                    "billed_amount",
                    "claim_type",
                ],
                "properties": {
                    "claim_id": {"type": "string", "pattern": r"^CLM-\d{4}-[A-Z0-9]+$"},
                    "patient_id": {"type": "string", "pattern": r"^PAT-[A-Z0-9-]+$"},
                    "provider_id": {"type": "string", "pattern": r"^PRV-[A-Z0-9-]+$"},
                    "provider_npi": {"type": "string", "pattern": r"^\d{10}$"},
                    "date_of_service": {"type": "string", "format": "date"},
                    "diagnosis_codes": {"type": "array", "minItems": 1},
                    "procedure_codes": {"type": "array", "minItems": 1},
                    "billed_amount": {"type": "number", "minimum": 0},
                    "claim_type": {"type": "string", "enum": ["professional", "institutional"]},
                    "fraud_indicator": {"type": "boolean"},
                    "red_flags": {"type": "array", "items": {"type": "string"}},
                },
            }
        }

    def validate_against_schema(self, data: Dict[str, Any], schema_name: str) -> List[str]:
        """
        Validate data against specified schema.

        Args:
            data: Data to validate
            schema_name: Name of schema to use

        Returns:
            List of validation error messages
        """
        if schema_name not in self.schemas:
            return [f"Schema '{schema_name}' not found"]

        schema = self.schemas[schema_name]
        validator = Draft7Validator(schema)
        errors = []

        for error in validator.iter_errors(data):
            errors.append(f"Field '{'.'.join(map(str, error.path))}': {error.message}")

        return errors
