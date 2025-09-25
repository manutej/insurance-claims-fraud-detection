"""
Unit tests for the validator module.
"""
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.validator import ClaimValidator, SchemaManager
from src.models.claim_models import (
    BaseClaim, MedicalClaim, PharmacyClaim, NoFaultClaim,
    ValidationError, ProcessingResult
)
from tests.fixtures.claim_factories import (
    ValidClaim, UpcodingFraudClaim, PhantomBillingClaim
)
from tests.test_config import BENCHMARKS


class TestClaimValidator:
    """Test the ClaimValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a ClaimValidator instance for testing."""
        return ClaimValidator()

    @pytest.fixture
    def custom_config_validator(self):
        """Create a validator with custom configuration."""
        config = {
            'max_daily_amount': '25000.00',
            'max_procedure_codes': 15
        }
        return ClaimValidator(config)

    @pytest.fixture
    def valid_claim_data(self):
        """Create valid claim data for testing."""
        return {
            'claim_id': 'CLM-2024-000001',
            'patient_id': 'PAT-00000001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today().isoformat(),
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': 250.0,
            'claim_type': 'professional',
            'fraud_indicator': False,
            'red_flags': []
        }

    @pytest.fixture
    def medical_claim_data(self):
        """Create medical claim data for testing."""
        return {
            'claim_id': 'CLM-2024-000002',
            'patient_id': 'PAT-00000002',
            'provider_id': 'PRV-002',
            'provider_npi': '9876543210',
            'date_of_service': date.today().isoformat(),
            'diagnosis_codes': ['M79.3', 'S13.4'],
            'procedure_codes': ['99213', '73721'],
            'billed_amount': 450.0,
            'claim_type': 'professional',
            'service_location': '11',
            'rendering_provider_npi': '1234567890',
            'fraud_indicator': False,
            'red_flags': []
        }

    @pytest.fixture
    def pharmacy_claim_data(self):
        """Create pharmacy claim data for testing."""
        return {
            'claim_id': 'CLM-2024-000003',
            'patient_id': 'PAT-00000003',
            'provider_id': 'PRV-003',
            'provider_npi': '1111111111',
            'date_of_service': date.today().isoformat(),
            'ndc_code': '12345-6789-01',
            'quantity': 30,
            'days_supply': 30,
            'billed_amount': 75.0,
            'claim_type': 'pharmacy',
            'fraud_indicator': False,
            'red_flags': []
        }

    @pytest.fixture
    def no_fault_claim_data(self):
        """Create no-fault claim data for testing."""
        accident_date = date.today() - timedelta(days=10)
        return {
            'claim_id': 'CLM-2024-000004',
            'patient_id': 'PAT-00000004',
            'provider_id': 'PRV-004',
            'provider_npi': '2222222222',
            'date_of_service': date.today().isoformat(),
            'accident_date': accident_date.isoformat(),
            'policy_number': 'POL-123456',
            'billed_amount': 1500.0,
            'claim_type': 'no_fault',
            'fraud_indicator': False,
            'red_flags': []
        }

    def test_validator_initialization(self, validator):
        """Test ClaimValidator initialization."""
        assert isinstance(validator.config, dict)
        assert isinstance(validator.errors, list)
        assert isinstance(validator.warnings, list)
        assert validator.max_daily_amount == Decimal('50000.00')
        assert validator.max_procedure_codes == 20
        assert isinstance(validator.common_fraud_patterns, dict)

    def test_validator_initialization_with_config(self, custom_config_validator):
        """Test validator initialization with custom config."""
        assert custom_config_validator.max_daily_amount == Decimal('25000.00')
        assert custom_config_validator.max_procedure_codes == 15

    @pytest.mark.unit
    def test_validate_schema_success(self, validator, valid_claim_data):
        """Test successful schema validation."""
        is_valid, errors = validator.validate_schema(valid_claim_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_missing_required_field(self, validator, valid_claim_data):
        """Test schema validation with missing required field."""
        # Remove required field
        del valid_claim_data['claim_id']

        is_valid, errors = validator.validate_schema(valid_claim_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any('claim_id' in error.field_name for error in errors)

    def test_validate_schema_invalid_data_type(self, validator, valid_claim_data):
        """Test schema validation with invalid data type."""
        # Invalid data type for billed_amount
        valid_claim_data['billed_amount'] = 'not_a_number'

        is_valid, errors = validator.validate_schema(valid_claim_data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_schema_malformed_data(self, validator):
        """Test schema validation with completely malformed data."""
        malformed_data = {'invalid': 'structure'}

        is_valid, errors = validator.validate_schema(malformed_data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_business_rules_valid_claim(self, validator, valid_claim_data):
        """Test business rules validation with valid claim."""
        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator.validate_business_rules(claim)

        # Should have no errors for valid claim
        assert len([e for e in errors if e.severity == 'error']) == 0

    def test_validate_billing_amounts_excessive(self, validator, valid_claim_data):
        """Test billing amount validation with excessive amount."""
        valid_claim_data['billed_amount'] = 75000.0  # Exceeds default max

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_billing_amounts(claim)

        assert len(errors) > 0
        assert any('exceeds daily maximum' in error.error_message for error in errors)

    def test_validate_billing_amounts_round_number(self, validator, valid_claim_data):
        """Test billing amount validation with suspicious round number."""
        valid_claim_data['billed_amount'] = 5000.0  # Suspicious round number

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_billing_amounts(claim)

        assert len(errors) > 0
        assert any('round number' in error.error_message for error in errors)

    def test_validate_dates_future_service_date(self, validator, valid_claim_data):
        """Test date validation with future service date."""
        future_date = date.today() + timedelta(days=30)
        valid_claim_data['date_of_service'] = future_date.isoformat()

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_dates(claim)

        assert len(errors) > 0
        assert any('future' in error.error_message for error in errors)
        assert any(error.severity == 'error' for error in errors)

    def test_validate_dates_old_service_date(self, validator, valid_claim_data):
        """Test date validation with very old service date."""
        old_date = date.today() - timedelta(days=800)  # Over 2 years
        valid_claim_data['date_of_service'] = old_date.isoformat()

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_dates(claim)

        assert len(errors) > 0
        assert any('over 2 years old' in error.error_message for error in errors)

    def test_validate_dates_weekend_service(self, validator, valid_claim_data):
        """Test date validation with weekend service."""
        # Find next Sunday
        today = date.today()
        days_ahead = 6 - today.weekday()  # Sunday is 6
        if days_ahead <= 0:
            days_ahead += 7
        sunday = today + timedelta(days=days_ahead)

        valid_claim_data['date_of_service'] = sunday.isoformat()
        valid_claim_data['claim_type'] = 'professional'

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_dates(claim)

        assert len(errors) > 0
        assert any('Sunday' in error.error_message for error in errors)

    def test_validate_provider_patterns_suspicious_npi(self, validator, valid_claim_data):
        """Test provider pattern validation with suspicious NPI."""
        valid_claim_data['provider_npi'] = '9999999999'  # Suspicious pattern

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_provider_patterns(claim)

        assert len(errors) > 0
        assert any('Suspicious NPI pattern' in error.error_message for error in errors)

    def test_validate_provider_patterns_test_provider(self, validator, valid_claim_data):
        """Test provider pattern validation with test provider ID."""
        valid_claim_data['provider_id'] = 'FRAUD-PROVIDER-001'

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_provider_patterns(claim)

        assert len(errors) > 0
        assert any('Test or fraud provider' in error.error_message for error in errors)

    def test_validate_fraud_patterns(self, validator, valid_claim_data):
        """Test fraud pattern validation."""
        valid_claim_data['red_flags'] = [
            'Service on Sunday when office closed',
            'No corresponding appointment records'
        ]

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = validator._validate_fraud_patterns(claim)

        assert len(errors) > 0
        assert any('fraud pattern detected' in error.error_message for error in errors)

    def test_validate_medical_claim_rules_excessive_procedures(self, validator, medical_claim_data):
        """Test medical claim validation with excessive procedure codes."""
        # Add many procedure codes
        medical_claim_data['procedure_codes'] = [f'9921{i}' for i in range(25)]

        from src.models.claim_models import claim_factory
        claim = claim_factory(medical_claim_data)

        errors = validator._validate_medical_claim_rules(claim)

        assert len(errors) > 0
        assert any('Too many procedure codes' in error.error_message for error in errors)

    def test_validate_medical_claim_rules_unbundling(self, validator, medical_claim_data):
        """Test medical claim validation for potential unbundling."""
        # Add multiple procedure codes
        medical_claim_data['procedure_codes'] = ['99213', '99214', '99215', '73721', '97110', '97112']

        from src.models.claim_models import claim_factory
        claim = claim_factory(medical_claim_data)

        errors = validator._validate_medical_claim_rules(claim)

        assert len(errors) > 0
        assert any('unbundling' in error.error_message for error in errors)

    def test_validate_pharmacy_claim_rules_excessive_days_supply(self, validator, pharmacy_claim_data):
        """Test pharmacy claim validation with excessive days supply."""
        pharmacy_claim_data['days_supply'] = 120  # Excessive

        from src.models.claim_models import claim_factory
        claim = claim_factory(pharmacy_claim_data)

        errors = validator._validate_pharmacy_claim_rules(claim)

        assert len(errors) > 0
        assert any('Excessive days supply' in error.error_message for error in errors)

    def test_validate_pharmacy_claim_rules_suspicious_quantity(self, validator, pharmacy_claim_data):
        """Test pharmacy claim validation with suspicious quantity."""
        pharmacy_claim_data['quantity'] = 1500  # Suspicious

        from src.models.claim_models import claim_factory
        claim = claim_factory(pharmacy_claim_data)

        errors = validator._validate_pharmacy_claim_rules(claim)

        assert len(errors) > 0
        assert any('Suspicious quantity' in error.error_message for error in errors)

    def test_validate_no_fault_claim_rules_invalid_accident_date(self, validator, no_fault_claim_data):
        """Test no-fault claim validation with invalid accident date."""
        # Accident date after service date
        accident_date = date.today() + timedelta(days=5)
        no_fault_claim_data['accident_date'] = accident_date.isoformat()

        from src.models.claim_models import claim_factory
        claim = claim_factory(no_fault_claim_data)

        errors = validator._validate_no_fault_claim_rules(claim)

        assert len(errors) > 0
        assert any('after service date' in error.error_message for error in errors)

    def test_validate_no_fault_claim_rules_old_accident(self, validator, no_fault_claim_data):
        """Test no-fault claim validation with very old accident."""
        # Accident over a year ago
        accident_date = date.today() - timedelta(days=400)
        no_fault_claim_data['accident_date'] = accident_date.isoformat()

        from src.models.claim_models import claim_factory
        claim = claim_factory(no_fault_claim_data)

        errors = validator._validate_no_fault_claim_rules(claim)

        assert len(errors) > 0
        assert any('days after accident' in error.error_message for error in errors)

    def test_check_diagnosis_procedure_mismatch(self, validator, medical_claim_data):
        """Test diagnosis-procedure mismatch detection."""
        # Emergency procedure with routine diagnosis
        medical_claim_data['procedure_codes'] = ['99285']  # Emergency
        medical_claim_data['diagnosis_codes'] = ['Z00.00']  # Routine checkup

        from src.models.claim_models import claim_factory
        claim = claim_factory(medical_claim_data)

        has_mismatch = validator._check_diagnosis_procedure_mismatch(claim)

        assert has_mismatch is True

    def test_validate_data_quality_duplicates(self, validator, valid_claim_data):
        """Test data quality validation for duplicate claims."""
        claims_data = [
            valid_claim_data.copy(),
            valid_claim_data.copy()  # Duplicate claim
        ]

        errors = validator.validate_data_quality(claims_data)

        assert len(errors) > 0
        assert any('Duplicate claim ID' in error.error_message for error in errors)

    def test_check_batch_patterns_excessive_provider_claims(self, validator):
        """Test batch pattern validation for excessive provider claims."""
        # Create many claims for same provider on same date
        claims_data = []
        for i in range(60):  # Exceeds threshold of 50
            claim_data = {
                'claim_id': f'CLM-2024-{i:06d}',
                'provider_id': 'PRV-001',
                'date_of_service': date.today().isoformat(),
                'billed_amount': 100.0
            }
            claims_data.append(claim_data)

        errors = validator._check_batch_patterns(claims_data)

        assert len(errors) > 0
        assert any('60 claims on' in error.error_message for error in errors)

    @pytest.mark.unit
    def test_validate_batch_success(self, validator, valid_claim_data):
        """Test successful batch validation."""
        claims_data = [valid_claim_data]

        result = validator.validate_batch(claims_data)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.processed_count == 1
        assert result.error_count >= 0  # May have warnings
        assert result.processing_time_seconds > 0

    def test_validate_batch_with_errors(self, validator):
        """Test batch validation with errors."""
        # Invalid claim data
        invalid_claim = {
            'claim_id': 'INVALID',
            'billed_amount': 'not_a_number'
        }

        claims_data = [invalid_claim]

        result = validator.validate_batch(claims_data)

        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.error_count > 0
        assert len(result.errors) > 0

    def test_validate_batch_mixed_valid_invalid(self, validator, valid_claim_data):
        """Test batch validation with mix of valid and invalid claims."""
        invalid_claim = {
            'claim_id': 'INVALID',
            'billed_amount': 'not_a_number'
        }

        claims_data = [valid_claim_data, invalid_claim]

        result = validator.validate_batch(claims_data)

        assert isinstance(result, ProcessingResult)
        assert result.processed_count >= 1  # At least the valid claim
        assert result.error_count > 0  # From invalid claim

    def test_validate_batch_exception_handling(self, validator):
        """Test batch validation exception handling."""
        # Data that will cause exceptions
        problematic_data = [
            None,  # Will cause TypeError
            {'claim_id': 'TEST'}  # Will cause validation errors
        ]

        result = validator.validate_batch(problematic_data)

        assert isinstance(result, ProcessingResult)
        assert result.error_count > 0

    @pytest.mark.performance
    def test_validation_performance(self, validator):
        """Test validation performance with larger dataset."""
        import time

        # Generate many valid claims
        claims_data = []
        for i in range(1000):
            claim_data = {
                'claim_id': f'CLM-2024-{i:06d}',
                'patient_id': f'PAT-{i:08d}',
                'provider_id': f'PRV-{i % 10:03d}',
                'provider_npi': '1234567890',
                'date_of_service': date.today().isoformat(),
                'diagnosis_codes': ['M79.3'],
                'procedure_codes': ['99213'],
                'billed_amount': 250.0,
                'claim_type': 'professional',
                'fraud_indicator': False,
                'red_flags': []
            }
            claims_data.append(claim_data)

        start_time = time.time()
        result = validator.validate_batch(claims_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 30  # 30 seconds for 1000 claims
        assert result.processed_count == 1000

        # Calculate throughput
        claims_per_second = len(claims_data) / processing_time
        assert claims_per_second > 30  # At least 30 claims/second

    def test_fraud_patterns_loading(self, validator):
        """Test fraud patterns loading."""
        patterns = validator.common_fraud_patterns

        assert 'phantom_billing_indicators' in patterns
        assert 'upcoding_indicators' in patterns
        assert 'unbundling_indicators' in patterns

        # Check that patterns are populated
        for pattern_type, pattern_list in patterns.items():
            assert isinstance(pattern_list, list)
            assert len(pattern_list) > 0

    def test_custom_config_validation(self, custom_config_validator, valid_claim_data):
        """Test validation with custom configuration."""
        # Use amount that exceeds custom max but not default max
        valid_claim_data['billed_amount'] = 30000.0

        from src.models.claim_models import claim_factory
        claim = claim_factory(valid_claim_data)

        errors = custom_config_validator._validate_billing_amounts(claim)

        assert len(errors) > 0
        assert any('25000' in error.error_message for error in errors)

    def test_edge_case_empty_batch(self, validator):
        """Test validation with empty batch."""
        result = validator.validate_batch([])

        assert isinstance(result, ProcessingResult)
        assert result.processed_count == 0
        assert result.error_count == 0

    def test_edge_case_none_values(self, validator):
        """Test validation with None values in claim data."""
        claim_with_nones = {
            'claim_id': 'CLM-TEST-001',
            'patient_id': None,
            'provider_id': None,
            'billed_amount': None
        }

        is_valid, errors = validator.validate_schema(claim_with_nones)

        assert is_valid is False
        assert len(errors) > 0


class TestSchemaManager:
    """Test the SchemaManager class."""

    @pytest.fixture
    def schema_manager(self):
        """Create a SchemaManager instance for testing."""
        return SchemaManager()

    def test_schema_manager_initialization(self, schema_manager):
        """Test SchemaManager initialization."""
        assert isinstance(schema_manager.schemas, dict)
        assert len(schema_manager.schemas) > 0
        assert 'medical_claim' in schema_manager.schemas

    def test_load_schemas(self, schema_manager):
        """Test schema loading."""
        schemas = schema_manager.schemas

        # Check medical claim schema
        medical_schema = schemas['medical_claim']
        assert 'type' in medical_schema
        assert 'required' in medical_schema
        assert 'properties' in medical_schema

        # Check required fields
        required_fields = medical_schema['required']
        assert 'claim_id' in required_fields
        assert 'patient_id' in required_fields
        assert 'provider_npi' in required_fields

    def test_validate_against_schema_success(self, schema_manager):
        """Test successful schema validation."""
        valid_data = {
            'claim_id': 'CLM-2024-TEST001',
            'patient_id': 'PAT-TEST001',
            'provider_id': 'PRV-TEST001',
            'provider_npi': '1234567890',
            'date_of_service': '2024-01-15',
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': 250.0,
            'claim_type': 'professional',
            'fraud_indicator': False,
            'red_flags': []
        }

        errors = schema_manager.validate_against_schema(valid_data, 'medical_claim')

        assert len(errors) == 0

    def test_validate_against_schema_missing_field(self, schema_manager):
        """Test schema validation with missing required field."""
        invalid_data = {
            'claim_id': 'CLM-2024-TEST001',
            # Missing required fields
            'billed_amount': 250.0
        }

        errors = schema_manager.validate_against_schema(invalid_data, 'medical_claim')

        assert len(errors) > 0
        assert any('required' in error.lower() for error in errors)

    def test_validate_against_schema_invalid_pattern(self, schema_manager):
        """Test schema validation with invalid pattern."""
        invalid_data = {
            'claim_id': 'INVALID-FORMAT',  # Doesn't match pattern
            'patient_id': 'PAT-TEST001',
            'provider_id': 'PRV-TEST001',
            'provider_npi': '1234567890',
            'date_of_service': '2024-01-15',
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': 250.0,
            'claim_type': 'professional',
            'fraud_indicator': False,
            'red_flags': []
        }

        errors = schema_manager.validate_against_schema(invalid_data, 'medical_claim')

        assert len(errors) > 0

    def test_validate_against_schema_nonexistent_schema(self, schema_manager):
        """Test validation against nonexistent schema."""
        data = {'test': 'data'}

        errors = schema_manager.validate_against_schema(data, 'nonexistent_schema')

        assert len(errors) == 1
        assert 'not found' in errors[0]

    def test_validate_against_schema_invalid_enum(self, schema_manager):
        """Test schema validation with invalid enum value."""
        invalid_data = {
            'claim_id': 'CLM-2024-TEST001',
            'patient_id': 'PAT-TEST001',
            'provider_id': 'PRV-TEST001',
            'provider_npi': '1234567890',
            'date_of_service': '2024-01-15',
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': 250.0,
            'claim_type': 'invalid_type',  # Invalid enum value
            'fraud_indicator': False,
            'red_flags': []
        }

        errors = schema_manager.validate_against_schema(invalid_data, 'medical_claim')

        assert len(errors) > 0

    def test_validate_against_schema_invalid_npi_pattern(self, schema_manager):
        """Test schema validation with invalid NPI pattern."""
        invalid_data = {
            'claim_id': 'CLM-2024-TEST001',
            'patient_id': 'PAT-TEST001',
            'provider_id': 'PRV-TEST001',
            'provider_npi': '123',  # Too short for NPI
            'date_of_service': '2024-01-15',
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': 250.0,
            'claim_type': 'professional',
            'fraud_indicator': False,
            'red_flags': []
        }

        errors = schema_manager.validate_against_schema(invalid_data, 'medical_claim')

        assert len(errors) > 0