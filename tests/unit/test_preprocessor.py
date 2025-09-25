"""
Unit tests for the preprocessor module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.preprocessor import ClaimPreprocessor
from src.models.claim_models import BaseClaim, MedicalClaim, PharmacyClaim, NoFaultClaim
from tests.fixtures.claim_factories import (
    ValidClaim, UpcodingFraudClaim, PhantomBillingClaim
)
from tests.test_config import CONFIG


class TestClaimPreprocessor:
    """Test the ClaimPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a ClaimPreprocessor instance for testing."""
        return ClaimPreprocessor()

    @pytest.fixture
    def custom_config_preprocessor(self):
        """Create a preprocessor with custom configuration."""
        config = {
            'normalize_amounts': False,
            'extract_temporal_features': False,
            'handle_missing_data': False,
            'encoding_strategy': 'label'
        }
        return ClaimPreprocessor(config)

    @pytest.fixture
    def sample_medical_claims(self):
        """Create sample medical claims for testing."""
        claims = []

        # Create valid medical claim
        claim_data = {
            'claim_id': 'CLM-2024-000001',
            'patient_id': 'PAT-00000001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today(),
            'diagnosis_codes': ['M79.3', 'S13.4'],
            'procedure_codes': ['99213', '73721'],
            'billed_amount': Decimal('250.00'),
            'claim_type': 'professional',
            'service_location': '11',
            'rendering_hours': Decimal('1.5'),
            'day_of_week': 'Monday',
            'fraud_indicator': False,
            'fraud_type': None,
            'red_flags': [],
            'notes': 'Patient complained of back pain'
        }
        claims.append(MedicalClaim(**claim_data))

        # Create fraud medical claim
        fraud_claim_data = {
            'claim_id': 'CLM-2024-000002',
            'patient_id': 'PAT-00000002',
            'provider_id': 'PRV-002',
            'provider_npi': '9876543210',
            'date_of_service': date.today(),
            'diagnosis_codes': ['E11.9'],
            'procedure_codes': ['99215', '99285'],
            'billed_amount': Decimal('5000.00'),
            'claim_type': 'professional',
            'service_location': '23',
            'rendering_hours': Decimal('3.0'),
            'day_of_week': 'Sunday',
            'fraud_indicator': True,
            'fraud_type': 'upcoding',
            'red_flags': ['excessive_billing', 'complexity_mismatch'],
            'notes': 'Emergency visit'
        }
        claims.append(MedicalClaim(**fraud_claim_data))

        return claims

    @pytest.fixture
    def sample_pharmacy_claims(self):
        """Create sample pharmacy claims for testing."""
        claims = []

        claim_data = {
            'claim_id': 'CLM-2024-000003',
            'patient_id': 'PAT-00000003',
            'provider_id': 'PRV-003',
            'provider_npi': '1111111111',
            'date_of_service': date.today(),
            'ndc_code': '12345-6789-01',
            'drug_name': 'Metformin',
            'quantity': 30,
            'days_supply': 30,
            'billed_amount': Decimal('75.00'),
            'claim_type': 'pharmacy',
            'prescriber_npi': '1234567890',
            'pharmacy_npi': '1111111111',
            'fill_date': date.today(),
            'fraud_indicator': False,
            'fraud_type': None,
            'red_flags': []
        }
        claims.append(PharmacyClaim(**claim_data))

        return claims

    @pytest.fixture
    def sample_no_fault_claims(self):
        """Create sample no-fault claims for testing."""
        claims = []

        accident_date = date.today() - timedelta(days=10)
        claim_data = {
            'claim_id': 'CLM-2024-000004',
            'patient_id': 'PAT-00000004',
            'provider_id': 'PRV-004',
            'provider_npi': '2222222222',
            'date_of_service': date.today(),
            'accident_date': accident_date,
            'vehicle_year': 2020,
            'vehicle_make': 'Toyota',
            'policy_number': 'POL-123456',
            'attorney_involved': False,
            'estimated_damage': Decimal('15000.00'),
            'billed_amount': Decimal('1500.00'),
            'claim_type': 'no_fault',
            'fraud_indicator': False,
            'fraud_type': None,
            'red_flags': []
        }
        claims.append(NoFaultClaim(**claim_data))

        return claims

    def test_preprocessor_initialization(self, preprocessor):
        """Test ClaimPreprocessor initialization."""
        assert isinstance(preprocessor.config, dict)
        assert isinstance(preprocessor.scalers, dict)
        assert isinstance(preprocessor.encoders, dict)
        assert isinstance(preprocessor.imputers, dict)
        assert isinstance(preprocessor.feature_columns, list)
        assert preprocessor.is_fitted is False

        # Check default configuration
        assert preprocessor.normalize_amounts is True
        assert preprocessor.extract_temporal_features is True
        assert preprocessor.handle_missing_data is True
        assert preprocessor.encoding_strategy == 'onehot'

    def test_preprocessor_initialization_with_config(self, custom_config_preprocessor):
        """Test preprocessor initialization with custom config."""
        assert custom_config_preprocessor.normalize_amounts is False
        assert custom_config_preprocessor.extract_temporal_features is False
        assert custom_config_preprocessor.handle_missing_data is False
        assert custom_config_preprocessor.encoding_strategy == 'label'

    @pytest.mark.unit
    def test_preprocess_claims_basic(self, preprocessor, sample_medical_claims):
        """Test basic claim preprocessing functionality."""
        df = preprocessor.preprocess_claims(sample_medical_claims)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_medical_claims)
        assert 'claim_id' in df.columns
        assert 'fraud_indicator' in df.columns
        assert preprocessor.is_fitted is True
        assert len(preprocessor.feature_columns) > 0

    def test_claims_to_dataframe_medical(self, preprocessor, sample_medical_claims):
        """Test converting medical claims to DataFrame."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_medical_claims)

        # Check basic columns
        expected_cols = [
            'claim_id', 'patient_id', 'provider_id', 'provider_npi',
            'date_of_service', 'billed_amount', 'claim_type',
            'fraud_indicator', 'fraud_type', 'red_flags_count'
        ]
        for col in expected_cols:
            assert col in df.columns

        # Check medical-specific columns
        medical_specific_cols = [
            'diagnosis_count', 'procedure_count', 'primary_diagnosis',
            'primary_procedure', 'service_location', 'rendering_hours'
        ]
        for col in medical_specific_cols:
            assert col in df.columns

        # Check data types and values
        assert df['billed_amount'].dtype in [np.float64, float]
        assert df['diagnosis_count'].iloc[0] == 2  # From test data
        assert df['procedure_count'].iloc[0] == 2

    def test_claims_to_dataframe_pharmacy(self, preprocessor, sample_pharmacy_claims):
        """Test converting pharmacy claims to DataFrame."""
        df = preprocessor._claims_to_dataframe(sample_pharmacy_claims)

        # Check pharmacy-specific columns
        pharmacy_specific_cols = [
            'ndc_code', 'drug_name', 'quantity', 'days_supply',
            'prescriber_npi', 'pharmacy_npi', 'fill_date'
        ]
        for col in pharmacy_specific_cols:
            assert col in df.columns

        assert df['quantity'].iloc[0] == 30
        assert df['days_supply'].iloc[0] == 30

    def test_claims_to_dataframe_no_fault(self, preprocessor, sample_no_fault_claims):
        """Test converting no-fault claims to DataFrame."""
        df = preprocessor._claims_to_dataframe(sample_no_fault_claims)

        # Check no-fault specific columns
        no_fault_specific_cols = [
            'accident_date', 'vehicle_year', 'vehicle_make',
            'attorney_involved', 'estimated_damage'
        ]
        for col in no_fault_specific_cols:
            assert col in df.columns

        assert df['vehicle_year'].iloc[0] == 2020
        assert df['attorney_involved'].iloc[0] is False

    def test_extract_basic_features(self, preprocessor, sample_medical_claims):
        """Test basic feature extraction."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_basic_features(df)

        # Check that basic features were created
        expected_features = [
            'amount_log', 'amount_rounded', 'amount_bin',
            'provider_prefix', 'npi_last_digit', 'patient_prefix'
        ]
        for feature in expected_features:
            assert feature in df.columns

        # Check feature values
        assert df['amount_log'].iloc[0] > 0  # log1p of positive amount
        assert df['amount_rounded'].iloc[0] in [0, 1]  # Binary flag
        assert df['npi_last_digit'].iloc[0] in range(10)  # Should be 0-9

    def test_extract_temporal_features(self, preprocessor, sample_medical_claims):
        """Test temporal feature extraction."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_temporal_features(df)

        # Check temporal features for date_of_service
        temporal_features = [
            'date_of_service_year', 'date_of_service_month',
            'date_of_service_day', 'date_of_service_weekday',
            'date_of_service_is_weekend', 'date_of_service_quarter',
            'date_of_service_days_since_epoch'
        ]
        for feature in temporal_features:
            assert feature in df.columns

        # Check feature values
        assert df['date_of_service_year'].iloc[0] == date.today().year
        assert df['date_of_service_month'].iloc[0] == date.today().month
        assert df['date_of_service_is_weekend'].iloc[0] in [0, 1]

    def test_extract_temporal_features_disabled(self, custom_config_preprocessor, sample_medical_claims):
        """Test temporal feature extraction when disabled."""
        df = custom_config_preprocessor._claims_to_dataframe(sample_medical_claims)
        df_before = df.copy()
        df = custom_config_preprocessor._extract_temporal_features(df)

        # Should not add temporal features when disabled
        assert len(df.columns) == len(df_before.columns)

    def test_extract_provider_features(self, preprocessor, sample_medical_claims):
        """Test provider feature extraction."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_provider_features(df)

        # Check provider features
        provider_features = [
            'provider_claim_count', 'provider_avg_amount',
            'provider_amount_std', 'provider_total_amount',
            'provider_fraud_rate', 'provider_risk_score',
            'provider_suspicious_npi', 'provider_test_id'
        ]
        for feature in provider_features:
            assert feature in df.columns

        # Check feature values
        assert df['provider_claim_count'].iloc[0] >= 1
        assert df['provider_suspicious_npi'].iloc[0] in [0, 1]
        assert df['provider_test_id'].iloc[0] in [0, 1]
        assert 0 <= df['provider_risk_score'].iloc[0] <= 1

    def test_extract_amount_features(self, preprocessor, sample_medical_claims):
        """Test amount-based feature extraction."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_amount_features(df)

        # Check amount features
        amount_features = [
            'amount_below_25th', 'amount_above_90th', 'amount_above_95th',
            'amount_zscore', 'amount_outlier'
        ]
        for feature in amount_features:
            assert feature in df.columns

        # Check feature values
        assert df['amount_below_25th'].iloc[0] in [0, 1]
        assert df['amount_above_90th'].iloc[0] in [0, 1]
        assert df['amount_outlier'].iloc[0] in [0, 1]

    def test_extract_medical_features(self, preprocessor, sample_medical_claims):
        """Test medical-specific feature extraction."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_medical_features(df)

        # Check medical features
        medical_features = [
            'multiple_diagnoses', 'many_procedures',
            'procedure_to_diagnosis_ratio', 'emergency_service', 'office_service'
        ]
        for feature in medical_features:
            assert feature in df.columns

        # Check feature values
        assert df['multiple_diagnoses'].iloc[0] in [0, 1]
        assert df['many_procedures'].iloc[0] in [0, 1]
        assert df['procedure_to_diagnosis_ratio'].iloc[0] >= 0

    def test_extract_medical_features_pharmacy(self, preprocessor, sample_pharmacy_claims):
        """Test medical feature extraction for pharmacy claims."""
        df = preprocessor._claims_to_dataframe(sample_pharmacy_claims)
        df = preprocessor._extract_medical_features(df)

        # Check pharmacy-specific features
        pharmacy_features = ['long_supply', 'excessive_supply', 'high_quantity']
        for feature in pharmacy_features:
            assert feature in df.columns

        assert df['long_supply'].iloc[0] in [0, 1]
        assert df['excessive_supply'].iloc[0] in [0, 1]
        assert df['high_quantity'].iloc[0] in [0, 1]

    def test_handle_missing_data(self, preprocessor, sample_medical_claims):
        """Test missing data handling."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)

        # Introduce missing values
        df.loc[0, 'billed_amount'] = np.nan
        df.loc[1, 'provider_id'] = np.nan

        df = preprocessor._handle_missing_data(df)

        # Check that missing values were imputed
        assert not df['billed_amount'].isnull().any()
        assert not df['provider_id'].isnull().any()

    def test_onehot_encode(self, preprocessor, sample_medical_claims):
        """Test one-hot encoding."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        categorical_cols = ['claim_type', 'service_location']

        df = preprocessor._onehot_encode(df, categorical_cols)

        # Check that original categorical columns are removed
        assert 'claim_type' not in df.columns
        assert 'service_location' not in df.columns

        # Check that encoded columns are created
        encoded_cols = [col for col in df.columns if 'claim_type_' in col or 'service_location_' in col]
        assert len(encoded_cols) > 0

    def test_label_encode(self, preprocessor, sample_medical_claims):
        """Test label encoding."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        categorical_cols = ['claim_type', 'service_location']

        df = preprocessor._label_encode(df, categorical_cols)

        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(df['claim_type'])
        assert pd.api.types.is_numeric_dtype(df['service_location'])

    def test_normalize_features(self, preprocessor, sample_medical_claims):
        """Test feature normalization."""
        df = preprocessor._claims_to_dataframe(sample_medical_claims)
        df = preprocessor._extract_basic_features(df)

        # Store original values
        original_amount = df['billed_amount'].copy()

        df = preprocessor._normalize_features(df)

        # Check that features were normalized (mean ≈ 0, std ≈ 1)
        normalized_amount = df['billed_amount']
        assert abs(normalized_amount.mean()) < 1e-10  # Should be close to 0
        assert abs(normalized_amount.std() - 1) < 1e-10  # Should be close to 1

    def test_preprocess_claims_complete_pipeline(self, preprocessor, sample_medical_claims):
        """Test complete preprocessing pipeline."""
        df = preprocessor.preprocess_claims(sample_medical_claims)

        # Check that all processing steps were applied
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_medical_claims)
        assert preprocessor.is_fitted is True

        # Check that features were created
        assert len(preprocessor.feature_columns) > 0

        # Check for expected feature types
        basic_features = [col for col in df.columns if 'amount_' in col]
        temporal_features = [col for col in df.columns if '_year' in col or '_month' in col]
        provider_features = [col for col in df.columns if 'provider_' in col]

        assert len(basic_features) > 0
        assert len(temporal_features) > 0
        assert len(provider_features) > 0

    def test_transform_new_data_success(self, preprocessor, sample_medical_claims):
        """Test transforming new data with fitted preprocessor."""
        # First fit the preprocessor
        preprocessor.preprocess_claims(sample_medical_claims)

        # Create new claims
        new_claims = sample_medical_claims[:1]  # Use first claim as new data

        # Transform new data
        df = preprocessor.transform_new_data(new_claims)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(new_claims)

        # Check that columns match training data
        expected_cols = ['claim_id'] + preprocessor.feature_columns
        if 'fraud_indicator' in df.columns:
            expected_cols.append('fraud_indicator')

        assert list(df.columns) == expected_cols

    def test_transform_new_data_not_fitted(self, preprocessor, sample_medical_claims):
        """Test transforming new data without fitting first."""
        with pytest.raises(ValueError) as excinfo:
            preprocessor.transform_new_data(sample_medical_claims)

        assert "must be fitted" in str(excinfo.value)

    def test_get_feature_importance_data(self, preprocessor, sample_medical_claims):
        """Test feature importance analysis."""
        df = preprocessor.preprocess_claims(sample_medical_claims)
        feature_info = preprocessor.get_feature_importance_data(df)

        assert isinstance(feature_info, dict)
        assert 'correlations' in feature_info
        assert 'statistics' in feature_info
        assert 'feature_count' in feature_info
        assert 'feature_names' in feature_info

        # Check correlations
        correlations = feature_info['correlations']
        assert isinstance(correlations, dict)
        for feature, corr in correlations.items():
            assert 0 <= corr <= 1  # Absolute correlation

        # Check statistics
        statistics = feature_info['statistics']
        assert isinstance(statistics, dict)
        for feature, stats in statistics.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'missing_pct' in stats

    def test_preprocessing_with_mixed_claim_types(self, preprocessor):
        """Test preprocessing with mixed claim types."""
        # Create mixed claims
        medical_claim_data = {
            'claim_id': 'CLM-MED-001',
            'patient_id': 'PAT-001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today(),
            'diagnosis_codes': ['M79.3'],
            'procedure_codes': ['99213'],
            'billed_amount': Decimal('250.00'),
            'claim_type': 'professional',
            'service_location': '11',
            'fraud_indicator': False
        }
        medical_claim = MedicalClaim(**medical_claim_data)

        pharmacy_claim_data = {
            'claim_id': 'CLM-PHARM-001',
            'patient_id': 'PAT-002',
            'provider_id': 'PRV-002',
            'provider_npi': '1111111111',
            'date_of_service': date.today(),
            'ndc_code': '12345-6789-01',
            'quantity': 30,
            'days_supply': 30,
            'billed_amount': Decimal('75.00'),
            'claim_type': 'pharmacy',
            'fraud_indicator': False
        }
        pharmacy_claim = PharmacyClaim(**pharmacy_claim_data)

        mixed_claims = [medical_claim, pharmacy_claim]

        df = preprocessor.preprocess_claims(mixed_claims)

        assert len(df) == 2
        assert preprocessor.is_fitted is True

        # Both claim types should be processed successfully
        assert df['claim_id'].tolist() == ['CLM-MED-001', 'CLM-PHARM-001']

    @pytest.mark.performance
    def test_preprocessing_performance(self, preprocessor):
        """Test preprocessing performance with larger dataset."""
        import time

        # Generate many claims
        claims = []
        for i in range(500):
            claim_data = {
                'claim_id': f'CLM-{i:06d}',
                'patient_id': f'PAT-{i:06d}',
                'provider_id': f'PRV-{i % 10:03d}',
                'provider_npi': '1234567890',
                'date_of_service': date.today(),
                'diagnosis_codes': ['M79.3'],
                'procedure_codes': ['99213'],
                'billed_amount': Decimal(str(100 + i)),
                'claim_type': 'professional',
                'service_location': '11',
                'fraud_indicator': i % 10 == 0  # 10% fraud rate
            }
            claims.append(MedicalClaim(**claim_data))

        start_time = time.time()
        df = preprocessor.preprocess_claims(claims)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 30  # 30 seconds for 500 claims
        assert len(df) == 500

        # Calculate throughput
        claims_per_second = len(claims) / processing_time
        assert claims_per_second > 15  # At least 15 claims/second

    def test_edge_case_empty_claims_list(self, preprocessor):
        """Test preprocessing with empty claims list."""
        empty_claims = []

        df = preprocessor.preprocess_claims(empty_claims)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert preprocessor.is_fitted is True

    def test_edge_case_single_claim(self, preprocessor, sample_medical_claims):
        """Test preprocessing with single claim."""
        single_claim = sample_medical_claims[:1]

        df = preprocessor.preprocess_claims(single_claim)

        assert len(df) == 1
        assert len(preprocessor.feature_columns) > 0

    def test_edge_case_missing_optional_fields(self, preprocessor):
        """Test preprocessing with minimal claim data."""
        minimal_claim_data = {
            'claim_id': 'CLM-MINIMAL-001',
            'patient_id': 'PAT-001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today(),
            'billed_amount': Decimal('100.00'),
            'claim_type': 'professional',
            'fraud_indicator': False
            # Missing many optional fields
        }
        minimal_claim = BaseClaim(**minimal_claim_data)

        df = preprocessor.preprocess_claims([minimal_claim])

        assert len(df) == 1
        assert preprocessor.is_fitted is True

    def test_custom_encoding_strategy(self):
        """Test preprocessor with different encoding strategies."""
        # Test label encoding
        label_config = {'encoding_strategy': 'label'}
        label_preprocessor = ClaimPreprocessor(label_config)

        claim_data = {
            'claim_id': 'CLM-001',
            'patient_id': 'PAT-001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today(),
            'billed_amount': Decimal('250.00'),
            'claim_type': 'professional',
            'fraud_indicator': False
        }
        claim = BaseClaim(**claim_data)

        df = label_preprocessor.preprocess_claims([claim])

        # With label encoding, categorical columns should remain but be numeric
        categorical_features = [col for col in df.columns if col.endswith('_professional') or col.endswith('_institutional')]
        # Should have fewer columns compared to one-hot encoding
        assert len(categorical_features) == 0  # No one-hot encoded columns

    def test_normalize_amounts_disabled(self):
        """Test preprocessor with normalization disabled."""
        config = {'normalize_amounts': False}
        preprocessor = ClaimPreprocessor(config)

        claim_data = {
            'claim_id': 'CLM-001',
            'patient_id': 'PAT-001',
            'provider_id': 'PRV-001',
            'provider_npi': '1234567890',
            'date_of_service': date.today(),
            'billed_amount': Decimal('1000.00'),
            'claim_type': 'professional',
            'fraud_indicator': False
        }
        claim = BaseClaim(**claim_data)

        df = preprocessor.preprocess_claims([claim])

        # Original amounts should be preserved (not normalized)
        assert df['billed_amount'].iloc[0] == 1000.0

    def test_fitted_state_persistence(self, preprocessor, sample_medical_claims):
        """Test that fitted state persists properly."""
        # Initially not fitted
        assert preprocessor.is_fitted is False

        # Fit the preprocessor
        df1 = preprocessor.preprocess_claims(sample_medical_claims)
        assert preprocessor.is_fitted is True

        # Should be able to transform new data
        df2 = preprocessor.transform_new_data(sample_medical_claims[:1])
        assert len(df2) == 1

        # Feature columns should be consistent
        assert list(df2.columns) == ['claim_id'] + preprocessor.feature_columns + ['fraud_indicator']

    def test_feature_consistency_across_transforms(self, preprocessor, sample_medical_claims):
        """Test that features are consistent across multiple transforms."""
        # Fit with training data
        df_train = preprocessor.preprocess_claims(sample_medical_claims)

        # Transform same data again
        df_test = preprocessor.transform_new_data(sample_medical_claims)

        # Should have same columns (except possibly different order)
        assert set(df_train.columns) == set(df_test.columns)
        assert len(df_train.columns) == len(df_test.columns)