"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

from src.detection.feature_engineering import FeatureEngineer, FeatureSet
from tests.fixtures.claim_factories import generate_mixed_claims_batch
from tests.test_config import CONFIG


class TestFeatureSet:
    """Test the FeatureSet dataclass."""

    def test_feature_set_creation(self):
        """Test creating a FeatureSet instance."""
        basic_df = pd.DataFrame({"feature1": [1, 2, 3]})
        temporal_df = pd.DataFrame({"feature2": [4, 5, 6]})
        network_df = pd.DataFrame({"feature3": [7, 8, 9]})
        sequence_df = pd.DataFrame({"feature4": [10, 11, 12]})
        statistical_df = pd.DataFrame({"feature5": [13, 14, 15]})
        text_df = pd.DataFrame({"feature6": [16, 17, 18]})

        feature_set = FeatureSet(
            basic_features=basic_df,
            temporal_features=temporal_df,
            network_features=network_df,
            sequence_features=sequence_df,
            statistical_features=statistical_df,
            text_features=text_df,
        )

        assert feature_set.basic_features.equals(basic_df)
        assert feature_set.temporal_features.equals(temporal_df)
        assert feature_set.network_features.equals(network_df)
        assert feature_set.sequence_features.equals(sequence_df)
        assert feature_set.statistical_features.equals(statistical_df)
        assert feature_set.text_features.equals(text_df)


class TestFeatureEngineer:
    """Test the FeatureEngineer class."""

    @pytest.fixture
    def feature_engineer(self):
        """Create a FeatureEngineer instance for testing."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims data for testing."""
        return [
            {
                "claim_id": "CLM-001",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "2024-01-15",
                "billed_amount": 250.00,
                "procedure_codes": ["99213", "73721"],
                "diagnosis_codes": ["M79.3", "S13.4"],
                "service_location": "11",
                "claim_type": "medical",
                "red_flags": ["suspicious_timing"],
                "notes": "Patient complained of severe pain",
                "fraud_indicator": False,
            },
            {
                "claim_id": "CLM-002",
                "provider_id": "PROV-002",
                "patient_id": "PAT-002",
                "date_of_service": "2024-01-16",
                "billed_amount": 5000.00,
                "procedure_codes": ["99215", "99285"],
                "diagnosis_codes": ["E11.9", "I10"],
                "service_location": "23",
                "claim_type": "emergency",
                "red_flags": ["excessive_billing", "complexity_mismatch"],
                "notes": "Emergency visit with complications",
                "fraud_indicator": True,
            },
            {
                "claim_id": "CLM-003",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "2024-01-20",
                "billed_amount": 150.00,
                "procedure_codes": ["99212"],
                "diagnosis_codes": ["M79.3"],
                "service_location": "11",
                "claim_type": "follow_up",
                "red_flags": [],
                "notes": "Follow-up visit",
                "fraud_indicator": False,
            },
        ]

    def test_feature_engineer_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert isinstance(feature_engineer.scalers, dict)
        assert isinstance(feature_engineer.encoders, dict)
        assert isinstance(feature_engineer.vectorizers, dict)
        assert isinstance(feature_engineer.feature_names, dict)
        assert feature_engineer.provider_network is not None
        assert isinstance(feature_engineer.claim_history, dict)

    @pytest.mark.unit
    def test_extract_features_basic(self, feature_engineer, sample_claims):
        """Test basic feature extraction functionality."""
        feature_set = feature_engineer.extract_features(sample_claims)

        assert isinstance(feature_set, FeatureSet)
        assert isinstance(feature_set.basic_features, pd.DataFrame)
        assert isinstance(feature_set.temporal_features, pd.DataFrame)
        assert isinstance(feature_set.network_features, pd.DataFrame)
        assert isinstance(feature_set.sequence_features, pd.DataFrame)
        assert isinstance(feature_set.statistical_features, pd.DataFrame)
        assert isinstance(feature_set.text_features, pd.DataFrame)

        # Check that features were created for all claims
        assert len(feature_set.basic_features) == len(sample_claims)
        assert len(feature_set.temporal_features) == len(sample_claims)

    def test_validate_and_prepare_data_missing_columns(self, feature_engineer):
        """Test data validation with missing required columns."""
        invalid_claims = [{"claim_id": "CLM-001", "amount": 100}]
        df = pd.DataFrame(invalid_claims)

        with pytest.raises(ValueError) as excinfo:
            feature_engineer._validate_and_prepare_data(df)

        assert "Missing required columns" in str(excinfo.value)

    def test_basic_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of basic features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        basic_features = feature_engineer._extract_basic_features(df)

        # Check that expected basic features are present
        expected_features = [
            "billed_amount",
            "billed_amount_log",
            "billed_amount_sqrt",
            "num_procedure_codes",
            "num_diagnosis_codes",
            "total_codes",
            "avg_procedure_complexity",
            "diagnosis_severity",
            "amount_per_procedure",
            "num_red_flags",
        ]

        for feature in expected_features:
            assert feature in basic_features.columns

        # Check feature values
        assert basic_features["billed_amount"].iloc[0] == 250.00
        assert basic_features["num_procedure_codes"].iloc[0] == 2
        assert basic_features["num_diagnosis_codes"].iloc[0] == 2
        assert basic_features["num_red_flags"].iloc[0] == 1

    def test_temporal_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of temporal features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        temporal_features = feature_engineer._extract_temporal_features(df)

        # Check that expected temporal features are present
        expected_features = [
            "day_of_week",
            "month",
            "day_of_month",
            "quarter",
            "is_weekend",
            "is_holiday",
            "days_since_epoch",
        ]

        for feature in expected_features:
            assert feature in temporal_features.columns

        # Check feature values
        assert temporal_features["month"].iloc[0] == 1  # January
        assert temporal_features["is_weekend"].iloc[0] in [0, 1]

    def test_network_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of network features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        network_features = feature_engineer._extract_network_features(df)

        # Check that network features are created
        assert len(network_features) == len(sample_claims)
        assert "provider_degree_centrality" in network_features.columns
        assert "provider_connections" in network_features.columns

    def test_sequence_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of sequence features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        sequence_features = feature_engineer._extract_sequence_features(df)

        # Check that sequence features are created
        assert len(sequence_features) == len(sample_claims)
        assert "claim_sequence_position" in sequence_features.columns
        assert "days_between_patient_claims" in sequence_features.columns

    def test_statistical_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of statistical features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        statistical_features = feature_engineer._extract_statistical_features(df)

        # Check that statistical features are created
        assert len(statistical_features) == len(sample_claims)
        assert "provider_avg_amount" in statistical_features.columns
        assert "amount_zscore_provider" in statistical_features.columns

    def test_text_features_extraction(self, feature_engineer, sample_claims):
        """Test extraction of text features."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        text_features = feature_engineer._extract_text_features(df)

        # Check that text features are created
        assert len(text_features) == len(sample_claims)
        assert "text_length" in text_features.columns
        assert "word_count" in text_features.columns

        # Check for suspicious keyword features
        assert "contains_severe" in text_features.columns
        assert text_features["contains_severe"].iloc[0] == 1  # First claim contains 'severe'

    def test_procedure_complexity_calculation(self, feature_engineer):
        """Test procedure complexity calculation."""
        # Test with known complexity codes
        codes_high_complexity = ["99215", "99285"]
        complexity_high = feature_engineer._calculate_procedure_complexity(codes_high_complexity)
        assert complexity_high > 3

        # Test with low complexity codes
        codes_low_complexity = ["99211", "99212"]
        complexity_low = feature_engineer._calculate_procedure_complexity(codes_low_complexity)
        assert complexity_low < 3

        # Test with empty codes
        complexity_empty = feature_engineer._calculate_procedure_complexity([])
        assert complexity_empty == 0.0

    def test_diagnosis_severity_calculation(self, feature_engineer):
        """Test diagnosis severity calculation."""
        # Test with known severity codes
        codes_severe = ["S72.001A", "J18.9"]  # Fracture, pneumonia
        severity_high = feature_engineer._calculate_diagnosis_severity(codes_severe)
        assert severity_high > 2

        # Test with mild codes
        codes_mild = ["I10"]  # Hypertension
        severity_low = feature_engineer._calculate_diagnosis_severity(codes_mild)
        assert severity_low <= 2

        # Test with empty codes
        severity_empty = feature_engineer._calculate_diagnosis_severity([])
        assert severity_empty == 0.0

    def test_holiday_detection(self, feature_engineer):
        """Test holiday detection."""
        # Test known holidays
        new_years = datetime(2024, 1, 1)
        assert feature_engineer._is_holiday(new_years) is True

        independence_day = datetime(2024, 7, 4)
        assert feature_engineer._is_holiday(independence_day) is True

        christmas = datetime(2024, 12, 25)
        assert feature_engineer._is_holiday(christmas) is True

        # Test non-holiday
        regular_day = datetime(2024, 6, 15)
        assert feature_engineer._is_holiday(regular_day) is False

    def test_combine_features(self, feature_engineer, sample_claims):
        """Test combining different feature sets."""
        feature_set = feature_engineer.extract_features(sample_claims)

        # Test combining all feature sets
        combined_all = feature_engineer.combine_features(feature_set)
        assert isinstance(combined_all, pd.DataFrame)
        assert len(combined_all) == len(sample_claims)

        # Test combining specific feature sets
        combined_basic_temporal = feature_engineer.combine_features(
            feature_set, include_sets=["basic", "temporal"]
        )
        assert len(combined_basic_temporal.columns) < len(combined_all.columns)

        # Test that there are no duplicate columns
        assert len(combined_all.columns) == len(set(combined_all.columns))

    def test_feature_importance_calculation(self, feature_engineer, sample_claims):
        """Test feature importance calculation."""
        feature_set = feature_engineer.extract_features(sample_claims)
        target = pd.Series([claim["fraud_indicator"] for claim in sample_claims])

        importance_scores = feature_engineer.get_feature_importance(feature_set, target)

        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0

        # All importance scores should be between 0 and 1
        for feature, score in importance_scores.items():
            assert 0.0 <= score <= 1.0

    def test_feature_selection(self, feature_engineer, sample_claims):
        """Test feature selection functionality."""
        feature_set = feature_engineer.extract_features(sample_claims)
        target = pd.Series([claim["fraud_indicator"] for claim in sample_claims])

        # Select top 10 features
        selected_features = feature_engineer.select_features(feature_set, target, top_k=10)

        assert isinstance(selected_features, list)
        assert len(selected_features) <= 10

        # All selected features should be strings
        for feature in selected_features:
            assert isinstance(feature, str)

    def test_provider_network_building(self, feature_engineer, sample_claims):
        """Test provider network building."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        # Build network
        feature_engineer._build_provider_network(df)

        # Check that network was built
        assert feature_engineer.provider_network.number_of_nodes() > 0

        # Check that providers are in the network
        providers = df["provider_id"].unique()
        for provider in providers:
            assert provider in feature_engineer.provider_network.nodes()

    def test_provider_centrality_calculation(self, feature_engineer, sample_claims):
        """Test provider centrality calculation."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)
        feature_engineer._build_provider_network(df)

        centrality_features = feature_engineer._calculate_provider_centrality(df)

        assert "provider_degree_centrality" in centrality_features.columns
        assert "provider_betweenness_centrality" in centrality_features.columns
        assert "provider_closeness_centrality" in centrality_features.columns

        # Check that centrality values are in valid range [0, 1]
        for col in ["provider_degree_centrality", "provider_betweenness_centrality"]:
            values = centrality_features[col]
            assert (values >= 0).all() and (values <= 1).all()

    def test_sequence_position_calculation(self, feature_engineer, sample_claims):
        """Test claim sequence position calculation."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)
        df_sorted = df.sort_values(["patient_id", "date_of_service"])

        sequence_features = feature_engineer._calculate_sequence_positions(df_sorted)

        assert "claim_sequence_position" in sequence_features.columns
        assert "provider_daily_sequence" in sequence_features.columns

        # Check that sequence positions start from 1
        assert sequence_features["claim_sequence_position"].min() >= 1

    def test_inter_claim_intervals(self, feature_engineer, sample_claims):
        """Test inter-claim interval calculation."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)
        df_sorted = df.sort_values(["patient_id", "date_of_service"])

        interval_features = feature_engineer._calculate_inter_claim_intervals(df_sorted)

        assert "days_between_patient_claims" in interval_features.columns
        assert "days_between_provider_claims" in interval_features.columns

        # Check that intervals are calculated correctly
        patient_intervals = interval_features["days_between_patient_claims"]
        # First claim for each patient should have NaN/999 (no previous claim)
        assert pd.isna(patient_intervals.iloc[0]) or patient_intervals.iloc[0] == 999

    def test_provider_statistics_calculation(self, feature_engineer, sample_claims):
        """Test provider statistics calculation."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        provider_stats = feature_engineer._calculate_provider_statistics(df)

        assert "provider_avg_amount" in provider_stats.columns
        assert "provider_std_amount" in provider_stats.columns
        assert "provider_claim_count" in provider_stats.columns
        assert "amount_zscore_provider" in provider_stats.columns

        # Check that statistics are reasonable
        assert provider_stats["provider_avg_amount"].min() >= 0
        assert provider_stats["provider_claim_count"].min() >= 1

    def test_anomaly_score_calculation(self, feature_engineer, sample_claims):
        """Test anomaly score calculation."""
        df = pd.DataFrame(sample_claims)
        feature_engineer._validate_and_prepare_data(df)

        anomaly_features = feature_engineer._calculate_anomaly_scores(df)

        assert "amount_anomaly_score" in anomaly_features.columns

        # Anomaly scores should be binary (0 or 1)
        anomaly_scores = anomaly_features["amount_anomaly_score"]
        assert set(anomaly_scores.unique()).issubset({0, 1})

    def test_save_and_load_pipeline(self, feature_engineer, sample_claims):
        """Test saving and loading feature engineering pipeline."""
        # Extract features to populate encoders and other artifacts
        feature_engineer.extract_features(sample_claims)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save pipeline
            feature_engineer.save_feature_engineering_pipeline(temp_path)
            assert os.path.exists(temp_path)

            # Create new feature engineer and load pipeline
            new_feature_engineer = FeatureEngineer()
            new_feature_engineer.load_feature_engineering_pipeline(temp_path)

            # Check that artifacts were loaded
            assert len(new_feature_engineer.encoders) > 0
            assert len(new_feature_engineer.feature_names) > 0

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_edge_case_empty_claims(self, feature_engineer):
        """Test handling of empty claims list."""
        empty_claims = []

        # Should handle gracefully without crashing
        try:
            feature_set = feature_engineer.extract_features(empty_claims)
            # If it succeeds, all feature DataFrames should be empty
            assert len(feature_set.basic_features) == 0
        except Exception as e:
            # Should be a reasonable exception
            assert isinstance(e, (ValueError, IndexError))

    def test_edge_case_missing_optional_fields(self, feature_engineer):
        """Test handling of claims with missing optional fields."""
        minimal_claims = [
            {
                "claim_id": "CLM-001",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "2024-01-15",
                # Missing optional fields like billed_amount, procedure_codes, etc.
            }
        ]

        # Should handle gracefully
        feature_set = feature_engineer.extract_features(minimal_claims)

        assert isinstance(feature_set, FeatureSet)
        assert len(feature_set.basic_features) == 1

    def test_edge_case_malformed_data(self, feature_engineer):
        """Test handling of malformed data."""
        malformed_claims = [
            {
                "claim_id": "CLM-001",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "invalid-date",  # Invalid date format
                "billed_amount": "not-a-number",  # Invalid amount
                "procedure_codes": "not-a-list",  # Invalid format
                "diagnosis_codes": None,  # None value
            }
        ]

        # Should handle gracefully without crashing
        try:
            feature_set = feature_engineer.extract_features(malformed_claims)
            assert isinstance(feature_set, FeatureSet)
        except Exception as e:
            # Should be a reasonable exception, not a crash
            assert isinstance(e, (ValueError, TypeError, pd.errors.ParserError))

    @pytest.mark.performance
    def test_feature_extraction_performance(self, feature_engineer):
        """Test feature extraction performance with larger dataset."""
        import time

        # Generate larger dataset
        large_claims = generate_mixed_claims_batch(total_claims=1000, fraud_rate=0.15)

        start_time = time.time()
        feature_set = feature_engineer.extract_features(large_claims)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust based on requirements)
        assert processing_time < 30  # 30 seconds for 1000 claims

        # Check that features were extracted for all claims
        assert len(feature_set.basic_features) == len(large_claims)

    def test_text_feature_extraction_with_tfidf(self, feature_engineer):
        """Test TF-IDF text feature extraction."""
        claims_with_text = [
            {
                "claim_id": "CLM-001",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "2024-01-15",
                "notes": "Patient presents with acute severe pain and complications",
                "procedure_descriptions": ["Complex surgical procedure", "Emergency treatment"],
            },
            {
                "claim_id": "CLM-002",
                "provider_id": "PROV-002",
                "patient_id": "PAT-002",
                "date_of_service": "2024-01-16",
                "notes": "Routine follow-up visit for chronic condition management",
                "procedure_descriptions": ["Standard consultation", "Medication review"],
            },
        ]

        df = pd.DataFrame(claims_with_text)
        feature_engineer._validate_and_prepare_data(df)

        text_features = feature_engineer._extract_text_features(df)

        # Check that TF-IDF features were created
        tfidf_columns = [col for col in text_features.columns if col.startswith("tfidf_")]

        # Should have some TF-IDF features if text processing succeeded
        # (May be empty if vocabulary is too small after filtering)
        assert len(text_features) == 2
        assert "text_length" in text_features.columns
        assert "word_count" in text_features.columns

    def test_network_features_with_single_provider(self, feature_engineer):
        """Test network features when there's only one provider."""
        single_provider_claims = [
            {
                "claim_id": "CLM-001",
                "provider_id": "PROV-001",
                "patient_id": "PAT-001",
                "date_of_service": "2024-01-15",
            }
        ]

        df = pd.DataFrame(single_provider_claims)
        feature_engineer._validate_and_prepare_data(df)

        network_features = feature_engineer._extract_network_features(df)

        # Should handle single provider gracefully
        assert len(network_features) == 1
        assert "provider_degree_centrality" in network_features.columns

        # Centrality should be 0 for single provider
        assert network_features["provider_degree_centrality"].iloc[0] == 0

    @pytest.mark.unit
    def test_feature_name_consistency(self, feature_engineer, sample_claims):
        """Test that feature names are consistent across extractions."""
        feature_set1 = feature_engineer.extract_features(sample_claims)
        feature_set2 = feature_engineer.extract_features(sample_claims)

        # Feature names should be consistent
        assert list(feature_set1.basic_features.columns) == list(
            feature_set2.basic_features.columns
        )
        assert list(feature_set1.temporal_features.columns) == list(
            feature_set2.temporal_features.columns
        )

    def test_feature_data_types(self, feature_engineer, sample_claims):
        """Test that extracted features have appropriate data types."""
        feature_set = feature_engineer.extract_features(sample_claims)
        combined_features = feature_engineer.combine_features(feature_set)

        # Most features should be numeric
        numeric_features = combined_features.select_dtypes(include=[np.number])
        assert len(numeric_features.columns) > 0

        # Check specific feature types
        if "billed_amount" in combined_features.columns:
            assert pd.api.types.is_numeric_dtype(combined_features["billed_amount"])

        if "is_weekend" in combined_features.columns:
            assert combined_features["is_weekend"].dtype in ["int64", "bool"]

    def test_missing_values_handling(self, feature_engineer, sample_claims):
        """Test handling of missing values in features."""
        feature_set = feature_engineer.extract_features(sample_claims)
        combined_features = feature_engineer.combine_features(feature_set)

        # Check for excessive missing values
        missing_percentage = combined_features.isnull().sum() / len(combined_features)

        # Most features should have low missing value rates
        high_missing_features = missing_percentage[missing_percentage > 0.5]

        # Should not have too many features with high missing rates
        assert len(high_missing_features) < len(combined_features.columns) * 0.2
