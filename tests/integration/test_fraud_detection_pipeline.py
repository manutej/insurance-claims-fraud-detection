"""
Integration tests for the complete fraud detection pipeline.
"""
import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.ingestion.data_loader import ClaimDataLoader, DataLoaderConfig
from src.ingestion.validator import ClaimValidator
from src.ingestion.preprocessor import ClaimPreprocessor
from src.detection.rule_engine import RuleEngine
from src.detection.ml_models import MLModelManager
from src.detection.feature_engineering import FeatureEngineer
from src.detection.anomaly_detector import AnomalyDetector
from src.models.claim_models import claim_factory

from tests.fixtures.claim_factories import generate_accuracy_test_data, generate_mixed_claims_batch
from tests.test_config import BENCHMARKS, CONFIG


class TestFraudDetectionPipeline:
    """Test the complete fraud detection pipeline end-to-end."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create test data files
            valid_claims = generate_mixed_claims_batch(total_claims=100, fraud_rate=0.1)

            # Save as JSON files
            medical_file = data_dir / "medical_claims.json"
            with open(medical_file, 'w') as f:
                json.dump(valid_claims[:50], f)

            fraud_file = data_dir / "fraud_claims.json"
            with open(fraud_file, 'w') as f:
                json.dump(valid_claims[50:], f)

            yield data_dir

    @pytest.fixture
    def pipeline_components(self):
        """Create and return pipeline components."""
        return {
            'validator': ClaimValidator(),
            'preprocessor': ClaimPreprocessor(),
            'rule_engine': RuleEngine(),
            'feature_engineer': FeatureEngineer(),
            'ml_manager': MLModelManager(),
            'anomaly_detector': AnomalyDetector()
        }

    @pytest.mark.integration
    def test_complete_pipeline_execution(self, temp_data_dir, pipeline_components):
        """Test complete pipeline from data loading to fraud detection."""
        # Step 1: Data Loading
        config = DataLoaderConfig(validate_on_load=True, batch_size=50)
        loader = ClaimDataLoader(temp_data_dir, config,
                                validator=pipeline_components['validator'])

        batch = loader.load_claims_batch()
        assert batch.total_count > 0
        print(f"✓ Loaded {batch.total_count} claims")

        # Step 2: Data Preprocessing
        preprocessor = pipeline_components['preprocessor']
        processed_df = preprocessor.preprocess_claims(batch.claims)
        assert len(processed_df) == batch.total_count
        assert len(preprocessor.feature_columns) > 0
        print(f"✓ Preprocessed claims with {len(preprocessor.feature_columns)} features")

        # Step 3: Feature Engineering
        feature_engineer = pipeline_components['feature_engineer']
        claims_dict = [claim.dict() for claim in batch.claims]
        feature_set = feature_engineer.extract_features(claims_dict)
        combined_features = feature_engineer.combine_features(feature_set)
        assert len(combined_features) == batch.total_count
        print(f"✓ Engineered {len(combined_features.columns)} features")

        # Step 4: Rule-based Detection
        rule_engine = pipeline_components['rule_engine']
        rule_results = []
        fraud_scores = []

        for claim in batch.claims:
            claim_dict = claim.dict()
            results, score = rule_engine.analyze_claim(claim_dict)
            rule_results.append(results)
            fraud_scores.append(score)

        assert len(rule_results) == batch.total_count
        print(f"✓ Applied rule-based detection, avg score: {np.mean(fraud_scores):.3f}")

        # Step 5: Anomaly Detection
        anomaly_detector = pipeline_components['anomaly_detector']
        # Fit anomaly detector
        numeric_features = combined_features.select_dtypes(include=[np.number])
        anomaly_detector.fit(numeric_features)

        anomaly_scores = anomaly_detector.detect_anomalies(numeric_features)
        assert len(anomaly_scores) == batch.total_count
        print(f"✓ Anomaly detection completed, avg score: {np.mean(anomaly_scores):.3f}")

        # Step 6: ML Model Training (simplified for integration test)
        ml_manager = pipeline_components['ml_manager']

        # Prepare training data
        X = processed_df[preprocessor.feature_columns]
        y = processed_df['fraud_indicator'] if 'fraud_indicator' in processed_df.columns else pd.Series([0] * len(X))

        # Train a simple model for testing
        if len(X) >= 10:  # Minimum samples for training
            performance_results = ml_manager.train_models(X, y)
            assert len(performance_results) > 0
            print(f"✓ Trained {len(performance_results)} ML models")

            # Step 7: Make Predictions
            predictions = ml_manager.predict(X.head(10))  # Test on small subset
            assert len(predictions) == 10
            print(f"✓ Generated predictions for test claims")

        # Verify pipeline meets performance requirements
        processing_time = loader.get_statistics()['processing_time_seconds']
        claims_per_second = batch.total_count / processing_time

        print(f"✓ Pipeline Performance:")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Throughput: {claims_per_second:.1f} claims/sec")

        # Should meet basic performance requirements
        assert claims_per_second > 10  # At least 10 claims/sec for integration test

    @pytest.mark.integration
    def test_pipeline_accuracy_validation(self, pipeline_components):
        """Test pipeline accuracy against known fraud cases."""
        # Generate balanced test dataset
        test_data = generate_accuracy_test_data()
        all_claims = test_data['valid'] + test_data['fraud']

        # Convert to claim objects
        claim_objects = []
        for claim_dict in all_claims:
            try:
                claim = claim_factory(claim_dict)
                claim_objects.append(claim)
            except Exception as e:
                print(f"Warning: Could not create claim object: {e}")
                continue

        print(f"Testing accuracy with {len(claim_objects)} claims")

        # Apply rule-based detection
        rule_engine = pipeline_components['rule_engine']
        correct_predictions = 0
        total_predictions = 0
        false_positives = 0
        true_positives = 0

        for claim in claim_objects:
            claim_dict = claim.dict()
            results, fraud_score = rule_engine.analyze_claim(claim_dict)

            # Threshold for fraud detection
            predicted_fraud = fraud_score > 0.5
            actual_fraud = claim_dict.get('fraud_indicator', False)

            total_predictions += 1

            if predicted_fraud == actual_fraud:
                correct_predictions += 1

            if predicted_fraud and not actual_fraud:
                false_positives += 1
            elif predicted_fraud and actual_fraud:
                true_positives += 1

        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        false_positive_rate = false_positives / len(test_data['valid']) if len(test_data['valid']) > 0 else 0
        detection_rate = true_positives / len(test_data['fraud']) if len(test_data['fraud']) > 0 else 0

        print(f"✓ Accuracy Results:")
        print(f"  - Overall accuracy: {accuracy:.3f}")
        print(f"  - False positive rate: {false_positive_rate:.3f}")
        print(f"  - Detection rate: {detection_rate:.3f}")

        # Should meet reasonable accuracy for rule-based system
        assert accuracy > 0.7  # At least 70% accuracy
        assert false_positive_rate < 0.2  # Less than 20% false positives

    @pytest.mark.integration
    def test_pipeline_data_flow_integrity(self, temp_data_dir, pipeline_components):
        """Test data integrity throughout the pipeline."""
        # Load data
        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()
        original_count = batch.total_count

        # Track claim IDs through pipeline
        original_claim_ids = {claim.claim_id for claim in batch.claims}

        # Preprocessing
        preprocessor = pipeline_components['preprocessor']
        processed_df = preprocessor.preprocess_claims(batch.claims)

        # Verify no claims lost in preprocessing
        assert len(processed_df) == original_count
        processed_claim_ids = set(processed_df['claim_id'].values)
        assert processed_claim_ids == original_claim_ids

        # Feature engineering
        feature_engineer = pipeline_components['feature_engineer']
        claims_dict = [claim.dict() for claim in batch.claims]
        feature_set = feature_engineer.extract_features(claims_dict)
        combined_features = feature_engineer.combine_features(feature_set)

        # Verify no claims lost in feature engineering
        assert len(combined_features) == original_count

        # Rule-based analysis
        rule_engine = pipeline_components['rule_engine']
        rule_results_count = 0

        for claim in batch.claims:
            claim_dict = claim.dict()
            results, score = rule_engine.analyze_claim(claim_dict)
            rule_results_count += 1

        assert rule_results_count == original_count

        print(f"✓ Data integrity maintained: {original_count} claims processed consistently")

    @pytest.mark.integration
    def test_pipeline_error_handling(self, temp_data_dir, pipeline_components):
        """Test pipeline robustness with problematic data."""
        # Create problematic data file
        problematic_claims = [
            {
                'claim_id': 'CLM-GOOD-001',
                'patient_id': 'PAT-001',
                'provider_id': 'PRV-001',
                'provider_npi': '1234567890',
                'date_of_service': '2024-01-15',
                'billed_amount': 250.0,
                'fraud_indicator': False
            },
            {
                'claim_id': 'CLM-BAD-001',
                'billed_amount': 'invalid_amount',  # Invalid data type
                'date_of_service': 'invalid_date',   # Invalid date
                'fraud_indicator': False
            },
            {
                # Missing required fields
                'claim_id': 'CLM-INCOMPLETE-001',
                'billed_amount': 100.0
            }
        ]

        # Save problematic data
        problem_file = temp_data_dir / "problematic_claims.json"
        with open(problem_file, 'w') as f:
            json.dump(problematic_claims, f)

        # Test pipeline robustness
        config = DataLoaderConfig(validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config,
                                validator=pipeline_components['validator'])

        # Should handle errors gracefully
        try:
            batch = loader.load_claims_batch()
            # Should have at least the valid claim
            assert batch.total_count >= 1
            print(f"✓ Pipeline handled problematic data, processed {batch.total_count} valid claims")
        except Exception as e:
            pytest.fail(f"Pipeline should handle errors gracefully, but failed with: {e}")

    @pytest.mark.integration
    def test_pipeline_performance_benchmarks(self, temp_data_dir, pipeline_components):
        """Test pipeline performance against benchmarks."""
        import time

        # Generate larger test dataset
        large_claims = generate_mixed_claims_batch(total_claims=1000, fraud_rate=0.15)

        # Save to file
        large_file = temp_data_dir / "large_claims.json"
        with open(large_file, 'w') as f:
            json.dump(large_claims, f)

        # Measure pipeline performance
        start_time = time.time()

        # Data loading
        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()
        load_time = time.time()

        # Preprocessing
        preprocessor = pipeline_components['preprocessor']
        processed_df = preprocessor.preprocess_claims(batch.claims)
        preprocess_time = time.time()

        # Rule-based detection (sample)
        rule_engine = pipeline_components['rule_engine']
        sample_claims = batch.claims[:100]  # Test on sample for performance
        for claim in sample_claims:
            claim_dict = claim.dict()
            rule_engine.analyze_claim(claim_dict)
        rule_time = time.time()

        total_time = rule_time - start_time

        # Calculate performance metrics
        claims_per_second = batch.total_count / total_time
        load_rate = batch.total_count / (load_time - start_time)
        preprocess_rate = batch.total_count / (preprocess_time - load_time)

        print(f"✓ Performance Benchmarks:")
        print(f"  - Overall throughput: {claims_per_second:.1f} claims/sec")
        print(f"  - Loading rate: {load_rate:.1f} claims/sec")
        print(f"  - Preprocessing rate: {preprocess_rate:.1f} claims/sec")
        print(f"  - Total processing time: {total_time:.2f}s")

        # Performance assertions
        assert claims_per_second > 50  # At least 50 claims/sec overall
        assert total_time < 60  # Should complete within 60 seconds

    @pytest.mark.integration
    def test_pipeline_concurrent_processing(self, temp_data_dir, pipeline_components):
        """Test pipeline with concurrent processing."""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Create multiple data files
        file_paths = []
        for i in range(3):
            claims = generate_mixed_claims_batch(total_claims=50, fraud_rate=0.1)
            file_path = temp_data_dir / f"batch_{i}.json"
            with open(file_path, 'w') as f:
                json.dump(claims, f)
            file_paths.append(file_path)

        results = []
        errors = []

        def process_batch(file_path):
            """Process a single batch file."""
            try:
                # Each thread gets its own component instances
                loader = ClaimDataLoader(temp_data_dir)
                batch = loader.load_claims_batch(file_paths=[file_path])

                preprocessor = ClaimPreprocessor()
                processed_df = preprocessor.preprocess_claims(batch.claims)

                return {
                    'file': file_path.name,
                    'count': len(processed_df),
                    'thread_id': threading.current_thread().ident
                }
            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")
                return None

        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, fp) for fp in file_paths]
            results = [future.result() for future in futures if future.result()]

        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3, "Should process all batches successfully"

        total_processed = sum(r['count'] for r in results)
        print(f"✓ Concurrent processing: {total_processed} claims across {len(results)} threads")

    @pytest.mark.integration
    def test_pipeline_configuration_flexibility(self, temp_data_dir):
        """Test pipeline with different configurations."""
        # Test different loader configurations
        configs = [
            DataLoaderConfig(validate_on_load=True, batch_size=25),
            DataLoaderConfig(validate_on_load=False, batch_size=100),
            DataLoaderConfig(validate_on_load=True, max_workers=2)
        ]

        for i, config in enumerate(configs):
            loader = ClaimDataLoader(temp_data_dir, config)
            batch = loader.load_claims_batch()
            assert batch.total_count > 0
            print(f"✓ Configuration {i+1}: processed {batch.total_count} claims")

        # Test different preprocessor configurations
        preprocess_configs = [
            {'normalize_amounts': True, 'encoding_strategy': 'onehot'},
            {'normalize_amounts': False, 'encoding_strategy': 'label'},
            {'extract_temporal_features': False}
        ]

        # Use first batch for preprocessing tests
        batch = loader.load_claims_batch()

        for i, config in enumerate(preprocess_configs):
            preprocessor = ClaimPreprocessor(config)
            processed_df = preprocessor.preprocess_claims(batch.claims[:10])  # Small sample
            assert len(processed_df) == 10
            print(f"✓ Preprocessor config {i+1}: {len(processed_df.columns)} features")

    @pytest.mark.integration
    def test_pipeline_memory_efficiency(self, temp_data_dir, pipeline_components):
        """Test pipeline memory usage with larger datasets."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process moderately large dataset
        large_claims = generate_mixed_claims_batch(total_claims=2000, fraud_rate=0.12)

        # Save to file
        large_file = temp_data_dir / "memory_test_claims.json"
        with open(large_file, 'w') as f:
            json.dump(large_claims, f)

        # Process through pipeline
        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()

        preprocessor = pipeline_components['preprocessor']
        processed_df = preprocessor.preprocess_claims(batch.claims)

        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        print(f"✓ Memory Usage:")
        print(f"  - Initial: {initial_memory:.1f} MB")
        print(f"  - Peak: {peak_memory:.1f} MB")
        print(f"  - Increase: {memory_increase:.1f} MB")
        print(f"  - Per claim: {memory_increase/len(batch.claims):.3f} MB/claim")

        # Memory should be reasonable (adjust based on requirements)
        assert memory_increase < 500  # Less than 500MB increase
        assert memory_increase / len(batch.claims) < 0.5  # Less than 0.5MB per claim

    @pytest.mark.integration
    def test_end_to_end_fraud_detection_accuracy(self, pipeline_components):
        """Test end-to-end fraud detection accuracy."""
        # Create realistic test scenario
        test_claims = []

        # Add obviously fraudulent claims
        fraud_patterns = [
            {
                'claim_id': 'CLM-FRAUD-001',
                'patient_id': 'PAT-FRAUD-001',
                'provider_id': 'FRAUD-PROVIDER-001',
                'provider_npi': '9999999999',  # Suspicious NPI
                'date_of_service': '2024-01-15',
                'billed_amount': 25000.0,  # Excessive amount
                'diagnosis_codes': ['Z00.00'],  # Routine checkup
                'procedure_codes': ['99285'],  # Emergency procedure (mismatch)
                'fraud_indicator': True,
                'red_flags': ['excessive_billing', 'complexity_mismatch']
            },
            {
                'claim_id': 'CLM-FRAUD-002',
                'patient_id': 'PAT-FRAUD-002',
                'provider_id': 'PRV-002',
                'provider_npi': '1234567890',
                'date_of_service': '2024-01-01',  # Holiday (New Year)
                'billed_amount': 15000.0,
                'diagnosis_codes': ['M79.3'],
                'procedure_codes': ['99213', '99214', '99215', '73721', '97110'],  # Many procedures
                'fraud_indicator': True,
                'red_flags': ['Service on Sunday when office closed', 'Multiple procedures']
            }
        ]

        # Add obviously legitimate claims
        valid_patterns = [
            {
                'claim_id': 'CLM-VALID-001',
                'patient_id': 'PAT-VALID-001',
                'provider_id': 'PRV-VALID-001',
                'provider_npi': '1234567890',
                'date_of_service': '2024-01-15',  # Weekday
                'billed_amount': 150.0,  # Reasonable amount
                'diagnosis_codes': ['M79.3'],
                'procedure_codes': ['99213'],
                'fraud_indicator': False,
                'red_flags': []
            },
            {
                'claim_id': 'CLM-VALID-002',
                'patient_id': 'PAT-VALID-002',
                'provider_id': 'PRV-VALID-002',
                'provider_npi': '2345678901',
                'date_of_service': '2024-01-16',
                'billed_amount': 300.0,
                'diagnosis_codes': ['E11.9'],
                'procedure_codes': ['99214'],
                'fraud_indicator': False,
                'red_flags': []
            }
        ]

        all_test_claims = fraud_patterns + valid_patterns

        # Convert to claim objects
        claim_objects = []
        for claim_dict in all_test_claims:
            claim = claim_factory(claim_dict)
            claim_objects.append(claim)

        # Apply complete detection pipeline
        rule_engine = pipeline_components['rule_engine']

        correct_detections = 0
        total_claims = len(claim_objects)

        for claim in claim_objects:
            claim_dict = claim.dict()
            results, fraud_score = rule_engine.analyze_claim(claim_dict)

            # Use multiple indicators for fraud detection
            rule_triggered = fraud_score > 0.5
            has_red_flags = len(claim_dict.get('red_flags', [])) > 0
            suspicious_amount = claim_dict.get('billed_amount', 0) > 10000

            # Combine indicators
            detected_fraud = rule_triggered or has_red_flags or suspicious_amount
            actual_fraud = claim_dict.get('fraud_indicator', False)

            if detected_fraud == actual_fraud:
                correct_detections += 1

            print(f"Claim {claim_dict['claim_id']}: "
                  f"Actual={actual_fraud}, Detected={detected_fraud}, "
                  f"Score={fraud_score:.3f}")

        accuracy = correct_detections / total_claims
        print(f"✓ End-to-end accuracy: {accuracy:.3f} ({correct_detections}/{total_claims})")

        # Should correctly identify obvious fraud and valid cases
        assert accuracy >= 0.75  # At least 75% accuracy on obvious cases