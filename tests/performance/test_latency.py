"""
Latency performance tests for fraud detection system.

Tests single claim processing latency against <100ms requirement.
"""
import pytest
import time
import statistics
from typing import List, Dict, Any
import tempfile
import json
from pathlib import Path

from src.ingestion.data_loader import ClaimDataLoader
from src.ingestion.validator import ClaimValidator
from src.ingestion.preprocessor import ClaimPreprocessor
from src.detection.rule_engine import RuleEngine
from src.detection.ml_models import MLModelManager
from src.detection.feature_engineering import FeatureEngineer
from src.detection.anomaly_detector import AnomalyDetector

from tests.fixtures.claim_factories import ValidClaim, UpcodingFraudClaim
from tests.test_config import BENCHMARKS


class TestLatencyPerformance:
    """Test latency performance of fraud detection components."""

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims for latency testing."""
        claims = []

        # Valid claims
        for i in range(50):
            claim = ValidClaim()
            claims.append(claim)

        # Fraud claims
        for i in range(50):
            claim = UpcodingFraudClaim()
            claims.append(claim)

        return claims

    @pytest.fixture
    def temp_data_file(self, sample_claims):
        """Create temporary data file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_claims, f)
            temp_path = f.name

        yield Path(temp_path)

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def measure_latency(self, func, *args, **kwargs) -> float:
        """Measure function execution latency in milliseconds."""
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds

    def measure_multiple_runs(self, func, runs: int = 10, *args, **kwargs) -> Dict[str, float]:
        """Measure function latency over multiple runs and return statistics."""
        latencies = []

        for _ in range(runs):
            latency = self.measure_latency(func, *args, **kwargs)
            latencies.append(latency)

        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95': sorted(latencies)[int(0.95 * len(latencies))],
            'p99': sorted(latencies)[int(0.99 * len(latencies))]
        }

    @pytest.mark.latency
    @pytest.mark.performance
    def test_data_loader_single_file_latency(self, temp_data_file):
        """Test data loader latency for single file."""
        loader = ClaimDataLoader(temp_data_file.parent)

        def load_single_file():
            return loader._load_single_file(temp_data_file)

        stats = self.measure_multiple_runs(load_single_file, runs=20)

        print(f"Data Loader Single File Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Should load file within reasonable time
        assert stats['p95'] < 50  # 95th percentile under 50ms
        assert stats['mean'] < 20  # Mean under 20ms

    @pytest.mark.latency
    @pytest.mark.performance
    def test_validator_single_claim_latency(self, sample_claims):
        """Test validator latency for single claim."""
        validator = ClaimValidator()
        test_claim = sample_claims[0]

        def validate_single_claim():
            return validator.validate_schema(test_claim)

        stats = self.measure_multiple_runs(validate_single_claim, runs=100)

        print(f"Validator Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Validation should be very fast
        assert stats['p95'] < 5   # 95th percentile under 5ms
        assert stats['mean'] < 2  # Mean under 2ms

    @pytest.mark.latency
    @pytest.mark.performance
    def test_rule_engine_single_claim_latency(self, sample_claims):
        """Test rule engine latency for single claim."""
        rule_engine = RuleEngine()
        test_claim = sample_claims[0]

        def analyze_single_claim():
            return rule_engine.analyze_claim(test_claim)

        stats = self.measure_multiple_runs(analyze_single_claim, runs=50)

        print(f"Rule Engine Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Rule analysis should meet the <100ms requirement
        assert stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS
        assert stats['mean'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS / 2

    @pytest.mark.latency
    @pytest.mark.performance
    def test_preprocessor_single_claim_latency(self, sample_claims):
        """Test preprocessor latency for single claim."""
        preprocessor = ClaimPreprocessor()

        # Convert first claim to claim object
        from src.models.claim_models import claim_factory
        test_claim_obj = claim_factory(sample_claims[0])

        def preprocess_single_claim():
            return preprocessor.preprocess_claims([test_claim_obj])

        stats = self.measure_multiple_runs(preprocess_single_claim, runs=20)

        print(f"Preprocessor Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Preprocessing should be reasonably fast
        assert stats['p95'] < 50  # 95th percentile under 50ms
        assert stats['mean'] < 20  # Mean under 20ms

    @pytest.mark.latency
    @pytest.mark.performance
    def test_feature_engineering_single_claim_latency(self, sample_claims):
        """Test feature engineering latency for single claim."""
        feature_engineer = FeatureEngineer()
        test_claim = [sample_claims[0]]  # Single claim in list

        def engineer_features():
            return feature_engineer.extract_features(test_claim)

        stats = self.measure_multiple_runs(engineer_features, runs=20)

        print(f"Feature Engineering Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Feature engineering should be reasonably fast
        assert stats['p95'] < 100  # 95th percentile under 100ms
        assert stats['mean'] < 50   # Mean under 50ms

    @pytest.mark.latency
    @pytest.mark.performance
    def test_ml_prediction_single_claim_latency(self, sample_claims):
        """Test ML model prediction latency for single claim."""
        # Prepare training data
        from src.models.claim_models import claim_factory
        import pandas as pd

        # Convert sample claims to objects and preprocess
        claim_objects = [claim_factory(claim) for claim in sample_claims[:20]]
        preprocessor = ClaimPreprocessor()
        processed_df = preprocessor.preprocess_claims(claim_objects)

        # Train a simple model
        ml_manager = MLModelManager()
        X = processed_df[preprocessor.feature_columns]
        y = processed_df['fraud_indicator'] if 'fraud_indicator' in processed_df.columns else pd.Series([0] * len(X))

        # Train only random forest for speed
        ml_manager.model_configs = {'random_forest': ml_manager.model_configs['random_forest']}
        ml_manager.train_models(X, y)

        # Test single prediction
        single_claim = X.head(1)

        def predict_single_claim():
            return ml_manager.predict(single_claim, 'random_forest')

        stats = self.measure_multiple_runs(predict_single_claim, runs=50)

        print(f"ML Prediction Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # ML prediction should meet the <100ms requirement
        assert stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS
        assert stats['mean'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS / 2

    @pytest.mark.latency
    @pytest.mark.performance
    def test_anomaly_detection_single_claim_latency(self, sample_claims):
        """Test anomaly detection latency for single claim."""
        # Prepare data
        from src.models.claim_models import claim_factory
        import pandas as pd
        import numpy as np

        claim_objects = [claim_factory(claim) for claim in sample_claims[:50]]
        preprocessor = ClaimPreprocessor()
        processed_df = preprocessor.preprocess_claims(claim_objects)

        # Get numeric features
        numeric_features = processed_df.select_dtypes(include=[np.number])

        # Train anomaly detector
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(numeric_features)

        # Test single detection
        single_claim = numeric_features.head(1)

        def detect_anomaly():
            return anomaly_detector.detect_anomalies(single_claim)

        stats = self.measure_multiple_runs(detect_anomaly, runs=50)

        print(f"Anomaly Detection Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # Anomaly detection should be fast
        assert stats['p95'] < 20  # 95th percentile under 20ms
        assert stats['mean'] < 10  # Mean under 10ms

    @pytest.mark.latency
    @pytest.mark.performance
    def test_end_to_end_single_claim_latency(self, sample_claims):
        """Test end-to-end latency for single claim processing."""
        # Set up components
        validator = ClaimValidator()
        preprocessor = ClaimPreprocessor()
        rule_engine = RuleEngine()

        # Prepare single claim
        from src.models.claim_models import claim_factory
        test_claim_dict = sample_claims[0]

        def process_claim_end_to_end():
            # Step 1: Validation
            is_valid, errors = validator.validate_schema(test_claim_dict)
            if not is_valid:
                return None

            # Step 2: Create claim object
            claim_obj = claim_factory(test_claim_dict)

            # Step 3: Preprocessing (minimal for single claim)
            processed_df = preprocessor.preprocess_claims([claim_obj])

            # Step 4: Rule-based analysis
            results, fraud_score = rule_engine.analyze_claim(test_claim_dict)

            return {
                'processed': True,
                'fraud_score': fraud_score,
                'features': len(processed_df.columns)
            }

        stats = self.measure_multiple_runs(process_claim_end_to_end, runs=20)

        print(f"End-to-End Single Claim Latency:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

        # End-to-end processing should meet the <100ms requirement
        assert stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS
        assert stats['mean'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS / 2

        print(f"✓ Meets latency requirement: {BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS}ms")

    @pytest.mark.latency
    @pytest.mark.performance
    def test_latency_under_load(self, sample_claims):
        """Test latency under concurrent load."""
        import threading
        import queue

        rule_engine = RuleEngine()
        results_queue = queue.Queue()

        def process_claim_worker(claim_data, worker_id):
            """Worker function for concurrent processing."""
            latencies = []
            for i in range(10):  # Process 10 claims per worker
                start_time = time.perf_counter()
                rule_engine.analyze_claim(claim_data)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000
                latencies.append(latency)

            results_queue.put({
                'worker_id': worker_id,
                'latencies': latencies,
                'mean_latency': statistics.mean(latencies),
                'max_latency': max(latencies)
            })

        # Start multiple worker threads
        num_workers = 5
        threads = []
        test_claim = sample_claims[0]

        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=process_claim_worker,
                args=(test_claim, worker_id)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        all_latencies = []
        worker_results = []

        while not results_queue.empty():
            result = results_queue.get()
            worker_results.append(result)
            all_latencies.extend(result['latencies'])

        # Calculate overall statistics
        overall_stats = {
            'mean': statistics.mean(all_latencies),
            'median': statistics.median(all_latencies),
            'p95': sorted(all_latencies)[int(0.95 * len(all_latencies))],
            'p99': sorted(all_latencies)[int(0.99 * len(all_latencies))],
            'max': max(all_latencies)
        }

        print(f"Latency Under Load ({num_workers} workers, {len(all_latencies)} total operations):")
        print(f"  Mean: {overall_stats['mean']:.2f}ms")
        print(f"  P95: {overall_stats['p95']:.2f}ms")
        print(f"  P99: {overall_stats['p99']:.2f}ms")
        print(f"  Max: {overall_stats['max']:.2f}ms")

        # Under load, latency should still be reasonable
        assert overall_stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS * 2  # Allow 2x under load
        assert overall_stats['mean'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS

    @pytest.mark.latency
    @pytest.mark.performance
    def test_latency_with_different_claim_sizes(self):
        """Test latency with different claim complexity."""
        rule_engine = RuleEngine()

        # Create claims of different complexity
        test_cases = [
            {
                'name': 'Simple',
                'claim': {
                    'claim_id': 'CLM-SIMPLE',
                    'patient_id': 'PAT-001',
                    'provider_id': 'PRV-001',
                    'provider_npi': '1234567890',
                    'date_of_service': '2024-01-15',
                    'billed_amount': 100.0,
                    'diagnosis_codes': ['M79.3'],
                    'procedure_codes': ['99213'],
                    'fraud_indicator': False,
                    'red_flags': []
                }
            },
            {
                'name': 'Complex',
                'claim': {
                    'claim_id': 'CLM-COMPLEX',
                    'patient_id': 'PAT-002',
                    'provider_id': 'PRV-002',
                    'provider_npi': '1234567890',
                    'date_of_service': '2024-01-15',
                    'billed_amount': 5000.0,
                    'diagnosis_codes': ['M79.3', 'S13.4', 'E11.9', 'I10', 'F32.9'],
                    'procedure_codes': ['99213', '99214', '99215', '73721', '97110'],
                    'fraud_indicator': True,
                    'red_flags': ['excessive_billing', 'complexity_mismatch', 'suspicious_timing'],
                    'notes': 'Complex case with multiple procedures and diagnoses'
                }
            },
            {
                'name': 'Very Complex',
                'claim': {
                    'claim_id': 'CLM-VERY-COMPLEX',
                    'patient_id': 'PAT-003',
                    'provider_id': 'PRV-003',
                    'provider_npi': '1234567890',
                    'date_of_service': '2024-01-15',
                    'billed_amount': 15000.0,
                    'diagnosis_codes': [f'M79.{i}' for i in range(10)],  # 10 diagnoses
                    'procedure_codes': [f'9921{i}' for i in range(15)],  # 15 procedures
                    'fraud_indicator': True,
                    'red_flags': [
                        'excessive_billing', 'complexity_mismatch', 'suspicious_timing',
                        'pattern_matching', 'unbundling', 'phantom_billing'
                    ],
                    'notes': 'Very complex case with many procedures, diagnoses, and red flags'
                }
            }
        ]

        results = {}

        for test_case in test_cases:
            claim_name = test_case['name']
            claim_data = test_case['claim']

            def analyze_claim():
                return rule_engine.analyze_claim(claim_data)

            stats = self.measure_multiple_runs(analyze_claim, runs=30)
            results[claim_name] = stats

            print(f"{claim_name} Claim Latency:")
            print(f"  Mean: {stats['mean']:.2f}ms")
            print(f"  P95: {stats['p95']:.2f}ms")

        # All claim types should meet latency requirements
        for claim_name, stats in results.items():
            assert stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS, \
                f"{claim_name} claim latency exceeds requirement"

        # Complex claims should not be significantly slower
        simple_latency = results['Simple']['mean']
        complex_latency = results['Complex']['mean']
        very_complex_latency = results['Very Complex']['mean']

        # Complex claims should be at most 3x slower than simple ones
        assert complex_latency < simple_latency * 3
        assert very_complex_latency < simple_latency * 5

    @pytest.mark.latency
    @pytest.mark.performance
    def test_cold_start_vs_warm_latency(self, sample_claims):
        """Test cold start vs warm execution latency."""
        test_claim = sample_claims[0]

        # Test cold start (first execution)
        rule_engine = RuleEngine()
        cold_start_latency = self.measure_latency(rule_engine.analyze_claim, test_claim)

        # Test warm execution (subsequent executions)
        warm_latencies = []
        for _ in range(10):
            latency = self.measure_latency(rule_engine.analyze_claim, test_claim)
            warm_latencies.append(latency)

        warm_mean = statistics.mean(warm_latencies)

        print(f"Cold Start vs Warm Latency:")
        print(f"  Cold start: {cold_start_latency:.2f}ms")
        print(f"  Warm mean: {warm_mean:.2f}ms")
        print(f"  Difference: {cold_start_latency - warm_mean:.2f}ms")

        # Cold start should not be excessively slow
        assert cold_start_latency < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS * 2
        # Warm execution should be faster than cold start
        assert warm_mean <= cold_start_latency

    @pytest.mark.latency
    @pytest.mark.performance
    def test_memory_impact_on_latency(self, sample_claims):
        """Test how memory usage affects latency."""
        import gc

        rule_engine = RuleEngine()
        test_claim = sample_claims[0]

        # Baseline latency
        gc.collect()  # Clean up memory
        baseline_stats = self.measure_multiple_runs(
            rule_engine.analyze_claim, runs=20, claim=test_claim
        )

        # Create memory pressure
        memory_hog = []
        for _ in range(10000):
            memory_hog.append([0] * 1000)  # Create large objects

        # Measure latency under memory pressure
        pressure_stats = self.measure_multiple_runs(
            rule_engine.analyze_claim, runs=20, claim=test_claim
        )

        # Clean up
        del memory_hog
        gc.collect()

        print(f"Memory Impact on Latency:")
        print(f"  Baseline mean: {baseline_stats['mean']:.2f}ms")
        print(f"  Under pressure mean: {pressure_stats['mean']:.2f}ms")
        print(f"  Impact: {pressure_stats['mean'] - baseline_stats['mean']:.2f}ms")

        # Memory pressure should not severely impact latency
        latency_increase = pressure_stats['mean'] - baseline_stats['mean']
        assert latency_increase < 50  # Less than 50ms increase
        assert pressure_stats['p95'] < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS * 1.5