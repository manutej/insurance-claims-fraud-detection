"""
Throughput performance tests for fraud detection system.

Tests system throughput against 1000 claims/sec requirement.
"""
import pytest
import time
import statistics
import tempfile
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading

from src.ingestion.data_loader import ClaimDataLoader, DataLoaderConfig
from src.ingestion.validator import ClaimValidator
from src.ingestion.preprocessor import ClaimPreprocessor
from src.detection.rule_engine import RuleEngine
from src.detection.ml_models import MLModelManager
from src.detection.feature_engineering import FeatureEngineer

from tests.fixtures.claim_factories import generate_mixed_claims_batch
from tests.test_config import BENCHMARKS


class TestThroughputPerformance:
    """Test throughput performance of fraud detection system."""

    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for throughput testing."""
        return generate_mixed_claims_batch(total_claims=5000, fraud_rate=0.15)

    @pytest.fixture
    def temp_data_files(self, large_dataset):
        """Create multiple temporary data files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Split dataset into multiple files for concurrent processing
            chunk_size = 1000
            file_paths = []

            for i in range(0, len(large_dataset), chunk_size):
                chunk = large_dataset[i:i + chunk_size]
                file_path = data_dir / f"batch_{i//chunk_size}.json"
                with open(file_path, 'w') as f:
                    json.dump(chunk, f)
                file_paths.append(file_path)

            yield data_dir, file_paths

    def measure_throughput(self, func, dataset_size: int, *args, **kwargs) -> dict:
        """Measure function throughput and return statistics."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = dataset_size / processing_time

        return {
            'processing_time': processing_time,
            'throughput': throughput,
            'dataset_size': dataset_size,
            'result': result
        }

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_data_loader_throughput(self, temp_data_files):
        """Test data loader throughput performance."""
        data_dir, file_paths = temp_data_files

        config = DataLoaderConfig(
            batch_size=1000,
            max_workers=4,
            validate_on_load=False  # Disable validation for pure loading test
        )
        loader = ClaimDataLoader(data_dir, config)

        def load_all_data():
            return loader.load_claims_batch()

        # Measure throughput
        start_time = time.perf_counter()
        batch = load_all_data()
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = batch.total_count / processing_time

        print(f"Data Loader Throughput:")
        print(f"  Claims processed: {batch.total_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")

        # Should meet minimum throughput for loading
        assert throughput > 2000  # At least 2000 claims/sec for loading

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_validator_throughput(self, large_dataset):
        """Test validator throughput performance."""
        validator = ClaimValidator()

        # Test batch validation
        start_time = time.perf_counter()
        result = validator.validate_batch(large_dataset[:2000])  # Test subset
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = result.processed_count / processing_time

        print(f"Validator Throughput:")
        print(f"  Claims processed: {result.processed_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")
        print(f"  Errors: {result.error_count}")

        # Validation should be very fast
        assert throughput > 5000  # At least 5000 claims/sec for validation

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_rule_engine_throughput(self, large_dataset):
        """Test rule engine throughput performance."""
        rule_engine = RuleEngine()

        # Process subset for throughput test
        test_data = large_dataset[:2000]

        start_time = time.perf_counter()

        results = []
        for claim_data in test_data:
            result, score = rule_engine.analyze_claim(claim_data)
            results.append((result, score))

        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = len(test_data) / processing_time

        print(f"Rule Engine Throughput:")
        print(f"  Claims processed: {len(test_data)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")

        # Should meet the throughput requirement
        assert throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_preprocessor_throughput(self, large_dataset):
        """Test preprocessor throughput performance."""
        from src.models.claim_models import claim_factory

        # Convert to claim objects (subset for testing)
        test_data = large_dataset[:1000]
        claim_objects = []

        for claim_data in test_data:
            try:
                claim_obj = claim_factory(claim_data)
                claim_objects.append(claim_obj)
            except Exception:
                continue  # Skip invalid claims

        preprocessor = ClaimPreprocessor()

        start_time = time.perf_counter()
        processed_df = preprocessor.preprocess_claims(claim_objects)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = len(claim_objects) / processing_time

        print(f"Preprocessor Throughput:")
        print(f"  Claims processed: {len(claim_objects)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")
        print(f"  Features generated: {len(processed_df.columns)}")

        # Preprocessing should be reasonably fast
        assert throughput > 100  # At least 100 claims/sec for preprocessing

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_feature_engineering_throughput(self, large_dataset):
        """Test feature engineering throughput performance."""
        feature_engineer = FeatureEngineer()

        # Test subset
        test_data = large_dataset[:1000]

        start_time = time.perf_counter()
        feature_set = feature_engineer.extract_features(test_data)
        combined_features = feature_engineer.combine_features(feature_set)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = len(test_data) / processing_time

        print(f"Feature Engineering Throughput:")
        print(f"  Claims processed: {len(test_data)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")
        print(f"  Features generated: {len(combined_features.columns)}")

        # Feature engineering should be reasonably fast
        assert throughput > 50  # At least 50 claims/sec for feature engineering

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_ml_prediction_throughput(self, large_dataset):
        """Test ML model prediction throughput."""
        from src.models.claim_models import claim_factory
        import pandas as pd

        # Prepare training data (small subset)
        train_data = large_dataset[:500]
        claim_objects = []

        for claim_data in train_data:
            try:
                claim_obj = claim_factory(claim_data)
                claim_objects.append(claim_obj)
            except Exception:
                continue

        # Preprocess and train
        preprocessor = ClaimPreprocessor()
        processed_df = preprocessor.preprocess_claims(claim_objects)

        ml_manager = MLModelManager()
        X = processed_df[preprocessor.feature_columns]
        y = processed_df['fraud_indicator'] if 'fraud_indicator' in processed_df.columns else pd.Series([0] * len(X))

        # Train only random forest for speed
        ml_manager.model_configs = {'random_forest': ml_manager.model_configs['random_forest']}
        ml_manager.train_models(X, y)

        # Test prediction throughput
        test_X = X.head(1000) if len(X) >= 1000 else X

        start_time = time.perf_counter()
        predictions = ml_manager.predict(test_X, 'random_forest')
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = len(test_X) / processing_time

        print(f"ML Prediction Throughput:")
        print(f"  Claims processed: {len(test_X)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")

        # ML predictions should be very fast
        assert throughput > 2000  # At least 2000 claims/sec for predictions

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_concurrent_processing_throughput(self, temp_data_files):
        """Test throughput with concurrent processing."""
        data_dir, file_paths = temp_data_files

        def process_file_batch(file_path):
            """Process a single file and return metrics."""
            loader = ClaimDataLoader(data_dir)
            rule_engine = RuleEngine()

            start_time = time.perf_counter()

            # Load file
            batch = loader.load_claims_batch(file_paths=[file_path])

            # Process claims
            results = []
            for claim in batch.claims:
                claim_dict = claim.dict()
                result, score = rule_engine.analyze_claim(claim_dict)
                results.append((result, score))

            end_time = time.perf_counter()

            return {
                'file': file_path.name,
                'claims_processed': len(results),
                'processing_time': end_time - start_time,
                'throughput': len(results) / (end_time - start_time)
            }

        # Test concurrent processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.perf_counter()
            futures = [executor.submit(process_file_batch, fp) for fp in file_paths]
            results = [future.result() for future in futures]
            end_time = time.perf_counter()

        # Calculate overall metrics
        total_claims = sum(r['claims_processed'] for r in results)
        total_time = end_time - start_time
        overall_throughput = total_claims / total_time

        print(f"Concurrent Processing Throughput:")
        print(f"  Files processed: {len(file_paths)}")
        print(f"  Total claims: {total_claims}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.1f} claims/sec")

        individual_throughputs = [r['throughput'] for r in results]
        print(f"  Average file throughput: {statistics.mean(individual_throughputs):.1f} claims/sec")

        # Concurrent processing should meet throughput requirements
        assert overall_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_streaming_throughput(self, temp_data_files):
        """Test throughput with streaming processing."""
        data_dir, file_paths = temp_data_files

        config = DataLoaderConfig(chunk_size=500)
        loader = ClaimDataLoader(data_dir, config)
        rule_engine = RuleEngine()

        total_processed = 0
        start_time = time.perf_counter()

        # Stream and process chunks
        for chunk in loader.stream_claims():
            for claim in chunk:
                claim_dict = claim.dict()
                rule_engine.analyze_claim(claim_dict)
                total_processed += 1

        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = total_processed / processing_time

        print(f"Streaming Throughput:")
        print(f"  Claims processed: {total_processed}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")

        # Streaming should maintain good throughput
        assert throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.8  # Allow 20% reduction for streaming

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_batch_size_impact_on_throughput(self, temp_data_files):
        """Test how batch size affects throughput."""
        data_dir, file_paths = temp_data_files

        batch_sizes = [100, 500, 1000, 2000]
        results = {}

        for batch_size in batch_sizes:
            config = DataLoaderConfig(batch_size=batch_size, validate_on_load=False)
            loader = ClaimDataLoader(data_dir, config)

            start_time = time.perf_counter()
            batch = loader.load_claims_batch()
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = batch.total_count / processing_time

            results[batch_size] = {
                'throughput': throughput,
                'processing_time': processing_time,
                'claims_count': batch.total_count
            }

            print(f"Batch size {batch_size}: {throughput:.1f} claims/sec")

        # Find optimal batch size
        best_batch_size = max(results.keys(), key=lambda k: results[k]['throughput'])
        best_throughput = results[best_batch_size]['throughput']

        print(f"Optimal batch size: {best_batch_size} ({best_throughput:.1f} claims/sec)")

        # Best throughput should meet requirements
        assert best_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_memory_efficient_throughput(self, large_dataset):
        """Test throughput with memory-efficient processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        rule_engine = RuleEngine()

        # Process in smaller chunks to maintain memory efficiency
        chunk_size = 200
        total_processed = 0
        start_time = time.perf_counter()

        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]

            for claim_data in chunk:
                rule_engine.analyze_claim(claim_data)
                total_processed += 1

            # Check memory usage periodically
            if i % (chunk_size * 5) == 0:  # Every 5 chunks
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory

                # Memory should not grow excessively
                assert memory_increase < 200, f"Memory usage increased by {memory_increase:.1f}MB"

        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = total_processed / processing_time
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory

        print(f"Memory-Efficient Throughput:")
        print(f"  Claims processed: {total_processed}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} claims/sec")
        print(f"  Memory increase: {total_memory_increase:.1f}MB")

        # Should maintain good throughput while being memory efficient
        assert throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.7  # Allow 30% reduction
        assert total_memory_increase < 150  # Less than 150MB increase

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_sustained_throughput(self, large_dataset):
        """Test sustained throughput over extended processing."""
        rule_engine = RuleEngine()

        # Process data in multiple waves to test sustained performance
        waves = 5
        wave_size = 1000
        wave_results = []

        for wave in range(waves):
            wave_data = large_dataset[wave * wave_size:(wave + 1) * wave_size]

            start_time = time.perf_counter()

            for claim_data in wave_data:
                rule_engine.analyze_claim(claim_data)

            end_time = time.perf_counter()

            wave_time = end_time - start_time
            wave_throughput = len(wave_data) / wave_time

            wave_results.append({
                'wave': wave + 1,
                'throughput': wave_throughput,
                'time': wave_time
            })

            print(f"Wave {wave + 1}: {wave_throughput:.1f} claims/sec")

        # Calculate statistics
        throughputs = [r['throughput'] for r in wave_results]
        mean_throughput = statistics.mean(throughputs)
        throughput_std = statistics.stdev(throughputs)
        min_throughput = min(throughputs)

        print(f"Sustained Throughput Analysis:")
        print(f"  Mean throughput: {mean_throughput:.1f} claims/sec")
        print(f"  Std deviation: {throughput_std:.1f} claims/sec")
        print(f"  Min throughput: {min_throughput:.1f} claims/sec")

        # Throughput should be sustained and not degrade significantly
        assert mean_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC
        assert min_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.8  # No more than 20% degradation
        assert throughput_std < mean_throughput * 0.2  # Standard deviation < 20% of mean

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_throughput_under_different_loads(self, large_dataset):
        """Test throughput under different system loads."""
        import threading
        import queue

        rule_engine = RuleEngine()

        def cpu_intensive_task():
            """Background CPU-intensive task to simulate load."""
            for _ in range(1000000):
                _ = sum(range(100))

        # Test scenarios
        scenarios = [
            {'name': 'No Load', 'background_threads': 0},
            {'name': 'Light Load', 'background_threads': 1},
            {'name': 'Medium Load', 'background_threads': 2},
            {'name': 'Heavy Load', 'background_threads': 4}
        ]

        results = {}

        for scenario in scenarios:
            scenario_name = scenario['name']
            num_threads = scenario['background_threads']

            # Start background load
            background_threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_intensive_task)
                thread.daemon = True
                thread.start()
                background_threads.append(thread)

            # Measure throughput under this load
            test_data = large_dataset[:1000]

            start_time = time.perf_counter()
            for claim_data in test_data:
                rule_engine.analyze_claim(claim_data)
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = len(test_data) / processing_time

            results[scenario_name] = throughput
            print(f"{scenario_name}: {throughput:.1f} claims/sec")

        # Throughput should degrade gracefully under load
        no_load_throughput = results['No Load']
        heavy_load_throughput = results['Heavy Load']

        degradation = (no_load_throughput - heavy_load_throughput) / no_load_throughput

        print(f"Throughput degradation under heavy load: {degradation * 100:.1f}%")

        # Even under heavy load, should maintain minimum throughput
        assert heavy_load_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.5  # 50% of baseline
        assert degradation < 0.7  # Less than 70% degradation

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_end_to_end_pipeline_throughput(self, temp_data_files):
        """Test end-to-end pipeline throughput."""
        data_dir, file_paths = temp_data_files

        # Set up complete pipeline
        config = DataLoaderConfig(batch_size=1000, validate_on_load=True)
        loader = ClaimDataLoader(data_dir, config)
        validator = ClaimValidator()
        preprocessor = ClaimPreprocessor()
        rule_engine = RuleEngine()

        start_time = time.perf_counter()

        # Step 1: Load data
        batch = loader.load_claims_batch()
        load_time = time.perf_counter()

        # Step 2: Preprocess claims (subset for performance)
        claims_to_process = batch.claims[:2000]
        processed_df = preprocessor.preprocess_claims(claims_to_process)
        preprocess_time = time.perf_counter()

        # Step 3: Apply rule engine (subset)
        rule_results = []
        for claim in claims_to_process[:1000]:
            claim_dict = claim.dict()
            result, score = rule_engine.analyze_claim(claim_dict)
            rule_results.append((result, score))

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        load_only_time = load_time - start_time
        preprocess_only_time = preprocess_time - load_time
        rule_only_time = end_time - preprocess_time

        overall_throughput = len(rule_results) / total_time
        load_throughput = batch.total_count / load_only_time
        preprocess_throughput = len(claims_to_process) / preprocess_only_time
        rule_throughput = len(rule_results) / rule_only_time

        print(f"End-to-End Pipeline Throughput:")
        print(f"  Overall: {overall_throughput:.1f} claims/sec")
        print(f"  Loading: {load_throughput:.1f} claims/sec")
        print(f"  Preprocessing: {preprocess_throughput:.1f} claims/sec")
        print(f"  Rule engine: {rule_throughput:.1f} claims/sec")
        print(f"  Total pipeline time: {total_time:.2f}s")

        # End-to-end pipeline should meet throughput requirements
        # (Allow lower throughput for complete pipeline)
        assert overall_throughput >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.3  # 30% of baseline

    @pytest.mark.throughput
    @pytest.mark.performance
    def test_throughput_scalability(self, large_dataset):
        """Test throughput scalability with increasing dataset sizes."""
        rule_engine = RuleEngine()

        dataset_sizes = [500, 1000, 2000, 4000]
        results = {}

        for size in dataset_sizes:
            test_data = large_dataset[:size]

            start_time = time.perf_counter()
            for claim_data in test_data:
                rule_engine.analyze_claim(claim_data)
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = size / processing_time

            results[size] = {
                'throughput': throughput,
                'processing_time': processing_time
            }

            print(f"Dataset size {size}: {throughput:.1f} claims/sec")

        # Throughput should remain relatively stable as dataset size increases
        throughputs = [results[size]['throughput'] for size in dataset_sizes]
        throughput_std = statistics.stdev(throughputs)
        mean_throughput = statistics.mean(throughputs)

        print(f"Throughput Scalability:")
        print(f"  Mean throughput: {mean_throughput:.1f} claims/sec")
        print(f"  Std deviation: {throughput_std:.1f} claims/sec")

        # Throughput should be stable across different dataset sizes
        assert throughput_std < mean_throughput * 0.3  # Less than 30% variation
        assert min(throughputs) >= BENCHMARKS.MIN_THROUGHPUT_CLAIMS_PER_SEC * 0.8  # Minimum 80% of requirement