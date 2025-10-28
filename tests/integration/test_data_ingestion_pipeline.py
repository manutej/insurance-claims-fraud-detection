"""
Integration tests for the data ingestion pipeline.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import asyncio

from src.ingestion.data_loader import ClaimDataLoader, DataLoaderConfig
from src.ingestion.validator import ClaimValidator
from src.ingestion.preprocessor import ClaimPreprocessor
from src.models.claim_models import claim_factory, ClaimBatch

from tests.fixtures.claim_factories import generate_mixed_claims_batch
from tests.test_config import BENCHMARKS


class TestDataIngestionPipeline:
    """Test the complete data ingestion pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with various test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create different types of claim files
            medical_claims = generate_mixed_claims_batch(total_claims=200, fraud_rate=0.1)
            pharmacy_claims = []
            for i in range(50):
                claim = {
                    "claim_id": f"CLM-PHARM-{i:03d}",
                    "patient_id": f"PAT-PHARM-{i:03d}",
                    "provider_id": f"PRV-PHARM-{i % 5:02d}",
                    "provider_npi": "1111111111",
                    "date_of_service": "2024-01-15",
                    "ndc_code": "12345-6789-01",
                    "quantity": 30,
                    "days_supply": 30,
                    "billed_amount": 75.0,
                    "claim_type": "pharmacy",
                    "fraud_indicator": i % 10 == 0,  # 10% fraud rate
                }
                pharmacy_claims.append(claim)

            # Save different file types
            (data_dir / "medical_claims.json").write_text(json.dumps(medical_claims))
            (data_dir / "pharmacy_claims.json").write_text(json.dumps(pharmacy_claims))
            (data_dir / "valid_claims" / "legitimate.json").parent.mkdir()
            (data_dir / "valid_claims" / "legitimate.json").write_text(
                json.dumps([c for c in medical_claims if not c.get("fraud_indicator", False)])
            )
            (data_dir / "fraudulent_claims" / "fraud.json").parent.mkdir()
            (data_dir / "fraudulent_claims" / "fraud.json").write_text(
                json.dumps([c for c in medical_claims if c.get("fraud_indicator", False)])
            )

            # Create large file for performance testing
            large_claims = generate_mixed_claims_batch(total_claims=1000, fraud_rate=0.15)
            (data_dir / "large_batch.json").write_text(json.dumps(large_claims))

            # Create problematic files
            (data_dir / "empty.json").write_text("[]")
            (data_dir / "malformed.json").write_text("{ invalid json")
            (data_dir / "README.txt").write_text("Not a JSON file")

            yield data_dir

    @pytest.mark.integration
    def test_complete_ingestion_pipeline(self, temp_data_dir):
        """Test complete data ingestion from files to processed claims."""
        # Configure ingestion pipeline
        config = DataLoaderConfig(
            batch_size=100, validate_on_load=True, preprocess_on_load=False, max_workers=2
        )

        validator = ClaimValidator()
        preprocessor = ClaimPreprocessor()
        loader = ClaimDataLoader(temp_data_dir, config, validator, preprocessor)

        # Step 1: Data Discovery
        file_summary = loader.get_file_summary()
        assert file_summary["total_files"] > 0
        assert "medical" in file_summary["files_by_type"]
        assert "pharmacy" in file_summary["files_by_type"]
        print(f"✓ Discovered {file_summary['total_files']} files")

        # Step 2: Batch Loading
        batch = loader.load_claims_batch()
        assert isinstance(batch, ClaimBatch)
        assert batch.total_count > 0
        assert len(batch.claims) == batch.total_count
        print(f"✓ Loaded {batch.total_count} claims in batch")

        # Step 3: Validation Results
        stats = loader.get_statistics()
        assert stats["files_processed"] > 0
        assert stats["claims_loaded"] > 0
        print(
            f"✓ Processing stats: {stats['files_processed']} files, "
            f"{stats['claims_loaded']} claims, {stats['validation_errors']} errors"
        )

        # Step 4: Data Preprocessing
        processed_df = preprocessor.preprocess_claims(batch.claims)
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == batch.total_count
        assert len(preprocessor.feature_columns) > 0
        print(
            f"✓ Preprocessed data: {len(processed_df)} rows, "
            f"{len(preprocessor.feature_columns)} features"
        )

        # Verify data quality
        assert processed_df["claim_id"].nunique() == len(processed_df)  # No duplicates
        assert not processed_df["claim_id"].isnull().any()  # No missing IDs

    @pytest.mark.integration
    def test_streaming_ingestion_pipeline(self, temp_data_dir):
        """Test streaming data ingestion pipeline."""
        config = DataLoaderConfig(chunk_size=50)
        loader = ClaimDataLoader(temp_data_dir, config)
        preprocessor = ClaimPreprocessor()

        total_processed = 0
        chunk_count = 0

        # Stream claims in chunks
        for chunk in loader.stream_claims(chunk_size=50):
            assert len(chunk) <= 50
            chunk_count += 1
            total_processed += len(chunk)

            # Process each chunk
            if len(chunk) > 0:
                processed_df = preprocessor.preprocess_claims(chunk)
                assert len(processed_df) == len(chunk)

        assert total_processed > 0
        assert chunk_count > 0
        print(f"✓ Streamed {total_processed} claims in {chunk_count} chunks")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_ingestion(self, temp_data_dir):
        """Test async streaming data ingestion."""
        config = DataLoaderConfig(chunk_size=100)
        loader = ClaimDataLoader(temp_data_dir, config)

        total_processed = 0
        chunk_count = 0

        # Async stream claims
        async for chunk in loader.stream_claims_async(chunk_size=100):
            assert len(chunk) <= 100
            chunk_count += 1
            total_processed += len(chunk)

        assert total_processed > 0
        assert chunk_count > 0
        print(f"✓ Async streamed {total_processed} claims in {chunk_count} chunks")

    @pytest.mark.integration
    def test_selective_data_ingestion(self, temp_data_dir):
        """Test ingestion of specific claim types and files."""
        loader = ClaimDataLoader(temp_data_dir)

        # Load only medical claims
        medical_claims = loader.load_specific_claim_type("medical")
        assert len(medical_claims) > 0
        assert all(hasattr(claim, "diagnosis_codes") for claim in medical_claims)
        print(f"✓ Loaded {len(medical_claims)} medical claims")

        # Load only pharmacy claims
        pharmacy_claims = loader.load_specific_claim_type("pharmacy")
        assert len(pharmacy_claims) > 0
        print(f"✓ Loaded {len(pharmacy_claims)} pharmacy claims")

        # Load from specific subdirectory
        valid_claims = loader.load_specific_claim_type("valid", subdirectory="valid_claims")
        assert len(valid_claims) > 0
        print(f"✓ Loaded {len(valid_claims)} valid claims from subdirectory")

        # Load specific files
        medical_file = temp_data_dir / "medical_claims.json"
        batch = loader.load_claims_batch(file_paths=[medical_file])
        assert batch.total_count > 0
        print(f"✓ Loaded {batch.total_count} claims from specific file")

    @pytest.mark.integration
    def test_validation_integration(self, temp_data_dir):
        """Test integration between loading and validation."""
        validator = ClaimValidator()
        config = DataLoaderConfig(validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config, validator=validator)

        # Load with validation enabled
        batch = loader.load_claims_batch()
        stats = loader.get_statistics()

        assert batch.total_count > 0
        print(f"✓ Validation integration: {stats['validation_errors']} errors found")

        # Test validation results
        if stats["validation_errors"] > 0:
            print(f"  - Found {stats['validation_errors']} validation errors as expected")

        # Test without validation
        config_no_validation = DataLoaderConfig(validate_on_load=False)
        loader_no_validation = ClaimDataLoader(temp_data_dir, config_no_validation)
        batch_no_validation = loader_no_validation.load_claims_batch()

        assert batch_no_validation.total_count >= batch.total_count
        print(f"✓ No validation: {batch_no_validation.total_count} claims loaded")

    @pytest.mark.integration
    def test_preprocessing_integration(self, temp_data_dir):
        """Test integration between loading and preprocessing."""
        config = DataLoaderConfig(preprocess_on_load=False)  # We'll preprocess manually
        loader = ClaimDataLoader(temp_data_dir, config)
        preprocessor = ClaimPreprocessor()

        # Load claims
        batch = loader.load_claims_batch()

        # Test different preprocessing configurations
        configs = [
            {"normalize_amounts": True, "encoding_strategy": "onehot"},
            {"normalize_amounts": False, "encoding_strategy": "label"},
            {"extract_temporal_features": False, "handle_missing_data": False},
        ]

        for i, preprocess_config in enumerate(configs):
            test_preprocessor = ClaimPreprocessor(preprocess_config)
            processed_df = test_preprocessor.preprocess_claims(batch.claims[:50])  # Test on subset

            assert len(processed_df) == 50
            assert len(test_preprocessor.feature_columns) > 0
            print(
                f"✓ Preprocessing config {i+1}: {len(test_preprocessor.feature_columns)} features"
            )

        # Test transform new data
        new_claims = batch.claims[50:60]
        transformed_df = preprocessor.transform_new_data(new_claims)
        assert len(transformed_df) == 10
        print(f"✓ Transformed {len(transformed_df)} new claims")

    @pytest.mark.integration
    def test_error_handling_and_recovery(self, temp_data_dir):
        """Test error handling and recovery in ingestion pipeline."""
        loader = ClaimDataLoader(temp_data_dir)

        # Test with malformed file included
        try:
            batch = loader.load_claims_batch()
            # Should succeed despite malformed files
            assert batch.total_count > 0
            print(f"✓ Pipeline recovered from errors, loaded {batch.total_count} claims")
        except Exception as e:
            pytest.fail(f"Pipeline should handle malformed files gracefully: {e}")

        # Test with nonexistent file
        try:
            nonexistent_file = temp_data_dir / "nonexistent.json"
            loader.load_claims_batch(file_paths=[nonexistent_file])
            pytest.fail("Should raise FileNotFoundError for nonexistent file")
        except FileNotFoundError:
            print("✓ Correctly raised error for nonexistent file")

        # Test with empty directory
        empty_dir = temp_data_dir / "empty_dir"
        empty_dir.mkdir()
        empty_loader = ClaimDataLoader(empty_dir)

        try:
            empty_loader.load_claims_batch()
            pytest.fail("Should raise ValueError for empty directory")
        except ValueError as e:
            assert "No valid claim files found" in str(e)
            print("✓ Correctly handled empty directory")

    @pytest.mark.integration
    def test_performance_benchmarks(self, temp_data_dir):
        """Test ingestion pipeline performance benchmarks."""
        import time

        config = DataLoaderConfig(batch_size=500, max_workers=4, validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config)

        # Test batch loading performance
        start_time = time.time()
        batch = loader.load_claims_batch()
        end_time = time.time()

        processing_time = end_time - start_time
        stats = loader.get_statistics()

        claims_per_second = batch.total_count / processing_time
        print(f"✓ Performance benchmarks:")
        print(f"  - Total claims: {batch.total_count}")
        print(f"  - Processing time: {processing_time:.2f}s")
        print(f"  - Throughput: {claims_per_second:.1f} claims/sec")
        print(f"  - Files processed: {stats['files_processed']}")

        # Performance assertions
        assert claims_per_second > 100  # At least 100 claims/sec
        assert processing_time < 30  # Complete within 30 seconds

        # Test streaming performance
        start_time = time.time()
        total_streamed = 0

        for chunk in loader.stream_claims(chunk_size=200):
            total_streamed += len(chunk)

        stream_time = time.time() - start_time
        stream_rate = total_streamed / stream_time

        print(f"  - Streaming rate: {stream_rate:.1f} claims/sec")
        assert stream_rate > 50  # At least 50 claims/sec for streaming

    @pytest.mark.integration
    def test_concurrent_ingestion(self, temp_data_dir):
        """Test concurrent data ingestion."""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        def load_data(file_pattern):
            """Load data in separate thread."""
            thread_id = threading.current_thread().ident
            try:
                loader = ClaimDataLoader(temp_data_dir)
                if file_pattern == "medical":
                    claims = loader.load_specific_claim_type("medical")
                elif file_pattern == "pharmacy":
                    claims = loader.load_specific_claim_type("pharmacy")
                else:
                    batch = loader.load_claims_batch()
                    claims = batch.claims

                return {
                    "thread_id": thread_id,
                    "pattern": file_pattern,
                    "count": len(claims),
                    "success": True,
                }
            except Exception as e:
                return {
                    "thread_id": thread_id,
                    "pattern": file_pattern,
                    "error": str(e),
                    "success": False,
                }

        # Test concurrent loading
        patterns = ["medical", "pharmacy", "all"]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(load_data, pattern) for pattern in patterns]
            results = [future.result() for future in futures]

        # Verify all succeeded
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        assert len(successful_results) == len(patterns), f"Failed results: {failed_results}"
        print(f"✓ Concurrent ingestion: {len(successful_results)} threads successful")

        for result in successful_results:
            print(f"  - Thread {result['thread_id']}: {result['count']} {result['pattern']} claims")

    @pytest.mark.integration
    def test_data_quality_validation(self, temp_data_dir):
        """Test data quality validation across ingestion pipeline."""
        validator = ClaimValidator()
        config = DataLoaderConfig(validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config, validator=validator)

        # Load and validate
        batch = loader.load_claims_batch()

        # Check for duplicates
        claim_ids = [claim.claim_id for claim in batch.claims]
        unique_ids = set(claim_ids)
        assert len(claim_ids) == len(unique_ids), "Found duplicate claim IDs"

        # Validate all claims can be converted to dictionaries
        valid_claims = 0
        for claim in batch.claims:
            try:
                claim_dict = claim.dict()
                assert "claim_id" in claim_dict
                assert "billed_amount" in claim_dict
                valid_claims += 1
            except Exception as e:
                print(f"Warning: Invalid claim {claim.claim_id}: {e}")

        print(f"✓ Data quality: {valid_claims}/{len(batch.claims)} claims valid")
        assert valid_claims == len(batch.claims), "All loaded claims should be valid"

        # Test batch validation
        claims_data = [claim.dict() for claim in batch.claims[:100]]  # Test subset
        validation_result = validator.validate_batch(claims_data)

        assert validation_result.processed_count > 0
        print(
            f"✓ Batch validation: {validation_result.processed_count} processed, "
            f"{validation_result.error_count} errors, "
            f"{validation_result.warnings_count} warnings"
        )

    @pytest.mark.integration
    def test_memory_efficient_ingestion(self, temp_data_dir):
        """Test memory-efficient ingestion of large datasets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Use streaming for large dataset
        config = DataLoaderConfig(chunk_size=100)
        loader = ClaimDataLoader(temp_data_dir, config)
        preprocessor = ClaimPreprocessor()

        total_processed = 0
        max_memory = initial_memory

        # Stream and process in chunks
        for chunk in loader.stream_claims():
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)

            # Process chunk
            if len(chunk) > 0:
                # Create new preprocessor for each chunk to avoid memory accumulation
                chunk_preprocessor = ClaimPreprocessor()
                processed_df = chunk_preprocessor.preprocess_claims(chunk)
                total_processed += len(processed_df)

        memory_increase = max_memory - initial_memory
        memory_per_claim = memory_increase / total_processed if total_processed > 0 else 0

        print(f"✓ Memory efficiency:")
        print(f"  - Initial memory: {initial_memory:.1f} MB")
        print(f"  - Peak memory: {max_memory:.1f} MB")
        print(f"  - Memory increase: {memory_increase:.1f} MB")
        print(f"  - Claims processed: {total_processed}")
        print(f"  - Memory per claim: {memory_per_claim:.3f} MB/claim")

        # Memory efficiency assertions
        assert memory_increase < 300  # Less than 300MB increase
        assert memory_per_claim < 0.2  # Less than 0.2MB per claim

    @pytest.mark.integration
    def test_ingestion_with_progress_tracking(self, temp_data_dir):
        """Test ingestion pipeline with progress tracking."""
        loader = ClaimDataLoader(temp_data_dir)

        progress_updates = []

        def progress_callback(current, total):
            progress_updates.append((current, total))
            print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

        # Load with progress tracking
        batch = loader.load_claims_batch(progress_callback=progress_callback)

        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == progress_updates[-1][1]  # Should end at 100%
        print(
            f"✓ Progress tracking: {len(progress_updates)} updates, "
            f"final: {progress_updates[-1]}"
        )

    @pytest.mark.integration
    def test_ingestion_statistics_and_monitoring(self, temp_data_dir):
        """Test statistics collection and monitoring capabilities."""
        validator = ClaimValidator()
        preprocessor = ClaimPreprocessor()
        config = DataLoaderConfig(validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config, validator, preprocessor)

        # Reset statistics
        loader.reset_statistics()
        initial_stats = loader.get_statistics()
        assert initial_stats["claims_loaded"] == 0

        # Load data and collect statistics
        batch = loader.load_claims_batch()
        final_stats = loader.get_statistics()

        # Verify statistics
        assert final_stats["files_processed"] > 0
        assert final_stats["claims_loaded"] > 0
        assert final_stats["processing_time_seconds"] > 0
        assert final_stats["claims_per_second"] > 0
        assert final_stats["avg_claims_per_file"] > 0

        print(f"✓ Statistics collection:")
        for key, value in final_stats.items():
            print(f"  - {key}: {value}")

        # Test file summary
        file_summary = loader.get_file_summary()
        assert file_summary["total_files"] > 0
        assert "files_by_type" in file_summary
        assert "total_size_mb" in file_summary
        assert len(file_summary["file_details"]) > 0

        print(
            f"✓ File summary: {file_summary['total_files']} files, "
            f"{file_summary['total_size_mb']:.2f} MB total"
        )
