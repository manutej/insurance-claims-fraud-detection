"""
Unit tests for the data loader module.
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import asyncio
from datetime import datetime

from src.ingestion.data_loader import (
    ClaimDataLoader, DataLoaderConfig,
    load_claims_from_directory, stream_claims_from_directory
)
from src.models.claim_models import BaseClaim, ClaimBatch
from tests.fixtures.claim_factories import ValidClaim, UpcodingFraudClaim
from tests.fixtures.mock_objects import MockValidator, MockPreprocessor
from tests.test_config import BENCHMARKS


class TestDataLoaderConfig:
    """Test the DataLoaderConfig class."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = DataLoaderConfig()

        assert config.batch_size == 1000
        assert config.max_workers == 4
        assert config.validate_on_load is True
        assert config.preprocess_on_load is False
        assert config.chunk_size == 10000
        assert config.max_file_size_mb == 500
        assert config.supported_extensions == ['.json']

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = DataLoaderConfig(
            batch_size=500,
            max_workers=8,
            validate_on_load=False,
            preprocess_on_load=True,
            chunk_size=5000,
            max_file_size_mb=1000,
            supported_extensions=['.json', '.csv']
        )

        assert config.batch_size == 500
        assert config.max_workers == 8
        assert config.validate_on_load is False
        assert config.preprocess_on_load is True
        assert config.chunk_size == 5000
        assert config.max_file_size_mb == 1000
        assert config.supported_extensions == ['.json', '.csv']


class TestClaimDataLoader:
    """Test the ClaimDataLoader class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_claims_data(self):
        """Create sample claims data for testing."""
        return [
            {
                'claim_id': 'CLM-001',
                'patient_id': 'PAT-001',
                'provider_npi': '1234567890',
                'service_date': '2024-01-15T00:00:00',
                'billed_amount': 250.0,
                'diagnosis_codes': ['M79.3'],
                'procedure_codes': ['99213'],
                'fraud_indicator': False
            },
            {
                'claim_id': 'CLM-002',
                'patient_id': 'PAT-002',
                'provider_npi': '9876543210',
                'service_date': '2024-01-16T00:00:00',
                'billed_amount': 5000.0,
                'diagnosis_codes': ['S13.4'],
                'procedure_codes': ['99285'],
                'fraud_indicator': True
            }
        ]

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator."""
        return MockValidator()

    @pytest.fixture
    def mock_preprocessor(self):
        """Create a mock preprocessor."""
        return MockPreprocessor()

    def test_loader_initialization(self, temp_data_dir):
        """Test ClaimDataLoader initialization."""
        loader = ClaimDataLoader(temp_data_dir)

        assert loader.data_directory == temp_data_dir
        assert isinstance(loader.config, DataLoaderConfig)
        assert loader.validator is not None
        assert loader.preprocessor is not None
        assert loader.stats['files_processed'] == 0
        assert loader.stats['claims_loaded'] == 0

    def test_loader_initialization_with_custom_config(self, temp_data_dir):
        """Test loader initialization with custom config."""
        config = DataLoaderConfig(batch_size=500, validate_on_load=False)
        loader = ClaimDataLoader(temp_data_dir, config)

        assert loader.config.batch_size == 500
        assert loader.config.validate_on_load is False

    def test_loader_initialization_nonexistent_directory(self):
        """Test loader initialization with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            ClaimDataLoader("/nonexistent/directory")

    def test_create_sample_data_files(self, temp_data_dir, sample_claims_data):
        """Helper method to create sample data files."""
        # Create medical claims file
        medical_file = temp_data_dir / "medical_claims.json"
        with open(medical_file, 'w') as f:
            json.dump(sample_claims_data, f)

        # Create fraud claims file
        fraud_file = temp_data_dir / "fraud_claims.json"
        with open(fraud_file, 'w') as f:
            json.dump([sample_claims_data[1]], f)

        return [medical_file, fraud_file]

    @pytest.mark.unit
    def test_load_claims_batch_success(self, temp_data_dir, sample_claims_data):
        """Test successful batch loading of claims."""
        # Create test files
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()

        assert isinstance(batch, ClaimBatch)
        assert len(batch.claims) >= 2  # At least 2 claims from our test data
        assert batch.batch_id is not None
        assert batch.processed_at is not None
        assert batch.total_count == len(batch.claims)

        # Check statistics
        assert loader.stats['files_processed'] > 0
        assert loader.stats['claims_loaded'] > 0

    def test_load_claims_batch_with_specific_files(self, temp_data_dir, sample_claims_data):
        """Test loading specific files."""
        files = self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch(file_paths=[files[0]])  # Only medical file

        assert isinstance(batch, ClaimBatch)
        assert len(batch.claims) == 2  # Both claims from medical file

    def test_load_claims_batch_nonexistent_file(self, temp_data_dir):
        """Test loading with nonexistent file."""
        loader = ClaimDataLoader(temp_data_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_claims_batch(file_paths=["/nonexistent/file.json"])

    def test_load_claims_batch_no_files(self, temp_data_dir):
        """Test loading when no files are found."""
        loader = ClaimDataLoader(temp_data_dir)

        with pytest.raises(ValueError) as excinfo:
            loader.load_claims_batch()

        assert "No valid claim files found" in str(excinfo.value)

    def test_load_single_file_success(self, temp_data_dir, sample_claims_data):
        """Test loading a single file."""
        test_file = temp_data_dir / "test_claims.json"
        with open(test_file, 'w') as f:
            json.dump(sample_claims_data, f)

        loader = ClaimDataLoader(temp_data_dir)
        claims = loader._load_single_file(test_file)

        assert len(claims) == 2
        assert all(isinstance(claim, BaseClaim) for claim in claims)

    def test_load_single_file_with_claims_wrapper(self, temp_data_dir, sample_claims_data):
        """Test loading file with claims wrapper structure."""
        test_file = temp_data_dir / "wrapped_claims.json"
        with open(test_file, 'w') as f:
            json.dump({"claims": sample_claims_data}, f)

        loader = ClaimDataLoader(temp_data_dir)
        claims = loader._load_single_file(test_file)

        assert len(claims) == 2

    def test_load_single_file_malformed_json(self, temp_data_dir):
        """Test loading file with malformed JSON."""
        test_file = temp_data_dir / "malformed.json"
        with open(test_file, 'w') as f:
            f.write("{ invalid json")

        loader = ClaimDataLoader(temp_data_dir)

        with pytest.raises(json.JSONDecodeError):
            loader._load_single_file(test_file)

    def test_discover_claim_files(self, temp_data_dir, sample_claims_data):
        """Test file discovery functionality."""
        # Create various test files
        medical_file = temp_data_dir / "medical_claims.json"
        pharmacy_file = temp_data_dir / "pharmacy_data.json"
        fraud_file = temp_data_dir / "fraud_claims.json"
        non_json = temp_data_dir / "readme.txt"

        for file in [medical_file, pharmacy_file, fraud_file]:
            with open(file, 'w') as f:
                json.dump(sample_claims_data, f)

        with open(non_json, 'w') as f:
            f.write("Not a JSON file")

        loader = ClaimDataLoader(temp_data_dir)
        discovered_files = loader._discover_claim_files()

        # Should find JSON files but not text file
        assert len(discovered_files) == 3
        assert all(file.suffix == '.json' for file in discovered_files)

    def test_discover_claim_files_with_type_filter(self, temp_data_dir, sample_claims_data):
        """Test file discovery with claim type filter."""
        # Create various test files
        medical_file = temp_data_dir / "medical_claims.json"
        pharmacy_file = temp_data_dir / "pharmacy_data.json"

        for file in [medical_file, pharmacy_file]:
            with open(file, 'w') as f:
                json.dump(sample_claims_data, f)

        loader = ClaimDataLoader(temp_data_dir)

        # Filter for medical claims only
        medical_files = loader._discover_claim_files(claim_types=['medical'])
        assert len(medical_files) == 1
        assert 'medical' in medical_files[0].name

        # Filter for pharmacy claims only
        pharmacy_files = loader._discover_claim_files(claim_types=['pharmacy'])
        assert len(pharmacy_files) == 1
        assert 'pharmacy' in pharmacy_files[0].name

    def test_stream_claims_success(self, temp_data_dir, sample_claims_data):
        """Test claim streaming functionality."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        chunks = list(loader.stream_claims(chunk_size=1))

        assert len(chunks) > 0
        # With chunk_size=1, each chunk should have 1 claim
        assert all(len(chunk) <= 1 for chunk in chunks)

    def test_stream_claims_large_chunks(self, temp_data_dir, sample_claims_data):
        """Test streaming with large chunk size."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        chunks = list(loader.stream_claims(chunk_size=10))

        # Should have fewer chunks with larger chunk size
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_claims_async(self, temp_data_dir, sample_claims_data):
        """Test async claim streaming."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        chunks = []

        async for chunk in loader.stream_claims_async(chunk_size=1):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(len(chunk) <= 1 for chunk in chunks)

    def test_load_specific_claim_type(self, temp_data_dir, sample_claims_data):
        """Test loading specific claim types."""
        # Create files with specific naming
        medical_file = temp_data_dir / "medical_claims.json"
        pharmacy_file = temp_data_dir / "pharmacy_claims.json"

        with open(medical_file, 'w') as f:
            json.dump(sample_claims_data, f)

        with open(pharmacy_file, 'w') as f:
            json.dump(sample_claims_data, f)

        loader = ClaimDataLoader(temp_data_dir)

        # Load medical claims
        medical_claims = loader.load_specific_claim_type('medical')
        assert len(medical_claims) > 0

        # Load pharmacy claims
        pharmacy_claims = loader.load_specific_claim_type('pharmacy')
        assert len(pharmacy_claims) > 0

    def test_load_specific_claim_type_invalid(self, temp_data_dir):
        """Test loading invalid claim type."""
        loader = ClaimDataLoader(temp_data_dir)

        with pytest.raises(ValueError) as excinfo:
            loader.load_specific_claim_type('invalid_type')

        assert "Unsupported claim type" in str(excinfo.value)

    def test_load_specific_claim_type_no_files(self, temp_data_dir):
        """Test loading specific type when no files exist."""
        loader = ClaimDataLoader(temp_data_dir)

        claims = loader.load_specific_claim_type('medical')
        assert claims == []

    def test_get_statistics(self, temp_data_dir, sample_claims_data):
        """Test statistics collection."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        loader.load_claims_batch()

        stats = loader.get_statistics()

        assert 'files_processed' in stats
        assert 'claims_loaded' in stats
        assert 'validation_errors' in stats
        assert 'processing_time_seconds' in stats
        assert 'avg_claims_per_file' in stats
        assert 'claims_per_second' in stats

        assert stats['files_processed'] > 0
        assert stats['claims_loaded'] > 0
        assert stats['processing_time_seconds'] > 0

    def test_reset_statistics(self, temp_data_dir):
        """Test statistics reset."""
        loader = ClaimDataLoader(temp_data_dir)

        # Manually set some stats
        loader.stats['files_processed'] = 5
        loader.stats['claims_loaded'] = 100

        loader.reset_statistics()

        assert loader.stats['files_processed'] == 0
        assert loader.stats['claims_loaded'] == 0
        assert loader.stats['validation_errors'] == 0
        assert loader.stats['processing_time'] == 0.0

    def test_get_file_summary(self, temp_data_dir, sample_claims_data):
        """Test file summary generation."""
        # Create various test files
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        loader = ClaimDataLoader(temp_data_dir)
        summary = loader.get_file_summary()

        assert 'total_files' in summary
        assert 'files_by_type' in summary
        assert 'total_size_mb' in summary
        assert 'file_details' in summary

        assert summary['total_files'] > 0
        assert isinstance(summary['files_by_type'], dict)
        assert summary['total_size_mb'] >= 0

        # Check file details
        for file_detail in summary['file_details']:
            assert 'path' in file_detail
            assert 'name' in file_detail
            assert 'type' in file_detail
            assert 'size_mb' in file_detail
            assert 'modified' in file_detail

    def test_validation_during_load(self, temp_data_dir, sample_claims_data, mock_validator):
        """Test validation during loading."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        config = DataLoaderConfig(validate_on_load=True)
        loader = ClaimDataLoader(temp_data_dir, config, validator=mock_validator)

        batch = loader.load_claims_batch()

        # Check that validation was called
        assert len(mock_validator.validate_calls) > 0
        assert isinstance(batch, ClaimBatch)

    def test_no_validation_during_load(self, temp_data_dir, sample_claims_data, mock_validator):
        """Test no validation during loading."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        config = DataLoaderConfig(validate_on_load=False)
        loader = ClaimDataLoader(temp_data_dir, config, validator=mock_validator)

        batch = loader.load_claims_batch()

        # Check that validation was not called
        assert len(mock_validator.validate_calls) == 0
        assert isinstance(batch, ClaimBatch)

    def test_progress_callback(self, temp_data_dir, sample_claims_data):
        """Test progress callback functionality."""
        self.test_create_sample_data_files(temp_data_dir, sample_claims_data)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        loader = ClaimDataLoader(temp_data_dir)
        loader.load_claims_batch(progress_callback=progress_callback)

        # Check that progress callback was called
        assert len(progress_calls) > 0
        # Last call should have current == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_file_size_filtering(self, temp_data_dir):
        """Test filtering files by size."""
        # Create a small config with very small max file size
        config = DataLoaderConfig(max_file_size_mb=0.001)  # Very small limit

        # Create a file larger than the limit
        large_file = temp_data_dir / "large_file.json"
        large_data = [ValidClaim() for _ in range(1000)]  # Should be > 0.001MB
        with open(large_file, 'w') as f:
            json.dump(large_data, f)

        loader = ClaimDataLoader(temp_data_dir, config)
        discovered_files = loader._discover_claim_files()

        # Large file should be filtered out
        assert len(discovered_files) == 0

    @pytest.mark.performance
    def test_batch_loading_performance(self, temp_data_dir):
        """Test batch loading performance."""
        import time

        # Create multiple files with moderate data
        for i in range(5):
            file_path = temp_data_dir / f"claims_{i}.json"
            claims_data = [ValidClaim() for _ in range(100)]  # 100 claims per file
            with open(file_path, 'w') as f:
                json.dump(claims_data, f)

        loader = ClaimDataLoader(temp_data_dir)

        start_time = time.time()
        batch = loader.load_claims_batch()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 10  # 10 seconds for 500 claims across 5 files
        assert len(batch.claims) == 500

        # Check performance metrics
        stats = loader.get_statistics()
        assert stats['claims_per_second'] > 50  # At least 50 claims/second

    def test_concurrent_file_processing(self, temp_data_dir):
        """Test concurrent file processing with ThreadPoolExecutor."""
        # Create multiple files
        for i in range(4):
            file_path = temp_data_dir / f"claims_{i}.json"
            claims_data = [ValidClaim() for _ in range(50)]
            with open(file_path, 'w') as f:
                json.dump(claims_data, f)

        config = DataLoaderConfig(max_workers=2)
        loader = ClaimDataLoader(temp_data_dir, config)

        batch = loader.load_claims_batch()

        assert len(batch.claims) == 200  # 4 files × 50 claims
        assert loader.stats['files_processed'] == 4

    def test_error_handling_corrupted_file(self, temp_data_dir):
        """Test error handling for corrupted files."""
        # Create good file
        good_file = temp_data_dir / "good_claims.json"
        with open(good_file, 'w') as f:
            json.dump([ValidClaim()], f)

        # Create corrupted file
        bad_file = temp_data_dir / "bad_claims.json"
        with open(bad_file, 'w') as f:
            f.write("{ invalid json content")

        loader = ClaimDataLoader(temp_data_dir)

        # Should still load good file despite bad file
        batch = loader.load_claims_batch()

        # Should have claims from good file
        assert len(batch.claims) >= 1
        # Should have processed only the good file
        assert loader.stats['files_processed'] == 1

    def test_edge_case_empty_files(self, temp_data_dir):
        """Test handling of empty files."""
        # Create empty file
        empty_file = temp_data_dir / "empty.json"
        with open(empty_file, 'w') as f:
            json.dump([], f)

        # Create file with claims
        claims_file = temp_data_dir / "claims.json"
        with open(claims_file, 'w') as f:
            json.dump([ValidClaim()], f)

        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()

        # Should handle empty file gracefully
        assert len(batch.claims) >= 1
        assert loader.stats['files_processed'] == 2

    def test_edge_case_unexpected_file_structure(self, temp_data_dir):
        """Test handling of unexpected file structures."""
        # Create file with unexpected structure
        weird_file = temp_data_dir / "weird.json"
        with open(weird_file, 'w') as f:
            json.dump({"unexpected": "structure"}, f)

        loader = ClaimDataLoader(temp_data_dir)
        batch = loader.load_claims_batch()

        # Should handle gracefully and return empty batch
        assert len(batch.claims) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_load_claims_from_directory(self, temp_data_dir):
        """Test convenience function for loading claims."""
        # Create test file
        test_file = temp_data_dir / "test_claims.json"
        with open(test_file, 'w') as f:
            json.dump([ValidClaim()], f)

        batch = load_claims_from_directory(temp_data_dir)

        assert isinstance(batch, ClaimBatch)
        assert len(batch.claims) >= 1

    def test_stream_claims_from_directory(self, temp_data_dir):
        """Test convenience function for streaming claims."""
        # Create test file
        test_file = temp_data_dir / "test_claims.json"
        with open(test_file, 'w') as f:
            json.dump([ValidClaim(), ValidClaim()], f)

        chunks = list(stream_claims_from_directory(temp_data_dir, chunk_size=1))

        assert len(chunks) >= 2
        assert all(len(chunk) <= 1 for chunk in chunks)

    def test_load_claims_with_validation_disabled(self, temp_data_dir):
        """Test loading with validation disabled."""
        # Create test file
        test_file = temp_data_dir / "test_claims.json"
        with open(test_file, 'w') as f:
            json.dump([ValidClaim()], f)

        batch = load_claims_from_directory(temp_data_dir, validate=False)

        assert isinstance(batch, ClaimBatch)
        assert len(batch.claims) >= 1

    def test_load_claims_with_type_filter(self, temp_data_dir):
        """Test loading with claim type filter."""
        # Create medical claims file
        medical_file = temp_data_dir / "medical_claims.json"
        with open(medical_file, 'w') as f:
            json.dump([ValidClaim()], f)

        # Create fraud claims file
        fraud_file = temp_data_dir / "fraud_claims.json"
        with open(fraud_file, 'w') as f:
            json.dump([UpcodingFraudClaim()], f)

        # Load only medical claims
        batch = load_claims_from_directory(temp_data_dir, claim_types=['medical'])

        assert isinstance(batch, ClaimBatch)
        assert len(batch.claims) >= 1