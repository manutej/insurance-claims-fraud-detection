"""
Test configuration module containing performance benchmarks and testing constants.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PerformanceBenchmarks:
    """Performance benchmark thresholds for fraud detection system."""

    # Accuracy Requirements
    MIN_ACCURACY: float = 0.94  # >94% accuracy requirement
    MAX_FALSE_POSITIVE_RATE: float = 0.038  # <3.8% false positive rate
    MIN_DETECTION_RATE: float = 0.08  # Minimum 8% fraud detection rate
    MAX_DETECTION_RATE: float = 0.15  # Maximum 15% fraud detection rate

    # Latency Requirements
    MAX_SINGLE_CLAIM_LATENCY_MS: float = 100.0  # <100ms for single claim
    MAX_BATCH_PROCESSING_HOURS: float = 4.0  # <4 hours per batch

    # Throughput Requirements
    MIN_THROUGHPUT_CLAIMS_PER_SEC: int = 1000  # 1000 claims/sec minimum

    # Test Coverage Requirements
    MIN_TEST_COVERAGE: float = 0.80  # 80% test coverage minimum

    # Memory and Resource Limits
    MAX_MEMORY_USAGE_MB: int = 512  # Maximum memory per process
    MAX_CPU_USAGE_PERCENT: float = 80.0  # Maximum CPU usage

    # Data Volume Limits for Testing
    SMALL_DATASET_SIZE: int = 100
    MEDIUM_DATASET_SIZE: int = 1000
    LARGE_DATASET_SIZE: int = 10000
    STRESS_DATASET_SIZE: int = 100000


@dataclass
class TestConfiguration:
    """Test configuration settings."""

    # Test Data Paths
    TEST_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "fixtures", "data")
    VALID_CLAIMS_PATH: str = os.path.join(TEST_DATA_DIR, "valid_claims.json")
    FRAUD_CLAIMS_PATH: str = os.path.join(TEST_DATA_DIR, "fraud_claims.json")
    MIXED_CLAIMS_PATH: str = os.path.join(TEST_DATA_DIR, "mixed_claims.json")

    # Test Database Configuration
    TEST_DB_URL: str = "sqlite:///:memory:"

    # Mock Service URLs
    MOCK_API_BASE_URL: str = "http://localhost:8080"

    # Test Timeouts
    DEFAULT_TIMEOUT_SECONDS: int = 30
    PERFORMANCE_TEST_TIMEOUT_SECONDS: int = 300

    # Random Seed for Reproducible Tests
    RANDOM_SEED: int = 42

    # Feature Engineering Test Parameters
    FEATURE_COLUMNS: list = None

    def __post_init__(self):
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                "billed_amount",
                "diagnosis_codes",
                "procedure_codes",
                "provider_npi",
                "patient_age",
                "days_between_service_and_claim",
            ]


# Global instances
BENCHMARKS = PerformanceBenchmarks()
CONFIG = TestConfiguration()

# Test Categories
TEST_CATEGORIES = {
    "unit": {
        "description": "Fast, isolated tests for individual functions/classes",
        "timeout": 5,
        "parallelizable": True,
    },
    "integration": {
        "description": "Tests for component interactions and data flow",
        "timeout": 30,
        "parallelizable": False,
    },
    "performance": {
        "description": "Latency, throughput, and resource usage tests",
        "timeout": 300,
        "parallelizable": False,
    },
    "accuracy": {
        "description": "Model accuracy and fraud detection rate validation",
        "timeout": 60,
        "parallelizable": True,
    },
    "security": {
        "description": "Security vulnerability and data privacy tests",
        "timeout": 30,
        "parallelizable": True,
    },
}

# Test Data Generation Configuration
TEST_DATA_CONFIG = {
    "claim_id_pattern": "CLM-{year}-{sequence:06d}",
    "patient_id_pattern": "PAT-{sequence:08d}",
    "provider_npi_range": (1000000000, 9999999999),
    "billed_amount_range": (50.0, 50000.0),
    "fraud_rate_in_test_data": 0.5,  # Higher than real-world for balanced testing
    "date_range_days": 365,
    "diagnosis_codes": [
        "M79.3",
        "S13.4",
        "M54.2",
        "G44.1",
        "M25.5",  # Common injury codes
        "Z51.11",
        "M17.0",
        "I25.10",
        "E11.9",
        "F32.9",  # Common chronic conditions
    ],
    "procedure_codes": [
        "99213",
        "99214",
        "99215",
        "73721",
        "97110",  # Common procedures
        "99283",
        "99284",
        "99285",
        "70553",
        "99281",  # Emergency procedures
    ],
}

# Error Tolerance Configuration
ERROR_TOLERANCE = {
    "floating_point_precision": 1e-6,
    "percentage_tolerance": 0.01,  # 1% tolerance for percentage calculations
    "timing_variance_percent": 0.2,  # 20% variance allowed in timing tests
}


def get_test_environment_info() -> Dict[str, Any]:
    """Get information about the test environment."""
    return {
        "python_version": os.sys.version,
        "platform": os.sys.platform,
        "cpu_count": os.cpu_count(),
        "test_data_dir": CONFIG.TEST_DATA_DIR,
        "benchmarks": BENCHMARKS.__dict__,
        "random_seed": CONFIG.RANDOM_SEED,
    }


def is_performance_test_enabled() -> bool:
    """Check if performance tests should be run."""
    return os.getenv("RUN_PERFORMANCE_TESTS", "false").lower() == "true"


def is_slow_test_enabled() -> bool:
    """Check if slow tests should be run."""
    return os.getenv("RUN_SLOW_TESTS", "false").lower() == "true"


def get_parallel_worker_count() -> int:
    """Get the number of parallel workers for tests."""
    return int(os.getenv("PYTEST_WORKERS", "4"))
