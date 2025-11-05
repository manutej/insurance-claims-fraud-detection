"""
Mock objects and fixtures for testing fraud detection components.
"""

import json
import tempfile
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, return_data: Optional[List[Dict]] = None):
        self.return_data = return_data or []
        self.load_calls = []

    def load_claims(self, file_path: str) -> List[Dict[str, Any]]:
        """Mock load_claims method."""
        self.load_calls.append(file_path)
        return self.return_data

    def load_batch(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Mock load_batch method."""
        return self.return_data[:batch_size]


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, validation_results: Optional[Dict] = None):
        self.validation_results = validation_results or {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }
        self.validate_calls = []

    def validate_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Mock validate_claim method."""
        self.validate_calls.append(claim)
        return self.validation_results

    def validate_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock validate_batch method."""
        return [self.validate_claim(claim) for claim in claims]


class MockPreprocessor:
    """Mock preprocessor for testing."""

    def __init__(self, processed_data: Optional[Any] = None):
        self.processed_data = processed_data
        self.preprocess_calls = []

    def preprocess(self, data: Any) -> Any:
        """Mock preprocess method."""
        self.preprocess_calls.append(data)
        return self.processed_data or data

    def transform_features(self, claims: List[Dict[str, Any]]) -> pd.DataFrame:
        """Mock transform_features method."""
        # Return a simple DataFrame for testing
        return pd.DataFrame(
            [
                {
                    "claim_id": claim.get("claim_id", "TEST-001"),
                    "billed_amount": claim.get("billed_amount", 100.0),
                    "fraud_indicator": claim.get("fraud_indicator", False),
                }
                for claim in claims
            ]
        )


class MockMLModel:
    """Mock ML model for testing."""

    def __init__(self, predictions: Optional[List] = None, probabilities: Optional[List] = None):
        self.predictions = predictions or [0, 1, 0, 1]  # Default pattern
        self.probabilities = probabilities or [0.1, 0.9, 0.2, 0.8]
        self.fit_calls = []
        self.predict_calls = []
        self.is_fitted = False

    def fit(self, X: Any, y: Any) -> "MockMLModel":
        """Mock fit method."""
        self.fit_calls.append((X, y))
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Mock predict method."""
        self.predict_calls.append(X)
        # Return predictions based on input size
        if hasattr(X, "__len__"):
            size = len(X)
        else:
            size = 1
        return np.array(self.predictions[:size])

    def predict_proba(self, X: Any) -> np.ndarray:
        """Mock predict_proba method."""
        if hasattr(X, "__len__"):
            size = len(X)
        else:
            size = 1
        probs = self.probabilities[:size]
        # Return 2D array with probabilities for both classes
        return np.array([[1 - p, p] for p in probs])

    def score(self, X: Any, y: Any) -> float:
        """Mock score method."""
        return 0.95  # Mock high accuracy


class MockRuleEngine:
    """Mock rule engine for testing."""

    def __init__(self, rule_results: Optional[Dict] = None):
        self.rule_results = rule_results or {
            "triggered_rules": [],
            "risk_score": 0.0,
            "fraud_probability": 0.0,
        }
        self.evaluate_calls = []

    def evaluate_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Mock evaluate_claim method."""
        self.evaluate_calls.append(claim)
        return self.rule_results

    def get_triggered_rules(self, claim: Dict[str, Any]) -> List[str]:
        """Mock get_triggered_rules method."""
        return self.rule_results.get("triggered_rules", [])


class MockAnomalyDetector:
    """Mock anomaly detector for testing."""

    def __init__(self, anomaly_scores: Optional[List] = None):
        self.anomaly_scores = anomaly_scores or [0.1, 0.9, 0.2, 0.8]
        self.fit_calls = []
        self.detect_calls = []

    def fit(self, X: Any) -> "MockAnomalyDetector":
        """Mock fit method."""
        self.fit_calls.append(X)
        return self

    def detect_anomalies(self, X: Any) -> np.ndarray:
        """Mock detect_anomalies method."""
        self.detect_calls.append(X)
        if hasattr(X, "__len__"):
            size = len(X)
        else:
            size = 1
        return np.array(self.anomaly_scores[:size])

    def is_anomaly(self, score: float, threshold: float = 0.5) -> bool:
        """Mock is_anomaly method."""
        return score > threshold


class MockFeatureEngineer:
    """Mock feature engineer for testing."""

    def __init__(self, engineered_features: Optional[pd.DataFrame] = None):
        self.engineered_features = engineered_features
        self.engineer_calls = []

    def engineer_features(self, claims: List[Dict[str, Any]]) -> pd.DataFrame:
        """Mock engineer_features method."""
        self.engineer_calls.append(claims)

        if self.engineered_features is not None:
            return self.engineered_features

        # Default mock features
        return pd.DataFrame(
            [
                {
                    "claim_id": claim.get("claim_id", f"TEST-{i:03d}"),
                    "amount_zscore": np.random.normal(0, 1),
                    "provider_claim_count": np.random.randint(1, 100),
                    "patient_claim_frequency": np.random.uniform(0, 1),
                    "diagnosis_rarity_score": np.random.uniform(0, 1),
                    "time_to_claim_days": np.random.randint(1, 30),
                }
                for i, claim in enumerate(claims)
            ]
        )


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.data = {}
        self.queries = []

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Mock database query execution."""
        self.queries.append((query, params))
        return []

    def insert_claim(self, claim: Dict[str, Any]) -> str:
        """Mock claim insertion."""
        claim_id = claim.get("claim_id", "MOCK-001")
        self.data[claim_id] = claim
        return claim_id

    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Mock claim retrieval."""
        return self.data.get(claim_id)


@pytest.fixture
def mock_claims_data():
    """Fixture providing mock claims data."""
    return [
        {
            "claim_id": "CLM-2024-000001",
            "patient_id": "PAT-00000001",
            "provider_npi": "1234567890",
            "service_date": "2024-01-15T00:00:00",
            "claim_date": "2024-01-20T00:00:00",
            "billed_amount": 250.00,
            "diagnosis_codes": ["M79.3"],
            "procedure_codes": ["99213"],
            "patient_age": 45,
            "fraud_indicator": False,
            "fraud_type": None,
            "red_flags": [],
        },
        {
            "claim_id": "CLM-2024-000002",
            "patient_id": "PAT-00000002",
            "provider_npi": "9876543210",
            "service_date": "2024-01-16T00:00:00",
            "claim_date": "2024-01-17T00:00:00",
            "billed_amount": 15000.00,
            "diagnosis_codes": ["S13.4", "M79.3"],
            "procedure_codes": ["99285"],
            "patient_age": 35,
            "fraud_indicator": True,
            "fraud_type": "upcoding",
            "red_flags": ["excessive_billing", "complexity_mismatch"],
        },
    ]


@pytest.fixture
def mock_temp_file():
    """Fixture providing a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_data = [
            {"claim_id": "TEST-001", "billed_amount": 100.0},
            {"claim_id": "TEST-002", "billed_amount": 200.0},
        ]
        json.dump(test_data, f)
        temp_path = f.name

    yield temp_path

    import os

    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_data_loader():
    """Fixture providing a mock data loader."""
    return MockDataLoader()


@pytest.fixture
def mock_validator():
    """Fixture providing a mock validator."""
    return MockValidator()


@pytest.fixture
def mock_preprocessor():
    """Fixture providing a mock preprocessor."""
    return MockPreprocessor()


@pytest.fixture
def mock_ml_model():
    """Fixture providing a mock ML model."""
    return MockMLModel()


@pytest.fixture
def mock_rule_engine():
    """Fixture providing a mock rule engine."""
    return MockRuleEngine()


@pytest.fixture
def mock_anomaly_detector():
    """Fixture providing a mock anomaly detector."""
    return MockAnomalyDetector()


@pytest.fixture
def mock_feature_engineer():
    """Fixture providing a mock feature engineer."""
    return MockFeatureEngineer()


@pytest.fixture
def mock_database():
    """Fixture providing a mock database."""
    return MockDatabase()


# Context managers for mocking external services
class MockAPIResponse:
    """Mock API response for testing external API calls."""

    def __init__(self, json_data: Dict, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def mock_external_api_success():
    """Context manager for mocking successful external API calls."""
    return patch("requests.get", return_value=MockAPIResponse({"status": "success"}))


def mock_external_api_failure():
    """Context manager for mocking failed external API calls."""
    return patch(
        "requests.get", return_value=MockAPIResponse({"error": "Service unavailable"}, 500)
    )
