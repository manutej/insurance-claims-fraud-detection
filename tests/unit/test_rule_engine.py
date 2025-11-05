"""
Unit tests for the fraud detection rule engine.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from src.detection.rule_engine import RuleEngine, FraudRule, RuleResult
from tests.fixtures.claim_factories import (
    ValidClaim,
    UpcodingFraudClaim,
    PhantomBillingClaim,
    UnbundlingFraudClaim,
    StagedAccidentClaim,
    PrescriptionFraudClaim,
)
from tests.test_config import BENCHMARKS


class TestFraudRule:
    """Test the FraudRule dataclass."""

    def test_fraud_rule_creation(self):
        """Test creating a FraudRule instance."""
        rule = FraudRule(
            name="test_rule",
            description="Test rule description",
            fraud_type="test_fraud",
            weight=0.8,
            threshold=0.7,
        )

        assert rule.name == "test_rule"
        assert rule.description == "Test rule description"
        assert rule.fraud_type == "test_fraud"
        assert rule.weight == 0.8
        assert rule.threshold == 0.7
        assert rule.enabled is True  # Default value

    def test_fraud_rule_disabled(self):
        """Test creating a disabled FraudRule."""
        rule = FraudRule(
            name="disabled_rule",
            description="Disabled rule",
            fraud_type="test_fraud",
            weight=0.5,
            threshold=0.6,
            enabled=False,
        )

        assert rule.enabled is False


class TestRuleResult:
    """Test the RuleResult dataclass."""

    def test_rule_result_creation(self):
        """Test creating a RuleResult instance."""
        evidence = ["High billing amount", "Suspicious timing"]
        result = RuleResult(
            rule_name="test_rule",
            triggered=True,
            score=0.8,
            details="Rule triggered with high confidence",
            evidence=evidence,
        )

        assert result.rule_name == "test_rule"
        assert result.triggered is True
        assert result.score == 0.8
        assert result.details == "Rule triggered with high confidence"
        assert result.evidence == evidence


class TestRuleEngine:
    """Test the RuleEngine class."""

    @pytest.fixture
    def rule_engine(self):
        """Create a RuleEngine instance for testing."""
        return RuleEngine()

    @pytest.fixture
    def valid_claim(self):
        """Create a valid claim for testing."""
        return ValidClaim()

    @pytest.fixture
    def upcoding_claim(self):
        """Create an upcoding fraud claim for testing."""
        return UpcodingFraudClaim()

    def test_rule_engine_initialization(self, rule_engine):
        """Test RuleEngine initialization."""
        assert len(rule_engine.rules) > 0
        assert len(rule_engine.thresholds) > 0
        assert isinstance(rule_engine.provider_history, dict)
        assert isinstance(rule_engine.patient_history, dict)
        assert isinstance(rule_engine.claim_patterns, dict)

    def test_default_rules_loaded(self, rule_engine):
        """Test that default rules are loaded correctly."""
        expected_rules = [
            "upcoding_complexity",
            "phantom_billing_schedule",
            "phantom_billing_location",
            "unbundling_detection",
            "staged_accident_pattern",
            "prescription_fraud_volume",
            "kickback_referral_pattern",
            "billing_frequency_anomaly",
            "amount_anomaly",
        ]

        for rule_name in expected_rules:
            assert rule_name in rule_engine.rules
            assert isinstance(rule_engine.rules[rule_name], FraudRule)

    def test_default_thresholds_loaded(self, rule_engine):
        """Test that default thresholds are loaded correctly."""
        expected_thresholds = [
            "max_daily_claims_per_provider",
            "max_amount_per_claim",
            "suspicious_amount_multiplier",
            "min_time_between_claims_minutes",
        ]

        for threshold in expected_thresholds:
            assert threshold in rule_engine.thresholds

    @pytest.mark.unit
    def test_analyze_valid_claim(self, rule_engine, valid_claim):
        """Test analyzing a valid claim."""
        results, fraud_score = rule_engine.analyze_claim(valid_claim)

        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(fraud_score, float)
        assert 0.0 <= fraud_score <= 1.0

        # Valid claims should have low fraud scores
        assert fraud_score < BENCHMARKS.MAX_FALSE_POSITIVE_RATE

    @pytest.mark.unit
    def test_analyze_fraud_claim(self, rule_engine, upcoding_claim):
        """Test analyzing a fraudulent claim."""
        results, fraud_score = rule_engine.analyze_claim(upcoding_claim)

        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(fraud_score, float)
        assert 0.0 <= fraud_score <= 1.0

        # Fraud claims should have higher fraud scores
        assert fraud_score > 0.5  # Should detect fraud

        # Check that at least one rule was triggered
        triggered_rules = [r for r in results if r.triggered]
        assert len(triggered_rules) > 0

    def test_upcoding_detection(self, rule_engine):
        """Test upcoding fraud detection."""
        claim = {
            "claim_id": "CLM-TEST-001",
            "procedure_codes": ["99215"],  # High complexity
            "diagnosis_codes": ["Z00.00"],  # Simple diagnosis
            "billed_amount": 15000.0,  # Excessive amount
            "patient_id": "PAT-001",
            "provider_id": "PROV-001",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should trigger upcoding rule
        upcoding_results = [r for r in results if r.rule_name == "upcoding_complexity"]
        assert len(upcoding_results) == 1
        assert upcoding_results[0].triggered

    def test_phantom_billing_detection(self, rule_engine):
        """Test phantom billing detection."""
        claim = {
            "claim_id": "CLM-TEST-002",
            "date_of_service": "2024-01-01",  # Holiday
            "day_of_week": "Sunday",
            "service_location": "11",  # Office (not emergency)
            "time_of_service": "02:00",  # Outside normal hours
            "patient_id": "PAT-002",
            "provider_id": "PROV-002",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should trigger phantom billing rule
        phantom_results = [r for r in results if r.rule_name == "phantom_billing_schedule"]
        assert len(phantom_results) == 1
        assert phantom_results[0].triggered

    def test_unbundling_detection(self, rule_engine):
        """Test unbundling fraud detection."""
        claim = {
            "claim_id": "CLM-TEST-003",
            "procedure_codes": ["45378", "45380", "45384"],  # Bundled procedures
            "date_of_service": "2024-01-15",
            "patient_id": "PAT-003",
            "provider_id": "PROV-003",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should trigger unbundling rule
        unbundling_results = [r for r in results if r.rule_name == "unbundling_detection"]
        assert len(unbundling_results) == 1
        assert unbundling_results[0].triggered

    def test_amount_anomaly_detection(self, rule_engine):
        """Test amount anomaly detection."""
        claim = {
            "claim_id": "CLM-TEST-004",
            "billed_amount": 25000.0,  # Excessive amount
            "procedure_codes": ["99213"],  # Simple procedure
            "patient_id": "PAT-004",
            "provider_id": "PROV-004",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should trigger amount anomaly rule
        amount_results = [r for r in results if r.rule_name == "amount_anomaly"]
        assert len(amount_results) == 1
        assert amount_results[0].triggered

    def test_billing_frequency_anomaly(self, rule_engine):
        """Test billing frequency anomaly detection."""
        # Add multiple claims for same provider on same day
        claim_base = {
            "date_of_service": "2024-01-15",
            "patient_id": "PAT-005",
            "provider_id": "PROV-005",
        }

        # Add 60 claims (exceeds threshold of 50)
        for i in range(60):
            claim = {**claim_base, "claim_id": f"CLM-TEST-{i:03d}"}
            rule_engine._update_claim_history(claim)

        # Test the last claim
        test_claim = {**claim_base, "claim_id": "CLM-TEST-FINAL"}
        results, fraud_score = rule_engine.analyze_claim(test_claim)

        # Should trigger billing frequency rule
        frequency_results = [r for r in results if r.rule_name == "billing_frequency_anomaly"]
        assert len(frequency_results) == 1
        assert frequency_results[0].triggered

    def test_prescription_fraud_detection(self, rule_engine):
        """Test prescription fraud detection."""
        claim = {
            "claim_id": "CLM-TEST-006",
            "procedure_codes": [
                "J1100",
                "J2001",
                "J3420",
                "J7799",
                "J2315",
                "J1170",
            ],  # Many prescriptions
            "diagnosis_codes": ["R50.9"],  # Inadequate diagnosis for controlled substances
            "patient_id": "PAT-006",
            "provider_id": "PROV-006",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should trigger prescription fraud rule
        prescription_results = [r for r in results if r.rule_name == "prescription_fraud_volume"]
        assert len(prescription_results) == 1
        assert prescription_results[0].triggered

    def test_config_loading(self, rule_engine):
        """Test loading configuration from file."""
        config_data = {
            "rules": {"upcoding_complexity": {"threshold": 0.5, "weight": 0.9, "enabled": False}},
            "thresholds": {"max_amount_per_claim": 5000},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            rule_engine.load_config("test_config.json")

        # Check that configuration was loaded
        upcoding_rule = rule_engine.rules["upcoding_complexity"]
        assert upcoding_rule.threshold == 0.5
        assert upcoding_rule.weight == 0.9
        assert upcoding_rule.enabled is False
        assert rule_engine.thresholds["max_amount_per_claim"] == 5000

    def test_config_loading_error(self, rule_engine, caplog):
        """Test handling of configuration loading errors."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            rule_engine.load_config("nonexistent.json")

        assert "Failed to load config" in caplog.text

    def test_disabled_rule_not_applied(self, rule_engine):
        """Test that disabled rules are not applied."""
        # Disable upcoding rule
        rule_engine.rules["upcoding_complexity"].enabled = False

        claim = {
            "claim_id": "CLM-TEST-007",
            "procedure_codes": ["99215"],
            "diagnosis_codes": ["Z00.00"],
            "billed_amount": 15000.0,
            "patient_id": "PAT-007",
            "provider_id": "PROV-007",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should not have upcoding results since rule is disabled
        upcoding_results = [r for r in results if r.rule_name == "upcoding_complexity"]
        assert len(upcoding_results) == 0

    def test_fraud_score_calculation(self, rule_engine):
        """Test fraud score calculation with multiple triggered rules."""
        claim = {
            "claim_id": "CLM-TEST-008",
            "procedure_codes": ["99215"],
            "diagnosis_codes": ["Z00.00"],
            "billed_amount": 25000.0,
            "date_of_service": "2024-01-01",  # Holiday
            "day_of_week": "Sunday",
            "service_location": "11",
            "patient_id": "PAT-008",
            "provider_id": "PROV-008",
        }

        results, fraud_score = rule_engine.analyze_claim(claim)

        # Should have multiple triggered rules
        triggered_rules = [r for r in results if r.triggered]
        assert len(triggered_rules) > 1

        # Fraud score should be weighted average
        assert 0.0 <= fraud_score <= 1.0
        assert fraud_score > 0.5  # Should be high due to multiple triggers

    def test_explanation_generation(self, rule_engine):
        """Test generating human-readable explanations."""
        results = [
            RuleResult(
                "upcoding_complexity", True, 0.8, "High complexity mismatch", ["Evidence 1"]
            ),
            RuleResult("amount_anomaly", True, 0.6, "Excessive amount", ["Evidence 2"]),
            RuleResult("phantom_billing_schedule", False, 0.2, "No scheduling issues", []),
        ]
        fraud_score = 0.75

        explanation = rule_engine.generate_explanation(results, fraud_score)

        assert "High" in explanation  # Should indicate high risk
        assert "upcoding_complexity" in explanation
        assert "amount_anomaly" in explanation
        assert "Evidence 1" in explanation
        assert "Evidence 2" in explanation
        assert "phantom_billing_schedule" not in explanation  # Not triggered

    def test_rule_statistics(self, rule_engine):
        """Test getting rule statistics."""
        stats = rule_engine.get_rule_statistics()

        assert "total_rules" in stats
        assert "enabled_rules" in stats
        assert "rules_by_type" in stats

        assert stats["total_rules"] > 0
        assert stats["enabled_rules"] <= stats["total_rules"]
        assert isinstance(stats["rules_by_type"], dict)

    @pytest.mark.parametrize(
        "procedure_codes,expected",
        [
            (["99213", "99214", "99215"], True),  # Sequential
            (["99213", "99280", "99215"], False),  # Not sequential
            (["99213"], False),  # Single code
            ([], False),  # Empty
        ],
    )
    def test_suspicious_procedure_progression(self, rule_engine, procedure_codes, expected):
        """Test detection of suspicious procedure progressions."""
        result = rule_engine._has_suspicious_procedure_progression(procedure_codes)
        assert result == expected

    @pytest.mark.parametrize(
        "date_str,expected",
        [
            ("2024-01-01", True),  # New Year's Day
            ("2024-07-04", True),  # Independence Day
            ("2024-12-25", True),  # Christmas
            ("2024-06-15", False),  # Regular day
        ],
    )
    def test_holiday_detection(self, rule_engine, date_str, expected):
        """Test holiday detection."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        result = rule_engine._is_holiday(date)
        assert result == expected

    def test_context_claims_usage(self, rule_engine):
        """Test using context claims for pattern analysis."""
        main_claim = {
            "claim_id": "CLM-MAIN",
            "procedure_codes": ["45378", "45380"],
            "date_of_service": "2024-01-15",
            "patient_id": "PAT-CONTEXT",
            "provider_id": "PROV-CONTEXT",
        }

        context_claims = [
            {
                "claim_id": "CLM-CONTEXT-1",
                "procedure_codes": ["45378"],  # Same procedure
                "date_of_service": "2024-01-15",  # Same date
                "patient_id": "PAT-CONTEXT",  # Same patient
                "provider_id": "PROV-CONTEXT",  # Same provider
            }
        ]

        results, fraud_score = rule_engine.analyze_claim(main_claim, context_claims)

        # Should detect unbundling due to context claims
        unbundling_results = [r for r in results if r.rule_name == "unbundling_detection"]
        assert len(unbundling_results) == 1
        assert unbundling_results[0].triggered

    def test_rule_execution_error_handling(self, rule_engine):
        """Test handling of rule execution errors."""
        # Create a malformed claim that might cause errors
        malformed_claim = {
            "claim_id": None,  # Invalid claim ID
            "billed_amount": "invalid",  # Invalid amount
            "procedure_codes": None,  # Invalid codes
        }

        # Should not raise exception, but return safe results
        results, fraud_score = rule_engine.analyze_claim(malformed_claim)

        assert isinstance(results, list)
        assert isinstance(fraud_score, float)
        assert 0.0 <= fraud_score <= 1.0

    @pytest.mark.performance
    def test_large_context_claims_performance(self, rule_engine, valid_claim):
        """Test performance with large number of context claims."""
        import time

        # Create large context claims list
        context_claims = [ValidClaim() for _ in range(1000)]

        start_time = time.time()
        results, fraud_score = rule_engine.analyze_claim(valid_claim, context_claims)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Should process within reasonable time
        assert (
            processing_time < BENCHMARKS.MAX_SINGLE_CLAIM_LATENCY_MS * 2
        )  # Allow 2x for large context
        assert isinstance(results, list)
        assert isinstance(fraud_score, float)

    def test_edge_case_empty_claim(self, rule_engine):
        """Test handling of empty claim."""
        empty_claim = {}

        results, fraud_score = rule_engine.analyze_claim(empty_claim)

        assert isinstance(results, list)
        assert isinstance(fraud_score, float)
        assert fraud_score >= 0.0

    def test_edge_case_none_values(self, rule_engine):
        """Test handling of None values in claim fields."""
        claim_with_nones = {
            "claim_id": "CLM-NONE-TEST",
            "billed_amount": None,
            "procedure_codes": None,
            "diagnosis_codes": None,
            "patient_id": None,
            "provider_id": None,
        }

        results, fraud_score = rule_engine.analyze_claim(claim_with_nones)

        assert isinstance(results, list)
        assert isinstance(fraud_score, float)
        assert fraud_score >= 0.0

    @pytest.mark.fraud_detection
    def test_accuracy_requirements(self, rule_engine):
        """Test that rule engine meets accuracy requirements."""
        # Generate test dataset
        valid_claims = [ValidClaim() for _ in range(100)]
        fraud_claims = [UpcodingFraudClaim() for _ in range(50)]

        # Test predictions
        correct_predictions = 0
        total_predictions = 0
        false_positives = 0

        for claim in valid_claims:
            _, fraud_score = rule_engine.analyze_claim(claim)
            total_predictions += 1
            if fraud_score < 0.5:  # Correctly identified as valid
                correct_predictions += 1
            else:  # False positive
                false_positives += 1

        for claim in fraud_claims:
            _, fraud_score = rule_engine.analyze_claim(claim)
            total_predictions += 1
            if fraud_score >= 0.5:  # Correctly identified as fraud
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        false_positive_rate = false_positives / len(valid_claims)

        # Should meet accuracy requirements
        assert accuracy >= BENCHMARKS.MIN_ACCURACY
        assert false_positive_rate <= BENCHMARKS.MAX_FALSE_POSITIVE_RATE
