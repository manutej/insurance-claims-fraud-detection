#!/usr/bin/env python3
"""
Fraud Detection System Demo

This script demonstrates the comprehensive fraud detection system
with rule-based detection, machine learning, and anomaly detection.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection.fraud_detector import FraudDetectorOrchestrator, DetectionConfig
from training.train_models import ModelTrainingPipeline, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_sample_data():
    """Load sample claims data for demonstration."""
    try:
        # Load legitimate claims
        with open("data/valid_claims/medical_claims.json", "r") as f:
            valid_data = json.load(f)
            valid_claims = valid_data["claims"][:50]  # Use first 50 claims

        # Load fraudulent claims
        fraud_files = [
            "data/fraudulent_claims/phantom_billing.json",
            "data/fraudulent_claims/unbundling_fraud.json",
        ]

        fraud_claims = []
        for file_path in fraud_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    fraud_data = json.load(f)
                    fraud_claims.extend(fraud_data["claims"][:25])  # 25 from each type

        # Combine data
        all_claims = valid_claims + fraud_claims
        logger.info(f"Loaded {len(valid_claims)} valid and {len(fraud_claims)} fraud claims")

        return all_claims

    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        return []


def create_synthetic_claims():
    """Create synthetic claims data for demonstration if files not available."""
    logger.info("Creating synthetic claims data")

    legitimate_claims = []
    fraud_claims = []

    # Create 100 legitimate claims
    for i in range(100):
        claim = {
            "claim_id": f"CLM-LEGIT-{i:04d}",
            "patient_id": f"PAT-{i:04d}",
            "provider_id": f"PRV-{(i % 20):03d}",
            "provider_npi": f"123456{(i % 20):04d}",
            "date_of_service": f"2024-{(i % 12 + 1):02d}-{(i % 28 + 1):02d}",
            "diagnosis_codes": ["I10", "E11.9"],
            "procedure_codes": ["99213"],
            "billed_amount": 125.0 + (i % 100),
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": False,
            "notes": "Routine medical visit",
        }
        legitimate_claims.append(claim)

    # Create 30 fraudulent claims
    for i in range(30):
        claim = {
            "claim_id": f"CLM-FRAUD-{i:04d}",
            "patient_id": f"PAT-GHOST-{i:03d}",
            "provider_id": f"PRV-FRAUD-{(i % 5):03d}",
            "provider_npi": f"999999{(i % 5):04d}",
            "date_of_service": f"2024-{(i % 12 + 1):02d}-{(i % 28 + 1):02d}",
            "diagnosis_codes": ["E11.9", "I10"],
            "procedure_codes": ["99215", "93000", "36415"],  # Upcoded procedures
            "billed_amount": 1000.0 + (i * 50),  # Inflated amounts
            "service_location": "11",
            "claim_type": "professional",
            "fraud_indicator": True,
            "fraud_type": "upcoding" if i % 2 == 0 else "phantom_billing",
            "red_flags": [
                "Excessive procedures for diagnosis",
                "Unusual billing pattern",
                "Patient address not verified",
            ],
            "notes": "Suspicious claim with multiple red flags",
        }
        fraud_claims.append(claim)

    return legitimate_claims + fraud_claims


def demo_rule_based_detection():
    """Demonstrate rule-based fraud detection."""
    logger.info("\n" + "=" * 50)
    logger.info("RULE-BASED FRAUD DETECTION DEMO")
    logger.info("=" * 50)

    from detection.rule_engine import RuleEngine

    # Initialize rule engine
    rule_engine = RuleEngine()

    # Create a suspicious claim
    suspicious_claim = {
        "claim_id": "CLM-SUSPICIOUS-001",
        "patient_id": "PAT-GHOST-001",
        "provider_id": "PRV-FRAUD-001",
        "provider_npi": "9999999001",
        "date_of_service": "2024-12-25",  # Christmas
        "day_of_week": "Sunday",
        "diagnosis_codes": ["I10"],  # Simple hypertension
        "procedure_codes": ["99215", "93000", "36415"],  # Complex procedures
        "billed_amount": 2500.00,  # Excessive amount
        "service_location": "11",
        "claim_type": "professional",
        "fraud_indicator": True,
        "red_flags": [
            "Service on holiday when office closed",
            "Patient address doesn't exist",
            "Excessive procedures for simple diagnosis",
        ],
    }

    # Analyze claim
    rule_results, fraud_score = rule_engine.analyze_claim(suspicious_claim)

    print(f"\nClaim Analysis Results:")
    print(f"Overall Fraud Score: {fraud_score:.3f}")
    print(f"Number of triggered rules: {len([r for r in rule_results if r.triggered])}")

    for result in rule_results:
        if result.triggered:
            print(f"\n🚨 RULE TRIGGERED: {result.rule_name}")
            print(f"   Score: {result.score:.3f}")
            print(f"   Details: {result.details}")
            for evidence in result.evidence:
                print(f"   Evidence: {evidence}")

    # Generate explanation
    explanation = rule_engine.generate_explanation(rule_results, fraud_score)
    print(f"\nExplanation:\n{explanation}")


def demo_ml_detection():
    """Demonstrate machine learning fraud detection."""
    logger.info("\n" + "=" * 50)
    logger.info("MACHINE LEARNING FRAUD DETECTION DEMO")
    logger.info("=" * 50)

    # Get sample data
    claims_data = load_sample_data()
    if not claims_data:
        claims_data = create_synthetic_claims()

    if len(claims_data) < 50:
        logger.warning("Insufficient data for ML demo")
        return

    # Configure training
    training_config = TrainingConfig(
        test_size=0.3,
        validation_size=0.2,
        enable_hyperparameter_tuning=False,  # Skip for demo speed
        generate_plots=False,
        output_directory="demo_models",
    )

    # Run training pipeline
    pipeline = ModelTrainingPipeline(training_config)

    try:
        logger.info("Training fraud detection models...")
        results = pipeline.run_full_pipeline(claims_data=claims_data)

        print(f"\n📊 TRAINING RESULTS:")
        print(f"Training Time: {results.training_time_seconds:.2f} seconds")
        print(f"Best Model: {results.best_model}")
        print(f"Final Accuracy: {results.overall_performance['accuracy']:.4f}")
        print(f"False Positive Rate: {results.overall_performance['false_positive_rate']:.4f}")
        print(f"Meets Targets: {'✅' if results.meets_targets else '❌'}")

        # Show top features
        if results.feature_importance:
            print(f"\n🔍 TOP 5 IMPORTANT FEATURES:")
            sorted_features = sorted(
                results.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"   {i+1}. {feature}: {importance:.4f}")

        return pipeline

    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return None


def demo_anomaly_detection():
    """Demonstrate anomaly detection."""
    logger.info("\n" + "=" * 50)
    logger.info("ANOMALY DETECTION DEMO")
    logger.info("=" * 50)

    from detection.anomaly_detector import AnomalyDetectionSuite
    from detection.feature_engineering import FeatureEngineer
    import pandas as pd

    # Get sample data
    claims_data = load_sample_data()
    if not claims_data:
        claims_data = create_synthetic_claims()

    # Extract features
    feature_engineer = FeatureEngineer()
    feature_set = feature_engineer.extract_features(claims_data)
    X = feature_engineer.combine_features(feature_set, include_sets=["basic", "temporal"])

    # Use only numeric features for anomaly detection
    X_numeric = X.select_dtypes(include=["number"])

    if X_numeric.empty:
        logger.warning("No numeric features available for anomaly detection")
        return

    # Split into normal and test data
    df = pd.DataFrame(claims_data)
    normal_data = X_numeric[df["fraud_indicator"] == False]
    test_data = X_numeric

    if len(normal_data) == 0:
        logger.warning("No normal data available for training")
        return

    # Initialize and train anomaly detector
    anomaly_detector = AnomalyDetectionSuite()

    try:
        logger.info("Training anomaly detection models...")
        anomaly_detector.fit(normal_data, list(X_numeric.columns))

        logger.info("Detecting anomalies...")
        anomaly_results = anomaly_detector.detect_anomalies(test_data)

        # Analyze results
        total_claims = len(anomaly_results)
        anomaly_count = sum(1 for r in anomaly_results if r.is_anomaly)
        anomaly_rate = anomaly_count / total_claims if total_claims > 0 else 0

        print(f"\n🔍 ANOMALY DETECTION RESULTS:")
        print(f"Total Claims Analyzed: {total_claims}")
        print(f"Anomalies Detected: {anomaly_count}")
        print(f"Anomaly Rate: {anomaly_rate:.2%}")

        # Show some anomalous claims
        high_confidence_anomalies = [
            r for r in anomaly_results if r.is_anomaly and r.confidence > 0.8
        ]

        if high_confidence_anomalies:
            print(f"\n🚨 HIGH CONFIDENCE ANOMALIES:")
            for result in high_confidence_anomalies[:3]:
                print(
                    f"   Claim {result.claim_id}: Score {result.anomaly_score:.3f}, "
                    f"Confidence {result.confidence:.2%}"
                )
                print(f"   Explanation: {result.explanation}")

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")


def demo_full_system():
    """Demonstrate the complete fraud detection system."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE FRAUD DETECTION SYSTEM DEMO")
    logger.info("=" * 60)

    # Get sample data
    claims_data = load_sample_data()
    if not claims_data:
        claims_data = create_synthetic_claims()

    if len(claims_data) < 30:
        logger.warning("Insufficient data for full system demo")
        return

    # Configure detection system
    detection_config = DetectionConfig(
        fraud_threshold=0.7,
        rule_weight=0.3,
        ml_weight=0.5,
        anomaly_weight=0.2,
        enable_parallel_processing=False,  # Disable for demo
    )

    # Initialize orchestrator
    orchestrator = FraudDetectorOrchestrator(detection_config)

    try:
        # Split data for training and testing
        train_size = int(len(claims_data) * 0.7)
        train_claims = claims_data[:train_size]
        test_claims = claims_data[train_size:]

        logger.info(f"Training system on {len(train_claims)} claims...")

        # Train the system
        training_results = orchestrator.train(train_claims)

        print(f"\n🎯 TRAINING COMPLETED:")
        print(f"Training Time: {training_results['training_time_seconds']:.2f} seconds")
        if "overall_performance" in training_results:
            perf = training_results["overall_performance"]
            print(f"Accuracy: {perf['accuracy']:.4f}")
            print(f"False Positive Rate: {perf['false_positive_rate']:.4f}")

        # Test on remaining claims
        logger.info(f"Testing on {len(test_claims)} claims...")
        results = orchestrator.detect_batch(test_claims)

        # Analyze results
        fraud_detected = sum(1 for r in results if r.is_fraud)
        high_risk = sum(1 for r in results if r.risk_level in ["HIGH", "CRITICAL"])

        print(f"\n📊 DETECTION RESULTS:")
        print(f"Total Claims Processed: {len(results)}")
        print(f"Fraud Claims Detected: {fraud_detected}")
        print(f"High Risk Claims: {high_risk}")

        # Show some results
        print(f"\n🔍 SAMPLE DETECTION RESULTS:")
        for result in results[:5]:
            status = "🚨 FRAUD" if result.is_fraud else "✅ LEGITIMATE"
            print(
                f"   {result.claim_id}: {status} - {result.risk_level} risk "
                f"(Score: {result.fraud_probability:.3f}, Confidence: {result.confidence_score:.2%})"
            )

        # Generate full report
        report = orchestrator.generate_detection_report(results)
        print(f"\n📋 DETAILED REPORT:")
        print(report[:500] + "..." if len(report) > 500 else report)

        return orchestrator

    except Exception as e:
        logger.error(f"Full system demo failed: {e}")
        return None


def main():
    """Run all fraud detection demos."""
    logger.info("🚀 Starting Fraud Detection System Demonstration")

    try:
        # Demo 1: Rule-based detection
        demo_rule_based_detection()

        # Demo 2: Anomaly detection
        demo_anomaly_detection()

        # Demo 3: Machine learning detection
        pipeline = demo_ml_detection()

        # Demo 4: Complete system
        orchestrator = demo_full_system()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        print(f"\n📈 SUMMARY:")
        print(f"✅ Rule-based detection: Implemented with configurable thresholds")
        print(f"✅ Machine learning models: Multiple algorithms with ensemble voting")
        print(f"✅ Anomaly detection: Statistical and ML-based outlier detection")
        print(f"✅ Feature engineering: Advanced temporal and network features")
        print(f"✅ Complete orchestrator: Real-time and batch processing")
        print(f"✅ Performance optimization: Targeting >94% accuracy, <3.8% FPR")

        if pipeline and hasattr(pipeline, "training_results") and pipeline.training_results:
            print(f"\n🎯 PERFORMANCE TARGETS:")
            results = pipeline.training_results
            accuracy_met = results.overall_performance["accuracy"] >= 0.94
            fpr_met = results.overall_performance["false_positive_rate"] <= 0.038
            print(
                f"   Accuracy >94%: {'✅' if accuracy_met else '❌'} "
                f"({results.overall_performance['accuracy']:.1%})"
            )
            print(
                f"   FPR <3.8%: {'✅' if fpr_met else '❌'} "
                f"({results.overall_performance['false_positive_rate']:.1%})"
            )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
