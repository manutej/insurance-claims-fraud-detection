"""
Main fraud detection orchestrator.

This module combines rule-based and ML approaches with confidence scoring,
explanation generation, and both real-time and batch processing modes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import detection modules
from .rule_engine import RuleEngine, RuleResult
from .ml_models import MLModelManager, PredictionResult, ModelPerformance
from .anomaly_detector import AnomalyDetectionSuite, AnomalyScore
from .feature_engineering import FeatureEngineer, FeatureSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FraudDetectionResult:
    """Comprehensive fraud detection result."""

    claim_id: str
    is_fraud: bool
    fraud_probability: float
    confidence_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    explanation: str
    evidence: List[str]

    # Component scores
    rule_score: float
    ml_score: float
    anomaly_score: float

    # Detailed results
    rule_results: List[Dict[str, Any]]
    ml_results: Dict[str, Any]
    anomaly_results: Dict[str, Any]

    # Metadata
    processing_time_ms: float
    model_version: str
    detection_timestamp: str


@dataclass
class DetectionConfig:
    """Configuration for fraud detection orchestrator."""

    # Threshold settings
    fraud_threshold: float = 0.7
    high_risk_threshold: float = 0.9
    medium_risk_threshold: float = 0.5

    # Weight settings for ensemble
    rule_weight: float = 0.3
    ml_weight: float = 0.5
    anomaly_weight: float = 0.2

    # Processing settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300

    # Feature settings
    feature_sets: List[str] = None
    top_k_features: int = 50

    # Model settings
    preferred_ml_model: str = "ensemble"
    enable_anomaly_detection: bool = True
    enable_rule_engine: bool = True

    # Performance requirements
    target_accuracy: float = 0.94
    target_false_positive_rate: float = 0.038

    def __post_init__(self):
        if self.feature_sets is None:
            self.feature_sets = ["basic", "temporal", "network", "statistical"]


class FraudDetectorOrchestrator:
    """
    Main fraud detection orchestrator that combines multiple detection methods.

    Integrates:
    - Rule-based detection
    - Machine learning models
    - Anomaly detection
    - Feature engineering
    - Confidence scoring
    - Explanation generation
    """

    def __init__(self, config: DetectionConfig = None):
        """Initialize fraud detection orchestrator."""
        self.config = config or DetectionConfig()

        # Component modules
        self.rule_engine = None
        self.ml_manager = None
        self.anomaly_detector = None
        self.feature_engineer = None

        # State
        self.is_trained = False
        self.model_version = "1.0.0"
        self.performance_metrics = {}
        self.feature_importance = {}

        # Threading
        self._lock = threading.Lock()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize detection components."""
        logger.info("Initializing fraud detection components")

        # Initialize rule engine
        if self.config.enable_rule_engine:
            self.rule_engine = RuleEngine()
            logger.info("Rule engine initialized")

        # Initialize ML manager
        self.ml_manager = MLModelManager(
            target_metrics={
                "accuracy": self.config.target_accuracy,
                "false_positive_rate": self.config.target_false_positive_rate,
            }
        )
        logger.info("ML model manager initialized")

        # Initialize anomaly detector
        if self.config.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetectionSuite()
            logger.info("Anomaly detector initialized")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        logger.info("Feature engineer initialized")

    def train(
        self, training_claims: List[Dict[str, Any]], validation_claims: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train the fraud detection system.

        Args:
            training_claims: List of training claims with fraud labels
            validation_claims: Optional validation claims

        Returns:
            Training results and performance metrics
        """
        logger.info(f"Training fraud detection system on {len(training_claims)} claims")
        start_time = time.time()

        try:
            # Convert to DataFrame
            train_df = pd.DataFrame(training_claims)

            # Validate required columns
            if "fraud_indicator" not in train_df.columns:
                raise ValueError("Training data must include 'fraud_indicator' column")

            # Extract features
            logger.info("Extracting features from training data")
            feature_set = self.feature_engineer.extract_features(
                training_claims, target_col="fraud_indicator"
            )

            # Combine features
            X = self.feature_engineer.combine_features(
                feature_set, include_sets=self.config.feature_sets
            )
            y = train_df["fraud_indicator"]

            # Feature selection
            if self.config.top_k_features:
                selected_features = self.feature_engineer.select_features(
                    feature_set, y, top_k=self.config.top_k_features
                )
                X = X[selected_features]
                logger.info(f"Selected {len(selected_features)} features")

            # Train ML models
            logger.info("Training machine learning models")
            ml_performance = self.ml_manager.train_models(X, y)

            # Train anomaly detector on normal samples
            if self.config.enable_anomaly_detection:
                logger.info("Training anomaly detector")
                normal_samples = X[y == 0]  # Non-fraud samples
                self.anomaly_detector.fit(normal_samples, list(X.columns))

            # Update rule engine with training data (for pattern analysis)
            if self.config.enable_rule_engine:
                logger.info("Updating rule engine patterns")
                for claim in training_claims:
                    self.rule_engine._update_claim_history(claim)

            # Calculate overall performance
            overall_performance = self._evaluate_ensemble_performance(X, y, validation_claims)

            # Store performance metrics
            self.performance_metrics = {
                "ml_performance": ml_performance,
                "overall_performance": overall_performance,
                "training_time_seconds": time.time() - start_time,
                "training_samples": len(training_claims),
                "feature_count": X.shape[1],
            }

            # Get feature importance
            self.feature_importance = self.ml_manager.get_feature_importance()

            self.is_trained = True
            logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

            return self.performance_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _evaluate_ensemble_performance(
        self, X: pd.DataFrame, y: pd.Series, validation_claims: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate ensemble performance on validation data."""
        if validation_claims:
            # Use provided validation data
            val_results = self.detect_batch(validation_claims)
            val_df = pd.DataFrame(validation_claims)
            y_true = val_df["fraud_indicator"]
            y_pred = [result.is_fraud for result in val_results]
            y_proba = [result.fraud_probability for result in val_results]
        else:
            # Use training data for evaluation (not ideal, but better than nothing)
            train_results = self.detect_batch([claim for claim in X.to_dict("records")])
            y_true = y
            y_pred = [result.is_fraud for result in train_results[: len(y)]]
            y_proba = [result.fraud_probability for result in train_results[: len(y)]]

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Calculate false positive rate
            from sklearn.metrics import confusion_matrix

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            # AUC if possible
            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = 0.5

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_positive_rate": fpr,
                "roc_auc": auc,
            }
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_positive_rate": 1.0,
                "roc_auc": 0.5,
            }

    def detect_single(
        self, claim: Dict[str, Any], context_claims: List[Dict[str, Any]] = None
    ) -> FraudDetectionResult:
        """
        Detect fraud in a single claim (real-time mode).

        Args:
            claim: Claim to analyze
            context_claims: Related claims for pattern analysis

        Returns:
            Fraud detection result
        """
        if not self.is_trained:
            raise RuntimeError("Fraud detector not trained. Call train() first.")

        start_time = time.time()

        try:
            # Extract features for this claim
            feature_set = self.feature_engineer.extract_features([claim])
            X = self.feature_engineer.combine_features(
                feature_set, include_sets=self.config.feature_sets
            )

            # Apply feature selection if configured
            if self.config.top_k_features and self.feature_importance:
                selected_features = list(self.feature_importance.keys())[
                    : self.config.top_k_features
                ]
                available_features = [f for f in selected_features if f in X.columns]
                if available_features:
                    X = X[available_features]

            # Component detections
            component_results = self._run_component_detections(claim, X, context_claims)

            # Combine results
            final_result = self._combine_detection_results(
                claim, component_results, time.time() - start_time
            )

            return final_result

        except Exception as e:
            logger.error(f"Single claim detection failed: {e}")
            # Return safe default
            return FraudDetectionResult(
                claim_id=claim.get("claim_id", "unknown"),
                is_fraud=True,  # Conservative approach
                fraud_probability=0.5,
                confidence_score=0.0,
                risk_level="MEDIUM",
                explanation=f"Detection failed: {str(e)}",
                evidence=[],
                rule_score=0.0,
                ml_score=0.0,
                anomaly_score=0.0,
                rule_results=[],
                ml_results={},
                anomaly_results={},
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version=self.model_version,
                detection_timestamp=datetime.now().isoformat(),
            )

    def detect_batch(
        self, claims: List[Dict[str, Any]], context_claims: List[Dict[str, Any]] = None
    ) -> List[FraudDetectionResult]:
        """
        Detect fraud in multiple claims (batch mode).

        Args:
            claims: Claims to analyze
            context_claims: Related claims for pattern analysis

        Returns:
            List of fraud detection results
        """
        if not self.is_trained:
            raise RuntimeError("Fraud detector not trained. Call train() first.")

        logger.info(f"Processing batch of {len(claims)} claims")
        start_time = time.time()

        try:
            if self.config.enable_parallel_processing and len(claims) > 10:
                # Parallel processing for large batches
                results = self._process_batch_parallel(claims, context_claims)
            else:
                # Sequential processing for small batches
                results = self._process_batch_sequential(claims, context_claims)

            processing_time = time.time() - start_time
            logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            logger.info(f"Average time per claim: {(processing_time/len(claims))*1000:.2f} ms")

            return results

        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            raise

    def _process_batch_sequential(
        self, claims: List[Dict[str, Any]], context_claims: List[Dict[str, Any]] = None
    ) -> List[FraudDetectionResult]:
        """Process claims sequentially."""
        results = []
        for claim in claims:
            try:
                result = self.detect_single(claim, context_claims)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process claim {claim.get('claim_id', 'unknown')}: {e}")
                # Add error result
                results.append(self._create_error_result(claim, str(e)))

        return results

    def _process_batch_parallel(
        self, claims: List[Dict[str, Any]], context_claims: List[Dict[str, Any]] = None
    ) -> List[FraudDetectionResult]:
        """Process claims in parallel."""
        results = [None] * len(claims)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_index = {
                executor.submit(self.detect_single, claim, context_claims): i
                for i, claim in enumerate(claims)
            }

            # Collect results
            for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to process claim at index {index}: {e}")
                    results[index] = self._create_error_result(claims[index], str(e))

        return results

    def _run_component_detections(
        self, claim: Dict[str, Any], X: pd.DataFrame, context_claims: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run all component detection methods."""
        component_results = {}

        # Rule-based detection
        if self.config.enable_rule_engine and self.rule_engine:
            try:
                rule_results, rule_score = self.rule_engine.analyze_claim(claim, context_claims)
                component_results["rules"] = {
                    "score": rule_score,
                    "results": [asdict(r) for r in rule_results],
                }
            except Exception as e:
                logger.error(f"Rule engine failed: {e}")
                component_results["rules"] = {"score": 0.0, "results": []}

        # ML prediction
        try:
            ml_predictions = self.ml_manager.predict(X, self.config.preferred_ml_model)
            if ml_predictions:
                ml_result = ml_predictions[0]
                component_results["ml"] = {
                    "score": ml_result.probability,
                    "prediction": ml_result.prediction,
                    "confidence": ml_result.confidence,
                    "explanation": ml_result.explanation,
                }
            else:
                component_results["ml"] = {
                    "score": 0.5,
                    "prediction": 0,
                    "confidence": 0.0,
                    "explanation": "No prediction",
                }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            component_results["ml"] = {
                "score": 0.5,
                "prediction": 0,
                "confidence": 0.0,
                "explanation": f"ML failed: {e}",
            }

        # Anomaly detection
        if self.config.enable_anomaly_detection and self.anomaly_detector:
            try:
                anomaly_results = self.anomaly_detector.detect_anomalies(X)
                if anomaly_results:
                    anomaly_result = anomaly_results[0]
                    component_results["anomaly"] = {
                        "score": anomaly_result.anomaly_score,
                        "is_anomaly": anomaly_result.is_anomaly,
                        "confidence": anomaly_result.confidence,
                        "explanation": anomaly_result.explanation,
                        "contributing_features": anomaly_result.contributing_features,
                    }
                else:
                    component_results["anomaly"] = {
                        "score": 0.0,
                        "is_anomaly": False,
                        "confidence": 0.0,
                        "explanation": "No anomaly detected",
                    }
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                component_results["anomaly"] = {
                    "score": 0.0,
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "explanation": f"Anomaly detection failed: {e}",
                }

        return component_results

    def _combine_detection_results(
        self, claim: Dict[str, Any], component_results: Dict[str, Any], processing_time: float
    ) -> FraudDetectionResult:
        """Combine results from all detection components."""
        # Extract component scores
        rule_score = component_results.get("rules", {}).get("score", 0.0)
        ml_score = component_results.get("ml", {}).get("score", 0.5)
        anomaly_score = component_results.get("anomaly", {}).get("score", 0.0)

        # Normalize anomaly score (convert to 0-1 scale)
        if anomaly_score > 1:
            anomaly_score = min(anomaly_score / 10.0, 1.0)  # Simple normalization

        # Calculate weighted ensemble score
        ensemble_score = (
            self.config.rule_weight * rule_score
            + self.config.ml_weight * ml_score
            + self.config.anomaly_weight * anomaly_score
        )

        # Determine fraud decision
        is_fraud = ensemble_score >= self.config.fraud_threshold

        # Calculate confidence based on agreement between methods
        scores = [rule_score, ml_score, anomaly_score]
        weights = [self.config.rule_weight, self.config.ml_weight, self.config.anomaly_weight]

        # Confidence is higher when methods agree
        weighted_variance = np.average((scores - ensemble_score) ** 2, weights=weights)
        confidence_score = max(0.0, 1.0 - weighted_variance)

        # Determine risk level
        if ensemble_score >= self.config.high_risk_threshold:
            risk_level = "CRITICAL"
        elif ensemble_score >= self.config.fraud_threshold:
            risk_level = "HIGH"
        elif ensemble_score >= self.config.medium_risk_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Generate explanation
        explanation = self._generate_comprehensive_explanation(
            ensemble_score, component_results, risk_level, confidence_score
        )

        # Collect evidence
        evidence = self._collect_evidence(component_results)

        return FraudDetectionResult(
            claim_id=claim.get("claim_id", "unknown"),
            is_fraud=is_fraud,
            fraud_probability=ensemble_score,
            confidence_score=confidence_score,
            risk_level=risk_level,
            explanation=explanation,
            evidence=evidence,
            rule_score=rule_score,
            ml_score=ml_score,
            anomaly_score=anomaly_score,
            rule_results=component_results.get("rules", {}).get("results", []),
            ml_results=component_results.get("ml", {}),
            anomaly_results=component_results.get("anomaly", {}),
            processing_time_ms=processing_time * 1000,
            model_version=self.model_version,
            detection_timestamp=datetime.now().isoformat(),
        )

    def _generate_comprehensive_explanation(
        self,
        ensemble_score: float,
        component_results: Dict[str, Any],
        risk_level: str,
        confidence_score: float,
    ) -> str:
        """Generate comprehensive explanation for the detection result."""
        explanation_parts = []

        # Overall assessment
        if ensemble_score >= self.config.fraud_threshold:
            explanation_parts.append(
                f"FRAUD DETECTED - {risk_level} RISK (Score: {ensemble_score:.3f})"
            )
        else:
            explanation_parts.append(
                f"LEGITIMATE CLAIM - {risk_level} RISK (Score: {ensemble_score:.3f})"
            )

        explanation_parts.append(f"Confidence: {confidence_score:.2%}")

        # Component contributions
        contributions = []

        if "rules" in component_results:
            rule_score = component_results["rules"]["score"]
            if rule_score > 0.5:
                contributions.append(f"Rules flagged (Score: {rule_score:.3f})")

        if "ml" in component_results:
            ml_score = component_results["ml"]["score"]
            if ml_score > 0.5:
                contributions.append(f"ML model flagged (Score: {ml_score:.3f})")

        if "anomaly" in component_results:
            anomaly_data = component_results["anomaly"]
            if anomaly_data.get("is_anomaly", False):
                contributions.append(
                    f"Anomaly detected (Score: {anomaly_data.get('score', 0):.3f})"
                )

        if contributions:
            explanation_parts.append("Contributing factors: " + "; ".join(contributions))

        return " | ".join(explanation_parts)

    def _collect_evidence(self, component_results: Dict[str, Any]) -> List[str]:
        """Collect evidence from all detection components."""
        evidence = []

        # Rule-based evidence
        if "rules" in component_results:
            for rule_result in component_results["rules"].get("results", []):
                if rule_result.get("triggered", False):
                    evidence.extend(rule_result.get("evidence", []))

        # ML evidence
        if "ml" in component_results:
            ml_explanation = component_results["ml"].get("explanation", "")
            if ml_explanation and "FRAUD" in ml_explanation:
                evidence.append(f"ML: {ml_explanation}")

        # Anomaly evidence
        if "anomaly" in component_results:
            anomaly_data = component_results["anomaly"]
            if anomaly_data.get("is_anomaly", False):
                evidence.append(
                    f"Anomaly: {anomaly_data.get('explanation', 'Unusual pattern detected')}"
                )
                contributing_features = anomaly_data.get("contributing_features", [])
                if contributing_features:
                    evidence.append(f"Anomalous features: {', '.join(contributing_features[:3])}")

        return evidence[:10]  # Limit to top 10 pieces of evidence

    def _create_error_result(
        self, claim: Dict[str, Any], error_message: str
    ) -> FraudDetectionResult:
        """Create error result for failed claim processing."""
        return FraudDetectionResult(
            claim_id=claim.get("claim_id", "unknown"),
            is_fraud=True,  # Conservative approach
            fraud_probability=0.5,
            confidence_score=0.0,
            risk_level="MEDIUM",
            explanation=f"Processing error: {error_message}",
            evidence=[],
            rule_score=0.0,
            ml_score=0.0,
            anomaly_score=0.0,
            rule_results=[],
            ml_results={},
            anomaly_results={},
            processing_time_ms=0.0,
            model_version=self.model_version,
            detection_timestamp=datetime.now().isoformat(),
        )

    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        return self.performance_metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update detection thresholds."""
        with self._lock:
            for key, value in new_thresholds.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated {key} to {value}")

    def save_model(self, directory: str) -> None:
        """Save the complete fraud detection model."""
        import os
        import json

        os.makedirs(directory, exist_ok=True)

        # Save components
        if self.rule_engine:
            # Save rule engine (if it has a save method)
            pass

        if self.ml_manager:
            self.ml_manager.save_models(os.path.join(directory, "ml_models"))

        if self.anomaly_detector:
            self.anomaly_detector.save_detectors(os.path.join(directory, "anomaly_detectors"))

        if self.feature_engineer:
            self.feature_engineer.save_feature_engineering_pipeline(
                os.path.join(directory, "feature_pipeline.pkl")
            )

        # Save orchestrator metadata
        metadata = {
            "model_version": self.model_version,
            "config": asdict(self.config),
            "performance_metrics": self.performance_metrics,
            "feature_importance": self.feature_importance,
            "is_trained": self.is_trained,
            "save_timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(directory, "orchestrator_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Fraud detection model saved to {directory}")

    def load_model(self, directory: str) -> None:
        """Load a complete fraud detection model."""
        import os
        import json

        # Load orchestrator metadata
        with open(os.path.join(directory, "orchestrator_metadata.json"), "r") as f:
            metadata = json.load(f)

        self.model_version = metadata["model_version"]
        self.performance_metrics = metadata["performance_metrics"]
        self.feature_importance = metadata["feature_importance"]
        self.is_trained = metadata["is_trained"]

        # Recreate config
        config_data = metadata["config"]
        self.config = DetectionConfig(**config_data)

        # Load components
        if self.ml_manager and os.path.exists(os.path.join(directory, "ml_models")):
            self.ml_manager.load_models(os.path.join(directory, "ml_models"))

        if self.anomaly_detector and os.path.exists(os.path.join(directory, "anomaly_detectors")):
            self.anomaly_detector.load_detectors(os.path.join(directory, "anomaly_detectors"))

        if self.feature_engineer and os.path.exists(
            os.path.join(directory, "feature_pipeline.pkl")
        ):
            self.feature_engineer.load_feature_engineering_pipeline(
                os.path.join(directory, "feature_pipeline.pkl")
            )

        logger.info(f"Fraud detection model loaded from {directory}")

    def generate_detection_report(self, results: List[FraudDetectionResult]) -> str:
        """Generate comprehensive detection report."""
        if not results:
            return "No detection results to report."

        total_claims = len(results)
        fraud_claims = sum(1 for r in results if r.is_fraud)
        fraud_rate = fraud_claims / total_claims

        # Risk level distribution
        risk_distribution = {}
        for result in results:
            risk_distribution[result.risk_level] = risk_distribution.get(result.risk_level, 0) + 1

        # Performance metrics
        avg_processing_time = np.mean([r.processing_time_ms for r in results])
        avg_confidence = np.mean([r.confidence_score for r in results])

        report = "FRAUD DETECTION BATCH REPORT\n"
        report += "=" * 40 + "\n\n"

        report += f"Total Claims Processed: {total_claims}\n"
        report += f"Fraud Claims Detected: {fraud_claims} ({fraud_rate:.2%})\n"
        report += f"Average Processing Time: {avg_processing_time:.2f} ms\n"
        report += f"Average Confidence: {avg_confidence:.2%}\n\n"

        report += "RISK LEVEL DISTRIBUTION:\n"
        for risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = risk_distribution.get(risk_level, 0)
            percentage = count / total_claims * 100
            report += f"  {risk_level}: {count} ({percentage:.1f}%)\n"

        # High-risk claims
        high_risk_claims = [r for r in results if r.risk_level in ["CRITICAL", "HIGH"]]
        if high_risk_claims:
            report += f"\nHIGH-RISK CLAIMS ({len(high_risk_claims)}):\n"
            report += "-" * 30 + "\n"

            for result in high_risk_claims[:10]:  # Show top 10
                report += f"Claim ID: {result.claim_id}\n"
                report += f"  Risk Level: {result.risk_level}\n"
                report += f"  Fraud Probability: {result.fraud_probability:.3f}\n"
                report += f"  Confidence: {result.confidence_score:.2%}\n"
                report += f"  Explanation: {result.explanation[:100]}...\n\n"

        # Performance vs targets
        if self.performance_metrics:
            overall_perf = self.performance_metrics.get("overall_performance", {})
            report += "PERFORMANCE VS TARGETS:\n"
            report += "-" * 25 + "\n"

            accuracy = overall_perf.get("accuracy", 0)
            fpr = overall_perf.get("false_positive_rate", 0)

            report += f"Accuracy: {accuracy:.3f} (Target: ≥{self.config.target_accuracy:.3f}) "
            report += f"{'✓' if accuracy >= self.config.target_accuracy else '✗'}\n"

            report += f"False Positive Rate: {fpr:.3f} (Target: ≤{self.config.target_false_positive_rate:.3f}) "
            report += f"{'✓' if fpr <= self.config.target_false_positive_rate else '✗'}\n"

        return report
