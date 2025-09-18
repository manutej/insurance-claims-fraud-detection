"""
Anomaly detection for fraud identification.

This module implements multiple anomaly detection algorithms including
Isolation Forest, Local Outlier Factor, Autoencoder, and statistical
methods to identify unusual patterns in insurance claims.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import joblib
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# Statistical methods
from scipy import stats
from scipy.spatial.distance import mahalanobis

# TensorFlow for autoencoder
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Autoencoder will not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Anomaly detection result for a single claim."""
    claim_id: str
    anomaly_score: float
    is_anomaly: bool
    method: str
    confidence: float
    explanation: str
    contributing_features: List[str]


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection algorithms."""
    contamination: float = 0.1  # Expected proportion of outliers
    random_state: int = 42
    n_jobs: int = -1
    threshold_percentile: float = 95  # Percentile threshold for statistical methods


class StatisticalAnomalyDetector:
    """Statistical anomaly detection methods."""

    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self.fitted_params = {}

    def fit(self, X: pd.DataFrame) -> None:
        """Fit statistical parameters on training data."""
        # Z-score parameters
        self.fitted_params['mean'] = X.mean()
        self.fitted_params['std'] = X.std()

        # IQR parameters
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.fitted_params['Q1'] = Q1
        self.fitted_params['Q3'] = Q3
        self.fitted_params['IQR'] = IQR

        # Modified Z-score parameters (using median)
        median = X.median()
        mad = np.median(np.abs(X - median), axis=0)
        self.fitted_params['median'] = median
        self.fitted_params['mad'] = mad

        # Mahalanobis distance parameters
        try:
            cov_matrix = np.cov(X.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            self.fitted_params['mean_vector'] = X.mean().values
            self.fitted_params['inv_cov_matrix'] = inv_cov_matrix
        except np.linalg.LinAlgError:
            logger.warning("Could not compute Mahalanobis distance parameters")
            self.fitted_params['inv_cov_matrix'] = None

    def detect_zscore_anomalies(self, X: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using Z-score method."""
        z_scores = np.abs((X - self.fitted_params['mean']) / self.fitted_params['std'])
        max_z_scores = z_scores.max(axis=1)
        return max_z_scores > threshold

    def detect_iqr_anomalies(self, X: pd.DataFrame, factor: float = 1.5) -> np.ndarray:
        """Detect anomalies using IQR method."""
        Q1 = self.fitted_params['Q1']
        Q3 = self.fitted_params['Q3']
        IQR = self.fitted_params['IQR']

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        return outliers.values

    def detect_modified_zscore_anomalies(self, X: pd.DataFrame, threshold: float = 3.5) -> np.ndarray:
        """Detect anomalies using modified Z-score (based on median)."""
        median = self.fitted_params['median']
        mad = self.fitted_params['mad']

        # Avoid division by zero
        mad_safe = np.where(mad == 0, 1e-8, mad)
        modified_z_scores = 0.6745 * np.abs((X - median) / mad_safe)
        max_modified_z_scores = modified_z_scores.max(axis=1)
        return max_modified_z_scores > threshold

    def detect_mahalanobis_anomalies(self, X: pd.DataFrame, threshold_percentile: float = 95) -> np.ndarray:
        """Detect anomalies using Mahalanobis distance."""
        if self.fitted_params['inv_cov_matrix'] is None:
            return np.zeros(len(X), dtype=bool)

        try:
            mean_vector = self.fitted_params['mean_vector']
            inv_cov_matrix = self.fitted_params['inv_cov_matrix']

            distances = []
            for _, row in X.iterrows():
                diff = row.values - mean_vector
                distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                distances.append(distance)

            distances = np.array(distances)
            threshold = np.percentile(distances, threshold_percentile)
            return distances > threshold

        except Exception as e:
            logger.error(f"Mahalanobis distance calculation failed: {e}")
            return np.zeros(len(X), dtype=bool)


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detection."""

    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

    def build_autoencoder(self, input_dim: int, encoding_dim: int = None) -> keras.Model:
        """Build autoencoder model."""
        if encoding_dim is None:
            encoding_dim = max(input_dim // 4, 10)

        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        # Autoencoder model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def fit(self, X: pd.DataFrame, validation_split: float = 0.2) -> None:
        """Train autoencoder on normal data."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available for autoencoder")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build model
        self.model = self.build_autoencoder(X_scaled.shape[1])

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )

        # Calculate reconstruction errors on training data to set threshold
        train_predictions = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - train_predictions, 2), axis=1)
        self.threshold = np.percentile(mse, self.config.threshold_percentile)

        logger.info(f"Autoencoder trained. Threshold: {self.threshold:.6f}")

    def detect_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using reconstruction error."""
        if self.model is None:
            raise RuntimeError("Autoencoder not fitted")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get reconstructions
        reconstructions = self.model.predict(X_scaled)

        # Calculate reconstruction errors
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

        # Determine anomalies
        anomalies = mse > self.threshold

        return anomalies, mse


class AnomalyDetectionSuite:
    """
    Comprehensive anomaly detection suite combining multiple algorithms.

    Implements:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - One-Class SVM
    - Statistical methods (Z-score, IQR, Modified Z-score, Mahalanobis)
    - Autoencoder (if TensorFlow available)
    - Ensemble voting
    """

    def __init__(self, config: AnomalyDetectorConfig = None):
        """Initialize anomaly detection suite."""
        self.config = config or AnomalyDetectorConfig()
        self.detectors = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.detection_thresholds = {}
        self.fitted = False

    def initialize_detectors(self, n_features: int) -> None:
        """Initialize all anomaly detection algorithms."""
        # Isolation Forest
        self.detectors['isolation_forest'] = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Local Outlier Factor
        self.detectors['lof'] = LocalOutlierFactor(
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs
        )

        # One-Class SVM
        self.detectors['one_class_svm'] = OneClassSVM(
            nu=self.config.contamination,
            gamma='scale'
        )

        # Elliptic Envelope (Robust covariance)
        self.detectors['elliptic_envelope'] = EllipticEnvelope(
            contamination=self.config.contamination,
            random_state=self.config.random_state
        )

        # DBSCAN clustering for outliers
        self.detectors['dbscan'] = DBSCAN(eps=0.5, min_samples=5)

        # Statistical methods
        self.detectors['statistical'] = StatisticalAnomalyDetector(self.config)

        # Autoencoder (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            self.detectors['autoencoder'] = AutoencoderAnomalyDetector(self.config)

        logger.info(f"Initialized {len(self.detectors)} anomaly detectors")

    def fit(self, X: pd.DataFrame, feature_names: List[str] = None) -> None:
        """
        Fit all anomaly detection algorithms.

        Args:
            X: Training data (should contain mostly normal samples)
            feature_names: Names of features
        """
        logger.info(f"Fitting anomaly detectors on {X.shape[0]} samples with {X.shape[1]} features")

        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Initialize detectors
        self.initialize_detectors(X.shape[1])

        # Scale features for algorithms that need it
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        # Fit each detector
        for name, detector in self.detectors.items():
            try:
                logger.info(f"Fitting {name}")

                if name == 'lof':
                    # LOF doesn't have separate fit method for novelty detection
                    detector.set_params(novelty=True)
                    detector.fit(X_scaled)
                elif name == 'dbscan':
                    # DBSCAN for outlier detection
                    labels = detector.fit_predict(X_scaled)
                    # Points labeled as -1 are outliers
                    outlier_ratio = np.sum(labels == -1) / len(labels)
                    logger.info(f"DBSCAN outlier ratio: {outlier_ratio:.3f}")
                elif name == 'statistical':
                    # Statistical methods
                    detector.fit(X_scaled_df)
                elif name == 'autoencoder' and TENSORFLOW_AVAILABLE:
                    # Autoencoder
                    detector.fit(X_scaled_df)
                else:
                    # Standard fit method
                    detector.fit(X_scaled)

            except Exception as e:
                logger.error(f"Failed to fit {name}: {e}")
                continue

        self.fitted = True
        logger.info("Anomaly detection suite fitted successfully")

    def detect_anomalies(self, X: pd.DataFrame) -> List[AnomalyScore]:
        """
        Detect anomalies using all fitted algorithms.

        Args:
            X: Data to analyze for anomalies

        Returns:
            List of AnomalyScore objects
        """
        if not self.fitted:
            raise RuntimeError("Anomaly detectors not fitted")

        logger.info(f"Detecting anomalies in {X.shape[0]} samples")

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        # Store results from each detector
        detection_results = {}
        anomaly_scores = {}

        # Apply each detector
        for name, detector in self.detectors.items():
            try:
                if name == 'isolation_forest':
                    predictions = detector.predict(X_scaled)
                    scores = detector.decision_function(X_scaled)
                    anomalies = predictions == -1
                    detection_results[name] = anomalies
                    anomaly_scores[name] = -scores  # Negative because IF uses negative scores for anomalies

                elif name == 'lof':
                    predictions = detector.predict(X_scaled)
                    scores = detector.decision_function(X_scaled)
                    anomalies = predictions == -1
                    detection_results[name] = anomalies
                    anomaly_scores[name] = -scores

                elif name == 'one_class_svm':
                    predictions = detector.predict(X_scaled)
                    scores = detector.decision_function(X_scaled)
                    anomalies = predictions == -1
                    detection_results[name] = anomalies
                    anomaly_scores[name] = -scores

                elif name == 'elliptic_envelope':
                    predictions = detector.predict(X_scaled)
                    anomalies = predictions == -1
                    detection_results[name] = anomalies
                    # Use Mahalanobis distance as score
                    anomaly_scores[name] = np.random.random(len(X))  # Placeholder

                elif name == 'dbscan':
                    labels = detector.fit_predict(X_scaled)
                    anomalies = labels == -1
                    detection_results[name] = anomalies
                    anomaly_scores[name] = np.where(anomalies, 1.0, 0.0)

                elif name == 'statistical':
                    # Combine multiple statistical methods
                    zscore_anomalies = detector.detect_zscore_anomalies(X_scaled_df)
                    iqr_anomalies = detector.detect_iqr_anomalies(X_scaled_df)
                    modified_zscore_anomalies = detector.detect_modified_zscore_anomalies(X_scaled_df)
                    mahalanobis_anomalies = detector.detect_mahalanobis_anomalies(X_scaled_df)

                    # Voting system for statistical methods
                    vote_count = (zscore_anomalies.astype(int) +
                                iqr_anomalies.astype(int) +
                                modified_zscore_anomalies.astype(int) +
                                mahalanobis_anomalies.astype(int))

                    anomalies = vote_count >= 2  # Majority vote
                    detection_results[name] = anomalies
                    anomaly_scores[name] = vote_count / 4.0  # Normalized vote score

                elif name == 'autoencoder' and TENSORFLOW_AVAILABLE:
                    anomalies, scores = detector.detect_anomalies(X_scaled_df)
                    detection_results[name] = anomalies
                    anomaly_scores[name] = scores

            except Exception as e:
                logger.error(f"Failed to apply {name}: {e}")
                continue

        # Combine results using ensemble voting
        ensemble_results = self._ensemble_voting(detection_results, anomaly_scores, X)

        return ensemble_results

    def _ensemble_voting(self, detection_results: Dict[str, np.ndarray],
                        anomaly_scores: Dict[str, np.ndarray],
                        X: pd.DataFrame) -> List[AnomalyScore]:
        """Combine results from multiple detectors using ensemble voting."""
        n_samples = X.shape[0]
        results = []

        for i in range(n_samples):
            # Collect votes and scores
            votes = []
            scores = []
            contributing_methods = []

            for method_name in detection_results.keys():
                if i < len(detection_results[method_name]):
                    vote = detection_results[method_name][i]
                    score = anomaly_scores[method_name][i] if method_name in anomaly_scores else 0.0

                    votes.append(vote)
                    scores.append(score)

                    if vote:
                        contributing_methods.append(method_name)

            # Calculate ensemble decision
            if votes:
                vote_ratio = np.mean(votes)
                avg_score = np.mean(scores)
                is_anomaly = vote_ratio >= 0.5  # Majority vote

                # Calculate confidence based on agreement
                confidence = max(vote_ratio, 1 - vote_ratio)

                # Generate explanation
                explanation = self._generate_anomaly_explanation(
                    i, is_anomaly, vote_ratio, contributing_methods, X
                )

                # Get contributing features (simplified)
                contributing_features = self._identify_contributing_features(
                    X.iloc[i], is_anomaly
                )

                # Get claim ID if available
                claim_id = X.index[i] if hasattr(X.index[i], '__str__') else f"claim_{i}"

                results.append(AnomalyScore(
                    claim_id=str(claim_id),
                    anomaly_score=avg_score,
                    is_anomaly=is_anomaly,
                    method='ensemble',
                    confidence=confidence,
                    explanation=explanation,
                    contributing_features=contributing_features
                ))

        return results

    def _generate_anomaly_explanation(self, index: int, is_anomaly: bool,
                                    vote_ratio: float, contributing_methods: List[str],
                                    X: pd.DataFrame) -> str:
        """Generate human-readable explanation for anomaly detection."""
        if is_anomaly:
            explanation = f"ANOMALY DETECTED (Confidence: {vote_ratio:.2f})"
            if contributing_methods:
                explanation += f" - Flagged by: {', '.join(contributing_methods)}"
        else:
            explanation = f"NORMAL (Confidence: {1-vote_ratio:.2f})"

        # Add feature-specific insights if available
        if len(self.feature_names) > 0 and index < len(X):
            sample = X.iloc[index]
            extreme_features = []

            # Find features with extreme values (simplified)
            for feature_name in self.feature_names:
                if feature_name in sample.index:
                    feature_value = sample[feature_name]
                    if hasattr(feature_value, '__abs__') and abs(feature_value) > 2:  # Simple threshold
                        extreme_features.append(feature_name)

            if extreme_features:
                explanation += f" - Extreme features: {', '.join(extreme_features[:3])}"

        return explanation

    def _identify_contributing_features(self, sample: pd.Series, is_anomaly: bool) -> List[str]:
        """Identify features contributing most to anomaly score."""
        contributing_features = []

        if not is_anomaly:
            return contributing_features

        # Simple approach: identify features with extreme values
        for feature_name in self.feature_names:
            if feature_name in sample.index:
                try:
                    value = float(sample[feature_name])
                    if abs(value) > 2:  # Simple threshold for "extreme"
                        contributing_features.append(feature_name)
                except (ValueError, TypeError):
                    continue

        # Return top 5 contributing features
        return contributing_features[:5]

    def get_method_performance(self, X: pd.DataFrame, y_true: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance of individual detection methods.

        Args:
            X: Test data
            y_true: True anomaly labels (if available)

        Returns:
            Performance metrics for each method
        """
        if not self.fitted:
            raise RuntimeError("Anomaly detectors not fitted")

        results = self.detect_anomalies(X)
        performance = {}

        if y_true is not None:
            # Group results by method
            method_predictions = {}
            for result in results:
                if result.method not in method_predictions:
                    method_predictions[result.method] = []
                method_predictions[result.method].append(result.is_anomaly)

            # Calculate metrics for each method
            for method, predictions in method_predictions.items():
                predictions = np.array(predictions)

                if len(predictions) == len(y_true):
                    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

                    performance[method] = {
                        'accuracy': accuracy_score(y_true, predictions),
                        'precision': precision_score(y_true, predictions, zero_division=0),
                        'recall': recall_score(y_true, predictions, zero_division=0),
                        'f1_score': f1_score(y_true, predictions, zero_division=0)
                    }

        return performance

    def save_detectors(self, directory: str) -> None:
        """Save fitted anomaly detectors."""
        import os
        import pickle

        os.makedirs(directory, exist_ok=True)

        # Save individual detectors (excluding neural network)
        for name, detector in self.detectors.items():
            if name == 'autoencoder' and TENSORFLOW_AVAILABLE:
                # Save autoencoder model and scaler separately
                if hasattr(detector, 'model') and detector.model:
                    detector.model.save(os.path.join(directory, f"{name}_model.h5"))
                    joblib.dump(detector.scaler, os.path.join(directory, f"{name}_scaler.pkl"))
                    joblib.dump(detector.threshold, os.path.join(directory, f"{name}_threshold.pkl"))
            else:
                joblib.dump(detector, os.path.join(directory, f"{name}.pkl"))

        # Save scaler and metadata
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

        metadata = {
            'feature_names': self.feature_names,
            'config': {
                'contamination': self.config.contamination,
                'random_state': self.config.random_state,
                'threshold_percentile': self.config.threshold_percentile
            },
            'fitted': self.fitted,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        logger.info(f"Anomaly detectors saved to {directory}")

    def load_detectors(self, directory: str) -> None:
        """Load fitted anomaly detectors."""
        import os
        import json
        import pickle

        # Load metadata
        with open(os.path.join(directory, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.fitted = metadata['fitted']

        # Reconstruct config
        config_data = metadata['config']
        self.config = AnomalyDetectorConfig(
            contamination=config_data['contamination'],
            random_state=config_data['random_state'],
            threshold_percentile=config_data['threshold_percentile']
        )

        # Load scaler
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))

        # Load individual detectors
        self.detectors = {}
        for filename in os.listdir(directory):
            if filename.endswith('.pkl') and filename != 'scaler.pkl':
                name = filename.replace('.pkl', '')
                if not name.endswith(('_scaler', '_threshold')):
                    self.detectors[name] = joblib.load(os.path.join(directory, filename))

        # Load autoencoder if available
        if TENSORFLOW_AVAILABLE:
            autoencoder_model_path = os.path.join(directory, "autoencoder_model.h5")
            autoencoder_scaler_path = os.path.join(directory, "autoencoder_scaler.pkl")
            autoencoder_threshold_path = os.path.join(directory, "autoencoder_threshold.pkl")

            if (os.path.exists(autoencoder_model_path) and
                os.path.exists(autoencoder_scaler_path) and
                os.path.exists(autoencoder_threshold_path)):

                autoencoder_detector = AutoencoderAnomalyDetector(self.config)
                autoencoder_detector.model = keras.models.load_model(autoencoder_model_path)
                autoencoder_detector.scaler = joblib.load(autoencoder_scaler_path)
                autoencoder_detector.threshold = joblib.load(autoencoder_threshold_path)

                self.detectors['autoencoder'] = autoencoder_detector

        logger.info(f"Anomaly detectors loaded from {directory}")

    def generate_anomaly_report(self, anomaly_results: List[AnomalyScore]) -> str:
        """Generate comprehensive anomaly detection report."""
        total_samples = len(anomaly_results)
        anomaly_count = sum(1 for result in anomaly_results if result.is_anomaly)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0

        report = "ANOMALY DETECTION REPORT\n"
        report += "=" * 30 + "\n\n"

        report += f"Total Samples Analyzed: {total_samples}\n"
        report += f"Anomalies Detected: {anomaly_count}\n"
        report += f"Anomaly Rate: {anomaly_rate:.2%}\n\n"

        # High confidence anomalies
        high_confidence_anomalies = [
            result for result in anomaly_results
            if result.is_anomaly and result.confidence > 0.8
        ]

        if high_confidence_anomalies:
            report += f"HIGH CONFIDENCE ANOMALIES ({len(high_confidence_anomalies)}):\n"
            report += "-" * 40 + "\n"

            for result in high_confidence_anomalies[:10]:  # Show top 10
                report += f"Claim ID: {result.claim_id}\n"
                report += f"  Score: {result.anomaly_score:.4f}\n"
                report += f"  Confidence: {result.confidence:.2%}\n"
                report += f"  Explanation: {result.explanation}\n"
                if result.contributing_features:
                    report += f"  Key Features: {', '.join(result.contributing_features[:3])}\n"
                report += "\n"

        # Summary statistics
        if anomaly_results:
            scores = [result.anomaly_score for result in anomaly_results]
            confidences = [result.confidence for result in anomaly_results]

            report += "STATISTICS:\n"
            report += "-" * 15 + "\n"
            report += f"Average Anomaly Score: {np.mean(scores):.4f}\n"
            report += f"Max Anomaly Score: {np.max(scores):.4f}\n"
            report += f"Average Confidence: {np.mean(confidences):.2%}\n"

        return report