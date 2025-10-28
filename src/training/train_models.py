"""
Model training pipeline for fraud detection.

This module provides a comprehensive training pipeline including data preparation,
feature selection, hyperparameter tuning, and model persistence with performance
monitoring and optimization.
"""

import logging
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import joblib

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    learning_curve,
    validation_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Import project modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from detection.feature_engineering import FeatureEngineer, FeatureSet
from detection.ml_models import MLModelManager, ModelConfig, ModelPerformance
from detection.anomaly_detector import AnomalyDetectionSuite, AnomalyDetectorConfig
from detection.fraud_detector import FraudDetectorOrchestrator, DetectionConfig

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Plotting features disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Data splits
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42

    # Feature engineering
    feature_sets: List[str] = None
    top_k_features: int = 50
    feature_selection_method: str = "importance"  # 'importance', 'correlation', 'mutual_info'

    # Model training
    enable_hyperparameter_tuning: bool = True
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    n_jobs: int = -1

    # Performance targets
    target_accuracy: float = 0.94
    target_false_positive_rate: float = 0.038
    target_processing_time_ms: float = 100.0

    # Output settings
    output_directory: str = "models"
    save_intermediate_results: bool = True
    generate_plots: bool = True
    save_training_data: bool = False

    def __post_init__(self):
        if self.feature_sets is None:
            self.feature_sets = ["basic", "temporal", "network", "statistical"]


@dataclass
class TrainingResults:
    """Results from training pipeline."""

    overall_performance: Dict[str, float]
    model_performances: Dict[str, ModelPerformance]
    feature_importance: Dict[str, float]
    selected_features: List[str]
    training_time_seconds: float
    best_model: str
    meets_targets: bool
    recommendations: List[str]


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline for fraud detection.

    Handles:
    - Data loading and preprocessing
    - Feature engineering and selection
    - Model training and evaluation
    - Hyperparameter tuning
    - Performance monitoring
    - Model persistence
    """

    def __init__(self, config: TrainingConfig = None):
        """Initialize training pipeline."""
        self.config = config or TrainingConfig()
        self.feature_engineer = FeatureEngineer()
        self.ml_manager = MLModelManager()
        self.anomaly_detector = AnomalyDetectionSuite()

        # Training state
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.feature_set = None
        self.selected_features = None
        self.training_results = None

        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)

    def load_data(
        self, data_path: str = None, claims_data: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load training data from file or list.

        Args:
            data_path: Path to data file (JSON/CSV)
            claims_data: List of claim dictionaries

        Returns:
            Loaded DataFrame
        """
        logger.info("Loading training data")

        if claims_data:
            df = pd.DataFrame(claims_data)
        elif data_path:
            if data_path.endswith(".json"):
                with open(data_path, "r") as f:
                    data = json.load(f)
                    if "claims" in data:
                        df = pd.DataFrame(data["claims"])
                    else:
                        df = pd.DataFrame(data)
            elif data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            raise ValueError("Either data_path or claims_data must be provided")

        # Validate required columns
        if "fraud_indicator" not in df.columns:
            raise ValueError("Data must contain 'fraud_indicator' column")

        logger.info(f"Loaded {len(df)} claims")
        logger.info(f"Fraud rate: {df['fraud_indicator'].mean():.2%}")

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by splitting into train/validation/test sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data splits")

        # Ensure we have both fraud and non-fraud cases
        fraud_count = df["fraud_indicator"].sum()
        total_count = len(df)

        logger.info(
            f"Dataset composition: {fraud_count} fraud, {total_count - fraud_count} legitimate"
        )

        if fraud_count == 0:
            raise ValueError("No fraud cases in dataset")
        if fraud_count == total_count:
            raise ValueError("No legitimate cases in dataset")

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            stratify=df["fraud_indicator"],
            random_state=self.config.random_state,
        )

        # Second split: separate validation from training
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df["fraud_indicator"],
            random_state=self.config.random_state,
        )

        logger.info(
            f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        self.training_data = train_df
        self.validation_data = val_df
        self.test_data = test_df

        return train_df, val_df, test_df

    def engineer_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineer features for all data splits.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data

        Returns:
            Feature matrices for train, validation, and test sets
        """
        logger.info("Engineering features")

        # Extract features from training data
        train_claims = train_df.to_dict("records")
        self.feature_set = self.feature_engineer.extract_features(
            train_claims, target_col="fraud_indicator"
        )

        # Combine features for training set
        X_train = self.feature_engineer.combine_features(
            self.feature_set, include_sets=self.config.feature_sets
        )

        # Apply same feature engineering to validation and test sets
        val_claims = val_df.to_dict("records")
        val_feature_set = self.feature_engineer.extract_features(val_claims)
        X_val = self.feature_engineer.combine_features(
            val_feature_set, include_sets=self.config.feature_sets
        )

        test_claims = test_df.to_dict("records")
        test_feature_set = self.feature_engineer.extract_features(test_claims)
        X_test = self.feature_engineer.combine_features(
            test_feature_set, include_sets=self.config.feature_sets
        )

        # Ensure consistent feature sets
        common_features = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
        common_features = list(common_features)

        X_train = X_train[common_features]
        X_val = X_val[common_features]
        X_test = X_test[common_features]

        logger.info(f"Engineered {len(common_features)} features")

        return X_train, X_val, X_test

    def select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select best features for training.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            X_test: Test features

        Returns:
            Feature-selected matrices
        """
        logger.info(f"Selecting top {self.config.top_k_features} features")

        if self.config.feature_selection_method == "importance":
            # Use feature importance from random forest
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            rf.fit(X_train, y_train)

            # Get feature importance
            feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
            selected_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in selected_features[: self.config.top_k_features]]

        elif self.config.feature_selection_method == "correlation":
            # Use correlation with target
            correlations = X_train.corrwith(y_train).abs()
            selected_features = correlations.nlargest(self.config.top_k_features).index.tolist()

        elif self.config.feature_selection_method == "mutual_info":
            # Use mutual information
            from sklearn.feature_selection import mutual_info_classif

            mi_scores = mutual_info_classif(X_train, y_train)
            mi_scores_dict = dict(zip(X_train.columns, mi_scores))
            selected_features = sorted(mi_scores_dict.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in selected_features[: self.config.top_k_features]]

        else:
            # Use all features
            selected_features = list(X_train.columns)

        self.selected_features = selected_features

        logger.info(
            f"Selected {len(selected_features)} features using {self.config.feature_selection_method}"
        )

        return X_train[selected_features], X_val[selected_features], X_test[selected_features]

    def train_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, ModelPerformance]:
        """
        Train all machine learning models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Model performance results
        """
        logger.info("Training machine learning models")

        # Set target metrics
        self.ml_manager.target_metrics = {
            "accuracy": self.config.target_accuracy,
            "false_positive_rate": self.config.target_false_positive_rate,
        }

        # Train models
        performance_results = self.ml_manager.train_models(
            X_train, y_train, validation_split=0.0  # We already have validation set
        )

        # Evaluate on validation set
        for model_name in self.ml_manager.trained_models.keys():
            try:
                model = self.ml_manager.trained_models[model_name]
                val_performance = self.ml_manager._evaluate_model(model, X_val, y_val, model_name)
                performance_results[f"{model_name}_val"] = val_performance
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on validation set: {e}")

        return performance_results

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for all models.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Best hyperparameters for each model
        """
        if not self.config.enable_hyperparameter_tuning:
            logger.info("Hyperparameter tuning disabled")
            return {}

        logger.info("Performing hyperparameter tuning")

        best_params = {}
        for model_name in self.ml_manager.model_configs.keys():
            try:
                logger.info(f"Tuning {model_name}")
                params = self.ml_manager.tune_hyperparameters(
                    X_train, y_train, model_name, cv_folds=self.config.cv_folds
                )
                best_params[model_name] = params
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")

        return best_params

    def train_anomaly_detector(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train anomaly detection models on normal samples.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Training anomaly detection models")

        # Use only non-fraud samples for training
        normal_samples = X_train[y_train == 0]

        if len(normal_samples) == 0:
            logger.warning("No normal samples available for anomaly detection training")
            return

        try:
            self.anomaly_detector.fit(normal_samples, list(X_train.columns))
            logger.info(f"Anomaly detector trained on {len(normal_samples)} normal samples")
        except Exception as e:
            logger.error(f"Anomaly detector training failed: {e}")

    def evaluate_final_performance(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate final performance on test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Final performance metrics
        """
        logger.info("Evaluating final performance on test set")

        # Create orchestrator for ensemble evaluation
        detection_config = DetectionConfig(
            target_accuracy=self.config.target_accuracy,
            target_false_positive_rate=self.config.target_false_positive_rate,
            feature_sets=self.config.feature_sets,
            top_k_features=self.config.top_k_features,
        )

        orchestrator = FraudDetectorOrchestrator(detection_config)
        orchestrator.ml_manager = self.ml_manager
        orchestrator.anomaly_detector = self.anomaly_detector
        orchestrator.feature_engineer = self.feature_engineer
        orchestrator.is_trained = True
        orchestrator.feature_importance = self.ml_manager.get_feature_importance()

        # Process test claims
        test_claims = []
        for i, (_, row) in enumerate(self.test_data.iterrows()):
            claim = row.to_dict()
            test_claims.append(claim)

        # Get predictions
        results = orchestrator.detect_batch(test_claims)

        # Calculate metrics
        y_pred = [result.is_fraud for result in results]
        y_proba = [result.fraud_probability for result in results]

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        performance = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Calculate false positive rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        performance["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate processing time
        processing_times = [result.processing_time_ms for result in results]
        performance["avg_processing_time_ms"] = np.mean(processing_times)

        return performance

    def run_full_pipeline(
        self, data_path: str = None, claims_data: List[Dict[str, Any]] = None
    ) -> TrainingResults:
        """
        Run the complete training pipeline.

        Args:
            data_path: Path to training data
            claims_data: Training data as list of dictionaries

        Returns:
            Complete training results
        """
        logger.info("Starting full training pipeline")
        start_time = time.time()

        try:
            # 1. Load data
            df = self.load_data(data_path, claims_data)

            # 2. Prepare data splits
            train_df, val_df, test_df = self.prepare_data(df)

            # 3. Engineer features
            X_train, X_val, X_test = self.engineer_features(train_df, val_df, test_df)
            y_train = train_df["fraud_indicator"]
            y_val = val_df["fraud_indicator"]
            y_test = test_df["fraud_indicator"]

            # 4. Select features
            X_train_selected, X_val_selected, X_test_selected = self.select_features(
                X_train, y_train, X_val, X_test
            )

            # 5. Hyperparameter tuning (optional)
            best_params = self.tune_hyperparameters(X_train_selected, y_train)

            # 6. Train ML models
            model_performances = self.train_models(X_train_selected, y_train, X_val_selected, y_val)

            # 7. Train anomaly detector
            self.train_anomaly_detector(X_train_selected, y_train)

            # 8. Final evaluation
            final_performance = self.evaluate_final_performance(X_test_selected, y_test)

            # 9. Generate results
            training_time = time.time() - start_time
            feature_importance = self.ml_manager.get_feature_importance()

            # Determine best model
            best_model = max(
                model_performances.items(),
                key=lambda x: x[1].accuracy if hasattr(x[1], "accuracy") else 0,
            )[0]

            # Check if targets are met
            meets_targets = (
                final_performance["accuracy"] >= self.config.target_accuracy
                and final_performance["false_positive_rate"]
                <= self.config.target_false_positive_rate
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(final_performance, model_performances)

            # Create results object
            self.training_results = TrainingResults(
                overall_performance=final_performance,
                model_performances=model_performances,
                feature_importance=feature_importance,
                selected_features=self.selected_features,
                training_time_seconds=training_time,
                best_model=best_model,
                meets_targets=meets_targets,
                recommendations=recommendations,
            )

            # 10. Save results
            self.save_results()

            # 11. Generate plots (optional)
            if self.config.generate_plots and PLOTTING_AVAILABLE:
                self.generate_plots(X_test_selected, y_test)

            logger.info(f"Training pipeline completed in {training_time:.2f} seconds")
            logger.info(f"Final accuracy: {final_performance['accuracy']:.4f}")
            logger.info(f"Final FPR: {final_performance['false_positive_rate']:.4f}")
            logger.info(f"Meets targets: {meets_targets}")

            return self.training_results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

    def _generate_recommendations(
        self, final_performance: Dict[str, float], model_performances: Dict[str, ModelPerformance]
    ) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []

        # Performance recommendations
        if final_performance["accuracy"] < self.config.target_accuracy:
            recommendations.append(
                f"Accuracy ({final_performance['accuracy']:.3f}) below target ({self.config.target_accuracy:.3f}). Consider feature engineering or ensemble methods."
            )

        if final_performance["false_positive_rate"] > self.config.target_false_positive_rate:
            recommendations.append(
                f"False positive rate ({final_performance['false_positive_rate']:.3f}) above target ({self.config.target_false_positive_rate:.3f}). Adjust classification thresholds."
            )

        if final_performance["avg_processing_time_ms"] > self.config.target_processing_time_ms:
            recommendations.append(
                f"Processing time ({final_performance['avg_processing_time_ms']:.1f}ms) above target. Consider feature reduction or model optimization."
            )

        # Model-specific recommendations
        best_accuracy = max(
            [p.accuracy for p in model_performances.values() if hasattr(p, "accuracy")]
        )
        if best_accuracy < 0.90:
            recommendations.append(
                "Overall model performance is low. Consider additional feature engineering or more training data."
            )

        # Feature recommendations
        if len(self.selected_features) < 20:
            recommendations.append("Feature count is low. Consider additional feature engineering.")

        if not recommendations:
            recommendations.append("Performance targets met. Model is ready for deployment.")

        return recommendations

    def save_results(self) -> None:
        """Save training results and models."""
        logger.info("Saving training results")

        # Save models
        self.ml_manager.save_models(os.path.join(self.config.output_directory, "ml_models"))

        if self.anomaly_detector:
            self.anomaly_detector.save_detectors(
                os.path.join(self.config.output_directory, "anomaly_detectors")
            )

        # Save feature engineering pipeline
        self.feature_engineer.save_feature_engineering_pipeline(
            os.path.join(self.config.output_directory, "feature_pipeline.pkl")
        )

        # Save training results
        if self.training_results:
            results_dict = asdict(self.training_results)
            # Convert model performances to serializable format
            serializable_performances = {}
            for name, perf in results_dict["model_performances"].items():
                if hasattr(perf, "__dict__"):
                    serializable_performances[name] = perf.__dict__
                else:
                    serializable_performances[name] = perf
            results_dict["model_performances"] = serializable_performances

            with open(
                os.path.join(self.config.output_directory, "training_results.json"), "w"
            ) as f:
                json.dump(results_dict, f, indent=2, default=str)

        # Save configuration
        config_dict = asdict(self.config)
        with open(os.path.join(self.config.output_directory, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save training data summaries (if enabled)
        if self.config.save_training_data:
            training_summary = {
                "total_samples": len(self.training_data) if self.training_data is not None else 0,
                "fraud_rate": (
                    self.training_data["fraud_indicator"].mean()
                    if self.training_data is not None
                    else 0
                ),
                "feature_count": len(self.selected_features) if self.selected_features else 0,
                "selected_features": self.selected_features or [],
            }

            with open(os.path.join(self.config.output_directory, "data_summary.json"), "w") as f:
                json.dump(training_summary, f, indent=2)

        logger.info(f"Results saved to {self.config.output_directory}")

    def generate_plots(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Generate training and evaluation plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return

        logger.info("Generating training plots")

        plot_dir = os.path.join(self.config.output_directory, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        try:
            # Feature importance plot
            if self.training_results and self.training_results.feature_importance:
                plt.figure(figsize=(12, 8))
                importance_items = list(self.training_results.feature_importance.items())
                importance_items.sort(key=lambda x: x[1], reverse=True)
                top_features = importance_items[:20]

                features, importances = zip(*top_features)
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.xlabel("Feature Importance")
                plt.title("Top 20 Feature Importances")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, "feature_importance.png"), dpi=300, bbox_inches="tight"
                )
                plt.close()

            # Model performance comparison
            if self.training_results and self.training_results.model_performances:
                plt.figure(figsize=(12, 6))
                models = []
                accuracies = []
                fprs = []

                for name, perf in self.training_results.model_performances.items():
                    if hasattr(perf, "accuracy") and hasattr(perf, "false_positive_rate"):
                        models.append(name)
                        accuracies.append(perf.accuracy)
                        fprs.append(perf.false_positive_rate)

                x = np.arange(len(models))
                width = 0.35

                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Accuracy bars
                ax1.bar(x - width / 2, accuracies, width, label="Accuracy", alpha=0.8)
                ax1.set_ylabel("Accuracy")
                ax1.set_ylim(0, 1)

                # FPR bars (on secondary axis)
                ax2 = ax1.twinx()
                ax2.bar(
                    x + width / 2, fprs, width, label="False Positive Rate", alpha=0.8, color="red"
                )
                ax2.set_ylabel("False Positive Rate")

                # Target lines
                ax1.axhline(
                    y=self.config.target_accuracy,
                    color="blue",
                    linestyle="--",
                    alpha=0.7,
                    label="Target Accuracy",
                )
                ax2.axhline(
                    y=self.config.target_false_positive_rate,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Target FPR",
                )

                ax1.set_xlabel("Models")
                ax1.set_xticks(x)
                ax1.set_xticklabels(models, rotation=45)
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")

                plt.title("Model Performance Comparison")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, "model_comparison.png"), dpi=300, bbox_inches="tight"
                )
                plt.close()

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

    def generate_training_report(self) -> str:
        """Generate comprehensive training report."""
        if not self.training_results:
            return "No training results available."

        report = "FRAUD DETECTION MODEL TRAINING REPORT\n"
        report += "=" * 50 + "\n\n"

        # Training overview
        report += "TRAINING OVERVIEW:\n"
        report += f"Training Time: {self.training_results.training_time_seconds:.2f} seconds\n"
        report += f"Best Model: {self.training_results.best_model}\n"
        report += f"Selected Features: {len(self.training_results.selected_features)}\n"
        report += (
            f"Meets Performance Targets: {'✓' if self.training_results.meets_targets else '✗'}\n\n"
        )

        # Final performance
        perf = self.training_results.overall_performance
        report += "FINAL PERFORMANCE (Test Set):\n"
        report += f"Accuracy: {perf['accuracy']:.4f} (Target: ≥{self.config.target_accuracy:.4f})\n"
        report += f"Precision: {perf['precision']:.4f}\n"
        report += f"Recall: {perf['recall']:.4f}\n"
        report += f"F1 Score: {perf['f1_score']:.4f}\n"
        report += f"ROC AUC: {perf['roc_auc']:.4f}\n"
        report += f"False Positive Rate: {perf['false_positive_rate']:.4f} (Target: ≤{self.config.target_false_positive_rate:.4f})\n"
        report += f"Avg Processing Time: {perf['avg_processing_time_ms']:.2f} ms\n\n"

        # Individual model performance
        report += "INDIVIDUAL MODEL PERFORMANCE:\n"
        for name, model_perf in self.training_results.model_performances.items():
            if hasattr(model_perf, "accuracy"):
                report += f"{name}: Accuracy={model_perf.accuracy:.4f}, FPR={model_perf.false_positive_rate:.4f}\n"

        report += "\n"

        # Top features
        if self.training_results.feature_importance:
            report += "TOP 10 FEATURES:\n"
            sorted_features = sorted(
                self.training_results.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                report += f"{i+1:2d}. {feature}: {importance:.4f}\n"
            report += "\n"

        # Recommendations
        if self.training_results.recommendations:
            report += "RECOMMENDATIONS:\n"
            for i, rec in enumerate(self.training_results.recommendations):
                report += f"{i+1}. {rec}\n"

        return report


def main():
    """Main function for running training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--config", type=str, help="Path to training configuration JSON")
    parser.add_argument("--output", type=str, default="models", help="Output directory")

    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_data = json.load(f)
        config = TrainingConfig(**config_data)
    else:
        config = TrainingConfig()

    config.output_directory = args.output

    # Run training
    pipeline = ModelTrainingPipeline(config)
    results = pipeline.run_full_pipeline(data_path=args.data)

    # Print report
    report = pipeline.generate_training_report()
    print(report)

    # Save report
    with open(os.path.join(config.output_directory, "training_report.txt"), "w") as f:
        f.write(report)

    print(f"\nTraining completed. Results saved to {config.output_directory}")


if __name__ == "__main__":
    main()
