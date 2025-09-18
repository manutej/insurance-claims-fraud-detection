"""
Machine learning models for fraud detection.

This module implements multiple ML algorithms including Random Forest,
XGBoost, Neural Networks, and ensemble methods optimized for fraud detection
with proper handling of class imbalance and performance optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import joblib
import json
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

# SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("Imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    use_class_weights: bool = True
    use_probability: bool = True
    scaling_required: bool = True


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    average_precision: float
    false_positive_rate: float
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass
class PredictionResult:
    """Result of fraud prediction."""
    prediction: int
    probability: float
    confidence: float
    explanation: str


class MLModelManager:
    """
    Machine learning model manager for fraud detection.

    Implements multiple algorithms with proper handling of:
    - Class imbalance
    - Feature scaling
    - Hyperparameter tuning
    - Model evaluation
    - Ensemble methods
    """

    def __init__(self, target_metrics: Dict[str, float] = None):
        """
        Initialize ML model manager.

        Args:
            target_metrics: Target performance metrics (accuracy >94%, FPR <3.8%)
        """
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.trained_models = {}
        self.ensemble_model = None

        # Target metrics from requirements
        self.target_metrics = target_metrics or {
            'accuracy': 0.94,
            'false_positive_rate': 0.038,
            'detection_rate_min': 0.08,
            'detection_rate_max': 0.15
        }

        self.model_configs = self._initialize_model_configs()

    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations."""
        configs = {
            'random_forest': ModelConfig(
                name='random_forest',
                algorithm='RandomForest',
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 42
                },
                use_class_weights=True,
                scaling_required=False
            ),
            'logistic_regression': ModelConfig(
                name='logistic_regression',
                algorithm='LogisticRegression',
                hyperparameters={
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'max_iter': 1000,
                    'random_state': 42
                },
                use_class_weights=True,
                scaling_required=True
            ),
            'svm': ModelConfig(
                name='svm',
                algorithm='SVM',
                hyperparameters={
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': 42
                },
                use_class_weights=True,
                scaling_required=True
            ),
            'mlp': ModelConfig(
                name='mlp',
                algorithm='MLP',
                hyperparameters={
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                    'random_state': 42
                },
                scaling_required=True
            )
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = ModelConfig(
                name='xgboost',
                algorithm='XGBoost',
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'logloss'
                },
                scaling_required=False
            )

        return configs

    def train_models(self, X: pd.DataFrame, y: pd.Series,
                    validation_split: float = 0.2) -> Dict[str, ModelPerformance]:
        """
        Train all configured models.

        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction for validation set

        Returns:
            Dictionary of model performance metrics
        """
        logger.info(f"Training models on {X.shape[0]} samples with {X.shape[1]} features")

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )

        # Handle class imbalance if library is available
        if IMBALANCED_LEARN_AVAILABLE:
            X_train, y_train = self._handle_class_imbalance(X_train, y_train)

        performance_results = {}

        # Train individual models
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name}")

                model = self._create_model(config, y_train)

                # Create pipeline with scaling if needed
                if config.scaling_required:
                    scaler = StandardScaler()
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('model', model)
                    ])
                    self.scalers[model_name] = scaler
                else:
                    pipeline = model

                # Train model
                pipeline.fit(X_train, y_train)
                self.trained_models[model_name] = pipeline

                # Evaluate model
                performance = self._evaluate_model(pipeline, X_val, y_val, model_name)
                performance_results[model_name] = performance

                logger.info(f"{model_name} - Accuracy: {performance.accuracy:.4f}, "
                           f"Precision: {performance.precision:.4f}, "
                           f"Recall: {performance.recall:.4f}, "
                           f"F1: {performance.f1_score:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # Train ensemble model
        if len(self.trained_models) > 1:
            logger.info("Training ensemble model")
            self._train_ensemble_model(X_train, y_train, X_val, y_val)

            if self.ensemble_model:
                ensemble_performance = self._evaluate_model(
                    self.ensemble_model, X_val, y_val, 'ensemble'
                )
                performance_results['ensemble'] = ensemble_performance

        # Train neural network if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            try:
                logger.info("Training neural network")
                nn_model = self._train_neural_network(X_train, y_train, X_val, y_val)
                if nn_model:
                    self.trained_models['neural_network'] = nn_model
                    nn_performance = self._evaluate_neural_network(nn_model, X_val, y_val)
                    performance_results['neural_network'] = nn_performance
            except Exception as e:
                logger.error(f"Failed to train neural network: {e}")

        logger.info(f"Training completed. {len(performance_results)} models trained.")
        return performance_results

    def _create_model(self, config: ModelConfig, y_train: pd.Series):
        """Create model instance based on configuration."""
        # Calculate class weights if needed
        class_weights = None
        if config.use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))

        # Create model based on algorithm
        if config.algorithm == 'RandomForest':
            return RandomForestClassifier(
                class_weight=class_weights,
                **config.hyperparameters
            )
        elif config.algorithm == 'LogisticRegression':
            return LogisticRegression(
                class_weight=class_weights,
                **config.hyperparameters
            )
        elif config.algorithm == 'SVM':
            return SVC(
                class_weight=class_weights,
                **config.hyperparameters
            )
        elif config.algorithm == 'MLP':
            return MLPClassifier(**config.hyperparameters)
        elif config.algorithm == 'XGBoost' and XGBOOST_AVAILABLE:
            # XGBoost handles class weights differently
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            params = config.hyperparameters.copy()
            params['scale_pos_weight'] = scale_pos_weight
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    def _handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE."""
        try:
            # Use SMOTE for oversampling minority class
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            logger.info(f"Applied SMOTE: {len(y_train)} -> {len(y_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)

        except Exception as e:
            logger.warning(f"SMOTE failed, using original data: {e}")
            return X_train, y_train

    def _evaluate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series,
                       model_name: str) -> ModelPerformance:
        """Evaluate model performance."""
        # Predictions
        y_pred = model.predict(X_val)

        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_val)
            # Convert to probabilities
            y_proba = 1 / (1 + np.exp(-y_proba))
        else:
            y_proba = y_pred.astype(float)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        try:
            roc_auc = roc_auc_score(y_val, y_proba)
            avg_precision = average_precision_score(y_val, y_proba)
        except ValueError:
            roc_auc = 0.5
            avg_precision = 0.5

        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Classification report
        report = classification_report(y_val, y_pred)

        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            false_positive_rate=fpr,
            confusion_matrix=confusion_matrix(y_val, y_pred),
            classification_report=report
        )

    def _train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Train ensemble model using voting classifier."""
        try:
            # Select best performing models for ensemble
            estimators = []

            for name, model in self.trained_models.items():
                if name in ['random_forest', 'logistic_regression', 'xgboost']:
                    estimators.append((name, model))

            if len(estimators) >= 2:
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft'  # Use probabilities
                )
                self.ensemble_model.fit(X_train, y_train)
                logger.info(f"Ensemble model created with {len(estimators)} estimators")
            else:
                logger.warning("Not enough models for ensemble")

        except Exception as e:
            logger.error(f"Failed to create ensemble model: {e}")

    def _train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series):
        """Train neural network using TensorFlow/Keras."""
        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers['neural_network'] = scaler

            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

            # Build model
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])

            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )

            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                class_weight=class_weight_dict,
                callbacks=[early_stopping],
                verbose=0
            )

            return model

        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            return None

    def _evaluate_neural_network(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> ModelPerformance:
        """Evaluate neural network performance."""
        # Scale validation data
        scaler = self.scalers['neural_network']
        X_val_scaled = scaler.transform(X_val)

        # Predictions
        y_proba = model.predict(X_val_scaled).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        avg_precision = average_precision_score(y_val, y_proba)

        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            false_positive_rate=fpr,
            confusion_matrix=confusion_matrix(y_val, y_pred),
            classification_report=classification_report(y_val, y_pred)
        )

    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble') -> List[PredictionResult]:
        """
        Make fraud predictions on new data.

        Args:
            X: Feature matrix
            model_name: Name of model to use for prediction

        Returns:
            List of prediction results
        """
        if model_name not in self.trained_models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not found")

        # Use ensemble model if available and requested
        if model_name == 'ensemble' and self.ensemble_model:
            model = self.ensemble_model
        elif model_name == 'ensemble':
            # Fall back to best individual model
            model_name = self._get_best_model()
            model = self.trained_models[model_name]
        else:
            model = self.trained_models[model_name]

        # Make predictions
        if model_name == 'neural_network':
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            probabilities = model.predict(X_scaled).flatten()
            predictions = (probabilities > 0.5).astype(int)
        else:
            predictions = model.predict(X)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = predictions.astype(float)

        # Create prediction results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = max(prob, 1 - prob)  # Distance from decision boundary
            explanation = self._generate_prediction_explanation(
                X.iloc[i], pred, prob, model_name
            )

            results.append(PredictionResult(
                prediction=int(pred),
                probability=float(prob),
                confidence=float(confidence),
                explanation=explanation
            ))

        return results

    def _get_best_model(self) -> str:
        """Get the name of the best performing model."""
        # This would be based on validation performance
        # For now, prefer random forest if available
        if 'random_forest' in self.trained_models:
            return 'random_forest'
        elif 'xgboost' in self.trained_models:
            return 'xgboost'
        else:
            return list(self.trained_models.keys())[0]

    def _generate_prediction_explanation(self, features: pd.Series, prediction: int,
                                       probability: float, model_name: str) -> str:
        """Generate explanation for prediction."""
        if prediction == 1:
            risk_level = "HIGH" if probability > 0.8 else "MEDIUM"
            explanation = f"FRAUD DETECTED ({risk_level} confidence: {probability:.2f})"
        else:
            confidence = 1 - probability
            risk_level = "LOW" if confidence > 0.8 else "MEDIUM"
            explanation = f"LEGITIMATE ({risk_level} confidence: {confidence:.2f})"

        explanation += f" [Model: {model_name}]"

        # Add key factors (simplified)
        if hasattr(self, 'feature_importance') and self.feature_importance:
            top_features = sorted(self.feature_importance.items(),
                                key=lambda x: x[1], reverse=True)[:3]
            explanation += f" Key factors: {[f[0] for f in top_features]}"

        return explanation

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           model_name: str, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.

        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to tune
            cv_folds: Number of cross-validation folds

        Returns:
            Best hyperparameters
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not configured")

        logger.info(f"Tuning hyperparameters for {model_name}")

        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            } if XGBOOST_AVAILABLE else {}
        }

        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.model_configs[model_name].hyperparameters

        # Create base model
        config = self.model_configs[model_name]
        base_model = self._create_model(config, y)

        # Perform grid search with cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Scale features if needed
        if config.scaling_required:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            grid_search.fit(X_scaled, y)
        else:
            grid_search.fit(X, y)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update model configuration
        self.model_configs[model_name].hyperparameters.update(grid_search.best_params_)

        return grid_search.best_params_

    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series,
                            cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.

        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds

        Returns:
            Cross-validation results for all models
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}

        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Cross-validating {model_name}")

                # Create model
                model = self._create_model(config, y)

                # Create pipeline with scaling if needed
                if config.scaling_required:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                else:
                    pipeline = model

                # Perform cross-validation
                scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                scores = {}

                for metric in scoring_metrics:
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
                    scores[metric] = {
                        'mean': np.mean(cv_scores),
                        'std': np.std(cv_scores),
                        'scores': cv_scores.tolist()
                    }

                cv_results[model_name] = scores

                logger.info(f"{model_name} CV results - "
                           f"Accuracy: {scores['accuracy']['mean']:.4f} ± {scores['accuracy']['std']:.4f}, "
                           f"F1: {scores['f1']['mean']:.4f} ± {scores['f1']['std']:.4f}")

            except Exception as e:
                logger.error(f"Cross-validation failed for {model_name}: {e}")
                continue

        return cv_results

    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance from trained models.

        Args:
            model_name: Specific model name, or None for ensemble

        Returns:
            Feature importance dictionary
        """
        if model_name and model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")

        # Use ensemble or best model if not specified
        if not model_name:
            if self.ensemble_model:
                model_name = 'ensemble'
            else:
                model_name = self._get_best_model()

        model = self.trained_models.get(model_name) or self.ensemble_model

        importance_dict = {}

        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            elif isinstance(model, Pipeline) and hasattr(model.named_steps['model'], 'feature_importances_'):
                # Pipeline with tree-based model
                importances = model.named_steps['model'].feature_importances_
            elif isinstance(model, VotingClassifier):
                # Ensemble model - average importances
                all_importances = []
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        all_importances.append(estimator.feature_importances_)
                    elif isinstance(estimator, Pipeline) and hasattr(estimator.named_steps['model'], 'feature_importances_'):
                        all_importances.append(estimator.named_steps['model'].feature_importances_)

                if all_importances:
                    importances = np.mean(all_importances, axis=0)
                else:
                    importances = np.ones(len(self.feature_names))
            else:
                # Default to uniform importance
                importances = np.ones(len(self.feature_names))

            # Create importance dictionary
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = float(importances[i])

        except Exception as e:
            logger.error(f"Could not extract feature importance: {e}")
            # Return uniform importance
            importance_dict = {name: 1.0 for name in self.feature_names}

        return importance_dict

    def save_models(self, directory: str) -> None:
        """Save all trained models and metadata."""
        import os
        import pickle

        os.makedirs(directory, exist_ok=True)

        # Save individual models
        for name, model in self.trained_models.items():
            if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                # Save TensorFlow model
                model.save(os.path.join(directory, f"{name}.h5"))
            else:
                # Save sklearn models
                joblib.dump(model, os.path.join(directory, f"{name}.pkl"))

        # Save ensemble model
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, os.path.join(directory, "ensemble.pkl"))

        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(directory, "scalers.pkl"))

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_configs': {name: {
                'name': config.name,
                'algorithm': config.algorithm,
                'hyperparameters': config.hyperparameters,
                'use_class_weights': config.use_class_weights,
                'scaling_required': config.scaling_required
            } for name, config in self.model_configs.items()},
            'target_metrics': self.target_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str) -> None:
        """Load trained models and metadata."""
        import os
        import pickle

        # Load metadata
        with open(os.path.join(directory, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.target_metrics = metadata['target_metrics']

        # Reconstruct model configs
        self.model_configs = {}
        for name, config_data in metadata['model_configs'].items():
            self.model_configs[name] = ModelConfig(**config_data)

        # Load individual models
        for name in metadata['model_configs'].keys():
            if name == 'neural_network':
                if TENSORFLOW_AVAILABLE and os.path.exists(os.path.join(directory, f"{name}.h5")):
                    self.trained_models[name] = keras.models.load_model(
                        os.path.join(directory, f"{name}.h5")
                    )
            else:
                model_path = os.path.join(directory, f"{name}.pkl")
                if os.path.exists(model_path):
                    self.trained_models[name] = joblib.load(model_path)

        # Load ensemble model
        ensemble_path = os.path.join(directory, "ensemble.pkl")
        if os.path.exists(ensemble_path):
            self.ensemble_model = joblib.load(ensemble_path)

        # Load scalers
        scalers_path = os.path.join(directory, "scalers.pkl")
        if os.path.exists(scalers_path):
            self.scalers = joblib.load(scalers_path)

        logger.info(f"Models loaded from {directory}")

    def check_target_metrics(self, performance_results: Dict[str, ModelPerformance]) -> Dict[str, bool]:
        """
        Check if models meet target performance metrics.

        Args:
            performance_results: Results from model evaluation

        Returns:
            Dictionary indicating which models meet targets
        """
        results = {}

        for model_name, performance in performance_results.items():
            meets_accuracy = performance.accuracy >= self.target_metrics['accuracy']
            meets_fpr = performance.false_positive_rate <= self.target_metrics['false_positive_rate']

            results[model_name] = {
                'meets_accuracy': meets_accuracy,
                'meets_fpr': meets_fpr,
                'meets_all_targets': meets_accuracy and meets_fpr,
                'actual_accuracy': performance.accuracy,
                'actual_fpr': performance.false_positive_rate
            }

        return results

    def generate_model_report(self, performance_results: Dict[str, ModelPerformance]) -> str:
        """Generate comprehensive model performance report."""
        report = "FRAUD DETECTION MODEL PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"

        # Target metrics
        report += "TARGET METRICS:\n"
        report += f"- Accuracy: >{self.target_metrics['accuracy']*100:.1f}%\n"
        report += f"- False Positive Rate: <{self.target_metrics['false_positive_rate']*100:.1f}%\n"
        report += f"- Detection Rate: {self.target_metrics['detection_rate_min']*100:.1f}%-{self.target_metrics['detection_rate_max']*100:.1f}%\n\n"

        # Individual model performance
        for model_name, performance in performance_results.items():
            report += f"MODEL: {model_name.upper()}\n"
            report += "-" * 30 + "\n"
            report += f"Accuracy: {performance.accuracy:.4f} ({performance.accuracy*100:.2f}%)\n"
            report += f"Precision: {performance.precision:.4f}\n"
            report += f"Recall: {performance.recall:.4f}\n"
            report += f"F1 Score: {performance.f1_score:.4f}\n"
            report += f"ROC AUC: {performance.roc_auc:.4f}\n"
            report += f"False Positive Rate: {performance.false_positive_rate:.4f} ({performance.false_positive_rate*100:.2f}%)\n"

            # Check if meets targets
            meets_accuracy = performance.accuracy >= self.target_metrics['accuracy']
            meets_fpr = performance.false_positive_rate <= self.target_metrics['false_positive_rate']

            report += f"Meets Accuracy Target: {'✓' if meets_accuracy else '✗'}\n"
            report += f"Meets FPR Target: {'✓' if meets_fpr else '✗'}\n"
            report += f"Overall Status: {'PASSED' if meets_accuracy and meets_fpr else 'NEEDS IMPROVEMENT'}\n\n"

        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"

        best_model = max(performance_results.items(),
                        key=lambda x: x[1].accuracy)
        report += f"Best performing model: {best_model[0]} (Accuracy: {best_model[1].accuracy:.4f})\n"

        failing_models = [name for name, perf in performance_results.items()
                         if perf.accuracy < self.target_metrics['accuracy'] or
                            perf.false_positive_rate > self.target_metrics['false_positive_rate']]

        if failing_models:
            report += f"Models needing improvement: {', '.join(failing_models)}\n"
            report += "Consider: hyperparameter tuning, feature engineering, or ensemble methods\n"

        return report