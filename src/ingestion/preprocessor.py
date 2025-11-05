"""
Data preprocessing module for insurance claims.

Handles normalization, feature extraction, categorical encoding, and missing data
for machine learning model preparation.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from collections import defaultdict
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from ..models.claim_models import (
    BaseClaim,
    MedicalClaim,
    PharmacyClaim,
    NoFaultClaim,
    ClaimType,
    FraudType,
)

logger = logging.getLogger(__name__)


class ClaimPreprocessor:
    """Comprehensive preprocessor for insurance claims data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Optional preprocessing configuration
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_columns = []
        self.is_fitted = False

        # Configuration defaults
        self.normalize_amounts = self.config.get("normalize_amounts", True)
        self.extract_temporal_features = self.config.get("extract_temporal_features", True)
        self.handle_missing_data = self.config.get("handle_missing_data", True)
        self.encoding_strategy = self.config.get(
            "encoding_strategy", "onehot"
        )  # 'onehot', 'label', 'target'

    def preprocess_claims(self, claims: List[BaseClaim]) -> pd.DataFrame:
        """
        Preprocess claims for machine learning.

        Args:
            claims: List of validated claim objects

        Returns:
            Preprocessed DataFrame ready for ML models
        """
        logger.info(f"Preprocessing {len(claims)} claims")

        # Convert claims to DataFrame
        df = self._claims_to_dataframe(claims)

        # Extract features
        df = self._extract_basic_features(df)
        df = self._extract_temporal_features(df)
        df = self._extract_provider_features(df)
        df = self._extract_amount_features(df)
        df = self._extract_medical_features(df)

        # Handle missing data
        if self.handle_missing_data:
            df = self._handle_missing_data(df)

        # Encode categorical variables
        df = self._encode_categorical_features(df)

        # Normalize numerical features
        if self.normalize_amounts:
            df = self._normalize_features(df)

        # Store feature columns for future use
        self.feature_columns = [
            col for col in df.columns if col not in ["claim_id", "fraud_indicator"]
        ]
        self.is_fitted = True

        logger.info(f"Preprocessing complete. Features extracted: {len(self.feature_columns)}")
        return df

    def _claims_to_dataframe(self, claims: List[BaseClaim]) -> pd.DataFrame:
        """Convert claims objects to pandas DataFrame."""
        data = []

        for claim in claims:
            # Base claim data
            claim_dict = {
                "claim_id": claim.claim_id,
                "patient_id": claim.patient_id,
                "provider_id": claim.provider_id,
                "provider_npi": claim.provider_npi,
                "date_of_service": claim.date_of_service,
                "billed_amount": float(claim.billed_amount),
                "claim_type": claim.claim_type,
                "fraud_indicator": claim.fraud_indicator,
                "fraud_type": claim.fraud_type.value if claim.fraud_type else None,
                "red_flags_count": len(claim.red_flags),
                "has_notes": claim.notes is not None and len(claim.notes) > 0,
            }

            # Type-specific data
            if isinstance(claim, MedicalClaim):
                claim_dict.update(
                    {
                        "diagnosis_count": len(claim.diagnosis_codes),
                        "procedure_count": len(claim.procedure_codes),
                        "primary_diagnosis": (
                            claim.diagnosis_codes[0] if claim.diagnosis_codes else None
                        ),
                        "primary_procedure": (
                            claim.procedure_codes[0] if claim.procedure_codes else None
                        ),
                        "service_location": claim.service_location,
                        "rendering_hours": (
                            float(claim.rendering_hours) if claim.rendering_hours else None
                        ),
                        "day_of_week": claim.day_of_week,
                    }
                )

            elif isinstance(claim, PharmacyClaim):
                claim_dict.update(
                    {
                        "ndc_code": claim.ndc_code,
                        "drug_name": claim.drug_name,
                        "quantity": claim.quantity,
                        "days_supply": claim.days_supply,
                        "prescriber_npi": claim.prescriber_npi,
                        "pharmacy_npi": claim.pharmacy_npi,
                        "fill_date": claim.fill_date,
                    }
                )

            elif isinstance(claim, NoFaultClaim):
                claim_dict.update(
                    {
                        "accident_date": claim.accident_date,
                        "vehicle_year": claim.vehicle_year,
                        "vehicle_make": claim.vehicle_make,
                        "attorney_involved": claim.attorney_involved,
                        "estimated_damage": (
                            float(claim.estimated_damage) if claim.estimated_damage else None
                        ),
                    }
                )

            data.append(claim_dict)

        return pd.DataFrame(data)

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from claims data."""
        logger.debug("Extracting basic features")

        # Amount-based features
        df["amount_log"] = np.log1p(df["billed_amount"])
        df["amount_rounded"] = (df["billed_amount"] % 100 == 0).astype(int)
        df["amount_bin"] = pd.cut(
            df["billed_amount"],
            bins=[0, 100, 500, 1000, 5000, float("inf")],
            labels=["low", "medium", "high", "very_high", "extreme"],
        )

        # Provider features
        df["provider_prefix"] = df["provider_id"].str.split("-").str[1]
        df["npi_last_digit"] = df["provider_npi"].str[-1].astype(int)

        # Patient features
        df["patient_prefix"] = df["patient_id"].str.split("-").str[1]

        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from dates."""
        if not self.extract_temporal_features:
            return df

        logger.debug("Extracting temporal features")

        # Convert date columns
        date_columns = ["date_of_service", "accident_date", "fill_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

                # Extract temporal components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df[f"{col}_is_weekend"] = (df[col].dt.weekday >= 5).astype(int)
                df[f"{col}_quarter"] = df[col].dt.quarter

                # Days since epoch (for trend analysis)
                epoch = pd.Timestamp("2020-01-01")
                df[f"{col}_days_since_epoch"] = (df[col] - epoch).dt.days

        # Calculate time differences
        if "accident_date" in df.columns and "date_of_service" in df.columns:
            df["days_accident_to_service"] = (df["date_of_service"] - df["accident_date"]).dt.days

        return df

    def _extract_provider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract provider-based features."""
        logger.debug("Extracting provider features")

        # Provider activity metrics
        provider_stats = (
            df.groupby("provider_npi")
            .agg(
                {
                    "claim_id": "count",
                    "billed_amount": ["mean", "std", "sum"],
                    "fraud_indicator": "mean",
                }
            )
            .reset_index()
        )

        provider_stats.columns = [
            "provider_npi",
            "provider_claim_count",
            "provider_avg_amount",
            "provider_amount_std",
            "provider_total_amount",
            "provider_fraud_rate",
        ]

        df = df.merge(provider_stats, on="provider_npi", how="left")

        # Provider risk scoring
        df["provider_risk_score"] = (
            df["provider_fraud_rate"] * 0.4
            + (df["provider_avg_amount"] / df["provider_avg_amount"].max()) * 0.3
            + (df["provider_claim_count"] / df["provider_claim_count"].max()) * 0.3
        )

        # Suspicious provider patterns
        df["provider_suspicious_npi"] = df["provider_npi"].str.startswith("9999999").astype(int)
        df["provider_test_id"] = df["provider_id"].str.contains("FRAUD|TEST", na=False).astype(int)

        return df

    def _extract_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amount-based statistical features."""
        logger.debug("Extracting amount features")

        # Amount percentiles
        amount_percentiles = df["billed_amount"].quantile([0.25, 0.5, 0.75, 0.9, 0.95])

        df["amount_below_25th"] = (df["billed_amount"] <= amount_percentiles[0.25]).astype(int)
        df["amount_above_90th"] = (df["billed_amount"] >= amount_percentiles[0.9]).astype(int)
        df["amount_above_95th"] = (df["billed_amount"] >= amount_percentiles[0.95]).astype(int)

        # Amount deviation from mean
        mean_amount = df["billed_amount"].mean()
        std_amount = df["billed_amount"].std()
        df["amount_zscore"] = (df["billed_amount"] - mean_amount) / std_amount
        df["amount_outlier"] = (np.abs(df["amount_zscore"]) > 3).astype(int)

        return df

    def _extract_medical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract medical-specific features."""
        logger.debug("Extracting medical features")

        # Medical claim features
        if "diagnosis_count" in df.columns:
            df["multiple_diagnoses"] = (df["diagnosis_count"] > 1).astype(int)
            df["many_procedures"] = (df["procedure_count"] > 5).astype(int)
            df["procedure_to_diagnosis_ratio"] = df["procedure_count"] / (df["diagnosis_count"] + 1)

        # Service location features
        if "service_location" in df.columns:
            emergency_locations = ["23"]  # Emergency room
            office_locations = ["11"]  # Office

            df["emergency_service"] = df["service_location"].isin(emergency_locations).astype(int)
            df["office_service"] = df["service_location"].isin(office_locations).astype(int)

        # Pharmacy features
        if "days_supply" in df.columns:
            df["long_supply"] = (df["days_supply"] > 30).astype(int)
            df["excessive_supply"] = (df["days_supply"] > 90).astype(int)

        if "quantity" in df.columns:
            df["high_quantity"] = (df["quantity"] > 100).astype(int)

        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using various imputation strategies."""
        logger.debug("Handling missing data")

        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove target and ID columns from imputation
        numerical_cols = [col for col in numerical_cols if col not in ["fraud_indicator"]]
        categorical_cols = [col for col in categorical_cols if col not in ["claim_id"]]

        # Impute numerical features
        if numerical_cols:
            if "numerical_imputer" not in self.imputers:
                self.imputers["numerical_imputer"] = SimpleImputer(strategy="median")
                df[numerical_cols] = self.imputers["numerical_imputer"].fit_transform(
                    df[numerical_cols]
                )
            else:
                df[numerical_cols] = self.imputers["numerical_imputer"].transform(
                    df[numerical_cols]
                )

        # Impute categorical features
        if categorical_cols:
            if "categorical_imputer" not in self.imputers:
                self.imputers["categorical_imputer"] = SimpleImputer(strategy="most_frequent")
                df[categorical_cols] = self.imputers["categorical_imputer"].fit_transform(
                    df[categorical_cols]
                )
            else:
                df[categorical_cols] = self.imputers["categorical_imputer"].transform(
                    df[categorical_cols]
                )

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.debug("Encoding categorical features")

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ["claim_id"]]

        if self.encoding_strategy == "onehot":
            df = self._onehot_encode(df, categorical_cols)
        elif self.encoding_strategy == "label":
            df = self._label_encode(df, categorical_cols)

        return df

    def _onehot_encode(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Apply one-hot encoding to categorical features."""
        for col in categorical_cols:
            if col in df.columns:
                encoder_key = f"onehot_{col}"

                if encoder_key not in self.encoders:
                    self.encoders[encoder_key] = OneHotEncoder(
                        sparse=False, drop="first", handle_unknown="ignore"
                    )
                    encoded = self.encoders[encoder_key].fit_transform(df[[col]])
                else:
                    encoded = self.encoders[encoder_key].transform(df[[col]])

                # Create column names
                feature_names = [
                    f"{col}_{cat}" for cat in self.encoders[encoder_key].categories_[0][1:]
                ]

                # Add encoded columns
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col])

        return df

    def _label_encode(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Apply label encoding to categorical features."""
        for col in categorical_cols:
            if col in df.columns:
                encoder_key = f"label_{col}"

                if encoder_key not in self.encoders:
                    self.encoders[encoder_key] = LabelEncoder()
                    df[col] = self.encoders[encoder_key].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[encoder_key].transform(df[col].astype(str))

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        logger.debug("Normalizing features")

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ["fraud_indicator"]]

        if numerical_cols:
            if "feature_scaler" not in self.scalers:
                self.scalers["feature_scaler"] = StandardScaler()
                df[numerical_cols] = self.scalers["feature_scaler"].fit_transform(
                    df[numerical_cols]
                )
            else:
                df[numerical_cols] = self.scalers["feature_scaler"].transform(df[numerical_cols])

        return df

    def transform_new_data(self, claims: List[BaseClaim]) -> pd.DataFrame:
        """
        Transform new claims data using fitted preprocessors.

        Args:
            claims: List of new claim objects

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")

        logger.info(f"Transforming {len(claims)} new claims")

        # Convert to DataFrame and apply same transformations
        df = self._claims_to_dataframe(claims)
        df = self._extract_basic_features(df)
        df = self._extract_temporal_features(df)
        df = self._extract_provider_features(df)
        df = self._extract_amount_features(df)
        df = self._extract_medical_features(df)

        # Apply fitted transformations
        if self.handle_missing_data:
            df = self._handle_missing_data(df)

        df = self._encode_categorical_features(df)

        if self.normalize_amounts:
            df = self._normalize_features(df)

        # Ensure same columns as training data
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        # Remove extra columns
        extra_cols = set(df.columns) - set(self.feature_columns + ["claim_id", "fraud_indicator"])
        df = df.drop(columns=list(extra_cols))

        # Reorder columns
        column_order = ["claim_id"] + self.feature_columns
        if "fraud_indicator" in df.columns:
            column_order.append("fraud_indicator")

        df = df[column_order]

        logger.info("Transformation complete")
        return df

    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate feature importance analysis data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Dictionary with feature analysis
        """
        feature_info = {}

        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ["fraud_indicator"]]

        if "fraud_indicator" in df.columns:
            # Correlation with fraud indicator
            correlations = {}
            for col in numerical_features:
                if col in df.columns:
                    corr = df[col].corr(df["fraud_indicator"])
                    if not pd.isna(corr):
                        correlations[col] = abs(corr)

            feature_info["correlations"] = correlations

        # Feature statistics
        feature_stats = {}
        for col in numerical_features:
            if col in df.columns:
                feature_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "missing_pct": df[col].isnull().mean() * 100,
                }

        feature_info["statistics"] = feature_stats
        feature_info["feature_count"] = len(self.feature_columns)
        feature_info["feature_names"] = self.feature_columns

        return feature_info
