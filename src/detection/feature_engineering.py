"""
Advanced feature engineering for fraud detection.

This module extracts and transforms features for machine learning models,
including temporal patterns, provider network analysis, claim sequences,
and cross-claim correlations.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features."""
    basic_features: pd.DataFrame
    temporal_features: pd.DataFrame
    network_features: pd.DataFrame
    sequence_features: pd.DataFrame
    statistical_features: pd.DataFrame
    text_features: pd.DataFrame


class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection.

    Extracts features across multiple dimensions:
    - Basic claim attributes
    - Temporal patterns and trends
    - Provider network analysis
    - Claim sequence patterns
    - Statistical aggregations
    - Text-based features
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_names = {}
        self.provider_network = nx.Graph()
        self.claim_history = defaultdict(list)
        self.feature_importance_cache = {}

    def extract_features(self, claims: List[Dict[str, Any]], target_col: str = 'fraud_indicator') -> FeatureSet:
        """
        Extract comprehensive features from claims data.

        Args:
            claims: List of claim dictionaries
            target_col: Name of target variable column

        Returns:
            FeatureSet containing all engineered features
        """
        logger.info(f"Extracting features from {len(claims)} claims")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(claims)

        # Ensure required columns exist
        self._validate_and_prepare_data(df)

        # Extract different feature types
        basic_features = self._extract_basic_features(df)
        temporal_features = self._extract_temporal_features(df)
        network_features = self._extract_network_features(df)
        sequence_features = self._extract_sequence_features(df)
        statistical_features = self._extract_statistical_features(df)
        text_features = self._extract_text_features(df)

        feature_set = FeatureSet(
            basic_features=basic_features,
            temporal_features=temporal_features,
            network_features=network_features,
            sequence_features=sequence_features,
            statistical_features=statistical_features,
            text_features=text_features
        )

        logger.info("Feature extraction completed")
        return feature_set

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> None:
        """Validate and prepare data for feature extraction."""
        required_cols = ['claim_id', 'provider_id', 'patient_id', 'date_of_service']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert date columns
        if 'date_of_service' in df.columns:
            df['date_of_service'] = pd.to_datetime(df['date_of_service'])

        # Fill missing values
        df['billed_amount'] = df.get('billed_amount', 0).fillna(0)
        df['procedure_codes'] = df.get('procedure_codes', []).fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        df['diagnosis_codes'] = df.get('diagnosis_codes', []).fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic claim features."""
        logger.info("Extracting basic features")

        features = pd.DataFrame(index=df.index)

        # Amount-based features
        features['billed_amount'] = df['billed_amount']
        features['billed_amount_log'] = np.log1p(df['billed_amount'])
        features['billed_amount_sqrt'] = np.sqrt(df['billed_amount'])

        # Code count features
        features['num_procedure_codes'] = df['procedure_codes'].apply(len)
        features['num_diagnosis_codes'] = df['diagnosis_codes'].apply(len)
        features['total_codes'] = features['num_procedure_codes'] + features['num_diagnosis_codes']

        # Code complexity features
        features['avg_procedure_complexity'] = df['procedure_codes'].apply(self._calculate_procedure_complexity)
        features['diagnosis_severity'] = df['diagnosis_codes'].apply(self._calculate_diagnosis_severity)

        # Service location encoding
        if 'service_location' in df.columns:
            le_location = LabelEncoder()
            features['service_location_encoded'] = le_location.fit_transform(
                df['service_location'].fillna('unknown')
            )
            self.encoders['service_location'] = le_location

        # Claim type encoding
        if 'claim_type' in df.columns:
            le_type = LabelEncoder()
            features['claim_type_encoded'] = le_type.fit_transform(
                df['claim_type'].fillna('unknown')
            )
            self.encoders['claim_type'] = le_type

        # Provider and patient encoding
        le_provider = LabelEncoder()
        le_patient = LabelEncoder()
        features['provider_encoded'] = le_provider.fit_transform(df['provider_id'])
        features['patient_encoded'] = le_patient.fit_transform(df['patient_id'])
        self.encoders['provider'] = le_provider
        self.encoders['patient'] = le_patient

        # Amount per procedure ratio
        features['amount_per_procedure'] = features['billed_amount'] / (features['num_procedure_codes'] + 1)

        # Red flags count
        if 'red_flags' in df.columns:
            features['num_red_flags'] = df['red_flags'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            features['num_red_flags'] = 0

        self.feature_names['basic'] = list(features.columns)
        return features

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal patterns and trends."""
        logger.info("Extracting temporal features")

        features = pd.DataFrame(index=df.index)

        # Basic time features
        features['day_of_week'] = df['date_of_service'].dt.dayofweek
        features['month'] = df['date_of_service'].dt.month
        features['day_of_month'] = df['date_of_service'].dt.day
        features['quarter'] = df['date_of_service'].dt.quarter

        # Weekend indicator
        features['is_weekend'] = (df['date_of_service'].dt.dayofweek >= 5).astype(int)

        # Holiday indicator (simplified)
        features['is_holiday'] = df['date_of_service'].apply(self._is_holiday).astype(int)

        # Time since epoch (for trend analysis)
        features['days_since_epoch'] = (df['date_of_service'] - datetime(2020, 1, 1)).dt.days

        # Provider temporal patterns
        provider_temporal = self._calculate_provider_temporal_patterns(df)
        features = features.join(provider_temporal)

        # Patient temporal patterns
        patient_temporal = self._calculate_patient_temporal_patterns(df)
        features = features.join(patient_temporal)

        # Claim timing patterns
        timing_features = self._extract_claim_timing_features(df)
        features = features.join(timing_features)

        self.feature_names['temporal'] = list(features.columns)
        return features

    def _extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract provider network analysis features."""
        logger.info("Extracting network features")

        features = pd.DataFrame(index=df.index)

        # Build provider network
        self._build_provider_network(df)

        # Provider centrality measures
        centrality_features = self._calculate_provider_centrality(df)
        features = features.join(centrality_features)

        # Provider clustering features
        clustering_features = self._calculate_provider_clustering(df)
        features = features.join(clustering_features)

        # Referral pattern features
        referral_features = self._calculate_referral_patterns(df)
        features = features.join(referral_features)

        # Provider collaboration features
        collaboration_features = self._calculate_provider_collaboration(df)
        features = features.join(collaboration_features)

        self.feature_names['network'] = list(features.columns)
        return features

    def _extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract claim sequence analysis features."""
        logger.info("Extracting sequence features")

        features = pd.DataFrame(index=df.index)

        # Sort by patient and date for sequence analysis
        df_sorted = df.sort_values(['patient_id', 'date_of_service'])

        # Sequence position features
        sequence_position = self._calculate_sequence_positions(df_sorted)
        features = features.join(sequence_position)

        # Inter-claim intervals
        interval_features = self._calculate_inter_claim_intervals(df_sorted)
        features = features.join(interval_features)

        # Procedure sequence patterns
        procedure_sequence = self._analyze_procedure_sequences(df_sorted)
        features = features.join(procedure_sequence)

        # Diagnosis progression features
        diagnosis_progression = self._analyze_diagnosis_progression(df_sorted)
        features = features.join(diagnosis_progression)

        # Provider switching patterns
        provider_switching = self._analyze_provider_switching(df_sorted)
        features = features.join(provider_switching)

        self.feature_names['sequence'] = list(features.columns)
        return features

    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical aggregation features."""
        logger.info("Extracting statistical features")

        features = pd.DataFrame(index=df.index)

        # Provider-level statistics
        provider_stats = self._calculate_provider_statistics(df)
        features = features.join(provider_stats)

        # Patient-level statistics
        patient_stats = self._calculate_patient_statistics(df)
        features = features.join(patient_stats)

        # Cross-claim correlations
        correlation_features = self._calculate_cross_claim_correlations(df)
        features = features.join(correlation_features)

        # Anomaly scores
        anomaly_features = self._calculate_anomaly_scores(df)
        features = features.join(anomaly_features)

        # Distribution-based features
        distribution_features = self._calculate_distribution_features(df)
        features = features.join(distribution_features)

        self.feature_names['statistical'] = list(features.columns)
        return features

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text-based features from notes and descriptions."""
        logger.info("Extracting text features")

        features = pd.DataFrame(index=df.index)

        # Combine text fields
        text_fields = ['notes', 'procedure_descriptions', 'diagnosis_descriptions']
        combined_text = []

        for idx, row in df.iterrows():
            text_parts = []
            for field in text_fields:
                if field in row and pd.notna(row[field]):
                    if isinstance(row[field], list):
                        text_parts.extend(row[field])
                    else:
                        text_parts.append(str(row[field]))
            combined_text.append(' '.join(text_parts))

        if combined_text and any(text.strip() for text in combined_text):
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )

            try:
                tfidf_matrix = vectorizer.fit_transform(combined_text)
                tfidf_features = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    index=df.index,
                    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )
                features = features.join(tfidf_features)
                self.vectorizers['tfidf'] = vectorizer
            except ValueError as e:
                logger.warning(f"Could not extract TF-IDF features: {e}")

        # Text length features
        features['text_length'] = [len(text) for text in combined_text]
        features['word_count'] = [len(text.split()) for text in combined_text]

        # Suspicious keyword indicators
        suspicious_keywords = ['urgent', 'emergency', 'critical', 'severe', 'complications']
        for keyword in suspicious_keywords:
            features[f'contains_{keyword}'] = [
                1 if keyword in text.lower() else 0 for text in combined_text
            ]

        self.feature_names['text'] = list(features.columns)
        return features

    def _calculate_procedure_complexity(self, codes: List[str]) -> float:
        """Calculate average procedure complexity."""
        if not codes:
            return 0.0

        # Simplified complexity scoring
        complexity_map = {
            '99211': 1, '99212': 2, '99213': 3, '99214': 4, '99215': 5,
            '99281': 1, '99282': 2, '99283': 3, '99284': 4, '99285': 5
        }

        complexities = [complexity_map.get(code, 3) for code in codes]
        return np.mean(complexities)

    def _calculate_diagnosis_severity(self, codes: List[str]) -> float:
        """Calculate diagnosis severity score."""
        if not codes:
            return 0.0

        # Simplified severity scoring based on code patterns
        severity_patterns = {
            'E11': 2,  # Diabetes
            'I10': 1,  # Hypertension
            'J18': 3,  # Pneumonia
            'S72': 4,  # Fractures
            'F32': 2   # Depression
        }

        severities = []
        for code in codes:
            for pattern, severity in severity_patterns.items():
                if code.startswith(pattern):
                    severities.append(severity)
                    break
            else:
                severities.append(2)  # Default severity

        return np.mean(severities)

    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday."""
        holidays = [
            (1, 1), (7, 4), (12, 25),  # Major holidays
            (1, 15), (2, 19), (5, 27), (9, 3), (10, 8), (11, 11), (11, 22)  # Federal holidays
        ]
        return (date.month, date.day) in holidays

    def _calculate_provider_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate provider-specific temporal patterns."""
        features = pd.DataFrame(index=df.index)

        # Claims per day by provider
        provider_daily_counts = df.groupby(['provider_id', 'date_of_service']).size()
        features['provider_claims_same_day'] = df.apply(
            lambda row: provider_daily_counts.get((row['provider_id'], row['date_of_service']), 0),
            axis=1
        )

        # Provider weekend ratio
        provider_weekend_ratio = df.groupby('provider_id').apply(
            lambda x: (x['date_of_service'].dt.dayofweek >= 5).mean()
        )
        features['provider_weekend_ratio'] = df['provider_id'].map(provider_weekend_ratio).fillna(0)

        # Provider time consistency
        provider_hour_consistency = self._calculate_provider_hour_consistency(df)
        features['provider_hour_consistency'] = df['provider_id'].map(provider_hour_consistency).fillna(0)

        return features

    def _calculate_patient_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate patient-specific temporal patterns."""
        features = pd.DataFrame(index=df.index)

        # Patient visit frequency
        patient_visit_counts = df.groupby('patient_id').size()
        features['patient_total_visits'] = df['patient_id'].map(patient_visit_counts).fillna(0)

        # Days since last visit
        df_sorted = df.sort_values(['patient_id', 'date_of_service'])
        features['days_since_last_visit'] = df_sorted.groupby('patient_id')['date_of_service'].diff().dt.days.fillna(999)

        # Patient provider diversity
        patient_provider_counts = df.groupby('patient_id')['provider_id'].nunique()
        features['patient_provider_diversity'] = df['patient_id'].map(patient_provider_counts).fillna(0)

        return features

    def _extract_claim_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract claim timing-related features."""
        features = pd.DataFrame(index=df.index)

        # Time of day (if available)
        if 'time_of_service' in df.columns:
            features['hour_of_service'] = pd.to_datetime(df['time_of_service'], format='%H:%M', errors='coerce').dt.hour
            features['is_after_hours'] = ((features['hour_of_service'] < 7) | (features['hour_of_service'] > 19)).astype(int)
        else:
            features['hour_of_service'] = 12  # Default to noon
            features['is_after_hours'] = 0

        # Rendering hours
        if 'rendering_hours' in df.columns:
            features['rendering_hours'] = df['rendering_hours']
            features['rendering_hours_log'] = np.log1p(df['rendering_hours'])
        else:
            features['rendering_hours'] = 1.0
            features['rendering_hours_log'] = 0.0

        return features

    def _build_provider_network(self, df: pd.DataFrame) -> None:
        """Build provider network graph."""
        self.provider_network.clear()

        # Add providers as nodes
        providers = df['provider_id'].unique()
        self.provider_network.add_nodes_from(providers)

        # Add edges based on shared patients
        patient_providers = df.groupby('patient_id')['provider_id'].apply(list)

        for patient_id, provider_list in patient_providers.items():
            if len(provider_list) > 1:
                for i in range(len(provider_list)):
                    for j in range(i + 1, len(provider_list)):
                        p1, p2 = provider_list[i], provider_list[j]
                        if self.provider_network.has_edge(p1, p2):
                            self.provider_network[p1][p2]['weight'] += 1
                        else:
                            self.provider_network.add_edge(p1, p2, weight=1)

    def _calculate_provider_centrality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate provider centrality measures."""
        features = pd.DataFrame(index=df.index)

        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.provider_network)
            features['provider_degree_centrality'] = df['provider_id'].map(degree_centrality).fillna(0)

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.provider_network)
            features['provider_betweenness_centrality'] = df['provider_id'].map(betweenness_centrality).fillna(0)

            # Closeness centrality
            if self.provider_network.number_of_nodes() > 1:
                closeness_centrality = nx.closeness_centrality(self.provider_network)
                features['provider_closeness_centrality'] = df['provider_id'].map(closeness_centrality).fillna(0)
            else:
                features['provider_closeness_centrality'] = 0

        except Exception as e:
            logger.warning(f"Could not calculate centrality measures: {e}")
            features['provider_degree_centrality'] = 0
            features['provider_betweenness_centrality'] = 0
            features['provider_closeness_centrality'] = 0

        return features

    def _calculate_provider_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate provider clustering features."""
        features = pd.DataFrame(index=df.index)

        try:
            # Clustering coefficient
            clustering = nx.clustering(self.provider_network)
            features['provider_clustering'] = df['provider_id'].map(clustering).fillna(0)

            # Local clustering
            features['provider_local_clustering'] = features['provider_clustering']

        except Exception as e:
            logger.warning(f"Could not calculate clustering measures: {e}")
            features['provider_clustering'] = 0
            features['provider_local_clustering'] = 0

        return features

    def _calculate_referral_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate referral pattern features."""
        features = pd.DataFrame(index=df.index)

        # Referral indicators (simplified)
        referral_codes = ['99241', '99242', '99243', '99244', '99245']

        features['is_referral'] = df['procedure_codes'].apply(
            lambda codes: 1 if any(code in referral_codes for code in codes) else 0
        )

        # Provider referral ratio
        provider_referral_ratio = df.groupby('provider_id')['is_referral'].mean()
        features['provider_referral_ratio'] = df['provider_id'].map(provider_referral_ratio).fillna(0)

        return features

    def _calculate_provider_collaboration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate provider collaboration features."""
        features = pd.DataFrame(index=df.index)

        # Number of collaborating providers
        provider_connections = {
            provider: len(list(self.provider_network.neighbors(provider)))
            for provider in self.provider_network.nodes()
        }
        features['provider_connections'] = df['provider_id'].map(provider_connections).fillna(0)

        # Collaboration strength
        provider_collaboration_strength = {}
        for provider in self.provider_network.nodes():
            total_weight = sum(
                self.provider_network[provider][neighbor].get('weight', 1)
                for neighbor in self.provider_network.neighbors(provider)
            )
            provider_collaboration_strength[provider] = total_weight

        features['provider_collaboration_strength'] = df['provider_id'].map(provider_collaboration_strength).fillna(0)

        return features

    def _calculate_sequence_positions(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Calculate position in claim sequences."""
        features = pd.DataFrame(index=df_sorted.index)

        # Position in patient's claim sequence
        features['claim_sequence_position'] = df_sorted.groupby('patient_id').cumcount() + 1

        # Position in provider's daily sequence
        features['provider_daily_sequence'] = df_sorted.groupby(
            ['provider_id', 'date_of_service']
        ).cumcount() + 1

        return features

    def _calculate_inter_claim_intervals(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Calculate intervals between claims."""
        features = pd.DataFrame(index=df_sorted.index)

        # Days between patient claims
        features['days_between_patient_claims'] = df_sorted.groupby('patient_id')['date_of_service'].diff().dt.days.fillna(999)

        # Days between provider claims
        features['days_between_provider_claims'] = df_sorted.groupby('provider_id')['date_of_service'].diff().dt.days.fillna(999)

        return features

    def _analyze_procedure_sequences(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Analyze procedure code sequences."""
        features = pd.DataFrame(index=df_sorted.index)

        # Procedure repetition patterns
        features['procedure_repetition_score'] = 0
        features['procedure_escalation_score'] = 0

        # This would be more sophisticated in practice
        for patient_id, group in df_sorted.groupby('patient_id'):
            if len(group) > 1:
                # Calculate procedure patterns for this patient
                pass  # Simplified for brevity

        return features

    def _analyze_diagnosis_progression(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Analyze diagnosis progression patterns."""
        features = pd.DataFrame(index=df_sorted.index)

        # Diagnosis consistency
        features['diagnosis_consistency'] = 0

        # Diagnosis escalation
        features['diagnosis_escalation'] = 0

        return features

    def _analyze_provider_switching(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Analyze provider switching patterns."""
        features = pd.DataFrame(index=df_sorted.index)

        # Provider switches for patient
        provider_switches = df_sorted.groupby('patient_id')['provider_id'].apply(
            lambda x: (x != x.shift()).sum() - 1
        ).fillna(0)

        features['patient_provider_switches'] = df_sorted['patient_id'].map(provider_switches)

        return features

    def _calculate_provider_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate provider-level statistical features."""
        features = pd.DataFrame(index=df.index)

        # Provider billing statistics
        provider_stats = df.groupby('provider_id')['billed_amount'].agg(['mean', 'std', 'count'])
        features['provider_avg_amount'] = df['provider_id'].map(provider_stats['mean']).fillna(0)
        features['provider_std_amount'] = df['provider_id'].map(provider_stats['std']).fillna(0)
        features['provider_claim_count'] = df['provider_id'].map(provider_stats['count']).fillna(0)

        # Z-score of current claim amount relative to provider
        features['amount_zscore_provider'] = (
            df['billed_amount'] - features['provider_avg_amount']
        ) / (features['provider_std_amount'] + 1e-8)

        return features

    def _calculate_patient_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate patient-level statistical features."""
        features = pd.DataFrame(index=df.index)

        # Patient billing statistics
        patient_stats = df.groupby('patient_id')['billed_amount'].agg(['mean', 'std', 'count'])
        features['patient_avg_amount'] = df['patient_id'].map(patient_stats['mean']).fillna(0)
        features['patient_std_amount'] = df['patient_id'].map(patient_stats['std']).fillna(0)
        features['patient_claim_count'] = df['patient_id'].map(patient_stats['count']).fillna(0)

        return features

    def _calculate_cross_claim_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations across claims."""
        features = pd.DataFrame(index=df.index)

        # Amount correlation with temporal features
        features['amount_day_correlation'] = 0  # Simplified
        features['amount_provider_correlation'] = 0  # Simplified

        return features

    def _calculate_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic anomaly scores."""
        features = pd.DataFrame(index=df.index)

        # Amount anomaly (IQR method)
        Q1 = df['billed_amount'].quantile(0.25)
        Q3 = df['billed_amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        features['amount_anomaly_score'] = np.where(
            (df['billed_amount'] < lower_bound) | (df['billed_amount'] > upper_bound),
            1, 0
        )

        return features

    def _calculate_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distribution-based features."""
        features = pd.DataFrame(index=df.index)

        # Percentile ranks
        features['amount_percentile'] = df['billed_amount'].rank(pct=True)

        # Provider percentile within specialty
        if 'provider_specialty' in df.columns:
            features['amount_percentile_specialty'] = df.groupby('provider_specialty')['billed_amount'].rank(pct=True)
        else:
            features['amount_percentile_specialty'] = features['amount_percentile']

        return features

    def _calculate_provider_hour_consistency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate provider hour consistency."""
        # Simplified implementation
        return {provider: 0.8 for provider in df['provider_id'].unique()}

    def combine_features(self, feature_set: FeatureSet, include_sets: List[str] = None) -> pd.DataFrame:
        """
        Combine feature sets into a single DataFrame.

        Args:
            feature_set: FeatureSet containing all features
            include_sets: List of feature set names to include

        Returns:
            Combined feature DataFrame
        """
        if include_sets is None:
            include_sets = ['basic', 'temporal', 'network', 'sequence', 'statistical', 'text']

        combined_features = pd.DataFrame()

        if 'basic' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.basic_features], axis=1)

        if 'temporal' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.temporal_features], axis=1)

        if 'network' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.network_features], axis=1)

        if 'sequence' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.sequence_features], axis=1)

        if 'statistical' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.statistical_features], axis=1)

        if 'text' in include_sets:
            combined_features = pd.concat([combined_features, feature_set.text_features], axis=1)

        # Remove duplicate columns
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features

    def get_feature_importance(self, feature_set: FeatureSet, target: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using correlation and mutual information.

        Args:
            feature_set: FeatureSet containing features
            target: Target variable

        Returns:
            Dictionary of feature importance scores
        """
        combined_features = self.combine_features(feature_set)

        # Remove non-numeric columns
        numeric_features = combined_features.select_dtypes(include=[np.number])

        importance_scores = {}

        # Correlation-based importance
        for col in numeric_features.columns:
            try:
                corr = abs(numeric_features[col].corr(target))
                importance_scores[col] = corr if not np.isnan(corr) else 0.0
            except:
                importance_scores[col] = 0.0

        return importance_scores

    def select_features(self, feature_set: FeatureSet, target: pd.Series, top_k: int = 50) -> List[str]:
        """
        Select top k most important features.

        Args:
            feature_set: FeatureSet containing features
            target: Target variable
            top_k: Number of top features to select

        Returns:
            List of selected feature names
        """
        importance_scores = self.get_feature_importance(feature_set, target)

        # Sort by importance and select top k
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:top_k]]

        logger.info(f"Selected {len(selected_features)} features out of {len(importance_scores)}")
        return selected_features

    def save_feature_engineering_pipeline(self, filepath: str) -> None:
        """Save feature engineering pipeline artifacts."""
        import pickle

        pipeline_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'vectorizers': self.vectorizers,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)

        logger.info(f"Feature engineering pipeline saved to {filepath}")

    def load_feature_engineering_pipeline(self, filepath: str) -> None:
        """Load feature engineering pipeline artifacts."""
        import pickle

        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)

        self.scalers = pipeline_data.get('scalers', {})
        self.encoders = pipeline_data.get('encoders', {})
        self.vectorizers = pipeline_data.get('vectorizers', {})
        self.feature_names = pipeline_data.get('feature_names', {})

        logger.info(f"Feature engineering pipeline loaded from {filepath}")