"""
Multi-factor confidence scoring algorithm for enrichment decisions.

Implements a 5-factor confidence scoring system:
1. Retrieval Quality (40%) - Quality of KB retrieval results
2. Source Diversity (20%) - Number of different KBs consulted
3. Temporal Relevance (15%) - Recency of source data
4. Cross-Validation (15%) - Agreement across multiple sources
5. Regulatory Citation (10%) - Regulatory KB confirmation

Formula:
    confidence = 0.40 * retrieval + 0.20 * diversity + 0.15 * temporal
                 + 0.15 * cross_val + 0.10 * regulatory
"""

import math
from typing import Dict, List, Any, Tuple
from collections import Counter

from src.rag.schemas import (
    EnrichmentEvidence,
    KnowledgeBaseType,
    EnrichmentQualityTier,
    ConfidenceFactors,
)


class ConfidenceScorer:
    """
    Multi-factor confidence scoring for enrichment decisions.

    Computes confidence scores based on five factors:
    - Retrieval quality
    - Source diversity
    - Temporal relevance
    - Cross-validation
    - Regulatory citation

    All methods return scores in range [0.0, 1.0].
    """

    # Confidence tier thresholds
    EXCELLENT_THRESHOLD = 0.90
    GOOD_THRESHOLD = 0.80
    ACCEPTABLE_THRESHOLD = 0.70

    # Factor weights (must sum to 1.0)
    WEIGHT_RETRIEVAL = 0.40
    WEIGHT_DIVERSITY = 0.20
    WEIGHT_TEMPORAL = 0.15
    WEIGHT_CROSS_VAL = 0.15
    WEIGHT_REGULATORY = 0.10

    def score_retrieval_quality(self, evidence: List[EnrichmentEvidence]) -> float:
        """
        Score quality of retrieval results.

        Factors:
        - Number of results (more is better up to a point)
        - Average relevance score
        - Average similarity distance (lower is better)

        Args:
            evidence: List of evidence from KB retrieval

        Returns:
            Retrieval quality score [0.0, 1.0]
        """
        if not evidence:
            return 0.0

        # Factor 1: Number of results (diminishing returns)
        num_results = len(evidence)
        result_bonus = min(1.0, num_results / 3.0)  # Optimal is 3+ results

        # Factor 2: Average relevance score
        avg_relevance = sum(e.relevance_score for e in evidence) / num_results

        # Factor 3: Average similarity distance (inverted - lower is better)
        avg_distance = sum(e.similarity_distance for e in evidence) / num_results
        distance_score = max(0.0, 1.0 - avg_distance)

        # Weighted combination
        # 50% relevance, 30% distance, 20% result count
        score = 0.50 * avg_relevance + 0.30 * distance_score + 0.20 * result_bonus

        return round(score, 4)

    def score_source_diversity(self, sources: List[KnowledgeBaseType]) -> float:
        """
        Score diversity of knowledge base sources consulted.

        More diverse sources indicate higher confidence through
        cross-validation and comprehensive retrieval.

        Args:
            sources: List of KB types consulted

        Returns:
            Source diversity score [0.0, 1.0]
        """
        if not sources:
            return 0.0

        # Count unique sources
        unique_sources = set(sources)
        num_unique = len(unique_sources)

        # Maximum possible is 4 KBs
        max_sources = len(KnowledgeBaseType)

        # Linear scaling with bonus for full coverage
        base_score = num_unique / max_sources
        if num_unique == max_sources:
            base_score = 1.0  # Perfect diversity bonus

        return round(base_score, 4)

    def score_temporal_relevance(self, age_days: float) -> float:
        """
        Score temporal relevance of source data.

        Newer data is more relevant, with exponential decay.

        Tiers:
        - <30 days: 0.90-1.00 (very recent)
        - 30-90 days: 0.75-0.90 (recent)
        - 90-180 days: 0.50-0.75 (moderate)
        - 180-365 days: 0.25-0.50 (old)
        - >365 days: 0.00-0.25 (very old)

        Args:
            age_days: Age of data in days

        Returns:
            Temporal relevance score [0.0, 1.0]

        Raises:
            ValueError: If age_days is negative
        """
        if age_days < 0:
            raise ValueError("age_days cannot be negative")

        if age_days == 0:
            return 1.0

        # Exponential decay with half-life of ~120 days
        half_life = 120.0
        score = math.exp(-age_days * math.log(2) / half_life)

        return round(score, 4)

    def score_cross_validation(
        self, retrieved_values: Dict[str, List[Tuple[Any, KnowledgeBaseType]]]
    ) -> float:
        """
        Score agreement across multiple sources.

        Higher scores when multiple sources agree on the same value.

        Args:
            retrieved_values: Dict mapping field names to list of
                (value, source_kb) tuples

        Returns:
            Cross-validation score [0.0, 1.0]
        """
        if not retrieved_values:
            return 0.0

        agreement_scores = []

        for field_name, values in retrieved_values.items():
            if not values:
                continue

            # Count occurrences of each value
            value_counts = Counter(v[0] for v in values)
            total_values = len(values)

            if total_values == 1:
                # Single source - medium confidence
                agreement_scores.append(0.50)
            else:
                # Calculate agreement ratio
                most_common_count = value_counts.most_common(1)[0][1]
                agreement_ratio = most_common_count / total_values

                # Perfect agreement (all sources agree) = 1.0
                # Majority agreement = 0.70-0.95
                # No agreement (all different) = 0.30
                if agreement_ratio == 1.0:
                    agreement_scores.append(1.0)
                elif agreement_ratio >= 0.75:
                    agreement_scores.append(0.85)
                elif agreement_ratio >= 0.50:
                    agreement_scores.append(0.70)
                else:
                    agreement_scores.append(0.40)

        if not agreement_scores:
            return 0.0

        # Average agreement across all fields
        avg_agreement = sum(agreement_scores) / len(agreement_scores)
        return round(avg_agreement, 4)

    def score_regulatory_citation(
        self, has_regulatory_confirmation: bool, regulatory_confidence: float = 0.0
    ) -> float:
        """
        Score regulatory knowledge base confirmation.

        Regulatory confirmation adds validation that the enrichment
        aligns with medical coding standards and regulations.

        Args:
            has_regulatory_confirmation: Whether regulatory KB confirmed
            regulatory_confidence: Confidence score from regulatory KB

        Returns:
            Regulatory citation score [0.0, 1.0]
        """
        if has_regulatory_confirmation:
            # Regulatory confirmation - scale by confidence
            return round(0.75 + (regulatory_confidence * 0.25), 4)
        elif regulatory_confidence > 0.70:
            # High confidence negative (conflict)
            return 0.20
        else:
            # No regulatory data available - neutral
            return 0.50

    def compute_overall_confidence(self, factors: Dict[str, float]) -> float:
        """
        Compute weighted overall confidence from individual factors.

        Weights:
        - Retrieval Quality: 40%
        - Source Diversity: 20%
        - Temporal Relevance: 15%
        - Cross-Validation: 15%
        - Regulatory Citation: 10%

        Args:
            factors: Dict with factor names and scores

        Returns:
            Weighted overall confidence [0.0, 1.0]
        """
        score = (
            factors.get("retrieval_quality", 0.0) * self.WEIGHT_RETRIEVAL
            + factors.get("source_diversity", 0.0) * self.WEIGHT_DIVERSITY
            + factors.get("temporal_relevance", 0.0) * self.WEIGHT_TEMPORAL
            + factors.get("cross_validation", 0.0) * self.WEIGHT_CROSS_VAL
            + factors.get("regulatory_citation", 0.0) * self.WEIGHT_REGULATORY
        )

        # Ensure bounds
        score = max(0.0, min(1.0, score))
        return round(score, 4)

    def compute_quality_tier(self, confidence: float) -> EnrichmentQualityTier:
        """
        Classify confidence score into quality tier.

        Tiers:
        - EXCELLENT: ≥ 0.90
        - GOOD: 0.80-0.89
        - ACCEPTABLE: 0.70-0.79
        - POOR: < 0.70

        Args:
            confidence: Overall confidence score

        Returns:
            Quality tier classification
        """
        if confidence >= self.EXCELLENT_THRESHOLD:
            return EnrichmentQualityTier.EXCELLENT
        elif confidence >= self.GOOD_THRESHOLD:
            return EnrichmentQualityTier.GOOD
        elif confidence >= self.ACCEPTABLE_THRESHOLD:
            return EnrichmentQualityTier.ACCEPTABLE
        else:
            return EnrichmentQualityTier.POOR

    def compute_all_factors(
        self,
        evidence: List[EnrichmentEvidence],
        sources: List[KnowledgeBaseType],
        age_days: float,
        retrieved_values: Dict[str, List[Tuple[Any, KnowledgeBaseType]]],
        has_regulatory: bool = False,
        regulatory_confidence: float = 0.0,
    ) -> ConfidenceFactors:
        """
        Compute all confidence factors in one call.

        Convenience method for computing all five factors at once.

        Args:
            evidence: Retrieval evidence
            sources: KB sources consulted
            age_days: Age of source data
            retrieved_values: Retrieved values for cross-validation
            has_regulatory: Regulatory confirmation flag
            regulatory_confidence: Regulatory confidence score

        Returns:
            ConfidenceFactors object with all factor scores
        """
        return ConfidenceFactors(
            retrieval_quality=self.score_retrieval_quality(evidence),
            source_diversity=self.score_source_diversity(sources),
            temporal_relevance=self.score_temporal_relevance(age_days),
            cross_validation=self.score_cross_validation(retrieved_values),
            regulatory_citation=self.score_regulatory_citation(
                has_regulatory, regulatory_confidence
            ),
        )
