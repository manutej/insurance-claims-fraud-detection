"""
Unit tests for Confidence Scoring component.

Test coverage for multi-factor confidence scoring algorithm,
quality tier classification, and score calibration.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from src.rag.schemas import (
    EnrichmentEvidence,
    KnowledgeBaseType,
    EnrichmentQualityTier,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def high_quality_evidence():
    """High-quality retrieval evidence."""
    return [
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.MEDICAL_CODING,
            document_id="doc_001",
            relevance_score=0.95,
            similarity_distance=0.05,
            content_snippet="CPT 99213 commonly associated with E11.9",
        ),
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.PROVIDER_PATTERN,
            document_id="doc_002",
            relevance_score=0.92,
            similarity_distance=0.08,
            content_snippet="Provider frequently bills 99213 with E11.9",
        ),
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.PATIENT_HISTORY,
            document_id="doc_003",
            relevance_score=0.90,
            similarity_distance=0.10,
            content_snippet="Patient history shows diabetes diagnosis",
        ),
    ]


@pytest.fixture
def medium_quality_evidence():
    """Medium-quality retrieval evidence."""
    return [
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.MEDICAL_CODING,
            document_id="doc_004",
            relevance_score=0.78,
            similarity_distance=0.22,
            content_snippet="CPT 99213 office visit",
        ),
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.PROVIDER_PATTERN,
            document_id="doc_005",
            relevance_score=0.75,
            similarity_distance=0.25,
            content_snippet="Common procedure code",
        ),
    ]


@pytest.fixture
def low_quality_evidence():
    """Low-quality retrieval evidence."""
    return [
        EnrichmentEvidence(
            source_kb=KnowledgeBaseType.MEDICAL_CODING,
            document_id="doc_006",
            relevance_score=0.55,
            similarity_distance=0.45,
            content_snippet="General medical codes",
        )
    ]


@pytest.fixture
def confidence_scorer():
    """Confidence scorer instance."""
    from src.rag.confidence_scoring import ConfidenceScorer

    return ConfidenceScorer()


# ============================================================================
# RETRIEVAL QUALITY SCORING TESTS
# ============================================================================


class TestRetrievalQualityScoring:
    """Test retrieval quality factor scoring."""

    def test_high_relevance_scores(self, confidence_scorer, high_quality_evidence):
        """Should score high when relevance scores are >0.90."""
        score = confidence_scorer.score_retrieval_quality(high_quality_evidence)
        assert score >= 0.85, f"Expected >=0.85, got {score}"
        assert score <= 1.0, "Score should not exceed 1.0"

    def test_medium_relevance_scores(self, confidence_scorer, medium_quality_evidence):
        """Should score medium when relevance scores are 0.70-0.90."""
        score = confidence_scorer.score_retrieval_quality(medium_quality_evidence)
        assert 0.60 <= score < 0.85, f"Expected 0.60-0.85, got {score}"

    def test_low_relevance_scores(self, confidence_scorer, low_quality_evidence):
        """Should score low when relevance scores are <0.70."""
        score = confidence_scorer.score_retrieval_quality(low_quality_evidence)
        assert score < 0.60, f"Expected <0.60, got {score}"

    def test_empty_evidence(self, confidence_scorer):
        """Should return 0.0 for empty evidence."""
        score = confidence_scorer.score_retrieval_quality([])
        assert score == 0.0, "Empty evidence should score 0.0"

    def test_considers_number_of_results(self, confidence_scorer):
        """Should give higher scores for more results."""
        single_result = [
            EnrichmentEvidence(
                source_kb=KnowledgeBaseType.MEDICAL_CODING,
                document_id="doc_001",
                relevance_score=0.90,
                similarity_distance=0.10,
                content_snippet="test",
            )
        ]
        multiple_results = [
            EnrichmentEvidence(
                source_kb=KnowledgeBaseType.MEDICAL_CODING,
                document_id=f"doc_{i}",
                relevance_score=0.90,
                similarity_distance=0.10,
                content_snippet="test",
            )
            for i in range(3)
        ]

        score_single = confidence_scorer.score_retrieval_quality(single_result)
        score_multiple = confidence_scorer.score_retrieval_quality(multiple_results)

        assert score_multiple > score_single, "More results should increase score"

    def test_considers_similarity_distance(self, confidence_scorer):
        """Should prefer lower similarity distances."""
        low_distance = [
            EnrichmentEvidence(
                source_kb=KnowledgeBaseType.MEDICAL_CODING,
                document_id="doc_001",
                relevance_score=0.90,
                similarity_distance=0.05,
                content_snippet="test",
            )
        ]
        high_distance = [
            EnrichmentEvidence(
                source_kb=KnowledgeBaseType.MEDICAL_CODING,
                document_id="doc_002",
                relevance_score=0.90,
                similarity_distance=0.40,
                content_snippet="test",
            )
        ]

        score_low = confidence_scorer.score_retrieval_quality(low_distance)
        score_high = confidence_scorer.score_retrieval_quality(high_distance)

        assert score_low > score_high, "Lower distance should score higher"


# ============================================================================
# SOURCE DIVERSITY SCORING TESTS
# ============================================================================


class TestSourceDiversityScoring:
    """Test source diversity factor scoring."""

    def test_all_four_kbs(self, confidence_scorer):
        """Should score highest when all 4 KBs consulted."""
        sources = [
            KnowledgeBaseType.PATIENT_HISTORY,
            KnowledgeBaseType.PROVIDER_PATTERN,
            KnowledgeBaseType.MEDICAL_CODING,
            KnowledgeBaseType.REGULATORY,
        ]
        score = confidence_scorer.score_source_diversity(sources)
        assert score == 1.0, "All 4 sources should score 1.0"

    def test_three_kbs(self, confidence_scorer):
        """Should score well with 3 different KBs."""
        sources = [
            KnowledgeBaseType.MEDICAL_CODING,
            KnowledgeBaseType.PROVIDER_PATTERN,
            KnowledgeBaseType.PATIENT_HISTORY,
        ]
        score = confidence_scorer.score_source_diversity(sources)
        assert 0.70 <= score < 1.0, f"3 sources should score 0.70-1.0, got {score}"

    def test_two_kbs(self, confidence_scorer):
        """Should score medium with 2 different KBs."""
        sources = [KnowledgeBaseType.MEDICAL_CODING, KnowledgeBaseType.PROVIDER_PATTERN]
        score = confidence_scorer.score_source_diversity(sources)
        assert 0.40 <= score < 0.70, f"2 sources should score 0.40-0.70, got {score}"

    def test_single_kb(self, confidence_scorer):
        """Should score low with only 1 KB."""
        sources = [KnowledgeBaseType.MEDICAL_CODING]
        score = confidence_scorer.score_source_diversity(sources)
        assert score < 0.40, f"1 source should score <0.40, got {score}"

    def test_empty_sources(self, confidence_scorer):
        """Should return 0.0 for empty sources."""
        score = confidence_scorer.score_source_diversity([])
        assert score == 0.0, "Empty sources should score 0.0"

    def test_duplicate_sources_ignored(self, confidence_scorer):
        """Should count unique sources only."""
        sources = [
            KnowledgeBaseType.MEDICAL_CODING,
            KnowledgeBaseType.MEDICAL_CODING,
            KnowledgeBaseType.MEDICAL_CODING,
        ]
        score = confidence_scorer.score_source_diversity(sources)
        assert score < 0.40, "Duplicates should not increase score"


# ============================================================================
# TEMPORAL RELEVANCE SCORING TESTS
# ============================================================================


class TestTemporalRelevanceScoring:
    """Test temporal relevance factor scoring."""

    def test_very_recent_data(self, confidence_scorer):
        """Should score highest for data <30 days old."""
        age_days = 15
        score = confidence_scorer.score_temporal_relevance(age_days)
        assert score >= 0.90, f"Recent data should score >=0.90, got {score}"

    def test_recent_data(self, confidence_scorer):
        """Should score well for data <90 days old."""
        age_days = 60
        score = confidence_scorer.score_temporal_relevance(age_days)
        # Exponential decay with 120-day half-life: ~0.71 for 60 days
        assert 0.65 <= score < 0.90, f"Recent data should score 0.65-0.90, got {score}"

    def test_moderately_old_data(self, confidence_scorer):
        """Should score medium for data <180 days old."""
        age_days = 120
        score = confidence_scorer.score_temporal_relevance(age_days)
        # Exponential decay: 120 days = half-life, score ~0.50
        assert 0.45 <= score < 0.75, f"Moderate data should score 0.45-0.75, got {score}"

    def test_old_data(self, confidence_scorer):
        """Should score low for data <365 days old."""
        age_days = 300
        score = confidence_scorer.score_temporal_relevance(age_days)
        # Exponential decay: 300 days with 120-day half-life ~0.18
        assert 0.15 <= score < 0.25, f"Old data should score 0.15-0.25, got {score}"

    def test_very_old_data(self, confidence_scorer):
        """Should score very low for data >365 days old."""
        age_days = 400
        score = confidence_scorer.score_temporal_relevance(age_days)
        assert score < 0.25, f"Very old data should score <0.25, got {score}"

    def test_zero_age(self, confidence_scorer):
        """Should handle zero age (current data)."""
        score = confidence_scorer.score_temporal_relevance(0)
        assert score == 1.0, "Current data should score 1.0"

    def test_negative_age_raises_error(self, confidence_scorer):
        """Should raise error for negative age."""
        with pytest.raises(ValueError, match="age.*negative"):
            confidence_scorer.score_temporal_relevance(-10)


# ============================================================================
# CROSS-VALIDATION SCORING TESTS
# ============================================================================


class TestCrossValidationScoring:
    """Test cross-validation factor scoring."""

    def test_perfect_agreement(self, confidence_scorer):
        """Should score 1.0 when all sources agree."""
        retrieved_values = {
            "diagnosis_codes": [
                ("E11.9", KnowledgeBaseType.MEDICAL_CODING),
                ("E11.9", KnowledgeBaseType.PROVIDER_PATTERN),
                ("E11.9", KnowledgeBaseType.PATIENT_HISTORY),
            ]
        }
        score = confidence_scorer.score_cross_validation(retrieved_values)
        assert score == 1.0, "Perfect agreement should score 1.0"

    def test_majority_agreement(self, confidence_scorer):
        """Should score high when majority agree."""
        retrieved_values = {
            "diagnosis_codes": [
                ("E11.9", KnowledgeBaseType.MEDICAL_CODING),
                ("E11.9", KnowledgeBaseType.PROVIDER_PATTERN),
                ("E11.9", KnowledgeBaseType.PATIENT_HISTORY),
                ("I10", KnowledgeBaseType.REGULATORY),
            ]
        }
        score = confidence_scorer.score_cross_validation(retrieved_values)
        assert 0.70 <= score < 1.0, f"Majority agreement should score 0.70-1.0, got {score}"

    def test_no_agreement(self, confidence_scorer):
        """Should score low when sources disagree."""
        retrieved_values = {
            "diagnosis_codes": [
                ("E11.9", KnowledgeBaseType.MEDICAL_CODING),
                ("I10", KnowledgeBaseType.PROVIDER_PATTERN),
                ("J45.9", KnowledgeBaseType.PATIENT_HISTORY),
                ("M79.3", KnowledgeBaseType.REGULATORY),
            ]
        }
        score = confidence_scorer.score_cross_validation(retrieved_values)
        assert score < 0.50, f"No agreement should score <0.50, got {score}"

    def test_single_source(self, confidence_scorer):
        """Should score medium with only one source."""
        retrieved_values = {"diagnosis_codes": [("E11.9", KnowledgeBaseType.MEDICAL_CODING)]}
        score = confidence_scorer.score_cross_validation(retrieved_values)
        assert 0.40 <= score <= 0.60, f"Single source should score 0.40-0.60, got {score}"

    def test_empty_values(self, confidence_scorer):
        """Should return 0.0 for empty values."""
        score = confidence_scorer.score_cross_validation({})
        assert score == 0.0, "Empty values should score 0.0"


# ============================================================================
# REGULATORY CITATION SCORING TESTS
# ============================================================================


class TestRegulatoryCitationScoring:
    """Test regulatory citation factor scoring."""

    def test_regulatory_confirmation(self, confidence_scorer):
        """Should score high when regulatory KB confirms."""
        has_regulatory_confirmation = True
        regulatory_confidence = 0.95
        score = confidence_scorer.score_regulatory_citation(
            has_regulatory_confirmation, regulatory_confidence
        )
        assert score >= 0.90, f"Regulatory confirmation should score >=0.90, got {score}"

    def test_no_regulatory_data(self, confidence_scorer):
        """Should score medium when no regulatory data available."""
        has_regulatory_confirmation = False
        regulatory_confidence = 0.0
        score = confidence_scorer.score_regulatory_citation(
            has_regulatory_confirmation, regulatory_confidence
        )
        assert 0.40 <= score <= 0.60, f"No regulatory data should score 0.40-0.60, got {score}"

    def test_regulatory_conflict(self, confidence_scorer):
        """Should score low when regulatory KB conflicts."""
        has_regulatory_confirmation = False
        regulatory_confidence = 0.85  # High confidence but negative
        score = confidence_scorer.score_regulatory_citation(
            has_regulatory_confirmation, regulatory_confidence
        )
        assert score < 0.30, f"Regulatory conflict should score <0.30, got {score}"


# ============================================================================
# OVERALL CONFIDENCE AGGREGATION TESTS
# ============================================================================


class TestOverallConfidenceAggregation:
    """Test overall confidence score computation."""

    def test_weighted_aggregation(self, confidence_scorer):
        """Should compute weighted average correctly."""
        factors = {
            "retrieval_quality": 0.90,
            "source_diversity": 0.80,
            "temporal_relevance": 0.70,
            "cross_validation": 0.85,
            "regulatory_citation": 0.75,
        }
        score = confidence_scorer.compute_overall_confidence(factors)

        # Verify weighted formula: 40% retrieval + 20% diversity + 15% temporal + 15% cross_val + 10% regulatory
        expected = (0.90 * 0.40) + (0.80 * 0.20) + (0.70 * 0.15) + (0.85 * 0.15) + (0.75 * 0.10)
        assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"

    def test_all_high_factors(self, confidence_scorer):
        """Should produce high score when all factors high."""
        factors = {
            "retrieval_quality": 0.95,
            "source_diversity": 0.90,
            "temporal_relevance": 0.92,
            "cross_validation": 0.94,
            "regulatory_citation": 0.90,
        }
        score = confidence_scorer.compute_overall_confidence(factors)
        assert score >= 0.90, f"All high factors should produce >=0.90, got {score}"

    def test_all_low_factors(self, confidence_scorer):
        """Should produce low score when all factors low."""
        factors = {
            "retrieval_quality": 0.40,
            "source_diversity": 0.30,
            "temporal_relevance": 0.35,
            "cross_validation": 0.40,
            "regulatory_citation": 0.30,
        }
        score = confidence_scorer.compute_overall_confidence(factors)
        assert score < 0.50, f"All low factors should produce <0.50, got {score}"

    def test_score_bounds(self, confidence_scorer):
        """Should always produce scores in [0.0, 1.0]."""
        test_cases = [
            {
                "retrieval_quality": 0.0,
                "source_diversity": 0.0,
                "temporal_relevance": 0.0,
                "cross_validation": 0.0,
                "regulatory_citation": 0.0,
            },
            {
                "retrieval_quality": 1.0,
                "source_diversity": 1.0,
                "temporal_relevance": 1.0,
                "cross_validation": 1.0,
                "regulatory_citation": 1.0,
            },
            {
                "retrieval_quality": 0.50,
                "source_diversity": 0.60,
                "temporal_relevance": 0.70,
                "cross_validation": 0.45,
                "regulatory_citation": 0.55,
            },
        ]
        for factors in test_cases:
            score = confidence_scorer.compute_overall_confidence(factors)
            assert 0.0 <= score <= 1.0, f"Score {score} outside bounds [0.0, 1.0]"


# ============================================================================
# QUALITY TIER CLASSIFICATION TESTS
# ============================================================================


class TestQualityTierClassification:
    """Test quality tier classification."""

    def test_excellent_tier(self, confidence_scorer):
        """Should classify >= 0.90 as EXCELLENT."""
        tier = confidence_scorer.compute_quality_tier(0.92)
        assert tier == EnrichmentQualityTier.EXCELLENT

    def test_good_tier(self, confidence_scorer):
        """Should classify 0.80-0.89 as GOOD."""
        tier = confidence_scorer.compute_quality_tier(0.85)
        assert tier == EnrichmentQualityTier.GOOD

    def test_acceptable_tier(self, confidence_scorer):
        """Should classify 0.70-0.79 as ACCEPTABLE."""
        tier = confidence_scorer.compute_quality_tier(0.75)
        assert tier == EnrichmentQualityTier.ACCEPTABLE

    def test_poor_tier(self, confidence_scorer):
        """Should classify < 0.70 as POOR."""
        tier = confidence_scorer.compute_quality_tier(0.65)
        assert tier == EnrichmentQualityTier.POOR

    def test_boundary_conditions(self, confidence_scorer):
        """Should handle boundary values correctly."""
        assert confidence_scorer.compute_quality_tier(0.90) == EnrichmentQualityTier.EXCELLENT
        assert confidence_scorer.compute_quality_tier(0.89) == EnrichmentQualityTier.GOOD
        assert confidence_scorer.compute_quality_tier(0.80) == EnrichmentQualityTier.GOOD
        assert confidence_scorer.compute_quality_tier(0.79) == EnrichmentQualityTier.ACCEPTABLE
        assert confidence_scorer.compute_quality_tier(0.70) == EnrichmentQualityTier.ACCEPTABLE
        assert confidence_scorer.compute_quality_tier(0.69) == EnrichmentQualityTier.POOR


# ============================================================================
# CONFIDENCE CALIBRATION TESTS
# ============================================================================


class TestConfidenceCalibration:
    """Test that confidence scores match actual accuracy."""

    def test_confidence_predicts_accuracy(self, confidence_scorer):
        """High confidence should correlate with high accuracy."""
        # This would require ground truth data
        # Placeholder test - should be implemented with actual data
        pytest.skip("Requires ground truth data for calibration testing")

    def test_calibration_curve(self, confidence_scorer):
        """Should produce well-calibrated confidence scores."""
        # Test that predicted confidence matches observed accuracy
        # For example: if we predict 0.80 confidence, we should be correct ~80% of the time
        pytest.skip("Requires ground truth data for calibration curve")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestConfidenceScoringIntegration:
    """Test complete confidence scoring workflow."""

    def test_end_to_end_scoring(self, confidence_scorer, high_quality_evidence):
        """Should compute all factors and produce final score."""
        # Simulate a complete enrichment
        sources = [KnowledgeBaseType.MEDICAL_CODING, KnowledgeBaseType.PROVIDER_PATTERN]
        age_days = 30
        retrieved_values = {
            "diagnosis_codes": [
                ("E11.9", KnowledgeBaseType.MEDICAL_CODING),
                ("E11.9", KnowledgeBaseType.PROVIDER_PATTERN),
            ]
        }

        # Compute all factors
        retrieval_quality = confidence_scorer.score_retrieval_quality(high_quality_evidence)
        source_diversity = confidence_scorer.score_source_diversity(sources)
        temporal_relevance = confidence_scorer.score_temporal_relevance(age_days)
        cross_validation = confidence_scorer.score_cross_validation(retrieved_values)
        regulatory = confidence_scorer.score_regulatory_citation(False, 0.5)

        # Compute overall
        factors = {
            "retrieval_quality": retrieval_quality,
            "source_diversity": source_diversity,
            "temporal_relevance": temporal_relevance,
            "cross_validation": cross_validation,
            "regulatory_citation": regulatory,
        }
        overall = confidence_scorer.compute_overall_confidence(factors)
        tier = confidence_scorer.compute_quality_tier(overall)

        # Verify all scores are valid
        assert 0.0 <= overall <= 1.0
        assert tier in EnrichmentQualityTier
        assert all(0.0 <= v <= 1.0 for v in factors.values())
