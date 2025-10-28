# Confidence Scoring Algorithm for RAG Fraud Detection

## Executive Summary

This document defines a comprehensive confidence scoring algorithm that assesses the quality and reliability of RAG-retrieved context for fraud detection. The algorithm combines retrieval scores, source diversity, temporal relevance, and cross-validation to produce an interpretable confidence score (0-1) with explainability.

**Key Components:**
- **Retrieval Quality Score** (40%): Semantic similarity + BM25 relevance
- **Source Diversity Score** (20%): Number of unique KBs and documents
- **Temporal Relevance Score** (15%): Recency of retrieved documents
- **Cross-Validation Score** (15%): Agreement across multiple sources
- **Regulatory Citation Score** (10%): Presence of authoritative sources

**Target**: >0.80 confidence for automatic decisions, 0.60-0.80 for human review, <0.60 reject

---

## Confidence Score Formula

### Overall Score

```
ConfidenceScore = (
    0.40 × RetrievalQualityScore +
    0.20 × SourceDiversityScore +
    0.15 × TemporalRelevanceScore +
    0.15 × CrossValidationScore +
    0.10 × RegulatoryCitationScore
)
```

---

## Component Calculations

### 1. Retrieval Quality Score (40%)

**Purpose**: Assess semantic and lexical relevance of retrieved documents.

**Formula**:
```
RetrievalQualityScore = (
    0.70 × SemanticScore +
    0.30 × BM25Score
)

SemanticScore = mean([cosine_similarity(query, doc) for doc in top_k])
BM25Score = mean([bm25_score(query, doc) for doc in top_k])
```

**Normalization**:
```python
def normalize_semantic_score(similarity: float) -> float:
    """Normalize cosine similarity (0-1) to quality score."""
    # Cosine similarity of 0.85+ is excellent
    # Below 0.70 is poor
    if similarity >= 0.85:
        return 1.0
    elif similarity >= 0.70:
        return (similarity - 0.70) / 0.15
    else:
        return 0.0

def normalize_bm25_score(bm25_score: float, max_score: float = 10.0) -> float:
    """Normalize BM25 score to 0-1 range."""
    return min(bm25_score / max_score, 1.0)
```

**Implementation**:
```python
def calculate_retrieval_quality_score(results: List[Dict]) -> float:
    """Calculate retrieval quality score."""
    if not results:
        return 0.0

    semantic_scores = [normalize_semantic_score(r['semantic_score']) for r in results]
    bm25_scores = [normalize_bm25_score(r['bm25_score']) for r in results]

    semantic_score = np.mean(semantic_scores)
    bm25_score = np.mean(bm25_scores)

    return 0.70 * semantic_score + 0.30 * bm25_score
```

---

### 2. Source Diversity Score (20%)

**Purpose**: Reward retrieval of evidence from multiple independent sources.

**Formula**:
```
SourceDiversityScore = (
    0.50 × KBDiversityScore +
    0.50 × DocumentDiversityScore
)

KBDiversityScore = unique_kbs_used / total_kbs_available

DocumentDiversityScore = min(unique_documents / 5, 1.0)
```

**Rationale**:
- Evidence from multiple KBs is more reliable
- At least 5 unique documents indicates thorough retrieval
- Diminishing returns after 5 documents

**Implementation**:
```python
def calculate_source_diversity_score(results: List[Dict]) -> float:
    """Calculate source diversity score."""
    if not results:
        return 0.0

    unique_kbs = len(set(r['source_kb'] for r in results))
    unique_docs = len(set(r['id'] for r in results))

    kb_diversity = unique_kbs / 5.0  # 5 KBs total
    doc_diversity = min(unique_docs / 5.0, 1.0)

    return 0.50 * kb_diversity + 0.50 * doc_diversity
```

---

### 3. Temporal Relevance Score (15%)

**Purpose**: Prioritize recent documents (regulatory updates, current fraud patterns).

**Formula**:
```
TemporalRelevanceScore = mean([
    temporal_decay(doc_age_days) for doc in results
])

temporal_decay(age_days) = exp(-age_days / 365)
```

**Decay Function**:
- Documents <1 year old: Full score (1.0)
- Documents 1-2 years old: Gradual decay
- Documents >3 years old: Minimal score (0.05)

**Implementation**:
```python
from datetime import datetime
import numpy as np

def calculate_temporal_relevance_score(results: List[Dict]) -> float:
    """Calculate temporal relevance score."""
    if not results:
        return 0.0

    current_date = datetime.now()
    decay_scores = []

    for result in results:
        doc_date = datetime.fromisoformat(result['metadata']['created_at'])
        age_days = (current_date - doc_date).days

        # Exponential decay with 365-day half-life
        decay_score = np.exp(-age_days / 365)
        decay_scores.append(decay_score)

    return np.mean(decay_scores)
```

---

### 4. Cross-Validation Score (15%)

**Purpose**: Measure agreement/consistency across multiple sources.

**Formula**:
```
CrossValidationScore = (
    0.60 × FraudTypeAgreementScore +
    0.40 × RiskScoreConsistencyScore
)

FraudTypeAgreementScore = (
    num_sources_agreeing_on_fraud_type / total_sources
)

RiskScoreConsistencyScore = 1 - (
    std_dev(risk_scores) / mean(risk_scores)
)
```

**Rationale**:
- Multiple sources identifying same fraud type increases confidence
- Consistent risk scores across sources indicates reliability

**Implementation**:
```python
from collections import Counter

def calculate_cross_validation_score(results: List[Dict]) -> float:
    """Calculate cross-validation score."""
    if not results:
        return 0.0

    # Extract fraud types and risk scores
    fraud_types = [r['payload'].get('fraud_type') for r in results if 'fraud_type' in r['payload']]
    risk_scores = [r['payload'].get('risk_score', r.get('score', 0)) for r in results]

    # Fraud type agreement
    if fraud_types:
        fraud_type_counts = Counter(fraud_types)
        most_common_fraud = fraud_type_counts.most_common(1)[0]
        fraud_agreement = most_common_fraud[1] / len(fraud_types)
    else:
        fraud_agreement = 0.0

    # Risk score consistency
    if len(risk_scores) >= 2:
        mean_risk = np.mean(risk_scores)
        std_risk = np.std(risk_scores)
        # Coefficient of variation (lower is better)
        cv = std_risk / mean_risk if mean_risk > 0 else 1.0
        risk_consistency = max(1.0 - cv, 0.0)
    else:
        risk_consistency = 0.5  # Neutral score for insufficient data

    return 0.60 * fraud_agreement + 0.40 * risk_consistency
```

---

### 5. Regulatory Citation Score (10%)

**Purpose**: Reward presence of authoritative regulatory sources.

**Formula**:
```
RegulatoryCitationScore = (
    0.50 × has_regulatory_source +
    0.30 × has_cms_source +
    0.20 × has_nfis_source
)
```

**Authoritative Sources**:
- NY DOF (Department of Financial Services)
- CMS (Centers for Medicare & Medicaid Services)
- NFIS (National Healthcare Anti-Fraud Association)
- OIG (Office of Inspector General)

**Implementation**:
```python
def calculate_regulatory_citation_score(results: List[Dict]) -> float:
    """Calculate regulatory citation score."""
    if not results:
        return 0.0

    authoritative_sources = {
        'ny_dof': False,
        'cms': False,
        'nfis': False,
        'oig': False
    }

    for result in results:
        payload = result.get('payload', {})
        regulatory_source = payload.get('regulatory_source', {})
        source_text = str(payload).lower()

        if 'ny dof' in source_text or 'department of financial services' in source_text:
            authoritative_sources['ny_dof'] = True
        if 'cms' in source_text or 'medicare' in source_text:
            authoritative_sources['cms'] = True
        if 'nfis' in source_text or 'anti-fraud association' in source_text:
            authoritative_sources['nfis'] = True
        if 'oig' in source_text or 'inspector general' in source_text:
            authoritative_sources['oig'] = True

    # Score based on presence of authoritative sources
    score = 0.0
    if any(authoritative_sources.values()):
        score += 0.50  # Has at least one regulatory source
    if authoritative_sources['cms']:
        score += 0.30  # CMS is highly authoritative
    if authoritative_sources['nfis'] or authoritative_sources['ny_dof']:
        score += 0.20  # NFIS or NY DOF

    return min(score, 1.0)
```

---

## Complete Confidence Scorer

```python
class ConfidenceScorer:
    """Calculate confidence scores for RAG retrievals."""

    def __init__(self):
        self.weights = {
            'retrieval_quality': 0.40,
            'source_diversity': 0.20,
            'temporal_relevance': 0.15,
            'cross_validation': 0.15,
            'regulatory_citation': 0.10
        }

    def score(self, results: List[Dict], query: str) -> Dict:
        """Calculate comprehensive confidence score."""
        # Calculate component scores
        retrieval_quality = calculate_retrieval_quality_score(results)
        source_diversity = calculate_source_diversity_score(results)
        temporal_relevance = calculate_temporal_relevance_score(results)
        cross_validation = calculate_cross_validation_score(results)
        regulatory_citation = calculate_regulatory_citation_score(results)

        # Calculate overall confidence
        confidence = (
            self.weights['retrieval_quality'] * retrieval_quality +
            self.weights['source_diversity'] * source_diversity +
            self.weights['temporal_relevance'] * temporal_relevance +
            self.weights['cross_validation'] * cross_validation +
            self.weights['regulatory_citation'] * regulatory_citation
        )

        # Generate explanation
        explanation = self._generate_explanation(
            confidence,
            retrieval_quality,
            source_diversity,
            temporal_relevance,
            cross_validation,
            regulatory_citation,
            results
        )

        return {
            'confidence_score': confidence,
            'component_scores': {
                'retrieval_quality': retrieval_quality,
                'source_diversity': source_diversity,
                'temporal_relevance': temporal_relevance,
                'cross_validation': cross_validation,
                'regulatory_citation': regulatory_citation
            },
            'explanation': explanation,
            'recommendation': self._get_recommendation(confidence)
        }

    def _generate_explanation(
        self,
        confidence: float,
        retrieval_quality: float,
        source_diversity: float,
        temporal_relevance: float,
        cross_validation: float,
        regulatory_citation: float,
        results: List[Dict]
    ) -> str:
        """Generate human-readable explanation."""
        explanation_parts = []

        # Overall assessment
        if confidence >= 0.80:
            explanation_parts.append("HIGH CONFIDENCE: Strong evidence from multiple reliable sources.")
        elif confidence >= 0.60:
            explanation_parts.append("MEDIUM CONFIDENCE: Sufficient evidence but requires review.")
        else:
            explanation_parts.append("LOW CONFIDENCE: Insufficient or conflicting evidence.")

        # Retrieval quality
        if retrieval_quality >= 0.80:
            explanation_parts.append(f"Retrieved documents highly relevant (score: {retrieval_quality:.2f}).")
        elif retrieval_quality < 0.60:
            explanation_parts.append(f"Low retrieval relevance (score: {retrieval_quality:.2f}) - results may be tangential.")

        # Source diversity
        unique_kbs = len(set(r['source_kb'] for r in results))
        unique_docs = len(set(r['id'] for r in results))
        explanation_parts.append(f"Evidence from {unique_kbs} knowledge bases, {unique_docs} documents.")

        # Cross-validation
        if cross_validation >= 0.70:
            explanation_parts.append("Strong agreement across sources.")
        elif cross_validation < 0.50:
            explanation_parts.append("Conflicting information across sources - requires investigation.")

        # Regulatory citation
        if regulatory_citation >= 0.50:
            explanation_parts.append("Supported by authoritative regulatory sources.")

        return " ".join(explanation_parts)

    def _get_recommendation(self, confidence: float) -> str:
        """Get action recommendation based on confidence."""
        if confidence >= 0.80:
            return "AUTOMATIC_DECISION"
        elif confidence >= 0.60:
            return "HUMAN_REVIEW_RECOMMENDED"
        else:
            return "REJECT_INSUFFICIENT_EVIDENCE"
```

---

## Confidence Thresholds

### Decision Matrix

| Confidence Range | Recommendation | Action |
|------------------|----------------|--------|
| 0.80 - 1.00 | Automatic Decision | Proceed with fraud flag/no-fraud determination |
| 0.60 - 0.79 | Human Review | Flag for expert review with RAG context |
| 0.40 - 0.59 | Insufficient Evidence | Request additional documentation |
| 0.00 - 0.39 | Reject | Unable to make determination, escalate |

---

## Example Output

```python
{
    "confidence_score": 0.87,
    "component_scores": {
        "retrieval_quality": 0.92,
        "source_diversity": 0.80,
        "temporal_relevance": 0.85,
        "cross_validation": 0.88,
        "regulatory_citation": 0.80
    },
    "explanation": "HIGH CONFIDENCE: Strong evidence from multiple reliable sources. Retrieved documents highly relevant (score: 0.92). Evidence from 4 knowledge bases, 8 documents. Strong agreement across sources. Supported by authoritative regulatory sources.",
    "recommendation": "AUTOMATIC_DECISION",
    "evidence": [
        {
            "source": "regulatory_guidance_kb",
            "document": "Upcoding Detection Rules",
            "quote": "Simple diagnosis (J00) billed with high complexity (99215) - risk weight 0.9",
            "score": 0.94,
            "kb": "regulatory_guidance"
        },
        {
            "source": "medical_coding_standards_kb",
            "document": "ICD-10 code J00 standards",
            "quote": "Common cold should NEVER be billed with 99215",
            "score": 0.91,
            "kb": "medical_coding_standards"
        },
        {
            "source": "provider_behavior_patterns_kb",
            "document": "Provider NPI hash_9a8b7c6d statistics",
            "quote": "Provider bills 90% of visits at 99215 (benchmark: <5%)",
            "score": 0.88,
            "kb": "provider_behavior_patterns"
        }
    ]
}
```

---

## Monitoring & Quality Assurance

### Key Metrics

**Confidence Score Distribution**:
- Target: >60% of claims with confidence ≥0.80
- Warning: >20% of claims with confidence <0.60

**Correlation with Fraud Detection Accuracy**:
- High confidence claims should have >95% accuracy
- Low confidence claims <70% accuracy (requires human review)

### Calibration

Periodically calibrate confidence scores against ground truth:

```python
def calibrate_confidence_scores(
    predictions: List[Dict],
    ground_truth: List[bool]
) -> Dict:
    """Calibrate confidence scores against actual fraud outcomes."""
    confidence_bins = [0.0, 0.4, 0.6, 0.8, 1.0]
    bin_accuracies = []

    for i in range(len(confidence_bins) - 1):
        low, high = confidence_bins[i], confidence_bins[i+1]

        # Get predictions in this confidence bin
        bin_predictions = [
            (pred, truth)
            for pred, truth in zip(predictions, ground_truth)
            if low <= pred['confidence_score'] < high
        ]

        if bin_predictions:
            correct = sum(pred['is_fraud'] == truth for pred, truth in bin_predictions)
            accuracy = correct / len(bin_predictions)
            bin_accuracies.append(accuracy)

    return {
        'bin_edges': confidence_bins,
        'bin_accuracies': bin_accuracies,
        'calibration_quality': 'GOOD' if all(acc >= 0.80 for acc in bin_accuracies) else 'NEEDS_ADJUSTMENT'
    }
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: Design Complete
