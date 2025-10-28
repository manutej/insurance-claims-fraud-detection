# RAG Integration Points for Insurance Fraud Detection

## Executive Summary

This document defines the integration architecture between the RAG knowledge base system and the fraud detection pipeline. The integration provides seamless context retrieval, LLM reasoning augmentation, and explainable fraud analysis with <100ms latency.

**Key Integration Points:**
1. **Fraud Detection Pipeline**: RAG context injection for incoming claims
2. **LLM Reasoning Engine**: Augmented prompt construction with retrieved evidence
3. **Explainability Module**: Citation generation and reasoning chains
4. **Feedback Loop**: Continuous learning from fraud analyst decisions
5. **Monitoring Dashboard**: Real-time RAG performance metrics

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Incoming Claim (JSON)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Integration Point 1: Fraud Detection Pipeline        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FraudDetector.analyze_claim(claim_json)                │  │
│  │  ↓                                                       │  │
│  │  1. Validate claim schema                               │  │
│  │  2. Extract features                                    │  │
│  │  3. *** RAG Context Retrieval *** ←─────────────┐      │  │
│  │  4. ML Model scoring                              │      │  │
│  │  5. LLM Reasoning (with RAG context)              │      │  │
│  │  6. Generate explanation                          │      │  │
│  └──────────────────────────────────────────────────│──────┘  │
└──────────────────────────────────────────────────────│──────────┘
                                                        │
                                  ┌─────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│              RAG Retrieval System                                 │
│                                                                   │
│  RetrievalPipeline.retrieve(claim)                              │
│  ↓                                                               │
│  1. Route query to relevant KBs                                 │
│  2. Generate embeddings                                         │
│  3. Hybrid search (semantic + BM25)                            │
│  4. Aggregate & rerank results                                 │
│  5. Calculate confidence scores                                 │
│  6. Format context for LLM                                      │
│                                                                   │
│  Returns: {                                                      │
│    "context": {...},                                            │
│    "confidence": 0.87,                                          │
│    "sources": [...]                                             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│         Integration Point 2: LLM Reasoning Engine                 │
│                                                                   │
│  Augmented Prompt = System Prompt + RAG Context + Claim         │
│  ↓                                                               │
│  LLM analyzes with:                                             │
│  • Patient history context                                      │
│  • Provider behavior patterns                                   │
│  • Medical coding standards                                     │
│  • Regulatory guidance                                          │
│  • Similar fraud patterns                                       │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│         Integration Point 3: Explainability Module                │
│                                                                   │
│  Generate human-readable explanation with:                       │
│  • Fraud score + confidence                                     │
│  • Key evidence (with citations)                                │
│  • Reasoning chain                                              │
│  • Regulatory violations (if any)                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Integration Point 1: Fraud Detection Pipeline

### API Interface

```python
from typing import Dict, Optional
from rag_retrieval import RetrievalPipeline
from confidence_scorer import ConfidenceScorer

class FraudDetector:
    """Main fraud detection class with RAG integration."""

    def __init__(
        self,
        ml_model,
        retrieval_pipeline: RetrievalPipeline,
        confidence_scorer: ConfidenceScorer,
        llm_client
    ):
        self.ml_model = ml_model
        self.retrieval = retrieval_pipeline
        self.scorer = confidence_scorer
        self.llm = llm_client

    async def analyze_claim(self, claim: Dict) -> Dict:
        """
        Analyze claim for fraud with RAG-augmented reasoning.

        Args:
            claim: Claim JSON with diagnosis codes, procedures, amounts, etc.

        Returns:
            {
                "claim_id": str,
                "fraud_score": float (0-1),
                "fraud_confidence": float (0-1),
                "is_fraud": bool,
                "fraud_type": str | None,
                "explanation": str,
                "evidence": List[Dict],
                "rag_context": Dict,
                "recommendation": str
            }
        """
        # 1. Validate claim schema
        self._validate_claim(claim)

        # 2. Extract ML features
        features = self._extract_features(claim)

        # 3. *** RAG CONTEXT RETRIEVAL ***
        rag_results = await self.retrieval.retrieve(claim)

        # 4. Calculate confidence scores
        confidence_result = self.scorer.score(
            results=rag_results.get('all_results', []),
            query=claim
        )

        # 5. ML Model scoring
        ml_fraud_score = self.ml_model.predict_proba(features)[0][1]

        # 6. LLM Reasoning (with RAG context)
        llm_analysis = await self._llm_analyze_with_context(
            claim,
            rag_results,
            confidence_result,
            ml_fraud_score
        )

        # 7. Generate final result
        result = {
            "claim_id": claim['claim_id'],
            "fraud_score": llm_analysis['fraud_score'],
            "fraud_confidence": confidence_result['confidence_score'],
            "is_fraud": llm_analysis['is_fraud'],
            "fraud_type": llm_analysis.get('fraud_type'),
            "explanation": llm_analysis['explanation'],
            "evidence": self._format_evidence(rag_results),
            "rag_context": {
                "confidence": confidence_result['confidence_score'],
                "component_scores": confidence_result['component_scores'],
                "sources_used": self._get_sources_summary(rag_results)
            },
            "recommendation": confidence_result['recommendation'],
            "ml_score": ml_fraud_score
        }

        return result
```

### Context Injection

```python
async def _retrieve_rag_context(self, claim: Dict) -> Dict:
    """Retrieve RAG context for claim."""
    # Query all relevant KBs
    results = await self.retrieval.retrieve(claim)

    # Format for easy consumption
    context = {
        "patient_history": results.get('patient_claim_history', []),
        "provider_patterns": results.get('provider_behavior_patterns', []),
        "coding_standards": results.get('medical_coding_standards', []),
        "regulatory_guidance": results.get('regulatory_guidance', []),
        "similar_frauds": results.get('claim_similarity_patterns', []),
        "all_results": []  # For confidence scoring
    }

    # Flatten for confidence scoring
    for kb_results in results.values():
        context['all_results'].extend(kb_results)

    return context
```

---

## Integration Point 2: LLM Reasoning Engine

### Augmented Prompt Construction

```python
async def _llm_analyze_with_context(
    self,
    claim: Dict,
    rag_context: Dict,
    confidence_result: Dict,
    ml_score: float
) -> Dict:
    """Analyze claim with LLM using RAG context."""

    # Construct augmented prompt
    system_prompt = """You are an expert insurance fraud detection analyst.
    Analyze the provided claim using the context from our knowledge bases.
    Provide a fraud assessment with clear reasoning and evidence citations."""

    user_prompt = self._construct_augmented_prompt(
        claim,
        rag_context,
        confidence_result,
        ml_score
    )

    # Call LLM
    response = await self.llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,  # Low temperature for consistency
        response_format={"type": "json_object"}
    )

    # Parse LLM response
    llm_result = json.loads(response.choices[0].message.content)

    return llm_result

def _construct_augmented_prompt(
    self,
    claim: Dict,
    rag_context: Dict,
    confidence_result: Dict,
    ml_score: float
) -> str:
    """Construct prompt with RAG context."""

    prompt = f"""# Claim Analysis Request

## Claim Details
- Claim ID: {claim['claim_id']}
- Patient ID: {claim.get('patient_id', 'N/A')}
- Provider NPI: {claim.get('provider_npi', 'N/A')}
- Date of Service: {claim.get('date_of_service', 'N/A')}
- Diagnosis Codes: {', '.join(claim.get('diagnosis_codes', []))}
- Diagnosis Descriptions: {', '.join(claim.get('diagnosis_descriptions', []))}
- Procedure Codes: {', '.join(claim.get('procedure_codes', []))}
- Procedure Descriptions: {', '.join(claim.get('procedure_descriptions', []))}
- Billed Amount: ${claim.get('billed_amount', 0)}

## ML Model Assessment
- Fraud Score: {ml_score:.2f}
- Interpretation: {"High Risk" if ml_score > 0.7 else "Medium Risk" if ml_score > 0.5 else "Low Risk"}

## Knowledge Base Context (Confidence: {confidence_result['confidence_score']:.2f})

"""

    # Add patient history
    if rag_context['patient_history']:
        prompt += "### Patient Claim History\n"
        for doc in rag_context['patient_history'][:2]:  # Top 2 results
            prompt += f"- {doc['payload']['embedding_text'][:200]}...\n"
        prompt += "\n"

    # Add provider patterns
    if rag_context['provider_patterns']:
        prompt += "### Provider Behavior Patterns\n"
        for doc in rag_context['provider_patterns'][:2]:
            prompt += f"- {doc['payload']['embedding_text'][:200]}...\n"
        prompt += "\n"

    # Add coding standards
    if rag_context['coding_standards']:
        prompt += "### Medical Coding Standards\n"
        for doc in rag_context['coding_standards'][:2]:
            prompt += f"- {doc['payload']['embedding_text'][:200]}...\n"
        prompt += "\n"

    # Add regulatory guidance
    if rag_context['regulatory_guidance']:
        prompt += "### Regulatory Guidance\n"
        for doc in rag_context['regulatory_guidance'][:2]:
            prompt += f"- {doc['payload']['embedding_text'][:200]}...\n"
        prompt += "\n"

    # Add similar fraud patterns
    if rag_context['similar_frauds']:
        prompt += "### Similar Fraud Patterns\n"
        for doc in rag_context['similar_frauds'][:3]:  # Top 3 matches
            fraud_type = doc['payload'].get('fraud_type', 'unknown')
            confidence = doc['payload'].get('fraud_confidence', 0)
            prompt += f"- {fraud_type.upper()} (confidence: {confidence:.2f}): {doc['payload']['embedding_text'][:150]}...\n"
        prompt += "\n"

    # Add analysis instructions
    prompt += """## Analysis Task

Based on the claim details, ML assessment, and knowledge base context, provide:

1. **Fraud Assessment**: Is this claim fraudulent? (yes/no)
2. **Fraud Score**: Overall fraud probability (0.0-1.0)
3. **Fraud Type**: If fraudulent, what type? (upcoding, phantom_billing, unbundling, etc.)
4. **Key Evidence**: 3-5 specific evidence points from the context
5. **Reasoning**: Step-by-step explanation of your analysis
6. **Regulatory Violations**: Any specific regulatory violations (with citations)
7. **Confidence**: Your confidence in this assessment (0.0-1.0)

Respond in JSON format with these fields:
{
    "is_fraud": bool,
    "fraud_score": float,
    "fraud_type": str | null,
    "evidence": [{"source": str, "quote": str, "relevance": str}],
    "reasoning": str,
    "regulatory_violations": [{"violation": str, "citation": str}],
    "confidence": float
}
"""

    return prompt
```

---

## Integration Point 3: Explainability Module

### Evidence Formatting

```python
def _format_evidence(self, rag_results: Dict) -> List[Dict]:
    """Format RAG results as evidence for explanation."""
    evidence = []

    # Collect top evidence from each KB
    for kb_name, results in rag_results.items():
        if kb_name == 'all_results':
            continue

        for result in results[:2]:  # Top 2 per KB
            evidence.append({
                "source_kb": kb_name,
                "document_id": result['id'],
                "quote": result['payload']['embedding_text'][:300],
                "relevance_score": result.get('score', 0),
                "rerank_score": result.get('rerank_score', 0),
                "metadata": result.get('payload', {}).get('metadata', {})
            })

    # Sort by rerank score
    evidence.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

    return evidence[:5]  # Return top 5 pieces of evidence
```

### Human-Readable Explanation

```python
def generate_explanation(
    self,
    fraud_result: Dict,
    rag_context: Dict,
    evidence: List[Dict]
) -> str:
    """Generate human-readable explanation."""

    explanation = f"""# Fraud Analysis for Claim {fraud_result['claim_id']}

## Assessment: {"FRAUD DETECTED" if fraud_result['is_fraud'] else "NO FRAUD DETECTED"}

**Fraud Score**: {fraud_result['fraud_score']:.2f} / 1.00
**Confidence**: {fraud_result['fraud_confidence']:.2f} / 1.00
**Fraud Type**: {fraud_result.get('fraud_type', 'N/A')}

## Reasoning

{fraud_result['explanation']}

## Supporting Evidence

"""

    # Add evidence with citations
    for i, ev in enumerate(evidence, 1):
        source_name = ev['source_kb'].replace('_', ' ').title()
        explanation += f"{i}. **{source_name}** (Relevance: {ev['relevance_score']:.2f})\n"
        explanation += f"   {ev['quote'][:200]}...\n\n"

    # Add regulatory violations if any
    if fraud_result.get('regulatory_violations'):
        explanation += "## Regulatory Violations\n\n"
        for violation in fraud_result['regulatory_violations']:
            explanation += f"- **{violation['violation']}**\n"
            explanation += f"  Citation: {violation['citation']}\n\n"

    # Add recommendation
    explanation += f"## Recommendation\n\n{rag_context['recommendation']}\n"

    return explanation
```

---

## Integration Point 4: Feedback Loop

### Analyst Feedback Collection

```python
class FeedbackCollector:
    """Collect and process fraud analyst feedback."""

    def __init__(self, qdrant_client, embedding_generator):
        self.qdrant = qdrant_client
        self.generator = embedding_generator

    async def record_feedback(
        self,
        claim_id: str,
        prediction: Dict,
        actual_outcome: Dict,
        analyst_notes: str
    ):
        """Record analyst feedback for continuous learning."""

        feedback = {
            "claim_id": claim_id,
            "predicted_fraud": prediction['is_fraud'],
            "actual_fraud": actual_outcome['is_fraud'],
            "predicted_type": prediction.get('fraud_type'),
            "actual_type": actual_outcome.get('fraud_type'),
            "prediction_confidence": prediction['fraud_confidence'],
            "was_correct": prediction['is_fraud'] == actual_outcome['is_fraud'],
            "analyst_notes": analyst_notes,
            "timestamp": datetime.now().isoformat()
        }

        # If prediction was incorrect, analyze why
        if not feedback['was_correct']:
            await self._analyze_failure(claim_id, prediction, actual_outcome)

        # Store feedback
        await self._store_feedback(feedback)

        # Update claim similarity KB with confirmed fraud
        if actual_outcome['is_fraud']:
            await self._add_to_fraud_kb(claim_id, actual_outcome)

    async def _add_to_fraud_kb(self, claim_id: str, fraud_details: Dict):
        """Add confirmed fraud to claim similarity KB."""
        # Retrieve original claim
        claim = await self._get_claim(claim_id)

        # Generate embedding
        embedding_text = self._construct_fraud_embedding_text(claim, fraud_details)
        embedding = await self.generator.embed_single(embedding_text)

        # Upsert to claim_similarity_patterns KB
        await self.qdrant.upsert(
            collection_name="claim_similarity_patterns",
            points=[
                {
                    "id": claim_id,
                    "vector": embedding,
                    "payload": {
                        "claim_id": claim_id,
                        "fraud_type": fraud_details['fraud_type'],
                        "fraud_confidence": 1.0,  # Confirmed by analyst
                        "claim_features": claim,
                        "analyst_notes": fraud_details.get('notes', ''),
                        "embedding_text": embedding_text,
                        "metadata": {
                            "created_at": datetime.now().isoformat(),
                            "verified_fraud": True
                        }
                    }
                }
            ]
        )
```

---

## Integration Point 5: Monitoring Dashboard

### Key Metrics

```python
class RAGMonitor:
    """Monitor RAG system performance."""

    def __init__(self):
        self.metrics = {
            "retrieval_latency": [],
            "confidence_scores": [],
            "cache_hit_rate": [],
            "kb_query_counts": {},
            "false_positives": 0,
            "false_negatives": 0
        }

    def record_retrieval(
        self,
        latency_ms: float,
        confidence: float,
        cache_hit: bool,
        kbs_queried: List[str]
    ):
        """Record retrieval metrics."""
        self.metrics["retrieval_latency"].append(latency_ms)
        self.metrics["confidence_scores"].append(confidence)
        self.metrics["cache_hit_rate"].append(1 if cache_hit else 0)

        for kb in kbs_queried:
            self.metrics["kb_query_counts"][kb] = self.metrics["kb_query_counts"].get(kb, 0) + 1

    def get_dashboard_data(self) -> Dict:
        """Get metrics for monitoring dashboard."""
        return {
            "latency_p50": np.percentile(self.metrics["retrieval_latency"], 50),
            "latency_p99": np.percentile(self.metrics["retrieval_latency"], 99),
            "avg_confidence": np.mean(self.metrics["confidence_scores"]),
            "cache_hit_rate": np.mean(self.metrics["cache_hit_rate"]),
            "kb_usage": self.metrics["kb_query_counts"],
            "false_positive_rate": self.metrics["false_positives"] / max(len(self.metrics["confidence_scores"]), 1),
            "false_negative_rate": self.metrics["false_negatives"] / max(len(self.metrics["confidence_scores"]), 1)
        }
```

---

## Performance Targets

| Integration Point | Latency Target | Acceptable | Critical |
|-------------------|----------------|------------|----------|
| RAG Context Retrieval | <60ms | <100ms | <150ms |
| LLM Analysis | <2s | <5s | <10s |
| Total Pipeline | <3s | <6s | <12s |
| Explanation Generation | <100ms | <200ms | <500ms |

---

## Error Handling

```python
async def analyze_claim_with_fallback(self, claim: Dict) -> Dict:
    """Analyze claim with fallback strategies."""
    try:
        # Try with RAG context
        return await self.analyze_claim(claim)

    except RAGTimeoutError:
        # Fallback: Use ML model only
        logger.warning(f"RAG timeout for claim {claim['claim_id']}, using ML fallback")
        return await self._ml_only_analysis(claim)

    except RAGLowConfidenceError:
        # Fallback: Flag for human review
        logger.warning(f"Low RAG confidence for claim {claim['claim_id']}")
        return {
            "claim_id": claim['claim_id'],
            "recommendation": "HUMAN_REVIEW_REQUIRED",
            "reason": "Insufficient context confidence"
        }

    except Exception as e:
        # Fallback: Conservative approach
        logger.error(f"Error analyzing claim {claim['claim_id']}: {e}")
        return {
            "claim_id": claim['claim_id'],
            "recommendation": "MANUAL_REVIEW",
            "reason": "System error - requires manual investigation"
        }
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: Design Complete - Ready for Implementation
