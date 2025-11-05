# Retrieval Patterns for Insurance Fraud Detection RAG

## Executive Summary

This document defines comprehensive retrieval patterns for the insurance fraud detection RAG system using Qdrant's hybrid search capabilities. The system combines semantic similarity (vector search) and lexical matching (BM25) with reranking to achieve >90% retrieval precision while maintaining <100ms latency.

**Key Strategies:**
- **Hybrid Search**: 70% semantic + 30% BM25 (tunable weights)
- **Reranking**: Cross-encoder reranking of top-10 results
- **Reciprocal Rank Fusion (RRF)**: Merge semantic and lexical rankings
- **Query Routing**: Intelligent KB selection based on fraud indicators
- **Result Aggregation**: Deduplication and relevance-based merging

---

## Hybrid Search Architecture

### Query Flow

```
Incoming Claim
    ↓
┌────────────────────────────────────────┐
│ Query Construction                     │
│ • Generate query embedding (1536d)    │
│ • Extract keywords for BM25           │
│ • Identify relevant KBs               │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Parallel Hybrid Search (5 KBs)        │
│                                        │
│ Each KB:                               │
│   Semantic Search (weight: 0.7) ──┐   │
│                                    │   │
│   BM25 Search (weight: 0.3) ──────┤   │
│                                    │   │
│   RRF Fusion ─────────────────────┘   │
│                                        │
│ Returns: top-k results per KB         │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Result Aggregation                     │
│ • Merge results from 5 KBs            │
│ • Deduplicate                          │
│ • Global RRF ranking                   │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Reranking (Cross-Encoder)             │
│ • Rerank top-10 by query relevance    │
│ • Calculate confidence scores          │
└────────┬───────────────────────────────┘
         │
         ▼
Top-k Results with Confidence Scores
```

---

## Query Templates

### 1. Patient History Lookup

**Use Case**: Retrieve historical claims for specific patient

```python
query_template = {
    "kb": "patient_claim_history",
    "query_type": "filter + semantic",
    "filter": {
        "must": [
            {"key": "patient_id", "match": {"value": "{patient_id_hash}"}}
        ]
    },
    "query_embedding": None,  # No semantic search needed for exact match
    "limit": 10,
    "score_threshold": None
}
```

**Example**:
```python
query = {
    "filter": {"must": [{"key": "patient_id", "match": {"value": "hash_8a7f9e2b"}}]},
    "limit": 10
}
results = qdrant_client.scroll(collection_name="patient_claim_history", scroll_filter=query["filter"])
```

---

### 2. Similar Patient Pattern Detection

**Use Case**: Find patients with similar claim patterns (doctor shopping, etc.)

```python
query_template = {
    "kb": "patient_claim_history",
    "query_type": "hybrid_search",
    "query_text": "{patient_narrative}",
    "query_embedding": "[1536d vector]",
    "fusion_params": {
        "semantic_weight": 0.7,
        "bm25_weight": 0.3
    },
    "filter": {
        "should": [
            {"key": "patient_patterns.provider_count_30d", "range": {"gte": 5}},
            {"key": "patient_patterns.early_refill_count", "range": {"gte": 2}}
        ]
    },
    "limit": 10,
    "score_threshold": 0.75
}
```

**Narrative Construction**:
```python
def construct_patient_query(claim):
    """Construct patient pattern query."""
    return f"""Patient with {claim['provider_count']} providers in 30 days.
    Diagnoses: {', '.join(claim['diagnosis_codes'])}.
    {len(claim['procedure_codes'])} procedures totaling ${claim['total_billed']}.
    Red flags: {', '.join(claim.get('red_flags', ['None']))}."""
```

---

### 3. Provider Behavior Lookup

**Use Case**: Retrieve provider billing patterns and compare to benchmarks

```python
query_template = {
    "kb": "provider_behavior_patterns",
    "query_type": "filter + semantic",
    "filter": {
        "must": [
            {"key": "provider_npi", "match": {"value": "{provider_npi_hash}"}}
        ]
    },
    "limit": 1
}
```

---

### 4. Similar Provider Pattern Detection

**Use Case**: Find providers with similar upcoding or phantom billing patterns

```python
query_template = {
    "kb": "provider_behavior_patterns",
    "query_type": "hybrid_search",
    "query_text": "{provider_narrative}",
    "query_embedding": "[1536d vector]",
    "filter": {
        "must": [
            {"key": "specialty", "match": {"value": "{specialty}"}}
        ],
        "should": [
            {"key": "benchmark_comparison.99215_rate.risk_score", "range": {"gte": 0.7}},
            {"key": "temporal_patterns.weekend_billing_rate", "range": {"gte": 0.05}}
        ]
    },
    "limit": 10,
    "score_threshold": 0.8
}
```

---

### 5. Medical Code Validation

**Use Case**: Validate if diagnosis-procedure combination is valid

```python
query_template = {
    "kb": "medical_coding_standards",
    "query_type": "filter + semantic",
    "filter": {
        "must": [
            {"key": "code", "match": {"value": "{icd10_code}"}}
        ]
    },
    "limit": 1
}

# Then check if procedure_code in result['valid_combinations']['valid_procedures']
```

---

### 6. Fraud Risk Combination Lookup

**Use Case**: Find high-risk diagnosis-procedure combinations

```python
query_template = {
    "kb": "medical_coding_standards",
    "query_type": "hybrid_search",
    "query_text": "Diagnosis {icd10_code} with procedure {cpt_code}",
    "filter": {
        "must": [
            {"key": "fraud_risk_combinations.risk_score", "range": {"gte": 0.7}}
        ]
    },
    "limit": 5,
    "score_threshold": 0.8
}
```

---

### 7. Regulatory Guidance Retrieval

**Use Case**: Retrieve fraud detection rules for specific fraud type

```python
query_template = {
    "kb": "regulatory_guidance",
    "query_type": "hybrid_search",
    "query_text": "{fraud_type} detection rules and examples",
    "filter": {
        "must": [
            {"key": "fraud_type", "match": {"value": "{fraud_type}"}}
        ]
    },
    "limit": 3,
    "score_threshold": 0.85
}
```

---

### 8. Similar Fraud Claim Detection

**Use Case**: Find claims similar to known fraudulent claims (primary fraud detection)

```python
query_template = {
    "kb": "claim_similarity_patterns",
    "query_type": "hybrid_search",
    "query_text": "{claim_narrative}",
    "query_embedding": "[1536d vector]",
    "filter": {
        "must": [
            {"key": "fraud_confidence", "range": {"gte": 0.7}}
        ]
    },
    "limit": 10,
    "score_threshold": 0.85
}
```

**Claim Narrative Construction**:
```python
def construct_claim_query(claim):
    """Construct claim similarity query."""
    return f"""Claim: {', '.join(claim['diagnosis_codes'])} diagnoses,
    {', '.join(claim['procedure_codes'])} procedures.
    Billed ${claim['billed_amount']}.
    Provider specialty: {claim.get('provider_specialty', 'unknown')}.
    Red flags: {', '.join(claim.get('red_flags', ['None']))}."""
```

---

## Reciprocal Rank Fusion (RRF)

### Algorithm

```python
def reciprocal_rank_fusion(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60  # RRF constant
) -> List[Dict]:
    """Merge semantic and BM25 results using RRF."""
    scores = {}

    # Calculate RRF scores for semantic results
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    # Calculate RRF scores for BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    # Sort by RRF score
    merged_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return merged_results
```

---

## Reranking

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank results using cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """Rerank results by query relevance."""
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result['embedding_text']) for result in results]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by relevance
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)

        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]
```

---

## Query Router

### Intelligent KB Selection

```python
class QueryRouter:
    """Route queries to relevant KBs."""

    def route(self, claim: Dict) -> Dict[str, bool]:
        """Determine which KBs to query."""
        route = {
            "patient_claim_history": True,  # Always query
            "provider_behavior_patterns": True,  # Always query
            "medical_coding_standards": True,  # Always query
            "regulatory_guidance": self._needs_regulatory(claim),
            "claim_similarity_patterns": self._needs_similarity(claim)
        }
        return route

    def _needs_regulatory(self, claim: Dict) -> bool:
        """Check if regulatory guidance is needed."""
        # Query regulatory KB if claim has red flags
        return len(claim.get('red_flags', [])) > 0

    def _needs_similarity(self, claim: Dict) -> bool:
        """Check if similarity search is needed."""
        # Always query for fraud detection
        return True
```

---

## Implementation

### Complete Retrieval Pipeline

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

class RetrievalPipeline:
    """Complete retrieval pipeline with hybrid search."""

    def __init__(self, qdrant_client: QdrantClient, embedding_generator, reranker):
        self.client = qdrant_client
        self.generator = embedding_generator
        self.reranker = reranker
        self.router = QueryRouter()

    async def retrieve(self, claim: Dict) -> Dict[str, List[Dict]]:
        """Retrieve relevant context from all KBs."""
        # 1. Route query to relevant KBs
        kb_routing = self.router.route(claim)

        # 2. Generate query embedding
        query_text = self._construct_query_text(claim)
        query_embedding = await self.generator.embed_single(query_text)

        # 3. Query all relevant KBs in parallel
        tasks = []
        for kb_name, should_query in kb_routing.items():
            if should_query:
                tasks.append(self._query_kb(kb_name, claim, query_embedding))

        kb_results = await asyncio.gather(*tasks)

        # 4. Aggregate results
        all_results = []
        for kb_name, results in zip([k for k, v in kb_routing.items() if v], kb_results):
            for result in results:
                result['source_kb'] = kb_name
                all_results.append(result)

        # 5. Deduplicate
        deduplicated = self._deduplicate_results(all_results)

        # 6. Rerank top-k
        reranked = self.reranker.rerank(query_text, deduplicated, top_k=10)

        # 7. Group by KB
        grouped_results = {}
        for result in reranked:
            kb = result['source_kb']
            if kb not in grouped_results:
                grouped_results[kb] = []
            grouped_results[kb].append(result)

        return grouped_results

    async def _query_kb(
        self,
        kb_name: str,
        claim: Dict,
        query_embedding: List[float]
    ) -> List[Dict]:
        """Query single KB with hybrid search."""
        # Construct filter based on KB and claim
        filter_condition = self._construct_filter(kb_name, claim)

        # Perform hybrid search
        results = self.client.search(
            collection_name=kb_name,
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=5,
            score_threshold=0.75,
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                'id': result.id,
                'score': result.score,
                'payload': result.payload,
                'embedding_text': result.payload.get('embedding_text', '')
            }
            for result in results
        ]

    def _construct_filter(self, kb_name: str, claim: Dict) -> Filter:
        """Construct filter conditions for KB query."""
        if kb_name == "patient_claim_history":
            # Filter by patient_id if available
            if 'patient_id_hash' in claim:
                return Filter(
                    must=[
                        FieldCondition(
                            key="patient_id",
                            match=MatchValue(value=claim['patient_id_hash'])
                        )
                    ]
                )

        elif kb_name == "provider_behavior_patterns":
            # Filter by provider NPI
            if 'provider_npi_hash' in claim:
                return Filter(
                    must=[
                        FieldCondition(
                            key="provider_npi",
                            match=MatchValue(value=claim['provider_npi_hash'])
                        )
                    ]
                )

        elif kb_name == "medical_coding_standards":
            # Filter by diagnosis codes
            if 'diagnosis_codes' in claim:
                return Filter(
                    should=[
                        FieldCondition(
                            key="code",
                            match=MatchValue(value=code)
                        )
                        for code in claim['diagnosis_codes']
                    ]
                )

        elif kb_name == "claim_similarity_patterns":
            # Filter by high confidence fraud
            return Filter(
                must=[
                    FieldCondition(
                        key="fraud_confidence",
                        range=Range(gte=0.7)
                    )
                ]
            )

        return None

    def _construct_query_text(self, claim: Dict) -> str:
        """Construct query text from claim."""
        diagnosis_desc = ', '.join([f"{c} ({d})" for c, d in zip(
            claim.get('diagnosis_codes', []),
            claim.get('diagnosis_descriptions', [])
        )])
        procedure_desc = ', '.join([f"{c} ({d})" for c, d in zip(
            claim.get('procedure_codes', []),
            claim.get('procedure_descriptions', [])
        )])

        query = f"""Claim analysis:
        Diagnoses: {diagnosis_desc}
        Procedures: {procedure_desc}
        Billed amount: ${claim.get('billed_amount', 0)}
        Red flags: {', '.join(claim.get('red_flags', ['None']))}"""

        return query

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results."""
        seen = set()
        deduplicated = []

        for result in results:
            # Use embedding_text as deduplication key
            key = result['embedding_text']
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated
```

---

## Performance Targets

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| Query Latency (per KB) | <10ms | <20ms | <50ms |
| Total Retrieval Latency | <60ms | <100ms | <150ms |
| Precision@10 | >90% | >85% | >75% |
| Recall@10 | >85% | >80% | >70% |
| Reranking Latency | <20ms | <40ms | <80ms |

---

## References

1. Qdrant Hybrid Search: https://qdrant.tech/articles/hybrid-search/
2. Reciprocal Rank Fusion: weaviate.io/blog/hybrid-search-explained
3. Cross-Encoder Reranking: medium.com/@ashpaklmulani/improve-retrieval-augmented-generation-rag-with-re-ranking

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: Design Complete
