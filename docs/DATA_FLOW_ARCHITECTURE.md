# Data Flow Architecture for Insurance Fraud Detection System

## Executive Summary

This document defines the end-to-end data flow architecture for the insurance fraud detection system, covering claim ingestion, enrichment, vector embedding, RAG-based fraud analysis, and result delivery. The architecture supports real-time processing, batch analysis, and continuous learning from feedback loops.

## 1. High-Level Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Architecture                          │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│  External   │
│  Data       │
│  Sources    │
└──────┬──────┘
       │
       │ Claims Data (JSON/XML/EDI)
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │   API       │  │   S3/SFTP    │  │   Message Queue        │   │
│  │   Gateway   │  │   Ingestion  │  │   (SQS/Kafka)          │   │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────────┘   │
│         │                │                      │                   │
│         └────────────────┴──────────────────────┘                   │
│                            │                                         │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │   Schema       │  │   Medical    │  │   Business Rules   │     │
│  │   Validation   │  │   Code       │  │   Validation       │     │
│  │   (Pydantic)   │  │   Validation │  │                    │     │
│  └────────┬───────┘  └──────┬───────┘  └─────────┬──────────┘     │
│           │                 │                     │                 │
│           └─────────────────┴─────────────────────┘                 │
│                            │                                         │
│                            ├─────► [Dead Letter Queue] (Invalid)    │
│                            │                                         │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENRICHMENT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │   Provider       │  │   Patient        │  │   Historical   │   │
│  │   Lookup         │  │   History        │  │   Patterns     │   │
│  │   (NPI Registry) │  │   Aggregation    │  │   Lookup       │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘   │
│           │                     │                      │            │
│           └─────────────────────┴──────────────────────┘            │
│                            │                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             Medical Code Description Service                  │  │
│  │  • ICD-10 → Human-readable diagnoses                         │  │
│  │  • CPT → Procedure descriptions                              │  │
│  │  • NDC → Medication details                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EMBEDDING LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Embedding Generation Service                     │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │  │
│  │  │  Claim Text    │  │  Medical Code  │  │  Provider     │  │  │
│  │  │  Embeddings    │  │  Embeddings    │  │  Embeddings   │  │  │
│  │  │  (1536-dim)    │  │  (768-dim)     │  │  (384-dim)    │  │  │
│  │  └────────┬───────┘  └────────┬───────┘  └───────┬───────┘  │  │
│  │           │                   │                   │           │  │
│  │           └───────────────────┴───────────────────┘           │  │
│  └───────────────────────────────┼───────────────────────────────┘  │
│                                  │                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Embedding Cache (Redis)                      │  │
│  │  • Provider embeddings (rarely change)                        │  │
│  │  • Common diagnosis code embeddings                           │  │
│  │  • TTL: 24 hours                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      Qdrant Cluster                           │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │  │
│  │  │ fraud_claims_kb │  │ provider_       │  │ medical_     │ │  │
│  │  │ (Claims)        │  │ patterns_kb     │  │ code_kb      │ │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Indexing Pipeline (Async)                        │  │
│  │  • Batch upserts (100 claims/batch)                           │  │
│  │  • Parallel processing                                        │  │
│  │  • Progress tracking                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FRAUD DETECTION LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              RAG Orchestration Service                        │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  1. Query Vector Store (similar fraud patterns)        │  │  │
│  │  │  2. Retrieve relevant context (top-k claims)           │  │  │
│  │  │  3. Construct LLM prompt with context                  │  │  │
│  │  │  4. Generate fraud analysis                            │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           Rule-Based Detection Engine (Parallel)              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │  │
│  │  │  Upcoding    │  │  Phantom     │  │  Provider       │    │  │
│  │  │  Detection   │  │  Billing     │  │  Network        │    │  │
│  │  │              │  │  Detection   │  │  Analysis       │    │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              ML Model Inference Service                       │  │
│  │  • Gradient Boosting Classifier                               │  │
│  │  • Neural Network (optional)                                  │  │
│  │  • Ensemble voting                                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RESULTS AGGREGATION                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Score Fusion & Ranking                           │  │
│  │  • RAG confidence score (0-1)                                 │  │
│  │  • Rule-based flags (binary)                                  │  │
│  │  • ML model probability (0-1)                                 │  │
│  │  • Weighted ensemble → Final fraud score                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE & DELIVERY                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐  │
│  │  PostgreSQL  │  │  S3 Archive   │  │  API Response          │  │
│  │  (Metadata)  │  │  (Raw Claims) │  │  (Real-time results)   │  │
│  └──────────────┘  └───────────────┘  └────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Audit Trail Service                          │  │
│  │  • All decisions logged                                       │  │
│  │  • Model versions tracked                                     │  │
│  │  • Human review outcomes                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEEDBACK LOOP                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           Human Review & Ground Truth Collection             │  │
│  │  • Analyst reviews flagged claims                            │  │
│  │  • Confirmed frauds → Training data                          │  │
│  │  • False positives → Model retraining                        │  │
│  └──────────────┬───────────────────────────────────────────────┘  │
│                 │                                                    │
│                 └──────► [Retraining Pipeline] ──► [Model Update]   │
│                 │                                                    │
│                 └──────► [KB Update] ──► [Vector Store Refresh]     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Components

### 2.1 Ingestion Layer

**Purpose**: Accept claims from multiple sources with different formats and protocols.

**Components:**

```python
# API Gateway (FastAPI)
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

@app.post("/api/v1/claims/submit")
async def submit_claim(
    claim: ClaimSubmission,
    background_tasks: BackgroundTasks
) -> dict:
    """
    Accept claim submission via REST API.
    Returns immediate acknowledgment and processes asynchronously.
    """
    # Generate claim ID
    claim_id = generate_claim_id()

    # Queue for processing
    await queue_client.send_message(
        queue_name="claims_ingestion",
        message={
            "claim_id": claim_id,
            "claim_data": claim.dict(),
            "source": "api",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Schedule async processing
    background_tasks.add_task(process_claim, claim_id)

    return {
        "claim_id": claim_id,
        "status": "accepted",
        "message": "Claim queued for processing"
    }


# S3 Event-Driven Ingestion
@app.post("/webhook/s3-claim-upload")
async def s3_claim_uploaded(event: S3Event) -> dict:
    """
    Triggered when claims uploaded to S3 bucket.
    """
    bucket = event.bucket_name
    key = event.object_key

    # Download and parse claim
    claim_data = await s3_client.get_object(bucket, key)

    # Route to processing queue
    await queue_client.send_message(
        queue_name="claims_ingestion",
        message={
            "claim_id": extract_claim_id(key),
            "s3_location": f"s3://{bucket}/{key}",
            "source": "s3",
            "timestamp": event.timestamp
        }
    )

    return {"status": "processed"}
```

**Message Queue (SQS/Kafka):**

```python
# Kafka Consumer for High Throughput
from aiokafka import AIOKafkaConsumer

async def consume_claims():
    """
    Consume claims from Kafka topic.
    """
    consumer = AIOKafkaConsumer(
        'claims.submitted',
        bootstrap_servers=['kafka:9092'],
        group_id='fraud-detection-processors',
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )

    await consumer.start()

    try:
        async for msg in consumer:
            claim_data = json.loads(msg.value.decode('utf-8'))

            # Process claim through validation pipeline
            await validation_pipeline.process(claim_data)

    finally:
        await consumer.stop()
```

### 2.2 Validation Layer

**Purpose**: Ensure data quality before processing using multi-stage validation.

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
from datetime import datetime

class ClaimValidation(BaseModel):
    """
    Pydantic model for claim validation.
    """
    claim_id: str = Field(..., regex=r'^CLM\d{10}$')
    patient_id: str = Field(..., regex=r'^PAT\d{8}$')
    provider_npi: str = Field(..., regex=r'^\d{10}$')
    diagnosis_codes: List[str] = Field(..., min_items=1, max_items=20)
    procedure_codes: List[str] = Field(..., min_items=1, max_items=50)
    claim_date: datetime
    service_date: datetime
    billed_amount: float = Field(..., gt=0, le=1000000)

    @validator('diagnosis_codes')
    def validate_icd10_codes(cls, codes):
        """Validate ICD-10 format."""
        icd10_pattern = re.compile(r'^[A-Z]\d{2}(\.\d{1,4})?$')
        for code in codes:
            if not icd10_pattern.match(code):
                raise ValueError(f"Invalid ICD-10 code: {code}")
        return codes

    @validator('procedure_codes')
    def validate_cpt_codes(cls, codes):
        """Validate CPT format."""
        cpt_pattern = re.compile(r'^\d{5}$')
        for code in codes:
            if not cpt_pattern.match(code):
                raise ValueError(f"Invalid CPT code: {code}")
        return codes

    @validator('service_date')
    def validate_service_date(cls, v, values):
        """Ensure service date is before claim date."""
        if 'claim_date' in values and v > values['claim_date']:
            raise ValueError("Service date cannot be after claim date")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Medical Code Validation Service
class MedicalCodeValidator:
    """
    Validates medical codes against official datasets.
    """
    def __init__(self):
        self.icd10_codes = self.load_icd10_codes()
        self.cpt_codes = self.load_cpt_codes()
        self.ndc_codes = self.load_ndc_codes()

    async def validate_claim(self, claim: dict) -> ValidationResult:
        """
        Comprehensive medical code validation.
        """
        errors = []
        warnings = []

        # Validate ICD-10 codes exist
        for code in claim['diagnosis_codes']:
            if code not in self.icd10_codes:
                errors.append(f"Unknown ICD-10 code: {code}")

        # Validate CPT codes exist
        for code in claim['procedure_codes']:
            if code not in self.cpt_codes:
                errors.append(f"Unknown CPT code: {code}")

        # Check for invalid combinations
        invalid_combos = self.check_code_combinations(
            claim['diagnosis_codes'],
            claim['procedure_codes']
        )
        if invalid_combos:
            warnings.extend(invalid_combos)

        # Check for gender-specific procedures
        gender_issues = self.validate_gender_specific_codes(
            claim['patient_gender'],
            claim['procedure_codes']
        )
        if gender_issues:
            errors.extend(gender_issues)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# Business Rules Validation
class BusinessRulesValidator:
    """
    Validates business logic rules.
    """
    async def validate_claim(self, claim: dict) -> ValidationResult:
        """
        Apply business rules validation.
        """
        rules_failed = []

        # Rule 1: Claim amount within reasonable range
        if claim['billed_amount'] > 50000:
            rules_failed.append({
                "rule": "HIGH_CLAIM_AMOUNT",
                "severity": "warning",
                "message": f"Claim amount ${claim['billed_amount']} exceeds normal threshold"
            })

        # Rule 2: Provider billing velocity
        provider_daily_claims = await self.get_provider_daily_claims(
            claim['provider_npi'],
            claim['claim_date']
        )
        if provider_daily_claims > 100:
            rules_failed.append({
                "rule": "HIGH_PROVIDER_VELOCITY",
                "severity": "warning",
                "message": f"Provider has {provider_daily_claims} claims today"
            })

        # Rule 3: Geographic impossibility
        if await self.check_geographic_impossibility(claim):
            rules_failed.append({
                "rule": "GEOGRAPHIC_IMPOSSIBILITY",
                "severity": "error",
                "message": "Patient location incompatible with service location"
            })

        return ValidationResult(
            valid=len([r for r in rules_failed if r['severity'] == 'error']) == 0,
            rules_failed=rules_failed
        )
```

### 2.3 Enrichment Layer

**Purpose**: Augment claims with contextual information from multiple data sources.

```python
class ClaimEnrichmentService:
    """
    Enriches claims with additional context.
    """
    def __init__(self):
        self.provider_registry = ProviderRegistry()
        self.patient_history_service = PatientHistoryService()
        self.fraud_pattern_db = FraudPatternDatabase()

    async def enrich_claim(self, claim: dict) -> EnrichedClaim:
        """
        Parallel enrichment from multiple sources.
        """
        # Fetch data in parallel
        provider_info, patient_history, historical_patterns = await asyncio.gather(
            self.get_provider_info(claim['provider_npi']),
            self.get_patient_history(claim['patient_id']),
            self.get_historical_patterns(claim)
        )

        # Enrich with medical code descriptions
        diagnosis_descriptions = await self.get_diagnosis_descriptions(
            claim['diagnosis_codes']
        )
        procedure_descriptions = await self.get_procedure_descriptions(
            claim['procedure_codes']
        )

        return EnrichedClaim(
            **claim,
            provider_info={
                "name": provider_info.name,
                "specialty": provider_info.specialty,
                "location": provider_info.location,
                "license_status": provider_info.license_status,
                "historical_fraud_flags": provider_info.fraud_history
            },
            patient_history={
                "previous_claims": patient_history.claim_count,
                "chronic_conditions": patient_history.chronic_conditions,
                "recent_procedures": patient_history.recent_procedures,
                "provider_relationships": patient_history.provider_network
            },
            diagnosis_descriptions=diagnosis_descriptions,
            procedure_descriptions=procedure_descriptions,
            similar_fraud_patterns=historical_patterns
        )

    async def get_provider_info(self, npi: str) -> ProviderInfo:
        """
        Lookup provider from NPI registry with caching.
        """
        # Check cache first
        cached = await redis_client.get(f"provider:{npi}")
        if cached:
            return ProviderInfo.parse_raw(cached)

        # Fetch from database
        provider = await db.fetch_one(
            "SELECT * FROM providers WHERE npi = $1",
            npi
        )

        # Cache for 24 hours
        await redis_client.setex(
            f"provider:{npi}",
            86400,
            provider.json()
        )

        return provider

    async def get_patient_history(self, patient_id: str) -> PatientHistory:
        """
        Aggregate patient history from claims database.
        """
        # Last 12 months of claims
        recent_claims = await db.fetch_all(
            """
            SELECT * FROM claims
            WHERE patient_id = $1
              AND claim_date >= NOW() - INTERVAL '12 months'
            ORDER BY claim_date DESC
            """,
            patient_id
        )

        # Aggregate patterns
        provider_network = set()
        procedure_frequency = {}

        for claim in recent_claims:
            provider_network.add(claim['provider_npi'])
            for proc in claim['procedure_codes']:
                procedure_frequency[proc] = procedure_frequency.get(proc, 0) + 1

        return PatientHistory(
            patient_id=patient_id,
            claim_count=len(recent_claims),
            provider_network=list(provider_network),
            procedure_frequency=procedure_frequency,
            chronic_conditions=self.extract_chronic_conditions(recent_claims)
        )
```

### 2.4 Embedding Layer

**Purpose**: Convert enriched claims into vector representations for similarity search.

```python
class EmbeddingService:
    """
    Generates embeddings for claims and medical codes.
    """
    def __init__(self):
        self.text_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.medical_code_model = BioMedicalEncoder()
        self.cache = RedisCache()

    async def generate_claim_embeddings(
        self,
        claim: EnrichedClaim
    ) -> ClaimEmbeddings:
        """
        Generate multi-modal embeddings for claim.
        """
        # Construct text representation
        claim_text = self.construct_claim_text(claim)

        # Check cache for text embedding
        cache_key = f"emb:text:{hash(claim_text)}"
        text_embedding = await self.cache.get(cache_key)

        if text_embedding is None:
            # Generate new embedding
            text_embedding = await self.text_model.aembed_query(claim_text)
            await self.cache.set(cache_key, text_embedding, ttl=86400)

        # Generate medical code embedding
        code_embedding = await self.medical_code_model.encode({
            "diagnosis": claim.diagnosis_codes,
            "procedure": claim.procedure_codes
        })

        # Generate provider embedding (cached by NPI)
        provider_embedding = await self.get_provider_embedding(
            claim.provider_npi
        )

        return ClaimEmbeddings(
            text=text_embedding,
            medical_codes=code_embedding,
            provider=provider_embedding
        )

    def construct_claim_text(self, claim: EnrichedClaim) -> str:
        """
        Construct comprehensive text representation.
        """
        parts = [
            f"Provider: {claim.provider_info['specialty']} in {claim.provider_info['location']}",
            f"Diagnoses: {', '.join(claim.diagnosis_descriptions)}",
            f"Procedures: {', '.join(claim.procedure_descriptions)}",
            f"Patient History: {claim.patient_history['previous_claims']} prior claims",
            f"Billed Amount: ${claim.billed_amount:,.2f}"
        ]

        if claim.patient_history.get('chronic_conditions'):
            parts.append(
                f"Chronic Conditions: {', '.join(claim.patient_history['chronic_conditions'])}"
            )

        return " | ".join(parts)

    async def batch_generate_embeddings(
        self,
        claims: List[EnrichedClaim],
        batch_size: int = 50
    ) -> List[ClaimEmbeddings]:
        """
        Generate embeddings in batches for efficiency.
        """
        embeddings = []

        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]

            # Parallel embedding generation
            batch_embeddings = await asyncio.gather(*[
                self.generate_claim_embeddings(claim)
                for claim in batch
            ])

            embeddings.extend(batch_embeddings)

            # Rate limiting
            await asyncio.sleep(0.1)

        return embeddings
```

### 2.5 Vector Store Indexing Pipeline

**Purpose**: Efficiently index embeddings into Qdrant for retrieval.

```python
class VectorIndexingPipeline:
    """
    Manages indexing of claim embeddings into Qdrant.
    """
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.progress_tracker = ProgressTracker()

    async def index_claims(
        self,
        claims: List[EnrichedClaim],
        embeddings: List[ClaimEmbeddings],
        batch_size: int = 100
    ) -> IndexingResult:
        """
        Index claims with embeddings into vector store.
        """
        total = len(claims)
        indexed = 0
        failed = []

        for i in range(0, total, batch_size):
            batch_claims = claims[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # Prepare points for Qdrant
            points = [
                PointStruct(
                    id=claim.claim_id,
                    vector={
                        "claim_text": emb.text,
                        "medical_codes": emb.medical_codes,
                        "provider": emb.provider
                    },
                    payload={
                        "claim_id": claim.claim_id,
                        "patient_id": claim.patient_id,
                        "provider_npi": claim.provider_npi,
                        "diagnosis_codes": claim.diagnosis_codes,
                        "procedure_codes": claim.procedure_codes,
                        "fraud_indicator": claim.fraud_indicator,
                        "fraud_type": claim.fraud_type,
                        "claim_date": claim.claim_date.isoformat(),
                        "billed_amount": float(claim.billed_amount),
                        "red_flags": claim.red_flags
                    }
                )
                for claim, emb in zip(batch_claims, batch_embeddings)
            ]

            try:
                # Batch upsert
                await self.qdrant_client.upsert(
                    collection_name="fraud_claims_kb",
                    points=points,
                    wait=False  # Async indexing
                )

                indexed += len(points)

                # Update progress
                await self.progress_tracker.update(indexed, total)

            except Exception as e:
                logger.error(f"Failed to index batch: {e}")
                failed.extend([c.claim_id for c in batch_claims])

        return IndexingResult(
            total=total,
            indexed=indexed,
            failed=failed,
            success_rate=indexed / total
        )
```

### 2.6 Fraud Detection Layer

**Purpose**: Multi-model fraud detection combining RAG, rules, and ML.

```python
class FraudDetectionOrchestrator:
    """
    Orchestrates multiple fraud detection methods.
    """
    def __init__(self):
        self.rag_service = RAGFraudDetectionService()
        self.rule_engine = RuleBasedDetectionEngine()
        self.ml_model = MLModelInferenceService()

    async def detect_fraud(
        self,
        claim: EnrichedClaim
    ) -> FraudDetectionResult:
        """
        Run all detection methods in parallel and aggregate results.
        """
        # Parallel detection
        rag_result, rule_result, ml_result = await asyncio.gather(
            self.rag_service.analyze(claim),
            self.rule_engine.evaluate(claim),
            self.ml_model.predict(claim)
        )

        # Aggregate scores with weights
        final_score = (
            rag_result.confidence * 0.4 +
            rule_result.score * 0.3 +
            ml_result.probability * 0.3
        )

        # Combine evidence
        all_flags = (
            rag_result.red_flags +
            rule_result.rules_triggered +
            ml_result.feature_importance
        )

        return FraudDetectionResult(
            claim_id=claim.claim_id,
            fraud_score=final_score,
            fraud_probability=final_score,
            is_fraud=final_score > 0.7,
            confidence=self.calculate_confidence([rag_result, rule_result, ml_result]),
            evidence={
                "rag_analysis": rag_result.explanation,
                "rules_triggered": rule_result.rules_triggered,
                "ml_prediction": ml_result.prediction_class,
                "similar_fraud_cases": rag_result.similar_cases
            },
            red_flags=all_flags,
            recommended_action=self.determine_action(final_score)
        )


class RAGFraudDetectionService:
    """
    RAG-based fraud detection using vector similarity.
    """
    def __init__(self):
        self.qdrant_client = QdrantClient()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    async def analyze(self, claim: EnrichedClaim) -> RAGResult:
        """
        Analyze claim using RAG approach.
        """
        # Generate query embedding
        query_embedding = await embedding_service.generate_claim_embeddings(claim)

        # Search for similar fraud cases
        similar_frauds = await self.qdrant_client.search(
            collection_name="fraud_claims_kb",
            query_vector=query_embedding.text,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="fraud_indicator",
                        match=MatchValue(value=True)
                    )
                ]
            ),
            limit=10,
            score_threshold=0.7
        )

        # Search for similar legitimate cases
        similar_legitimate = await self.qdrant_client.search(
            collection_name="fraud_claims_kb",
            query_vector=query_embedding.text,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="fraud_indicator",
                        match=MatchValue(value=False)
                    )
                ]
            ),
            limit=5,
            score_threshold=0.7
        )

        # Construct LLM prompt
        prompt = self.construct_fraud_analysis_prompt(
            claim,
            similar_frauds,
            similar_legitimate
        )

        # Generate analysis
        response = await self.llm.ainvoke(prompt)

        return RAGResult(
            confidence=self.calculate_rag_confidence(similar_frauds, similar_legitimate),
            explanation=response.content,
            similar_cases=[s.id for s in similar_frauds],
            red_flags=self.extract_red_flags(response.content)
        )

    def construct_fraud_analysis_prompt(
        self,
        claim: EnrichedClaim,
        similar_frauds: List[ScoredPoint],
        similar_legitimate: List[ScoredPoint]
    ) -> str:
        """
        Construct comprehensive prompt for LLM analysis.
        """
        fraud_examples = "\n".join([
            f"- {f.payload['fraud_type']}: {f.payload['diagnosis_codes']} "
            f"(Score: {f.score:.2f})"
            for f in similar_frauds
        ])

        legit_examples = "\n".join([
            f"- Legitimate: {l.payload['diagnosis_codes']} (Score: {l.score:.2f})"
            for l in similar_legitimate
        ])

        return f"""
Analyze this insurance claim for potential fraud:

**Current Claim:**
- Provider: {claim.provider_info['specialty']} ({claim.provider_npi})
- Diagnoses: {', '.join(claim.diagnosis_descriptions)}
- Procedures: {', '.join(claim.procedure_descriptions)}
- Billed Amount: ${claim.billed_amount:,.2f}
- Patient History: {claim.patient_history['previous_claims']} prior claims

**Similar Confirmed Fraud Cases:**
{fraud_examples}

**Similar Legitimate Cases:**
{legit_examples}

**Provider History:**
- Historical fraud flags: {claim.provider_info['historical_fraud_flags']}

Based on this context, analyze:
1. Likelihood this claim is fraudulent (0-100%)
2. Primary fraud indicators (if any)
3. Comparison to similar cases
4. Recommended action

Provide a detailed analysis.
"""
```

### 2.7 Results Storage and Delivery

```python
class ResultsStorageService:
    """
    Persists fraud detection results and delivers to consumers.
    """
    def __init__(self):
        self.db = AsyncDatabase()
        self.s3_client = S3Client()
        self.audit_service = AuditTrailService()

    async def store_result(
        self,
        result: FraudDetectionResult
    ) -> None:
        """
        Store detection result with full audit trail.
        """
        # Store in PostgreSQL
        await self.db.execute(
            """
            INSERT INTO fraud_detection_results (
                claim_id, fraud_score, fraud_probability,
                is_fraud, confidence, evidence, red_flags,
                recommended_action, model_version, detected_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
            result.claim_id,
            result.fraud_score,
            result.fraud_probability,
            result.is_fraud,
            result.confidence,
            json.dumps(result.evidence),
            result.red_flags,
            result.recommended_action,
            settings.MODEL_VERSION
        )

        # Archive full claim to S3
        await self.s3_client.put_object(
            bucket=settings.ARCHIVE_BUCKET,
            key=f"claims/{result.claim_id}/detection_result.json",
            body=result.json()
        )

        # Log to audit trail
        await self.audit_service.log_detection(
            claim_id=result.claim_id,
            result=result,
            user="system",
            timestamp=datetime.utcnow()
        )

    async def deliver_result(
        self,
        result: FraudDetectionResult
    ) -> DeliveryStatus:
        """
        Deliver results to downstream systems.
        """
        delivery_tasks = []

        # Send high-priority alerts for confirmed fraud
        if result.is_fraud and result.confidence > 0.8:
            delivery_tasks.append(
                self.send_fraud_alert(result)
            )

        # Update case management system
        delivery_tasks.append(
            self.update_case_management(result)
        )

        # Publish to event stream
        delivery_tasks.append(
            self.publish_to_kafka(result)
        )

        # Execute deliveries in parallel
        await asyncio.gather(*delivery_tasks)

        return DeliveryStatus(
            claim_id=result.claim_id,
            delivered_at=datetime.utcnow(),
            channels=["alert", "case_management", "kafka"]
        )
```

## 3. Async Processing Pipeline

### 3.1 Worker Architecture

```python
# Celery worker for distributed processing
from celery import Celery

celery_app = Celery(
    'fraud_detection',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery_app.task(bind=True, max_retries=3)
def process_claim_task(self, claim_data: dict) -> dict:
    """
    Celery task for claim processing.
    """
    try:
        # Run full pipeline
        result = asyncio.run(full_pipeline(claim_data))
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Claim processing failed: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


async def full_pipeline(claim_data: dict) -> FraudDetectionResult:
    """
    Execute full claim processing pipeline.
    """
    # 1. Validation
    validated_claim = await validation_service.validate(claim_data)

    # 2. Enrichment
    enriched_claim = await enrichment_service.enrich(validated_claim)

    # 3. Embedding
    embeddings = await embedding_service.generate(enriched_claim)

    # 4. Indexing
    await indexing_service.index(enriched_claim, embeddings)

    # 5. Fraud Detection
    result = await fraud_detection.detect(enriched_claim)

    # 6. Storage
    await storage_service.store(result)

    # 7. Delivery
    await delivery_service.deliver(result)

    return result
```

### 3.2 Performance Optimization

```python
# Batch processing for efficiency
async def process_claim_batch(claims: List[dict]) -> List[FraudDetectionResult]:
    """
    Process multiple claims efficiently.
    """
    # Parallel validation
    validated = await asyncio.gather(*[
        validation_service.validate(claim)
        for claim in claims
    ])

    # Parallel enrichment
    enriched = await asyncio.gather(*[
        enrichment_service.enrich(claim)
        for claim in validated
    ])

    # Batch embedding generation
    embeddings = await embedding_service.batch_generate(enriched)

    # Batch indexing
    await indexing_service.batch_index(enriched, embeddings)

    # Parallel fraud detection
    results = await asyncio.gather(*[
        fraud_detection.detect(claim)
        for claim in enriched
    ])

    # Batch storage
    await storage_service.batch_store(results)

    return results
```

## 4. Cache Layer Design

```python
# Multi-tier caching strategy
class CacheLayerService:
    """
    Implements multi-tier caching for performance.
    """
    def __init__(self):
        self.l1_cache = {}  # In-memory (process-level)
        self.l2_cache = RedisClient()  # Distributed cache
        self.l3_cache = None  # Vector store cache (handled by Qdrant)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get from cache with fallthrough.
        """
        # L1: In-memory
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: Redis
        value = await self.l2_cache.get(key)
        if value:
            # Populate L1
            self.l1_cache[key] = value
            return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> None:
        """
        Set in all cache tiers.
        """
        # L1: In-memory
        self.l1_cache[key] = value

        # L2: Redis with TTL
        await self.l2_cache.setex(key, ttl, value)
```

## 5. Monitoring and Observability

### 5.1 Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
claims_processed = Counter(
    'claims_processed_total',
    'Total claims processed',
    ['status', 'source']
)

processing_latency = Histogram(
    'claim_processing_seconds',
    'Time to process claim',
    ['stage']
)

fraud_detected = Counter(
    'fraud_detected_total',
    'Total fraud cases detected',
    ['fraud_type', 'confidence_tier']
)

queue_depth = Gauge(
    'claims_queue_depth',
    'Number of claims in processing queue'
)

# Usage in pipeline
async def process_with_metrics(claim: dict) -> FraudDetectionResult:
    """
    Process claim with metric collection.
    """
    with processing_latency.labels(stage='total').time():
        try:
            # Validation
            with processing_latency.labels(stage='validation').time():
                validated = await validation_service.validate(claim)

            # Enrichment
            with processing_latency.labels(stage='enrichment').time():
                enriched = await enrichment_service.enrich(validated)

            # Detection
            with processing_latency.labels(stage='detection').time():
                result = await fraud_detection.detect(enriched)

            # Record result
            claims_processed.labels(
                status='success',
                source=claim.get('source', 'unknown')
            ).inc()

            if result.is_fraud:
                fraud_detected.labels(
                    fraud_type=result.evidence.get('fraud_type', 'unknown'),
                    confidence_tier=self.get_confidence_tier(result.confidence)
                ).inc()

            return result

        except Exception as e:
            claims_processed.labels(
                status='error',
                source=claim.get('source', 'unknown')
            ).inc()
            raise
```

### 5.2 Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/api/v1/claims/submit")
async def submit_claim_with_tracing(claim: ClaimSubmission):
    """
    API endpoint with distributed tracing.
    """
    with tracer.start_as_current_span("submit_claim") as span:
        span.set_attribute("claim.id", claim.claim_id)
        span.set_attribute("claim.amount", claim.billed_amount)

        # Validation span
        with tracer.start_as_current_span("validate_claim"):
            validated = await validation_service.validate(claim)

        # Enrichment span
        with tracer.start_as_current_span("enrich_claim"):
            enriched = await enrichment_service.enrich(validated)

        # Detection span
        with tracer.start_as_current_span("detect_fraud"):
            result = await fraud_detection.detect(enriched)

        span.set_attribute("fraud.detected", result.is_fraud)
        span.set_attribute("fraud.score", result.fraud_score)

        return result
```

## 6. Error Handling and Recovery

```python
class ErrorRecoveryService:
    """
    Handles errors and recovery strategies.
    """
    def __init__(self):
        self.dlq = DeadLetterQueue()
        self.retry_policy = RetryPolicy()

    async def handle_validation_error(
        self,
        claim: dict,
        error: ValidationError
    ) -> None:
        """
        Handle validation failures.
        """
        # Log error
        logger.error(f"Validation failed for claim {claim.get('claim_id')}: {error}")

        # Send to DLQ for manual review
        await self.dlq.send_message(
            queue_name="validation_failures",
            message={
                "claim": claim,
                "error": str(error),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Alert data quality team
        await self.send_alert(
            severity="warning",
            message=f"Claim validation failed: {error}"
        )

    async def handle_processing_error(
        self,
        claim: dict,
        error: Exception,
        retry_count: int
    ) -> None:
        """
        Handle processing failures with retry logic.
        """
        if retry_count < 3:
            # Retry with exponential backoff
            await asyncio.sleep(2 ** retry_count)
            return await process_claim_task.retry(
                args=[claim],
                countdown=60 * (2 ** retry_count)
            )
        else:
            # Max retries exceeded, send to DLQ
            await self.dlq.send_message(
                queue_name="processing_failures",
                message={
                    "claim": claim,
                    "error": str(error),
                    "retry_count": retry_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
```

## 7. Performance Benchmarks

### 7.1 Target Latencies

```
Ingestion: <10ms (API acknowledgment)
Validation: <50ms per claim
Enrichment: <200ms per claim
Embedding: <100ms per claim (cached)
Vector Search: <80ms p95
Fraud Detection: <500ms total per claim
Storage: <50ms per claim

End-to-End: <1s p95 for real-time processing
Batch Processing: >1000 claims/minute
```

### 7.2 Throughput Targets

```
Development: 10 claims/second
Staging: 50 claims/second
Production: 200 claims/second (peak: 500/sec)
```

## Related Documents

- [VECTOR_STORE_DESIGN.md](./VECTOR_STORE_DESIGN.md) - Vector database architecture
- [DATA_VALIDATION_PIPELINE.md](./DATA_VALIDATION_PIPELINE.md) - Validation strategy details
- [INFRASTRUCTURE_REQUIREMENTS.md](./INFRASTRUCTURE_REQUIREMENTS.md) - Resource requirements
