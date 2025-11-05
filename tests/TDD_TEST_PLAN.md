# TDD Test Plan: Test-Driven Development Strategy

## Executive Summary

This document outlines the Test-Driven Development (TDD) approach for implementing the insurance fraud detection system with RAG enrichment. Following TDD principles ensures code quality, maintainability, and alignment with requirements from the outset.

## 1. TDD Principles

### 1.1 Red-Green-Refactor Cycle

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  1. RED: Write Failing Test                    │
│     - Define expected behavior                  │
│     - Write test with assertions                │
│     - Run test (should fail)                    │
│                                                 │
│  2. GREEN: Write Minimum Code                  │
│     - Implement just enough to pass             │
│     - Run test (should pass)                    │
│     - No refactoring yet                        │
│                                                 │
│  3. REFACTOR: Improve Code Quality             │
│     - Optimize implementation                   │
│     - Improve readability                       │
│     - Ensure tests still pass                   │
│                                                 │
│  4. REPEAT: Next feature/requirement            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 1.2 TDD Rules

1. **Write NO production code** without a failing test
2. **Write only enough test** to demonstrate a failure
3. **Write only enough production code** to pass the test
4. **Tests must be fast** (<1 second for unit tests)
5. **Tests must be independent** (no shared state)
6. **Tests must be deterministic** (same input → same output)

---

## 2. Implementation Order (TDD Sequence)

### 2.1 Phase 1: Core Data Models (Week 1)

**Goal**: Implement foundational data structures

#### Step 1.1: Claim Validation Models

**RED - Write Failing Test**:
```python
def test_valid_claim_creation():
    """Should create a valid medical claim."""
    claim_data = {
        "claim_id": "CLM-TEST-001",
        "patient_id": "PAT-10001",
        "provider_npi": "1234567890",
        "diagnosis_codes": ["E11.9"],
        "procedure_codes": ["99213"],
        "billed_amount": 125.00,
        "date_of_service": "2024-03-15"
    }
    claim = MedicalClaim(**claim_data)
    assert claim.claim_id == "CLM-TEST-001"
    assert claim.billed_amount == Decimal("125.00")
```

**GREEN - Implement Minimum Code**:
```python
class MedicalClaim(BaseModel):
    claim_id: str
    patient_id: str
    provider_npi: str
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    billed_amount: Decimal
    date_of_service: date
```

**REFACTOR**: Add validators, proper types, documentation

---

#### Step 1.2: Invalid Data Handling

**RED - Write Failing Test**:
```python
def test_invalid_npi_rejected():
    """Should reject invalid NPI format."""
    claim_data = {
        "claim_id": "CLM-TEST-001",
        "provider_npi": "123",  # Invalid: should be 10 digits
        # ... other fields
    }
    with pytest.raises(ValidationError) as exc_info:
        MedicalClaim(**claim_data)
    assert "provider_npi" in str(exc_info.value)
```

**GREEN - Add Validation**:
```python
class MedicalClaim(BaseModel):
    provider_npi: str = Field(..., regex=r"^\d{10}$")
```

**REFACTOR**: Add comprehensive validators for all fields

---

### 2.2 Phase 2: RAG Enrichment System (Week 2-3)

**Goal**: Implement knowledge-based claim enrichment

#### Step 2.1: Knowledge Base Retrieval

**RED - Write Failing Test**:
```python
def test_retrieve_diagnosis_from_kb():
    """Should retrieve diagnosis information from KB."""
    kb = KnowledgeBase()
    diagnosis = kb.get_diagnosis_info("E11.9")

    assert diagnosis is not None
    assert diagnosis["code"] == "E11.9"
    assert "diabetes" in diagnosis["description"].lower()
    assert len(diagnosis["common_procedures"]) > 0
```

**GREEN - Implement KB**:
```python
class KnowledgeBase:
    def __init__(self):
        self.diagnoses = self._load_diagnosis_kb()

    def get_diagnosis_info(self, code: str) -> Optional[Dict]:
        return self.diagnoses.get(code)
```

**REFACTOR**: Add caching, error handling, performance optimization

---

#### Step 2.2: Missing Data Detection

**RED - Write Failing Test**:
```python
def test_detect_missing_diagnosis():
    """Should detect when diagnosis codes are missing."""
    incomplete_claim = {
        "claim_id": "CLM-TEST-001",
        "procedure_codes": ["99213"],
        # Missing diagnosis_codes
    }
    detector = MissingDataDetector()
    missing_fields = detector.detect(incomplete_claim)

    assert "diagnosis_codes" in missing_fields
    assert len(missing_fields) == 1
```

**GREEN - Implement Detector**:
```python
class MissingDataDetector:
    def detect(self, claim: dict) -> List[str]:
        missing = []
        if "diagnosis_codes" not in claim or not claim["diagnosis_codes"]:
            missing.append("diagnosis_codes")
        return missing
```

**REFACTOR**: Check all required fields, add metadata

---

#### Step 2.3: Enrichment Logic

**RED - Write Failing Test**:
```python
def test_enrich_missing_diagnosis_from_procedure():
    """Should infer diagnosis from procedure code."""
    incomplete_claim = {
        "claim_id": "CLM-TEST-001",
        "procedure_codes": ["99213"],
        "billed_amount": 125.00
    }
    enricher = ClaimEnricher(kb=knowledge_base)
    enriched_claim = enricher.enrich(incomplete_claim)

    assert "diagnosis_codes" in enriched_claim
    assert len(enriched_claim["diagnosis_codes"]) > 0
    assert "enrichment_metadata" in enriched_claim
    assert enriched_claim["enrichment_metadata"]["diagnosis_confidence"] > 0.7
```

**GREEN - Implement Enricher**:
```python
class ClaimEnricher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def enrich(self, claim: dict) -> dict:
        enriched = claim.copy()
        if "diagnosis_codes" not in claim:
            # Infer from procedure
            procedure = claim["procedure_codes"][0]
            diagnosis = self.kb.get_common_diagnosis_for_procedure(procedure)
            enriched["diagnosis_codes"] = [diagnosis]
            enriched["enrichment_metadata"] = {"diagnosis_confidence": 0.85}
        return enriched
```

**REFACTOR**: Handle multiple procedures, confidence scoring, error cases

---

### 2.3 Phase 3: Rule-Based Fraud Detection (Week 3-4)

**Goal**: Implement fraud detection rules

#### Step 3.1: Upcoding Detection

**RED - Write Failing Test**:
```python
def test_detect_upcoding_simple_diagnosis_complex_procedure():
    """Should detect upcoding: common cold as complex visit."""
    claim = {
        "claim_id": "CLM-TEST-001",
        "diagnosis_codes": ["J00"],  # Common cold
        "procedure_codes": ["99215"],  # High complexity
        "billed_amount": 325.00
    }
    rule_engine = RuleEngine()
    results, fraud_score = rule_engine.analyze_claim(claim)

    assert fraud_score > 0.7
    assert any(r.rule_name == "upcoding_complexity" and r.triggered for r in results)
```

**GREEN - Implement Rule**:
```python
class RuleEngine:
    def analyze_claim(self, claim: dict):
        results = []
        score = 0.0

        # Check upcoding
        if self._is_simple_diagnosis(claim["diagnosis_codes"]):
            if self._is_complex_procedure(claim["procedure_codes"]):
                results.append(RuleResult(
                    rule_name="upcoding_complexity",
                    triggered=True,
                    score=0.8,
                    details="Simple diagnosis with complex procedure"
                ))
                score = 0.8

        return results, score
```

**REFACTOR**: Add multiple rules, proper scoring, evidence collection

---

#### Step 3.2: Phantom Billing Detection

**RED - Write Failing Test**:
```python
def test_detect_phantom_billing_outside_hours():
    """Should detect service billed outside operating hours."""
    claim = {
        "claim_id": "CLM-TEST-001",
        "date_of_service": "2024-03-15",
        "time_of_service": "02:00:00",  # 2 AM
        "service_location": "11",  # Office
        "procedure_codes": ["99213"]
    }
    rule_engine = RuleEngine()
    results, fraud_score = rule_engine.analyze_claim(claim)

    assert fraud_score > 0.8
    assert any(r.rule_name == "phantom_billing_schedule" and r.triggered for r in results)
```

**GREEN - Implement Rule**:
```python
def _check_phantom_billing_schedule(self, rule, claim):
    score = 0.0
    evidence = []

    if "time_of_service" in claim:
        hour = int(claim["time_of_service"].split(":")[0])
        if hour < 6 or hour > 22:
            score = 0.9
            evidence.append(f"Service at {claim['time_of_service']} outside normal hours")

    return RuleResult(rule.name, score > 0.8, score, "Phantom billing check", evidence)
```

**REFACTOR**: Add weekend checks, holiday checks, provider availability

---

### 2.4 Phase 4: ML Models (Week 4-5)

**Goal**: Implement machine learning fraud detection

#### Step 4.1: Feature Engineering

**RED - Write Failing Test**:
```python
def test_extract_statistical_features():
    """Should extract statistical features from claim."""
    claim = {
        "billed_amount": 325.00,
        "procedure_codes": ["99215", "94060"],
        "diagnosis_codes": ["E11.9"]
    }
    feature_engineer = FeatureEngineer()
    features = feature_engineer.extract_features(claim)

    assert "billed_amount" in features
    assert "procedure_count" in features
    assert features["procedure_count"] == 2
    assert "diagnosis_count" in features
    assert features["diagnosis_count"] == 1
```

**GREEN - Implement Feature Extraction**:
```python
class FeatureEngineer:
    def extract_features(self, claim: dict) -> dict:
        return {
            "billed_amount": claim["billed_amount"],
            "procedure_count": len(claim.get("procedure_codes", [])),
            "diagnosis_count": len(claim.get("diagnosis_codes", []))
        }
```

**REFACTOR**: Add temporal, provider, patient features

---

#### Step 4.2: Model Training and Prediction

**RED - Write Failing Test**:
```python
def test_random_forest_predicts_fraud():
    """Should predict fraud using Random Forest."""
    model = RandomForestFraudModel()
    model.train(training_data, training_labels)

    fraud_claim = {
        "billed_amount": 325.00,
        "procedure_count": 1,
        "diagnosis_code_E11.9": 0,
        "procedure_code_99215": 1
    }
    prediction = model.predict(fraud_claim)

    assert prediction["fraud_probability"] > 0.7
    assert prediction["prediction"] == "fraud"
```

**GREEN - Implement Model**:
```python
class RandomForestFraudModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features: dict) -> dict:
        X = self._features_to_array(features)
        prob = self.model.predict_proba(X)[0][1]
        return {
            "fraud_probability": prob,
            "prediction": "fraud" if prob > 0.5 else "legitimate"
        }
```

**REFACTOR**: Add cross-validation, hyperparameter tuning, model persistence

---

### 2.5 Phase 5: Integration and End-to-End (Week 5-6)

**Goal**: Integrate all components

#### Step 5.1: Complete Fraud Detection Pipeline

**RED - Write Failing Test**:
```python
def test_end_to_end_fraud_detection():
    """Should detect fraud from raw claim to final score."""
    raw_claim = load_test_claim("upcoding_fraud.json")

    # Complete pipeline
    fraud_detector = FraudDetector(
        enricher=claim_enricher,
        rule_engine=rule_engine,
        ml_models=[rf_model, xgb_model]
    )
    result = fraud_detector.detect(raw_claim)

    assert result.fraud_detected is True
    assert result.fraud_score > 0.7
    assert "upcoding" in result.fraud_types
    assert len(result.explanation) > 0
```

**GREEN - Implement Pipeline**:
```python
class FraudDetector:
    def __init__(self, enricher, rule_engine, ml_models):
        self.enricher = enricher
        self.rule_engine = rule_engine
        self.ml_models = ml_models

    def detect(self, claim: dict) -> FraudResult:
        # 1. Enrich if needed
        if self._needs_enrichment(claim):
            claim = self.enricher.enrich(claim)

        # 2. Extract features
        features = self.feature_engineer.extract_features(claim)

        # 3. Run rules
        rule_results, rule_score = self.rule_engine.analyze_claim(claim)

        # 4. Run ML models
        ml_scores = [model.predict(features) for model in self.ml_models]

        # 5. Aggregate scores
        final_score = self._aggregate_scores(rule_score, ml_scores)

        return FraudResult(
            fraud_detected=final_score > 0.5,
            fraud_score=final_score,
            fraud_types=self._identify_fraud_types(rule_results),
            explanation=self._generate_explanation(rule_results, ml_scores)
        )
```

**REFACTOR**: Optimize pipeline, add error handling, logging

---

## 3. TDD Best Practices

### 3.1 Test Organization

```
tests/
├── unit/                   # Fast, isolated tests
│   ├── rag/
│   │   ├── test_knowledge_base.py
│   │   ├── test_embeddings.py
│   │   ├── test_retrieval.py
│   │   └── test_enrichment.py
│   ├── detection/
│   │   ├── test_rule_engine.py
│   │   ├── test_ml_models.py
│   │   └── test_feature_engineering.py
│   └── models/
│       └── test_claim_models.py
├── integration/            # Component interaction tests
│   ├── test_rag_pipeline.py
│   ├── test_fraud_detection_pipeline.py
│   └── test_enrichment_fraud_detection.py
└── e2e/                    # Full workflow tests
    └── test_fraud_detection_e2e.py
```

### 3.2 Test Naming Conventions

**Pattern**: `test_<component>_<action>_<expected_outcome>`

Examples:
```python
def test_claim_validator_with_invalid_npi_raises_error():
    pass

def test_enricher_with_missing_diagnosis_adds_diagnosis():
    pass

def test_rule_engine_with_upcoding_pattern_flags_fraud():
    pass
```

### 3.3 Assertion Best Practices

**Good Assertions**:
```python
# Specific and informative
assert claim.provider_npi == "1234567890", \
    f"Expected NPI '1234567890', got '{claim.provider_npi}'"

# Multiple focused assertions
assert result.fraud_detected is True
assert result.fraud_score > 0.7
assert "upcoding" in result.fraud_types
```

**Bad Assertions**:
```python
# Too vague
assert result

# Too broad
assert result.fraud_detected and result.fraud_score > 0.7 and "upcoding" in result.fraud_types
```

### 3.4 Test Fixtures

**Use pytest fixtures for reusable test data**:
```python
@pytest.fixture
def valid_claim():
    """Valid claim for testing."""
    return CompleteTestClaim(
        claim_id="CLM-TEST-001",
        patient_id="PAT-10001",
        provider_npi="1234567890",
        diagnosis_codes=["E11.9"],
        procedure_codes=["99213"],
        billed_amount=Decimal("125.00"),
        date_of_service=date.today()
    )

@pytest.fixture
def upcoding_claim():
    """Upcoding fraud claim for testing."""
    return TestClaimFactory.create_upcoding_claim()

@pytest.fixture
def knowledge_base():
    """Mock knowledge base for testing."""
    return MockKnowledgeBase()
```

---

## 4. TDD Workflow Examples

### 4.1 Example: Implementing Diagnosis-Procedure Validation

**Iteration 1: RED**
```python
def test_validate_compatible_diagnosis_procedure_pair():
    """Should validate diabetes diagnosis with routine office visit."""
    validator = DiagnosisProcedureValidator(kb=knowledge_base)
    is_valid = validator.is_compatible("E11.9", "99213")
    assert is_valid is True
```
Run: ❌ FAIL (function doesn't exist)

**Iteration 1: GREEN**
```python
class DiagnosisProcedureValidator:
    def __init__(self, kb):
        self.kb = kb

    def is_compatible(self, diagnosis: str, procedure: str) -> bool:
        return True  # Simplest implementation to pass
```
Run: ✅ PASS

**Iteration 2: RED**
```python
def test_validate_incompatible_diagnosis_procedure_pair():
    """Should reject colonoscopy for common cold."""
    validator = DiagnosisProcedureValidator(kb=knowledge_base)
    is_valid = validator.is_compatible("J00", "45378")  # Cold + colonoscopy
    assert is_valid is False
```
Run: ❌ FAIL (returns True for everything)

**Iteration 2: GREEN**
```python
def is_compatible(self, diagnosis: str, procedure: str) -> bool:
    diagnosis_info = self.kb.get_diagnosis_info(diagnosis)
    if diagnosis_info and procedure in diagnosis_info["common_procedures"]:
        return True
    return False
```
Run: ✅ PASS

**Iteration 3: REFACTOR**
```python
def is_compatible(self, diagnosis: str, procedure: str) -> bool:
    """Check if diagnosis-procedure pair is medically compatible."""
    diagnosis_info = self.kb.get_diagnosis_info(diagnosis)

    if not diagnosis_info:
        logger.warning(f"Unknown diagnosis code: {diagnosis}")
        return False

    common_procedures = diagnosis_info.get("common_procedures", [])
    return procedure in common_procedures
```
Run: ✅ PASS (all tests still pass)

---

## 5. TDD Metrics and Monitoring

### 5.1 Development Velocity

Track TDD progress:
```python
# Daily metrics
tests_written_today = 15
tests_passing_today = 15
code_coverage = 85%
avg_time_per_test = 45 seconds
```

### 5.2 TDD Quality Indicators

- **Test-to-Code Ratio**: 1.5:1 (1.5 lines of test per line of production code)
- **Test Execution Time**: <2 minutes for all unit tests
- **Test Pass Rate**: >99% (after initial RED phase)
- **Code Coverage**: >80% from TDD alone

---

## 6. Common TDD Pitfalls and Solutions

### 6.1 Pitfall: Writing Too Much Test Code

**Problem**: Complex test setup obscures test intent

**Solution**: Use factories and fixtures
```python
# Bad: Verbose test setup
def test_fraud_detection():
    claim = {
        "claim_id": "CLM-001",
        "patient_id": "PAT-001",
        # ... 20 more fields
    }

# Good: Use factory
def test_fraud_detection():
    claim = TestClaimFactory.create_upcoding_claim()
```

### 6.2 Pitfall: Testing Implementation Instead of Behavior

**Problem**: Tests break when refactoring

**Solution**: Test behavior, not implementation
```python
# Bad: Tests implementation
def test_rule_engine_calls_upcoding_check():
    rule_engine._check_upcoding_complexity.assert_called_once()

# Good: Tests behavior
def test_rule_engine_detects_upcoding():
    result = rule_engine.analyze_claim(upcoding_claim)
    assert result.fraud_detected is True
```

### 6.3 Pitfall: Slow Tests

**Problem**: Test suite takes too long

**Solution**: Mock external dependencies
```python
# Bad: Real database access
def test_load_claim_from_db():
    claim = database.load_claim("CLM-001")

# Good: Mock database
@patch('database.load_claim')
def test_load_claim_from_db(mock_load):
    mock_load.return_value = test_claim
    claim = database.load_claim("CLM-001")
```

---

## 7. TDD Success Criteria

### 7.1 Code Quality Metrics

- ✅ All production code has corresponding tests
- ✅ Test coverage >80%
- ✅ All tests pass
- ✅ No skipped tests in main branch
- ✅ Test execution time <2 minutes

### 7.2 Development Process Metrics

- ✅ RED-GREEN-REFACTOR cycle followed for all features
- ✅ Commits show test-first pattern
- ✅ Code reviews verify TDD approach
- ✅ CI/CD pipeline runs tests on every commit

---

## 8. TDD Implementation Checklist

### 8.1 Before Starting Development

- [ ] Review requirements and acceptance criteria
- [ ] Design test fixtures and factories
- [ ] Set up test environment
- [ ] Configure pytest and coverage tools
- [ ] Create test stubs for planned features

### 8.2 During Development (Per Feature)

- [ ] Write failing test (RED)
- [ ] Run test to confirm failure
- [ ] Write minimum code to pass (GREEN)
- [ ] Run test to confirm passing
- [ ] Refactor code while keeping tests passing (REFACTOR)
- [ ] Commit with descriptive message
- [ ] Update documentation

### 8.3 After Development

- [ ] Run full test suite
- [ ] Check code coverage
- [ ] Review test quality
- [ ] Update test plan
- [ ] Document lessons learned

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**TDD Approach**: Red-Green-Refactor
**Target Coverage**: >80%
