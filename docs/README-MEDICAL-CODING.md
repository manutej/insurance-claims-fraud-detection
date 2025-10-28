# Medical Coding Standards & Fraud Detection Documentation

This directory contains comprehensive research and implementation guides for medical coding standards (ICD-10, CPT) and fraud detection in insurance claims.

## Documents Overview

### 1. MEDICAL-CODING-STANDARDS-RESEARCH.md
**Complete Research Documentation** (3000+ lines)

Comprehensive research on:
- ICD-10-CM diagnosis codes (format, hierarchy, official sources)
- CPT procedure codes (structure, modifiers, categories)
- HCPCS codes for specialized services
- Authoritative reference sources (CMS, NIH, AHA)
- Python validation libraries (simple-icd-10-cm, icdcodex)
- API-based code validation approaches
- Diagnosis-procedure combination validation strategies
- Common fraud patterns (upcoding, unbundling, phantom billing, etc.)
- Medical necessity validation rules
- NCCI bundling rules and conflicts
- Integration recommendations for your fraud detection system

**Key Findings:**
- **ICD-10-CM API:** https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search (free, official)
- **Best Library:** simple-icd-10-cm for Python validation
- **CPT Challenge:** No free official API, requires manual reference database
- **Fraud Patterns:** Upcoding most common, requires diagnosis-procedure validation

### 2. MEDICAL-CODING-IMPLEMENTATION-GUIDE.md
**Step-by-Step Implementation** (2000+ lines)

Practical implementation guide with:
- Installation instructions for required libraries
- Complete Python module implementations:
  - `src/medical_coding/icd10_validator.py` - ICD-10-CM validation
  - `src/medical_coding/cpt_validator.py` - CPT code validation
  - `src/medical_coding/combo_validator.py` - Diagnosis-procedure validation
- Integration with existing `src/ingestion/validator.py`
- Unit test examples
- Performance optimization techniques
- Caching strategies for high-volume claims
- Error handling patterns

**Implementation Timeline:**
- Phase 1 (Week 1-2): Install dependencies, basic ICD-10 validation
- Phase 2 (Week 3-4): CPT reference database, combination validation
- Phase 3 (Week 5-6): Fraud detection integration, network analysis

### 3. MEDICAL-CODING-REFERENCE.md
**Complete Reference Tables** (1500+ lines)

Quick reference material:
- Common diagnosis codes by category (diabetes, hypertension, infections)
- CPT codes with complexity levels and typical procedures
- CPT modifiers and valid use cases
- Place of service codes
- NCCI bundling rules (mutual exclusions, add-on codes)
- Valid diagnosis-procedure combinations
- Fraud risk thresholds and scoring
- Red flag weights for pattern detection
- Provider benchmark data (normal billing patterns)
- CMS reference URLs

**Quick Access:**
- Copy-paste code definitions for testing
- Pre-built fraud detection rules
- Peer benchmark data for anomaly detection

---

## How to Use These Documents

### For Architects/Managers:
1. Read **MEDICAL-CODING-STANDARDS-RESEARCH.md** - Executive Summary section
2. Review key findings and integration recommendations
3. Understand fraud patterns and detection challenges

### For Implementation Teams:
1. Follow **MEDICAL-CODING-IMPLEMENTATION-GUIDE.md** step-by-step
2. Create module files in `src/medical_coding/`
3. Write unit tests from provided examples
4. Integrate with existing validator
5. Use **MEDICAL-CODING-REFERENCE.md** for test data

### For Fraud Analysts:
1. Review **MEDICAL-CODING-REFERENCE.md** - Fraud Detection Patterns
2. Understand red flags and risk scoring
3. Use provider benchmarks for anomaly detection
4. Create investigation guidelines based on patterns

### For Developers:
1. Implement modules from Implementation Guide
2. Use Reference for test data and validation rules
3. Set up caching for performance
4. Add monitoring for code validation metrics

---

## Key Findings Summary

### Authoritative Sources

| Source | Type | Cost | Best For |
|--------|------|------|----------|
| NIH Clinical Table Search Service | API | Free | ICD-10-CM validation |
| CMS ICD-10 Files | Data Files | Free | Bulk validation, offline |
| simple-icd-10-cm | Python Library | Free | Application integration |
| AHA Coding Clinic | Guidance | Subscription | Complex coding questions |
| Local MAC | Guidance | Free | Regional interpretation |

### Python Libraries

**Recommended:** `simple-icd-10-cm`
```bash
pip install simple-icd-10-cm
```

**Features:**
- Validates ICD-10-CM codes
- Returns code descriptions
- Hierarchical navigation (parent/children)
- Fast performance
- Latest 2025 code set

**Limitations:**
- Diagnosis codes only (no CPT)
- No clinical validation rules
- No fraud pattern detection

### Fraud Detection Capabilities

**Three-Tier Strategy:**

```
Tier 1: Schema Validation (Basic)
├─ Format checks (regex patterns)
├─ Type validation
└─ Required fields

Tier 2: Code Validation (Intermediate)
├─ ICD-10-CM existence
├─ CPT code existence
└─ Modifier validation

Tier 3: Clinical Validation (Advanced)
├─ Diagnosis-procedure combinations
├─ Medical necessity rules
├─ NCCI bundling rules
└─ Fraud pattern detection
```

**Current Gap:** Your existing validator is at Tier 1, needs Tier 2 and 3

### Top Fraud Patterns to Detect

1. **Upcoding** (Most Common)
   - Simple diagnosis + complex procedure
   - Example: Routine physical (Z00) billed with ER emergency code (99285)
   - Detection: Diagnosis-procedure combination rules

2. **Unbundling**
   - Component procedures without parent
   - Multiple E/M codes same day
   - Example: Bill 99213 + 99214 + 99215 same day (violates NCCI)
   - Detection: NCCI bundling rules

3. **Phantom Billing**
   - Services never rendered
   - Weekend/holiday services when closed
   - Detection: Temporal analysis, date rules

4. **Prescription Fraud**
   - Early refills, doctor shopping
   - Example: Refill controlled substance 5 days early
   - Detection: Frequency analysis, prescriber patterns

---

## Integration Checklist

### Phase 1: Discovery (Completed)
- [x] Research medical coding standards
- [x] Identify Python libraries
- [x] Review authoritative sources
- [x] Document fraud patterns

### Phase 2: Implementation (Ready)
- [ ] Install simple-icd-10-cm
- [ ] Create icd10_validator.py module
- [ ] Create cpt_validator.py module
- [ ] Create combo_validator.py module
- [ ] Write unit tests
- [ ] Integrate with validator.py
- [ ] Test with sample claims
- [ ] Performance testing

### Phase 3: Enhancement (Planning)
- [ ] Add monitoring dashboard
- [ ] Create fraud alert system
- [ ] Build provider benchmark database
- [ ] Implement network analysis
- [ ] Create investigation workflows
- [ ] Document compliance audit trail

---

## File Structure

```
docs/
├── README-MEDICAL-CODING.md (THIS FILE)
├── MEDICAL-CODING-STANDARDS-RESEARCH.md (3000 lines)
├── MEDICAL-CODING-IMPLEMENTATION-GUIDE.md (2000 lines)
└── MEDICAL-CODING-REFERENCE.md (1500 lines)

src/medical_coding/ (To be created)
├── __init__.py
├── icd10_validator.py
├── cpt_validator.py
└── combo_validator.py

tests/unit/ (To be created)
├── test_icd10_validator.py
├── test_cpt_validator.py
└── test_combo_validator.py
```

---

## Quick Start Commands

### Install Dependencies
```bash
pip install simple-icd-10-cm requests
```

### Test ICD-10 Code
```python
from icd10cm import ICD10CM

icd = ICD10CM()
print(icd.get_description('E11.9'))  # Type 2 diabetes without complications
```

### Search ICD-10 Via API
```bash
curl "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms=diabetes&count=5"
```

### Validate Claim
```python
from src.medical_coding.combo_validator import DiagnosisProcedureValidator

validator = DiagnosisProcedureValidator()
result = validator.validate_claim_codes(
    diagnoses=['E11.9', 'I10'],
    procedures=['99213', '80053']
)
print(f"Valid: {result['valid']}")
```

---

## Next Steps

1. **Read MEDICAL-CODING-STANDARDS-RESEARCH.md** for complete context
2. **Review MEDICAL-CODING-IMPLEMENTATION-GUIDE.md** for code examples
3. **Reference MEDICAL-CODING-REFERENCE.md** during implementation
4. **Install dependencies** (simple-icd-10-cm)
5. **Create medical_coding modules** with provided code
6. **Write and run tests**
7. **Integrate with existing validator**
8. **Deploy and monitor**

---

## References

### Official Resources
- **CMS:** https://www.cms.gov/medicare/coding-billing/icd-10-codes
- **CDC:** https://www.cdc.gov/nchs/icd/icd10cm.htm
- **AMA:** https://www.ama-assn.org/practice-management/cpt

### APIs
- **NIH Clinical Table Search:** https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html
- **ICD-10 API:** https://www.icd10api.com/

### Python Libraries
- **simple-icd-10-cm:** https://github.com/StefanoTrv/simple_icd_10_CM
- **PyPI:** https://pypi.org/project/simple-icd-10-cm/

---

## Contact & Support

### For Questions:
- **ICD-10-CM:** Contact CDC (nchsicd10cm@cdc.gov)
- **CPT Codes:** Contact AMA or your MAC (Medicare Administrative Contractor)
- **Implementation:** Review code examples in MEDICAL-CODING-IMPLEMENTATION-GUIDE.md

### For Updates:
- CMS publishes updates annually (October 1)
- Monitor https://www.cms.gov/medicare/coding-billing/icd-10-codes for new releases
- Update simple-icd-10-cm library annually

---

**Documentation Version:** 1.0
**Created:** 2025-10-28
**Status:** Complete and ready for implementation
**Effort:** 8-10 weeks to fully implement all three phases

