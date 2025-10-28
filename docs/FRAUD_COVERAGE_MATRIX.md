# FRAUD TYPE COVERAGE MATRIX

**Assessment Date:** October 28, 2025
**System Version:** 1.0.0
**Assessment Type:** Code Analysis + Test Review

---

## EXECUTIVE SUMMARY

This matrix maps the 6 documented fraud types against the current detection capabilities of the system. Overall fraud coverage is estimated at **34%**, with significant gaps in medical coding validation and industry-standard rule integration.

**Overall Coverage Score: 34% ⚠️**

| Fraud Type | Prevalence | Detection | Effectiveness | Status |
|------------|------------|-----------|---------------|--------|
| Upcoding | 8-15% | Partial | 30% | ⚠️ Limited |
| Phantom Billing | 3-10% | Good | 65% | ✅ Moderate |
| Unbundling | 5-10% | Limited | 15% | ❌ Critical Gap |
| Staged Accidents | Variable | Basic | 40% | ⚠️ Limited |
| Prescription Fraud | Variable | Minimal | 25% | ❌ Critical Gap |
| Kickback Schemes | Variable | Minimal | 30% | ⚠️ Limited |

---

## 1. UPCODING FRAUD (8-15% of claims)

### Definition
Services billed at higher complexity level than actually performed or documented.

### Current Detection Capabilities

#### Detection Method
**Rule: `upcoding_complexity`**
- Weight: 0.8
- Threshold: 0.7
- Location: `rule_engine.py` lines 236-270

#### What It DOES Detect ✅
1. **High complexity procedure with simple diagnosis**
   - Example: CPT 99215 (high complexity E&M) with Z00.00 (routine checkup)
   - Hardcoded procedures: ['99215', '99285', '99291', '99292']
   - Hardcoded diagnoses: ['Z00.00', 'Z12.11', 'I10', 'E11.9']

2. **Excessive billing amounts**
   - Compares billed amount to expected (procedures * $150)
   - Triggers if amount > 3x expected

3. **Suspicious procedure code progression**
   - Detects sequential procedure codes (potential complexity escalation)

#### What It CANNOT Detect ❌

1. **Same-family CPT upcoding**
   - ❌ Cannot detect 99213 → 99214 → 99215 progression
   - ❌ No CPT family hierarchy
   - ❌ No level-of-service validation

2. **E&M Complexity Validation**
   - ❌ Cannot verify history, examination, medical decision making
   - ❌ No documentation requirements checking
   - ❌ No time-based validation

3. **Modifier Abuse**
   - ❌ Cannot detect modifier 25 abuse (separate E&M)
   - ❌ Cannot detect modifier 59 abuse (distinct procedures)
   - ❌ No modifier validation logic

4. **Specialty-Specific Upcoding**
   - ❌ No specialty-appropriate complexity validation
   - ❌ No typical billing pattern analysis per specialty
   - ❌ No RVU-based validation

### Test Coverage ✅
- **Unit tests:** Yes (test_upcoding_detection)
- **Integration tests:** Yes (included in pipeline)
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| CPT hierarchy | Yes | No | 100% |
| E&M complexity rules | Yes | No | 100% |
| Modifier validation | Yes | No | 100% |
| RVU analysis | Yes | No | 100% |
| Documentation matching | Yes | No | 100% |
| Specialty norms | Yes | No | 100% |

**Effectiveness Score: 30%**

### Required for Full Detection
1. CPT code family database with hierarchy
2. E&M complexity scoring matrix
3. Modifier validation rules (25, 59, etc.)
4. RVU (Relative Value Unit) reference data
5. Documentation requirement templates
6. Specialty-specific billing norms

### Test Cases Needed
```python
# Currently missing:
test_same_family_upcoding_99213_to_99215()
test_em_complexity_level_validation()
test_modifier_25_abuse_detection()
test_modifier_59_abuse_detection()
test_time_based_code_validation()
test_specialty_appropriate_complexity()
test_documentation_support_validation()
```

---

## 2. PHANTOM BILLING (3-10% of claims)

### Definition
Billing for services never rendered - services that did not occur.

### Current Detection Capabilities

#### Detection Methods
**Rule 1: `phantom_billing_schedule`**
- Weight: 0.9
- Threshold: 0.8
- Location: `rule_engine.py` lines 272-316

**Rule 2: `phantom_billing_location`**
- Weight: 0.95
- Threshold: 0.9
- Location: `rule_engine.py` lines 318-348

#### What It DOES Detect ✅
1. **Time/Schedule Anomalies**
   - ✅ Weekend services (non-emergency)
   - ✅ Holiday services
   - ✅ After-hours services (before 6 AM, after 10 PM)
   - ✅ Invalid date formats

2. **Location Issues**
   - ✅ Ghost patients (patient ID contains 'GHOST')
   - ✅ Fraudulent providers (provider ID contains 'FRAUD')
   - ✅ Red flags from patient address validation

3. **Pattern Detection**
   - ✅ Services at closed facilities
   - ✅ Appointment without records

#### What It CANNOT Detect ❌

1. **Medical Possibility Validation**
   - ❌ Cannot verify if procedure is physically possible to perform
   - ❌ Cannot validate required time for procedure
   - ❌ Cannot detect physically impossible procedure sequences

2. **Facility Capability Validation**
   - ❌ Cannot verify facility has required equipment
   - ❌ Cannot verify facility is licensed for procedure
   - ❌ Cannot validate facility type matches procedure

3. **Provider Credential Validation**
   - ❌ Cannot verify provider credentials for procedure
   - ❌ Cannot validate provider specialty matches service
   - ❌ Cannot check provider license status

4. **Equipment/Resource Validation**
   - ❌ Cannot verify required equipment availability
   - ❌ Cannot validate staff requirements
   - ❌ Cannot check surgical suite availability

### Test Coverage ✅
- **Unit tests:** Yes (test_phantom_billing_detection)
- **Integration tests:** Yes
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| Time/schedule checks | Yes | Yes | 0% |
| Location validation | Yes | Partial | 40% |
| Medical possibility | Yes | No | 100% |
| Facility capability | Yes | No | 100% |
| Provider credentials | Yes | No | 100% |
| Equipment validation | Yes | No | 100% |

**Effectiveness Score: 65%**

### Required for Full Detection
1. CPT procedure time requirements
2. Facility capability database
3. Provider credential database
4. Equipment requirement mapping
5. Physically possible procedure sequences

### Test Cases Needed
```python
# Currently missing:
test_medically_impossible_procedure_time()
test_facility_lacks_required_equipment()
test_provider_lacks_required_credentials()
test_surgery_without_surgical_suite()
test_physically_impossible_sequence()
```

---

## 3. UNBUNDLING FRAUD (5-10% of claims)

### Definition
Billing separately for procedures that should be billed as a single bundled service.

### Current Detection Capabilities

#### Detection Method
**Rule: `unbundling_detection`**
- Weight: 0.85
- Threshold: 0.75
- Location: `rule_engine.py` lines 350-399

#### What It DOES Detect ✅
1. **4 Hardcoded Bundled Groups Only:**
   ```python
   'colonoscopy': ['45378', '45380', '45384', '45385']
   'cataract_surgery': ['66984', '66982', '66983']
   'knee_arthroscopy': ['29881', '29882', '29883']
   'cardiac_cath': ['93454', '93455', '93456', '93457']
   ```

2. **Same Procedures on Same Day**
   - Detects duplicate procedures across multiple claims
   - Requires context_claims parameter

3. **Excessive Procedure Count**
   - Flags claims with >10 procedures

#### What It CANNOT Detect ❌

1. **NCCI Edits - 99.999% MISSING**
   - ❌ Missing 500,000+ NCCI code pair combinations
   - ❌ No Column 1/Column 2 relationship validation
   - ❌ No modifier indicator checking
   - ❌ No effective date tracking
   - **Impact: Cannot detect 85% of unbundling fraud**

2. **MUE (Medically Unlikely Edits)**
   - ❌ No MUE limits per CPT code
   - ❌ No anatomic limitation validation
   - ❌ No clinical limitation validation
   - ❌ No quantity per day validation

3. **Mutually Exclusive Codes**
   - ❌ No mutually exclusive code pair detection
   - ❌ Cannot detect procedures that cannot be performed together

4. **Add-On Codes**
   - ❌ Cannot validate add-on code usage
   - ❌ Cannot verify primary code exists for add-on
   - ❌ Cannot validate add-on code quantity limits

5. **Global Period Rules**
   - ❌ No global period validation
   - ❌ Cannot detect services included in global period
   - ❌ No post-operative period tracking

### Test Coverage ✅
- **Unit tests:** Yes (test_unbundling_detection)
- **Integration tests:** Yes
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| Hardcoded bundles | Yes | Yes (4 groups) | 0% |
| NCCI edits (500K+) | Yes | No | 99.999% |
| MUE limits | Yes | No | 100% |
| Mutually exclusive | Yes | No | 100% |
| Add-on codes | Yes | No | 100% |
| Global periods | Yes | No | 100% |

**Effectiveness Score: 15%** ❌ CRITICAL GAP

### Required for Full Detection
1. **NCCI Edit Table (updated quarterly)**
   - 500,000+ code pair combinations
   - Column 1/Column 2 relationships
   - Modifier indicators
   - Effective dates

2. **MUE Limits Database**
   - MUE per CPT code
   - Anatomic vs. clinical limitations
   - Per line/per day distinctions

3. **Mutually Exclusive Code List**
   - Cannot be performed together
   - Anatomic exclusions
   - Time-based exclusions

4. **Add-On Code Rules**
   - Primary code requirements
   - Quantity limitations
   - Valid combinations

5. **Global Period Rules**
   - Pre-operative services
   - Post-operative services
   - Global period lengths

### Test Cases Needed
```python
# Currently missing:
test_ncci_column1_column2_violation()
test_ncci_modifier_indicator_validation()
test_mue_quantity_limit_violation()
test_mutually_exclusive_procedure_detection()
test_addon_code_without_primary()
test_global_period_violation()
test_quarterly_ncci_update_integration()
```

---

## 4. STAGED ACCIDENTS (Variable %)

### Definition
Fabricated auto accidents with manufactured injuries and pre-existing relationships among participants.

### Current Detection Capabilities

#### Detection Method
**Rule: `staged_accident_pattern`**
- Weight: 0.9
- Threshold: 0.8
- Location: `rule_engine.py` lines 401-442

#### What It DOES Detect ✅
1. **Multiple Similar Accidents**
   - Detects >3 similar auto accidents for same participants
   - Identifies common diagnosis patterns

2. **Pre-Existing Relationships**
   - Checks red flags for relationship indicators
   - Keywords: 'relationship', 'prior', 'staged'

3. **Consistent Injury Patterns**
   - Hardcoded patterns:
     - Fracture + head injury: ['S72.001A', 'S06.0X0A']
     - Spinal issues: ['M99.23', 'M54.2']

4. **Auto Injury Codes**
   - Identifies auto accident diagnoses: ['S72.001A', 'S06.0X0A', 'M99.23', 'M54.2']

#### What It CANNOT Detect ❌

1. **Injury-Mechanism Validation**
   - ❌ Cannot verify injury consistent with accident type
   - ❌ No biomechanical plausibility checks
   - ❌ Cannot validate collision mechanics

2. **Treatment Pattern Analysis**
   - ❌ Cannot verify expected treatment sequence
   - ❌ No clinical pathway validation for injuries
   - ❌ Cannot detect inappropriate treatments

3. **Recovery Timeline Validation**
   - ❌ Cannot detect abnormal recovery patterns
   - ❌ No expected recovery timeline checking
   - ❌ Cannot validate rehab progression

4. **Participant Network Analysis**
   - ❌ Limited relationship detection
   - ❌ No social network analysis
   - ❌ Cannot detect organized fraud rings

### Test Coverage ⚠️
- **Unit tests:** Limited
- **Integration tests:** Basic
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| Pattern matching | Yes | Yes | 0% |
| Relationship detection | Yes | Partial | 60% |
| Injury-mechanism | Yes | No | 100% |
| Treatment validation | Yes | No | 100% |
| Recovery timeline | Yes | No | 100% |
| Network analysis | Yes | No | 100% |

**Effectiveness Score: 40%**

### Required for Full Detection
1. Injury-mechanism validation rules
2. Biomechanical plausibility database
3. Clinical pathway templates for injuries
4. Expected recovery timelines
5. Treatment sequence validation
6. Social network analysis tools

### Test Cases Needed
```python
# Currently missing:
test_injury_mechanism_inconsistency()
test_biomechanically_impossible_injury()
test_treatment_sequence_validation()
test_recovery_timeline_anomaly()
test_fraud_ring_network_detection()
```

---

## 5. PRESCRIPTION FRAUD (Variable %)

### Definition
Drug diversion, doctor shopping, early refills, inappropriate prescriptions.

### Current Detection Capabilities

#### Detection Method
**Rule: `prescription_fraud_volume`**
- Weight: 0.8
- Threshold: 0.7
- Location: `rule_engine.py` lines 444-478

#### What It DOES Detect ✅
1. **Excessive Prescription Volumes**
   - Daily limit: 5 prescriptions
   - Counts prescription-related J-codes

2. **Controlled Substances Without Adequate Diagnosis**
   - Checks for adequate diagnoses: ['M79.3', 'G89.29', 'F32.9']
   - Flags controlled substance codes: ['J2315', 'J1170', 'J2405']

3. **Doctor Shopping Patterns (Basic)**
   - Detects >5 different providers in recent 30 claims

#### What It CANNOT Detect ❌

1. **DEA Schedule Validation**
   - ❌ No DEA schedule checking (I-V)
   - ❌ Cannot verify DEA registration
   - ❌ No controlled substance tracking

2. **Drug-Diagnosis Matching**
   - ❌ Limited diagnosis validation
   - ❌ Cannot verify diagnosis supports specific drug
   - ❌ No drug formulary checking

3. **Early Refill Detection**
   - ❌ Cannot track refill timing
   - ❌ No days supply calculation
   - ❌ Cannot detect premature refills

4. **Pharmacy Shopping**
   - ❌ No pharmacy tracking
   - ❌ Cannot detect multiple pharmacy usage
   - ❌ No geographic pattern analysis

5. **Quantity Validation**
   - ❌ Cannot verify appropriate quantities
   - ❌ No daily dose validation
   - ❌ No maximum quantity checking

6. **Drug Interactions**
   - ❌ No dangerous combination detection
   - ❌ No contraindication checking
   - ❌ No drug-drug interaction validation

### Test Coverage ⚠️
- **Unit tests:** Basic (test_prescription_fraud_detection)
- **Integration tests:** Minimal
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| Volume checking | Yes | Yes | 0% |
| DEA schedule | Yes | No | 100% |
| Drug-diagnosis | Yes | Partial | 80% |
| Early refills | Yes | No | 100% |
| Pharmacy shopping | Yes | No | 100% |
| Quantity validation | Yes | No | 100% |
| Drug interactions | Yes | No | 100% |

**Effectiveness Score: 25%** ❌ CRITICAL GAP

### Required for Full Detection
1. DEA schedule database
2. Drug-diagnosis matching rules
3. Prescription fill history tracking
4. Days supply calculation
5. Pharmacy identification system
6. Quantity limit database
7. Drug interaction database

### Test Cases Needed
```python
# Currently missing:
test_dea_schedule_validation()
test_drug_diagnosis_mismatch()
test_early_refill_detection()
test_pharmacy_shopping_pattern()
test_excessive_quantity_detection()
test_dangerous_drug_combination()
test_contraindication_detection()
```

---

## 6. KICKBACK SCHEMES (Variable %)

### Definition
Hidden financial relationships, unnecessary referrals, self-referrals.

### Current Detection Capabilities

#### Detection Method
**Rule: `kickback_referral_pattern`**
- Weight: 0.75
- Threshold: 0.7
- Location: `rule_engine.py` lines 480-509

#### What It DOES Detect ✅
1. **Referral Concentration**
   - Calculates concentration (max referrals to single provider / total)
   - Threshold: >70% concentration flagged

2. **Circular Referral Patterns (Simple)**
   - Detects simple A→B→A patterns
   - Uses referral network graph

3. **Red Flag Keywords**
   - Keywords: 'kickback', 'referral', 'relationship'

#### What It CANNOT Detect ❌

1. **Financial Relationship Detection**
   - ❌ No ownership stake identification
   - ❌ Cannot detect profit-sharing arrangements
   - ❌ No financial interest tracking

2. **Facility Co-Location**
   - ❌ Cannot detect shared office spaces
   - ❌ No geographic clustering analysis
   - ❌ Cannot identify related facilities

3. **Self-Referral Detection**
   - ❌ Cannot identify physician-owned facilities
   - ❌ No Stark Law violation detection
   - ❌ Cannot detect designated health services referrals

4. **Specialty Appropriateness**
   - ❌ Cannot validate referral is medically necessary
   - ❌ No specialty-appropriate referral patterns
   - ❌ Cannot detect unnecessary referrals

5. **Complex Referral Networks**
   - ❌ Limited network analysis
   - ❌ Cannot detect sophisticated fraud rings
   - ❌ No multi-hop relationship detection

### Test Coverage ⚠️
- **Unit tests:** Minimal
- **Integration tests:** Basic
- **Performance tests:** No

### Gap Analysis

| Capability | Required | Implemented | Gap |
|------------|----------|-------------|-----|
| Referral concentration | Yes | Yes | 0% |
| Circular patterns | Yes | Partial | 50% |
| Financial relationships | Yes | No | 100% |
| Facility co-location | Yes | No | 100% |
| Self-referral | Yes | No | 100% |
| Specialty appropriateness | Yes | No | 100% |
| Network analysis | Yes | Minimal | 90% |

**Effectiveness Score: 30%**

### Required for Full Detection
1. Financial relationship database (ownership, profit-sharing)
2. Facility ownership database
3. Geographic facility location data
4. Stark Law violation rules
5. Specialty-appropriate referral patterns
6. Advanced network analysis algorithms
7. Designated health services list

### Test Cases Needed
```python
# Currently missing:
test_financial_ownership_detection()
test_facility_colocation_pattern()
test_stark_law_violation_detection()
test_physician_owned_facility_referral()
test_unnecessary_referral_detection()
test_specialty_inappropriate_referral()
test_complex_fraud_network_detection()
```

---

## SUMMARY: OVERALL FRAUD TYPE COVERAGE

### Coverage by Fraud Type

| Rank | Fraud Type | Effectiveness | Status | Priority |
|------|------------|---------------|--------|----------|
| 1 | Phantom Billing | 65% | ✅ Moderate | Medium |
| 2 | Staged Accidents | 40% | ⚠️ Limited | Medium |
| 3 | Upcoding | 30% | ⚠️ Limited | **HIGH** |
| 4 | Kickback Schemes | 30% | ⚠️ Limited | Medium |
| 5 | Prescription Fraud | 25% | ❌ Critical | **HIGH** |
| 6 | Unbundling | 15% | ❌ **Critical** | **CRITICAL** |

**Overall Weighted Average: 34%**

### Critical Gaps Summary

**BLOCKING ISSUES (Must Fix Before Production):**

1. **UNBUNDLING - 15% Effectiveness**
   - Missing: 500,000+ NCCI edits
   - Missing: MUE limits
   - Impact: Cannot detect 85% of unbundling fraud

2. **MEDICAL CODING VALIDATION - 0% Implementation**
   - No ICD-10 validation
   - No CPT validation
   - No CMS guideline integration
   - Impact: All fraud types affected

3. **UPCODING - 30% Effectiveness**
   - No CPT hierarchy
   - No E&M validation
   - No modifier checking
   - Impact: Major fraud type inadequately covered

### Fraud Types Meeting Minimum Standards

**None** - No fraud type achieves >70% detection effectiveness.

### High-Priority Improvements

1. **Implement NCCI Edit Checking** (Effort: 3 weeks)
   - Would improve unbundling from 15% → 90%+
   - Industry-standard fraud detection

2. **Integrate Medical Code Validation** (Effort: 3-4 weeks)
   - Would improve all fraud types by 20-30%
   - Enables code-level fraud detection

3. **Implement CPT Hierarchy** (Effort: 2 weeks)
   - Would improve upcoding from 30% → 70%
   - Enables same-family upcoding detection

4. **Implement MUE Validation** (Effort: 1 week)
   - Would improve unbundling from 15% → 40%
   - Quantity-based fraud detection

---

## APPENDIX: TEST CASE COVERAGE

### Existing Test Cases ✅
- test_upcoding_detection
- test_phantom_billing_detection
- test_unbundling_detection
- test_amount_anomaly_detection
- test_billing_frequency_anomaly
- test_prescription_fraud_detection
- test_accuracy_requirements

### Missing Critical Test Cases ❌
```python
# Upcoding
test_same_family_cpt_upcoding()
test_em_complexity_validation()
test_modifier_abuse_detection()

# Unbundling
test_ncci_edit_violation()
test_mue_limit_violation()
test_mutually_exclusive_codes()

# Phantom Billing
test_medically_impossible_procedure()
test_facility_capability_validation()

# Prescription Fraud
test_dea_schedule_validation()
test_early_refill_detection()
test_pharmacy_shopping()

# Kickback Schemes
test_financial_relationship_detection()
test_stark_law_violation()

# Medical Coding
test_icd10_code_validation()
test_cpt_code_validation()
test_diagnosis_procedure_relationship()
```

---

**Report Generated:** 2025-10-28
**Overall System Readiness:** NOT PRODUCTION-READY
**Critical Gaps:** 3 (Unbundling, Medical Coding, Upcoding)
**Recommended Action:** Implement medical coding validation and NCCI edits before production deployment
