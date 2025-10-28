# Fraud Pattern Test Cases

## Executive Summary

This document provides comprehensive test cases for all six fraud patterns detected by the insurance fraud detection system. Each fraud type includes multiple test scenarios covering obvious fraud, moderate suspicion, subtle patterns, and edge cases.

## Table of Contents

1. [Upcoding Test Cases](#1-upcoding-test-cases)
2. [Phantom Billing Test Cases](#2-phantom-billing-test-cases)
3. [Unbundling Test Cases](#3-unbundling-test-cases)
4. [Staged Accident Test Cases](#4-staged-accident-test-cases)
5. [Prescription Fraud Test Cases](#5-prescription-fraud-test-cases)
6. [Kickback Scheme Test Cases](#6-kickback-scheme-test-cases)
7. [Multi-Pattern Test Cases](#7-multi-pattern-test-cases)
8. [Edge Cases and Boundary Conditions](#8-edge-cases-and-boundary-conditions)

---

## 1. Upcoding Test Cases

### 1.1 Obvious Upcoding

#### TC-UP-001: Simple Diagnosis, Complex Procedure
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.85-0.95)

**Scenario**: Common cold billed as high-complexity office visit

```json
{
  "claim_id": "CLM-TEST-UP-001",
  "diagnosis_codes": ["J00"],
  "diagnosis_descriptions": ["Acute nasopharyngitis (common cold)"],
  "procedure_codes": ["99215"],
  "procedure_descriptions": ["Office visit, established patient, high complexity"],
  "billed_amount": 325.00,
  "actual_service": "99212",
  "actual_amount": 75.00,
  "rendering_hours": 0.25
}
```

**Expected Detection**:
- Fraud Type: Upcoding
- Triggered Rules: `upcoding_complexity`, `amount_anomaly`
- Red Flags:
  - Simple diagnosis (cold) with complex procedure code
  - Billed amount 4.3x expected for diagnosis
  - Rendering time inconsistent with complexity

---

#### TC-UP-002: Procedure Complexity Escalation
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.80-0.90)

**Scenario**: Routine diabetes follow-up billed as comprehensive visit

```json
{
  "claim_id": "CLM-TEST-UP-002",
  "diagnosis_codes": ["E11.9"],
  "diagnosis_descriptions": ["Type 2 diabetes mellitus without complications"],
  "procedure_codes": ["99215"],
  "procedure_descriptions": ["Office visit, high complexity"],
  "billed_amount": 295.00,
  "actual_service": "99213",
  "actual_amount": 125.00,
  "provider_pattern": "90% of visits billed as 99215"
}
```

**Expected Detection**:
- Fraud Type: Upcoding
- Triggered Rules: `upcoding_complexity`, `provider_upcoding_pattern`
- Red Flags:
  - Routine follow-up billed at highest complexity
  - Provider consistently bills maximum codes
  - Amount exceeds typical for diabetes follow-up

---

#### TC-UP-003: Physical Therapy Upcoding
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.70-0.85)

**Scenario**: Simple muscle pain billed with multiple therapy codes

```json
{
  "claim_id": "CLM-TEST-UP-003",
  "diagnosis_codes": ["M79.3"],
  "diagnosis_descriptions": ["Myalgia"],
  "procedure_codes": ["97140", "97110", "97112", "97530"],
  "procedure_descriptions": [
    "Manual therapy",
    "Therapeutic exercises",
    "Neuromuscular reeducation",
    "Therapeutic activities"
  ],
  "billed_amount": 485.00,
  "actual_service": "97110",
  "actual_amount": 125.00,
  "rendering_hours": 1.0,
  "total_therapy_time": 45
}
```

**Expected Detection**:
- Fraud Type: Upcoding
- Triggered Rules: `upcoding_complexity`, `time_duration_mismatch`
- Red Flags:
  - Multiple therapy codes for simple pain
  - Total billed time exceeds actual appointment
  - Simple diagnosis doesn't support extensive therapy

---

### 1.2 Moderate Upcoding

#### TC-UP-004: Diagnostic Test Overutilization
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.65-0.80)

**Scenario**: Simple cough with unnecessary pulmonary function test

```json
{
  "claim_id": "CLM-TEST-UP-004",
  "diagnosis_codes": ["R05"],
  "diagnosis_descriptions": ["Cough"],
  "procedure_codes": ["99213", "94060"],
  "procedure_descriptions": [
    "Office visit, moderate complexity",
    "Bronchodilation responsiveness"
  ],
  "billed_amount": 285.00,
  "medical_necessity": "questionable"
}
```

**Expected Detection**:
- Fraud Type: Upcoding
- Triggered Rules: `upcoding_complexity`, `medical_necessity`
- Red Flags:
  - Pulmonary function test for simple cough
  - No medical necessity documentation
  - Test typically for asthma/COPD diagnosis

---

### 1.3 Subtle Upcoding

#### TC-UP-005: Borderline Complexity Escalation
**Severity**: Subtle
**Expected Detection**: MAYBE (Score: 0.50-0.70)

**Scenario**: Moderately complex visit billed as high complexity

```json
{
  "claim_id": "CLM-TEST-UP-005",
  "diagnosis_codes": ["E11.9", "I10", "E78.5"],
  "diagnosis_descriptions": [
    "Type 2 diabetes mellitus",
    "Essential hypertension",
    "Hyperlipidemia"
  ],
  "procedure_codes": ["99215"],
  "procedure_descriptions": ["Office visit, high complexity"],
  "billed_amount": 275.00,
  "rendering_hours": 0.75,
  "number_of_diagnoses": 3
}
```

**Expected Detection**:
- Fraud Type: Upcoding (borderline)
- Triggered Rules: Possibly `upcoding_complexity` (lower confidence)
- Notes:
  - Multiple chronic conditions may justify higher complexity
  - Time supports moderate complexity
  - Could be legitimate or mild upcoding

---

### 1.4 Valid High-Complexity Claims (Negative Cases)

#### TC-UP-NEG-001: Legitimate High Complexity
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.3)

**Scenario**: Pneumonia emergency visit with imaging

```json
{
  "claim_id": "CLM-TEST-UP-NEG-001",
  "diagnosis_codes": ["J18.9"],
  "diagnosis_descriptions": ["Pneumonia, unspecified organism"],
  "procedure_codes": ["99285", "71046"],
  "procedure_descriptions": [
    "Emergency dept visit, high severity",
    "Chest X-ray, 2 views"
  ],
  "billed_amount": 1250.00,
  "service_location": "23",
  "rendering_hours": 1.5
}
```

**Expected Detection**:
- Fraud Type: None
- Triggered Rules: None
- Notes: Legitimate emergency presentation requiring high-complexity care

---

## 2. Phantom Billing Test Cases

### 2.1 Obvious Phantom Billing

#### TC-PB-001: Service Outside Operating Hours
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.90-1.0)

**Scenario**: Office visit billed at 2:00 AM

```json
{
  "claim_id": "CLM-TEST-PB-001",
  "diagnosis_codes": ["Z00.00"],
  "diagnosis_descriptions": ["Encounter for general adult medical examination"],
  "procedure_codes": ["99213"],
  "procedure_descriptions": ["Office visit"],
  "billed_amount": 125.00,
  "date_of_service": "2024-03-15",
  "time_of_service": "02:00:00",
  "service_location": "11",
  "provider_hours": "9am-5pm Mon-Fri"
}
```

**Expected Detection**:
- Fraud Type: Phantom Billing
- Triggered Rules: `phantom_billing_schedule`
- Red Flags:
  - Office visit at 2 AM
  - Provider office closed
  - Non-emergency service outside normal hours

---

#### TC-PB-002: Ghost Patient
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.95-1.0)

**Scenario**: Service billed for non-existent patient

```json
{
  "claim_id": "CLM-TEST-PB-002",
  "patient_id": "PAT-GHOST-001",
  "diagnosis_codes": ["M79.3"],
  "diagnosis_descriptions": ["Myalgia"],
  "procedure_codes": ["97110"],
  "procedure_descriptions": ["Therapeutic exercises"],
  "billed_amount": 95.00,
  "patient_address_verification": "failed",
  "patient_contact_verification": "failed"
}
```

**Expected Detection**:
- Fraud Type: Phantom Billing
- Triggered Rules: `phantom_billing_location`, `ghost_patient_detection`
- Red Flags:
  - Patient address doesn't exist
  - No contact information verified
  - No appointment records

---

#### TC-PB-003: Weekend Non-Emergency Services
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.70-0.85)

**Scenario**: Routine office visit billed on Sunday

```json
{
  "claim_id": "CLM-TEST-PB-003",
  "diagnosis_codes": ["Z00.00"],
  "diagnosis_descriptions": ["General medical examination"],
  "procedure_codes": ["99213"],
  "procedure_descriptions": ["Office visit"],
  "billed_amount": 125.00,
  "date_of_service": "2024-03-17",
  "day_of_week": "Sunday",
  "service_location": "11"
}
```

**Expected Detection**:
- Fraud Type: Phantom Billing
- Triggered Rules: `phantom_billing_schedule`
- Red Flags:
  - Routine office visit on Sunday
  - Non-emergency service
  - Office typically closed weekends

---

### 2.2 Moderate Phantom Billing

#### TC-PB-004: Service During Provider Absence
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.75-0.90)

**Scenario**: Services billed while provider on vacation

```json
{
  "claim_id": "CLM-TEST-PB-004",
  "provider_id": "PRV-20001",
  "diagnosis_codes": ["E11.9"],
  "procedure_codes": ["99213"],
  "billed_amount": 125.00,
  "date_of_service": "2024-07-15",
  "provider_vacation_dates": ["2024-07-10", "2024-07-20"],
  "no_coverage_provider": true
}
```

**Expected Detection**:
- Fraud Type: Phantom Billing
- Triggered Rules: `phantom_billing_schedule`, `provider_availability`
- Red Flags:
  - Provider documented on vacation
  - No substitute provider documented
  - Patient has no appointment record

---

### 2.3 Valid Weekend/After-Hours Services (Negative Cases)

#### TC-PB-NEG-001: Legitimate Emergency Service
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.2)

**Scenario**: Emergency room visit on Sunday

```json
{
  "claim_id": "CLM-TEST-PB-NEG-001",
  "diagnosis_codes": ["S52.501A"],
  "diagnosis_descriptions": ["Unspecified fracture of lower end of right radius"],
  "procedure_codes": ["99285", "73100"],
  "procedure_descriptions": ["Emergency dept visit, high severity", "Wrist X-ray"],
  "billed_amount": 850.00,
  "date_of_service": "2024-03-17",
  "day_of_week": "Sunday",
  "service_location": "23"
}
```

**Expected Detection**:
- Fraud Type: None
- Notes: Legitimate emergency care on weekend

---

## 3. Unbundling Test Cases

### 3.1 Obvious Unbundling

#### TC-UB-001: Colonoscopy Procedure Unbundling
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.85-0.95)

**Scenario**: Colonoscopy components billed separately

```json
{
  "claim_id": "CLM-TEST-UB-001",
  "diagnosis_codes": ["K59.00"],
  "diagnosis_descriptions": ["Constipation, unspecified"],
  "procedure_codes": ["45378", "45380", "45384"],
  "procedure_descriptions": [
    "Colonoscopy, diagnostic",
    "Colonoscopy with biopsy",
    "Colonoscopy with lesion removal"
  ],
  "billed_amount": 4200.00,
  "should_be_bundled_as": "45385",
  "expected_bundled_amount": 1800.00,
  "date_of_service": "2024-03-15"
}
```

**Expected Detection**:
- Fraud Type: Unbundling
- Triggered Rules: `unbundling_detection`, `bundled_procedures`
- Red Flags:
  - Multiple colonoscopy codes same day
  - Procedures should be bundled under single code
  - Billed amount 2.3x expected

---

#### TC-UB-002: Physical Therapy Session Unbundling
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.80-0.90)

**Scenario**: Single therapy session split into multiple codes

```json
{
  "claim_id": "CLM-TEST-UB-002",
  "diagnosis_codes": ["M54.5"],
  "diagnosis_descriptions": ["Low back pain"],
  "procedure_codes": ["97110", "97112", "97140", "97530"],
  "procedure_descriptions": [
    "Therapeutic exercises",
    "Neuromuscular reeducation",
    "Manual therapy",
    "Therapeutic activities"
  ],
  "billed_amount": 440.00,
  "rendering_hours": 1.0,
  "appointment_duration": 45,
  "individual_procedure_minutes": [15, 15, 15, 15]
}
```

**Expected Detection**:
- Fraud Type: Unbundling
- Triggered Rules: `unbundling_detection`, `time_duration_mismatch`
- Red Flags:
  - Four therapy codes for 45-minute session
  - Each code typically requires 15+ minutes
  - Total claimed time exceeds appointment

---

#### TC-UB-003: Same-Day Duplicate Procedures
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.90-0.95)

**Scenario**: Same procedure billed twice on same day

```json
{
  "claim_id": "CLM-TEST-UB-003",
  "patient_id": "PAT-10001",
  "date_of_service": "2024-03-15",
  "procedure_codes": ["99213"],
  "billed_amount": 125.00,
  "related_claims": [
    {
      "claim_id": "CLM-TEST-UB-003-B",
      "patient_id": "PAT-10001",
      "date_of_service": "2024-03-15",
      "procedure_codes": ["99213"],
      "billed_amount": 125.00,
      "provider_id": "PRV-20001",
      "time_difference_minutes": 30
    }
  ]
}
```

**Expected Detection**:
- Fraud Type: Unbundling
- Triggered Rules: `unbundling_detection`, `duplicate_claims`
- Red Flags:
  - Identical procedure billed twice same day
  - Same provider, patient, date
  - Insufficient time between services

---

### 3.2 Valid Multiple Procedures (Negative Cases)

#### TC-UB-NEG-001: Legitimate Multiple Procedures
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.3)

**Scenario**: Distinct procedures performed in same session

```json
{
  "claim_id": "CLM-TEST-UB-NEG-001",
  "diagnosis_codes": ["C50.911", "Z85.3"],
  "diagnosis_descriptions": [
    "Malignant neoplasm of breast",
    "Personal history of malignant neoplasm"
  ],
  "procedure_codes": ["19307", "38525"],
  "procedure_descriptions": [
    "Mastectomy, modified radical",
    "Biopsy or excision of lymph nodes"
  ],
  "billed_amount": 8500.00,
  "rendering_hours": 3.5
}
```

**Expected Detection**:
- Fraud Type: None
- Notes: Legitimate surgical procedures performed together

---

## 4. Staged Accident Test Cases

### 4.1 Obvious Staged Accidents

#### TC-SA-001: Identical Injury Pattern
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.85-0.95)

**Scenario**: Multiple patients with identical injuries from same provider

```json
{
  "claim_id": "CLM-TEST-SA-001",
  "patient_id": "PAT-30001",
  "diagnosis_codes": ["S72.001A", "S06.0X0A", "M99.23"],
  "diagnosis_descriptions": [
    "Fracture of unspecified part of neck of right femur",
    "Concussion without loss of consciousness",
    "Subluxation of cervical region"
  ],
  "accident_date": "2024-03-10",
  "accident_location": "Intersection of Main St and 1st Ave",
  "attorney_involved": true,
  "attorney_id": "ATT-001",
  "similar_accidents": [
    {
      "patient_id": "PAT-30002",
      "diagnosis_codes": ["S72.001A", "S06.0X0A", "M99.23"],
      "accident_location": "Intersection of Main St and 1st Ave",
      "attorney_id": "ATT-001",
      "days_apart": 15
    },
    {
      "patient_id": "PAT-30003",
      "diagnosis_codes": ["S72.001A", "S06.0X0A", "M99.23"],
      "accident_location": "Same intersection",
      "attorney_id": "ATT-001",
      "days_apart": 30
    }
  ]
}
```

**Expected Detection**:
- Fraud Type: Staged Accident
- Triggered Rules: `staged_accident_pattern`, `identical_injury_pattern`
- Red Flags:
  - Three accidents with identical injury patterns
  - Same location, same attorney
  - Suspiciously similar timing

---

#### TC-SA-002: Pre-Existing Relationships
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.80-0.90)

**Scenario**: Patients knew providers/attorneys before accident

```json
{
  "claim_id": "CLM-TEST-SA-002",
  "patient_id": "PAT-30004",
  "provider_id": "PRV-30001",
  "accident_date": "2024-03-15",
  "pre_existing_relationships": [
    "Patient visited same provider 6 times in year before accident",
    "Patient and provider share same address (family)",
    "Attorney is patient's neighbor"
  ],
  "attorney_involved": true,
  "treatment_started": "Same day as accident"
}
```

**Expected Detection**:
- Fraud Type: Staged Accident
- Triggered Rules: `staged_accident_pattern`, `pre_existing_relationship`
- Red Flags:
  - Family relationship between patient and provider
  - Pre-existing relationship with attorney
  - Immediate treatment suggests pre-planning

---

### 4.2 Valid Accident Claims (Negative Cases)

#### TC-SA-NEG-001: Legitimate Auto Accident
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.3)

```json
{
  "claim_id": "CLM-TEST-SA-NEG-001",
  "diagnosis_codes": ["S52.501A"],
  "diagnosis_descriptions": ["Fracture of lower end of right radius"],
  "accident_date": "2024-03-15",
  "accident_location": "Highway 101, northbound",
  "police_report_number": "PR-2024-123456",
  "attorney_involved": false,
  "treatment_started": "2024-03-16",
  "emergency_room_visit": true
}
```

**Expected Detection**:
- Fraud Type: None
- Notes: Legitimate accident with police report and appropriate treatment

---

## 5. Prescription Fraud Test Cases

### 5.1 Obvious Prescription Fraud

#### TC-PF-001: Doctor Shopping
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.90-0.95)

**Scenario**: Patient obtains same controlled substance from multiple providers

```json
{
  "claim_id": "CLM-TEST-PF-001",
  "patient_id": "PAT-40001",
  "diagnosis_codes": ["G89.29"],
  "diagnosis_descriptions": ["Other chronic pain"],
  "procedure_codes": ["J2315"],
  "procedure_descriptions": ["Injection, oxycodone"],
  "prescriber_npi": "1234567890",
  "fill_date": "2024-03-15",
  "patient_prescription_history_30_days": [
    {"prescriber_npi": "1234567891", "drug": "oxycodone", "days_ago": 7},
    {"prescriber_npi": "1234567892", "drug": "oxycodone", "days_ago": 14},
    {"prescriber_npi": "1234567893", "drug": "oxycodone", "days_ago": 21},
    {"prescriber_npi": "1234567894", "drug": "oxycodone", "days_ago": 28}
  ]
}
```

**Expected Detection**:
- Fraud Type: Prescription Fraud
- Triggered Rules: `prescription_fraud_volume`, `doctor_shopping_pattern`
- Red Flags:
  - Five different providers in 30 days
  - All for same controlled substance
  - Overlapping prescriptions

---

#### TC-PF-002: Early Refill Pattern
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.85-0.95)

**Scenario**: Prescriptions refilled too early repeatedly

```json
{
  "claim_id": "CLM-TEST-PF-002",
  "patient_id": "PAT-40002",
  "drug_name": "Hydrocodone/Acetaminophen",
  "days_supply": 30,
  "fill_date": "2024-03-15",
  "last_fill_date": "2024-02-26",
  "days_since_last_fill": 18,
  "expected_days_between_fills": 30,
  "early_refill_pattern_6_months": [
    {"days_early": 12},
    {"days_early": 10},
    {"days_early": 15},
    {"days_early": 8},
    {"days_early": 12}
  ]
}
```

**Expected Detection**:
- Fraud Type: Prescription Fraud
- Triggered Rules: `prescription_fraud_volume`, `early_refill_pattern`
- Red Flags:
  - Refilled 12 days early (40% of supply)
  - Pattern of early refills over 6 months
  - Indicates diversion or addiction

---

#### TC-PF-003: Controlled Substance Without Diagnosis
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.75-0.85)

**Scenario**: Opioid prescription with inadequate diagnosis

```json
{
  "claim_id": "CLM-TEST-PF-003",
  "diagnosis_codes": ["M79.3"],
  "diagnosis_descriptions": ["Myalgia"],
  "procedure_codes": ["J2315"],
  "procedure_descriptions": ["Injection, oxycodone"],
  "quantity": 120,
  "days_supply": 30,
  "no_prior_conservative_treatment": true,
  "first_visit_with_provider": true
}
```

**Expected Detection**:
- Fraud Type: Prescription Fraud
- Triggered Rules: `controlled_substance_diagnosis_mismatch`
- Red Flags:
  - High-dose opioid for simple muscle pain
  - No prior conservative treatment
  - First visit with provider

---

### 5.2 Valid Prescription Claims (Negative Cases)

#### TC-PF-NEG-001: Legitimate Chronic Pain Management
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.3)

```json
{
  "claim_id": "CLM-TEST-PF-NEG-001",
  "diagnosis_codes": ["G89.29", "M48.06"],
  "diagnosis_descriptions": [
    "Other chronic pain",
    "Spinal stenosis, lumbar region"
  ],
  "drug_name": "Morphine Sulfate",
  "quantity": 60,
  "days_supply": 30,
  "patient_treatment_history": "2+ years with same provider",
  "pain_management_agreement": true,
  "regular_monitoring": true
}
```

**Expected Detection**:
- Fraud Type: None
- Notes: Legitimate chronic pain management with proper documentation

---

## 6. Kickback Scheme Test Cases

### 6.1 Obvious Kickback Schemes

#### TC-KB-001: High Referral Concentration
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.80-0.90)

**Scenario**: Provider refers 95% of patients to single specialist

```json
{
  "claim_id": "CLM-TEST-KB-001",
  "provider_id": "PRV-50001",
  "diagnosis_codes": ["M79.3"],
  "procedure_codes": ["99213"],
  "referred_to_provider": "PRV-50002",
  "referral_specialty": "Pain Management",
  "provider_referral_statistics_12_months": {
    "total_referrals": 200,
    "referrals_to_PRV-50002": 190,
    "referral_concentration": 0.95,
    "next_highest_referrals": 5,
    "financial_relationship": "Co-owners of medical building"
  }
}
```

**Expected Detection**:
- Fraud Type: Kickback Scheme
- Triggered Rules: `kickback_referral_pattern`, `referral_concentration`
- Red Flags:
  - 95% referral concentration
  - Financial relationship between providers
  - Unusual referral pattern

---

#### TC-KB-002: Circular Referral Pattern
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.85-0.95)

**Scenario**: Providers refer patients back and forth

```json
{
  "claim_id": "CLM-TEST-KB-002",
  "patient_id": "PAT-50001",
  "referral_chain": [
    {"provider": "PRV-50001", "service": "Initial consultation", "date": "2024-01-10"},
    {"provider": "PRV-50002", "service": "Specialist consult", "date": "2024-01-17"},
    {"provider": "PRV-50003", "service": "Diagnostic testing", "date": "2024-01-24"},
    {"provider": "PRV-50001", "service": "Follow-up", "date": "2024-02-01"},
    {"provider": "PRV-50002", "service": "Re-evaluation", "date": "2024-02-08"}
  ],
  "total_billed": 3500.00,
  "medical_necessity_questionable": true,
  "providers_share_billing_office": true
}
```

**Expected Detection**:
- Fraud Type: Kickback Scheme
- Triggered Rules: `kickback_referral_pattern`, `circular_referral`
- Red Flags:
  - Circular referral pattern
  - Providers share billing office
  - Questionable medical necessity

---

#### TC-KB-003: Unnecessary Referrals
**Severity**: Moderate
**Expected Detection**: TRUE (Score: 0.70-0.85)

**Scenario**: Routine conditions referred to expensive specialists

```json
{
  "claim_id": "CLM-TEST-KB-003",
  "diagnosis_codes": ["E11.9"],
  "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
  "provider_id": "PRV-50001",
  "referred_to_provider": "PRV-50002",
  "referral_specialty": "Endocrinology",
  "patient_condition": "Stable, well-controlled diabetes",
  "medical_necessity": "questionable",
  "typical_treatment": "Primary care management sufficient"
}
```

**Expected Detection**:
- Fraud Type: Kickback Scheme
- Triggered Rules: `unnecessary_referral_pattern`
- Red Flags:
  - Stable condition doesn't require specialist
  - Pattern of unnecessary referrals
  - Financial incentive suspected

---

### 6.2 Valid Referral Patterns (Negative Cases)

#### TC-KB-NEG-001: Legitimate Specialist Referral
**Severity**: Legitimate
**Expected Detection**: FALSE (Score: 0.0-0.3)

```json
{
  "claim_id": "CLM-TEST-KB-NEG-001",
  "diagnosis_codes": ["C50.911"],
  "diagnosis_descriptions": ["Malignant neoplasm of breast"],
  "provider_id": "PRV-50001",
  "referred_to_provider": "PRV-50002",
  "referral_specialty": "Oncology",
  "medical_necessity": "clear",
  "referral_pattern": "Appropriate specialist care"
}
```

**Expected Detection**:
- Fraud Type: None
- Notes: Appropriate specialist referral for cancer diagnosis

---

## 7. Multi-Pattern Test Cases

### 7.1 Combined Fraud Patterns

#### TC-MP-001: Upcoding + Unbundling
**Severity**: Obvious
**Expected Detection**: TRUE (Score: 0.90-0.95)

```json
{
  "claim_id": "CLM-TEST-MP-001",
  "diagnosis_codes": ["M79.3"],
  "diagnosis_descriptions": ["Myalgia"],
  "procedure_codes": ["99215", "97140", "97110", "97112", "97530"],
  "billed_amount": 725.00,
  "actual_service": "99213 + 97110",
  "actual_amount": 200.00,
  "rendering_hours": 1.5
}
```

**Expected Detection**:
- Fraud Types: Upcoding, Unbundling
- Multiple triggered rules
- High fraud score from combined patterns

---

## 8. Edge Cases and Boundary Conditions

### 8.1 Borderline Cases

#### TC-EDGE-001: High but Justified Billing
**Severity**: Legitimate (borderline)
**Expected Detection**: FALSE (Score: 0.3-0.5)

```json
{
  "claim_id": "CLM-TEST-EDGE-001",
  "diagnosis_codes": ["I50.9", "N18.9", "E11.9", "I10", "E78.5"],
  "procedure_codes": ["99215"],
  "billed_amount": 295.00,
  "rendering_hours": 1.25,
  "number_of_diagnoses": 5,
  "complexity_justification": "Multiple chronic conditions requiring extensive review"
}
```

**Expected Detection**:
- Fraud Type: None (borderline)
- Notes: Borderline case where high complexity may be justified

---

### 8.2 Missing Data Scenarios

#### TC-EDGE-002: Incomplete Claim Data
**Severity**: Varies
**Expected Detection**: Depends on enrichment

```json
{
  "claim_id": "CLM-TEST-EDGE-002",
  "procedure_codes": ["99213"],
  "billed_amount": 125.00,
  "diagnosis_codes": null,
  "diagnosis_descriptions": null,
  "requires_enrichment": true
}
```

**Expected Behavior**:
- Trigger RAG enrichment
- Detection depends on enriched data quality
- Confidence scores should reflect uncertainty

---

## 9. Test Coverage Matrix

| Fraud Type | Obvious | Moderate | Subtle | Legitimate | Edge Cases |
|-----------|---------|----------|--------|------------|------------|
| Upcoding | 3 cases | 1 case | 1 case | 1 case | 1 case |
| Phantom Billing | 3 cases | 1 case | - | 1 case | - |
| Unbundling | 3 cases | - | - | 1 case | - |
| Staged Accident | 2 cases | - | - | 1 case | - |
| Prescription Fraud | 3 cases | - | - | 1 case | - |
| Kickback Scheme | 3 cases | - | - | 1 case | - |
| Multi-Pattern | 1 case | - | - | - | - |

**Total Test Cases**: 35+

## 10. Test Execution Guidelines

### 10.1 Test Data Loading

```python
from tests.fixtures.TEST_FIXTURES_SCHEMA import (
    UpcodingTestCase,
    PhantomBillingTestCase,
    generate_test_batch
)

# Load specific test case
upcoding_test = UpcodingTestCase.parse_file("TC-UP-001.json")

# Generate batch for testing
test_batch = generate_test_batch(
    batch_id="FRAUD_PATTERNS",
    total_claims=100,
    fraud_rate=0.15
)
```

### 10.2 Assertion Templates

```python
def assert_fraud_detected(result, expected):
    """Assert fraud detection meets expectations."""
    assert result.fraud_detected == expected.should_detect_fraud, \
        f"Expected fraud_detected={expected.should_detect_fraud}"

    assert expected.expected_score_range[0] <= result.fraud_score <= expected.expected_score_range[1], \
        f"Score {result.fraud_score} outside expected range {expected.expected_score_range}"

    if expected.expected_fraud_types:
        assert set(result.fraud_types).intersection(set(expected.expected_fraud_types)), \
            f"Expected fraud types {expected.expected_fraud_types}, got {result.fraud_types}"

    if expected.expected_rules_triggered:
        for rule in expected.expected_rules_triggered:
            assert rule in result.triggered_rules, \
                f"Expected rule '{rule}' to be triggered"
```

### 10.3 Performance Benchmarks

- **Per-test execution time**: <500ms
- **Batch processing time**: <5 seconds for 100 claims
- **Memory usage**: <500MB for full test suite

## 11. Test Maintenance

### 11.1 Adding New Test Cases

1. Define test case in this document
2. Create JSON test data file
3. Add Pydantic model if needed
4. Implement pytest test function
5. Update coverage matrix

### 11.2 Review Cycle

- **Monthly**: Review and update test cases
- **Quarterly**: Add new fraud patterns from real-world data
- **Annually**: Comprehensive test suite audit

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Test Coverage**: 35+ cases across 6 fraud types
