# Medical Coding Reference

Complete reference for ICD-10, CPT codes, and fraud detection patterns.

---

## ICD-10-CM Code Reference

### Common Diagnosis Codes by Category

#### Diabetes (E10-E14)

| Code | Description | Severity | Red Flags |
|------|-------------|----------|-----------|
| E10.9 | Type 1 diabetes without complications | Low | None |
| E11.9 | Type 2 diabetes without complications | Low | None |
| E11.21 | Type 2 diabetes with neuropathy | Medium | Requires neurologist referral |
| E11.22 | Type 2 diabetes with diabetic kidney disease | High | Requires nephrologist, specialized tests |
| E13.9 | Other specified diabetes | Low | Requires clarification |

**Valid Procedures for E11.9:**
- 99213, 99214 (Office visit)
- 80053 (Metabolic panel)
- 81000 (Urinalysis)

**Invalid Procedures for E11.9:**
- 99285, 99286 (ER emergency codes)
- 27447 (Knee replacement)
- 70450 (CT scan - unless diabetic complication)

#### Hypertension (I10-I16)

| Code | Description | RVU Value | Typical Procedures |
|------|-------------|-----------|-------------------|
| I10 | Essential hypertension | 0.00 | 99213, 36415 |
| I11.9 | HTN with stage 5 CKD | 0.00 | 99214, 80053, 36415 |
| I12.9 | HTN with CKD stage 5 | 0.00 | 99214, 80053 |

#### Common Cold (J00-J06)

| Code | Description | Complexity | Max Procedures |
|------|-------------|-----------|-----------------|
| J00 | Acute nasopharyngitis | Minimal | 1-2 |
| J01.0 | Acute sinusitis | Low | 1-2 |
| J02.0 | Strep throat | Low | 1-2 |

**Important:** These should NEVER be billed with complex procedures or high ER codes

#### Routine Visits (Z00-Z13)

| Code | Description | Allowed Procedures |
|------|-------------|-------------------|
| Z00.00 | Encounter for general adult medical exam | 99201-99202, 99211-99212 |
| Z01.00 | Encounter for vision exam | 92004, 92012 |
| Z12.11 | Encounter for screening for colon cancer | 45378 (colonoscopy) |
| Z13.9 | Encounter for screening | 99201-99202 |

**Fraud Warning:** These with ED codes (99285-99286) = potential upcoding

---

## CPT Code Reference

### Office Visit/E&M Codes (99201-99215)

#### New Patient Codes

| Code | Complexity | Time (min) | Risk Level |
|------|-----------|-----------|-----------|
| 99201 | Minimal | 10 | Low - overuse common |
| 99202 | Low | 20 | Low |
| 99203 | Moderate | 30 | Medium |
| 99204 | High | 40 | Medium |
| 99205 | Very High | 50+ | High - overuse common |

#### Established Patient Codes

| Code | Complexity | Time (min) | Fraud Risk |
|------|-----------|-----------|-----------|
| 99211 | Minimal | 5 | High - nurse visit often billed as MD |
| 99212 | Low | 10 | Medium |
| 99213 | Moderate | 20 | Low - most appropriate |
| 99214 | High | 30 | Medium - overuse common |
| 99215 | Very High | 40 | High - overuse common |

**Upcoding Alert:** Provider billing 99215 >50% of time (vs. specialty average 15-20%) = fraud red flag

### Emergency Department Codes (99281-99285)

| Code | Severity | Typical Diagnosis |
|------|----------|------------------|
| 99281 | Self-limited/minor | Laceration, minor contusion |
| 99282 | Minor | Earache, sore throat |
| 99283 | Moderate | Cough, minor trauma |
| 99284 | High | Chest pain, moderate trauma |
| 99285 | Very High | Trauma, breathing difficulty |

**Fraud Alert:** Non-ER diagnosis (Z00, J00) with 99284-99285 = likely upcoding

### Laboratory Codes

| Code | Description | Typical Cost | Frequency |
|------|-------------|--------------|-----------|
| 80053 | Comprehensive metabolic panel | $50 | Annual |
| 80055 | Obstetric panel | $75 | Once per pregnancy |
| 81000 | Urinalysis | $15 | PRN or annual |
| 85025 | Complete blood count | $25 | Annual |
| 36415 | Venipuncture (single) | $3 | Per draw |

### Surgery Codes

| Code | Description | Complexity | Component Codes |
|------|-------------|-----------|-----------------|
| 27447 | Total knee arthroplasty | 5 | 27410, 27411, 27412 |
| 27450 | Knee arthroplasty revision | 5 | 27445, 27446 |
| 71101 | Chest X-ray, 2 views | 2 | 71100, 71020 |

**Unbundling Alert:** Billing 27447 + 27410 same day without modifier 59 = fraud

---

## CPT Modifiers Reference

### Common Modifiers

| Modifier | Meaning | Use Case |
|----------|---------|----------|
| 25 | Significant, separately identifiable E/M | E/M + procedure same day |
| 26 | Professional component | Professional only (no facility) |
| 50 | Bilateral procedure | Both sides performed |
| 59 | Distinct procedural service | Multiple related procedures |
| 76 | Repeat procedure same day | Same code performed twice |
| 77 | Repeat procedure different provider | Same code, different provider |
| 91 | Repeat clinical diagnostic laboratory | Same lab test repeated |
| LT | Left side | Unilateral left |
| RT | Right side | Unilateral right |
| E1 | Upper left eyelid | Specific eye codes |
| E2 | Lower left eyelid | Specific eye codes |

**Invalid Use:** Modifier 59 without clear clinical reason = fraud indicator

---

## Place of Service Codes

| Code | Location | Examples | Frequency Alert |
|------|----------|----------|-----------------|
| 11 | Office | MD office, clinic | Normal |
| 12 | Home | House visit | Monitor provider |
| 21 | Inpatient hospital | Admits, hospital stay | Monitor IMR |
| 22 | Outpatient hospital | Hospital OP | Normal |
| 23 | Emergency department | ER, trauma | High fraud risk |
| 31 | Skilled nursing facility | SNF, nursing home | Monitor |
| 41 | Ambulance (Land) | EMS transport | Monitor cost |
| 42 | Ambulance (Air) | Helicopter EMS | Verify necessity |
| 51 | Inpatient psychiatric | Psychiatric hospital | Monitor |
| 81 | Durable medical equipment | DME supplier | Monitor fraud |

**Temporal Fraud:** Professional service (Code 11) on Sunday when office closed = phantom billing

---

## NCCI Bundling Rules (National Correct Coding Initiative)

### Mutually Exclusive Pairs (Cannot Bill Together)

| Procedure 1 | Procedure 2 | Reason |
|-----------|-----------|--------|
| 99213 | 99214 | Cannot bill two office visit codes same day |
| 99213 | 99215 | Cannot bill two office visit codes same day |
| 27447 | 27410 | Component procedure cannot be billed with main procedure |
| 27447 | 27411 | Component procedure cannot be billed with main procedure |
| 36415 | 36416 | Add-on code, only with parent code |

**Exception:** Modifier 59 (distinct procedural service) may allow, but requires documentation

### Add-On Codes (Parent Code Required)

| Add-On Code | Required Parent | Restriction |
|-----------|----------------|------------|
| 27411 | 27447 | Cannot bill alone |
| 36416 | 36415 | Cannot bill without venipuncture |
| 99292 | 99291 | Must be second or subsequent hour of ICU |

---

## Fraud Detection Patterns

### Pattern 1: Classic Upcoding

**Indicators:**
- Simple diagnosis (Z00-Z13) + high complexity procedure (99215, 99285)
- Routine complaint + ER visit codes
- Office visit + imaging/specialty procedures

**Example Claims:**
```json
{
  "claim_id": "CLM-FRAUD-001",
  "diagnosis_codes": ["Z00"],           // Routine physical
  "procedure_codes": ["99285", "70450"], // ER code + CT scan
  "billed_amount": 1500.00,              // Very high for routine
  "red_flags": [
    "Simple diagnosis with emergency codes",
    "Billed amount significantly above normal"
  ]
}
```

**Detection Rules:**
```python
UPCODING_TRIGGERS = {
    ('Z00', '99285'): 0.9,  # Risk score
    ('Z00', '99286'): 0.9,
    ('Z01', '99285'): 0.8,
    ('J00', '99284'): 0.7,
    ('J00', '99285'): 0.9
}
```

### Pattern 2: Unbundling

**Indicators:**
- Multiple E/M codes same day (99213 + 99214 + 99215)
- Component procedures without parent code
- Related procedures billed separately

**Example Claims:**
```json
{
  "claim_id": "CLM-FRAUD-002",
  "diagnosis_codes": ["E11.9"],
  "procedure_codes": ["99213", "99214", "99215", "80053"],
  "billed_amount": 850.00,
  "red_flags": [
    "3 E/M codes same day - violates NCCI",
    "Unusual procedure combination"
  ]
}
```

**Detection Rules:**
```python
# Cannot bill multiple E/M codes same day
if sum(1 for p in procedures if p.startswith('99')) > 1:
    risk += 0.6

# Component codes require parent
if '27410' in procedures and '27447' not in procedures:
    risk += 0.8
```

### Pattern 3: Phantom Billing

**Indicators:**
- Service on weekend/holiday
- Service when office closed
- Identical charges for multiple patients
- Patient denies service

**Example Claims:**
```json
{
  "claim_id": "CLM-FRAUD-003",
  "date_of_service": "2024-03-17",  // Sunday
  "procedure_codes": ["99213"],
  "billed_amount": 100.00,
  "red_flags": [
    "Professional service on Sunday",
    "Office typically closed on weekends"
  ]
}
```

**Detection Rules:**
```python
# Check if office closed
if date_of_service.weekday() == 6:  # Sunday
    if claim_type == 'professional':
        risk += 0.7

# Check for identical charges pattern
if duplicate_amount_count > 5:
    risk += 0.6
```

### Pattern 4: Prescription Fraud (Controlled Substances)

**Indicators:**
- Early refills (refill before days supply exhausted)
- Multiple prescribers (doctor shopping)
- Multiple pharmacies (pharmacy hopping)
- High-dose opioids without cancer diagnosis

**Detection Rules:**
```python
# Early refill check
days_since_last_fill = (fill_date - last_fill_date).days
if days_since_last_fill < (days_supply - 7):
    risk += 0.5

# Doctor shopping
prescribers_30_days = count_unique_prescribers(last_30_days)
if prescribers_30_days > 5:
    risk += 0.8

# Pharmacy hopping
pharmacies_30_days = count_unique_pharmacies(last_30_days)
if pharmacies_30_days > 3 and controlled_substance:
    risk += 0.7

# High-dose opioid without cancer
if daily_dose_mme > 90 and not has_cancer_diagnosis:
    risk += 0.6
```

---

## Fraud Risk Thresholds

### Scoring System

```python
FRAUD_RISK_LEVELS = {
    0.0 - 0.2: "Low Risk",        # Typical claim
    0.2 - 0.4: "Monitor",         # Investigate if pattern emerges
    0.4 - 0.6: "Medium Risk",     # Review before payment
    0.6 - 0.8: "High Risk",       # Hold for investigation
    0.8 - 1.0: "Critical Risk"    # Block and escalate
}
```

### Red Flag Weights

| Red Flag | Weight | Fraud Type |
|----------|--------|-----------|
| Simple diagnosis + complex procedure | 0.30 | Upcoding |
| Multiple E/M same day | 0.30 | Unbundling |
| Service on weekend | 0.20 | Phantom |
| Component without parent | 0.25 | Unbundling |
| Early refill (controlled) | 0.25 | Prescription |
| Doctor shopping (5+ prescribers) | 0.35 | Prescription |
| Invalid diagnosis-procedure combo | 0.25 | Misc |
| Excessive procedures (>5) | 0.20 | Unbundling |

---

## Valid Diagnosis-Procedure Combinations

### Diabetes Management

```
E11.9 (Type 2 diabetes):
  Valid:
    - 99213, 99214 (Office visit)
    - 80053 (Metabolic panel)
    - 81000 (Urinalysis)
    - 36415 (Venipuncture)

  Invalid:
    - 99285, 99286 (ER codes)
    - 27447 (Knee replacement)
    - 70450 (CT scan)
    - 93000 (EKG - unless diabetic neuropathy)

Risk Levels:
  High Risk: Z00 + 99215
  Medium Risk: Z00 + 99214
  Low Risk: E11.9 + 99213
```

### Hypertension Management

```
I10 (Essential hypertension):
  Valid:
    - 99213, 99214 (Office visit)
    - 36415 (Blood pressure check)
    - 80053 (Metabolic panel)

  Invalid:
    - 99285, 99286 (ER codes for routine HTN)
    - 27447 (Orthopedic procedures)
    - 70450 (CT imaging without specific indication)

Risk Levels:
  High Risk: I10 + 99285
  Medium Risk: I10 + 99215
  Low Risk: I10 + 99213 + 36415
```

### Respiratory Conditions

```
J00 (Common cold):
  Valid:
    - 99211, 99212 (Minimal complexity)
    - 81000 (Urinalysis if fever)

  Invalid:
    - 99284, 99285 (ER high complexity)
    - 70450 (CT imaging)
    - 93000 (EKG)

Risk Levels:
  Critical Risk: J00 + 99285 + 70450
  High Risk: J00 + 99284
  Medium Risk: J00 + 99213
  Low Risk: J00 + 99212
```

---

## Provider Peer Benchmarks

### Average Billing Patterns (Established Patient Office Visits)

| Specialty | 99211 % | 99212 % | 99213 % | 99214 % | 99215 % |
|-----------|---------|---------|---------|---------|---------|
| Family Medicine | 5% | 15% | 60% | 15% | 5% |
| Internal Medicine | 3% | 10% | 65% | 18% | 4% |
| Pediatrics | 8% | 20% | 55% | 12% | 5% |
| Emergency Medicine | 2% | 8% | 25% | 35% | 30% |

**Fraud Detection:** Provider billing >40% 99215 when specialty average is 5% = HIGH RISK

### Average Claim Amounts by Service

| Service Type | Low | Normal | High | Critical |
|-------------|-----|--------|------|----------|
| Office visit (99213) | $50 | $75-100 | $125 | >$200 |
| ER visit (99285) | $200 | $300-400 | $600 | >$800 |
| Total knee replacement | $25k | $40-50k | $75k | >$100k |
| Diabetes follow-up | $75 | $100-150 | $200 | >$250 |

**Fraud Detection:** 99213 billed at $200 when normal is $100 = investigate

---

## CMS Reference URLs

### Code Lookups
- **ICD-10-CM Search:** https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search
- **CMS Code Files:** https://www.cms.gov/medicare/coding-billing/icd-10-codes
- **ICD-10 API:** https://www.icd10api.com/

### Official Guidelines
- **FY 2026 ICD-10-CM Guidelines:** https://www.cms.gov/files/document/fy-2026-icd-10-cm-official-guidelines.pdf
- **NCCI Edits:** https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-edits
- **Medicare Fee Schedule:** https://www.cms.gov/apps/physician-fee-schedule/

### Fraud Resources
- **OIG Fraud Alerts:** https://oig.hhs.gov/
- **Coding Clinic:** https://www.ahacoding.org/
- **MAC Guidance:** Contact your regional MAC

---

## Quick Reference Checklist

### For Each Claim, Verify:

- [ ] Diagnosis codes valid (ICD-10-CM format)
- [ ] Procedure codes valid (CPT format)
- [ ] Diagnosis-procedure combination valid
- [ ] No multiple E/M codes same day (unless modifier 25)
- [ ] No component codes without parent
- [ ] Service date not on weekend (for professional)
- [ ] Service date within policy effective dates
- [ ] Billed amount within normal range for procedure
- [ ] Provider not on OIG exclusion list
- [ ] Modifiers used appropriately

### Red Flags Checklist:

- [ ] Simple diagnosis + complex procedure = UPCODING
- [ ] Multiple E/M codes same day = UNBUNDLING
- [ ] Service on weekend/holiday = PHANTOM
- [ ] Early refills on controlled drugs = PRESCRIPTION FRAUD
- [ ] Doctor shopping (5+ prescribers/30 days) = FRAUD
- [ ] Unusual procedure combinations = UNBUNDLING
- [ ] Billed amount significantly above average = UPCODING/FRAUD
- [ ] Component without parent procedure = UNBUNDLING

---

**Version:** 1.0
**Last Updated:** 2025-10-28
**Status:** Reference Ready

