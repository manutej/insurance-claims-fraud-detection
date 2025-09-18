# Insurance Claims Fraud Detection - Sample Data

## Overview
This directory contains sample healthcare insurance claims data for training and testing fraud detection algorithms. The data is based on real fraud patterns identified in research documents from NY State Department of Financial Services, academic papers, and industry reports.

## Directory Structure

```
data/
├── valid_claims/       # Legitimate insurance claims
│   ├── medical_claims.json    # Regular medical services
│   ├── pharmacy_claims.json   # Prescription drug claims
│   └── no_fault_claims.json   # Auto accident medical claims
├── fraudulent_claims/  # Various types of fraudulent claims
│   ├── upcoding_fraud.json    # Services billed at higher complexity
│   ├── phantom_billing.json   # Services never rendered
│   ├── unbundling_fraud.json  # Single procedures split into multiple
│   ├── staged_accidents.json  # Fabricated auto accidents
│   ├── prescription_fraud.json # Drug diversion schemes
│   └── kickback_schemes.json  # Referral kickback fraud
└── raw/
    └── mixed_claims.json       # Mixed dataset for testing
```

## Fraud Types and Patterns

### 1. Upcoding (8-15% of claims)
- Billing for more expensive procedures than performed
- Common pattern: Simple diagnosis (common cold) billed at highest complexity
- Red flags: 90%+ of claims at maximum complexity level

### 2. Phantom Billing (3-10% of claims)
- Services billed but never rendered
- Ghost patients with sequential SSNs
- Services on days when facility closed
- Billing for deceased patients

### 3. Unbundling (5-10% of claims)
- Single procedure billed as multiple components
- Total exceeds bundled rate by 200-300%
- Add-on codes billed as primary procedures

### 4. Staged Accidents (No-Fault)
- Multiple passengers with identical injuries
- Same medical clinic and attorney
- Treatment begins before police report
- Damage inconsistent with injuries

### 5. Prescription Diversion
- Multiple prescribers in 30 days (doctor shopping)
- Traveling >50 miles between pharmacies
- Early refill attempts
- Controlled substances only

### 6. Kickback Schemes
- 95%+ referrals to single facility
- Hidden financial relationships
- Unnecessary procedures ordered
- Self-referral to owned facilities

## Data Fields

### Medical Claims
- `claim_id`: Unique identifier
- `patient_id`: Patient identifier
- `provider_id`: Provider identifier
- `provider_npi`: National Provider Identifier
- `date_of_service`: Service date
- `diagnosis_codes`: ICD-10 diagnosis codes
- `procedure_codes`: CPT procedure codes
- `billed_amount`: Amount billed
- `fraud_indicator`: Boolean flag
- `fraud_type`: Type of fraud (if applicable)
- `red_flags`: List of suspicious indicators

### Pharmacy Claims
- `ndc_code`: National Drug Code
- `drug_name`: Medication name
- `quantity`: Pills/units dispensed
- `days_supply`: Duration of prescription
- `refill_number`: Refill count

### No-Fault Claims
- `accident_date`: Date of accident
- `police_report_number`: Police report reference
- `vehicle_info`: Vehicle and insurance details

## Usage

### Loading Data (Python)
```python
import json

# Load valid claims
with open('data/valid_claims/medical_claims.json', 'r') as f:
    valid_medical = json.load(f)

# Load fraudulent claims
with open('data/fraudulent_claims/upcoding_fraud.json', 'r') as f:
    fraud_upcoding = json.load(f)

# Load mixed test set
with open('data/raw/mixed_claims.json', 'r') as f:
    test_data = json.load(f)
```

## Statistics

- **Total Sample Claims**: ~35 detailed examples
- **Fraud Rate in Mixed Set**: 50% (for balanced testing)
- **Real-World Fraud Rate**: 8-15% (per research)
- **Potential Savings**: $3B+ annually (scaled from NY data)

## Common ICD-10 Codes Used

- **E11.9**: Type 2 diabetes mellitus
- **I10**: Essential hypertension
- **J18.9**: Pneumonia
- **M54.5**: Low back pain
- **N39.0**: Urinary tract infection
- **S13.4XXA**: Sprain of cervical spine

## Common CPT Codes Used

- **99213-99215**: Office visits (various complexity)
- **97110**: Therapeutic exercises
- **97140**: Manual therapy
- **70450**: CT head
- **80053**: Comprehensive metabolic panel

## Detection Targets

Based on research, the fraud detection system should aim for:
- **Accuracy**: >94%
- **False Positive Rate**: <3.8%
- **Processing Time**: <4 hours per batch
- **Detection Rate**: Identify 8-15% of claims as suspicious

## References

- NY State Department of Financial Services Healthcare Fraud Reports
- FBI Healthcare Fraud Statistics
- Academic research on insurance fraud patterns
- Industry reports on fraud detection methods

## Note

This is synthetic data created for testing purposes. All provider NPIs, patient IDs, and claim IDs are fictitious. Real fraud patterns are based on documented schemes from official sources.