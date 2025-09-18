# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Insurance Claims Fraud Detection dataset and analysis project focused on healthcare insurance claims. The project contains sample data representing various types of legitimate and fraudulent insurance claims based on real fraud patterns identified by NY State Department of Financial Services and industry research.

## Project Structure

```
insurance_claims/
├── .claude/                # Claude configuration
│   ├── agents/             # Specialized agent configurations
│   │   ├── enterprise-architect.md      # Requirements engineering agent
│   │   └── software-testing-manager.md  # Test management agent
│   └── settings.local.json
├── data/                   # Insurance claims datasets
│   ├── valid_claims/       # Legitimate claims samples
│   ├── fraudulent_claims/  # Fraud pattern samples
│   └── raw/               # Mixed test datasets
└── docs/                   # Research documentation (PDFs)
```

## Data Architecture

### Core Data Schema

Claims follow a structured JSON format with these key fields:
- `claim_id`: Unique identifier
- `patient_id`: Patient identifier
- `provider_npi`: National Provider Identifier
- `diagnosis_codes`: ICD-10 diagnosis codes
- `procedure_codes`: CPT procedure codes
- `billed_amount`: Claimed amount
- `fraud_indicator`: Boolean fraud flag
- `fraud_type`: Type of fraud (if applicable)
- `red_flags`: List of suspicious indicators

### Fraud Patterns in Dataset

1. **Upcoding** (8-15% of claims) - Services billed at higher complexity than performed
2. **Phantom Billing** (3-10%) - Services billed but never rendered
3. **Unbundling** (5-10%) - Single procedures split into multiple claims
4. **Staged Accidents** - Fabricated auto accidents with consistent patterns
5. **Prescription Fraud** - Drug diversion and doctor shopping
6. **Kickback Schemes** - Hidden financial relationships and unnecessary referrals

## Working with the Data

### Loading Claims Data

Since this is a data-only project currently, use standard JSON parsing:

```python
import json

# Load specific claim types
with open('data/valid_claims/medical_claims.json', 'r') as f:
    valid_claims = json.load(f)

with open('data/fraudulent_claims/upcoding_fraud.json', 'r') as f:
    fraud_claims = json.load(f)

# Load mixed dataset for testing
with open('data/raw/mixed_claims.json', 'r') as f:
    test_data = json.load(f)
```

## Key Detection Metrics

When developing fraud detection algorithms, target these benchmarks:
- **Accuracy**: >94%
- **False Positive Rate**: <3.8%
- **Detection Rate**: 8-15% of claims flagged as suspicious
- **Processing Time**: <4 hours per batch

## Available Agents

The project includes two specialized Claude agents:

1. **enterprise-architect**: Transforms use cases into detailed business requirements
2. **software-testing-manager**: Develops test strategies and manages quality assurance

Use these agents when:
- Converting fraud detection use cases into formal requirements
- Planning testing strategies for fraud detection systems
- Analyzing test results and prioritizing bug fixes

## Important Considerations

- All data is synthetic but based on documented real-world fraud patterns
- Provider NPIs and patient IDs are fictitious
- Fraud rates in mixed dataset (50%) are higher than real-world (8-15%) for balanced testing
- Research documents in `/docs` provide context on actual fraud schemes and detection methods