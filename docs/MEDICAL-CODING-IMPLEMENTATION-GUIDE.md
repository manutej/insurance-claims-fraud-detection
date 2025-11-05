# Medical Coding Implementation Guide

## Quick Start: Implementing Medical Code Validation

This guide provides step-by-step instructions to integrate medical coding validation into your insurance claims system.

---

## Table of Contents

1. [Installation](#installation)
2. [Basic ICD-10 Validation](#basic-icd-10-validation)
3. [CPT Code Validation](#cpt-code-validation)
4. [Diagnosis-Procedure Validation](#diagnosis-procedure-validation)
5. [Fraud Detection Integration](#fraud-detection-integration)
6. [Testing Strategy](#testing-strategy)

---

## Installation

### Step 1: Add Dependencies

```bash
# Install simple-icd-10-cm for diagnosis code validation
pip install simple-icd-10-cm

# Install requests for API calls (already in requirements)
pip install requests
```

### Step 2: Update requirements.txt

Add to `/Users/manu/Documents/CETI/CODING_PROJECTS/insurance_claims/requirements.txt`:

```
simple-icd-10-cm>=1.0.0
requests>=2.28.0
```

---

## Basic ICD-10 Validation

### Create New Module: `src/medical_coding/icd10_validator.py`

```python
"""
ICD-10-CM Code Validation Module

Provides functions to validate ICD-10-CM diagnosis codes against
the official CMS code set.
"""

import logging
from typing import List, Dict, Tuple, Optional
from icd10cm import ICD10CM

logger = logging.getLogger(__name__)


class ICD10Validator:
    """Validates ICD-10-CM diagnosis codes."""

    def __init__(self):
        """Initialize ICD10 validator with code reference."""
        try:
            self.icd = ICD10CM()
            logger.info("ICD-10-CM code reference loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ICD-10-CM reference: {e}")
            self.icd = None

    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate single ICD-10-CM code.

        Args:
            code: ICD-10-CM code to validate (e.g., "E11.9")

        Returns:
            (is_valid, description)
        """
        if not self.icd:
            return False, "ICD reference not available"

        try:
            if self.icd.exists(code):
                description = self.icd.get_description(code)
                return True, description
            return False, "Code not found in ICD-10-CM"
        except Exception as e:
            logger.error(f"Error validating code {code}: {e}")
            return False, f"Validation error: {str(e)}"

    def validate_codes_batch(self, codes: List[str]) -> Dict[str, Dict]:
        """
        Validate multiple ICD-10-CM codes.

        Args:
            codes: List of ICD-10-CM codes

        Returns:
            Dictionary mapping code -> {valid, description, parent}
        """
        results = {}
        for code in codes:
            is_valid, description = self.validate_code(code)
            results[code] = {
                'valid': is_valid,
                'description': description,
                'parent': self._get_parent(code) if is_valid else None
            }
        return results

    def get_code_hierarchy(self, code: str) -> Dict:
        """
        Get code hierarchy (parent codes).

        Args:
            code: ICD-10-CM code (e.g., "E11.9")

        Returns:
            Dictionary with code hierarchy information
        """
        if not self.icd or not self.icd.exists(code):
            return {'code': code, 'valid': False}

        hierarchy = {'code': code, 'valid': True}
        parents = []

        current = code
        while current:
            try:
                parent = self.icd.get_parent(current)
                if parent:
                    parents.append(parent)
                    current = parent
                else:
                    break
            except Exception:
                break

        hierarchy['parents'] = parents
        hierarchy['depth'] = len(parents)

        return hierarchy

    def _get_parent(self, code: str) -> Optional[str]:
        """Get parent code safely."""
        try:
            if self.icd:
                return self.icd.get_parent(code)
        except Exception:
            pass
        return None

    def get_description(self, code: str) -> Optional[str]:
        """Get code description."""
        is_valid, description = self.validate_code(code)
        return description if is_valid else None


# Module-level instance for convenience
_icd10_validator = None


def get_icd10_validator() -> ICD10Validator:
    """Get or create singleton ICD10 validator instance."""
    global _icd10_validator
    if _icd10_validator is None:
        _icd10_validator = ICD10Validator()
    return _icd10_validator


def validate_diagnosis_code(code: str) -> Tuple[bool, str]:
    """
    Convenience function to validate single diagnosis code.

    Args:
        code: ICD-10-CM code

    Returns:
        (is_valid, description_or_error)
    """
    validator = get_icd10_validator()
    return validator.validate_code(code)


def validate_diagnosis_codes(codes: List[str]) -> Dict[str, Dict]:
    """
    Convenience function to validate multiple diagnosis codes.

    Args:
        codes: List of ICD-10-CM codes

    Returns:
        Dictionary with validation results
    """
    validator = get_icd10_validator()
    return validator.validate_codes_batch(codes)
```

### Create Test File: `tests/unit/test_icd10_validator.py`

```python
"""Tests for ICD-10-CM validator."""

import pytest
from src.medical_coding.icd10_validator import (
    ICD10Validator,
    validate_diagnosis_code,
    validate_diagnosis_codes
)


class TestICD10Validator:
    """Test ICD-10-CM code validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ICD10Validator()

    def test_valid_code(self, validator):
        """Test validation of valid code."""
        is_valid, description = validator.validate_code('E11.9')
        assert is_valid is True
        assert 'Type 2 diabetes' in description

    def test_invalid_code(self, validator):
        """Test validation of invalid code."""
        is_valid, description = validator.validate_code('XXX99')
        assert is_valid is False

    def test_batch_validation(self, validator):
        """Test batch validation."""
        codes = ['E11.9', 'I10', 'J00', 'INVALID']
        results = validator.validate_codes_batch(codes)

        assert results['E11.9']['valid'] is True
        assert results['I10']['valid'] is True
        assert results['INVALID']['valid'] is False

    def test_code_hierarchy(self, validator):
        """Test getting code hierarchy."""
        hierarchy = validator.get_code_hierarchy('E11.9')
        assert hierarchy['valid'] is True
        assert 'parents' in hierarchy
        assert hierarchy['depth'] > 0

    def test_convenience_function(self):
        """Test module-level convenience function."""
        is_valid, description = validate_diagnosis_code('E11.9')
        assert is_valid is True

    def test_batch_convenience_function(self):
        """Test batch convenience function."""
        results = validate_diagnosis_codes(['E11.9', 'I10'])
        assert results['E11.9']['valid'] is True
        assert results['I10']['valid'] is True
```

---

## CPT Code Validation

### Create CPT Reference Module: `src/medical_coding/cpt_validator.py`

```python
"""
CPT Procedure Code Validation Module

Since no official free CPT API exists, this module maintains
a reference database loaded from authoritative sources.
"""

import json
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class CPTValidator:
    """Validates CPT procedure codes and modifiers."""

    # Common CPT codes and descriptions (minimal reference)
    # In production, load from CMS files or database
    COMMON_CPT_CODES = {
        # Office/Outpatient E/M Codes
        '99201': {'desc': 'Office visit, new patient, minimal', 'category': 'E/M', 'complexity': 1},
        '99202': {'desc': 'Office visit, new patient, low', 'category': 'E/M', 'complexity': 2},
        '99203': {'desc': 'Office visit, new patient, moderate', 'category': 'E/M', 'complexity': 3},
        '99204': {'desc': 'Office visit, new patient, high', 'category': 'E/M', 'complexity': 4},
        '99205': {'desc': 'Office visit, new patient, very high', 'category': 'E/M', 'complexity': 5},

        '99211': {'desc': 'Office visit, established patient, minimal', 'category': 'E/M', 'complexity': 1},
        '99212': {'desc': 'Office visit, established patient, low', 'category': 'E/M', 'complexity': 2},
        '99213': {'desc': 'Office visit, established patient, moderate', 'category': 'E/M', 'complexity': 3},
        '99214': {'desc': 'Office visit, established patient, high', 'category': 'E/M', 'complexity': 4},
        '99215': {'desc': 'Office visit, established patient, very high', 'category': 'E/M', 'complexity': 5},

        # Emergency Department E/M Codes
        '99281': {'desc': 'ED visit, self-limited', 'category': 'E/M', 'complexity': 1},
        '99282': {'desc': 'ED visit, minor', 'category': 'E/M', 'complexity': 2},
        '99283': {'desc': 'ED visit, moderate', 'category': 'E/M', 'complexity': 3},
        '99284': {'desc': 'ED visit, high', 'category': 'E/M', 'complexity': 4},
        '99285': {'desc': 'ED visit, very high', 'category': 'E/M', 'complexity': 5},

        # Lab Codes
        '80053': {'desc': 'Comprehensive metabolic panel', 'category': 'Lab', 'complexity': 2},
        '81000': {'desc': 'Urinalysis', 'category': 'Lab', 'complexity': 1},

        # Orthopedic Surgery
        '27447': {'desc': 'Total knee arthroplasty', 'category': 'Surgery', 'complexity': 5},
        '27410': {'desc': 'Knee arthroplasty component', 'category': 'Surgery', 'complexity': 4},

        # Vascular
        '36415': {'desc': 'Venipuncture', 'category': 'Procedure', 'complexity': 1},
        '36416': {'desc': 'Venipuncture, multi-draw', 'category': 'Procedure', 'complexity': 1},
    }

    # Valid CPT modifiers
    VALID_MODIFIERS = {
        '25': 'Significant, separately identifiable E/M service',
        '26': 'Professional component',
        '50': 'Bilateral procedure',
        '59': 'Distinct procedural service',
        'LT': 'Left side',
        'RT': 'Right side',
        'E1': 'Upper left eyelid',
        'E2': 'Lower left eyelid',
    }

    def __init__(self, custom_cpt_file: Optional[str] = None):
        """
        Initialize CPT validator.

        Args:
            custom_cpt_file: Optional path to custom CPT reference file
        """
        self.cpt_codes = self.COMMON_CPT_CODES.copy()

        if custom_cpt_file:
            self._load_custom_cpt_codes(custom_cpt_file)

        logger.info(f"CPT validator initialized with {len(self.cpt_codes)} codes")

    def validate_code(self, code: str) -> bool:
        """Check if CPT code is valid."""
        return code in self.cpt_codes

    def validate_codes_batch(self, codes: List[str]) -> Dict[str, Dict]:
        """Validate multiple CPT codes."""
        results = {}
        for code in codes:
            results[code] = {
                'valid': self.validate_code(code),
                'info': self.cpt_codes.get(code, {})
            }
        return results

    def get_code_info(self, code: str) -> Optional[Dict]:
        """Get information about a CPT code."""
        return self.cpt_codes.get(code)

    def validate_modifier(self, modifier: str) -> bool:
        """Check if modifier is valid."""
        return modifier in self.VALID_MODIFIERS

    def validate_modifiers_batch(self, modifiers: List[str]) -> Dict[str, bool]:
        """Validate multiple modifiers."""
        return {mod: self.validate_modifier(mod) for mod in modifiers}

    def get_modifier_description(self, modifier: str) -> Optional[str]:
        """Get modifier description."""
        return self.VALID_MODIFIERS.get(modifier)

    def _load_custom_cpt_codes(self, filepath: str):
        """Load custom CPT codes from file."""
        try:
            with open(filepath, 'r') as f:
                custom_codes = json.load(f)
                self.cpt_codes.update(custom_codes)
                logger.info(f"Loaded {len(custom_codes)} custom CPT codes")
        except Exception as e:
            logger.error(f"Failed to load custom CPT codes: {e}")


# Module-level instance
_cpt_validator = None


def get_cpt_validator() -> CPTValidator:
    """Get or create singleton CPT validator instance."""
    global _cpt_validator
    if _cpt_validator is None:
        _cpt_validator = CPTValidator()
    return _cpt_validator


def validate_procedure_code(code: str) -> bool:
    """Convenience function to validate single CPT code."""
    validator = get_cpt_validator()
    return validator.validate_code(code)


def validate_procedure_codes(codes: List[str]) -> Dict[str, Dict]:
    """Convenience function to validate multiple CPT codes."""
    validator = get_cpt_validator()
    return validator.validate_codes_batch(codes)
```

---

## Diagnosis-Procedure Validation

### Create Combination Validator: `src/medical_coding/combo_validator.py`

```python
"""
Diagnosis-Procedure Combination Validator

Validates that diagnosis and procedure codes are clinically
appropriate combinations.
"""

import logging
from typing import List, Dict, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class UpcodeRisk(Enum):
    """Upcoding risk levels."""
    NONE = 0.0
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8


class DiagnosisProcedureValidator:
    """Validates diagnosis-procedure combinations."""

    # Simple diagnoses that shouldn't have complex procedures
    ROUTINE_DIAGNOSES = {
        'Z00': 'Encounter for general adult medical examination',
        'Z01': 'Encounter for other special examination',
        'Z12': 'Encounter for screening for malignant neoplasm',
        'Z13': 'Encounter for screening for other diseases',
    }

    # Complex procedures
    HIGH_COMPLEXITY_PROCEDURES = {
        '99285', '99286', '99291', '99292',  # ER highest complexity
        '27447', '27450',  # Joint replacement
        '70450', '71260',  # Imaging
    }

    # Valid diagnosis-procedure combinations
    # Format: 'diagnosis_prefix': ['allowed_procedures']
    VALID_COMBINATIONS = {
        'Z00': ['99201', '99202', '99211', '99212'],  # Routine visit
        'Z01': ['99201', '99202', '99211', '99212'],
        'E11': ['99213', '99214', '80053', '81000'],  # Diabetes care
        'I10': ['99213', '99214', '85025', '36415'],  # Hypertension care
        'J00': ['99212', '99213'],  # Common cold - simple visit only
    }

    # NCCI Bundling rules (simplified)
    NCCI_BUNDLES = {
        ('99213', '99214'): "Cannot bill multiple E/M codes same day",
        ('99213', '99215'): "Cannot bill multiple E/M codes same day",
        ('27447', '27410'): "27410 is component of 27447",
        ('36415', '36416'): "36416 is add-on to 36415",
    }

    def validate_combination(self, diagnosis: str, procedure: str) -> Tuple[bool, str]:
        """
        Validate single diagnosis-procedure combination.

        Args:
            diagnosis: ICD-10-CM code
            procedure: CPT code

        Returns:
            (is_valid, message)
        """
        # Extract diagnosis prefix
        diag_prefix = diagnosis.split('.')[0] if '.' in diagnosis else diagnosis

        # Check if combination is in invalid list
        if not self._is_valid_combo(diag_prefix, procedure):
            return False, f"Invalid combination: {diagnosis} + {procedure}"

        return True, "Valid combination"

    def check_upcoding_risk(self, diagnosis: str, procedure: str) -> Tuple[float, str]:
        """
        Check for potential upcoding.

        Args:
            diagnosis: ICD-10-CM code
            procedure: CPT code

        Returns:
            (risk_score 0-1, explanation)
        """
        # Check if simple diagnosis with complex procedure
        if self._is_routine_diagnosis(diagnosis):
            if procedure in self.HIGH_COMPLEXITY_PROCEDURES:
                return 0.8, f"Simple diagnosis '{diagnosis}' billed with complex procedure '{procedure}'"

        return 0.0, "No upcoding risk detected"

    def check_unbundling_risk(self, procedures: List[str],
                             modifiers: List[str] = None) -> Tuple[float, List[str]]:
        """
        Check for potential unbundling.

        Args:
            procedures: List of CPT codes
            modifiers: List of CPT modifiers

        Returns:
            (risk_score 0-1, list of suspicious patterns)
        """
        modifiers = modifiers or []
        flags = []
        risk_score = 0.0

        # Check for known bundled pairs
        for i, proc1 in enumerate(procedures):
            for proc2 in procedures[i+1:]:
                pair = tuple(sorted([proc1, proc2]))
                if pair in self.NCCI_BUNDLES:
                    # Check if distinct procedural service modifier (59) is used
                    if '59' not in modifiers and 'LT' not in modifiers and 'RT' not in modifiers:
                        flags.append(self.NCCI_BUNDLES[pair])
                        risk_score += 0.3

        # Check for multiple E/M codes (cannot bill same day)
        em_codes = [p for p in procedures if p.startswith('99')]
        if len(em_codes) > 1:
            flags.append(f"Multiple E/M codes: {em_codes} - cannot bill same day")
            risk_score += 0.4

        # Check for excessive procedures
        if len(procedures) > 5:
            flags.append(f"Excessive procedures ({len(procedures)}) may indicate unbundling")
            risk_score += 0.2

        return min(risk_score, 1.0), flags

    def _is_routine_diagnosis(self, diagnosis: str) -> bool:
        """Check if diagnosis is routine (low severity)."""
        diag_prefix = diagnosis.split('.')[0] if '.' in diagnosis else diagnosis
        return diag_prefix in self.ROUTINE_DIAGNOSES

    def _is_valid_combo(self, diagnosis_prefix: str, procedure: str) -> bool:
        """Check if combination is in valid list."""
        if diagnosis_prefix not in self.VALID_COMBINATIONS:
            return True  # Unknown diagnosis, allow

        allowed = self.VALID_COMBINATIONS[diagnosis_prefix]
        return procedure in allowed

    def validate_claim_codes(self, diagnoses: List[str],
                            procedures: List[str]) -> Dict:
        """
        Comprehensive claim code validation.

        Args:
            diagnoses: List of ICD-10-CM codes
            procedures: List of CPT codes

        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'upcoding_risk': 0.0,
            'unbundling_risk': 0.0
        }

        # Check combinations
        for diagnosis in diagnoses:
            for procedure in procedures:
                is_valid, message = self.validate_combination(diagnosis, procedure)
                if not is_valid:
                    result['errors'].append(message)
                    result['valid'] = False

                # Check upcoding
                risk, explanation = self.check_upcoding_risk(diagnosis, procedure)
                if risk > 0.5:
                    result['warnings'].append(f"Upcoding risk: {explanation}")
                    result['upcoding_risk'] = max(result['upcoding_risk'], risk)

        # Check unbundling
        unbundling_risk, flags = self.check_unbundling_risk(procedures)
        if unbundling_risk > 0.3:
            result['warnings'].extend(flags)
            result['unbundling_risk'] = unbundling_risk

        return result
```

---

## Fraud Detection Integration

### Update Validator: `src/ingestion/validator.py`

Add to your existing validator:

```python
# Add imports at top
from src.medical_coding.icd10_validator import validate_diagnosis_codes
from src.medical_coding.cpt_validator import validate_procedure_codes
from src.medical_coding.combo_validator import DiagnosisProcedureValidator

class EnhancedClaimValidator(ClaimValidator):
    """Extended validator with medical coding checks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.combo_validator = DiagnosisProcedureValidator()

    def validate_medical_codes(self, claim: MedicalClaim) -> List[ValidationError]:
        """Validate ICD-10-CM and CPT codes."""
        errors = []

        # Validate diagnosis codes
        diag_results = validate_diagnosis_codes(claim.diagnosis_codes)
        for code, result in diag_results.items():
            if not result['valid']:
                errors.append(ValidationError(
                    field_name='diagnosis_codes',
                    error_message=f'Invalid diagnosis code: {code}',
                    claim_id=claim.claim_id,
                    severity='error'
                ))

        # Validate procedure codes
        proc_results = validate_procedure_codes(claim.procedure_codes)
        for code, result in proc_results.items():
            if not result['valid']:
                errors.append(ValidationError(
                    field_name='procedure_codes',
                    error_message=f'Unknown procedure code: {code}',
                    claim_id=claim.claim_id,
                    severity='warning'
                ))

        # Validate combinations
        combo_result = self.combo_validator.validate_claim_codes(
            claim.diagnosis_codes,
            claim.procedure_codes
        )

        for error in combo_result['errors']:
            errors.append(ValidationError(
                field_name='diagnosis_procedure_combo',
                error_message=error,
                claim_id=claim.claim_id,
                severity='error'
            ))

        for warning in combo_result['warnings']:
            errors.append(ValidationError(
                field_name='diagnosis_procedure_combo',
                error_message=warning,
                claim_id=claim.claim_id,
                severity='warning'
            ))

        return errors

    def _validate_medical_claim_rules(self, claim: MedicalClaim) -> List[ValidationError]:
        """Override to include medical coding validation."""
        errors = super()._validate_medical_claim_rules(claim)

        # Add medical code validation
        errors.extend(self.validate_medical_codes(claim))

        # Check upcoding specifically
        if self.combo_validator.combo_validator.check_upcoding_risk > 0.5:
            errors.append(ValidationError(
                field_name='fraud_risk',
                error_message='High upcoding risk detected',
                claim_id=claim.claim_id,
                severity='warning'
            ))

        return errors
```

---

## Testing Strategy

### Run Tests

```bash
# Test ICD-10 validation
pytest tests/unit/test_icd10_validator.py -v

# Test CPT validation
pytest tests/unit/test_cpt_validator.py -v

# Test combo validation
pytest tests/unit/test_combo_validator.py -v

# Test full validator
pytest tests/unit/test_validator.py -v
```

### Example Test Cases

```python
# Test Case 1: Valid medical claim
valid_claim = {
    'claim_id': 'CLM-2024-001234',
    'diagnosis_codes': ['E11.9', 'I10'],  # Valid codes
    'procedure_codes': ['99213', '80053'],  # Valid codes, valid combo
    'billed_amount': 150.00
}

# Test Case 2: Invalid diagnosis code
invalid_diag_claim = {
    'claim_id': 'CLM-2024-001235',
    'diagnosis_codes': ['XXX99'],  # Invalid
    'procedure_codes': ['99213'],
    'billed_amount': 150.00
}

# Test Case 3: Upcoding pattern
upcode_claim = {
    'claim_id': 'CLM-2024-001236',
    'diagnosis_codes': ['Z00'],  # Routine exam
    'procedure_codes': ['99285'],  # ER highest complexity
    'billed_amount': 500.00
}

# Test Case 4: Unbundling pattern
unbundle_claim = {
    'claim_id': 'CLM-2024-001237',
    'diagnosis_codes': ['E11.9'],
    'procedure_codes': ['99213', '99214', '99215'],  # Multiple E/M same day
    'billed_amount': 400.00
}
```

---

## Performance Considerations

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def validate_code_cached(code: str) -> bool:
    """Cache validation results."""
    validator = get_icd10_validator()
    is_valid, _ = validator.validate_code(code)
    return is_valid
```

### Batch Processing

```python
def validate_claim_batch(claims: List[Dict]) -> List[Dict]:
    """Validate multiple claims efficiently."""
    results = []
    validator = EnhancedClaimValidator()

    for claim_data in claims:
        try:
            claim = claim_factory(claim_data)
            errors = validator.validate_medical_codes(claim)
            results.append({
                'claim_id': claim.claim_id,
                'valid': len(errors) == 0,
                'errors': errors
            })
        except Exception as e:
            logger.error(f"Validation failed: {e}")

    return results
```

---

## Next Steps

1. **Install dependencies**
2. **Create medical coding modules**
3. **Write and run tests**
4. **Integrate with validator.py**
5. **Create monitoring dashboard**
6. **Document custom CPT codes**

---

**Version:** 1.0
**Status:** Ready for implementation
**Last Updated:** 2025-10-28

