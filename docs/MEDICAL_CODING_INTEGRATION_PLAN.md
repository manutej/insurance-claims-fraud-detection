# MEDICAL CODING INTEGRATION PLAN

**Project:** Insurance Fraud Detection System Enhancement
**Document Version:** 1.0
**Date:** October 28, 2025
**Status:** Implementation Plan
**Priority:** P0 - CRITICAL

---

## EXECUTIVE SUMMARY

This plan details the integration of comprehensive medical coding validation into the fraud detection system. Medical coding validation is a **critical blocking issue** preventing production deployment. Current system effectiveness is only **34%** primarily due to the lack of medical domain knowledge encoded in ICD-10, CPT, NCCI, and MUE standards.

**Project Goals:**
- Integrate ICD-10 diagnosis code validation
- Integrate CPT procedure code validation
- Implement NCCI (National Correct Coding Initiative) edit checking
- Implement MUE (Medically Unlikely Edits) validation
- Integrate CMS fee schedules and RVU data
- Establish medical necessity validation

**Expected Impact:**
- Unbundling detection: 15% → 90%+ effectiveness
- Upcoding detection: 30% → 70%+ effectiveness
- Overall system effectiveness: 34% → 70%+
- Production-ready in 11-16 weeks

**Timeline:** 6 weeks (critical path)
**Team Required:** 2-3 developers, 1 medical coding consultant
**Budget:** $50K-75K (data acquisition + personnel)

---

## 1. PROJECT SCOPE

### 1.1 In Scope

**Medical Code Validation:**
- [x] ICD-10 diagnosis code structure and existence validation
- [x] CPT procedure code structure and existence validation
- [x] Medical code hierarchy (chapter, category, family)
- [x] Code validity checking (active vs. obsolete)
- [x] Age/gender-specific validation
- [x] Laterality and specificity validation

**Industry Standard Rules:**
- [x] NCCI edit checking (500,000+ code pairs)
- [x] MUE (Medically Unlikely Edits) limits
- [x] Mutually exclusive code detection
- [x] Add-on code validation
- [x] Modifier validation (25, 59, etc.)

**Clinical Validation:**
- [x] Diagnosis-procedure relationship validation
- [x] Medical necessity checking (basic)
- [x] Expected diagnosis patterns
- [x] Specialty-appropriate procedures

**Fee Schedule Integration:**
- [x] Medicare Physician Fee Schedule (MPFS)
- [x] RVU (Relative Value Unit) data
- [x] Geographic adjustment factors (GPCI)
- [x] Facility vs. non-facility pricing

### 1.2 Out of Scope (Future Phases)

- Natural language processing of medical notes
- Image/radiology report analysis
- Real-time claims adjudication
- Payer-specific edits beyond Medicare
- International code systems (ICD-11, other countries)
- Clinical pathways and treatment protocols (detailed)

### 1.3 Assumptions

1. CMS data is publicly available or can be licensed
2. Existing MEDICAL_CODE_MAPPING.json file is accurate
3. System will use PostgreSQL or similar for code tables
4. Redis or similar for caching frequently accessed codes
5. Quarterly updates can be automated
6. Medical coding consultant available for validation

---

## 2. CURRENT STATE ANALYSIS

### 2.1 Available Resources

**Existing Data Files:**
```
data/MEDICAL_CODE_MAPPING.json (33 KB)
└── Contains: ICD-10 codes, CPT codes, bundling info
    Status: NOT INTEGRATED
    Quality: Unknown, needs validation
```

**Existing Code Modules:**
```
src/detection/rule_engine.py
├── Hardcoded complexity scoring (6 CPT codes)
├── Hardcoded bundling rules (4 groups)
└── Basic fee schedule (6 codes + $100 default)

src/detection/feature_engineering.py
├── Placeholder procedure complexity function
├── Placeholder diagnosis severity function
└── No medical code hierarchy features
```

### 2.2 Current Medical Coding Usage

**Where Medical Codes Are Used:**

1. **rule_engine.py:**
   - Line 236-270: Upcoding detection (hardcoded)
   - Line 361-366: Unbundling detection (4 bundles)
   - Line 723-738: Fee schedule (6 codes)

2. **feature_engineering.py:**
   - Line 139: Procedure complexity (hardcoded 8 codes)
   - Line 140: Diagnosis severity (hardcoded patterns)

3. **No Usage Of:**
   - Code existence validation
   - Code hierarchy
   - NCCI edits
   - MUE limits
   - Medical necessity rules

---

## 3. REQUIRED DATA SOURCES

### 3.1 ICD-10 Code Database

**Source:** CMS ICD-10-CM (Clinical Modification)
**URL:** https://www.cms.gov/medicare/coding-billing/icd-10-codes
**Update Frequency:** Annual (October 1)
**Data Size:** ~72,000 codes

**Required Data Elements:**
```json
{
  "code": "E11.9",
  "description": "Type 2 diabetes mellitus without complications",
  "chapter": "04",
  "chapter_name": "Endocrine, nutritional and metabolic diseases",
  "category": "E11",
  "valid_from": "2015-10-01",
  "valid_to": null,
  "age_restrictions": null,
  "gender_restrictions": null,
  "requires_laterality": false,
  "manifestation_code": false,
  "etiology_code": false
}
```

**Acquisition Plan:**
1. Download ICD-10-CM files from CMS (Week 1)
2. Parse and structure data (Week 1)
3. Load into PostgreSQL database (Week 1)
4. Create indexes for fast lookup (Week 1)
5. Validate against existing claims (Week 2)

**Cost:** Free (public domain)

---

### 3.2 CPT Code Database

**Source:** American Medical Association (AMA)
**License Required:** YES ($$$)
**Update Frequency:** Annual (January 1)
**Data Size:** ~10,000 codes

**Required Data Elements:**
```json
{
  "code": "99214",
  "description": "Office visit E&M, established patient, 30-39 min",
  "category": "Evaluation & Management",
  "family": "Office Visits",
  "level": 4,
  "typical_time_minutes": 30,
  "work_rvu": 1.50,
  "facility_rvu": 1.80,
  "modifiers_allowed": ["25", "59"],
  "global_days": 0,
  "status": "active",
  "effective_date": "2023-01-01"
}
```

**Acquisition Plan:**
1. License CPT database from AMA (Week 1) **[BLOCKING]**
2. Parse and structure data (Week 1)
3. Load into PostgreSQL database (Week 1)
4. Create code family hierarchies (Week 2)
5. Validate against existing claims (Week 2)

**Cost:** ~$10K-$15K annual license

**Alternative:** Use existing MEDICAL_CODE_MAPPING.json for MVP, acquire full database later

---

### 3.3 NCCI Edit Table

**Source:** CMS National Correct Coding Initiative
**URL:** https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci
**Update Frequency:** Quarterly
**Data Size:** ~500,000 code pair edits

**Required Data Elements:**
```json
{
  "column1_code": "43235",
  "column2_code": "43239",
  "modifier_indicator": "1",
  "effective_date": "2024-01-01",
  "deletion_date": null,
  "rationale": "Component of comprehensive code"
}
```

**Modifier Indicators:**
- **0:** Never bundle separately (always bundled)
- **1:** May bill separately with appropriate modifier
- **9:** Not applicable (deleted or special)

**Acquisition Plan:**
1. Download NCCI edit files from CMS (Week 2)
2. Parse quarterly update files (Week 2)
3. Load into PostgreSQL with indexes (Week 2)
4. Implement lookup optimization (Week 3)
5. Test with known violations (Week 3)

**Cost:** Free (public domain)

---

### 3.4 MUE (Medically Unlikely Edits) Table

**Source:** CMS Medically Unlikely Edits
**URL:** https://www.cms.gov/medicare/coding-billing/mue
**Update Frequency:** Quarterly
**Data Size:** ~10,000 entries

**Required Data Elements:**
```json
{
  "cpt_code": "36415",
  "mue_value": 1,
  "mue_adjudication_indicator": "2",
  "rationale": "Anatomic consideration (single venipuncture)",
  "effective_date": "2024-01-01"
}
```

**MUE Adjudication Indicators:**
- **1:** Per date of service (clinical)
- **2:** Per date of service (anatomic)
- **3:** Per date of service (policy)

**Acquisition Plan:**
1. Download MUE files from CMS (Week 2)
2. Parse and structure data (Week 2)
3. Load into PostgreSQL (Week 2)
4. Implement validation logic (Week 3)
5. Test with known violations (Week 3)

**Cost:** Free (public domain)

---

### 3.5 Medicare Fee Schedule (MPFS)

**Source:** CMS Physician Fee Schedule
**URL:** https://www.cms.gov/medicare/payment/fee-schedules
**Update Frequency:** Annual (January 1)
**Data Size:** ~10,000 entries

**Required Data Elements:**
```json
{
  "cpt_code": "99214",
  "work_rvu": 1.50,
  "practice_expense_rvu": 1.23,
  "malpractice_rvu": 0.10,
  "total_rvu": 2.83,
  "conversion_factor": 33.8872,
  "national_payment": 95.89,
  "facility_payment": 85.00,
  "non_facility_payment": 95.89
}
```

**Acquisition Plan:**
1. Download MPFS from CMS (Week 3)
2. Parse and structure data (Week 3)
3. Load into PostgreSQL (Week 3)
4. Integrate into amount validation (Week 4)

**Cost:** Free (public domain)

---

## 4. TECHNICAL ARCHITECTURE

### 4.1 New Module Structure

```
src/
├── validation/  # NEW MODULE
│   ├── __init__.py
│   ├── medical_code_validator.py      # Main validation orchestrator
│   ├── icd10_validator.py             # ICD-10 specific validation
│   ├── cpt_validator.py               # CPT specific validation
│   ├── ncci_validator.py              # NCCI edit checking
│   ├── mue_validator.py               # MUE limit checking
│   ├── medical_necessity_validator.py # Diagnosis-procedure relationships
│   └── fee_schedule_validator.py      # Fee and RVU validation
│
├── data/  # NEW MODULE
│   ├── __init__.py
│   ├── code_database.py               # Database interface
│   ├── code_cache.py                  # Redis caching layer
│   └── code_updater.py                # Quarterly update automation
│
├── detection/  # ENHANCED EXISTING
│   ├── rule_engine.py                 # Enhanced with medical validation
│   ├── feature_engineering.py         # Enhanced with medical features
│   ├── fraud_detector.py              # Enhanced with validation integration
│   └── ...
```

### 4.2 Database Schema

**PostgreSQL Tables:**

```sql
-- ICD-10 Codes
CREATE TABLE icd10_codes (
    code VARCHAR(10) PRIMARY KEY,
    description TEXT NOT NULL,
    chapter VARCHAR(2),
    category VARCHAR(3),
    valid_from DATE,
    valid_to DATE,
    age_min INT,
    age_max INT,
    gender_restriction CHAR(1),
    requires_laterality BOOLEAN,
    manifestation_code BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_icd10_chapter ON icd10_codes(chapter);
CREATE INDEX idx_icd10_category ON icd10_codes(category);

-- CPT Codes
CREATE TABLE cpt_codes (
    code VARCHAR(5) PRIMARY KEY,
    description TEXT NOT NULL,
    category VARCHAR(50),
    family VARCHAR(50),
    level INT,
    typical_time_minutes INT,
    work_rvu DECIMAL(6,2),
    modifiers_allowed TEXT[], -- Array of valid modifiers
    global_days INT,
    status VARCHAR(20),
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_cpt_category ON cpt_codes(category);
CREATE INDEX idx_cpt_family ON cpt_codes(family);

-- NCCI Edits
CREATE TABLE ncci_edits (
    id SERIAL PRIMARY KEY,
    column1_code VARCHAR(5) NOT NULL,
    column2_code VARCHAR(5) NOT NULL,
    modifier_indicator CHAR(1) NOT NULL,
    effective_date DATE NOT NULL,
    deletion_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(column1_code, column2_code, effective_date)
);
CREATE INDEX idx_ncci_col1 ON ncci_edits(column1_code);
CREATE INDEX idx_ncci_col2 ON ncci_edits(column2_code);
CREATE INDEX idx_ncci_dates ON ncci_edits(effective_date, deletion_date);

-- MUE Limits
CREATE TABLE mue_limits (
    cpt_code VARCHAR(5) PRIMARY KEY,
    mue_value INT NOT NULL,
    adjudication_indicator CHAR(1) NOT NULL,
    rationale TEXT,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fee Schedule
CREATE TABLE fee_schedule (
    cpt_code VARCHAR(5) PRIMARY KEY,
    work_rvu DECIMAL(6,2),
    practice_expense_rvu DECIMAL(6,2),
    malpractice_rvu DECIMAL(6,2),
    total_rvu DECIMAL(6,2),
    national_payment DECIMAL(8,2),
    facility_payment DECIMAL(8,2),
    non_facility_payment DECIMAL(8,2),
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Medical Necessity Rules (simplified)
CREATE TABLE medical_necessity_rules (
    id SERIAL PRIMARY KEY,
    diagnosis_code VARCHAR(10) NOT NULL,
    procedure_code VARCHAR(5) NOT NULL,
    is_covered BOOLEAN DEFAULT true,
    requires_prior_auth BOOLEAN DEFAULT false,
    age_restrictions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_med_nec_dx ON medical_necessity_rules(diagnosis_code);
CREATE INDEX idx_med_nec_px ON medical_necessity_rules(procedure_code);
```

### 4.3 Caching Strategy

**Redis Cache Structure:**
```python
# Cache keys
ICD10:{code} → ValidationResult (TTL: 1 hour)
CPT:{code} → ValidationResult (TTL: 1 hour)
NCCI:{code1}:{code2}:{date} → NCCIResult (TTL: 24 hours)
MUE:{code} → MUELimit (TTL: 24 hours)
FEE:{code} → FeeSchedule (TTL: 24 hours)

# Cache invalidation on quarterly updates
# Implement versioned cache keys for easy invalidation
```

---

## 5. IMPLEMENTATION PLAN

### 5.1 Phase 1: Foundation (Weeks 1-2)

**Week 1: Data Acquisition & Setup**

**Day 1-2: Database Setup**
- [ ] Create PostgreSQL database
- [ ] Implement schema (see section 4.2)
- [ ] Set up indexes and constraints
- [ ] Configure backup strategy

**Day 3-5: Data Acquisition**
- [ ] Download ICD-10-CM from CMS
- [ ] License CPT database from AMA (or use existing file)
- [ ] Download NCCI edit files from CMS
- [ ] Download MUE files from CMS
- [ ] Download MPFS from CMS

**Week 2: Data Loading & Validation**

**Day 1-3: Data Parsing**
```python
# New script: scripts/load_medical_codes.py

def load_icd10_codes():
    """Parse and load ICD-10 codes into database."""
    # Parse CMS ICD-10 files
    # Transform to database format
    # Bulk insert with error handling
    # Validate load completeness

def load_cpt_codes():
    """Parse and load CPT codes into database."""
    # Parse AMA CPT files (or MEDICAL_CODE_MAPPING.json)
    # Extract family relationships
    # Load into database
    # Validate completeness

def load_ncci_edits():
    """Parse and load NCCI edits into database."""
    # Parse CMS NCCI quarterly files
    # Handle effective dates
    # Bulk insert 500K+ records
    # Create optimized indexes

def load_mue_limits():
    """Parse and load MUE limits into database."""
    # Parse CMS MUE files
    # Load with adjudication indicators
    # Validate completeness

def load_fee_schedule():
    """Parse and load Medicare fee schedule."""
    # Parse CMS MPFS files
    # Calculate RVUs and payment amounts
    # Load into database
```

**Day 4-5: Validation**
- [ ] Verify record counts
- [ ] Sample 100 random codes for accuracy
- [ ] Test lookup performance
- [ ] Document any data quality issues

---

### 5.2 Phase 2: Core Validation Modules (Weeks 3-4)

**Week 3: ICD-10 & CPT Validators**

```python
# src/validation/icd10_validator.py

class ICD10Validator:
    """Validate ICD-10 diagnosis codes."""

    def __init__(self, db_connection, cache):
        self.db = db_connection
        self.cache = cache

    def validate(self, code: str, patient_age: int = None,
                patient_gender: str = None) -> ValidationResult:
        """
        Validate ICD-10 code.

        Args:
            code: ICD-10 code (e.g., "E11.9")
            patient_age: Patient age for age-specific validation
            patient_gender: Patient gender (M/F) for gender-specific validation

        Returns:
            ValidationResult with validation status and details
        """
        # Check cache first
        cache_key = f"ICD10:{code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Validate code structure (regex pattern)
        if not self._validate_structure(code):
            return ValidationResult(
                valid=False,
                code=code,
                reason="Invalid ICD-10 code format"
            )

        # Check code exists in database
        code_data = self._lookup_code(code)
        if not code_data:
            return ValidationResult(
                valid=False,
                code=code,
                reason="ICD-10 code not found in database"
            )

        # Check validity dates
        if not self._is_currently_valid(code_data):
            return ValidationResult(
                valid=False,
                code=code,
                reason=f"Code is obsolete (valid until {code_data['valid_to']})"
            )

        # Age-specific validation
        if patient_age and code_data['age_restrictions']:
            if not self._validate_age(code_data, patient_age):
                return ValidationResult(
                    valid=False,
                    code=code,
                    reason=f"Code not valid for age {patient_age}"
                )

        # Gender-specific validation
        if patient_gender and code_data['gender_restriction']:
            if code_data['gender_restriction'] != patient_gender:
                return ValidationResult(
                    valid=False,
                    code=code,
                    reason=f"Code not valid for gender {patient_gender}"
                )

        # All validations passed
        result = ValidationResult(
            valid=True,
            code=code,
            description=code_data['description'],
            chapter=code_data['chapter'],
            category=code_data['category']
        )

        # Cache result
        self.cache.set(cache_key, result, ttl=3600)

        return result

    def _validate_structure(self, code: str) -> bool:
        """Validate ICD-10 code structure using regex."""
        import re
        # Pattern: Letter followed by 2 digits, optional decimal and more digits
        pattern = r'^[A-Z][0-9]{2}(\.[0-9]{1,4})?$'
        return bool(re.match(pattern, code))

    def _lookup_code(self, code: str) -> Optional[dict]:
        """Look up code in database."""
        query = "SELECT * FROM icd10_codes WHERE code = %s"
        return self.db.fetch_one(query, (code,))

    def _is_currently_valid(self, code_data: dict) -> bool:
        """Check if code is currently valid based on dates."""
        from datetime import date
        today = date.today()

        valid_from = code_data['valid_from']
        valid_to = code_data['valid_to']

        if valid_from and today < valid_from:
            return False
        if valid_to and today > valid_to:
            return False

        return True

    def _validate_age(self, code_data: dict, patient_age: int) -> bool:
        """Validate age restrictions."""
        age_min = code_data.get('age_min')
        age_max = code_data.get('age_max')

        if age_min and patient_age < age_min:
            return False
        if age_max and patient_age > age_max:
            return False

        return True
```

```python
# src/validation/cpt_validator.py

class CPTValidator:
    """Validate CPT procedure codes."""

    def __init__(self, db_connection, cache):
        self.db = db_connection
        self.cache = cache

    def validate(self, code: str, modifiers: List[str] = None) -> ValidationResult:
        """
        Validate CPT code.

        Args:
            code: CPT code (e.g., "99214")
            modifiers: List of modifiers (e.g., ["25", "59"])

        Returns:
            ValidationResult with validation status and details
        """
        # Check cache
        cache_key = f"CPT:{code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Validate code structure (5 digits)
        if not self._validate_structure(code):
            return ValidationResult(
                valid=False,
                code=code,
                reason="Invalid CPT code format (must be 5 digits)"
            )

        # Look up code
        code_data = self._lookup_code(code)
        if not code_data:
            return ValidationResult(
                valid=False,
                code=code,
                reason="CPT code not found in database"
            )

        # Check if code is active
        if code_data['status'] != 'active':
            return ValidationResult(
                valid=False,
                code=code,
                reason=f"Code status: {code_data['status']}"
            )

        # Validate modifiers if provided
        modifier_validation = True
        invalid_modifiers = []
        if modifiers:
            allowed_modifiers = code_data.get('modifiers_allowed', [])
            for modifier in modifiers:
                if modifier not in allowed_modifiers:
                    modifier_validation = False
                    invalid_modifiers.append(modifier)

        if not modifier_validation:
            return ValidationResult(
                valid=False,
                code=code,
                reason=f"Invalid modifiers: {invalid_modifiers}"
            )

        # All validations passed
        result = ValidationResult(
            valid=True,
            code=code,
            description=code_data['description'],
            category=code_data['category'],
            family=code_data['family'],
            level=code_data['level'],
            work_rvu=code_data['work_rvu']
        )

        # Cache result
        self.cache.set(cache_key, result, ttl=3600)

        return result

    def get_code_family(self, code: str) -> Optional[str]:
        """Get the family for a CPT code."""
        code_data = self._lookup_code(code)
        return code_data['family'] if code_data else None

    def get_code_level(self, code: str) -> Optional[int]:
        """Get the complexity level for a CPT code."""
        code_data = self._lookup_code(code)
        return code_data['level'] if code_data else None
```

**Week 4: NCCI & MUE Validators**

```python
# src/validation/ncci_validator.py

class NCCIValidator:
    """Validate against NCCI edits."""

    def __init__(self, db_connection, cache):
        self.db = db_connection
        self.cache = cache

    def check_code_pair(self, code1: str, code2: str,
                       date_of_service: date,
                       modifiers: List[str] = None) -> NCCIResult:
        """
        Check if code pair violates NCCI edit.

        Args:
            code1: First CPT code
            code2: Second CPT code
            date_of_service: Date of service
            modifiers: Modifiers used on codes

        Returns:
            NCCIResult indicating if edit exists and if bundling required
        """
        # Check cache
        cache_key = f"NCCI:{code1}:{code2}:{date_of_service}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Look up edit in database
        edit = self._lookup_edit(code1, code2, date_of_service)

        if not edit:
            # No edit exists, codes can be billed separately
            result = NCCIResult(
                has_edit=False,
                can_bill_separately=True,
                modifier_allowed=False
            )
        else:
            # Edit exists, check modifier indicator
            modifier_indicator = edit['modifier_indicator']

            if modifier_indicator == '0':
                # Never bill separately
                result = NCCIResult(
                    has_edit=True,
                    can_bill_separately=False,
                    modifier_allowed=False,
                    column1_code=code1,
                    column2_code=code2,
                    reason="Component of comprehensive service"
                )
            elif modifier_indicator == '1':
                # Can bill separately with appropriate modifier
                has_appropriate_modifier = self._check_modifiers(modifiers)
                result = NCCIResult(
                    has_edit=True,
                    can_bill_separately=has_appropriate_modifier,
                    modifier_allowed=True,
                    modifier_used=has_appropriate_modifier,
                    column1_code=code1,
                    column2_code=code2,
                    reason="Requires modifier 59 or X-modifier for separate billing"
                )
            else:  # modifier_indicator == '9'
                # N/A or deleted
                result = NCCIResult(
                    has_edit=False,
                    can_bill_separately=True,
                    modifier_allowed=False
                )

        # Cache result
        self.cache.set(cache_key, result, ttl=86400)  # 24 hours

        return result

    def analyze_claim_procedures(self, procedures: List[str],
                                date_of_service: date,
                                modifiers_by_code: Dict[str, List[str]] = None
                                ) -> List[NCCIViolation]:
        """
        Analyze all procedure combinations in a claim.

        Args:
            procedures: List of CPT codes in claim
            date_of_service: Date of service
            modifiers_by_code: Dict mapping code to its modifiers

        Returns:
            List of NCCI violations found
        """
        violations = []
        modifiers_by_code = modifiers_by_code or {}

        # Check all pairs
        for i in range(len(procedures)):
            for j in range(i + 1, len(procedures)):
                code1 = procedures[i]
                code2 = procedures[j]
                modifiers = modifiers_by_code.get(code2, [])

                # Check both directions
                result1 = self.check_code_pair(code1, code2, date_of_service, modifiers)
                result2 = self.check_code_pair(code2, code1, date_of_service, modifiers)

                if result1.has_edit and not result1.can_bill_separately:
                    violations.append(NCCIViolation(
                        code1=code1,
                        code2=code2,
                        reason=result1.reason,
                        modifier_indicator=result1.modifier_indicator
                    ))
                elif result2.has_edit and not result2.can_bill_separately:
                    violations.append(NCCIViolation(
                        code1=code2,
                        code2=code1,
                        reason=result2.reason,
                        modifier_indicator=result2.modifier_indicator
                    ))

        return violations

    def _lookup_edit(self, code1: str, code2: str, dos: date) -> Optional[dict]:
        """Look up NCCI edit in database."""
        query = """
            SELECT * FROM ncci_edits
            WHERE column1_code = %s
              AND column2_code = %s
              AND effective_date <= %s
              AND (deletion_date IS NULL OR deletion_date > %s)
            ORDER BY effective_date DESC
            LIMIT 1
        """
        return self.db.fetch_one(query, (code1, code2, dos, dos))

    def _check_modifiers(self, modifiers: List[str]) -> bool:
        """Check if appropriate NCCI modifier is present."""
        ncci_modifiers = ['59', 'XE', 'XS', 'XP', 'XU']  # NCCI override modifiers
        return any(m in ncci_modifiers for m in (modifiers or []))
```

```python
# src/validation/mue_validator.py

class MUEValidator:
    """Validate against MUE (Medically Unlikely Edits) limits."""

    def __init__(self, db_connection, cache):
        self.db = db_connection
        self.cache = cache

    def validate_quantity(self, code: str, quantity: int,
                         adjudication_type: str = 'per_day') -> MUEResult:
        """
        Validate quantity against MUE limit.

        Args:
            code: CPT code
            quantity: Quantity/units billed
            adjudication_type: 'per_day' or 'per_line'

        Returns:
            MUEResult indicating if quantity exceeds limit
        """
        # Check cache
        cache_key = f"MUE:{code}"
        cached = self.cache.get(cache_key)
        if cached:
            mue_limit = cached
        else:
            # Look up MUE limit
            mue_data = self._lookup_mue(code)
            if not mue_data:
                # No MUE limit defined
                return MUEResult(
                    has_limit=False,
                    within_limit=True
                )

            mue_limit = mue_data
            self.cache.set(cache_key, mue_limit, ttl=86400)

        # Check quantity against limit
        limit_value = mue_limit['mue_value']
        adjudication_indicator = mue_limit['adjudication_indicator']

        if quantity > limit_value:
            return MUEResult(
                has_limit=True,
                within_limit=False,
                limit_value=limit_value,
                actual_quantity=quantity,
                adjudication_indicator=adjudication_indicator,
                reason=mue_limit['rationale'],
                recommendation=f"Quantity {quantity} exceeds MUE limit of {limit_value}"
            )
        else:
            return MUEResult(
                has_limit=True,
                within_limit=True,
                limit_value=limit_value,
                actual_quantity=quantity
            )

    def _lookup_mue(self, code: str) -> Optional[dict]:
        """Look up MUE limit in database."""
        query = "SELECT * FROM mue_limits WHERE cpt_code = %s"
        return self.db.fetch_one(query, (code,))
```

---

### 5.3 Phase 3: Integration (Weeks 5-6)

**Week 5: Integrate into Fraud Detection**

**Task 1: Enhance Rule Engine**
```python
# src/detection/rule_engine.py - ENHANCED

class RuleEngine:
    def __init__(self, config_file: Optional[str] = None):
        # ... existing code ...

        # NEW: Add medical validators
        self.icd10_validator = ICD10Validator(db_connection, cache)
        self.cpt_validator = CPTValidator(db_connection, cache)
        self.ncci_validator = NCCIValidator(db_connection, cache)
        self.mue_validator = MUEValidator(db_connection, cache)
        self.medical_necessity_validator = MedicalNecessityValidator(db_connection)

    def _check_upcoding_complexity(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """ENHANCED upcoding detection with medical validation."""
        evidence = []
        score = 0.0

        procedure_codes = claim.get('procedure_codes', [])
        diagnosis_codes = claim.get('diagnosis_codes', [])

        # NEW: Validate all codes first
        for code in procedure_codes:
            validation = self.cpt_validator.validate(code)
            if not validation.valid:
                evidence.append(f"Invalid CPT code: {code} - {validation.reason}")
                score += 0.8  # High score for invalid code

        for code in diagnosis_codes:
            validation = self.icd10_validator.validate(code)
            if not validation.valid:
                evidence.append(f"Invalid ICD-10 code: {code} - {validation.reason}")
                score += 0.7

        # NEW: Check for same-family upcoding
        for proc_code in procedure_codes:
            family = self.cpt_validator.get_code_family(proc_code)
            level = self.cpt_validator.get_code_level(proc_code)

            if family and level:
                # Check if level is appropriate for diagnosis
                expected_level = self._get_expected_level(diagnosis_codes, family)
                if level > expected_level:
                    evidence.append(
                        f"Upcoding detected: {proc_code} (level {level}) "
                        f"exceeds expected level {expected_level} for diagnoses"
                    )
                    score += 0.6

        # ... rest of existing logic ...

        return RuleResult(rule.name, score >= rule.threshold, score, details, evidence)

    def _check_unbundling(self, rule: FraudRule, claim: Dict[str, Any],
                         context_claims: List[Dict[str, Any]]) -> RuleResult:
        """ENHANCED unbundling detection with NCCI validation."""
        evidence = []
        score = 0.0

        procedure_codes = claim.get('procedure_codes', [])
        date_of_service = claim.get('date_of_service')
        modifiers_by_code = claim.get('modifiers_by_code', {})

        # NEW: Check NCCI edits
        ncci_violations = self.ncci_validator.analyze_claim_procedures(
            procedure_codes,
            datetime.strptime(date_of_service, '%Y-%m-%d').date(),
            modifiers_by_code
        )

        if ncci_violations:
            for violation in ncci_violations:
                evidence.append(
                    f"NCCI violation: {violation.code1} and {violation.code2} "
                    f"should be bundled - {violation.reason}"
                )
                score += 0.7

        # NEW: Check MUE limits
        for code in procedure_codes:
            quantity = claim.get('quantities', {}).get(code, 1)
            mue_result = self.mue_validator.validate_quantity(code, quantity)

            if not mue_result.within_limit:
                evidence.append(
                    f"MUE violation: {code} quantity {quantity} "
                    f"exceeds limit of {mue_result.limit_value}"
                )
                score += 0.5

        # ... rest of existing logic ...

        return RuleResult(rule.name, score >= rule.threshold, score, details, evidence)
```

**Task 2: Enhance Feature Engineering**
```python
# src/detection/feature_engineering.py - ENHANCED

class FeatureEngineer:
    def __init__(self):
        # ... existing code ...

        # NEW: Add medical validators
        self.icd10_validator = ICD10Validator(db_connection, cache)
        self.cpt_validator = CPTValidator(db_connection, cache)

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ENHANCED with medical code features."""
        features = pd.DataFrame(index=df.index)

        # ... existing features ...

        # NEW: Medical code validation features
        features['has_invalid_icd10'] = df['diagnosis_codes'].apply(
            lambda codes: any(
                not self.icd10_validator.validate(code).valid
                for code in codes
            )
        ).astype(int)

        features['has_invalid_cpt'] = df['procedure_codes'].apply(
            lambda codes: any(
                not self.cpt_validator.validate(code).valid
                for code in codes
            )
        ).astype(int)

        # NEW: Code hierarchy features
        features['icd10_chapter'] = df['diagnosis_codes'].apply(
            lambda codes: self._get_primary_chapter(codes)
        )

        features['cpt_category'] = df['procedure_codes'].apply(
            lambda codes: self._get_primary_category(codes)
        )

        # NEW: Medical necessity feature
        features['medical_necessity_score'] = df.apply(
            lambda row: self._calculate_medical_necessity(
                row['diagnosis_codes'],
                row['procedure_codes']
            ),
            axis=1
        )

        return features
```

**Week 6: Testing & Validation**

**Comprehensive Testing:**
```python
# tests/unit/test_medical_code_validation.py

class TestMedicalCodeValidation:
    """Test medical code validation."""

    def test_valid_icd10_code(self):
        """Test validation of valid ICD-10 code."""
        validator = ICD10Validator(db, cache)
        result = validator.validate("E11.9")
        assert result.valid
        assert result.description == "Type 2 diabetes mellitus without complications"

    def test_invalid_icd10_code(self):
        """Test validation of invalid ICD-10 code."""
        validator = ICD10Validator(db, cache)
        result = validator.validate("INVALID")
        assert not result.valid
        assert "Invalid ICD-10 code format" in result.reason

    def test_ncci_edit_violation(self):
        """Test NCCI edit detection."""
        validator = NCCIValidator(db, cache)
        # Known NCCI edit pair
        result = validator.check_code_pair("43235", "43239", date(2024, 1, 1))
        assert result.has_edit
        assert not result.can_bill_separately

    def test_ncci_edit_with_modifier(self):
        """Test NCCI edit with appropriate modifier."""
        validator = NCCIValidator(db, cache)
        result = validator.check_code_pair(
            "43235", "43239", date(2024, 1, 1), modifiers=["59"]
        )
        assert result.has_edit
        assert result.can_bill_separately  # Modifier 59 allows separate billing

    def test_mue_limit_violation(self):
        """Test MUE limit detection."""
        validator = MUEValidator(db, cache)
        result = validator.validate_quantity("36415", quantity=2)
        assert not result.within_limit  # MUE limit is 1

    def test_enhanced_upcoding_detection(self):
        """Test enhanced upcoding with medical validation."""
        rule_engine = RuleEngine()
        claim = {
            'claim_id': 'TEST-001',
            'procedure_codes': ['99215'],  # Level 5 E&M
            'diagnosis_codes': ['Z00.00'],  # Routine checkup
            'billed_amount': 500.0
        }
        results, score = rule_engine.analyze_claim(claim)
        assert score > 0.5  # Should detect upcoding
```

**Integration Testing:**
```python
# tests/integration/test_medical_code_integration.py

class TestMedicalCodeIntegration:
    """Test integration of medical code validation into pipeline."""

    def test_end_to_end_with_ncci_validation(self):
        """Test complete pipeline with NCCI validation."""
        # Generate claim with known NCCI violation
        claim = generate_unbundling_claim_with_ncci_violation()

        # Process through pipeline
        detector = FraudDetectorOrchestrator()
        result = detector.detect_single(claim)

        # Should detect unbundling
        assert result.is_fraud
        assert result.risk_level in ['HIGH', 'CRITICAL']
        assert any('NCCI' in e for e in result.evidence)

    def test_performance_with_medical_validation(self):
        """Test that medical validation doesn't slow down pipeline."""
        claims = generate_mixed_claims_batch(1000)
        detector = FraudDetectorOrchestrator()

        start_time = time.time()
        results = detector.detect_batch(claims)
        end_time = time.time()

        processing_time = end_time - start_time
        claims_per_second = len(claims) / processing_time

        # Should still meet performance requirements
        assert claims_per_second > 50
        assert processing_time < 60
```

---

## 6. DATA UPDATE STRATEGY

### 6.1 Quarterly NCCI/MUE Updates

**Automated Update Process:**
```python
# src/data/code_updater.py

class QuarterlyCodeUpdater:
    """Automated quarterly updates for NCCI and MUE."""

    def __init__(self, db_connection):
        self.db = db_connection
        self.cms_urls = {
            'ncci': 'https://www.cms.gov/medicare/coding-billing/ncci',
            'mue': 'https://www.cms.gov/medicare/coding-billing/mue'
        }

    def check_for_updates(self) -> Dict[str, bool]:
        """Check CMS website for new quarterly files."""
        updates_available = {}

        # Check NCCI
        latest_ncci_version = self._get_latest_cms_version('ncci')
        current_ncci_version = self._get_current_db_version('ncci')
        updates_available['ncci'] = latest_ncci_version > current_ncci_version

        # Check MUE
        latest_mue_version = self._get_latest_cms_version('mue')
        current_mue_version = self._get_current_db_version('mue')
        updates_available['mue'] = latest_mue_version > current_mue_version

        return updates_available

    def download_and_apply_updates(self):
        """Download and apply quarterly updates."""
        updates = self.check_for_updates()

        if updates['ncci']:
            logger.info("NCCI update available, downloading...")
            self._download_ncci_update()
            self._apply_ncci_update()
            logger.info("NCCI update applied successfully")

        if updates['mue']:
            logger.info("MUE update available, downloading...")
            self._download_mue_update()
            self._apply_mue_update()
            logger.info("MUE update applied successfully")

        # Invalidate cache after updates
        self._invalidate_cache()

    def _apply_ncci_update(self):
        """Apply NCCI update to database."""
        # Begin transaction
        with self.db.transaction():
            # Mark deleted edits
            self._mark_deleted_ncci_edits()

            # Add new edits
            self._add_new_ncci_edits()

            # Update version tracking
            self._update_version_tracking('ncci')

    def _invalidate_cache(self):
        """Invalidate Redis cache after updates."""
        cache = self._get_cache_connection()
        # Increment cache version to invalidate all cached codes
        cache.incr('code_version')
        logger.info("Cache invalidated after code updates")
```

**Cron Job Configuration:**
```bash
# Run on the 1st of each quarter (January, April, July, October)
0 2 1 1,4,7,10 * /usr/local/bin/python /app/scripts/quarterly_update.py

# Or use more sophisticated scheduling with APScheduler in Python
```

### 6.2 Annual Updates (ICD-10, CPT)

**Annual Update Process:**
```python
# Similar to quarterly but runs annually
# October 1 for ICD-10
# January 1 for CPT

class AnnualCodeUpdater:
    """Automated annual updates for ICD-10 and CPT."""

    def update_icd10_codes(self):
        """Update ICD-10 codes (October 1 annually)."""
        # Download new ICD-10-CM files
        # Mark obsolete codes with valid_to date
        # Add new codes with valid_from date
        # Update code descriptions if changed
        # Invalidate cache

    def update_cpt_codes(self):
        """Update CPT codes (January 1 annually)."""
        # Download new CPT files (requires license)
        # Mark deleted codes as inactive
        # Add new codes
        # Update descriptions and RVUs
        # Invalidate cache
```

---

## 7. TESTING STRATEGY

### 7.1 Unit Tests

**Test Coverage Requirements: >90%**

```python
# Test each validator independently
tests/unit/test_icd10_validator.py         # ICD-10 validation
tests/unit/test_cpt_validator.py           # CPT validation
tests/unit/test_ncci_validator.py          # NCCI edit checking
tests/unit/test_mue_validator.py           # MUE limit checking
tests/unit/test_medical_necessity.py       # Medical necessity validation
```

### 7.2 Integration Tests

```python
# Test integration with fraud detection
tests/integration/test_medical_validation_pipeline.py
tests/integration/test_code_validation_performance.py
tests/integration/test_quarterly_update_process.py
```

### 7.3 Performance Tests

**Performance Requirements:**
- Single code validation: <10ms
- NCCI pair checking: <20ms
- Claim with 5 procedures: <100ms
- Batch of 1000 claims: <60 seconds

```python
# tests/performance/test_validation_latency.py

def test_icd10_validation_latency():
    """Test ICD-10 validation latency."""
    validator = ICD10Validator(db, cache)

    start = time.time()
    for _ in range(1000):
        validator.validate("E11.9")
    end = time.time()

    avg_latency = (end - start) / 1000 * 1000  # ms
    assert avg_latency < 10  # <10ms average

def test_ncci_checking_latency():
    """Test NCCI checking latency."""
    validator = NCCIValidator(db, cache)

    start = time.time()
    for _ in range(1000):
        validator.check_code_pair("43235", "43239", date.today())
    end = time.time()

    avg_latency = (end - start) / 1000 * 1000  # ms
    assert avg_latency < 20  # <20ms average
```

### 7.4 Validation Tests

**Test with Known Fraud Cases:**
```python
# tests/validation/test_known_fraud_cases.py

def test_cms_cert_fraud_cases():
    """Test against CMS CERT fraud cases."""
    # Use publicly available CMS CERT (Comprehensive Error Rate Testing) data
    cert_fraud_cases = load_cms_cert_data()

    detector = FraudDetectorOrchestrator()
    results = detector.detect_batch(cert_fraud_cases)

    # Calculate detection rate
    detected = sum(1 for r in results if r.is_fraud)
    detection_rate = detected / len(cert_fraud_cases)

    # Should detect >80% of known fraud
    assert detection_rate > 0.80

def test_oig_fraud_patterns():
    """Test against OIG (Office of Inspector General) fraud patterns."""
    # Use OIG fraud pattern examples
    oig_patterns = load_oig_fraud_patterns()

    detector = FraudDetectorOrchestrator()

    for pattern in oig_patterns:
        result = detector.detect_single(pattern)
        # Should flag all OIG patterns as fraud
        assert result.is_fraud
```

---

## 8. DEPLOYMENT PLAN

### 8.1 Deployment Phases

**Phase 1: Development Environment (Week 3)**
- Deploy code validators to dev
- Load test data
- Run comprehensive testing

**Phase 2: Staging Environment (Week 5)**
- Deploy to staging with production-like data
- Performance testing at scale
- Load balancing and caching validation

**Phase 3: Production Deployment (Week 6)**
- Phased rollout:
  1. Enable for 10% of traffic
  2. Monitor performance and accuracy
  3. Increase to 50% if metrics good
  4. Full rollout after 48 hours

### 8.2 Monitoring & Alerts

**Key Metrics to Monitor:**
```yaml
# Prometheus metrics
medical_code_validation_latency_ms:
  target: <10ms (p95)
  alert: >20ms

ncci_checking_latency_ms:
  target: <20ms (p95)
  alert: >40ms

code_validation_error_rate:
  target: <0.1%
  alert: >1%

cache_hit_rate:
  target: >95%
  alert: <90%

database_query_latency_ms:
  target: <50ms (p95)
  alert: >100ms

quarterly_update_status:
  target: success
  alert: failure
```

**Alerting:**
- PagerDuty for critical issues
- Slack for warnings
- Email for quarterly update notifications

### 8.3 Rollback Plan

**Rollback Triggers:**
- Code validation latency >100ms (p95)
- Error rate >5%
- Accuracy drops >10%
- Database connection issues

**Rollback Process:**
1. Feature flag to disable medical validation
2. Fall back to hardcoded rules
3. Investigate issue
4. Fix and redeploy

---

## 9. SUCCESS CRITERIA

### 9.1 Technical Success Criteria

- [ ] All 72,000 ICD-10 codes loaded and validated
- [ ] All 10,000 CPT codes loaded and validated
- [ ] All 500,000+ NCCI edits loaded and operational
- [ ] MUE limits loaded for all applicable codes
- [ ] Code validation latency <10ms (p95)
- [ ] NCCI checking latency <20ms (p95)
- [ ] Cache hit rate >95%
- [ ] Database uptime >99.9%
- [ ] Test coverage >90%

### 9.2 Functional Success Criteria

- [ ] Invalid ICD-10 codes detected and flagged
- [ ] Invalid CPT codes detected and flagged
- [ ] NCCI violations detected with >95% accuracy
- [ ] MUE violations detected with >95% accuracy
- [ ] Unbundling detection improved from 15% → 90%+
- [ ] Upcoding detection improved from 30% → 70%+
- [ ] Overall fraud coverage improved from 34% → 70%+

### 9.3 Business Success Criteria

- [ ] System meets production readiness requirements
- [ ] Accuracy >94%
- [ ] False positive rate <3.8%
- [ ] Processing time <4 hours per batch
- [ ] Quarterly updates automated
- [ ] Compliance with CMS standards

---

## 10. RISKS & MITIGATION

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CPT license unavailable | Low | Critical | Use MEDICAL_CODE_MAPPING.json as fallback |
| NCCI data format changes | Medium | High | Implement flexible parser, version checking |
| Database performance issues | Medium | High | Optimize indexes, implement caching |
| Code validation too slow | Medium | Medium | Aggressive caching, query optimization |
| Quarterly update fails | Low | Medium | Automated rollback, alerts, manual override |

### 10.2 Data Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MEDICAL_CODE_MAPPING.json inaccurate | Medium | High | Validate against CMS sources, flag discrepancies |
| CMS data errors | Low | Medium | Cross-reference multiple sources, manual review |
| Historical data migration issues | Medium | Medium | Thorough testing, gradual rollout |

### 10.3 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Timeline slips | Medium | High | Prioritize P0 features, reduce scope if needed |
| Resource unavailability | Low | Medium | Cross-train team, have backup resources |
| Scope creep | Medium | Medium | Strict scope management, defer P2/P3 features |

---

## 11. COST ESTIMATE

### 11.1 Data Acquisition Costs

| Item | Cost | Frequency | Annual Cost |
|------|------|-----------|-------------|
| CPT Database License | $10K-$15K | Annual | $10K-$15K |
| ICD-10 Database | Free (CMS) | - | $0 |
| NCCI Edits | Free (CMS) | - | $0 |
| MUE Limits | Free (CMS) | - | $0 |
| Fee Schedule | Free (CMS) | - | $0 |
| **Total Data** | | | **$10K-$15K** |

### 11.2 Infrastructure Costs

| Item | Cost | Frequency | Annual Cost |
|------|------|-----------|-------------|
| PostgreSQL Database | $200/mo | Monthly | $2.4K |
| Redis Cache | $100/mo | Monthly | $1.2K |
| Backup Storage | $50/mo | Monthly | $0.6K |
| Monitoring Tools | $100/mo | Monthly | $1.2K |
| **Total Infrastructure** | | | **$5.4K** |

### 11.3 Personnel Costs (6 weeks)

| Role | Rate | Hours | Total |
|------|------|-------|-------|
| Senior Developer | $150/hr | 240 hrs | $36K |
| Developer | $100/hr | 240 hrs | $24K |
| Medical Coding Consultant | $200/hr | 40 hrs | $8K |
| **Total Personnel** | | | **$68K** |

### 11.4 Total Project Cost

| Category | Amount |
|----------|--------|
| Data Acquisition | $10K-$15K |
| Infrastructure (6 months) | $2.7K |
| Personnel (6 weeks) | $68K |
| **Total** | **$80K-$86K** |

---

## 12. TIMELINE & MILESTONES

### Gantt Chart Overview

```
Week 1: [===========] Database Setup & Data Acquisition
Week 2: [===========] Data Loading & Validation
Week 3: [===========] ICD-10 & CPT Validators
Week 4: [===========] NCCI & MUE Validators
Week 5: [===========] Integration & Testing
Week 6: [===========] Deployment & Validation
```

### Key Milestones

| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Database & Data Ready | All CMS data downloaded, DB schema created |
| 2 | Data Loaded | All codes loaded, validated, indexed |
| 3 | Core Validators Complete | ICD-10 & CPT validation working |
| 4 | NCCI/MUE Complete | All validators implemented, tested |
| 5 | Integration Complete | Fraud detection using medical validation |
| 6 | Production Deploy | System live, meeting performance targets |

---

## 13. MAINTENANCE PLAN

### 13.1 Ongoing Maintenance Tasks

**Quarterly (Every 3 months):**
- Update NCCI edits (January, April, July, October)
- Update MUE limits (January, April, July, October)
- Review and optimize database indexes
- Performance tuning based on metrics

**Annually (October 1):**
- Update ICD-10 codes
- Refresh code hierarchies
- Update medical necessity rules

**Annually (January 1):**
- Update CPT codes (requires new license)
- Update RVU values
- Refresh fee schedules

**Continuous:**
- Monitor performance metrics
- Review and optimize caching
- Database backup and maintenance
- Security patches and updates

### 13.2 Support Model

**Tier 1: Automated**
- Quarterly updates (automated)
- Cache management (automated)
- Performance monitoring (automated)

**Tier 2: Engineering Team**
- Code validation issues
- Performance optimization
- Bug fixes

**Tier 3: Medical Coding Consultant**
- Medical necessity rules updates
- New fraud pattern identification
- Validation of edge cases

---

## 14. CONCLUSION

This medical coding integration plan provides a comprehensive roadmap for transforming the fraud detection system from 34% to 70%+ effectiveness. The 6-week implementation timeline is aggressive but achievable with dedicated resources.

**Critical Success Factors:**
1. CPT license acquisition (Week 1) - BLOCKING
2. NCCI database load (Week 2-3) - HIGH IMPACT
3. Performance optimization (Weeks 4-5) - CRITICAL
4. Comprehensive testing (Week 6) - REQUIRED

**Expected Outcomes:**
- Unbundling detection: 15% → 90%+ effectiveness
- Upcoding detection: 30% → 70%+ effectiveness
- Overall system: 34% → 70%+ effectiveness
- Production-ready in 6 weeks

**Next Steps:**
1. Approve plan and budget ($80K-$86K)
2. Assign development team (2-3 developers)
3. Engage medical coding consultant
4. Acquire CPT license from AMA
5. Begin Week 1 implementation

---

**Document Status:** Final Draft
**Approval Required:** Technical Lead, Product Owner, CFO
**Start Date:** [To Be Determined]
**Target Completion:** 6 weeks from start
**Project Manager:** [To Be Assigned]
