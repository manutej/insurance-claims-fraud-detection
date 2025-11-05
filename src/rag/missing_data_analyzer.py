"""
Missing Data Analyzer for Insurance Claims Fraud Detection.

This module analyzes claims for missing fields and suspicious submission patterns
that may indicate fraud. Missing data can be a signal for various fraud types:
- Incomplete submissions to hide fraud details
- Systematic omission of specific fields
- Unusual submission timing patterns

Key Components:
- MissingFieldDetector: Identifies missing/incomplete fields
- SuspiciousSubmissionPatternDetector: Analyzes submission patterns
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import Counter, defaultdict
import statistics


class MissingFieldDetector:
    """
    Detects missing fields in insurance claims and assesses their criticality.

    This detector identifies which required and optional fields are missing
    from a claim and scores how critical each missing field is for fraud
    detection purposes.
    """

    # Define critical fields and their importance scores (0.0-1.0)
    FIELD_CRITICALITY = {
        # Absolutely critical fields (0.90-1.00)
        "diagnosis_codes": 0.95,
        "procedure_codes": 0.95,
        "billed_amount": 0.90,
        "date_of_service": 0.90,
        "provider_npi": 1.00,
        "patient_id": 1.00,
        "claim_id": 1.00,

        # Highly important fields (0.70-0.89)
        "diagnosis_descriptions": 0.80,
        "procedure_descriptions": 0.80,
        "provider_specialty": 0.75,
        "service_location": 0.75,
        "claim_type": 0.85,

        # Important fields (0.50-0.69)
        "treatment_type": 0.60,
        "days_supply": 0.55,  # Important for pharmacy claims
        "medical_necessity": 0.65,
        "provider_id": 0.60,

        # Moderately important fields (0.30-0.49)
        "service_location_desc": 0.25,
        "rendering_hours": 0.40,

        # Optional/supplementary fields (0.10-0.29)
        "notes": 0.15,
        "red_flags": 0.20,
        "day_of_week": 0.10,
    }

    # Fields that should not be None or empty
    REQUIRED_FIELDS = {
        "claim_id",
        "patient_id",
        "provider_npi",
        "date_of_service",
        "billed_amount",
        "claim_type",
    }

    # Fields that are conditionally required based on claim type
    CONDITIONAL_FIELDS = {
        "professional": ["diagnosis_codes", "procedure_codes"],
        "pharmacy": ["ndc_code", "drug_name", "days_supply"],
        "no_fault": ["accident_date", "medical_treatment_type"],
    }

    def detect_missing_fields(self, claim: Dict[str, Any]) -> List[str]:
        """
        Identifies missing or incomplete fields in a claim.

        A field is considered missing if:
        - It's not present in the claim dictionary
        - Its value is None
        - Its value is an empty list (for list fields)
        - Its value is an empty string (for string fields)

        Args:
            claim: Dictionary containing claim data

        Returns:
            List of field names that are missing
        """
        missing_fields = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if not self._is_field_present(claim, field):
                missing_fields.append(field)

        # Check conditional fields based on claim type
        claim_type = claim.get("claim_type", "professional")
        conditional_fields = self.CONDITIONAL_FIELDS.get(claim_type, [])

        for field in conditional_fields:
            if not self._is_field_present(claim, field):
                missing_fields.append(field)

        # Check other important fields from criticality map
        for field in self.FIELD_CRITICALITY:
            if field not in missing_fields and not self._is_field_present(claim, field):
                # Only add if it's a reasonably critical field
                if self.FIELD_CRITICALITY[field] >= 0.50:
                    # Skip fields that are claim-type specific
                    # For example, days_supply is only for pharmacy claims
                    if field == "days_supply" and claim_type != "pharmacy":
                        continue
                    missing_fields.append(field)

        return missing_fields

    def _is_field_present(self, claim: Dict[str, Any], field: str) -> bool:
        """
        Checks if a field is present and has a valid value.

        Args:
            claim: Claim dictionary
            field: Field name to check

        Returns:
            True if field is present and valid, False otherwise
        """
        if field not in claim:
            return False

        value = claim[field]

        # None is always missing
        if value is None:
            return False

        # Empty lists are missing
        if isinstance(value, list) and len(value) == 0:
            return False

        # Empty strings are missing
        if isinstance(value, str) and value.strip() == "":
            return False

        return True

    def assess_missing_criticality(self, missing_fields: List[str]) -> Dict[str, float]:
        """
        Assesses the criticality of each missing field.

        Args:
            missing_fields: List of missing field names

        Returns:
            Dictionary mapping field name to criticality score (0.0-1.0)
            Higher scores indicate more critical missing data
        """
        criticality = {}

        for field in missing_fields:
            # Get base criticality score
            base_score = self.FIELD_CRITICALITY.get(field, 0.50)

            # Boost score if it's a required field
            if field in self.REQUIRED_FIELDS:
                base_score = min(1.0, base_score * 1.1)

            criticality[field] = round(base_score, 2)

        return criticality

    def compute_missing_data_percentage(self, claim: Dict[str, Any]) -> float:
        """
        Computes the percentage of important fields that are missing.

        This considers all fields in the criticality map that have a
        criticality score >= 0.50 (important fields).

        Args:
            claim: Claim dictionary

        Returns:
            Percentage of important fields missing (0.0-1.0)
        """
        # Get all important fields (criticality >= 0.50)
        important_fields = [
            field for field, score in self.FIELD_CRITICALITY.items()
            if score >= 0.50
        ]

        # Add required fields
        all_important = set(important_fields) | self.REQUIRED_FIELDS

        # Count how many are missing
        missing_count = 0
        for field in all_important:
            if not self._is_field_present(claim, field):
                missing_count += 1

        # Calculate percentage
        if len(all_important) == 0:
            return 0.0

        percentage = missing_count / len(all_important)
        return round(percentage, 3)


class SuspiciousSubmissionPatternDetector:
    """
    Detects suspicious patterns in claim submission behavior.

    Analyzes historical claim submission patterns for providers and patients
    to identify anomalies that may indicate fraud, such as:
    - Frequent submission of incomplete claims
    - Systematic omission of specific fields
    - Unusual submission timing (weekends, nights)
    - Temporal pattern anomalies
    """

    def detect_provider_submission_pattern(
        self,
        provider_npi: str,
        historical_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes a provider's historical submission patterns.

        Args:
            provider_npi: Provider NPI identifier
            historical_claims: List of historical claims from this provider

        Returns:
            Dictionary containing:
            - missing_rate: Overall rate of missing data
            - missing_field_types: Counter of which fields are missing
            - temporal_pattern: Submission timing analysis
            - suspicious_score: Overall suspicion score (0.0-1.0)
        """
        if not historical_claims:
            return {
                "missing_rate": 0.0,
                "missing_field_types": {},
                "temporal_pattern": {},
                "suspicious_score": 0.0,
            }

        detector = MissingFieldDetector()
        missing_counter = Counter()
        total_missing = 0
        total_fields_checked = 0

        # Analyze each claim for missing fields
        for claim in historical_claims:
            missing_fields = detector.detect_missing_fields(claim)
            total_missing += len(missing_fields)

            # Count which fields are commonly missing
            for field in missing_fields:
                missing_counter[field] += 1

            # Count important fields checked
            total_fields_checked += len(detector.REQUIRED_FIELDS)
            total_fields_checked += len([
                f for f in detector.FIELD_CRITICALITY
                if detector.FIELD_CRITICALITY[f] >= 0.50
            ])

        # Calculate missing rate
        missing_rate = total_missing / total_fields_checked if total_fields_checked > 0 else 0.0

        # Calculate suspicious score based on missing rate and field types
        suspicious_score = self._calculate_provider_suspicion(
            missing_rate,
            missing_counter,
            len(historical_claims)
        )

        return {
            "missing_rate": round(missing_rate, 3),
            "missing_field_types": dict(missing_counter),
            "temporal_pattern": self._analyze_temporal_patterns(historical_claims),
            "suspicious_score": round(suspicious_score, 3),
        }

    def detect_patient_submission_pattern(
        self,
        patient_id: str,
        historical_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes a patient's historical submission patterns.

        Args:
            patient_id: Patient identifier
            historical_claims: List of historical claims from this patient

        Returns:
            Dictionary with pattern analysis similar to provider pattern
        """
        if not historical_claims:
            return {
                "missing_rate": 0.0,
                "missing_field_types": {},
                "suspicious_score": 0.0,
            }

        detector = MissingFieldDetector()
        missing_counter = Counter()
        total_missing = 0
        total_fields_checked = 0

        for claim in historical_claims:
            missing_fields = detector.detect_missing_fields(claim)
            total_missing += len(missing_fields)

            for field in missing_fields:
                missing_counter[field] += 1

            total_fields_checked += len(detector.REQUIRED_FIELDS) + 5  # Approx important fields

        missing_rate = total_missing / total_fields_checked if total_fields_checked > 0 else 0.0

        # Patients typically shouldn't have high missing rates
        # Higher threshold than providers
        suspicious_score = min(1.0, missing_rate * 2.5) if missing_rate > 0.20 else 0.0

        return {
            "missing_rate": round(missing_rate, 3),
            "missing_field_types": dict(missing_counter),
            "suspicious_score": round(suspicious_score, 3),
        }

    def detect_temporal_pattern(
        self,
        claim: Dict[str, Any],
        similar_historical_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detects unusual temporal submission patterns.

        Args:
            claim: Current claim being analyzed
            similar_historical_claims: Similar historical claims for comparison

        Returns:
            Dictionary with temporal pattern analysis
        """
        temporal_info = {}

        # Check if submission timestamp exists
        submission_timestamp = claim.get("submission_timestamp")
        if submission_timestamp:
            if isinstance(submission_timestamp, str):
                try:
                    submission_dt = datetime.fromisoformat(submission_timestamp)
                except ValueError:
                    submission_dt = None
            else:
                submission_dt = submission_timestamp

            if submission_dt:
                # Check day of week
                day_of_week = submission_dt.weekday()
                is_weekend = day_of_week >= 5  # Saturday or Sunday

                # Check time of day
                hour = submission_dt.hour
                is_night_time = hour < 6 or hour > 22  # Between 10 PM and 6 AM

                temporal_info["day_of_week"] = day_of_week
                temporal_info["hour"] = hour
                temporal_info["is_weekend"] = is_weekend
                temporal_info["is_night_time"] = is_night_time

                # Check if unusual
                temporal_anomaly = is_weekend or is_night_time
                temporal_info["temporal_anomaly"] = temporal_anomaly
                temporal_info["suspicious"] = temporal_anomaly

        # Analyze date of service pattern
        date_of_service = claim.get("date_of_service")
        if date_of_service:
            temporal_info["date_of_service"] = date_of_service

            # Check if date is far in the past (late submission)
            # This could indicate data quality issues or fraud

        return temporal_info

    def assess_submission_suspicion(
        self,
        provider_npi: str,
        patient_id: str,
        claim: Dict[str, Any],
        provider_history: Optional[List[Dict[str, Any]]] = None,
        patient_history: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Calculates an overall submission suspicion score.

        Combines provider patterns, patient patterns, and current claim
        characteristics to produce a unified suspicion score.

        Args:
            provider_npi: Provider identifier
            patient_id: Patient identifier
            claim: Current claim
            provider_history: Historical claims from provider
            patient_history: Historical claims from patient

        Returns:
            Suspicion score (0.0-1.0), higher = more suspicious
        """
        suspicion_factors = []

        # Analyze provider pattern
        if provider_history:
            provider_pattern = self.detect_provider_submission_pattern(
                provider_npi,
                provider_history
            )
            suspicion_factors.append(provider_pattern["suspicious_score"])

        # Analyze patient pattern
        if patient_history:
            patient_pattern = self.detect_patient_submission_pattern(
                patient_id,
                patient_history
            )
            suspicion_factors.append(patient_pattern["suspicious_score"])

        # Analyze current claim
        detector = MissingFieldDetector()
        missing_fields = detector.detect_missing_fields(claim)

        if missing_fields:
            criticality = detector.assess_missing_criticality(missing_fields)
            # Average criticality of missing fields
            avg_criticality = sum(criticality.values()) / len(criticality)
            suspicion_factors.append(avg_criticality)

        # Temporal pattern
        temporal_pattern = self.detect_temporal_pattern(claim, [])
        if temporal_pattern.get("temporal_anomaly"):
            suspicion_factors.append(0.60)  # Moderate suspicion boost

        # Calculate overall score
        if suspicion_factors:
            # Use average, but weight towards lower scores if all factors are low
            overall_score = sum(suspicion_factors) / len(suspicion_factors)

            # If all factors are low (< 0.40), apply a dampening factor
            all_low = all(score < 0.40 for score in suspicion_factors)
            if all_low:
                overall_score = overall_score * 0.80  # Dampen low scores
            # Boost if multiple high factors present
            elif len(suspicion_factors) >= 3 and overall_score > 0.50:
                overall_score = min(1.0, overall_score * 1.15)

            return round(overall_score, 3)

        return 0.0

    def _calculate_provider_suspicion(
        self,
        missing_rate: float,
        missing_fields_counter: Counter,
        claim_count: int
    ) -> float:
        """
        Calculates provider suspicion score based on missing data patterns.

        Args:
            missing_rate: Overall missing data rate
            missing_fields_counter: Counter of missing field types
            claim_count: Number of claims analyzed

        Returns:
            Suspicion score (0.0-1.0)
        """
        # Base suspicion from missing rate
        base_suspicion = min(1.0, missing_rate * 3.0)

        # Check for systematic missing patterns (same fields always missing)
        if missing_fields_counter and claim_count > 0:
            # If certain fields are missing in >50% of claims, it's systematic
            systematic_count = sum(
                1 for count in missing_fields_counter.values()
                if count / claim_count > 0.50
            )

            if systematic_count > 0:
                # Boost suspicion for systematic omission
                base_suspicion = min(1.0, base_suspicion + (systematic_count * 0.15))

        return base_suspicion

    def _analyze_temporal_patterns(
        self,
        historical_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes temporal submission patterns across historical claims.

        Args:
            historical_claims: List of historical claims

        Returns:
            Dictionary with temporal pattern statistics
        """
        if not historical_claims:
            return {}

        weekday_counts = Counter()
        hour_counts = Counter()

        for claim in historical_claims:
            submission_timestamp = claim.get("submission_timestamp")
            if submission_timestamp:
                if isinstance(submission_timestamp, str):
                    try:
                        dt = datetime.fromisoformat(submission_timestamp)
                        weekday_counts[dt.weekday()] += 1
                        hour_counts[dt.hour] += 1
                    except ValueError:
                        pass

        return {
            "weekday_distribution": dict(weekday_counts),
            "hour_distribution": dict(hour_counts),
            "total_analyzed": len(historical_claims),
        }
