"""
Rule-based fraud detection engine for insurance claims.

This module implements configurable rules for detecting various types of fraud
including upcoding, phantom billing, unbundling, staged accidents, and other
suspicious patterns.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FraudRule:
    """Represents a fraud detection rule."""

    name: str
    description: str
    fraud_type: str
    weight: float
    threshold: float
    enabled: bool = True


@dataclass
class RuleResult:
    """Result of applying a fraud detection rule."""

    rule_name: str
    triggered: bool
    score: float
    details: str
    evidence: List[str]


class RuleEngine:
    """
    Configurable rule-based fraud detection engine.

    Implements rules for all major fraud types:
    - Upcoding: Services billed at higher complexity than performed
    - Phantom Billing: Services billed but never rendered
    - Unbundling: Single procedures split into multiple claims
    - Staged Accidents: Fabricated auto accidents
    - Prescription Fraud: Drug diversion and doctor shopping
    - Kickback Schemes: Hidden financial relationships
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the rule engine with configurable thresholds."""
        self.rules = self._initialize_default_rules()
        self.thresholds = self._initialize_default_thresholds()
        self.provider_history = defaultdict(list)
        self.patient_history = defaultdict(list)
        self.claim_patterns = defaultdict(list)

        if config_file:
            self.load_config(config_file)

    def _initialize_default_rules(self) -> Dict[str, FraudRule]:
        """Initialize default fraud detection rules."""
        return {
            "upcoding_complexity": FraudRule(
                name="upcoding_complexity",
                description="Detect upcoding based on procedure complexity vs diagnosis",
                fraud_type="upcoding",
                weight=0.8,
                threshold=0.7,
            ),
            "phantom_billing_schedule": FraudRule(
                name="phantom_billing_schedule",
                description="Detect services billed outside normal hours",
                fraud_type="phantom_billing",
                weight=0.9,
                threshold=0.8,
            ),
            "phantom_billing_location": FraudRule(
                name="phantom_billing_location",
                description="Detect services at impossible locations",
                fraud_type="phantom_billing",
                weight=0.95,
                threshold=0.9,
            ),
            "unbundling_detection": FraudRule(
                name="unbundling_detection",
                description="Detect artificially separated procedures",
                fraud_type="unbundling",
                weight=0.85,
                threshold=0.75,
            ),
            "staged_accident_pattern": FraudRule(
                name="staged_accident_pattern",
                description="Detect suspicious accident patterns",
                fraud_type="staged_accident",
                weight=0.9,
                threshold=0.8,
            ),
            "prescription_fraud_volume": FraudRule(
                name="prescription_fraud_volume",
                description="Detect excessive prescription volumes",
                fraud_type="prescription_fraud",
                weight=0.8,
                threshold=0.7,
            ),
            "kickback_referral_pattern": FraudRule(
                name="kickback_referral_pattern",
                description="Detect suspicious referral patterns",
                fraud_type="kickback_scheme",
                weight=0.75,
                threshold=0.7,
            ),
            "billing_frequency_anomaly": FraudRule(
                name="billing_frequency_anomaly",
                description="Detect unusual billing frequencies",
                fraud_type="general",
                weight=0.6,
                threshold=0.6,
            ),
            "amount_anomaly": FraudRule(
                name="amount_anomaly",
                description="Detect suspicious billing amounts",
                fraud_type="general",
                weight=0.7,
                threshold=0.65,
            ),
        }

    def _initialize_default_thresholds(self) -> Dict[str, Any]:
        """Initialize default detection thresholds."""
        return {
            "max_daily_claims_per_provider": 50,
            "max_hourly_claims_per_provider": 8,
            "max_amount_per_claim": 10000,
            "suspicious_amount_multiplier": 3.0,
            "min_time_between_claims_minutes": 15,
            "max_procedures_per_claim": 10,
            "suspicious_weekend_ratio": 0.3,
            "max_patient_visits_per_day": 3,
            "upcoding_complexity_threshold": 2.0,
            "phantom_billing_hours": {"start": 6, "end": 22},  # 6 AM  # 10 PM
            "unbundling_time_window_hours": 24,
            "staged_accident_similarity_threshold": 0.8,
            "prescription_daily_limit": 5,
            "referral_concentration_threshold": 0.7,
        }

    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if "rules" in config:
                for rule_name, rule_config in config["rules"].items():
                    if rule_name in self.rules:
                        self.rules[rule_name].threshold = rule_config.get(
                            "threshold", self.rules[rule_name].threshold
                        )
                        self.rules[rule_name].weight = rule_config.get(
                            "weight", self.rules[rule_name].weight
                        )
                        self.rules[rule_name].enabled = rule_config.get(
                            "enabled", self.rules[rule_name].enabled
                        )

            if "thresholds" in config:
                self.thresholds.update(config["thresholds"])

            logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")

    def analyze_claim(
        self, claim: Dict[str, Any], context_claims: List[Dict[str, Any]] = None
    ) -> Tuple[List[RuleResult], float]:
        """
        Analyze a single claim for fraud indicators.

        Args:
            claim: The claim to analyze
            context_claims: Related claims for pattern analysis

        Returns:
            Tuple of (rule results, overall fraud score)
        """
        results = []
        context_claims = context_claims or []

        # Update history for pattern analysis
        self._update_claim_history(claim)

        # Apply all enabled rules
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue

            result = self._apply_rule(rule, claim, context_claims)
            results.append(result)

        # Calculate overall fraud score
        fraud_score = self._calculate_fraud_score(results)

        return results, fraud_score

    def _apply_rule(
        self, rule: FraudRule, claim: Dict[str, Any], context_claims: List[Dict[str, Any]]
    ) -> RuleResult:
        """Apply a specific rule to a claim."""
        try:
            if rule.name == "upcoding_complexity":
                return self._check_upcoding_complexity(rule, claim)
            elif rule.name == "phantom_billing_schedule":
                return self._check_phantom_billing_schedule(rule, claim)
            elif rule.name == "phantom_billing_location":
                return self._check_phantom_billing_location(rule, claim)
            elif rule.name == "unbundling_detection":
                return self._check_unbundling(rule, claim, context_claims)
            elif rule.name == "staged_accident_pattern":
                return self._check_staged_accident(rule, claim, context_claims)
            elif rule.name == "prescription_fraud_volume":
                return self._check_prescription_fraud(rule, claim)
            elif rule.name == "kickback_referral_pattern":
                return self._check_kickback_scheme(rule, claim, context_claims)
            elif rule.name == "billing_frequency_anomaly":
                return self._check_billing_frequency(rule, claim)
            elif rule.name == "amount_anomaly":
                return self._check_amount_anomaly(rule, claim)
            else:
                return RuleResult(rule.name, False, 0.0, "Rule not implemented", [])

        except Exception as e:
            logger.error(f"Error applying rule {rule.name}: {e}")
            return RuleResult(rule.name, False, 0.0, f"Rule execution error: {e}", [])

    def _check_upcoding_complexity(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for upcoding based on procedure complexity vs diagnosis."""
        evidence = []
        score = 0.0

        procedure_codes = claim.get("procedure_codes", [])
        diagnosis_codes = claim.get("diagnosis_codes", [])
        billed_amount = claim.get("billed_amount", 0)

        # Check for high-complexity procedures with simple diagnoses
        high_complexity_procedures = ["99215", "99285", "99291", "99292"]
        simple_diagnoses = ["Z00.00", "Z12.11", "I10", "E11.9"]

        for proc_code in procedure_codes:
            if proc_code in high_complexity_procedures:
                for diag_code in diagnosis_codes:
                    if diag_code in simple_diagnoses:
                        evidence.append(
                            f"High complexity procedure {proc_code} for simple diagnosis {diag_code}"
                        )
                        score += 0.3

        # Check for excessive billing amounts relative to procedures
        expected_amount = len(procedure_codes) * 150  # Average procedure cost
        if billed_amount > expected_amount * self.thresholds["suspicious_amount_multiplier"]:
            evidence.append(f"Billed amount ${billed_amount} excessive for procedures")
            score += 0.4

        # Check procedure code patterns
        if self._has_suspicious_procedure_progression(procedure_codes):
            evidence.append("Suspicious procedure code progression detected")
            score += 0.3

        triggered = score >= rule.threshold
        details = f"Upcoding score: {score:.2f}, Evidence: {len(evidence)} items"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_phantom_billing_schedule(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for services billed outside normal operating hours."""
        evidence = []
        score = 0.0

        date_of_service = claim.get("date_of_service", "")
        day_of_week = claim.get("day_of_week", "")

        try:
            service_date = datetime.strptime(date_of_service, "%Y-%m-%d")

            # Check for weekend services (unless emergency)
            service_location = claim.get("service_location", "")
            if (
                day_of_week in ["Saturday", "Sunday"] and service_location != "23"
            ):  # Not emergency room
                evidence.append(f"Service on {day_of_week} for non-emergency location")
                score += 0.4

            # Check for holiday services
            if self._is_holiday(service_date):
                evidence.append(f"Service on holiday: {service_date.strftime('%Y-%m-%d')}")
                score += 0.3

            # Check for suspicious time patterns
            if "time_of_service" in claim:
                hour = int(claim["time_of_service"].split(":")[0])
                normal_hours = self.thresholds["phantom_billing_hours"]
                if hour < normal_hours["start"] or hour > normal_hours["end"]:
                    evidence.append(f"Service at {claim['time_of_service']} outside normal hours")
                    score += 0.5

        except ValueError as e:
            evidence.append(f"Invalid date format: {date_of_service}")
            score += 0.2

        # Check for non-existent patient addresses
        red_flags = claim.get("red_flags", [])
        for flag in red_flags:
            if "address" in flag.lower() and "exist" in flag.lower():
                evidence.append("Patient address validation failed")
                score += 0.6

        triggered = score >= rule.threshold
        details = f"Phantom billing schedule score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_phantom_billing_location(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for services at impossible or suspicious locations."""
        evidence = []
        score = 0.0

        service_location = claim.get("service_location", "")
        provider_id = claim.get("provider_id", "")
        patient_id = claim.get("patient_id", "")

        # Check for ghost patients/providers
        if "GHOST" in patient_id or "FRAUD" in provider_id:
            evidence.append("Ghost patient or fraudulent provider ID detected")
            score += 0.9

        # Check for impossible location combinations
        impossible_combinations = [
            ("11", "23"),  # Office and ER on same day
            ("21", "11"),  # Inpatient and office same day
        ]

        # Check red flags for location issues
        red_flags = claim.get("red_flags", [])
        for flag in red_flags:
            if any(word in flag.lower() for word in ["appointment", "records", "closed"]):
                evidence.append(f"Location issue: {flag}")
                score += 0.3

        triggered = score >= rule.threshold
        details = f"Phantom billing location score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_unbundling(
        self, rule: FraudRule, claim: Dict[str, Any], context_claims: List[Dict[str, Any]]
    ) -> RuleResult:
        """Check for artificially separated procedures (unbundling)."""
        evidence = []
        score = 0.0

        procedure_codes = claim.get("procedure_codes", [])
        claim_date = claim.get("date_of_service", "")
        patient_id = claim.get("patient_id", "")
        provider_id = claim.get("provider_id", "")

        # Known bundled procedure groups
        bundled_groups = {
            "colonoscopy": ["45378", "45380", "45384", "45385"],
            "cataract_surgery": ["66984", "66982", "66983"],
            "knee_arthroscopy": ["29881", "29882", "29883"],
            "cardiac_cath": ["93454", "93455", "93456", "93457"],
        }

        # Check for procedures that should be bundled
        for group_name, codes in bundled_groups.items():
            if len(set(procedure_codes) & set(codes)) > 1:
                evidence.append(f"Multiple {group_name} procedures that should be bundled")
                score += 0.4

        # Check for same procedures across multiple claims on same day
        if context_claims:
            same_day_claims = [
                c
                for c in context_claims
                if (
                    c.get("date_of_service") == claim_date
                    and c.get("patient_id") == patient_id
                    and c.get("provider_id") == provider_id
                    and c.get("claim_id") != claim.get("claim_id")
                )
            ]

            for other_claim in same_day_claims:
                other_procedures = other_claim.get("procedure_codes", [])
                common_procedures = set(procedure_codes) & set(other_procedures)
                if common_procedures:
                    evidence.append(
                        f"Duplicate procedures across claims: {list(common_procedures)}"
                    )
                    score += 0.5

        # Check for excessive number of procedures
        if len(procedure_codes) > self.thresholds["max_procedures_per_claim"]:
            evidence.append(f"Excessive procedures in single claim: {len(procedure_codes)}")
            score += 0.3

        triggered = score >= rule.threshold
        details = f"Unbundling score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_staged_accident(
        self, rule: FraudRule, claim: Dict[str, Any], context_claims: List[Dict[str, Any]]
    ) -> RuleResult:
        """Check for staged accident patterns."""
        evidence = []
        score = 0.0

        diagnosis_codes = claim.get("diagnosis_codes", [])
        claim_type = claim.get("claim_type", "")

        # Auto accident injury codes
        auto_injury_codes = ["S72.001A", "S06.0X0A", "M99.23", "M54.2"]

        if claim_type == "auto" or any(code in auto_injury_codes for code in diagnosis_codes):
            # Check for suspicious patterns in auto claims

            # Multiple similar accidents
            if context_claims:
                similar_accidents = [
                    c
                    for c in context_claims
                    if (
                        c.get("claim_type") == "auto"
                        and len(set(c.get("diagnosis_codes", [])) & set(diagnosis_codes)) > 0
                    )
                ]

                if len(similar_accidents) > 3:
                    evidence.append(f"Multiple similar auto accidents: {len(similar_accidents)}")
                    score += 0.4

            # Check for pre-existing relationships
            red_flags = claim.get("red_flags", [])
            for flag in red_flags:
                if any(word in flag.lower() for word in ["relationship", "prior", "staged"]):
                    evidence.append(f"Staged accident indicator: {flag}")
                    score += 0.5

            # Check for consistent injury patterns
            if self._has_consistent_injury_pattern(diagnosis_codes):
                evidence.append("Consistent injury pattern across multiple claims")
                score += 0.3

        triggered = score >= rule.threshold
        details = f"Staged accident score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_prescription_fraud(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for prescription fraud patterns."""
        evidence = []
        score = 0.0

        procedure_codes = claim.get("procedure_codes", [])
        diagnosis_codes = claim.get("diagnosis_codes", [])

        # Prescription-related procedure codes
        prescription_codes = ["J1100", "J2001", "J3420", "J7799"]
        controlled_substance_codes = ["J2315", "J1170", "J2405"]

        # Check for excessive prescription volumes
        prescription_count = len([code for code in procedure_codes if code in prescription_codes])
        if prescription_count > self.thresholds["prescription_daily_limit"]:
            evidence.append(f"Excessive prescription count: {prescription_count}")
            score += 0.5

        # Check for controlled substances with inadequate diagnosis
        for code in procedure_codes:
            if code in controlled_substance_codes:
                if not self._has_adequate_diagnosis_for_controlled_substance(diagnosis_codes):
                    evidence.append(f"Controlled substance {code} without adequate diagnosis")
                    score += 0.6

        # Check for doctor shopping patterns
        patient_id = claim.get("patient_id", "")
        if self._shows_doctor_shopping_pattern(patient_id):
            evidence.append("Doctor shopping pattern detected")
            score += 0.7

        triggered = score >= rule.threshold
        details = f"Prescription fraud score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_kickback_scheme(
        self, rule: FraudRule, claim: Dict[str, Any], context_claims: List[Dict[str, Any]]
    ) -> RuleResult:
        """Check for kickback and referral fraud patterns."""
        evidence = []
        score = 0.0

        provider_id = claim.get("provider_id", "")

        # Check for excessive referral concentration
        if context_claims:
            referral_pattern = self._analyze_referral_patterns(provider_id, context_claims)
            if (
                referral_pattern["concentration"]
                > self.thresholds["referral_concentration_threshold"]
            ):
                evidence.append(
                    f"High referral concentration: {referral_pattern['concentration']:.2f}"
                )
                score += 0.4

        # Check for circular referral patterns
        if self._has_circular_referrals(provider_id, context_claims):
            evidence.append("Circular referral pattern detected")
            score += 0.5

        # Check for suspicious billing relationships
        red_flags = claim.get("red_flags", [])
        for flag in red_flags:
            if any(word in flag.lower() for word in ["kickback", "referral", "relationship"]):
                evidence.append(f"Kickback indicator: {flag}")
                score += 0.6

        triggered = score >= rule.threshold
        details = f"Kickback scheme score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_billing_frequency(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for unusual billing frequency patterns."""
        evidence = []
        score = 0.0

        provider_id = claim.get("provider_id", "")
        patient_id = claim.get("patient_id", "")
        date_of_service = claim.get("date_of_service", "")

        # Check provider's daily claim volume
        provider_claims_today = len(
            [
                c
                for c in self.provider_history[provider_id]
                if c.get("date_of_service") == date_of_service
            ]
        )

        if provider_claims_today > self.thresholds["max_daily_claims_per_provider"]:
            evidence.append(f"Excessive daily claims: {provider_claims_today}")
            score += 0.4

        # Check patient visit frequency
        patient_visits_today = len(
            [
                c
                for c in self.patient_history[patient_id]
                if c.get("date_of_service") == date_of_service
            ]
        )

        if patient_visits_today > self.thresholds["max_patient_visits_per_day"]:
            evidence.append(f"Excessive patient visits: {patient_visits_today}")
            score += 0.3

        triggered = score >= rule.threshold
        details = f"Billing frequency score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _check_amount_anomaly(self, rule: FraudRule, claim: Dict[str, Any]) -> RuleResult:
        """Check for suspicious billing amounts."""
        evidence = []
        score = 0.0

        billed_amount = claim.get("billed_amount", 0)
        procedure_codes = claim.get("procedure_codes", [])

        # Check for excessive amounts
        if billed_amount > self.thresholds["max_amount_per_claim"]:
            evidence.append(f"Excessive claim amount: ${billed_amount}")
            score += 0.5

        # Check amount vs procedure consistency
        expected_amount = self._calculate_expected_amount(procedure_codes)
        if expected_amount > 0 and billed_amount > expected_amount * 2:
            evidence.append(
                f"Amount ${billed_amount} significantly exceeds expected ${expected_amount}"
            )
            score += 0.4

        # Check for round number bias
        if billed_amount % 100 == 0 and billed_amount > 500:
            evidence.append(f"Suspiciously round billing amount: ${billed_amount}")
            score += 0.2

        triggered = score >= rule.threshold
        details = f"Amount anomaly score: {score:.2f}"

        return RuleResult(rule.name, triggered, score, details, evidence)

    def _calculate_fraud_score(self, results: List[RuleResult]) -> float:
        """Calculate overall fraud score from rule results."""
        if not results:
            return 0.0

        weighted_score = 0.0
        total_weight = 0.0

        for result in results:
            if result.rule_name in self.rules:
                rule_weight = self.rules[result.rule_name].weight
                weighted_score += result.score * rule_weight
                total_weight += rule_weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _update_claim_history(self, claim: Dict[str, Any]) -> None:
        """Update claim history for pattern analysis."""
        provider_id = claim.get("provider_id", "")
        patient_id = claim.get("patient_id", "")

        if provider_id:
            self.provider_history[provider_id].append(claim)
            # Keep only recent history (last 100 claims)
            if len(self.provider_history[provider_id]) > 100:
                self.provider_history[provider_id] = self.provider_history[provider_id][-100:]

        if patient_id:
            self.patient_history[patient_id].append(claim)
            # Keep only recent history (last 50 claims)
            if len(self.patient_history[patient_id]) > 50:
                self.patient_history[patient_id] = self.patient_history[patient_id][-50:]

    def _has_suspicious_procedure_progression(self, procedure_codes: List[str]) -> bool:
        """Check for suspicious procedure code progressions."""
        # Sequential procedure codes might indicate upcoding
        if len(procedure_codes) < 2:
            return False

        numeric_codes = []
        for code in procedure_codes:
            try:
                numeric_codes.append(int(re.sub(r"\D", "", code)))
            except ValueError:
                continue

        if len(numeric_codes) < 2:
            return False

        # Check for sequential patterns
        sorted_codes = sorted(numeric_codes)
        sequential_count = 0
        for i in range(1, len(sorted_codes)):
            if sorted_codes[i] - sorted_codes[i - 1] == 1:
                sequential_count += 1

        return sequential_count >= len(sorted_codes) - 1

    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday."""
        # Simple holiday check - can be expanded
        holidays = [(1, 1), (7, 4), (12, 25)]  # New Year's Day  # Independence Day  # Christmas
        return (date.month, date.day) in holidays

    def _has_consistent_injury_pattern(self, diagnosis_codes: List[str]) -> bool:
        """Check for consistent injury patterns across claims."""
        # This would be enhanced with historical data analysis
        common_staged_patterns = [
            ["S72.001A", "S06.0X0A"],  # Fracture + head injury
            ["M99.23", "M54.2"],  # Spinal issues
        ]

        for pattern in common_staged_patterns:
            if all(code in diagnosis_codes for code in pattern):
                return True

        return False

    def _has_adequate_diagnosis_for_controlled_substance(self, diagnosis_codes: List[str]) -> bool:
        """Check if diagnosis supports controlled substance prescription."""
        adequate_diagnoses = ["M79.3", "G89.29", "F32.9"]  # Myalgia  # Chronic pain  # Depression
        return any(code in adequate_diagnoses for code in diagnosis_codes)

    def _shows_doctor_shopping_pattern(self, patient_id: str) -> bool:
        """Check for doctor shopping patterns."""
        if patient_id not in self.patient_history:
            return False

        recent_claims = self.patient_history[patient_id][-30:]  # Last 30 claims
        providers = set(claim.get("provider_id", "") for claim in recent_claims)

        # More than 5 different providers in recent history might indicate shopping
        return len(providers) > 5

    def _analyze_referral_patterns(
        self, provider_id: str, context_claims: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze referral patterns for kickback detection."""
        if not context_claims:
            return {"concentration": 0.0}

        # Count referrals by target provider
        referral_counts = Counter()
        total_referrals = 0

        for claim in context_claims:
            if claim.get("provider_id") == provider_id:
                # Look for referral indicators in procedure codes
                procedure_codes = claim.get("procedure_codes", [])
                if any("99" in code for code in procedure_codes):  # Consultation codes
                    referred_to = claim.get("referred_to_provider", "")
                    if referred_to:
                        referral_counts[referred_to] += 1
                        total_referrals += 1

        if total_referrals == 0:
            return {"concentration": 0.0}

        # Calculate concentration (max referrals to single provider / total)
        max_referrals = max(referral_counts.values()) if referral_counts else 0
        concentration = max_referrals / total_referrals

        return {"concentration": concentration}

    def _has_circular_referrals(
        self, provider_id: str, context_claims: List[Dict[str, Any]]
    ) -> bool:
        """Check for circular referral patterns."""
        # This would require more sophisticated graph analysis
        # Simplified implementation
        referral_network = defaultdict(set)

        for claim in context_claims:
            referring_provider = claim.get("provider_id", "")
            referred_to = claim.get("referred_to_provider", "")
            if referring_provider and referred_to:
                referral_network[referring_provider].add(referred_to)

        # Check for simple circular patterns
        for target in referral_network.get(provider_id, set()):
            if provider_id in referral_network.get(target, set()):
                return True

        return False

    def _calculate_expected_amount(self, procedure_codes: List[str]) -> float:
        """Calculate expected amount based on procedure codes."""
        # Simplified fee schedule
        fee_schedule = {
            "99213": 125.0,
            "99214": 185.0,
            "99215": 250.0,
            "99285": 350.0,
            "71046": 85.0,
            "93000": 45.0,
        }

        total = 0.0
        for code in procedure_codes:
            total += fee_schedule.get(code, 100.0)  # Default fee

        return total

    def generate_explanation(self, results: List[RuleResult], fraud_score: float) -> str:
        """Generate human-readable explanation of fraud detection results."""
        if fraud_score < 0.3:
            risk_level = "Low"
        elif fraud_score < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        explanation = f"Fraud Risk Assessment: {risk_level} (Score: {fraud_score:.2f})\n\n"

        triggered_rules = [r for r in results if r.triggered]
        if triggered_rules:
            explanation += "Triggered Fraud Rules:\n"
            for result in triggered_rules:
                explanation += f"- {result.rule_name}: {result.details}\n"
                for evidence in result.evidence:
                    explanation += f"  • {evidence}\n"
        else:
            explanation += "No fraud rules triggered.\n"

        return explanation

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule performance."""
        stats = {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "rules_by_type": defaultdict(int),
        }

        for rule in self.rules.values():
            stats["rules_by_type"][rule.fraud_type] += 1

        return stats
