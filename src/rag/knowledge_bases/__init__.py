"""
Knowledge Base implementations for RAG enrichment system.

This package contains 4 production-ready knowledge bases:
- PatientClaimHistoryKB: Patient claim patterns and doctor shopping detection
- ProviderBehaviorPatternKB: Provider billing patterns and anomaly detection
- MedicalCodingStandardsKB: ICD-10/CPT validation and bundling rules
- RegulatoryGuidanceKB: Fraud patterns and regulatory compliance

All KBs follow TDD approach with >90% test coverage.
"""

from src.rag.knowledge_bases.base_kb import BaseKnowledgeBase, KBDocument, KBStatistics
from src.rag.knowledge_bases.patient_kb import (
    PatientClaimHistoryKB,
    PatientClaimDocument,
    PatientHistoryBuilder,
    PatientHistoryRetriever,
)
from src.rag.knowledge_bases.provider_kb import (
    ProviderBehaviorPatternKB,
    ProviderProfileDocument,
    ProviderPatternBuilder,
    ProviderPatternRetriever,
)
from src.rag.knowledge_bases.medical_coding_kb import (
    MedicalCodingStandardsKB,
    MedicalCodeDocument,
    MedicalCodingBuilder,
    MedicalCodingValidator,
    BundlingRuleChecker,
)
from src.rag.knowledge_bases.regulatory_kb import (
    RegulatoryGuidanceKB,
    FraudPatternDocument,
    RegulatoryGuidanceDocument,
    RegulationBuilder,
    FraudPatternRetriever,
)

__all__ = [
    # Base classes
    "BaseKnowledgeBase",
    "KBDocument",
    "KBStatistics",
    # Patient KB
    "PatientClaimHistoryKB",
    "PatientClaimDocument",
    "PatientHistoryBuilder",
    "PatientHistoryRetriever",
    # Provider KB
    "ProviderBehaviorPatternKB",
    "ProviderProfileDocument",
    "ProviderPatternBuilder",
    "ProviderPatternRetriever",
    # Medical Coding KB
    "MedicalCodingStandardsKB",
    "MedicalCodeDocument",
    "MedicalCodingBuilder",
    "MedicalCodingValidator",
    "BundlingRuleChecker",
    # Regulatory KB
    "RegulatoryGuidanceKB",
    "FraudPatternDocument",
    "RegulatoryGuidanceDocument",
    "RegulationBuilder",
    "FraudPatternRetriever",
]
