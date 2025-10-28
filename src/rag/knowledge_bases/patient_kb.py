"""
Patient Claim History Knowledge Base.

Tracks patient claim patterns, detects doctor shopping, and identifies temporal anomalies.

Components:
- PatientClaimDocument: Pydantic model for patient history
- PatientHistoryBuilder: Load and process patient data
- PatientHistoryRetriever: Find similar patient patterns
- PatientClaimHistoryKB: Complete knowledge base implementation
"""

import json
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
from qdrant_client import QdrantClient

from src.rag.knowledge_bases.base_kb import BaseKnowledgeBase, KBDocument

logger = logging.getLogger(__name__)


class PatientClaimDocument(BaseModel):
    """Patient claim history document with fraud indicators."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patient_id: str = Field(..., description="Unique patient identifier (hashed)")
    claim_sequence: List[Dict[str, Any]] = Field(
        ..., description="Chronological sequence of claims"
    )
    patient_patterns: Dict[str, Any] = Field(
        ..., description="Derived behavioral patterns"
    )
    red_flags: List[str] = Field(
        default_factory=list, description="List of red flag indicators"
    )
    temporal_analysis: Dict[str, Any] = Field(
        ..., description="Temporal pattern analysis"
    )

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        """Validate patient_id is not empty."""
        if not v or len(v) == 0:
            raise ValueError("patient_id cannot be empty")
        return v

    def generate_embedding_text(self) -> str:
        """
        Generate text representation for embedding.

        Returns:
            Natural language description of patient history
        """
        # Extract key information
        num_claims = len(self.claim_sequence)
        num_providers = self.patient_patterns.get("provider_count_90d", 0)
        avg_amount = self.patient_patterns.get("avg_claim_amount", 0)

        # Get top diagnoses
        all_diagnoses = []
        for claim in self.claim_sequence:
            all_diagnoses.extend(claim.get("diagnosis_codes", []))
        top_diagnoses = [code for code, _ in Counter(all_diagnoses).most_common(3)]

        # Get top procedures
        all_procedures = []
        for claim in self.claim_sequence:
            all_procedures.extend(claim.get("procedure_codes", []))
        top_procedures = [code for code, _ in Counter(all_procedures).most_common(3)]

        # Calculate time span
        start_date = self.temporal_analysis.get("date_range_start", "")
        end_date = self.temporal_analysis.get("date_range_end", "")

        # Build narrative
        parts = [
            f"Patient has {num_claims} claims",
            f"across {num_providers} providers" if num_providers != 1 else "with 1 provider",
        ]

        if start_date and end_date:
            start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            days = (end - start).days
            parts.append(f"over {days} days")

        if top_diagnoses:
            parts.append(f"Primary diagnoses: {', '.join(top_diagnoses)}")

        if top_procedures:
            parts.append(f"Common procedures: {', '.join(top_procedures)}")

        parts.append(f"Average claim amount ${avg_amount:.2f}")

        if self.red_flags:
            parts.append(f"Red flags: {'; '.join(self.red_flags)}")
        else:
            parts.append("No red flags detected")

        return ". ".join(parts) + "."


class PatientHistoryBuilder:
    """Build patient history documents from raw claims data."""

    def __init__(self):
        """Initialize builder."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_patient_document(self, patient_data: Dict[str, Any]) -> PatientClaimDocument:
        """
        Build PatientClaimDocument from raw data.

        Args:
            patient_data: Dict containing patient_id, claim_sequence, etc.

        Returns:
            PatientClaimDocument instance
        """
        # If patterns not pre-calculated, calculate them
        if "patient_patterns" not in patient_data:
            patient_data["patient_patterns"] = self.calculate_patient_patterns(
                patient_data["claim_sequence"]
            )

        # If temporal analysis not provided, calculate it
        if "temporal_analysis" not in patient_data:
            patient_data["temporal_analysis"] = self.analyze_temporal_patterns(
                patient_data["claim_sequence"]
            )

        # Detect red flags
        if "red_flags" not in patient_data or not patient_data["red_flags"]:
            patient_data["red_flags"] = self.detect_red_flags(
                patient_data["patient_patterns"], patient_data["temporal_analysis"]
            )

        return PatientClaimDocument(**patient_data)

    def calculate_patient_patterns(self, claim_sequence: List[Dict]) -> Dict[str, Any]:
        """
        Calculate patient behavioral patterns.

        Args:
            claim_sequence: List of claim dictionaries

        Returns:
            Dict with pattern metrics
        """
        if not claim_sequence:
            return self._empty_patterns()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(claim_sequence)
        df["date_of_service"] = pd.to_datetime(df["date_of_service"])

        # Current date reference (use latest claim date)
        current_date = df["date_of_service"].max()

        # Calculate time windows
        date_30d = current_date - pd.Timedelta(days=30)
        date_90d = current_date - pd.Timedelta(days=90)

        df_30d = df[df["date_of_service"] >= date_30d]
        df_90d = df[df["date_of_service"] >= date_90d]

        # Calculate metrics
        patterns = {
            "provider_count_30d": df_30d["provider_npi"].nunique() if len(df_30d) > 0 else 0,
            "provider_count_90d": df_90d["provider_npi"].nunique() if len(df_90d) > 0 else 0,
            "total_claims_30d": len(df_30d),
            "total_claims_90d": len(df_90d),
            "avg_claim_amount": df["billed_amount"].mean() if len(df) > 0 else 0.0,
            "diagnosis_diversity": self._calculate_diversity(df, "diagnosis_codes"),
            "pharmacy_count_30d": 0,  # Would need prescription data
            "controlled_substance_prescriptions": 0,  # Would need drug codes
            "early_refill_count": 0,  # Would need prescription data
        }

        return patterns

    def analyze_temporal_patterns(self, claim_sequence: List[Dict]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in claims.

        Args:
            claim_sequence: List of claim dictionaries

        Returns:
            Dict with temporal metrics
        """
        if not claim_sequence:
            return {
                "date_range_start": None,
                "date_range_end": None,
                "claim_frequency_days": 0.0,
                "geographic_impossibility_count": 0,
            }

        df = pd.DataFrame(claim_sequence)
        df["date_of_service"] = pd.to_datetime(df["date_of_service"])
        df = df.sort_values("date_of_service")

        # Calculate date ranges
        start_date = df["date_of_service"].min()
        end_date = df["date_of_service"].max()

        # Calculate frequency
        if len(df) > 1:
            total_days = (end_date - start_date).days
            claim_frequency = total_days / (len(df) - 1) if len(df) > 1 else 0.0
        else:
            claim_frequency = 0.0

        return {
            "date_range_start": start_date.isoformat(),
            "date_range_end": end_date.isoformat(),
            "claim_frequency_days": claim_frequency,
            "geographic_impossibility_count": 0,  # Would need location data
        }

    def detect_red_flags(
        self, patient_patterns: Dict[str, Any], temporal_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Detect red flag indicators for fraud.

        Args:
            patient_patterns: Patient behavioral patterns
            temporal_analysis: Temporal metrics

        Returns:
            List of red flag descriptions
        """
        red_flags = []

        # Doctor shopping (5+ providers in 30 days)
        provider_count_30d = patient_patterns.get("provider_count_30d", 0)
        if provider_count_30d >= 5:
            red_flags.append(f"Doctor shopping: {provider_count_30d} providers in 30 days")

        # Pharmacy hopping (3+ pharmacies in 30 days)
        pharmacy_count_30d = patient_patterns.get("pharmacy_count_30d", 0)
        if pharmacy_count_30d >= 3:
            red_flags.append(f"Pharmacy hopping: {pharmacy_count_30d} pharmacies in 30 days")

        # Early refills (2+ instances)
        early_refill_count = patient_patterns.get("early_refill_count", 0)
        if early_refill_count >= 2:
            red_flags.append(f"Early refills: {early_refill_count} instances")

        # Excessive controlled substances
        controlled_count = patient_patterns.get("controlled_substance_prescriptions", 0)
        if controlled_count >= 4:
            red_flags.append(
                f"Excessive controlled substances: {controlled_count} prescriptions in 90 days"
            )

        # Geographic impossibilities
        geo_impossible = temporal_analysis.get("geographic_impossibility_count", 0)
        if geo_impossible > 0:
            red_flags.append(f"Geographic impossibilities: {geo_impossible} detected")

        return red_flags

    def _calculate_diversity(self, df: pd.DataFrame, column: str) -> float:
        """Calculate diversity score for a list column."""
        if len(df) == 0:
            return 0.0

        # Extract all values from list column
        all_values = []
        for values in df[column]:
            if isinstance(values, list):
                all_values.extend(values)

        if len(all_values) == 0:
            return 0.0

        # Calculate diversity as unique_values / total_values
        unique_count = len(set(all_values))
        total_count = len(all_values)

        return unique_count / total_count if total_count > 0 else 0.0

    def _empty_patterns(self) -> Dict[str, Any]:
        """Return empty pattern dict."""
        return {
            "provider_count_30d": 0,
            "provider_count_90d": 0,
            "total_claims_30d": 0,
            "total_claims_90d": 0,
            "avg_claim_amount": 0.0,
            "diagnosis_diversity": 0.0,
            "pharmacy_count_30d": 0,
            "controlled_substance_prescriptions": 0,
            "early_refill_count": 0,
        }


class PatientHistoryRetriever:
    """Retrieve and search patient history patterns."""

    def __init__(self, qdrant_client: QdrantClient, openai_api_key: str):
        """
        Initialize retriever.

        Args:
            qdrant_client: Qdrant client instance
            openai_api_key: OpenAI API key
        """
        self.kb = PatientClaimHistoryKB(
            qdrant_client=qdrant_client, openai_api_key=openai_api_key
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def index_patients(self, documents: List[PatientClaimDocument]) -> None:
        """
        Index patient documents.

        Args:
            documents: List of PatientClaimDocument instances
        """
        # Convert to KBDocument format
        kb_docs = []
        for doc in documents:
            kb_doc = KBDocument(
                id=doc.patient_id,
                embedding_text=doc.generate_embedding_text(),
                metadata={
                    "patient_patterns": doc.patient_patterns,
                    "red_flags": doc.red_flags,
                    "temporal_analysis": doc.temporal_analysis,
                },
            )
            kb_docs.append(kb_doc)

        self.kb.upsert_documents(kb_docs)

    def find_similar_patterns(
        self, query_doc: PatientClaimDocument, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find patients with similar patterns.

        Args:
            query_doc: Query patient document
            limit: Maximum number of results

        Returns:
            List of similar patient records
        """
        query_text = query_doc.generate_embedding_text()
        return self.kb.search(query_text=query_text, limit=limit)


class PatientClaimHistoryKB(BaseKnowledgeBase):
    """Patient Claim History Knowledge Base."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_api_key: str,
        collection_name: str = "patient_claim_history",
    ):
        """
        Initialize Patient Claim History KB.

        Args:
            qdrant_client: Qdrant client instance
            openai_api_key: OpenAI API key
            collection_name: Name of Qdrant collection
        """
        super().__init__(
            collection_name=collection_name,
            qdrant_client=qdrant_client,
            openai_api_key=openai_api_key,
            vector_size=1536,
        )
        self.builder = PatientHistoryBuilder()

    def build(self, data_source: str) -> None:
        """
        Build KB from JSON file of patient claims.

        Args:
            data_source: Path to JSON file with patient data
        """
        logger.info(f"Building Patient Claim History KB from {data_source}")

        # Load data
        with open(data_source, "r") as f:
            patient_records = json.load(f)

        # Build documents
        documents = []
        for record in patient_records:
            doc = self.builder.build_patient_document(record)

            # Convert to KBDocument
            kb_doc = KBDocument(
                id=doc.patient_id,
                embedding_text=doc.generate_embedding_text(),
                metadata={
                    "patient_patterns": doc.patient_patterns,
                    "red_flags": doc.red_flags,
                    "temporal_analysis": doc.temporal_analysis,
                    "claim_count": len(doc.claim_sequence),
                },
            )
            documents.append(kb_doc)

        # Index documents
        self.upsert_documents(documents)

        logger.info(f"Indexed {len(documents)} patient records")

    def validate(self) -> bool:
        """
        Validate KB completeness.

        Returns:
            True if validation passes
        """
        stats = self.get_statistics()

        # Check minimum documents
        if stats.total_documents == 0:
            logger.warning("No documents in collection")
            return False

        # Check vector dimensions
        if stats.vector_dimensions != 1536:
            logger.error(f"Invalid vector dimensions: {stats.vector_dimensions}")
            return False

        logger.info(f"Validation passed: {stats.total_documents} documents indexed")
        return True
