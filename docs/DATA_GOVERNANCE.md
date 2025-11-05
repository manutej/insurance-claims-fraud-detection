# Data Governance for Insurance Claims Fraud Detection System

## Executive Summary

This document establishes comprehensive data governance policies for the insurance fraud detection system, covering knowledge base versioning, enrichment decision tracking, audit trails for regulatory compliance, and complete data lineage. These policies ensure transparency, accountability, and regulatory compliance (HIPAA, SOX, state insurance regulations).

## 1. Data Governance Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Data Governance Architecture                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│   Data Lineage       │   │   Version Control    │   │   Audit Trail        │
│   Tracking           │   │   Management         │   │   System             │
│                      │   │                      │   │                      │
│  • Source tracking   │   │  • KB versions       │   │  • All decisions     │
│  • Transformation    │   │  • Model versions    │   │  • User actions      │
│    logging           │   │  • Config versions   │   │  • System events     │
│  • Data provenance   │   │  • Schema versions   │   │  • Access logs       │
└──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
           │                          │                           │
           └──────────────────────────┴───────────────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                            │
    ┌───────────▼───────────┐              ┌───────────────▼──────────┐
    │  Compliance Engine    │              │  Data Quality Management │
    │                       │              │                          │
    │  • HIPAA compliance   │              │  • Quality metrics       │
    │  • SOX compliance     │              │  • Validation rules      │
    │  • State regulations  │              │  • Error detection       │
    │  • Retention policies │              │  • Remediation tracking  │
    └───────────────────────┘              └──────────────────────────┘
```

## 2. Knowledge Base Versioning

### 2.1 Version Control Strategy

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

class VersionType(Enum):
    MAJOR = "major"  # Breaking changes (e.g., schema change)
    MINOR = "minor"  # New features (e.g., new fraud patterns)
    PATCH = "patch"  # Bug fixes (e.g., corrected embeddings)

@dataclass
class KnowledgeBaseVersion:
    """
    Tracks versions of the fraud detection knowledge base.
    """
    version: str  # Semantic versioning: major.minor.patch
    version_type: VersionType
    created_at: datetime
    created_by: str
    description: str
    changes: List[Dict[str, any]]
    schema_version: str
    embedding_model_version: str
    total_claims: int
    fraud_claims: int
    legitimate_claims: int
    checksum: str  # SHA-256 of vector store snapshot

    # Metadata
    git_commit: Optional[str] = None
    deployment_environment: str = "staging"  # staging, production
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None

    # Compatibility
    compatible_with: List[str] = None  # List of compatible system versions
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None


class KnowledgeBaseVersionManager:
    """
    Manages knowledge base versions and changes.
    """
    def __init__(self):
        self.version_db = VersionDatabase()
        self.snapshot_service = SnapshotService()

    async def create_version(
        self,
        changes: List[Dict[str, any]],
        version_type: VersionType,
        description: str,
        created_by: str
    ) -> KnowledgeBaseVersion:
        """
        Create new knowledge base version.
        """
        # Get current version
        current_version = await self.version_db.get_latest_version()

        # Calculate new version number
        new_version = self._increment_version(
            current_version.version if current_version else "0.0.0",
            version_type
        )

        # Create snapshot of vector store
        snapshot = await self.snapshot_service.create_snapshot(
            collection_name="fraud_claims_kb"
        )

        # Count current claims
        stats = await self._get_kb_statistics()

        # Create version record
        version = KnowledgeBaseVersion(
            version=new_version,
            version_type=version_type,
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description,
            changes=changes,
            schema_version=settings.SCHEMA_VERSION,
            embedding_model_version=settings.EMBEDDING_MODEL,
            total_claims=stats['total'],
            fraud_claims=stats['fraud'],
            legitimate_claims=stats['legitimate'],
            checksum=snapshot.checksum,
            deployment_environment=settings.ENVIRONMENT
        )

        # Store version record
        await self.version_db.save_version(version)

        # Tag snapshot with version
        await self.snapshot_service.tag_snapshot(
            snapshot.snapshot_id,
            f"v{new_version}"
        )

        logger.info(f"Created KB version {new_version}")

        return version

    def _increment_version(
        self,
        current: str,
        version_type: VersionType
    ) -> str:
        """
        Increment version number according to semantic versioning.
        """
        major, minor, patch = map(int, current.split('.'))

        if version_type == VersionType.MAJOR:
            return f"{major + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{major}.{minor + 1}.0"
        else:  # PATCH
            return f"{major}.{minor}.{patch + 1}"

    async def rollback_to_version(
        self,
        version: str,
        reason: str,
        initiated_by: str
    ) -> RollbackResult:
        """
        Rollback knowledge base to specific version.
        """
        # Get version record
        target_version = await self.version_db.get_version(version)

        if not target_version:
            raise ValueError(f"Version {version} not found")

        # Find snapshot
        snapshot = await self.snapshot_service.get_snapshot_by_tag(
            f"v{version}"
        )

        # Restore snapshot
        await self.snapshot_service.restore_snapshot(
            snapshot_id=snapshot.snapshot_id,
            collection_name="fraud_claims_kb"
        )

        # Log rollback
        await self.version_db.log_rollback(
            from_version=await self.version_db.get_latest_version(),
            to_version=target_version,
            reason=reason,
            initiated_by=initiated_by,
            timestamp=datetime.utcnow()
        )

        return RollbackResult(
            success=True,
            previous_version=target_version.version,
            rolled_back_to=version,
            timestamp=datetime.utcnow()
        )

    async def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> VersionComparison:
        """
        Compare two knowledge base versions.
        """
        v1 = await self.version_db.get_version(version1)
        v2 = await self.version_db.get_version(version2)

        return VersionComparison(
            version1=v1,
            version2=v2,
            claims_added=v2.total_claims - v1.total_claims,
            fraud_claims_added=v2.fraud_claims - v1.fraud_claims,
            changes=self._diff_changes(v1.changes, v2.changes),
            schema_changes=v1.schema_version != v2.schema_version,
            embedding_model_changes=v1.embedding_model_version != v2.embedding_model_version
        )
```

### 2.2 Version Tagging and Metadata

```python
class VersionMetadata:
    """
    Extended metadata for KB versions.
    """
    def __init__(self):
        self.metadata_store = MetadataStore()

    async def tag_version(
        self,
        version: str,
        tags: List[str],
        metadata: Dict[str, any]
    ) -> None:
        """
        Add tags and metadata to version.
        """
        await self.metadata_store.upsert({
            "version": version,
            "tags": tags,
            "metadata": metadata,
            "tagged_at": datetime.utcnow()
        })

    async def get_versions_by_tag(
        self,
        tag: str
    ) -> List[KnowledgeBaseVersion]:
        """
        Retrieve versions by tag.
        """
        return await self.metadata_store.query({
            "tags": {"$contains": tag}
        })


# Example usage
"""
# Creating a new version
version_manager = KnowledgeBaseVersionManager()

await version_manager.create_version(
    changes=[
        {
            "type": "claims_added",
            "count": 5000,
            "fraud_type": "upcoding",
            "date_range": "2024-01-01 to 2024-03-31"
        },
        {
            "type": "embeddings_regenerated",
            "reason": "Updated to OpenAI text-embedding-3-large",
            "affected_claims": 15000
        }
    ],
    version_type=VersionType.MINOR,
    description="Added Q1 2024 fraud patterns, updated embedding model",
    created_by="data-engineering-team"
)

# Tagging version
await version_metadata.tag_version(
    version="1.5.0",
    tags=["production", "q1-2024", "validated"],
    metadata={
        "validation_accuracy": 0.94,
        "validation_date": "2024-04-01",
        "deployment_date": "2024-04-05"
    }
)
"""
```

## 3. Enrichment Decision Tracking

### 3.1 Enrichment Audit Log

```python
from enum import Enum

class EnrichmentSource(Enum):
    NPI_REGISTRY = "npi_registry"
    PATIENT_HISTORY = "patient_history"
    FRAUD_PATTERNS = "fraud_patterns"
    MEDICAL_CODE_LOOKUP = "medical_code_lookup"
    GEOGRAPHIC_DATA = "geographic_data"
    PROVIDER_NETWORK = "provider_network"

@dataclass
class EnrichmentDecision:
    """
    Records enrichment decision for a claim.
    """
    claim_id: str
    enrichment_id: str  # Unique ID for this enrichment
    timestamp: datetime
    source: EnrichmentSource
    data_added: Dict[str, any]
    confidence_score: float  # 0-1
    decision_rationale: str
    data_quality_score: float
    user_or_system: str  # Who/what made the decision

    # Provenance
    source_system: str
    source_record_id: Optional[str]
    source_timestamp: Optional[datetime]

    # Validation
    validated: bool = False
    validation_timestamp: Optional[datetime] = None
    validation_result: Optional[str] = None


class EnrichmentTracker:
    """
    Tracks all enrichment decisions and data sources.
    """
    def __init__(self):
        self.enrichment_db = EnrichmentDatabase()

    async def log_enrichment(
        self,
        claim_id: str,
        source: EnrichmentSource,
        data_added: Dict[str, any],
        confidence_score: float,
        decision_rationale: str,
        source_system: str,
        source_record_id: Optional[str] = None
    ) -> EnrichmentDecision:
        """
        Log an enrichment decision.
        """
        enrichment = EnrichmentDecision(
            claim_id=claim_id,
            enrichment_id=self._generate_enrichment_id(),
            timestamp=datetime.utcnow(),
            source=source,
            data_added=data_added,
            confidence_score=confidence_score,
            decision_rationale=decision_rationale,
            data_quality_score=await self._assess_data_quality(data_added),
            user_or_system="system",  # or actual user ID
            source_system=source_system,
            source_record_id=source_record_id,
            source_timestamp=datetime.utcnow()
        )

        await self.enrichment_db.insert(enrichment)

        logger.info(
            f"Logged enrichment for claim {claim_id} "
            f"from {source.value} (confidence: {confidence_score:.2f})"
        )

        return enrichment

    async def get_claim_enrichment_history(
        self,
        claim_id: str
    ) -> List[EnrichmentDecision]:
        """
        Retrieve all enrichments for a claim.
        """
        return await self.enrichment_db.query({
            "claim_id": claim_id
        }, sort_by="timestamp")

    async def validate_enrichment(
        self,
        enrichment_id: str,
        validation_result: str,
        validated_by: str
    ) -> None:
        """
        Mark enrichment as validated.
        """
        await self.enrichment_db.update(
            {"enrichment_id": enrichment_id},
            {
                "$set": {
                    "validated": True,
                    "validation_timestamp": datetime.utcnow(),
                    "validation_result": validation_result,
                    "validated_by": validated_by
                }
            }
        )

    async def get_enrichment_statistics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, any]:
        """
        Get enrichment statistics for reporting.
        """
        results = await self.enrichment_db.aggregate([
            {
                "$match": {
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": "$source",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"},
                    "avg_quality": {"$avg": "$data_quality_score"}
                }
            }
        ])

        return {
            result['_id']: {
                "count": result['count'],
                "average_confidence": result['avg_confidence'],
                "average_quality": result['avg_quality']
            }
            for result in results
        }
```

### 3.2 Enrichment Approval Workflow

```python
class EnrichmentApprovalWorkflow:
    """
    Manages approval workflow for high-risk enrichments.
    """
    def __init__(self):
        self.approval_service = ApprovalService()
        self.notification_service = NotificationService()

    async def request_approval(
        self,
        enrichment: EnrichmentDecision,
        reviewer: str,
        priority: str = "normal"
    ) -> ApprovalRequest:
        """
        Create approval request for enrichment decision.
        """
        # Check if approval is required
        if not self._requires_approval(enrichment):
            return ApprovalRequest(
                status="AUTO_APPROVED",
                enrichment_id=enrichment.enrichment_id
            )

        # Create approval request
        request = ApprovalRequest(
            enrichment_id=enrichment.enrichment_id,
            claim_id=enrichment.claim_id,
            requested_by="system",
            reviewer=reviewer,
            priority=priority,
            created_at=datetime.utcnow(),
            status="PENDING",
            enrichment_details=enrichment
        )

        await self.approval_service.create_request(request)

        # Notify reviewer
        await self.notification_service.notify(
            user=reviewer,
            message=f"Enrichment approval required for claim {enrichment.claim_id}",
            priority=priority
        )

        return request

    def _requires_approval(self, enrichment: EnrichmentDecision) -> bool:
        """
        Determine if enrichment requires human approval.
        """
        # Require approval if:
        # 1. Low confidence score
        if enrichment.confidence_score < 0.7:
            return True

        # 2. Low data quality
        if enrichment.data_quality_score < 0.8:
            return True

        # 3. High-risk source
        high_risk_sources = [
            EnrichmentSource.FRAUD_PATTERNS,
            EnrichmentSource.PROVIDER_NETWORK
        ]
        if enrichment.source in high_risk_sources:
            return True

        return False
```

## 4. Audit Trail System

### 4.1 Comprehensive Audit Logging

```python
from enum import Enum

class AuditEventType(Enum):
    CLAIM_SUBMITTED = "claim_submitted"
    CLAIM_VALIDATED = "claim_validated"
    CLAIM_ENRICHED = "claim_enriched"
    FRAUD_DETECTED = "fraud_detected"
    FRAUD_REVIEWED = "fraud_reviewed"
    MODEL_INFERENCE = "model_inference"
    KB_SEARCH = "kb_search"
    USER_ACCESS = "user_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"

@dataclass
class AuditEvent:
    """
    Comprehensive audit event record.
    """
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]

    # Resource information
    resource_type: str  # claim, model, kb, etc.
    resource_id: str
    action: str  # create, read, update, delete, execute

    # Event details
    details: Dict[str, any]
    status: str  # success, failure, partial
    error_message: Optional[str] = None

    # Context
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    parent_event_id: Optional[str] = None  # For chained events

    # Compliance
    hipaa_logged: bool = True
    retention_period_days: int = 2555  # 7 years for HIPAA
    pii_contains: bool = False


class AuditTrailService:
    """
    Manages comprehensive audit trail.
    """
    def __init__(self):
        self.audit_db = AuditDatabase()
        self.encryption_service = EncryptionService()

    async def log_event(
        self,
        event_type: AuditEventType,
        resource_type: str,
        resource_id: str,
        action: str,
        details: Dict[str, any],
        user_id: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event.
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            ip_address=self._get_request_ip(),
            user_agent=self._get_user_agent(),
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details,
            status=status,
            error_message=error_message,
            session_id=self._get_session_id(),
            request_id=self._get_request_id()
        )

        # Encrypt sensitive data
        if event.pii_contains:
            event.details = await self.encryption_service.encrypt(
                event.details
            )

        # Store in audit database
        await self.audit_db.insert(event)

        # Stream to SIEM if high-risk event
        if self._is_high_risk_event(event):
            await self._stream_to_siem(event)

        return event

    async def log_fraud_detection(
        self,
        claim_id: str,
        result: FraudDetectionResult,
        user_id: Optional[str] = None
    ) -> AuditEvent:
        """
        Log fraud detection event.
        """
        return await self.log_event(
            event_type=AuditEventType.FRAUD_DETECTED,
            resource_type="claim",
            resource_id=claim_id,
            action="detect_fraud",
            details={
                "fraud_score": result.fraud_score,
                "is_fraud": result.is_fraud,
                "confidence": result.confidence,
                "model_version": settings.MODEL_VERSION,
                "kb_version": settings.KB_VERSION,
                "red_flags": result.red_flags,
                "recommended_action": result.recommended_action
            },
            user_id=user_id,
            status="success"
        )

    async def log_human_review(
        self,
        claim_id: str,
        reviewer_id: str,
        decision: str,
        rationale: str
    ) -> AuditEvent:
        """
        Log human review decision.
        """
        return await self.log_event(
            event_type=AuditEventType.FRAUD_REVIEWED,
            resource_type="claim",
            resource_id=claim_id,
            action="human_review",
            details={
                "reviewer_id": reviewer_id,
                "decision": decision,
                "rationale": rationale,
                "review_timestamp": datetime.utcnow().isoformat()
            },
            user_id=reviewer_id,
            status="success"
        )

    async def get_audit_trail(
        self,
        resource_id: str,
        resource_type: Optional[str] = None
    ) -> List[AuditEvent]:
        """
        Retrieve complete audit trail for resource.
        """
        query = {"resource_id": resource_id}
        if resource_type:
            query["resource_type"] = resource_type

        return await self.audit_db.query(
            query,
            sort_by="timestamp",
            order="asc"
        )

    async def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[AuditEventType]] = None
    ) -> AuditReport:
        """
        Generate compliance audit report.
        """
        query = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }

        if event_types:
            query["event_type"] = {"$in": event_types}

        events = await self.audit_db.query(query)

        return AuditReport(
            start_date=start_date,
            end_date=end_date,
            total_events=len(events),
            events_by_type=self._group_by_type(events),
            events_by_user=self._group_by_user(events),
            failure_events=self._filter_failures(events),
            high_risk_events=self._filter_high_risk(events),
            generated_at=datetime.utcnow()
        )

    def _is_high_risk_event(self, event: AuditEvent) -> bool:
        """
        Determine if event is high-risk for SIEM alerting.
        """
        high_risk_types = [
            AuditEventType.DATA_EXPORT,
            AuditEventType.CONFIGURATION_CHANGE
        ]

        return (
            event.event_type in high_risk_types or
            event.status == "failure" or
            event.action == "delete"
        )
```

### 4.2 Access Control Auditing

```python
class AccessControlAuditor:
    """
    Audits all access to sensitive data.
    """
    def __init__(self):
        self.audit_service = AuditTrailService()

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        access_type: str,  # read, write, delete
        justification: str
    ) -> None:
        """
        Log data access for compliance.
        """
        await self.audit_service.log_event(
            event_type=AuditEventType.USER_ACCESS,
            resource_type=resource_type,
            resource_id=resource_id,
            action=access_type,
            details={
                "justification": justification,
                "access_method": self._get_access_method()
            },
            user_id=user_id
        )

    async def log_phi_access(
        self,
        user_id: str,
        patient_id: str,
        claim_ids: List[str],
        purpose: str
    ) -> None:
        """
        Log Protected Health Information (PHI) access for HIPAA.
        """
        await self.audit_service.log_event(
            event_type=AuditEventType.USER_ACCESS,
            resource_type="phi",
            resource_id=patient_id,
            action="read",
            details={
                "claim_ids": claim_ids,
                "purpose": purpose,
                "hipaa_compliance": True,
                "breach_notification_required": False
            },
            user_id=user_id
        )

    async def detect_anomalous_access(
        self,
        user_id: str,
        time_window_hours: int = 24
    ) -> List[AnomalyAlert]:
        """
        Detect unusual access patterns.
        """
        # Get user's recent access
        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_access = await self.audit_service.audit_db.query({
            "user_id": user_id,
            "timestamp": {"$gte": cutoff},
            "event_type": AuditEventType.USER_ACCESS
        })

        alerts = []

        # Check for unusual volume
        if len(recent_access) > 1000:  # Threshold
            alerts.append(AnomalyAlert(
                user_id=user_id,
                alert_type="HIGH_VOLUME_ACCESS",
                severity="high",
                details=f"User accessed {len(recent_access)} records in {time_window_hours}h",
                timestamp=datetime.utcnow()
            ))

        # Check for unusual time of access
        unusual_time_access = [
            e for e in recent_access
            if e.timestamp.hour < 6 or e.timestamp.hour > 22
        ]
        if unusual_time_access:
            alerts.append(AnomalyAlert(
                user_id=user_id,
                alert_type="UNUSUAL_TIME_ACCESS",
                severity="medium",
                details=f"User accessed data during off-hours {len(unusual_time_access)} times",
                timestamp=datetime.utcnow()
            ))

        return alerts
```

## 5. Data Lineage Tracking

### 5.1 Lineage Graph

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LineageNode:
    """
    Represents a node in data lineage graph.
    """
    node_id: str
    node_type: str  # source, transformation, enrichment, model, output
    resource_id: str
    resource_type: str
    timestamp: datetime
    metadata: Dict[str, any]

@dataclass
class LineageEdge:
    """
    Represents relationship between lineage nodes.
    """
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # derived_from, enriched_by, processed_by
    transformation: Optional[str] = None
    metadata: Dict[str, any] = None


class DataLineageTracker:
    """
    Tracks complete data lineage for claims.
    """
    def __init__(self):
        self.lineage_db = LineageDatabase()
        self.graph_builder = LineageGraphBuilder()

    async def track_claim_lineage(
        self,
        claim_id: str,
        source: str,
        source_system: str
    ) -> LineageNode:
        """
        Create root lineage node for claim.
        """
        node = LineageNode(
            node_id=self._generate_node_id(),
            node_type="source",
            resource_id=claim_id,
            resource_type="claim",
            timestamp=datetime.utcnow(),
            metadata={
                "source": source,
                "source_system": source_system,
                "ingestion_method": "api"  # or batch, edi, etc.
            }
        )

        await self.lineage_db.insert_node(node)

        return node

    async def track_transformation(
        self,
        source_node_id: str,
        transformation_type: str,
        transformation_details: Dict[str, any],
        output_resource_id: str
    ) -> LineageNode:
        """
        Track data transformation step.
        """
        # Create target node
        target_node = LineageNode(
            node_id=self._generate_node_id(),
            node_type="transformation",
            resource_id=output_resource_id,
            resource_type="claim",
            timestamp=datetime.utcnow(),
            metadata={
                "transformation_type": transformation_type,
                "transformation_details": transformation_details
            }
        )

        await self.lineage_db.insert_node(target_node)

        # Create edge
        edge = LineageEdge(
            edge_id=self._generate_edge_id(),
            source_node_id=source_node_id,
            target_node_id=target_node.node_id,
            relationship_type="derived_from",
            transformation=transformation_type,
            metadata=transformation_details
        )

        await self.lineage_db.insert_edge(edge)

        return target_node

    async def track_enrichment(
        self,
        claim_node_id: str,
        enrichment_source: str,
        enrichment_data: Dict[str, any]
    ) -> LineageNode:
        """
        Track enrichment step.
        """
        enrichment_node = LineageNode(
            node_id=self._generate_node_id(),
            node_type="enrichment",
            resource_id=enrichment_source,
            resource_type="enrichment_source",
            timestamp=datetime.utcnow(),
            metadata={
                "source": enrichment_source,
                "data_added": list(enrichment_data.keys())
            }
        )

        await self.lineage_db.insert_node(enrichment_node)

        edge = LineageEdge(
            edge_id=self._generate_edge_id(),
            source_node_id=enrichment_node.node_id,
            target_node_id=claim_node_id,
            relationship_type="enriched_by",
            metadata={"enrichment_fields": list(enrichment_data.keys())}
        )

        await self.lineage_db.insert_edge(edge)

        return enrichment_node

    async def track_model_inference(
        self,
        claim_node_id: str,
        model_version: str,
        model_type: str,
        prediction: Dict[str, any]
    ) -> LineageNode:
        """
        Track model inference step.
        """
        model_node = LineageNode(
            node_id=self._generate_node_id(),
            node_type="model",
            resource_id=f"{model_type}:{model_version}",
            resource_type="ml_model",
            timestamp=datetime.utcnow(),
            metadata={
                "model_version": model_version,
                "model_type": model_type,
                "prediction": prediction
            }
        )

        await self.lineage_db.insert_node(model_node)

        edge = LineageEdge(
            edge_id=self._generate_edge_id(),
            source_node_id=claim_node_id,
            target_node_id=model_node.node_id,
            relationship_type="processed_by",
            metadata={
                "model_version": model_version,
                "inference_timestamp": datetime.utcnow().isoformat()
            }
        )

        await self.lineage_db.insert_edge(edge)

        return model_node

    async def get_lineage_graph(
        self,
        claim_id: str
    ) -> LineageGraph:
        """
        Retrieve complete lineage graph for claim.
        """
        # Get all nodes related to claim
        nodes = await self.lineage_db.get_nodes_for_resource(
            resource_id=claim_id
        )

        # Get all edges between nodes
        node_ids = [n.node_id for n in nodes]
        edges = await self.lineage_db.get_edges_for_nodes(node_ids)

        # Build graph
        graph = self.graph_builder.build_graph(nodes, edges)

        return graph

    async def visualize_lineage(
        self,
        claim_id: str,
        output_format: str = "mermaid"
    ) -> str:
        """
        Generate visual representation of lineage.
        """
        graph = await self.get_lineage_graph(claim_id)

        if output_format == "mermaid":
            return self._generate_mermaid_diagram(graph)
        elif output_format == "graphviz":
            return self._generate_graphviz_diagram(graph)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_mermaid_diagram(self, graph: LineageGraph) -> str:
        """
        Generate Mermaid diagram of lineage.
        """
        lines = ["graph TD"]

        for node in graph.nodes:
            node_label = f"{node.node_type}:{node.resource_id}"
            lines.append(f"    {node.node_id}[{node_label}]")

        for edge in graph.edges:
            edge_label = edge.relationship_type
            lines.append(
                f"    {edge.source_node_id} -->|{edge_label}| {edge.target_node_id}"
            )

        return "\n".join(lines)
```

### 5.2 Impact Analysis

```python
class LineageImpactAnalyzer:
    """
    Analyzes impact of data changes through lineage.
    """
    def __init__(self):
        self.lineage_tracker = DataLineageTracker()

    async def analyze_impact(
        self,
        changed_resource_id: str,
        change_type: str
    ) -> ImpactAnalysis:
        """
        Analyze downstream impact of data change.
        """
        # Get lineage graph
        graph = await self.lineage_tracker.get_lineage_graph(changed_resource_id)

        # Find all downstream nodes
        downstream_nodes = self._get_downstream_nodes(
            graph,
            changed_resource_id
        )

        # Categorize impact
        impacted_claims = [
            n for n in downstream_nodes
            if n.node_type == "claim"
        ]

        impacted_models = [
            n for n in downstream_nodes
            if n.node_type == "model"
        ]

        impacted_outputs = [
            n for n in downstream_nodes
            if n.node_type == "output"
        ]

        return ImpactAnalysis(
            changed_resource=changed_resource_id,
            change_type=change_type,
            total_impacted_resources=len(downstream_nodes),
            impacted_claims=impacted_claims,
            impacted_models=impacted_models,
            impacted_outputs=impacted_outputs,
            requires_reprocessing=len(impacted_outputs) > 0,
            estimated_reprocessing_time=self._estimate_reprocessing_time(
                len(downstream_nodes)
            )
        )

    def _get_downstream_nodes(
        self,
        graph: LineageGraph,
        start_node_id: str
    ) -> List[LineageNode]:
        """
        Get all nodes downstream from start node.
        """
        visited = set()
        queue = [start_node_id]
        downstream = []

        while queue:
            current_id = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            # Get outgoing edges
            outgoing_edges = [
                e for e in graph.edges
                if e.source_node_id == current_id
            ]

            for edge in outgoing_edges:
                target_node = graph.get_node(edge.target_node_id)
                downstream.append(target_node)
                queue.append(edge.target_node_id)

        return downstream
```

## 6. Compliance and Retention

### 6.1 HIPAA Compliance

```python
class HIPAAComplianceManager:
    """
    Manages HIPAA compliance requirements.
    """
    def __init__(self):
        self.audit_service = AuditTrailService()
        self.encryption_service = EncryptionService()
        self.retention_service = RetentionPolicyService()

    async def ensure_phi_protection(
        self,
        claim_data: Dict[str, any]
    ) -> PHIProtectionResult:
        """
        Ensure PHI is properly protected.
        """
        # Identify PHI fields
        phi_fields = self._identify_phi_fields(claim_data)

        # Encrypt PHI at rest
        encrypted_data = await self.encryption_service.encrypt_fields(
            claim_data,
            phi_fields
        )

        # Log PHI handling
        await self.audit_service.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            resource_type="phi",
            resource_id=claim_data.get('claim_id'),
            action="encrypt",
            details={
                "phi_fields": phi_fields,
                "encryption_algorithm": "AES-256-GCM",
                "key_id": self.encryption_service.current_key_id
            }
        )

        return PHIProtectionResult(
            protected=True,
            phi_fields=phi_fields,
            encryption_applied=True
        )

    def _identify_phi_fields(self, data: Dict[str, any]) -> List[str]:
        """
        Identify fields containing PHI.
        """
        phi_field_patterns = [
            "patient_id",
            "patient_name",
            "patient_ssn",
            "patient_dob",
            "patient_address",
            "diagnosis_codes",  # Contains health information
            "procedure_codes"    # Contains health information
        ]

        return [
            field for field in phi_field_patterns
            if field in data
        ]
```

### 6.2 Retention Policies

```python
class RetentionPolicyService:
    """
    Manages data retention policies.
    """
    def __init__(self):
        self.retention_db = RetentionPolicyDatabase()

    async def apply_retention_policy(
        self,
        resource_type: str,
        resource_id: str
    ) -> RetentionPolicy:
        """
        Apply retention policy to resource.
        """
        # Get policy for resource type
        policy = await self.retention_db.get_policy(resource_type)

        # Calculate retention expiry
        expiry_date = datetime.utcnow() + timedelta(days=policy.retention_days)

        # Store retention metadata
        await self.retention_db.store_retention_metadata({
            "resource_type": resource_type,
            "resource_id": resource_id,
            "policy_name": policy.policy_name,
            "retention_days": policy.retention_days,
            "expiry_date": expiry_date,
            "legal_hold": False,
            "created_at": datetime.utcnow()
        })

        return policy

    async def check_retention_expiry(self) -> List[str]:
        """
        Check for resources past retention period.
        """
        expired = await self.retention_db.query({
            "expiry_date": {"$lte": datetime.utcnow()},
            "legal_hold": False,
            "deleted": False
        })

        return [r['resource_id'] for r in expired]

    async def archive_expired_data(
        self,
        resource_ids: List[str]
    ) -> ArchiveResult:
        """
        Archive data past retention period.
        """
        archived_count = 0

        for resource_id in resource_ids:
            # Move to cold storage
            await self._move_to_cold_storage(resource_id)

            # Mark as archived
            await self.retention_db.update(
                {"resource_id": resource_id},
                {"$set": {"archived": True, "archived_at": datetime.utcnow()}}
            )

            archived_count += 1

        return ArchiveResult(
            total_resources=len(resource_ids),
            archived_count=archived_count,
            archive_timestamp=datetime.utcnow()
        )
```

## 7. Monitoring and Reporting

### 7.1 Governance Metrics

```python
class GovernanceMetricsCollector:
    """
    Collects governance and compliance metrics.
    """
    async def collect_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> GovernanceMetrics:
        """
        Collect comprehensive governance metrics.
        """
        return GovernanceMetrics(
            period_start=start_date,
            period_end=end_date,
            audit_events_total=await self._count_audit_events(start_date, end_date),
            enrichment_decisions=await self._count_enrichments(start_date, end_date),
            kb_versions_created=await self._count_kb_versions(start_date, end_date),
            compliance_violations=await self._count_violations(start_date, end_date),
            data_quality_score=await self._calculate_avg_quality(start_date, end_date),
            retention_compliance_rate=await self._calculate_retention_compliance()
        )

    async def generate_compliance_report(
        self,
        report_type: str,
        period: str
    ) -> ComplianceReport:
        """
        Generate compliance report for auditors.
        """
        # Implementation depends on specific compliance requirements
        pass
```

## Related Documents

- [DATA_FLOW_ARCHITECTURE.md](./DATA_FLOW_ARCHITECTURE.md) - Data processing pipelines
- [DATA_VALIDATION_PIPELINE.md](./DATA_VALIDATION_PIPELINE.md) - Validation processes
- [INFRASTRUCTURE_REQUIREMENTS.md](./INFRASTRUCTURE_REQUIREMENTS.md) - Infrastructure specs
