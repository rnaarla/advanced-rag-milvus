"""
Compliance and Auditing Module
Document lineage, versioning semantics, and audit-grade metadata
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
from enum import Enum
from datetime import timedelta


class AuditEventType(Enum):
    """Types of auditable events"""
    DOCUMENT_INGESTION = "document_ingestion"
    CHUNK_CREATED = "chunk_created"
    INDEX_UPDATE = "index_update"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVALUATION = "evaluation"
    DRIFT_DETECTION = "drift_detection"
    VERSION_UPDATE = "version_update"


@dataclass
class AuditLog:
    """Audit log entry"""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Event-specific data
    event_data: Dict[str, Any]
    
    # Lineage tracking
    parent_event_id: Optional[str]
    related_event_ids: List[str]
    
    # Compliance metadata
    compliance_flags: List[str]
    retention_policy: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_data": self.event_data,
            "parent_event_id": self.parent_event_id,
            "related_event_ids": self.related_event_ids,
            "compliance_flags": self.compliance_flags,
            "retention_policy": self.retention_policy
        }


@dataclass
class DocumentVersion:
    """Document version metadata"""
    doc_id: str
    version: str
    version_hash: str
    created_at: str
    created_by: Optional[str]
    
    # Change tracking
    change_type: str  # "create", "update", "delete"
    previous_version: Optional[str]
    changes_summary: Optional[str]
    
    # Content metadata
    chunk_count: int
    total_tokens: int
    
    # Compliance
    retention_until: Optional[str]
    classification: str  # "public", "internal", "confidential", "restricted"


class ComplianceManager:
    """
    Manages compliance requirements including:
    - Document lineage and provenance
    - Version control semantics
    - Audit logging
    - Retention policies
    - Access control metadata
    """
    
    def __init__(
        self,
        enable_audit: bool = True,
        enable_versioning: bool = True,
        storage_backend: Optional[Any] = None,
        retention_days: int = 90
    ):
        """
        Args:
            enable_audit: Enable audit logging
            enable_versioning: Enable document versioning
            storage_backend: Backend for persistent storage (DB, S3, etc.)
            retention_days: Default retention period
        """
        self.enable_audit = enable_audit
        self.enable_versioning = enable_versioning
        self.storage_backend = storage_backend
        self.retention_days = retention_days
        
        # In-memory stores (use persistent storage in production)
        self.audit_logs: List[AuditLog] = []
        self.document_versions: Dict[str, List[DocumentVersion]] = {}
        self.lineage_graph: Dict[str, List[str]] = {}  # event_id -> child event_ids
        # Compliance flags (tenant_id -> set(doc_id)); None for default tenant
        self.legal_holds: Dict[Optional[str], set[str]] = {}
        
        # Session tracking
        self.current_session_id = self._generate_session_id()
    
    async def log_ingestion(
        self,
        document_count: int,
        chunk_count: int,
        report: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> AuditLog:
        """Log document ingestion event"""
        if not self.enable_audit:
            return None
        
        event_id = self._generate_event_id()
        
        audit_log = AuditLog(
            event_id=event_id,
            event_type=AuditEventType.DOCUMENT_INGESTION,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self.current_session_id,
            event_data={
                "document_count": document_count,
                "chunk_count": chunk_count,
                "ingestion_report": report
            },
            parent_event_id=None,
            related_event_ids=[],
            compliance_flags=["data_ingestion"],
            retention_policy=f"{self.retention_days}_days"
        )
        
        self._store_audit_log(audit_log)
        return audit_log
    
    async def log_retrieval(
        self,
        query: str,
        chunk_id: str,
        score: float,
        latency_ms: float,
        user_id: Optional[str] = None
    ) -> AuditLog:
        """Log retrieval event"""
        if not self.enable_audit:
            return None
        
        event_id = self._generate_event_id()
        
        audit_log = AuditLog(
            event_id=event_id,
            event_type=AuditEventType.RETRIEVAL,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self.current_session_id,
            event_data={
                "query": query,
                "chunk_id": chunk_id,
                "score": score,
                "latency_ms": latency_ms
            },
            parent_event_id=None,
            related_event_ids=[],
            compliance_flags=["data_access"],
            retention_policy=f"{self.retention_days}_days"
        )
        
        self._store_audit_log(audit_log)
        return audit_log
    
    async def create_version(
        self,
        doc_id: str,
        content: str,
        change_type: str,
        chunk_count: int,
        total_tokens: int,
        user_id: Optional[str] = None,
        classification: str = "internal"
    ) -> DocumentVersion:
        """Create a new document version"""
        if not self.enable_versioning:
            return None
        
        # Get previous version
        previous_versions = self.document_versions.get(doc_id, [])
        previous_version = previous_versions[-1].version if previous_versions else None
        
        # Generate new version
        version_number = len(previous_versions) + 1
        version = f"v{version_number}"
        version_hash = self._hash_content(content)
        
        doc_version = DocumentVersion(
            doc_id=doc_id,
            version=version,
            version_hash=version_hash,
            created_at=datetime.now().isoformat(),
            created_by=user_id,
            change_type=change_type,
            previous_version=previous_version,
            changes_summary=f"{change_type.capitalize()} operation",
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            retention_until=self._calculate_retention_date(),
            classification=classification
        )
        
        # Store version
        if doc_id not in self.document_versions:
            self.document_versions[doc_id] = []
        self.document_versions[doc_id].append(doc_version)
        
        # Log version creation
        if self.enable_audit:
            event_id = self._generate_event_id()
            audit_log = AuditLog(
                event_id=event_id,
                event_type=AuditEventType.VERSION_UPDATE,
                timestamp=datetime.now().isoformat(),
                user_id=user_id,
                session_id=self.current_session_id,
                event_data={
                    "doc_id": doc_id,
                    "version": version,
                    "version_hash": version_hash,
                    "change_type": change_type
                },
                parent_event_id=None,
                related_event_ids=[],
                compliance_flags=["version_control"],
                retention_policy=f"{self.retention_days}_days"
            )
            self._store_audit_log(audit_log)
        
        return doc_version

    async def apply_legal_hold(self, doc_id: str, tenant_id: Optional[str] = None) -> None:
        """
        Apply a legal hold to a document.

        In-memory implementation:
        - Marks the (tenant_id, doc_id) pair as under legal hold.
        - Future retention or deletion logic can consult this flag to skip
          automatic deletion.
        """
        key = tenant_id
        holds = self.legal_holds.setdefault(key, set())
        holds.add(doc_id)

    async def forget_document(self, doc_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Right-to-forget hook (in-memory implementation).

        Behaviour:
        - If the (tenant_id, doc_id) is under legal hold, do nothing and return a flag.
        - Otherwise, clear in-memory version history and emit an audit log
          entry describing the redaction event.
        """
        holds = self.legal_holds.get(tenant_id) or set()
        if doc_id in holds:
            return {
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "forgotten": False,
                "reason": "legal_hold",
            }

        removed_versions = self.document_versions.pop(doc_id, [])

        if self.enable_audit and removed_versions:
            event_id = self._generate_event_id()
            audit_log = AuditLog(
                event_id=event_id,
                event_type=AuditEventType.VERSION_UPDATE,
                timestamp=datetime.now().isoformat(),
                user_id=None,
                session_id=self.current_session_id,
                event_data={
                    "doc_id": doc_id,
                    "action": "forget",
                    "removed_versions": [v.version for v in removed_versions],
                },
                parent_event_id=None,
                related_event_ids=[],
                compliance_flags=["right_to_forget"],
                retention_policy=f"{self.retention_days}_days",
            )
            self._store_audit_log(audit_log)

        return {
            "doc_id": doc_id,
            "forgotten": bool(removed_versions),
            "reason": "removed" if removed_versions else "not_found",
        }
    
    def get_document_lineage(self, doc_id: str) -> List[DocumentVersion]:
        """Get complete version history for a document"""
        return self.document_versions.get(doc_id, [])
    
    def get_version(self, doc_id: str, version: str) -> Optional[DocumentVersion]:
        """Get specific version of a document"""
        versions = self.document_versions.get(doc_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def track_lineage(
        self,
        parent_event_id: str,
        child_event_id: str
    ):
        """Track lineage relationship between events"""
        if parent_event_id not in self.lineage_graph:
            self.lineage_graph[parent_event_id] = []
        self.lineage_graph[parent_event_id].append(child_event_id)
    
    def get_event_lineage(
        self,
        event_id: str,
        depth: int = -1
    ) -> Dict[str, Any]:
        """
        Get lineage tree for an event
        
        Args:
            event_id: Root event ID
            depth: Maximum depth (-1 for unlimited)
            
        Returns:
            Lineage tree structure
        """
        def build_tree(node_id: str, current_depth: int) -> Dict:
            if depth != -1 and current_depth >= depth:
                return {"event_id": node_id, "children": []}
            
            children = self.lineage_graph.get(node_id, [])
            return {
                "event_id": node_id,
                "children": [
                    build_tree(child_id, current_depth + 1)
                    for child_id in children
                ]
            }
        
        return build_tree(event_id, 0)
    
    def query_audit_logs(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        compliance_flag: Optional[str] = None
    ) -> List[AuditLog]:
        """Query audit logs with filters"""
        results = self.audit_logs
        
        if event_type:
            results = [log for log in results if log.event_type == event_type]
        
        if user_id:
            results = [log for log in results if log.user_id == user_id]
        
        if start_time:
            results = [log for log in results if log.timestamp >= start_time]
        
        if end_time:
            results = [log for log in results if log.timestamp <= end_time]
        
        if compliance_flag:
            results = [
                log for log in results
                if compliance_flag in log.compliance_flags
            ]
        
        return results
    
    def generate_compliance_report(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Generate compliance report for a time period"""
        logs = self.query_audit_logs(
            start_time=start_date,
            end_time=end_date
        )
        
        # Aggregate statistics
        event_counts = {}
        for log in logs:
            event_type = log.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # User activity
        user_activity = {}
        for log in logs:
            if log.user_id:
                user_activity[log.user_id] = user_activity.get(log.user_id, 0) + 1
        
        # Compliance flags
        compliance_flags = {}
        for log in logs:
            for flag in log.compliance_flags:
                compliance_flags[flag] = compliance_flags.get(flag, 0) + 1
        
        return {
            "period_start": start_date,
            "period_end": end_date,
            "total_events": len(logs),
            "event_counts": event_counts,
            "user_activity": user_activity,
            "compliance_flags": compliance_flags,
            "version_updates": len([
                log for log in logs
                if log.event_type == AuditEventType.VERSION_UPDATE
            ])
        }
    
    def verify_data_integrity(
        self,
        doc_id: str,
        expected_hash: str
    ) -> bool:
        """Verify data integrity using version hashes"""
        versions = self.document_versions.get(doc_id, [])
        if not versions:
            return False
        
        latest_version = versions[-1]
        return latest_version.version_hash == expected_hash
    
    def _store_audit_log(self, audit_log: AuditLog):
        """Store audit log (in-memory or persistent storage)"""
        self.audit_logs.append(audit_log)
        
        # In production, persist to storage backend
        if self.storage_backend:
            self.storage_backend.store(audit_log.to_dict())
        
        # Enforce retention policy
        self._enforce_retention_policy()
    
    def _enforce_retention_policy(self):
        """Remove logs older than retention period"""
        if not self.audit_logs:
            return
        
        # Use a small grace period to avoid pruning logs created within the same second
        cutoff_date = datetime.now() - timedelta(days=self.retention_days, seconds=1)
        cutoff_str = cutoff_date.isoformat()
        
        self.audit_logs = [
            log for log in self.audit_logs
            if log.timestamp >= cutoff_str
        ]
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().isoformat()
        random_part = hashlib.sha256(
            f"{timestamp}{id(self)}".encode()
        ).hexdigest()[:16]
        return f"evt_{random_part}"
    
    def _generate_session_id(self) -> str:
        """Generate session ID"""
        timestamp = datetime.now().isoformat()
        random_part = hashlib.sha256(
            f"session_{timestamp}".encode()
        ).hexdigest()[:12]
        return f"sess_{random_part}"
    
    def _hash_content(self, content: str) -> str:
        """Generate content hash for integrity verification"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _calculate_retention_date(self) -> str:
        """Calculate retention expiration date"""
        retention_date = datetime.now() + timedelta(days=self.retention_days)
        return retention_date.isoformat()
    
    async def close(self):
        """Cleanup and finalize audit logs"""
        # In production, ensure all logs are persisted
        if self.storage_backend:
            for log in self.audit_logs:
                self.storage_backend.store(log.to_dict())
        
        print(f"Finalized {len(self.audit_logs)} audit log entries")
