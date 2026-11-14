"""Retention policy enforcement for stored resume documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db.models import DocumentStatus, ResumeDocument
from backend.security import log_security_event
from backend.storage.adapter import StorageAdapter, StorageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionSweepSummary:
    inspected_count: int
    expired_count: int
    failed_count: int


def enforce_document_retention(
    db: Session,
    storage: StorageAdapter,
    *,
    now: datetime | None = None,
    actor_recruiter_id: UUID | None = None,
    limit: int = 200,
) -> RetentionSweepSummary:
    """Purge physical files and mark metadata EXPIRED for documents past retention_until."""
    reference_time = now if now is not None else datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    candidates = list(
        db.scalars(
            select(ResumeDocument)
            .where(
                ResumeDocument.status.in_(
                    [DocumentStatus.AVAILABLE, DocumentStatus.QUARANTINED]
                ),
                ResumeDocument.retention_until.is_not(None),
                ResumeDocument.retention_until <= reference_time,
            )
            .order_by(ResumeDocument.retention_until.asc())
            .limit(limit)
        ).all()
    )

    expired_count = 0
    failed_count = 0

    for document in candidates:
        try:
            if document.status != DocumentStatus.QUARANTINED:
                document.status = DocumentStatus.QUARANTINED
                db.commit()

            storage.delete_file(document.storage_key)

            document.status = DocumentStatus.EXPIRED
            document.deleted_at = reference_time
            db.commit()
            expired_count += 1
            log_security_event(
                "retention.document_expired",
                outcome="success",
                recruiter_id=actor_recruiter_id,
                resource_type="resume_document",
                resource_id=document.id,
            )
        except (StorageError, SQLAlchemyError):
            db.rollback()
            failed_count += 1
            logger.error(
                "Retention purge failed for resume document %s.",
                document.id,
            )

    return RetentionSweepSummary(
        inspected_count=len(candidates),
        expired_count=expired_count,
        failed_count=failed_count,
    )
