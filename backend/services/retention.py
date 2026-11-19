"""Retention policy enforcement for stored resume documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db.models import ConsentRecord, ConsentStatus, DocumentStatus, ResumeDocument
from backend.security import log_security_event
from backend.storage.adapter import StorageAdapter, StorageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionSweepSummary:
    inspected_count: int
    expired_count: int
    failed_count: int
    consents_expired: int = 0


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
                db=db,
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


def expire_consents(db: Session, *, now: datetime | None = None) -> int:
    """Mark granted consents whose expiry time has passed as EXPIRED.

    Access checks already treat a consent past ``expires_at`` as invalid even before this
    runs; the sweep keeps the stored status truthful for reports and the UI.
    """
    reference_time = now if now is not None else datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    stale = db.scalars(
        select(ConsentRecord).where(
            ConsentRecord.status == ConsentStatus.GRANTED,
            ConsentRecord.expires_at.is_not(None),
            ConsentRecord.expires_at <= reference_time,
        )
    ).all()
    for record in stale:
        record.status = ConsentStatus.EXPIRED
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error("Consent expiry sweep failed.")
        return 0
    return len(stale)


def run_retention_cycle(
    db: Session,
    storage: StorageAdapter,
    *,
    now: datetime | None = None,
    actor_recruiter_id: UUID | None = None,
) -> RetentionSweepSummary:
    """Run the full housekeeping pass: purge expired resumes, then expire consents."""
    documents = enforce_document_retention(
        db, storage, now=now, actor_recruiter_id=actor_recruiter_id
    )
    consents_expired = expire_consents(db, now=now)
    return RetentionSweepSummary(
        inspected_count=documents.inspected_count,
        expired_count=documents.expired_count,
        failed_count=documents.failed_count,
        consents_expired=consents_expired,
    )
