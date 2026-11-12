"""Resume document lifecycle operations."""

from __future__ import annotations

import hashlib
import logging
import ntpath
import os
import shutil
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator
from uuid import UUID, uuid4

import pymupdf
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db.models import (
    Application,
    Candidate,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    Recruiter,
    RecruiterStatus,
    ResumeDocument,
)
from backend.storage.adapter import StorageAdapter, StorageError


logger = logging.getLogger(__name__)
MAX_RESUME_BYTES = 10 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
ALLOWED_MEDIA_TYPES = {"", "application/pdf", "application/octet-stream"}


class ResumeDocumentError(Exception):
    """Base error for safe resume lifecycle failures."""


class ResumeUploadError(ResumeDocumentError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ResumeNotFoundError(ResumeDocumentError):
    pass


class CandidateNotFoundError(ResumeDocumentError):
    pass


class InvalidConsentError(ResumeDocumentError):
    pass


class DevelopmentRecruiterError(ResumeDocumentError):
    pass


class ResumeStorageUnavailableError(ResumeDocumentError):
    pass


class ResumeDatabaseError(ResumeDocumentError):
    pass


def sanitize_original_filename(filename: str | None) -> str:
    supplied_name = (filename or "").replace("\x00", "")
    basename = ntpath.basename(supplied_name.replace("/", "\\"))
    safe_name = "".join(
        character
        for character in basename
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        and character not in '<>:"/\\|?*'
    ).strip(" .")
    if not safe_name or safe_name.casefold() in {".", ".."}:
        raise ResumeUploadError("A valid PDF filename is required.", 415)
    if not safe_name.casefold().endswith(".pdf"):
        raise ResumeUploadError("Only PDF resumes are accepted.", 415)
    if len(safe_name) > 255:
        safe_name = safe_name[:251].rstrip(" .") + ".pdf"
    return safe_name


def _validate_pdf(stream: BinaryIO) -> None:
    stream.seek(0)
    header = stream.read(1024)
    if b"%PDF-" not in header:
        raise ResumeUploadError("The uploaded file is not a valid PDF.", 422)

    stream.seek(0)
    try:
        document = pymupdf.open(stream=stream.read(), filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, OSError, ValueError):
        raise ResumeUploadError("The uploaded PDF is malformed or unreadable.", 422) from None

    try:
        if document.is_encrypted or document.needs_pass:
            raise ResumeUploadError(
                "Password-protected PDFs cannot be stored.", 422
            )
        if document.page_count < 1:
            raise ResumeUploadError("The uploaded PDF contains no pages.", 422)
    except ResumeUploadError:
        raise
    except (RuntimeError, OSError, ValueError):
        raise ResumeUploadError("The uploaded PDF is malformed or unreadable.", 422) from None
    finally:
        document.close()
    stream.seek(0)


def _expiry_is_active(expires_at: datetime | None, now: datetime) -> bool:
    if expires_at is None:
        return True
    normalized_expiry = (
        expires_at.replace(tzinfo=timezone.utc)
        if expires_at.tzinfo is None
        else expires_at
    )
    return normalized_expiry > now


class ResumeDocumentService:
    def __init__(self, storage: StorageAdapter) -> None:
        self.storage = storage

    @staticmethod
    def validate_development_recruiter(
        db: Session, recruiter_id: UUID
    ) -> Recruiter:
        recruiter = db.get(Recruiter, recruiter_id)
        if recruiter is None or recruiter.status != RecruiterStatus.ACTIVE:
            raise DevelopmentRecruiterError(
                "The configured development recruiter is unavailable."
            )
        return recruiter

    @staticmethod
    def _active_storage_consent(
        db: Session, candidate_id: UUID, consent_id: UUID
    ) -> ConsentRecord:
        consent = db.get(ConsentRecord, consent_id)
        now = datetime.now(timezone.utc)
        if (
            consent is None
            or consent.candidate_id != candidate_id
            or consent.purpose != ConsentPurpose.DOCUMENT_STORAGE
            or consent.status != ConsentStatus.GRANTED
            or consent.withdrawn_at is not None
            or not _expiry_is_active(consent.expires_at, now)
        ):
            raise InvalidConsentError(
                "A valid, active document-storage consent is required."
            )
        return consent

    def upload(
        self,
        db: Session,
        candidate_id: UUID,
        consent_id: UUID,
        recruiter_id: UUID,
        source: BinaryIO,
        filename: str | None,
        media_type: str | None,
    ) -> ResumeDocument:
        safe_filename = sanitize_original_filename(filename)
        normalized_media_type = (media_type or "").split(";", 1)[0].strip().lower()
        if normalized_media_type not in ALLOWED_MEDIA_TYPES:
            raise ResumeUploadError(
                "The uploaded file must have a PDF media type.", 415
            )

        candidate = db.get(Candidate, candidate_id)
        if candidate is None:
            raise CandidateNotFoundError("Candidate not found.")
        recruiter = self.validate_development_recruiter(db, recruiter_id)
        consent = self._active_storage_consent(db, candidate_id, consent_id)

        hasher = hashlib.sha256()
        byte_size = 0
        with tempfile.TemporaryFile(mode="w+b") as staged:
            while True:
                chunk = source.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                byte_size += len(chunk)
                if byte_size > MAX_RESUME_BYTES:
                    raise ResumeUploadError(
                        "The PDF exceeds the 10 MB upload limit.", 413
                    )
                hasher.update(chunk)
                staged.write(chunk)
            if byte_size == 0:
                raise ResumeUploadError("The uploaded PDF is empty.", 422)

            _validate_pdf(staged)
            storage_key = str(uuid4())
            try:
                self.storage.store_file(staged, storage_key)
            except StorageError:
                logger.error("Resume storage failed during upload.")
                raise ResumeStorageUnavailableError(
                    "Resume storage is temporarily unavailable."
                ) from None

        document = ResumeDocument(
            candidate_id=candidate.id,
            uploaded_by_recruiter_id=recruiter.id,
            consent_record_id=consent.id,
            storage_key=storage_key,
            original_filename=safe_filename,
            media_type="application/pdf",
            byte_size=byte_size,
            sha256=hasher.hexdigest(),
            status=DocumentStatus.AVAILABLE,
        )
        try:
            db.add(document)
            db.flush()
            db.refresh(document)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            try:
                self.storage.delete_file(storage_key)
            except StorageError:
                logger.error(
                    "Resume metadata persistence failed and compensating file cleanup failed."
                )
            logger.error(
                "Resume metadata persistence failed after storing a private file."
            )
            raise ResumeDatabaseError(
                "Unable to save resume metadata."
            ) from None
        return document

    @staticmethod
    def list_for_candidate(
        db: Session, candidate_id: UUID, limit: int, offset: int
    ) -> list[ResumeDocument]:
        if db.get(Candidate, candidate_id) is None:
            raise CandidateNotFoundError("Candidate not found.")
        return list(
            db.scalars(
                select(ResumeDocument)
                .where(
                    ResumeDocument.candidate_id == candidate_id,
                    ResumeDocument.status != DocumentStatus.DELETED,
                )
                .order_by(ResumeDocument.uploaded_at.desc(), ResumeDocument.id)
                .limit(limit)
                .offset(offset)
            ).all()
        )

    @staticmethod
    def get_metadata(db: Session, resume_id: UUID) -> ResumeDocument:
        document = db.get(ResumeDocument, resume_id)
        if document is None:
            raise ResumeNotFoundError("Resume document not found.")
        return document

    def open_download(self, db: Session, resume_id: UUID) -> tuple[ResumeDocument, BinaryIO]:
        document = self.get_metadata(db, resume_id)
        if document.status != DocumentStatus.AVAILABLE:
            raise ResumeNotFoundError("Resume document is not available.")
        if document.consent_record_id is None:
            raise InvalidConsentError(
                "A valid, active document-storage consent is required."
            )
        self._active_storage_consent(
            db, document.candidate_id, document.consent_record_id
        )
        try:
            stored_file = self.storage.open_file(document.storage_key)
        except FileNotFoundError:
            logger.error("Resume metadata references an unavailable private file.")
            raise ResumeStorageUnavailableError(
                "The stored resume is temporarily unavailable."
            ) from None
        except StorageError:
            logger.error("Resume storage failed during download.")
            raise ResumeStorageUnavailableError(
                "The stored resume is temporarily unavailable."
            ) from None
        return document, stored_file

    @contextmanager
    def materialize_for_screening(
        self,
        db: Session,
        resume_id: UUID,
        application_id: UUID,
        recruiter_id: UUID,
    ) -> Iterator[Path]:
        """Temporarily materialize an authorized application's available PDF."""
        document = self.get_metadata(db, resume_id)
        application = db.get(Application, application_id)
        recruiter = self.validate_development_recruiter(db, recruiter_id)
        if (
            application is None
            or application.candidate_id != document.candidate_id
            or application.job is None
            or application.job.created_by_recruiter_id != recruiter.id
        ):
            raise ResumeNotFoundError(
                "Resume document is not available for this application."
            )
        if document.status != DocumentStatus.AVAILABLE:
            raise ResumeNotFoundError("Resume document is not available.")
        if document.consent_record_id is None:
            raise InvalidConsentError(
                "A valid, active document-storage consent is required."
            )
        consent = self._active_storage_consent(
            db, document.candidate_id, document.consent_record_id
        )
        if (
            consent.application_id is not None
            and consent.application_id != application.id
        ):
            raise InvalidConsentError(
                "A valid, active document-storage consent is required."
            )

        with tempfile.TemporaryDirectory(prefix="smarthire-screening-") as temp_dir:
            pdf_path = Path(temp_dir) / "resume.pdf"
            try:
                with self.storage.open_file(document.storage_key) as stored_file:
                    with pdf_path.open("xb") as temporary_file:
                        shutil.copyfileobj(
                            stored_file, temporary_file, length=64 * 1024
                        )
            except FileNotFoundError:
                logger.error("Resume metadata references an unavailable private file.")
                raise ResumeStorageUnavailableError(
                    "The stored resume is temporarily unavailable."
                ) from None
            except StorageError:
                logger.error("Resume storage failed during screening materialization.")
                raise ResumeStorageUnavailableError(
                    "The stored resume is temporarily unavailable."
                ) from None
            yield pdf_path

    def delete(self, db: Session, resume_id: UUID) -> None:
        document = self.get_metadata(db, resume_id)
        if document.status == DocumentStatus.DELETED:
            return

        if document.status != DocumentStatus.QUARANTINED:
            document.status = DocumentStatus.QUARANTINED
            try:
                db.commit()
            except SQLAlchemyError:
                db.rollback()
                logger.error("Unable to mark the resume unavailable for deletion.")
                raise ResumeDatabaseError(
                    "Unable to begin resume deletion."
                ) from None

        try:
            self.storage.delete_file(document.storage_key)
        except StorageError:
            logger.error("Physical resume deletion failed; document remains retryable.")
            raise ResumeStorageUnavailableError(
                "Resume deletion could not be completed; retry later."
            ) from None

        document.status = DocumentStatus.DELETED
        document.deleted_at = datetime.now(timezone.utc)
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.error(
                "Resume file was deleted but metadata finalization failed; retry is safe."
            )
            raise ResumeDatabaseError(
                "Resume file was removed, but metadata finalization must be retried."
            ) from None

    def orphaned_storage_keys(self, db: Session) -> list[str]:
        """Return keys without metadata for a future internal reconciliation job."""
        referenced_keys = set(db.scalars(select(ResumeDocument.storage_key)).all())
        try:
            stored_keys = self.storage.list_keys()
        except StorageError:
            logger.error("Private storage listing failed during reconciliation.")
            raise ResumeStorageUnavailableError(
                "Resume storage reconciliation is temporarily unavailable."
            ) from None
        return sorted(set(stored_keys) - referenced_keys)
