"""Pydantic request schemas for future recruiter and screening endpoints.

These schemas define validation boundaries only; no CRUD endpoints or database
writes are added in this infrastructure phase.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.db.models import (
    ApplicationStatus,
    ConsentPurpose,
    ConsentStatus,
    DocumentStatus,
    EvidenceKind,
    FitLabel,
    JobStatus,
    RecruiterStatus,
    RequirementKind,
    VerificationStatus,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RecruiterCreate(BaseModel):
    email_normalized: str = Field(min_length=3, max_length=320)
    display_name: str = Field(min_length=1, max_length=200)
    status: RecruiterStatus = RecruiterStatus.ACTIVE


class JobCreate(BaseModel):
    created_by_recruiter_id: UUID
    title: str = Field(min_length=1, max_length=300)
    department: str | None = Field(default=None, max_length=200)
    location: str | None = Field(default=None, max_length=300)
    status: JobStatus = JobStatus.DRAFT


class JobDescriptionVersionCreate(BaseModel):
    job_id: UUID
    version: int = Field(gt=0)
    description: str = Field(min_length=1)
    created_by_recruiter_id: UUID


class CandidateCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    email_normalized: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=50)


class ApplicationCreate(BaseModel):
    candidate_id: UUID
    job_id: UUID
    status: ApplicationStatus = ApplicationStatus.RECEIVED
    applied_at: datetime | None = None
    source: str | None = Field(default=None, max_length=100)


class ResumeDocumentCreate(BaseModel):
    candidate_id: UUID
    uploaded_by_recruiter_id: UUID
    consent_record_id: UUID | None = None
    storage_key: str = Field(min_length=1, max_length=500)
    original_filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(gt=0)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    status: DocumentStatus = DocumentStatus.QUARANTINED
    retention_until: datetime | None = None


class ScreeningResultCreate(BaseModel):
    application_id: UUID
    resume_document_id: UUID
    job_description_version_id: UUID
    run_number: int = Field(gt=0)
    final_label: FitLabel
    model_label: FitLabel
    model_probabilities: dict[str, float] = Field(min_length=1)
    decision_policy_version: str = Field(min_length=1, max_length=120)
    model_artifact_version: str = Field(min_length=1, max_length=200)
    decision_explanation: str | None = None
    required_group_count: int = Field(default=0, ge=0)
    matched_required_group_count: int = Field(default=0, ge=0)
    required_skill_coverage: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts_and_probabilities(self) -> ScreeningResultCreate:
        if self.matched_required_group_count > self.required_group_count:
            raise ValueError("Matched required groups cannot exceed the total.")
        if not all(0 <= probability <= 1 for probability in self.model_probabilities.values()):
            raise ValueError("Model probabilities must be between zero and one.")
        if abs(sum(self.model_probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("Model probabilities must sum to approximately one.")
        return self


class ScreeningEvidenceCreate(BaseModel):
    screening_result_id: UUID
    skill_id: UUID
    evidence_kind: EvidenceKind
    source_reference: str | None = Field(default=None, max_length=500)


class ConsentRecordCreate(BaseModel):
    candidate_id: UUID
    application_id: UUID | None = None
    purpose: ConsentPurpose
    status: ConsentStatus
    captured_at: datetime
    withdrawn_at: datetime | None = None
    expires_at: datetime | None = None
    notice_version: str = Field(min_length=1, max_length=100)
    capture_method: str = Field(min_length=1, max_length=100)
    proof_reference: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_withdrawal_timestamp(self) -> ConsentRecordCreate:
        if (self.status == ConsentStatus.WITHDRAWN) != (self.withdrawn_at is not None):
            raise ValueError("withdrawn_at must be set exactly when consent is withdrawn.")
        return self


class VerificationRecordCreate(BaseModel):
    application_id: UUID
    screening_result_id: UUID | None = None
    reviewer_recruiter_id: UUID | None = None
    claim_category: str = Field(min_length=1, max_length=100)
    claim_text: str = Field(min_length=1)
    status: VerificationStatus = VerificationStatus.PENDING
    evidence_storage_key: str | None = Field(default=None, max_length=500)
    reviewer_notes: str | None = None
    reviewed_at: datetime | None = None


__all__ = [
    "ApplicationCreate",
    "CandidateCreate",
    "ConsentRecordCreate",
    "JobCreate",
    "JobDescriptionVersionCreate",
    "RecruiterCreate",
    "ResumeDocumentCreate",
    "ScreeningEvidenceCreate",
    "ScreeningResultCreate",
    "VerificationRecordCreate",
]
