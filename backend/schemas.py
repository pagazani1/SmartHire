"""Pydantic request and response schemas for recruiter and screening APIs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from src.skill_extractor import SKILL_VOCABULARY


_CANONICAL_SKILLS = {skill.casefold(): skill for skill in SKILL_VOCABULARY}


def _validate_skill_groups(groups: list[list[str]]) -> list[list[str]]:
    canonical_groups: list[list[str]] = []
    for group in groups:
        if not group or len(group) > 20:
            raise ValueError("Each skill group must contain between 1 and 20 skills.")
        canonical_group = []
        for skill in group:
            canonical = _CANONICAL_SKILLS.get(skill.strip().casefold())
            if canonical is None:
                raise ValueError(
                    f"'{skill}' is not in the supported canonical skill vocabulary."
                )
            canonical_group.append(canonical)
        if len(set(canonical_group)) != len(canonical_group):
            raise ValueError("A skill may only appear once in a requirement group.")
        canonical_groups.append(canonical_group)
    if len(canonical_groups) > 100:
        raise ValueError("At most 100 requirement groups may be provided.")
    return canonical_groups


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
    description: str = Field(min_length=1, max_length=50000)
    required_skill_groups: list[list[str]] = Field(default_factory=list, max_length=100)
    preferred_skill_groups: list[list[str]] = Field(default_factory=list, max_length=100)

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("department", "location")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("required_skill_groups", "preferred_skill_groups")
    @classmethod
    def canonicalize_skill_groups(cls, value: list[list[str]]) -> list[list[str]]:
        return _validate_skill_groups(value)


class JobUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    department: str | None = Field(default=None, max_length=200)
    location: str | None = Field(default=None, max_length=300)
    status: JobStatus | None = None
    description: str | None = Field(default=None, max_length=50000)
    required_skill_groups: list[list[str]] | None = Field(default=None, max_length=100)
    preferred_skill_groups: list[list[str]] | None = Field(default=None, max_length=100)

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("department", "location")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("required_skill_groups", "preferred_skill_groups")
    @classmethod
    def canonicalize_optional_skill_groups(
        cls, value: list[list[str]] | None
    ) -> list[list[str]] | None:
        return None if value is None else _validate_skill_groups(value)

    @model_validator(mode="after")
    def validate_changes(self) -> JobUpdate:
        if not self.model_fields_set:
            raise ValueError("At least one job field must be provided.")
        for field in ("title", "description", "status"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null.")
        for field in ("required_skill_groups", "preferred_skill_groups"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null.")
        return self


class JobRequirementGroupRead(BaseModel):
    kind: RequirementKind
    ordinal: int
    display_text: str | None
    skills: list[str]


class JobRead(ORMModel):
    id: UUID
    created_by_recruiter_id: UUID
    title: str
    department: str | None
    location: str | None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    description: str | None
    description_version: int | None
    required_skill_groups: list[list[str]]
    preferred_skill_groups: list[list[str]]


class JobDescriptionVersionCreate(BaseModel):
    job_id: UUID
    version: int = Field(gt=0)
    description: str = Field(min_length=1)
    created_by_recruiter_id: UUID


class CandidateCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=50)
    job_id: UUID

    @field_validator("display_name", mode="before")
    @classmethod
    def strip_display_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("email", "phone")
    @classmethod
    def normalize_optional_contact_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CandidateRead(ORMModel):
    id: UUID
    display_name: str
    email: str | None
    phone: str | None
    created_at: datetime
    updated_at: datetime


class ApplicationRead(ORMModel):
    id: UUID
    job_id: UUID
    status: ApplicationStatus
    created_at: datetime


class CandidateCreatedResponse(BaseModel):
    candidate: CandidateRead
    application: ApplicationRead


class CandidateDetailResponse(CandidateRead):
    applications: list[ApplicationRead]


class ScreeningHistoryRead(BaseModel):
    id: UUID
    application_id: UUID
    resume_document_id: UUID
    job_description_version_id: UUID
    created_at: datetime
    final_label: FitLabel
    model_label: FitLabel
    model_probabilities: dict[str, float]
    decision_policy_version: str
    decision_explanation: str | None
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    required_group_count: int
    matched_required_group_count: int
    required_skill_coverage: float | None


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


class ResumeDocumentRead(ORMModel):
    id: UUID
    candidate_id: UUID
    original_filename: str
    media_type: str
    byte_size: int
    status: DocumentStatus
    uploaded_at: datetime
    deleted_at: datetime | None
    retention_until: datetime | None


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
    "ApplicationRead",
    "ApplicationCreate",
    "CandidateCreate",
    "CandidateCreatedResponse",
    "CandidateDetailResponse",
    "CandidateRead",
    "ConsentRecordCreate",
    "JobCreate",
    "JobRead",
    "JobRequirementGroupRead",
    "JobUpdate",
    "JobDescriptionVersionCreate",
    "RecruiterCreate",
    "ResumeDocumentCreate",
    "ResumeDocumentRead",
    "ScreeningEvidenceCreate",
    "ScreeningResultCreate",
    "ScreeningHistoryRead",
    "VerificationRecordCreate",
]
