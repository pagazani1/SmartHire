"""Normalized SQLAlchemy models for recruiter and screening workflows."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class RecruiterStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"


class JobStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


class ApplicationStatus(str, Enum):
    RECEIVED = "received"
    IN_REVIEW = "in_review"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ARCHIVED = "archived"


class DocumentStatus(str, Enum):
    QUARANTINED = "quarantined"
    AVAILABLE = "available"
    DELETED = "deleted"
    EXPIRED = "expired"


class FitLabel(str, Enum):
    GOOD_FIT = "Good Fit"
    POTENTIAL_FIT = "Potential Fit"
    NO_FIT = "No Fit"


class RequirementKind(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class EvidenceKind(str, Enum):
    RESUME_DETECTED = "resume_detected"
    JD_REQUIRED = "jd_required"
    JD_PREFERRED = "jd_preferred"
    JD_UNCLASSIFIED = "jd_unclassified"
    MATCHED = "matched"
    MISSING = "missing"


class ConsentPurpose(str, Enum):
    SCREENING = "screening"
    DOCUMENT_STORAGE = "document_storage"
    BACKGROUND_VERIFICATION = "background_verification"
    CONTACT = "contact"


class ConsentStatus(str, Enum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class VerificationStatus(str, Enum):
    PENDING = "Pending"
    VERIFIED = "Verified"
    UNABLE_TO_VERIFY = "Unable to Verify"
    DISCREPANCY_FOUND = "Discrepancy Found"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


def _enum(enum_class: type[Enum], name: str) -> SAEnum:
    return SAEnum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda members: [member.value for member in members],
    )


class Recruiter(TimestampMixin, Base):
    __tablename__ = "recruiters"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_normalized: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[RecruiterStatus] = mapped_column(
        _enum(RecruiterStatus, "recruiter_status"),
        nullable=False,
        default=RecruiterStatus.ACTIVE,
        server_default=RecruiterStatus.ACTIVE.value,
    )

    jobs: Mapped[list[Job]] = relationship(back_populates="created_by")
    job_description_versions: Mapped[list[JobDescriptionVersion]] = relationship(
        back_populates="created_by"
    )
    uploaded_documents: Mapped[list[ResumeDocument]] = relationship(
        back_populates="uploaded_by"
    )
    reviewed_verifications: Mapped[list[VerificationRecord]] = relationship(
        back_populates="reviewer"
    )


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_recruiter_status_updated", "created_by_recruiter_id", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_by_recruiter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("recruiters.id", ondelete="RESTRICT"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    department: Mapped[str | None] = mapped_column(String(200))
    location: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"),
        nullable=False,
        default=JobStatus.DRAFT,
        server_default=JobStatus.DRAFT.value,
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_by: Mapped[Recruiter] = relationship(back_populates="jobs")
    description_versions: Mapped[list[JobDescriptionVersion]] = relationship(
        back_populates="job", cascade="save-update, merge"
    )
    applications: Mapped[list[Application]] = relationship(back_populates="job")


class JobDescriptionVersion(TimestampMixin, Base):
    __tablename__ = "job_description_versions"
    __table_args__ = (
        UniqueConstraint("job_id", "version", name="uq_job_description_versions_job_version"),
        CheckConstraint("version > 0", name="version_positive"),
        Index("ix_job_description_versions_job_version", "job_id", "version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_recruiter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("recruiters.id", ondelete="RESTRICT"), nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="description_versions")
    created_by: Mapped[Recruiter] = relationship(back_populates="job_description_versions")
    requirement_groups: Mapped[list[JobRequirementGroup]] = relationship(
        back_populates="job_description_version", cascade="save-update, merge"
    )
    screening_results: Mapped[list[ScreeningResult]] = relationship(
        back_populates="job_description_version"
    )


class Candidate(TimestampMixin, Base):
    __tablename__ = "candidates"
    __table_args__ = (Index("ix_candidates_email_normalized", "email_normalized"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    email_normalized: Mapped[str | None] = mapped_column(String(320))
    phone: Mapped[str | None] = mapped_column(String(50))
    anonymized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    applications: Mapped[list[Application]] = relationship(back_populates="candidate")
    resume_documents: Mapped[list[ResumeDocument]] = relationship(back_populates="candidate")
    consent_records: Mapped[list[ConsentRecord]] = relationship(back_populates="candidate")


class Application(TimestampMixin, Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", name="uq_applications_job_candidate"),
        Index("ix_applications_job_status_created", "job_id", "status", "created_at"),
        Index("ix_applications_candidate_created", "candidate_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("candidates.id", ondelete="RESTRICT"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[ApplicationStatus] = mapped_column(
        _enum(ApplicationStatus, "application_status"),
        nullable=False,
        default=ApplicationStatus.RECEIVED,
        server_default=ApplicationStatus.RECEIVED.value,
    )
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(String(100))

    candidate: Mapped[Candidate] = relationship(back_populates="applications")
    job: Mapped[Job] = relationship(back_populates="applications")
    screening_results: Mapped[list[ScreeningResult]] = relationship(back_populates="application")
    consent_records: Mapped[list[ConsentRecord]] = relationship(back_populates="application")
    verification_records: Mapped[list[VerificationRecord]] = relationship(
        back_populates="application"
    )


class ConsentRecord(CreatedAtMixin, Base):
    __tablename__ = "consent_records"
    __table_args__ = (
        CheckConstraint(
            "(status = 'withdrawn' AND withdrawn_at IS NOT NULL) "
            "OR (status != 'withdrawn' AND withdrawn_at IS NULL)",
            name="withdrawal_timestamp_matches_status",
        ),
        Index("ix_consent_records_candidate_purpose_status", "candidate_id", "purpose", "status"),
        Index("ix_consent_records_application_purpose_status", "application_id", "purpose", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("candidates.id", ondelete="RESTRICT"), nullable=False
    )
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("applications.id", ondelete="RESTRICT")
    )
    purpose: Mapped[ConsentPurpose] = mapped_column(
        _enum(ConsentPurpose, "consent_purpose"), nullable=False
    )
    status: Mapped[ConsentStatus] = mapped_column(
        _enum(ConsentStatus, "consent_status"), nullable=False
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notice_version: Mapped[str] = mapped_column(String(100), nullable=False)
    capture_method: Mapped[str] = mapped_column(String(100), nullable=False)
    proof_reference: Mapped[str | None] = mapped_column(String(500))

    candidate: Mapped[Candidate] = relationship(back_populates="consent_records")
    application: Mapped[Application | None] = relationship(back_populates="consent_records")


class ResumeDocument(TimestampMixin, Base):
    __tablename__ = "resume_documents"
    __table_args__ = (
        CheckConstraint("byte_size > 0", name="byte_size_positive"),
        Index("ix_resume_documents_candidate_uploaded", "candidate_id", "uploaded_at"),
        Index("ix_resume_documents_status_retention", "status", "retention_until"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("candidates.id", ondelete="RESTRICT"), nullable=False
    )
    uploaded_by_recruiter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("recruiters.id", ondelete="RESTRICT"), nullable=False
    )
    consent_record_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("consent_records.id", ondelete="RESTRICT")
    )
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"),
        nullable=False,
        default=DocumentStatus.QUARANTINED,
        server_default=DocumentStatus.QUARANTINED.value,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    candidate: Mapped[Candidate] = relationship(back_populates="resume_documents")
    uploaded_by: Mapped[Recruiter] = relationship(back_populates="uploaded_documents")
    consent_record: Mapped[ConsentRecord | None] = relationship()
    screening_results: Mapped[list[ScreeningResult]] = relationship(
        back_populates="resume_document"
    )


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    vocabulary_version: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    requirement_options: Mapped[list[JobRequirementOption]] = relationship(back_populates="skill")
    screening_evidence: Mapped[list[ScreeningEvidence]] = relationship(back_populates="skill")


class JobRequirementGroup(Base):
    __tablename__ = "job_requirement_groups"
    __table_args__ = (
        UniqueConstraint(
            "job_description_version_id",
            "kind",
            "ordinal",
            name="uq_job_requirement_groups_version_kind_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_description_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("job_description_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[RequirementKind] = mapped_column(
        _enum(RequirementKind, "requirement_kind"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    display_text: Mapped[str | None] = mapped_column(Text)

    job_description_version: Mapped[JobDescriptionVersion] = relationship(
        back_populates="requirement_groups"
    )
    options: Mapped[list[JobRequirementOption]] = relationship(
        back_populates="requirement_group", cascade="save-update, merge"
    )
    screening_results: Mapped[list[ScreeningRequirementResult]] = relationship(
        back_populates="requirement_group"
    )


class JobRequirementOption(Base):
    __tablename__ = "job_requirement_options"

    requirement_group_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("job_requirement_groups.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("skills.id", ondelete="RESTRICT"), primary_key=True
    )

    requirement_group: Mapped[JobRequirementGroup] = relationship(back_populates="options")
    skill: Mapped[Skill] = relationship(back_populates="requirement_options")


class ScreeningResult(Base):
    __tablename__ = "screening_results"
    __table_args__ = (
        UniqueConstraint("application_id", "run_number", name="uq_screening_results_application_run"),
        UniqueConstraint(
            "application_id",
            "idempotency_key",
            name="uq_screening_results_application_idempotency_key",
        ),
        CheckConstraint("run_number > 0", name="run_number_positive"),
        CheckConstraint(
            "required_group_count >= 0 AND matched_required_group_count >= 0 "
            "AND matched_required_group_count <= required_group_count",
            name="required_group_counts_valid",
        ),
        CheckConstraint(
            "required_skill_coverage IS NULL OR "
            "(required_skill_coverage >= 0 AND required_skill_coverage <= 1)",
            name="required_skill_coverage_range",
        ),
        Index("ix_screening_results_application_created", "application_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    resume_document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("resume_documents.id", ondelete="RESTRICT"), nullable=False
    )
    job_description_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("job_description_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    final_label: Mapped[FitLabel] = mapped_column(_enum(FitLabel, "final_fit_label"), nullable=False)
    model_label: Mapped[FitLabel] = mapped_column(_enum(FitLabel, "model_fit_label"), nullable=False)
    model_probabilities: Mapped[dict[str, float]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    decision_policy_version: Mapped[str] = mapped_column(String(120), nullable=False)
    model_artifact_version: Mapped[str] = mapped_column(String(200), nullable=False)
    decision_explanation: Mapped[str | None] = mapped_column(Text)
    required_group_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matched_required_group_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    required_skill_coverage: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    application: Mapped[Application] = relationship(back_populates="screening_results")
    resume_document: Mapped[ResumeDocument] = relationship(back_populates="screening_results")
    job_description_version: Mapped[JobDescriptionVersion] = relationship(
        back_populates="screening_results"
    )
    evidence: Mapped[list[ScreeningEvidence]] = relationship(back_populates="screening_result")
    requirement_results: Mapped[list[ScreeningRequirementResult]] = relationship(
        back_populates="screening_result"
    )
    verification_records: Mapped[list[VerificationRecord]] = relationship(
        back_populates="screening_result"
    )


class ScreeningEvidence(Base):
    __tablename__ = "screening_evidence"
    __table_args__ = (
        UniqueConstraint(
            "screening_result_id",
            "skill_id",
            "evidence_kind",
            name="uq_screening_evidence_result_skill_kind",
        ),
        Index("ix_screening_evidence_skill_kind", "skill_id", "evidence_kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    screening_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("screening_results.id", ondelete="RESTRICT"), nullable=False
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("skills.id", ondelete="RESTRICT"), nullable=False
    )
    evidence_kind: Mapped[EvidenceKind] = mapped_column(
        _enum(EvidenceKind, "screening_evidence_kind"), nullable=False
    )
    source_reference: Mapped[str | None] = mapped_column(String(500))

    screening_result: Mapped[ScreeningResult] = relationship(back_populates="evidence")
    skill: Mapped[Skill] = relationship(back_populates="screening_evidence")


class ScreeningRequirementResult(Base):
    __tablename__ = "screening_requirement_results"
    __table_args__ = (
        UniqueConstraint(
            "screening_result_id",
            "requirement_group_id",
            name="uq_screening_requirement_results_result_group",
        ),
        Index(
            "ix_screening_requirement_results_group_matched",
            "requirement_group_id",
            "matched",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    screening_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("screening_results.id", ondelete="RESTRICT"), nullable=False
    )
    requirement_group_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("job_requirement_groups.id", ondelete="RESTRICT"),
    )
    requirement_snapshot: Mapped[dict[str, object] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    matched: Mapped[bool] = mapped_column(Boolean, nullable=False)
    matched_skill_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("skills.id", ondelete="RESTRICT")
    )

    screening_result: Mapped[ScreeningResult] = relationship(back_populates="requirement_results")
    requirement_group: Mapped[JobRequirementGroup | None] = relationship(
        back_populates="screening_results"
    )
    matched_skill: Mapped[Skill | None] = relationship()


class VerificationRecord(TimestampMixin, Base):
    __tablename__ = "verification_records"
    __table_args__ = (
        Index("ix_verification_records_application_status_updated", "application_id", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("applications.id", ondelete="RESTRICT"), nullable=False
    )
    screening_result_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("screening_results.id", ondelete="RESTRICT")
    )
    reviewer_recruiter_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("recruiters.id", ondelete="RESTRICT")
    )
    claim_category: Mapped[str] = mapped_column(String(100), nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[VerificationStatus] = mapped_column(
        _enum(VerificationStatus, "verification_status"),
        nullable=False,
        default=VerificationStatus.PENDING,
        server_default=VerificationStatus.PENDING.value,
    )
    evidence_storage_key: Mapped[str | None] = mapped_column(String(500))
    reviewer_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    application: Mapped[Application] = relationship(back_populates="verification_records")
    screening_result: Mapped[ScreeningResult | None] = relationship(
        back_populates="verification_records"
    )
    reviewer: Mapped[Recruiter | None] = relationship(back_populates="reviewed_verifications")
    events: Mapped[list[VerificationEvent]] = relationship(back_populates="verification_record")


class VerificationEvent(Base):
    __tablename__ = "verification_events"
    __table_args__ = (Index("ix_verification_events_record_created", "verification_record_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    verification_record_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("verification_records.id", ondelete="RESTRICT"), nullable=False
    )
    changed_by_recruiter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("recruiters.id", ondelete="RESTRICT"), nullable=False
    )
    old_status: Mapped[VerificationStatus | None] = mapped_column(
        _enum(VerificationStatus, "verification_event_old_status")
    )
    new_status: Mapped[VerificationStatus] = mapped_column(
        _enum(VerificationStatus, "verification_event_new_status"), nullable=False
    )
    change_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    verification_record: Mapped[VerificationRecord] = relationship(back_populates="events")
    changed_by: Mapped[Recruiter] = relationship()


__all__ = [
    "Application",
    "ApplicationStatus",
    "Candidate",
    "ConsentPurpose",
    "ConsentRecord",
    "ConsentStatus",
    "DocumentStatus",
    "EvidenceKind",
    "FitLabel",
    "Job",
    "JobDescriptionVersion",
    "JobRequirementGroup",
    "JobRequirementOption",
    "JobStatus",
    "Recruiter",
    "RecruiterStatus",
    "RequirementKind",
    "ResumeDocument",
    "ScreeningEvidence",
    "ScreeningRequirementResult",
    "ScreeningResult",
    "Skill",
    "VerificationEvent",
    "VerificationRecord",
    "VerificationStatus",
]
