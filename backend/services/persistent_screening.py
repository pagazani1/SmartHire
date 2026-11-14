"""Persistent resume screening service with explainable decision policy."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from backend.db.models import (
    Application,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    EvidenceKind,
    FitLabel,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    RequirementKind,
    ResumeDocument,
    ScreeningEvidence,
    ScreeningRequirementResult,
    ScreeningResult,
    Skill,
)
from backend.schemas import (
    PersistentScreeningResponse,
    ScreeningEvidenceRead,
    ScreeningRequirementResultRead,
    ScreeningRequirementSnapshot,
)
from backend.services.resume_documents import ResumeDocumentService
from src.decision_policy import apply_decision_policy
from src.resume_analyzer import analyze_resume


class PersistentScreeningError(Exception):
    """Base exception for persistent screening errors."""


class ScreeningNotFoundError(PersistentScreeningError):
    """The requested application, resume, or JD version was not found."""


class ScreeningConsentError(PersistentScreeningError):
    """Required screening or document-storage consent is missing or inactive."""


class ScreeningValidationError(PersistentScreeningError):
    """The supplied resume or JD version cannot be used for this application."""


class ScreeningConflictError(PersistentScreeningError):
    """The idempotency key conflicts with an earlier screening request."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_active_consent(
    consent: ConsentRecord,
    *,
    candidate_id: UUID,
    purpose: ConsentPurpose,
    application_id: UUID,
) -> bool:
    """Check that a consent record is active and applies to this candidate/application."""
    if consent.candidate_id != candidate_id:
        return False

    if consent.purpose != purpose:
        return False

    if consent.status != ConsentStatus.GRANTED:
        return False

    if consent.withdrawn_at is not None:
        return False

    if consent.application_id is not None:
        if consent.application_id != application_id:
            return False

    if consent.expires_at is not None:
        expiry = consent.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        if expiry <= _utc_now():
            return False

    return True


def _require_active_screening_consent(
    db: Session,
    *,
    candidate_id: UUID,
    application_id: UUID,
) -> None:
    """Require at least one active SCREENING consent for this application."""
    records = db.scalars(
        select(ConsentRecord).where(
            ConsentRecord.candidate_id == candidate_id,
            ConsentRecord.purpose == ConsentPurpose.SCREENING,
        )
    ).all()

    if not any(
        _is_active_consent(
            record,
            candidate_id=candidate_id,
            purpose=ConsentPurpose.SCREENING,
            application_id=application_id,
        )
        for record in records
    ):
        raise ScreeningConsentError(
            "Active screening consent is required before screening this resume."
        )


def _require_active_document_consent(
    db: Session,
    *,
    resume: ResumeDocument,
    application_id: UUID,
) -> None:
    """Validate the exact document-storage consent linked to the resume."""
    if resume.consent_record_id is None:
        raise ScreeningConsentError(
            "The resume does not have a linked document-storage consent record."
        )

    consent = db.get(ConsentRecord, resume.consent_record_id)
    if consent is None or not _is_active_consent(
        consent,
        candidate_id=resume.candidate_id,
        purpose=ConsentPurpose.DOCUMENT_STORAGE,
        application_id=application_id,
    ):
        raise ScreeningConsentError(
            "Active document-storage consent is required for this resume."
        )


def _validate_screening_inputs(
    db: Session,
    *,
    application_id: UUID,
    resume_document_id: UUID,
    job_description_version_id: UUID,
) -> tuple[Application, ResumeDocument, JobDescriptionVersion]:
    """Validate that the resume and JD version belong to the application."""
    application = db.get(Application, application_id)
    if application is None:
        raise ScreeningNotFoundError("Application not found.")

    resume = db.get(ResumeDocument, resume_document_id)
    if resume is None:
        raise ScreeningNotFoundError("Resume document not found.")

    jd_version = db.get(JobDescriptionVersion, job_description_version_id)
    if jd_version is None:
        raise ScreeningNotFoundError("Job description version not found.")

    if resume.candidate_id != application.candidate_id:
        raise ScreeningValidationError(
            "The selected resume does not belong to this application's candidate."
        )

    if jd_version.job_id != application.job_id:
        raise ScreeningValidationError(
            "The selected job description version does not belong to this application."
        )

    if resume.status != DocumentStatus.AVAILABLE:
        raise ScreeningValidationError(
            "The selected resume is not available for screening."
        )

    _require_active_document_consent(
        db,
        resume=resume,
        application_id=application_id,
    )

    _require_active_screening_consent(
        db,
        candidate_id=application.candidate_id,
        application_id=application_id,
    )

    return application, resume, jd_version


def _get_model_artifact_version() -> str:
    """Return a deterministic fingerprint of the two model artifacts."""
    project_root = Path(__file__).resolve().parents[2]
    artifact_paths = (
        project_root / "models" / "tfidf_vectorizer.joblib",
        project_root / "models" / "logistic_regression.joblib",
    )

    digest = hashlib.sha256()
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise PersistentScreeningError(
                "A required screening model artifact is unavailable."
            )

        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)

    return f"sha256:{digest.hexdigest()}"


def _build_structured_skill_analysis(
    analysis: dict[str, Any],
    requirement_groups: list[JobRequirementGroup],
) -> dict[str, Any]:
    """Build skill-analysis input using persisted structured JD requirements."""
    resume_skills = set(analysis.get("resume_skills", []))

    required_groups: list[set[str]] = []
    preferred_groups: list[set[str]] = []

    required_skills: set[str] = set()
    preferred_skills: set[str] = set()

    for group in requirement_groups:
        group_skills = {
            option.skill.canonical_name
            for option in group.options
            if option.skill is not None and option.skill.is_active
        }

        if not group_skills:
            continue

        if group.kind == RequirementKind.REQUIRED:
            required_groups.append(group_skills)
            required_skills.update(group_skills)
        elif group.kind == RequirementKind.PREFERRED:
            preferred_groups.append(group_skills)
            preferred_skills.update(group_skills)

    jd_skills = required_skills | preferred_skills
    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills

    matched_required = resume_skills & required_skills
    missing_required = required_skills - resume_skills

    matched_preferred = resume_skills & preferred_skills
    missing_preferred = preferred_skills - resume_skills

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "unclassified_jd_skills": set(),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "required_skill_groups": required_groups,
        "preferred_skill_groups": preferred_groups,
    }


def _apply_structured_requirements(
    analysis: dict[str, Any],
    requirement_groups: list[JobRequirementGroup],
) -> dict[str, Any]:
    """Reapply the existing decision policy using structured JD requirements."""
    skill_analysis = _build_structured_skill_analysis(
        analysis,
        requirement_groups,
    )

    model_prediction = {
        "label": analysis["model_label"],
        "probs": analysis["model_probs"],
    }

    decision = apply_decision_policy(
        model_prediction,
        skill_analysis,
    )

    result = dict(analysis)
    result.update(decision)

    result["resume_skills"] = sorted(skill_analysis["resume_skills"])
    result["jd_skills"] = sorted(skill_analysis["jd_skills"])
    result["required_skills"] = sorted(skill_analysis["required_skills"])
    result["preferred_skills"] = sorted(skill_analysis["preferred_skills"])
    result["unclassified_jd_skills"] = []
    result["matched_skills"] = sorted(skill_analysis["matched_skills"])
    result["missing_skills"] = sorted(skill_analysis["missing_skills"])
    result["matched_required_skills"] = sorted(
        skill_analysis["matched_required_skills"]
    )
    result["missing_required_skills"] = sorted(
        skill_analysis["missing_required_skills"]
    )
    result["matched_preferred_skills"] = sorted(
        skill_analysis["matched_preferred_skills"]
    )
    result["missing_preferred_skills"] = sorted(
        skill_analysis["missing_preferred_skills"]
    )
    result["required_skill_groups"] = decision["decision_evidence"][
        "required_skill_groups"
    ]
    result["preferred_skill_groups"] = decision["decision_evidence"][
        "preferred_skill_groups"
    ]

    return result


def _find_existing_by_idempotency_key(
    db: Session,
    *,
    application_id: UUID,
    idempotency_key: str,
) -> ScreeningResult | None:
    return db.scalar(
        select(ScreeningResult)
        .options(
            selectinload(ScreeningResult.evidence).selectinload(ScreeningEvidence.skill),
            selectinload(ScreeningResult.requirement_results)
            .selectinload(ScreeningRequirementResult.requirement_group)
            .selectinload(JobRequirementGroup.options)
            .selectinload(JobRequirementOption.skill),
            selectinload(ScreeningResult.requirement_results).selectinload(
                ScreeningRequirementResult.matched_skill
            ),
        )
        .where(
            ScreeningResult.application_id == application_id,
            ScreeningResult.idempotency_key == idempotency_key,
        )
    )


def _load_screening_result(db: Session, result_id: UUID) -> ScreeningResult:
    result = db.scalar(
        select(ScreeningResult)
        .options(
            selectinload(ScreeningResult.evidence).selectinload(ScreeningEvidence.skill),
            selectinload(ScreeningResult.requirement_results)
            .selectinload(ScreeningRequirementResult.requirement_group)
            .selectinload(JobRequirementGroup.options)
            .selectinload(JobRequirementOption.skill),
            selectinload(ScreeningResult.requirement_results).selectinload(
                ScreeningRequirementResult.matched_skill
            ),
        )
        .where(ScreeningResult.id == result_id)
    )
    if result is None:
        raise PersistentScreeningError("Screening result could not be reloaded.")
    return result


def _get_or_create_skills(db: Session, skill_names: set[str]) -> dict[str, Skill]:
    """Retrieve existing skills by canonical name and insert missing ones."""
    if not skill_names:
        return {}

    existing_skills = {
        skill.canonical_name: skill
        for skill in db.scalars(
            select(Skill).where(Skill.canonical_name.in_(skill_names))
        ).all()
    }

    new_skills = [
        Skill(canonical_name=name, is_active=True)
        for name in skill_names
        if name not in existing_skills
    ]
    if new_skills:
        db.add_all(new_skills)
        db.flush()
        for skill in new_skills:
            existing_skills[skill.canonical_name] = skill

    return existing_skills


def screen_application(
    db: Session,
    *,
    application_id: UUID,
    resume_document_id: UUID,
    job_description_version_id: UUID,
    idempotency_key: str | None = None,
    recruiter_id: UUID,
    resume_service: ResumeDocumentService,
) -> ScreeningResult:
    """Execute and atomically persist explainable screening for an application."""
    normalized_key = idempotency_key.strip() if idempotency_key else None

    # Step 1: Pre-inference validation and fast-path idempotency check.
    _validate_screening_inputs(
        db,
        application_id=application_id,
        resume_document_id=resume_document_id,
        job_description_version_id=job_description_version_id,
    )

    if normalized_key is not None:
        existing = _find_existing_by_idempotency_key(
            db,
            application_id=application_id,
            idempotency_key=normalized_key,
        )
        if existing is not None:
            if (
                existing.resume_document_id == resume_document_id
                and existing.job_description_version_id == job_description_version_id
            ):
                return existing
            raise ScreeningConflictError(
                "An existing screening run exists for this idempotency key with different inputs."
            )

    # Load structured requirement groups from the database version.
    jd_version = db.get(JobDescriptionVersion, job_description_version_id)
    if jd_version is None:
        raise ScreeningNotFoundError("Job description version not found.")

    structured_groups = list(
        db.scalars(
            select(JobRequirementGroup)
            .options(
                selectinload(JobRequirementGroup.options).selectinload(
                    JobRequirementOption.skill
                )
            )
            .where(
                JobRequirementGroup.job_description_version_id
                == job_description_version_id
            )
            .order_by(JobRequirementGroup.kind, JobRequirementGroup.ordinal)
        ).all()
    )
    jd_text = jd_version.description
    has_structured_groups = len(structured_groups) > 0

    # Step 2: Extraction and inference outside of database write lock.
    model_artifact_version = _get_model_artifact_version()

    with resume_service.materialize_for_screening(
        db=db,
        resume_id=resume_document_id,
        application_id=application_id,
        recruiter_id=recruiter_id,
    ) as pdf_path:
        analysis = analyze_resume(pdf_path, jd_text)

    if has_structured_groups:
        final_analysis = _apply_structured_requirements(analysis, structured_groups)
    else:
        final_analysis = analysis

    # Step 3: Atomic persistence with revalidation and race resolution.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Recheck idempotency before persistence.
            if normalized_key is not None:
                existing = _find_existing_by_idempotency_key(
                    db,
                    application_id=application_id,
                    idempotency_key=normalized_key,
                )
                if existing is not None:
                    if (
                        existing.resume_document_id == resume_document_id
                        and existing.job_description_version_id
                        == job_description_version_id
                    ):
                        return existing
                    raise ScreeningConflictError(
                        "An existing screening run exists for this idempotency key with different inputs."
                    )

            # Re-validate inputs and active consents right before committing.
            _validate_screening_inputs(
                db,
                application_id=application_id,
                resume_document_id=resume_document_id,
                job_description_version_id=job_description_version_id,
            )

            # Compute next run number for this application.
            max_run = db.scalar(
                select(func.coalesce(func.max(ScreeningResult.run_number), 0))
                .where(ScreeningResult.application_id == application_id)
            )
            run_number = (max_run or 0) + 1

            # Collect all skill names to ensure they exist as Skill rows.
            all_skill_names: set[str] = set()
            all_skill_names.update(final_analysis.get("resume_skills", []))
            all_skill_names.update(final_analysis.get("jd_skills", []))
            all_skill_names.update(final_analysis.get("matched_skills", []))
            all_skill_names.update(final_analysis.get("missing_skills", []))
            all_skill_names.update(final_analysis.get("required_skills", []))
            all_skill_names.update(final_analysis.get("preferred_skills", []))
            all_skill_names.update(final_analysis.get("unclassified_jd_skills", []))

            for group in structured_groups:
                for option in group.options:
                    if option.skill is not None:
                        all_skill_names.add(option.skill.canonical_name)

            skills_by_name = _get_or_create_skills(db, all_skill_names)

            decision_evidence = final_analysis["decision_evidence"]
            screening_result = ScreeningResult(
                application_id=application_id,
                idempotency_key=normalized_key,
                resume_document_id=resume_document_id,
                job_description_version_id=job_description_version_id,
                run_number=run_number,
                final_label=FitLabel(final_analysis["label"]),
                model_label=FitLabel(final_analysis["model_label"]),
                model_probabilities=dict(final_analysis["model_probs"]),
                decision_policy_version=str(final_analysis["decision_policy"]),
                model_artifact_version=model_artifact_version,
                decision_explanation=final_analysis.get("decision_explanation"),
                required_group_count=int(
                    decision_evidence["required_skill_group_count"]
                ),
                matched_required_group_count=int(
                    decision_evidence["matched_required_group_count"]
                ),
                required_skill_coverage=decision_evidence["required_skill_coverage"],
            )
            db.add(screening_result)
            db.flush()

            # Persist explainable evidence without raw resume text or file paths.
            evidence_pairs: set[tuple[str, EvidenceKind]] = set()
            for name in final_analysis.get("resume_skills", []):
                evidence_pairs.add((name, EvidenceKind.RESUME_DETECTED))
            for name in final_analysis.get("required_skills", []):
                evidence_pairs.add((name, EvidenceKind.JD_REQUIRED))
            for name in final_analysis.get("preferred_skills", []):
                evidence_pairs.add((name, EvidenceKind.JD_PREFERRED))
            for name in final_analysis.get("unclassified_jd_skills", []):
                evidence_pairs.add((name, EvidenceKind.JD_UNCLASSIFIED))
            for name in final_analysis.get("matched_skills", []):
                evidence_pairs.add((name, EvidenceKind.MATCHED))
            for name in final_analysis.get("missing_skills", []):
                evidence_pairs.add((name, EvidenceKind.MISSING))

            for skill_name, kind in evidence_pairs:
                skill = skills_by_name[skill_name]
                evidence_record = ScreeningEvidence(
                    screening_result_id=screening_result.id,
                    skill_id=skill.id,
                    evidence_kind=kind,
                    source_reference=None,
                )
                db.add(evidence_record)

            # Persist requirement results and snapshots.
            resume_skill_set = set(final_analysis.get("resume_skills", []))

            if has_structured_groups:
                for group in structured_groups:
                    group_skills = sorted(
                        option.skill.canonical_name
                        for option in group.options
                        if option.skill is not None and option.skill.is_active
                    )
                    if not group_skills:
                        continue

                    matched_name = next(
                        (name for name in group_skills if name in resume_skill_set),
                        None,
                    )
                    matched = matched_name is not None
                    matched_skill_id = (
                        skills_by_name[matched_name].id if matched else None
                    )

                    snapshot = {
                        "canonical_identifier": str(group.id),
                        "name": group.display_text or " or ".join(group_skills),
                        "kind": group.kind.value,
                        "skills": group_skills,
                        "source": "structured",
                    }

                    req_result = ScreeningRequirementResult(
                        screening_result_id=screening_result.id,
                        requirement_group_id=group.id,
                        requirement_snapshot=snapshot,
                        matched=matched,
                        matched_skill_id=matched_skill_id,
                    )
                    db.add(req_result)
            else:
                for i, group_skills in enumerate(
                    final_analysis.get("required_skill_groups", [])
                ):
                    skills_list = sorted(group_skills)
                    if not skills_list:
                        continue

                    matched_name = next(
                        (name for name in skills_list if name in resume_skill_set),
                        None,
                    )
                    matched = matched_name is not None
                    matched_skill_id = (
                        skills_by_name[matched_name].id if matched else None
                    )

                    snapshot = {
                        "canonical_identifier": f"jd-required-{i}",
                        "name": " or ".join(skills_list),
                        "kind": RequirementKind.REQUIRED.value,
                        "skills": skills_list,
                        "source": "jd_text_fallback",
                    }

                    req_result = ScreeningRequirementResult(
                        screening_result_id=screening_result.id,
                        requirement_group_id=None,
                        requirement_snapshot=snapshot,
                        matched=matched,
                        matched_skill_id=matched_skill_id,
                    )
                    db.add(req_result)

                for j, group_skills in enumerate(
                    final_analysis.get("preferred_skill_groups", [])
                ):
                    skills_list = sorted(group_skills)
                    if not skills_list:
                        continue

                    matched_name = next(
                        (name for name in skills_list if name in resume_skill_set),
                        None,
                    )
                    matched = matched_name is not None
                    matched_skill_id = (
                        skills_by_name[matched_name].id if matched else None
                    )

                    snapshot = {
                        "canonical_identifier": f"jd-preferred-{j}",
                        "name": " or ".join(skills_list),
                        "kind": RequirementKind.PREFERRED.value,
                        "skills": skills_list,
                        "source": "jd_text_fallback",
                    }

                    req_result = ScreeningRequirementResult(
                        screening_result_id=screening_result.id,
                        requirement_group_id=None,
                        requirement_snapshot=snapshot,
                        matched=matched,
                        matched_skill_id=matched_skill_id,
                    )
                    db.add(req_result)

            db.commit()
            return _load_screening_result(db, screening_result.id)

        except IntegrityError as exc:
            db.rollback()
            # If concurrent request used the same idempotency key:
            if normalized_key is not None:
                existing = _find_existing_by_idempotency_key(
                    db,
                    application_id=application_id,
                    idempotency_key=normalized_key,
                )
                if existing is not None:
                    if (
                        existing.resume_document_id == resume_document_id
                        and existing.job_description_version_id
                        == job_description_version_id
                    ):
                        return existing
                    raise ScreeningConflictError(
                        "An existing screening run exists for this idempotency key with different inputs."
                    ) from exc

            # If run_number collision occurred due to concurrent requests, retry.
            if attempt == max_retries - 1:
                raise PersistentScreeningError(
                    "Unable to allocate a unique screening run number after concurrent attempts."
                ) from exc

    raise PersistentScreeningError("Screening persistence failed unexpectedly.")


def format_screening_response(
    result: ScreeningResult,
) -> PersistentScreeningResponse:
    """Format a database ScreeningResult into the strict PersistentScreeningResponse schema."""
    coverage = result.required_skill_coverage
    coverage_percentage = (
        round(coverage * 100.0, 4) if coverage is not None else None
    )

    evidence_reads = [
        ScreeningEvidenceRead(
            skill=ev.skill.canonical_name,
            evidence_kind=ev.evidence_kind,
        )
        for ev in sorted(
            result.evidence,
            key=lambda item: (item.evidence_kind.value, item.skill.canonical_name),
        )
    ]

    requirement_reads: list[ScreeningRequirementResultRead] = []
    for req_result in sorted(
        result.requirement_results,
        key=lambda item: (
            item.requirement_group.ordinal
            if item.requirement_group is not None
            else 0,
            item.id,
        ),
    ):
        matched_skill_name = (
            req_result.matched_skill.canonical_name
            if req_result.matched_skill is not None
            else None
        )

        if req_result.requirement_snapshot is not None:
            snapshot = ScreeningRequirementSnapshot.model_validate(
                req_result.requirement_snapshot
            )
            requirement_reads.append(
                ScreeningRequirementResultRead(
                    requirement_group_id=req_result.requirement_group_id,
                    requirement_snapshot=snapshot,
                    kind=snapshot.kind,
                    skills=snapshot.skills,
                    matched=req_result.matched,
                    matched_skill=matched_skill_name,
                )
            )
        elif req_result.requirement_group is not None:
            group = req_result.requirement_group
            skills = sorted(
                opt.skill.canonical_name
                for opt in group.options
                if opt.skill is not None and opt.skill.is_active
            )
            requirement_reads.append(
                ScreeningRequirementResultRead(
                    requirement_group_id=group.id,
                    requirement_snapshot=None,
                    kind=group.kind,
                    skills=skills,
                    matched=req_result.matched,
                    matched_skill=matched_skill_name,
                )
            )

    return PersistentScreeningResponse(
        id=result.id,
        application_id=result.application_id,
        resume_document_id=result.resume_document_id,
        job_description_version_id=result.job_description_version_id,
        run_number=result.run_number,
        final_label=result.final_label,
        model_label=result.model_label,
        model_probabilities=result.model_probabilities,
        decision_policy_version=result.decision_policy_version,
        model_artifact_version=result.model_artifact_version,
        decision_explanation=result.decision_explanation,
        required_group_count=result.required_group_count,
        matched_required_group_count=result.matched_required_group_count,
        missing_required_group_count=max(
            0, result.required_group_count - result.matched_required_group_count
        ),
        required_skill_coverage=coverage,
        required_skill_coverage_percentage=coverage_percentage,
        evidence=evidence_reads,
        requirement_results=requirement_reads,
    )
