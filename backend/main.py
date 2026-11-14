"""FastAPI endpoints for SmartHire resume analysis."""

import ipaddress
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator
from urllib.parse import quote
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload
from starlette.responses import StreamingResponse

from backend.db.models import (
    Application,
    Candidate,
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    Recruiter,
    RequirementKind,
    ResumeDocument,
    Skill,
    ScreeningRequirementResult,
    ScreeningResult,
)
from backend.db.session import dispose_engine, get_db
from backend.schemas import (
    ApplicationRead,
    CandidateCreate,
    CandidateCreatedResponse,
    CandidateDetailResponse,
    CandidateRead,
    JobCreate,
    JobRead,
    JobUpdate,
    PersistentScreeningRequest,
    PersistentScreeningResponse,
    ResumeDocumentRead,
    ScreeningHistoryRead,
    ScreeningRequirementSnapshot,
)
from backend.services.persistent_screening import (
    PersistentScreeningError,
    ScreeningConflictError,
    ScreeningConsentError,
    ScreeningNotFoundError,
    ScreeningValidationError,
    format_screening_response,
    screen_application,
)
from backend.services.resume_documents import (
    CandidateNotFoundError,
    DevelopmentRecruiterError,
    InvalidConsentError,
    ResumeDatabaseError,
    ResumeDocumentService,
    ResumeNotFoundError,
    ResumeStorageUnavailableError,
    ResumeUploadError,
)
from backend.storage.adapter import StorageError
from backend.storage.local_filesystem import LocalFilesystemStorage
from predict import validate_text_input
from src.pdf_extractor import PDFExtractionError
from src.resume_analyzer import analyze_resume


def _job_statement():
    return select(Job).options(
        selectinload(Job.description_versions)
        .selectinload(JobDescriptionVersion.requirement_groups)
        .selectinload(JobRequirementGroup.options)
        .selectinload(JobRequirementOption.skill)
    )


def _latest_job_description(job: Job) -> JobDescriptionVersion | None:
    if not job.description_versions:
        return None
    return max(job.description_versions, key=lambda version: version.version)


def _job_response(job: Job) -> JobRead:
    description_version = _latest_job_description(job)
    required_groups: list[list[str]] = []
    preferred_groups: list[list[str]] = []
    if description_version is not None:
        for group in sorted(
            description_version.requirement_groups,
            key=lambda item: (item.kind.value, item.ordinal),
        ):
            names = sorted(option.skill.canonical_name for option in group.options)
            if group.kind == RequirementKind.REQUIRED:
                required_groups.append(names)
            else:
                preferred_groups.append(names)

    return JobRead(
        id=job.id,
        created_by_recruiter_id=job.created_by_recruiter_id,
        title=job.title,
        department=job.department,
        location=job.location,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        description=(
            description_version.description
            if description_version is not None
            else None
        ),
        description_version=(
            description_version.version if description_version is not None else None
        ),
        description_version_id=(
            description_version.id if description_version is not None else None
        ),
        required_skill_groups=required_groups,
        preferred_skill_groups=preferred_groups,
    )


def _add_requirement_groups(
    db: Session,
    description_version: JobDescriptionVersion,
    kind: RequirementKind,
    groups: list[list[str]],
) -> None:
    skill_names = {name for group in groups for name in group}
    skills_by_name = {
        skill.canonical_name: skill
        for skill in db.scalars(
            select(Skill).where(Skill.canonical_name.in_(skill_names))
        ).all()
    } if skill_names else {}

    for ordinal, names in enumerate(groups):
        requirement_group = JobRequirementGroup(
            job_description_version=description_version,
            kind=kind,
            ordinal=ordinal,
            display_text=" or ".join(names),
        )
        for name in names:
            skill = skills_by_name.get(name)
            if skill is None:
                skill = Skill(canonical_name=name)
                skills_by_name[name] = skill
                db.add(skill)
            requirement_group.options.append(JobRequirementOption(skill=skill))
        db.add(requirement_group)


def _stored_requirement_groups(
    description_version: JobDescriptionVersion | None,
    kind: RequirementKind,
) -> list[list[str]]:
    if description_version is None:
        return []
    groups = [
        group
        for group in description_version.requirement_groups
        if group.kind == kind
    ]
    return [
        sorted(option.skill.canonical_name for option in group.options)
        for group in sorted(groups, key=lambda item: item.ordinal)
    ]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Dispose any lazily-created database pool during application shutdown."""
    try:
        yield
    finally:
        dispose_engine()


app = FastAPI(title="SmartHire API", lifespan=lifespan)


def require_resume_management_access(request: Request) -> None:
    """Limit this unauthenticated milestone to explicitly enabled loopback use."""
    development_mode = os.getenv("SMART_HIRE_DEVELOPMENT_MODE", "").strip().casefold()
    if development_mode not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="Resume management is available only in configured development mode.",
        )
    client_host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="Resume management is restricted to loopback clients.",
        )


def _resume_service() -> ResumeDocumentService:
    try:
        storage = LocalFilesystemStorage()
    except StorageError:
        raise HTTPException(
            status_code=503,
            detail="Private resume storage is unavailable.",
        ) from None
    return ResumeDocumentService(storage)


def _configured_development_recruiter_id() -> UUID:
    recruiter_value = os.getenv("SMART_HIRE_DEVELOPMENT_RECRUITER_ID", "").strip()
    try:
        return UUID(recruiter_value)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=503,
            detail="A valid development recruiter identity is not configured.",
        ) from None


def _resume_http_error(error: Exception) -> HTTPException:
    if isinstance(error, CandidateNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, ResumeNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, InvalidConsentError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, DevelopmentRecruiterError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, ResumeUploadError):
        return HTTPException(status_code=error.status_code, detail=str(error))
    if isinstance(error, ResumeStorageUnavailableError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, ResumeDatabaseError):
        return HTTPException(status_code=500, detail=str(error))
    if isinstance(error, SQLAlchemyError):
        return HTTPException(
            status_code=500,
            detail="Unable to complete the resume database operation.",
        )
    return HTTPException(status_code=500, detail="Resume operation failed.")


def _resume_response(document: ResumeDocument) -> ResumeDocumentRead:
    return ResumeDocumentRead.model_validate(document)


class AnalysisResponse(BaseModel):
    label: str
    probs: dict[str, float]
    model_label: str | None = None
    model_probs: dict[str, float] = Field(default_factory=dict)
    decision_policy: str | None = None
    decision_explanation: str | None = None
    decision_evidence: dict[str, Any] = Field(default_factory=dict)
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    detected_resume_skills: list[str] = Field(default_factory=list)
    detected_jd_skills: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_skill_groups: list[list[str]] = Field(default_factory=list)
    preferred_skill_groups: list[list[str]] = Field(default_factory=list)
    matched_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    missing_preferred_skills: list[str] = Field(default_factory=list)
    unclassified_jd_skills: list[str] = Field(default_factory=list)
    total_jd_skills: int = 0
    coverage_metrics: dict[str, int | float | None] = Field(default_factory=dict)
    extraction_limitations: list[str] = Field(default_factory=list)


@app.get("/api/health/")
def health_check() -> dict[str, str]:
    """Return a simple backend liveness status."""
    return {"status": "ok"}


@app.post(
    "/api/jobs/",
    response_model=JobRead,
    status_code=status.HTTP_201_CREATED,
    tags=["jobs"],
    summary="Create a job",
)
def create_job(
    job_data: JobCreate,
    db: Annotated[Session, Depends(get_db)],
) -> JobRead:
    """Create a job and its first immutable description version."""
    try:
        recruiter = db.get(Recruiter, job_data.created_by_recruiter_id)
        if recruiter is None:
            raise HTTPException(status_code=404, detail="Recruiter not found.")

        job = Job(
            created_by_recruiter_id=recruiter.id,
            title=job_data.title,
            department=job_data.department,
            location=job_data.location,
            status=job_data.status,
        )
        description_version = JobDescriptionVersion(
            job=job,
            version=1,
            description=job_data.description,
            created_by=recruiter,
        )
        db.add_all([job, description_version])
        _add_requirement_groups(
            db,
            description_version,
            RequirementKind.REQUIRED,
            job_data.required_skill_groups,
        )
        _add_requirement_groups(
            db,
            description_version,
            RequirementKind.PREFERRED,
            job_data.preferred_skill_groups,
        )
        db.commit()
        db.refresh(job)
        db.refresh(description_version)
        return _job_response(job)
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Job conflicts with an existing database record.",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to complete the job database operation.",
        ) from None


@app.get(
    "/api/jobs/",
    response_model=list[JobRead],
    tags=["jobs"],
    summary="List jobs",
)
def list_jobs(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[JobRead]:
    """Return jobs newest first with their latest description and requirements."""
    try:
        jobs = db.scalars(
            _job_statement()
            .order_by(Job.created_at.desc(), Job.id)
            .limit(limit)
            .offset(offset)
        ).all()
        return [_job_response(job) for job in jobs]
    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve jobs.",
        ) from None


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobRead,
    tags=["jobs"],
    summary="Get a job",
)
def get_job(
    job_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> JobRead:
    """Return one job and its latest description version."""
    try:
        job = db.scalar(_job_statement().where(Job.id == job_id))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve the job.",
        ) from None
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@app.put(
    "/api/jobs/{job_id}",
    response_model=JobRead,
    tags=["jobs"],
    summary="Update a job",
)
def update_job(
    job_id: UUID,
    job_data: JobUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> JobRead:
    """Update job fields and append a description version when content changes."""
    try:
        job = db.scalar(
            _job_statement()
            .where(Job.id == job_id)
            .with_for_update()
        )
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        changes = job_data.model_dump(exclude_unset=True)
        latest_version = _latest_job_description(job)
        current_required = _stored_requirement_groups(
            latest_version, RequirementKind.REQUIRED
        )
        current_preferred = _stored_requirement_groups(
            latest_version, RequirementKind.PREFERRED
        )
        next_required = changes.get("required_skill_groups", current_required)
        next_preferred = changes.get("preferred_skill_groups", current_preferred)
        next_description = changes.get(
            "description",
            latest_version.description if latest_version is not None else None,
        )

        requirements_changed = (
            next_required != current_required or next_preferred != current_preferred
        )
        description_changed = (
            "description" in changes
            and (
                latest_version is None
                or changes["description"] != latest_version.description
            )
        )
        create_version = description_changed or requirements_changed
        if create_version and next_description is None:
            raise HTTPException(
                status_code=422,
                detail="A description is required to create a job description version.",
            )

        for field in ("title", "department", "location", "status"):
            if field in changes:
                setattr(job, field, changes[field])

        if create_version:
            description_version = JobDescriptionVersion(
                job=job,
                version=(latest_version.version + 1 if latest_version else 1),
                description=next_description,
                created_by_recruiter_id=job.created_by_recruiter_id,
            )
            db.add(description_version)
            _add_requirement_groups(
                db,
                description_version,
                RequirementKind.REQUIRED,
                next_required,
            )
            _add_requirement_groups(
                db,
                description_version,
                RequirementKind.PREFERRED,
                next_preferred,
            )

        db.commit()
        db.refresh(job)
        return _job_response(job)
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Job update conflicts with an existing database record.",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to update the job.",
        ) from None


@app.post(
    "/api/candidates/",
    response_model=CandidateCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["candidates"],
    summary="Create a candidate and application",
)
def create_candidate(
    candidate_data: CandidateCreate,
    db: Annotated[Session, Depends(get_db)],
) -> CandidateCreatedResponse:
    """Create a candidate and link them to an existing job in one transaction."""
    job = db.get(Job, candidate_data.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    candidate = Candidate(
        display_name=candidate_data.display_name,
        email=candidate_data.email,
        email_normalized=(
            candidate_data.email.casefold()
            if candidate_data.email is not None
            else None
        ),
        phone=candidate_data.phone,
    )
    application = Application(candidate=candidate, job=job)
    db.add(application)

    try:
        db.commit()
        db.refresh(candidate)
        db.refresh(application)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Candidate application conflicts with an existing record.",
        ) from None

    return CandidateCreatedResponse(
        candidate=CandidateRead.model_validate(candidate),
        application=ApplicationRead.model_validate(application),
    )


@app.get(
    "/api/candidates/",
    response_model=list[CandidateRead],
    tags=["candidates"],
    summary="List candidates",
)
def list_candidates(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[CandidateRead]:
    """List candidates in reverse creation order with bounded pagination."""
    candidates = db.scalars(
        select(Candidate)
        .order_by(Candidate.created_at.desc(), Candidate.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return [CandidateRead.model_validate(candidate) for candidate in candidates]


@app.get(
    "/api/candidates/{candidate_id}",
    response_model=CandidateDetailResponse,
    tags=["candidates"],
    summary="Get candidate details",
)
def get_candidate(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> CandidateDetailResponse:
    """Retrieve a candidate profile and linked applications."""
    candidate = db.scalar(
        select(Candidate)
        .where(Candidate.id == candidate_id)
        .options(selectinload(Candidate.applications))
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return CandidateDetailResponse.model_validate(candidate)


def _screening_history_item(result: ScreeningResult) -> ScreeningHistoryRead:
    matched_required_skills: set[str] = set()
    missing_required_skills: set[str] = set()
    for requirement_result in result.requirement_results:
        group = requirement_result.requirement_group
        if group is None:
            snapshot = ScreeningRequirementSnapshot.model_validate(
                requirement_result.requirement_snapshot
            )
            if snapshot.kind != RequirementKind.REQUIRED:
                continue
            if requirement_result.matched:
                if requirement_result.matched_skill is not None:
                    matched_required_skills.add(
                        requirement_result.matched_skill.canonical_name
                    )
            else:
                missing_required_skills.update(snapshot.skills)
            continue
        if group.kind != RequirementKind.REQUIRED:
            continue
        if requirement_result.matched:
            if requirement_result.matched_skill is not None:
                matched_required_skills.add(
                    requirement_result.matched_skill.canonical_name
                )
        else:
            missing_required_skills.update(
                option.skill.canonical_name for option in group.options
            )

    return ScreeningHistoryRead(
        id=result.id,
        application_id=result.application_id,
        resume_document_id=result.resume_document_id,
        job_description_version_id=result.job_description_version_id,
        created_at=result.created_at,
        final_label=result.final_label,
        model_label=result.model_label,
        model_probabilities=result.model_probabilities,
        decision_policy_version=result.decision_policy_version,
        decision_explanation=result.decision_explanation,
        matched_required_skills=sorted(matched_required_skills),
        missing_required_skills=sorted(missing_required_skills),
        required_group_count=result.required_group_count,
        matched_required_group_count=result.matched_required_group_count,
        required_skill_coverage=result.required_skill_coverage,
    )


@app.get(
    "/api/candidates/{candidate_id}/screenings/",
    response_model=list[ScreeningHistoryRead],
    tags=["candidates"],
    summary="Get candidate screening history",
)
def get_candidate_screenings(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScreeningHistoryRead]:
    """Retrieve persisted screening records; the legacy analysis route is stateless."""
    if db.get(Candidate, candidate_id) is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    screening_results = db.scalars(
        select(ScreeningResult)
        .join(ScreeningResult.application)
        .where(Application.candidate_id == candidate_id)
        .options(
            selectinload(ScreeningResult.requirement_results)
            .selectinload(ScreeningRequirementResult.requirement_group)
            .selectinload(JobRequirementGroup.options)
            .selectinload(JobRequirementOption.skill),
            selectinload(ScreeningResult.requirement_results).selectinload(
                ScreeningRequirementResult.matched_skill
            ),
        )
        .order_by(ScreeningResult.created_at.desc(), ScreeningResult.id)
        .limit(limit)
        .offset(offset)
    ).all()

    return [_screening_history_item(result) for result in screening_results]


@app.post(
    "/api/applications/{application_id}/screenings/",
    response_model=PersistentScreeningResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["screenings"],
    summary="Screen a candidate application with persistent explainable history",
    dependencies=[Depends(require_resume_management_access)],
)
def create_application_screening(
    application_id: UUID,
    request: PersistentScreeningRequest,
    db: Annotated[Session, Depends(get_db)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> PersistentScreeningResponse:
    key = idempotency_key.strip()
    if not key:
        raise HTTPException(
            status_code=422,
            detail="A non-empty Idempotency-Key header is required.",
        )
    if len(key) > 255:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key header must not exceed 255 characters.",
        )

    resume_service = _resume_service()
    recruiter_id = _configured_development_recruiter_id()

    try:
        screening_result = screen_application(
            db=db,
            application_id=application_id,
            resume_document_id=request.resume_document_id,
            job_description_version_id=request.job_description_version_id,
            idempotency_key=key,
            recruiter_id=recruiter_id,
            resume_service=resume_service,
        )
    except ScreeningNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ScreeningConsentError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ScreeningConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ScreeningValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ResumeNotFoundError, CandidateNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidConsentError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except DevelopmentRecruiterError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ResumeStorageUnavailableError, StorageError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to extract text from the uploaded PDF. "
                "Provide a valid text-based PDF; OCR is not available."
            ),
        ) from exc
    except (ResumeUploadError, ResumeDatabaseError) as exc:
        raise _resume_http_error(exc) from exc
    except PersistentScreeningError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Database operation failed during screening.",
        )

    return format_screening_response(screening_result)


@app.get(
    "/api/applications/{application_id}/screenings/",
    response_model=list[ScreeningHistoryRead],
    tags=["screenings"],
    summary="List persistent screening history for an application",
    dependencies=[Depends(require_resume_management_access)],
)
def get_application_screenings(
    application_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScreeningHistoryRead]:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail="Application not found.")

    screening_results = db.scalars(
        select(ScreeningResult)
        .where(ScreeningResult.application_id == application_id)
        .options(
            selectinload(ScreeningResult.requirement_results)
            .selectinload(ScreeningRequirementResult.requirement_group)
            .selectinload(JobRequirementGroup.options)
            .selectinload(JobRequirementOption.skill),
            selectinload(ScreeningResult.requirement_results).selectinload(
                ScreeningRequirementResult.matched_skill
            ),
        )
        .order_by(ScreeningResult.created_at.desc(), ScreeningResult.id)
        .limit(limit)
        .offset(offset)
    ).all()

    return [_screening_history_item(result) for result in screening_results]


@app.post(
    "/api/candidates/{candidate_id}/resumes/",
    response_model=ResumeDocumentRead,
    status_code=status.HTTP_201_CREATED,
    tags=["resumes"],
    summary="Upload a private resume document",
    dependencies=[Depends(require_resume_management_access)],
)
def upload_resume(
    candidate_id: UUID,
    file: Annotated[UploadFile, File(description="PDF resume")],
    consent_record_id: Annotated[UUID, Form()],
    db: Annotated[Session, Depends(get_db)],
) -> ResumeDocumentRead:
    """Validate consent and store a private PDF with metadata only in the database."""
    try:
        service = _resume_service()
        recruiter_id = _configured_development_recruiter_id()
        document = service.upload(
            db=db,
            candidate_id=candidate_id,
            consent_id=consent_record_id,
            recruiter_id=recruiter_id,
            source=file.file,
            filename=file.filename,
            media_type=file.content_type,
        )
    except (
        CandidateNotFoundError,
        DevelopmentRecruiterError,
        InvalidConsentError,
        ResumeDatabaseError,
        ResumeStorageUnavailableError,
        ResumeUploadError,
        SQLAlchemyError,
    ) as exc:
        raise _resume_http_error(exc) from None
    finally:
        file.file.close()
    return _resume_response(document)


@app.get(
    "/api/candidates/{candidate_id}/resumes/",
    response_model=list[ResumeDocumentRead],
    tags=["resumes"],
    summary="List a candidate's private resume metadata",
    dependencies=[Depends(require_resume_management_access)],
)
def list_candidate_resumes(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ResumeDocumentRead]:
    try:
        documents = ResumeDocumentService.list_for_candidate(
            db, candidate_id, limit, offset
        )
    except (CandidateNotFoundError, SQLAlchemyError) as exc:
        raise _resume_http_error(exc) from None
    return [_resume_response(document) for document in documents]


@app.get(
    "/api/resumes/{resume_id}",
    response_model=ResumeDocumentRead,
    tags=["resumes"],
    summary="Get private resume metadata",
    dependencies=[Depends(require_resume_management_access)],
)
def get_resume_metadata(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> ResumeDocumentRead:
    try:
        document = ResumeDocumentService.get_metadata(db, resume_id)
    except (ResumeNotFoundError, SQLAlchemyError) as exc:
        raise _resume_http_error(exc) from None
    return _resume_response(document)


@app.get(
    "/api/resumes/{resume_id}/download",
    tags=["resumes"],
    summary="Download a private resume",
    dependencies=[Depends(require_resume_management_access)],
)
def download_resume(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    service = _resume_service()
    try:
        document, stored_file = service.open_download(db, resume_id)
    except (
        ResumeNotFoundError,
        ResumeStorageUnavailableError,
        InvalidConsentError,
        SQLAlchemyError,
    ) as exc:
        raise _resume_http_error(exc) from None

    def stream_file():
        try:
            while True:
                chunk = stored_file.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            stored_file.close()

    encoded_filename = quote(document.original_filename, safe="")
    return StreamingResponse(
        stream_file(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                'attachment; filename="resume.pdf"; '
                f"filename*=UTF-8''{encoded_filename}"
            ),
            "Content-Length": str(document.byte_size),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.delete(
    "/api/resumes/{resume_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["resumes"],
    summary="Soft-delete a private resume and remove its file",
    dependencies=[Depends(require_resume_management_access)],
)
def delete_resume(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    service = _resume_service()
    try:
        service.delete(db, resume_id)
    except (
        ResumeNotFoundError,
        ResumeDatabaseError,
        ResumeStorageUnavailableError,
        SQLAlchemyError,
    ) as exc:
        raise _resume_http_error(exc) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/analyze/", response_model=AnalysisResponse)
def analyze(
    file: Annotated[UploadFile, File(description="PDF resume to analyze")],
    jd_text: Annotated[str, Form(description="Job description text")] = "",
) -> AnalysisResponse:
    """Temporarily store an uploaded PDF and return its resume analysis."""
    try:
        validate_text_input(jd_text, "Job description")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = None
    analysis_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="smarthire-upload-") as temp_dir:
            pdf_path = Path(temp_dir) / "resume.pdf"
            with pdf_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    destination.write(chunk)

            try:
                result = analyze_resume(pdf_path, jd_text)
            except (PDFExtractionError, FileNotFoundError):
                analysis_error = "pdf"
            except Exception:
                analysis_error = "internal"
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Resume analysis failed due to an internal server error.",
        ) from None

    if analysis_error == "pdf":
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to extract text from the uploaded PDF. "
                "Provide a valid text-based PDF; OCR is not available."
            ),
        )
    if analysis_error == "internal":
        raise HTTPException(
            status_code=500,
            detail="Resume analysis failed due to an internal server error.",
        )

    return AnalysisResponse(**result)