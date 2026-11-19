"""FastAPI endpoints for SmartHire resume analysis."""

import tempfile
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, AsyncIterator
from urllib.parse import quote
from uuid import UUID

from fastapi.responses import JSONResponse
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
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload
from starlette.responses import StreamingResponse

from backend.db.models import (
    Application,
    Candidate,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    Recruiter,
    RecruiterRole,
    RecruiterStatus,
    RequirementKind,
    ResumeDocument,
    Skill,
    ScreeningRequirementResult,
    ScreeningResult,
)
from backend.db.session import dispose_engine, get_db
from backend.schemas import (
    ApplicationRead,
    CandidateAnonymizeResponse,
    CandidateCreate,
    CandidateCreatedResponse,
    CandidateDetailResponse,
    CandidateRead,
    ConsentGrantRequest,
    ConsentRead,
    JobCreate,
    JobRead,
    JobUpdate,
    LoginRequest,
    PersistentScreeningRequest,
    PersistentScreeningResponse,
    RecruiterAccountCreate,
    RecruiterRead,
    SkillVocabularyResponse,
    RetentionSweepResponse,
    ResumeDocumentRead,
    ScreeningHistoryRead,
    ScreeningRequirementSnapshot,
    TokenResponse,
)
from backend.rate_limit import account_login_limiter, client_login_limiter
from backend.security import (
    burn_password_verification_time,
    create_access_token,
    ensure_application_access,
    ensure_candidate_access,
    ensure_job_access,
    ensure_resume_access,
    get_current_recruiter,
    hash_password,
    is_development_mode_enabled,
    is_loopback_request,
    log_security_event,
    recruiter_owns_candidate,
    validate_runtime_configuration,
    verify_password,
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
    MAX_RESUME_BYTES,
    CandidateNotFoundError,
    DevelopmentRecruiterError,
    InvalidConsentError,
    ResumeDatabaseError,
    ResumeDocumentService,
    ResumeNotFoundError,
    ResumeStorageUnavailableError,
    ResumeUploadError,
)
from backend.services.retention import run_retention_cycle
from backend.storage.adapter import StorageError
from backend.storage.local_filesystem import LocalFilesystemStorage
from predict import validate_text_input
from src.pdf_extractor import PDFExtractionError
from src.skill_extractor import SKILL_VOCABULARY
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
    """Validate production settings at startup; dispose the database pool at shutdown."""
    validate_runtime_configuration()
    try:
        yield
    finally:
        dispose_engine()


app = FastAPI(title="SmartHire API", lifespan=lifespan)
logger = logging.getLogger("smarthire.api")


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request, exc: OperationalError) -> JSONResponse:
    """Turn a database connection failure into a clear 503 instead of an opaque 500.

    Only the exception class is logged: the message can contain connection details.
    """
    logger.error("Database operation failed: %s", exc.__class__.__name__)
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The database is unavailable. Check that PostgreSQL is running and "
                "that DATABASE_URL in .env is correct."
            )
        },
    )




def require_resume_management_access(request: Request) -> None:
    """Verify explicit local development mode and loopback origin when called directly."""
    if not is_development_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Private resume management endpoints require explicit development mode.",
        )
    if not is_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Private resume management endpoints are restricted to loopback clients.",
        )


def _get_candidate_or_404(db: Session, candidate_id: UUID) -> Candidate:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return candidate


def _aware_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _resume_service() -> ResumeDocumentService:
    try:
        storage = LocalFilesystemStorage()
    except StorageError:
        raise HTTPException(
            status_code=503,
            detail="Private resume storage is unavailable.",
        ) from None
    return ResumeDocumentService(storage)


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
    "/api/auth/login",
    response_model=TokenResponse,
    tags=["auth"],
    summary="Authenticate a recruiter and issue a signed bearer token",
)
def login_recruiter(
    credentials: LoginRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    normalized_email = credentials.email.strip().casefold()
    client_ip = request.client.host if request.client is not None else "unknown"
    account_key = f"{normalized_email}|{client_ip}"

    retry_after = max(
        account_login_limiter.retry_after(account_key),
        client_login_limiter.retry_after(client_ip),
    )
    if retry_after:
        log_security_event(
            "auth.login_rate_limited",
            outcome="denied",
            reason="too_many_failures",
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed sign-in attempts. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    recruiter = db.scalar(
        select(Recruiter).where(Recruiter.email_normalized == normalized_email)
    )
    password_ok = False
    if recruiter is not None and recruiter.password_hash:
        password_ok = verify_password(credentials.password, recruiter.password_hash)
    else:
        # Spend the same time as a real check so unknown accounts are not detectable.
        burn_password_verification_time(credentials.password)

    if (
        recruiter is None
        or recruiter.status != RecruiterStatus.ACTIVE
        or not password_ok
    ):
        account_login_limiter.record_failure(account_key)
        client_login_limiter.record_failure(client_ip)
        log_security_event(
            "auth.login_failed",
            outcome="denied",
            recruiter_id=recruiter.id if recruiter is not None else None,
            reason="invalid_credentials_or_inactive",
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    account_login_limiter.reset(account_key)
    token, expires_at = create_access_token(recruiter)
    log_security_event(
        "auth.login_success",
        outcome="success",
        recruiter_id=recruiter.id,
        db=db,
    )
    return TokenResponse(
        access_token=token,
        expires_at=expires_at,
        recruiter=RecruiterRead.model_validate(recruiter),
    )


@app.get(
    "/api/auth/me",
    response_model=RecruiterRead,
    tags=["auth"],
    summary="Return the currently authenticated recruiter profile",
)
def read_authenticated_recruiter(
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> RecruiterRead:
    return RecruiterRead.model_validate(current_recruiter)


@app.post(
    "/api/admin/retention/sweep",
    response_model=RetentionSweepResponse,
    tags=["admin"],
    summary="Purge resume documents past their retention deadline",
)
def run_retention_sweep(
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> RetentionSweepResponse:
    if current_recruiter.role != RecruiterRole.ADMIN:
        log_security_event(
            "rbac.retention_sweep_denied",
            outcome="denied",
            recruiter_id=current_recruiter.id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges are required to trigger retention sweeps.",
        )

    try:
        storage = LocalFilesystemStorage()
    except StorageError:
        raise HTTPException(
            status_code=503,
            detail="Private resume storage is unavailable.",
        ) from None

    summary = run_retention_cycle(
        db,
        storage,
        actor_recruiter_id=current_recruiter.id,
    )
    return RetentionSweepResponse(
        inspected_count=summary.inspected_count,
        expired_count=summary.expired_count,
        failed_count=summary.failed_count,
        consents_expired=summary.consents_expired,
    )


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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> JobRead:
    """Create a job and its first immutable description version."""
    owner_id = job_data.created_by_recruiter_id or current_recruiter.id
    if owner_id != current_recruiter.id and current_recruiter.role != RecruiterRole.ADMIN:
        log_security_event(
            "rbac.job_create_denied",
            outcome="denied",
            recruiter_id=current_recruiter.id,
            resource_type="recruiter",
            resource_id=owner_id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create jobs for your own account.",
        )
    try:
        recruiter = db.get(Recruiter, owner_id)
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
        log_security_event(
            "job.created",
            outcome="success",
            recruiter_id=current_recruiter.id,
            resource_type="job",
            resource_id=job.id,
            db=db,
        )
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[JobRead]:
    """Return jobs newest first with their latest description and requirements."""
    try:
        statement = _job_statement()
        if current_recruiter.role != RecruiterRole.ADMIN:
            statement = statement.where(
                Job.created_by_recruiter_id == current_recruiter.id
            )
        jobs = db.scalars(
            statement.order_by(Job.created_at.desc(), Job.id)
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
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
    ensure_job_access(current_recruiter, job, db)
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
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
        ensure_job_access(current_recruiter, job, db)

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
                created_by_recruiter_id=current_recruiter.id,
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> CandidateCreatedResponse:
    """Create a candidate and link them to an existing job in one transaction."""
    job = db.get(Job, candidate_data.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    ensure_job_access(current_recruiter, job, db)

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

    log_security_event(
        "candidate.created",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="candidate",
        resource_id=candidate.id,
        db=db,
    )
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[CandidateRead]:
    """List candidates in reverse creation order with bounded pagination."""
    statement = select(Candidate)
    if current_recruiter.role != RecruiterRole.ADMIN:
        # Only candidates who applied to one of this recruiter's jobs.
        statement = statement.where(
            Candidate.id.in_(
                select(Application.candidate_id)
                .join(Job, Application.job_id == Job.id)
                .where(Job.created_by_recruiter_id == current_recruiter.id)
            )
        )
    candidates = db.scalars(
        statement.order_by(Candidate.created_at.desc(), Candidate.id)
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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> CandidateDetailResponse:
    """Retrieve a candidate profile and the applications to this recruiter's jobs."""
    candidate = db.scalar(
        select(Candidate)
        .where(Candidate.id == candidate_id)
        .options(selectinload(Candidate.applications).selectinload(Application.job))
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    ensure_candidate_access(db, current_recruiter, candidate_id)
    response = CandidateDetailResponse.model_validate(candidate)
    if current_recruiter.role != RecruiterRole.ADMIN:
        visible_ids = {
            application.id
            for application in candidate.applications
            if application.job is not None
            and application.job.created_by_recruiter_id == current_recruiter.id
        }
        response.applications = [
            application
            for application in response.applications
            if application.id in visible_ids
        ]
    return response


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
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScreeningHistoryRead]:
    """Retrieve persisted screening records; the legacy analysis route is stateless."""
    _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)

    history_statement = (
        select(ScreeningResult)
        .join(ScreeningResult.application)
        .where(Application.candidate_id == candidate_id)
    )
    if current_recruiter.role != RecruiterRole.ADMIN:
        # Never reveal screenings the candidate had for another recruiter's jobs.
        history_statement = history_statement.join(
            Job, Application.job_id == Job.id
        ).where(Job.created_by_recruiter_id == current_recruiter.id)

    screening_results = db.scalars(
        history_statement
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
)
def create_application_screening(
    application_id: UUID,
    request: PersistentScreeningRequest,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
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

    application = db.get(Application, application_id)
    if application is not None:
        ensure_application_access(current_recruiter, application)

    resume_service = _resume_service()

    try:
        screening_result = screen_application(
            db=db,
            application_id=application_id,
            resume_document_id=request.resume_document_id,
            job_description_version_id=request.job_description_version_id,
            idempotency_key=key,
            recruiter_id=current_recruiter.id,
            resume_service=resume_service,
        )
    except ScreeningNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ScreeningConsentError as exc:
        log_security_event(
            "screening.consent_denied",
            outcome="denied",
            recruiter_id=current_recruiter.id,
            resource_type="application",
            resource_id=application_id,
            db=db,
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ScreeningConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ScreeningValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ResumeNotFoundError, CandidateNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidConsentError as exc:
        log_security_event(
            "screening.consent_denied",
            outcome="denied",
            recruiter_id=current_recruiter.id,
            resource_type="application",
            resource_id=application_id,
            db=db,
        )
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

    log_security_event(
        "screening.executed",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="screening_result",
        resource_id=screening_result.id,
        db=db,
    )
    return format_screening_response(screening_result)


@app.get(
    "/api/applications/{application_id}/screenings/",
    response_model=list[ScreeningHistoryRead],
    tags=["screenings"],
    summary="List persistent screening history for an application",
)
def get_application_screenings(
    application_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScreeningHistoryRead]:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    ensure_application_access(current_recruiter, application)

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
)
def upload_resume(
    candidate_id: UUID,
    file: Annotated[UploadFile, File(description="PDF resume")],
    consent_record_id: Annotated[UUID, Form()],
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> ResumeDocumentRead:
    """Validate consent and store a private PDF with metadata only in the database."""
    _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)
    try:
        service = _resume_service()
        document = service.upload(
            db=db,
            candidate_id=candidate_id,
            consent_id=consent_record_id,
            recruiter_id=current_recruiter.id,
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

    log_security_event(
        "resume.uploaded",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="resume_document",
        resource_id=document.id,
        db=db,
    )
    return _resume_response(document)


@app.get(
    "/api/candidates/{candidate_id}/resumes/",
    response_model=list[ResumeDocumentRead],
    tags=["resumes"],
    summary="List a candidate's private resume metadata",
)
def list_candidate_resumes(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ResumeDocumentRead]:
    _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)
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
)
def get_resume_metadata(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> ResumeDocumentRead:
    try:
        document = ResumeDocumentService.get_metadata(db, resume_id)
    except (ResumeNotFoundError, SQLAlchemyError) as exc:
        raise _resume_http_error(exc) from None
    ensure_resume_access(db, current_recruiter, document)
    return _resume_response(document)


@app.get(
    "/api/resumes/{resume_id}/download",
    tags=["resumes"],
    summary="Download a private resume",
)
def download_resume(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> StreamingResponse:
    service = _resume_service()
    try:
        document_meta = ResumeDocumentService.get_metadata(db, resume_id)
    except (ResumeNotFoundError, SQLAlchemyError) as exc:
        raise _resume_http_error(exc) from None
    ensure_resume_access(db, current_recruiter, document_meta)

    try:
        document, stored_file = service.open_download(db, resume_id)
    except (
        ResumeNotFoundError,
        ResumeStorageUnavailableError,
        InvalidConsentError,
        SQLAlchemyError,
    ) as exc:
        raise _resume_http_error(exc) from None

    log_security_event(
        "resume.downloaded",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="resume_document",
        resource_id=document.id,
        db=db,
    )

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
)
def delete_resume(
    resume_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> Response:
    service = _resume_service()
    try:
        document_meta = ResumeDocumentService.get_metadata(db, resume_id)
    except (ResumeNotFoundError, SQLAlchemyError) as exc:
        raise _resume_http_error(exc) from None
    ensure_resume_access(db, current_recruiter, document_meta)

    try:
        service.delete(db, resume_id)
    except (
        ResumeNotFoundError,
        ResumeDatabaseError,
        ResumeStorageUnavailableError,
        SQLAlchemyError,
    ) as exc:
        raise _resume_http_error(exc) from None

    log_security_event(
        "resume.deleted",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="resume_document",
        resource_id=resume_id,
        db=db,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/api/skills/",
    response_model=SkillVocabularyResponse,
    tags=["jobs"],
    summary="List the skill names that job requirements can use",
    dependencies=[Depends(get_current_recruiter)],
)
def list_supported_skills() -> SkillVocabularyResponse:
    return SkillVocabularyResponse(skills=sorted(set(SKILL_VOCABULARY), key=str.casefold))


@app.post(
    "/api/admin/recruiters",
    response_model=RecruiterRead,
    status_code=status.HTTP_201_CREATED,
    tags=["admin"],
    summary="Create a recruiter account (admin only)",
)
def create_recruiter_account(
    account: RecruiterAccountCreate,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> RecruiterRead:
    """Accounts are created by an admin; there is no open self-registration."""
    if current_recruiter.role != RecruiterRole.ADMIN:
        log_security_event(
            "rbac.recruiter_create_denied",
            outcome="denied",
            recruiter_id=current_recruiter.id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges are required to create recruiter accounts.",
        )

    already_exists = db.scalar(
        select(Recruiter.id).where(Recruiter.email_normalized == account.email)
    )
    if already_exists is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A recruiter with this email already exists.",
        )

    recruiter = Recruiter(
        email_normalized=account.email,
        display_name=account.display_name,
        role=account.role,
        status=RecruiterStatus.ACTIVE,
        password_hash=hash_password(account.password),
    )
    db.add(recruiter)
    try:
        db.commit()
        db.refresh(recruiter)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A recruiter with this email already exists.",
        ) from None

    log_security_event(
        "recruiter.created",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="recruiter",
        resource_id=recruiter.id,
        db=db,
    )
    return RecruiterRead.model_validate(recruiter)


@app.post(
    "/api/candidates/{candidate_id}/consents/",
    response_model=ConsentRead,
    status_code=status.HTTP_201_CREATED,
    tags=["consent"],
    summary="Record a candidate's consent",
)
def grant_candidate_consent(
    candidate_id: UUID,
    consent_data: ConsentGrantRequest,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> ConsentRead:
    """Record consent that the candidate gave (the recruiter attests to it).

    The capture time is set by the server. This is a record of an attestation, not
    proof of consent; keep the evidence referenced by ``proof_reference`` elsewhere.
    """
    candidate = _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)
    if candidate.anonymized_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Consent cannot be recorded for an anonymized candidate.",
        )

    if consent_data.application_id is not None:
        application = db.get(Application, consent_data.application_id)
        if application is None or application.candidate_id != candidate_id:
            raise HTTPException(
                status_code=422,
                detail="The application does not belong to this candidate.",
            )

    now = datetime.now(timezone.utc)
    expires_at = (
        _aware_utc(consent_data.expires_at)
        if consent_data.expires_at is not None
        else None
    )
    if expires_at is not None and expires_at <= now:
        raise HTTPException(status_code=422, detail="expires_at must be in the future.")

    record = ConsentRecord(
        candidate_id=candidate_id,
        application_id=consent_data.application_id,
        purpose=consent_data.purpose,
        status=ConsentStatus.GRANTED,
        captured_at=now,
        expires_at=expires_at,
        notice_version=consent_data.notice_version.strip(),
        capture_method=consent_data.capture_method.strip(),
        proof_reference=consent_data.proof_reference,
    )
    db.add(record)
    try:
        db.commit()
        db.refresh(record)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Unable to record the consent."
        ) from None

    log_security_event(
        "consent.granted",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="consent_record",
        resource_id=record.id,
        db=db,
    )
    return ConsentRead.model_validate(record)


@app.get(
    "/api/candidates/{candidate_id}/consents/",
    response_model=list[ConsentRead],
    tags=["consent"],
    summary="List a candidate's consent records",
)
def list_candidate_consents(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> list[ConsentRead]:
    _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)
    records = db.scalars(
        select(ConsentRecord)
        .where(ConsentRecord.candidate_id == candidate_id)
        .order_by(ConsentRecord.captured_at.desc(), ConsentRecord.id)
    ).all()
    return [ConsentRead.model_validate(record) for record in records]


@app.post(
    "/api/consents/{consent_id}/withdraw",
    response_model=ConsentRead,
    tags=["consent"],
    summary="Withdraw a consent record",
)
def withdraw_consent(
    consent_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> ConsentRead:
    """Withdraw consent. Screening and resume access that require it stop immediately."""
    record = db.get(ConsentRecord, consent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Consent record not found.")
    ensure_candidate_access(db, current_recruiter, record.candidate_id)

    if record.status != ConsentStatus.WITHDRAWN:
        record.status = ConsentStatus.WITHDRAWN
        record.withdrawn_at = datetime.now(timezone.utc)
        try:
            db.commit()
            db.refresh(record)
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(
                status_code=500, detail="Unable to withdraw the consent."
            ) from None
        log_security_event(
            "consent.withdrawn",
            outcome="success",
            recruiter_id=current_recruiter.id,
            resource_type="consent_record",
            resource_id=record.id,
            db=db,
        )
    return ConsentRead.model_validate(record)


@app.post(
    "/api/candidates/{candidate_id}/anonymize",
    response_model=CandidateAnonymizeResponse,
    tags=["candidates"],
    summary="Erase a candidate's personal data and resume files",
)
def anonymize_candidate(
    candidate_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    current_recruiter: Annotated[Recruiter, Depends(get_current_recruiter)],
) -> CandidateAnonymizeResponse:
    """Delete resume files, withdraw consents and blank the candidate's contact details.

    Application and screening rows are kept (without any identity) so job statistics and
    the audit trail stay intact. If any resume file cannot be deleted, nothing about the
    candidate is changed and the request can be retried safely.
    """
    candidate = _get_candidate_or_404(db, candidate_id)
    ensure_candidate_access(db, current_recruiter, candidate_id)

    if candidate.anonymized_at is not None:
        return CandidateAnonymizeResponse(
            candidate_id=candidate.id,
            anonymized_at=_aware_utc(candidate.anonymized_at),
            resumes_deleted=0,
            consents_withdrawn=0,
        )

    service = _resume_service()
    document_ids = db.scalars(
        select(ResumeDocument.id).where(
            ResumeDocument.candidate_id == candidate_id,
            ResumeDocument.status != DocumentStatus.DELETED,
        )
    ).all()
    resumes_deleted = 0
    for document_id in document_ids:
        try:
            service.delete(db, document_id)
        except (
            ResumeNotFoundError,
            ResumeDatabaseError,
            ResumeStorageUnavailableError,
            SQLAlchemyError,
        ):
            log_security_event(
                "candidate.anonymize_failed",
                outcome="error",
                recruiter_id=current_recruiter.id,
                resource_type="candidate",
                resource_id=candidate_id,
                reason="resume_deletion_failed",
                db=db,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "A resume file could not be deleted, so the candidate was not "
                    "anonymized. Please retry."
                ),
            ) from None
        resumes_deleted += 1

    now = datetime.now(timezone.utc)
    consents = db.scalars(
        select(ConsentRecord).where(
            ConsentRecord.candidate_id == candidate_id,
            ConsentRecord.status != ConsentStatus.WITHDRAWN,
        )
    ).all()
    for consent in consents:
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = now

    candidate.display_name = "Anonymized candidate"
    candidate.email = None
    candidate.email_normalized = None
    candidate.phone = None
    candidate.anonymized_at = now
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Unable to anonymize the candidate."
        ) from None

    log_security_event(
        "candidate.anonymized",
        outcome="success",
        recruiter_id=current_recruiter.id,
        resource_type="candidate",
        resource_id=candidate_id,
        db=db,
    )
    return CandidateAnonymizeResponse(
        candidate_id=candidate_id,
        anonymized_at=now,
        resumes_deleted=resumes_deleted,
        consents_withdrawn=len(consents),
    )


@app.post(
    "/api/analyze/",
    response_model=AnalysisResponse,
    dependencies=[Depends(get_current_recruiter)],
)
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
            total_bytes = 0
            header = b""
            with pdf_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_RESUME_BYTES:
                        analysis_error = "too_large"
                        break
                    if not header:
                        header = chunk[:1024]
                    destination.write(chunk)

            if analysis_error is None and b"%PDF-" not in header:
                analysis_error = "pdf"

            if analysis_error is None:
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

    if analysis_error == "too_large":
        raise HTTPException(
            status_code=413,
            detail=(
                "The uploaded PDF exceeds the "
                f"{MAX_RESUME_BYTES // (1024 * 1024)} MiB size limit."
            ),
        )
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