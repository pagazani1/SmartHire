"""FastAPI endpoints for SmartHire resume analysis."""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from backend.db.models import (
    Application,
    Candidate,
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    Recruiter,
    RequirementKind,
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
    ScreeningHistoryRead,
)
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

    history: list[ScreeningHistoryRead] = []
    for result in screening_results:
        matched_required_skills: set[str] = set()
        missing_required_skills: set[str] = set()
        for requirement_result in result.requirement_results:
            group = requirement_result.requirement_group
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

        history.append(
            ScreeningHistoryRead(
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
        )
    return history


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
