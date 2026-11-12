"""FastAPI endpoints for SmartHire resume analysis."""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.db.session import dispose_engine
from predict import validate_text_input
from src.pdf_extractor import PDFExtractionError
from src.resume_analyzer import analyze_resume


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
