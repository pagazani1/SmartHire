"""Reusable PDF resume analysis using SmartHire's existing ML and skill logic."""

from pathlib import Path

from predict import load_artifacts, predict, validate_text_input
from src.decision_policy import apply_decision_policy
from src.pdf_extractor import PDFExtractionError, extract_text_from_pdf
from src.skill_extractor import analyse_skills


def analyze_resume(pdf_path: str | Path, jd_text: object) -> dict:
    """Extract a PDF resume and analyze it against a job description.

    The returned ``label`` is the recruiter-facing policy decision.
    ``model_label`` and ``model_probs`` preserve the unchanged text-model
    result; ``probs`` remains an alias for those original probabilities.
    Raw resume text is not returned or printed.

    Raises:
        ValueError: The job description is not a non-empty string.
        FileNotFoundError: The PDF path does not exist.
        PDFExtractionError: The PDF cannot be read or contains no extractable
            text. Scanned PDFs require OCR, which is not implemented.

    Artifact-loading and model-inference exceptions are propagated from the
    existing inference functions.
    """
    validated_jd = validate_text_input(jd_text, "Job description")
    resume_text = extract_text_from_pdf(pdf_path)
    vectorizer, model = load_artifacts()
    model_prediction = predict(resume_text, validated_jd, vectorizer, model)
    skills = analyse_skills(resume_text, validated_jd)
    decision = apply_decision_policy(model_prediction, skills)

    return {
        **decision,
        "resume_skills": sorted(skills["resume_skills"]),
        "matched_skills": sorted(skills["matched_skills"]),
        "missing_skills": sorted(skills["missing_skills"]),
        "detected_resume_skills": sorted(skills["resume_skills"]),
        "detected_jd_skills": sorted(skills["jd_skills"]),
        "required_skills": sorted(skills["required_skills"]),
        "preferred_skills": sorted(skills["preferred_skills"]),
        "required_skill_groups": [
            sorted(group) for group in skills["required_skill_groups"]
        ],
        "preferred_skill_groups": [
            sorted(group) for group in skills["preferred_skill_groups"]
        ],
        "matched_required_skills": sorted(skills["matched_required_skills"]),
        "missing_required_skills": sorted(skills["missing_required_skills"]),
        "matched_preferred_skills": sorted(skills["matched_preferred_skills"]),
        "missing_preferred_skills": sorted(skills["missing_preferred_skills"]),
        "unclassified_jd_skills": sorted(skills["unclassified_jd_skills"]),
        "total_jd_skills": skills["total_jd_skills"],
        "coverage_metrics": skills["coverage_metrics"],
        "extraction_limitations": skills["extraction_limitations"],
    }
