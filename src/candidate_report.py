"""
src/candidate_report.py  –  SmartHire Unified Candidate Report
---------------------------------------------------------------
Combines three independent information sources into a single, human-readable
report for a reviewer:

  1. ML Fit Prediction   – predicted label and class probabilities from the
                           saved Logistic Regression model via predict.py
  2. Skill Analysis      – matched and missing skills via skill_extractor.py
  3. Background Checks   – claim verification records via background_verification.py

DESIGN PRINCIPLES
-----------------
- Each section stays strictly separate.  The ML fit category NEVER influences
  verification status, and verification status NEVER changes the predicted label.
- No claim is automatically marked Verified.  Only a human reviewer can do that.
- This module reuses existing functions and classes; it does NOT duplicate logic.
- No new dependencies are required (only Python stdlib + whatever predict.py
  already uses: joblib, scikit-learn).

USAGE
-----
  # With a loaded model (for full reports including ML prediction):
  from src.candidate_report import CandidateReport
  import joblib

  vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
  model      = joblib.load("models/logistic_regression.joblib")

  report = CandidateReport(
      candidate_name="Jane Doe",
      resume_text=<resume string>,
      jd_text=<job description string>,
      verifier=<BackgroundVerifier instance>,
      vectorizer=vectorizer,
      model=model,
  )
  print(report.build())

  # Without a model (skill analysis + verification only):
  report = CandidateReport(
      candidate_name="Jane Doe",
      resume_text=<resume string>,
      jd_text=<job description string>,
      verifier=<BackgroundVerifier instance>,
  )
  print(report.build())
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from typing import Optional

# ---------------------------------------------------------------------------
# SmartHire modules (reused, not duplicated)
# ---------------------------------------------------------------------------
# predict() and load_artifacts() from predict.py handle ML inference.
# We import predict() directly so this module never calls model.fit() and
# never touches the saved .joblib artifacts itself.
from predict import predict as _ml_predict

# Rule-based skill analysis (separate from ML model).
from src.skill_extractor import analyse_skills

# Verification records — human-controlled only.
from src.background_verification import BackgroundVerifier


# ---------------------------------------------------------------------------
# CandidateReport
# ---------------------------------------------------------------------------

class CandidateReport:
    """
    Aggregates ML prediction, skill analysis, and background verification
    data for a single candidate into one readable report.

    Parameters
    ----------
    candidate_name : str
        A display name for the report header.
    resume_text : str
        Raw resume text.  Must be a non-empty string.
    jd_text : str
        Raw job description text.  Must be a non-empty string.
    verifier : BackgroundVerifier, optional
        A populated BackgroundVerifier instance.  If omitted, the
        Background Verification section will note that no claims were recorded.
    vectorizer : optional
        A fitted TF-IDF vectorizer (loaded via joblib).  If omitted, the ML
        prediction section is skipped and a notice is shown instead.
    model : optional
        A trained classifier with predict() and predict_proba() (loaded via
        joblib).  If omitted, the ML prediction section is skipped.

    Notes
    -----
    - Passing vectorizer or model alone (without the other) will skip the
      ML section and show a configuration notice.
    - resume_text and jd_text are validated on construction; ValueError is
      raised for empty / whitespace-only inputs.
    """

    def __init__(
        self,
        candidate_name: str = "",
        resume_text: str = "",
        jd_text: str = "",
        verifier: Optional[BackgroundVerifier] = None,
        vectorizer=None,
        model=None,
    ) -> None:
        # Input validation — we want to surface bad inputs early.
        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text must not be empty.")
        if not jd_text or not jd_text.strip():
            raise ValueError("jd_text must not be empty.")

        self.candidate_name: str = candidate_name.strip()
        self._resume_text: str   = resume_text
        self._jd_text: str       = jd_text
        self._verifier: Optional[BackgroundVerifier] = verifier
        self._vectorizer         = vectorizer
        self._model              = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> str:
        """
        Build and return the full unified candidate report as a plain string.

        The report has three numbered sections:
          Section 1 – ML Fit Prediction
          Section 2 – Skill Analysis
          Section 3 – Background Verification

        Each section is self-contained; a reviewer can read them independently.
        """
        outer_sep = "=" * 65
        inner_sep = "-" * 65

        lines: list[str] = [
            outer_sep,
            "  SMARTHIRE  –  UNIFIED CANDIDATE REPORT",
        ]
        if self.candidate_name:
            lines.append(f"  Candidate : {self.candidate_name}")
        lines.append(outer_sep)

        # ── Section 1: ML Fit Prediction ───────────────────────────────
        lines.append("")
        lines.append("  SECTION 1 OF 3  –  ML FIT PREDICTION")
        lines.append(inner_sep)
        lines.extend(self._build_prediction_section())

        # ── Section 2: Skill Analysis ──────────────────────────────────
        lines.append("")
        lines.append("  SECTION 2 OF 3  –  SKILL ANALYSIS")
        lines.append(inner_sep)
        lines.extend(self._build_skill_section())

        # ── Section 3: Background Verification ────────────────────────
        lines.append("")
        lines.append("  SECTION 3 OF 3  –  BACKGROUND VERIFICATION")
        lines.append(inner_sep)
        lines.extend(self._build_verification_section())

        # ── Footer ──────────────────────────────────────────────────────
        lines.append("")
        lines.append(outer_sep)
        lines.append("  IMPORTANT REMINDERS")
        lines.append(inner_sep)
        lines.append("  * The ML fit category is based on statistical patterns in")
        lines.append("    training data.  It is one signal, not a final decision.")
        lines.append("  * Skill matches are keyword detections, not validated facts.")
        lines.append("  * Verification statuses are set only by human reviewers after")
        lines.append("    consulting supporting evidence.")
        lines.append("  * Never automatically reject a candidate on the basis of any")
        lines.append("    single section of this report.")
        lines.append(outer_sep)

        return "\n".join(lines)

    def prediction_result(self) -> Optional[dict]:
        """
        Return the raw ML prediction dict (label + probs), or None if
        the vectorizer / model was not supplied.

        Useful for programmatic access without parsing the text report.
        """
        if self._vectorizer is None or self._model is None:
            return None
        return _ml_predict(self._resume_text, self._jd_text,
                           self._vectorizer, self._model)

    def skill_result(self) -> dict:
        """
        Return the raw skill analysis dict (resume_skills, matched_skills,
        missing_skills).

        Useful for programmatic access without parsing the text report.
        """
        return analyse_skills(self._resume_text, self._jd_text)

    def verification_claims(self) -> list[dict]:
        """
        Return all verification claims as a list of dicts, or an empty list
        if no verifier was supplied.

        Useful for programmatic access without parsing the text report.
        """
        if self._verifier is None:
            return []
        return self._verifier.all_claims()

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _build_prediction_section(self) -> list[str]:
        """Return lines for the ML prediction section."""
        lines: list[str] = []

        if self._vectorizer is None or self._model is None:
            lines.append("  [Not available]  No vectorizer/model was supplied.")
            lines.append("  Run with vectorizer and model arguments to include")
            lines.append("  ML predictions in this report.")
            return lines

        result = _ml_predict(
            self._resume_text, self._jd_text, self._vectorizer, self._model
        )
        label  = result["label"]
        probs  = result["probs"]

        icons = {
            "Good Fit":      "[GOOD FIT]",
            "Potential Fit": "[POTENTIAL FIT]",
            "No Fit":        "[NO FIT]",
        }
        icon = icons.get(label, "[?]")

        lines.append(f"  Predicted category : {icon}")
        lines.append("")
        lines.append("  Confidence scores (sorted highest first):")
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar    = "#" * int(prob * 25)
            marker = "  << predicted" if cls == label else ""
            lines.append(f"    {cls:<16}  {prob:.1%}  {bar}{marker}")

        lines.append("")
        lines.append("  NOTE: This score reflects statistical similarity to training")
        lines.append("  data.  It is not a measure of candidate quality or character.")
        return lines

    def _build_skill_section(self) -> list[str]:
        """Return lines for the skill analysis section."""
        lines: list[str] = []

        result         = analyse_skills(self._resume_text, self._jd_text)
        resume_skills  = sorted(result["resume_skills"])
        matched_skills = sorted(result["matched_skills"])
        missing_skills = sorted(result["missing_skills"])

        lines.append(f"  Skills found in resume  ({len(resume_skills)}):")
        if resume_skills:
            for skill in resume_skills:
                marker = "[+]" if skill in result["matched_skills"] else "[ ]"
                lines.append(f"    {marker}  {skill}")
        else:
            lines.append("    (none detected)")

        lines.append("")
        lines.append(f"  Matched skills – in both resume and JD  ({len(matched_skills)}):")
        if matched_skills:
            for skill in matched_skills:
                lines.append(f"    [+]  {skill}")
        else:
            lines.append("    (none)")

        lines.append("")
        lines.append(f"  JD skills not detected in resume  ({len(missing_skills)}):")
        if missing_skills:
            for skill in missing_skills:
                lines.append(f"    [-]  {skill}")
        else:
            lines.append("    (none  –  great coverage!)")

        lines.append("")
        lines.append("  NOTE: [+] means the keyword was detected; it does not mean")
        lines.append("  the skill has been validated.  [-] means the keyword was")
        lines.append("  absent from the resume text, not that the candidate lacks it.")
        return lines

    def _build_verification_section(self) -> list[str]:
        """Return lines for the background verification section."""
        lines: list[str] = []

        if self._verifier is None:
            lines.append("  [Not available]  No BackgroundVerifier was supplied.")
            lines.append("  Create and populate a BackgroundVerifier, then pass it")
            lines.append("  to CandidateReport(..., verifier=<instance>).")
            return lines

        claims = self._verifier.all_claims()
        lines.append(f"  Total claims recorded : {len(claims)}")

        if not claims:
            lines.append("  (No claims have been recorded yet.)")
        else:
            # Status summary line
            from collections import Counter
            counts = Counter(c["status"] for c in claims)
            summary_parts = [f"{s}: {counts[s]}" for s in
                             ("Pending", "Verified", "Unable to Verify", "Discrepancy Found")
                             if s in counts]
            lines.append("  Status summary  :  " + "  |  ".join(summary_parts))
            lines.append("")

            for claim in claims:
                lines.append(f"  Claim #{claim['claim_id']}  [{claim['status']}]")
                lines.append(f"    Statement  : {claim['claim_text']}")
                lines.append(f"    Category   : {claim['category']}")
                if claim["evidence"]:
                    ev_lines = claim["evidence"].splitlines()
                    lines.append(f"    Evidence   : {ev_lines[0]}")
                    for ev in ev_lines[1:]:
                        lines.append(f"                 {ev}")
                else:
                    lines.append("    Evidence   : (none recorded)")
                if claim["reviewer_notes"]:
                    note_lines = claim["reviewer_notes"].splitlines()
                    lines.append(f"    Notes      : {note_lines[0]}")
                    for n in note_lines[1:]:
                        lines.append(f"                 {n}")
                date_str = (
                    str(claim["review_date"]) if claim["review_date"]
                    else "not yet reviewed"
                )
                lines.append(f"    Review date: {date_str}")
                lines.append("  " + "." * 61)

        lines.append("")
        lines.append("  NOTE: 'Verified' status is set only by a human reviewer.")
        lines.append("  'Unable to Verify' does not mean a claim is false.")
        lines.append("  'Discrepancy Found' requires further review; it is not a")
        lines.append("  finding of dishonesty.  Verification is independent of")
        lines.append("  the ML fit prediction.")
        return lines


# ---------------------------------------------------------------------------
# Demo  (python src/candidate_report.py)
# ---------------------------------------------------------------------------

def _build_sample_verifier() -> BackgroundVerifier:
    """Return a BackgroundVerifier populated with realistic sample claims."""
    v = BackgroundVerifier(candidate_name="Alex Rivera (sample)")

    # All claims start Pending – a reviewer must act to change them
    deg_id  = v.add_claim("BSc Computer Science, State University (2021)",
                           category="Degree / Education")
    cert_id = v.add_claim("AWS Certified Solutions Architect (2023)",
                           category="Certification")
    exp_id  = v.add_claim("Software Engineer Intern, TechCorp (2022)",
                           category="Internship")
    k8s_id  = v.add_claim("5 years of production Kubernetes experience",
                           category="Skill")

    # Reviewer actions (human-controlled only)
    v.update_claim(cert_id, status="Verified",
                   evidence="Official AWS certificate PDF reviewed; "
                            "ID confirmed on aws.training portal.",
                   reviewer_notes="Valid. Expires 2026-04-10.")

    v.update_claim(deg_id,  status="Unable to Verify",
                   evidence="Two emails to State University registrar; no reply in 10 days.",
                   reviewer_notes="Ask candidate for an official transcript.")

    v.update_claim(k8s_id, status="Discrepancy Found",
                   evidence="Reference (former manager) states ~1 year, not 5.",
                   reviewer_notes="Needs clarification; not an automatic finding of dishonesty.")

    # exp_id intentionally left as Pending
    return v


_SAMPLE_RESUME = """
Alex Rivera
Email: alex@example.com | GitHub: github.com/alexrivera

SUMMARY
Python developer with 3 years of experience building backend services
and data pipelines. Experienced with Docker, AWS, FastAPI, and PostgreSQL.
Familiar with Machine Learning workflows using scikit-learn and Pandas.

EDUCATION
BSc Computer Science, State University, 2021

CERTIFICATIONS
AWS Certified Solutions Architect (2023)

EXPERIENCE
Software Engineer Intern, TechCorp, Jun-Aug 2022
  - Built ETL pipelines in Python/Pandas
  - Deployed microservices on Docker and AWS ECS

SKILLS
Python, FastAPI, Docker, PostgreSQL, AWS, Git, scikit-learn, Pandas, NumPy
"""

_SAMPLE_JD = """
We are looking for a Backend Engineer with:
- Strong Python skills (3+ years)
- Experience with FastAPI or Django
- Familiarity with Docker and Kubernetes
- AWS or GCP cloud experience
- PostgreSQL or MySQL database skills
- Bonus: Machine Learning experience (scikit-learn, PyTorch)
- Must know Git and CI/CD pipelines
"""


def _demo() -> None:
    """
    Run a demonstration without loading the ML model.
    Shows skill analysis and background verification sections.
    The ML section will display a 'not available' notice because no
    model is loaded (avoids requiring joblib in a quick demo).
    """
    verifier = _build_sample_verifier()

    report = CandidateReport(
        candidate_name="Alex Rivera (sample – no ML model loaded)",
        resume_text=_SAMPLE_RESUME,
        jd_text=_SAMPLE_JD,
        verifier=verifier,
        # vectorizer and model intentionally omitted to keep demo self-contained
    )

    print(report.build())


if __name__ == "__main__":
    # When run directly as `python src/candidate_report.py` the project root
    # is not automatically on sys.path.  We add it here so that `predict`
    # (a root-level module) can be found.  This guard only fires for the demo;
    # importing the module normally from the project root is unaffected.
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    _demo()
