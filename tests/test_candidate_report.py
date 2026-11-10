"""
tests/test_candidate_report.py
-------------------------------
Unit tests for src/candidate_report.py

Run from the project root with:
    python -m unittest tests/test_candidate_report.py -v

All tests use only Python's built-in unittest and the existing SmartHire
modules.  No ML models are loaded – the tests exercise the report-building
logic and the integration between CandidateReport and its collaborators.

Test classes:
    TC-01  TestReportConstruction       – valid inputs, invalid inputs
    TC-02  TestMLSectionWithoutModel    – graceful 'not available' notice
    TC-03  TestSkillSectionContent      – skill keywords appear correctly
    TC-04  TestVerificationSectionContent – claims flow into the report
    TC-05  TestNoVerifierSupplied       – missing verifier handled gracefully
    TC-06  TestReportStructure          – section headers and footer present
    TC-07  TestProgrammaticAccessors    – prediction_result / skill_result /
                                         verification_claims return correct types
    TC-08  TestSeparationInvariant      – ML label never alters verification;
                                         verification never alters ML result
"""

import unittest

from src.candidate_report import CandidateReport
from src.background_verification import BackgroundVerifier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal but non-trivial texts (no ML model needed for these tests)
_RESUME = (
    "Python developer with 3 years of experience. "
    "Proficient in Docker, AWS, FastAPI, PostgreSQL, and scikit-learn. "
    "BSc Computer Science, State University (2021). "
    "AWS Certified Solutions Architect (2023)."
)

_JD = (
    "Looking for a Python backend engineer. "
    "Must know Docker, FastAPI, and PostgreSQL. "
    "AWS or GCP experience is required. "
    "Bonus: scikit-learn and Kubernetes."
)


def _make_verifier() -> BackgroundVerifier:
    """Return a BackgroundVerifier with two sample claims (one Verified, one Pending)."""
    v = BackgroundVerifier(candidate_name="Test Candidate")
    v.add_claim("BSc Computer Science", category="Degree / Education")  # stays Pending
    cert_id = v.add_claim("AWS Certified Solutions Architect", category="Certification")
    v.update_claim(cert_id, status="Verified",
                   evidence="Certificate PDF reviewed by HR.",
                   reviewer_notes="Valid until 2026.")
    return v


def _make_report(**kwargs) -> CandidateReport:
    """
    Factory that creates a CandidateReport with sensible defaults and lets
    individual tests override specific arguments.
    """
    defaults = dict(
        candidate_name="Test Candidate",
        resume_text=_RESUME,
        jd_text=_JD,
        verifier=_make_verifier(),
    )
    defaults.update(kwargs)
    return CandidateReport(**defaults)


# ---------------------------------------------------------------------------
# TC-01  Construction: valid and invalid inputs
# ---------------------------------------------------------------------------

class TestReportConstruction(unittest.TestCase):
    """TC-01  CandidateReport raises ValueError for empty resume/JD."""

    def test_valid_inputs_construct_without_error(self):
        report = _make_report()
        self.assertIsInstance(report, CandidateReport)

    def test_empty_resume_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            CandidateReport(resume_text="", jd_text=_JD)
        self.assertIn("resume_text", str(ctx.exception).lower())

    def test_whitespace_resume_raises_value_error(self):
        with self.assertRaises(ValueError):
            CandidateReport(resume_text="   ", jd_text=_JD)

    def test_empty_jd_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            CandidateReport(resume_text=_RESUME, jd_text="")
        self.assertIn("jd_text", str(ctx.exception).lower())

    def test_whitespace_jd_raises_value_error(self):
        with self.assertRaises(ValueError):
            CandidateReport(resume_text=_RESUME, jd_text="\t\n")

    def test_candidate_name_is_optional(self):
        # No name supplied – should not raise
        report = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        built  = report.build()
        self.assertIsInstance(built, str)

    def test_verifier_is_optional(self):
        # No verifier – should not raise; section shows 'not available'
        report = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        built  = report.build()
        self.assertIn("not available", built.lower())


# ---------------------------------------------------------------------------
# TC-02  ML section without a model
# ---------------------------------------------------------------------------

class TestMLSectionWithoutModel(unittest.TestCase):
    """TC-02  When no model/vectorizer is supplied the ML section says 'not available'."""

    def test_no_model_shows_not_available(self):
        report = _make_report(vectorizer=None, model=None)
        built  = report.build()
        self.assertIn("not available", built.lower())

    def test_only_vectorizer_no_model_shows_not_available(self):
        # Supplying only one of the two should still skip ML
        import joblib
        try:
            vec = joblib.load("models/tfidf_vectorizer.joblib")
        except Exception:
            self.skipTest("Model artifacts not available in this environment.")
        report = _make_report(vectorizer=vec, model=None)
        self.assertIn("not available", report.build().lower())

    def test_prediction_result_returns_none_without_model(self):
        report = _make_report(vectorizer=None, model=None)
        self.assertIsNone(report.prediction_result())


# ---------------------------------------------------------------------------
# TC-03  Skill section content
# ---------------------------------------------------------------------------

class TestSkillSectionContent(unittest.TestCase):
    """TC-03  Skills that appear in the resume/JD should show up in the report."""

    def setUp(self):
        self.report = _make_report()
        self.built  = self.report.build()

    def test_report_contains_skill_section_header(self):
        self.assertIn("SKILL ANALYSIS", self.built)

    def test_matched_skill_python_appears(self):
        # "Python" is in both _RESUME and _JD so it should show as matched
        self.assertIn("Python", self.built)

    def test_matched_skill_docker_appears(self):
        self.assertIn("Docker", self.built)

    def test_matched_skill_fastapi_appears(self):
        self.assertIn("FastAPI", self.built)

    def test_missing_skill_kubernetes_is_noted(self):
        # Kubernetes is in the JD but not in the resume text
        self.assertIn("Kubernetes", self.built)

    def test_skill_section_contains_match_markers(self):
        # [+] and [-] markers should both appear
        self.assertIn("[+]", self.built)
        self.assertIn("[-]", self.built)

    def test_skill_note_about_keyword_detection(self):
        # Disclaimer about keyword-only detection must be present
        self.assertIn("keyword", self.built.lower())

    # Programmatic accessor tests
    def test_skill_result_returns_dict(self):
        result = self.report.skill_result()
        self.assertIsInstance(result, dict)
        self.assertIn("resume_skills",  result)
        self.assertIn("matched_skills", result)
        self.assertIn("missing_skills", result)

    def test_skill_result_matched_contains_python(self):
        result = self.report.skill_result()
        self.assertIn("Python", result["matched_skills"])

    def test_skill_result_missing_contains_kubernetes(self):
        result = self.report.skill_result()
        self.assertIn("Kubernetes", result["missing_skills"])


# ---------------------------------------------------------------------------
# TC-04  Verification section content
# ---------------------------------------------------------------------------

class TestVerificationSectionContent(unittest.TestCase):
    """TC-04  Claims from a BackgroundVerifier appear in the report correctly."""

    def setUp(self):
        self.verifier = _make_verifier()
        self.report   = _make_report(verifier=self.verifier)
        self.built    = self.report.build()

    def test_report_contains_verification_section_header(self):
        self.assertIn("BACKGROUND VERIFICATION", self.built)

    def test_pending_claim_appears_in_report(self):
        self.assertIn("BSc Computer Science", self.built)
        self.assertIn("Pending", self.built)

    def test_verified_claim_appears_in_report(self):
        self.assertIn("AWS Certified Solutions Architect", self.built)
        self.assertIn("Verified", self.built)

    def test_evidence_text_appears_in_report(self):
        self.assertIn("Certificate PDF reviewed by HR", self.built)

    def test_total_claims_count_in_report(self):
        # _make_verifier adds exactly 2 claims
        self.assertIn("2", self.built)

    def test_status_summary_line_present(self):
        # Summary like "Pending: 1  |  Verified: 1" should appear
        self.assertIn("Verified: 1", self.built)
        self.assertIn("Pending: 1",  self.built)

    def test_verification_disclaimer_present(self):
        self.assertIn("human reviewer", self.built.lower())

    def test_unable_to_verify_appears_when_present(self):
        v = BackgroundVerifier()
        cid = v.add_claim("Google PM Certificate")
        v.update_claim(cid, status="Unable to Verify",
                       evidence="No response from issuer.")
        r = _make_report(verifier=v)
        self.assertIn("Unable to Verify", r.build())

    def test_discrepancy_found_appears_when_present(self):
        v = BackgroundVerifier()
        cid = v.add_claim("5 years Kubernetes")
        v.update_claim(cid, status="Discrepancy Found",
                       evidence="Reference says 1 year.")
        r = _make_report(verifier=v)
        self.assertIn("Discrepancy Found", r.build())

    def test_claim_status_is_never_auto_verified(self):
        """
        Adding a claim to the verifier must not automatically mark it Verified.
        Only an explicit reviewer update_claim() call can do that.
        """
        v = BackgroundVerifier()
        v.add_claim("Self-taught Machine Learning")  # no update_claim call
        r = _make_report(verifier=v)
        built = r.build()
        # 'Pending' must appear; 'Verified' must NOT appear for this claim
        self.assertIn("Pending", built)
        # Count occurrences – 'Verified' only in the disclaimer text, not as a status
        verified_as_status = built.count("[Verified]")
        self.assertEqual(verified_as_status, 0,
            "No claim should be auto-verified in the report.")

    # Programmatic accessor tests
    def test_verification_claims_returns_list(self):
        claims = self.report.verification_claims()
        self.assertIsInstance(claims, list)
        self.assertEqual(len(claims), 2)

    def test_verification_claims_preserves_status(self):
        claims = self.report.verification_claims()
        statuses = {c["claim_text"]: c["status"] for c in claims}
        self.assertEqual(statuses["BSc Computer Science"],         "Pending")
        self.assertEqual(statuses["AWS Certified Solutions Architect"], "Verified")


# ---------------------------------------------------------------------------
# TC-05  No verifier supplied
# ---------------------------------------------------------------------------

class TestNoVerifierSupplied(unittest.TestCase):
    """TC-05  CandidateReport gracefully handles a missing BackgroundVerifier."""

    def test_report_builds_without_verifier(self):
        r = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        built = r.build()
        self.assertIsInstance(built, str)
        self.assertGreater(len(built), 0)

    def test_report_shows_not_available_notice_for_verification(self):
        r = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        self.assertIn("not available", r.build().lower())

    def test_verification_claims_returns_empty_list_without_verifier(self):
        r = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        self.assertEqual(r.verification_claims(), [])

    def test_skill_section_still_works_without_verifier(self):
        # Skills should be extracted even if verification is absent
        r    = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        built = r.build()
        self.assertIn("Python",   built)
        self.assertIn("Docker",   built)
        self.assertIn("FastAPI",  built)


# ---------------------------------------------------------------------------
# TC-06  Report structure – section headers and footer
# ---------------------------------------------------------------------------

class TestReportStructure(unittest.TestCase):
    """TC-06  The report contains all required structural elements."""

    def setUp(self):
        self.built = _make_report().build()

    def test_report_is_a_string(self):
        self.assertIsInstance(self.built, str)

    def test_report_is_non_empty(self):
        self.assertGreater(len(self.built), 100)

    def test_section_1_header_present(self):
        self.assertIn("SECTION 1 OF 3", self.built)
        self.assertIn("ML FIT PREDICTION", self.built)

    def test_section_2_header_present(self):
        self.assertIn("SECTION 2 OF 3", self.built)
        self.assertIn("SKILL ANALYSIS", self.built)

    def test_section_3_header_present(self):
        self.assertIn("SECTION 3 OF 3", self.built)
        self.assertIn("BACKGROUND VERIFICATION", self.built)

    def test_footer_reminders_present(self):
        self.assertIn("IMPORTANT REMINDERS", self.built)

    def test_candidate_name_in_header(self):
        self.assertIn("Test Candidate", self.built)

    def test_report_without_candidate_name_still_builds(self):
        r = CandidateReport(resume_text=_RESUME, jd_text=_JD)
        self.assertIsInstance(r.build(), str)


# ---------------------------------------------------------------------------
# TC-07  Programmatic accessors
# ---------------------------------------------------------------------------

class TestProgrammaticAccessors(unittest.TestCase):
    """TC-07  prediction_result, skill_result, verification_claims return correct types."""

    def setUp(self):
        self.report = _make_report()

    def test_prediction_result_is_none_without_model(self):
        self.assertIsNone(self.report.prediction_result())

    def test_skill_result_keys_are_correct(self):
        r = self.report.skill_result()
        self.assertIn("resume_skills",  r)
        self.assertIn("matched_skills", r)
        self.assertIn("missing_skills", r)

    def test_skill_result_values_are_sets(self):
        r = self.report.skill_result()
        self.assertIsInstance(r["resume_skills"],  set)
        self.assertIsInstance(r["matched_skills"], set)
        self.assertIsInstance(r["missing_skills"], set)

    def test_verification_claims_returns_list_of_dicts(self):
        claims = self.report.verification_claims()
        self.assertIsInstance(claims, list)
        for c in claims:
            self.assertIsInstance(c, dict)
            self.assertIn("claim_id",   c)
            self.assertIn("status",     c)
            self.assertIn("claim_text", c)

    def test_verification_claims_are_copies(self):
        """Mutating a returned claim dict must not corrupt the verifier state."""
        claims = self.report.verification_claims()
        original_status = claims[0]["status"]
        claims[0]["status"] = "Hacked"   # mutate the returned copy
        # Retrieve again – original state must be intact
        fresh = self.report.verification_claims()
        self.assertEqual(fresh[0]["status"], original_status)


# ---------------------------------------------------------------------------
# TC-08  Separation invariant
# ---------------------------------------------------------------------------

class TestSeparationInvariant(unittest.TestCase):
    """
    TC-08  The ML prediction and verification statuses must be completely
    independent.  Changing one must never affect the other.
    """

    def test_verification_status_does_not_change_skill_result(self):
        v = _make_verifier()
        r = _make_report(verifier=v)

        skills_before = r.skill_result()

        # Update a verification claim
        claims = v.all_claims()
        v.update_claim(claims[0]["claim_id"], status="Discrepancy Found",
                       evidence="Some discrepancy noted.")

        skills_after = r.skill_result()
        # Skill results are based on text only – must be unchanged
        self.assertEqual(skills_before["matched_skills"],
                         skills_after["matched_skills"])
        self.assertEqual(skills_before["missing_skills"],
                         skills_after["missing_skills"])

    def test_report_does_not_auto_promote_claim_to_verified(self):
        """
        Building the report must not silently change any claim status.
        A claim added as Pending must remain Pending after build().
        """
        v = BackgroundVerifier()
        cid = v.add_claim("Python expert", category="Skill")

        r = _make_report(verifier=v)
        r.build()   # trigger full report assembly

        # Status must still be Pending – report build must be side-effect free
        self.assertEqual(v.get_claim(cid)["status"], "Pending")

    def test_ml_label_absent_from_verification_status(self):
        """
        Whatever the ML label is, the verification section must not contain
        that label as a verification status.  Statuses are strictly one of
        the four VALID_STATUSES values.
        """
        from src.background_verification import VALID_STATUSES
        v = BackgroundVerifier()
        v.add_claim("Some claim")
        r = _make_report(verifier=v)
        claims = r.verification_claims()
        for c in claims:
            self.assertIn(c["status"], VALID_STATUSES)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
