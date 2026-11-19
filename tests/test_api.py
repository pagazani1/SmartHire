import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from backend.main import app
from backend.security import get_current_recruiter
from src.pdf_extractor import NoExtractableTextError
from src.resume_analyzer import analyze_resume as run_resume_analyzer


class TestSmartHireAPI(unittest.TestCase):
    def setUp(self):
        # These tests cover analysis behaviour only; authentication is tested separately.
        app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(id=None)
        self.addCleanup(app.dependency_overrides.pop, get_current_recruiter, None)
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_pdf(self, filename: str = "synthetic.pdf") -> Path:
        pdf_path = self.temp_path / filename
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic Python developer")
        document.save(pdf_path)
        document.close()
        return pdf_path

    def _post_pdf(self, pdf_path: Path, jd_text: str = "Python engineering role"):
        with pdf_path.open("rb") as pdf_file:
            return self.client.post(
                "/api/analyze/",
                files={"file": (pdf_path.name, pdf_file, "application/pdf")},
                data={"jd_text": jd_text},
            )

    def test_health_check(self):
        response = self.client.get("/api/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_successful_analysis_returns_structured_result_and_cleans_temp_file(self):
        pdf_path = self._create_pdf()
        analyzer_result = {
            "label": "Potential Fit",
            "probs": {
                "Good Fit": 0.1,
                "No Fit": 0.2,
                "Potential Fit": 0.7,
            },
            "resume_skills": ["Python"],
            "matched_skills": ["Python"],
            "missing_skills": ["AWS"],
        }
        observed_temp_path = None

        def fake_analyze(path, jd_text):
            nonlocal observed_temp_path
            observed_temp_path = Path(path)
            self.assertTrue(observed_temp_path.is_file())
            self.assertEqual(observed_temp_path.read_bytes(), pdf_path.read_bytes())
            self.assertEqual(jd_text, "Python engineering role")
            return analyzer_result

        with patch("backend.main.analyze_resume", side_effect=fake_analyze):
            response = self._post_pdf(pdf_path)

        self.assertEqual(response.status_code, 200)
        response_body = response.json()
        self.assertEqual(
            {key: response_body[key] for key in analyzer_result},
            analyzer_result,
        )
        self.assertEqual(response_body["detected_jd_skills"], [])
        self.assertEqual(response_body["coverage_metrics"], {})
        self.assertIsNotNone(observed_temp_path)
        self.assertFalse(observed_temp_path.exists())
        self.assertFalse(observed_temp_path.parent.exists())

    def test_analysis_response_includes_additive_skill_evidence(self):
        pdf_path = self._create_pdf()
        analyzer_result = {
            "label": "Potential Fit",
            "probs": {"Good Fit": 0.1, "No Fit": 0.2, "Potential Fit": 0.7},
            "model_label": "Potential Fit",
            "model_probs": {
                "Good Fit": 0.1,
                "No Fit": 0.2,
                "Potential Fit": 0.7,
            },
            "decision_policy": "required_skill_group_coverage_v2",
            "decision_explanation": "Synthetic policy explanation.",
            "decision_evidence": {
                "structured_evidence_available": True,
                "matched_required_group_count": 1,
                "required_skill_group_count": 1,
                "missing_required_group_count": 0,
                "required_skill_coverage": 1.0,
                "required_skill_coverage_percentage": 100.0,
                "missing_required_skill_groups": [],
            },
            "resume_skills": ["Python"],
            "matched_skills": ["Python"],
            "missing_skills": ["AWS"],
            "detected_resume_skills": ["Python"],
            "detected_jd_skills": ["AWS", "Python"],
            "required_skills": ["Python"],
            "preferred_skills": [],
            "required_skill_groups": [["Python"]],
            "preferred_skill_groups": [],
            "matched_required_skills": ["Python"],
            "missing_required_skills": [],
            "matched_preferred_skills": [],
            "missing_preferred_skills": [],
            "unclassified_jd_skills": ["AWS"],
            "total_jd_skills": 2,
            "coverage_metrics": {
                "required_skill_coverage": 1.0,
                "required_skill_coverage_percentage": 100.0,
                "preferred_skill_coverage": None,
                "overall_skill_coverage": 0.5,
            },
            "extraction_limitations": ["Keyword matching is not competence validation."],
        }

        with patch("backend.main.analyze_resume", return_value=analyzer_result):
            response = self._post_pdf(pdf_path)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), analyzer_result)

    def test_missing_file_returns_validation_error(self):
        response = self.client.post(
            "/api/analyze/",
            data={"jd_text": "Python engineering role"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("file", response.json()["detail"][0]["loc"])

    def test_missing_job_description_returns_validation_error(self):
        pdf_path = self._create_pdf()
        with pdf_path.open("rb") as pdf_file:
            response = self.client.post(
                "/api/analyze/",
                files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Job description cannot be empty. Please provide some text.")

    def test_empty_or_whitespace_job_description_returns_validation_error(self):
        pdf_path = self._create_pdf()
        for jd_text in ["", " \t\n "]:
            with self.subTest(jd_text=jd_text):
                with pdf_path.open("rb") as pdf_file:
                    response = self.client.post(
                        "/api/analyze/",
                        files={
                            "file": (pdf_path.name, pdf_file, "application/pdf"),
                            "jd_text": (None, jd_text),
                        },
                    )
                self.assertEqual(response.status_code, 422)
                self.assertIn("cannot be empty", response.json()["detail"])

    def test_invalid_pdf_upload_returns_bad_request_and_cleans_temp_file(self):
        # A file without a PDF signature is now rejected before analysis runs
        # (the response contract is unchanged). Temp-file cleanup for the
        # analysis-time failure path is covered by the extraction-error test.
        temp_root = Path(tempfile.gettempdir())
        before = set(temp_root.glob("smarthire-upload-*"))

        with patch("backend.main.analyze_resume") as analyze:
            response = self.client.post(
                "/api/analyze/",
                files={"file": ("invalid.pdf", b"not a PDF", "application/pdf")},
                data={"jd_text": "Python engineering role"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unable to extract text", response.json()["detail"])
        self.assertIn("OCR is not available", response.json()["detail"])
        analyze.assert_not_called()
        self.assertEqual(set(temp_root.glob("smarthire-upload-*")), before)

    def test_extraction_error_returns_safe_message_and_cleans_temp_file(self):
        pdf_path = self._create_pdf()
        observed_temp_path = None

        def fake_analyze(path, _jd_text):
            nonlocal observed_temp_path
            observed_temp_path = Path(path)
            raise NoExtractableTextError("Sensitive parser detail")

        with patch("backend.main.analyze_resume", side_effect=fake_analyze):
            response = self._post_pdf(pdf_path)

        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Sensitive parser detail", response.text)
        self.assertIsNotNone(observed_temp_path)
        self.assertFalse(observed_temp_path.exists())
        self.assertFalse(observed_temp_path.parent.exists())

    def test_unexpected_error_is_sanitized_and_temp_file_is_cleaned(self):
        pdf_path = self._create_pdf()
        observed_temp_path = None

        def fake_analyze(path, _jd_text):
            nonlocal observed_temp_path
            observed_temp_path = Path(path)
            raise RuntimeError("Sensitive internal exception detail")

        with patch("backend.main.analyze_resume", side_effect=fake_analyze):
            response = self._post_pdf(pdf_path)

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("Sensitive internal exception detail", response.text)
        self.assertEqual(
            response.json()["detail"],
            "Resume analysis failed due to an internal server error.",
        )
        self.assertIsNotNone(observed_temp_path)
        self.assertFalse(observed_temp_path.exists())
        self.assertFalse(observed_temp_path.parent.exists())


if __name__ == "__main__":
    unittest.main()