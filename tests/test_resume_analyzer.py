import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

import config
from src.resume_analyzer import analyze_resume


class RecordingVectorizer:
    def __init__(self):
        self.transformed_texts = None

    def transform(self, texts):
        self.transformed_texts = texts
        return texts


class FixedClassifier:
    classes_ = ["Good Fit", "No Fit", "Potential Fit"]

    def predict(self, features):
        return ["Potential Fit"]

    def predict_proba(self, features):
        return [[0.1, 0.2, 0.7]]


class TestAnalyzeResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "synthetic_resume.pdf"
        self.resume_text = "Synthetic Python developer with Docker experience."
        self.jd_text = "Seeking a Python engineer with AWS and Docker."
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), self.resume_text)
        document.save(self.pdf_path)
        document.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analyze_resume_composes_existing_inference_and_skill_results(self):
        vectorizer = RecordingVectorizer()
        model = FixedClassifier()
        expected_skills = {
            "resume_skills": {"Python", "Docker"},
            "matched_skills": {"Python", "Docker"},
            "missing_skills": {"AWS"},
            "jd_skills": {"Python", "Docker", "AWS"},
            "required_skills": {"Python"},
            "preferred_skills": {"Docker"},
            "required_skill_groups": [{"Python"}],
            "preferred_skill_groups": [{"Docker"}],
            "matched_required_skills": {"Python"},
            "missing_required_skills": set(),
            "matched_preferred_skills": {"Docker"},
            "missing_preferred_skills": set(),
            "unclassified_jd_skills": {"AWS"},
            "total_jd_skills": 3,
            "coverage_metrics": {
                "required_skill_coverage": 1.0,
                "required_skill_coverage_percentage": 100.0,
                "preferred_skill_coverage": 1.0,
                "overall_skill_coverage": 2 / 3,
                "required_skill_count": 1,
                "preferred_skill_count": 1,
                "unclassified_skill_count": 1,
                "matched_skill_count": 2,
                "missing_skill_count": 1,
            },
            "extraction_limitations": ["Keyword matching is not competence validation."],
        }

        with (
            patch("src.resume_analyzer.load_artifacts", return_value=(vectorizer, model)),
            patch(
                "src.resume_analyzer.analyse_skills",
                return_value=expected_skills,
            ),
        ):
            result = analyze_resume(self.pdf_path, self.jd_text)

        self.assertEqual(
            vectorizer.transformed_texts,
            ["synthetic python developer with docker experience. seeking a python engineer with aws and docker."],
        )
        self.assertEqual(result["label"], "Good Fit")
        self.assertEqual(result["model_label"], "Potential Fit")
        self.assertEqual(
            result["probs"],
            {"Good Fit": 0.1, "No Fit": 0.2, "Potential Fit": 0.7},
        )
        self.assertEqual(result["model_probs"], result["probs"])
        self.assertEqual(
            result["decision_evidence"]["matched_required_skill_groups"],
            [["Python"]],
        )
        self.assertIn("keyword evidence", result["decision_explanation"])
        self.assertEqual(result["resume_skills"], ["Docker", "Python"])
        self.assertEqual(result["matched_skills"], ["Docker", "Python"])
        self.assertEqual(result["missing_skills"], ["AWS"])
        self.assertEqual(result["detected_resume_skills"], ["Docker", "Python"])
        self.assertEqual(result["detected_jd_skills"], ["AWS", "Docker", "Python"])
        self.assertEqual(result["required_skills"], ["Python"])
        self.assertEqual(result["preferred_skills"], ["Docker"])
        self.assertEqual(result["required_skill_groups"], [["Python"]])
        self.assertEqual(result["preferred_skill_groups"], [["Docker"]])
        self.assertEqual(result["matched_required_skills"], ["Python"])
        self.assertEqual(result["missing_required_skills"], [])
        self.assertEqual(result["matched_preferred_skills"], ["Docker"])
        self.assertEqual(result["missing_preferred_skills"], [])
        self.assertEqual(result["unclassified_jd_skills"], ["AWS"])
        self.assertEqual(result["total_jd_skills"], 3)
        self.assertEqual(
            result["coverage_metrics"]["overall_skill_coverage"],
            2 / 3,
        )
        self.assertEqual(
            result["coverage_metrics"]["required_skill_coverage_percentage"],
            100.0,
        )
        self.assertEqual(result["extraction_limitations"], expected_skills["extraction_limitations"])
        self.assertNotIn("resume_text", result)

    def test_invalid_job_description_is_rejected_before_pdf_extraction(self):
        for jd_text in ["", " \t\n ", None, 123]:
            with self.subTest(jd_text=jd_text):
                with patch("src.resume_analyzer.extract_text_from_pdf") as extract:
                    with self.assertRaisesRegex(ValueError, "Job description"):
                        analyze_resume(self.pdf_path, jd_text)
                extract.assert_not_called()

    def test_pdf_extraction_errors_are_propagated(self):
        with (
            patch(
                "src.resume_analyzer.extract_text_from_pdf",
                side_effect=FileNotFoundError("Synthetic missing PDF"),
            ),
            patch("src.resume_analyzer.load_artifacts") as load,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Synthetic missing PDF"):
                analyze_resume(self.pdf_path, self.jd_text)
        load.assert_not_called()

    def test_model_inference_errors_are_propagated(self):
        with (
            patch(
                "src.resume_analyzer.load_artifacts",
                return_value=(RecordingVectorizer(), FixedClassifier()),
            ),
            patch(
                "src.resume_analyzer.predict",
                side_effect=RuntimeError("Synthetic inference failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Synthetic inference failure"):
                analyze_resume(self.pdf_path, self.jd_text)

    @unittest.skipUnless(
        config.TFIDF_VECTORIZER_PATH.exists()
        and config.LOGISTIC_REGRESSION_PATH.exists(),
        "Existing TF-IDF and Logistic Regression artifacts are unavailable.",
    )
    def test_complete_workflow_with_existing_saved_artifacts(self):
        with (
            patch(
                "predict.VECTORIZER_PATH",
                str(config.TFIDF_VECTORIZER_PATH),
            ),
            patch(
                "predict.MODEL_PATH",
                str(config.LOGISTIC_REGRESSION_PATH),
            ),
        ):
            result = analyze_resume(
                self.pdf_path,
                "Synthetic role seeking Python, Docker, and Kubernetes skills.",
            )

        self.assertIn(result["label"], {"Good Fit", "No Fit", "Potential Fit"})
        self.assertEqual(
            set(result["probs"]),
            {"Good Fit", "No Fit", "Potential Fit"},
        )
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0, places=6)
        self.assertEqual(result["resume_skills"], ["Docker", "Python"])
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("Kubernetes", result["missing_skills"])

    @unittest.skipUnless(
        config.TFIDF_VECTORIZER_PATH.exists()
        and config.LOGISTIC_REGRESSION_PATH.exists()
        and all(
            (config.BASE_DIR / "test_resumes" / f"SmartHire_{case}_Match_Resume.pdf").exists()
            for case in ("Strong", "Partial", "Weak")
        ),
        "Production model artifacts or synthetic resume fixtures are unavailable.",
    )
    def test_decision_policy_on_all_synthetic_match_levels(self):
        jd_text = (
            "Python Backend Developer Intern\n"
            "Required Skills:\n"
            "- Python\n"
            "- FastAPI or Flask\n"
            "- REST APIs\n"
            "- SQL\n"
            "- SQLite\n"
            "- Pandas\n"
            "- NumPy\n"
            "- scikit-learn\n"
            "- Git\n"
            "- GitHub\n"
            "- machine learning\n"
            "- text classification"
        )
        expected = {
            "Strong": ("Good Fit", 12, 12),
            "Partial": ("No Fit", 4, 12),
            "Weak": ("No Fit", 0, 12),
        }
        with (
            patch("predict.VECTORIZER_PATH", str(config.TFIDF_VECTORIZER_PATH)),
            patch("predict.MODEL_PATH", str(config.LOGISTIC_REGRESSION_PATH)),
        ):
            for case, (label, matched_count, required_count) in expected.items():
                with self.subTest(case=case):
                    pdf_path = (
                        config.BASE_DIR
                        / "test_resumes"
                        / f"SmartHire_{case}_Match_Resume.pdf"
                    )
                    result = analyze_resume(pdf_path, jd_text)
                    self.assertEqual(result["label"], label)
                    self.assertEqual(result["model_probs"], result["probs"])
                    self.assertAlmostEqual(sum(result["model_probs"].values()), 1.0)
                    evidence = result["decision_evidence"]
                    self.assertEqual(
                        evidence["matched_required_group_count"], matched_count
                    )
                    self.assertEqual(
                        evidence["required_skill_group_count"], required_count
                    )
                    self.assertEqual(
                        evidence["missing_required_group_count"],
                        required_count - matched_count,
                    )
                    self.assertAlmostEqual(
                        evidence["required_skill_coverage_percentage"],
                        matched_count / required_count * 100,
                    )
                    self.assertTrue(evidence["structured_evidence_available"])


if __name__ == "__main__":
    unittest.main()
