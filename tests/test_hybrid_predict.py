import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.hybrid_predict import predict_hybrid
from src.structured_classifier import (
    CATEGORICAL_FEATURE_COLUMNS,
    HYBRID_EXPERIMENT_ID,
    NUMERIC_FEATURE_COLUMNS,
    HybridArtifactError,
    create_hybrid_bundle,
    extract_structured_features,
    fit_hybrid_estimators,
    load_hybrid_bundle,
    save_hybrid_bundle,
    validate_hybrid_bundle,
)
from src.structured_classifier import build_combined_text


def _small_fitted_bundle():
    labels = ["Good Fit", "No Fit", "Potential Fit"]
    rows = []
    for label in labels:
        for index in range(6):
            if label == "Good Fit":
                resume = (
                    "Python FastAPI Docker SQL experienced engineer "
                    f"goodprofile{index}"
                )
            elif label == "Potential Fit":
                resume = (
                    "Python SQL junior developer "
                    f"partialprofile{index}"
                )
            else:
                resume = f"Ruby designer unrelatedprofile{index}"
            jd = (
                "Requirements: Python, FastAPI or Flask, SQL. "
                "Preferred Skills: Docker. "
                f"role{label.replace(' ', '')}{index}"
            )
            rows.append(
                {
                    "resume": resume,
                    "jd": jd,
                    "resume_domain": "Software Engineering",
                    "jd_domain": "Software Engineering",
                    "label": label,
                    "resume_group": f"resume_{label}_{index}",
                }
            )
    training_rows = pd.DataFrame(rows)
    training_rows["combined_text"] = [
        build_combined_text(resume, jd)
        for resume, jd in zip(training_rows["resume"], training_rows["jd"])
    ]
    features = extract_structured_features(training_rows)
    estimators = fit_hybrid_estimators(
        training_rows,
        features,
        labels,
        oof_splits=3,
        regularization=1.0,
    )
    return create_hybrid_bundle(
        estimators,
        {
            "experiment_version": "test",
            "training_sampled": True,
            "max_training_rows_per_resume_label": 5,
        },
    )


class TestHybridPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _small_fitted_bundle()

    def test_inference_returns_structured_and_probability_evidence(self):
        result = predict_hybrid(
            "Experienced Python FastAPI and SQL developer.",
            "Requirements: Python and FastAPI or Flask. Preferred: Docker.",
            bundle=self.bundle,
            resume_domain="Software Engineering",
            jd_domain="Software Engineering",
        )

        self.assertIn("Python", result["structured_skill_evidence"]["detected_resume_skills"])
        self.assertEqual(
            result["structured_skill_evidence"]["coverage_metrics"][
                "required_skill_coverage"
            ],
            1.0,
        )
        self.assertIn(
            ["FastAPI", "Flask"],
            result["structured_skill_evidence"]["matched_required_skill_groups"],
        )
        self.assertEqual(result["structured_features"]["domain_compatible"], 1)
        self.assertEqual(set(result["text_model_probabilities"]), set(self.bundle["label_order"]))
        self.assertEqual(set(result["hybrid_probabilities"]), set(self.bundle["label_order"]))
        self.assertAlmostEqual(sum(result["text_model_probabilities"].values()), 1.0)
        self.assertAlmostEqual(sum(result["hybrid_probabilities"].values()), 1.0)
        self.assertEqual(
            result["predicted_label"],
            max(result["hybrid_probabilities"], key=result["hybrid_probabilities"].get),
        )
        self.assertEqual(result["experiment_id"], HYBRID_EXPERIMENT_ID)

    def test_bundle_save_load_produces_identical_prediction(self):
        resume = "Experienced Python FastAPI and SQL developer."
        jd = "Requirements: Python and FastAPI or Flask. Preferred: Docker."
        before = predict_hybrid(resume, jd, bundle=self.bundle)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_hybrid_bundle(
                self.bundle,
                Path(temp_dir) / "hybrid_bundle.joblib",
            )
            loaded = load_hybrid_bundle(path)
            after = predict_hybrid(resume, jd, bundle=loaded)

        self.assertEqual(before, after)

    def test_feature_ordering_is_recorded_and_enforced(self):
        schema = self.bundle["feature_schema"]
        self.assertEqual(
            schema["structured_feature_columns"],
            [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS],
        )
        self.assertEqual(
            schema["fusion_feature_columns"],
            [
                *schema["structured_feature_columns"],
                *[
                    f"text_probability__{label}"
                    for label in self.bundle["label_order"]
                ],
            ],
        )
        incompatible = dict(self.bundle)
        incompatible["feature_schema"] = dict(schema)
        incompatible["feature_schema"]["structured_feature_columns"] = list(
            reversed(schema["structured_feature_columns"])
        )
        with self.assertRaisesRegex(HybridArtifactError, "ordering is incompatible"):
            validate_hybrid_bundle(incompatible)

    def test_empty_and_non_string_text_inputs_are_rejected(self):
        for resume, jd in [
            ("", "Python role"),
            (" \t\n", "Python role"),
            (None, "Python role"),
            ("Python resume", ""),
            ("Python resume", 123),
        ]:
            with self.subTest(resume=resume, jd=jd):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    predict_hybrid(resume, jd, bundle=self.bundle)

    def test_invalid_domain_and_conflicting_bundle_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Resume domain"):
            predict_hybrid(
                "Python developer",
                "Python role",
                bundle=self.bundle,
                resume_domain=42,
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.joblib"
            save_hybrid_bundle(self.bundle, path)
            with self.assertRaisesRegex(ValueError, "either bundle or bundle_path"):
                predict_hybrid(
                    "Python developer",
                    "Python role",
                    bundle=self.bundle,
                    bundle_path=path,
                )

    def test_missing_and_incompatible_artifacts_have_clear_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.joblib"
            with self.assertRaisesRegex(HybridArtifactError, "not found"):
                load_hybrid_bundle(missing_path)

            incompatible_path = Path(temp_dir) / "incompatible.joblib"
            joblib.dump({"format_version": 999}, incompatible_path)
            with self.assertRaisesRegex(HybridArtifactError, "invalid or missing"):
                load_hybrid_bundle(incompatible_path)

    def test_unsupported_format_version_is_rejected(self):
        incompatible = dict(self.bundle)
        incompatible["format_version"] = 999
        with self.assertRaisesRegex(HybridArtifactError, "format is incompatible"):
            validate_hybrid_bundle(incompatible)

    def test_scikit_learn_minor_version_mismatch_is_rejected(self):
        incompatible = dict(self.bundle)
        incompatible["sklearn_version"] = "0.0.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_hybrid_bundle(
                incompatible,
                Path(temp_dir) / "version_mismatch.joblib",
            )
            with self.assertRaisesRegex(HybridArtifactError, "scikit-learn"):
                load_hybrid_bundle(path)


if __name__ == "__main__":
    unittest.main()
