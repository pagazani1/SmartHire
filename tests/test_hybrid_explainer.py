import math
import unittest

import numpy as np

from src.hybrid_explainer import HybridExplanationError, explain_hybrid
from src.hybrid_predict import predict_hybrid
from src.structured_classifier import load_hybrid_bundle


_RESUME = (
    "Python developer with experience in FastAPI, Flask, REST APIs, SQL, "
    "SQLite, Pandas, NumPy, scikit-learn, Git, and GitHub."
)
_JD = (
    "Python Backend Developer Intern. Requirements: Python, FastAPI or Flask, "
    "REST APIs, SQL, SQLite, Pandas, NumPy, scikit-learn, Git, GitHub, "
    "machine learning, and text classification."
)


class _PredictorWithoutPipeline:
    classes_ = np.array(["Good Fit", "No Fit", "Potential Fit"])

    def predict_proba(self, features):
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))

    def predict(self, features):
        return np.repeat("Potential Fit", len(features))


class TestHybridExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_hybrid_bundle()

    def test_feature_and_class_alignment_and_logit_reconstruction(self):
        result = explain_hybrid(_RESUME, _JD, bundle=self.bundle, top_k_text_terms=5)

        self.assertEqual(result["class_order"], self.bundle["label_order"])
        self.assertEqual(
            list(result["hybrid_logit_explanations"]),
            self.bundle["label_order"],
        )
        for label in self.bundle["label_order"]:
            explanation = result["hybrid_logit_explanations"][label]
            self.assertTrue(explanation["reconstruction_verified"])
            self.assertLessEqual(
                abs(explanation["reconstruction_error"]),
                1e-10 + 1e-9 * abs(explanation["model_logit"]),
            )
            self.assertAlmostEqual(
                explanation["reconstructed_logit"],
                explanation["intercept"]
                + sum(
                    item["contribution"]
                    for item in explanation["contributions"]
                ),
                places=10,
            )
            self.assertEqual(
                len(explanation["contributions"]),
                len(
                    self.bundle["estimators"]["hybrid_model"]
                    .named_steps["preprocessing"]
                    .get_feature_names_out()
                ),
            )
            self.assertEqual(
                sum(
                    item["feature_type"] == "text_probability"
                    for item in explanation["contributions"]
                ),
                len(self.bundle["label_order"]),
            )

    def test_positive_and_negative_hybrid_and_text_contributions_are_reported(self):
        result = explain_hybrid(_RESUME, _JD, bundle=self.bundle, top_k_text_terms=8)
        hybrid_contributions = [
            item["contribution"]
            for class_result in result["hybrid_logit_explanations"].values()
            for item in class_result["contributions"]
        ]
        self.assertTrue(any(value > 0 for value in hybrid_contributions))
        self.assertTrue(any(value < 0 for value in hybrid_contributions))

        text_contributions = [
            item["contribution"]
            for class_result in result["text_model_explanations"].values()
            for key in ("top_positive_terms", "top_negative_terms")
            for item in class_result[key]
        ]
        self.assertTrue(any(value > 0 for value in text_contributions))
        self.assertTrue(any(value < 0 for value in text_contributions))
        for class_result in result["text_model_explanations"].values():
            for item in class_result["top_positive_terms"]:
                self.assertGreater(item["contribution"], 0)
                self.assertAlmostEqual(
                    item["tfidf_value"] * item["coefficient"],
                    item["contribution"],
                )
            for item in class_result["top_negative_terms"]:
                self.assertLess(item["contribution"], 0)

    def test_unavailable_features_and_unknown_domains_are_handled(self):
        result = explain_hybrid(
            "Worked on general projects.",
            "Seeking a person for a general role.",
            bundle=self.bundle,
            resume_domain="Unseen Domain",
            jd_domain="Another Unseen Domain",
        )
        contributions = result["hybrid_logit_explanations"]["No Fit"][
            "contributions"
        ]
        for item in contributions:
            self.assertTrue(math.isfinite(item["transformed_value"]))
            self.assertTrue(math.isfinite(item["coefficient"]))
            self.assertTrue(math.isfinite(item["contribution"]))
        for domain_prefix in (
            "categorical__resume_domain_",
            "categorical__jd_domain_",
        ):
            domain_values = [
                item["transformed_value"]
                for item in contributions
                if item["feature_name"].startswith(domain_prefix)
            ]
            self.assertTrue(domain_values)
            self.assertTrue(all(value == 0.0 for value in domain_values))
        self.assertIsNone(
            result["prediction"]["structured_features"][
                "required_skill_coverage"
            ]
        )

    def test_explanation_preserves_existing_prediction_exactly(self):
        before = predict_hybrid(_RESUME, _JD, bundle=self.bundle)
        explanation = explain_hybrid(_RESUME, _JD, bundle=self.bundle)
        self.assertEqual(explanation["prediction"], before)
        self.assertEqual(explanation["predicted_label"], before["predicted_label"])
        self.assertEqual(
            explanation["hybrid_probabilities"],
            before["hybrid_probabilities"],
        )

    def test_empty_invalid_inputs_and_invalid_top_k_are_rejected(self):
        for resume, jd in [
            ("", _JD),
            (" \t\n", _JD),
            (None, _JD),
            (_RESUME, ""),
            (_RESUME, 123),
        ]:
            with self.subTest(resume=resume, jd=jd):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    explain_hybrid(resume, jd, bundle=self.bundle)
        for invalid_top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=invalid_top_k):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    explain_hybrid(
                        _RESUME,
                        _JD,
                        bundle=self.bundle,
                        top_k_text_terms=invalid_top_k,
                    )

    def test_unavailable_linear_coefficients_raise_clear_error(self):
        bundle = dict(self.bundle)
        bundle["estimators"] = dict(self.bundle["estimators"])
        bundle["estimators"]["hybrid_model"] = _PredictorWithoutPipeline()
        with self.assertRaisesRegex(
            HybridExplanationError,
            "preprocessing pipeline",
        ):
            explain_hybrid(_RESUME, _JD, bundle=bundle)


if __name__ == "__main__":
    unittest.main()
