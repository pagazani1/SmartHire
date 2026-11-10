import unittest

import numpy as np
import pandas as pd

from src.structured_classifier import (
    NUMERIC_FEATURE_COLUMNS,
    build_combined_text,
    extract_structured_features,
    generate_oof_text_probabilities,
    make_structured_classifier,
    sample_group_label_rows,
)


class TestStructuredFeatures(unittest.TestCase):
    def test_features_capture_required_alternative_as_one_group(self):
        data = pd.DataFrame(
            {
                "resume": ["Python FastAPI developer with SQL experience."],
                "jd": [
                    "Requirements:\n"
                    "- Python\n"
                    "- FastAPI or Flask\n"
                    "- SQL\n"
                    "Preferred Skills:\n"
                    "- Docker"
                ],
                "resume_domain": ["Software Engineering"],
                "jd_domain": ["Software Engineering"],
            }
        )

        features = extract_structured_features(data)

        self.assertEqual(features.iloc[0]["required_skill_group_count"], 3)
        self.assertEqual(features.iloc[0]["matched_required_group_count"], 3)
        self.assertEqual(features.iloc[0]["missing_required_group_count"], 0)
        self.assertEqual(features.iloc[0]["required_skill_coverage"], 1.0)
        self.assertEqual(features.iloc[0]["preferred_skill_group_count"], 1)
        self.assertEqual(features.iloc[0]["preferred_skill_coverage"], 0.0)
        self.assertEqual(features.iloc[0]["matched_skill_count"], 3)
        self.assertEqual(features.iloc[0]["missing_skill_count"], 2)
        self.assertEqual(features.iloc[0]["domain_compatible"], 1)

    def test_undefined_coverage_is_distinguished_from_zero(self):
        data = pd.DataFrame(
            {
                "resume": ["General engineer."],
                "jd": ["We value collaborative teammates."],
                "resume_domain": ["A"],
                "jd_domain": ["B"],
            }
        )

        features = extract_structured_features(data)

        self.assertTrue(np.isnan(features.iloc[0]["required_skill_coverage"]))
        self.assertEqual(features.iloc[0]["required_coverage_available"], 0)
        self.assertTrue(np.isnan(features.iloc[0]["preferred_skill_coverage"]))
        self.assertEqual(features.iloc[0]["preferred_coverage_available"], 0)
        self.assertEqual(features.iloc[0]["domain_compatible"], 0)

    def test_feature_output_does_not_contain_source_text(self):
        data = pd.DataFrame(
            {
                "resume": ["PRIVATE RESUME STRING"],
                "jd": ["PRIVATE JOB DESCRIPTION STRING"],
                "resume_domain": ["A"],
                "jd_domain": ["A"],
            }
        )

        features = extract_structured_features(data)

        self.assertNotIn("resume", features.columns)
        self.assertNotIn("jd", features.columns)
        self.assertNotIn("PRIVATE RESUME STRING", features.to_string())
        self.assertNotIn("PRIVATE JOB DESCRIPTION STRING", features.to_string())

    def test_missing_source_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "resume_domain"):
            extract_structured_features(
                pd.DataFrame({"resume": [], "jd": []})
            )

    def test_combined_text_uses_existing_cleaning(self):
        self.assertEqual(
            build_combined_text("  PYTHON  developer ", " FastAPI role "),
            "python developer fastapi role",
        )

    def test_group_label_sampling_is_reproducible_and_preserves_combinations(self):
        data = pd.DataFrame(
            {
                "resume_group": ["r1"] * 8 + ["r2"] * 3,
                "label": ["Good Fit"] * 4 + ["No Fit"] * 4 + ["Potential Fit"] * 3,
                "row_value": range(11),
            }
        )

        first = sample_group_label_rows(data, max_rows_per_group_label=2, random_state=9)
        second = sample_group_label_rows(data, max_rows_per_group_label=2, random_state=9)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            set(zip(first["resume_group"], first["label"])),
            set(zip(data["resume_group"], data["label"])),
        )
        self.assertTrue(
            first.groupby(["resume_group", "label"]).size().max() <= 2
        )

    def test_structured_classifier_fits_and_predicts_without_training_text(self):
        features = pd.DataFrame(
            {
                "resume_skill_count": [1, 2, 0, 3, 1, 0],
                "jd_skill_count": [2, 2, 2, 3, 1, 3],
                "required_skill_group_count": [1, 2, 2, 3, 1, 2],
                "matched_required_group_count": [1, 2, 0, 3, 0, 0],
                "missing_required_group_count": [0, 0, 2, 0, 1, 2],
                "required_skill_coverage": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                "required_coverage_available": [1, 1, 1, 1, 1, 1],
                "preferred_skill_group_count": [0, 0, 1, 0, 1, 0],
                "matched_preferred_group_count": [0, 0, 0, 0, 1, 0],
                "missing_preferred_group_count": [0, 0, 1, 0, 0, 0],
                "preferred_skill_coverage": [np.nan, np.nan, 0.0, np.nan, 1.0, np.nan],
                "preferred_coverage_available": [0, 0, 1, 0, 1, 0],
                "matched_skill_count": [1, 2, 0, 3, 1, 0],
                "missing_skill_count": [1, 0, 2, 0, 0, 3],
                "unclassified_jd_skill_count": [0, 0, 0, 0, 0, 1],
                "domain_compatible": [1, 1, 0, 1, 0, 0],
                "resume_domain": ["A", "A", "B", "A", "C", "B"],
                "jd_domain": ["A", "A", "C", "A", "B", "A"],
            }
        )
        self.assertEqual(
            list(NUMERIC_FEATURE_COLUMNS),
            [
                column
                for column in features.columns
                if column not in {"resume_domain", "jd_domain"}
            ],
        )
        labels = np.array(
            ["Good Fit", "Good Fit", "No Fit", "Good Fit", "Potential Fit", "No Fit"]
        )

        model = make_structured_classifier()
        model.fit(features, labels)
        predictions = model.predict(features)

        self.assertEqual(len(predictions), len(labels))
        self.assertTrue(set(predictions).issubset(set(labels)))


class TestOutOfFoldTextProbabilities(unittest.TestCase):
    def test_oof_probabilities_are_finite_and_grouped(self):
        texts = np.array(
            [
                f"shared resume jd phrase class{index // 4} group{index}"
                for index in range(12)
            ],
            dtype=object,
        )
        labels = np.array(
            ["Good Fit"] * 4 + ["No Fit"] * 4 + ["Potential Fit"] * 4,
            dtype=object,
        )
        groups = np.array([f"resume_{index}" for index in range(12)], dtype=object)
        class_labels = ["Good Fit", "No Fit", "Potential Fit"]

        probabilities = generate_oof_text_probabilities(
            texts,
            labels,
            groups,
            class_labels,
            n_splits=3,
        )

        self.assertEqual(probabilities.shape, (12, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_oof_rejects_mismatched_input_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            generate_oof_text_probabilities(
                ["text one", "text two"],
                ["Good Fit"],
                ["resume one", "resume two"],
                ["Good Fit", "No Fit"],
                n_splits=2,
            )


if __name__ == "__main__":
    unittest.main()
