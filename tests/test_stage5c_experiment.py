import unittest

import numpy as np
import pandas as pd

from evaluate_stage5c import (
    _class_probabilities,
    _extract_stage5c_features,
    _make_pair_vectorizer,
    _pair_training_matrices,
)
from sklearn.linear_model import LogisticRegression
from scipy import sparse


class TestStage5CRepresentations(unittest.TestCase):
    def test_pair_representation_keeps_resume_and_jd_blocks_distinct(self):
        resumes = [
            "Python FastAPI backend developer",
            "Python SQL developer",
            "Java web developer",
        ]
        jds = [
            "Python FastAPI SQL internship",
            "Python SQL backend role",
            "Java web internship",
        ]
        vectorizer = _make_pair_vectorizer()

        resume_matrix, jd_matrix, separate, interaction = _pair_training_matrices(
            vectorizer,
            resumes,
            jds,
            fit=True,
        )

        vocabulary_width = len(vectorizer.vocabulary_)
        self.assertEqual(separate.shape, (3, vocabulary_width * 2))
        self.assertEqual(interaction.shape, (3, vocabulary_width * 3 + 1))
        np.testing.assert_allclose(
            np.asarray(resume_matrix.multiply(jd_matrix).sum(axis=1)).ravel(),
            interaction[:, -1].toarray().ravel(),
        )

    def test_unavailable_skill_overlap_is_not_encoded_as_zero(self):
        data = pd.DataFrame(
            {
                "resume": ["General engineering experience"],
                "jd": ["Collaborative team member wanted"],
                "resume_domain": [None],
                "jd_domain": ["Technology"],
            }
        )

        features = _extract_stage5c_features(data)

        self.assertTrue(np.isnan(features.loc[0, "required_skill_coverage"]))
        self.assertEqual(features.loc[0, "required_coverage_available"], 0)
        self.assertTrue(np.isnan(features.loc[0, "resume_jd_skill_overlap_ratio"]))
        self.assertEqual(features.loc[0, "resume_jd_skill_overlap_available"], 0)
        self.assertTrue(np.isnan(features.loc[0, "resume_jd_skill_jaccard"]))
        self.assertEqual(features.loc[0, "resume_jd_skill_jaccard_available"], 0)

    def test_probability_columns_follow_requested_label_order(self):
        features = sparse.csr_matrix(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.1, 0.2],
                [0.9, 0.8],
                [0.2, 0.1],
                [0.8, 0.9],
            ]
        )
        targets = np.array(
            ["Good Fit", "No Fit", "Good Fit", "No Fit", "Good Fit", "No Fit"]
        )
        model = LogisticRegression(random_state=42).fit(features, targets)

        probabilities = _class_probabilities(
            model,
            features[:2],
            ["No Fit", "Good Fit"],
        )

        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        raw_probabilities = model.predict_proba(features[:2])
        class_indexes = {
            label: index for index, label in enumerate(model.classes_)
        }
        np.testing.assert_allclose(
            probabilities[:, 0],
            raw_probabilities[:, class_indexes["No Fit"]],
        )
        np.testing.assert_allclose(
            probabilities[:, 1],
            raw_probabilities[:, class_indexes["Good Fit"]],
        )


if __name__ == "__main__":
    unittest.main()
