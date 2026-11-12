import unittest

from src.decision_policy import DECISION_POLICY_ID, apply_decision_policy


class TestDecisionPolicy(unittest.TestCase):
    def setUp(self):
        self.model_prediction = {
            "label": "No Fit",
            "probs": {
                "Good Fit": 0.1,
                "No Fit": 0.8,
                "Potential Fit": 0.1,
            },
        }

    def test_alternative_requirement_counts_as_one_group(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": {"Flask"},
                "required_skill_groups": [{"FastAPI", "Flask"}],
                "preferred_skill_groups": [{"Docker"}],
                "matched_skills": {"Flask"},
                "missing_skills": {"FastAPI", "Docker"},
                "matched_required_skills": {"Flask"},
                "missing_required_skills": set(),
                "matched_preferred_skills": set(),
                "missing_preferred_skills": {"Docker"},
            },
        )

        self.assertEqual(result["label"], "Good Fit")
        self.assertEqual(result["model_label"], "No Fit")
        self.assertEqual(result["model_probs"], self.model_prediction["probs"])
        self.assertEqual(result["probs"], self.model_prediction["probs"])
        self.assertEqual(
            result["decision_evidence"]["matched_required_skill_groups"],
            [["FastAPI", "Flask"]],
        )
        self.assertEqual(
            result["decision_evidence"]["missing_preferred_skill_groups"],
            [["Docker"]],
        )
        self.assertEqual(result["decision_policy"], DECISION_POLICY_ID)
        self.assertIn("preferred skills do not override", result["decision_explanation"])

    def test_missing_skill_information_falls_back_to_text_model(self):
        result = apply_decision_policy(
            self.model_prediction,
            {"required_skill_groups": [{"Python"}]},
        )

        self.assertEqual(result["label"], "No Fit")
        self.assertFalse(
            result["decision_evidence"]["structured_evidence_available"]
        )
        self.assertIsNone(result["decision_evidence"]["required_skill_coverage"])
        self.assertIn("falls back", result["decision_explanation"])

    def test_no_classified_requirements_falls_back_to_text_model(self):
        result = apply_decision_policy(
            self.model_prediction,
            {"resume_skills": {"Python"}, "required_skill_groups": []},
        )

        self.assertEqual(result["label"], "No Fit")
        self.assertFalse(
            result["decision_evidence"]["structured_evidence_available"]
        )


if __name__ == "__main__":
    unittest.main()
