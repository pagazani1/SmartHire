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

    def evaluate_coverage(self, total, matched):
        groups = [{f"Skill {index:02}"} for index in range(total)]
        resume_skills = {f"Skill {index:02}" for index in range(matched)}
        return apply_decision_policy(
            self.model_prediction,
            {"resume_skills": resume_skills, "required_skill_groups": groups},
        )

    def test_percentage_threshold_boundaries(self):
        cases = [
            (100, 100, "Good Fit", 100.0),
            (20, 17, "Good Fit", 85.0),
            (19, 16, "Potential Fit", 16 / 19 * 100),
            (2, 1, "Potential Fit", 50.0),
            (19, 9, "No Fit", 9 / 19 * 100),
            (10, 0, "No Fit", 0.0),
        ]
        for total, matched, label, percentage in cases:
            with self.subTest(total=total, matched=matched):
                result = self.evaluate_coverage(total, matched)
                self.assertEqual(result["label"], label)
                evidence = result["decision_evidence"]
                self.assertEqual(evidence["required_skill_group_count"], total)
                self.assertEqual(evidence["matched_required_group_count"], matched)
                self.assertAlmostEqual(
                    evidence["required_skill_coverage_percentage"], percentage
                )
                self.assertIn(f"{matched} out of {total}", result["decision_explanation"])

    def test_alternative_requirement_counts_once_and_either_skill_matches(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": {"Flask"},
                "required_skill_groups": [{"FastAPI", "Flask"}],
                "preferred_skill_groups": [{"Docker"}],
            },
        )

        self.assertEqual(result["label"], "Good Fit")
        self.assertEqual(
            result["decision_evidence"]["matched_required_skill_groups"],
            [["FastAPI", "Flask"]],
        )
        self.assertEqual(result["decision_evidence"]["required_skill_coverage"], 1.0)
        self.assertIn("preferred skills do not affect", result["decision_explanation"])

    def test_preferred_skills_do_not_change_required_coverage(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": {"Python", "Docker", "AWS"},
                "required_skill_groups": [{"Python"}, {"SQL"}],
                "preferred_skill_groups": [{"Docker"}, {"AWS"}],
            },
        )

        self.assertEqual(result["label"], "Potential Fit")
        self.assertEqual(result["decision_evidence"]["required_skill_coverage"], 0.5)
        self.assertEqual(
            result["decision_evidence"]["matched_preferred_skill_groups"],
            [["Docker"], ["AWS"]],
        )

    def test_duplicate_skills_and_requirement_groups_do_not_inflate_coverage(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": {"Python"},
                "required_skill_groups": [
                    {"Python", "Python"},
                    {"Python"},
                    {"SQL"},
                    {"SQL", "sql"},
                ],
            },
        )

        evidence = result["decision_evidence"]
        self.assertEqual(evidence["required_skill_group_count"], 2)
        self.assertEqual(evidence["matched_required_group_count"], 1)
        self.assertEqual(evidence["required_skill_coverage"], 0.5)
        self.assertEqual(result["label"], "Potential Fit")

    def test_missing_skill_information_falls_back_to_text_model(self):
        result = apply_decision_policy(
            self.model_prediction,
            {"required_skill_groups": [{"Python"}]},
        )

        evidence = result["decision_evidence"]
        self.assertEqual(result["label"], "No Fit")
        self.assertFalse(evidence["structured_evidence_available"])
        self.assertEqual(evidence["required_skill_evidence_status"], "unavailable")
        self.assertIsNone(evidence["required_skill_coverage"])
        self.assertEqual(evidence["missing_required_group_count"], 0)
        self.assertEqual(evidence["missing_required_skill_groups"], [])
        self.assertIn("falls back", result["decision_explanation"])

    def test_unreliable_requirement_groups_fall_back_to_text_model(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": set(),
                "required_skill_groups": [{"Python"}, set()],
            },
        )

        evidence = result["decision_evidence"]
        self.assertEqual(result["label"], "No Fit")
        self.assertEqual(evidence["required_skill_evidence_status"], "unavailable")
        self.assertIsNone(evidence["required_skill_coverage"])

    def test_unreliable_resume_skill_evidence_falls_back_to_text_model(self):
        result = apply_decision_policy(
            self.model_prediction,
            {
                "resume_skills": ["Python", None],
                "required_skill_groups": [{"Python"}],
            },
        )

        evidence = result["decision_evidence"]
        self.assertEqual(result["label"], "No Fit")
        self.assertEqual(evidence["required_skill_evidence_status"], "unavailable")
        self.assertEqual(evidence["missing_required_group_count"], 0)
        self.assertIsNone(evidence["required_skill_coverage"])

    def test_no_required_groups_falls_back_without_zero_division(self):
        result = apply_decision_policy(
            self.model_prediction,
            {"resume_skills": set(), "required_skill_groups": []},
        )

        evidence = result["decision_evidence"]
        self.assertEqual(result["label"], "No Fit")
        self.assertEqual(evidence["required_skill_evidence_status"], "no_required_groups")
        self.assertIsNone(evidence["required_skill_coverage"])
        self.assertIsNone(evidence["required_skill_coverage_percentage"])
        self.assertIn("No extractable required skill groups", result["decision_explanation"])

    def test_model_prediction_and_probabilities_are_preserved(self):
        result = self.evaluate_coverage(20, 17)

        self.assertEqual(result["model_label"], "No Fit")
        self.assertEqual(result["model_probs"], self.model_prediction["probs"])
        self.assertEqual(result["probs"], self.model_prediction["probs"])
        self.assertEqual(result["decision_policy"], DECISION_POLICY_ID)


if __name__ == "__main__":
    unittest.main()
