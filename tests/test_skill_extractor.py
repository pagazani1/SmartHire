import unittest

from src.skill_extractor import (
    analyse_skills,
    extract_jd_skills,
    extract_resume_skills,
    extract_skills,
)


class TestSkillCompatibilityAnalysis(unittest.TestCase):
    def test_full_match_calculates_all_coverage(self):
        result = analyse_skills(
            "Python, Docker, and AWS experience.",
            "Required Skills:\nPython\nDocker\nPreferred Skills:\nAWS",
        )

        self.assertEqual(result["matched_required_skills"], {"Python", "Docker"})
        self.assertEqual(result["missing_required_skills"], set())
        self.assertEqual(result["matched_preferred_skills"], {"AWS"})
        self.assertEqual(result["missing_preferred_skills"], set())
        self.assertEqual(result["coverage_metrics"]["required_skill_coverage"], 1.0)
        self.assertEqual(result["coverage_metrics"]["preferred_skill_coverage"], 1.0)
        self.assertEqual(result["coverage_metrics"]["overall_skill_coverage"], 1.0)

    def test_partial_match_reports_missing_required_and_preferred_skills(self):
        result = analyse_skills(
            "Python developer.",
            "Required Skills: Python, Docker\nPreferred Skills: AWS",
        )

        self.assertEqual(result["matched_required_skills"], {"Python"})
        self.assertEqual(result["missing_required_skills"], {"Docker"})
        self.assertEqual(result["missing_preferred_skills"], {"AWS"})
        self.assertEqual(result["total_jd_skills"], 3)
        self.assertAlmostEqual(result["coverage_metrics"]["required_skill_coverage"], 0.5)
        self.assertEqual(result["coverage_metrics"]["preferred_skill_coverage"], 0.0)
        self.assertAlmostEqual(result["coverage_metrics"]["overall_skill_coverage"], 1 / 3)

    def test_weak_match_has_zero_coverage_without_inferring_fit_label(self):
        result = analyse_skills(
            "Ruby and MySQL experience.",
            "Required: Python, Docker. Preferred: AWS.",
        )

        self.assertEqual(result["matched_skills"], set())
        self.assertEqual(result["coverage_metrics"]["overall_skill_coverage"], 0.0)
        self.assertEqual(result["coverage_metrics"]["required_skill_coverage"], 0.0)
        self.assertEqual(result["coverage_metrics"]["preferred_skill_coverage"], 0.0)
        self.assertNotIn("label", result)

    def test_required_and_preferred_sections_are_distinguished(self):
        result = extract_jd_skills(
            "Requirements:\nPython\nDocker\n"
            "Preferred Qualifications:\nAWS\n"
            "Responsibilities:\nJava"
        )

        self.assertEqual(result["required_skills"], {"Python", "Docker"})
        self.assertEqual(result["preferred_skills"], {"AWS"})
        self.assertEqual(result["unclassified_jd_skills"], {"Java"})

    def test_requirement_phrases_and_unclassified_mentions(self):
        result = extract_jd_skills(
            "Must have Python experience. Docker is preferred. "
            "Experience with AWS."
        )

        self.assertEqual(result["preferred_skills"], {"Docker"})
        self.assertEqual(result["required_skills"], {"Python", "AWS"})
        self.assertEqual(result["unclassified_jd_skills"], set())

    def test_exact_backend_intern_requirements_structure(self):
        jd_text = (
            "Python Backend Developer Intern\n"
            "Requirements:\n"
            "- Good understanding of Python programming\n"
            "- Experience building REST APIs using FastAPI or Flask\n"
            "- Knowledge of SQL and database operations\n"
            "- Familiarity with Pandas, NumPy, and scikit-learn\n"
            "- Understanding of Git and GitHub\n"
            "- Basic knowledge of machine learning and text classification"
        )
        result = extract_jd_skills(jd_text)

        self.assertEqual(result["required_skills"], {
            "Python", "REST API", "FastAPI", "Flask", "SQL", "Pandas",
            "NumPy", "scikit-learn", "Git", "GitHub", "Machine Learning",
            "Text Classification",
        })
        self.assertEqual(len(result["jd_skills"]), 12)
        self.assertEqual(result["preferred_skills"], set())
        self.assertEqual(result["unclassified_jd_skills"], set())
        self.assertIn({"FastAPI", "Flask"}, result["required_skill_groups"])

    def test_headings_support_markdown_colons_and_bullet_formats(self):
        result = extract_jd_skills(
            "- Requirements:\n"
            "* Python\n"
            "1. Docker\n"
            "- Preferred Skills:\n"
            "- AWS"
        )

        self.assertEqual(result["required_skills"], {"Python", "Docker"})
        self.assertEqual(result["preferred_skills"], {"AWS"})
        self.assertEqual(result["unclassified_jd_skills"], set())

    def test_preferred_phrases_override_required_section(self):
        result = extract_jd_skills(
            "Requirements:\n"
            "- Python required\n"
            "- Docker is preferred\n"
            "- AWS would be an advantage\n"
            "- Nice to have: Kubernetes\n"
            "- Bonus: TensorFlow\n"
            "- Desirable: PyTorch"
        )

        self.assertEqual(result["required_skills"], {"Python"})
        self.assertEqual(
            result["preferred_skills"],
            {"Docker", "AWS", "Kubernetes", "TensorFlow", "PyTorch"},
        )
        self.assertEqual(result["unclassified_jd_skills"], set())

    def test_requirement_phrases_classify_without_a_requirements_heading(self):
        result = extract_jd_skills(
            "Should have Python experience. "
            "Good understanding of Docker. "
            "Experience with AWS. Knowledge of SQL. "
            "Familiarity with Pandas. Understanding of Git. "
            "Basic knowledge of machine learning."
        )

        self.assertEqual(
            result["required_skills"],
            {"Python", "Docker", "AWS", "SQL", "Pandas", "Git", "Machine Learning"},
        )
        self.assertEqual(result["unclassified_jd_skills"], set())

    def test_ambiguous_mentions_remain_unclassified(self):
        result = extract_jd_skills(
            "Our team uses Python and Docker. "
            "Candidates may have exposure to AWS."
        )

        self.assertEqual(result["required_skills"], set())
        self.assertEqual(result["preferred_skills"], set())
        self.assertEqual(result["unclassified_jd_skills"], {"Python", "Docker", "AWS"})

    def test_or_alternatives_count_as_one_required_group_for_coverage(self):
        result = analyse_skills(
            "Built APIs using FastAPI.",
            "Requirements:\nFastAPI or Flask",
        )

        self.assertIn({"FastAPI", "Flask"}, result["required_skill_groups"])
        self.assertEqual(
            result["coverage_metrics"]["required_skill_count"],
            1,
        )
        self.assertEqual(
            result["coverage_metrics"]["required_skill_coverage"],
            1.0,
        )
        self.assertEqual(result["missing_required_skills"], set())

    def test_mixed_required_preferred_and_ambiguous_mentions(self):
        result = extract_jd_skills(
            "Requirements:\nPython\n"
            "Preferred: Docker\n"
            "Responsibilities:\n"
            "Our team also uses AWS."
        )

        self.assertEqual(result["required_skills"], {"Python"})
        self.assertEqual(result["preferred_skills"], {"Docker"})
        self.assertEqual(result["unclassified_jd_skills"], {"AWS"})

    def test_negated_mentions_are_not_counted(self):
        resume_skills = extract_resume_skills(
            "No experience with Python. Java is not used. "
            "Docker was used for deployments."
        )
        jd_skills = extract_jd_skills(
            "Python is not required. Never used Java. Docker experience required."
        )

        self.assertEqual(resume_skills, {"Docker"})
        self.assertEqual(jd_skills["jd_skills"], {"Docker"})
        self.assertEqual(jd_skills["required_skills"], {"Docker"})

    def test_aspirational_mentions_are_not_counted_as_demonstrated_skills(self):
        skills = extract_resume_skills(
            "Interested in learning Python and eager to learn JavaScript. "
            "I am currently learning AWS. Built production services with Docker."
        )

        self.assertEqual(skills, {"Docker"})

    def test_negated_contractions_do_not_count(self):
        skills = extract_resume_skills(
            "Python isn't required for my role, and I don't know Java."
        )
        self.assertEqual(skills, set())

    def test_aliases_are_canonicalized_and_duplicates_removed(self):
        resume_skills = extract_resume_skills(
            "JS and JavaScript; sklearn and scikit-learn; "
            "Postgres and PostgreSQL."
        )
        jd_skills = extract_jd_skills(
            "Requirements: JavaScript, scikit-learn, PostgreSQL"
        )

        self.assertEqual(
            resume_skills,
            {"JavaScript", "scikit-learn", "PostgreSQL"},
        )
        self.assertEqual(
            jd_skills["jd_skills"],
            {"JavaScript", "scikit-learn", "PostgreSQL"},
        )
        self.assertEqual(extract_resume_skills("Node.js developer."), {"Node.js"})

    def test_empty_and_invalid_inputs_return_empty_evidence(self):
        self.assertEqual(extract_skills(" \t\n"), set())
        self.assertEqual(extract_skills(None), set())
        self.assertEqual(extract_resume_skills(42), set())
        self.assertEqual(
            extract_jd_skills(None),
            {
                "jd_skills": set(),
                "required_skills": set(),
                "preferred_skills": set(),
                "unclassified_jd_skills": set(),
                "required_skill_groups": [],
                "preferred_skill_groups": [],
            },
        )
        result = analyse_skills(None, 42)
        self.assertEqual(result["resume_skills"], set())
        self.assertEqual(result["jd_skills"], set())
        self.assertIsNone(result["coverage_metrics"]["overall_skill_coverage"])
        self.assertIsNone(result["coverage_metrics"]["required_skill_coverage"])

    def test_job_description_without_identifiable_skills_has_undefined_coverage(self):
        result = analyse_skills(
            "Experienced engineer.",
            "We are looking for a thoughtful and collaborative teammate.",
        )

        self.assertEqual(result["jd_skills"], set())
        self.assertEqual(result["total_jd_skills"], 0)
        self.assertIsNone(result["coverage_metrics"]["overall_skill_coverage"])
        self.assertIsNone(result["coverage_metrics"]["preferred_skill_coverage"])
        self.assertIsNone(result["coverage_metrics"]["required_skill_coverage"])


if __name__ == "__main__":
    unittest.main()
