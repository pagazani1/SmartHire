"""Tests that the evaluation harness runs and its arithmetic is consistent.

These use tiny synthetic text pairs. They prove the harness works; they say nothing
about how accurate the model or the policy is on real data.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_decision_policy import (
    LABELS,
    build_report,
    classification_summary,
    main,
    majority_baseline,
    render_markdown,
    score_pairs,
    threshold_sweep,
)

JD = (
    "We are hiring a backend engineer. Required skills: Python, FastAPI and PostgreSQL. "
    "Preferred: Docker. Responsibilities include building and maintaining REST APIs."
)
STRONG = (
    "Backend engineer with five years of experience building REST APIs in Python "
    "with FastAPI, PostgreSQL databases and Docker containers in production systems."
)
WEAK = (
    "Graphic designer experienced with brand identity, illustration, typography and "
    "print layout for marketing campaigns across several agencies and studios."
)


def _frame() -> pd.DataFrame:
    rows = [
        (STRONG, JD, "Good Fit"),
        (STRONG, JD, "Good Fit"),
        (WEAK, JD, "No Fit"),
        (WEAK, JD, "No Fit"),
        (WEAK, JD, "No Fit"),
        (STRONG, JD, "Potential Fit"),
    ]
    return pd.DataFrame(rows, columns=["resume", "jd", "label"])


class EvaluationHarnessTests(unittest.TestCase):
    def test_scoring_and_report_are_internally_consistent(self):
        scored = score_pairs(_frame())
        self.assertEqual(len(scored) + scored.attrs["errors"], 6)
        report = build_report(scored)
        self.assertEqual(report["rows_scored"], len(scored))
        self.assertEqual(sum(report["evidence_status_counts"].values()), len(scored))
        confusion = report["policy_final_label"]["confusion_actual_by_predicted"]
        self.assertEqual(sum(sum(row.values()) for row in confusion.values()), len(scored))
        self.assertEqual(report["majority_baseline"]["always_predicts"], "No Fit")
        # The clearly matching resume must be judged by the policy, not by fallback.
        self.assertGreater(sum(1 for s in scored["evidence_status"] if s == "available"), 0)

    def test_baseline_and_summary_helpers(self):
        truth = ["No Fit", "No Fit", "Good Fit"]
        baseline = majority_baseline(truth)
        self.assertAlmostEqual(baseline["accuracy"], 2 / 3)
        summary = classification_summary(truth, ["No Fit", "Good Fit", "Good Fit"])
        self.assertEqual(summary["per_class"]["Good Fit"]["support"], 1)

    def test_threshold_sweep_marks_current_thresholds(self):
        reliable = pd.DataFrame(
            {
                "truth": ["Good Fit", "Potential Fit", "No Fit", "No Fit"],
                "required_matched": [10, 6, 2, 0],
                "required_total": [10, 10, 10, 10],
            }
        )
        sweep = threshold_sweep(reliable)
        current = [row for row in sweep if row["is_current"]]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["macro_f1"], 1.0)

    def test_cli_writes_report_without_any_resume_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "pairs.csv"
            _frame().to_csv(data, index=False)
            self.assertEqual(main(["--data", str(data), "--output-dir", tmp]), 0)
            report = (Path(tmp) / "policy_validation_report.md").read_text(encoding="utf-8")
            metrics = json.loads((Path(tmp) / "policy_validation_metrics.json").read_text())
        self.assertIn("required_skill_group_coverage_v2", report)
        self.assertNotIn("graphic designer", report.lower())
        self.assertNotIn("Graphic designer", json.dumps(metrics))
        self.assertEqual(set(LABELS), {"Good Fit", "Potential Fit", "No Fit"})
        self.assertIn("Headline comparison", render_markdown(metrics, "pairs.csv"))


if __name__ == "__main__":
    unittest.main()
