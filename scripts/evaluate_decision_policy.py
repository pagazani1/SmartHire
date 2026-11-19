"""Evaluate the text model and the deterministic decision policy separately.

Run this on YOUR labelled file (it is not in the repository):

    python -m scripts.evaluate_decision_policy --data data/raw/raw_test.parquet

The file needs the columns ``resume``, ``jd`` and ``label`` (labels: "Good Fit",
"Potential Fit", "No Fit"). Use ``--sample 2000`` for a quick first run.

What it reports, all computed from the file you pass in (nothing is hard-coded):

* accuracy and macro-F1 of the text model, of the policy, and of an
  "always predict the most common class" baseline;
* how often the model and the policy disagree, and in which direction;
* how often the policy falls back to the model (no usable required skills);
* a sensitivity sweep over the two thresholds, with the current 85% / 50% values marked.

IMPORTANT LIMITS. The dataset labels were not produced by the policy, and nothing here
proves that a label is "correct" for a real hiring decision. The sweep is evidence to
discuss, never an instruction to change the thresholds. Skill groups here come from the
JD text extractor, which differs from the recruiter-defined groups used by persistent
screening. Reports contain counts only, never resume or job text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from predict import load_artifacts_cached, predict
from src.decision_policy import DECISION_POLICY_ID, apply_decision_policy
from src.skill_extractor import analyse_skills

LABELS = ["Good Fit", "Potential Fit", "No Fit"]
CURRENT_GOOD = 0.85
CURRENT_POTENTIAL = 0.50
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def score_pairs(frame: pd.DataFrame, *, resume_col="resume", jd_col="jd", label_col="label") -> pd.DataFrame:
    """Run the model and the policy on every (resume, jd) row; return per-row outcomes only."""
    vectorizer, model = load_artifacts_cached()
    rows: list[dict[str, Any]] = []
    errors = 0
    for resume, jd, truth in zip(frame[resume_col], frame[jd_col], frame[label_col]):
        try:
            prediction = predict(str(resume), str(jd), vectorizer, model)
            skills = analyse_skills(str(resume), str(jd))
            decision = apply_decision_policy(prediction, skills)
        except Exception:  # noqa: BLE001 - count and continue; never log text
            errors += 1
            continue
        evidence = decision["decision_evidence"]
        rows.append(
            {
                "truth": str(truth),
                "model_label": decision["model_label"],
                "policy_label": decision["label"],
                "evidence_status": evidence["required_skill_evidence_status"],
                "required_total": evidence["required_skill_group_count"],
                "required_matched": evidence["matched_required_group_count"],
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["errors"] = errors
    return result


def classification_summary(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    confusion = {
        actual: {
            predicted: int(sum(1 for t, p in zip(y_true, y_pred) if t == actual and p == predicted))
            for predicted in LABELS
        }
        for actual in LABELS
    }
    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
        if y_true
        else None,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_actual_by_predicted": confusion,
    }


def majority_baseline(y_true: list[str]) -> dict[str, Any]:
    if not y_true:
        return {"rows": 0, "accuracy": None, "macro_f1": None}
    majority = Counter(y_true).most_common(1)[0][0]
    summary = classification_summary(y_true, [majority] * len(y_true))
    summary["always_predicts"] = majority
    return summary


def _label_from_ratio(matched: int, total: int, good: float, potential: float) -> str:
    ratio = matched / total
    if ratio >= good - 1e-12:
        return "Good Fit"
    if ratio >= potential - 1e-12:
        return "Potential Fit"
    return "No Fit"


def threshold_sweep(reliable: pd.DataFrame) -> list[dict[str, Any]]:
    """Macro-F1 of coverage-threshold rules on rows with reliable required-group evidence."""
    if reliable.empty:
        return []
    truth = reliable["truth"].tolist()
    matched = reliable["required_matched"].tolist()
    total = reliable["required_total"].tolist()
    grid = []
    for good in np.arange(0.60, 0.96, 0.05):
        for potential in np.arange(0.20, 0.71, 0.05):
            if potential >= good:
                continue
            predicted = [_label_from_ratio(m, t, good, potential) for m, t in zip(matched, total)]
            grid.append(
                {
                    "good_threshold": round(float(good), 2),
                    "potential_threshold": round(float(potential), 2),
                    "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(truth, predicted)),
                    "is_current": bool(abs(good - CURRENT_GOOD) < 1e-9 and abs(potential - CURRENT_POTENTIAL) < 1e-9),
                }
            )
    return sorted(grid, key=lambda row: row["macro_f1"], reverse=True)


def bootstrap_macro_f1(reliable: pd.DataFrame, *, repeats: int = 200, seed: int = 7) -> dict[str, float] | None:
    """95% bootstrap interval for the CURRENT thresholds, so differences can be judged as noise or not."""
    if len(reliable) < 30:
        return None
    rng = np.random.default_rng(seed)
    truth = reliable["truth"].to_numpy()
    predicted = np.array(
        [
            _label_from_ratio(m, t, CURRENT_GOOD, CURRENT_POTENTIAL)
            for m, t in zip(reliable["required_matched"], reliable["required_total"])
        ]
    )
    scores = []
    for _ in range(repeats):
        index = rng.integers(0, len(truth), len(truth))
        scores.append(f1_score(truth[index], predicted[index], labels=LABELS, average="macro", zero_division=0))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def build_report(scored: pd.DataFrame) -> dict[str, Any]:
    truth = scored["truth"].tolist()
    reliable = scored[scored["evidence_status"] == "available"]
    disagreement = Counter(
        f"model={m} / policy={p}"
        for m, p in zip(scored["model_label"], scored["policy_label"])
        if m != p
    )
    sweep = threshold_sweep(reliable)
    current = next((row for row in sweep if row["is_current"]), None)
    return {
        "decision_policy": DECISION_POLICY_ID,
        "rows_scored": len(scored),
        "rows_failed": int(scored.attrs.get("errors", 0)),
        "model": classification_summary(truth, scored["model_label"].tolist()),
        "policy_final_label": classification_summary(truth, scored["policy_label"].tolist()),
        "majority_baseline": majority_baseline(truth),
        "evidence_status_counts": dict(Counter(scored["evidence_status"])),
        "fallback_rate": float((scored["evidence_status"] != "available").mean()) if len(scored) else None,
        "model_policy_agreement": float((scored["model_label"] == scored["policy_label"]).mean()) if len(scored) else None,
        "disagreements": dict(disagreement.most_common()),
        "policy_on_reliable_rows_only": classification_summary(
            reliable["truth"].tolist(), reliable["policy_label"].tolist()
        ),
        "model_on_same_reliable_rows": classification_summary(
            reliable["truth"].tolist(), reliable["model_label"].tolist()
        ),
        "current_thresholds_bootstrap_95ci_macro_f1": bootstrap_macro_f1(reliable),
        "threshold_sweep_top5": sweep[:5],
        "threshold_sweep_current": current,
        "threshold_sweep_rank_of_current": (sweep.index(current) + 1) if current in sweep else None,
        "threshold_sweep_size": len(sweep),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(report: dict[str, Any], source: str) -> str:
    model, policy, base = report["model"], report["policy_final_label"], report["majority_baseline"]
    lines = [
        "# Decision policy vs text model: evaluation",
        "",
        f"Source file: `{source}`. Rows scored: {report['rows_scored']} (failed: {report['rows_failed']}).",
        f"Policy under test: `{report['decision_policy']}` (85% / 50% thresholds, unchanged).",
        "",
        "These numbers describe only the file that was evaluated. The dataset labels were not",
        "created by this policy and are not proof of real-world suitability.",
        "",
        "## Headline comparison (all scored rows)",
        "",
        "| Predictor | Accuracy | Macro-F1 |",
        "|---|---|---|",
        f"| Always `{base.get('always_predicts')}` (baseline) | {_fmt(base['accuracy'])} | {_fmt(base['macro_f1'])} |",
        f"| Text model (`model_label`) | {_fmt(model['accuracy'])} | {_fmt(model['macro_f1'])} |",
        f"| Policy final label (`label`) | {_fmt(policy['accuracy'])} | {_fmt(policy['macro_f1'])} |",
        "",
        "Accuracy alone is misleading when one class dominates; compare macro-F1 as well.",
        "",
        "## Fallback behaviour",
        "",
        f"Rows where the policy could not use required-skill evidence and fell back to the model: "
        f"**{_fmt(report['fallback_rate'])}** of rows.",
        f"Evidence status counts: `{json.dumps(report['evidence_status_counts'], sort_keys=True)}`.",
        "",
        "## Model vs policy disagreement",
        "",
        f"The two labels agree on {_fmt(report['model_policy_agreement'])} of rows. Most common disagreements:",
        "",
    ]
    for key, count in list(report["disagreements"].items())[:8]:
        lines.append(f"- {key}: {count}")
    reliable_policy = report["policy_on_reliable_rows_only"]
    reliable_model = report["model_on_same_reliable_rows"]
    lines += [
        "",
        "## Rows with reliable required-skill evidence only",
        "",
        "| Predictor | Rows | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
        f"| Text model | {reliable_model['rows']} | {_fmt(reliable_model['accuracy'])} | {_fmt(reliable_model['macro_f1'])} |",
        f"| Policy | {reliable_policy['rows']} | {_fmt(reliable_policy['accuracy'])} | {_fmt(reliable_policy['macro_f1'])} |",
        "",
    ]
    ci = report["current_thresholds_bootstrap_95ci_macro_f1"]
    if ci:
        lines.append(f"Bootstrap 95% interval for the policy's macro-F1 here: {ci['low']:.3f} to {ci['high']:.3f}.")
        lines.append("")
    lines += ["## Threshold sensitivity (discussion only)", ""]
    current = report["threshold_sweep_current"]
    if current:
        lines.append(
            f"Current 85%/50% ranks {report['threshold_sweep_rank_of_current']} of "
            f"{report['threshold_sweep_size']} tried pairs (macro-F1 {current['macro_f1']:.3f})."
        )
        lines += ["", "| Good ≥ | Potential ≥ | Macro-F1 | Accuracy |", "|---|---|---|---|"]
        for row in report["threshold_sweep_top5"]:
            marker = " (current)" if row["is_current"] else ""
            lines.append(
                f"| {row['good_threshold']:.2f}{marker} | {row['potential_threshold']:.2f} | "
                f"{row['macro_f1']:.3f} | {row['accuracy']:.3f} |"
            )
        lines += [
            "",
            "A different pair scoring higher on this file is NOT a reason to change the policy by",
            "itself: check that the gap exceeds the bootstrap interval, that it holds on a separate",
            "file, and that the labels mean what recruiters need. Changing thresholds requires a new",
            "policy id and a migration plan.",
        ]
    else:
        lines.append("Not enough rows with reliable evidence to run the sweep.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="parquet or csv with columns resume, jd, label")
    parser.add_argument("--sample", type=int, default=0, help="score a random sample of N rows (0 = all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluation_results" / "policy_validation"))
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        parser.error(f"file not found: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"resume", "jd", "label"} - set(frame.columns)
    if missing:
        parser.error(f"missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["resume", "jd", "label"])
    if args.sample and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=args.seed)

    scored = score_pairs(frame)
    report = build_report(scored)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "policy_validation_report.md").write_text(render_markdown(report, path.name), encoding="utf-8")
    print(f"Wrote {out_dir / 'policy_validation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
