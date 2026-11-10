"""Grouped experiment comparing text, structured, and hybrid classifiers."""

from __future__ import annotations

import json
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import config
from src.preprocessing import clean_text
from src.structured_classifier import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TEXT_PROBABILITY_PREFIX,
    build_combined_text,
    extract_structured_features,
    fit_text_logistic_regression,
    generate_oof_text_probabilities,
    make_structured_classifier,
    predict_text_probabilities,
    probability_feature_frame,
    sample_group_label_rows,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "structured"
REPORT_PATH = BASE_DIR / "evaluation_results" / "structured_model_comparison.md"
HOLDOUT_SEED = 2026
HOLDOUT_SIZE = 0.2
OUTER_FOLDS = 3
INNER_FOLDS = 3
MAX_TRAIN_ROWS_PER_RESUME_LABEL = 5
TRAIN_SAMPLE_SEED = 6842
REGULARIZATION = 1.0
MODEL_NAMES = ("text_logistic_regression", "structured_only", "hybrid")


def _evaluate(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "prediction_distribution": {
            label: int(np.count_nonzero(y_pred == label)) for label in labels
        },
    }


def _group_counts(
    train_groups: pd.Series,
    validation_groups: pd.Series,
) -> dict[str, int]:
    train_unique = set(train_groups)
    validation_unique = set(validation_groups)
    overlap = train_unique.intersection(validation_unique)
    if overlap:
        raise AssertionError(f"Resume group overlap detected: {len(overlap)}")
    return {
        "train_resume_groups": len(train_unique),
        "validation_resume_groups": len(validation_unique),
        "resume_group_overlap": len(overlap),
    }


def _prepare_feature_frame(
    features: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    probability_frame = probability_feature_frame(probabilities, labels)
    combined = pd.concat(
        [
            features.reset_index(drop=True),
            probability_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    return combined, [
        f"{TEXT_PROBABILITY_PREFIX}{label}" for label in labels
    ]


def _feature_availability(features: pd.DataFrame) -> dict[str, Any]:
    row_count = len(features)
    summaries = {
        "required_coverage_defined_rows": int(
            features["required_coverage_available"].sum()
        ),
        "preferred_coverage_defined_rows": int(
            features["preferred_coverage_available"].sum()
        ),
        "rows_with_any_detected_jd_skill": int(
            (features["jd_skill_count"] > 0).sum()
        ),
        "rows_with_any_detected_resume_skill": int(
            (features["resume_skill_count"] > 0).sum()
        ),
        "rows_with_any_matched_skill": int(
            (features["matched_skill_count"] > 0).sum()
        ),
        "resume_domain_categories": int(
            features["resume_domain"].nunique(dropna=False)
        ),
        "jd_domain_categories": int(features["jd_domain"].nunique(dropna=False)),
        "domain_compatibility_rows": int(features["domain_compatible"].sum()),
        "rows": row_count,
    }
    summaries["required_coverage_defined_fraction"] = (
        summaries["required_coverage_defined_rows"] / row_count
    )
    summaries["preferred_coverage_defined_fraction"] = (
        summaries["preferred_coverage_defined_rows"] / row_count
    )
    return summaries


def _fit_predict_fold(
    development: pd.DataFrame,
    structured_features: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    labels: list[str],
    fold_number: int,
) -> dict[str, np.ndarray]:
    full_train_rows = development.iloc[train_indices]
    validation_rows = development.iloc[validation_indices]
    validation_feature_rows = structured_features.iloc[validation_indices]
    group_metadata = _group_counts(
        full_train_rows["resume_group"],
        validation_rows["resume_group"],
    )
    train_rows = sample_group_label_rows(
        full_train_rows,
        max_rows_per_group_label=MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        random_state=TRAIN_SAMPLE_SEED + fold_number,
    )
    train_feature_rows = structured_features.loc[train_rows.index]
    train_groups = train_rows["resume_group"]

    text_vectorizer, text_model = fit_text_logistic_regression(
        train_rows["combined_text"].to_numpy(),
        train_rows["label"].to_numpy(),
    )
    validation_text_probabilities = predict_text_probabilities(
        text_vectorizer,
        text_model,
        validation_rows["combined_text"].to_numpy(),
        labels,
    )
    baseline_predictions = text_model.predict(
        text_vectorizer.transform(validation_rows["combined_text"].to_numpy())
    )
    del text_vectorizer, text_model

    structured_model = make_structured_classifier(regularization=REGULARIZATION)
    structured_model.fit(train_feature_rows, train_rows["label"])
    structured_predictions = structured_model.predict(validation_feature_rows)
    del structured_model

    oof_probabilities = generate_oof_text_probabilities(
        train_rows["combined_text"].to_numpy(),
        train_rows["label"].to_numpy(),
        train_groups.to_numpy(),
        labels,
        n_splits=INNER_FOLDS,
    )
    fusion_train_features, probability_columns = _prepare_feature_frame(
        train_feature_rows,
        oof_probabilities,
        labels,
    )
    fusion_validation_features, _ = _prepare_feature_frame(
        validation_feature_rows,
        validation_text_probabilities,
        labels,
    )
    hybrid_model = make_structured_classifier(
        numeric_columns=[*NUMERIC_FEATURE_COLUMNS, *probability_columns],
        regularization=REGULARIZATION,
    )
    hybrid_model.fit(fusion_train_features, train_rows["label"])
    hybrid_predictions = hybrid_model.predict(fusion_validation_features)

    return {
        "text_logistic_regression": baseline_predictions,
        "structured_only": structured_predictions,
        "hybrid": hybrid_predictions,
        "validation_indices": validation_indices,
        "group_metadata": group_metadata,
        "sampled_train_rows": len(train_rows),
    }


def _run_cross_validation(
    development: pd.DataFrame,
    structured_features: pd.DataFrame,
    labels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    splitter = GroupKFold(n_splits=OUTER_FOLDS)
    fold_records: list[dict[str, Any]] = []
    groups = development["resume_group"].to_numpy()
    targets = development["label"].to_numpy()

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(development, targets, groups=groups),
        start=1,
    ):
        print(
            f"[CV] Outer fold {fold_number}/{OUTER_FOLDS}: "
            f"training rows={len(train_indices)}, "
            f"validation rows={len(validation_indices)}"
        )
        fold_predictions = _fit_predict_fold(
            development,
            structured_features,
            train_indices,
            validation_indices,
            labels,
            fold_number,
        )
        validation_rows = development.iloc[validation_indices]
        group_metadata = fold_predictions["group_metadata"]
        for model_name in MODEL_NAMES:
            metrics = _evaluate(
                validation_rows["label"].to_numpy(),
                fold_predictions[model_name],
                labels,
            )
            fold_records.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_rows": fold_predictions["sampled_train_rows"],
                    "full_train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    **group_metadata,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    **{
                        f"recall_{label}": metrics["classification_report"][label][
                            "recall"
                        ]
                        for label in labels
                    },
                }
            )

    fold_frame = pd.DataFrame(fold_records)
    summaries: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        rows = fold_frame.loc[fold_frame["model"] == model_name]
        summaries[model_name] = {}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            summaries[model_name][f"{metric}_mean"] = float(rows[metric].mean())
            summaries[model_name][f"{metric}_std"] = float(
                rows[metric].std(ddof=0)
            )
        for label in labels:
            metric = f"recall_{label}"
            summaries[model_name][f"{metric}_mean"] = float(rows[metric].mean())
            summaries[model_name][f"{metric}_std"] = float(
                rows[metric].std(ddof=0)
            )
    return fold_records, summaries


def _fit_final_models(
    development: pd.DataFrame,
    final_holdout: pd.DataFrame,
    development_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    labels: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], int]:
    sampled_development = sample_group_label_rows(
        development,
        max_rows_per_group_label=MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        random_state=TRAIN_SAMPLE_SEED,
    )
    sampled_development_features = development_features.loc[
        sampled_development.index
    ]
    y_development = sampled_development["label"].to_numpy()
    y_holdout = final_holdout["label"].to_numpy()

    text_vectorizer, text_model = fit_text_logistic_regression(
        sampled_development["combined_text"].to_numpy(),
        y_development,
    )
    holdout_text_probabilities = predict_text_probabilities(
        text_vectorizer,
        text_model,
        final_holdout["combined_text"].to_numpy(),
        labels,
    )
    text_predictions = text_model.predict(
        text_vectorizer.transform(final_holdout["combined_text"].to_numpy())
    )
    del text_vectorizer, text_model

    structured_model = make_structured_classifier(regularization=REGULARIZATION)
    structured_model.fit(sampled_development_features, y_development)
    structured_predictions = structured_model.predict(holdout_features)

    print(f"[FINAL] Generating {OUTER_FOLDS}-fold grouped OOF LR probabilities...")
    development_oof_probabilities = generate_oof_text_probabilities(
        sampled_development["combined_text"].to_numpy(),
        y_development,
        sampled_development["resume_group"].to_numpy(),
        labels,
        n_splits=OUTER_FOLDS,
    )
    fusion_development_features, probability_columns = _prepare_feature_frame(
        sampled_development_features,
        development_oof_probabilities,
        labels,
    )
    fusion_holdout_features, _ = _prepare_feature_frame(
        holdout_features,
        holdout_text_probabilities,
        labels,
    )
    hybrid_model = make_structured_classifier(
        numeric_columns=[*NUMERIC_FEATURE_COLUMNS, *probability_columns],
        regularization=REGULARIZATION,
    )
    hybrid_model.fit(fusion_development_features, y_development)
    hybrid_predictions = hybrid_model.predict(fusion_holdout_features)

    predictions = {
        "text_logistic_regression": text_predictions,
        "structured_only": structured_predictions,
        "hybrid": hybrid_predictions,
    }
    metrics = {
        model_name: _evaluate(y_holdout, model_predictions, labels)
        for model_name, model_predictions in predictions.items()
    }
    return metrics, predictions, len(sampled_development)


def _write_report(
    *,
    split_metadata: dict[str, Any],
    feature_availability: dict[str, Any],
    class_distribution: dict[str, Any],
    cv_summary: dict[str, dict[str, float]],
    final_metrics: dict[str, dict[str, Any]],
    labels: list[str],
) -> None:
    lines = [
        "# SmartHire Structured Classifier Experiment",
        "",
        "## Scope and limitations",
        "",
        "This is a separate exploratory evaluation. The existing Logistic",
        "Regression model, prediction endpoint, saved artifacts, datasets, and",
        "existing evaluation outputs were not modified. No saved model was",
        "overwritten. The text baseline was freshly fit using the existing",
        "TF-IDF and Logistic Regression settings.",
        "",
        "The dataset has no explicit skill annotations. Skill features below",
        "come from the fixed keyword vocabulary and heuristic JD requirement",
        "classification; they are not validated evidence of competence.",
        "No fit label is hardcoded from skill coverage.",
        "",
        "The exact fictional strong, partial, and weak resume texts requested",
        "for synthetic checks were not found in repository files or supplied",
        "with this task. Their feature values were therefore not independently",
        "recomputed. The user-provided strong/partial summary values are not",
        "treated as experimental results. Unit tests cover controlled inputs",
        "only and do not establish generalization.",
        "",
        "## Data and leakage controls",
        "",
        f"- Rows: {split_metadata['total_rows']:,}; resume groups: "
        f"{split_metadata['total_resume_groups']:,}.",
        f"- Final grouped holdout: {split_metadata['holdout_rows']:,} rows, "
        f"{split_metadata['holdout_resume_groups']:,} resume groups; "
        f"development: {split_metadata['development_rows']:,} rows, "
        f"{split_metadata['development_resume_groups']:,} groups.",
        f"- Holdout: GroupShuffleSplit test_size={HOLDOUT_SIZE}, "
        f"random_state={HOLDOUT_SEED}; overlap="
        f"{split_metadata['holdout_group_overlap']}.",
        "",
        "### Label support",
        "",
        "| Class | Development rows | Final holdout rows |",
        "|---|---:|---:|",
    ]
    for label in labels:
        lines.append(
            f"| {label} | {class_distribution['development'][label]} | "
            f"{class_distribution['final_holdout'][label]} |"
        )
    lines.extend(
        [
            "",
        f"- Development CV: {OUTER_FOLDS}-fold GroupKFold by normalized "
        "resume identity. Hybrid-training text probabilities are generated "
        f"by {INNER_FOLDS}-fold grouped OOF within each outer training fold.",
        f"- Final hybrid training probabilities: {OUTER_FOLDS}-fold grouped "
        "OOF over sampled development rows only. Final holdout predictions "
        "use the text model fit on that same sample.",
        "- TF-IDF, domain encoders, imputers, scalers, and classifiers are fit "
        "only on their applicable training partition.",
        f"- Due to full-corpus TF-IDF memory exhaustion, each training "
        f"partition uses a fixed random sample capped at "
        f"{MAX_TRAIN_ROWS_PER_RESUME_LABEL} rows per resume-group/label "
        "combination. Each available group-label combination is preserved. "
        f"The final development sample contains "
        f"{split_metadata['final_training_sample_rows']:,} rows. Validation "
        "and final holdout rows remain complete.",
        "- Final holdout results are reported once and were not used for "
        "feature or parameter selection.",
        "",
        ]
    )
    lines.extend(
        [
        "### Derived feature availability across all rows",
        "",
        "| Feature evidence | Rows | Share |",
        "|---|---:|---:|",
        f"| Rows with detected resume skills | "
        f"{feature_availability['rows_with_any_detected_resume_skill']:,} | "
        f"{feature_availability['rows_with_any_detected_resume_skill'] / feature_availability['rows']:.3f} |",
        f"| Rows with detected JD skills | "
        f"{feature_availability['rows_with_any_detected_jd_skill']:,} | "
        f"{feature_availability['rows_with_any_detected_jd_skill'] / feature_availability['rows']:.3f} |",
        f"| Rows with required coverage defined | "
        f"{feature_availability['required_coverage_defined_rows']:,} | "
        f"{feature_availability['required_coverage_defined_fraction']:.3f} |",
        f"| Rows with preferred coverage defined | "
        f"{feature_availability['preferred_coverage_defined_rows']:,} | "
        f"{feature_availability['preferred_coverage_defined_fraction']:.3f} |",
        f"| Rows with any keyword skill match | "
        f"{feature_availability['rows_with_any_matched_skill']:,} | "
        f"{feature_availability['rows_with_any_matched_skill'] / feature_availability['rows']:.3f} |",
        f"| Resume / JD domain categories | "
        f"{feature_availability['resume_domain_categories']} / "
        f"{feature_availability['jd_domain_categories']} | — |",
        "",
        "Required/preferred coverage is undefined when no such detected group",
        "exists. The model imputer maps undefined numeric values to zero,",
        "while the availability indicator and group count preserve the",
        "difference between absent annotations and measured zero coverage.",
        "",
        ]
    )
    lines.extend(
        [
        "## Models and fixed configuration",
        "",
        "- **Text Logistic Regression baseline:** existing combined cleaned "
        "resume+JD TF-IDF, fit within each training partition, with the "
        "repository's `config.py` vectorizer and LR settings.",
        "- **Structured-only:** skill counts/coverage/group measures plus "
        "resume and JD domains and their equality indicator.",
        "- **Hybrid:** same structured features plus class probabilities "
        "generated out-of-fold from the text Logistic Regression for fusion "
        "training, and from a development-fitted text LR for validation.",
        f"- Structured and fusion models use regularized Logistic Regression "
        f"(C={REGULARIZATION}, class_weight={config.CLASS_WEIGHT}); no "
        "hyperparameter tuning was performed.",
        "",
        "## Development grouped cross-validation",
        "",
        "Scores are computed from each outer validation fold. Reported mean "
        f"and standard deviation are unweighted across the {OUTER_FOLDS} folds.",
        "",
        "| Model | Accuracy mean ± SD | Macro-F1 mean ± SD | Weighted-F1 mean ± SD | Good Fit recall mean ± SD | Potential Fit recall mean ± SD |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        summary = cv_summary[model_name]
        lines.append(
            f"| {model_name} | "
            f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | "
            f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | "
            f"{summary['weighted_f1_mean']:.4f} ± {summary['weighted_f1_std']:.4f} | "
            f"{summary['recall_Good Fit_mean']:.4f} ± {summary['recall_Good Fit_std']:.4f} | "
            f"{summary['recall_Potential Fit_mean']:.4f} ± {summary['recall_Potential Fit_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Fold-level accuracy, macro-F1, weighted-F1, class recall, rows,",
            "group counts, and overlap checks are saved in",
            "`evaluation_results/structured/fold_metrics.csv`.",
            "",
            "## Untouched final grouped holdout",
            "",
            "| Model | Accuracy | Macro-F1 | Weighted-F1 | Good Fit recall | Potential Fit recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        report = final_metrics[model_name]["classification_report"]
        lines.append(
            f"| {model_name} | "
            f"{final_metrics[model_name]['accuracy']:.4f} | "
            f"{final_metrics[model_name]['macro_f1']:.4f} | "
            f"{final_metrics[model_name]['weighted_f1']:.4f} | "
            f"{report['Good Fit']['recall']:.4f} | "
            f"{report['Potential Fit']['recall']:.4f} |"
        )
    for model_name in MODEL_NAMES:
        lines.extend(
            [
                "",
                f"### {model_name}: per-class metrics",
                "",
                "| Class | Precision | Recall | F1 | Support | Predicted |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        report = final_metrics[model_name]["classification_report"]
        for label in labels:
            values = report[label]
            predicted_count = final_metrics[model_name][
                "prediction_distribution"
            ][label]
            lines.append(
                f"| {label} | {values['precision']:.4f} | "
                f"{values['recall']:.4f} | {values['f1-score']:.4f} | "
                f"{int(values['support'])} | {predicted_count} |"
            )
        matrix = final_metrics[model_name]["confusion_matrix"]
        lines.extend(
            [
                "",
                "Confusion matrix (rows=actual, columns=predicted):",
                "",
                "| Actual \\ Predicted | " + " | ".join(labels) + " |",
                "|---|" + "---:|" * len(labels),
            ]
        )
        for label, row in zip(labels, matrix):
            lines.append(
                "| " + label + " | " + " | ".join(map(str, row)) + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The structured and hybrid scores are experimental comparisons on "
            "the specified grouped split, not evidence that heuristic keyword "
            "coverage measures competence or that the system is ready to "
            "replace the current prediction pipeline. Compare final-holdout "
            "metrics with the fold variation, class recall, and the domain/"
            "feature availability caveats above. A single fixed holdout and "
            "only a few hundred independent resume groups limit certainty.",
            "",
            "The previously saved evaluation used another split/seed and is "
            "historical context only; it is not the final holdout for this "
            "experiment and was not used for tuning. It reported Logistic "
            "Regression accuracy 0.7218 and Macro-F1 0.6060 on its prior "
            "seed-42 split; these are not directly comparable with the "
            "current sampled-training, seed-2026 results.",
            "",
            "No candidate-level or raw resume/JD text is included in this "
            "report or the aggregate experiment outputs.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_structured_experiment() -> dict[str, Any]:
    required_columns = {
        "resume",
        "jd",
        "label",
        "resume_domain",
        "jd_domain",
    }
    print("[LOAD] Loading raw train and test Parquet files...")
    raw_train = pd.read_parquet(config.RAW_TRAIN_PATH)
    raw_test = pd.read_parquet(config.RAW_TEST_PATH)
    missing_columns = required_columns.difference(raw_train.columns).union(
        required_columns.difference(raw_test.columns)
    )
    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    data = pd.concat([raw_train, raw_test], ignore_index=True, sort=False)
    data = data.loc[data["label"].notna()].copy().reset_index(drop=True)
    data["label"] = data["label"].astype(str)
    labels = sorted(data["label"].unique().tolist())
    if len(labels) != 3:
        raise ValueError(f"Expected three classification labels, found {len(labels)}.")
    data["resume_group"] = data["resume"].fillna("").astype(str).map(clean_text)
    data["combined_text"] = [
        build_combined_text(resume, jd)
        for resume, jd in zip(data["resume"], data["jd"])
    ]

    print("[FEATURES] Extracting deterministic skill/domain features...")
    structured_features = extract_structured_features(data)
    availability = _feature_availability(structured_features)
    data = data.drop(columns=["resume", "jd"])
    del raw_train, raw_test
    gc.collect()
    print(
        "[FEATURES] "
        f"required coverage defined={availability['required_coverage_defined_rows']}/"
        f"{availability['rows']}; preferred coverage defined="
        f"{availability['preferred_coverage_defined_rows']}/{availability['rows']}"
    )

    holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_SIZE,
        random_state=HOLDOUT_SEED,
    )
    development_indices, holdout_indices = next(
        holdout_splitter.split(
            data,
            data["label"],
            groups=data["resume_group"],
        )
    )
    development = data.iloc[development_indices].reset_index(drop=True)
    final_holdout = data.iloc[holdout_indices].reset_index(drop=True)
    development_features = structured_features.iloc[
        development_indices
    ].reset_index(drop=True)
    holdout_features = structured_features.iloc[holdout_indices].reset_index(
        drop=True
    )
    split_metadata = {
        "total_rows": len(data),
        "total_resume_groups": int(data["resume_group"].nunique()),
        "development_rows": len(development),
        "development_resume_groups": int(development["resume_group"].nunique()),
        "holdout_rows": len(final_holdout),
        "holdout_resume_groups": int(final_holdout["resume_group"].nunique()),
        "holdout_group_overlap": len(
            set(development["resume_group"]).intersection(
                set(final_holdout["resume_group"])
            )
        ),
        "holdout_seed": HOLDOUT_SEED,
        "holdout_size": HOLDOUT_SIZE,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "max_training_rows_per_resume_label": MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        "training_sample_seed": TRAIN_SAMPLE_SEED,
    }
    if split_metadata["holdout_group_overlap"] != 0:
        raise AssertionError("Resume-group overlap in final holdout split.")
    if set(development["label"].unique()) != set(labels):
        raise ValueError("Development split does not contain every target class.")
    if set(final_holdout["label"].unique()) != set(labels):
        raise ValueError("Final holdout does not contain every target class.")
    print(
        "[SPLIT] "
        f"development rows={len(development)}, groups="
        f"{split_metadata['development_resume_groups']}; holdout rows="
        f"{len(final_holdout)}, groups={split_metadata['holdout_resume_groups']}; "
        f"overlap={split_metadata['holdout_group_overlap']}"
    )

    class_distribution = {
        partition: {
            str(label): int(count)
            for label, count in frame["label"]
            .value_counts()
            .reindex(labels, fill_value=0)
            .items()
        }
        for partition, frame in (
            ("development", development),
            ("final_holdout", final_holdout),
        )
    }
    print("[CV] Running nested grouped evaluation...")
    fold_records, cv_summary = _run_cross_validation(
        development,
        development_features,
        labels,
    )

    print("[FINAL] Fitting all final models on development data...")
    final_metrics, _, final_training_sample_rows = _fit_final_models(
        development,
        final_holdout,
        development_features,
        holdout_features,
        labels,
    )
    split_metadata["final_training_sample_rows"] = final_training_sample_rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "label": label,
                "precision": metrics["classification_report"][label]["precision"],
                "recall": metrics["classification_report"][label]["recall"],
                "f1": metrics["classification_report"][label]["f1-score"],
                "support": metrics["classification_report"][label]["support"],
                "predicted_count": metrics["prediction_distribution"][label],
            }
            for model_name, metrics in final_metrics.items()
            for label in labels
        ]
    ).to_csv(RESULTS_DIR / "final_per_class_metrics.csv", index=False)
    for model_name, metrics in final_metrics.items():
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=pd.Index(labels, name="actual"),
            columns=pd.Index(labels, name="predicted"),
        ).to_csv(RESULTS_DIR / f"final_confusion_matrix_{model_name}.csv")
    pd.DataFrame(
        {
            "actual_count": final_holdout["label"].value_counts().reindex(
                labels, fill_value=0
            ),
            **{
                f"{model_name}_predicted_count": pd.Series(
                    metrics["prediction_distribution"]
                ).reindex(labels, fill_value=0)
                for model_name, metrics in final_metrics.items()
            },
        }
    ).to_csv(RESULTS_DIR / "final_class_distributions.csv", index_label="label")
    pd.DataFrame(
        [
            {"feature": feature, "available_rows": int(structured_features[feature].notna().sum()),
             "unique_values": int(structured_features[feature].nunique(dropna=False))}
            for feature in [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS]
        ]
    ).to_csv(RESULTS_DIR / "feature_availability.csv", index=False)
    (RESULTS_DIR / "structured_metrics.json").write_text(
        json.dumps(
            {
                "split": split_metadata,
                "feature_availability": availability,
                "class_distribution": class_distribution,
                "cross_validation": {
                    "folds": fold_records,
                    "summary": cv_summary,
                },
                "final_holdout_models": final_metrics,
                "configuration": {
                    "regularization_C": REGULARIZATION,
                    "class_weight": config.CLASS_WEIGHT,
                    "logreg_max_iter": config.LOGREG_MAX_ITER,
                    "max_training_rows_per_resume_label": (
                        MAX_TRAIN_ROWS_PER_RESUME_LABEL
                    ),
                    "training_sample_seed": TRAIN_SAMPLE_SEED,
                    "outer_folds": OUTER_FOLDS,
                    "inner_folds": INNER_FOLDS,
                    "tfidf_max_features": config.TFIDF_MAX_FEATURES,
                    "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
                    "tfidf_min_df": config.TFIDF_MIN_DF,
                    "tfidf_max_df": config.TFIDF_MAX_DF,
                    "tfidf_sublinear_tf": config.TFIDF_SUBLINEAR_TF,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        split_metadata=split_metadata,
        feature_availability=availability,
        class_distribution=class_distribution,
        cv_summary=cv_summary,
        final_metrics=final_metrics,
        labels=labels,
    )
    print(f"[SAVED] Aggregate results: {RESULTS_DIR}")
    print(f"[SAVED] Comparison report: {REPORT_PATH}")
    for model_name, metrics in final_metrics.items():
        print(
            f"[FINAL] {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
    return final_metrics


if __name__ == "__main__":
    run_structured_experiment()
