"""Reproduce the Stage 2 final fit and save a separate inference bundle."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

import config
from src.preprocessing import clean_text
from src.structured_classifier import (
    HYBRID_ARTIFACT_DIR,
    HYBRID_BUNDLE_PATH,
    build_combined_text,
    build_hybrid_fusion_features,
    create_hybrid_bundle,
    extract_structured_features,
    fit_hybrid_estimators,
    predict_text_probabilities,
    sample_group_label_rows,
    save_hybrid_bundle,
)


BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "evaluation_results" / "structured" / "structured_metrics.json"
README_PATH = HYBRID_ARTIFACT_DIR / "README.md"
HOLDOUT_SEED = 2026
HOLDOUT_SIZE = 0.2
INNER_FOLDS = 3
MAX_TRAIN_ROWS_PER_RESUME_LABEL = 5
TRAIN_SAMPLE_SEED = 6842
REGULARIZATION = 1.0
EXPERIMENT_VERSION = "stage-3a-persisted-final-fit-v1"


def _read_stage2_metrics() -> dict[str, Any]:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Stage 2 metrics are required to verify reproduction: {METRICS_PATH}"
        ) from exc


def _prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    required_columns = {
        "resume",
        "jd",
        "label",
        "resume_domain",
        "jd_domain",
    }
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

    print("[FEATURES] Deriving Stage 2 structured features...")
    all_features = extract_structured_features(data)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_SIZE,
        random_state=HOLDOUT_SEED,
    )
    development_indices, holdout_indices = next(
        splitter.split(data, data["label"], groups=data["resume_group"])
    )
    development = data.iloc[development_indices].copy().reset_index(drop=True)
    holdout = data.iloc[holdout_indices].copy().reset_index(drop=True)
    development_features = all_features.iloc[development_indices].copy().reset_index(
        drop=True
    )
    holdout_features = all_features.iloc[holdout_indices].copy().reset_index(
        drop=True
    )
    overlap = set(development["resume_group"]).intersection(
        set(holdout["resume_group"])
    )
    if overlap:
        raise AssertionError(f"Resume group leakage found: {len(overlap)} groups.")
    metadata = {
        "total_rows": len(data),
        "total_resume_groups": int(data["resume_group"].nunique()),
        "development_rows": len(development),
        "development_resume_groups": int(development["resume_group"].nunique()),
        "holdout_rows": len(holdout),
        "holdout_resume_groups": int(holdout["resume_group"].nunique()),
        "holdout_group_overlap": len(overlap),
    }
    del raw_train, raw_test, data, all_features
    gc.collect()
    return (
        development,
        development_features,
        labels,
        {
            **metadata,
            "holdout": holdout,
            "holdout_features": holdout_features,
        },
    )


def _verify_against_stage2(
    estimators: dict[str, Any],
    holdout: pd.DataFrame,
    holdout_features: pd.DataFrame,
    labels: list[str],
    expected_metrics: dict[str, Any],
) -> dict[str, list[list[int]]]:
    predictions: dict[str, np.ndarray] = {}
    text_features = estimators["text_vectorizer"].transform(
        holdout["combined_text"].to_numpy()
    )
    predictions["text_logistic_regression"] = estimators["text_model"].predict(
        text_features
    )
    predictions["structured_only"] = estimators["structured_model"].predict(
        holdout_features
    )
    text_probabilities = predict_text_probabilities(
        estimators["text_vectorizer"],
        estimators["text_model"],
        holdout["combined_text"].to_numpy(),
        labels,
    )
    fusion_features = build_hybrid_fusion_features(
        holdout_features,
        text_probabilities,
        labels,
    )
    predictions["hybrid"] = estimators["hybrid_model"].predict(fusion_features)

    verified_matrices: dict[str, list[list[int]]] = {}
    expected_models = expected_metrics.get("final_holdout_models", {})
    for model_name, model_predictions in predictions.items():
        expected = expected_models.get(model_name)
        if expected is None:
            raise RuntimeError(
                f"Stage 2 metrics are missing model {model_name!r}."
            )
        actual_matrix = confusion_matrix(
            holdout["label"],
            model_predictions,
            labels=labels,
        )
        expected_matrix = np.asarray(expected["confusion_matrix"], dtype=int)
        if not np.array_equal(actual_matrix, expected_matrix):
            raise RuntimeError(
                f"Reproduced {model_name} confusion matrix does not match "
                "the saved Stage 2 result; refusing to save the bundle."
            )
        verified_matrices[model_name] = actual_matrix.tolist()
    return verified_matrices


def train_and_save_hybrid_bundle() -> Path:
    """Recreate Stage 2's capped development fit and save verified estimators."""
    expected_metrics = _read_stage2_metrics()
    expected_split = expected_metrics.get("split", {})
    if (
        expected_split.get("holdout_seed") != HOLDOUT_SEED
        or expected_split.get("holdout_size") != HOLDOUT_SIZE
        or expected_split.get("training_sample_seed") != TRAIN_SAMPLE_SEED
        or expected_split.get("max_training_rows_per_resume_label")
        != MAX_TRAIN_ROWS_PER_RESUME_LABEL
    ):
        raise RuntimeError(
            "Current training configuration does not match the saved Stage 2 protocol."
        )

    development, development_features, labels, split_data = _prepare_dataset()
    holdout = split_data.pop("holdout")
    holdout_features = split_data.pop("holdout_features")
    if split_data["holdout_group_overlap"] != 0:
        raise AssertionError("Final holdout contains overlapping resume groups.")
    for key in ("total_rows", "total_resume_groups", "development_rows",
                "development_resume_groups", "holdout_rows", "holdout_resume_groups"):
        if split_data[key] != expected_split[key]:
            raise RuntimeError(
                f"Recreated Stage 2 split differs for {key}: "
                f"{split_data[key]} != {expected_split[key]}."
            )

    sampled_development = sample_group_label_rows(
        development,
        max_rows_per_group_label=MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        random_state=TRAIN_SAMPLE_SEED,
    )
    sampled_features = development_features.loc[sampled_development.index]
    expected_sample_count = expected_split.get("final_training_sample_rows")
    if len(sampled_development) != expected_sample_count:
        raise RuntimeError(
            "Capped training sample size differs from Stage 2 metrics: "
            f"{len(sampled_development)} != {expected_sample_count}."
        )
    print(
        f"[TRAIN] Fitting on {len(sampled_development)} capped development rows "
        f"from {sampled_development['resume_group'].nunique()} resume groups."
    )
    estimators = fit_hybrid_estimators(
        sampled_development,
        sampled_features,
        labels,
        oof_splits=INNER_FOLDS,
        regularization=REGULARIZATION,
    )
    print("[VERIFY] Comparing reproduced holdout confusion matrices...")
    verified_matrices = _verify_against_stage2(
        estimators,
        holdout,
        holdout_features,
        labels,
        expected_metrics,
    )
    print("[VERIFY] Text, structured-only, and hybrid matrices match Stage 2.")

    training_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_size": HOLDOUT_SIZE,
        "training_sample_seed": TRAIN_SAMPLE_SEED,
        "max_training_rows_per_resume_label": MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        "inner_oof_folds": INNER_FOLDS,
        "regularization_C": REGULARIZATION,
        "class_weight": config.CLASS_WEIGHT,
        "logreg_max_iter": config.LOGREG_MAX_ITER,
        "tfidf_max_features": config.TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
        "tfidf_min_df": config.TFIDF_MIN_DF,
        "tfidf_max_df": config.TFIDF_MAX_DF,
        "tfidf_sublinear_tf": config.TFIDF_SUBLINEAR_TF,
        "total_rows": split_data["total_rows"],
        "total_resume_groups": split_data["total_resume_groups"],
        "development_rows": split_data["development_rows"],
        "development_resume_groups": split_data["development_resume_groups"],
        "holdout_rows": split_data["holdout_rows"],
        "holdout_resume_groups": split_data["holdout_resume_groups"],
        "holdout_group_overlap": split_data["holdout_group_overlap"],
        "sampled_development_rows": len(sampled_development),
        "sampled_development_resume_groups": int(
            sampled_development["resume_group"].nunique()
        ),
        "training_sampled": True,
        "experimental_only": True,
        "stage2_holdout_confusion_matrices_verified": True,
        "verified_confusion_matrices": verified_matrices,
        "dataset_name": config.HF_DATASET_NAME,
        "scikit_learn_version": sklearn.__version__,
    }
    bundle = create_hybrid_bundle(estimators, training_config)
    bundle_path = save_hybrid_bundle(bundle, HYBRID_BUNDLE_PATH)
    README_PATH.write_text(
        "# Experimental SmartHire Hybrid Model\n\n"
        "This bundle is for offline experimental inference only. It is not "
        "integrated into FastAPI or the production prediction path.\n\n"
        "The text, structured, and fusion estimators were reproduced using "
        "the Stage 2 grouped split and capped development sample (maximum "
        "five rows per resume-group/label combination). Hybrid fusion was "
        "trained on grouped out-of-fold text probabilities. The reproduced "
        "holdout confusion matrices were checked against the saved Stage 2 "
        "aggregate metrics before this bundle was saved.\n\n"
        "Keyword-derived features are heuristic and do not establish "
        "candidate competence. The model must not be used as an employment "
        "decision without further independent validation and appropriate "
        "human review.\n\n"
        "Load through `src.hybrid_predict.predict_hybrid`; bundle metadata "
        "includes feature ordering, labels, configuration, and versions.\n",
        encoding="utf-8",
    )
    print(f"[SAVED] Experimental inference bundle: {bundle_path}")
    print(f"[SAVED] Experimental model notes: {README_PATH}")
    return bundle_path


if __name__ == "__main__":
    train_and_save_hybrid_bundle()
