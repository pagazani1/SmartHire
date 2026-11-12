"""Compare resume/JD representations with grouped train/validation/holdout data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.preprocessing import clean_text
from src.structured_classifier import (
    build_combined_text,
    build_hybrid_fusion_features,
    extract_structured_features,
    fit_hybrid_estimators,
    fit_text_logistic_regression,
    make_structured_classifier,
    predict_text_probabilities,
    sample_group_label_rows,
)


LABELS = ["Good Fit", "No Fit", "Potential Fit"]
HOLDOUT_SEED = 2026
VALIDATION_SEED = 2027
TRAINING_SAMPLE_SEED = 6842
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION_OF_DEVELOPMENT = 0.25
MAX_ROWS_PER_RESUME_LABEL = 5
STACKING_FOLDS = 3
REGULARIZATION = 1.0


def _read_data() -> pd.DataFrame:
    columns = ["resume", "jd", "label", "resume_domain", "jd_domain"]
    train = pd.read_parquet(config.RAW_TRAIN_PATH, columns=columns)
    test = pd.read_parquet(config.RAW_TEST_PATH, columns=columns)
    data = pd.concat([train, test], ignore_index=True)
    data = data.loc[data["label"].notna()].copy().reset_index(drop=True)
    data["label"] = data["label"].astype(str)
    data["resume"] = data["resume"].fillna("").astype(str)
    data["jd"] = data["jd"].fillna("").astype(str)
    data["resume_group"] = data["resume"].map(clean_text)
    data["resume_clean"] = data["resume"].map(clean_text)
    data["jd_clean"] = data["jd"].map(clean_text)
    data["combined_text"] = [
        build_combined_text(resume, jd)
        for resume, jd in zip(data["resume"], data["jd"])
    ]
    return data


def _split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_FRACTION,
        random_state=HOLDOUT_SEED,
    )
    development_indices, holdout_indices = next(
        outer.split(data, data["label"], groups=data["resume_group"])
    )
    development = data.iloc[development_indices].copy()
    holdout = data.iloc[holdout_indices].copy()

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION_OF_DEVELOPMENT,
        random_state=VALIDATION_SEED,
    )
    train_indices, validation_indices = next(
        inner.split(
            development,
            development["label"],
            groups=development["resume_group"],
        )
    )
    train = development.iloc[train_indices].copy()
    validation = development.iloc[validation_indices].copy()

    group_sets = [
        set(frame["resume_group"])
        for frame in (train, validation, holdout)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise AssertionError("Resume-group overlap exists between data splits.")
    if set(data["label"].unique()) != set(LABELS):
        raise ValueError("The dataset does not contain the expected three labels.")
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("holdout", holdout),
    ):
        if set(frame["label"].unique()) != set(LABELS):
            raise ValueError(f"{name} split does not contain all three labels.")
    return train, validation, holdout


def _cap_training_rows(data: pd.DataFrame) -> pd.DataFrame:
    return sample_group_label_rows(
        data,
        max_rows_per_group_label=MAX_ROWS_PER_RESUME_LABEL,
        random_state=TRAINING_SAMPLE_SEED,
    )


def _make_separate_text_pipeline() -> Pipeline:
    vectorizer_options = {
        "max_features": config.TFIDF_MAX_FEATURES,
        "ngram_range": config.TFIDF_NGRAM_RANGE,
        "min_df": config.TFIDF_MIN_DF,
        "max_df": config.TFIDF_MAX_DF,
        "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
    }
    representation = ColumnTransformer(
        transformers=[
            (
                "resume_tfidf",
                TfidfVectorizer(**vectorizer_options),
                "resume_clean",
            ),
            (
                "jd_tfidf",
                TfidfVectorizer(**vectorizer_options),
                "jd_clean",
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline(
        steps=[("representation", representation), ("classifier", classifier)]
    )


def _fit_hybrid(
    training: pd.DataFrame,
    structured: pd.DataFrame,
) -> dict:
    return fit_hybrid_estimators(
        training,
        structured.loc[training.index],
        LABELS,
        oof_splits=STACKING_FOLDS,
        regularization=REGULARIZATION,
    )


def _predict_hybrid(
    estimators: dict,
    rows: pd.DataFrame,
    structured: pd.DataFrame,
) -> np.ndarray:
    probabilities = predict_text_probabilities(
        estimators["text_vectorizer"],
        estimators["text_model"],
        rows["combined_text"].to_numpy(),
        LABELS,
    )
    fusion_features = build_hybrid_fusion_features(
        structured.loc[rows.index],
        probabilities,
        LABELS,
    )
    return estimators["hybrid_model"].predict(fusion_features)


def _print_metrics(
    name: str,
    rows: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    truth = rows["label"].to_numpy()
    report = classification_report(
        truth,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    print(f"\n[{name}]")
    print(f"rows={len(rows)}")
    print(f"accuracy={accuracy_score(truth, predictions):.6f}")
    print(
        "macro_f1="
        f"{f1_score(truth, predictions, labels=LABELS, average='macro', zero_division=0):.6f}"
    )
    print(
        "weighted_f1="
        f"{f1_score(truth, predictions, labels=LABELS, average='weighted', zero_division=0):.6f}"
    )
    for label in LABELS:
        scores = report[label]
        print(
            f"{label}: precision={scores['precision']:.6f} "
            f"recall={scores['recall']:.6f} f1={scores['f1-score']:.6f} "
            f"support={int(scores['support'])}"
        )
    print("confusion_matrix_labels=" + repr(LABELS))
    print("confusion_matrix=" + repr(confusion_matrix(truth, predictions, labels=LABELS).tolist()))


def main() -> None:
    data = _read_data()
    train, validation, holdout = _split_data(data)
    train_sample = _cap_training_rows(train)
    development_sample = _cap_training_rows(
        pd.concat([train, validation], axis=0).sort_index()
    )
    labels = train_sample["label"].to_numpy()
    if set(labels) != set(LABELS) or set(development_sample["label"]) != set(LABELS):
        raise ValueError("Capped training sample does not contain all three labels.")

    print("Grouped split and fixed experiment configuration")
    print(
        f"all_rows={len(data)} all_resume_groups={data['resume_group'].nunique()} "
        f"train_rows={len(train)} train_groups={train['resume_group'].nunique()} "
        f"train_sample_rows={len(train_sample)} "
        f"validation_rows={len(validation)} "
        f"validation_groups={validation['resume_group'].nunique()} "
        f"holdout_rows={len(holdout)} "
        f"holdout_groups={holdout['resume_group'].nunique()}"
    )
    print("class_order=" + repr(LABELS))
    print("training_row_cap_per_resume_label=" + str(MAX_ROWS_PER_RESUME_LABEL))
    print("holdout_used_for_model_selection=False")
    print("structured_feature_extraction=deterministic_existing_skill_rules")

    structured = extract_structured_features(data)

    print("\nFitting candidates on the capped training split for validation.")
    combined_vectorizer, combined_model = fit_text_logistic_regression(
        train_sample["combined_text"].to_numpy(), labels
    )
    separate_model = _make_separate_text_pipeline()
    separate_model.fit(train_sample, labels)
    structured_model = make_structured_classifier(
        regularization=REGULARIZATION
    )
    structured_model.fit(structured.loc[train_sample.index], labels)
    hybrid_train = _fit_hybrid(train_sample, structured)

    validation_predictions = {
        "A_production_combined_tfidf_lr_validation": combined_model.predict(
            combined_vectorizer.transform(validation["combined_text"].to_numpy())
        ),
        "B_separate_resume_jd_tfidf_lr_validation": separate_model.predict(validation),
        "C_structured_skill_domain_lr_validation": structured_model.predict(
            structured.loc[validation.index]
        ),
        "D_oof_text_plus_structured_hybrid_validation": _predict_hybrid(
            hybrid_train, validation, structured
        ),
    }
    for name, predictions in validation_predictions.items():
        _print_metrics(name, validation, predictions)

    print("\nRefitting the fixed candidates on capped train+validation data.")
    combined_vectorizer, combined_model = fit_text_logistic_regression(
        development_sample["combined_text"].to_numpy(),
        development_sample["label"].to_numpy(),
    )
    separate_model = _make_separate_text_pipeline()
    separate_model.fit(
        development_sample,
        development_sample["label"].to_numpy(),
    )
    structured_model = make_structured_classifier(
        regularization=REGULARIZATION
    )
    structured_model.fit(
        structured.loc[development_sample.index],
        development_sample["label"].to_numpy(),
    )
    hybrid_development = _fit_hybrid(development_sample, structured)

    holdout_predictions = {
        "A_production_combined_tfidf_lr_holdout": combined_model.predict(
            combined_vectorizer.transform(holdout["combined_text"].to_numpy())
        ),
        "B_separate_resume_jd_tfidf_lr_holdout": separate_model.predict(holdout),
        "C_structured_skill_domain_lr_holdout": structured_model.predict(
            structured.loc[holdout.index]
        ),
        "D_oof_text_plus_structured_hybrid_holdout": _predict_hybrid(
            hybrid_development, holdout, structured
        ),
    }
    for name, predictions in holdout_predictions.items():
        _print_metrics(name, holdout, predictions)


if __name__ == "__main__":
    main()
