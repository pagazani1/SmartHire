import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

import config
from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "evaluation_results"


def _report_evaluation(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    reports_file,
) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report_data = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    matrix_path = RESULTS_DIR / f"confusion_matrix_{name}.csv"
    matrix_df.to_csv(matrix_path)

    print(f"\n{name.replace('_', ' ').title()}")
    print(f"Accuracy:   {accuracy:.6f}")
    print(f"Macro-F1:   {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")
    print("Classification report:")
    print(report_text)
    print(f"Confusion matrix (rows=actual, columns=predicted):\n{matrix_df}")
    print(f"Saved confusion matrix: {matrix_path}")

    reports_file.write(f"\n{name.replace('_', ' ').title()}\n")
    reports_file.write(f"Accuracy: {accuracy:.6f}\n")
    reports_file.write(f"Macro-F1: {macro_f1:.6f}\n")
    reports_file.write(f"Weighted-F1: {weighted_f1:.6f}\n")
    reports_file.write("Classification report:\n")
    reports_file.write(report_text)
    reports_file.write(f"\nConfusion matrix (rows=actual, columns=predicted):\n{matrix_df}\n")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_data,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
    }


def run_unseen_resume_evaluation() -> None:
    required_columns = {"resume", "jd", "label"}
    print("[LOAD] Reading raw train and test Parquet files...")
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
    print(f"[LOAD] Combined raw data shape: {data.shape}")
    print("[LOAD] Missing values before handling:")
    print(data[["resume", "jd", "label"]].isna().sum().to_string())

    missing_labels = int(data["label"].isna().sum())
    if missing_labels:
        print(f"[DATA] Dropping {missing_labels} rows with missing labels.")
        data = data.loc[data["label"].notna()].copy()
    if data.empty:
        raise ValueError("No rows with valid labels remain after loading the data.")

    missing_resumes = int(data["resume"].isna().sum())
    missing_jds = int(data["jd"].isna().sum())
    if missing_resumes or missing_jds:
        print(
            "[DATA] Replacing missing resume/JD text with empty strings: "
            f"resume={missing_resumes}, jd={missing_jds}"
        )
    data["resume"] = data["resume"].fillna("").astype(str)
    data["jd"] = data["jd"].fillna("").astype(str)

    data["resume_group"] = data["resume"].map(clean_text)
    data["jd_clean"] = data["jd"].map(clean_text)
    data["combined_text"] = data["resume_group"] + " " + data["jd_clean"]

    labels = sorted(data["label"].astype(str).unique().tolist())
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        splitter.split(data, data["label"], groups=data["resume_group"])
    )
    train_df = data.iloc[train_indices].copy()
    test_df = data.iloc[test_indices].copy()

    train_groups = set(train_df["resume_group"])
    test_groups = set(test_df["resume_group"])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Resume groups overlap between splits: {len(overlap)}"

    print(
        "[SPLIT] "
        f"Train rows={len(train_df)}, test rows={len(test_df)}; "
        f"train resume groups={len(train_groups)}, "
        f"test resume groups={len(test_groups)}, overlap={len(overlap)}"
    )

    y_train = train_df["label"]
    y_test = test_df["label"]
    train_distribution = y_train.value_counts().reindex(labels, fill_value=0)
    test_distribution = y_test.value_counts().reindex(labels, fill_value=0)
    distribution_df = pd.DataFrame(
        {
            "train_count": train_distribution,
            "train_fraction": train_distribution / len(y_train),
            "test_count": test_distribution,
            "test_fraction": test_distribution / len(y_test),
        }
    )
    print("\n[DATA] Class distributions (counts and fractions):")
    print(distribution_df.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    distribution_path = RESULTS_DIR / "class_distributions.csv"
    distribution_df.to_csv(distribution_path, index_label="label")

    print("\n[TF-IDF] Fitting a fresh vectorizer on training text only...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )
    X_train = vectorizer.fit_transform(train_df["combined_text"])
    X_test = vectorizer.transform(test_df["combined_text"])
    print(f"[TF-IDF] X_train={X_train.shape}, X_test={X_test.shape}")

    results = {
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_resume_groups": len(train_groups),
            "test_resume_groups": len(test_groups),
            "resume_group_overlap": len(overlap),
        },
        "class_distributions": {
            str(label): {
                "train_count": int(distribution_df.loc[label, "train_count"]),
                "train_fraction": float(distribution_df.loc[label, "train_fraction"]),
                "test_count": int(distribution_df.loc[label, "test_count"]),
                "test_fraction": float(distribution_df.loc[label, "test_fraction"]),
            }
            for label in labels
        },
        "vectorizer": {
            "max_features": config.TFIDF_MAX_FEATURES,
            "ngram_range": list(config.TFIDF_NGRAM_RANGE),
            "min_df": config.TFIDF_MIN_DF,
            "max_df": config.TFIDF_MAX_DF,
            "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            "fitted_features": X_train.shape[1],
        },
        "models": {},
    }

    reports_path = RESULTS_DIR / "classification_reports.txt"
    with reports_path.open("w", encoding="utf-8") as reports_file:
        majority_label = y_train.mode().iloc[0]
        majority_predictions = pd.Series(majority_label, index=y_test.index)
        results["models"]["majority_baseline"] = _report_evaluation(
            "majority_baseline", y_test, majority_predictions, labels, reports_file
        )
        print(f"[BASELINE] Majority class selected from training labels: {majority_label}")

        logistic_regression = LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER,
            class_weight=config.CLASS_WEIGHT,
            random_state=config.RANDOM_STATE,
        )
        print("\n[TRAIN] Fitting fresh Logistic Regression...")
        logistic_regression.fit(X_train, y_train)
        logistic_regression_predictions = logistic_regression.predict(X_test)
        logistic_regression_probabilities = logistic_regression.predict_proba(X_test)
        results["models"]["logistic_regression"] = _report_evaluation(
            "logistic_regression",
            y_test,
            pd.Series(logistic_regression_predictions, index=y_test.index),
            labels,
            reports_file,
        )

        linear_svc = LinearSVC(
            class_weight=config.CLASS_WEIGHT,
            random_state=config.RANDOM_STATE,
        )
        print("\n[TRAIN] Fitting fresh LinearSVC...")
        linear_svc.fit(X_train, y_train)
        linear_svc_predictions = linear_svc.predict(X_test)
        results["models"]["linear_svc"] = _report_evaluation(
            "linear_svc",
            y_test,
            pd.Series(linear_svc_predictions, index=y_test.index),
            labels,
            reports_file,
        )

        predictions_df = pd.DataFrame(
            {
                "anonymous_row_id": [
                    f"test_{row_number:06d}"
                    for row_number in range(1, len(test_df) + 1)
                ],
                "actual_label": y_test.to_numpy(),
                "logistic_regression_predicted_label": logistic_regression_predictions,
                "linear_svc_predicted_label": linear_svc_predictions,
            },
            index=y_test.index,
        )
        for class_index, class_label in enumerate(logistic_regression.classes_):
            predictions_df[
                f"logistic_regression_probability_{class_label}"
            ] = logistic_regression_probabilities[:, class_index]

        predictions_path = RESULTS_DIR / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

    metrics_path = RESULTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    print("\n[SAVED] Evaluation outputs:")
    print(f"  Class distributions: {distribution_path}")
    print(f"  Classification reports: {reports_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Row-level predictions: {predictions_path}")
    for model_name in results["models"]:
        print(f"  Confusion matrix: {RESULTS_DIR / f'confusion_matrix_{model_name}.csv'}")


if __name__ == "__main__":
    run_unseen_resume_evaluation()
