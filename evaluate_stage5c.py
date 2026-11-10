"""Run controlled Stage 5C grouped experiments and save isolated artifacts."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    build_combined_text,
    build_hybrid_fusion_features,
    extract_structured_features,
    make_structured_classifier,
    sample_group_label_rows,
)
from src.skill_extractor import extract_jd_skills


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "evaluation_results" / "structured" / "structured_metrics.json"
ARTIFACT_DIR = config.MODELS_DIR / "stage5c_experimental"
REPORT_PATH = ROOT / "evaluation_results" / "stage5c_model_comparison.md"
EXPERIMENT_ID = "smarthire-stage5c-separate-text-skill-fusion-v1"

HOLDOUT_SEED = 2026
HOLDOUT_SIZE = 0.2
OUTER_FOLDS = 3
INNER_FOLDS = 3
MAX_TRAIN_ROWS_PER_RESUME_LABEL = 5
TRAIN_SAMPLE_SEED = 6842
REGULARIZATION = 1.0
STREAM_BATCH_SIZE = 1024
MIN_FREE_RAM_BYTES = 256 * 1024 * 1024

INPUT_COLUMNS = ["resume", "jd", "label", "resume_domain", "jd_domain", "source"]
MODEL_NAMES = (
    "text_baseline",
    "separate_resume_jd",
    "separate_with_interaction",
    "structured_skill",
    "combined_fusion",
)

STAGE5C_EXTRA_NUMERIC_COLUMNS = [
    "resume_jd_skill_overlap_ratio",
    "resume_jd_skill_overlap_available",
    "resume_jd_skill_jaccard",
    "resume_jd_skill_jaccard_available",
    "required_alternative_group_count",
    "preferred_alternative_group_count",
]
STAGE5C_NUMERIC_COLUMNS = [
    *NUMERIC_FEATURE_COLUMNS,
    *STAGE5C_EXTRA_NUMERIC_COLUMNS,
]
class Stage5CResourceLimitError(RuntimeError):
    """Raised when continuing an experiment would risk exhausting memory."""


_RESOURCE_STATE: dict[str, int] = {
    "peak_observed_rss_bytes": 0,
    "minimum_observed_available_ram_bytes": 2**63 - 1,
}


def _observe_resources(*, enforce_limit: bool = True) -> None:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    _RESOURCE_STATE["peak_observed_rss_bytes"] = max(
        _RESOURCE_STATE["peak_observed_rss_bytes"],
        rss,
    )
    _RESOURCE_STATE["minimum_observed_available_ram_bytes"] = min(
        _RESOURCE_STATE["minimum_observed_available_ram_bytes"],
        available,
    )
    if enforce_limit and available < MIN_FREE_RAM_BYTES:
        raise Stage5CResourceLimitError(
            "Available system memory fell below the 256 MiB safety floor."
        )


def _read_reference_metrics() -> dict[str, Any]:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Reference Stage 2 metrics are required: {METRICS_PATH}"
        ) from exc


def _extract_stage5c_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic overlap/alternative features to existing skill features."""
    features = extract_structured_features(frame)
    jd_cache = {
        jd: extract_jd_skills(jd)
        for jd in frame["jd"].fillna("").astype(str).unique()
    }
    alternative_counts: list[tuple[int, int]] = []
    for jd in frame["jd"].fillna("").astype(str):
        analysis = jd_cache[jd]
        alternative_counts.append(
            (
                sum(len(group) > 1 for group in analysis["required_skill_groups"]),
                sum(len(group) > 1 for group in analysis["preferred_skill_groups"]),
            )
        )

    required_skill_counts = features["jd_skill_count"].to_numpy(dtype=float)
    resume_skill_counts = features["resume_skill_count"].to_numpy(dtype=float)
    overlap_counts = features["matched_skill_count"].to_numpy(dtype=float)
    union_counts = resume_skill_counts + required_skill_counts - overlap_counts
    has_jd_skills = required_skill_counts > 0
    has_skill_union = union_counts > 0
    features["resume_jd_skill_overlap_ratio"] = np.divide(
        overlap_counts,
        required_skill_counts,
        out=np.full(len(features), np.nan),
        where=has_jd_skills,
    )
    features["resume_jd_skill_overlap_available"] = has_jd_skills.astype(int)
    features["resume_jd_skill_jaccard"] = np.divide(
        overlap_counts,
        union_counts,
        out=np.full(len(features), np.nan),
        where=has_skill_union,
    )
    features["resume_jd_skill_jaccard_available"] = has_skill_union.astype(int)
    features["required_alternative_group_count"] = [
        item[0] for item in alternative_counts
    ]
    features["preferred_alternative_group_count"] = [
        item[1] for item in alternative_counts
    ]
    return features


def _scan_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream source rows, group IDs and deterministic features without full text loading."""
    feature_frames: list[pd.DataFrame] = []
    labels: list[str] = []
    group_values: list[str] = []
    active_row_ids: list[int] = []
    sources: list[str] = []
    group_intern: dict[str, str] = {}
    global_row_id = 0
    active_position = 0
    parquet_paths = (config.RAW_TRAIN_PATH, config.RAW_TEST_PATH)
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(
            columns=INPUT_COLUMNS,
            batch_size=STREAM_BATCH_SIZE,
        ):
            _observe_resources()
            frame = record_batch.to_pandas()
            row_ids = np.arange(
                global_row_id,
                global_row_id + len(frame),
                dtype=np.int64,
            )
            global_row_id += len(frame)
            valid = frame["label"].notna().to_numpy()
            if not valid.any():
                continue
            active = frame.loc[valid].copy()
            normalized_resumes = active["resume"].fillna("").astype(str).map(clean_text)
            canonical_groups = [
                group_intern.setdefault(text, text)
                for text in normalized_resumes
            ]
            group_values.extend(canonical_groups)
            active_row_ids.extend(row_ids[valid].tolist())
            labels.extend(active["label"].astype(str).tolist())
            sources.extend(active["source"].fillna("__missing__").astype(str).tolist())

            active_features = _extract_stage5c_features(active)
            active_features.index = np.arange(
                active_position,
                active_position + len(active_features),
                dtype=np.int64,
            )
            active_position += len(active_features)
            feature_frames.append(active_features)
            del frame, active, normalized_resumes, active_features, record_batch
        del parquet
        gc.collect()

    sorted_group_values = sorted(group_intern)
    group_to_code = {group: index for index, group in enumerate(sorted_group_values)}
    group_codes = np.fromiter(
        (group_to_code[group] for group in group_values),
        dtype=np.int32,
        count=len(group_values),
    )
    feature_frame = pd.concat(feature_frames, axis=0).sort_index()
    if len(feature_frame) != len(labels):
        raise AssertionError("Streamed structured features do not align with labels.")
    metadata = {
        "rows": len(labels),
        "resume_groups": len(sorted_group_values),
        "labels": sorted(set(labels)),
        "source_categories": sorted(set(sources)),
        "source_label_counts": (
            pd.crosstab(pd.Series(sources, name="source"), pd.Series(labels, name="label"))
            .reindex(columns=sorted(set(labels)), fill_value=0)
            .to_dict(orient="index")
        ),
    }
    return (
        feature_frame,
        np.asarray(labels, dtype=object),
        group_codes,
        np.asarray(active_row_ids, dtype=np.int64),
        metadata,
    )


def _load_rows(global_row_ids: np.ndarray) -> pd.DataFrame:
    wanted = set(int(value) for value in global_row_ids)
    found_frames: list[pd.DataFrame] = []
    global_row_id = 0
    for path in (config.RAW_TRAIN_PATH, config.RAW_TEST_PATH):
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(
            columns=INPUT_COLUMNS,
            batch_size=STREAM_BATCH_SIZE,
        ):
            _observe_resources()
            row_ids = np.arange(
                global_row_id,
                global_row_id + record_batch.num_rows,
                dtype=np.int64,
            )
            mask = np.fromiter(
                (int(value) in wanted for value in row_ids),
                dtype=bool,
                count=len(row_ids),
            )
            if mask.any():
                selected = record_batch.filter(pa.array(mask))
                part = selected.to_pandas()
                part["__row_id"] = row_ids[mask]
                found_frames.append(part)
                del selected, part
            global_row_id += record_batch.num_rows
            del record_batch
        del parquet
    if not found_frames:
        return pd.DataFrame(columns=[*INPUT_COLUMNS, "__row_id"])
    result = pd.concat(found_frames, ignore_index=True).sort_values("__row_id")
    if len(result) != len(wanted):
        raise RuntimeError(
            f"Requested {len(wanted)} source rows, but loaded {len(result)}."
        )
    return result.reset_index(drop=True)


def _iter_selected_batches(
    wanted_global_row_ids: np.ndarray,
) -> Iterator[pd.DataFrame]:
    wanted = set(int(value) for value in wanted_global_row_ids)
    global_row_id = 0
    for path in (config.RAW_TRAIN_PATH, config.RAW_TEST_PATH):
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(
            columns=INPUT_COLUMNS,
            batch_size=STREAM_BATCH_SIZE,
        ):
            _observe_resources()
            row_ids = np.arange(
                global_row_id,
                global_row_id + record_batch.num_rows,
                dtype=np.int64,
            )
            mask = np.fromiter(
                (int(value) in wanted for value in row_ids),
                dtype=bool,
                count=len(row_ids),
            )
            if mask.any():
                selected = record_batch.filter(pa.array(mask))
                part = selected.to_pandas()
                part["__row_id"] = row_ids[mask]
                yield part
                del selected, part
            global_row_id += record_batch.num_rows
            del record_batch
        del parquet


def _make_pair_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )


def _pair_training_matrices(
    vectorizer: TfidfVectorizer,
    resumes: list[str],
    jds: list[str],
    *,
    fit: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    normalized_resumes = [clean_text(value) for value in resumes]
    normalized_jds = [clean_text(value) for value in jds]
    if fit:
        vectorizer.fit([*normalized_resumes, *normalized_jds])
    resume_matrix = vectorizer.transform(normalized_resumes).tocsr()
    jd_matrix = vectorizer.transform(normalized_jds).tocsr()
    interaction = resume_matrix.multiply(jd_matrix).tocsr()
    cosine_similarity = np.asarray(interaction.sum(axis=1)).reshape(-1, 1)
    resume_jd_cosine = sparse.csr_matrix(cosine_similarity)
    separate = sparse.hstack(
        [resume_matrix, jd_matrix],
        format="csr",
    )
    separate_with_interaction = sparse.hstack(
        [resume_matrix, jd_matrix, interaction, resume_jd_cosine],
        format="csr",
    )
    return resume_matrix, jd_matrix, separate, separate_with_interaction


def _class_probabilities(
    model: LogisticRegression,
    features: sparse.spmatrix,
    labels: list[str],
) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((features.shape[0], len(labels)), dtype=float)
    label_indexes = {label: index for index, label in enumerate(labels)}
    for model_column, label in enumerate(model.classes_):
        aligned[:, label_indexes[str(label)]] = raw[:, model_column]
    return aligned


def _fit_pair_models(
    training_rows: pd.DataFrame,
    labels: list[str],
) -> tuple[TfidfVectorizer, LogisticRegression, LogisticRegression]:
    vectorizer = _make_pair_vectorizer()
    _, _, separate, interaction = _pair_training_matrices(
        vectorizer,
        training_rows["resume"].fillna("").astype(str).tolist(),
        training_rows["jd"].fillna("").astype(str).tolist(),
        fit=True,
    )
    target = training_rows["label"].astype(str).to_numpy()
    baseline = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    baseline.fit(separate, target)
    del separate
    _observe_resources()
    interaction_model = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    interaction_model.fit(interaction, target)
    del interaction
    _observe_resources()
    if set(str(value) for value in baseline.classes_) != set(labels):
        raise ValueError("Separate text classifier is missing one or more labels.")
    return vectorizer, baseline, interaction_model


def _generate_pair_oof_probabilities(
    training_rows: pd.DataFrame,
    group_codes: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Generate pair-text probabilities with each row scored outside its group."""
    target = training_rows["label"].astype(str).to_numpy()
    output = np.full((len(training_rows), len(labels)), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(training_rows)),
        target,
        groups=group_codes,
    ):
        if set(group_codes[train_indices]).intersection(group_codes[validation_indices]):
            raise AssertionError("Resume-group overlap in Stage 5C fusion OOF.")
        fold_train = training_rows.iloc[train_indices]
        vectorizer = _make_pair_vectorizer()
        _, _, _, train_features = _pair_training_matrices(
            vectorizer,
            fold_train["resume"].fillna("").astype(str).tolist(),
            fold_train["jd"].fillna("").astype(str).tolist(),
            fit=True,
        )
        model = LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER,
            class_weight=config.CLASS_WEIGHT,
            random_state=config.RANDOM_STATE,
        )
        model.fit(train_features, target[train_indices])
        validation = training_rows.iloc[validation_indices]
        _, _, _, validation_features = _pair_training_matrices(
            vectorizer,
            validation["resume"].fillna("").astype(str).tolist(),
            validation["jd"].fillna("").astype(str).tolist(),
            fit=False,
        )
        output[validation_indices] = _class_probabilities(
            model,
            validation_features,
            labels,
        )
        del vectorizer, model, train_features, validation_features
        gc.collect()
        _observe_resources()
    if not np.isfinite(output).all():
        raise AssertionError("Stage 5C fusion OOF probabilities are incomplete.")
    return output


def _evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    report = classification_report(
        truth,
        prediction,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
                "predicted": int(np.count_nonzero(prediction == label)),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }


def _add_prediction_batch(
    *,
    rows: pd.DataFrame,
    all_features: pd.DataFrame,
    global_to_active: dict[int, int],
    validation_prediction_offsets: dict[int, int],
    baseline_vectorizer: TfidfVectorizer,
    baseline_model: LogisticRegression,
    pair_vectorizer: TfidfVectorizer,
    pair_baseline_model: LogisticRegression,
    pair_interaction_model: LogisticRegression,
    structured_model: Any,
    fusion_model: Any,
    labels: list[str],
    predictions: dict[str, np.ndarray],
) -> None:
    global_ids = rows["__row_id"].to_numpy(dtype=np.int64)
    active_positions = np.fromiter(
        (global_to_active[int(row_id)] for row_id in global_ids),
        dtype=np.int64,
        count=len(global_ids),
    )
    offsets = np.fromiter(
        (validation_prediction_offsets[int(position)] for position in active_positions),
        dtype=np.int64,
        count=len(active_positions),
    )
    combined_texts = [
        build_combined_text(resume, jd)
        for resume, jd in zip(rows["resume"], rows["jd"])
    ]
    baseline_features = baseline_vectorizer.transform(combined_texts)
    predictions["text_baseline"][offsets] = baseline_model.predict(baseline_features)
    del baseline_features, combined_texts

    _, _, separate_features, interaction_features = _pair_training_matrices(
        pair_vectorizer,
        rows["resume"].fillna("").astype(str).tolist(),
        rows["jd"].fillna("").astype(str).tolist(),
        fit=False,
    )
    predictions["separate_resume_jd"][offsets] = pair_baseline_model.predict(
        separate_features
    )
    pair_probabilities = _class_probabilities(
        pair_interaction_model,
        interaction_features,
        labels,
    )
    predictions["separate_with_interaction"][offsets] = (
        pair_interaction_model.predict(interaction_features)
    )

    structured_features = all_features.loc[active_positions]
    predictions["structured_skill"][offsets] = structured_model.predict(
        structured_features
    )
    fusion_features = build_hybrid_fusion_features(
        structured_features,
        pair_probabilities,
        labels,
    )
    predictions["combined_fusion"][offsets] = fusion_model.predict(fusion_features)
    del (
        separate_features,
        interaction_features,
        pair_probabilities,
        structured_features,
        fusion_features,
    )
    _observe_resources()


def _evaluate_partition(
    *,
    active_positions: np.ndarray,
    active_row_ids: np.ndarray,
    labels_array: np.ndarray,
    all_features: pd.DataFrame,
    trained_models: dict[str, Any],
    labels: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    truth = labels_array[active_positions]
    predictions = {
        model_name: np.empty(len(active_positions), dtype=object)
        for model_name in MODEL_NAMES
    }
    global_to_active = {
        int(row_id): int(active_position)
        for active_position, row_id in enumerate(active_row_ids)
    }
    validation_prediction_offsets = {
        int(active_position): offset
        for offset, active_position in enumerate(active_positions)
    }
    wanted_global_ids = active_row_ids[active_positions]
    seen = 0
    for rows in _iter_selected_batches(wanted_global_ids):
        _add_prediction_batch(
            rows=rows,
            all_features=all_features,
            global_to_active=global_to_active,
            validation_prediction_offsets=validation_prediction_offsets,
            baseline_vectorizer=trained_models["baseline_vectorizer"],
            baseline_model=trained_models["baseline_model"],
            pair_vectorizer=trained_models["pair_vectorizer"],
            pair_baseline_model=trained_models["pair_baseline_model"],
            pair_interaction_model=trained_models["pair_interaction_model"],
            structured_model=trained_models["structured_model"],
            fusion_model=trained_models["fusion_model"],
            labels=labels,
            predictions=predictions,
        )
        seen += len(rows)
        del rows
    if seen != len(active_positions):
        raise RuntimeError(
            f"Scored {seen} rows, expected {len(active_positions)}."
        )
    results = {
        model_name: _evaluate(truth, predictions[model_name], labels)
        for model_name in MODEL_NAMES
    }
    return predictions, results


def _fit_all_models(
    *,
    sampled_rows: pd.DataFrame,
    sampled_features: pd.DataFrame,
    sampled_group_codes: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    _observe_resources()
    texts = [
        build_combined_text(resume, jd)
        for resume, jd in zip(sampled_rows["resume"], sampled_rows["jd"])
    ]
    baseline_vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )
    baseline_matrix = baseline_vectorizer.fit_transform(texts)
    baseline_model = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    baseline_model.fit(baseline_matrix, sampled_rows["label"].astype(str).to_numpy())
    del baseline_matrix, texts

    pair_vectorizer, pair_baseline_model, pair_interaction_model = _fit_pair_models(
        sampled_rows,
        labels,
    )
    structured_model = make_structured_classifier(
        numeric_columns=STAGE5C_NUMERIC_COLUMNS,
        regularization=REGULARIZATION,
    )
    structured_model.fit(sampled_features, sampled_rows["label"].astype(str))

    oof_probabilities = _generate_pair_oof_probabilities(
        sampled_rows,
        sampled_group_codes,
        labels,
    )
    fusion_training_features = build_hybrid_fusion_features(
        sampled_features,
        oof_probabilities,
        labels,
    )
    probability_columns = [f"text_probability__{label}" for label in labels]
    fusion_model = make_structured_classifier(
        numeric_columns=[*STAGE5C_NUMERIC_COLUMNS, *probability_columns],
        regularization=REGULARIZATION,
    )
    fusion_model.fit(
        fusion_training_features,
        sampled_rows["label"].astype(str),
    )
    del oof_probabilities, fusion_training_features
    _observe_resources()
    return {
        "baseline_vectorizer": baseline_vectorizer,
        "baseline_model": baseline_model,
        "pair_vectorizer": pair_vectorizer,
        "pair_baseline_model": pair_baseline_model,
        "pair_interaction_model": pair_interaction_model,
        "structured_model": structured_model,
        "fusion_model": fusion_model,
    }


def _write_report(
    *,
    status: str,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    lines = [
        "# Stage 5C: Controlled Resume–JD Matching Experiments",
        "",
        "## Status and methodology",
        "",
        f"Experiment status: **{status}**.",
        "",
        "The experiment uses the existing raw train/test Parquet inputs, "
        "the seed-2026 20% resume-group holdout, three grouped development "
        "folds, and the existing cap of five rows per resume-group/label "
        "combination. Sampling and all fitted transforms are restricted to "
        "the applicable development/training fold. The final holdout was "
        "scored once after the model protocol was fixed. No synthetic resume "
        "fixtures were used for fitting or selection.",
        "",
        "Compared models: existing concatenated text baseline (reference "
        "metrics from the matching saved Stage 2 run), separate resume/JD "
        "TF-IDF blocks, separate blocks plus token-overlap and cosine "
        "interaction, structured skill interaction features, and a combined "
        "fusion classifier trained on grouped OOF probabilities from the "
        "interaction text model.",
        "",
        "The pair vectorizer is fitted on resume and JD documents from the "
        "training partition only, creating a shared vocabulary. Resume and JD "
        "matrices remain separate. The interaction model additionally uses "
        "their element-wise TF-IDF overlap and cosine similarity. Structured "
        "features reuse the rule-based skill extractor and include required/"
        "preferred coverage, matched/missing counts, skill overlap ratios, "
        "Jaccard overlap, and alternative-group counts.",
        "",
        "Unavailable required/preferred coverage and overlap ratios remain "
        "NaN with explicit availability indicators; the fitted pipeline "
        "imputes only within each training partition.",
        "",
    ]
    if error:
        lines.extend(
            [
                "## Stopped experiment",
                "",
                f"The experiment stopped safely: {error}",
                "",
                "No final model bundle was saved. Any completed folds are "
                "reported only if present in the status metadata. No "
                "evaluation protocol or data split was altered.",
                "",
            ]
        )
    elif metrics is not None:
        _append_metrics_report(lines, metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _append_metrics_report(lines: list[str], metrics: dict[str, Any]) -> None:
    lines.extend(["## Split and resource measurements", ""])
    split = metrics["split"]
    lines.extend(
        [
            f"- Rows: {split['rows']:,}; resume groups: {split['resume_groups']:,}.",
            f"- Development: {split['development_rows']:,} rows / "
            f"{split['development_resume_groups']:,} groups; final holdout: "
            f"{split['holdout_rows']:,} rows / {split['holdout_resume_groups']:,} groups.",
            f"- Holdout overlap: {split['holdout_group_overlap']}; seed: "
            f"{HOLDOUT_SEED}; holdout fraction: {HOLDOUT_SIZE}.",
            f"- Capped development training rows: "
            f"{split['sampled_development_rows']:,}; cap: "
            f"{MAX_TRAIN_ROWS_PER_RESUME_LABEL} rows/group/label; sampling seed: "
            f"{TRAIN_SAMPLE_SEED}.",
            f"- Runtime: {metrics['runtime_seconds']:.1f} seconds.",
            f"- Peak observed process RSS: "
            f"{metrics['resources']['peak_observed_rss_bytes'] / (1024**3):.2f} GiB.",
            f"- Minimum observed available system RAM: "
            f"{metrics['resources']['minimum_observed_available_ram_bytes'] / (1024**3):.2f} GiB.",
            "",
            "## Development grouped cross-validation",
            "",
            "| Model | Accuracy mean ± SD | Macro-F1 mean ± SD | Weighted-F1 mean ± SD | Good Fit P/R/F1 | Potential Fit P/R/F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        summary = metrics["cv_summary"][model_name]
        good = summary["per_class"]["Good Fit"]
        potential = summary["per_class"]["Potential Fit"]
        lines.append(
            f"| {model_name} | {summary['accuracy']['mean']:.4f} ± "
            f"{summary['accuracy']['std']:.4f} | "
            f"{summary['macro_f1']['mean']:.4f} ± "
            f"{summary['macro_f1']['std']:.4f} | "
            f"{summary['weighted_f1']['mean']:.4f} ± "
            f"{summary['weighted_f1']['std']:.4f} | "
            f"{good['precision']['mean']:.3f}/{good['recall']['mean']:.3f}/"
            f"{good['f1']['mean']:.3f} | "
            f"{potential['precision']['mean']:.3f}/"
            f"{potential['recall']['mean']:.3f}/"
            f"{potential['f1']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Per-fold metrics, per-class scores, and confusion matrices are "
            "saved under `models/stage5c_experimental/`.",
            "",
            "## Single final grouped holdout",
            "",
            "| Model | Accuracy | Macro-F1 | Weighted-F1 | Good Fit P/R/F1 | Potential Fit P/R/F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        result = metrics["holdout_metrics"][model_name]
        good = result["per_class"]["Good Fit"]
        potential = result["per_class"]["Potential Fit"]
        lines.append(
            f"| {model_name} | {result['accuracy']:.4f} | "
            f"{result['macro_f1']:.4f} | {result['weighted_f1']:.4f} | "
            f"{good['precision']:.3f}/{good['recall']:.3f}/{good['f1']:.3f} | "
            f"{potential['precision']:.3f}/{potential['recall']:.3f}/"
            f"{potential['f1']:.3f} |"
        )
    lines.extend(["", "### Holdout per-class metrics", ""])
    for model_name in MODEL_NAMES:
        result = metrics["holdout_metrics"][model_name]
        lines.extend(
            [
                f"#### {model_name}",
                "",
                "| Class | Precision | Recall | F1 | Support | Predicted |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label in metrics["labels"]:
            score = result["per_class"][label]
            lines.append(
                f"| {label} | {score['precision']:.4f} | {score['recall']:.4f} | "
                f"{score['f1']:.4f} | {score['support']} | {score['predicted']} |"
            )
        lines.extend(
            [
                "",
                "Confusion matrix (rows=actual, columns=predicted; label order "
                f"{', '.join(metrics['labels'])}):",
                "",
                "```text",
                json.dumps(result["confusion_matrix"]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Ablations",
            "",
            "The separate-resume/JD model versus its interaction variant "
            "measures the value of adding token-wise overlap and cosine "
            "features. The structured-skill model versus combined fusion "
            "measures the added value of grouped OOF pair-text probabilities. "
            "The concatenated text baseline is the existing Stage 2 result "
            "on the same split and capped protocol.",
            "",
            "Do not interpret the single holdout as a tuning signal. Model "
            "selection should be based on the development grouped-fold "
            "results and a predeclared metric/guardrail policy.",
            "",
            "## Recommendation and limitations",
            "",
        ]
    )
    recommendation = metrics.get("recommendation", "No model selected.")
    lines.append(recommendation)
    lines.extend(
        [
            "",
            "The skill extractor is heuristic and keyword-based; it does not "
            "verify skill proficiency or candidate competence. No ground-truth "
            "skill annotations are available. Results are experimental and "
            "must not be used as automatic employment decisions.",
            "",
            "The development sample is capped and shifts row-level class "
            "prevalence. The final holdout is complete, grouped, and was not "
            "used to tune this experiment. Pairwise cosine similarity is "
            "lexical TF-IDF overlap, not semantic similarity.",
            "",
        ]
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def _summarize_cv(
    folds: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        records = [item["metrics"][model_name] for item in folds]
        summary[model_name] = {
            metric: _mean_std([record[metric] for record in records])
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        }
        summary[model_name]["per_class"] = {
            label: {
                metric: _mean_std(
                    [record["per_class"][label][metric] for record in records]
                )
                for metric in ("precision", "recall", "f1")
            }
            for label in folds[0]["metrics"][model_name]["per_class"]
        }
    return summary


def _run() -> dict[str, Any]:
    started = time.perf_counter()
    reference = _read_reference_metrics()
    expected_split = reference["split"]
    _observe_resources()

    features, labels_array, group_codes, active_row_ids, data_metadata = _scan_dataset()
    if len(active_row_ids) != len(labels_array):
        raise AssertionError("Row identifiers and streamed dataset labels do not align.")
    labels = sorted(set(str(value) for value in labels_array))
    if labels != sorted(reference["final_holdout_models"]["text_logistic_regression"]["confusion_matrix_labels"]):
        raise ValueError("Current raw dataset class labels differ from Stage 2.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_SIZE,
        random_state=HOLDOUT_SEED,
    )
    development_positions, holdout_positions = next(
        splitter.split(
            np.zeros(len(labels_array)),
            labels_array,
            groups=group_codes,
        )
    )
    development_groups = set(group_codes[development_positions])
    holdout_groups = set(group_codes[holdout_positions])
    group_overlap = development_groups.intersection(holdout_groups)
    if group_overlap:
        raise AssertionError("Resume group overlap in Stage 5C final holdout.")
    split = {
        "rows": len(labels_array),
        "resume_groups": int(len(np.unique(group_codes))),
        "development_rows": len(development_positions),
        "development_resume_groups": len(development_groups),
        "holdout_rows": len(holdout_positions),
        "holdout_resume_groups": len(holdout_groups),
        "holdout_group_overlap": len(group_overlap),
    }
    for key, expected_key in (
        ("rows", "total_rows"),
        ("resume_groups", "total_resume_groups"),
        ("development_rows", "development_rows"),
        ("development_resume_groups", "development_resume_groups"),
        ("holdout_rows", "holdout_rows"),
        ("holdout_resume_groups", "holdout_resume_groups"),
    ):
        if split[key] != expected_split[expected_key]:
            raise RuntimeError(
                f"Stage 5C split does not match Stage 2 for {key}: "
                f"{split[key]} != {expected_split[expected_key]}."
            )

    development_metadata = pd.DataFrame(
        {
            "resume_group": group_codes[development_positions],
            "label": labels_array[development_positions],
        },
        index=np.arange(len(development_positions)),
    )
    final_sample_local = sample_group_label_rows(
        development_metadata,
        MAX_TRAIN_ROWS_PER_RESUME_LABEL,
        TRAIN_SAMPLE_SEED,
    ).index.to_numpy(dtype=np.int64)
    if len(final_sample_local) != expected_split["final_training_sample_rows"]:
        raise RuntimeError("Stage 5C capped sample differs from reference size.")
    final_training_positions = development_positions[final_sample_local]
    if int(len(final_training_positions)) != int(expected_split["final_training_sample_rows"]):
        raise AssertionError("Unexpected Stage 5C final development sample size.")

    outer_splitter = GroupKFold(n_splits=OUTER_FOLDS)
    fold_records: list[dict[str, Any]] = []
    for fold_number, (train_local, validation_local) in enumerate(
        outer_splitter.split(
            development_positions,
            labels_array[development_positions],
            groups=group_codes[development_positions],
        ),
        start=1,
    ):
        _observe_resources()
        full_train_positions = development_positions[train_local]
        validation_positions = development_positions[validation_local]
        train_local_metadata = pd.DataFrame(
            {
                "resume_group": group_codes[full_train_positions],
                "label": labels_array[full_train_positions],
            },
            index=np.arange(len(train_local)),
        )
        sampled_fold_local = sample_group_label_rows(
            train_local_metadata,
            MAX_TRAIN_ROWS_PER_RESUME_LABEL,
            TRAIN_SAMPLE_SEED + fold_number,
        ).index.to_numpy(dtype=np.int64)
        training_positions = full_train_positions[sampled_fold_local]
        train_global_ids = active_row_ids[training_positions]
        training_rows = _load_rows(train_global_ids).set_index("__row_id")
        training_features = features.loc[training_positions]
        sample_group_codes = group_codes[training_positions]
        models = _fit_all_models(
            sampled_rows=training_rows,
            sampled_features=training_features,
            sampled_group_codes=sample_group_codes,
            labels=labels,
        )

        validation_predictions = {
            model_name: np.empty(len(validation_positions), dtype=object)
            for model_name in MODEL_NAMES
        }
        truth = labels_array[validation_positions]
        global_to_active = {
            int(row_id): int(active_position)
            for active_position, row_id in enumerate(active_row_ids)
        }
        validation_offsets = {
            int(active_position): index
            for index, active_position in enumerate(validation_positions)
        }
        wanted_global_ids = active_row_ids[validation_positions]
        seen = 0
        for validation_rows in _iter_selected_batches(wanted_global_ids):
            _add_prediction_batch(
                rows=validation_rows,
                all_features=features,
                global_to_active=global_to_active,
                validation_prediction_offsets=validation_offsets,
                baseline_vectorizer=models["baseline_vectorizer"],
                baseline_model=models["baseline_model"],
                pair_vectorizer=models["pair_vectorizer"],
                pair_baseline_model=models["pair_baseline_model"],
                pair_interaction_model=models["pair_interaction_model"],
                structured_model=models["structured_model"],
                fusion_model=models["fusion_model"],
                labels=labels,
                predictions=validation_predictions,
            )
            seen += len(validation_rows)
            del validation_rows
        if seen != len(validation_positions):
            raise RuntimeError(
                f"Fold {fold_number} scored {seen} rows, expected "
                f"{len(validation_positions)}."
            )
        fold_metrics = {
            model_name: _evaluate(truth, validation_predictions[model_name], labels)
            for model_name in MODEL_NAMES
        }
        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": len(training_positions),
                "validation_rows": len(validation_positions),
                "train_resume_groups": int(len(set(group_codes[training_positions]))),
                "validation_resume_groups": int(
                    len(set(group_codes[validation_positions]))
                ),
                "resume_group_overlap": 0,
                "metrics": fold_metrics,
            }
        )
        del models, training_rows, training_features, validation_predictions
        gc.collect()
        _observe_resources()

    cv_summary = _summarize_cv(fold_records)

    final_training_global_ids = active_row_ids[final_training_positions]
    final_training_rows = _load_rows(final_training_global_ids).set_index("__row_id")
    final_training_features = features.loc[final_training_positions]
    final_models = _fit_all_models(
        sampled_rows=final_training_rows,
        sampled_features=final_training_features,
        sampled_group_codes=group_codes[final_training_positions],
        labels=labels,
    )
    _, holdout_metrics = _evaluate_partition(
        active_positions=holdout_positions,
        active_row_ids=active_row_ids,
        labels_array=labels_array,
        all_features=features,
        trained_models=final_models,
        labels=labels,
    )
    runtime_seconds = time.perf_counter() - started
    _observe_resources(enforce_limit=False)
    resource_summary = {
        **_RESOURCE_STATE,
        "safety_floor_bytes": MIN_FREE_RAM_BYTES,
    }

    existing_baseline = reference["final_holdout_models"]["text_logistic_regression"]
    reproduced_baseline = holdout_metrics["text_baseline"]
    if not np.array_equal(
        np.asarray(reproduced_baseline["confusion_matrix"], dtype=int),
        np.asarray(existing_baseline["confusion_matrix"], dtype=int),
    ):
        raise RuntimeError(
            "The freshly reproduced text baseline does not match the saved "
            "Stage 2 holdout confusion matrix; refusing to publish a comparison."
        )

    holdout_matrices = {
        model: holdout_metrics[model]["confusion_matrix"] for model in MODEL_NAMES
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "experiment_id": EXPERIMENT_ID,
        "label_order": labels,
        "feature_schema": {
            "structured_numeric_columns": STAGE5C_NUMERIC_COLUMNS,
            "structured_categorical_columns": CATEGORICAL_FEATURE_COLUMNS,
            "pair_text_representation": [
                "resume_tfidf",
                "jd_tfidf",
                "elementwise_tfidf_overlap",
                "resume_jd_tfidf_cosine_similarity",
            ],
            "pair_feature_width": int(len(final_models["pair_vectorizer"].vocabulary_)),
        },
        "estimators": {
            "baseline_vectorizer": final_models["baseline_vectorizer"],
            "baseline_model": final_models["baseline_model"],
            "pair_vectorizer": final_models["pair_vectorizer"],
            "pair_baseline_model": final_models["pair_baseline_model"],
            "pair_interaction_model": final_models["pair_interaction_model"],
            "structured_model": final_models["structured_model"],
            "fusion_model": final_models["fusion_model"],
        },
        "configuration": {
            "holdout_seed": HOLDOUT_SEED,
            "holdout_size": HOLDOUT_SIZE,
            "outer_folds": OUTER_FOLDS,
            "inner_oof_folds": INNER_FOLDS,
            "training_sample_seed": TRAIN_SAMPLE_SEED,
            "max_rows_per_resume_label": MAX_TRAIN_ROWS_PER_RESUME_LABEL,
            "regularization_C": REGULARIZATION,
            "class_weight": config.CLASS_WEIGHT,
            "tfidf_max_features": config.TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
            "tfidf_min_df": config.TFIDF_MIN_DF,
            "tfidf_max_df": config.TFIDF_MAX_DF,
            "tfidf_sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            "split": split,
            "sampled_development_rows": int(len(final_training_positions)),
            "dataset_name": config.HF_DATASET_NAME,
            "source_categories": data_metadata["source_categories"],
            "experimental_only": True,
        },
    }
    metrics = {
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "labels": labels,
        "split": {
            **split,
            "sampled_development_rows": int(len(final_training_positions)),
            "sampled_development_resume_groups": int(
                len(set(group_codes[final_training_positions]))
            ),
            "holdout_seed": HOLDOUT_SEED,
            "holdout_size": HOLDOUT_SIZE,
            "outer_folds": OUTER_FOLDS,
            "inner_oof_folds": INNER_FOLDS,
        },
        "source_label_counts": data_metadata["source_label_counts"],
        "feature_availability": {
            column: int(features[column].notna().sum())
            for column in STAGE5C_NUMERIC_COLUMNS
        },
        "folds": fold_records,
        "cv_summary": cv_summary,
        "holdout_metrics": holdout_metrics,
        "runtime_seconds": runtime_seconds,
        "resources": resource_summary,
        "recommendation": (
            "No experiment should be selected from a single synthetic example. "
            "Use development grouped-fold results as the primary comparison; "
            "retain the existing baseline unless a prespecified development "
            "metric and class-wise guardrails support a candidate. The final "
            "holdout results are a one-time estimate, not a tuning signal."
        ),
    }

    joblib.dump(bundle, ARTIFACT_DIR / "stage5c_bundle.joblib")
    (ARTIFACT_DIR / "stage5c_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "fold": record["fold"],
                "model": model_name,
                "train_rows": record["train_rows"],
                "validation_rows": record["validation_rows"],
                "train_resume_groups": record["train_resume_groups"],
                "validation_resume_groups": record["validation_resume_groups"],
                "resume_group_overlap": record["resume_group_overlap"],
                "accuracy": record["metrics"][model_name]["accuracy"],
                "macro_f1": record["metrics"][model_name]["macro_f1"],
                "weighted_f1": record["metrics"][model_name]["weighted_f1"],
                **{
                    f"{label}_{metric}": record["metrics"][model_name]["per_class"][label][metric]
                    for label in labels
                    for metric in ("precision", "recall", "f1")
                },
            }
            for record in fold_records
            for model_name in MODEL_NAMES
        ]
    ).to_csv(ARTIFACT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "label": label,
                **holdout_metrics[model_name]["per_class"][label],
            }
            for model_name in MODEL_NAMES
            for label in labels
        ]
    ).to_csv(ARTIFACT_DIR / "holdout_per_class_metrics.csv", index=False)
    for model_name, matrix in holdout_matrices.items():
        pd.DataFrame(
            matrix,
            index=pd.Index(labels, name="actual"),
            columns=pd.Index(labels, name="predicted"),
        ).to_csv(ARTIFACT_DIR / f"holdout_confusion_{model_name}.csv")
    _write_report(status="completed", metrics=metrics)
    del final_models
    return metrics


def run_stage5c_experiment() -> dict[str, Any]:
    try:
        return _run()
    except (Stage5CResourceLimitError, MemoryError) as exc:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "stopped_resource_limit",
            "error": str(exc),
            "resources": dict(_RESOURCE_STATE),
            "safety_floor_bytes": MIN_FREE_RAM_BYTES,
        }
        (ARTIFACT_DIR / "stage5c_status.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        _write_report(
            status="stopped before completion",
            error=str(exc),
        )
        return failure


if __name__ == "__main__":
    result = run_stage5c_experiment()
    print(
        f"Stage 5C status={result['status']}; artifacts={ARTIFACT_DIR}; "
        f"report={REPORT_PATH}"
    )
