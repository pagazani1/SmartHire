"""Feature extraction and models for the separate structured experiment."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.preprocessing import clean_text
from src.skill_extractor import extract_jd_skills, extract_resume_skills


NUMERIC_FEATURE_COLUMNS = [
    "resume_skill_count",
    "jd_skill_count",
    "required_skill_group_count",
    "matched_required_group_count",
    "missing_required_group_count",
    "required_skill_coverage",
    "required_coverage_available",
    "preferred_skill_group_count",
    "matched_preferred_group_count",
    "missing_preferred_group_count",
    "preferred_skill_coverage",
    "preferred_coverage_available",
    "matched_skill_count",
    "missing_skill_count",
    "unclassified_jd_skill_count",
    "domain_compatible",
]
CATEGORICAL_FEATURE_COLUMNS = ["resume_domain", "jd_domain"]
TEXT_PROBABILITY_PREFIX = "text_probability__"
HYBRID_BUNDLE_FORMAT_VERSION = 1
HYBRID_EXPERIMENT_ID = "smarthire-stage2-grouped-hybrid-v1"
HYBRID_ARTIFACT_DIR = config.MODELS_DIR / "hybrid_experimental"
HYBRID_BUNDLE_PATH = HYBRID_ARTIFACT_DIR / "hybrid_bundle.joblib"


class HybridArtifactError(RuntimeError):
    """Raised when an experimental hybrid inference bundle is unavailable."""


def build_combined_text(resume: object, jd: object) -> str:
    """Create the existing normalized resume-plus-JD TF-IDF input."""
    resume_text = "" if pd.isna(resume) else str(resume)
    jd_text = "" if pd.isna(jd) else str(jd)
    return f"{clean_text(resume_text)} {clean_text(jd_text)}".strip()


def extract_structured_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive deterministic keyword/domain features without returning source text.

    Skill rules and aliases come from the existing fixed skill extractor. No
    labels are used during extraction. Undefined coverage is represented as
    missing and is accompanied by an explicit availability indicator.
    """
    required = {"resume", "jd", "resume_domain", "jd_domain"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns for structured features: "
            + ", ".join(sorted(missing))
        )

    resumes = data["resume"].fillna("").astype(str)
    jds = data["jd"].fillna("").astype(str)
    resume_cache = {
        text: extract_resume_skills(text) for text in resumes.unique()
    }
    jd_cache = {text: extract_jd_skills(text) for text in jds.unique()}

    rows: list[dict[str, Any]] = []
    for resume_text, jd_text, resume_domain, jd_domain in zip(
        resumes,
        jds,
        data["resume_domain"],
        data["jd_domain"],
    ):
        resume_skills = resume_cache[resume_text]
        jd_analysis = jd_cache[jd_text]
        jd_skills = jd_analysis["jd_skills"]
        required_groups = jd_analysis["required_skill_groups"]
        preferred_groups = jd_analysis["preferred_skill_groups"]
        matched_required = sum(
            bool(group.intersection(resume_skills)) for group in required_groups
        )
        matched_preferred = sum(
            bool(group.intersection(resume_skills)) for group in preferred_groups
        )
        matched_skills = resume_skills.intersection(jd_skills)
        normalized_resume_domain = (
            "" if pd.isna(resume_domain) else str(resume_domain).strip()
        )
        normalized_jd_domain = (
            "" if pd.isna(jd_domain) else str(jd_domain).strip()
        )

        rows.append(
            {
                "resume_skill_count": len(resume_skills),
                "jd_skill_count": len(jd_skills),
                "required_skill_group_count": len(required_groups),
                "matched_required_group_count": matched_required,
                "missing_required_group_count": (
                    len(required_groups) - matched_required
                ),
                "required_skill_coverage": (
                    matched_required / len(required_groups)
                    if required_groups
                    else np.nan
                ),
                "required_coverage_available": int(bool(required_groups)),
                "preferred_skill_group_count": len(preferred_groups),
                "matched_preferred_group_count": matched_preferred,
                "missing_preferred_group_count": (
                    len(preferred_groups) - matched_preferred
                ),
                "preferred_skill_coverage": (
                    matched_preferred / len(preferred_groups)
                    if preferred_groups
                    else np.nan
                ),
                "preferred_coverage_available": int(bool(preferred_groups)),
                "matched_skill_count": len(matched_skills),
                "missing_skill_count": len(jd_skills - resume_skills),
                "unclassified_jd_skill_count": len(
                    jd_analysis["unclassified_jd_skills"]
                ),
                "domain_compatible": int(
                    bool(normalized_resume_domain)
                    and normalized_resume_domain.casefold()
                    == normalized_jd_domain.casefold()
                ),
                "resume_domain": normalized_resume_domain or "__missing__",
                "jd_domain": normalized_jd_domain or "__missing__",
            }
        )

    return pd.DataFrame(rows, index=data.index)


def sample_group_label_rows(
    data: pd.DataFrame,
    max_rows_per_group_label: int,
    random_state: int,
) -> pd.DataFrame:
    """Bound training rows per resume/label while preserving each group-label."""
    required = {"resume_group", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing columns for grouped sampling: " + ", ".join(sorted(missing))
        )
    if max_rows_per_group_label < 1:
        raise ValueError("max_rows_per_group_label must be at least one.")

    generator = np.random.default_rng(random_state)
    selected_positions: list[int] = []
    grouped_positions = data.groupby(
        ["resume_group", "label"],
        sort=True,
        dropna=False,
    ).indices.values()
    for positions in grouped_positions:
        positions = np.asarray(positions, dtype=int)
        if len(positions) > max_rows_per_group_label:
            positions = generator.choice(
                positions,
                size=max_rows_per_group_label,
                replace=False,
            )
        selected_positions.extend(positions.tolist())
    return data.iloc[sorted(selected_positions)]


def build_hybrid_fusion_features(
    structured_features: pd.DataFrame,
    text_probabilities: np.ndarray,
    class_labels: Sequence[str],
) -> pd.DataFrame:
    """Append class-aligned base-text probabilities to structured features."""
    probability_frame = probability_feature_frame(
        text_probabilities,
        class_labels,
        index=structured_features.index,
    )
    return pd.concat([structured_features, probability_frame], axis=1)


def make_structured_classifier(
    numeric_columns: Sequence[str] = NUMERIC_FEATURE_COLUMNS,
    regularization: float = 1.0,
) -> Pipeline:
    """Build a fold-fit preprocessing and regularized Logistic Regression model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=regularization,
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def fit_text_logistic_regression(
    combined_texts: Sequence[str],
    labels: Sequence[str],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit the existing text LR recipe, with TF-IDF learned only from inputs."""
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )
    features = vectorizer.fit_transform(combined_texts)
    model = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    model.fit(features, labels)
    return vectorizer, model


def predict_text_probabilities(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    combined_texts: Sequence[str],
    class_labels: Sequence[str],
) -> np.ndarray:
    """Predict probabilities aligned to the stable requested class order."""
    probabilities = model.predict_proba(vectorizer.transform(combined_texts))
    aligned = np.zeros((len(combined_texts), len(class_labels)), dtype=float)
    label_indexes = {label: index for index, label in enumerate(class_labels)}
    for model_column, label in enumerate(model.classes_):
        aligned[:, label_indexes[str(label)]] = probabilities[:, model_column]
    return aligned


def generate_oof_text_probabilities(
    combined_texts: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    class_labels: Sequence[str],
    n_splits: int = 5,
) -> np.ndarray:
    """Generate LR probabilities with every row scored outside its resume group."""
    texts = np.asarray(combined_texts, dtype=object)
    target = np.asarray(labels, dtype=object)
    group_values = np.asarray(groups, dtype=object)
    if not (len(texts) == len(target) == len(group_values)):
        raise ValueError("Text, label, and group arrays must have equal lengths.")
    if len(pd.unique(group_values)) < n_splits:
        raise ValueError("There are fewer resume groups than OOF folds.")

    output = np.full((len(texts), len(class_labels)), np.nan, dtype=float)
    assigned = np.zeros(len(texts), dtype=bool)
    splitter = GroupKFold(n_splits=n_splits)
    for train_indices, validation_indices in splitter.split(
        texts, target, groups=group_values
    ):
        train_groups = set(group_values[train_indices])
        validation_groups = set(group_values[validation_indices])
        if train_groups.intersection(validation_groups):
            raise AssertionError("Resume-group overlap detected in OOF fold.")
        vectorizer, model = fit_text_logistic_regression(
            texts[train_indices], target[train_indices]
        )
        output[validation_indices] = predict_text_probabilities(
            vectorizer,
            model,
            texts[validation_indices],
            class_labels,
        )
        assigned[validation_indices] = True

    if not assigned.all() or not np.isfinite(output).all():
        raise AssertionError("OOF probabilities are missing or non-finite.")
    return output


def probability_feature_frame(
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Name LR probability columns for the structured fusion model."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_labels):
        raise ValueError("Probability shape does not match the class labels.")
    return pd.DataFrame(
        probabilities,
        columns=[
            f"{TEXT_PROBABILITY_PREFIX}{label}" for label in class_labels
        ],
        index=index,
    )


def fit_hybrid_estimators(
    training_rows: pd.DataFrame,
    structured_features: pd.DataFrame,
    class_labels: Sequence[str],
    *,
    oof_splits: int,
    regularization: float,
) -> dict[str, Any]:
    """Fit the Stage 2 final-model protocol on the already capped sample."""
    if len(training_rows) != len(structured_features):
        raise ValueError("Training rows and structured features must align.")
    if list(training_rows.index) != list(structured_features.index):
        raise ValueError("Training rows and structured feature indexes must match.")
    required = {"combined_text", "label", "resume_group"}
    missing = required.difference(training_rows.columns)
    if missing:
        raise ValueError(
            "Missing training columns: " + ", ".join(sorted(missing))
        )
    actual_labels = set(training_rows["label"].astype(str))
    if actual_labels != set(class_labels):
        raise ValueError("Training labels do not match the saved class ordering.")
    expected_structured_columns = [
        *NUMERIC_FEATURE_COLUMNS,
        *CATEGORICAL_FEATURE_COLUMNS,
    ]
    if list(structured_features.columns) != expected_structured_columns:
        raise ValueError("Structured feature columns do not match the expected schema.")

    labels = [str(label) for label in class_labels]
    target = training_rows["label"].astype(str).to_numpy()
    vectorizer, text_model = fit_text_logistic_regression(
        training_rows["combined_text"].to_numpy(),
        target,
    )
    structured_model = make_structured_classifier(
        regularization=regularization
    )
    structured_model.fit(structured_features, target)

    oof_probabilities = generate_oof_text_probabilities(
        training_rows["combined_text"].to_numpy(),
        target,
        training_rows["resume_group"].to_numpy(),
        labels,
        n_splits=oof_splits,
    )
    fusion_training_features = build_hybrid_fusion_features(
        structured_features,
        oof_probabilities,
        labels,
    )
    probability_columns = [
        f"{TEXT_PROBABILITY_PREFIX}{label}" for label in labels
    ]
    fusion_numeric_columns = [*NUMERIC_FEATURE_COLUMNS, *probability_columns]
    hybrid_model = make_structured_classifier(
        numeric_columns=fusion_numeric_columns,
        regularization=regularization,
    )
    hybrid_model.fit(fusion_training_features, target)

    return {
        "text_vectorizer": vectorizer,
        "text_model": text_model,
        "structured_model": structured_model,
        "hybrid_model": hybrid_model,
        "structured_feature_columns": expected_structured_columns,
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "fusion_feature_columns": list(fusion_training_features.columns),
        "fusion_numeric_feature_columns": fusion_numeric_columns,
        "label_order": labels,
    }


def create_hybrid_bundle(
    fitted_estimators: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Wrap fitted estimators with a validated inference schema and metadata."""
    required = {
        "text_vectorizer",
        "text_model",
        "structured_model",
        "hybrid_model",
        "structured_feature_columns",
        "numeric_feature_columns",
        "categorical_feature_columns",
        "fusion_feature_columns",
        "fusion_numeric_feature_columns",
        "label_order",
    }
    missing = required.difference(fitted_estimators)
    if missing:
        raise ValueError(
            "Missing fitted hybrid components: " + ", ".join(sorted(missing))
        )
    labels = list(fitted_estimators["label_order"])
    for estimator_name in ("text_model", "structured_model", "hybrid_model"):
        estimator_classes = [
            str(label)
            for label in fitted_estimators[estimator_name].classes_
        ]
        if set(estimator_classes) != set(labels):
            raise ValueError(
                f"{estimator_name} classes do not match the bundle labels."
            )
    return {
        "format_version": HYBRID_BUNDLE_FORMAT_VERSION,
        "experiment_id": HYBRID_EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "training_config": dict(training_config),
        "label_order": labels,
        "feature_schema": {
            "structured_feature_columns": list(
                fitted_estimators["structured_feature_columns"]
            ),
            "numeric_feature_columns": list(
                fitted_estimators["numeric_feature_columns"]
            ),
            "categorical_feature_columns": list(
                fitted_estimators["categorical_feature_columns"]
            ),
            "fusion_feature_columns": list(
                fitted_estimators["fusion_feature_columns"]
            ),
            "fusion_numeric_feature_columns": list(
                fitted_estimators["fusion_numeric_feature_columns"]
            ),
        },
        "estimators": {
            estimator_name: fitted_estimators[estimator_name]
            for estimator_name in (
                "text_vectorizer",
                "text_model",
                "structured_model",
                "hybrid_model",
            )
        },
    }


def save_hybrid_bundle(
    bundle: dict[str, Any],
    bundle_path: str | Path = HYBRID_BUNDLE_PATH,
) -> Path:
    """Atomically persist an experimental bundle without touching old models."""
    validate_hybrid_bundle(bundle, check_sklearn_version=False)
    target = Path(bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, target)
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise HybridArtifactError(
            f"Unable to save experimental hybrid bundle to {target}."
        ) from exc
    return target


def validate_hybrid_bundle(
    bundle: Any,
    *,
    check_sklearn_version: bool = True,
) -> dict[str, Any]:
    """Validate bundle version, estimator presence, classes, and schema."""
    required_top_level = {
        "format_version",
        "experiment_id",
        "sklearn_version",
        "label_order",
        "feature_schema",
        "estimators",
        "training_config",
    }
    if not isinstance(bundle, dict) or not required_top_level.issubset(bundle):
        raise HybridArtifactError(
            "Experimental hybrid bundle is invalid or missing required metadata."
        )
    if bundle["format_version"] != HYBRID_BUNDLE_FORMAT_VERSION:
        raise HybridArtifactError(
            "Experimental hybrid bundle format is incompatible. "
            f"Expected version {HYBRID_BUNDLE_FORMAT_VERSION}, "
            f"found {bundle['format_version']}."
        )
    if bundle["experiment_id"] != HYBRID_EXPERIMENT_ID:
        raise HybridArtifactError("Artifact is not a supported SmartHire hybrid experiment.")
    if check_sklearn_version:
        saved_version = str(bundle["sklearn_version"]).split(".")[:2]
        current_version = sklearn.__version__.split(".")[:2]
        if saved_version != current_version:
            raise HybridArtifactError(
                "Experimental hybrid bundle was trained with scikit-learn "
                f"{bundle['sklearn_version']}; current version is "
                f"{sklearn.__version__}. Use a compatible environment."
            )

    estimators = bundle["estimators"]
    required_estimators = {
        "text_vectorizer",
        "text_model",
        "structured_model",
        "hybrid_model",
    }
    if not isinstance(estimators, dict) or not required_estimators.issubset(estimators):
        raise HybridArtifactError(
            "Experimental hybrid bundle is missing one or more fitted estimators."
        )
    schema = bundle["feature_schema"]
    required_schema = {
        "structured_feature_columns",
        "numeric_feature_columns",
        "categorical_feature_columns",
        "fusion_feature_columns",
        "fusion_numeric_feature_columns",
    }
    if not isinstance(schema, dict) or not required_schema.issubset(schema):
        raise HybridArtifactError(
            "Experimental hybrid bundle has an incompatible feature schema."
        )
    if list(schema["structured_feature_columns"]) != [
        *NUMERIC_FEATURE_COLUMNS,
        *CATEGORICAL_FEATURE_COLUMNS,
    ]:
        raise HybridArtifactError(
            "Experimental hybrid bundle structured feature ordering is incompatible."
        )
    if list(schema["numeric_feature_columns"]) != list(NUMERIC_FEATURE_COLUMNS):
        raise HybridArtifactError(
            "Experimental hybrid bundle numeric feature ordering is incompatible."
        )
    if list(schema["categorical_feature_columns"]) != list(
        CATEGORICAL_FEATURE_COLUMNS
    ):
        raise HybridArtifactError(
            "Experimental hybrid bundle categorical feature ordering is incompatible."
        )
    labels = list(bundle["label_order"])
    expected_probability_columns = [
        f"{TEXT_PROBABILITY_PREFIX}{label}" for label in labels
    ]
    expected_fusion_columns = [
        *schema["structured_feature_columns"],
        *expected_probability_columns,
    ]
    if list(schema["fusion_feature_columns"]) != expected_fusion_columns:
        raise HybridArtifactError(
            "Experimental hybrid bundle fusion feature ordering is incompatible."
        )
    if list(schema["fusion_numeric_feature_columns"]) != [
        *NUMERIC_FEATURE_COLUMNS,
        *expected_probability_columns,
    ]:
        raise HybridArtifactError(
            "Experimental hybrid bundle fusion numeric feature ordering is incompatible."
        )
    for estimator_name in ("text_model", "structured_model", "hybrid_model"):
        estimator = estimators[estimator_name]
        if not hasattr(estimator, "classes_"):
            raise HybridArtifactError(
                f"Experimental hybrid bundle {estimator_name} is not fitted."
            )
        estimator_classes = [str(label) for label in estimator.classes_]
        if set(estimator_classes) != set(labels):
            raise HybridArtifactError(
                f"Experimental hybrid bundle {estimator_name} labels are incompatible."
            )
    if not hasattr(estimators["text_vectorizer"], "vocabulary_"):
        raise HybridArtifactError(
            "Experimental hybrid bundle text vectorizer is not fitted."
        )
    return bundle


def load_hybrid_bundle(
    bundle_path: str | Path = HYBRID_BUNDLE_PATH,
) -> dict[str, Any]:
    """Load and validate the separate experimental inference bundle."""
    path = Path(bundle_path)
    if not path.is_file():
        raise HybridArtifactError(
            f"Experimental hybrid bundle not found: {path}. "
            "Run train_hybrid_experimental.py to create it."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise HybridArtifactError(
            f"Unable to load experimental hybrid bundle from {path}; "
            "the artifact may be corrupt or incompatible."
        ) from exc
    return validate_hybrid_bundle(bundle)
