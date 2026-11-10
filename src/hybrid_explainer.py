"""Local logit explanations for the saved experimental hybrid classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.hybrid_predict import predict_hybrid
from src.structured_classifier import (
    HYBRID_BUNDLE_PATH,
    build_combined_text,
    build_hybrid_fusion_features,
    extract_structured_features,
    load_hybrid_bundle,
    validate_hybrid_bundle,
)


_LOGIT_ATOL = 1e-10
_LOGIT_RTOL = 1e-9


class HybridExplanationError(RuntimeError):
    """Raised when a fitted bundle cannot produce a faithful logit explanation."""


def _validate_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _validate_domain(value: object, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")
    return value


def _pipeline_components(
    estimator: Any,
    name: str,
) -> tuple[Any, Any]:
    if not hasattr(estimator, "named_steps"):
        raise HybridExplanationError(
            f"Saved {name} estimator does not expose a fitted preprocessing pipeline."
        )
    steps = estimator.named_steps
    if "preprocessing" not in steps or "classifier" not in steps:
        raise HybridExplanationError(
            f"Saved {name} pipeline must contain preprocessing and classifier steps."
        )
    classifier = steps["classifier"]
    if not all(hasattr(classifier, attr) for attr in ("coef_", "intercept_", "classes_")):
        raise HybridExplanationError(
            f"Saved {name} classifier does not expose fitted linear coefficients."
        )
    return steps["preprocessing"], classifier


def _dense_single_row(values: Any, feature_count: int, name: str) -> np.ndarray:
    dense = values.toarray() if sparse.issparse(values) else np.asarray(values)
    if dense.shape != (1, feature_count):
        raise HybridExplanationError(
            f"Transformed {name} features have shape {dense.shape}; "
            f"expected (1, {feature_count})."
        )
    if not np.isfinite(dense).all():
        raise HybridExplanationError(
            f"Transformed {name} features contain non-finite values."
        )
    return dense[0].astype(float, copy=False)


def _linear_logits(classifier: Any, transformed: Any, name: str) -> np.ndarray:
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercepts = np.asarray(classifier.intercept_, dtype=float)
    class_labels = [str(label) for label in classifier.classes_]
    if coefficients.ndim != 2 or coefficients.shape[0] != len(class_labels):
        raise HybridExplanationError(
            f"Saved {name} coefficient rows do not match its classes."
        )
    if intercepts.shape != (len(class_labels),):
        raise HybridExplanationError(
            f"Saved {name} intercepts do not match its classes."
        )
    scores = np.asarray(classifier.decision_function(transformed), dtype=float)
    if scores.shape != (1, len(class_labels)):
        raise HybridExplanationError(
            f"Saved {name} decision scores have unsupported shape {scores.shape}."
        )
    return scores[0]


def _verify_reconstruction(
    values: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    model_logit: float,
    name: str,
) -> tuple[float, float]:
    contributions = values * coefficients
    reconstructed = float(intercept + contributions.sum())
    error = reconstructed - float(model_logit)
    if not np.isclose(
        reconstructed,
        model_logit,
        rtol=_LOGIT_RTOL,
        atol=_LOGIT_ATOL,
    ):
        raise HybridExplanationError(
            f"{name} logit reconstruction failed: error={error:.3g}."
        )
    return reconstructed, error


def explain_hybrid(
    resume_text: object,
    jd_text: object,
    *,
    bundle: dict[str, Any] | None = None,
    bundle_path: str | Path | None = None,
    resume_domain: object = None,
    jd_domain: object = None,
    top_k_text_terms: int = 10,
) -> dict[str, Any]:
    """Explain hybrid logits without changing the existing prediction.

    Hybrid contributions are additive terms in class logits, not causal
    effects or direct increments to class probabilities. The optional text
    term lists similarly explain the text model's logits, not skill evidence.
    """
    resume = _validate_text(resume_text, "Resume")
    jd = _validate_text(jd_text, "Job description")
    resume_domain_value = _validate_domain(resume_domain, "Resume domain")
    jd_domain_value = _validate_domain(jd_domain, "Job description domain")
    if bundle is not None and bundle_path is not None:
        raise ValueError("Provide either bundle or bundle_path, not both.")
    if (
        isinstance(top_k_text_terms, bool)
        or not isinstance(top_k_text_terms, int)
        or top_k_text_terms < 1
    ):
        raise ValueError("top_k_text_terms must be a positive integer.")

    active_bundle = (
        validate_hybrid_bundle(bundle)
        if bundle is not None
        else load_hybrid_bundle(bundle_path or HYBRID_BUNDLE_PATH)
    )
    prediction = predict_hybrid(
        resume,
        jd,
        bundle=active_bundle,
        resume_domain=resume_domain_value,
        jd_domain=jd_domain_value,
    )
    estimators = active_bundle["estimators"]
    schema = active_bundle["feature_schema"]
    labels = list(active_bundle["label_order"])

    input_row = pd.DataFrame(
        [
            {
                "resume": resume,
                "jd": jd,
                "resume_domain": resume_domain_value,
                "jd_domain": jd_domain_value,
            }
        ]
    )
    structured = extract_structured_features(input_row)
    structured_columns = list(schema["structured_feature_columns"])
    if list(structured.columns) != structured_columns:
        raise HybridExplanationError(
            "Current structured feature ordering does not match the saved bundle."
        )
    structured = structured.loc[:, structured_columns]

    combined_text = build_combined_text(resume, jd)
    vectorizer = estimators["text_vectorizer"]
    text_model = estimators["text_model"]
    text_matrix = vectorizer.transform([combined_text])
    text_scores = _linear_logits(text_model, text_matrix, "text")
    text_labels = [str(label) for label in text_model.classes_]
    if set(text_labels) != set(labels):
        raise HybridExplanationError(
            "Saved text model classes do not match the bundle class order."
        )
    text_score_by_label = {
        label: float(text_scores[index])
        for index, label in enumerate(text_labels)
    }

    text_coefficients = np.asarray(text_model.coef_, dtype=float)
    text_intercepts = np.asarray(text_model.intercept_, dtype=float)
    text_terms = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    if text_coefficients.shape[1] != len(text_terms):
        raise HybridExplanationError(
            "Text-model coefficients do not match the saved vectorizer vocabulary."
        )
    text_row = text_matrix.getrow(0) if sparse.issparse(text_matrix) else sparse.csr_matrix(text_matrix)
    text_explanations: dict[str, Any] = {}
    for class_index, label in enumerate(text_labels):
        indices = text_row.indices
        tfidf_values = text_row.data.astype(float, copy=False)
        term_coefficients = text_coefficients[class_index, indices]
        term_contributions = tfidf_values * term_coefficients
        positive_order = np.argsort(-term_contributions)
        negative_order = np.argsort(term_contributions)

        def term_records(order: np.ndarray, positive: bool) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for position in order:
                contribution = float(term_contributions[position])
                if (positive and contribution <= 0) or (
                    not positive and contribution >= 0
                ):
                    continue
                feature_index = int(indices[position])
                records.append(
                    {
                        "class": label,
                        "term": str(text_terms[feature_index]),
                        "tfidf_value": float(tfidf_values[position]),
                        "coefficient": float(term_coefficients[position]),
                        "contribution": contribution,
                    }
                )
                if len(records) == top_k_text_terms:
                    break
            return records

        text_reconstructed, text_error = _verify_reconstruction(
            np.asarray(text_row.toarray(), dtype=float)[0],
            text_coefficients[class_index],
            float(text_intercepts[class_index]),
            text_score_by_label[label],
            f"text model class {label}",
        )
        text_explanations[label] = {
            "intercept": float(text_intercepts[class_index]),
            "model_logit": text_score_by_label[label],
            "reconstructed_logit": text_reconstructed,
            "reconstruction_error": text_error,
            "top_positive_terms": term_records(positive_order, positive=True),
            "top_negative_terms": term_records(negative_order, positive=False),
        }

    ordered_text_probabilities = np.asarray(
        [[prediction["text_model_probabilities"][label] for label in labels]],
        dtype=float,
    )
    fusion_features = build_hybrid_fusion_features(
        structured,
        ordered_text_probabilities,
        labels,
    )
    fusion_columns = list(schema["fusion_feature_columns"])
    if list(fusion_features.columns) != fusion_columns:
        raise HybridExplanationError(
            "Current hybrid feature ordering does not match the saved bundle."
        )

    hybrid_model = estimators["hybrid_model"]
    preprocessing, classifier = _pipeline_components(hybrid_model, "hybrid")
    transformed = preprocessing.transform(fusion_features)
    feature_names = [str(name) for name in preprocessing.get_feature_names_out()]
    transformed_values = _dense_single_row(
        transformed,
        len(feature_names),
        "hybrid",
    )
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercepts = np.asarray(classifier.intercept_, dtype=float)
    hybrid_labels = [str(label) for label in classifier.classes_]
    if set(hybrid_labels) != set(labels):
        raise HybridExplanationError(
            "Saved hybrid model classes do not match the bundle class order."
        )
    if coefficients.shape[1] != len(feature_names):
        raise HybridExplanationError(
            "Hybrid coefficients do not match transformed feature names."
        )
    hybrid_scores = _linear_logits(classifier, transformed, "hybrid")
    hybrid_score_by_label = {
        label: float(hybrid_scores[index])
        for index, label in enumerate(hybrid_labels)
    }
    hybrid_explanations: dict[str, Any] = {}
    for class_index, label in enumerate(hybrid_labels):
        reconstructed, error = _verify_reconstruction(
            transformed_values,
            coefficients[class_index],
            float(intercepts[class_index]),
            hybrid_score_by_label[label],
            f"hybrid model class {label}",
        )
        records = []
        for feature_name, value, coefficient in zip(
            feature_names,
            transformed_values,
            coefficients[class_index],
        ):
            feature_type = (
                "text_probability"
                if feature_name.startswith("numeric__text_probability__")
                else "structured"
            )
            records.append(
                {
                    "class": label,
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "transformed_value": float(value),
                    "coefficient": float(coefficient),
                    "contribution": float(value * coefficient),
                }
            )
        hybrid_explanations[label] = {
            "intercept": float(intercepts[class_index]),
            "contributions": records,
            "reconstructed_logit": reconstructed,
            "model_logit": hybrid_score_by_label[label],
            "reconstruction_error": error,
            "reconstruction_verified": True,
        }

    return {
        "experiment_id": active_bundle["experiment_id"],
        "class_order": labels,
        "predicted_label": prediction["predicted_label"],
        "hybrid_probabilities": prediction["hybrid_probabilities"],
        "prediction": prediction,
        "hybrid_logit_explanations": hybrid_explanations,
        "text_model_explanations": text_explanations,
        "explanation_notes": [
            "Hybrid contributions are additive model-logit terms, not causal effects or direct probability increments.",
            "TF-IDF terms explain text-model logits only; they are not skill-extractor matches or evidence of competence.",
            "Skill matches are keyword-derived and do not establish candidate competence.",
        ],
    }


__all__ = ["HybridExplanationError", "explain_hybrid"]
