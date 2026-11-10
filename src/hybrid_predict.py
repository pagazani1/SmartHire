"""Inference helpers for the separate experimental SmartHire hybrid model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.structured_classifier import (
    HYBRID_BUNDLE_PATH,
    HybridArtifactError,
    build_combined_text,
    build_hybrid_fusion_features,
    extract_structured_features,
    load_hybrid_bundle,
    predict_text_probabilities,
    validate_hybrid_bundle,
)
from src.skill_extractor import analyse_skills


def _validate_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _validate_domain(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")
    return value


def _ordered_probabilities(
    estimator: Any,
    features: pd.DataFrame,
    labels: list[str],
) -> dict[str, float]:
    model_probabilities = estimator.predict_proba(features)[0]
    model_labels = [str(label) for label in estimator.classes_]
    aligned = {label: 0.0 for label in labels}
    for model_index, label in enumerate(model_labels):
        aligned[label] = float(model_probabilities[model_index])
    return aligned


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _skill_evidence(resume_text: str, jd_text: str) -> dict[str, Any]:
    analysis = analyse_skills(resume_text, jd_text)
    resume_skills = analysis["resume_skills"]

    def groups_with_evidence(groups: list[set[str]]) -> tuple[list[list[str]], list[list[str]]]:
        matched: list[list[str]] = []
        missing: list[list[str]] = []
        for group in groups:
            ordered_group = sorted(group)
            if group.intersection(resume_skills):
                matched.append(ordered_group)
            else:
                missing.append(ordered_group)
        return matched, missing

    matched_required_groups, missing_required_groups = groups_with_evidence(
        analysis["required_skill_groups"]
    )
    matched_preferred_groups, missing_preferred_groups = groups_with_evidence(
        analysis["preferred_skill_groups"]
    )
    return {
        "detected_resume_skills": sorted(analysis["resume_skills"]),
        "detected_jd_skills": sorted(analysis["jd_skills"]),
        "required_skill_groups": [
            sorted(group) for group in analysis["required_skill_groups"]
        ],
        "matched_required_skill_groups": matched_required_groups,
        "missing_required_skill_groups": missing_required_groups,
        "preferred_skill_groups": [
            sorted(group) for group in analysis["preferred_skill_groups"]
        ],
        "matched_preferred_skill_groups": matched_preferred_groups,
        "missing_preferred_skill_groups": missing_preferred_groups,
        "unclassified_jd_skills": sorted(analysis["unclassified_jd_skills"]),
        "coverage_metrics": analysis["coverage_metrics"],
        "extraction_limitations": analysis["extraction_limitations"],
    }


def predict_hybrid(
    resume_text: object,
    jd_text: object,
    *,
    bundle: dict[str, Any] | None = None,
    bundle_path: str | Path | None = None,
    resume_domain: object = None,
    jd_domain: object = None,
) -> dict[str, Any]:
    """Return skill evidence, structured features, text scores, and hybrid output.

    The production prediction endpoint is not affected. If no in-memory bundle
    is provided, the experimental bundle is loaded from ``bundle_path`` or
    ``models/hybrid_experimental/hybrid_bundle.joblib``.
    """
    validated_resume = _validate_text(resume_text, "Resume")
    validated_jd = _validate_text(jd_text, "Job description")
    validated_resume_domain = _validate_domain(resume_domain, "Resume domain")
    validated_jd_domain = _validate_domain(jd_domain, "Job description domain")
    if bundle is not None and bundle_path is not None:
        raise ValueError("Provide either bundle or bundle_path, not both.")

    active_bundle = (
        validate_hybrid_bundle(bundle)
        if bundle is not None
        else load_hybrid_bundle(bundle_path or HYBRID_BUNDLE_PATH)
    )
    estimators = active_bundle["estimators"]
    schema = active_bundle["feature_schema"]
    labels = list(active_bundle["label_order"])

    input_row = pd.DataFrame(
        [
            {
                "resume": validated_resume,
                "jd": validated_jd,
                "resume_domain": validated_resume_domain,
                "jd_domain": validated_jd_domain,
            }
        ]
    )
    structured_features = extract_structured_features(input_row)
    structured_columns = list(schema["structured_feature_columns"])
    if list(structured_features.columns) != structured_columns:
        raise HybridArtifactError(
            "Current structured feature ordering does not match the saved bundle."
        )
    structured_features = structured_features.loc[:, structured_columns]

    combined_text = build_combined_text(validated_resume, validated_jd)
    text_probabilities = predict_text_probabilities(
        estimators["text_vectorizer"],
        estimators["text_model"],
        [combined_text],
        labels,
    )
    fusion_features = build_hybrid_fusion_features(
        structured_features,
        text_probabilities,
        labels,
    )
    fusion_columns = list(schema["fusion_feature_columns"])
    if list(fusion_features.columns) != fusion_columns:
        raise HybridArtifactError(
            "Current fusion feature ordering does not match the saved bundle."
        )

    hybrid_probabilities = _ordered_probabilities(
        estimators["hybrid_model"],
        fusion_features,
        labels,
    )
    predicted_label = str(estimators["hybrid_model"].predict(fusion_features)[0])
    structured_feature_values = {
        name: _json_scalar(value)
        for name, value in structured_features.iloc[0].to_dict().items()
    }
    return {
        "structured_skill_evidence": _skill_evidence(
            validated_resume,
            validated_jd,
        ),
        "structured_features": structured_feature_values,
        "text_model_probabilities": {
            label: float(text_probabilities[0, index])
            for index, label in enumerate(labels)
        },
        "hybrid_probabilities": hybrid_probabilities,
        "predicted_label": predicted_label,
        "experiment_id": active_bundle["experiment_id"],
    }


__all__ = ["HybridArtifactError", "predict_hybrid"]
