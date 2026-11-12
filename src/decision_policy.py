"""Explainable recruiter-facing decision policy over existing model evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


DECISION_POLICY_ID = "required_skill_group_coverage_v1"


def _names(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return []
    return sorted({str(item) for item in value if isinstance(item, str)})


def _groups(value: object) -> list[set[str]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    groups: list[set[str]] = []
    for group in value:
        names = _names(group)
        if not names:
            return None
        groups.append(set(names))
    return groups


def apply_decision_policy(
    model_prediction: dict[str, Any],
    skill_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Assign a final label while preserving the original text-model output.

    Classified required groups drive the decision when available. Alternative
    skills within a group are satisfied by any detected alternative. Without
    usable required-group evidence, the model label is retained.
    """
    model_label = str(model_prediction["label"])
    model_probs = dict(model_prediction["probs"])
    required_groups = _groups(skill_analysis.get("required_skill_groups"))
    resume_skills_value = skill_analysis.get("resume_skills")
    resume_skills = (
        set(_names(resume_skills_value))
        if isinstance(resume_skills_value, Iterable)
        and not isinstance(resume_skills_value, str)
        else None
    )
    evidence_available = required_groups is not None and bool(required_groups) and resume_skills is not None

    matched_required_groups: list[set[str]] = []
    missing_required_groups: list[set[str]] = []
    if evidence_available:
        for group in required_groups:
            if group.intersection(resume_skills):
                matched_required_groups.append(group)
            else:
                missing_required_groups.append(group)
        if not missing_required_groups:
            final_label = "Good Fit"
            reason = "Every detected required-skill group has a keyword match."
        elif matched_required_groups:
            final_label = "Potential Fit"
            reason = "Some required-skill groups match, but others are missing."
        else:
            final_label = "No Fit"
            reason = "No detected required-skill group has a keyword match."
    else:
        final_label = model_label
        reason = (
            "No usable classified required-skill groups were available; "
            "the final category falls back to the text-model prediction."
        )

    preferred_groups = _groups(skill_analysis.get("preferred_skill_groups"))
    preferred_matched_groups: list[set[str]] = []
    preferred_missing_groups: list[set[str]] = []
    if preferred_groups is not None and resume_skills is not None:
        for group in preferred_groups:
            if group.intersection(resume_skills):
                preferred_matched_groups.append(group)
            else:
                preferred_missing_groups.append(group)

    def format_groups(groups: list[set[str]]) -> list[list[str]]:
        return [sorted(group) for group in groups]

    matched_skills = _names(skill_analysis.get("matched_skills"))
    missing_skills = _names(skill_analysis.get("missing_skills"))
    matched_required_skills = _names(skill_analysis.get("matched_required_skills"))
    missing_required_skills = _names(skill_analysis.get("missing_required_skills"))
    matched_preferred_skills = _names(skill_analysis.get("matched_preferred_skills"))
    missing_preferred_skills = _names(skill_analysis.get("missing_preferred_skills"))
    required_count = len(required_groups) if required_groups is not None else 0
    matched_count = len(matched_required_groups)
    required_coverage = (
        matched_count / required_count if evidence_available and required_count else None
    )

    explanation = (
        f"{reason} Required coverage: {matched_count}/{required_count}"
        + (
            f" ({required_coverage:.1%})."
            if required_coverage is not None
            else "."
        )
        + f" The text model predicted {model_label}; its probabilities are unchanged."
    )
    if preferred_groups:
        explanation += (
            f" Preferred groups matched: {len(preferred_matched_groups)}"
            f"/{len(preferred_groups)}; preferred skills do not override"
            " required-skill coverage."
        )
    explanation += (
        " Skill matches are keyword evidence, not proof of proficiency."
    )

    return {
        "label": final_label,
        "probs": model_probs,
        "model_label": model_label,
        "model_probs": model_probs,
        "decision_policy": DECISION_POLICY_ID,
        "decision_explanation": explanation,
        "decision_evidence": {
            "structured_evidence_available": evidence_available,
            "required_skill_group_count": required_count,
            "matched_required_group_count": matched_count,
            "missing_required_group_count": len(missing_required_groups),
            "required_skill_coverage": required_coverage,
            "required_skill_groups": (
                format_groups(required_groups) if required_groups is not None else []
            ),
            "matched_required_skill_groups": format_groups(matched_required_groups),
            "missing_required_skill_groups": format_groups(missing_required_groups),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_required_skills": matched_required_skills,
            "missing_required_skills": missing_required_skills,
            "preferred_skill_groups": (
                format_groups(preferred_groups) if preferred_groups is not None else []
            ),
            "matched_preferred_skill_groups": format_groups(
                preferred_matched_groups
            ),
            "missing_preferred_skill_groups": format_groups(
                preferred_missing_groups
            ),
            "matched_preferred_skills": matched_preferred_skills,
            "missing_preferred_skills": missing_preferred_skills,
        },
    }


__all__ = ["DECISION_POLICY_ID", "apply_decision_policy"]
