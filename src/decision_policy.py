"""Explainable recruiter-facing decision policy over existing model evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DECISION_POLICY_ID = "required_skill_group_coverage_v2"


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
    seen_groups: set[frozenset[str]] = set()
    for group in value:
        if (
            not isinstance(group, Iterable)
            or isinstance(group, (str, Mapping))
        ):
            return None
        group_names = list(group)
        if any(not isinstance(name, str) for name in group_names):
            return None
        names = {name.strip() for name in group_names if name.strip()}
        if not names:
            return None
        normalized_key = frozenset(name.casefold() for name in names)
        if normalized_key in seen_groups:
            continue
        seen_groups.add(normalized_key)
        groups.append(names)
    return groups


def apply_decision_policy(
    model_prediction: dict[str, Any],
    skill_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Classify by exact required-group coverage while preserving model output.

    At least 85% coverage is Good Fit, at least 50% and below 85% is
    Potential Fit, and below 50% is No Fit. Any matched alternative satisfies
    its group. Missing or absent structured evidence falls back to the model.
    """
    model_label = str(model_prediction["label"])
    model_probs = dict(model_prediction["probs"])
    required_groups = _groups(skill_analysis.get("required_skill_groups"))
    resume_skills_value = skill_analysis.get("resume_skills")
    resume_skill_values = (
        list(resume_skills_value)
        if isinstance(resume_skills_value, Iterable)
        and not isinstance(resume_skills_value, (str, Mapping))
        else None
    )
    resume_skills = (
        {skill.strip() for skill in resume_skill_values}
        if resume_skill_values is not None
        and all(
            isinstance(skill, str) and skill.strip()
            for skill in resume_skill_values
        )
        else None
    )
    normalized_resume_skills = (
        {skill.casefold() for skill in resume_skills}
        if resume_skills is not None
        else None
    )
    evidence_available = (
        required_groups is not None
        and bool(required_groups)
        and normalized_resume_skills is not None
    )

    matched_required_groups: list[set[str]] = []
    missing_required_groups: list[set[str]] = []
    if evidence_available:
        for group in required_groups:
            if {skill.casefold() for skill in group} & normalized_resume_skills:
                matched_required_groups.append(group)
            else:
                missing_required_groups.append(group)

    required_count = len(required_groups) if required_groups is not None else 0
    matched_count = len(matched_required_groups)
    required_coverage = (
        matched_count / required_count if evidence_available else None
    )
    required_coverage_percentage = (
        required_coverage * 100 if required_coverage is not None else None
    )

    if evidence_available:
        if matched_count * 100 >= required_count * 85:
            final_label = "Good Fit"
            classification_reason = "coverage is at least 85%"
        elif matched_count * 100 >= required_count * 50:
            final_label = "Potential Fit"
            classification_reason = "coverage is at least 50% and below 85%"
        else:
            final_label = "No Fit"
            classification_reason = "coverage is below 50%"
        percentage_text = f"{required_coverage_percentage:.2f}".rstrip("0").rstrip(".")
        explanation = (
            f"Candidate matched {matched_count} out of {required_count} required "
            f"skill groups ({percentage_text}%). Since the {classification_reason}, "
            f"the candidate is classified as {final_label}."
        )
        evidence_status = "available"
    else:
        final_label = model_label
        evidence_status = (
            "no_required_groups" if required_groups == [] else "unavailable"
        )
        if evidence_status == "no_required_groups":
            explanation = (
                "No extractable required skill groups were identified; the final "
                "category falls back to the text-model prediction. Required skill "
                "coverage and missing required groups cannot be determined."
            )
        else:
            explanation = (
                "Reliable required-skill evidence is unavailable; the final "
                "category falls back to the text-model prediction. Required skill "
                "coverage and missing required groups cannot be determined."
            )

    preferred_groups = _groups(skill_analysis.get("preferred_skill_groups"))
    preferred_matched_groups: list[set[str]] = []
    preferred_missing_groups: list[set[str]] = []
    if preferred_groups is not None and normalized_resume_skills is not None:
        for group in preferred_groups:
            if {skill.casefold() for skill in group} & normalized_resume_skills:
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

    if evidence_available:
        explanation += (
            f" The text model predicted {model_label}; its probabilities are unchanged."
        )
    if preferred_groups:
        explanation += (
            f" Preferred groups matched: {len(preferred_matched_groups)}"
            f"/{len(preferred_groups)}; preferred skills do not affect required"
            " coverage."
        )
    explanation += " Skill matches are keyword evidence, not proof of proficiency."

    return {
        "label": final_label,
        "probs": model_probs,
        "model_label": model_label,
        "model_probs": model_probs,
        "decision_policy": DECISION_POLICY_ID,
        "decision_explanation": explanation,
        "decision_evidence": {
            "structured_evidence_available": evidence_available,
            "required_skill_evidence_status": evidence_status,
            "required_skill_group_count": required_count,
            "matched_required_group_count": matched_count,
            "missing_required_group_count": len(missing_required_groups),
            "required_skill_coverage": required_coverage,
            "required_skill_coverage_percentage": required_coverage_percentage,
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
