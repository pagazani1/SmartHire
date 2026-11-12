# Production Fit Decision Policy

## Policy

The existing text-model prediction remains available as `model_label` and
`model_probs`. Its probabilities are not altered. When required skill groups
and resume skill evidence are reliable, the recruiter-facing `label` is based
on the exact required-group coverage:

```text
coverage_percentage = matched_required_skill_groups
                     / total_required_skill_groups * 100
```

The thresholds are applied to the unrounded matched/total counts:

- **Good Fit:** 85% through 100%, inclusive.
- **Potential Fit:** 50% through below 85%.
- **No Fit:** below 50%.

For example, 17/20 (85%) is Good Fit; 16/19 (about 84.21%) is Potential Fit;
and 9/19 (about 47.37%) is No Fit. Display rounding does not affect a decision.

An alternative such as “FastAPI or Flask” is one requirement group. Matching
either alternative satisfies the group. Duplicate skill names within a group
and duplicate requirement groups are counted only once. Preferred groups are
reported separately and never contribute to required coverage.

If there are no classified required groups, or if required groups or resume
skill evidence are unavailable or malformed, the final label falls back to
the existing text-model label. Coverage is returned as `null`, and the
response distinguishes unavailable evidence from a reliable evaluation with
missing groups. The model probabilities and prediction remain unchanged in
all cases.

The response includes the matched, total, and missing required groups,
required coverage as both a 0–1 ratio (`required_skill_coverage`) and
percentage (`required_skill_coverage_percentage`), and a short explanation
of the final category.

## Limitations

This is a deterministic recruiter-facing screening policy, not a learned or
validated measure of candidate suitability. Skill extraction detects
keywords and limited alternative phrasing; it cannot establish proficiency,
experience depth, recency, or context. A missing keyword is not proof that a
candidate lacks a skill, and a matched keyword is not proof of competence.
False or incomplete extraction can therefore affect the rule-based label.

Unclassified requirements are not part of the coverage denominator. Preferred
skills and text-model probabilities are shown for context but do not override
the explicit required-coverage rule. Human review remains necessary.
