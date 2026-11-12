# Production Fit Decision Policy

## Policy

The existing text-model prediction remains available as `model_label` and
`model_probs`. Its probabilities are not altered. The recruiter-facing `label`
is assigned by the following rule when the existing skill extractor finds at
least one classified required-skill group in the job description:

1. **Good Fit:** every required group is matched by at least one detected
   resume skill.
2. **Potential Fit:** at least one, but not every, required group is matched.
3. **No Fit:** no required group is matched.

An alternative such as “FastAPI or Flask” is one group. Matching either
alternative satisfies that group. Preferred groups, matched/missing skill
names, and the original text-model result are returned as supporting evidence
in the decision explanation; preferred skills do not compensate for an unmet
required group and do not change these category rules.

If the JD has no classified required groups, or structured requirement
evidence is unavailable, the final label falls back to the existing text-model
label. The response still includes the available skill evidence and states
that the structured policy could not determine a category.

## Limitations

This is a deterministic recruiter-facing screening policy, not a learned or
validated measure of candidate suitability. Skill extraction detects
keywords and limited alternative phrasing; it cannot establish proficiency,
experience depth, recency, or context. A missing keyword is not proof that a
candidate lacks a skill, and a matched keyword is not proof of competence.
False or incomplete extraction can therefore affect the rule-based label.

Complete coverage means only that every classified required group had a
detected keyword match. It must not be interpreted as a hiring recommendation.
Unclassified requirements are not part of the coverage denominator. Preferred
skills and text-model probabilities are shown for context but do not override
the explicit required-coverage rule. Human review remains necessary.
