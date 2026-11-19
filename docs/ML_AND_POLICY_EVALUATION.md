# ML model and decision policy: evaluation and decisions

This document separates what was **measured**, what is **reasoned**, and what is **not yet known**.

## 1. What was measured (from the committed reports)

Source: `evaluation_results/` (grouped holdout, 18,913 pairs; produced before this review).

| Predictor | Accuracy | Macro-F1 |
|---|---|---|
| Always predict "No Fit" | 0.731 | 0.282 |
| Logistic Regression (production model) | 0.722 | 0.606 |
| LinearSVC (not used in production) | 0.729 | 0.569 |

- The model's accuracy is **slightly below** the trivial "always No Fit" baseline, because
  73% of rows are "No Fit". Its macro-F1 is far higher, so it has real signal, but accuracy alone
  must never be quoted as evidence that it works.
- Per class (Logistic Regression F1): Good Fit 0.511, Potential Fit 0.478, No Fit 0.829.
  "Potential Fit" is the weakest class: of 3,475 true Potential Fit rows, 1,128 were predicted No Fit
  and 448 Good Fit.
- Confidence carries information: accuracy was 0.909 when the top probability was at least 0.75
  (7,263 rows), 0.648 between 0.50 and 0.75 (8,785 rows) and 0.475 below 0.50 (2,865 rows).

Not measured: any comparison of the **policy** with these labels. The repository does not contain
the labelled rows, so that evaluation could not be run here (see section 4).

## 2. Reasoned limitations (not measured)

- The model sees `resume + " " + job description` as one bag of words. It cannot represent *which
  required skills the resume satisfies*, only which words appear anywhere in the pair. This is why a
  strong match can still be predicted "No Fit". The skill-group matcher does this comparison
  directly, which is why the policy, not the model, decides the final label.
- The labels have no documented independent adjudication, so even a perfect model would only
  reproduce the dataset's conventions.

## 3. Decisions taken in this phase

| Topic | Decision | Reason |
|---|---|---|
| Decision policy and thresholds | **Unchanged** (`required_skill_group_coverage_v2`, 85% / 50%) | No labelled evidence was available to justify a change. |
| Model retraining / replacing TF-IDF | **Not done** | No training data in the repository; an unmeasured "improvement" would be a guess. |
| Model role in the UI | Shown as **"Text-model reference"**, never as the decision | It is a weak signal and can contradict the policy. |
| Model/policy contradiction | UI explains that the policy decides and why the model may differ | The previous display showed two labels without context. |
| Fallback (no usable required skills) | Behaviour unchanged, but the UI shows a **"Manual review needed"** banner | The final label is then only the model's guess; recruiters must know. |
| Evaluation | Added `scripts/evaluate_decision_policy.py` | Lets the model and the policy be scored separately on your own labelled file. |

A different fallback (for example "Insufficient evidence" instead of the model label) would change
API behaviour and needs a new policy id. It is recommended as a future change once the evaluation
below has been run.

## 4. How to produce the missing evidence

```bash
python -m scripts.evaluate_decision_policy --data data/raw/raw_test.parquet --sample 2000
```

Columns required: `resume`, `jd`, `label`. Output goes to `evaluation_results/policy_validation/`:
accuracy and macro-F1 for the model, the policy and the baseline; the share of rows where the policy
fell back to the model; model-vs-policy disagreement by direction; and a threshold sensitivity sweep
with the current 85% / 50% marked and a bootstrap interval. Reports contain counts only, no text.

Interpretation rules: a different threshold pair is only worth discussing if it beats the current one by
more than the bootstrap interval **and** holds on a second file. Any change requires a new policy id,
a documented migration plan and a version bump in stored screening results.

## 5. Known limitations that remain

- Skill extraction is keyword-based. It cannot judge proficiency, recency or context.
- Policy quality on real hiring outcomes is unknown.
- Experiments (`models/hybrid_experimental`, `stage5c_experimental`, structured models) are kept
  untouched for reference and are not used in production.
