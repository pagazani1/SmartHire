# Hybrid Model Explainability Investigation

## Scope and evidence

This is a read-only investigation of the saved experimental hybrid model and
the three fictional synthetic resume evaluations previously recorded in
`hybrid_synthetic_validation.md`. The three PDF fixtures were passed through
the existing text extractor and `predict_hybrid()` with the same Python
Backend Developer Intern job description. Only derived skill and model
outputs are reported here; extracted resume text and personal details are
omitted.

The fitted bundle was inspected but not retrained, tuned, or saved. The
production API, existing model artifacts, and prior evaluation results were
not changed. No tests were run because this was a read-only investigation.

Measured outputs describe these three synthetic inputs only. They are not
evidence of real-world hiring performance, and keyword matches do not
establish candidate competence.

## Architecture and prediction flow

1. **Resume input:** For these PDF tests, `extract_text_from_pdf()` extracts
   the page text. `predict_hybrid()` accepts the resulting resume text and the
   job description as non-empty strings; the inference function itself does
   not read PDFs.
2. **Bundle load and validation:** The fitted experimental bundle contains a
   text vectorizer, text Logistic Regression model, structured-feature
   Logistic Regression pipeline, hybrid fusion Logistic Regression pipeline,
   class order, and feature schema. Inference validates the bundle format,
   experiment ID, scikit-learn compatibility, model classes, and feature
   ordering.
3. **Structured skill evidence and features:** The skill extractor searches a
   fixed vocabulary using case-insensitive patterns, aliases, and limited
   context rules for negation and aspirational mentions. It canonicalizes
   detected skills, classifies JD skills, and groups explicit alternatives.
   `extract_structured_features()` converts these results into counts,
   coverage and availability values, and domain fields. It does not use the
   model label.
4. **Text representation:** `build_combined_text()` applies `clean_text()` to
   resume and JD text separately, then joins them. Cleaning lowercases,
   replaces double quotes, and collapses whitespace. The saved TF-IDF
   vectorizer transforms this combined text; it is not refit at inference.
   Its Logistic Regression model produces class probabilities, aligned to
   the bundle's stable label order.
5. **Fusion:** The structured features and the three text-model class
   probabilities are concatenated in the recorded order. The hybrid
   classifier pipeline imputes numeric values, scales numeric columns, and
   one-hot encodes categorical domain columns before Logistic Regression.
   Training generated grouped out-of-fold text probabilities for fusion
   fitting; inference supplies probabilities from the final saved text model.
6. **Output:** The hybrid classifier returns its class probabilities and
   `predict()` label. The public `predict_hybrid()` result includes skill
   evidence, raw structured feature values, text-model probabilities, hybrid
   probabilities, label, and experiment ID. It does not return structured-only
   classifier probabilities or per-feature contributions.

## Three-resume comparison

The structured-only probabilities below are from the saved model's
`structured_model` pipeline using the same structured feature generator.
Probabilities are rounded to six decimal places and are model outputs, not
calibrated hiring probabilities.

| Synthetic case | Structured summary | Required coverage | Text LR (Good / No / Potential) | Structured LR (Good / No / Potential) | Hybrid (Good / No / Potential) | Hybrid label |
|---|---|---:|---|---|---|---|
| Strong | Resume skills 13; matched skills 13; required groups 12; matched/missing required groups 12/0 | 1.000000 | 0.150888 / 0.550818 / 0.298293 | 0.239967 / 0.000308 / 0.759724 | 0.316166 / 0.000209 / 0.683624 | Potential Fit |
| Partial | Resume skills 5; matched skills 4; required groups 12; matched/missing required groups 4/8 | 0.333333 | 0.093778 / 0.595668 / 0.310554 | 0.076593 / 0.345060 / 0.578347 | 0.070920 / 0.434505 / 0.494574 | Potential Fit |
| Weak | Resume skills 0; matched skills 0; required groups 12; matched/missing required groups 0/12 | 0.000000 | 0.101736 / 0.511482 / 0.386782 | 0.005540 / 0.934265 / 0.060195 | 0.006160 / 0.936966 / 0.056874 | No Fit |

Other structured feature values:

| Feature | Strong | Partial | Weak |
|---|---:|---:|---:|
| JD skill count | 13 | 13 | 13 |
| Required skill group count | 12 | 12 | 12 |
| Matched required group count | 12 | 4 | 0 |
| Missing required group count | 0 | 8 | 12 |
| Required coverage available | 1 | 1 | 1 |
| Preferred skill group count | 0 | 0 | 0 |
| Matched preferred group count | 0 | 0 | 0 |
| Missing preferred group count | 0 | 0 | 0 |
| Preferred coverage | unavailable (`null`) | unavailable (`null`) | unavailable (`null`) |
| Preferred coverage available | 0 | 0 | 0 |
| Missing skill count | 0 | 9 | 13 |
| Unclassified JD skill count | 0 | 0 | 0 |
| Domain compatibility feature | 0 | 0 | 0 |
| Resume/JD domain values | missing / missing | missing / missing | missing / missing |

The JD has 13 detected canonical skills but 12 required groups because
FastAPI and Flask are represented as one alternative group. The strong case
has 13 matched skill names, since its detected resume includes both framework
alternatives. A `domain_compatible` value of zero here means the comparison
could not be established: both domains were left missing. It does not show
that the resume and job domains are incompatible. Preferred coverage is
unavailable because the JD produced no preferred groups.

## Why the strong case received Potential Fit

**Measured facts:**

- Its heuristic skill analysis matched all 12 required groups, giving
  `required_skill_coverage = 1.0`.
- The text Logistic Regression's largest score was **No Fit (0.550818)**.
- The structured-only classifier's largest score was **Potential Fit
  (0.759724)**.
- The fusion model's largest score was **Potential Fit (0.683624)**, compared
  with **Good Fit (0.316166)** and **No Fit (0.000209)**.

The final label is the hybrid Logistic Regression result over its learned
fusion inputs; it is not selected by a rule that maps full skill coverage to
Good Fit. The text and structured branches disagree, and the fused output
places Potential Fit above Good Fit. Full keyword coverage therefore does
not force a Good Fit result.

**Not established by these observations:** They do not show which individual
feature caused Potential Fit to win, or why the training data associates this
particular combination with that class. The class probabilities and feature
values alone are not a per-feature explanation.

## What can and cannot be explained now

### Directly available

- The input processing steps and deterministic structured features listed
  above.
- Matched and missing keyword skill groups and their limitations.
- Text-model, structured-model, and hybrid class probabilities, when each
  model is queried from the saved bundle.
- The final predicted class and the comparison of the branch-level outputs.
- Fitted Logistic Regression coefficients are present in the saved estimators.
  Inspection found coefficient shapes of **3 × 20,000** for text LR,
  **3 × 40** for structured LR, and **3 × 43** for hybrid LR. The
  vectorizer exposes its vocabulary, and the structured preprocessing
  pipelines expose transformed feature names.

### Not currently returned or attributable by the public interface

- `predict_hybrid()` does not return the structured-only probabilities,
  coefficient values, feature weights, token weights, or local explanation.
- The output does not attribute the hybrid label to individual skills, counts,
  domains, or text tokens.
- A skill occurrence is evidence of a text match under the extractor's rules,
  not evidence of actual experience, proficiency, or competence.
- The fact that a feature is present alongside a prediction is not evidence
  that it caused that prediction.

The stored Logistic Regression weights make a transparent *logit
decomposition* technically possible: transform an example using the fitted
pipeline and combine each transformed value with its class coefficient. The
hybrid pipeline's weights apply to scaled numeric and one-hot encoded
structured features plus the text branch's class probabilities. The text
model's weights apply to TF-IDF terms. Such a decomposition would explain
additive contributions to model logits for that exact representation, not
causal effects or additive changes in final probabilities. Softmax
probabilities depend jointly on all class logits. No such decomposition is
currently implemented or emitted by the inference interface.

## Technical limitations

- Both text and skill analysis use surface text. The vectorizer has no explicit
  resume/JD interaction features; it sees a normalized concatenation, so term
  presence does not itself encode whether a resume claim satisfies a
  particular requirement.
- The skill engine is vocabulary- and rule-based. It can miss synonyms or
  context and can count mentions without determining their depth or truth.
- The structured classifier uses heuristic features without gold skill
  annotations. Its coverage values inherit the extractor's limitations.
- Domains were not supplied in these tests, so domain compatibility was not
  informative.
- The text model favors No Fit for all three synthetic cases, while the
  structured model favors Potential Fit for the strong and partial cases.
  This branch disagreement is measured; it does not identify a causal
  explanation.
- Fusion training used grouped out-of-fold text probabilities, while
  inference uses probabilities from the final text model fitted on the capped
  development sample. This is the implemented protocol; the effect of any
  difference in probability distributions has not been measured here.
- The model is experimental and trained on a capped sample. Three fictional
  resumes do not validate it for real-world hiring.

## Recommendations for a future explainability implementation

1. Add a read-only explanation function that reuses the saved bundle and
   inference transformations; do not alter model fitting or predictions.
2. For the hybrid LR, calculate and report per-class additive logit
   contributions from each transformed structured feature and each text
   probability input, plus the intercept. Include the transformed feature
   value, coefficient, contribution, and class. Do not label these as causal
   effects or as direct probability increments.
3. For the text LR, optionally show top positive and negative TF-IDF terms per
   class, mapped through the fitted vocabulary and actual vectorized input.
   Keep this separate from skill-extractor evidence because the two
   representations are not equivalent.
4. Clearly distinguish deterministic skill evidence, branch scores, and
   model-weight-based attribution in any recruiter-facing display. Include
   missing-domain and unavailable-coverage states rather than interpreting
   them as negative evidence.
5. Validate any explanation output against the model's logits and predicted
   probabilities, and test stability and failure cases before exposing it to
   recruiters. Continue to require human review; do not use synthetic
   examples as validation evidence for employment decisions.
