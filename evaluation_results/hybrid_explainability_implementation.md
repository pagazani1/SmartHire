# Hybrid Explainability Implementation

## Architecture

Added `src/hybrid_explainer.py` as a read-only layer over the saved
experimental bundle. It calls the existing `predict_hybrid()` interface and
returns that prediction unchanged alongside class-aligned logit explanations.
It does not fit, tune, save, or mutate any estimator, and it is not connected
to the production API.

The explanation uses the existing inference transformations:

1. Structured features are created by `extract_structured_features()` and
   ordered according to the saved bundle schema.
2. The resume and JD are normalized and joined by `build_combined_text()`;
   the fitted vectorizer transforms this string.
3. Text LR class probabilities are combined with the structured inputs using
   `build_hybrid_fusion_features()`.
4. The saved hybrid pipeline's fitted preprocessing step imputes and scales
   numeric columns and one-hot encodes categorical domain columns. Its
   transformed feature names and values are used directly with the fitted LR
   coefficients.
5. Class rows are mapped from each estimator's `classes_` to the stable label
   order in bundle metadata; class order is not guessed or assumed.

## Explanation methodology

For each class, the hybrid explanation reports the fitted classifier
intercept and every transformed input term:

`contribution = transformed feature value * fitted class coefficient`

This includes each transformed structured feature and each of the three
text-model probability inputs. Each contribution includes its class, feature
name, feature type, transformed value, coefficient, and signed contribution.
The intercept plus all contributions reconstructs the saved hybrid LR's
`decision_function()` logit. The implementation raises
`HybridExplanationError` if the pipeline, feature schema, coefficients, or
reconstruction do not agree.

The optional text-model explanation uses the fitted vectorizer and the actual
transformed combined resume/JD input. It reports up to `top_k_text_terms`
positive and negative term contributions for each class, with the term,
TF-IDF value, coefficient, and contribution. The complete text logit is also
reconstructed and verified, although only the highest-magnitude signed terms
are returned in those lists.

These are additive terms in model logits, **not causal effects or direct
increments to class probabilities**. The resulting probabilities are a
joint softmax over class logits.

## Example output structure

```json
{
  "class_order": ["Good Fit", "No Fit", "Potential Fit"],
  "predicted_label": "Potential Fit",
  "hybrid_probabilities": {
    "Good Fit": 0.3,
    "No Fit": 0.1,
    "Potential Fit": 0.6
  },
  "hybrid_logit_explanations": {
    "Potential Fit": {
      "intercept": 0.2,
      "contributions": [
        {
          "class": "Potential Fit",
          "feature_name": "numeric__required_skill_coverage",
          "feature_type": "structured",
          "transformed_value": 1.1,
          "coefficient": 0.4,
          "contribution": 0.44
        },
        {
          "class": "Potential Fit",
          "feature_name": "numeric__text_probability__Good Fit",
          "feature_type": "text_probability",
          "transformed_value": 0.3,
          "coefficient": -0.2,
          "contribution": -0.06
        }
      ],
      "reconstructed_logit": 0.58,
      "model_logit": 0.58,
      "reconstruction_error": 0.0,
      "reconstruction_verified": true
    }
  },
  "text_model_explanations": {
    "Potential Fit": {
      "intercept": 0.1,
      "model_logit": 0.5,
      "reconstructed_logit": 0.5,
      "reconstruction_error": 0.0,
      "top_positive_terms": [],
      "top_negative_terms": []
    }
  }
}
```

Values above illustrate the schema only; they are not measured predictions.
The result also contains the unchanged `predict_hybrid()` output and notes
clarifying the scope of the explanations.

## Validation

Added `tests/test_hybrid_explainer.py` with six focused tests. Along with the
existing hybrid prediction tests, `python -m unittest discover -s tests -p
'test_hybrid*.py'` passed **14 tests**.

The tests cover class and feature alignment, contribution signs, logit
reconstruction, TF-IDF contribution calculations, unknown domain categories,
missing coverage values, invalid inputs, unavailable linear-pipeline
handling, and exact prediction output equality before and after explanation.

The three supplied synthetic PDF cases were also run through the saved bundle
and explainer as a non-training smoke check. For all three cases, predicted
labels and hybrid probability dictionaries were exactly equal to the existing
inference outputs. For all three classes in each model:

- Maximum absolute hybrid-logit reconstruction error across the three cases:
  `8.881784197001252e-16`.
- Maximum absolute text-logit reconstruction error across the three cases:
  `3.3306690738754696e-16`.

These values are within floating-point rounding error and the configured
reconstruction tolerances (`atol=1e-10`, `rtol=1e-9`). The synthetic cases
were not used to fit or tune anything.

## Limitations

- Coefficient contributions explain this fitted linear model in its
  transformed representation. They do not prove that a feature caused a
  class prediction, establish the correctness of a training label, or show
  what would happen under a real-world intervention.
- A positive logit contribution is not a probability-point increase.
- Structured categorical values appear as one-hot encoded transformed
  columns, not necessarily as one intuitive domain-level explanation.
- Missing numeric values are represented after the saved pipeline's
  imputation/scaling transformation; unknown categories follow the fitted
  encoder's `handle_unknown="ignore"` behavior.
- Text terms are TF-IDF features over combined resume and JD text. They are
  separate from the rule-based skill extractor and do not prove skill or
  competence.
- The model and skill features remain experimental and based on capped
  training data with heuristic features. These explanations do not validate
  the model for real hiring decisions.

## Privacy and human review

The explainer does not include or persist the raw resume or JD text. However,
top TF-IDF terms are derived from supplied text and may themselves contain
sensitive or identifying words. Treat explanation results as sensitive
candidate data: restrict access, avoid logging or unnecessary retention, and
review displayed terms before exposing them to a recruiter.

Use explanations only to support human review. They must not be presented as
proof of competence or used as an automatic employment decision. A recruiter
should be able to inspect the underlying evidence and challenge an incorrect
keyword match or model output.
