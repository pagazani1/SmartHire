# SmartHire Hybrid Model Validation

## Result

Stage 3 inference validation could not be performed because the Stage 2
evaluator did not save a trained hybrid model or the text-model artifacts
required by the hybrid. The requested fictional strong, partial, and weak
resume examples were also not available as persistent fixtures. No training
was rerun, no models or endpoints were changed, and no predictions were
fabricated.

## Artifact availability

The Stage 2 implementation in `evaluate_structured_models.py` builds and fits
the text Logistic Regression, structured-only classifier, and hybrid
classifier in memory to calculate aggregate evaluation outputs. The final fit
function returns metrics and predictions only. It does not serialize the
fitted estimators, TF-IDF vectorizer, or a hybrid inference bundle.

The `evaluation_results/structured/` directory contains:

- `structured_metrics.json`
- `fold_metrics.csv`
- `feature_availability.csv`
- `final_per_class_metrics.csv`
- `final_class_distributions.csv`
- Three final confusion-matrix CSVs

It does **not** contain a hybrid estimator, text TF-IDF vectorizer, fitted
text Logistic Regression estimator, or saved row-level holdout/synthetic
predictions. The existing files under `models/` are the original application
artifacts (`logistic_regression.joblib`, `tfidf_vectorizer.joblib`, and
`linear_svc.joblib`); they are not the Stage 2 sampled/fold-fitted hybrid
components and were not used as substitutes.

### What must be persisted for reproducible hybrid inference

1. The fitted hybrid preprocessing/classifier `Pipeline`, including its
   numeric imputers/scaler, categorical encoders, feature-column order, and
   class order.
2. The fitted base text Logistic Regression model and its matching fitted
   `TfidfVectorizer`, trained on the same Stage 2 sampled development rows.
3. The exact feature schema/version and the skill-extractor implementation
   version used to derive inference features.
4. Training metadata: dataset/split identity, holdout and sample seeds,
   sampling cap, model hyperparameters, TF-IDF configuration, and the OOF
   protocol used to create hybrid training probabilities.

The hybrid was trained with grouped out-of-fold text probabilities, while
inference would ordinarily use probabilities from the final text model fitted
on all sampled development rows. This train/inference stacking difference
should be documented and addressed or validated before the model is exposed
to an application. Do not pair a newly fit text model with a hybrid estimator
trained against a different text feature space or sampling procedure.

Inference can be reproduced only by rerunning the explicitly configured
training procedure on the same data and saving those fitted components.
That retraining was not performed for this validation task.

## Synthetic fixture audit

No persistent strong-, partial-, and weak-match resume triplet or associated
PDF fixtures were found. Existing PDF tests create generic synthetic PDFs
inside temporary test directories, including a short synthetic Python/Docker
resume; those files are removed by test cleanup. Existing code tests also
contain generic candidate-report sample text, but they do not label it as the
requested strong, partial, or weak validation set. Those examples are not a
reproducible, labeled triplet for this task.

Consequently, no synthetic feature profiles or hybrid predictions are
reported. The previously supplied user summary (12 required groups with
coverage 1.0 for a strong example, and 4 matched groups with coverage 0.3333
for a partial example) is not independently reproducible from repository
fixtures. A weak example was not supplied in a persistent fixture either.

| Requested example | Fixture found? | Required skill groups | Matched groups | Missing groups | Required coverage | Hybrid label/scores |
|---|---|---:|---:|---:|---:|---|
| Strong | No | Not available | Not available | Not available | Not available | Not available |
| Partial | No | Not available | Not available | Not available | Not available | Not available |
| Weak | No | Not available | Not available | Not available | Not available | Not available |

No raw resume or job-description text is included in this report.

## Stage 2 evaluation context

These are aggregate results from the Stage 2 grouped evaluation, not
predictions for the unavailable synthetic examples:

| Model | Final grouped-holdout accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Text Logistic Regression | 0.6177 | 0.4896 | 0.6544 |
| Structured-only | 0.5621 | 0.4108 | 0.6119 |
| Hybrid | 0.6567 | 0.5355 | 0.6914 |

The final holdout used 129 resume groups and had zero group overlap with the
development set. Stage 2 training used a capped sample of 6,462 development
rows after a full-row text-vectorizer fit failed from memory exhaustion. The
reported evaluation therefore measures that capped-training protocol. These
results do not verify inference for the strong, partial, or weak synthetic
examples and do not establish real-world generalization.

## Interpretation and limitations

- The apparent No Fit outputs reported for synthetic resumes cannot be
  reproduced or compared against the Stage 2 hybrid because those inputs,
  feature rows, predictions, and fitted hybrid artifacts are unavailable.
- A skill-coverage value is an extractor-derived keyword feature. It neither
  establishes actual candidate competence nor dictates a class label.
- No score or expected class has been inferred from coverage, and no
  threshold or override was introduced.
- No hybrid-specific inference test was added: there is no persisted
  estimator bundle to load. Existing feature-extraction and classifier-fit
  tests validate components, not saved-artifact inference.
- The production FastAPI endpoint and existing Logistic Regression
  prediction model remain unchanged.

## Validation actions performed

- Inspected the structured feature/model implementation and the evaluation
  training/saving path.
- Checked `models/` and `evaluation_results/structured/` for fitted model
  artifacts and checked test/PDF fixtures for the named synthetic examples.
- Did not run training, serialize models, modify existing evaluation
  artifacts, change the API, or print raw resume/job-description content.
- No synthetic prediction or artifact-based inference test could be run
  because the required inputs and fitted hybrid components were absent.
