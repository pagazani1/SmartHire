# Stage 5B: Model Error Diagnosis and Improvement

## Scope and evidence labels

This report diagnoses the saved experimental hybrid model, the Stage 2
aggregate evaluation, and read-only aggregates of the local raw Parquet
datasets. No model was fitted or tuned, no artifact was changed, and the
synthetic resumes were used only to inspect existing inference and
explanations. Raw resume and job-description text is not included.

- **Measured:** direct outputs from the saved model, explainability function,
  evaluation files, or aggregate dataset counts.
- **Interpretation / hypothesis:** a plausible explanation that is not proven
  by the available data.
- **Not available:** a requested comparison that cannot be computed without
  new model fitting or additional labels.

The three synthetic resumes are fictional and cannot validate the classifier
for real-world hiring. Keyword coverage does not demonstrate candidate
competence.

## Root-cause investigation

### Why the text Logistic Regression predicts No Fit for the strong synthetic case

**Measured for the supplied strong PDF and Stage 3B JD:**

- Text LR probabilities: Good Fit **0.150888**, No Fit **0.550818**, Potential
  Fit **0.298293**; its largest score is No Fit.
- The saved text model's raw class logits were Good Fit **-0.658800**, No Fit
  **0.636064**, Potential Fit **0.022737**.
- The strongest positive text-logit term contributions for No Fit were:

  | Term | Contribution |
  |---|---:|
  | `apis` | +0.12124 |
  | `data` | +0.08598 |
  | `sql` | +0.07084 |
  | `python` | +0.04695 |
  | `are looking` | +0.03923 |

- Among the strongest negative No Fit contributions were `learn` (-0.03413),
  `automated tests` (-0.02977), `data driven` (-0.02605), `pandas`
  (-0.01622), and `of machine` (-0.01594). For Potential Fit, positive terms
  included `pandas` (+0.02042), `numpy` (+0.01622), and `ml` (+0.01576), while
  `apis` (-0.06213), `sql` (-0.04841), `backend` (-0.04656), `rest`
  (-0.04200), and `rest apis` (-0.03174) were negative.

**Confirmed representation issue:** `build_combined_text()` lowercases and
concatenates cleaned resume and JD text with a space. The saved
`TfidfVectorizer` uses unigram/bigram features. It has no document-boundary
feature, skill-match feature, or explicit resume-term-by-JD-term interaction.
Thus, terms from either document contribute to the same pooled feature
dimensions. The inference explanation can show a term's signed LR contribution
for the combined input, but cannot attribute that contribution uniquely to a
resume versus the JD.

**Interpretation:** JD language contributes to the No Fit score for this
example: `are looking` and requirement-related vocabulary occur in the
provided JD, and their combined-input terms have positive No Fit
contributions. Frequent terms such as `python`, `sql`, and `apis` are not
reliable matching evidence in this representation; the model has learned
class-conditional weights from their training associations. The current
explanation does not prove which document instance caused a feature value
when the term is present in both.

This is a demonstrated limitation of the representation. It does not by
itself establish why the dataset labels or fitted coefficient signs arose.

### Dataset labels and source-specific patterns

**Measured from the local raw train and test Parquet files:**

- Combined rows: **93,733**, across **643** normalized-resume groups.
- `source` has two values: `generated_smart` and `synthetic_test`.
- Exact normalized resume+JD pairs with more than one distinct label:
  **0**.
- Normalized resume groups occurring with more than one label: **623**.
  Because these groups can pair the same resume with different JDs, this is
  not by itself a label contradiction; compatibility can be JD-dependent.
- Class prevalence differs by source:

  | Source | Good Fit | No Fit | Potential Fit |
  |---|---:|---:|---:|
  | `generated_smart` | 7.24% | 75.38% | 17.38% |
  | `synthetic_test` | 12.20% | 61.69% | 26.12% |

No exact repeated-pair label conflicts were observed in the scan. This does
**not** establish that the labels are semantically correct: the repository
contains no independent gold standard or human adjudication for fit quality.
The source-specific class differences are a measured association, not proof
of source-caused labels or source leakage. Source identity was not held out
as a validation group, and this investigation did not audit source-specific
text templates.

### Is the capped training sample representative?

**Measured using the persisted protocol's `GroupShuffleSplit` seed 2026 and
group/label cap seed 6842, without fitting any estimators:**

- Development: **74,550** rows; capped sample: **6,462** rows (**8.67%**).
- The sample preserves all **1,408** development resume-group/label
  combinations, but keeps at most **5** rows from any combination.
- Development rows per group/label combination: median **39**, 90th
  percentile **138**, maximum **199**.
- Row-level class prevalence shifts substantially:

  | Partition | Good Fit | No Fit | Potential Fit |
  |---|---:|---:|---:|
  | Full dataset | 7.97% | 73.37% | 18.66% |
  | Development rows | 8.31% | 72.96% | 18.74% |
  | Capped development sample | 24.90% | 39.77% | 35.33% |

The sample is therefore **not representative of the development rows by row
frequency or class prevalence**. It preserves group/label support but
downweights group/label combinations that contain many repeated rows and
overrepresents combinations with few rows. The model additionally uses
`class_weight="balanced"`, calculated on the sampled labels.

**Interpretation / risk:** This cap was introduced after full-data TF-IDF
memory exhaustion and is a reproducibility constraint, not a statistically
neutral sample. Its effect on calibration and generalization has not been
isolated. The final holdout remains complete and grouped, but the development
estimator sees this altered distribution.

### Why the hybrid model favors Potential Fit over Good Fit

For the strong case, the existing outputs are:

- Structured LR: Good Fit **0.239967**, No Fit **0.000308**, Potential Fit
  **0.759724**.
- Hybrid: Good Fit **0.316166**, No Fit **0.000209**, Potential Fit
  **0.683624**.
- Detected required coverage: **12/12**.

The hybrid pipeline applies the fitted preprocessing (`SimpleImputer`,
`StandardScaler` for numeric columns, and `OneHotEncoder` for domains) before
its regularized multinomial Logistic Regression. Its final logits are
therefore based on standardized numeric features, encoded domains, and
standardized text-probability inputs; it does not apply a rule mapping
100% coverage to Good Fit.

**Measured local additive logit decomposition for this strong case:**

| Class | Intercept | Sum of structured contributions | Sum of text-probability contributions | Reconstructed logit |
|---|---:|---:|---:|---:|
| Good Fit | -0.318232 | +3.351235 | -0.850186 | 2.182818 |
| No Fit | -0.065222 | -5.985802 | +0.914249 | -5.136776 |
| Potential Fit | +0.383454 | +2.634567 | -0.064064 | 2.953958 |

For this case, Potential Fit's logit exceeds Good Fit's by about **0.771140**.
The relative difference decomposes into approximately **+0.701686** from the
intercept difference, **-0.716668** from the structured-contribution
difference, and **+0.786122** from the text-probability-contribution
difference. These terms sum to the measured logit gap. The structured
features support both Good Fit and Potential Fit; their net contribution is
larger for Good Fit in this input. The transformed text-probability inputs
shift the relative logit toward Potential Fit compared with Good Fit.

Among the largest individual structured terms for this case, standardized
`matched_skill_count` contributes **+3.259042** to Good Fit,
**-4.943125** to No Fit, and **+1.684083** to Potential Fit. Standardized
`matched_required_group_count` contributes **-1.728683** to No Fit and
**+1.442683** to Potential Fit. These are model-logit terms for one input,
not causal effects or probability increments.

This local explanation accounts numerically for the fitted scores; it does
not show that the learned coefficients are substantively correct or that
Potential Fit is the appropriate hiring category.

### Does the fusion model rely excessively on either branch?

The saved pipeline permits an exact local additive decomposition. On the
strong synthetic example, the signed sums show structured features dominate
the Potential Fit logit contribution (**+2.634567**) while text-probability
inputs sum to **-0.064064**; for Good Fit the respective sums are
**+3.351235** and **-0.850186**. For No Fit, they are **-5.985802** and
**+0.914249**.

This is **not enough to conclude that the model globally or excessively
relies on either branch**. The sums include cancellations; structured and
probability features have different meanings; and one synthetic input is not
a representative evaluation. A global branch reliance claim needs grouped
validation analysis, including ablation or carefully designed permutation
tests on development folds. No such analysis was run in this read-only phase.

### OOF training probabilities versus final inference probabilities

The training code generates three-fold `GroupKFold` out-of-fold text
probabilities for the sampled development rows and uses these as fusion
training inputs. The saved text LR used at inference is fit on the complete
6,462-row capped development sample. Thus, OOF fusion-training probabilities
come from fold models trained on subsets, while inference probabilities come
from the final full-sample text model.

**Not available:** neither the per-row OOF probabilities nor their
distribution summaries are persisted in the bundle or saved aggregate
evaluation outputs. Measuring the OOF-versus-final probability distribution
gap would require refitting the OOF text models or adding saved OOF outputs;
that was not done because retraining was prohibited. Consequently, a
distribution mismatch is a plausible risk, not a measured cause of this
synthetic prediction.

The existing grouped evaluation does train fold-specific text models for
validation and scores each fold's validation rows using that fold's text
model. It reports aggregate classification metrics but does not persist the
per-row OOF/validation probability distributions needed for this comparison.

## Potential sources of bias or leakage

### Measured

- The cap changes row-level class prevalence substantially, as shown above.
- Source-specific label prevalence differs (`generated_smart` versus
  `synthetic_test`).
- The grouped split prevents exact normalized-resume overlap between
  development and final holdout; the saved Stage 2 report records zero
  overlap and **129** holdout resume groups.
- No exact normalized resume+JD pair with conflicting labels was found.

### Risks not established as occurring

- Resume-grouping stops exact normalized-resume leakage but does not prove
  that near-duplicate resumes, duplicated templates, or source-specific
  language cannot cross groups.
- Row-level source prevalence differences may let a text model learn source
  or template cues associated with class. This has not been tested with a
  source-held-out evaluation or source-conditioned model analysis.
- Human or synthetic label noise may remain even though exact duplicate
  pairs did not conflict; no independent label audit is available.
- The final fusion training and inference base-text models differ in fit
  scope (OOF fold model versus full capped sample), as described above.

No leakage or label error is asserted without direct evidence.

## Improvement proposals, ranked

Each proposal is an experiment, not a promised improvement. Do not select
models or thresholds using the three synthetic resumes.

### 1. Audit data source and label quality

- **Experiment:** Sample records stratified by source, label, and normalized
  resume group for blinded human review of whether labels match the stated
  resume/JD pair. Review same-resume/different-JD label patterns and
  source-specific templates. Keep raw text access restricted.
- **Expected benefit:** Determine whether text/source correlations or noisy
  labels are limiting the target rather than the classifier alone.
- **Risks:** Human judgments are subjective; source strata may not represent
  future data; access to candidate-like text requires privacy controls.
- **Validation:** Use at least two independent reviewers, agreement and
  adjudication protocol, document annotation criteria, then report
  source-stratified error rates on resume-grouped folds. Do not revise labels
  based only on model disagreement.

### 2. Add explicit resume–JD interaction features

- **Experiment:** Compare matched/unmatched canonical skill groups, counts,
  and coverage with distinct resume and JD skill vectors; preserve
  required/preferred/unclassified distinctions and alternatives. Add
  interaction features only from training-fold data or deterministic
  extraction rules.
- **Expected benefit:** Represent actual overlap rather than pooled
  occurrence of terms. The current strong-case text explanation shows
  requirement vocabulary can push No Fit even when corresponding terms are
  present in the resume.
- **Risks:** Keyword extraction misses synonyms, context, and proficiency;
  heuristic coverage may appear more authoritative than it is; label
  associations can still be weak.
- **Validation:** Compare against current text-only and hybrid baselines on
  identical grouped outer folds and the reserved untouched grouped holdout.
  Report class-wise precision/recall/F1, especially Good Fit and Potential
  Fit, plus source-stratified results and coverage-defined support.

### 3. Separate resume and JD text representations

- **Experiment:** Compare pooled concatenated TF-IDF with separate resume and
  JD TF-IDF blocks, plus explicitly constructed overlap or interaction terms.
  Fit all vocabularies and transformations within each training fold.
- **Expected benefit:** Retain which document contributed each term and allow
  matching features to be learned explicitly.
- **Risks:** Increases dimensionality and memory; may capture spurious
  co-occurrences or source artifacts; separate feature blocks still do not
  guarantee semantic compatibility.
- **Validation:** Use the same group-aware nested CV and held-out group set;
  compare macro-F1, class metrics, calibration, memory, and source-held-out
  stress tests. No holdout-driven feature selection.

### 4. Evaluate the cap and training distribution

- **Experiment:** Compare the current deterministic cap with memory-feasible
  alternatives such as per-group sampling or chunked sparse-vectorization;
  preserve resume groups and label support. A sampling strategy must be
  fixed before final evaluation.
- **Expected benefit:** Determine whether larger or more representative
  development data improves stability and probability behavior.
- **Risks:** Full TF-IDF can exceed available memory; repeated rows may not be
  independent; preserving row prevalence alone can reintroduce group
  domination.
- **Validation:** On development data only, compare row and group/label
  composition, fold variation, per-source performance, and resource use.
  Freeze the selected procedure before evaluating the untouched grouped
  holdout once.

### 5. Test semantic resume–JD similarity as an added feature

- **Experiment:** Compare a separately trained or fixed semantic encoder
  similarity feature against lexical and interaction baselines. If the
  encoder is learned or fine-tuned, fit it only within training folds.
- **Expected benefit:** Potentially recognize paraphrases and related
  experience not captured by exact keyword overlap.
- **Risks:** Similarity can reward topical resemblance without requirement
  satisfaction; pretrained encoders may encode bias, add deployment cost, and
  need licensing/privacy review; cosine similarity is not a fit probability.
- **Validation:** Pre-register the similarity method and compare its
  incremental value on identical grouped folds and untouched holdout. Include
  source and subgroup analysis, calibration, and human-reviewed error cases.

### 6. Compare class-weighting and decision strategies

- **Experiment:** Compare current balanced weights with alternatives and
  class-specific decision strategies using development-only grouped nested
  validation.
- **Expected benefit:** Explore the observed tradeoff among Good Fit,
  Potential Fit, and No Fit errors.
- **Risks:** May increase one class's recall while reducing precision or
  worsening other classes; thresholds can encode consequential policy
  choices.
- **Validation:** Predefine the primary metric and acceptable class-wise
  constraints, report fold distributions and confusion matrices, then lock
  all choices before the final holdout. No rule should force Good Fit based
  on skill coverage.

### 7. Assess probability calibration after the model is stable

- **Experiment:** Evaluate multiclass log loss, Brier score, and
  classwise reliability curves. If calibration is justified, fit the
  calibrator using group-disjoint development predictions only.
- **Expected benefit:** Determine whether reported probabilities correspond
  to empirical frequencies and improve their interpretation.
- **Risks:** Calibration cannot correct ranking errors or label noise and
  can overfit small classes; the current sample and source shifts may make
  calibration unstable.
- **Validation:** Use group-aware out-of-fold predictions for calibration
  development, then evaluate once on the untouched grouped holdout. Report
  calibration by class and source; never calibrate on the final holdout.

## Proposed validation protocol

1. Keep the existing final holdout assignment fixed: combine the prescribed
   raw data, normalize resume identity with `clean_text()`, and use
   `GroupShuffleSplit(test_size=0.2, random_state=2026)`. Assert zero resume
   group overlap.
2. Make all feature, sampling, model, and decision choices inside development
   data using grouped outer folds. Fit text vocabularies, scalers, encoders,
   semantic components if learned, and calibration transforms only on their
   corresponding training portions.
3. Generate fusion base-model probabilities out-of-fold by resume group for
   every training row. Persist aggregate probability-distribution summaries
   and compare fold-OOF against final-model validation probabilities without
   exposing per-resume text.
4. Predefine primary and guardrail metrics. Report accuracy, macro-F1,
   weighted-F1, per-class precision/recall/F1, confusion matrices,
   prediction distributions, probability calibration, fold variation,
   source-stratified metrics, group counts, and overlap checks.
5. Address label/source quality with a documented blinded review protocol.
   Do not assume the `source` field is a label or use it as a predictive
   feature without explicit justification and a source-shift evaluation.
6. Lock the modeling protocol before scoring the untouched grouped holdout
   once. Keep the three synthetic resumes outside training, tuning,
   thresholding, and model selection; use them only for transparent
   post-selection smoke checks.

## Limitations

- The dataset has no explicit, independently verified skill annotations or
  demonstrated ground truth for competence.
- Exact duplicate-pair label conflicts were absent in the aggregate scan,
  but semantic label validity was not auditable from counts.
- Source-specific class prevalence is measured; source-caused artifacts or
  leakage are hypotheses requiring targeted tests.
- The capped training sample materially changes row-level class prevalence
  and representation of repeated group/label cells.
- OOF probability arrays were not saved, so their distribution relative to
  final inference probabilities remains unmeasured.
- The strong-case feature decomposition explains the saved model arithmetic
  for that input only. It does not establish a causal or generally reliable
  explanation.
- The hybrid remains experimental; none of these proposals constitutes
  evidence of improved real-world hiring validity.
