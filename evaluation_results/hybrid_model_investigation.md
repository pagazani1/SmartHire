# SmartHire Hybrid Classification Investigation

## Scope and evidence labels

This is a read-only investigation of the current data schema, training and
inference implementation, saved evaluation outputs, and skill extractor.
Dataset checks read only labels and domain columns or calculated aggregate
counts; no resume or job-description text was printed or saved. No model was
retrained and no source, model, dataset, or existing evaluation output was
changed.

- **Measured / confirmed** means directly observed in source or calculated
  from the existing prediction/evaluation artifacts and dataset metadata.
- **Likely cause / hypothesis** means plausible but not proven by the
  available artifacts.
- Recommendations below are proposals, not measured performance gains.

## Executive summary

The reported synthetic classifications are plausible given the current
feature design, but this investigation cannot verify those specific examples:
their texts, predictions, and probabilities are not present in the saved
evaluation artifacts.

The measured results do **not** show Logistic Regression predicting No Fit
more often than its prevalence in the held-out set. It predicted No Fit for
63.95% of rows versus 73.07% actual No Fit. However, No Fit is still the most
frequent predicted class in absolute terms, and it remains the class with the
highest F1. LinearSVC predicted No Fit at 74.32%, close to the 73.07% actual
share.

The current classifier sees a single TF-IDF bag of words made by concatenating
resume and JD text. It has no explicit matched-skill, coverage, experience, or
resume/JD semantic-similarity features. The raw dataset has useful label,
resume-domain, JD-domain, and source columns, but no explicit skill
annotations. The existing skill extractor can derive a bounded set of
keywords, but does not judge skill requirement strength or experience.

## 1. Dataset and labels

### Confirmed dataset structure

`src/data_loader.py` reads local `raw_train.parquet` and `raw_test.parquet`
when both exist, otherwise downloads the configured Hugging Face dataset and
caches both splits. The observed raw schema is:

| Column | Role |
|---|---|
| `resume` | Resume text |
| `jd` | Job description text |
| `label` | Classification target |
| `source` | Source metadata |
| `resume_domain` | Resume domain/category |
| `jd_domain` | Job-description domain/category |

Across both raw splits there are **93,733 rows**: **68,774 No Fit**,
**17,489 Potential Fit**, and **7,470 Good Fit**. The official raw split has
80,017 train rows and 13,716 test rows. Both splits had zero missing values
among `label`, `resume_domain`, and `jd_domain` in the aggregate schema check.

There are **643 distinct resume texts** and **351 distinct JD texts** in the
combined data. The same resume text commonly appears with more than one label:
623 resume-text groups have multiple labels, including 484 with all three
labels. Likewise, 343 JD-text groups have multiple labels, including 289 with
all three labels. Each exact resume/JD pair appeared once in this dataset
check, so there is no repeated-pair evidence here from which to estimate
same-pair label disagreement.

### Domain information

The combined dataset contains **12 distinct `resume_domain` values**, **12
distinct `jd_domain` values**, and **134 distinct domain pairs**. All 12
resume-domain groups and all 12 JD-domain groups contain more than one label;
88 of the 134 domain pairs contain more than one label. Thus domain columns
are available and may provide useful context, but neither a domain alone nor
most domain pairs uniquely determine the label. The current TF-IDF classifier
does not include these domain columns in its text features. `src/evaluate.py`
does use `resume_domain` in a separate domain-majority baseline.

### Label reliability and sufficiency

The data provides a substantial number of labeled resume/JD rows and supports
supervised experiments. The repeated resumes and JDs with different labels
also demonstrate that compatibility depends on the pairing, not just on a
resume or JD considered independently.

However, the inspected schema and reports do not document label definitions,
annotation instructions, annotator agreement, label provenance by `source`,
or a validation process. Therefore the labels' reliability and the semantic
meaning of the boundary between Good Fit, Potential Fit, and No Fit cannot be
confirmed. Dataset size alone does not establish label quality. Before using
the labels to train a hybrid model, confirm how the categories were assigned
and whether “required” versus “preferred” skills are represented by the
labeling rubric.

## 2. Current representation and training

`src/preprocessing.py` applies `clean_text()` independently to resume and JD,
then builds:

```text
resume_clean + " " + jd_clean
```

`get_tfidf_features()` uses `TfidfVectorizer` with 20,000 maximum features,
unigrams and bigrams, `min_df=2`, `max_df=0.95`, and sublinear term frequency.
It fits on the training text in the standard pipeline and transforms the test
text. `src/models.py` uses Logistic Regression with `max_iter=1000`,
`class_weight="balanced"`, and `random_state=42`. The saved unseen-resume
evaluation uses those same LR settings and fits a fresh vectorizer on that
evaluation split's training rows.

The combined representation retains words from both texts, but it does not
keep them as explicitly separate feature channels and does not directly
compute “JD requires skill X and resume has skill X.” TF-IDF counts weighted
terms and short n-grams; it does not by itself encode semantic equivalence,
negation, whether a JD skill is mandatory, years of experience, or the
relationship between a specific resume claim and a specific requirement.
There is no explicit cross-attention or pairwise skill-match feature in this
model.

`predict.py` and `src/resume_analyzer.py` reuse this same preprocessor and
saved Logistic Regression model. `analyze_resume()` adds a separate skill
summary, but that summary is returned for display and is not fed into the
classifier.

## 3. Measured evaluation evidence

The saved `predictions.csv` contains 18,913 unseen-resume test rows. The
evaluation used a single `GroupShuffleSplit(test_size=0.2, random_state=42)`
over the combined raw train/test records, grouping by cleaned resume text:
74,820 training rows and 18,913 test rows, with 514 train resume groups, 129
test groups, and zero group overlap. Test label supports were 1,618 Good Fit,
13,820 No Fit, and 3,475 Potential Fit.

### Model-level scores

| Model | Accuracy | Macro-F1 | Incorrect rows | Predicted No Fit share |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.721832 | 0.605955 | 5,261 | 63.95% |
| LinearSVC | 0.729287 | 0.568731 | 5,120 | 74.32% |
| Actual test labels | — | — | — | 73.07% |

### Logistic Regression per-class scores

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Good Fit | 0.432479 | 0.625464 | 0.511369 | 1,618 |
| No Fit | 0.888053 | 0.777207 | 0.828941 | 13,820 |
| Potential Fit | 0.424073 | 0.546475 | 0.477556 | 3,475 |

No Fit has the highest class F1, while both positive fit categories have
substantially lower F1. The Logistic Regression confusion matrix (rows =
actual; columns = predicted) is:

| Actual \ Predicted | Good Fit | No Fit | Potential Fit |
|---|---:|---:|---:|
| Good Fit | 1,012 | 226 | 380 |
| No Fit | 880 | 10,741 | 2,199 |
| Potential Fit | 448 | 1,128 | 1,899 |

For actual Potential Fit, 1,128 rows were called No Fit and 448 Good Fit.
Among predictions called Potential Fit, 2,199 were actually No Fit and 380
Good Fit. The saved probabilities agree with LR's predicted label argmax on
all 18,913 rows. These counts establish meaningful confusion between
categories; they do not prove why a particular synthetic resume was called
No Fit.

### Finding A: Is No Fit disproportionately predicted?

**Measured answer:** It depends on the model and on what “disproportionately”
means.

- Logistic Regression's No Fit prediction share (63.95%) is below its actual
  share (73.07%). It does not overpredict No Fit relative to the test-set
  prevalence, though it still predicts No Fit more often than either other
  label.
- LinearSVC's No Fit share (74.32%) is only modestly above its actual share
  (73.07%).
- Both models' No Fit class F1 is higher than Good Fit or Potential Fit.
  Logistic Regression's No Fit precision is 0.888 and recall is 0.777;
  LinearSVC's are 0.834 and 0.848 respectively.

Therefore the broad claim that both classifiers simply predict an excessive
share of No Fit is not supported by these aggregate results. The actual
majority class and class-specific difficulty are important context. The
specific user-reported synthetic examples are not available in
`predictions.csv`, so this report cannot independently reproduce them.

## 4. Skill/domain information and candidate features

The raw dataset has domains but no explicit required-skill list,
resume-skill annotation, years-of-experience label, or semantic similarity
score. `src/skill_extractor.py` has a manually curated technical-skill
vocabulary and returns sets of detected canonical names. It is not a
comprehensive ontology and does not distinguish required from preferred
skills, infer proficiency/years, recognize all synonyms, or resolve context
and negation. For example, simple keyword presence can overstate a skill that
is only mentioned as a JD requirement in a resume or negated in text.

The existing extraction output can be used as an initial, explicitly
heuristic feature source, but not as ground-truth skill labels.

### Proposed structured features

For each resume/JD pair, compute features independently and retain an
auditable definition:

1. **JD skill inventory:** canonicalized skills detected in the JD, ideally
   divided into required and preferred groups using labeled sections or
   conservative requirement-phrase rules. If required/preferred status cannot
   be confidently parsed, record an unknown/weak-confidence status rather
   than treating every JD mention as mandatory.
2. **Resume skill inventory:** canonicalized skills detected in resume
   context, excluding negated or purely aspirational mentions where feasible.
3. **Required skill count:** number of distinct skills marked required.
4. **Matched required skill count:** distinct intersection of resume skills and
   required JD skills.
5. **Required coverage:** matched required count divided by required skill
   count, with an explicit convention for zero or unknown required skills.
6. **Preferred-skill coverage:** separate count and ratio; do not silently
   mix optional and mandatory requirements.
7. **Weighted skill coverage:** optional weights for explicit “required” and
   “preferred” categories, fixed before evaluation or learned only inside
   training folds.
8. **Experience indicators:** parsed minimum/actual years, total experience
   and skill-specific experience only where extraction confidence is
   measurable. Raw regex counts alone can misread dates, ranges, and
   unrelated numbers; start with conservative features and missing-value
   indicators.
9. **Resume/JD semantic similarity:** cosine similarity between separately
   encoded resume and JD, optionally supplemented by section-level or
   requirement-to-resume-sentence matches.
10. **Existing TF-IDF model scores:** LR probability for each class, plus
    optionally decision margins from LinearSVC. These are stacking/meta
    features and must be generated out-of-fold for training examples.
11. **Domain features:** resume/JD domains and their match/compatibility
    interaction, encoded from the observed domain columns. Fit encoders using
    training folds only and handle unseen categories explicitly.

Required/matched counts and coverage are strongly related; avoid treating
them as independent proofs of compatibility. Include missingness/confidence
indicators and inspect feature leakage and dependence.

## 5. Structured model and hybrid architecture recommendation

Use a transparent two-branch design:

1. **Text branch:** retain the current TF-IDF Logistic Regression as a
   baseline probability source. Add an optional local sentence-embedding
   similarity feature (or a controlled, local semantic model) computed from
   resume and JD separately.
2. **Structured branch:** skill counts/coverage, requirement type, cautiously
   parsed experience, domain relation, extraction confidence, and missingness.
3. **Fusion:** train a regularized, interpretable classifier on structured
   features plus out-of-fold text-model probabilities. Start with Logistic
   Regression or a small tree-based model and compare against simpler
   rule-based coverage baselines. Do not make a hard “all skills matched =
   Good Fit” rule unless labeling policy validates that behavior.
4. **Output:** keep three labels and per-class scores initially. Calibrate
   probabilities on group-separated validation data if probabilities will be
   communicated as confidence. A low-confidence/review outcome may be safer
   than forcing a label, but that policy requires product approval and
   validation.
5. **Explainability:** return detected/matched/missing skill evidence as
   separate evidence, and explicitly distinguish those rules from model
   reasoning.

Do not replace the current model solely on the reported synthetic examples.
First establish a labeled test set containing reviewed examples of high,
partial, and low compatibility, with label definitions and preferably
independent reviewer agreement.

## 6. Sentence-transformers assessment

Sentence embeddings are a reasonable **candidate experiment**, not a
confirmed fix. Separate resume and JD embeddings followed by cosine similarity
can represent paraphrases and related terms better than exact lexical overlap.
Embedding similarity still does not determine whether an individual
requirement is mandatory, whether a candidate has the required experience,
or whether a semantic match is sufficient for Good Fit.

The selected workspace environment did not list `sentence_transformers` or
`transformers` as installed modules, and `requirements.txt` does not declare
them. No embedding model was run or benchmarked. Introducing a model would
require a dependency/model selection, local model distribution or controlled
download policy, runtime/memory measurement, license review, and privacy
review. Prefer local inference for resumes unless an external processing
service is explicitly approved. Compare against a no-new-dependency TF-IDF
cosine baseline first.

## 7. Leakage-safe evaluation methodology

1. **Freeze the evaluation protocol and labels.** Define Good/Potential/No Fit
   criteria before tuning. Keep the existing saved unseen-resume predictions
   as historical evidence, not as a tuning target.
2. **Split by resume identity.** Prefer a stable resume ID if available;
   currently the evaluator groups on normalized resume text. Ensure every
   row for one group stays within one partition. The previous single
   `GroupShuffleSplit` achieved zero overlap but is one random split and does
   not stratify labels.
3. **Reserve a final grouped holdout.** Split resume groups into development
   and final test sets once. Do not inspect final test examples to tune
   thresholds, features, prompts, or architecture.
4. **Tune within development groups.** Use `StratifiedGroupKFold` where
   supported and feasible, or repeated `GroupShuffleSplit` with reported
   class/group coverage. Report folds with missing-class cases rather than
   silently hiding them.
5. **Fit text preprocessing inside each fold.** `TfidfVectorizer`,
   domain/category encoders, feature scaling, feature selection, and any
   learned transformations must fit only on that fold's training partition.
6. **Generate stacking probabilities out-of-fold.** For each development
   training example, create LR/SVC scores from a model that did not train on
   that example or any row from its resume group. Fit the fusion model on
   these out-of-fold scores and structured features. At inference/final test,
   refit the text model on the full development training groups, then score
   the untouched held-out groups.
7. **Avoid test-guided heuristics.** Freeze skill aliases, required/preferred
   parsing rules, thresholds, and feature choices within the development
   workflow; do not derive them from final-test performance.
8. **Report more than accuracy.** Include per-class precision/recall/F1,
   macro-F1, confusion matrices, supports, prediction distribution,
   probability calibration (if exposed), group counts, split seeds, and
   confidence intervals or variation across grouped folds. Compare class
   prevalence and the majority baseline.
9. **Run ablations.** Compare current TF-IDF alone, structured features alone,
   lexical skill coverage, TF-IDF + structured fusion, and semantic features
   only if added. Attribute any observed gains to the measured holdout/fold
   results; do not assume improvement in advance.
10. **Audit group and source robustness.** Check performance across resume
    domains, JD domains, sources, resume lengths, and number of extracted
    skills, while avoiding groups too small for meaningful conclusions.

The existing `run_cross_validation()` is not sufficient for this purpose: it
receives a TF-IDF matrix vectorized before fold assignment and uses ordinary
`cross_val_score` without resume groups. The separate unseen-resume evaluator
does fit TF-IDF on its training partition and holds out cleaned resume groups,
but it reports only one split and is not a substitute for grouped development
cross-validation plus a final untouched holdout.

## 8. Likely causes for the reported classifications (hypotheses)

- The current model does not explicitly compare required JD skills against
  resume skills; a large skill count in the resume is not a guaranteed
  positive compatibility signal.
- Concatenation and bag-of-words TF-IDF may emphasize frequent lexical
  patterns and class-specific wording rather than the relationship between
  each requirement and evidence in the resume.
- The label mapping/rubric may not equate skill coverage with Good Fit. A
  strong skill match can still be No Fit under the source label policy.
- A new synthetic example may differ in wording, layout, length, or vocabulary
  from the training examples; no saved prediction row or model score for the
  reported examples was provided to determine whether this occurred.
- Missing or inconsistent label semantics, noisy keyword extraction, and
  domain variation may contribute, but are unverified without reviewed
  examples and label provenance.

These are explanations to test, not conclusions about the particular
synthetic resumes.

## 9. Risks and limitations

- Current No Fit prevalence is high (73.07% of the grouped evaluation test
  rows), so raw accuracy can conceal weaker positive-class performance.
- There are 643 distinct resume texts across the raw data; unseen-resume
  generalization is measured over relatively few independent resume groups
  compared with row count.
- Skill extraction is a bounded keyword ruleset, not a validated skill
  ontology or evidence/experience parser.
- The raw schema does not contain explicit required-skill annotations or
  documented label confidence/annotator agreement.
- One grouped split can vary by random seed; `GroupShuffleSplit` does not
  guarantee label stratification.
- Sentence-transformers is not present in the inspected environment, and no
  semantic model performance, resource use, or license was evaluated.
- This investigation is not an experiment. It makes no claim that a hybrid
  model or semantic embeddings will improve any metric.
