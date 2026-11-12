# Production No Fit Prediction Investigation

**Date:** 2026-11-12
**Scope:** Read-only production-model diagnosis plus isolated, reproducible
experiments. No existing source, model, data, test, backend, frontend, or
evaluation artifact was changed. The only additions are this report and
`run_production_representation_experiment.py`.

## Executive summary

- The API's production prediction is from the saved 20,000-feature TF-IDF
  vectorizer and Logistic Regression classifier. The separate skill report is
  not an input to this classifier.
- The exact browser request/JD was not retained, so the reported
  `No Fit 51.0% / Potential Fit 44.2% / Good Fit 4.8%` cannot be reproduced
  exactly. On the fixed, section-formatted JD below, the same strong-match PDF
  returns **Potential Fit**, with 13 matched skills, zero missing skills, and
  12/12 required-group coverage. A second fixed, prose-style JD returns **No
  Fit** with 12 matched skills, zero missing, and 100% coverage of the
  requirements the extractor classifies.
- This is not evidence that the response fields contradict each other. The
  text classifier and keyword skill extractor are separate computations.
  The classifier has no explicit skill-match or coverage features.
- On the controlled grouped holdout, the hybrid candidate improved the
  combined-text baseline's accuracy, macro-F1, weighted-F1, Good Fit recall,
  and Potential Fit recall. This is promising evidence, not sufficient reason
  to replace production: training used a capped sample, the data contains only
  643 resume groups, labels lack documented independent adjudication, and
  this holdout had already been evaluated in earlier project reports.

## Production path and saved artifacts

`backend/main.py` accepts the PDF and `jd_text`, writes the PDF temporarily,
and calls `src.resume_analyzer.analyze_resume()`. That function extracts text,
loads the inference artifacts, calls `predict()`, and independently calls
`analyse_skills()`.

The production artifact paths in `predict.py` are:

- `models/tfidf_vectorizer.joblib`
- `models/logistic_regression.joblib`

Read-only loading confirmed the artifacts are a scikit-learn
`TfidfVectorizer` and `LogisticRegression`, each using 20,000 features. The
classifier classes are `Good Fit`, `No Fit`, and `Potential Fit`.
`predict()` applies the shared `clean_text()` normalization, concatenates
cleaned resume then cleaned JD, calls `transform()` (not `fit_transform()`),
then obtains the label and probabilities from `predict()` and
`predict_proba()`. The returned label is the model's highest-probability
class.

The saved artifacts have no embedded data/configuration fingerprint. Their
types, class set, and feature dimension are compatible, but their exact
training provenance cannot be proven from the artifacts alone.

## PDF and reproducible request

The local [strong-match PDF](../test_resumes/SmartHire_Strong_Match_Resume.pdf)
was successfully parsed as 888 characters. Its extracted content includes a
Python/backend summary; skills covering Python, SQL, FastAPI, Flask, REST APIs,
SQLite, Pandas, NumPy, scikit-learn, machine learning, text classification,
Git, and GitHub; and two corresponding project descriptions. The extracted
text contains the Unicode replacement character `�` in place of separators
in the project headings and education date range. No exact cleaned resume
string in the 93,733-row raw dataset matched the extracted PDF text.

The text extractor returned (preserving extracted line breaks; the contact
line is synthetic as marked in the source PDF):

```text
JAMIE RIVERA
fictional.candidate@example.test | +1 (555) 010-0000 | Portfolio: example.test
SUMMARY
Computer Science graduate focused on Python backend engineering and machine learning. Built and tested REST APIs and
data-driven applications.
SKILLS
Programming: Python, SQL
Backend: FastAPI, Flask, REST APIs
Databases: SQLite
Data & ML: Pandas, NumPy, scikit-learn, Machine Learning, text classification
Tools: Git, GitHub
PROJECTS
Inventory REST API � Python, FastAPI, SQLite. Designed REST endpoints, SQL queries, request validation, and
automated tests.
Text Classification Project � Python, Pandas, NumPy, scikit-learn. Built a TF-IDF text classification prototype and
evaluated it using precision, recall, and F1-score.
EDUCATION
BSc Computer Science, Fictional Technical University, 2022�2026
Fictional sample resume created solely for SmartHire testing. All details are invented.
```

The exact JD used for the 13-skill reproducible run was:

```text
Python Backend Developer Intern
Required Skills:
- Python
- FastAPI or Flask
- REST APIs
- SQL
- SQLite
- Pandas
- NumPy
- scikit-learn
- Git
- GitHub
- machine learning
- text classification
```

Production API result for that PDF/JD:

| Label | Probability | Frontend-style display |
|---|---:|---:|
| Good Fit | 0.0560510689 | 5.6% |
| No Fit | 0.4711556530 | 47.1% |
| Potential Fit | 0.4727932781 | 47.3% |

The label is **Potential Fit**. Skill output is 13 matched, 0 missing,
12 required groups matched out of 12, and 100% required and overall coverage.
The two detected FastAPI/Flask alternatives count as one requirement group.

For comparison, a second fixed JD using the longer prose/requirement phrasing
from `tests/test_skill_extractor.py` (with the same required-skill concepts,
but without the explicit SQLite bullet) returned **No Fit**:
Good Fit `0.0430079415`, No Fit `0.5303511556`, Potential Fit `0.4266409029`.
It returned 12 matched skills, 0 missing, and 100% coverage across 11
classified required groups. This shows wording changes can change both skill
classification and model scores. Neither fixed request reproduces the exact
browser percentages, and the browser JD was not saved; therefore the exact
reported request remains unverified.

## Feature evidence for a No Fit result

For the reproducible No Fit request, the model's reconstructed class logits
were:

| Class | Logit |
|---|---:|
| Good Fit | -1.602237486 |
| No Fit | 0.909917074 |
| Potential Fit | 0.692320412 |

The `No Fit` logit is largest. Its strongest positive TF-IDF contributions
were:

| Feature | Contribution | Present in |
|---|---:|---|
| `sql` | +0.554020 | Resume and JD |
| `data` | +0.544352 | Resume and JD |
| `python` | +0.376655 | Resume and JD |
| `test` | +0.204717 | Resume |
| `engineering` | +0.174902 | Resume |
| `apis` | +0.161420 | Resume and JD |
| `computer science` | +0.115646 | Resume |
| `programming` | +0.115103 | Resume and JD |
| `using` | +0.108710 | Resume and JD |
| `rest` | +0.080759 | Resume and JD |

Strong negative contributions to the No Fit logit included `backend`
(-0.156486), `automated tests` (-0.093274), `github` (-0.091390),
`classification` (-0.080469), `learn` (-0.066517), and `pandas`
(-0.062688).

These are signed contributions to this class's linear logit, not causal
explanations or skill judgments. Common terms such as `sql`, `data`, and
`python` occur in both documents and are pooled into the same TF-IDF feature;
the model cannot distinguish which occurrence came from the resume versus
the JD. The positive No Fit associations outweigh the model's negative
associations for this input. This is consistent with the learned text model,
not evidence that the skill extractor's `missing_skills` output was ignored
incorrectly.

The frontend keeps the backend response and displays each probability as
`probability * 100`, rounded to one decimal place in
`frontend/src/components/AnalysisResults.jsx`. It does not derive or change
the predicted category.

## Training, dataset, leakage, and preprocessing

The available raw files contain 93,733 rows: 80,017 in `raw_train.parquet`
and 13,716 in `raw_test.parquet`. Class counts and proportions are:

| Label | Rows | Share |
|---|---:|---:|
| No Fit | 68,774 | 73.3722% |
| Potential Fit | 17,489 | 18.6583% |
| Good Fit | 7,470 | 7.9694% |

There are 643 distinct resume texts and 93,733 distinct resume/JD pairs.
Resume texts repeat with different JDs and labels, so row-wise random
validation would leak resume information. The official raw train/test files
have **476 normalized resume groups in common**; treating that original split
as an unseen-resume test would therefore be invalid.

Observed safeguards and risks:

- The normal training and inference text-normalization path uses the same
  `clean_text()` and resume-then-JD concatenation. No direct preprocessing
  mismatch was found.
- `src/preprocessing.py` fits TF-IDF on training text and transforms test
  text when refitting. Its default cache reuse is not fingerprinted to the
  supplied dataset/configuration, so stale processed matrices or a stale
  model/vectorizer pairing remains a provenance risk, not a confirmed cause
  of this prediction.
- `run_group_kfold.py` groups by JD rather than resume and consumes a
  precomputed TF-IDF matrix, so it does not provide a clean unseen-resume
  estimate and its fold validation text has influenced the earlier
  vectorizer fit.
- `evaluate_unseen_resumes.py` does create a fresh resume-grouped split and
  fits TF-IDF on that split's training rows. Its saved Logistic Regression
  test results were Accuracy `0.721832`, Macro-F1 `0.605955`, Good Fit recall
  `0.625464`, and Potential Fit recall `0.546475`. Those full-training results
  use a different split and training set size from the controlled experiment
  below; they are contextual, not a direct row-for-row comparison.
- Labels are imbalanced even though the Logistic Regression uses
  `class_weight="balanced"`. Reweighting training loss does not establish
  probability calibration or correct label semantics. The project does not
  record a gold-standard label rubric or independent human adjudication for
  these examples.

## Controlled experiment

The repeatable experiment is implemented in
[`run_production_representation_experiment.py`](../run_production_representation_experiment.py).
It reads the raw datasets but writes no model, matrix, prediction, or report
artifacts.

Protocol:

1. Combine the raw dataset rows; group by normalized resume text.
2. Make a fixed 20% resume-group holdout (`GroupShuffleSplit`, seed 2026).
3. Split the remaining development groups into train and validation
   (`GroupShuffleSplit`, 25% of development, seed 2027), giving approximately
   60/20/20 train/validation/holdout by rows.
4. Cap each training resume/label pair to five rows using the existing
   Stage 2 sampler (seed 6842). This yields 4,802 training rows for the
   validation comparison and 6,462 rows for final refits. Validation and
   holdout retain all their rows.
5. Keep all model settings fixed. Select no model or parameter using the
   holdout. Fit vectorizers on each training split only; fit structured
   imputation, scaling, and category encoders inside their training pipeline.
   The existing deterministic skill-rule extraction uses a fixed vocabulary,
   not corpus-fitted parameters. Hybrid stacking probabilities are generated
   out-of-fold with resume-grouped folds.
6. After the validation comparison, refit all four fixed candidates on
   capped train+validation data and evaluate once on the final holdout.

The code asserts no resume-group overlap between train, validation, and
holdout and checks that every split has all three labels. The final holdout
was **not used for model selection in this run**. Important caveat: this same
seed-2026 holdout was already evaluated in the existing Stage 2 report, so it
is not an unseen or globally untouched test set for future production claims.

### A-D comparison

- **A — production recipe:** combined resume+JD TF-IDF, Logistic Regression.
- **B — separated text:** independently fitted resume TF-IDF and JD TF-IDF
  channels, concatenated before Logistic Regression.
- **C — structured:** existing skill counts/coverage and domain features with
  the existing fold-fit structured Logistic Regression.
- **D — hybrid:** combined-text Logistic Regression probabilities plus
  structured features; training fusion uses out-of-fold text probabilities
  grouped by resume, following the existing hybrid implementation.

#### Validation metrics

Validation supports: Good Fit 1,806; No Fit 13,500; Potential Fit 3,622.
Each per-class cell is `Precision / Recall / F1`.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Good Fit P/R/F1 | No Fit P/R/F1 | Potential Fit P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.631657 | 0.524716 | 0.655398 | 0.322424 / 0.698228 / 0.441140 | 0.854512 / 0.683926 / 0.759761 | 0.347104 / 0.403644 / 0.373245 |
| B | 0.601543 | 0.519723 | 0.633191 | 0.319448 / 0.730343 / 0.444482 | 0.874567 / 0.616667 / 0.723316 | 0.329924 / 0.480950 / 0.391373 |
| C | 0.573806 | 0.443061 | 0.605625 | 0.238674 / 0.749723 / 0.362080 | 0.862614 / 0.647407 / 0.739675 | 0.245597 / 0.211761 / 0.227428 |
| D | 0.659182 | 0.558417 | 0.684674 | 0.351992 / 0.748616 / 0.478838 | 0.895235 / 0.700074 / 0.785717 | 0.369536 / 0.462176 / 0.410697 |

Validation confusion matrices (rows = actual; columns = predicted; order =
Good Fit, No Fit, Potential Fit):

| Model | Confusion matrix |
|---|---|
| A | `[[1261, 241, 304], [1821, 9233, 2446], [829, 1331, 1462]]` |
| B | `[[1319, 145, 342], [1979, 8325, 3196], [831, 1049, 1742]]` |
| C | `[[1354, 277, 175], [2579, 8740, 2181], [1740, 1115, 767]]` |
| D | `[[1352, 154, 300], [1493, 9451, 2556], [996, 952, 1674]]` |

#### Final grouped holdout metrics

Holdout supports: Good Fit 1,277; No Fit 14,386; Potential Fit 3,520.
Each per-class cell is `Precision / Recall / F1`.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Good Fit P/R/F1 | No Fit P/R/F1 | Potential Fit P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.617734 | 0.489625 | 0.654427 | 0.239403 / 0.703211 / 0.357200 | 0.876986 / 0.660086 / 0.753232 | 0.316247 / 0.413636 / 0.358444 |
| B | 0.613877 | 0.500806 | 0.653631 | 0.248881 / 0.740016 / 0.372487 | 0.895628 / 0.636452 / 0.744118 | 0.324424 / 0.475852 / 0.385811 |
| C | 0.562112 | 0.410771 | 0.611908 | 0.174848 / 0.675020 / 0.277751 | 0.885015 / 0.633463 / 0.738403 | 0.204247 / 0.229545 / 0.216158 |
| D | 0.656675 | 0.535470 | 0.691396 | 0.288509 / 0.739233 / 0.415036 | 0.914713 / 0.688864 / 0.785884 | 0.343313 / 0.495170 / 0.405490 |

Holdout confusion matrices (same actual/predicted label order):

| Model | Confusion matrix |
|---|---|
| A | `[[898, 172, 207], [1949, 9496, 2941], [904, 1160, 1456]]` |
| B | `[[945, 115, 217], [1959, 9156, 3271], [893, 952, 1675]]` |
| C | `[[862, 216, 199], [2324, 9113, 2949], [1744, 968, 808]]` |
| D | `[[944, 103, 230], [1372, 9910, 3104], [956, 821, 1743]]` |

The controlled run's A, C, and D holdout metrics and matrices exactly match
the persisted Stage 2 metrics for the same seed-2026 split and capped fit,
which provides an independent reproduction check. B is the new candidate on
that split.

### Experiment interpretation

- B slightly improves holdout Macro-F1 over A (`0.500806` vs `0.489625`) and
  improves Good Fit and Potential Fit recall, while reducing accuracy and
  No Fit recall. The separated channels alone do not establish an overall
  improvement.
- C alone has substantially lower accuracy and Macro-F1 and weak Potential
  Fit recall. Skill extraction coverage is unavailable for many dataset rows;
  the saved Stage 2 feature audit found required coverage defined for only
  38.2% of rows. These heuristic features are useful evidence, but insufficient
  as a standalone classifier here.
- D improves on A in all three aggregate holdout metrics and in all three
  class recalls on this split. It also led on the validation aggregate
  metrics. This makes the existing hybrid a **promising research candidate**,
  not yet a production-ready replacement.
- The original experimental hybrid training caps the development data to
  6,462 rows (8.67% of its 74,550 development rows). Sampling by
  resume/label preserves group-label support but changes row frequency and
  class prevalence. This is a known limitation in
  `evaluation_results/stage5b_model_diagnosis.md`.
- The holdout has 19,183 rows but only 129 resume groups, with repeated
  resume/JD observations. Reported row-level metrics should not be mistaken
  for performance on 19,183 independent candidates.

## Diagnosis

The strongest supported explanation is **model/feature limitation combined
with dataset and evaluation limitations**, not a demonstrated frontend/API
software bug:

1. The model learns class associations from a pooled bag of resume+JD terms;
   it does not model requirement-to-resume matches or requirement importance.
2. The No Fit class is 73.37% of the dataset. Balanced class weights do not
   encode skill coverage or guarantee calibrated probabilities.
3. The dataset has only 643 unique resume texts across 93,733 rows, and its
   label rubric and independent label quality are undocumented.
4. The standard raw split leaks 476 resume groups; grouped evaluation is
   necessary. The current experiment did group the candidate splits, but its
   final holdout had already been seen in existing reports.
5. The same strong resume and skill coverage can yield Potential Fit or No Fit
   under different fixed JD wording. The exact original browser JD and
   response were unavailable, so the specific displayed percentages cannot
   be assigned to one verified input here.

No evidence indicates that the frontend changed the returned prediction or
that the PDF skills were used as the classifier's features. The PDF extraction
does contain replacement characters, but the tested skill names and project
text were extracted; there is no evidence this alone caused the class result.

## Recommendation and next evidence needed

Do not replace production artifacts on this study alone. Keep A as the
production baseline and evaluate D further on a newly collected, frozen,
resume-group-disjoint dataset with:

- a documented fit-label rubric and independent human review/adjudication;
- more independent resumes, particularly Good Fit and Potential Fit examples;
- source/time-separated evaluation and source-stratified error analysis;
- a holdout not previously used to compare these candidates;
- probability calibration and class-specific error/cost analysis;
- explicit human-review safeguards and validation that skill features do not
  encode unwanted proxies.

No threshold was changed, and the synthetic PDF was neither used for fitting
nor for model selection.

## Commands used

From the repository root, the main controlled experiment is reproducible with:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe .\run_production_representation_experiment.py
```

The focused PDF/API and artifact inspections were read-only Python snippets
using the same `.venv` interpreter. No dependency installation, model
training-to-disk, or production API changes were made.
