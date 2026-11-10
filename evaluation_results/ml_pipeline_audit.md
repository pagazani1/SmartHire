# SmartHire ML Pipeline Audit

**Scope:** Read-only review of the nine requested source files. No training was
run, no model artifacts were loaded, and no resume/job-description records were
read or printed. A small synthetic-string check was run against
`extract_skills`; no project files were changed by that check.

## Summary

- **Preprocessing parity:** The main training path and `predict.py` use the
  same `clean_text()` function and the same `resume_clean + " " + jd_clean`
  feature construction. This part is consistent when preprocessing is
  recomputed.
- **Artifact loading:** The inference code uses the same vectorizer/model
  filenames configured by `config.py`. Both files were present during the
  audit. `predict.py` resolves them relative to the process working directory,
  unlike `config.py`, which resolves paths relative to the project. Actual
  deserialization and model/vectorizer compatibility were not tested.
- **Input validation:** The CLI rejects blank strings, but the callable
  `predict()` function does not validate its arguments. `clean_text()` converts
  non-string values using `str()`, so a programmatic `None` can become literal
  text rather than a clear input error.
- **Skill extraction:** Matching is case-insensitive, the vocabulary includes
  custom patterns for special-character skills and a `Set` deduplicates
  repeated identical skill names. Some custom patterns are unanchored, and
  one comment promises uppercase-only NLP matching despite case-insensitive
  compilation. Synthetic checks confirmed these behaviors.
- **API readiness:** There is a reusable `predict()` function and separate
  artifact loading and skill-analysis functions, but validation and artifact
  paths are CLI/cwd dependent. A thin reusable inference service should own
  validation, artifact paths, prediction, and structured errors before API
  integration.

## File-wise findings

### `config.py`

- Defines project-rooted `Path` values for raw/processed data and model
  artifacts, including `tfidf_vectorizer.joblib` and
  `logistic_regression.joblib`.
- Defines the TF-IDF and Logistic Regression hyperparameters consumed by the
  training and evaluation code.
- **Potential risk:** Importing this module creates the raw-data,
  processed-data, and model directories. This is a startup side effect, which
  may be undesirable in read-only or server contexts.
- It does not configure inference input validation, artifact versioning, or
  provenance metadata.

### `main.py`

- Orchestrates data loading, preprocessing, TF-IDF generation, baselines,
  classifier training/evaluation, and optional cross-validation.
- With the default flags, calls permit cached processed data, cached TF-IDF
  matrices/vectorizer, and cached models to be reused.
- **Confirmed risk:** Cached artifacts are not tied to a raw-data fingerprint
  or training configuration. `force_download` does not itself force
  reprocessing, refitting TF-IDF, or retraining. A download can therefore be
  followed by stale processed/features/models unless the corresponding force
  flags are also used.
- `run_pipeline()` is an orchestration function, not an inference entry point.

### `predict.py`

- `load_artifacts()` calls `joblib.load()` on the Logistic Regression and
  vectorizer files. Their basenames match the paths configured in
  `config.py`.
- `predict()` cleans both inputs with `clean_text()`, concatenates resume then
  JD with one space, calls the fitted vectorizer's `transform()` (not
  `fit_transform()`), and returns the predicted label plus class probabilities.
- `validate_input()` catches empty/whitespace strings when called by the
  interactive `main()` flow.
- **Confirmed issue:** `predict()` does not call `validate_input()`. A caller
  such as a future API can pass blank values directly; `clean_text(None)`
  stringifies the value rather than rejecting it. `validate_input(None, ...)`
  itself raises an attribute error at `.strip()`. The CLI's `sys.exit()` error
  behavior is also not a suitable service-layer validation contract.
- **Confirmed path inconsistency:** Artifact paths are relative to the current
  working directory, whereas configured paths are rooted at the project.
  Running the script with a different working directory can resolve the
  artifact paths incorrectly.
- **Potential risk:** There is no explicit check that the loaded vectorizer
  and model are a compatible artifact pair, or that their expected classes
  match the service contract. Loading errors are allowed to propagate, but are
  not translated into structured application errors.
- The prediction, input, and display logic are in the same CLI module. The
  `predict()` function is reusable, but callers must separately load
  artifacts and remember to validate inputs.

### `src/data_loader.py`

- Reads both local raw Parquet files when present; otherwise downloads the
  configured Hugging Face dataset and writes train/test Parquet caches.
- **Potential risk:** It does not validate required columns or row schemas
  itself; failures surface later in preprocessing. If only one local cache is
  missing, it downloads and overwrites both.
- Loading may require network access when either cache is absent. This should
  not be part of a request-time inference service.

### `src/preprocessing.py`

- `clean_text()` stringifies the value, lowercases it, replaces double quotes
  with spaces, collapses whitespace, and trims it.
- `preprocess_data()` applies that function independently to `resume` and
  `jd`, then joins the cleaned values in resume-first order. This matches the
  composition in `predict.py`.
- **Confirmed risk:** The processed-data cache is returned whenever both
  processed files exist and `force_reprocess` is false, regardless of the
  supplied DataFrames or their source/version.
- **Confirmed risk:** `get_tfidf_features()` reuses cached sparse matrices and
  vectorizer when all three files exist, without checking that the matrices
  align with the currently supplied labels/rows or that they were built from
  the current processed data.
- Missing values passed through `clean_text()` become strings such as `"nan"`
  or `"none"` rather than empty text. The unseen-resume evaluator handles
  missing resume/JD values before calling the cleaner, but this is not
  consistent with the regular preprocessing path.
- **Potential risk:** The fitted TF-IDF vectorizer and trained classifier are
  saved as separate files, so they can be independently stale.

### `src/models.py`

- `train_logistic_regression()` and `train_linear_svc()` load existing model
  files unless `force_retrain=True`; otherwise they fit and save new models.
- Logistic Regression hyperparameters match `config.py`.
- **Confirmed risk:** An existing model is accepted without validating its
  training data, vectorizer version, class set, or feature contract. If model
  and vectorizer caches are from different runs, inference can fail or
  produce semantically invalid predictions even if dimensions happen to fit.
- These functions combine training and cache-loading behavior; a request-time
  service should load only a deliberately versioned artifact bundle rather
  than call training helpers.

### `src/evaluate.py`

- Provides majority/domain baselines, test-set metrics, confusion-matrix
  output, and cross-validation.
- **Confirmed evaluation concern:** `run_cross_validation()` receives a
  precomputed `X_train` whose TF-IDF vectorizer was fit before folds are made.
  Validation-fold text therefore influenced vocabulary/IDF statistics.
- **Confirmed evaluation concern:** `cross_val_score(..., cv=cv_folds)` does
  not receive resume groups. For repeated resumes, folds are not guaranteed
  to keep the same resume entirely on one side. This does not measure
  generalization to unseen resumes.
- The confusion-matrix label order is `Good Fit`, `Potential Fit`, `No Fit`,
  while other reports and evaluation outputs may use a different order. The
  counts remain interpretable only if the displayed labels/order are kept
  together.

### `src/skill_extractor.py`

- Uses a fixed skill vocabulary, case-insensitive regular expressions, custom
  expressions for punctuation-bearing names, and a set of canonical labels.
- **Confirmed behavior:** A repeated skill is returned once because results
  are stored in a `Set`; occurrence counts and contextual evidence are lost.
- **Confirmed bug:** The `NLP` custom pattern comment says it is an
  uppercase-only acronym guard, but patterns compile with `re.IGNORECASE`.
  A synthetic check of lowercase `nlp` returned the `NLP` skill.
- **Confirmed matching risk:** Several custom patterns for punctuation-bearing
  skills do not include boundary guards. Synthetic checks found substring
  matches such as `.NET` within a larger token and `Node.js` followed by an
  extra letter. `ASP.NET` also matches both `ASP.NET` and `.NET`.
- Some formatting variants are explicitly supported (for example
  `scikit-learn`/`scikit learn` and `T-SQL`/`T SQL`); others are exact-pattern
  matches and may miss variants such as `node js`.
- The built-in self-test exercises ordinary, special-character, empty,
  overlap, and case examples, but it does not assert most edge-case outcomes.
  The existing `tests/` files do not contain a dedicated skill-extractor test
  module.

### `evaluate_unseen_resumes.py`

- Combines raw train/test rows, groups on cleaned resume text, makes a
  `GroupShuffleSplit`, fits a fresh TF-IDF vectorizer on training text, and
  fits fresh Logistic Regression and LinearSVC models.
- Its `clean_text()` use and resume/JD concatenation align with training and
  inference. It explicitly normalizes missing resume/JD values before
  cleaning.
- It writes to `evaluation_results/`, including files that are outputs of the
  evaluation pipeline. Running it can overwrite those evaluation outputs.
- This script is an evaluation/training program, not a production inference
  function; it should not be invoked by an API request.

## Confirmed issues

1. **Inference callable does not enforce non-empty text.** Validation only
   occurs in the interactive CLI flow, and `clean_text()` stringifies
   non-string values.
2. **Artifact paths depend on current working directory.** `predict.py` uses
   relative literals rather than the project-rooted paths in `config.py`.
3. **Cache reuse has no provenance/alignment validation.** Processed
   DataFrames, TF-IDF matrices/vectorizer, and classifier models can be
   reused independently without confirming they belong together.
4. **Regular cross-validation is not leakage-safe for unseen resumes.** The
   TF-IDF transform is precomputed before CV and no resume groups are passed.
5. **NLP case comment and behavior disagree.** The implementation accepts
   lowercase `nlp` due to `re.IGNORECASE`.
6. **Some special-character skill patterns can match substrings.** This was
   confirmed with synthetic strings; actual resume/JD content was not used.
7. **No dedicated ML/inference tests were found in the inspected `tests/`
   directory.** The visible test files focus on candidate reports and
   background verification. `skill_extractor.py` has a manual self-test
   block, not a dedicated automated test module.

## Potential risks (not established as production failures)

- Existing artifact files were present, but this audit did not load them.
  Therefore successful deserialization, sklearn-version compatibility, and
  model/vectorizer compatibility remain unverified.
- There is no single versioned artifact manifest or checksum that ensures the
  vectorizer and classifier are paired.
- The training data loader and preprocessing path have no explicit schema
  contract or consistent null-text policy.
- Inference functions have no explicit document-size limits or structured
  error types, which should be considered for API resource and response
  handling.
- Skill detection reports keyword presence only; synonyms, negation, context,
  proficiency, and repeated evidence are not modeled.
- Importing `config.py` creates directories, which can create filesystem
  side effects when modules are imported by a server.

## Recommended fixes, in priority order

1. **P0 — Create a validated inference service boundary.** Validate both
   inputs inside the callable prediction API, reject non-strings and
   whitespace-only values with stable `ValueError`/request-validation errors,
   and avoid `sys.exit()` outside the CLI wrapper.
2. **P0 — Make artifact resolution deterministic.** Use project/config-based
   paths or an injected artifact directory, and load a versioned bundle that
   records the matching vectorizer/model versions and expected feature/class
   contract.
3. **P1 — Make cache reuse coherent.** Add a data/config fingerprint or
   explicit cache manifest; verify row counts, feature dimensions, labels,
   and artifact provenance before reusing data, matrices, or models. Couple
   vectorizer and classifier artifacts.
4. **P1 — Correct evaluation methodology.** For cross-validation, fit TF-IDF
   inside each fold (for example, as part of an sklearn `Pipeline`) and use a
   group-aware splitter when the goal is unseen-resume performance. Keep the
   independent GroupShuffleSplit evaluation as a separate holdout.
5. **P1 — Define consistent missing-text/schema handling.** Validate required
   columns and label values at ingestion and consistently reject or normalize
   null/empty resume and JD fields before cleaning.
6. **P2 — Tighten skill patterns and document matching semantics.** Add
   boundaries/negative lookarounds for punctuation skills, decide whether
   acronyms are case-sensitive, normalize supported aliases, and preserve
   `Set` deduplication only if occurrence counts are not needed.
7. **P2 — Separate server-safe imports and operations.** Avoid directory
   creation merely from importing configuration, and keep downloading,
   training, plotting, and result-file writes out of request-time inference.

## Tests required before backend integration

These are recommendations; **they were not run as part of this read-only
audit**.

1. Unit-test preprocessing for lowercase/whitespace/quote handling,
   resume-then-JD order, and null/non-string behavior.
2. Unit-test the inference function for empty, whitespace, `None`, and
   non-string inputs; verify stable structured errors without process exit.
3. Test artifact loading from a working directory other than the project
   root, missing/corrupt files, and mismatched vectorizer/model metadata.
4. Use small synthetic fitted artifacts to test that `predict()` transforms
   but never refits, returns a valid class, and maps probabilities to the
   model's `classes_` correctly.
5. Unit-test skill matching boundaries and variants: C/C++/C#, `.NET` versus
   `ASP.NET`, Node.js punctuation, Java versus JavaScript, acronym casing,
   alias normalization, and repeated mentions.
6. Test cache invalidation when input data or TF-IDF settings change, and
   verify cached matrix rows/labels/model feature counts remain aligned.
7. Test group-aware CV with repeated synthetic resume groups and assert no
   group appears in both train and validation folds; fit preprocessing only
   within each fold.
8. Add an end-to-end inference contract test covering validation, artifact
   loading, prediction output schema, skill analysis separation, and expected
   API error responses without printing or persisting submitted text.

## Verification performed

- Source files and existing test filenames were inspected read-only.
- A synthetic-string check exercised selected skill-regex cases; it printed
  only the synthetic test case names and detected skill names.
- Model artifact file names and sizes were checked, but model/vectorizer
  deserialization was not attempted.
- No training or project test suite was run.
