# Hybrid Synthetic Resume Validation

## Method

Validated the three supplied fictional PDFs in `test_resumes/` against the
same Python Backend Developer Intern job description specified for this test.
All three files existed, were non-empty, and were successfully read with
`extract_text_from_pdf()` from `src/pdf_extractor.py`. The saved
`models/hybrid_experimental/hybrid_bundle.joblib` loaded successfully through
`load_hybrid_bundle()` and passed the bundle's compatibility checks.

Each extracted resume was passed to the existing `predict_hybrid()` interface.
The structured-only classifier scores are not included in that function's
return value; they were computed with the saved `structured_model` estimator
and the same `extract_structured_features()` function used by the training and
inference pipeline. No resume text was printed or saved in this report.

The optional domain inputs were not supplied, so domain-related values used
the inference function's default missing-domain behavior. No model was
retrained or tuned.

## Results

Probabilities are shown to six decimal places. They are model outputs, not
calibrated measures of hiring certainty.

### Strong match

- **Predicted category:** Potential Fit
- **Required skill coverage:** 12/12 (1.000000)
- **Detected resume skills:** FastAPI, Flask, Git, GitHub, Machine Learning,
  NumPy, Pandas, Python, REST API, SQL, SQLite, Text Classification,
  scikit-learn
- **Matched required groups:** FastAPI OR Flask; Git; GitHub; Machine Learning;
  NumPy; Pandas; Python; REST API; SQL; SQLite; Text Classification;
  scikit-learn
- **Missing required groups:** None
- **Text Logistic Regression probabilities:** Good Fit 0.150888; No Fit
  0.550818; Potential Fit 0.298293
- **Structured classifier probabilities:** Good Fit 0.239967; No Fit 0.000308;
  Potential Fit 0.759724
- **Hybrid probabilities:** Good Fit 0.316166; No Fit 0.000209; Potential Fit
  0.683624

### Partial match

- **Predicted category:** Potential Fit
- **Required skill coverage:** 4/12 (0.333333)
- **Detected resume skills:** Git, JavaScript, Pandas, Python, SQL
- **Matched required groups:** Git; Pandas; Python; SQL
- **Missing required groups:** FastAPI OR Flask; GitHub; Machine Learning;
  NumPy; REST API; SQLite; Text Classification; scikit-learn
- **Text Logistic Regression probabilities:** Good Fit 0.093778; No Fit
  0.595668; Potential Fit 0.310554
- **Structured classifier probabilities:** Good Fit 0.076593; No Fit 0.345060;
  Potential Fit 0.578347
- **Hybrid probabilities:** Good Fit 0.070920; No Fit 0.434505; Potential Fit
  0.494574

### Weak match

- **Predicted category:** No Fit
- **Required skill coverage:** 0/12 (0.000000)
- **Detected resume skills:** None
- **Matched required groups:** None
- **Missing required groups:** FastAPI OR Flask; Git; GitHub; Machine Learning;
  NumPy; Pandas; Python; REST API; SQL; SQLite; Text Classification;
  scikit-learn
- **Text Logistic Regression probabilities:** Good Fit 0.101736; No Fit
  0.511482; Potential Fit 0.386782
- **Structured classifier probabilities:** Good Fit 0.005540; No Fit 0.934265;
  Potential Fit 0.060195
- **Hybrid probabilities:** Good Fit 0.006160; No Fit 0.936966; Potential Fit
  0.056874

## Comparison

| Synthetic case | Matched required groups | Required coverage | Hybrid prediction | Hybrid probabilities (Good / No / Potential) |
|---|---:|---:|---|---|
| Strong | 12/12 | 1.000000 | Potential Fit | 0.316166 / 0.000209 / 0.683624 |
| Partial | 4/12 | 0.333333 | Potential Fit | 0.070920 / 0.434505 / 0.494574 |
| Weak | 0/12 | 0.000000 | No Fit | 0.006160 / 0.936966 / 0.056874 |

The prediction ordering is directionally consistent with the three supplied
examples: the strong case has complete detected required-skill coverage and
the largest Potential Fit score; the partial case has lower coverage and a
closer Potential Fit/No Fit score; and the weak case has no detected required
skills and a dominant No Fit score. The strong case is predicted as Potential
Fit rather than Good Fit. These are observations on these fixtures only.

For all three cases, each probability distribution (text, structured, and
hybrid) summed to 1 within floating-point precision (approximately 1.0).

## Limitations

- These are three fictional synthetic test cases, not a representative sample
  and not evidence of real-world hiring performance or generalization.
- Skill detection is keyword-based and does not establish a candidate's
  competence, depth of experience, or ability to perform the work.
- The hybrid model is experimental and was trained on the capped Stage 2
  sample. Predictions should not be used as automatic employment decisions.
- Probability outputs should not be interpreted as calibrated probabilities
  of job performance or hiring success.
- The optional domain features were left missing because no domain inputs were
  provided for these resumes or the supplied job description.

## Verification

- PDF extraction and hybrid inference succeeded for all three supplied files.
- The saved hybrid bundle loaded successfully.
- Probability sums were checked for all three model outputs and all cases.
- Focused tests passed: `test_hybrid_predict.py` (8 tests) and
  `test_pdf_extractor.py` (10 tests), 18 total.
- No model training, tuning, production API, original model artifact, or
  existing evaluation output was changed.
