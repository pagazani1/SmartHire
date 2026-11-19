# SmartHire

Explainable resume screening for recruiters. SmartHire matches a PDF resume against a
job's required and preferred skills and explains the result: which skill groups were
matched, which are missing, and how the final **Good Fit / Potential Fit / No Fit**
label was reached.

> **Status: working local demo, not production.** Not ready for real candidate data. See
> [Known limitations](#known-limitations) and [docs/SECURITY_AND_PRIVACY.md](docs/SECURITY_AND_PRIVACY.md).

## How a decision is made

1. Text is extracted from a text-based PDF (scanned PDFs are not supported; there is no OCR).
2. Skills are detected with a rule-based vocabulary (aliases, negation and
   "aspirational" phrasing are filtered).
3. Required skills form **groups**. "Python OR Java" is one group, satisfied by either.
4. The decision policy (`required_skill_group_coverage_v2`) uses the share of required
   groups matched: **85% or more = Good Fit, 50% to under 85% = Potential Fit, under 50% = No Fit.**
   Comparisons use exact integers; preferred skills never change the label.
5. A TF-IDF + Logistic Regression model also predicts a label. It is reported
   separately (`model_label`, `model_probs`) and does **not** change the policy decision
   when skill evidence is available.

Skill matches are keyword evidence, not proof of competence. Human review is required.

## What you can do

Sign in as a recruiter, create jobs with required skill groups, add candidates, record and
withdraw consent, upload PDF resumes, screen a candidate against the exact job version, read the
explanation (matched and missing skills, coverage, decision policy), review the screening history,
and erase a candidate's personal data. Recruiters see only their own jobs and candidates.
Accounts are created by an administrator; there is no public registration.

## Quick start

Full instructions, environment variables, a demo script and troubleshooting are in
**[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**.

```powershell
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
copy .env.example .env                      # set DATABASE_URL and SMART_HIRE_AUTH_SECRET
alembic upgrade head
python -m scripts.create_recruiter --email you@example.com --name "Your Name" --role admin
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000    # terminal 1
cd frontend; npm ci; npm run dev                                   # terminal 2: http://localhost:5173
```

## Project layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI app, auth, SQLAlchemy models, screening, consent, retention and storage services |
| `src/` | Screening logic: PDF extraction, skill extractor, decision policy, analyzer |
| `frontend/` | React 19 + Vite + Tailwind UI (with Vitest tests, including a live end-to-end test) |
| `migrations/` | Alembic migrations (PostgreSQL) |
| `scripts/` | `create_recruiter`, `run_retention_sweep`, `demo_smoke_test`, `evaluate_decision_policy` |
| `models/` | Saved production model artifacts |
| `evaluation_results/` | Reports from model experiments |
| `docs/` | Setup guide, API, security notes, ML evaluation, architecture, demo files |

## Tests

```powershell
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .
cd frontend; npm test; npm run build
```

`requirements-training.txt` adds packages for retraining/evaluating models. Model artifacts are
Python pickles (`joblib`): only load files you created yourself.

## Known limitations

- The text model is weak (about 72% accuracy, below the 73% of always answering "No Fit";
  macro-F1 about 0.61) and is shown only as a reference. The final label comes from required-skill
  coverage. See [docs/ML_AND_POLICY_EVALUATION.md](docs/ML_AND_POLICY_EVALUATION.md).
- The 85% / 50% thresholds have not been validated against hand-labelled outcomes.
- Skill matching is keyword-based and limited to a fixed vocabulary; there is no OCR.
- No encryption at rest, HTTPS setup, legal review of consent/retention, or penetration test.
- Not deployed anywhere; resume storage is local disk only.
