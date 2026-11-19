# Local setup and demo guide

Tested with Python 3.12, Node 22, PostgreSQL 16. Commands are for Windows PowerShell; on
macOS/Linux use `source .venv/bin/activate` and `cp` instead of `copy`.

## 1. One-time setup

**1. Create the database** (PostgreSQL running; use SQL Shell (psql) or `psql -U postgres`):

```sql
CREATE USER smarthire_user WITH PASSWORD 'choose-a-password';
CREATE DATABASE smarthire OWNER smarthire_user;
```

**2. Backend**, from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

**3. Edit `.env`:** put your password in `DATABASE_URL`, and set the signing secret:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste the output after SMART_HIRE_AUTH_SECRET=
```

**4. Create the tables and your admin account:**

```powershell
alembic upgrade head
python -m scripts.create_recruiter --email you@example.com --name "Your Name" --role admin
```

You will be asked for a password (minimum 10 characters; typing is hidden).

**5. Frontend:**

```powershell
cd frontend
npm ci
```

## 2. Start the app (two terminals)

```powershell
# Terminal 1: backend (from the project folder, virtual environment active)
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: frontend
cd frontend
npm run dev
```

Open http://localhost:5173 and sign in. The first start takes about 10 seconds while the model
loads. API documentation is at http://127.0.0.1:8000/docs.

## 3. Environment variables

| Variable | Needed? | If missing or wrong |
|---|---|---|
| `DATABASE_URL` | Required | Startup of any database feature fails with "DATABASE_URL is not configured". A wrong password or stopped PostgreSQL returns HTTP 503 "The database is unavailable". |
| `SMART_HIRE_AUTH_SECRET` (32+ chars) | Required for sign-in | Sign-in returns 503 "Authentication signing secret is not configured". |
| `SMART_HIRE_RESUME_STORAGE_ROOT` | Optional | Uses the per-user application data folder. |
| `SMART_HIRE_RESUME_RETENTION_DAYS` | Optional (180) | Invalid values fall back to 180; `0` removes the time limit. |
| `SMART_HIRE_ENV` | Optional | Set `production` to refuse unsafe settings at startup. |
| `SMART_HIRE_DEVELOPMENT_MODE`, `SMART_HIRE_DEVELOPMENT_RECRUITER_ID` | Leave off | Only for API tools; the web app always asks you to sign in. |

## 4. Demonstration script (about 10 minutes)

Sample files are in `docs/demo/`. The results below are what the real system returned for
exactly these inputs.

1. **Sign in** with your admin account. Try a wrong password first: you get "Invalid email or
   password." The **Dashboard** shows your checklist.
2. **Jobs → New job.** Title `Demo Backend Engineer`; description `Backend engineer building APIs
   with Python and PostgreSQL, deployed with Docker.` Required skill groups (one per line):
   `Python | Java`, `PostgreSQL`, `Docker`. Preferred: `AWS`. Note that `Python | Java` is one
   group, satisfied by either. Try adding a line `REST`: the form refuses it and lists the
   supported skills. Remove it and click **Create job**.
3. **Candidates → Add.** Name `Asha Rao`, applying for your job. In her panel click **Record
   consent** twice: once for *Store the resume document*, once for *Screen the candidate*.
   Notice **Upload** is disabled until storage consent exists.
4. **Upload** `docs/demo/sample_resume_strong.pdf`, then click **Screen this candidate**.
5. **Run Persistent Screening.** Expected: **Good Fit**, 3 of 3 required groups (100%), preferred
   skill AWS matched. A blue note explains that the text model said *Potential Fit* but the
   skill coverage decides. Click **Run** again: a new run is recorded automatically, and the
   history table shows **2 runs**.
6. **Different outcomes.** Repeat steps 3 to 5 for `Ravi Menon` with `sample_resume_partial.pdf`
   (expected **Potential Fit**, 2 of 3, PostgreSQL missing) and `Meera Iyer` with
   `sample_resume_weak.pdf` (expected **No Fit**, 0 of 3). The text model predicted *Potential Fit*
   for all three, including the graphic designer: that is why it is only a reference.
7. **Consent withdrawal.** On Asha's panel click **Withdraw** next to the screening consent and
   confirm. Screen her again: it is refused with "Active screening consent is required…".
8. **Resume deletion and erasure.** Click **Delete** on her resume, then **Erase this candidate's
   data** and confirm. Her name and contact details disappear; her screening history stays.
9. **Access control.** In a terminal run
   `python -m scripts.create_recruiter --email other@example.com --name "Other"`, sign out, sign in
   as that user: the job and candidates you created are not visible.
10. **Quick resume check** (sidebar): upload a PDF and paste a job description for a one-off
    result that stores nothing.

If a job has no required skill groups, results show an amber **Manual review needed** banner
because the label then falls back to the text model's guess.

## 5. Tests

```powershell
# Backend (SQLite; no setup needed)
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .

# Backend against PostgreSQL (use a SCRATCH database: its tables are dropped after every test)
$env:SMART_HIRE_TEST_DATABASE_URL="postgresql+psycopg://smarthire_user:PASSWORD@localhost:5432/smarthire_test"
python -m unittest tests.test_phase4_security

# Frontend
cd frontend; npm test; npm run build

# Whole workflow against your running database (creates "smoke-" rows; use a development database)
python -m scripts.demo_smoke_test

# Real UI against the running servers (backend + frontend started, a recruiter account, a small PDF)
cd frontend
$env:E2E_BASE_URL="http://127.0.0.1:5173"; $env:E2E_EMAIL="you@example.com"
$env:E2E_PASSWORD="your-password"; $env:E2E_PDF="..\docs\demo\sample_resume_strong.pdf"
npm run test:e2e
```

Evaluate the model and policy on your own labelled data: see
[ML_AND_POLICY_EVALUATION.md](ML_AND_POLICY_EVALUATION.md).

## 6. Housekeeping

```powershell
python -m scripts.run_retention_sweep    # schedule daily with Task Scheduler
```

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "The SmartHire server is not responding" at sign-in | Backend not running or not on port 8000. Start Terminal 1. |
| "The database is unavailable" | PostgreSQL stopped, or `DATABASE_URL` wrong. Test with `psql`. |
| "Authentication signing secret is not configured" | Set `SMART_HIRE_AUTH_SECRET` (32+ characters) in `.env`, restart the backend. |
| "Too many failed sign-in attempts" | Wait 15 minutes, or restart the backend (the counter is in memory). |
| `alembic upgrade head` fails to connect | Check `DATABASE_URL`; create the database first (section 1). |
| Upload says "Record an active document-storage consent first" | Record consent for *Store the resume document* in the candidate's panel. |
| Job form rejects a skill | Only names from the supported list work; open "Supported skills" in the form. |
| A scanned PDF gives no results | There is no OCR. Use a text-based PDF. |

## 8. Not ready for real candidate data

See [SECURITY_AND_PRIVACY.md](SECURITY_AND_PRIVACY.md): no encryption at rest, no HTTPS setup, no legal
review of consent and retention, no penetration test, not deployed anywhere.
