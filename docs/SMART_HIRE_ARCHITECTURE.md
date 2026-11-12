# SmartHire Architecture Audit and Database Design

**Status:** Architecture baseline. The initial PostgreSQL/SQLAlchemy/Alembic
infrastructure is implemented as documented in [DATABASE_SETUP.md](./DATABASE_SETUP.md).
The screening API remains stateless; persistence workflows still require
separate implementation and policy approval.

## 1. Executive summary

SmartHire is currently a local, stateless resume-screening prototype:

- A synchronous FastAPI service exposes health and PDF-analysis endpoints.
- The analysis path extracts PDF text, runs the existing combined-text
  TF-IDF/Logistic Regression classifier, extracts skill evidence, and applies
  the production required-skill-group decision policy.
- A React 19/Vite 6/Tailwind 4 frontend submits one PDF and one job
  description to the API and renders the final assessment and skill evidence.
- The API does not persist recruiters, jobs, candidates, applications,
  documents, screening runs, or verification records.
- Authentication is not implemented. The current upload is temporarily stored
  and deleted after analysis.
- `src/background_verification.py` and `src/candidate_report.py` implement
  in-memory verification and report assembly; they are not database modules.

The recommended next persistence layer is PostgreSQL, introduced behind
repositories and versioned Alembic migrations. Keep the existing screening
pipeline and `/api/analyze/` response behavior intact. Add persistent
application-scoped workflows as new endpoints, initially opt-in, after
consent, authorization, retention, and storage policies are approved.

## 2. Current architecture audit

### Backend and screening flow

`backend/main.py` defines the FastAPI application and these endpoints:

| Endpoint | Request | Response / behavior |
|---|---|---|
| `GET /api/health/` | None | `{"status":"ok"}` |
| `POST /api/analyze/` | `multipart/form-data`: `file` (PDF), `jd_text` (string) | `AnalysisResponse`; 422 for invalid/missing JD or file, 400 for unreadable/no-text PDF, sanitized 500 for internal failures |

For `/api/analyze/`, the backend streams the upload to a temporary directory,
calls `src.resume_analyzer.analyze_resume()`, then removes the directory. It
does not retain the PDF or extracted text. The response includes the final
`label`, original model `probs`, `model_label`, `model_probs`, decision policy
and explanation, matched/missing skill information, grouped required and
preferred skills, coverage metrics, and extraction limitations.

`src/resume_analyzer.py` composes existing functions:

1. `src.pdf_extractor.extract_text_from_pdf()` extracts text from text-based
   PDFs; OCR is not available.
2. `predict.predict()` loads `models/tfidf_vectorizer.joblib` and
   `models/logistic_regression.joblib`, cleans and combines resume/JD text,
   transforms it, and returns the original model label and probabilities.
3. `src.skill_extractor.analyse_skills()` independently extracts skills and
   classifies/group requirements using a fixed vocabulary and rules.
4. `src.decision_policy.apply_decision_policy()` uses required skill group
   coverage for the final recruiter-facing label when evidence is available.
   The model output is preserved separately. Policy details are in
   [production_decision_policy.md](../evaluation_results/production_decision_policy.md).

Do not change these existing algorithms as part of database implementation.
Persist the returned output as a screening record; do not reimplement
extraction or prediction in repositories or API handlers.

### Frontend

`frontend/` contains the Vite entry point (`src/main.jsx`), application shell
(`src/App.jsx`), and reusable components under `src/components/`. The current
`App` submits `file` and `jd_text` to `/api/analyze/`; `AnalysisResults.jsx`
renders the final classification, assessment summary, matched/missing skills,
coverage, and caveats. It is a single-analysis workflow and has no candidate,
job, or application navigation or persistence.

The Vite development server proxies `/api` to `http://127.0.0.1:8000`.
Production hosting and API origin configuration remain deployment decisions.

### Models, dependencies, tests

The production model artifacts are local Joblib files under `models/`.
`models/hybrid_experimental/` contains a separate experimental bundle and
must not be treated as the production classifier. `data/raw/` and
`data/processed/` are local data/cache locations and are Git-ignored.

`requirements.txt` includes FastAPI, Pydantic transitively, PyMuPDF,
scikit-learn, Joblib, data/ML packages, and multipart support. It does **not**
declare SQLAlchemy, Alembic, a PostgreSQL driver, or another database client.
`frontend/package.json` declares React, Vite, Tailwind, and the React Vite
plugin; no frontend database/auth client exists.

Tests use Python `unittest` and FastAPI `TestClient`. Existing coverage
includes PDF extraction, prediction, skill extraction, the decision policy,
the API contract, candidate-report assembly, and in-memory background
verification. There are no database, migration, persistence, authorization,
or frontend automated-test modules at present.

### Existing persistence and verification functionality

No database connection, ORM, SQL schema, migration setup, or repository layer
was found. `src/background_verification.py` stores claims in a Python list for
the lifetime of a `BackgroundVerifier` instance. `src/candidate_report.py`
combines prediction, skill analysis, and that verifier into a plain-text
report. Neither module persists data between requests or process restarts.
The verification code explicitly requires human review and keeps verification
status independent from fit classification; retain that separation.

## 3. Proposed relational model

### Design choices

- **Database:** PostgreSQL as the first durable store. It provides transactions,
  constraints, indexing, UUIDs, and JSONB for immutable model-output
  snapshots. JSONB is supplementary; core relationships and queryable skill
  evidence remain relational.
- **Keys:** UUID primary keys generated by the application or database.
- **Time:** UTC `timestamptz` columns for creation, update, and review events.
- **Ownership:** `jobs.created_by_recruiter_id` and application-level access
  checks establish ownership. For organizations with multiple recruiters,
  introduce an `organizations` table and scope all records to a tenant before
  enabling multi-tenant access.
- **Deletion:** Prefer retention-driven soft deletion/anonymization and
  explicit purge workflows over broad cascading deletes. Use restrictive
  foreign keys for records that must remain auditable.
- **Snapshots:** A screening references the exact immutable job-description
  version and resume-document version analyzed. Re-running a screen creates
  a new result rather than overwriting old evidence.

The following is a logical schema, not executable migration SQL. `NOT NULL`,
`CHECK`, `UNIQUE`, and foreign-key rules are part of the proposal.

### Core records

| Table | Important columns and constraints |
|---|---|
| `recruiters` | `id UUID PK`; `email_normalized TEXT NOT NULL UNIQUE`; `display_name TEXT NOT NULL`; future `password_hash` or external identity subject; `status CHECK` (e.g. active/disabled); `created_at`, `updated_at`, optional `disabled_at`. Never store plaintext passwords. Authentication is out of scope for this phase. |
| `jobs` | `id UUID PK`; `created_by_recruiter_id UUID NOT NULL FK recruiters`; `title TEXT NOT NULL`; `department`, `location` optional; `status CHECK` (draft/open/paused/closed/archived); `created_at`, `updated_at`, optional `archived_at`. |
| `job_description_versions` | `id UUID PK`; `job_id UUID NOT NULL FK jobs`; `version INTEGER NOT NULL CHECK (version > 0)`; `description TEXT NOT NULL`; `created_by_recruiter_id UUID NOT NULL FK recruiters`; `created_at`; `UNIQUE(job_id, version)`. Published versions are immutable. |
| `candidates` | `id UUID PK`; `display_name TEXT NOT NULL`; optional email/contact fields only when needed; `created_at`, `updated_at`, optional `anonymized_at`. Do not impose global uniqueness on candidate email by default: it creates identity/tenant privacy and account-linking problems. If matching is required, define a controlled, tenant-scoped normalized identifier policy. |
| `applications` | `id UUID PK`; `candidate_id UUID NOT NULL FK candidates`; `job_id UUID NOT NULL FK jobs`; `status CHECK` (e.g. received/in_review/interview/offer/rejected/withdrawn/archived); `applied_at`, `created_at`, `updated_at`; optional `source`; `UNIQUE(job_id, candidate_id)` for one application row per candidate/job, with a defined reopen workflow. If repeat application rounds are required, add `round_number` and use `UNIQUE(job_id, candidate_id, round_number)`. |
| `candidate_consents` | `id UUID PK`; `candidate_id UUID NOT NULL FK candidates`; optional `application_id UUID FK applications`; `purpose CHECK` (screening/document storage/background verification/contact); `status CHECK` (granted/withdrawn/expired); `captured_at`, `withdrawn_at`, optional `expires_at`; `notice_version`, `capture_method`, and minimal proof/reference. A consent event is append-only; do not overwrite a grant when consent is withdrawn. |
| `resume_documents` | `id UUID PK`; `candidate_id UUID NOT NULL FK candidates`; `uploaded_by_recruiter_id UUID NOT NULL FK recruiters`; optional `consent_id UUID FK candidate_consents`; `storage_key TEXT NOT NULL UNIQUE`; `original_filename TEXT` (untrusted display metadata); `media_type`, `byte_size CHECK (byte_size > 0)`, scoped `sha256`, `status CHECK` (quarantined/available/deleted/expired), `uploaded_at`, optional `deleted_at` and `retention_until`. Store the binary in private object storage, not a database blob or public web directory. |
| `screening_results` | `id UUID PK`; `application_id UUID NOT NULL FK applications`; `resume_document_id UUID NOT NULL FK resume_documents`; `job_description_version_id UUID NOT NULL FK job_description_versions`; `run_number INTEGER NOT NULL CHECK (run_number > 0)`; `final_label CHECK` (Good Fit/Potential Fit/No Fit); `model_label CHECK` (same classes); `model_probabilities JSONB NOT NULL`; `decision_policy_version TEXT NOT NULL`; `model_artifact_version TEXT NOT NULL`; `decision_explanation TEXT`; `required_group_count`, `matched_required_group_count` with `0 <= matched <= total`; optional coverage ratio; `created_at`; `UNIQUE(application_id, run_number)`. Keep this as an immutable record of each run. |
| `skills` | `id UUID PK`; `canonical_name TEXT NOT NULL UNIQUE`; optional `vocabulary_version` and active flag. Seed/maintain from the existing canonical skill vocabulary rather than adding a second parser. |
| `job_requirement_groups` | `id UUID PK`; `job_description_version_id UUID NOT NULL FK job_description_versions`; `kind CHECK` (required/preferred); `ordinal INTEGER NOT NULL`; optional display text; `UNIQUE(job_description_version_id, kind, ordinal)`. Each alternative group (e.g. FastAPI OR Flask) is one row. |
| `job_requirement_options` | `requirement_group_id UUID NOT NULL FK job_requirement_groups`; `skill_id UUID NOT NULL FK skills`; composite `PRIMARY KEY(requirement_group_id, skill_id)`. A group with multiple options expresses OR alternatives. |
| `screening_requirement_results` | `screening_result_id UUID NOT NULL FK screening_results`; `requirement_group_id UUID NOT NULL FK job_requirement_groups`; `matched BOOLEAN NOT NULL`; optional `matched_skill_id UUID FK skills`; composite `PRIMARY KEY(screening_result_id, requirement_group_id)`. Enforce that the group belongs to the screening's referenced job-description version in service logic or a database trigger. |
| `screening_skill_evidence` | `screening_result_id UUID NOT NULL FK screening_results`; `skill_id UUID NOT NULL FK skills`; `evidence_kind CHECK` (resume_detected/JD_required/JD_preferred/JD_unclassified/matched/missing); optional source span/reference; composite `PRIMARY KEY(screening_result_id, skill_id, evidence_kind)`. Store only necessary evidence, not full extracted resume text by default. |
| `background_verification_records` | `id UUID PK`; `application_id UUID NOT NULL FK applications`; optional `screening_result_id UUID FK screening_results`; `claim_category TEXT NOT NULL`; `claim_text TEXT NOT NULL`; `status CHECK` (Pending/Verified/Unable to Verify/Discrepancy Found); `reviewer_recruiter_id UUID FK recruiters`; optional private evidence `storage_key` and reviewer notes; `created_at`, `updated_at`, optional `reviewed_at`. Only a human reviewer may set a reviewed status. |
| `background_verification_events` | `id UUID PK`; `verification_record_id UUID NOT NULL FK background_verification_records`; `changed_by_recruiter_id UUID NOT NULL FK recruiters`; `old_status`, `new_status`, optional change note; `created_at`. Append-only history of review transitions. |

### Relationships

```text
recruiters 1 ── * jobs
jobs 1 ── * job_description_versions
job_description_versions 1 ── * job_requirement_groups
job_requirement_groups * ── * skills (via job_requirement_options)

candidates 1 ── * applications * ── 1 jobs
candidates 1 ── * resume_documents
candidates 1 ── * candidate_consents
applications 1 ── * screening_results
screening_results * ── 1 resume_documents
screening_results * ── 1 job_description_versions
screening_results 1 ── * screening_requirement_results
screening_results 1 ── * screening_skill_evidence * ── 1 skills
applications 1 ── * background_verification_records
background_verification_records 1 ── * background_verification_events
```

### Important indexes

- `recruiters(email_normalized)` unique.
- `jobs(created_by_recruiter_id, status, updated_at DESC)`.
- `job_description_versions(job_id, version DESC)` unique.
- `applications(job_id, status, created_at DESC)` and
  `applications(candidate_id, created_at DESC)`; unique application key as
  described above.
- `resume_documents(candidate_id, uploaded_at DESC)` and
  `(status, retention_until)` for lifecycle processing. Avoid a globally
  searchable hash index; scope duplicate detection to a candidate/tenant.
- `screening_results(application_id, created_at DESC)` and unique
  `(application_id, run_number)`.
- `screening_requirement_results(requirement_group_id, matched)` and
  `screening_skill_evidence(skill_id, evidence_kind)` where analytics require
  them.
- `background_verification_records(application_id, status, updated_at DESC)`
  and `background_verification_events(verification_record_id, created_at)`.
- Consent lookup indexes on `(candidate_id, purpose, status)` and
  `(application_id, purpose, status)`; ensure every processing request checks
  current consent and withdrawal state.

Use database constraints for stable invariants and indexes for actual query
paths. Avoid putting the only copy of requirements or skill results in opaque
JSON; retain JSONB only for exact API/model snapshots and forward compatibility.

## 4. Connecting screening to records

1. Recruiter creates a job; the backend creates an immutable job-description
   version and stores the JD's structured requirement groups/options using the
   same skill extractor output. Existing `skill_extractor` remains the source
   of canonical names and alternative grouping.
2. A candidate/application is created under an explicit consent and document
   handling flow. The PDF is validated and placed in private quarantine/object
   storage; metadata is inserted into `resume_documents`.
3. A screening request identifies an application, document, and exact job
   version. Authorization must verify recruiter access and the relevant
   consent before processing.
4. The service passes the document's temporary local path and versioned JD
   text to the existing `analyze_resume()` function. Keep extraction, model
   probabilities, and decision policy unchanged.
5. In one database transaction, store a new immutable `screening_results`
   row, its normalized group/skill evidence, and the exact model and policy
   versions. Do not silently overwrite previous screenings when a model,
   policy, document, or JD changes.
6. Return the existing response fields for UI compatibility, adding stable
   screening/application identifiers only on the new persistent workflow.
   The current Analysis Results component can continue rendering the same
   response shape; a future candidate/job dashboard can query stored records.
7. Verification records are created and transitioned separately by a human
   reviewer. They must never change `final_label` or model probabilities.

Keep `/api/analyze/` stateless and compatible in the first persistence release.
Add persistent endpoints such as jobs, candidates, applications, and
application-scoped screenings only after authorization and consent are
implemented. If the existing endpoint is later changed to persist, explicitly
version/document that behavior and provide a clear retention notice.

## 5. Proposed future backend structure

Keep current modules in place during the transition; do not rename or move
them. Add bounded modules around the new persistence use cases:

```text
backend/
  main.py                       # existing app and legacy endpoints
  api/
    dependencies.py             # DB transaction, current recruiter, authorization
    routers/
      health.py
      screening.py              # existing stateless route + new persistent route
      jobs.py
      candidates.py
      applications.py
      verification.py
    schemas/
      screening.py
      jobs.py
      candidates.py
      applications.py
      verification.py
  core/
    config.py                   # environment-backed settings; no secrets in Git
    security.py                 # future authentication/authorization primitives
  db/
    base.py
    session.py
    models/                     # SQLAlchemy mappings
    repositories/               # persistence/query boundary
  services/
    screening_service.py        # composes existing resume_analyzer
    document_service.py         # validation, quarantine, storage, lifecycle
    consent_service.py
    verification_service.py
  migrations/                   # Alembic environment and versioned revisions

src/                            # existing ML/PDF/skill/decision modules, reused
tests/
  api/
  services/
  repositories/
  migrations/
```

This is a target, not a request to reorganize current files. Route handlers
should validate and authorize, then call a service. Services should own
transactions and call repositories. Repositories should not contain ML or
decision-policy logic. For the existing synchronous FastAPI handlers, a
synchronous SQLAlchemy session is the lowest-change initial approach; choose
async only if deployment and load requirements justify its extra complexity.

## 6. Migration strategy

1. **Approve product and governance decisions:** PostgreSQL deployment,
   tenant model, candidate consent language/purposes, access roles, retention
   periods, data residency, deletion/export requirements, and audit needs.
2. **Add dependencies only after approval:** SQLAlchemy 2.x, Alembic, and a
   PostgreSQL driver (for example psycopg). Pin compatible versions and add
   test-only database support as needed. No dependency is installed or
   required by this audit.
3. **Configure database access:** load connection configuration from secret
   management/environment, enforce TLS and least-privilege DB credentials,
   establish health/readiness behavior, and ensure tests use an isolated
   database. Never commit connection strings or production credentials.
4. **Create baseline migrations:** create core tables, constraints, indexes,
   and versioned requirements. Test clean install, upgrade, and rollback
   strategy against an ephemeral PostgreSQL instance; do not rely on SQLite
   to validate PostgreSQL-specific behavior.
5. **Add repositories/services without changing inference:** test CRUD,
   transaction rollback, authorization predicates, retention filtering, and
   error mapping independently of ML.
6. **Keep the legacy analysis route stateless:** verify its exact request,
   response, status codes, temporary-file cleanup, and model outputs remain
   compatible. Introduce persistent endpoints behind consent and
   authorization.
7. **Add opt-in persistence:** first persist job/application-scoped screenings
   for new workflows. Preserve a new immutable row for every rerun, including
   model/policy version metadata. Do not backfill synthetic candidate records
   or invent consent for previous uploads.
8. **Operationalize lifecycle and audit:** add document expiration/deletion
   jobs, consent withdrawal handling, candidate export/anonymization, access
   audit logs, backup/restore drills, and monitoring before production use.
9. **Only then build recruiter workflows:** job/application lists, candidate
   detail, screening history, and human verification screens can consume the
   new APIs. Keep frontend `AnalysisResults` as a reusable screening-summary
   component.

## 7. Reuse rather than rewrite

- `src/pdf_extractor.py`: current PDF validation, text extraction, cleanup,
  and explicit no-OCR errors.
- `predict.py` and saved model artifacts: production TF-IDF/Logistic
  Regression inference and unmodified probability output.
- `src/skill_extractor.py`: canonical skill vocabulary, required/preferred
  classification, alternative requirement groups, and coverage.
- `src/decision_policy.py`: final recruiter-facing category and explanation;
  persist its policy identifier with each run.
- `src/resume_analyzer.py`: current orchestration boundary for a PDF/JD
  screening.
- `backend/main.py` response schema and API error handling: preserve the
  existing contract when adding persistent routes.
- `frontend/src/components/AnalysisResults.jsx`: reusable final assessment,
  matched/missing skills, and caveats.
- `frontend/src/components/FileUpload.jsx` and
  `JobDescriptionField.jsx`: upload and JD input controls.
- `frontend/src/components/Sidebar.jsx` and `App.jsx`: existing layout and
  recruiter workspace shell; extend navigation when persistent pages are
  approved.
- `src/background_verification.py` and `src/candidate_report.py`: business
  semantics, status vocabulary, and human-review separation are useful
  references. Replace only their in-memory storage boundary with a persistence
  service after preserving tests and status invariants; do not treat them as
  current database implementations.

## 8. Security, privacy, consent, and storage risks

- **Authentication and authorization are absent.** Do not expose persistent
  recruiter or candidate records until every route enforces an authenticated
  recruiter identity and ownership/tenant-scoped access. Avoid insecure
  direct-object-reference access by UUID.
- **Candidate privacy and consent:** identify lawful basis and jurisdictional
  obligations with counsel. Capture informed, purpose-specific consent where
  required before storing or analyzing documents; explain automated analysis,
  retention, sharing, and withdrawal. Withdrawal must block future processing
  and start the approved deletion/retention workflow.
- **Data minimization:** avoid storing extracted full resume text unless an
  explicit product need and retention policy justify it. Store skill evidence
  and screening snapshots sufficient for audit. Avoid government IDs,
  sensitive demographic data, and unnecessary verification evidence.
- **Document security:** store PDFs in private encrypted object storage with
  random keys, short-lived signed access, encryption in transit/at rest,
  malware/content checks, size/type limits, and retention deletion. Treat
  filename, MIME type, and PDF content as untrusted. Do not serve from a public
  static directory or log resume/JD contents.
- **PII protection:** minimize candidate contact fields; encrypt especially
  sensitive fields where appropriate; restrict access; use masked logs; do not
  globally deduplicate people by email or expose one recruiter's data to
  another.
- **Auditability:** record who viewed, screened, downloaded, changed
  verification, exported, or deleted sensitive records. Keep verification
  status human-controlled and separate from fit classification.
- **Fairness and decision risk:** labels and keyword matches are imperfect
  decision support, not proof of qualifications. Preserve explanation,
  model/policy version and human review. Review applicable employment-AI,
  anti-discrimination, accessibility, and notice requirements before
  deployment.
- **Secrets and operations:** use secret management, least privilege, backups
  with tested restore, encryption, key rotation, incident response, and
  retention enforcement. Do not commit credentials or production data.
- **Cross-tenant analytics:** analytics must be authorization-scoped and
  aggregate carefully; prevent small cohorts or exports from revealing
  personal data.

## 9. Remaining implementation sequence

1. Confirm PostgreSQL, single-tenant versus organization tenancy, consent
   workflow, retention policy, and whether initial use is internal-only.
2. Approve the schema and API boundary; specify user roles and ownership
   rules before designing authentication.
3. Before deployment, provision and validate an isolated PostgreSQL
   integration-test database; only SQLite migration round-trips are covered
   by the current ordinary test suite.
4. Review and apply the initial migration in a new, explicitly selected
   database after backup and migration-SQL review.
5. Implement repositories and services with transaction and authorization
   tests; add private object storage and deletion lifecycle.
6. Add persistent job/candidate/application/screening APIs while keeping
   `GET /api/health/` and current stateless `POST /api/analyze/` behavior
   unchanged.
7. Persist screening snapshots and normalized requirement/skill evidence;
   verify exact model probabilities and policy version round-trip.
8. Add reviewer-managed verification persistence and append-only status
   history, preserving the existing human-only semantics.
9. Extend the frontend incrementally using the current layout and result
   component; add API/UI tests and accessibility checks.
10. Complete security, privacy, threat-model, retention, backup/restore, and
    deployment reviews before production rollout.

## 10. Decision and open questions

The initial infrastructure uses PostgreSQL, SQLAlchemy, and Alembic as
requested. No PostgreSQL connection was configured or tested during this
implementation, and no candidate or screening data is persisted. The
following workflow and policy decisions remain open:

- single recruiter workspace versus organization/tenant support;
- whether candidates enter applications themselves or recruiters create
  candidate records;
- exact application lifecycle and reapplication behavior;
- consent and lawful-basis requirements for screening, document retention,
  and background verification;
- retention periods and deletion/anonymization obligations;
- private object-storage provider, region, malware scanning, and access
  controls;
- authentication provider, roles, and audit requirements;
- which screening evidence must be queryable versus retained only in the
  immutable response snapshot.

The ORM/migration layer is only a foundation: authentication, authorization,
repositories, consent enforcement, private document storage, retention jobs,
and persistent endpoints remain unimplemented. Keep the existing screening
pipeline and API stateless until those controls are designed and tested.
