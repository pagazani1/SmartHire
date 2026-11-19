# SmartHire API

The interactive OpenAPI specification is available at `/docs` while the
FastAPI service is running.

## Authentication

Every route except `GET /api/health/` and `POST /api/auth/login` requires a signed-in
recruiter. There is no public registration: accounts are created by an administrator with
`python -m scripts.create_recruiter` or `POST /api/admin/recruiters`.

- `POST /api/auth/login` with `{"email", "password"}` returns `{"access_token", "expires_at",
  "recruiter"}`. Send the token as `Authorization: Bearer <token>`. Tokens last 8 hours.
  Wrong credentials, unknown emails and disabled accounts all return the same `401`.
  After 5 failed attempts for one email and client address in 15 minutes, further attempts
  return `429` with a `Retry-After` header. A missing or short `SMART_HIRE_AUTH_SECRET`
  returns `503`.
- `GET /api/auth/me` returns the current recruiter.
- Any route returns `401` for a missing, invalid or expired token, and `503` with a plain
  message when the database cannot be reached.

**Ownership.** A recruiter sees only their own jobs, and only candidates who applied to one
of their jobs (with only the applications and screenings for those jobs). Other recruiters'
data returns `403`; unknown ids return `404`. Admins can see everything. The owner of a new
job is always the signed-in recruiter (`created_by_recruiter_id` is optional; only an admin
may name someone else).

## Job management

Skill names in requirement groups must come from the supported vocabulary (`GET /api/skills/`).
An unknown name returns `422` with a message naming it.

### `GET /api/skills/`

Returns `{"skills": [...]}`: every skill name that can be used in required or preferred groups
(145 at the time of writing, sorted case-insensitively).

### `POST /api/jobs/`

Creates a job and its first description version in one transaction. The
supported fields are `created_by_recruiter_id`, `title`, optional
`department`, `location`, and `status`, required `description`, and optional
`required_skill_groups` / `preferred_skill_groups`. Each inner skill group is
an OR group; separate groups are independently considered. Skill names must
match the existing canonical skill vocabulary. Employment type and experience
level are not currently represented in the database schema.

Request:

```json
{
  "created_by_recruiter_id": "00000000-0000-4000-8000-000000000001",
  "title": "Backend Engineer",
  "department": "Engineering",
  "location": "Remote",
  "status": "open",
  "description": "Build and maintain backend APIs.",
  "required_skill_groups": [["Python"], ["FastAPI", "Flask"]],
  "preferred_skill_groups": [["PostgreSQL"]]
}
```

Returns `201 Created` with the job, current description version, and grouped
requirements. Returns `404` if the recruiter does not exist, `409` for a
database conflict, and `422` for invalid fields or unsupported skills.

### `GET /api/jobs/`

Lists jobs newest first. Supports `limit` (default `50`, range `1`–`100`) and
`offset` (default `0`):

```text
GET /api/jobs/?limit=25&offset=0
```

Each job includes the latest description and requirement groups.

### `GET /api/jobs/{job_id}`

Returns one job with its latest description and required/preferred skill
groups. Returns `404` if the job does not exist.

### `PUT /api/jobs/{job_id}`

Updates any supplied editable fields (`title`, `department`, `location`,
`status`, `description`, `required_skill_groups`, and
`preferred_skill_groups`). At least one field is required. Omitted values are
preserved; nullable fields may be set to `null` where supported. A changed
description or requirement group creates a new immutable
`JobDescriptionVersion`; previous versions remain stored. Scalar-only changes
do not create a description version. Returns `404` for an unknown job, `409`
for a database conflict, and `422` for invalid updates.

## Candidate and application management

### `POST /api/candidates/`

Creates a candidate and an application for an existing job in a single
database transaction. `job_id` must reference a row already in `jobs`; this
API does not create recruiters or jobs.

Request:

```json
{
  "display_name": "Synthetic Candidate",
  "email": "candidate@example.test",
  "phone": "+1 555 0100",
  "job_id": "00000000-0000-4000-8000-000000000001"
}
```

`email` and `phone` are optional; surrounding whitespace is removed. When an
email is supplied, the server stores a case-folded copy in
`email_normalized` for lookup. Candidate emails are not globally unique.

Success response (`201 Created`):

```json
{
  "candidate": {
    "id": "00000000-0000-4000-8000-000000000002",
    "display_name": "Synthetic Candidate",
    "email": "candidate@example.test",
    "phone": "+1 555 0100",
    "created_at": "2026-11-12T12:00:00Z",
    "updated_at": "2026-11-12T12:00:00Z"
  },
  "application": {
    "id": "00000000-0000-4000-8000-000000000003",
    "job_id": "00000000-0000-4000-8000-000000000001",
    "status": "received",
    "created_at": "2026-11-12T12:00:00Z"
  }
}
```

Errors: `404` when `job_id` does not exist, `409` for a database uniqueness
conflict, and `422` for invalid request fields.

### `GET /api/candidates/`

Returns candidate profiles, newest first. Supports `limit` (default `50`,
range `1`–`100`) and `offset` (default `0`):

```text
GET /api/candidates/?limit=25&offset=0
```

### `GET /api/candidates/{candidate_id}`

Returns the candidate profile and linked applications. Returns `404` when
the candidate ID does not exist.

### `GET /api/candidates/{candidate_id}/screenings/`

Returns persisted screening history, newest first, with pagination parameters
matching the candidate list. Each record includes the final classification,
original model classification and probabilities, decision-policy version and
explanation, required skill evidence, and coverage counts.

Analyses submitted to `POST /api/analyze/` do not appear in this history: that
legacy endpoint remains stateless. Persistent screening writes are not
implemented yet. Returns `404` when the candidate does not exist.

## Consent

Consent is recorded by the recruiter on the candidate's behalf. SmartHire stores the record and
the server-side time it was saved; it does not prove the candidate agreed, so keep your evidence
elsewhere and reference it in `proof_reference`. Wording and legal basis need review before any
real candidate data is used.

- `POST /api/candidates/{id}/consents/` with `{"purpose", "notice_version", "capture_method",
  "proof_reference"?, "application_id"?, "expires_at"?}` returns `201`. `purpose` is one of
  `screening`, `document_storage`, `contact`, `background_verification`. The status and capture
  time cannot be supplied by the client. `expires_at` must be in the future. `409` for an
  erased candidate.
- `GET /api/candidates/{id}/consents/` lists them, newest first.
- `POST /api/consents/{consent_id}/withdraw` withdraws a consent (idempotent). Screening, resume
  upload and resume download that depend on it are refused immediately afterwards (`403`).

A consent past its `expires_at` is treated as invalid even before the housekeeping job marks its
stored status as `expired`.

## Candidate erasure

### `POST /api/candidates/{candidate_id}/anonymize`

Deletes the candidate's resume files, withdraws their consents, and replaces their name, email
and phone with blanks (`anonymized_at` is set). Application and screening rows remain, without
identity, for statistics and audit. If any file cannot be deleted the call returns `503` and
nothing about the candidate is changed, so it can be retried. Repeating it is safe.

## Persistent screening

### `POST /api/applications/{application_id}/screenings/`

Body: `{"resume_document_id", "job_description_version_id"}`. Header `Idempotency-Key` is
required (max 255 characters). Requires active `screening` and `document_storage` consents, a
resume belonging to the application's candidate, and a job description version of the
application's job. The result stores the requirement groups as they were at the time.
Repeating the same key with the same inputs returns the original run; the same key with
different inputs returns `409`. A new key creates the next `run_number`.

The response includes `final_label` (decided by required-group coverage, policy
`required_skill_group_coverage_v2`), `model_label` (a weak reference signal that can differ),
coverage counts, matched and missing skills, `decision_explanation` and evidence rows. If a job
has no usable required groups, coverage is `null` and the label falls back to the model: the UI
shows this as "manual review needed".

### `GET /api/applications/{application_id}/screenings/`

Lists past runs, newest first. `GET /api/candidates/{id}/screenings/` lists a candidate's runs for
the recruiter's own jobs.

## Administration

- `POST /api/admin/recruiters` (admin only) creates an account: `{"email", "display_name",
  "password" (min 10 characters), "role"?}`. `409` if the email exists.
- `POST /api/admin/retention/sweep` (admin only) runs the housekeeping pass: it deletes resume
  files past `retention_until`, then marks lapsed consents `expired`. The same pass can be
  scheduled with `python -m scripts.run_retention_sweep`.

## Private resume documents

Uploading requires access to the candidate (see ownership above) and a consent record for that
same candidate with purpose `document_storage`, status `granted`, no withdrawal timestamp, and
no expired `expires_at`. The uploader is the signed-in recruiter. Each stored resume gets a
`retention_until` date (default 180 days, `SMART_HIRE_RESUME_RETENTION_DAYS`, or the consent
expiry if sooner). Files are limited to 10 MiB. A `.pdf` extension and PDF-compatible MIME type
are hints only; the server also checks the PDF signature, parseability, page count, and
encryption. Encrypted PDFs are rejected. Valid scanned PDFs can be stored, but the analyzer
does not provide OCR.

### `POST /api/candidates/{candidate_id}/resumes/`

Submit `multipart/form-data` with:

- `file`: PDF resume
- `consent_record_id`: UUID of an active document-storage consent

Returns `201 Created` with metadata only:

```json
{
  "id": "00000000-0000-4000-8000-000000000001",
  "candidate_id": "00000000-0000-4000-8000-000000000002",
  "original_filename": "resume.pdf",
  "media_type": "application/pdf",
  "byte_size": 12345,
  "status": "available",
  "uploaded_at": "2026-11-12T12:00:00Z",
  "deleted_at": null,
  "retention_until": null
}
```

The response does not include the storage key or filesystem path. Returns
`404` for an unknown candidate, `403` for missing/invalid consent, `413` when
the streamed upload exceeds 10 MiB, `415` for a non-PDF filename or
incompatible media type, `422` for empty/invalid/encrypted PDF data, and
sanitized `500`/`503` errors for database or storage failures.

### `GET /api/candidates/{candidate_id}/resumes/`

Returns resume metadata newest first, excluding soft-deleted documents by
default. Accepts `limit` (default `50`, range `1`–`100`) and `offset`
(default `0`). Returns `404` for an unknown candidate.

### `GET /api/resumes/{resume_id}`

Returns metadata only, including for a soft-deleted record. It never returns
a storage key, absolute path, or public URL.

### `GET /api/resumes/{resume_id}/download`

Streams an available PDF through the backend with an attachment filename and
`Cache-Control: private, no-store`. Deleted, quarantined, expired, or
unavailable files cannot be downloaded. No public object URL is created.

### `DELETE /api/resumes/{resume_id}`

Returns `204 No Content` after deleting the private file and retaining the
database metadata row with status `deleted` and `deleted_at`. Screening-result
foreign keys and historical metadata are preserved. The document is first
made unavailable; if physical deletion fails, it remains quarantined so a
later retry can complete deletion. If the file is removed but the final
database update fails, retrying is safe.

The database and filesystem cannot participate in one atomic transaction.
Upload compensates for metadata persistence failures by attempting to delete
the stored object. A failed compensation may leave an orphan; an internal
reconciliation operation can identify storage keys without database metadata.
There is no public cleanup endpoint.

## Stateless analysis (quick check)

### `POST /api/analyze/`

Requires sign-in. The request and response contract is otherwise unchanged. Uploads over 10 MiB
return `413`, and files without a PDF signature return `400`.

The existing contract is unchanged. Submit `multipart/form-data` containing:

- `file`: PDF resume
- `jd_text`: non-empty job description

The response contains the final decision, original model probabilities,
skill evidence, decision explanation, and extraction limitations. The PDF
is temporarily stored and deleted after analysis. This route does not create
candidate, application, document, or screening records.

When reliable required skill groups and resume skill evidence are available,
the final `label` uses the required-group coverage policy: at least 85% is
`Good Fit`, at least 50% and below 85% is `Potential Fit`, and below 50% is
`No Fit`. The decision uses the exact matched/total counts, not a rounded
display percentage. Alternative skills in one group count as one requirement;
preferred skills do not contribute. The response's `decision_evidence`
contains `matched_required_group_count`, `required_skill_group_count`,
`missing_required_group_count`, `required_skill_coverage` (0–1),
`required_skill_coverage_percentage` (0–100), the matched and missing groups,
and `required_skill_evidence_status`. If there are no required groups or
evidence is unavailable, the final label falls back to `model_label` and
coverage is `null`. `model_label`, `model_probs`, and `probs` preserve the
original text-model result. See
[production_decision_policy.md](../evaluation_results/production_decision_policy.md)
for the policy and limitations.

## Security and deployment status

Authentication, per-recruiter ownership, consent capture and withdrawal, erasure, retention and a
durable audit trail (`audit_events`) are implemented. See
[SECURITY_AND_PRIVACY.md](SECURITY_AND_PRIVACY.md) for what is and is not covered. This is a local
demo build: it has not been deployed, penetration-tested or reviewed by a lawyer.
