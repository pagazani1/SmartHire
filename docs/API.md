# SmartHire API

The interactive OpenAPI specification is available at `/docs` while the
FastAPI service is running.

## Job management

Jobs are associated with an existing recruiter through
`created_by_recruiter_id`. There is no public recruiter-creation endpoint.
For local development only, create or reuse the stable seed recruiter with:

```text
python -m scripts.seed_dev_recruiter --development
```

The command refuses non-local PostgreSQL hosts. Use the printed recruiter ID
when creating a job. Authentication and recruiter ownership enforcement are
not implemented; do not expose these endpoints to untrusted clients.

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
legacy endpoint remains stateless. Persistent screening writes are deferred
until private document storage is selected and implemented. Returns `404`
when the candidate does not exist.

## Existing stateless analysis

### `POST /api/analyze/`

The existing contract is unchanged. Submit `multipart/form-data` containing:

- `file`: PDF resume
- `jd_text`: non-empty job description

The response contains the final decision, original model probabilities,
skill evidence, decision explanation, and extraction limitations. The PDF
is temporarily stored and deleted after analysis. This route does not create
candidate, application, document, or screening records.

## Security and deployment status

Authentication and authorization are not implemented. Candidate endpoints
return personal profile fields and do not filter by recruiter ownership.
Do not expose them on an untrusted network or use real candidate records
until access control, consent, retention, and audit policies are implemented.
The handlers do not log candidate profile values.
