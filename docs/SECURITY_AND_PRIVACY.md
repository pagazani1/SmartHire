# Security and privacy: what exists and what does not

This describes the current local-demo build honestly. It is not a compliance statement.

## Implemented and tested

| Control | Detail |
|---|---|
| Sign-in | Email and password; PBKDF2-HMAC-SHA256 with a random salt (310,000 iterations); signed 8-hour tokens; role re-read from the database on every request, so disabling an account takes effect immediately. |
| No public registration | Accounts come from `python -m scripts.create_recruiter` (hidden password prompt, never a command-line argument) or the admin-only `POST /api/admin/recruiters`. |
| Every data route is protected | Jobs, candidates, consents, resumes, screenings, skills, analysis and admin routes all return 401 without a valid token (tested for each route). |
| Ownership | A recruiter sees only their own jobs and the candidates who applied to them, with only the applications and screenings for those jobs. The job owner always comes from the token, never from the request body. Admins see all. |
| Login hardening | Failures are rate-limited per account+client and per client (HTTP 429); unknown email and wrong password look identical, including in timing. |
| Development shortcut | The no-login developer mode is off by default, needs a direct loopback connection, and is ignored when any proxy header (`X-Forwarded-For`, etc.) is present. With `SMART_HIRE_ENV=production` the server refuses to start if it is enabled or if the secret is missing. |
| Consent | Recorded, listed and withdrawn through the API and UI. Screening and resume upload/download require an active (not withdrawn, not expired) consent; withdrawal takes effect immediately. The capture time is set by the server. |
| Resume storage | Private directory outside the repository, random file names, no symlink following, SHA-256 integrity check, 10 MiB limit, PDF signature/parse/encryption checks. |
| Retention | Each resume gets a `retention_until` date (default 180 days, configurable, never later than the consent expiry). `scripts/run_retention_sweep.py` and the admin endpoint delete expired files and mark lapsed consents `expired`. |
| Erasure | `POST /api/candidates/{id}/anonymize` deletes resume files, withdraws consents and blanks name, email and phone. If a file cannot be deleted nothing is changed. |
| Audit trail | Sign-ins (success, failure, lockout), denied access, consent changes, uploads, screenings, erasure and retention actions are written to the `audit_events` table and the log. Entries hold identifiers only: no names, emails, passwords or resume text. |
| Error handling | Database outages return a plain 503 without connection details. |

## Not implemented or not verified

- **Encryption at rest.** Candidate names, emails and phone numbers are stored in plaintext in
  PostgreSQL, and resume files are stored unencrypted on disk. Rely on disk encryption and
  database access controls.
- **Transport security.** No HTTPS or production CORS configuration is included. Put the app
  behind a TLS-terminating reverse proxy before any non-local use, and set `SMART_HIRE_ENV=production`.
- **Token revocation.** Tokens are stateless; signing out removes the token from the browser but
  an already-issued token stays valid until it expires (8 hours). Disabling the account stops it.
- **Token storage in the browser.** The token is kept in `sessionStorage`, which scripts on the
  same page can read. An `httpOnly` cookie would be stronger but needs CSRF protection.
- **Rate limiting scope.** The login limiter is in memory for one server process. Multiple workers
  need a shared store or proxy-level limiting.
- **Multi-recruiter job sharing.** There is none: a candidate shared by two recruiters' jobs is
  seen by each recruiter only through their own job.
- **Legal basis.** The consent wording, retention period, erasure scope (screening rows are kept
  without identity) and audit retention have not been reviewed by a lawyer. In India, check the
  DPDP Act; elsewhere, the local equivalent.
- **Penetration testing, dependency auditing and load testing** have not been done.
- **Database-level guarantee** that a screening's resume belongs to the application's candidate.
  The service checks it (and tests cover it) but there is no composite foreign key.
- **Background verification** tables and code exist from earlier work, are not exposed through the
  API or UI, and should be treated as unused.

## Operating notes

- Never commit `.env`. Rotate `SMART_HIRE_AUTH_SECRET` to sign everyone out.
- Back up the database and the resume directory together; restoring one without the other leaves
  dangling records.
- Run `python -m scripts.run_retention_sweep` on a schedule (daily is typical).
