"""End-to-end check of the whole recruiter workflow against a real database.

    python -m scripts.demo_smoke_test

Uses DATABASE_URL (run `alembic upgrade head` first). It creates clearly-labelled test rows
("smoke-...") and leaves them in place, so point it at a development or scratch database,
never a database holding real candidate data.

Flow: admin account -> login -> job with required groups -> candidate -> consent -> PDF upload
-> screening (with idempotent replay) -> second run -> history -> a second recruiter is
refused on every route -> consent withdrawal blocks
screening -> erasure -> retention sweep -> anonymous access is refused.
"""

from __future__ import annotations

import os
import secrets
import sys
import tempfile
import uuid

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

STEPS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAILED: {message}")
        raise SystemExit(1)
    STEPS.append(message)
    print(f"  ok  {message}")


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    contents = document.tobytes()
    document.close()
    return contents


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("Set DATABASE_URL first (see .env.example).")
        return 2
    os.environ.setdefault("SMART_HIRE_AUTH_SECRET", secrets.token_urlsafe(48))
    os.environ.pop("SMART_HIRE_DEVELOPMENT_MODE", None)
    storage_dir = None
    if not os.environ.get("SMART_HIRE_RESUME_STORAGE_ROOT"):
        storage_dir = tempfile.TemporaryDirectory()
        os.environ["SMART_HIRE_RESUME_STORAGE_ROOT"] = storage_dir.name

    from backend.db.models import RecruiterRole
    from backend.db.session import get_engine
    from backend.main import app
    from scripts.create_recruiter import create_or_update_recruiter

    tag = uuid.uuid4().hex[:8]
    email, password = f"smoke-{tag}@example.test", secrets.token_urlsafe(16)
    with Session(get_engine()) as session:
        create_or_update_recruiter(
            session, email=email, display_name="Smoke Admin", role=RecruiterRole.ADMIN, password=password
        )

    print("SmartHire workflow smoke test")
    with TestClient(app) as client:
        check(client.get("/api/jobs/").status_code == 401, "anonymous request to /api/jobs/ is refused (401)")
        check(client.post("/api/auth/login", json={"email": email, "password": "wrong-password-x"}).status_code == 401,
              "wrong password is refused (401)")
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        check(login.status_code == 200, "recruiter can sign in")
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        job = client.post("/api/jobs/", headers=headers, json={
            "title": f"Smoke Backend Engineer {tag}", "status": "open",
            "description": "Backend role building APIs with Python and PostgreSQL, deployed with Docker.",
            "required_skill_groups": [["Python", "Java"], ["PostgreSQL"], ["Docker"]],
            "preferred_skill_groups": [["AWS"]],
        })
        check(job.status_code == 201, "job with required skill groups is created")
        job = job.json()

        created = client.post("/api/candidates/", headers=headers, json={
            "job_id": job["id"], "display_name": f"Smoke Candidate {tag}", "email": f"cand-{tag}@example.test"})
        check(created.status_code == 201, "candidate is added to the job")
        candidate_id = created.json()["candidate"]["id"]
        application_id = created.json()["application"]["id"]

        def consent(purpose):
            return client.post(f"/api/candidates/{candidate_id}/consents/", headers=headers, json={
                "purpose": purpose, "notice_version": "v1", "capture_method": "smoke test"})

        storage_consent, screening_consent = consent("document_storage"), consent("screening")
        check(storage_consent.status_code == 201 and screening_consent.status_code == 201,
              "storage and screening consents are recorded")

        pdf = make_pdf("Backend engineer. Five years of Python development. Designed APIs "
                       "backed by PostgreSQL databases. Deployed services with Docker.")
        upload = client.post(f"/api/candidates/{candidate_id}/resumes/", headers=headers,
                             files={"file": ("resume.pdf", pdf, "application/pdf")},
                             data={"consent_record_id": storage_consent.json()["id"]})
        check(upload.status_code == 201, "PDF resume is uploaded and stored privately")
        resume_id = upload.json()["id"]
        check(upload.json()["retention_until"] is not None, "the resume has an automatic retention deadline")

        body = {"resume_document_id": resume_id, "job_description_version_id": job["description_version_id"]}
        key = f"smoke-{tag}-1"
        first = client.post(f"/api/applications/{application_id}/screenings/", headers={**headers, "Idempotency-Key": key}, json=body)
        check(first.status_code in (200, 201), f"screening runs (HTTP {first.status_code})")
        result = first.json()
        check(result["final_label"] in ("Good Fit", "Potential Fit", "No Fit"), f"final label returned: {result['final_label']}")
        check(result["required_group_count"] == 3, "all three required groups were evaluated")
        print(f"      coverage {result['required_skill_coverage_percentage']}%, text-model reference: {result['model_label']}")

        replay = client.post(f"/api/applications/{application_id}/screenings/", headers={**headers, "Idempotency-Key": key}, json=body)
        check(replay.json()["id"] == result["id"], "repeating the same Idempotency-Key returns the same run")
        second = client.post(f"/api/applications/{application_id}/screenings/", headers={**headers, "Idempotency-Key": f"smoke-{tag}-2"}, json=body)
        check(second.json()["run_number"] == 2, "a new key creates run #2")
        history = client.get(f"/api/applications/{application_id}/screenings/", headers=headers).json()
        check(len(history) == 2, "screening history lists both runs")

        # A second, ordinary recruiter must not reach the first recruiter's data.
        other_email, other_password = f"smoke-other-{tag}@example.test", secrets.token_urlsafe(16)
        with Session(get_engine()) as session:
            create_or_update_recruiter(
                session, email=other_email, display_name="Smoke Other", role=RecruiterRole.RECRUITER, password=other_password
            )
        other = client.post("/api/auth/login", json={"email": other_email, "password": other_password})
        other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
        check(client.get(f"/api/jobs/{job['id']}", headers=other_headers).status_code == 403, "another recruiter cannot read the job (403)")
        check(job["id"] not in [j["id"] for j in client.get("/api/jobs/", headers=other_headers).json()], "the job is absent from the other recruiter's list")
        check(client.get(f"/api/candidates/{candidate_id}", headers=other_headers).status_code == 403, "another recruiter cannot read the candidate (403)")
        check(client.get(f"/api/candidates/{candidate_id}/screenings/", headers=other_headers).status_code == 403, "another recruiter cannot read screening history (403)")
        check(client.post(f"/api/applications/{application_id}/screenings/", headers={**other_headers, "Idempotency-Key": f"smoke-{tag}-x"}, json=body).status_code == 403,
              "another recruiter cannot run a screening (403)")
        check(client.get(f"/api/resumes/{resume_id}/download", headers=other_headers).status_code == 403, "another recruiter cannot download the resume (403)")
        check(client.post("/api/admin/recruiters", headers=other_headers, json={"email": "x@example.test", "display_name": "X", "password": "a-long-password-1"}).status_code == 403,
              "a non-admin cannot create accounts (403)")

        withdrawn = client.post(f"/api/consents/{screening_consent.json()['id']}/withdraw", headers=headers)
        check(withdrawn.status_code == 200, "screening consent is withdrawn")
        blocked = client.post(f"/api/applications/{application_id}/screenings/", headers={**headers, "Idempotency-Key": f"smoke-{tag}-3"}, json=body)
        check(blocked.status_code == 403, "screening is blocked after consent withdrawal (403)")

        erased = client.post(f"/api/candidates/{candidate_id}/anonymize", headers=headers)
        check(erased.status_code == 200 and erased.json()["resumes_deleted"] == 1, "erasure deletes the resume file")
        profile = client.get(f"/api/candidates/{candidate_id}", headers=headers).json()
        check(profile["display_name"] == "Anonymized candidate" and profile["email"] is None, "personal data is removed")
        check(len(client.get(f"/api/applications/{application_id}/screenings/", headers=headers).json()) == 2,
              "screening history survives erasure")

        sweep = client.post("/api/admin/retention/sweep", headers=headers)
        check(sweep.status_code == 200, "retention sweep runs")

    if storage_dir is not None:
        storage_dir.cleanup()
    print(f"\nSMOKE TEST PASSED ({len(STEPS)} checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
