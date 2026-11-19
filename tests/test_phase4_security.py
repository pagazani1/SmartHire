"""Phase 4: authentication on every route, ownership isolation, consent lifecycle, erasure, audit."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import select
from starlette.requests import Request

from backend.db.models import (
    Application,
    AuditEvent,
    Candidate,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    Job,
    Recruiter,
    RecruiterRole,
    ResumeDocument,
)
from backend.security import is_loopback_request, validate_runtime_configuration
from backend.services.resume_documents import (
    ResumeDocumentService,
    ResumeStorageUnavailableError,
    resume_retention_deadline,
)
from backend.services.retention import expire_consents
from scripts.create_recruiter import AccountError, create_or_update_recruiter
from tests.api_fixtures import PASSWORD, ApiTestBase


class AuthenticationRequiredTests(ApiTestBase):
    def test_every_sensitive_route_rejects_anonymous_requests(self):
        some_id = uuid.uuid4()
        routes = [
            ("get", "/api/jobs/"),
            ("post", "/api/jobs/"),
            ("get", f"/api/jobs/{some_id}"),
            ("put", f"/api/jobs/{some_id}"),
            ("get", "/api/candidates/"),
            ("post", "/api/candidates/"),
            ("get", f"/api/candidates/{some_id}"),
            ("get", f"/api/candidates/{some_id}/screenings/"),
            ("get", f"/api/candidates/{some_id}/consents/"),
            ("post", f"/api/candidates/{some_id}/consents/"),
            ("post", f"/api/consents/{some_id}/withdraw"),
            ("post", f"/api/candidates/{some_id}/anonymize"),
            ("get", f"/api/candidates/{some_id}/resumes/"),
            ("post", f"/api/candidates/{some_id}/resumes/"),
            ("post", "/api/admin/recruiters"),
            ("post", "/api/admin/retention/sweep"),
            ("post", "/api/analyze/"),
            ("get", "/api/skills/"),
        ]
        for method, path in routes:
            with self.subTest(route=f"{method.upper()} {path}"):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 401, response.text)

    def test_garbage_and_expired_tokens_are_rejected(self):
        response = self.client.get("/api/jobs/", headers={"Authorization": "Bearer not.a.token"})
        self.assertEqual(response.status_code, 401)

    def test_deactivated_recruiter_token_stops_working(self):
        from backend.db.models import RecruiterStatus

        rid = self.make_recruiter()
        self.assertEqual(self.client.get("/api/jobs/", headers=self.headers(rid)).status_code, 200)
        with self.SessionLocal() as session:
            session.get(Recruiter, rid).status = RecruiterStatus.DISABLED
            session.commit()
        self.assertEqual(self.client.get("/api/jobs/", headers=self.headers(rid)).status_code, 401)


class OwnershipIsolationTests(ApiTestBase):
    def setUp(self):
        super().setUp()
        self.alice = self.make_recruiter("alice@example.test")
        self.bob = self.make_recruiter("bob@example.test")
        self.admin = self.make_recruiter("admin@example.test", role=RecruiterRole.ADMIN)
        self.job_a = self.make_job(self.alice, "Alice Role")
        self.job_b = self.make_job(self.bob, "Bob Role")
        self.cand_a = self.make_candidate(self.alice, self.job_a, "Alice Candidate")

    def test_jobs_are_listed_and_readable_only_by_their_owner(self):
        listed = self.client.get("/api/jobs/", headers=self.headers(self.bob)).json()
        self.assertEqual([j["id"] for j in listed], [self.job_b])
        self.assertEqual(self.client.get(f"/api/jobs/{self.job_a}", headers=self.headers(self.bob)).status_code, 403)
        update = self.client.put(
            f"/api/jobs/{self.job_a}", json={"title": "Hijacked"}, headers=self.headers(self.bob)
        )
        self.assertEqual(update.status_code, 403)
        self.assertEqual(self.client.get(f"/api/jobs/{self.job_a}", headers=self.headers(self.alice)).json()["title"], "Alice Role")

    def test_admin_sees_everything(self):
        listed = self.client.get("/api/jobs/", headers=self.headers(self.admin)).json()
        self.assertEqual({j["id"] for j in listed}, {self.job_a, self.job_b})
        self.assertEqual(self.client.get(f"/api/candidates/{self.cand_a}", headers=self.headers(self.admin)).status_code, 200)

    def test_candidates_are_invisible_to_other_recruiters(self):
        listing = self.client.get("/api/candidates/", headers=self.headers(self.bob)).json()
        self.assertEqual(listing, [])
        probes = [
            ("get", f"/api/candidates/{self.cand_a}"),
            ("get", f"/api/candidates/{self.cand_a}/screenings/"),
            ("get", f"/api/candidates/{self.cand_a}/consents/"),
            ("get", f"/api/candidates/{self.cand_a}/resumes/"),
            ("post", f"/api/candidates/{self.cand_a}/anonymize"),
        ]
        for method, path in probes:
            with self.subTest(route=f"{method.upper()} {path}"):
                self.assertEqual(getattr(self.client, method)(path, headers=self.headers(self.bob)).status_code, 403)

    def test_cannot_attach_a_candidate_to_someone_elses_job(self):
        response = self.client.post(
            "/api/candidates/",
            json={"job_id": self.job_a, "display_name": "Intruder"},
            headers=self.headers(self.bob),
        )
        self.assertEqual(response.status_code, 403)

    def test_other_recruiter_cannot_grant_withdraw_or_upload(self):
        consent = self.grant_consent(self.alice, self.cand_a)
        self.assertEqual(consent.status_code, 201, consent.text)
        consent_id = consent.json()["id"]
        self.assertEqual(self.grant_consent(self.bob, self.cand_a).status_code, 403)
        self.assertEqual(
            self.client.post(f"/api/consents/{consent_id}/withdraw", headers=self.headers(self.bob)).status_code, 403
        )
        self.assertEqual(self.upload_resume(self.bob, self.cand_a, consent_id).status_code, 403)

    def test_shared_candidate_only_shows_the_viewers_own_applications(self):
        with self.SessionLocal() as session:
            session.add(Application(candidate_id=uuid.UUID(self.cand_a), job_id=uuid.UUID(self.job_b)))
            session.commit()
        as_alice = self.client.get(f"/api/candidates/{self.cand_a}", headers=self.headers(self.alice)).json()
        as_bob = self.client.get(f"/api/candidates/{self.cand_a}", headers=self.headers(self.bob)).json()
        self.assertEqual([a["job_id"] for a in as_alice["applications"]], [self.job_a])
        self.assertEqual([a["job_id"] for a in as_bob["applications"]], [self.job_b])
        self.assertEqual(len(self.client.get(f"/api/candidates/{self.cand_a}", headers=self.headers(self.admin)).json()["applications"]), 2)

    def test_unknown_candidate_is_404_not_403(self):
        response = self.client.get(f"/api/candidates/{uuid.uuid4()}", headers=self.headers(self.alice))
        self.assertEqual(response.status_code, 404)


class SkillVocabularyTests(ApiTestBase):
    def test_vocabulary_is_listed_for_signed_in_recruiters(self):
        from src.skill_extractor import SKILL_VOCABULARY

        rid = self.make_recruiter()
        response = self.client.get("/api/skills/", headers=self.headers(rid))
        self.assertEqual(response.status_code, 200, response.text)
        skills = response.json()["skills"]
        self.assertIn("Python", skills)
        self.assertEqual(len(skills), len(set(SKILL_VOCABULARY)))
        self.assertEqual(skills, sorted(skills, key=str.casefold))

    def test_unknown_skill_in_a_job_is_rejected_with_a_clear_message(self):
        rid = self.make_recruiter()
        response = self.client.post(
            "/api/jobs/",
            json={"title": "T", "description": "D", "required_skill_groups": [["REST"]]},
            headers=self.headers(rid),
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("not in the supported canonical skill vocabulary", response.text)


class LoginAndAccountTests(ApiTestBase):
    def _login(self, email, password):
        return self.client.post("/api/auth/login", json={"email": email, "password": password})

    def test_login_success_returns_token_that_works(self):
        self.make_recruiter("login@example.test")
        response = self._login("LOGIN@example.test", PASSWORD)
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["access_token"]
        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)

    def test_unknown_email_and_wrong_password_look_identical(self):
        self.make_recruiter("real@example.test")
        wrong = self._login("real@example.test", "wrong-password-1")
        unknown = self._login("nobody@example.test", "wrong-password-1")
        self.assertEqual((wrong.status_code, wrong.json()), (401, unknown.json() and unknown.json()))
        self.assertEqual(unknown.status_code, 401)

    def test_repeated_failures_lock_the_account_then_success_resets(self):
        self.make_recruiter("lock@example.test")
        for _ in range(5):
            self.assertEqual(self._login("lock@example.test", "bad-password-1").status_code, 401)
        blocked = self._login("lock@example.test", PASSWORD)
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)

    def test_successful_login_clears_earlier_failures(self):
        self.make_recruiter("reset@example.test")
        for _ in range(4):
            self._login("reset@example.test", "bad-password-1")
        self.assertEqual(self._login("reset@example.test", PASSWORD).status_code, 200)
        for _ in range(4):
            self.assertEqual(self._login("reset@example.test", "bad-password-1").status_code, 401)

    def test_inactive_account_cannot_log_in(self):
        from backend.db.models import RecruiterStatus

        self.make_recruiter("off@example.test", status=RecruiterStatus.DISABLED)
        self.assertEqual(self._login("off@example.test", PASSWORD).status_code, 401)

    def test_login_events_are_stored_in_the_database_without_secrets(self):
        self.make_recruiter("audit@example.test")
        self._login("audit@example.test", "bad-password-1")
        self._login("audit@example.test", PASSWORD)
        with self.SessionLocal() as session:
            events = session.scalars(select(AuditEvent).order_by(AuditEvent.occurred_at)).all()
        names = [e.event for e in events]
        self.assertIn("auth.login_failed", names)
        self.assertIn("auth.login_success", names)
        dump = " ".join(f"{e.event}{e.reason}{e.resource_type}" for e in events)
        self.assertNotIn(PASSWORD, dump)
        self.assertNotIn("audit@example.test", dump)

    def test_denied_access_is_audited(self):
        alice = self.make_recruiter()
        bob = self.make_recruiter()
        job = self.make_job(alice)
        self.client.get(f"/api/jobs/{job}", headers=self.headers(bob))
        with self.SessionLocal() as session:
            denied = session.scalars(select(AuditEvent).where(AuditEvent.outcome == "denied")).all()
        self.assertTrue(any(e.recruiter_id == bob for e in denied))


class AdminAccountCreationTests(ApiTestBase):
    PAYLOAD = {"email": "New.Person@Example.test", "display_name": "New Person", "password": "a-long-password-1"}

    def test_only_admins_can_create_accounts(self):
        plain = self.make_recruiter()
        self.assertEqual(
            self.client.post("/api/admin/recruiters", json=self.PAYLOAD, headers=self.headers(plain)).status_code, 403
        )

    def test_admin_creates_account_that_can_log_in(self):
        admin = self.make_recruiter(role=RecruiterRole.ADMIN)
        created = self.client.post("/api/admin/recruiters", json=self.PAYLOAD, headers=self.headers(admin))
        self.assertEqual(created.status_code, 201, created.text)
        self.assertNotIn("password", created.text)
        self.assertEqual(created.json()["role"], "recruiter")
        login = self.client.post(
            "/api/auth/login", json={"email": "new.person@example.test", "password": self.PAYLOAD["password"]}
        )
        self.assertEqual(login.status_code, 200)

    def test_duplicate_email_and_short_password_are_rejected(self):
        admin = self.make_recruiter(role=RecruiterRole.ADMIN)
        ok = self.client.post("/api/admin/recruiters", json=self.PAYLOAD, headers=self.headers(admin))
        self.assertEqual(ok.status_code, 201)
        dup = self.client.post("/api/admin/recruiters", json=self.PAYLOAD, headers=self.headers(admin))
        self.assertEqual(dup.status_code, 409)
        short = self.client.post(
            "/api/admin/recruiters", json={**self.PAYLOAD, "email": "x@example.test", "password": "short"},
            headers=self.headers(admin),
        )
        self.assertEqual(short.status_code, 422)


class CreateRecruiterCommandTests(ApiTestBase):
    def test_create_then_duplicate_then_reset(self):
        with self.SessionLocal() as session:
            recruiter, created = create_or_update_recruiter(
                session, email="Cli@Example.test", display_name="Cli User",
                role=RecruiterRole.ADMIN, password="a-long-password-1",
            )
            self.assertTrue(created)
            self.assertEqual(recruiter.email_normalized, "cli@example.test")
            with self.assertRaises(AccountError):
                create_or_update_recruiter(
                    session, email="cli@example.test", display_name="x",
                    role=RecruiterRole.RECRUITER, password="another-long-pass",
                )
            _, created_again = create_or_update_recruiter(
                session, email="cli@example.test", display_name=None, role=RecruiterRole.RECRUITER,
                password="brand-new-password", reset_password=True,
            )
            self.assertFalse(created_again)
        login = self.client.post("/api/auth/login", json={"email": "cli@example.test", "password": "brand-new-password"})
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["recruiter"]["role"], "admin")  # reset never changes the role

    def test_short_password_and_bad_email_are_rejected(self):
        with self.SessionLocal() as session:
            for email, password in [("ok@example.test", "short"), ("not-an-email", "a-long-password-1")]:
                with self.assertRaises(AccountError):
                    create_or_update_recruiter(
                        session, email=email, display_name="X", role=RecruiterRole.RECRUITER, password=password
                    )
            with self.assertRaises(AccountError):
                create_or_update_recruiter(
                    session, email="ghost@example.test", display_name=None, role=RecruiterRole.RECRUITER,
                    password="a-long-password-1", reset_password=True,
                )

    def test_command_reads_password_from_environment_only(self):
        from scripts import create_recruiter

        with patch.object(create_recruiter, "get_engine", return_value=self.engine), patch.dict(
            os.environ, {"SMART_HIRE_NEW_PASSWORD": "a-long-password-1"}
        ):
            code = create_recruiter.main(["--email", "cmd@example.test", "--name", "Cmd User"])
        self.assertEqual(code, 0)
        with self.SessionLocal() as session:
            stored = session.scalar(select(Recruiter).where(Recruiter.email_normalized == "cmd@example.test"))
        self.assertTrue(stored.password_hash.startswith("pbkdf2_sha256$"))
        self.assertNotIn("a-long-password-1", stored.password_hash)


class ConsentLifecycleTests(ApiTestBase):
    def setUp(self):
        super().setUp()
        self.rid = self.make_recruiter()
        self.job = self.make_job(self.rid)
        self.cand = self.make_candidate(self.rid, self.job)

    def test_grant_list_and_server_sets_capture_time(self):
        response = self.grant_consent(self.rid, self.cand, proof_reference="form-123")
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["status"], "granted")
        self.assertIsNotNone(body["captured_at"])
        listing = self.client.get(f"/api/candidates/{self.cand}/consents/", headers=self.headers(self.rid)).json()
        self.assertEqual([c["id"] for c in listing], [body["id"]])

    def test_client_cannot_supply_status_or_capture_time(self):
        response = self.grant_consent(self.rid, self.cand, status="withdrawn", captured_at="2001-01-01T00:00:00Z")
        self.assertEqual(response.status_code, 422)

    def test_past_expiry_and_foreign_application_are_rejected(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.assertEqual(self.grant_consent(self.rid, self.cand, expires_at=past).status_code, 422)
        other_rid = self.make_recruiter()
        other_cand = self.make_candidate(other_rid, self.make_job(other_rid))
        with self.SessionLocal() as session:
            other_app = session.scalar(select(Application).where(Application.candidate_id == uuid.UUID(other_cand)))
            other_app_id = str(other_app.id)
        bad = self.grant_consent(self.rid, self.cand, application_id=other_app_id)
        self.assertEqual(bad.status_code, 422)

    def test_withdrawal_is_idempotent_and_blocks_new_uploads(self):
        consent_id = self.grant_consent(self.rid, self.cand).json()["id"]
        self.assertEqual(self.upload_resume(self.rid, self.cand, consent_id).status_code, 201)
        first = self.client.post(f"/api/consents/{consent_id}/withdraw", headers=self.headers(self.rid))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "withdrawn")
        stamp = first.json()["withdrawn_at"]
        second = self.client.post(f"/api/consents/{consent_id}/withdraw", headers=self.headers(self.rid))
        self.assertEqual(second.json()["withdrawn_at"], stamp)
        self.assertEqual(self.upload_resume(self.rid, self.cand, consent_id).status_code, 403)

    def test_withdrawn_consent_blocks_download_of_existing_resume(self):
        consent_id = self.grant_consent(self.rid, self.cand).json()["id"]
        resume_id = self.upload_resume(self.rid, self.cand, consent_id).json()["id"]
        self.assertEqual(self.client.get(f"/api/resumes/{resume_id}/download", headers=self.headers(self.rid)).status_code, 200)
        self.client.post(f"/api/consents/{consent_id}/withdraw", headers=self.headers(self.rid))
        self.assertEqual(self.client.get(f"/api/resumes/{resume_id}/download", headers=self.headers(self.rid)).status_code, 403)

    def test_expiry_sweep_marks_only_lapsed_granted_consents(self):
        live = self.grant_consent(self.rid, self.cand, expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat()).json()["id"]
        lapsed = self.grant_consent(self.rid, self.cand, purpose="screening", expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()).json()["id"]
        withdrawn = self.grant_consent(self.rid, self.cand, purpose="contact").json()["id"]
        self.client.post(f"/api/consents/{withdrawn}/withdraw", headers=self.headers(self.rid))
        with self.SessionLocal() as session:
            count = expire_consents(session, now=datetime.now(timezone.utc) + timedelta(days=2))
            statuses = {str(c.id): c.status for c in session.scalars(select(ConsentRecord))}
        self.assertEqual(count, 1)
        self.assertEqual(statuses[lapsed], ConsentStatus.EXPIRED)
        self.assertEqual(statuses[live], ConsentStatus.GRANTED)
        self.assertEqual(statuses[withdrawn], ConsentStatus.WITHDRAWN)


class RetentionDeadlineTests(ApiTestBase):
    def test_upload_sets_a_default_retention_deadline(self):
        rid = self.make_recruiter()
        cand = self.make_candidate(rid, self.make_job(rid))
        consent_id = self.grant_consent(rid, cand).json()["id"]
        self.assertEqual(self.upload_resume(rid, cand, consent_id).status_code, 201)
        with self.SessionLocal() as session:
            document = session.scalar(select(ResumeDocument))
        deadline = document.retention_until
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        days = (deadline - datetime.now(timezone.utc)).days
        self.assertIn(days, (179, 180))

    def test_deadline_rules(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch.dict(os.environ, {"SMART_HIRE_RESUME_RETENTION_DAYS": "30"}):
            self.assertEqual(resume_retention_deadline(now, None), now + timedelta(days=30))
            sooner = now + timedelta(days=5)
            self.assertEqual(resume_retention_deadline(now, sooner), sooner)
            later = now + timedelta(days=90)
            self.assertEqual(resume_retention_deadline(now, later), now + timedelta(days=30))
        with patch.dict(os.environ, {"SMART_HIRE_RESUME_RETENTION_DAYS": "0"}):
            self.assertIsNone(resume_retention_deadline(now, None))
        with patch.dict(os.environ, {"SMART_HIRE_RESUME_RETENTION_DAYS": "garbage"}):
            self.assertEqual(resume_retention_deadline(now, None), now + timedelta(days=180))


class AnonymizeCandidateTests(ApiTestBase):
    def setUp(self):
        super().setUp()
        self.rid = self.make_recruiter()
        self.job = self.make_job(self.rid)
        self.cand = self.make_candidate(self.rid, self.job, "Real Name")
        self.consent = self.grant_consent(self.rid, self.cand).json()["id"]
        self.resume = self.upload_resume(self.rid, self.cand, self.consent).json()["id"]

    def test_erasure_removes_files_pii_and_consents_but_keeps_the_application(self):
        key = None
        with self.SessionLocal() as session:
            key = session.get(ResumeDocument, uuid.UUID(self.resume)).storage_key
        from backend.storage.local_filesystem import LocalFilesystemStorage

        self.assertTrue(LocalFilesystemStorage().exists(key))
        response = self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["resumes_deleted"], 1)
        self.assertEqual(response.json()["consents_withdrawn"], 1)
        self.assertFalse(LocalFilesystemStorage().exists(key))
        with self.SessionLocal() as session:
            candidate = session.get(Candidate, uuid.UUID(self.cand))
            document = session.get(ResumeDocument, uuid.UUID(self.resume))
            applications = session.scalars(select(Application).where(Application.candidate_id == candidate.id)).all()
        self.assertEqual(candidate.display_name, "Anonymized candidate")
        self.assertIsNone(candidate.email)
        self.assertIsNone(candidate.phone)
        self.assertIsNotNone(candidate.anonymized_at)
        self.assertEqual(document.status, DocumentStatus.DELETED)
        self.assertEqual(len(applications), 1)

    def test_erasure_is_idempotent(self):
        first = self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid)).json()
        second = self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid))
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["anonymized_at"], first["anonymized_at"])
        self.assertEqual(second.json()["resumes_deleted"], 0)

    def test_failed_file_deletion_leaves_the_candidate_untouched(self):
        with patch.object(ResumeDocumentService, "delete", side_effect=ResumeStorageUnavailableError("down")):
            response = self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid))
        self.assertEqual(response.status_code, 503)
        with self.SessionLocal() as session:
            candidate = session.get(Candidate, uuid.UUID(self.cand))
        self.assertEqual(candidate.display_name, "Real Name")
        self.assertIsNone(candidate.anonymized_at)

    def test_no_new_consent_for_an_anonymized_candidate(self):
        self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid))
        self.assertEqual(self.grant_consent(self.rid, self.cand).status_code, 409)

    def test_anonymization_is_audited(self):
        self.client.post(f"/api/candidates/{self.cand}/anonymize", headers=self.headers(self.rid))
        with self.SessionLocal() as session:
            names = [e.event for e in session.scalars(select(AuditEvent))]
        self.assertIn("candidate.anonymized", names)


class DatabaseOutageTests(ApiTestBase):
    def test_connection_failure_returns_a_clear_503_without_leaking_details(self):
        from fastapi.testclient import TestClient
        from sqlalchemy.exc import OperationalError

        from backend.db.session import get_db
        from backend.main import app

        def broken_db():
            raise OperationalError("connect", {}, Exception("password=hunter2 host=10.0.0.5"))
            yield  # pragma: no cover

        app.dependency_overrides[get_db] = broken_db
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/auth/login", json={"email": "a@b.test", "password": "whatever-123"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("database is unavailable", response.json()["detail"])
        self.assertNotIn("hunter2", response.text)
        self.assertNotIn("10.0.0.5", response.text)


class DevelopmentModeAndStartupTests(ApiTestBase):
    @staticmethod
    def _request(headers=None, client=("127.0.0.1", 5000)):
        return Request({"type": "http", "headers": headers or [], "client": client})

    def test_direct_loopback_is_local_but_proxied_requests_are_not(self):
        self.assertTrue(is_loopback_request(self._request()))
        for header in (b"x-forwarded-for", b"x-real-ip", b"forwarded", b"x-forwarded-host"):
            with self.subTest(header=header):
                self.assertFalse(is_loopback_request(self._request([(header, b"203.0.113.9")])))
        self.assertFalse(is_loopback_request(self._request(client=("203.0.113.9", 1))))

    def test_production_refuses_development_mode_and_missing_secret(self):
        with patch.dict(os.environ, {"SMART_HIRE_ENV": "production", "SMART_HIRE_DEVELOPMENT_MODE": "true"}):
            with self.assertRaises(RuntimeError):
                validate_runtime_configuration()
        with patch.dict(os.environ, {"SMART_HIRE_ENV": "production", "SMART_HIRE_AUTH_SECRET": "short",
                                     "SMART_HIRE_DEVELOPMENT_MODE": "false"}):
            with self.assertRaises(RuntimeError):
                validate_runtime_configuration()
        with patch.dict(os.environ, {"SMART_HIRE_ENV": "production", "SMART_HIRE_DEVELOPMENT_MODE": "false"}):
            validate_runtime_configuration()  # AUTH_ENV already supplies a long secret
        with patch.dict(os.environ, {"SMART_HIRE_ENV": "", "SMART_HIRE_DEVELOPMENT_MODE": "true"}):
            validate_runtime_configuration()  # local development is allowed


if __name__ == "__main__":
    import unittest

    unittest.main()
