"""Tests for Phase 7 authentication, RBAC ownership, audit logging, and retention enforcement."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.base import Base
from backend.db.models import (
    Application,
    Candidate,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    Job,
    Recruiter,
    RecruiterRole,
    RecruiterStatus,
    ResumeDocument,
)
from backend.db.session import get_db
from backend.main import app
from backend.security import create_access_token, hash_password, verify_password
from backend.services.retention import enforce_document_retention
from backend.storage.local_filesystem import LocalFilesystemStorage


class SecurityHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "security_test.sqlite"
        self.storage_dir = Path(self.temp_dir.name) / "private_resumes"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.SessionTesting = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

        def override_get_db():
            db = self.SessionTesting()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        self.env_patcher = patch.dict(
            os.environ,
            {
                "SMART_HIRE_AUTH_SECRET": "test-secret-key-at-least-32-characters-long!",
                "SMART_HIRE_DEVELOPMENT_MODE": "false",
                "SMART_HIRE_PRIVATE_STORAGE_DIR": str(self.storage_dir),
            },
        )
        self.env_patcher.start()

        with self.SessionTesting() as db:
            self.recruiter_a = Recruiter(
                email_normalized="alice@smarthire.example",
                display_name="Alice Recruiter",
                role=RecruiterRole.RECRUITER,
                password_hash=hash_password("StrongPass123!"),
                status=RecruiterStatus.ACTIVE,
            )
            self.recruiter_b = Recruiter(
                email_normalized="bob@smarthire.example",
                display_name="Bob Recruiter",
                role=RecruiterRole.RECRUITER,
                password_hash=hash_password("OtherPass123!"),
                status=RecruiterStatus.ACTIVE,
            )
            self.admin = Recruiter(
                email_normalized="admin@smarthire.example",
                display_name="Admin User",
                role=RecruiterRole.ADMIN,
                password_hash=hash_password("AdminPass123!"),
                status=RecruiterStatus.ACTIVE,
            )
            db.add_all([self.recruiter_a, self.recruiter_b, self.admin])
            db.flush()

            self.job_a = Job(
                created_by_recruiter_id=self.recruiter_a.id,
                title="Backend Engineer",
            )
            self.candidate = Candidate(
                display_name="Jane Candidate",
                email="jane@example.com",
            )
            db.add_all([self.job_a, self.candidate])
            db.flush()

            self.application_a = Application(
                candidate_id=self.candidate.id,
                job_id=self.job_a.id,
            )
            self.storage_consent = ConsentRecord(
                candidate_id=self.candidate.id,
                purpose=ConsentPurpose.DOCUMENT_STORAGE,
                status=ConsentStatus.GRANTED,
                notice_version="v1",
                capture_method="portal",
            )
            db.add_all([self.application_a, self.storage_consent])
            db.commit()

            self.recruiter_a_id = self.recruiter_a.id
            self.recruiter_b_id = self.recruiter_b.id
            self.admin_id = self.admin.id
            self.application_a_id = self.application_a.id
            self.candidate_id = self.candidate.id

    def tearDown(self) -> None:
        self.env_patcher.stop()
        app.dependency_overrides.clear()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_password_hashing_and_verification(self) -> None:
        hashed = hash_password("MySecretPassword99")
        self.assertTrue(verify_password("MySecretPassword99", hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))
        self.assertFalse(verify_password("MySecretPassword99", None))

    def test_login_issues_bearer_token_and_me_endpoint_returns_profile(self) -> None:
        login_res = self.client.post(
            "/api/auth/login",
            json={"email": "ALICE@smarthire.example", "password": "StrongPass123!"},
        )
        self.assertEqual(login_res.status_code, 200)
        payload = login_res.json()
        self.assertEqual(payload["token_type"], "bearer")
        self.assertEqual(payload["recruiter"]["id"], str(self.recruiter_a_id))

        me_res = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
        )
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email_normalized"], "alice@smarthire.example")

    def test_unauthenticated_and_tampered_tokens_are_rejected_when_dev_mode_off(self) -> None:
        res_missing = self.client.get(
            f"/api/applications/{self.application_a_id}/screenings/"
        )
        self.assertEqual(res_missing.status_code, 401)

        with self.SessionTesting() as db:
            alice = db.get(Recruiter, self.recruiter_a_id)
            token, _ = create_access_token(alice)

        tampered = token[:-4] + "XXXX"
        res_tampered = self.client.get(
            f"/api/applications/{self.application_a_id}/screenings/",
            headers={"Authorization": f"Bearer {tampered}"},
        )
        self.assertEqual(res_tampered.status_code, 401)

    def test_recruiter_ownership_blocks_cross_recruiter_access(self) -> None:
        with self.SessionTesting() as db:
            alice = db.get(Recruiter, self.recruiter_a_id)
            bob = db.get(Recruiter, self.recruiter_b_id)
            alice_token, _ = create_access_token(alice)
            bob_token, _ = create_access_token(bob)

        # Alice owns job_a -> application_a: allowed
        ok_res = self.client.get(
            f"/api/applications/{self.application_a_id}/screenings/",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        self.assertEqual(ok_res.status_code, 200)

        # Bob does not own job_a -> 403 Forbidden
        forbidden_res = self.client.get(
            f"/api/applications/{self.application_a_id}/screenings/",
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        self.assertEqual(forbidden_res.status_code, 403)

    def test_retention_sweep_purges_expired_resume_and_updates_status(self) -> None:
        storage = LocalFilesystemStorage()
        storage.store_file(BytesIO(b"%PDF-1.4\n1 0 obj<<>>endobj\n%%EOF"), "11111111-2222-3333-4444-555555555555")

        with self.SessionTesting() as db:
            doc = ResumeDocument(
                candidate_id=self.candidate_id,
                uploaded_by_recruiter_id=self.recruiter_a_id,
                consent_record_id=self.storage_consent.id,
                storage_key="11111111-2222-3333-4444-555555555555",
                original_filename="old_resume.pdf",
                media_type="application/pdf",
                byte_size=32,
                status=DocumentStatus.AVAILABLE,
                retention_until=datetime.now(timezone.utc) - timedelta(days=1),
            )
            db.add(doc)
            db.commit()
            doc_id = doc.id

            admin = db.get(Recruiter, self.admin_id)
            admin_token, _ = create_access_token(admin)

        sweep_res = self.client.post(
            "/api/admin/retention/sweep",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(sweep_res.status_code, 200)
        self.assertEqual(sweep_res.json()["expired_count"], 1)

        with self.SessionTesting() as db:
            refreshed = db.get(ResumeDocument, doc_id)
            self.assertEqual(refreshed.status, DocumentStatus.EXPIRED)
            self.assertIsNotNone(refreshed.deleted_at)


if __name__ == "__main__":
    unittest.main()