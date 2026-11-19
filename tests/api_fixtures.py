"""Base class for API tests that run against a throw-away SQLite database and storage folder."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.db.base import Base
from backend.db.models import Recruiter, RecruiterRole, RecruiterStatus
from backend.db.session import get_db
from backend.main import app
from backend.rate_limit import account_login_limiter, client_login_limiter
from backend.security import hash_password
from tests.auth_helpers import AUTH_ENV, bearer_headers

PASSWORD = "correct-horse-battery"


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        self._db_dir = tempfile.TemporaryDirectory()
        self._storage_dir = tempfile.TemporaryDirectory()
        # Set SMART_HIRE_TEST_DATABASE_URL to a SCRATCH PostgreSQL database to run these
        # tests against PostgreSQL. All tables in it are dropped after every test.
        self._pg_url = os.environ.get("SMART_HIRE_TEST_DATABASE_URL")
        if self._pg_url:
            self.engine = create_engine(self._pg_url)
            Base.metadata.drop_all(self.engine)
        else:
            self.engine = create_engine(f"sqlite:///{Path(self._db_dir.name, 'api.sqlite').as_posix()}")

            @event.listens_for(self.engine, "connect")
            def _fk_on(connection, _record):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        def override_get_db():
            with self.SessionLocal() as session:
                yield session

        self._previous_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_db] = override_get_db
        self._env = patch.dict(
            os.environ,
            {**AUTH_ENV, "SMART_HIRE_RESUME_STORAGE_ROOT": self._storage_dir.name},
        )
        self._env.start()
        account_login_limiter.clear()
        client_login_limiter.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self._previous_overrides)
        self._env.stop()
        self.engine.dispose()
        if self._pg_url:
            cleanup_engine = create_engine(self._pg_url)
            Base.metadata.drop_all(cleanup_engine)
            cleanup_engine.dispose()
        self._db_dir.cleanup()
        self._storage_dir.cleanup()
        account_login_limiter.clear()
        client_login_limiter.clear()

    # ---- builders -------------------------------------------------------------------
    def make_recruiter(self, email=None, role=RecruiterRole.RECRUITER, password=PASSWORD,
                       status=RecruiterStatus.ACTIVE):
        with self.SessionLocal() as session:
            recruiter = Recruiter(
                email_normalized=email or f"{uuid4().hex[:8]}@example.test",
                display_name="Synthetic Recruiter",
                role=role,
                status=status,
                password_hash=hash_password(password) if password else None,
            )
            session.add(recruiter)
            session.commit()
            return recruiter.id

    @staticmethod
    def headers(recruiter_id):
        return bearer_headers(recruiter_id)

    def make_job(self, owner_id, title="Python Backend Engineer"):
        response = self.client.post(
            "/api/jobs/",
            json={
                "title": title,
                "description": "Build REST APIs. Required: Python and PostgreSQL.",
                "required_skill_groups": [["Python"], ["PostgreSQL"]],
            },
            headers=self.headers(owner_id),
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def make_candidate(self, owner_id, job_id, name="Synthetic Candidate"):
        response = self.client.post(
            "/api/candidates/",
            json={"job_id": job_id, "display_name": name, "email": f"{uuid4().hex[:6]}@candidate.test"},
            headers=self.headers(owner_id),
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["candidate"]["id"]

    def grant_consent(self, owner_id, candidate_id, purpose="document_storage", **extra):
        payload = {"purpose": purpose, "notice_version": "v1", "capture_method": "recorded_by_recruiter"}
        payload.update(extra)
        return self.client.post(
            f"/api/candidates/{candidate_id}/consents/", json=payload, headers=self.headers(owner_id)
        )

    @staticmethod
    def pdf_bytes():
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic resume: Python and PostgreSQL experience.")
        contents = document.tobytes()
        document.close()
        return contents

    def upload_resume(self, owner_id, candidate_id, consent_id):
        return self.client.post(
            f"/api/candidates/{candidate_id}/resumes/",
            files={"file": ("resume.pdf", self.pdf_bytes(), "application/pdf")},
            data={"consent_record_id": str(consent_id)},
            headers=self.headers(owner_id),
        )
