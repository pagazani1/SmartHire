import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from backend.db.base import Base
from backend.db.models import (
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    Recruiter,
    RecruiterRole,
)
from backend.db.session import get_db
from backend.main import app
from scripts.seed_dev_recruiter import seed_development_recruiter
from tests.auth_helpers import AUTH_ENV, bearer_headers


class TestJobAPI(unittest.TestCase):
    def setUp(self):
        self.database_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.database_dir.name) / "job-api.sqlite"
        self.engine = create_engine(f"sqlite:///{database_path.as_posix()}")

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        with self.SessionLocal() as session:
            self.recruiter, _ = seed_development_recruiter(session)
            self.recruiter_id = self.recruiter.id

        def override_get_db():
            with self.SessionLocal() as session:
                yield session

        self.original_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = override_get_db
        self.auth_env = patch.dict(os.environ, AUTH_ENV)
        self.auth_env.start()
        self.client = TestClient(app)
        self.client.headers.update(bearer_headers(self.recruiter_id))

    def tearDown(self):
        self.client.close()
        self.auth_env.stop()
        if self.original_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = self.original_override
        self.engine.dispose()
        self.database_dir.cleanup()

    def _create_job(self, **overrides):
        payload = {
            "created_by_recruiter_id": str(self.recruiter_id),
            "title": "Backend Engineer",
            "description": "Build APIs with Python and Flask.",
            "department": "Engineering",
            "location": "Remote",
            "required_skill_groups": [["Python"], ["FastAPI", "Flask"]],
            "preferred_skill_groups": [["PostgreSQL"]],
        }
        payload.update(overrides)
        return self.client.post("/api/jobs/", json=payload)

    def test_create_job_and_first_description_version(self):
        response = self._create_job()

        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["title"], "Backend Engineer")
        self.assertEqual(body["description"], "Build APIs with Python and Flask.")
        self.assertEqual(body["description_version"], 1)
        self.assertEqual(body["required_skill_groups"], [["Python"], ["FastAPI", "Flask"]])
        self.assertEqual(body["preferred_skill_groups"], [["PostgreSQL"]])
        with self.SessionLocal() as session:
            job = session.get(Job, UUID(body["id"]))
            self.assertEqual(job.created_by_recruiter_id, self.recruiter_id)
            self.assertEqual(len(job.description_versions), 1)
            self.assertEqual(len(job.description_versions[0].requirement_groups), 3)

    def test_create_job_validates_required_and_supported_fields(self):
        missing_title = self._create_job(title=None)
        blank_title = self._create_job(title="   ")
        missing_description = self.client.post(
            "/api/jobs/",
            json={"created_by_recruiter_id": str(self.recruiter_id), "title": "Role"},
        )
        unsupported_skill = self._create_job(required_skill_groups=[["NotARealSkill"]])
        invalid_status = self._create_job(status="not_a_status")

        self.assertEqual(missing_title.status_code, 422)
        self.assertEqual(blank_title.status_code, 422)
        self.assertEqual(missing_description.status_code, 422)
        self.assertEqual(unsupported_skill.status_code, 422)
        self.assertEqual(invalid_status.status_code, 422)

    def test_create_job_cannot_name_another_recruiter(self):
        response = self._create_job(created_by_recruiter_id=str(uuid4()))

        self.assertEqual(response.status_code, 403)
        with self.SessionLocal() as session:
            self.assertEqual(session.scalar(select(Job)), None)

    def test_admin_naming_an_unknown_recruiter_gets_not_found(self):
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            recruiter.role = RecruiterRole.ADMIN
            session.commit()
        response = self._create_job(created_by_recruiter_id=str(uuid4()))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Recruiter not found.")

    def test_created_by_defaults_to_the_authenticated_recruiter(self):
        payload = {
            "title": "Data Engineer",
            "description": "Build pipelines with Python.",
            "required_skill_groups": [["Python"]],
        }
        response = self.client.post("/api/jobs/", json=payload)

        self.assertEqual(response.status_code, 201, response.text)
        with self.SessionLocal() as session:
            job = session.scalar(select(Job))
            self.assertEqual(job.created_by_recruiter_id, self.recruiter_id)

    def test_list_jobs_and_pagination(self):
        first = self._create_job(title="First Role")
        second = self._create_job(title="Second Role")
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)

        page = self.client.get("/api/jobs/?limit=1&offset=0")
        next_page = self.client.get("/api/jobs/?limit=1&offset=1")
        invalid_page = self.client.get("/api/jobs/?limit=101")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(next_page.status_code, 200)
        self.assertEqual(len(page.json()), 1)
        self.assertEqual(len(next_page.json()), 1)
        self.assertNotEqual(page.json()[0]["id"], next_page.json()[0]["id"])
        self.assertEqual(invalid_page.status_code, 422)

    def test_get_existing_and_missing_job(self):
        created = self._create_job()
        job_id = UUID(created.json()["id"])

        found = self.client.get(f"/api/jobs/{job_id}")
        missing = self.client.get(f"/api/jobs/{uuid4()}")
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["id"], str(job_id))
        self.assertEqual(found.json()["description_version"], 1)
        self.assertEqual(missing.status_code, 404)

    def test_update_scalar_fields_preserves_description_version(self):
        created = self._create_job()
        job_id = UUID(created.json()["id"])

        updated = self.client.put(
            f"/api/jobs/{job_id}",
            json={"title": "Senior Backend Engineer", "location": None, "status": "open"},
        )

        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["title"], "Senior Backend Engineer")
        self.assertIsNone(updated.json()["location"])
        self.assertEqual(updated.json()["status"], "open")
        self.assertEqual(updated.json()["description_version"], 1)
        with self.SessionLocal() as session:
            versions = session.scalars(
                select(JobDescriptionVersion).where(
                    JobDescriptionVersion.job_id == job_id
                )
            ).all()
            self.assertEqual(len(versions), 1)

    def test_description_update_appends_version_and_preserves_history(self):
        created = self._create_job()
        job_id = UUID(created.json()["id"])
        original_description = created.json()["description"]

        updated = self.client.put(
            f"/api/jobs/{job_id}",
            json={"description": "A revised role description."},
        )

        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["description"], "A revised role description.")
        self.assertEqual(updated.json()["description_version"], 2)
        with self.SessionLocal() as session:
            versions = session.scalars(
                select(JobDescriptionVersion)
                .where(JobDescriptionVersion.job_id == job_id)
                .order_by(JobDescriptionVersion.version)
            ).all()
            self.assertEqual([version.version for version in versions], [1, 2])
            self.assertEqual(versions[0].description, original_description)

    def test_requirement_update_appends_version_and_preserves_previous_groups(self):
        created = self._create_job()
        job_id = UUID(created.json()["id"])

        updated = self.client.put(
            f"/api/jobs/{job_id}",
            json={"required_skill_groups": [["Python", "Django"]]},
        )

        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["description_version"], 2)
        self.assertEqual(updated.json()["required_skill_groups"], [["Django", "Python"]])
        self.assertEqual(
            updated.json()["preferred_skill_groups"],
            [["PostgreSQL"]],
        )
        with self.SessionLocal() as session:
            versions = session.scalars(
                select(JobDescriptionVersion)
                .where(JobDescriptionVersion.job_id == job_id)
                .order_by(JobDescriptionVersion.version)
            ).all()
            self.assertEqual(len(versions), 2)
            original_alternative_group = session.scalar(
                select(JobRequirementGroup).where(
                    JobRequirementGroup.job_description_version_id == versions[0].id,
                    JobRequirementGroup.kind == "required",
                    JobRequirementGroup.ordinal == 1,
                )
            )
            self.assertEqual(
                sorted(
                    option.skill.canonical_name
                    for option in original_alternative_group.options
                ),
                ["FastAPI", "Flask"],
            )

    def test_update_same_description_does_not_create_new_version(self):
        created = self._create_job()
        job_id = UUID(created.json()["id"])

        updated = self.client.put(
            f"/api/jobs/{job_id}",
            json={"description": created.json()["description"]},
        )

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["description_version"], 1)

    def test_invalid_empty_or_null_updates_return_validation_error(self):
        job_id = self._create_job().json()["id"]

        for payload in ({}, {"title": "  "}, {"title": None}, {"description": None}):
            with self.subTest(payload=payload):
                response = self.client.put(f"/api/jobs/{job_id}", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_update_missing_job_returns_not_found(self):
        response = self.client.put(f"/api/jobs/{uuid4()}", json={"title": "Role"})
        self.assertEqual(response.status_code, 404)

    def test_seed_development_recruiter_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = create_engine(
                f"sqlite:///{(Path(directory) / 'seed.sqlite').as_posix()}"
            )
            Base.metadata.create_all(engine)
            SeedSession = sessionmaker(bind=engine, expire_on_commit=False)
            try:
                with SeedSession() as session:
                    first, created = seed_development_recruiter(session)
                    first_id = first.id
                    second, created_again = seed_development_recruiter(session)
                    self.assertTrue(created)
                    self.assertFalse(created_again)
                    self.assertEqual(second.id, first_id)
                    self.assertEqual(
                        session.scalar(
                            select(Recruiter.id).where(Recruiter.id == first_id)
                        ),
                        first_id,
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
