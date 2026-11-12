import tempfile
import unittest
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from backend.db.base import Base
from backend.db.models import (
    Application,
    Candidate,
    DocumentStatus,
    FitLabel,
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    Recruiter,
    RequirementKind,
    ResumeDocument,
    ScreeningRequirementResult,
    ScreeningResult,
    Skill,
)
from backend.db.session import get_db
from backend.main import app


class TestCandidateAPI(unittest.TestCase):
    def setUp(self):
        self.database_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.database_dir.name) / "candidate-api.sqlite"
        self.engine = create_engine(f"sqlite:///{database_path.as_posix()}")

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        def override_get_db():
            with self.SessionLocal() as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)
        self.recruiter_id, self.job_id = self._create_job()

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.pop(get_db, None)
        self.engine.dispose()
        self.database_dir.cleanup()

    def _create_job(self):
        with self.SessionLocal() as session:
            recruiter = Recruiter(
                email_normalized="recruiter@example.test",
                display_name="Test Recruiter",
            )
            job = Job(title="Backend Engineer", created_by=recruiter)
            session.add(job)
            session.flush()
            ids = recruiter.id, job.id
            session.commit()
        return ids

    def _create_screening_history(self) -> Candidate:
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            job = session.get(Job, self.job_id)
            candidate = Candidate(display_name="Synthetic Candidate")
            application = Application(candidate=candidate, job=job)
            description = JobDescriptionVersion(
                job=job,
                version=1,
                description="Python and SQL backend role",
                created_by=recruiter,
            )
            resume = ResumeDocument(
                candidate=candidate,
                uploaded_by=recruiter,
                storage_key=f"test/{uuid4()}",
                original_filename="synthetic.pdf",
                media_type="application/pdf",
                byte_size=100,
                status=DocumentStatus.AVAILABLE,
            )
            python_skill = Skill(canonical_name="Python")
            sql_skill = Skill(canonical_name="SQL")
            python_group = JobRequirementGroup(
                job_description_version=description,
                kind=RequirementKind.REQUIRED,
                ordinal=0,
            )
            sql_group = JobRequirementGroup(
                job_description_version=description,
                kind=RequirementKind.REQUIRED,
                ordinal=1,
            )
            python_group.options.append(JobRequirementOption(skill=python_skill))
            sql_group.options.append(JobRequirementOption(skill=sql_skill))
            result = ScreeningResult(
                application=application,
                resume_document=resume,
                job_description_version=description,
                run_number=1,
                final_label=FitLabel.POTENTIAL_FIT,
                model_label=FitLabel.NO_FIT,
                model_probabilities={
                    "Good Fit": 0.1,
                    "Potential Fit": 0.2,
                    "No Fit": 0.7,
                },
                decision_policy_version="required_skill_group_coverage_v1",
                model_artifact_version="test-artifact",
                decision_explanation="One of two required groups matched.",
                required_group_count=2,
                matched_required_group_count=1,
                required_skill_coverage=0.5,
            )
            session.add(result)
            session.flush()
            result.requirement_results.extend(
                [
                    ScreeningRequirementResult(
                        requirement_group=python_group,
                        matched=True,
                        matched_skill=python_skill,
                    ),
                    ScreeningRequirementResult(
                        requirement_group=sql_group,
                        matched=False,
                    ),
                ]
            )
            session.flush()
            candidate_id = candidate.id
            session.commit()
        with self.SessionLocal() as session:
            return session.get(Candidate, candidate_id)

    def test_create_candidate_and_application(self):
        response = self.client.post(
            "/api/candidates/",
            json={
                "display_name": "  Synthetic Candidate  ",
                "email": "  TEST@example.test ",
                "phone": "  +1 555 0100  ",
                "job_id": str(self.job_id),
            },
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["candidate"]["display_name"], "Synthetic Candidate")
        self.assertEqual(body["candidate"]["email"], "TEST@example.test")
        self.assertEqual(body["application"]["job_id"], str(self.job_id))
        with self.SessionLocal() as session:
            candidate = session.get(Candidate, UUID(body["candidate"]["id"]))
            self.assertEqual(candidate.email_normalized, "test@example.test")
            self.assertEqual(len(candidate.applications), 1)

    def test_create_candidate_validates_name_and_job(self):
        blank_name = self.client.post(
            "/api/candidates/",
            json={"display_name": "  ", "job_id": str(self.job_id)},
        )
        unknown_job = self.client.post(
            "/api/candidates/",
            json={"display_name": "Synthetic Candidate", "job_id": str(uuid4())},
        )

        self.assertEqual(blank_name.status_code, 422)
        self.assertEqual(unknown_job.status_code, 404)
        with self.SessionLocal() as session:
            self.assertEqual(session.scalar(select(Candidate)), None)

    def test_candidate_list_and_detail_include_linked_application(self):
        created = self.client.post(
            "/api/candidates/",
            json={"display_name": "Synthetic Candidate", "job_id": str(self.job_id)},
        ).json()

        listed = self.client.get("/api/candidates/?limit=25&offset=0")
        detail = self.client.get(f"/api/candidates/{created['candidate']['id']}")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(
            detail.json()["applications"][0]["id"],
            created["application"]["id"],
        )
        self.assertEqual(self.client.get("/api/candidates/?limit=101").status_code, 422)

    def test_candidate_and_screening_history_return_not_found(self):
        missing_id = uuid4()
        self.assertEqual(
            self.client.get(f"/api/candidates/{missing_id}").status_code,
            404,
        )
        self.assertEqual(
            self.client.get(f"/api/candidates/{missing_id}/screenings/").status_code,
            404,
        )

    def test_screening_history_returns_decisions_and_required_skill_evidence(self):
        candidate = self._create_screening_history()

        response = self.client.get(f"/api/candidates/{candidate.id}/screenings/")

        self.assertEqual(response.status_code, 200)
        record = response.json()[0]
        self.assertEqual(record["final_label"], "Potential Fit")
        self.assertEqual(record["model_label"], "No Fit")
        self.assertEqual(
            record["model_probabilities"],
            {"Good Fit": 0.1, "Potential Fit": 0.2, "No Fit": 0.7},
        )
        self.assertEqual(record["matched_required_skills"], ["Python"])
        self.assertEqual(record["missing_required_skills"], ["SQL"])
        self.assertEqual(record["required_skill_coverage"], 0.5)

    def test_screening_history_is_empty_before_any_result_is_persisted(self):
        candidate = self.client.post(
            "/api/candidates/",
            json={"display_name": "Synthetic Candidate", "job_id": str(self.job_id)},
        ).json()["candidate"]

        response = self.client.get(
            f"/api/candidates/{candidate['id']}/screenings/"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


if __name__ == "__main__":
    unittest.main()
