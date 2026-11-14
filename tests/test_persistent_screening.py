"""Comprehensive test suite for persistent explainable resume screening."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from backend.db.base import Base
from backend.db.models import (
    Application,
    Candidate,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    FitLabel,
    Job,
    JobDescriptionVersion,
    JobRequirementGroup,
    JobRequirementOption,
    Recruiter,
    RequirementKind,
    ResumeDocument,
    ScreeningEvidence,
    ScreeningRequirementResult,
    ScreeningResult,
    Skill,
)
from backend.db.session import get_db
from backend.main import app
from backend.services.resume_documents import ResumeDocumentService
from backend.storage.local_filesystem import LocalFilesystemStorage


class TestPersistentScreening(unittest.TestCase):
    def setUp(self):
        self.database_dir = tempfile.TemporaryDirectory()
        self.storage_dir = tempfile.TemporaryDirectory()

        database_path = Path(self.database_dir.name) / "screening-test.sqlite"
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

        self.previous_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_db] = override_get_db

        self.previous_environment = {
            name: os.environ.get(name)
            for name in (
                "SMART_HIRE_RESUME_STORAGE_ROOT",
                "SMART_HIRE_DEVELOPMENT_MODE",
                "SMART_HIRE_DEVELOPMENT_RECRUITER_ID",
            )
        }
        os.environ["SMART_HIRE_RESUME_STORAGE_ROOT"] = self.storage_dir.name
        os.environ["SMART_HIRE_DEVELOPMENT_MODE"] = "true"

        # Initialize recruiter, candidate, job, JD version, consents, and default resume.
        with self.SessionLocal() as session:
            self.recruiter = Recruiter(
                email_normalized="screening-recruiter@example.test",
                display_name="Screening Recruiter",
            )
            self.candidate = Candidate(
                email="candidate@example.test",
                display_name="Candidate One",
            )
            session.add_all([self.recruiter, self.candidate])
            session.flush()

            self.recruiter_id = self.recruiter.id
            self.candidate_id = self.candidate.id

            self.job = Job(
                title="Python Backend Engineer",
                created_by=self.recruiter,
            )
            session.add(self.job)
            session.flush()
            self.job_id = self.job.id

            self.application = Application(
                candidate=self.candidate,
                job=self.job,
            )
            session.add(self.application)
            session.flush()
            self.application_id = self.application.id

            self.jd_version = JobDescriptionVersion(
                job=self.job,
                version=1,
                description="Requirements:\n- Python\n- SQL\n\nPreferred:\n- Docker\n",
                created_by=self.recruiter,
            )
            session.add(self.jd_version)
            session.flush()
            self.jd_version_id = self.jd_version.id

            # Active Document Storage consent linked to candidate
            self.storage_consent = ConsentRecord(
                candidate=self.candidate,
                purpose=ConsentPurpose.DOCUMENT_STORAGE,
                status=ConsentStatus.GRANTED,
                notice_version="v1.0",
                capture_method="web_form",
            )
            # Active Screening consent linked to candidate & application
            self.screening_consent = ConsentRecord(
                candidate=self.candidate,
                application=self.application,
                purpose=ConsentPurpose.SCREENING,
                status=ConsentStatus.GRANTED,
                notice_version="v1.0",
                capture_method="web_form",
            )
            session.add_all([self.storage_consent, self.screening_consent])
            session.flush()
            self.storage_consent_id = self.storage_consent.id
            self.screening_consent_id = self.screening_consent.id

            session.commit()

        os.environ["SMART_HIRE_DEVELOPMENT_RECRUITER_ID"] = str(self.recruiter_id)
        self.client = TestClient(app, client=("127.0.0.1", 12345))

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self.previous_overrides)
        for name, value in self.previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.engine.dispose()
        self.database_dir.cleanup()
        self.storage_dir.cleanup()

    @staticmethod
    def _create_pdf_bytes(text: str) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), text)
        data = doc.tobytes()
        doc.close()
        return data

    def _upload_resume(
        self,
        text: str = "Experienced Python and SQL developer with Docker skills.",
        candidate_id: UUID | None = None,
        consent_id: UUID | None = None,
    ) -> UUID:
        pdf_bytes = self._create_pdf_bytes(text)
        response = self.client.post(
            f"/api/candidates/{candidate_id or self.candidate_id}/resumes/",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
            data={"consent_record_id": str(consent_id or self.storage_consent_id)},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return UUID(response.json()["id"])

    def _add_structured_requirement_groups(
        self,
        groups: list[tuple[RequirementKind, list[str]]],
        jd_version_id: UUID | None = None,
    ) -> None:
        with self.SessionLocal() as session:
            version = session.get(
                JobDescriptionVersion, jd_version_id or self.jd_version_id
            )
            for ordinal, (kind, skill_names) in enumerate(groups):
                resolved_skills = []
                for name in skill_names:
                    skill = session.scalar(
                        select(Skill).where(Skill.canonical_name == name)
                    )
                    if skill is None:
                        skill = Skill(canonical_name=name, is_active=True)
                        session.add(skill)
                        session.flush()
                    resolved_skills.append(skill)

                group = JobRequirementGroup(
                    job_description_version=version,
                    kind=kind,
                    ordinal=ordinal,
                    display_text=" or ".join(skill_names),
                )
                for skill in resolved_skills:
                    group.options.append(JobRequirementOption(skill=skill))
                session.add(group)
            session.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_missing_or_empty_idempotency_key_header_is_rejected(self):
        resume_id = self._upload_resume()

        # Missing header -> 422
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
        )
        self.assertEqual(res.status_code, 422)

        # Empty header -> 422
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "   "},
        )
        self.assertEqual(res.status_code, 422)

    def test_screening_fails_when_application_does_not_exist(self):
        resume_id = self._upload_resume()
        res = self.client.post(
            f"/api/applications/{uuid4()}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-app-not-found"},
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("Application not found", res.json()["detail"])

    def test_screening_fails_when_resume_document_does_not_exist(self):
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(uuid4()),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-resume-not-found"},
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("Resume document not found", res.json()["detail"])

    def test_screening_fails_when_jd_version_does_not_exist(self):
        resume_id = self._upload_resume()
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(uuid4()),
            },
            headers={"Idempotency-Key": "key-jd-not-found"},
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("Job description version not found", res.json()["detail"])

    def test_screening_rejects_resume_belonging_to_another_candidate(self):
        with self.SessionLocal() as session:
            other_candidate = Candidate(
                email="other@example.test", display_name="Other Candidate"
            )
            other_storage_consent = ConsentRecord(
                candidate=other_candidate,
                purpose=ConsentPurpose.DOCUMENT_STORAGE,
                status=ConsentStatus.GRANTED,
                notice_version="v1.0",
                capture_method="web_form",
            )
            session.add_all([other_candidate, other_storage_consent])
            session.flush()
            other_candidate_id = other_candidate.id
            other_consent_id = other_storage_consent.id
            session.commit()

        other_resume_id = self._upload_resume(
            candidate_id=other_candidate_id, consent_id=other_consent_id
        )

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(other_resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-mismatch-cand"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("does not belong to this application's candidate", res.json()["detail"])

    def test_screening_rejects_jd_version_belonging_to_another_job(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            other_job = Job(title="Other Job", created_by=recruiter)
            session.add(other_job)
            session.flush()
            other_jd_version = JobDescriptionVersion(
                job=other_job,
                version=1,
                description="Java architect role",
                created_by=recruiter,
            )
            session.add(other_jd_version)
            session.flush()
            other_jd_version_id = other_jd_version.id
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(other_jd_version_id),
            },
            headers={"Idempotency-Key": "key-mismatch-job"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("does not belong to this application", res.json()["detail"])

    def test_screening_rejects_unavailable_resume_document(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            resume = session.get(ResumeDocument, resume_id)
            resume.status = DocumentStatus.DELETED
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-unavailable-doc"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("not available for screening", res.json()["detail"])

    # ──────────────────────────────────────────────────────────────────────────
    # Consent checks
    # ──────────────────────────────────────────────────────────────────────────

    def test_screening_rejects_missing_screening_consent(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.screening_consent_id)
            session.delete(consent)
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-missing-screen-consent"},
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Active screening consent is required", res.json()["detail"])

    def test_screening_rejects_withdrawn_screening_consent(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.screening_consent_id)
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.now(timezone.utc)
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-withdrawn-consent"},
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Active screening consent is required", res.json()["detail"])

    def test_screening_rejects_expired_screening_consent(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.screening_consent_id)
            consent.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-expired-consent"},
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Active screening consent is required", res.json()["detail"])

    def test_screening_rejects_screening_consent_scoped_to_another_application(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            candidate = session.get(Candidate, self.candidate_id)
            recruiter = session.get(Recruiter, self.recruiter_id)
            other_job = Job(title="Secondary Role", created_by=recruiter)
            session.add(other_job)
            session.flush()

            other_app = Application(candidate=candidate, job=other_job)
            session.add(other_app)
            session.flush()

            consent = session.get(ConsentRecord, self.screening_consent_id)
            consent.application_id = other_app.id
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-wrong-app-consent"},
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Active screening consent is required", res.json()["detail"])

    def test_screening_rejects_withdrawn_document_storage_consent(self):
        resume_id = self._upload_resume()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.storage_consent_id)
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.now(timezone.utc)
            session.commit()

        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "key-storage-consent-withdrawn"},
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Active document-storage consent is required", res.json()["detail"])

    # ──────────────────────────────────────────────────────────────────────────
    # Idempotency tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_idempotency_same_key_and_same_inputs_returns_cached_screening(self):
        resume_id = self._upload_resume("Python and SQL developer.")
        payload = {
            "resume_document_id": str(resume_id),
            "job_description_version_id": str(self.jd_version_id),
        }
        headers = {"Idempotency-Key": "idemp-key-1"}

        res1 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers=headers,
        )
        self.assertEqual(res1.status_code, 201)
        data1 = res1.json()

        res2 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers=headers,
        )
        self.assertIn(res2.status_code, (200, 201))
        data2 = res2.json()

        self.assertEqual(data1["id"], data2["id"])
        self.assertEqual(data1["run_number"], data2["run_number"])
        self.assertEqual(data1["run_number"], 1)

    def test_idempotency_conflict_when_same_key_used_with_different_resume(self):
        resume_id1 = self._upload_resume("Python developer.")
        resume_id2 = self._upload_resume("Java developer.")

        headers = {"Idempotency-Key": "idemp-key-conflict-resume"}
        res1 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id1),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers=headers,
        )
        self.assertEqual(res1.status_code, 201)

        res2 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id2),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers=headers,
        )
        self.assertEqual(res2.status_code, 409)
        self.assertIn("different inputs", res2.json()["detail"])

    def test_idempotency_conflict_when_same_key_used_with_different_jd_version(self):
        resume_id = self._upload_resume("Python developer.")
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            job = session.get(Job, self.job_id)
            jd_v2 = JobDescriptionVersion(
                job=job,
                version=2,
                description="Python developer with FastAPI experience.",
                created_by=recruiter,
            )
            session.add(jd_v2)
            session.flush()
            jd_v2_id = jd_v2.id
            session.commit()

        headers = {"Idempotency-Key": "idemp-key-conflict-jd"}
        res1 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers=headers,
        )
        self.assertEqual(res1.status_code, 201)

        res2 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(jd_v2_id),
            },
            headers=headers,
        )
        self.assertEqual(res2.status_code, 409)

    def test_different_idempotency_keys_create_separate_runs(self):
        resume_id = self._upload_resume("Python developer.")
        payload = {
            "resume_document_id": str(resume_id),
            "job_description_version_id": str(self.jd_version_id),
        }

        res1 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers={"Idempotency-Key": "run-key-1"},
        )
        self.assertEqual(res1.status_code, 201)
        self.assertEqual(res1.json()["run_number"], 1)

        res2 = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers={"Idempotency-Key": "run-key-2"},
        )
        self.assertEqual(res2.status_code, 201)
        self.assertEqual(res2.json()["run_number"], 2)
        self.assertNotEqual(res1.json()["id"], res2.json()["id"])

    # ──────────────────────────────────────────────────────────────────────────
    # Classification & Structured Requirements
    # ──────────────────────────────────────────────────────────────────────────

    def test_structured_requirements_good_fit_at_100_percent(self):
        self._add_structured_requirement_groups(
            [
                (RequirementKind.REQUIRED, ["Python"]),
                (RequirementKind.REQUIRED, ["SQL"]),
            ]
        )
        resume_id = self._upload_resume("Experienced in Python and SQL.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "structured-good-fit"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["final_label"], "Good Fit")
        self.assertEqual(data["required_group_count"], 2)
        self.assertEqual(data["matched_required_group_count"], 2)
        self.assertEqual(data["missing_required_group_count"], 0)
        self.assertEqual(data["required_skill_coverage"], 1.0)
        self.assertEqual(data["required_skill_coverage_percentage"], 100.0)

        # Check requirement results snapshots have source == "structured"
        for req in data["requirement_results"]:
            self.assertEqual(req["requirement_snapshot"]["source"], "structured")
            self.assertTrue(req["matched"])

    def test_structured_requirements_potential_fit_at_50_percent(self):
        self._add_structured_requirement_groups(
            [
                (RequirementKind.REQUIRED, ["Python"]),
                (RequirementKind.REQUIRED, ["C++"]),
            ]
        )
        resume_id = self._upload_resume("Skilled in Python programming.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "structured-potential-fit"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["final_label"], "Potential Fit")
        self.assertEqual(data["required_group_count"], 2)
        self.assertEqual(data["matched_required_group_count"], 1)
        self.assertEqual(data["missing_required_group_count"], 1)
        self.assertEqual(data["required_skill_coverage"], 0.5)
        self.assertEqual(data["required_skill_coverage_percentage"], 50.0)

    def test_structured_requirements_no_fit_below_50_percent(self):
        self._add_structured_requirement_groups(
            [
                (RequirementKind.REQUIRED, ["Go"]),
                (RequirementKind.REQUIRED, ["Rust"]),
            ]
        )
        resume_id = self._upload_resume("Python and SQL developer.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "structured-no-fit"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["final_label"], "No Fit")
        self.assertEqual(data["required_group_count"], 2)
        self.assertEqual(data["matched_required_group_count"], 0)
        self.assertEqual(data["missing_required_group_count"], 2)
        self.assertEqual(data["required_skill_coverage"], 0.0)

    def test_alternative_skills_satisfy_group_as_one_requirement(self):
        # Python OR Java in one group
        self._add_structured_requirement_groups(
            [
                (RequirementKind.REQUIRED, ["Python", "Java"]),
            ]
        )
        # Candidate has only Java
        resume_id = self._upload_resume("Java software engineer with Spring experience.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "alternative-skill-test"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["final_label"], "Good Fit")
        self.assertEqual(data["required_group_count"], 1)
        self.assertEqual(data["matched_required_group_count"], 1)
        self.assertEqual(data["required_skill_coverage"], 1.0)
        req_res = data["requirement_results"][0]
        self.assertTrue(req_res["matched"])
        self.assertEqual(req_res["matched_skill"], "Java")

    def test_preferred_requirements_do_not_affect_required_coverage(self):
        self._add_structured_requirement_groups(
            [
                (RequirementKind.REQUIRED, ["Python"]),
                (RequirementKind.REQUIRED, ["SQL"]),
                (RequirementKind.PREFERRED, ["Docker"]),
            ]
        )
        # Candidate matches Python and Docker (1 of 2 required matched)
        resume_id = self._upload_resume("Python developer with Docker containers.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "preferred-req-test"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        # Coverage must remain 1/2 = 50%, NOT 2/3
        self.assertEqual(data["required_group_count"], 2)
        self.assertEqual(data["matched_required_group_count"], 1)
        self.assertEqual(data["required_skill_coverage"], 0.5)
        self.assertEqual(data["final_label"], "Potential Fit")
        self.assertIn("preferred skills do not affect required coverage", data["decision_explanation"])

    def test_structured_only_preferred_does_not_fall_back_to_text(self):
        # Only preferred requirement, 0 required
        self._add_structured_requirement_groups(
            [
                (RequirementKind.PREFERRED, ["Docker"]),
            ]
        )
        resume_id = self._upload_resume("Python and Docker specialist.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "structured-preferred-only"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["required_group_count"], 0)
        self.assertEqual(data["matched_required_group_count"], 0)
        self.assertIsNone(data["required_skill_coverage"])
        # Final label falls back to model label
        self.assertEqual(data["final_label"], data["model_label"])
        self.assertIn("falls back to the text-model prediction", data["decision_explanation"])

    def test_fallback_to_text_extraction_when_structured_groups_absent(self):
        # No structured groups added to self.jd_version
        resume_id = self._upload_resume("Senior Python developer proficient in SQL and Docker.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "text-fallback-test"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertGreater(len(data["requirement_results"]), 0)
        for req in data["requirement_results"]:
            self.assertIsNone(req["requirement_group_id"])
            self.assertEqual(req["requirement_snapshot"]["source"], "jd_text_fallback")

    # ──────────────────────────────────────────────────────────────────────────
    # Privacy and persistence checks
    # ──────────────────────────────────────────────────────────────────────────

    def test_screening_does_not_persist_or_expose_private_storage_details(self):
        resume_id = self._upload_resume("Python and SQL.")
        res = self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json={
                "resume_document_id": str(resume_id),
                "job_description_version_id": str(self.jd_version_id),
            },
            headers={"Idempotency-Key": "privacy-test-key"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()

        # Response must not contain internal storage attributes or raw text
        self.assertNotIn("storage_key", data)
        self.assertNotIn("storage_path", data)
        self.assertNotIn("raw_resume_text", data)
        self.assertNotIn("idempotency_key", data)

        with self.SessionLocal() as session:
            evidences = session.scalars(
                select(ScreeningEvidence).where(
                    ScreeningEvidence.screening_result_id == UUID(data["id"])
                )
            ).all()
            for evidence in evidences:
                self.assertIsNone(evidence.source_reference)

    # ──────────────────────────────────────────────────────────────────────────
    # History endpoints
    # ──────────────────────────────────────────────────────────────────────────

    def test_application_screening_history_ordered_newest_first_and_paginated(self):
        resume_id = self._upload_resume("Python and SQL.")
        payload = {
            "resume_document_id": str(resume_id),
            "job_description_version_id": str(self.jd_version_id),
        }

        # Create two screening runs
        self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers={"Idempotency-Key": "hist-key-1"},
        )
        self.client.post(
            f"/api/applications/{self.application_id}/screenings/",
            json=payload,
            headers={"Idempotency-Key": "hist-key-2"},
        )

        res = self.client.get(f"/api/applications/{self.application_id}/screenings/")
        self.assertEqual(res.status_code, 200)
        history = res.json()
        self.assertEqual(len(history), 2)
        # Newest first
        self.assertGreaterEqual(history[0]["created_at"], history[1]["created_at"])

        # Pagination test
        page1 = self.client.get(
            f"/api/applications/{self.application_id}/screenings/?limit=1&offset=0"
        ).json()
        page2 = self.client.get(
            f"/api/applications/{self.application_id}/screenings/?limit=1&offset=1"
        ).json()
        self.assertEqual(len(page1), 1)
        self.assertEqual(len(page2), 1)
        self.assertNotEqual(page1[0]["id"], page2[0]["id"])

    def test_application_screening_history_404_for_unknown_application(self):
        res = self.client.get(f"/api/applications/{uuid4()}/screenings/")
        self.assertEqual(res.status_code, 404)
        self.assertIn("Application not found", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
