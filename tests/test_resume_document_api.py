import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pymupdf
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from backend.db.base import Base
from backend.db.models import (
    Application,
    Candidate,
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DocumentStatus,
    Job,
    JobDescriptionVersion,
    Recruiter,
    RecruiterStatus,
    ResumeDocument,
    FitLabel,
    ScreeningResult,
)
from backend.db.session import get_db
from backend.main import app, require_resume_management_access
from backend.services.resume_documents import (
    ResumeDatabaseError,
    ResumeDocumentService,
    ResumeNotFoundError,
    ResumeStorageUnavailableError,
)
from backend.storage.adapter import StorageError
from backend.storage.local_filesystem import LocalFilesystemStorage


class FailingDeleteStorage:
    def __init__(self, storage: LocalFilesystemStorage) -> None:
        self.storage = storage
        self.fail_delete = True

    def store_file(self, source, key):
        self.storage.store_file(source, key)

    def open_file(self, key):
        return self.storage.open_file(key)

    def delete_file(self, key):
        if self.fail_delete:
            raise StorageError("delete failed")
        self.storage.delete_file(key)

    def exists(self, key):
        return self.storage.exists(key)

    def list_keys(self):
        return self.storage.list_keys()


class TestResumeDocumentAPI(unittest.TestCase):
    def setUp(self):
        self.database_dir = tempfile.TemporaryDirectory()
        self.storage_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.database_dir.name) / "resume-api.sqlite"
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
        with self.SessionLocal() as session:
            self.recruiter = Recruiter(
                email_normalized="resume-test@example.test",
                display_name="Synthetic Recruiter",
            )
            self.candidate = Candidate(display_name="Synthetic Candidate")
            session.add_all([self.recruiter, self.candidate])
            session.flush()
            self.recruiter_id = self.recruiter.id
            self.candidate_id = self.candidate.id
            owned_job = Job(
                title="Owned Synthetic Role", created_by_recruiter_id=self.recruiter_id
            )
            session.add(owned_job)
            session.flush()
            session.add(Application(candidate_id=self.candidate_id, job_id=owned_job.id))
            self.consent_id = self._add_consent(
                session, self.candidate_id, ConsentPurpose.DOCUMENT_STORAGE
            )
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
    def _add_consent(
        session,
        candidate_id,
        purpose,
        status=ConsentStatus.GRANTED,
        expires_at=None,
    ):
        consent = ConsentRecord(
            candidate_id=candidate_id,
            purpose=purpose,
            status=status,
            withdrawn_at=(
                datetime.now(timezone.utc)
                if status == ConsentStatus.WITHDRAWN
                else None
            ),
            expires_at=expires_at,
            notice_version="test-notice",
            capture_method="test",
        )
        session.add(consent)
        session.flush()
        return consent.id

    @staticmethod
    def _pdf_bytes(encrypted=False):
        document = pymupdf.open()
        document.new_page()
        options = {}
        if encrypted:
            options = {
                "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                "owner_pw": "synthetic-owner",
                "user_pw": "synthetic-user",
            }
        contents = document.tobytes(**options)
        document.close()
        return contents

    def _upload(self, contents=None, filename="resume.pdf", media_type="application/pdf",
                candidate_id=None, consent_id=None):
        return self.client.post(
            f"/api/candidates/{candidate_id or self.candidate_id}/resumes/",
            files={
                "file": (
                    filename,
                    contents if contents is not None else self._pdf_bytes(),
                    media_type,
                )
            },
            data={"consent_record_id": str(consent_id or self.consent_id)},
        )

    def _upload_record(self):
        response = self._upload()
        self.assertEqual(response.status_code, 201, response.text)
        return UUID(response.json()["id"])

    def test_upload_stores_valid_pdf_metadata_and_random_key(self):
        pdf_bytes = self._pdf_bytes()
        response = self._upload(contents=pdf_bytes)

        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["original_filename"], "resume.pdf")
        self.assertEqual(body["media_type"], "application/pdf")
        self.assertEqual(body["status"], "available")
        self.assertNotIn("storage_key", body)
        self.assertNotIn("sha256", body)
        with self.SessionLocal() as session:
            document = session.get(ResumeDocument, UUID(body["id"]))
            self.assertEqual(document.byte_size, len(pdf_bytes))
            self.assertEqual(len(document.sha256), 64)
            self.assertEqual(UUID(document.storage_key).version, 4)
            self.assertTrue(LocalFilesystemStorage().exists(document.storage_key))

    def test_upload_rejects_files_over_10_mib(self):
        response = self._upload(contents=b"%PDF-" + b"x" * (10 * 1024 * 1024))
        self.assertEqual(response.status_code, 413)

    def test_upload_rejects_invalid_extension_and_mime_hints(self):
        invalid_extension = self._upload(filename="resume.txt")
        invalid_mime = self._upload(media_type="image/png")
        self.assertEqual(invalid_extension.status_code, 415)
        self.assertEqual(invalid_mime.status_code, 415)

    def test_upload_rejects_invalid_signature_malformed_and_encrypted_pdfs(self):
        empty = self._upload(contents=b"")
        invalid_signature = self._upload(contents=b"not a pdf")
        malformed = self._upload(contents=b"%PDF-1.7\nnot a parseable document")
        encrypted = self._upload(contents=self._pdf_bytes(encrypted=True))

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(invalid_signature.status_code, 422)
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(encrypted.status_code, 422)

    def test_upload_rejects_missing_candidate_and_consent(self):
        unknown_candidate = self._upload(candidate_id=uuid4())
        missing_consent = self.client.post(
            f"/api/candidates/{self.candidate_id}/resumes/",
            files={"file": ("resume.pdf", self._pdf_bytes(), "application/pdf")},
            data={},
        )

        self.assertEqual(unknown_candidate.status_code, 404)
        self.assertEqual(missing_consent.status_code, 422)

    def test_upload_requires_candidate_bound_active_storage_consent(self):
        with self.SessionLocal() as session:
            other_candidate = Candidate(display_name="Other Synthetic Candidate")
            session.add(other_candidate)
            session.flush()
            other_consent_id = self._add_consent(
                session, other_candidate.id, ConsentPurpose.DOCUMENT_STORAGE
            )
            wrong_purpose_id = self._add_consent(
                session, self.candidate_id, ConsentPurpose.SCREENING
            )
            withdrawn_id = self._add_consent(
                session,
                self.candidate_id,
                ConsentPurpose.DOCUMENT_STORAGE,
                status=ConsentStatus.WITHDRAWN,
            )
            expired_id = self._add_consent(
                session,
                self.candidate_id,
                ConsentPurpose.DOCUMENT_STORAGE,
                expires_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
            session.commit()

        for consent_id in (
            other_consent_id,
            wrong_purpose_id,
            withdrawn_id,
            expired_id,
        ):
            with self.subTest(consent_id=consent_id):
                response = self._upload(consent_id=consent_id)
                self.assertEqual(response.status_code, 403)

    def test_duplicate_filenames_are_safe_and_storage_keys_are_unique(self):
        first = self._upload()
        second = self._upload()
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(first.json()["original_filename"], second.json()["original_filename"])
        with self.SessionLocal() as session:
            documents = session.scalars(
                select(ResumeDocument).order_by(ResumeDocument.id)
            ).all()
            self.assertEqual(len(documents), 2)
            self.assertNotEqual(documents[0].storage_key, documents[1].storage_key)
            self.assertTrue(all(UUID(item.storage_key).version == 4 for item in documents))

    def test_upload_uses_configured_active_recruiter_not_client_input(self):
        response = self.client.post(
            f"/api/candidates/{self.candidate_id}/resumes/",
            files={"file": ("resume.pdf", self._pdf_bytes(), "application/pdf")},
            data={
                "consent_record_id": str(self.consent_id),
                "recruiter_id": str(uuid4()),
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        with self.SessionLocal() as session:
            document = session.get(ResumeDocument, UUID(response.json()["id"]))
            self.assertEqual(document.uploaded_by_recruiter_id, self.recruiter_id)

    def test_upload_requires_configured_active_development_recruiter(self):
        os.environ["SMART_HIRE_DEVELOPMENT_RECRUITER_ID"] = ""
        missing_identity = self._upload()
        self.assertEqual(missing_identity.status_code, 503)

        with self.SessionLocal() as session:
            disabled_recruiter = Recruiter(
                email_normalized="disabled-recruiter@example.test",
                display_name="Disabled Synthetic Recruiter",
                status=RecruiterStatus.DISABLED,
            )
            session.add(disabled_recruiter)
            session.commit()
            disabled_recruiter_id = disabled_recruiter.id
        os.environ["SMART_HIRE_DEVELOPMENT_RECRUITER_ID"] = str(disabled_recruiter_id)
        disabled_identity = self._upload()
        self.assertEqual(disabled_identity.status_code, 503)

    def test_download_and_soft_delete(self):
        contents = self._pdf_bytes()
        response = self._upload(contents=contents)
        self.assertEqual(response.status_code, 201, response.text)
        resume_id = UUID(response.json()["id"])
        downloaded = self.client.get(f"/api/resumes/{resume_id}/download")
        self.assertEqual(downloaded.status_code, 200)
        self.assertEqual(downloaded.content, contents)
        self.assertEqual(downloaded.headers["content-type"], "application/pdf")
        self.assertIn("attachment", downloaded.headers["content-disposition"])
        self.assertNotIn(self.storage_dir.name, downloaded.headers["content-disposition"])

        deleted = self.client.delete(f"/api/resumes/{resume_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/api/resumes/{resume_id}/download").status_code, 404)
        metadata = self.client.get(f"/api/resumes/{resume_id}")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.json()["status"], "deleted")
        with self.SessionLocal() as session:
            document = session.get(ResumeDocument, resume_id)
            self.assertIsNotNone(document)
            self.assertIsNotNone(document.deleted_at)

    def _assert_download_denied_without_consent_details(self, resume_id):
        response = self.client.get(f"/api/resumes/{resume_id}/download")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"],
            "A valid, active document-storage consent is required.",
        )
        self.assertNotIn(self.storage_dir.name, response.text)

    def test_withdrawn_consent_blocks_resume_download(self):
        resume_id = self._upload_record()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.consent_id)
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.now(timezone.utc)
            session.commit()

        self._assert_download_denied_without_consent_details(resume_id)

    def test_expired_consent_blocks_resume_download(self):
        resume_id = self._upload_record()
        with self.SessionLocal() as session:
            consent = session.get(ConsentRecord, self.consent_id)
            consent.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            session.commit()

        self._assert_download_denied_without_consent_details(resume_id)

    def test_consent_for_another_candidate_blocks_resume_download(self):
        resume_id = self._upload_record()
        with self.SessionLocal() as session:
            other_candidate = Candidate(display_name="Other Synthetic Candidate")
            session.add(other_candidate)
            session.flush()
            other_consent_id = self._add_consent(
                session, other_candidate.id, ConsentPurpose.DOCUMENT_STORAGE
            )
            document = session.get(ResumeDocument, resume_id)
            document.consent_record_id = other_consent_id
            session.commit()

        self._assert_download_denied_without_consent_details(resume_id)

    def test_missing_consent_blocks_resume_download(self):
        resume_id = self._upload_record()
        with self.SessionLocal() as session:
            document = session.get(ResumeDocument, resume_id)
            document.consent_record_id = None
            session.commit()

        self._assert_download_denied_without_consent_details(resume_id)

    def test_delete_is_idempotent(self):
        resume_id = self._upload_record()
        self.assertEqual(self.client.delete(f"/api/resumes/{resume_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/resumes/{resume_id}").status_code, 204)

    def test_delete_preserves_screening_history_and_document_metadata(self):
        resume_id = self._upload_record()
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            job = Job(title="Historical Synthetic Role", created_by=recruiter)
            application = Application(candidate_id=self.candidate_id, job=job)
            version = JobDescriptionVersion(
                job=job,
                version=1,
                description="Synthetic screening job description",
                created_by=recruiter,
            )
            screening = ScreeningResult(
                application=application,
                resume_document_id=resume_id,
                job_description_version=version,
                run_number=1,
                final_label=FitLabel.POTENTIAL_FIT,
                model_label=FitLabel.NO_FIT,
                model_probabilities={
                    "Good Fit": 0.1,
                    "Potential Fit": 0.2,
                    "No Fit": 0.7,
                },
                decision_policy_version="test-policy",
                model_artifact_version="test-model",
            )
            session.add(screening)
            session.commit()
            screening_id = screening.id

        self.assertEqual(self.client.delete(f"/api/resumes/{resume_id}").status_code, 204)
        with self.SessionLocal() as session:
            self.assertIsNotNone(session.get(ScreeningResult, screening_id))
            document = session.get(ResumeDocument, resume_id)
            self.assertEqual(document.status, DocumentStatus.DELETED)
            self.assertEqual(document.original_filename, "resume.pdf")

    def test_listing_paginates_and_excludes_deleted_documents(self):
        ids = [self._upload_record() for _ in range(3)]
        self.client.delete(f"/api/resumes/{ids[0]}")

        page = self.client.get(
            f"/api/candidates/{self.candidate_id}/resumes/?limit=1&offset=0"
        )
        next_page = self.client.get(
            f"/api/candidates/{self.candidate_id}/resumes/?limit=1&offset=1"
        )
        self.assertEqual(page.status_code, 200)
        self.assertEqual(next_page.status_code, 200)
        self.assertEqual(len(page.json()), 1)
        self.assertEqual(len(next_page.json()), 1)
        self.assertNotEqual(page.json()[0]["id"], next_page.json()[0]["id"])
        self.assertTrue(
            all(item["status"] != "deleted" for item in page.json() + next_page.json())
        )
        self.assertEqual(
            self.client.get(
                f"/api/candidates/{uuid4()}/resumes/"
            ).status_code,
            404,
        )

    def test_failed_physical_delete_is_retryable(self):
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            actual_storage = LocalFilesystemStorage()
            failing_storage = FailingDeleteStorage(actual_storage)
            service = ResumeDocumentService(failing_storage)
            document = service.upload(
                session,
                self.candidate_id,
                self.consent_id,
                recruiter.id,
                io.BytesIO(self._pdf_bytes()),
                "retry.pdf",
                "application/pdf",
            )
            document_id = document.id
            storage_key = document.storage_key

            with self.assertRaises(ResumeStorageUnavailableError):
                service.delete(session, document_id)
            session.refresh(document)
            self.assertEqual(document.status, DocumentStatus.QUARANTINED)
            self.assertIsNone(document.deleted_at)
            self.assertTrue(actual_storage.exists(storage_key))

            failing_storage.fail_delete = False
            service.delete(session, document_id)
            session.refresh(document)
            self.assertEqual(document.status, DocumentStatus.DELETED)
            self.assertIsNotNone(document.deleted_at)
            self.assertFalse(actual_storage.exists(storage_key))

    def test_screening_materialization_checks_association_and_cleans_temp_file(self):
        storage = LocalFilesystemStorage()
        service = ResumeDocumentService(storage)
        with self.SessionLocal() as session:
            recruiter = session.get(Recruiter, self.recruiter_id)
            job = Job(title="Synthetic Role", created_by_recruiter_id=recruiter.id)
            application = Application(candidate_id=self.candidate_id, job=job)
            session.add(application)
            session.flush()
            application_id = application.id
            document = service.upload(
                session,
                self.candidate_id,
                self.consent_id,
                recruiter.id,
                io.BytesIO(self._pdf_bytes()),
                "screening.pdf",
                "application/pdf",
            )
            document_id = document.id

            with service.materialize_for_screening(
                session, document_id, application_id, recruiter.id
            ) as pdf_path:
                self.assertTrue(pdf_path.is_file())
                self.assertEqual(pdf_path.stat().st_size, document.byte_size)
            self.assertFalse(pdf_path.exists())

            other_candidate = Candidate(display_name="Different Synthetic Candidate")
            other_job = Job(title="Another Synthetic Role", created_by_recruiter_id=recruiter.id)
            other_application = Application(candidate=other_candidate, job=other_job)
            session.add(other_application)
            session.flush()
            with self.assertRaises(ResumeNotFoundError):
                with service.materialize_for_screening(
                    session, document_id, other_application.id, recruiter.id
                ):
                    self.fail("A different candidate's application must be rejected.")

    def test_database_failure_after_storage_compensates_file(self):
        storage = LocalFilesystemStorage()
        service = ResumeDocumentService(storage)
        with self.SessionLocal() as session:
            with patch.object(session, "commit", side_effect=SQLAlchemyError("synthetic")):
                with self.assertRaises(ResumeDatabaseError):
                    service.upload(
                        session,
                        self.candidate_id,
                        self.consent_id,
                        self.recruiter_id,
                        io.BytesIO(self._pdf_bytes()),
                        "rollback.pdf",
                        "application/pdf",
                    )
        self.assertEqual(storage.list_keys(), [])

    def test_storage_rejects_path_traversal_and_symlink_escape(self):
        storage = LocalFilesystemStorage()
        with self.assertRaises(StorageError):
            storage.exists("../outside")

        outside_file = Path(self.database_dir.name) / "outside.pdf"
        outside_file.write_bytes(self._pdf_bytes())
        symlink_path = Path(self.storage_dir.name) / str(uuid4())
        try:
            symlink_path.symlink_to(outside_file)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is not supported in this environment.")
        with self.assertRaises(StorageError):
            storage.open_file(symlink_path.name)
        self.assertTrue(outside_file.exists())

    def test_storage_rejects_repository_roots_and_reconciliation_finds_orphans(self):
        repository_root = Path(__file__).resolve().parents[1]
        with self.assertRaises(StorageError):
            LocalFilesystemStorage(repository_root / "private-resumes")

        storage = LocalFilesystemStorage()
        orphan_key = str(uuid4())
        storage.store_file(io.BytesIO(b"orphan test"), orphan_key)
        with self.SessionLocal() as session:
            service = ResumeDocumentService(storage)
            self.assertEqual(service.orphaned_storage_keys(session), [orphan_key])

    def test_development_guard_requires_explicit_mode_and_loopback(self):
        def request_from(host):
            return Request(
                {
                    "type": "http",
                    "headers": [],
                    "client": (host, 12345),
                    "method": "GET",
                    "path": "/",
                    "query_string": b"",
                    "scheme": "http",
                    "server": ("localhost", 8000),
                }
            )

        with patch.dict(os.environ, {"SMART_HIRE_DEVELOPMENT_MODE": "false"}):
            with self.assertRaises(HTTPException) as disabled:
                require_resume_management_access(request_from("127.0.0.1"))
            self.assertEqual(disabled.exception.status_code, 403)

        with patch.dict(os.environ, {"SMART_HIRE_DEVELOPMENT_MODE": "true"}):
            with self.assertRaises(HTTPException) as remote:
                require_resume_management_access(request_from("192.0.2.10"))
            self.assertEqual(remote.exception.status_code, 403)
            self.assertIsNone(
                require_resume_management_access(request_from("127.0.0.1"))
            )


if __name__ == "__main__":
    unittest.main()
