import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, configure_mappers, sessionmaker

from backend.db.base import Base
from backend.db.models import Application, Candidate, ConsentRecord, Job, Recruiter
from backend.db.session import get_db


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_test_engine():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


class TestDatabaseModels(unittest.TestCase):
    def setUp(self):
        configure_mappers()
        self.engine = create_test_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def test_models_import_and_relationships_configure(self):
        expected_tables = {
            "recruiters",
            "jobs",
            "job_description_versions",
            "candidates",
            "applications",
            "resume_documents",
            "screening_results",
            "screening_evidence",
            "consent_records",
            "verification_records",
        }
        self.assertTrue(expected_tables.issubset(Base.metadata.tables))
        self.assertIn("verification_events", Base.metadata.tables)
        self.assertNotIn("updated_at", ConsentRecord.__table__.c)

        with self.SessionLocal() as session:
            recruiter = Recruiter(
                email_normalized="recruiter@example.test",
                display_name="Test Recruiter",
            )
            job = Job(title="Backend Engineer", created_by=recruiter)
            candidate = Candidate(display_name="Test Candidate")
            application = Application(candidate=candidate, job=job)
            session.add_all([application])
            session.flush()

            self.assertIs(job.created_by, recruiter)
            self.assertIn(job, recruiter.jobs)
            self.assertIs(application.candidate, candidate)
            self.assertIs(application.job, job)

    def test_foreign_key_constraints_are_enforced(self):
        with self.SessionLocal() as session:
            session.add(
                Job(
                    title="Orphaned job",
                    created_by_recruiter_id=uuid4(),
                )
            )
            with self.assertRaises(IntegrityError):
                session.flush()

    def test_application_job_candidate_pair_is_unique(self):
        with self.SessionLocal() as session:
            recruiter = Recruiter(
                email_normalized="recruiter@example.test",
                display_name="Test Recruiter",
            )
            job = Job(title="Backend Engineer", created_by=recruiter)
            candidate = Candidate(display_name="Test Candidate")
            session.add_all([job, candidate])
            session.flush()
            session.add(Application(candidate=candidate, job=job))
            session.flush()
            session.add(Application(candidate=candidate, job=job))

            with self.assertRaises(IntegrityError):
                session.flush()

    def test_recruiter_email_is_unique(self):
        with self.SessionLocal() as session:
            session.add_all(
                [
                    Recruiter(
                        email_normalized="recruiter@example.test",
                        display_name="First",
                    ),
                    Recruiter(
                        email_normalized="recruiter@example.test",
                        display_name="Second",
                    ),
                ]
            )
            with self.assertRaises(IntegrityError):
                session.flush()


class TestDatabaseDependency(unittest.TestCase):
    def test_dependency_rolls_back_and_closes_session_on_error(self):
        engine = create_test_engine()
        try:
            generator = get_db()
            with patch("backend.db.session.get_engine", return_value=engine):
                session = next(generator)
                session.add(
                    Recruiter(
                        email_normalized="rollback@example.test",
                        display_name="Rollback Test",
                    )
                )
                with patch.object(session, "close", wraps=session.close) as close:
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        generator.throw(RuntimeError("test failure"))
                    close.assert_called_once()

            with Session(engine) as verify_session:
                count = verify_session.scalar(
                    select(func.count()).select_from(Recruiter)
                )
            self.assertEqual(count, 0)
        finally:
            engine.dispose()

    def test_dependency_yields_and_closes_session(self):
        engine = create_test_engine()
        try:
            with patch("backend.db.session.get_engine", return_value=engine):
                generator = get_db()
                session = next(generator)
                self.assertIsInstance(session, Session)
                with patch.object(session, "close", wraps=session.close) as close:
                    with self.assertRaises(StopIteration):
                        next(generator)
                    close.assert_called_once()
        finally:
            engine.dispose()


class TestAlembicConfiguration(unittest.TestCase):
    def test_alembic_upgrade_and_downgrade_on_disposable_sqlite_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "migration-test.sqlite"
            database_url = f"sqlite:///{database_path.as_posix()}"
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))

            with patch.dict(os.environ, {"DATABASE_URL": database_url}):
                command.upgrade(config, "head")
                command.check(config)

                engine = create_engine(database_url)
                try:
                    with engine.connect() as connection:
                        table_names = set(
                            engine.dialect.get_table_names(connection)
                        )
                finally:
                    engine.dispose()

                self.assertTrue(set(Base.metadata.tables).issubset(table_names))

                command.downgrade(config, "base")

                engine = create_engine(database_url)
                try:
                    with engine.connect() as connection:
                        remaining_tables = set(
                            engine.dialect.get_table_names(connection)
                        )
                finally:
                    engine.dispose()

                self.assertFalse(set(Base.metadata.tables) & remaining_tables)


if __name__ == "__main__":
    unittest.main()
