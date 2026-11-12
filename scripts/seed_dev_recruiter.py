"""Create the stable local-development recruiter needed to create jobs."""

from __future__ import annotations

import argparse

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from backend.db.models import Recruiter, RecruiterStatus
from backend.db.session import get_database_url, get_engine


DEV_RECRUITER_EMAIL = "local-recruiter@smarthire.test"
DEV_RECRUITER_NAME = "Local Development Recruiter"
LOCAL_DATABASE_HOSTS = {"localhost", "127.0.0.1", "::1"}


def seed_development_recruiter(session: Session) -> tuple[Recruiter, bool]:
    """Return the existing local seed recruiter or create it once."""
    recruiter = session.scalar(
        select(Recruiter).where(
            Recruiter.email_normalized == DEV_RECRUITER_EMAIL
        )
    )
    if recruiter is not None:
        return recruiter, False

    recruiter = Recruiter(
        email_normalized=DEV_RECRUITER_EMAIL,
        display_name=DEV_RECRUITER_NAME,
        status=RecruiterStatus.ACTIVE,
    )
    session.add(recruiter)
    session.commit()
    session.refresh(recruiter)
    return recruiter, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        action="store_true",
        required=True,
        help="confirm this is an explicitly requested local development seed",
    )
    args = parser.parse_args()
    if not args.development:
        parser.error("this utility is only for local development")

    database_url = make_url(get_database_url())
    if database_url.get_backend_name() != "sqlite" and (
        database_url.host not in LOCAL_DATABASE_HOSTS
    ):
        parser.error("refusing to seed a non-local database")

    with Session(get_engine()) as session:
        recruiter, created = seed_development_recruiter(session)
    action = "Created" if created else "Already exists"
    print(f"{action} local development recruiter ID: {recruiter.id}")


if __name__ == "__main__":
    main()
