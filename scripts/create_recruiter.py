"""Create (or reset the password of) a recruiter account from the command line.

There is deliberately no public registration. Examples:

    python -m scripts.create_recruiter --email me@example.com --name "Aruni Mishra" --role admin
    python -m scripts.create_recruiter --email me@example.com --reset-password

The password is read from a hidden prompt, never from a command-line argument, so it does not
end up in shell history or process lists. For automation only, set SMART_HIRE_NEW_PASSWORD.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Recruiter, RecruiterRole, RecruiterStatus
from backend.db.session import get_engine
from backend.schemas import RecruiterAccountCreate
from backend.security import hash_password


class AccountError(Exception):
    """A problem the operator can fix (bad email, short password, duplicate account...)."""


def create_or_update_recruiter(
    session: Session,
    *,
    email: str,
    display_name: str | None,
    role: RecruiterRole,
    password: str,
    reset_password: bool = False,
) -> tuple[Recruiter, bool]:
    """Return ``(recruiter, created)``. An existing account is only changed with reset_password."""
    try:
        validated = RecruiterAccountCreate(
            email=email,
            display_name=display_name or "Recruiter",
            password=password,
            role=role,
        )
    except ValidationError as exc:
        problems = "; ".join(error["msg"] for error in exc.errors())
        raise AccountError(f"Invalid account details: {problems}") from None

    existing = session.scalar(
        select(Recruiter).where(Recruiter.email_normalized == validated.email)
    )
    if existing is not None:
        if not reset_password:
            raise AccountError(
                "An account with this email already exists. Use --reset-password to change its password."
            )
        existing.password_hash = hash_password(validated.password)
        existing.status = RecruiterStatus.ACTIVE
        session.commit()
        return existing, False

    if reset_password:
        raise AccountError("No account with this email exists, so there is no password to reset.")
    recruiter = Recruiter(
        email_normalized=validated.email,
        display_name=validated.display_name,
        role=validated.role,
        status=RecruiterStatus.ACTIVE,
        password_hash=hash_password(validated.password),
    )
    session.add(recruiter)
    session.commit()
    session.refresh(recruiter)
    return recruiter, True


def _read_password() -> str:
    env_password = os.getenv("SMART_HIRE_NEW_PASSWORD")
    if env_password:
        return env_password
    first = getpass.getpass("Password (min 10 characters): ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise AccountError("The two passwords do not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", help="display name (required for a new account)")
    parser.add_argument("--role", choices=[r.value for r in RecruiterRole], default=RecruiterRole.RECRUITER.value)
    parser.add_argument("--reset-password", action="store_true", help="change the password of an existing account")
    args = parser.parse_args(argv)

    if not args.reset_password and not args.name:
        parser.error("--name is required when creating a new account")
    try:
        password = _read_password()
        with Session(get_engine()) as session:
            recruiter, created = create_or_update_recruiter(
                session,
                email=args.email,
                display_name=args.name,
                role=RecruiterRole(args.role),
                password=password,
                reset_password=args.reset_password,
            )
    except AccountError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    action = "Created" if created else "Updated password for"
    print(f"{action} {args.role} account {args.email.strip().lower()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
