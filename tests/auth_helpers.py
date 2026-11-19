"""Shared helpers so API tests can authenticate as a recruiter."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from backend.db.models import RecruiterRole
from backend.security import create_access_token

AUTH_ENV = {
    "SMART_HIRE_AUTH_SECRET": "test-secret-key-at-least-32-characters-long!",
    "SMART_HIRE_DEVELOPMENT_MODE": "false",
}


def auth_env_patcher():
    """Return a started-by-caller patcher that sets the signing secret for tests."""
    return patch.dict(os.environ, AUTH_ENV)


def bearer_headers(recruiter_id, role=RecruiterRole.RECRUITER):
    """Authorization header for a recruiter id (the route re-reads the role from the database)."""
    token, _ = create_access_token(SimpleNamespace(id=recruiter_id, role=role))
    return {"Authorization": f"Bearer {token}"}
