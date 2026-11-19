"""Authentication, RBAC ownership enforcement, and privacy-safe security audit logging."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.db.models import (
    Application,
    AuditEvent,
    Job,
    Recruiter,
    RecruiterRole,
    RecruiterStatus,
    ResumeDocument,
)
from backend.db.session import get_db

audit_logger = logging.getLogger("smarthire.security.audit")
_bearer_scheme = HTTPBearer(auto_error=False)

DEFAULT_TOKEN_TTL_MINUTES = 480
PBKDF2_ITERATIONS = 310_000


def log_security_event(
    event: str,
    *,
    outcome: str,
    recruiter_id: UUID | None = None,
    resource_type: str | None = None,
    resource_id: UUID | None = None,
    reason: str | None = None,
    db: Session | None = None,
) -> None:
    """Emit a privacy-safe audit entry to the log and, when ``db`` is given, to the database.

    Never records PII, file paths or resume text. The database copy is written with its
    own short session so it survives a rollback of the request that triggered it, and a
    failure to write it never breaks the request.
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "outcome": outcome,
        "recruiter_id": str(recruiter_id) if recruiter_id else None,
        "resource_type": resource_type,
        "resource_id": str(resource_id) if resource_id else None,
    }
    if reason:
        payload["reason"] = reason
    audit_logger.info(json.dumps(payload, sort_keys=True))

    if db is None:
        return
    try:
        with Session(bind=db.get_bind()) as audit_session:
            audit_session.add(
                AuditEvent(
                    event=event[:100],
                    outcome=outcome[:20],
                    recruiter_id=recruiter_id,
                    resource_type=resource_type[:50] if resource_type else None,
                    resource_id=resource_id,
                    reason=reason[:200] if reason else None,
                )
            )
            audit_session.commit()
    except Exception:  # noqa: BLE001 - auditing must never break the request
        audit_logger.warning("Audit event could not be stored in the database.")


def hash_password(plain_password: str) -> str:
    """Hash a recruiter password using PBKDF2-HMAC-SHA256 with a random salt."""
    if len(plain_password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}$"
        f"{base64.urlsafe_b64encode(salt).decode('ascii')}$"
        f"{base64.urlsafe_b64encode(digest).decode('ascii')}"
    )


def verify_password(plain_password: str, stored_hash: str | None) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    if not stored_hash or not plain_password:
        return False
    try:
        algorithm, iter_str, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected_digest = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
    except (ValueError, TypeError):
        return False

    candidate_digest = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate_digest, expected_digest)


def _auth_secret() -> bytes:
    secret = os.getenv("SMART_HIRE_AUTH_SECRET", "").strip()
    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication signing secret is not configured (minimum 32 characters).",
        )
    return secret.encode("utf-8")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def create_access_token(
    recruiter: Recruiter,
    *,
    expires_delta: timedelta | None = None,
) -> tuple[str, datetime]:
    """Create a signed HMAC-SHA256 bearer token for an active recruiter."""
    now = datetime.now(timezone.utc)
    expires_at = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=DEFAULT_TOKEN_TTL_MINUTES)
    )
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "sub": str(recruiter.id),
        "role": recruiter.role.value,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    signing_input = (
        f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))}."
        f"{_b64url_encode(json.dumps(claims, separators=(',', ':')).encode('utf-8'))}"
    )
    signature = hmac.new(
        _auth_secret(),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64url_encode(signature)}", expires_at


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate signature and expiration of an HMAC-SHA256 bearer token."""
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token format.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    signing_input = f"{parts[0]}.{parts[1]}"
    try:
        provided_sig = _b64url_decode(parts[2])
        claims = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    expected_sig = hmac.new(
        _auth_secret(),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token signature.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    exp = claims.get("exp")
    if not isinstance(exp, int) or datetime.now(timezone.utc).timestamp() >= exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims


def is_development_mode_enabled() -> bool:
    return os.getenv("SMART_HIRE_DEVELOPMENT_MODE", "").strip().casefold() in {
        "1",
        "true",
        "yes",
    }


_FORWARDING_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")


def is_loopback_request(request: Request) -> bool:
    """True only for a direct local connection.

    A reverse proxy makes every request look like it comes from 127.0.0.1, so any
    request carrying proxy forwarding headers is never treated as loopback.
    """
    if any(request.headers.get(name) for name in _FORWARDING_HEADERS):
        return False
    client_host = request.client.host if request.client is not None else ""
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return False


def validate_runtime_configuration() -> None:
    """Fail fast at startup when production settings are unsafe."""
    environment = os.getenv("SMART_HIRE_ENV", "").strip().casefold()
    if environment not in {"production", "prod"}:
        return
    if is_development_mode_enabled():
        raise RuntimeError(
            "SMART_HIRE_DEVELOPMENT_MODE must not be enabled when SMART_HIRE_ENV=production."
        )
    if len(os.getenv("SMART_HIRE_AUTH_SECRET", "").strip()) < 32:
        raise RuntimeError(
            "SMART_HIRE_AUTH_SECRET (minimum 32 characters) is required in production."
        )


_DUMMY_PASSWORD_HASH: str | None = None


def burn_password_verification_time(plain_password: str) -> None:
    """Spend the same time as a real check so unknown accounts are not detectable by timing."""
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(18))
    verify_password(plain_password, _DUMMY_PASSWORD_HASH)


def get_current_recruiter(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> Recruiter:
    """Resolve the authenticated active recruiter via Bearer token or explicit local dev mode."""
    if credentials is not None and credentials.credentials.strip():
        claims = decode_access_token(credentials.credentials.strip())
        try:
            recruiter_id = UUID(str(claims.get("sub", "")))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token subject is not a valid recruiter identifier.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None

        recruiter = db.get(Recruiter, recruiter_id)
        if recruiter is None or recruiter.status != RecruiterStatus.ACTIVE:
            log_security_event(
                "auth.token_rejected",
                outcome="denied",
                recruiter_id=recruiter_id,
                reason="inactive_or_missing_recruiter",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Recruiter account is inactive or unavailable.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return recruiter

    if is_development_mode_enabled():
        if not is_loopback_request(request):
            log_security_event(
                "auth.dev_fallback_rejected",
                outcome="denied",
                reason="non_loopback_client",
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Development mode authentication is restricted to loopback clients.",
            )
        recruiter_value = os.getenv("SMART_HIRE_DEVELOPMENT_RECRUITER_ID", "").strip()
        try:
            dev_recruiter_id = UUID(recruiter_value)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="A valid development recruiter identity is not configured.",
            ) from None
        recruiter = db.get(Recruiter, dev_recruiter_id)
        if recruiter is None or recruiter.status != RecruiterStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The configured development recruiter is unavailable.",
            )
        return recruiter

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials are required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def ensure_job_access(
    recruiter: Recruiter, job: Job, db: Session | None = None
) -> None:
    """Enforce that only the owning recruiter or an admin may manage a job."""
    if recruiter.role == RecruiterRole.ADMIN:
        return
    if job.created_by_recruiter_id != recruiter.id:
        log_security_event(
            "rbac.job_access_denied",
            outcome="denied",
            recruiter_id=recruiter.id,
            resource_type="job",
            resource_id=job.id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access or modify this job.",
        )


def ensure_application_access(
    recruiter: Recruiter, application: Application, db: Session | None = None
) -> None:
    """Enforce recruiter ownership over an application via its parent job."""
    if recruiter.role == RecruiterRole.ADMIN:
        return
    if application.job is None or application.job.created_by_recruiter_id != recruiter.id:
        log_security_event(
            "rbac.application_access_denied",
            outcome="denied",
            recruiter_id=recruiter.id,
            resource_type="application",
            resource_id=application.id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this application.",
        )


def ensure_resume_access(
    db: Session,
    recruiter: Recruiter,
    document: ResumeDocument,
) -> None:
    """Enforce that the recruiter uploaded the resume, owns a linked job, or is an admin."""
    if recruiter.role == RecruiterRole.ADMIN:
        return
    if document.uploaded_by_recruiter_id == recruiter.id:
        return
    has_candidate_job = (
        db.query(Application)
        .join(Job, Application.job_id == Job.id)
        .filter(
            Application.candidate_id == document.candidate_id,
            Job.created_by_recruiter_id == recruiter.id,
        )
        .first()
        is not None
    )
    if not has_candidate_job:
        log_security_event(
            "rbac.resume_access_denied",
            outcome="denied",
            recruiter_id=recruiter.id,
            resource_type="resume_document",
            resource_id=document.id,
            db=db,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resume document.",
        )


def recruiter_owns_candidate(db: Session, recruiter: Recruiter, candidate_id: UUID) -> bool:
    """A recruiter owns a candidate when the candidate applied to one of their jobs."""
    if recruiter.role == RecruiterRole.ADMIN:
        return True
    return (
        db.query(Application.id)
        .join(Job, Application.job_id == Job.id)
        .filter(
            Application.candidate_id == candidate_id,
            Job.created_by_recruiter_id == recruiter.id,
        )
        .first()
        is not None
    )


def ensure_candidate_access(db: Session, recruiter: Recruiter, candidate_id: UUID) -> None:
    """Enforce that the candidate applied to the recruiter's job (admins may access all)."""
    if recruiter_owns_candidate(db, recruiter, candidate_id):
        return
    log_security_event(
        "rbac.candidate_access_denied",
        outcome="denied",
        recruiter_id=recruiter.id,
        resource_type="candidate",
        resource_id=candidate_id,
        db=db,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have permission to access this candidate.",
    )
