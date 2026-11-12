"""Lazy SQLAlchemy engine, session factory, and FastAPI session dependency."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import ArgumentError
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

_engine: Engine | None = None
_engine_lock = Lock()


def get_database_url() -> str:
    """Return a normalized DATABASE_URL without logging its credentials."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url or not database_url.strip():
        raise RuntimeError(
            "DATABASE_URL is not configured. Copy .env.example to .env "
            "and set the local PostgreSQL connection URL."
        )

    try:
        url = make_url(database_url.strip())
    except ArgumentError as exc:
        raise RuntimeError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc

    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


def get_engine() -> Engine:
    """Create the process engine on first use; importing the app never connects."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                url = make_url(get_database_url())
                connect_args = (
                    {"check_same_thread": False}
                    if url.get_backend_name() == "sqlite"
                    else {}
                )
                _engine = create_engine(
                    url,
                    pool_pre_ping=True,
                    connect_args=connect_args,
                )
    return _engine


def dispose_engine() -> None:
    """Dispose the process engine during application shutdown."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped session and always roll back/close safely."""
    session = SessionLocal(bind=get_engine())
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
