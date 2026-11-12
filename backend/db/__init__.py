"""Database infrastructure for SmartHire."""

from backend.db.base import Base
from backend.db.session import SessionLocal, get_db

__all__ = ["Base", "SessionLocal", "get_db"]
