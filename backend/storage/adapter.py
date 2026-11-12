"""Provider-neutral interface for private stored objects."""

from __future__ import annotations

from typing import BinaryIO, Protocol


class StorageError(Exception):
    """Raised when a private storage operation fails safely."""


class StorageAdapter(Protocol):
    def store_file(self, source: BinaryIO, key: str) -> None:
        """Store a stream under an opaque key without exposing a public URL."""

    def open_file(self, key: str) -> BinaryIO:
        """Open an object for reading."""

    def delete_file(self, key: str) -> None:
        """Delete an object; deleting a missing object is idempotent."""

    def exists(self, key: str) -> bool:
        """Return whether a regular stored object exists."""

    def list_keys(self) -> list[str]:
        """List opaque object keys for internal orphan reconciliation."""
