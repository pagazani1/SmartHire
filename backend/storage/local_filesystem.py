"""Private local-filesystem storage for development."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from backend.storage.adapter import StorageError


def default_storage_root() -> Path:
    configured_root = os.getenv("SMART_HIRE_RESUME_STORAGE_ROOT")
    if configured_root and configured_root.strip():
        return Path(configured_root).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "SmartHire" / "private-resumes"
    return Path.home() / ".local" / "share" / "smarthire" / "private-resumes"


class LocalFilesystemStorage:
    """Store opaque UUID-keyed files outside the repository and web roots."""

    def __init__(self, root: str | Path | None = None) -> None:
        requested_root = Path(root) if root is not None else default_storage_root()
        try:
            if requested_root.is_symlink():
                raise StorageError("Resume storage root must not be a symbolic link.")
            self.root = requested_root.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise StorageError("Unable to resolve private resume storage.") from exc
        repository_root = Path(__file__).resolve().parents[2]
        if self.root == repository_root or repository_root in self.root.parents:
            raise StorageError("Resume storage root must be outside the repository.")
        public_directory_names = {"htdocs", "public", "static", "www", "wwwroot"}
        if any(part.casefold() in public_directory_names for part in self.root.parts):
            raise StorageError("Resume storage root must not be a public web directory.")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink() or not self.root.is_dir():
                raise StorageError("Resume storage root is not a private directory.")
            if os.name != "nt":
                self.root.chmod(0o700)
        except OSError as exc:
            raise StorageError("Unable to initialize private resume storage.") from exc

    @staticmethod
    def _validate_key(key: str) -> str:
        try:
            parsed = UUID(key)
        except (ValueError, TypeError, AttributeError) as exc:
            raise StorageError("Invalid private storage key.") from exc
        if str(parsed) != key.lower():
            raise StorageError("Invalid private storage key.")
        return str(parsed)

    def _object_path(self, key: str) -> Path:
        safe_key = self._validate_key(key)
        path = self.root / safe_key
        try:
            if path.is_symlink():
                raise StorageError("Stored object is not a regular file.")
            if path.exists() and path.resolve(strict=True).parent != self.root:
                raise StorageError("Stored object resolves outside private storage.")
        except OSError as exc:
            raise StorageError("Unable to validate stored object.") from exc
        return path

    def store_file(self, source: BinaryIO, key: str) -> None:
        destination = self._object_path(key)
        staging_path: Path | None = None
        created_destination = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".staging-",
                dir=self.root,
                delete=False,
            ) as staging:
                staging_path = Path(staging.name)
                if hasattr(source, "seek"):
                    source.seek(0)
                shutil.copyfileobj(source, staging, length=64 * 1024)
                staging.flush()
                os.fsync(staging.fileno())

            with staging_path.open("rb") as staged, destination.open("xb") as stored:
                created_destination = True
                shutil.copyfileobj(staged, stored, length=64 * 1024)
                stored.flush()
                os.fsync(stored.fileno())
            if os.name != "nt":
                destination.chmod(0o600)
        except (OSError, shutil.Error) as exc:
            if created_destination:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            raise StorageError("Unable to store the private resume.") from exc
        finally:
            if staging_path is not None:
                try:
                    staging_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def open_file(self, key: str) -> BinaryIO:
        path = self._object_path(key)
        try:
            resolved_path = path.resolve(strict=True)
            if resolved_path.parent != self.root or not resolved_path.is_file():
                raise StorageError("Stored object is not a regular private file.")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved_path, flags)
            return os.fdopen(descriptor, "rb")
        except FileNotFoundError:
            raise
        except StorageError:
            raise
        except OSError as exc:
            raise StorageError("Unable to open the private resume.") from exc

    def delete_file(self, key: str) -> None:
        path = self._object_path(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Unable to delete the private resume.") from exc

    def exists(self, key: str) -> bool:
        path = self._object_path(key)
        try:
            return path.is_file()
        except OSError as exc:
            raise StorageError("Unable to inspect the private resume.") from exc

    def list_keys(self) -> list[str]:
        try:
            keys = []
            for path in self.root.iterdir():
                if path.name.startswith(".staging-"):
                    continue
                try:
                    key = self._validate_key(path.name)
                except StorageError:
                    continue
                if path.is_symlink():
                    raise StorageError("Private storage contains an invalid object.")
                if path.is_file():
                    keys.append(key)
            return sorted(keys)
        except OSError as exc:
            raise StorageError("Unable to list private resume objects.") from exc
