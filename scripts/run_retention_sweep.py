"""Housekeeping job: purge resumes past their retention date and expire old consents.

    python -m scripts.run_retention_sweep

Run it on a schedule (Windows Task Scheduler daily, or cron). It prints counts only.
Exit code 1 means at least one resume file could not be deleted; it is retried next run.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.db.session import dispose_engine, get_engine
from backend.services.retention import run_retention_cycle
from backend.storage.local_filesystem import LocalFilesystemStorage


def main() -> int:
    storage = LocalFilesystemStorage()
    try:
        with Session(get_engine()) as session:
            summary = run_retention_cycle(session, storage)
    finally:
        dispose_engine()
    print(
        f"Resumes inspected: {summary.inspected_count}, purged: {summary.expired_count}, "
        f"failed: {summary.failed_count}; consents expired: {summary.consents_expired}."
    )
    return 1 if summary.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
