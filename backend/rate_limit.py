"""Small in-memory failure limiter used to slow down password guessing.

State lives in this process only. That is adequate for a single-instance
deployment; with several workers or servers, move the counters to a shared
store such as Redis (or enforce limits at the reverse proxy).
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Callable


class FailureLimiter:
    """Block a key after ``max_failures`` failures inside ``window_seconds``."""

    def __init__(
        self,
        max_failures: int,
        window_seconds: float,
        *,
        time_func: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._time = time_func
        self._max_keys = max_keys
        self._failures: dict[str, deque[float]] = {}
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> deque[float] | None:
        attempts = self._failures.get(key)
        if attempts is None:
            return None
        while attempts and now - attempts[0] >= self.window_seconds:
            attempts.popleft()
        if not attempts:
            del self._failures[key]
            return None
        return attempts

    def retry_after(self, key: str) -> int:
        """Return whole seconds until the key may try again (0 when not blocked)."""
        with self._lock:
            now = self._time()
            attempts = self._prune(key, now)
            if attempts is None or len(attempts) < self.max_failures:
                return 0
            return max(1, int(self.window_seconds - (now - attempts[0])) + 1)

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = self._time()
            if key not in self._failures and len(self._failures) >= self._max_keys:
                for stale in list(self._failures):
                    self._prune(stale, now)
                if len(self._failures) >= self._max_keys:
                    self._failures.pop(next(iter(self._failures)))
            self._failures.setdefault(key, deque()).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._failures.clear()


# 5 wrong passwords per account+client, and 30 failures per client, per 15 minutes.
account_login_limiter = FailureLimiter(max_failures=5, window_seconds=15 * 60)
client_login_limiter = FailureLimiter(max_failures=30, window_seconds=15 * 60)
