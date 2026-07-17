"""cache.py — Lightweight LRU cache with optional TTL for ML results."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class LRUCache:
    """Least-Recently-Used cache with optional per-entry TTL.

    Thread-safe for concurrent reads/writes. Designed for high-hit-rate,
    low-contention scenarios such as caching embedding vectors and reranker scores.
    """

    def __init__(self, maxsize: int = 1000, default_ttl: float | None = None):
        self._data: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> object | None:
        with self._lock:
            if key not in self._data:
                return None
            value, expires = self._data[key]
            if expires < time.time():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: object, ttl: float | None = None) -> None:
        with self._lock:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
            expires = time.time() + (ttl if ttl is not None else self._default_ttl or 3.15e8)
            self._data[key] = (value, expires)
            self._data.move_to_end(key)

    def invalidate(self, pattern: str | None = None) -> int:
        """Remove all entries whose key contains *pattern* (or everything)."""
        with self._lock:
            if pattern is None:
                count = len(self._data)
                self._data.clear()
                return count
            keys = [k for k in self._data if pattern in k]
            for k in keys:
                del self._data[k]
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            _now = time.time()
            expired = [k for k, (_, e) in self._data.items() if e < _now]
            for k in expired:
                del self._data[k]
            return len(self._data)
