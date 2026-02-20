"""
Lightweight file-based caching helpers.

Each helper follows the same contract:
  - is_cache_fresh(path, ttl)  → bool
  - The caller is responsible for reading / writing; helpers only check age.

This keeps I/O logic out of service modules and makes the TTL policy easy to test.
"""

import os
from datetime import datetime


def cache_age_hours(path: str) -> float:
    """Return how many hours old *path* is (mtime-based). Returns ∞ if missing."""
    if not os.path.exists(path):
        return float("inf")
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).total_seconds() / 3600


def cache_age_days(path: str) -> float:
    """Return how many days old *path* is (mtime-based). Returns ∞ if missing."""
    return cache_age_hours(path) / 24


def is_fresh_hours(path: str, ttl_hours: float) -> bool:
    """True if *path* exists and is younger than *ttl_hours*."""
    return cache_age_hours(path) < ttl_hours


def is_fresh_days(path: str, ttl_days: float) -> bool:
    """True if *path* exists and is younger than *ttl_days*."""
    return cache_age_days(path) < ttl_days


def ensure_cache_dirs(*dirs: str) -> None:
    """Create cache directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)