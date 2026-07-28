"""
Timestamp serialization for API responses.

Every `created_at`/`updated_at` in this app is stored NAIVE and in UTC: the
containers run on a UTC clock and SQLite's CURRENT_TIMESTAMP is UTC. Handing that
to a client as a bare "2026-07-28T08:26:03" is ambiguous — JavaScript parses a
date-time with no offset as LOCAL time, so a browser at UTC+5:30 read every row as
5.5 hours older than it really was, and a run recorded 6 minutes earlier rendered
as "5h ago".

iso_utc() tags the offset so the client converts instead of guessing. Timestamps
that already carry a timezone are passed through untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone


def iso_utc(dt: datetime | None) -> str | None:
    """Return an ISO-8601 string with an explicit UTC offset, or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
