"""Verbatim persistence of adapter responses.

The rule, learned the hard way: **persist the raw response first, normalize
second.** Vendor schemas drift, fields get renamed, tables get quarantined --
and when that happens you want the original bytes still on disk so the whole
history is re-parseable rather than lost.

For GovGreed this file carries extra weight. Historical backfill is an
institutional-tier feature, so the API offers no cheap way to backtest signal
history. Snapshotting every response from day one builds our own backtest
dataset for free, which is why the pull job logs before any trading logic runs.

Layout: ``data/raw/<source>/<YYYY-MM-DD>/<endpoint>.jsonl.gz``, one JSON object
per line.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from lab.config import get_settings
from lab.timeutil import to_utc, utcnow


def _safe(part: str) -> str:
    """Make an endpoint path safe for a filename."""
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in part.strip("/"))
    return out or "root"


def raw_dir(source: str, day: datetime | None = None) -> Path:
    day = to_utc(day) if day else utcnow()
    root = get_settings().paths.raw / _safe(source) / day.strftime("%Y-%m-%d")
    root.mkdir(parents=True, exist_ok=True)
    return root


def save(
    source: str,
    endpoint: str,
    payload: Any,
    *,
    request_id: str | None = None,
    fetched_at: datetime | None = None,
    params: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Append one response, verbatim, and return the file it landed in."""
    fetched = to_utc(fetched_at) if fetched_at else utcnow()
    path = raw_dir(source, fetched) / f"{_safe(endpoint)}.jsonl.gz"
    record = {
        "fetched_at": fetched.isoformat(),
        "source": source,
        "endpoint": endpoint,
        "request_id": request_id,
        "params": dict(params or {}),
        "meta": dict(meta or {}),
        "payload": payload,
    }
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    with gzip.open(path, "at", encoding="utf-8") as fh:
        fh.write(line)
    return path


def iter_raw(
    source: str,
    endpoint: str | None = None,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterator[dict[str, Any]]:
    """Replay stored snapshots in fetch order.

    Corrupt or truncated lines are skipped rather than killing a replay -- a
    half-written line from a crashed pull should cost one response, not the
    whole history.
    """
    base = get_settings().paths.raw / _safe(source)
    if not base.exists():
        return
    lo = to_utc(since) if since else None
    hi = to_utc(until) if until else None
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        pattern = f"{_safe(endpoint)}.jsonl.gz" if endpoint else "*.jsonl.gz"
        for f in sorted(day_dir.glob(pattern)):
            try:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        fetched = to_utc(rec.get("fetched_at") or day_dir.name)
                        if lo and fetched < lo:
                            continue
                        if hi and fetched > hi:
                            continue
                        rec["_fetched_at"] = fetched
                        rec["_file"] = str(f)
                        yield rec
            except (OSError, EOFError, gzip.BadGzipFile):
                continue


def load(
    source: str,
    endpoint: str | None = None,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    return list(iter_raw(source, endpoint, since=since, until=until))


def summary(source: str | None = None) -> list[dict[str, Any]]:
    """What we have snapshotted, for `lab data coverage` and the console."""
    root = get_settings().paths.raw
    if not root.exists():
        return []
    sources = [root / _safe(source)] if source else sorted(p for p in root.iterdir() if p.is_dir())
    out: list[dict[str, Any]] = []
    for src in sources:
        if not src.is_dir():
            continue
        counts: dict[str, dict[str, Any]] = {}
        for day_dir in sorted(src.iterdir()):
            if not day_dir.is_dir():
                continue
            for f in day_dir.glob("*.jsonl.gz"):
                key = f.name.removesuffix(".jsonl.gz")
                slot = counts.setdefault(
                    key, {"source": src.name, "endpoint": key, "files": 0, "bytes": 0,
                           "first_day": day_dir.name, "last_day": day_dir.name}
                )
                slot["files"] += 1
                slot["bytes"] += f.stat().st_size
                slot["first_day"] = min(slot["first_day"], day_dir.name)
                slot["last_day"] = max(slot["last_day"], day_dir.name)
        out.extend(counts.values())
    return out
