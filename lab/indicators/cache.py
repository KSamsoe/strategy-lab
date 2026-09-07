"""Indicator cache keyed by ``(ticker, indicator, params, data_version)``.

A sweep recomputes the same SMA over the same bars a few thousand times, so the
cache is what makes parameter search cheap. The interesting part of the key is
``data_version``: when the store restates a partition its digest changes, every
key derived from it changes with it, and stale series become unreachable rather
than silently wrong. That is the whole reason the version is in the key instead
of an mtime check -- restated data must invalidate results, not just newer data.

Params hash order-independently (canonical JSON, sorted keys, numeric types
normalized) so ``{"n": 20, "field": "close"}`` and ``{"field": "close", "n": 20}``
are one cache entry rather than two.

Entries are parquet under ``data/cache/indicators/``, written to a temp file and
renamed, because a half-written cache entry that reads back as valid data is
worse than no cache at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lab.config import get_settings
from lab.indicators.computed import compute

_SUBDIR = "indicators"
_KIND = b"lab_kind"
_NAME = b"lab_name"
_FREQ = b"lab_freq"

_hits = 0
_misses = 0


def cache_key(ticker: str, indicator: str, params: Mapping[str, Any], data_version: str) -> str:
    """A filesystem-safe key: ``<TICKER>_<indicator>_<32 hex>``.

    The readable prefix is there so :func:`clear` can take one, and so a human
    staring at ``data/cache/indicators`` can tell what they are looking at; the
    digest carries the params and the data version.
    """
    tk = str(ticker).strip().upper()
    ind = str(indicator).strip().lower()
    ver = str(data_version).strip()
    if not tk:
        raise ValueError("ticker must be a non-empty string")
    if not ind:
        raise ValueError("indicator must be a non-empty string")
    if not ver:
        # No version means the key cannot notice a restatement. Refuse rather
        # than hand back results from data that no longer exists.
        raise ValueError("data_version must be a non-empty string")
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ValueError(f"params must be a mapping, got {type(params).__name__}")
    payload = "|".join([tk, ind, _canonical(params), ver])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{_slug(tk)}_{_slug(ind)}_{digest}"


def get(key: str) -> pd.Series | pd.DataFrame | None:
    """Read an entry, or ``None`` on a miss. Counts the hit/miss for stats()."""
    global _hits, _misses
    path = _path(key)
    if not path.exists():
        _misses += 1
        return None
    try:
        table = pq.read_table(path)
        value = _from_table(table)
    except Exception:
        # A corrupt entry is a miss, not an outage: drop it and recompute.
        path.unlink(missing_ok=True)
        _misses += 1
        return None
    _hits += 1
    return value


def put(key: str, value: pd.Series | pd.DataFrame) -> None:
    if not isinstance(value, (pd.Series, pd.DataFrame)):
        raise ValueError(f"cache value must be a Series or DataFrame, got {type(value).__name__}")
    path = _path(key)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(value, pd.Series):
        name = value.name
        frame = value.to_frame(name="value" if name is None else str(name))
        meta = {_KIND: b"series"}
        if name is not None:
            meta[_NAME] = str(name).encode("utf-8")
    else:
        bad = [c for c in value.columns if not isinstance(c, str)]
        if bad:
            raise ValueError(f"cached frame columns must be strings; got {bad}")
        frame = value
        meta = {_KIND: b"frame"}

    # Parquet has no slot for a DatetimeIndex freq, and a cache hit that returns
    # a subtly different object from a cache miss is a bug generator.
    freq = getattr(getattr(frame.index, "freq", None), "freqstr", None)
    if freq:
        meta[_FREQ] = freq.encode("utf-8")

    table = pa.Table.from_pandas(frame, preserve_index=True)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})

    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def cached_indicator(
    ticker: str,
    name: str,
    df: pd.DataFrame,
    data_version: str,
    **params: Any,
) -> pd.Series | pd.DataFrame:
    """``compute(name, df, **params)`` with the result memoized on disk."""
    key = cache_key(ticker, name, params, data_version)
    hit = get(key)
    if hit is not None:
        return hit
    value = compute(name, df, **params)
    put(key, value)
    return value


def clear(prefix: str | None = None) -> int:
    """Delete cache entries whose key starts with ``prefix``. Returns the count."""
    directory = _dir()
    if not directory.exists():
        return 0
    removed = 0
    for path in sorted(directory.glob("*.parquet")):
        if prefix and not path.stem.startswith(prefix):
            continue
        path.unlink(missing_ok=True)
        removed += 1
    if not prefix:
        for stale in directory.glob("*.tmp"):
            stale.unlink(missing_ok=True)
    return removed


def stats() -> dict[str, int]:
    directory = _dir()
    entries, total = 0, 0
    if directory.exists():
        for path in directory.glob("*.parquet"):
            entries += 1
            total += path.stat().st_size
    return {"entries": entries, "bytes": total, "hits": _hits, "misses": _misses}


def reset_stats() -> None:
    """Zero the hit/miss counters. Process-local; tests and the console's
    per-run stats want a baseline."""
    global _hits, _misses
    _hits = 0
    _misses = 0


# --- internals ---------------------------------------------------------------


def _dir() -> Path:
    # Resolved per call, never at import: settings are env-driven and tests
    # repoint LAB_DATA_DIR between cases.
    return get_settings().paths.cache / _SUBDIR


def _path(key: str) -> Path:
    k = str(key).strip()
    if not k or k != _slug(k):
        raise ValueError(f"bad cache key {key!r}; expected the output of cache_key()")
    return _dir() / f"{k}.parquet"


def _slug(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(text))


def _canonical(params: Mapping[str, Any]) -> str:
    return json.dumps(
        {str(k): _plain(v) for k, v in params.items()},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _plain(value: Any) -> Any:
    """Normalize a param value to something JSON-stable.

    numpy scalars from a YAML-driven grid must hash the same as the python
    scalars a hand-written call passes, or the cache misses forever.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _from_table(table: pa.Table) -> pd.Series | pd.DataFrame:
    meta = table.schema.metadata or {}
    frame = table.to_pandas()

    freq = meta.get(_FREQ)
    if freq is not None and isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index.freq = freq.decode("utf-8")
        except (ValueError, TypeError):
            pass  # the stored freq no longer validates against the index; harmless

    if meta.get(_KIND) != b"series":
        return frame
    out = frame.iloc[:, 0]
    raw = meta.get(_NAME)
    out.name = raw.decode("utf-8") if raw is not None else None
    return out


__all__: Iterable[str] = [
    "cache_key",
    "cached_indicator",
    "clear",
    "get",
    "put",
    "reset_stats",
    "stats",
]
