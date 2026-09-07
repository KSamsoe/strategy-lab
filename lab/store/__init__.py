"""The data spine: a two-timestamp, point-in-time store over Parquet + DuckDB.

Import surface is flat on purpose -- ``from lab.store import read_bars, as_of``
should be all a strategy author or notebook ever needs to remember.
"""

from __future__ import annotations

from lab.store import duck, raw
from lab.store.duck import connect, describe, query, register_views
from lab.store.parquet_io import (
    DEDUPE_KEYS,
    coverage,
    data_version,
    read_bars,
    read_signals,
    table_path,
    write_bars,
    write_events,
    write_frame,
)
from lab.store.schema import (
    BARS,
    BARS_SCHEMA,
    PARTITION_KEYS,
    SCHEMAS,
    SIGNALS,
    SIGNALS_SCHEMA,
    Bar,
    Event,
    empty_frame,
)

__all__ = [
    "BARS",
    "BARS_SCHEMA",
    "Bar",
    "DEDUPE_KEYS",
    "Event",
    "PARTITION_KEYS",
    "SCHEMAS",
    "SIGNALS",
    "SIGNALS_SCHEMA",
    "connect",
    "coverage",
    "data_version",
    "describe",
    "duck",
    "empty_frame",
    "query",
    "raw",
    "read_bars",
    "read_signals",
    "register_views",
    "table_path",
    "write_bars",
    "write_events",
    "write_frame",
]
