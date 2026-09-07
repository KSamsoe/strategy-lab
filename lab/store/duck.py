"""DuckDB façade over the Parquet store.

Same files, second door. The backtester reads through :mod:`lab.store.parquet_io`
because it needs typed frames and the ``as_of`` barrier applied for it; ad-hoc
research reads through here because SQL over hive-partitioned Parquet is the
fastest way to answer a question nobody anticipated. Both see identical bytes,
so a notebook finding and a backtest can never disagree about the data.

The one thing this layer does *not* do is apply the knowledge-time barrier.
Anything reachable from ``bars``/``signals`` here is the raw store, future
knowledge included -- which is correct for provenance and coverage queries and
wrong for anything feeding a strategy. Research SQL that will inform a backtest
must carry its own ``WHERE knowledge_time <= ...``.
"""

from __future__ import annotations

from contextlib import closing
from typing import Any, Mapping, Sequence

import pandas as pd

from lab.store.parquet_io import _PART_GLOB, table_path
from lab.store.schema import BARS, PARTITION_KEYS, SCHEMAS, SIGNALS

_VIEWS = (BARS, SIGNALS)

#: DuckDB types for the hive columns, which arrive from directory names as text.
_PARTITION_CASTS = {"source": "VARCHAR", "ticker": "VARCHAR", "year": "INTEGER"}


def _glob(table: str) -> str | None:
    root = table_path(table)
    if not root.exists() or next(root.glob(_PART_GLOB), None) is None:
        return None
    pattern = f"{root.as_posix()}/{_PART_GLOB}"
    return pattern.replace("'", "''")


def _select_list(table: str) -> str:
    parts = []
    for field in SCHEMAS[table]:
        if field.name in PARTITION_KEYS:
            parts.append(f'CAST("{field.name}" AS {_PARTITION_CASTS[field.name]}) AS "{field.name}"')
        else:
            parts.append(f'"{field.name}"')
    return ", ".join(parts)


def register_views(con: Any) -> None:
    """Register ``bars`` and ``signals`` on ``con``.

    An empty store is a normal state -- a fresh clone has one -- so a missing
    table becomes a zero-row view with the right column types rather than a
    "no files found" error. Every caller then gets the same SQL surface whether
    or not `lab pull` has run yet.
    """
    for table in _VIEWS:
        pattern = _glob(table)
        if pattern is None:
            empty = f"_empty_{table}"
            con.register(empty, SCHEMAS[table].empty_table())
            con.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM "{empty}"')
            continue
        con.execute(
            f'CREATE OR REPLACE VIEW "{table}" AS SELECT {_select_list(table)} '
            f"FROM read_parquet('{pattern}', hive_partitioning=1, union_by_name=1)"
        )


def connect() -> Any:
    """A fresh in-memory connection with the store's views registered.

    In-memory and per-call by design: the durable state is the Parquet tree, so
    a connection is a disposable query plan holder and two of them can never
    disagree or lock each other out.
    """
    import duckdb

    con = duckdb.connect(":memory:")
    register_views(con)
    return con


def query(sql: str, params: Sequence[Any] | Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run ``sql`` against the store and return a pandas frame.

    ``params`` is positional (``?``) for a sequence, named (``$name``) for a
    mapping. Use it -- string-formatting a ticker into SQL is how a research
    query becomes an injection in the API layer later.
    """
    with closing(connect()) as con:
        if params is None:
            rel = con.execute(sql)
        elif isinstance(params, Mapping):
            rel = con.execute(sql, dict(params))
        else:
            rel = con.execute(sql, list(params))
        if rel.description is None:
            return pd.DataFrame()
        return rel.fetchdf()


def describe() -> pd.DataFrame:
    """One row per view: what is in the store and over what span."""
    rows: list[dict[str, Any]] = []
    with closing(connect()) as con:
        for table in _VIEWS:
            r = con.execute(
                f'SELECT count(*), count(DISTINCT source), count(DISTINCT ticker), '
                f'min(event_time), max(event_time), max(knowledge_time) FROM "{table}"'
            ).fetchone()
            rows.append(
                {
                    "table": table,
                    "rows": int(r[0]),
                    "sources": int(r[1]),
                    "tickers": int(r[2]),
                    "first": r[3],
                    "last": r[4],
                    "last_known": r[5],
                }
            )
    return pd.DataFrame(rows)
