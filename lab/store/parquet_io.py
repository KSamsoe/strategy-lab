"""Point-in-time Parquet store.

The store's one non-negotiable property: **every read takes ``as_of`` and
applies ``knowledge_time <= as_of`` before anything else**. A row we could not
have known yet is not filtered out late, or clamped, or warned about -- it is
structurally unreachable, because the barrier sits under the query rather than
on top of it. Everything downstream (indicators, ``DataView``, the backtester)
inherits that guarantee for free instead of re-deriving it.

Layout is hive: ``data/parquet/<table>/source=<s>/ticker=<T>/year=<Y>/part-<n>.parquet``
where ``year`` comes from ``event_time``. The three partition keys live only in
the directory names, never inside the files, so DuckDB's ``hive_partitioning=1``
and pyarrow's dataset discovery both reconstruct them without duplicate columns.
Partitioning on ticker+year is what keeps a three-ticker query from touching the
rest of the store.

Re-pulls restate rather than duplicate: a write merges into the partitions it
touches (read-merge-rewrite of those directories only) and the *later* row wins
on the dedupe key. Vendors revise history; the store has to survive that without
growing a second copy of every bar.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from lab.config import get_settings
from lab.store.schema import (
    BARS,
    PARTITION_KEYS,
    SCHEMAS,
    SIGNALS,
    Bar,
    Event,
    empty_frame,
)
from lab.timeutil import to_utc

_TS = pa.timestamp("us", tz="UTC")

#: Last write wins on these. Bars are identified by their slot in the grid;
#: events by the vendor-stable ``uid`` the adapter minted for the underlying fact.
DEDUPE_KEYS: dict[str, list[str]] = {
    BARS: ["source", "ticker", "timeframe", "event_time"],
    SIGNALS: ["source", "uid"],
}

_TIME_COLUMNS = ("event_time", "knowledge_time")
_COMPRESSION = "zstd"
_PART_GLOB = "source=*/ticker=*/year=*/*.parquet"


# --- paths -------------------------------------------------------------------


def _check_table(table: str) -> str:
    if table not in SCHEMAS:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(SCHEMAS)}")
    return table


def table_path(table: str) -> Path:
    return get_settings().paths.parquet / _check_table(table)


def _tmp_dir() -> Path:
    """Staging area for atomic rewrites, deliberately *outside* any table root
    so a half-written file can never be picked up by dataset discovery."""
    d = get_settings().paths.parquet / ".tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _part_value(value: Any, what: str) -> str:
    s = str(value)
    if not s or any(c in s for c in "/\\="):
        raise ValueError(f"{what} {s!r} is not usable as a partition directory name")
    return s


def _part_dir(table: str, source: str, ticker: str, year: int) -> Path:
    return (
        table_path(table)
        / f"source={_part_value(source, 'source')}"
        / f"ticker={_part_value(ticker, 'ticker')}"
        / f"year={int(year)}"
    )


def _next_part_index(part_dir: Path) -> int:
    n = -1
    for f in part_dir.glob("part-*.parquet"):
        stem = f.name[len("part-") : -len(".parquet")]
        if stem.isdigit():
            n = max(n, int(stem))
    return n + 1


# --- schema helpers ----------------------------------------------------------


def _file_schema(table: str) -> pa.Schema:
    """The schema as written into the files: partition keys are omitted because
    hive directories already carry them."""
    schema = SCHEMAS[_check_table(table)]
    return pa.schema([f for f in schema if f.name not in PARTITION_KEYS])


def _partitioning() -> ds.Partitioning:
    # Explicit types, so ``year`` comes back int32 rather than an inferred
    # dictionary<string> that would then need casting on every read.
    return ds.partitioning(
        pa.schema([pa.field("source", pa.string()), pa.field("ticker", pa.string()), pa.field("year", pa.int32())]),
        flavor="hive",
    )


# --- writing -----------------------------------------------------------------


def _normalize(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a caller-shaped frame into exactly the stored schema.

    Raises rather than repairs: a frame missing ``knowledge_time``, or carrying
    a row that claims to have been known before it happened, is a bug upstream
    and silently fixing it here would hide the very thing the store exists to
    make visible.
    """
    schema = SCHEMAS[table]
    out = df.copy()

    required = [f.name for f in schema if not f.nullable and f.name != "year"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{table} write is missing required column(s): {missing}")

    for name in schema.names:
        if name not in out.columns:
            out[name] = None

    for col in _TIME_COLUMNS:
        out[col] = pd.to_datetime(out[col], utc=True).dt.as_unit("us")
    if out[list(_TIME_COLUMNS)].isna().any().any():
        raise ValueError(f"{table} write has null event_time/knowledge_time")

    bad = out["knowledge_time"] < out["event_time"]
    if bool(bad.any()):
        first = out.loc[bad].iloc[0]
        raise ValueError(
            f"knowledge_time {first['knowledge_time']} precedes event_time "
            f"{first['event_time']} for {first['source']}/{first['ticker']}"
        )

    out["source"] = out["source"].astype("string").str.strip()
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    if out["source"].isna().any() or out["ticker"].isna().any():
        raise ValueError(f"{table} write has null source/ticker")
    # ``year`` is derived, never trusted from the caller: it is the partition key
    # and must agree with event_time or reads prune the wrong directories.
    out["year"] = out["event_time"].dt.year.astype("int32")

    key = [c for c in DEDUPE_KEYS[table] if c in out.columns]
    out = out.drop_duplicates(subset=key, keep="last")
    return out[schema.names].reset_index(drop=True)


def _to_arrow(table: str, part: pd.DataFrame) -> pa.Table:
    fs = _file_schema(table)
    return pa.Table.from_pandas(part[fs.names], schema=fs, preserve_index=False)


def _read_partition(table: str, part_dir: Path) -> pa.Table | None:
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None
    # Explicit schema so a partition written before a schema addition reads back
    # with nulls in the new column instead of failing to concat.
    return ds.dataset([str(f) for f in files], format="parquet", schema=_file_schema(table)).to_table()


def _write_partition(table: str, part_dir: Path, new: pa.Table, *, dedupe: bool) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_partition(table, part_dir) if dedupe else None

    if existing is None:
        merged, replace_all = new, False
    else:
        combined = pa.concat_tables([existing, new]).to_pandas()
        key = [c for c in DEDUPE_KEYS[table] if c in combined.columns]
        combined = combined.drop_duplicates(subset=key, keep="last")
        merged = pa.Table.from_pandas(combined, schema=_file_schema(table), preserve_index=False)
        replace_all = True

    tmp = _tmp_dir() / f"{uuid.uuid4().hex}.parquet"
    try:
        pq.write_table(merged, tmp, compression=_COMPRESSION)
        if replace_all:
            for stale in sorted(part_dir.glob("*.parquet")):
                stale.unlink()
            target = part_dir / "part-0.parquet"
        else:
            target = part_dir / f"part-{_next_part_index(part_dir)}.parquet"
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def write_frame(table: str, df: pd.DataFrame, *, dedupe: bool = True) -> int:
    """Persist ``df`` into ``table``; returns the number of rows written.

    Only the partitions the frame touches are read back and rewritten, so
    appending one day of bars costs one small file per (source, ticker) rather
    than a full-table rewrite.
    """
    _check_table(table)
    if df is None or len(df) == 0:
        return 0

    norm = _normalize(table, df)
    if norm.empty:
        return 0

    for (source, ticker, year), part in norm.groupby(PARTITION_KEYS, sort=True):
        _write_partition(
            table,
            _part_dir(table, str(source), str(ticker), int(year)),
            _to_arrow(table, part),
            dedupe=dedupe,
        )
    return int(len(norm))


def write_bars(bars: Iterable[Bar], *, dedupe: bool = True) -> int:
    rows = [b.as_row() for b in bars]
    if not rows:
        return 0
    return write_frame(BARS, pd.DataFrame(rows), dedupe=dedupe)


def write_events(events: Iterable[Event], *, dedupe: bool = True) -> int:
    rows = [e.as_row() for e in events]
    if not rows:
        return 0
    return write_frame(SIGNALS, pd.DataFrame(rows), dedupe=dedupe)


# --- reading -----------------------------------------------------------------


def _tickers_list(tickers: Sequence[str] | str | None) -> list[str] | None:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        tickers = [tickers]
    out = sorted({t.strip().upper() for t in tickers if str(t).strip()})
    if not out:
        raise ValueError("tickers was given but resolved to an empty list")
    return out


def _dataset(table: str) -> ds.Dataset | None:
    root = table_path(table)
    if not root.exists():
        return None
    if next(root.glob(_PART_GLOB), None) is None:
        return None
    return ds.dataset(str(root), format="parquet", partitioning=_partitioning())


def _scalar(ts: datetime) -> pa.Scalar:
    return pa.scalar(to_utc(ts), type=_TS)


def _read(
    table: str,
    *,
    tickers: Sequence[str] | str | None = None,
    source: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    as_of: datetime | None = None,
    equals: dict[str, str] | None = None,
) -> pd.DataFrame:
    schema = SCHEMAS[_check_table(table)]
    # Validate arguments before the empty-store shortcut: a typo'd query should
    # fail the same way whether or not `lab pull` has run yet.
    tick = _tickers_list(tickers)
    dataset = _dataset(table)
    if dataset is None:
        return empty_frame(table)

    expr: ds.Expression | None = None

    def _add(e: ds.Expression) -> None:
        nonlocal expr
        expr = e if expr is None else (expr & e)

    # The barrier goes on first. Order does not change the row set pyarrow
    # returns, but it does make the intent unmistakable in a stack trace and in
    # the pushed-down predicate a reader sees while debugging.
    if as_of is not None:
        _add(ds.field("knowledge_time") <= _scalar(as_of))

    if source is not None:
        _add(ds.field("source") == str(source))
    if tick is not None:
        _add(ds.field("ticker").isin(tick))
    if start is not None:
        lo = to_utc(start)
        # ``year`` is a partition key, so this prunes directories, not rows.
        _add((ds.field("year") >= lo.year) & (ds.field("event_time") >= _scalar(lo)))
    if end is not None:
        hi = to_utc(end)
        _add((ds.field("year") <= hi.year) & (ds.field("event_time") <= _scalar(hi)))
    for col, value in (equals or {}).items():
        if value is not None:
            _add(ds.field(col) == str(value))

    tbl = dataset.to_table(filter=expr) if expr is not None else dataset.to_table()
    df = tbl.select(schema.names).cast(schema).to_pandas()
    if df.empty:
        return empty_frame(table)
    return df.sort_values(["event_time", "ticker"], kind="stable").reset_index(drop=True)


def read_bars(
    tickers: Sequence[str] | str | None = None,
    *,
    timeframe: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    as_of: datetime | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Bars with ``knowledge_time <= as_of``, ``start <= event_time <= end``.

    ``as_of=None`` means "everything on disk" and is only correct outside a
    backtest -- the engine always passes it.
    """
    return _read(
        BARS,
        tickers=tickers,
        source=source,
        start=start,
        end=end,
        as_of=as_of,
        equals={"timeframe": timeframe} if timeframe else None,
    )


def read_signals(
    source: str | None = None,
    *,
    tickers: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    as_of: datetime | None = None,
    kind: str | None = None,
) -> pd.DataFrame:
    """Alt-data events. This is where ``as_of`` earns its keep: disclosure lag
    means ``knowledge_time`` can be weeks past ``event_time``, so a read
    windowed on ``event_time`` alone would serve rows nobody could have had."""
    return _read(
        SIGNALS,
        tickers=tickers,
        source=source,
        start=start,
        end=end,
        as_of=as_of,
        equals={"kind": kind} if kind else None,
    )


# --- provenance --------------------------------------------------------------


def _contributing_files(
    table: str, *, tickers: Sequence[str] | None = None, source: str | None = None
) -> list[Path]:
    root = table_path(table)
    if not root.exists():
        return []
    src_glob = f"source={_part_value(source, 'source')}" if source else "source=*"
    tick = _tickers_list(tickers)
    tick_globs = [f"ticker={_part_value(t, 'ticker')}" for t in tick] if tick else ["ticker=*"]
    files: set[Path] = set()
    for tg in tick_globs:
        files.update(root.glob(f"{src_glob}/{tg}/year=*/*.parquet"))
    return sorted(files)


def data_version(
    table: str = BARS, *, tickers: Sequence[str] | None = None, source: str | None = None
) -> str:
    """A 16-char digest of the partition files behind a query.

    Stamped on every run so a result stays attributable after the vendor
    restates history: same bytes on disk, same string; one revised bar anywhere
    in scope, different string, and the old run is visibly no longer
    reproducible rather than quietly wrong.

    The digest inputs are the store-relative posix path, byte size, mtime_ns and
    row count of each file, sorted -- so it is independent of filesystem
    enumeration order. It is *not* independent of a byte-identical rewrite:
    re-writing the same rows moves mtime and therefore the version. That is the
    conservative direction of error (a false "data changed", never a false
    "data unchanged"), which is the one worth having.
    """
    root = get_settings().paths.parquet
    parts: list[str] = []
    for f in _contributing_files(table, tickers=tickers, source=source):
        st = f.stat()
        try:
            rows = pq.ParquetFile(f).metadata.num_rows
        except (OSError, pa.ArrowInvalid):
            rows = -1
        parts.append(f"{f.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}|{rows}")
    parts.sort()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def coverage(table: str = BARS) -> pd.DataFrame:
    """What the store actually holds: one row per (source, ticker, sub-type).

    For ``bars`` the third column is ``timeframe``; for ``signals`` it carries
    ``kind``, the analogous sub-type dimension, so callers rendering the table
    do not need a branch per table.
    """
    _check_table(table)
    dataset = _dataset(table)
    cols = ["source", "ticker", "timeframe", "rows", "first", "last"]
    if dataset is None:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    sub = "timeframe" if table == BARS else "kind"
    tbl = dataset.to_table(columns=["source", "ticker", sub, "event_time"])
    df = tbl.to_pandas()
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    grouped = (
        df.groupby(["source", "ticker", sub], dropna=False)["event_time"]
        .agg(rows="size", first="min", last="max")
        .reset_index()
        .rename(columns={sub: "timeframe"})
    )
    return grouped[cols].sort_values(["source", "ticker", "timeframe"]).reset_index(drop=True)
