"""Canonical event schema.

Everything the lab stores -- an OHLCV bar, a GovGreed composite signal, an
insider score -- normalizes into a record carrying **two timestamps**:

    event_time      when the thing happened in the world
    knowledge_time  when we could actually have known it

For bars these are effectively equal. For alt-data they are emphatically not:
a congressional trade has an ``event_time`` up to 45 days before its disclosure
``knowledge_time``. Conflating them is the single most common source of
fraudulent-looking backtests, so the store indexes on ``knowledge_time`` and the
backtester serves data by it. Nothing downstream is allowed to see a row whose
``knowledge_time`` is in the future relative to ``ctx.now``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import pyarrow as pa

from lab.timeutil import to_utc

# --- table names -------------------------------------------------------------
BARS = "bars"
SIGNALS = "signals"

#: Columns present on every stored table, in canonical order.
COMMON_COLUMNS = ["event_time", "knowledge_time", "source", "ticker"]


BARS_SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("knowledge_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("timeframe", pa.string(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("vwap", pa.float64()),
        pa.field("trade_count", pa.float64()),
        pa.field("adjusted", pa.bool_()),
        pa.field("year", pa.int32(), nullable=False),
    ]
)

#: Alt-data lands here. ``payload`` keeps the normalized-but-source-shaped body
#: as a JSON string so schema drift at the vendor cannot break the table.
SIGNALS_SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("knowledge_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("direction", pa.string()),
        pa.field("tier", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("sector", pa.string()),
        pa.field("fresh", pa.bool_()),
        pa.field("uid", pa.string(), nullable=False),
        pa.field("payload", pa.string()),
        pa.field("request_id", pa.string()),
        pa.field("year", pa.int32(), nullable=False),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {BARS: BARS_SCHEMA, SIGNALS: SIGNALS_SCHEMA}

#: Partition layout on disk: ``<table>/source=.../ticker=.../year=.../part.parquet``
PARTITION_KEYS = ["source", "ticker", "year"]


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. ``event_time`` is the bar's *open* timestamp."""

    event_time: datetime
    knowledge_time: datetime
    source: str
    ticker: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: float | None = None
    adjusted: bool = True

    def as_row(self) -> dict[str, Any]:
        et, kt = to_utc(self.event_time), to_utc(self.knowledge_time)
        return {
            "event_time": et,
            "knowledge_time": kt,
            "source": self.source,
            "ticker": self.ticker.upper(),
            "timeframe": self.timeframe,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "vwap": None if self.vwap is None else float(self.vwap),
            "trade_count": None if self.trade_count is None else float(self.trade_count),
            "adjusted": bool(self.adjusted),
            "year": et.year,
        }


@dataclass(frozen=True, slots=True)
class Event:
    """A non-bar datum: an alt-data signal, score, or annotation.

    ``uid`` must be stable for the same underlying fact so that re-pulls
    deduplicate instead of double-counting. ``payload`` carries whatever the
    source gave us that does not fit the typed columns.
    """

    event_time: datetime
    knowledge_time: datetime
    source: str
    ticker: str
    kind: str
    uid: str
    direction: str | None = None
    tier: str | None = None
    score: float | None = None
    sector: str | None = None
    fresh: bool | None = None
    request_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        import json

        et, kt = to_utc(self.event_time), to_utc(self.knowledge_time)
        if kt < et:
            # Knowing something before it happened is always a bug upstream.
            raise ValueError(
                f"knowledge_time {kt} precedes event_time {et} for {self.source}/{self.ticker}"
            )
        return {
            "event_time": et,
            "knowledge_time": kt,
            "source": self.source,
            "ticker": self.ticker.upper(),
            "kind": self.kind,
            "direction": self.direction,
            "tier": self.tier,
            "score": None if self.score is None else float(self.score),
            "sector": self.sector,
            "fresh": self.fresh,
            "uid": self.uid,
            "payload": json.dumps(dict(self.payload), default=str, sort_keys=True),
            "request_id": self.request_id,
            "year": et.year,
        }


def empty_frame(table: str):
    """An empty pandas frame with the right dtypes for ``table``."""
    return SCHEMAS[table].empty_table().to_pandas()
