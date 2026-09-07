"""Fetched indicators: alt-data placed onto the bar timeline.

Alt-data arrives already scored -- a GovGreed composite, an insider tier -- so
there is nothing to compute. The only job here is *when* each score becomes
visible, and the answer is always ``knowledge_time``, never ``event_time``.

That distinction is the entire reason the store carries two timestamps. A
congressional trade has an ``event_time`` up to 45 days before its disclosure
``knowledge_time``; aligning on ``event_time`` produces a strategy that trades
on filings six weeks before they exist and a backtest that looks like a
printing press. An event lands on the first bar whose timestamp is at or after
its ``knowledge_time``, which is the same ``knowledge_time <= now`` rule the
engine's ``DataView`` enforces -- stated once here, in bar-index terms.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lab.timeutil import to_utc

#: Aggregations allowed when several events land on the same bar. Anything
#: else is a typo, and a silently-ignored aggregation is a silently-wrong
#: signal.
AGGS = frozenset({"last", "first", "mean", "median", "sum", "min", "max", "count", "std"})


def align(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    field: str = "score",
    agg: str = "last",
    ffill_limit: int | None = None,
) -> pd.Series:
    """Place ``events[field]`` onto the bar timeline ``index`` by knowledge time.

    Each event lands on the first bar at or after its ``knowledge_time``.
    Events known only after the last bar are dropped -- they are the future.
    Bars with no news carry the last known value, forward-filled for at most
    ``ffill_limit`` bars (``None`` = unlimited, ``0`` = no fill at all), so a
    score persists as state until it is restated rather than blinking off.
    """
    idx = _bar_index(index)
    if not isinstance(events, pd.DataFrame):
        raise ValueError(f"events must be a DataFrame, got {type(events).__name__}")
    agg = str(agg).strip().lower()
    if agg not in AGGS:
        raise ValueError(f"unknown agg {agg!r}; expected one of {sorted(AGGS)}")
    limit = _limit(ffill_limit)

    if len(idx) == 0:
        return pd.Series(np.array([], dtype="float64"), index=idx, name=field)
    if events.empty:
        return _blank(idx, field)
    if "knowledge_time" not in events.columns:
        raise ValueError(
            "events frame has no 'knowledge_time' column; alignment on event_time "
            "would be look-ahead"
        )
    if field not in events.columns:
        raise ValueError(
            f"events frame has no column {field!r}; columns are "
            f"{[str(c) for c in events.columns]}"
        )

    # Vectorized equivalent of lab.timeutil.to_utc over a column: naive values
    # are read as UTC, aware ones converted.
    kt = pd.to_datetime(events["knowledge_time"], utc=True)
    if kt.isna().any():
        raise ValueError("events carry a null knowledge_time; refusing to guess when")

    order = np.argsort(kt.to_numpy(), kind="stable")  # stable: ties keep frame order
    kt_sorted = kt.to_numpy()[order]
    values = events[field].to_numpy()[order]

    # side="left" -> the first bar t with t >= knowledge_time, i.e. the first
    # bar on which "knowledge_time <= t" holds.
    pos = np.searchsorted(idx.to_numpy(), kt_sorted, side="left")
    visible = pos < len(idx)
    if not visible.any():
        return _blank(idx, field)

    bucketed = pd.Series(values[visible], index=pos[visible])
    reduced = getattr(bucketed.groupby(level=0), agg)()
    placed = pd.Series(
        reduced.to_numpy(), index=idx[reduced.index.to_numpy()], name=field
    )

    out = placed.reindex(idx)
    if limit is None:
        out = out.ffill()
    elif limit > 0:
        out = out.ffill(limit=limit)
    out.name = field
    out.index.name = getattr(index, "name", None)
    return out


def series(
    source: str,
    ticker: str,
    index: pd.DatetimeIndex,
    *,
    field: str = "score",
    as_of: datetime | None = None,
    **query: Any,
) -> pd.Series:
    """Read ``source`` signals for one ticker from the store and align them.

    ``agg`` and ``ffill_limit`` are forwarded to :func:`align`; every other
    keyword goes to ``lab.store.parquet_io.read_signals`` and is checked
    against its signature first, so a mistyped filter fails loudly instead of
    quietly widening the query.
    """
    idx = _bar_index(index)
    tk = str(ticker).strip().upper()
    if not tk:
        raise ValueError("ticker must be a non-empty string")
    src = str(source).strip() if source is not None else None

    agg = query.pop("agg", "last")
    ffill_limit = query.pop("ffill_limit", None)
    cutoff = to_utc(as_of) if as_of is not None else None

    # Lazy: the store is a sibling module and the import cost is real in a sweep
    # that never touches alt-data.
    from lab.store import parquet_io

    accepted = set(inspect.signature(parquet_io.read_signals).parameters)
    kwargs: dict[str, Any] = {"tickers": [tk], **query}
    if cutoff is not None:
        kwargs["as_of"] = cutoff
    unsupported = set(kwargs) - accepted
    if unsupported:
        raise ValueError(
            f"read_signals does not accept {sorted(unsupported)}; accepts {sorted(accepted)}"
        )

    frame = parquet_io.read_signals(src, **kwargs)
    if frame is None or len(frame) == 0:
        return _blank(idx, field)

    # Re-apply both filters locally. The store owns the look-ahead barrier, but
    # this is the cheapest possible second lock on the one door that must never
    # be left open.
    if "ticker" in frame.columns:
        frame = frame[frame["ticker"].astype("string").str.upper() == tk]
    if cutoff is not None and "knowledge_time" in frame.columns:
        frame = frame[pd.to_datetime(frame["knowledge_time"], utc=True) <= cutoff]
    if frame.empty:
        return _blank(idx, field)

    return align(frame, idx, field=field, agg=agg, ffill_limit=ffill_limit)


# --- helpers -----------------------------------------------------------------


def _bar_index(index: Any) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        idx = index
    else:
        try:
            idx = pd.DatetimeIndex(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"index must be datetime-like: {exc}") from None
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if not idx.is_monotonic_increasing:
        raise ValueError("bar index must be sorted ascending to align by knowledge_time")
    return idx


def _limit(ffill_limit: int | None) -> int | None:
    if ffill_limit is None:
        return None
    if isinstance(ffill_limit, bool) or not isinstance(ffill_limit, (int, np.integer)):
        raise ValueError(f"ffill_limit must be an integer or None, got {ffill_limit!r}")
    limit = int(ffill_limit)
    if limit < 0:
        raise ValueError(f"ffill_limit must be >= 0, got {limit}")
    return limit


def _blank(idx: pd.DatetimeIndex, field: str) -> pd.Series:
    return pd.Series(np.full(len(idx), np.nan), index=idx, name=field, dtype="float64")


__all__: Iterable[str] = ["AGGS", "align", "series"]
