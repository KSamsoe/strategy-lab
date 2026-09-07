"""Yahoo Finance bars, via the unofficial ``yfinance`` package.

**Never load-bearing.** yfinance scrapes an undocumented endpoint that Yahoo
changes without notice: it rate-limits without saying so, silently returns short
or empty frames, restates adjusted history, and has broken outright more than
once. Treat it as a convenience bootstrap for getting *some* daily history on an
arbitrary ticker while you are still exploring. Anything that matters -- a paper
run, a live run, a result you intend to believe -- should come from Alpaca or
another adapter with a contract behind it.

``adjusted=True`` throughout (``auto_adjust``), so OHLC is split- and
dividend-adjusted and every restatement changes the history. That is exactly why
``data_version`` exists downstream.
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from typing import Any, Iterator, Sequence

import pandas as pd

from lab.adapters.base import BaseAdapter
from lab.store.schema import Bar
from lab.timeutil import (
    is_intraday,
    parse_timeframe,
    session_close_utc,
    session_date,
    session_open_utc,
    to_utc,
    utcnow,
)

SOURCE = "yfinance"

#: Our timeframe strings mapped onto Yahoo's interval vocabulary.
INTERVALS: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
    "90m": "90m",
    "1d": "1d",
}

_FIELDS = ("open", "high", "low", "close", "volume")


def _clean_tickers(tickers: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(tickers, str):
        tickers = [tickers]
    out = tuple(t.strip().upper() for t in tickers if str(t).strip())
    if not out:
        raise ValueError("no tickers given")
    return out


def _frames_by_ticker(df: pd.DataFrame, syms: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split whatever shape ``yf.download`` returned into one frame per ticker.

    Multi-ticker downloads come back with a MultiIndex on the columns, and which
    level holds the ticker depends on ``group_by`` and on the yfinance version.
    Single-ticker downloads are flat on old versions and MultiIndex on new ones.
    Sniff rather than assume.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return {syms[0]: df} if len(syms) == 1 else {}

    wanted = set(syms)
    level = 0
    for candidate in range(df.columns.nlevels):
        if wanted & {str(v).upper() for v in df.columns.get_level_values(candidate)}:
            level = candidate
            break

    out: dict[str, pd.DataFrame] = {}
    for sym in syms:
        try:
            sub = df.xs(sym, axis=1, level=level)
        except KeyError:
            continue
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = sub.columns.get_level_values(-1)
        out[sym] = sub
    return out


def _normalize_columns(sub: pd.DataFrame) -> pd.DataFrame:
    renamed = sub.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in sub.columns})
    if "close" not in renamed.columns and "adj_close" in renamed.columns:
        renamed = renamed.rename(columns={"adj_close": "close"})
    return renamed


def _day_of(stamp: pd.Timestamp) -> date:
    # Daily rows come back tz-naive and already keyed by session date; intraday
    # rows are tz-aware in exchange time and need the ET conversion.
    return stamp.date() if stamp.tzinfo is None else session_date(stamp)


class YFinanceAdapter(BaseAdapter):
    """Daily and intraday OHLCV from Yahoo. Bars only, no alt-data."""

    name = SOURCE
    provides = frozenset({"bars"})

    def __init__(
        self,
        *,
        source: str = SOURCE,
        chunk_size: int = 50,
        persist_raw: bool = False,
        prepost: bool = False,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.source = source
        self.chunk_size = int(chunk_size)
        # Bars are re-fetchable and bulky; the verbatim-snapshot discipline
        # earns its keep on alt-data, where the vendor offers no backfill.
        self.persist_raw = bool(persist_raw)
        self.prepost = bool(prepost)
        self.last_call: datetime | None = None
        self.calls_made = 0

    def available(self) -> tuple[bool, str]:
        try:
            found = importlib.util.find_spec("yfinance") is not None
        except (ImportError, ValueError):  # a half-installed package
            found = False
        if not found:
            return False, "yfinance not installed (pip install 'strategy-lab[data]')"
        return True, ""

    def info(self):
        got = super().info()
        got.last_call = self.last_call
        got.detail = {"calls_made": self.calls_made, "unofficial": True}
        return got

    def interval(self, timeframe: str) -> str:
        tf = str(timeframe).strip().lower()
        parse_timeframe(tf)  # rejects junk with a useful message
        if tf not in INTERVALS:
            raise ValueError(f"yfinance has no interval for {tf!r}; known: {sorted(INTERVALS)}")
        return INTERVALS[tf]

    def fetch_bars(
        self,
        tickers: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        self.require_available()
        syms = _clean_tickers(tickers)
        tf = str(timeframe).strip().lower()
        interval = self.interval(tf)
        step = parse_timeframe(tf)
        lo, hi = to_utc(start), to_utc(end)
        if lo > hi:
            raise ValueError(f"start {lo.isoformat()} is after end {hi.isoformat()}")

        import yfinance as yf  # noqa: PLC0415 - optional dep, imported on use only

        for chunk in (syms[i : i + self.chunk_size] for i in range(0, len(syms), self.chunk_size)):
            frame = yf.download(
                tickers=list(chunk),
                start=lo.date(),
                # Yahoo's `end` is exclusive; nudge it so the caller's last day
                # is actually included.
                end=hi.date() + timedelta(days=1),
                interval=interval,
                auto_adjust=True,
                actions=False,
                prepost=self.prepost,
                progress=False,
                threads=False,
                group_by="ticker",
            )
            self.calls_made += 1
            self.last_call = utcnow()
            if frame is None or frame.empty:
                continue
            if self.persist_raw:
                self._snapshot(chunk, tf, lo, hi, frame)
            for sym, sub in _frames_by_ticker(frame, chunk).items():
                yield from self._bars_from(sym, tf, step, _normalize_columns(sub))

    def _bars_from(
        self, sym: str, tf: str, step: timedelta, sub: pd.DataFrame
    ) -> Iterator[Bar]:
        if not {"open", "high", "low", "close"}.issubset(sub.columns):
            return
        daily = not is_intraday(tf)
        for raw_ts, row in sub.iterrows():
            values = [row.get(f) for f in _FIELDS]
            if any(v is None or pd.isna(v) for v in values[:4]):
                continue  # Yahoo pads non-trading rows with NaN on multi-pulls
            o, h, l, c, v = (float(x) if x is not None and not pd.isna(x) else 0.0 for x in values)
            stamp = pd.Timestamp(raw_ts)
            day = _day_of(stamp)
            if daily:
                event_time = session_open_utc(day)
                knowledge_time = session_close_utc(day)
            else:
                event_time = to_utc(stamp)
                # You know a bar once it has closed -- and never later than the
                # session close, since Yahoo truncates the last bar of the day.
                knowledge_time = min(event_time + step, session_close_utc(day))
                if knowledge_time < event_time:
                    knowledge_time = event_time + step
            yield Bar(
                event_time=event_time,
                knowledge_time=knowledge_time,
                source=self.source,
                ticker=sym,
                timeframe=tf,
                open=o,
                high=max(h, o, c),
                low=min(l, o, c),
                close=c,
                volume=v,
                adjusted=True,
            )

    def _snapshot(
        self,
        chunk: Sequence[str],
        tf: str,
        lo: datetime,
        hi: datetime,
        frame: pd.DataFrame,
    ) -> None:
        from lab.store import raw  # noqa: PLC0415 - keeps the import graph shallow

        payload: Any
        try:
            payload = frame.to_json(orient="split", date_format="iso")
        except (TypeError, ValueError):
            payload = frame.astype(str).to_dict()
        raw.save(
            self.source,
            "download",
            payload,
            params={
                "tickers": list(chunk),
                "timeframe": tf,
                "start": lo.isoformat(),
                "end": hi.isoformat(),
            },
        )
