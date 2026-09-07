"""Alpaca historical bars via ``alpaca-py``.

The free data tier gives daily and minute bars plus a live stream, which is
enough for v1 and comes with an actual contract behind it -- so this, not
yfinance, is the source anything load-bearing should run on.

Two shapes of failure are handled as *state* rather than exceptions: the
optional dependency may be absent, and the API keys may be absent. Either way
``available()`` returns ``(False, reason)`` and construction still succeeds, so
``lab adapters`` prints a table instead of a traceback. Credentials are read
lazily from :func:`lab.config.get_settings` at call time, so a test that swaps
the environment and clears the settings cache sees the change.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from typing import Any, Iterator, Sequence

from lab.adapters.base import BaseAdapter
from lab.config import get_settings
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

SOURCE = "alpaca"

#: Our timeframe unit letters mapped onto alpaca-py's ``TimeFrameUnit`` names.
_UNITS = {"m": "Minute", "h": "Hour", "d": "Day"}

#: Range chunking. Alpaca paginates internally, but a single request covering a
#: decade of minute bars still buffers an absurd amount before it returns, and a
#: transient failure loses the whole thing.
_WINDOW_DAYS = {"m": 30, "h": 180, "d": 730}


def _clean_tickers(tickers: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(tickers, str):
        tickers = [tickers]
    out = tuple(t.strip().upper() for t in tickers if str(t).strip())
    if not out:
        raise ValueError("no tickers given")
    return out


def _split_timeframe(timeframe: str) -> tuple[int, str]:
    tf = str(timeframe).strip().lower()
    parse_timeframe(tf)  # rejects junk with a useful message
    unit = tf[-1]
    if unit not in _UNITS:
        raise ValueError(f"alpaca has no timeframe for {tf!r}; expected e.g. '1d', '5m', '1h'")
    return int(tf[:-1]), unit


def _jsonable(obj: Any) -> Any:
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                pass
    return getattr(obj, "__dict__", str(obj))


class AlpacaAdapter(BaseAdapter):
    """Historical OHLCV from Alpaca's market-data API. Bars only."""

    name = SOURCE
    provides = frozenset({"bars"})

    def __init__(
        self,
        *,
        key_id: str | None = None,
        secret_key: str | None = None,
        feed: str = "iex",
        adjustment: str = "all",
        source: str = SOURCE,
        chunk_tickers: int = 100,
        persist_raw: bool = False,
    ) -> None:
        if chunk_tickers < 1:
            raise ValueError("chunk_tickers must be >= 1")
        self._key_id = key_id
        self._secret_key = secret_key
        self.feed = feed.strip().lower()
        self.adjustment = adjustment.strip().lower()
        self.source = source
        self.chunk_tickers = int(chunk_tickers)
        # See the note in the yfinance adapter: bars are re-fetchable, so the
        # verbatim snapshot is opt-in here rather than mandatory.
        self.persist_raw = bool(persist_raw)
        self.last_call: datetime | None = None
        self.calls_made = 0

    # -- availability --------------------------------------------------------
    def credentials(self) -> tuple[str | None, str | None]:
        settings = get_settings()
        return (
            self._key_id or settings.alpaca_key_id,
            self._secret_key or settings.alpaca_secret_key,
        )

    def available(self) -> tuple[bool, str]:
        try:
            found = importlib.util.find_spec("alpaca") is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return False, "alpaca-py not installed (pip install 'strategy-lab[data]')"
        key_id, secret = self.credentials()
        if not key_id:
            return False, "no ALPACA_API_KEY_ID"
        if not secret:
            return False, "no ALPACA_API_SECRET_KEY"
        return True, ""

    def info(self):
        got = super().info()
        got.last_call = self.last_call
        got.detail = {"feed": self.feed, "adjustment": self.adjustment, "calls_made": self.calls_made}
        return got

    # -- client --------------------------------------------------------------
    def _client(self):
        self.require_available()
        from alpaca.data.historical import StockHistoricalDataClient  # noqa: PLC0415

        key_id, secret = self.credentials()
        return StockHistoricalDataClient(key_id, secret)

    def _timeframe(self, timeframe: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: PLC0415

        amount, unit = _split_timeframe(timeframe)
        return TimeFrame(amount, getattr(TimeFrameUnit, _UNITS[unit]))

    def _request_kwargs(self) -> dict[str, Any]:
        from alpaca.data.enums import Adjustment, DataFeed  # noqa: PLC0415

        extra: dict[str, Any] = {}
        adjustment = {a.value: a for a in Adjustment}.get(self.adjustment)
        if adjustment is not None:
            extra["adjustment"] = adjustment
        feed = {f.value: f for f in DataFeed}.get(self.feed)
        if feed is not None:
            extra["feed"] = feed
        return extra

    # -- bars ----------------------------------------------------------------
    def fetch_bars(
        self,
        tickers: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        syms = _clean_tickers(tickers)
        tf = str(timeframe).strip().lower()
        _, unit = _split_timeframe(tf)
        step = parse_timeframe(tf)
        lo, hi = to_utc(start), to_utc(end)
        if lo > hi:
            raise ValueError(f"start {lo.isoformat()} is after end {hi.isoformat()}")

        from alpaca.data.requests import StockBarsRequest  # noqa: PLC0415

        client = self._client()
        alpaca_tf = self._timeframe(tf)
        extra = self._request_kwargs()
        window = timedelta(days=_WINDOW_DAYS[unit])
        daily = not is_intraday(tf)

        for group in (
            syms[i : i + self.chunk_tickers] for i in range(0, len(syms), self.chunk_tickers)
        ):
            cursor = lo
            while cursor <= hi:
                stop = min(cursor + window, hi)
                request = StockBarsRequest(
                    symbol_or_symbols=list(group),
                    timeframe=alpaca_tf,
                    start=cursor,
                    end=stop,
                    **extra,
                )
                payload = client.get_stock_bars(request)
                self.calls_made += 1
                self.last_call = utcnow()
                data = getattr(payload, "data", None) or {}
                if self.persist_raw and data:
                    self._snapshot(group, tf, cursor, stop, data)
                for sym in group:
                    for row in data.get(sym, ()):
                        bar = self._to_bar(sym, tf, step, daily, row)
                        if bar is not None:
                            yield bar
                if stop >= hi:
                    break
                cursor = stop + timedelta(microseconds=1)

    def _to_bar(self, sym: str, tf: str, step: timedelta, daily: bool, row: Any) -> Bar | None:
        stamp = getattr(row, "timestamp", None)
        close = getattr(row, "close", None)
        if stamp is None or close is None:
            return None
        event_time = to_utc(stamp)
        day = session_date(event_time)
        if daily:
            # Alpaca stamps daily bars at midnight ET; pin them to the session
            # open so daily bars from every source line up in the store.
            event_time = session_open_utc(day)
            knowledge_time = session_close_utc(day)
        else:
            knowledge_time = min(event_time + step, session_close_utc(day))
            if knowledge_time < event_time:
                knowledge_time = event_time + step
        o = float(getattr(row, "open", close))
        h = float(getattr(row, "high", close))
        l = float(getattr(row, "low", close))
        c = float(close)
        vwap = getattr(row, "vwap", None)
        trade_count = getattr(row, "trade_count", None)
        return Bar(
            event_time=event_time,
            knowledge_time=knowledge_time,
            source=self.source,
            ticker=sym,
            timeframe=tf,
            open=o,
            high=max(h, o, c),
            low=min(l, o, c),
            close=c,
            volume=float(getattr(row, "volume", 0.0) or 0.0),
            vwap=None if vwap is None else float(vwap),
            trade_count=None if trade_count is None else float(trade_count),
            adjusted=self.adjustment != "raw",
        )

    def _snapshot(
        self,
        group: Sequence[str],
        tf: str,
        lo: datetime,
        hi: datetime,
        data: dict[str, Any],
    ) -> None:
        from lab.store import raw  # noqa: PLC0415

        raw.save(
            self.source,
            "stock_bars",
            {sym: [_jsonable(r) for r in rows] for sym, rows in data.items()},
            params={
                "tickers": list(group),
                "timeframe": tf,
                "start": lo.isoformat(),
                "end": hi.isoformat(),
                "feed": self.feed,
                "adjustment": self.adjustment,
            },
        )
