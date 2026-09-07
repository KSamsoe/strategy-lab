"""The context: the only door between a strategy and the data.

Two jobs, both load-bearing.

**Refuse the future.** Every read filters on ``knowledge_time <= now``. Not by
convention -- structurally, in one place, with no way around it, because a
strategy cannot reach the store directly. Look-ahead stops being a discipline
the author must sustain over hundreds of edits and becomes a property of the
plumbing. In ``strict`` mode an out-of-range request raises ``LookAheadError``
rather than silently clamping, since a loud crash beats a plausibly-wrong curve.

**Record what it served.** Because every value a strategy sees passes through
here, capturing the *input tape* is a wrapper rather than a change to strategy
code. That tape is what makes "why does this position exist" a lookup instead of
an investigation.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lab.engine.events import Intent
from lab.engine.protocols import LookAheadError, PortfolioView
from lab.store.schema import Event
from lab.timeutil import ET, REGULAR_CLOSE, REGULAR_OPEN, is_intraday, session_date, to_utc

#: How much of a history slice the input tape keeps. Full slices would bloat the
#: journal by orders of magnitude; the summary plus the tail is what a human or
#: an agent actually reads when reconstructing a decision.
TAPE_TAIL = 5


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 8)
    if isinstance(v, (pd.Timestamp, datetime)):
        return to_utc(v).isoformat()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _summarize_series(s: pd.Series) -> dict[str, Any]:
    tail = s.dropna().tail(TAPE_TAIL)
    return {
        "n": int(len(s)),
        "first": _jsonable(s.index[0]) if len(s) else None,
        "last": _jsonable(s.index[-1]) if len(s) else None,
        "tail": [
            {"t": _jsonable(idx), "v": _jsonable(val)} for idx, val in tail.items()
        ],
    }


def regular_hours_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop pre-market and after-hours bars from an intraday frame.

    Alpaca returns extended-hours bars by default, and they are a trap for a
    backtest. They are thin -- a few hundred shares against tens of thousands in
    the regular session -- so a fill model prices them as if they were liquid,
    and there are more of them per day than the annualization factor assumes,
    which quietly inflates every Sharpe. The live runner already fires only
    inside regular hours, so keeping them would also make the backtest and the
    live path disagree about what a bar even is.

    Filtered on read rather than on ingest: the store keeps what the vendor
    sent, and a strategy that genuinely wants the overnight session can ask.
    """
    if frame is None or frame.empty:
        return frame
    index = pd.DatetimeIndex(frame.index)
    et = index.tz_convert(ET)
    minutes = et.hour * 60 + et.minute
    open_m = REGULAR_OPEN.hour * 60 + REGULAR_OPEN.minute
    close_m = REGULAR_CLOSE.hour * 60 + REGULAR_CLOSE.minute
    # Bars are stamped at their close, so the first regular bar closes *after*
    # the open and the last closes exactly at 16:00.
    keep = (minutes > open_m) & (minutes <= close_m)
    return frame[keep]


def available_bar_sources(tickers: Sequence[str], timeframe: str = "1d") -> list[str]:
    """Which adapters have bars for these tickers at this cadence."""
    from lab.store import parquet_io

    try:
        cov = parquet_io.coverage()
    except Exception:
        return []
    if cov is None or len(cov) == 0:
        return []
    wanted = {t.upper() for t in tickers}
    sub = cov[cov["ticker"].isin(wanted)]
    if "timeframe" in sub.columns:
        sub = sub[sub["timeframe"] == timeframe]
    return sorted({str(x) for x in sub["source"].unique()})


def require_one_bar_source(tickers: Sequence[str], timeframe: str = "1d") -> str | None:
    """Resolve the bar source, or refuse to guess.

    The store deliberately keeps one row per (source, ticker, timeframe,
    event_time) so a yfinance bootstrap and a real Alpaca pull can coexist. That
    is correct for storage and catastrophic for a *reader* that ignores it: two
    sources for one ticker interleave into a price series that jumps between two
    unrelated universes, and a backtest over it produces confident numbers about
    nothing at all.

    So an ambiguous store is an error, not a default. It fires only in the
    situation that would otherwise be silently wrong -- one source, or none, and
    nothing happens.
    """
    found = available_bar_sources(tickers, timeframe)
    if len(found) <= 1:
        return found[0] if found else None
    raise ValueError(
        "the store holds bars for these tickers from more than one source "
        f"({', '.join(found)}), and mixing them would interleave unrelated price "
        "series into one nonsensical curve. Name the one you mean with `source:` "
        "in the backtest config (e.g. `source: " + found[0] + "`), or drop the "
        "partitions you do not want from data/parquet/bars/."
    )


class DataView:
    """Bar and signal access for one run, bound by ``knowledge_time``.

    Backtests preload frames into memory (the whole point of the coarse
    daily/minute cadence is that this fits comfortably); live mode delegates to
    the store. Either way the filtering rule is identical, which is what lets
    one runner serve both.
    """

    def __init__(
        self,
        bars: Mapping[str, pd.DataFrame],
        signals: pd.DataFrame | None = None,
        timeframe: str = "1d",
    ) -> None:
        self.timeframe = timeframe
        self._bars: dict[str, pd.DataFrame] = {}
        #: Per-ticker knowledge_time as int64 nanoseconds, plus whether it is
        #: non-decreasing. When it is -- which is the normal case, since a bar
        #: becomes knowable when it closes -- the visible slice is a prefix and
        #: searchsorted finds it in log time instead of masking the whole frame
        #: on every single read. At 2k bars x 20 tickers x a few reads per bar
        #: that is the difference between a snappy backtest and a coffee break.
        self._kt: dict[str, np.ndarray] = {}
        self._kt_sorted: dict[str, bool] = {}
        for ticker, df in bars.items():
            key = ticker.upper()
            prepared = self._prepare_bars(df)
            self._bars[key] = prepared
            kt = prepared["knowledge_time"].to_numpy(dtype="datetime64[ns]", copy=False) \
                if len(prepared) else np.array([], dtype="datetime64[ns]")
            self._kt[key] = kt
            self._kt_sorted[key] = bool(len(kt) <= 1 or (np.diff(kt) >= np.timedelta64(0)).all())
        self._signals = self._prepare_signals(signals)
        self._sig_kt = (
            self._signals["knowledge_time"].to_numpy(dtype="datetime64[ns]", copy=False)
            if len(self._signals)
            else np.array([], dtype="datetime64[ns]")
        )

    # --- preparation -------------------------------------------------------

    @staticmethod
    def _prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "knowledge_time"]
            )
        out = df.copy()
        if "event_time" in out.columns:
            out = out.set_index("event_time")
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
        if "knowledge_time" not in out.columns:
            # No stated knowledge time means the bar was knowable when it printed.
            out["knowledge_time"] = out.index
        else:
            out["knowledge_time"] = pd.to_datetime(out["knowledge_time"], utc=True)
        return out.sort_index()

    @staticmethod
    def _prepare_signals(df: pd.DataFrame | None) -> pd.DataFrame:
        cols = [
            "event_time", "knowledge_time", "source", "ticker", "kind",
            "direction", "tier", "score", "sector", "fresh", "uid", "payload",
        ]
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols)
        out = df.copy()
        for c in ("event_time", "knowledge_time"):
            out[c] = pd.to_datetime(out[c], utc=True)
        return out.sort_values("knowledge_time").reset_index(drop=True)

    # --- reads, all gated on `upto` ---------------------------------------

    def visible_len(self, ticker: str, upto: datetime) -> int:
        """Length of the knowable *prefix* of ``ticker`` at ``upto``.

        A prefix, not a count. Callers slice with it -- ``ctx.indicator``
        computes over the whole series once and cuts here -- so with
        out-of-order knowledge times it has to stop at the first unknowable
        row. Counting knowable rows instead would slide a restated future bar
        inside the window and feed it to every indicator, which is precisely
        the look-ahead this class exists to make impossible.
        """
        key = ticker.upper()
        df = self._bars.get(key)
        if df is None or df.empty:
            return 0
        cutoff = np.datetime64(to_utc(upto).replace(tzinfo=None), "ns")
        if self._kt_sorted.get(key, False):
            return int(np.searchsorted(self._kt[key], cutoff, side="right"))
        unknowable = np.flatnonzero(self._kt[key] > cutoff)
        return int(unknowable[0]) if unknowable.size else len(self._kt[key])

    def _visible(self, ticker: str, upto: datetime) -> pd.DataFrame:
        key = ticker.upper()
        df = self._bars.get(key)
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        if self._kt_sorted.get(key, False):
            return df.iloc[: self.visible_len(key, upto)]
        # Out-of-order knowledge times (a restated bar) fall back to a mask;
        # correctness first, and this path is rare enough not to matter.
        return df[df["knowledge_time"] <= pd.Timestamp(to_utc(upto))]

    def bars(self, ticker: str, n: int, upto: datetime) -> pd.DataFrame:
        vis = self._visible(ticker, upto)
        return vis.tail(n) if n and n > 0 else vis

    def history(self, ticker: str, field: str, n: int, upto: datetime) -> pd.Series:
        vis = self._visible(ticker, upto)
        if vis.empty or field not in vis.columns:
            return pd.Series(dtype="float64", name=field)
        s = vis[field]
        return (s.tail(n) if n and n > 0 else s).rename(field)

    def price(self, ticker: str, upto: datetime) -> float | None:
        vis = self._visible(ticker, upto)
        if vis.empty:
            return None
        value = vis["close"].iloc[-1]
        return None if pd.isna(value) else float(value)

    def bar_at(self, ticker: str, ts: datetime) -> dict[str, Any] | None:
        """The bar stamped exactly ``ts``, ignoring the knowledge filter.

        Used by the fill engine, never by a strategy: filling an order needs the
        bar the market actually printed, not what the strategy was allowed to
        see when it decided.
        """
        df = self._bars.get(ticker.upper())
        if df is None or df.empty:
            return None
        key = pd.Timestamp(to_utc(ts))
        if key not in df.index:
            return None
        row = df.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row.to_dict()

    def signals(self, source: str | None, upto: datetime, **query: Any) -> list[Event]:
        df = self._signals
        if df is None or len(df) == 0:
            return []
        # Signals are stored sorted by knowledge_time, so the knowable set is a
        # prefix; trim it first and filter the (usually much smaller) remainder.
        cutoff = np.datetime64(to_utc(upto).replace(tzinfo=None), "ns")
        k = int(np.searchsorted(self._sig_kt, cutoff, side="right"))
        if k == 0:
            return []
        df = df.iloc[:k]
        mask = pd.Series(True, index=df.index)
        if source:
            mask &= df["source"] == source
        for key in ("ticker", "kind", "tier", "direction", "sector"):
            val = query.pop(key, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                vals = {str(v).upper() if key == "ticker" else str(v) for v in val}
                mask &= df[key].isin(vals)
            else:
                val = str(val).upper() if key == "ticker" else val
                mask &= df[key] == val
        min_score = query.pop("min_score", None)
        if min_score is not None:
            mask &= df["score"].fillna(-np.inf) >= float(min_score)
        fresh = query.pop("fresh", None)
        if fresh is not None:
            mask &= df["fresh"] == bool(fresh)
        since = query.pop("since", None)
        if since is not None:
            mask &= df["knowledge_time"] >= pd.Timestamp(to_utc(since))
        lookback_days = query.pop("lookback_days", None)
        if lookback_days is not None:
            floor = pd.Timestamp(to_utc(upto)) - pd.Timedelta(days=float(lookback_days))
            mask &= df["knowledge_time"] >= floor

        sub = df[mask]
        limit = query.pop("limit", None)
        if limit:
            sub = sub.tail(int(limit))
        return [self._to_event(r) for _, r in sub.iterrows()]

    @staticmethod
    def _to_event(row: Mapping[str, Any]) -> Event:
        import json

        payload = row.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                payload = {"_raw": payload}
        return Event(
            event_time=to_utc(row["event_time"]),
            knowledge_time=to_utc(row["knowledge_time"]),
            source=str(row.get("source") or ""),
            ticker=str(row.get("ticker") or ""),
            kind=str(row.get("kind") or ""),
            uid=str(row.get("uid") or ""),
            direction=row.get("direction") or None,
            tier=row.get("tier") or None,
            score=None if pd.isna(row.get("score")) else float(row.get("score")),
            sector=row.get("sector") or None,
            fresh=None if row.get("fresh") is None or pd.isna(row.get("fresh")) else bool(row.get("fresh")),
            request_id=row.get("request_id") or None,
            payload=payload or {},
        )

    # --- shape -------------------------------------------------------------

    def timestamps(self) -> list[datetime]:
        """The union of all bar timestamps: the backtest's clock ticks."""
        stamps: set[pd.Timestamp] = set()
        for df in self._bars.values():
            stamps.update(df.index)
        return [ts.to_pydatetime() for ts in sorted(stamps)]

    def tickers(self) -> list[str]:
        return sorted(self._bars)

    def frame(self, ticker: str) -> pd.DataFrame:
        return self._bars.get(ticker.upper(), pd.DataFrame())

    def bars_at(self, ts: datetime) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ticker in self._bars:
            bar = self.bar_at(ticker, ts)
            if bar is not None:
                out[ticker] = bar
        return out

    def closes_at(self, ts: datetime) -> dict[str, float]:
        out: dict[str, float] = {}
        for ticker, bar in self.bars_at(ts).items():
            close = bar.get("close")
            if close is not None and not pd.isna(close):
                out[ticker] = float(close)
        return out

    @classmethod
    def from_store(
        cls,
        tickers: Sequence[str],
        timeframe: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        sources: Sequence[str] | None = None,
        as_of: datetime | None = None,
        bar_source: str | None = None,
        regular_hours: bool = True,
    ) -> "DataView":
        """Build a view over the store.

        ``bar_source`` picks which adapter's bars to trade. It is required
        whenever the store holds bars for the same tickers from more than one
        source -- see :func:`require_one_bar_source`.
        """
        from lab.store import parquet_io

        if bar_source is None:
            bar_source = require_one_bar_source(list(tickers), timeframe)
        bars_df = parquet_io.read_bars(
            list(tickers), timeframe=timeframe, start=start, end=end, as_of=as_of,
            source=bar_source,
        )
        intraday = is_intraday(timeframe)
        by_ticker: dict[str, pd.DataFrame] = {}
        if len(bars_df):
            for ticker, grp in bars_df.groupby("ticker"):
                frame = grp.set_index("event_time")
                if intraday and regular_hours:
                    frame = regular_hours_only(frame)
                by_ticker[str(ticker).upper()] = frame
        for t in tickers:
            by_ticker.setdefault(t.upper(), pd.DataFrame())

        signals_df = None
        for source in sources or []:
            part = parquet_io.read_signals(source, start=start, end=end, as_of=as_of)
            if len(part):
                signals_df = part if signals_df is None else pd.concat([signals_df, part])
        return cls(by_ticker, signals_df, timeframe=timeframe)


class EngineContext:
    """The ``Context`` implementation handed to a strategy each bar."""

    def __init__(
        self,
        *,
        run_id: str,
        strategy: str,
        params: Mapping[str, Any],
        universe: Sequence[str],
        portfolio: PortfolioView,
        data: DataView,
        timeframe: str = "1d",
        strict: bool = True,
        capture: bool = True,
        indicator_data_version: str = "",
    ) -> None:
        self.run_id = run_id
        self.strategy = strategy
        self.params = dict(params)
        self.universe = [t.upper() for t in universe]
        self.portfolio = portfolio
        self.data = data
        self.timeframe = timeframe
        self.strict = strict
        self.capture = capture
        self._data_version = indicator_data_version
        self._now: datetime | None = None
        self._intents: list[Intent] = []
        self._tape: dict[str, Any] = {}
        self._logs: list[dict[str, Any]] = []
        self._indicators: dict[tuple, Any] = {}
        self._causality_checked: set[tuple] = set()

    # --- clock -------------------------------------------------------------

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise RuntimeError("context time not set; the runner must call set_now()")
        return self._now

    @property
    def session(self) -> date:
        return session_date(self.now)

    def set_now(self, ts: datetime) -> None:
        self._now = to_utc(ts)

    def reset_bar(self) -> None:
        self._intents = []
        self._tape = {}
        self._logs = []

    def _guard(self, when: datetime | None) -> None:
        if when is None or self._now is None:
            return
        if to_utc(when) > self._now:
            msg = (
                f"{self.strategy} asked for data at {to_utc(when).isoformat()} "
                f"but ctx.now is {self._now.isoformat()}"
            )
            if self.strict:
                raise LookAheadError(msg)
            self._log_internal(level="warning", event="look_ahead_suppressed", detail=msg)

    # --- reads -------------------------------------------------------------

    def history(
        self, ticker: str, field: str = "close", n: int = 100, timeframe: str | None = None
    ) -> pd.Series:
        if timeframe and timeframe != self.timeframe:
            raise ValueError(
                f"context is bound to timeframe {self.timeframe!r}; "
                f"cross-timeframe reads are not supported in v1"
            )
        s = self.data.history(ticker, field, n, self.now)
        if self.capture:
            self._tape.setdefault("history", {})[f"{ticker.upper()}.{field}"] = _summarize_series(s)
        return s

    def bars(self, ticker: str, n: int = 100, timeframe: str | None = None) -> pd.DataFrame:
        if timeframe and timeframe != self.timeframe:
            raise ValueError(f"context is bound to timeframe {self.timeframe!r}")
        df = self.data.bars(ticker, n, self.now)
        if self.capture:
            self._tape.setdefault("bars", {})[ticker.upper()] = {
                "n": int(len(df)),
                "last": _jsonable(df.index[-1]) if len(df) else None,
                "last_close": _jsonable(df["close"].iloc[-1]) if len(df) and "close" in df else None,
            }
        return df

    def price(self, ticker: str) -> float | None:
        p = self.data.price(ticker, self.now)
        if self.capture:
            self._tape.setdefault("prices", {})[ticker.upper()] = _jsonable(p)
        return p

    def indicator(self, ticker: str, name: str, **params: Any) -> pd.Series:
        """Computed or fetched indicator, truncated to the visible window.

        Computed **once over the whole series** and then sliced at ``now``,
        rather than recomputed from scratch on every bar. That is sound only
        because every registered indicator is causal -- value at row *i* depends
        on no row after *i* -- which the indicator suite proves mechanically for
        the entire registry. Since it *looks* like a look-ahead footgun, strict
        mode also verifies it per (ticker, indicator, params) on first use by
        recomputing over the truncated prefix and comparing. One check per key,
        not per bar, so the guarantee costs nothing at scale.

        Recomputing rather than maintaining streaming state is itself a
        deliberate choice: mathematically identical at daily-to-minute cadence,
        far simpler, and it means the live path and the backtest path run
        literally the same code.
        """
        from lab.indicators import computed

        params.pop("lookback", None)  # a hint for the old per-bar path; harmless now
        column = params.pop("column", None)

        key = (ticker.upper(), name, tuple(sorted(params.items(), key=lambda kv: kv[0])))
        full = self._indicators.get(key)
        if full is None:
            frame = self.data.frame(ticker)
            if frame is None or frame.empty:
                return pd.Series(dtype="float64", name=name)
            full = computed.compute(name, frame, **params)
            self._indicators[key] = full
            if self.strict:
                self._assert_causal(ticker, name, params, full, frame)

        result = full
        if isinstance(result, pd.DataFrame):
            result = result[column] if column and column in result else result.iloc[:, 0]
        series = result.iloc[: self.data.visible_len(ticker, self.now)].rename(name)

        if self.capture:
            last = series.dropna()
            self._tape.setdefault("indicators", {})[f"{ticker.upper()}.{name}"] = {
                "params": {k: _jsonable(v) for k, v in params.items()},
                "value": _jsonable(last.iloc[-1]) if len(last) else None,
                "n": int(len(series)),
            }
        return series

    def _assert_causal(
        self, ticker: str, name: str, params: Mapping[str, Any], full: Any, frame: pd.DataFrame
    ) -> None:
        """Prove, once per key, that slicing the full series is safe.

        Recompute the indicator over only the rows visible right now and compare
        the last value against the same position in the full-series result. A
        non-causal indicator -- anything that peeks forward or back-fills its
        warm-up -- disagrees here and gets a ``LookAheadError`` instead of a
        silently optimistic backtest.
        """
        from lab.indicators import computed

        key = (ticker.upper(), name, tuple(sorted(params.items(), key=lambda kv: kv[0])))
        if key in self._causality_checked:
            return
        self._causality_checked.add(key)

        k = self.data.visible_len(ticker, self.now)
        if k < 2 or k >= len(frame):
            return  # nothing hidden yet, so nothing to prove

        try:
            truncated = computed.compute(name, frame.iloc[:k], **params)
        except Exception:
            return  # too little data for a shorter window; not evidence of peeking

        def _tail(obj: Any) -> Any:
            if isinstance(obj, pd.DataFrame):
                obj = obj.iloc[:, 0]
            return obj

        a = _tail(full).iloc[k - 1]
        b = _tail(truncated).iloc[-1]
        if pd.isna(a) and pd.isna(b):
            return
        if pd.isna(a) != pd.isna(b) or not math.isclose(
            float(a), float(b), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise LookAheadError(
                f"indicator {name!r} on {ticker.upper()} is not causal: over the full "
                f"series row {k - 1} is {a!r}, but computed over only the first {k} "
                f"rows it is {b!r}. It reads data from the future."
            )

    def signals(self, source: str, **query: Any) -> list[Event]:
        events = self.data.signals(source, self.now, **query)
        if self.capture:
            captured = self._tape.setdefault("signals", [])
            for e in events:
                captured.append(
                    {
                        "source": e.source,
                        "ticker": e.ticker,
                        "kind": e.kind,
                        "tier": e.tier,
                        "direction": e.direction,
                        "score": _jsonable(e.score),
                        "fresh": e.fresh,
                        "event_time": _jsonable(e.event_time),
                        "knowledge_time": _jsonable(e.knowledge_time),
                        "uid": e.uid,
                    }
                )
        return events

    # --- writes ------------------------------------------------------------

    def order_target_pct(
        self, ticker: str, pct: float, tag: str = "", reason: str = "", **meta: Any
    ) -> None:
        t = ticker.upper()
        if self.universe and t not in self.universe:
            raise ValueError(f"{t} is not in this run's universe")
        if not math.isfinite(float(pct)):
            raise ValueError(f"target_pct for {t} must be finite, got {pct!r}")
        # Last intent per ticker wins, so a strategy can revise within a bar.
        self._intents = [i for i in self._intents if i.ticker != t]
        self._intents.append(
            Intent(ticker=t, target_pct=float(pct), tag=tag, reason=reason, meta=dict(meta))
        )

    def close(self, ticker: str, tag: str = "exit", reason: str = "") -> None:
        self.order_target_pct(ticker, 0.0, tag=tag, reason=reason)

    def log(self, **fields: Any) -> None:
        self._logs.append(
            {"at": self._now.isoformat() if self._now else None}
            | {k: _jsonable(v) for k, v in fields.items()}
        )

    def _log_internal(self, **fields: Any) -> None:
        self._logs.append({"_engine": True} | {k: _jsonable(v) for k, v in fields.items()})

    # --- runner-facing -----------------------------------------------------

    def drain_intents(self) -> list[Intent]:
        out, self._intents = self._intents, []
        return out

    def input_tape(self) -> dict[str, Any]:
        return dict(self._tape)

    def logs(self) -> list[dict[str, Any]]:
        return list(self._logs)
