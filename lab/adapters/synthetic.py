"""Deterministic synthetic market data: bars and lagged alt-data signals.

This adapter exists so the entire platform can be exercised end to end with no
network and no credentials -- the test suite, CI, and the offline demo all run
on it. That makes *reproducibility*, not realism, the binding requirement.

Three properties are load-bearing:

1. **Bit-identical output.** The same ``(seed, ticker, start, end, timeframe)``
   produces the same rows in any process on any day. All randomness is seeded
   from a BLAKE2b digest of those inputs; Python's built-in ``hash()`` is salted
   per process and must never appear here.
2. **Per-ticker independence.** Each ticker draws from its own stream keyed on
   ``(seed, ticker)``, so adding a symbol to a universe does not perturb the
   history of the ones already there -- otherwise every stored partition would
   silently restate whenever the universe grew.
3. **Window independence.** Paths are generated forward from a fixed
   :data:`ANCHOR` date, and each trading day consumes a fixed-size block of
   draws, so extending the horizon appends rows rather than renumbering them.
   Re-pulling an overlapping range restates identically instead of conflicting.

The price process is geometric Brownian motion around a per-ticker log-drift
line, with a very weak Ornstein-Uhlenbeck pull (``anchor_pull``, half-life on
the order of a decade) toward that line. Pure GBM run from 1990 wanders far
enough to produce sub-penny or five-figure quotes; the pull bounds the excursion
without being fast enough for any realistic strategy horizon to trade against.
Set ``anchor_pull=0.0`` for textbook GBM.

Signals model the congressional-disclosure shape deliberately: ``event_time`` is
the notional trade date, ``knowledge_time`` is the disclosure, and the gap
between them runs out to ``disclosure_lag_days`` (45 by default, matching the
STOCK Act deadline). The signal's direction is correlated with the forward
return measured **from the trade date**, so a strategy that respects
``knowledge_time`` earns a small residual edge while one that peeks at
``event_time`` captures the whole pre-disclosure move. That asymmetry is what
makes the look-ahead tests mean something.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Iterator, Sequence

import numpy as np

from lab.adapters.base import BaseAdapter
from lab.store.schema import Bar, Event
from lab.timeutil import (
    is_intraday,
    is_trading_day,
    parse_timeframe,
    session_close_utc,
    session_open_utc,
    to_utc,
    trading_days,
    utcnow,
)

SOURCE = "synthetic"

#: First session the generator knows about. Requests before this raise, rather
#: than silently returning a shorter history than the caller asked for.
ANCHOR = date(1990, 1, 2)

TRADING_DAYS_PER_YEAR = 252

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "XOM", "JNJ", "PG", "UNH", "SPY",
)

SECTORS: tuple[str, ...] = (
    "Information Technology", "Health Care", "Financials",
    "Consumer Discretionary", "Communication Services", "Industrials",
    "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials",
)

SIGNAL_KINDS: tuple[str, ...] = ("congress_trade", "insider_trade", "committee_conflict")

#: Forward window used to colour a signal's direction, in trading days past the
#: disclosure. Long enough that the post-disclosure tail is tradeable, short
#: enough that the pre-disclosure move dominates it.
_POST_DISCLOSURE_BARS = 10


def _seed_of(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary parts.

    ``hash()`` is salted per interpreter (PYTHONHASHSEED), so it cannot appear
    anywhere in a path that must reproduce across processes.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _clean_tickers(tickers: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(tickers, str):
        tickers = [tickers]
    out = tuple(t.strip().upper() for t in tickers if str(t).strip())
    if not out:
        raise ValueError("no tickers given")
    return out


@dataclass(frozen=True, slots=True)
class _Profile:
    """Per-ticker character. Mild spread in drift and vol so cross-sectional
    strategies have something real to rank."""

    s0: float       # price at ANCHOR
    nu: float       # annual log drift, already net of the variance drag
    sigma: float    # annual log volatility
    base_volume: float
    gap: float      # overnight gap size, as a fraction of daily vol
    span: float     # intrabar range size, as a fraction of daily vol


@lru_cache(maxsize=4096)
def _profile(seed: int, ticker: str) -> _Profile:
    rng = np.random.default_rng(_seed_of(seed, ticker, "profile"))
    return _Profile(
        s0=float(np.exp(rng.uniform(np.log(8.0), np.log(120.0)))),
        nu=float(rng.uniform(0.01, 0.12)),
        sigma=float(rng.uniform(0.16, 0.45)),
        base_volume=float(np.exp(rng.uniform(np.log(1.5e5), np.log(2.5e7)))),
        gap=float(rng.uniform(0.15, 0.55)),
        span=float(rng.uniform(0.45, 1.15)),
    )


def sector_for(ticker: str, seed: int = 0) -> str:
    """A stable sector label, so the risk gate's sector caps can be demoed."""
    return SECTORS[_seed_of(seed, ticker.strip().upper(), "sector") % len(SECTORS)]


@lru_cache(maxsize=64)
def _calendar(last_year: int) -> tuple[date, ...]:
    return tuple(trading_days(ANCHOR, date(last_year, 12, 31)))


@dataclass(frozen=True, slots=True)
class _Path:
    days: tuple[date, ...]
    index: dict[date, int]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@lru_cache(maxsize=512)
def _daily_path(seed: int, ticker: str, last_year: int, pull: float) -> _Path:
    """The ticker's whole daily history from ANCHOR through ``last_year``.

    Cached per calendar year rather than per request so that overlapping pulls
    reuse one array; the year rounding is safe precisely because the draw
    stream is prefix-stable.
    """
    days = _calendar(last_year)
    n = len(days)
    p = _profile(seed, ticker)

    # One row of draws per day. Sizing the block per-day (rather than drawing
    # five separate length-n vectors) is what keeps the stream prefix-stable:
    # extending the horizon appends rows and leaves earlier ones untouched.
    draws = np.random.default_rng(_seed_of(seed, ticker, "daily")).standard_normal((n, 5))

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    sd = p.sigma * np.sqrt(dt)
    steps = np.arange(1, n + 1, dtype=np.float64)

    shocks = sd * draws[:, 0]
    decay = 1.0 - pull * dt
    if decay >= 1.0:
        deviation = np.cumsum(shocks)
    else:
        # AR(1) in closed form: d_i = sum_j decay^(i-j) e_j. decay^-n stays
        # near 3 for a 36-year path at the default pull, so no scaling trouble.
        weight = decay**steps
        deviation = weight * np.cumsum(shocks / weight)

    close = np.exp(np.log(p.s0) + p.nu * steps * dt + deviation)
    prev = np.concatenate(([p.s0], close[:-1]))
    open_ = prev * np.exp(p.gap * sd * draws[:, 1])

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(p.span * sd * np.abs(draws[:, 2]))
    low = body_lo * np.exp(-p.span * sd * np.abs(draws[:, 3]))

    # Volume co-moves with the size of the move, which is what makes any
    # volume-aware indicator behave rather than see white noise.
    raw_vol = p.base_volume * np.exp(0.42 * draws[:, 4] - 0.088) * (1.0 + 0.7 * np.abs(draws[:, 0]))
    volume = np.maximum(np.rint(raw_vol), 1.0)

    if not np.isfinite(close).all() or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError(f"synthetic path for {ticker} diverged; lower sigma or raise anchor_pull")

    return _Path(
        days=days,
        index={d: i for i, d in enumerate(days)},
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _session_slots(day: date, step: timedelta) -> list[tuple[datetime, datetime]]:
    """Intraday bar boundaries, clipped to the regular session.

    The final slot is truncated at the close rather than dropped, so a 1h
    request still ends the day at 16:00 ET the way a real vendor does.
    """
    opened, closed = session_open_utc(day), session_close_utc(day)
    slots: list[tuple[datetime, datetime]] = []
    ts = opened
    while ts < closed:
        slots.append((ts, min(ts + step, closed)))
        ts += step
    return slots


def _forward_return(path: _Path, day: date, bars: int) -> float:
    i = path.index.get(day)
    if i is None:
        return 0.0
    j = min(i + max(bars, 1), len(path.close) - 1)
    return float(path.close[j] / path.close[i] - 1.0)


class SyntheticAdapter(BaseAdapter):
    """Seeded market data. No network, no credentials, always available."""

    name = SOURCE
    provides = frozenset({"bars", "signals"})

    def __init__(
        self,
        *,
        seed: int = 7,
        tickers: Sequence[str] | None = None,
        disclosure_lag_days: int = 45,
        min_disclosure_lag_days: int = 2,
        signal_rate: float = 0.05,
        signal_edge: float = 0.15,
        anchor_pull: float = 0.06,
        source: str = SOURCE,
    ) -> None:
        if int(seed) != seed:
            raise ValueError("seed must be an integer")
        if min_disclosure_lag_days < 0:
            raise ValueError("min_disclosure_lag_days must be >= 0")
        if disclosure_lag_days < min_disclosure_lag_days:
            raise ValueError("disclosure_lag_days must be >= min_disclosure_lag_days")
        if not 0.0 <= signal_rate <= 1.0:
            raise ValueError("signal_rate must be in [0, 1]")
        if not 0.0 <= signal_edge <= 1.0:
            raise ValueError("signal_edge must be in [0, 1]")
        if anchor_pull < 0.0:
            raise ValueError("anchor_pull must be >= 0")

        self.seed = int(seed)
        self.tickers = _clean_tickers(tickers) if tickers else DEFAULT_UNIVERSE
        self.disclosure_lag_days = int(disclosure_lag_days)
        self.min_disclosure_lag_days = int(min_disclosure_lag_days)
        self.signal_rate = float(signal_rate)
        self.signal_edge = float(signal_edge)
        self.anchor_pull = float(anchor_pull)
        self.source = source

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self):
        got = super().info()
        got.detail = {
            "seed": self.seed,
            "anchor": ANCHOR.isoformat(),
            "disclosure_lag_days": self.disclosure_lag_days,
            "signal_rate": self.signal_rate,
            "signal_edge": self.signal_edge,
            "universe": list(self.tickers),
        }
        return got

    def sector(self, ticker: str) -> str:
        return sector_for(ticker, self.seed)

    # -- bars ----------------------------------------------------------------
    def fetch_bars(
        self,
        tickers: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        """Yield bars for each ticker, chronologically, ticker by ticker.

        ``start``/``end`` bound the session date inclusively. A bound given at
        midnight covers that whole session; one with a time of day is honoured
        to the second, so a live-style "last two hours" request works too.
        """
        syms = _clean_tickers(tickers)
        tf = str(timeframe).strip().lower()
        step = parse_timeframe(tf)
        lo, hi = to_utc(start), to_utc(end)
        if lo > hi:
            raise ValueError(f"start {lo.isoformat()} is after end {hi.isoformat()}")
        if lo.date() < ANCHOR:
            raise ValueError(f"synthetic history starts at {ANCHOR.isoformat()}, got {lo.date()}")

        daily = step == timedelta(days=1)
        if not daily and not is_intraday(tf):
            raise ValueError(f"synthetic supports '1d' and intraday timeframes, not {tf!r}")

        hi_bound = (
            hi
            if hi.time() != time(0, 0)
            else hi.replace(hour=23, minute=59, second=59, microsecond=999_999)
        )
        days = trading_days(lo.date(), hi.date())
        if not days:
            return
        last_year = max(days[-1].year, lo.year)

        for sym in syms:
            path = _daily_path(self.seed, sym, last_year, self.anchor_pull)
            for day in days:
                i = path.index[day]
                if daily:
                    yield from self._daily_bar(sym, tf, path, i, day, lo, hi_bound)
                else:
                    yield from self._intraday_bars(sym, tf, step, path, i, day, lo, hi_bound)

    def _daily_bar(
        self,
        sym: str,
        tf: str,
        path: _Path,
        i: int,
        day: date,
        lo: datetime,
        hi: datetime,
    ) -> Iterator[Bar]:
        event_time = session_open_utc(day)
        if not lo <= event_time <= hi:
            return
        o, h, l, c = float(path.open[i]), float(path.high[i]), float(path.low[i]), float(path.close[i])
        v = float(path.volume[i])
        yield Bar(
            event_time=event_time,
            # A daily bar is only knowable once the session has closed.
            knowledge_time=session_close_utc(day),
            source=self.source,
            ticker=sym,
            timeframe=tf,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            vwap=(h + l + c) / 3.0,
            trade_count=max(1.0, round(v / 220.0)),
        )

    def _intraday_bars(
        self,
        sym: str,
        tf: str,
        step: timedelta,
        path: _Path,
        i: int,
        day: date,
        lo: datetime,
        hi: datetime,
    ) -> Iterator[Bar]:
        """Bridge the day's open to its close so intraday and daily agree.

        Each session gets its own counter-seeded stream keyed on the date, which
        is what keeps intraday output window-independent without generating
        every minute back to 1990. The bridge pins the last bar's close to the
        daily close exactly; intrabar extremes are drawn independently, so they
        need not coincide with the daily bar's own high and low.
        """
        slots = _session_slots(day, step)
        n = len(slots)
        if n == 0:
            return
        p = _profile(self.seed, sym)
        draws = np.random.default_rng(
            _seed_of(self.seed, sym, tf, day.toordinal())
        ).standard_normal((n, 4))

        sd_bar = p.sigma * np.sqrt(1.0 / TRADING_DAYS_PER_YEAR) / np.sqrt(n)
        walk = np.cumsum(draws[:, 0]) * sd_bar
        frac = np.arange(1, n + 1, dtype=np.float64) / n
        bridge = walk - frac * walk[-1]

        x0, xn = np.log(float(path.open[i])), np.log(float(path.close[i]))
        levels = np.exp(x0 + frac * (xn - x0) + bridge)
        opens = np.concatenate(([np.exp(x0)], levels[:-1]))

        body_hi = np.maximum(opens, levels)
        body_lo = np.minimum(opens, levels)
        highs = body_hi * np.exp(0.5 * p.span * sd_bar * np.abs(draws[:, 1]))
        lows = body_lo * np.exp(-0.5 * p.span * sd_bar * np.abs(draws[:, 2]))

        # Classic U-shaped session volume: heavy at the open and the close.
        shape = 1.0 + 3.0 * (2.0 * (frac - 0.5 / n) - 1.0) ** 2
        shape = shape / shape.sum()
        vols = np.maximum(
            np.rint(float(path.volume[i]) * shape * np.exp(0.3 * draws[:, 3] - 0.045)), 1.0
        )

        for k, (opened, closed) in enumerate(slots):
            if not lo <= opened <= hi:
                continue
            o, h, l, c = float(opens[k]), float(highs[k]), float(lows[k]), float(levels[k])
            v = float(vols[k])
            yield Bar(
                event_time=opened,
                knowledge_time=closed,
                source=self.source,
                ticker=sym,
                timeframe=tf,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                vwap=(h + l + c) / 3.0,
                trade_count=max(1.0, round(v / 180.0)),
            )

    # -- signals -------------------------------------------------------------
    def fetch_signals(
        self, start: datetime | None = None, end: datetime | None = None, **query: Any
    ) -> Iterator[Event]:
        """Yield alt-data events that became *knowable* inside the window.

        ``start``/``end`` bound ``knowledge_time``, not ``event_time`` -- the
        window means "what could I have learned here", which is the only useful
        question for a source whose facts surface weeks after they happen. The
        scan therefore reaches back an extra ``disclosure_lag_days`` to catch
        trades whose disclosure lands inside the window.

        Recognised ``query`` keys: ``tickers``, ``kind``/``kinds``,
        ``min_score``, ``limit``.
        """
        hi = to_utc(end) if end is not None else utcnow()
        lo = to_utc(start) if start is not None else hi - timedelta(days=365)
        if lo > hi:
            raise ValueError(f"start {lo.isoformat()} is after end {hi.isoformat()}")

        syms = _clean_tickers(query.get("tickers") or self.tickers)
        wanted = query.get("kinds") or ([query["kind"]] if query.get("kind") else None)
        kinds = {str(k) for k in wanted} if wanted else None
        min_score = float(query.get("min_score", 0.0))
        limit = query.get("limit")
        limit = int(limit) if limit is not None else None

        scan_lo = max(ANCHOR, (lo - timedelta(days=self.disclosure_lag_days + 10)).date())
        scan_hi = hi.date()
        if scan_lo > scan_hi:
            return
        days = trading_days(scan_lo, scan_hi)
        if not days:
            return
        # Directions are coloured by a forward return that runs past the
        # disclosure, so the path must extend beyond the requested window.
        last_year = scan_hi.year + 1

        out: list[Event] = []
        for sym in syms:
            path = _daily_path(self.seed, sym, last_year, self.anchor_pull)
            for day in days:
                ev = self._signal_for(sym, path, day, lo, hi)
                if ev is None:
                    continue
                if kinds is not None and ev.kind not in kinds:
                    continue
                if ev.score is not None and ev.score < min_score:
                    continue
                out.append(ev)

        out.sort(key=lambda e: (e.knowledge_time, e.ticker, e.uid))
        if limit is not None:
            out = out[:limit]
        yield from out

    def _signal_for(
        self, sym: str, path: _Path, day: date, lo: datetime, hi: datetime
    ) -> Event | None:
        # One independent stream per (ticker, day): whether an event fires here
        # cannot depend on what happened on any other day or ticker, which is
        # what makes the scan window irrelevant to the output.
        rng = np.random.default_rng(_seed_of(self.seed, sym, "signal", day.toordinal()))
        if float(rng.random()) >= self.signal_rate:
            return None

        span = self.disclosure_lag_days - self.min_disclosure_lag_days
        # Skewed toward the deadline: real filings cluster at the last legal day.
        lag = self.min_disclosure_lag_days + span * float(rng.random()) ** 0.35
        disclosure = day + timedelta(days=max(1, int(round(lag))))
        while not is_trading_day(disclosure):
            disclosure += timedelta(days=1)

        knowledge_time = session_close_utc(disclosure)
        if not lo <= knowledge_time <= hi:
            return None

        kind = SIGNAL_KINDS[int(rng.integers(0, len(SIGNAL_KINDS)))]
        i, j = path.index.get(day), path.index.get(disclosure)
        horizon = (j - i if i is not None and j is not None else 0) + _POST_DISCLOSURE_BARS
        forward = _forward_return(path, day, horizon)
        direction = "BUY" if forward >= 0 else "SELL"
        if float(rng.random()) > 0.5 + self.signal_edge / 2.0:
            direction = "SELL" if direction == "BUY" else "BUY"

        score = float(rng.uniform(0.30, 1.0))
        tier = "A" if score >= 0.80 else "B" if score >= 0.60 else "C"
        notional = int(round(float(np.exp(rng.uniform(np.log(1e3), np.log(2.5e6)))), -3))
        actor = f"actor-{_seed_of(self.seed, sym, day.toordinal(), 'actor') % 0xFFFFFF:06x}"

        return Event(
            event_time=session_close_utc(day),
            knowledge_time=knowledge_time,
            source=self.source,
            ticker=sym,
            kind=kind,
            uid=f"{self.source}:{kind}:{sym}:{day.isoformat()}",
            direction=direction,
            tier=tier,
            score=round(score, 6),
            sector=self.sector(sym),
            fresh=(disclosure - day).days <= 14,
            payload={
                "lag_days": (disclosure - day).days,
                "traded_on": day.isoformat(),
                "disclosed_on": disclosure.isoformat(),
                "notional_usd": notional,
                "actor": actor,
            },
        )
