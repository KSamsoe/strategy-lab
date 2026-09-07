"""Two clocks, one engine.

The backtester and the live runner differ in exactly two places: which clock
drives the loop and which broker receives the orders. Everything between --
context, strategy, gate, portfolio accounting -- is the same code. Keeping the
clocks behind one tiny protocol is what makes that true rather than aspirational.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta
from typing import Iterator, Sequence

from lab.timeutil import (
    ET,
    UTC,
    is_trading_day,
    next_trading_day,
    parse_timeframe,
    session_close_utc,
    session_open_utc,
    to_et,
    to_utc,
    utcnow,
)


class SimClock:
    """Walks a precomputed list of bar timestamps as fast as the CPU allows."""

    def __init__(self, timestamps: Sequence[datetime]) -> None:
        self._ts = [to_utc(t) for t in timestamps]
        if any(b < a for a, b in zip(self._ts, self._ts[1:])):
            raise ValueError("SimClock timestamps must be non-decreasing")
        self._i = -1

    @property
    def now(self) -> datetime:
        if self._i < 0:
            return self._ts[0] if self._ts else utcnow()
        return self._ts[min(self._i, len(self._ts) - 1)]

    @property
    def index(self) -> int:
        return self._i

    def __len__(self) -> int:
        return len(self._ts)

    def __iter__(self) -> Iterator[datetime]:
        for i, ts in enumerate(self._ts):
            self._i = i
            yield ts

    def advance(self) -> datetime | None:
        if self._i + 1 >= len(self._ts):
            return None
        self._i += 1
        return self._ts[self._i]

    def peek(self, offset: int = 1) -> datetime | None:
        j = self._i + offset
        return self._ts[j] if 0 <= j < len(self._ts) else None


class WallClock:
    """Fires once per real bar, on the trading calendar.

    Daily strategies fire at ``at_time`` Eastern on each session; intraday ones
    fire at each bar boundary inside regular hours. Iteration blocks until the
    next fire time, and is interruptible via :meth:`stop` so a runner can shut
    down cleanly instead of being killed mid-sleep.
    """

    def __init__(
        self,
        timeframe: str = "1d",
        at_time: time | str | None = None,
        *,
        calendar: bool = True,
        max_fires: int | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.interval = parse_timeframe(timeframe)
        self.calendar = calendar
        self.max_fires = max_fires
        self._stop = threading.Event()
        self._now: datetime | None = None
        self._fires = 0

        if isinstance(at_time, str):
            hh, _, mm = at_time.partition(":")
            at_time = time(int(hh), int(mm or 0))
        # Daily strategies decide shortly after the open by default: late enough
        # that the opening auction has printed, early enough to act on it.
        self.at_time: time = at_time or time(9, 35)

    @property
    def now(self) -> datetime:
        return self._now or utcnow()

    def stop(self) -> None:
        self._stop.set()

    def next_fire(self, after: datetime | None = None) -> datetime:
        ref = to_utc(after) if after else utcnow()
        if self.interval >= timedelta(days=1):
            return self._next_daily_fire(ref)
        return self._next_intraday_fire(ref)

    def _next_daily_fire(self, ref: datetime) -> datetime:
        d: date = to_et(ref).date()
        candidate = datetime.combine(d, self.at_time, tzinfo=ET).astimezone(UTC)
        if candidate <= ref or (self.calendar and not is_trading_day(d)):
            d = next_trading_day(d) if self.calendar else d + timedelta(days=1)
            candidate = datetime.combine(d, self.at_time, tzinfo=ET).astimezone(UTC)
        return candidate

    def _next_intraday_fire(self, ref: datetime) -> datetime:
        d = to_et(ref).date()
        for _ in range(10):  # at most a long holiday weekend away
            if not self.calendar or is_trading_day(d):
                open_utc = session_open_utc(d)
                close_utc = session_close_utc(d)
                # Bars are stamped at their close, so the first fire is one
                # interval after the open.
                t = open_utc + self.interval
                while t <= close_utc:
                    if t > ref:
                        return t
                    t += self.interval
            d = next_trading_day(d) if self.calendar else d + timedelta(days=1)
        raise RuntimeError("could not find a next intraday fire time")

    def __iter__(self) -> Iterator[datetime]:
        while not self._stop.is_set():
            if self.max_fires is not None and self._fires >= self.max_fires:
                return
            target = self.next_fire(self._now)
            while not self._stop.is_set():
                remaining = (target - utcnow()).total_seconds()
                if remaining <= 0:
                    break
                # Wake often enough that stop() is responsive without busy-waiting.
                self._stop.wait(min(remaining, 1.0))
            if self._stop.is_set():
                return
            self._now = target
            self._fires += 1
            yield target


class ManualClock:
    """A clock a test or the live runner's ``step()`` drives by hand."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = to_utc(start) if start else utcnow()

    @property
    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> datetime:
        self._now = to_utc(ts)
        return self._now

    def __iter__(self) -> Iterator[datetime]:
        yield self._now
