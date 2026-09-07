"""Time handling. One rule: every timestamp inside the lab is timezone-aware
UTC. Conversion to US/Eastern happens only at display and calendar edges.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

UTC = timezone.utc
ET = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def to_utc(ts) -> datetime:
    """Coerce anything date-like to an aware UTC datetime.

    Naive inputs are assumed UTC; a naive *date* becomes midnight UTC.
    """
    if isinstance(ts, datetime):
        return ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
    if isinstance(ts, date):
        return datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").to_pydatetime()


def to_et(ts) -> datetime:
    return to_utc(ts).astimezone(ET)


def session_date(ts) -> date:
    """The trading session a timestamp belongs to, in Eastern terms."""
    return to_et(ts).date()


# --- calendar ----------------------------------------------------------------
# A pragmatic US-equities calendar: weekends plus the fixed and observed federal
# market holidays. Good enough for daily/minute research; swap for
# pandas_market_calendars if half-days ever matter.

_FIXED_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth (from 2022)
    (7, 4),    # Independence Day
    (12, 25),  # Christmas
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _easter(year: int) -> date:
    # Anonymous Gregorian algorithm; Good Friday is Easter - 2 days.
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def market_holidays(year: int) -> set[date]:
    days = {_observed(date(year, m, d)) for (m, d) in _FIXED_HOLIDAYS if not (m == 6 and year < 2022)}
    days.add(_nth_weekday(year, 1, 0, 3))    # MLK Day
    days.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday
    days.add(_easter(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))      # Memorial Day
    days.add(_nth_weekday(year, 9, 0, 1))    # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving
    return days


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in market_holidays(d.year)


def trading_days(start: date, end: date) -> list[date]:
    out, cur = [], start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def next_trading_day(d: date) -> date:
    cur = d + timedelta(days=1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def session_open_utc(d: date) -> datetime:
    return datetime.combine(d, REGULAR_OPEN, tzinfo=ET).astimezone(UTC)


def session_close_utc(d: date) -> datetime:
    return datetime.combine(d, REGULAR_CLOSE, tzinfo=ET).astimezone(UTC)


def parse_timeframe(tf: str) -> timedelta:
    """'1d' -> 1 day, '5m' -> 5 minutes, '1h' -> 1 hour."""
    tf = tf.strip().lower()
    unit, value = tf[-1], tf[:-1]
    if not value.isdigit():
        raise ValueError(f"bad timeframe {tf!r}; expected e.g. '1d', '15m', '1h'")
    n = int(value)
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "m":
        return timedelta(minutes=n)
    raise ValueError(f"unsupported timeframe unit {unit!r} in {tf!r}")


def is_intraday(tf: str) -> bool:
    return parse_timeframe(tf) < timedelta(days=1)
