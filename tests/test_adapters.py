"""Adapter tests. Every one of these runs offline.

The synthetic tests additionally run with ``socket.socket`` sabotaged, because
"no network" is a promise the offline demo depends on, not an aspiration.
"""

from __future__ import annotations

import importlib.util
import math
import socket
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.adapters import AlpacaAdapter, SyntheticAdapter, YFinanceAdapter
from lab.adapters.base import AdapterUnavailable, get_adapter, list_adapters
from lab.adapters.synthetic import ANCHOR, DEFAULT_UNIVERSE
from lab.config import reset_settings_cache
from lab.store.schema import Bar, Event
from lab.timeutil import (
    is_trading_day,
    parse_timeframe,
    session_close_utc,
    session_date,
    session_open_utc,
)

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)
SYMS = ["AAPL", "MSFT", "XOM"]


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket construction explode, then run the test anyway."""

    def blocked(*_args, **_kwargs):
        raise AssertionError("adapter attempted network access")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def rows(bars):
    return [b.as_row() for b in bars]


# --- synthetic: determinism ---------------------------------------------------


def test_same_seed_is_identical(no_network) -> None:
    a = list(SyntheticAdapter(seed=42).fetch_bars(SYMS, "1d", START, END))
    b = list(SyntheticAdapter(seed=42).fetch_bars(SYMS, "1d", START, END))
    assert a and rows(a) == rows(b)


def test_different_seed_differs(no_network) -> None:
    a = list(SyntheticAdapter(seed=42).fetch_bars(SYMS, "1d", START, END))
    b = list(SyntheticAdapter(seed=43).fetch_bars(SYMS, "1d", START, END))
    assert len(a) == len(b)
    assert rows(a) != rows(b)


def test_identical_across_processes() -> None:
    """The strong form of the promise: another interpreter, same bytes."""
    code = (
        "import hashlib, json, datetime as dt\n"
        "from lab.adapters.synthetic import SyntheticAdapter\n"
        "u = dt.timezone.utc\n"
        "bars = SyntheticAdapter(seed=42).fetch_bars("
        "['AAPL','MSFT','XOM'], '1d', dt.datetime(2022,1,1,tzinfo=u), dt.datetime(2023,12,31,tzinfo=u))\n"
        "h = hashlib.blake2b(digest_size=16)\n"
        "for b in bars:\n"
        "    h.update(json.dumps(b.as_row(), default=str, sort_keys=True).encode())\n"
        "print(h.hexdigest())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr

    import hashlib
    import json

    here = hashlib.blake2b(digest_size=16)
    for bar in SyntheticAdapter(seed=42).fetch_bars(SYMS, "1d", START, END):
        here.update(json.dumps(bar.as_row(), default=str, sort_keys=True).encode())
    assert out.stdout.strip() == here.hexdigest()


def test_adding_a_ticker_does_not_perturb_the_others(no_network) -> None:
    adapter = SyntheticAdapter(seed=11)
    alone = rows(adapter.fetch_bars(["AAA"], "1d", START, END))
    crowd = [r for r in rows(adapter.fetch_bars(["AAA", "ZZZZ", "QQ"], "1d", START, END))
             if r["ticker"] == "AAA"]
    assert alone and alone == crowd


def test_window_is_independent(no_network) -> None:
    """A sub-range must restate, not conflict, with the full pull."""
    adapter = SyntheticAdapter(seed=5)
    full = {(r["ticker"], r["event_time"]): r for r in rows(adapter.fetch_bars(SYMS, "1d", START, END))}
    slice_ = rows(
        adapter.fetch_bars(SYMS, "1d", datetime(2023, 3, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC))
    )
    assert slice_
    for row in slice_:
        assert full[(row["ticker"], row["event_time"])] == row


def test_signals_are_deterministic(no_network) -> None:
    kwargs = dict(seed=3, tickers=SYMS)
    a = [e.as_row() for e in SyntheticAdapter(**kwargs).fetch_signals(START, END)]
    b = [e.as_row() for e in SyntheticAdapter(**kwargs).fetch_signals(START, END)]
    assert a and a == b


# --- synthetic: bar shape -----------------------------------------------------


@pytest.mark.parametrize("timeframe", ["1d", "1h", "5m"])
def test_ohlc_is_well_formed(no_network, timeframe: str) -> None:
    end = END if timeframe == "1d" else datetime(2022, 2, 1, tzinfo=UTC)
    bars = list(SyntheticAdapter(seed=9).fetch_bars(SYMS, timeframe, START, end))
    assert bars
    for bar in bars:
        assert bar.low <= min(bar.open, bar.close)
        assert max(bar.open, bar.close) <= bar.high
        assert bar.low > 0.0
        assert bar.volume > 0.0
        assert all(math.isfinite(v) for v in (bar.open, bar.high, bar.low, bar.close, bar.volume))
        assert bar.low <= bar.vwap <= bar.high
        assert bar.knowledge_time > bar.event_time
        assert bar.ticker == bar.ticker.upper()
        assert bar.timeframe == timeframe


def test_daily_bars_land_only_on_trading_days(no_network) -> None:
    bars = list(SyntheticAdapter(seed=9).fetch_bars(SYMS, "1d", START, END))
    assert bars
    for bar in bars:
        day = session_date(bar.event_time)
        assert is_trading_day(day)
        assert bar.event_time == session_open_utc(day)
        assert bar.knowledge_time == session_close_utc(day)


@pytest.mark.parametrize("timeframe", ["1h", "5m", "30m"])
def test_intraday_bars_stay_inside_the_session(no_network, timeframe: str) -> None:
    end = datetime(2022, 1, 31, tzinfo=UTC)
    bars = list(SyntheticAdapter(seed=9).fetch_bars(["AAPL"], timeframe, START, end))
    assert bars
    seen = set()
    for bar in bars:
        day = session_date(bar.event_time)
        assert is_trading_day(day)
        assert session_open_utc(day) <= bar.event_time < session_close_utc(day)
        assert bar.knowledge_time <= session_close_utc(day)
        assert bar.knowledge_time <= bar.event_time + parse_timeframe(timeframe)
        seen.add((bar.ticker, bar.event_time))
    assert len(seen) == len(bars)  # no duplicate timestamps


def test_intraday_last_bar_closes_the_session(no_network) -> None:
    day = datetime(2022, 1, 5, tzinfo=UTC)
    bars = list(SyntheticAdapter(seed=9).fetch_bars(["AAPL"], "1h", day, day))
    assert bars[-1].knowledge_time == session_close_utc(day.date())
    daily = list(SyntheticAdapter(seed=9).fetch_bars(["AAPL"], "1d", day, day))
    # The intraday path is a bridge across the daily bar, so the endpoints agree.
    assert bars[0].open == pytest.approx(daily[0].open)
    assert bars[-1].close == pytest.approx(daily[0].close)


def test_bounds_with_a_time_of_day_are_honoured(no_network) -> None:
    """A midnight bound means the whole session; a real time means that instant."""
    adapter = SyntheticAdapter(seed=9)
    day = datetime(2022, 1, 5, tzinfo=UTC)
    whole = list(adapter.fetch_bars(["AAPL"], "1h", day, day))
    partial = list(
        adapter.fetch_bars(["AAPL"], "1h", day, datetime(2022, 1, 5, 16, 0, tzinfo=UTC))
    )
    assert 0 < len(partial) < len(whole)
    assert rows(partial) == rows(whole[: len(partial)])
    assert all(b.event_time <= datetime(2022, 1, 5, 16, 0, tzinfo=UTC) for b in partial)


def test_tickers_differ_in_drift_and_vol(no_network) -> None:
    """Cross-sectional strategies need something to rank."""
    adapter = SyntheticAdapter(seed=4)
    totals = []
    for sym in DEFAULT_UNIVERSE[:6]:
        closes = [b.close for b in adapter.fetch_bars([sym], "1d", START, END)]
        totals.append(closes[-1] / closes[0])
    assert max(totals) - min(totals) > 0.10
    assert len(set(round(t, 6) for t in totals)) == len(totals)


# --- synthetic: signals and the disclosure lag --------------------------------


def test_signals_carry_a_real_disclosure_lag(no_network) -> None:
    adapter = SyntheticAdapter(seed=3, tickers=DEFAULT_UNIVERSE, disclosure_lag_days=45)
    events = list(adapter.fetch_signals(START, END))
    assert len(events) > 50

    lags = []
    for ev in events:
        assert isinstance(ev, Event)
        assert ev.knowledge_time >= ev.event_time
        ev.as_row()  # the schema itself refuses knowledge before the event
        lags.append((ev.knowledge_time - ev.event_time).days)

    assert min(lags) >= 1
    # Rounding a lag forward onto the next trading day can add a few days.
    assert max(lags) <= 45 + 5
    assert max(lags) > 20, "a 45-day disclosure regime must actually produce long lags"
    assert sum(lags) / len(lags) > 10


def test_disclosure_lag_is_configurable(no_network) -> None:
    tight = SyntheticAdapter(seed=3, tickers=DEFAULT_UNIVERSE, disclosure_lag_days=3,
                             min_disclosure_lag_days=1)
    lags = [(e.knowledge_time - e.event_time).days for e in tight.fetch_signals(START, END)]
    assert lags
    assert max(lags) <= 3 + 5
    loose = SyntheticAdapter(seed=3, tickers=DEFAULT_UNIVERSE, disclosure_lag_days=45)
    loose_lags = [(e.knowledge_time - e.event_time).days for e in loose.fetch_signals(START, END)]
    assert sum(loose_lags) / len(loose_lags) > sum(lags) / len(lags)


def test_signals_are_knowledge_time_ordered_and_uniquely_keyed(no_network) -> None:
    events = list(SyntheticAdapter(seed=3).fetch_signals(START, END))
    assert events
    assert events == sorted(events, key=lambda e: (e.knowledge_time, e.ticker, e.uid))
    assert len({e.uid for e in events}) == len(events)
    for ev in events:
        assert START <= ev.knowledge_time <= END
        assert ev.source == "synthetic"
        assert ev.direction in {"BUY", "SELL"}
        assert ev.tier in {"A", "B", "C"}
        assert 0.0 <= (ev.score or 0.0) <= 1.0
        traded = date.fromisoformat(ev.payload["traded_on"])
        disclosed = date.fromisoformat(ev.payload["disclosed_on"])
        assert ev.payload["lag_days"] == (disclosed - traded).days
        # The UTC delta can sit an hour short of the calendar gap across a DST
        # boundary, since both stamps are session closes in Eastern time.
        assert abs(ev.payload["lag_days"] - (ev.knowledge_time - ev.event_time).days) <= 1


def test_signal_query_filters(no_network) -> None:
    adapter = SyntheticAdapter(seed=3)
    only = list(adapter.fetch_signals(START, END, tickers=["AAPL"], kind="congress_trade", limit=5))
    assert 0 < len(only) <= 5
    assert {e.ticker for e in only} == {"AAPL"}
    assert {e.kind for e in only} == {"congress_trade"}
    graded = list(adapter.fetch_signals(START, END, min_score=0.8))
    assert graded and all((e.score or 0.0) >= 0.8 for e in graded)


# --- synthetic: bad input -----------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ([], "1d", START, END),
        (SYMS, "1x", START, END),
        (SYMS, "3d", START, END),
        (SYMS, "1d", END, START),
        (SYMS, "1d", datetime(1970, 1, 1, tzinfo=UTC), END),
    ],
)
def test_bad_bar_requests_raise_value_error(args) -> None:
    with pytest.raises(ValueError):
        list(SyntheticAdapter().fetch_bars(*args))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"disclosure_lag_days": -1},
        {"disclosure_lag_days": 1, "min_disclosure_lag_days": 5},
        {"signal_rate": 1.5},
        {"signal_edge": -0.1},
        {"anchor_pull": -1.0},
    ],
)
def test_bad_construction_raises_value_error(kwargs) -> None:
    with pytest.raises(ValueError):
        SyntheticAdapter(**kwargs)


def test_anchor_is_the_documented_floor() -> None:
    bars = list(SyntheticAdapter().fetch_bars(["AAPL"], "1d", ANCHOR, datetime(1990, 1, 10, tzinfo=UTC)))
    assert bars and all(isinstance(b, Bar) for b in bars)


# --- registry -----------------------------------------------------------------


def test_registry_resolves_offline(no_network) -> None:
    assert isinstance(get_adapter("synthetic"), SyntheticAdapter)
    assert isinstance(get_adapter("SYNTHETIC"), SyntheticAdapter)
    assert isinstance(get_adapter("yfinance"), YFinanceAdapter)
    assert isinstance(get_adapter("alpaca"), AlpacaAdapter)
    assert isinstance(get_adapter("synthetic", seed=99), SyntheticAdapter)


def test_unknown_adapter_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_adapter("bloomberg")


def test_list_adapters_never_raises(no_network) -> None:
    infos = list_adapters()
    names = {i.name for i in infos}
    assert {"synthetic", "yfinance", "alpaca"} <= names
    for info in infos:
        assert isinstance(info.available, bool)
        assert isinstance(info.reason, str)
        assert info.available or info.reason, "an unavailable adapter must say why"
        info.to_dict()


def test_synthetic_is_always_available(no_network) -> None:
    assert SyntheticAdapter().available() == (True, "")
    assert SyntheticAdapter().provides == frozenset({"bars", "signals"})


# --- yfinance / alpaca: availability only, never a request --------------------


@pytest.mark.parametrize("cls", [YFinanceAdapter, AlpacaAdapter])
def test_optional_adapters_report_state_without_raising(no_network, cls) -> None:
    adapter = cls()
    ok, reason = adapter.available()
    assert isinstance(ok, bool) and isinstance(reason, str)
    assert ok or reason
    assert adapter.provides == frozenset({"bars"})
    assert adapter.info().to_dict()["name"] == adapter.name
    if not ok:
        with pytest.raises(AdapterUnavailable):
            adapter.require_available()


def test_alpaca_is_unavailable_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("alpaca") is None:
        pytest.skip("alpaca-py not installed")
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    reset_settings_cache()
    try:
        assert AlpacaAdapter().available() == (False, "no ALPACA_API_KEY_ID")
        assert AlpacaAdapter(key_id="k").available() == (False, "no ALPACA_API_SECRET_KEY")
        assert AlpacaAdapter(key_id="k", secret_key="s").available() == (True, "")
    finally:
        reset_settings_cache()


def test_alpaca_rejects_bad_timeframes_before_any_client(no_network) -> None:
    from lab.adapters.alpaca import _split_timeframe

    assert _split_timeframe("5m") == (5, "m")
    assert _split_timeframe("1D") == (1, "d")
    with pytest.raises(ValueError):
        _split_timeframe("1y")


def test_yfinance_interval_mapping(no_network) -> None:
    adapter = YFinanceAdapter()
    assert adapter.interval("1d") == "1d"
    assert adapter.interval("60m") == "1h"
    assert adapter.interval("5m") == "5m"
    with pytest.raises(ValueError):
        adapter.interval("1w")
    with pytest.raises(ValueError):
        adapter.interval("nonsense")


def test_yfinance_frame_splitting_handles_multiindex(no_network) -> None:
    """The multi-ticker frame shape is the part most likely to rot."""
    import pandas as pd

    from lab.adapters.yfinance import _frames_by_ticker

    index = pd.date_range("2024-01-02", periods=3, freq="D")
    columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]])
    frame = pd.DataFrame(1.0, index=index, columns=columns)
    split = _frames_by_ticker(frame, ["AAPL", "MSFT"])
    assert set(split) == {"AAPL", "MSFT"}
    assert list(split["AAPL"].columns) == ["Open", "High", "Low", "Close", "Volume"]

    flipped = frame.swaplevel(axis=1).sort_index(axis=1)
    assert set(_frames_by_ticker(flipped, ["AAPL", "MSFT"])) == {"AAPL", "MSFT"}

    flat = pd.DataFrame(1.0, index=index, columns=["Open", "High", "Low", "Close", "Volume"])
    assert set(_frames_by_ticker(flat, ["AAPL"])) == {"AAPL"}


def test_yfinance_daily_normalizes_to_the_session(no_network) -> None:
    import pandas as pd

    adapter = YFinanceAdapter()
    frame = pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-03")]),
    )
    bar = next(iter(adapter._bars_from("AAPL", "1d", timedelta(days=1), frame)))
    assert bar.event_time == session_open_utc(bar.event_time.date())
    assert bar.knowledge_time == session_close_utc(bar.event_time.date())
    assert bar.adjusted is True

    intraday = pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-03 15:30", tz="America/New_York")]),
    )
    hour = next(iter(adapter._bars_from("AAPL", "1h", timedelta(hours=1), intraday)))
    # One bar interval later, clipped at the close rather than spilling past it.
    assert hour.knowledge_time == session_close_utc(session_date(hour.event_time))
