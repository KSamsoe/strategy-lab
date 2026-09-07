"""Shared fixtures for the whole suite.

Three things live here, all of them things every engine test needs and none of
them worth re-deriving per module.

**Path isolation, autouse.** ``get_settings()`` is process-cached and resolves
``data/`` and ``runs/`` from the environment, so a single test that forgets to
redirect them writes the developer's real registry, journal and artifact
directory. Redirecting is therefore not opt-in. Cached sqlite connections are
closed on both edges of the fixture: ``lab.registry.db`` caches one connection
per (thread, path), and a connection left open on the previous test's tmpdir
would keep serving reads from a directory pytest has already deleted.

**A deterministic bar builder.** Closes come from a fixed cycle of returns, not
an RNG, so a failure reproduces from the test name alone and two runs of the
same backtest are comparable byte for byte. ``knowledge_lag`` is a first-class
argument because the look-ahead barrier is the property most of these tests
exist to pin down.

**A strategy factory.** The runner only ever touches ``LoadedStrategy`` through
``.name``, ``.params`` and ``.hook()``, so a test strategy is a closure in a
throwaway module rather than a file on disk.
"""

from __future__ import annotations

import types
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import pytest

from lab.config import Paths, get_settings, reset_settings_cache
from lab.engine.context import DataView
from lab.engine.loader import LoadedStrategy
from lab.timeutil import UTC, session_close_utc, trading_days

#: First bar of the deterministic tape: 16:00 ET on Tue 2 Jan 2024, i.e. a
#: daily bar stamped at its close on a real session.
START = date(2024, 1, 2)

#: A fixed return cycle. Mixed signs and coprime-ish length so a moving average
#: over any small window keeps moving and a momentum rule actually trades.
_RETURNS: tuple[float, ...] = (0.010, -0.004, 0.012, -0.008, 0.006, 0.015, -0.011, 0.003, -0.002, 0.009)


# --- isolation ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterable[Paths]:
    """Point every on-disk lab location at this test's tmpdir."""
    from lab.registry import db

    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(tmp_path / "runs"))
    # A kill switch left engaged in the developer's environment would block
    # every gate evaluation in the suite; the tests that want one set it.
    monkeypatch.delenv("LAB_KILL_SWITCH", raising=False)
    monkeypatch.delenv("LAB_KILL_FILE", raising=False)
    # Credentials are scrubbed for the same reason, and it is not hypothetical:
    # `lab.config` loads .env at import, so a developer who has actually
    # configured Alpaca or an agent backend would otherwise see availability
    # tests flip and, worse, could have a test reach a real endpoint. The suite
    # must behave identically on a machine with every key and a machine with
    # none; tests that need a credential set it themselves.
    for var in (
        "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "ALPACA_PAPER",
        "ANTHROPIC_API_KEY", "GOVGREED_API_KEY",
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "LAB_AGENT_API_KEY",
        "OPENROUTER_REFERER", "OPENROUTER_TITLE",
        "LAB_AGENT_BASE_URL", "LAB_AGENT_MODEL",
        "LAB_UI_TOKEN", "LAB_ALERT_WEBHOOK",
    ):
        monkeypatch.delenv(var, raising=False)
    # Pin the agent backend to the one with no credentials rather than leaving
    # it on `auto`: `auto` would find the `claude` CLI on a developer machine and
    # a test that touches the agent layer would spawn real subprocesses, spend
    # real subscription quota, and take minutes. Tests that want a backend set
    # this themselves.
    monkeypatch.setenv("LAB_AGENT_PROVIDER", "anthropic")
    reset_settings_cache()
    db.close_all()
    try:
        yield get_settings().paths
    finally:
        db.close_all()
        reset_settings_cache()


@pytest.fixture()
def paths(isolated_paths: Paths) -> Paths:
    return isolated_paths


# --- deterministic bars -------------------------------------------------------


def bar_timestamps(n: int, *, start: date = START) -> list[datetime]:
    """``n`` daily bar closes on consecutive real sessions."""
    if n <= 0:
        return []
    span = trading_days(start, start + timedelta(days=n * 2 + 14))
    if len(span) < n:  # pragma: no cover - only if the calendar changes shape
        raise ValueError(f"cannot find {n} sessions after {start}")
    return [session_close_utc(d) for d in span[:n]]


def make_bars(
    ticker: str = "AAA",
    n: int = 40,
    *,
    start: date = START,
    base: float = 100.0,
    phase: int = 0,
    knowledge_lag: timedelta = timedelta(0),
    volume: float = 1_000_000.0,
    closes: Sequence[float] | None = None,
    timestamps: Sequence[datetime] | None = None,
) -> pd.DataFrame:
    """An OHLCV frame indexed by ``event_time``, with a ``knowledge_time`` column.

    ``knowledge_lag`` shifts knowability away from the print: the default of
    zero means a bar is knowable the instant it closes, which is what the
    runner assumes for bars it decides on.
    """
    index = list(timestamps) if timestamps is not None else bar_timestamps(n, start=start)
    n = len(index)
    if closes is None:
        series: list[float] = []
        price = float(base)
        for i in range(n):
            price *= 1.0 + _RETURNS[(i + phase) % len(_RETURNS)]
            series.append(round(price, 4))
        closes = series
    if len(closes) != n:
        raise ValueError(f"{len(closes)} closes for {n} timestamps")

    opens = [closes[0]] + list(closes[:-1])
    frame = pd.DataFrame(
        {
            "open": [round(o, 4) for o in opens],
            "high": [round(max(o, c) * 1.004, 4) for o, c in zip(opens, closes)],
            "low": [round(min(o, c) * 0.996, 4) for o, c in zip(opens, closes)],
            "close": list(closes),
            "volume": [float(volume)] * n,
            "knowledge_time": [ts + knowledge_lag for ts in index],
        },
        index=pd.DatetimeIndex(index, name="event_time"),
    )
    frame.attrs["ticker"] = ticker.upper()
    return frame


def make_signals(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """A signals frame in store shape. ``lag_days`` sets the disclosure delay."""
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        r = dict(row)
        event_time = r.pop("event_time")
        lag = r.pop("lag_days", 45)
        knowledge_time = r.pop("knowledge_time", None) or event_time + timedelta(days=lag)
        ticker = str(r.pop("ticker", "AAA")).upper()
        out.append(
            {
                "event_time": event_time,
                "knowledge_time": knowledge_time,
                "source": r.pop("source", "govgreed"),
                "ticker": ticker,
                "kind": r.pop("kind", "congress_trade"),
                "direction": r.pop("direction", "buy"),
                "tier": r.pop("tier", "A"),
                "score": r.pop("score", 0.8),
                "sector": r.pop("sector", "tech"),
                "fresh": r.pop("fresh", True),
                "uid": r.pop("uid", f"uid-{ticker}-{i}"),
                "payload": r.pop("payload", "{}"),
                "request_id": r.pop("request_id", None),
            }
            | r
        )
    return pd.DataFrame(out)


def make_view(
    tickers: Sequence[str] = ("AAA", "BBB"),
    n: int = 40,
    *,
    start: date = START,
    knowledge_lag: timedelta = timedelta(0),
    signals: pd.DataFrame | None = None,
    timeframe: str = "1d",
    **bar_kwargs: Any,
) -> DataView:
    """A ``DataView`` over ``len(tickers)`` deterministic, phase-shifted tapes."""
    bars = {
        t.upper(): make_bars(
            t, n, start=start, phase=i * 3, knowledge_lag=knowledge_lag, **bar_kwargs
        )
        for i, t in enumerate(tickers)
    }
    return DataView(bars, signals, timeframe=timeframe)


@pytest.fixture()
def bars_factory() -> Callable[..., pd.DataFrame]:
    return make_bars


@pytest.fixture()
def view_factory() -> Callable[..., DataView]:
    return make_view


@pytest.fixture()
def signals_factory() -> Callable[..., pd.DataFrame]:
    return make_signals


@pytest.fixture()
def view() -> DataView:
    """Two tickers, 40 sessions, bars knowable the moment they close."""
    return make_view()


# --- strategies ---------------------------------------------------------------


class Probe:
    """A strategy whose ``on_bar`` is whatever the test passed in."""

    def __init__(self, on_bar: Callable[[Any], None], params: Mapping[str, Any] | None = None):
        self.params = dict(params or {})
        self._on_bar = on_bar
        self.bars_seen: list[datetime] = []
        self.fills_seen: list[Any] = []
        self.started = 0
        self.stopped = 0

    def on_bar(self, ctx: Any) -> None:
        self.bars_seen.append(ctx.now)
        self._on_bar(ctx)


def make_strategy(
    on_bar: Callable[[Any], None],
    *,
    name: str = "probe",
    params: Mapping[str, Any] | None = None,
    on_start: Callable[[Any], None] | None = None,
    on_fill: Callable[[Any, Any], None] | None = None,
    on_stop: Callable[[Any], None] | None = None,
    track_fills: bool = False,
) -> LoadedStrategy:
    """Wrap a callable as the ``LoadedStrategy`` the runner expects.

    Optional hooks are attached only when asked for, so ``loaded.hook("on_fill")``
    reports ``None`` for a strategy that does not define one -- the same shape a
    real module gives.
    """
    instance = Probe(on_bar, params)
    if on_start is not None:
        instance.on_start = _counted(instance, "started", on_start)
    if on_stop is not None:
        instance.on_stop = _counted(instance, "stopped", on_stop)
    if on_fill is not None or track_fills:
        def _on_fill(ctx: Any, fill: Any, _fn=on_fill, _inst=instance) -> None:
            _inst.fills_seen.append(fill)
            if _fn is not None:
                _fn(ctx, fill)

        instance.on_fill = _on_fill

    module = types.ModuleType(f"lab_test_strategy_{name}")
    module.NAME = name
    module.PARAMS = dict(params or {})
    return LoadedStrategy(
        name=name,
        instance=instance,
        module=module,
        path=Path(f"<generated:{name}>"),
        params=dict(params or {}),
        source=f"# generated test strategy {name}\n",
        source_hash=f"test{abs(hash(name)) % 10**8:08d}",
        doc="generated test strategy",
    )


def _counted(instance: Probe, attr: str, fn: Callable[[Any], None]) -> Callable[[Any], None]:
    def wrapper(ctx: Any) -> None:
        setattr(instance, attr, getattr(instance, attr) + 1)
        fn(ctx)

    return wrapper


@pytest.fixture()
def strategy_factory() -> Callable[..., LoadedStrategy]:
    return make_strategy


# --- a store with data in it --------------------------------------------------


def seed_store(
    tickers: Sequence[str] = ("AAA", "BBB", "CCC"),
    *,
    days: int = 400,
    seed: int = 11,
    signals: bool = False,
    start: date = START,
) -> dict[str, Any]:
    """Write deterministic synthetic bars into the *isolated* parquet store.

    ``isolated_paths`` is autouse, so the developer's real ``data/`` is never
    visible to a test. Anything exercising the store end to end -- sweeps,
    ``fast_screen``, ``DataView.from_store``, the CLI, the API -- therefore has
    to put its own bars there first, which is what this does.

    Returns the handful of facts callers need to build a matching config.
    """
    from datetime import datetime as _dt

    from lab.adapters.synthetic import SyntheticAdapter
    from lab.store import parquet_io

    tickers = [t.upper() for t in tickers]
    lo = _dt(start.year, start.month, start.day, tzinfo=UTC)
    hi = lo + timedelta(days=int(days * 1.5))

    adapter = SyntheticAdapter(seed=seed)
    rows = parquet_io.write_bars(adapter.fetch_bars(tickers, "1d", lo, hi))

    n_signals = 0
    if signals:
        n_signals = parquet_io.write_events(
            adapter.fetch_signals(lo, hi, tickers=tickers)
        )

    return {
        "tickers": tickers,
        "start": lo,
        "end": hi,
        "bars": rows,
        "signals": n_signals,
        "data_version": parquet_io.data_version(tickers=tickers),
    }


@pytest.fixture()
def seeded_store() -> dict[str, Any]:
    return seed_store()


@pytest.fixture()
def seeded_store_factory() -> Callable[..., dict[str, Any]]:
    return seed_store
