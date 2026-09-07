"""Look-ahead barrier tests.

This is the most important file in the suite, because the barrier is the most
important code in the repo: everything else the platform claims -- that a
backtest resembles the live path, that an alt-data edge is real, that a Sharpe
means anything -- is void if a strategy can see one bar into the future.

The barrier tests are parametrized over *every* read method the ``Context``
protocol declares, and a separate test asserts that parametrization is complete.
Adding a read to the protocol without adding it here fails the suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

import pandas as pd
import pytest

from lab.engine.context import DataView, EngineContext
from lab.engine.events import Intent
from lab.engine.portfolio import Portfolio, ReadOnlyPortfolio
from lab.engine.protocols import Context, LookAheadError
from lab.indicators import computed
from lab.timeutil import UTC

from tests.conftest import bar_timestamps, make_bars, make_signals, make_view

#: Bars print at their close but are only knowable a day later, so at any
#: ``ctx.now`` the newest bar on the tape is *not* the newest visible bar. A
#: barrier that is off by one shows up immediately.
LAG = timedelta(days=1)

N_BARS = 12
TS = bar_timestamps(N_BARS)
DISCLOSURE_LAG = timedelta(days=45)


def build_context(
    view: DataView,
    *,
    now: datetime | None = None,
    strict: bool = True,
    capture: bool = True,
    universe: tuple[str, ...] = ("AAA", "BBB"),
    params: dict[str, Any] | None = None,
) -> EngineContext:
    ctx = EngineContext(
        run_id="r_test",
        strategy="probe",
        params=params or {},
        universe=list(universe),
        portfolio=ReadOnlyPortfolio(Portfolio(100_000.0)),
        data=view,
        strict=strict,
        capture=capture,
    )
    if now is not None:
        ctx.set_now(now)
    return ctx


# --- fixtures -----------------------------------------------------------------


@pytest.fixture()
def bars_df() -> pd.DataFrame:
    return make_bars("AAA", N_BARS, knowledge_lag=LAG)


@pytest.fixture()
def signals_df() -> pd.DataFrame:
    return make_signals(
        [
            {"ticker": "AAA", "event_time": TS[0], "lag_days": 45, "uid": "sig-early"},
            {"ticker": "AAA", "event_time": TS[3], "lag_days": 45, "uid": "sig-late"},
            {"ticker": "BBB", "event_time": TS[1], "lag_days": 0, "uid": "sig-instant"},
        ]
    )


@pytest.fixture()
def barrier_view(bars_df: pd.DataFrame, signals_df: pd.DataFrame) -> DataView:
    return DataView({"AAA": bars_df, "BBB": make_bars("BBB", N_BARS, knowledge_lag=LAG)}, signals_df)


# --- the barrier, over every read method --------------------------------------


def _closes(obj: Any) -> set[float]:
    return {round(float(v), 4) for v in obj if v == v}


READERS: dict[str, Callable[[EngineContext], set[Any]]] = {
    # sma(1) is the identity, so an indicator leak shows up as a close that the
    # raw readers would have refused to serve.
    "indicator": lambda ctx: _closes(ctx.indicator("AAA", "sma", n=1)),
    "history": lambda ctx: _closes(ctx.history("AAA", "close", 500)),
    "bars": lambda ctx: _closes(ctx.bars("AAA", 500)["close"]),
    "price": lambda ctx: _closes([ctx.price("AAA")] if ctx.price("AAA") is not None else []),
    "signals": lambda ctx: {e.uid for e in ctx.signals("govgreed")},
}


def _expected(name: str, bars: pd.DataFrame, signals: pd.DataFrame, now: datetime) -> set[Any]:
    """The barrier, reimplemented independently of the code under test."""
    if name == "signals":
        visible = signals[
            (signals["knowledge_time"] <= now) & (signals["source"] == "govgreed")
        ]
        return set(visible["uid"])
    visible = bars[bars["knowledge_time"] <= now]
    if name == "price":
        return _closes(visible["close"].iloc[-1:]) if len(visible) else set()
    return _closes(visible["close"])


@pytest.mark.parametrize("reader", sorted(READERS))
@pytest.mark.parametrize("at", [0, 1, 5, N_BARS - 1])
def test_every_read_stops_at_the_knowledge_barrier(
    reader: str, at: int, barrier_view: DataView, bars_df: pd.DataFrame, signals_df: pd.DataFrame
):
    now = TS[at]
    ctx = build_context(barrier_view, now=now)
    assert READERS[reader](ctx) == _expected(reader, bars_df, signals_df, now)


@pytest.mark.parametrize("reader", sorted(READERS))
def test_no_read_serves_the_bar_it_is_standing_on_when_it_is_not_yet_knowable(
    reader: str, barrier_view: DataView, bars_df: pd.DataFrame
):
    """The sharpest form of the bug: the current bar leaking into the decision."""
    now = TS[6]
    todays_close = round(float(bars_df["close"].iloc[6]), 4)
    served = READERS[reader](build_context(barrier_view, now=now))
    assert todays_close not in served


def test_the_barrier_parametrization_covers_the_whole_protocol():
    """A new read method on ``Context`` must arrive with a barrier test."""
    writes = {"order_target_pct", "close", "log"}
    state = {"params", "now", "session", "universe", "portfolio"}
    declared = set(Context.__protocol_attrs__) - writes - state
    assert declared == set(READERS), (
        "Context gained or lost a read method; add it to READERS so the "
        "look-ahead barrier is proven for it too"
    )


def test_a_bar_with_a_future_knowledge_time_is_invisible(barrier_view: DataView, bars_df):
    ctx = build_context(barrier_view, now=TS[0])
    assert len(ctx.history("AAA", "close", 500)) == 0
    assert ctx.price("AAA") is None
    assert ctx.bars("AAA", 500).empty

    ctx.set_now(TS[0] + LAG)
    assert len(ctx.history("AAA", "close", 500)) == 1
    assert ctx.price("AAA") == pytest.approx(float(bars_df["close"].iloc[0]))


def test_alt_data_becomes_visible_exactly_on_its_knowledge_time():
    """A congressional trade is invisible until its disclosure, not its trade."""
    traded = TS[0]
    disclosed = traded + DISCLOSURE_LAG
    signals = make_signals([{"ticker": "AAA", "event_time": traded, "lag_days": 45}])
    view = DataView({"AAA": make_bars("AAA", N_BARS)}, signals)
    ctx = build_context(view)

    ctx.set_now(disclosed - timedelta(microseconds=1))
    assert ctx.signals("govgreed") == []

    ctx.set_now(disclosed)
    seen = ctx.signals("govgreed")
    assert len(seen) == 1
    assert seen[0].event_time == traded
    assert seen[0].knowledge_time == disclosed
    assert (seen[0].knowledge_time - seen[0].event_time).days == 45


def test_signal_query_filters_apply_under_the_barrier(barrier_view: DataView):
    ctx = build_context(barrier_view, now=TS[3] + DISCLOSURE_LAG)
    assert {e.uid for e in ctx.signals("govgreed")} == {"sig-early", "sig-late", "sig-instant"}
    assert {e.uid for e in ctx.signals("govgreed", ticker="AAA")} == {"sig-early", "sig-late"}
    assert {e.uid for e in ctx.signals("govgreed", ticker="bbb")} == {"sig-instant"}
    assert ctx.signals("govgreed", min_score=0.99) == []
    assert ctx.signals("nobody-here") == []
    # A lookback window is measured from `now`, and can only ever narrow.
    recent = ctx.signals("govgreed", lookback_days=1)
    assert {e.uid for e in recent} <= {"sig-early", "sig-late", "sig-instant"}


# --- strict mode --------------------------------------------------------------


@pytest.fixture()
def noncausal_indicator() -> Iterator[str]:
    """An indicator that reads one row into the future, registered for one test."""
    name = "_test_noncausal_shift"

    @computed.register(name)
    def _peek(df: pd.DataFrame) -> pd.Series:
        return df["close"].shift(-1)

    try:
        yield name
    finally:
        computed.INDICATORS.pop(name, None)


def test_strict_mode_catches_a_noncausal_indicator(noncausal_indicator: str):
    """The compute-once-then-slice optimization is sound only for causal
    indicators, so strict mode proves causality per key on first use."""
    view = make_view(("AAA",), N_BARS)
    ctx = build_context(view, now=TS[5], universe=("AAA",))

    with pytest.raises(LookAheadError) as excinfo:
        ctx.indicator("AAA", noncausal_indicator)
    assert "not causal" in str(excinfo.value)
    assert "future" in str(excinfo.value)


def test_a_causal_indicator_passes_the_same_check():
    view = make_view(("AAA",), N_BARS)
    ctx = build_context(view, now=TS[5], universe=("AAA",))
    series = ctx.indicator("AAA", "sma", n=3)
    assert len(series) == 6
    assert series.index[-1] == pd.Timestamp(TS[5])


def test_non_strict_mode_tolerates_the_noncausal_indicator(noncausal_indicator: str):
    """Strictness is the switch; the check is not silently always-on."""
    view = make_view(("AAA",), N_BARS)
    ctx = build_context(view, now=TS[5], universe=("AAA",), strict=False)
    assert len(ctx.indicator("AAA", noncausal_indicator)) == 6


def test_the_guard_raises_in_strict_mode_and_only_warns_otherwise():
    view = make_view(("AAA",), N_BARS)
    future = TS[5] + timedelta(days=30)

    strict = build_context(view, now=TS[5], universe=("AAA",))
    with pytest.raises(LookAheadError) as excinfo:
        strict._guard(future)
    assert "ctx.now is" in str(excinfo.value)
    strict._guard(TS[4])  # a past timestamp is fine
    strict._guard(None)

    lenient = build_context(view, now=TS[5], universe=("AAA",), strict=False)
    lenient._guard(future)
    logs = lenient.logs()
    assert logs and logs[-1]["event"] == "look_ahead_suppressed"


def test_reading_before_the_clock_is_set_is_an_error():
    ctx = build_context(make_view(("AAA",), N_BARS), universe=("AAA",))
    with pytest.raises(RuntimeError, match="context time not set"):
        ctx.price("AAA")


def test_unknown_indicator_kwargs_are_rejected():
    ctx = build_context(make_view(("AAA",), N_BARS), now=TS[5], universe=("AAA",))
    with pytest.raises(ValueError, match="takes no parameter"):
        ctx.indicator("AAA", "sma", n=3, windw=4)


def test_cross_timeframe_reads_are_refused():
    ctx = build_context(make_view(("AAA",), N_BARS), now=TS[5], universe=("AAA",))
    with pytest.raises(ValueError, match="timeframe"):
        ctx.history("AAA", "close", 10, timeframe="5m")
    with pytest.raises(ValueError, match="timeframe"):
        ctx.bars("AAA", 10, timeframe="5m")


# --- out-of-order knowledge times ---------------------------------------------


def _restated_view(restated_row: int = 3) -> tuple[DataView, pd.DataFrame]:
    """A tape where one bar is restated far in the future -- the fallback path."""
    bars = make_bars("AAA", N_BARS)
    kt = list(bars["knowledge_time"])
    kt[restated_row] = TS[-1] + timedelta(days=3_650)
    bars["knowledge_time"] = kt
    return DataView({"AAA": bars}), bars


def test_out_of_order_knowledge_times_take_the_masking_fallback():
    view, bars = _restated_view()
    assert view._kt_sorted["AAA"] is False, "this tape must exercise the slow path"

    now = TS[8]
    ctx = build_context(view, now=now, universe=("AAA",))
    served = _closes(ctx.history("AAA", "close", 500))
    expected = _closes(bars.loc[bars["knowledge_time"] <= now, "close"])
    assert served == expected
    assert round(float(bars["close"].iloc[3]), 4) not in served
    assert len(served) == 8, "eight of the nine printed bars were knowable"


def test_a_restated_bar_never_reaches_an_indicator():
    """``visible_len`` bounds a *prefix*: counting knowable rows instead would
    slide the restated bar into every indicator window."""
    view, bars = _restated_view()
    now = TS[8]
    ctx = build_context(view, now=now, universe=("AAA",))

    served = _closes(ctx.indicator("AAA", "sma", n=1))
    assert round(float(bars["close"].iloc[3]), 4) not in served
    assert view.visible_len("AAA", now) == 3, "the knowable prefix ends at the restatement"
    assert served == _closes(bars["close"].iloc[:3])


def test_the_restated_bar_appears_once_it_is_knowable():
    view, bars = _restated_view()
    later = TS[-1] + timedelta(days=3_651)
    ctx = build_context(view, now=later, universe=("AAA",))
    assert len(ctx.history("AAA", "close", 500)) == N_BARS
    assert view.visible_len("AAA", later) == N_BARS


def test_bar_at_ignores_the_barrier_for_the_fill_engine():
    """Filling needs the bar the market printed, not what the strategy could see."""
    view = DataView({"AAA": make_bars("AAA", N_BARS, knowledge_lag=LAG)})
    assert view.bar_at("AAA", TS[6])["close"] == pytest.approx(
        float(make_bars("AAA", N_BARS)["close"].iloc[6])
    )
    assert view.bar_at("AAA", TS[6] + timedelta(hours=1)) is None
    assert view.bar_at("ZZZ", TS[6]) is None


def test_dataview_shape_helpers():
    view = make_view(("AAA", "bbb"), N_BARS)
    assert view.tickers() == ["AAA", "BBB"]
    assert view.timestamps() == TS
    assert set(view.bars_at(TS[2])) == {"AAA", "BBB"}
    assert set(view.closes_at(TS[2])) == {"AAA", "BBB"}
    assert view.frame("ZZZ").empty
    assert view.visible_len("ZZZ", TS[2]) == 0


def test_an_empty_tape_serves_nothing_rather_than_raising():
    view = DataView({"AAA": pd.DataFrame()})
    ctx = build_context(view, now=TS[5], universe=("AAA",))
    assert ctx.price("AAA") is None
    assert ctx.bars("AAA", 10).empty
    assert list(ctx.history("AAA", "close", 10)) == []
    assert list(ctx.indicator("AAA", "sma", n=3)) == []
    assert list(ctx.history("AAA", "no_such_field", 10)) == []


# --- intents ------------------------------------------------------------------


def test_the_universe_guard_refuses_a_ticker_the_run_cannot_touch():
    ctx = build_context(make_view(("AAA", "BBB"), N_BARS), now=TS[5])
    ctx.order_target_pct("aaa", 0.1)  # case is normalized, not rejected
    assert [i.ticker for i in ctx.drain_intents()] == ["AAA"]

    with pytest.raises(ValueError, match="not in this run's universe"):
        ctx.order_target_pct("ZZZ", 0.1)
    with pytest.raises(ValueError, match="not in this run's universe"):
        ctx.close("ZZZ")
    assert ctx.drain_intents() == []


@pytest.mark.parametrize("pct", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_target_is_refused(pct: float):
    ctx = build_context(make_view(("AAA",), N_BARS), now=TS[5], universe=("AAA",))
    with pytest.raises(ValueError, match="must be finite"):
        ctx.order_target_pct("AAA", pct)


def test_last_intent_per_ticker_wins():
    ctx = build_context(make_view(("AAA", "BBB"), N_BARS), now=TS[5])
    ctx.order_target_pct("AAA", 0.10, tag="first")
    ctx.order_target_pct("BBB", 0.20, tag="other")
    ctx.order_target_pct("AAA", 0.05, tag="revised", reason="changed my mind")

    intents = ctx.drain_intents()
    assert len(intents) == 2
    by_ticker = {i.ticker: i for i in intents}
    assert by_ticker["AAA"].target_pct == pytest.approx(0.05)
    assert by_ticker["AAA"].tag == "revised"
    assert by_ticker["BBB"].target_pct == pytest.approx(0.20)
    assert ctx.drain_intents() == [], "draining is destructive"


def test_close_is_a_zero_target():
    ctx = build_context(make_view(("AAA",), N_BARS), now=TS[5], universe=("AAA",))
    ctx.order_target_pct("AAA", 0.4)
    ctx.close("AAA", reason="stop")
    (intent,) = ctx.drain_intents()
    assert isinstance(intent, Intent)
    assert intent.target_pct == 0.0
    assert intent.tag == "exit"
    assert intent.reason == "stop"


def test_session_follows_the_eastern_calendar():
    ctx = build_context(make_view(("AAA",), N_BARS), now=TS[0], universe=("AAA",))
    assert ctx.session.isoformat() == "2024-01-02"
    assert ctx.now.tzinfo is UTC


# --- the input tape -----------------------------------------------------------


def test_the_input_tape_captures_every_value_served(barrier_view: DataView, bars_df):
    now = TS[7]
    ctx = build_context(barrier_view, now=now)
    ctx.history("AAA", "close", 5)
    ctx.bars("AAA", 5)
    ctx.price("AAA")
    ctx.indicator("AAA", "sma", n=2)
    ctx.signals("govgreed")
    ctx.log(note="hello", value=1.5)

    tape = ctx.input_tape()
    assert set(tape) == {"history", "bars", "prices", "indicators", "signals"}

    visible_close = round(float(bars_df.loc[bars_df["knowledge_time"] <= now, "close"].iloc[-1]), 4)
    assert tape["prices"]["AAA"] == pytest.approx(visible_close)
    assert tape["bars"]["AAA"]["last_close"] == pytest.approx(visible_close)
    assert tape["history"]["AAA.close"]["n"] == 5
    assert tape["history"]["AAA.close"]["tail"][-1]["v"] == pytest.approx(visible_close)
    assert tape["indicators"]["AAA.sma"]["params"] == {"n": 2}
    assert tape["indicators"]["AAA.sma"]["value"] is not None
    # Only the zero-lag signal has been disclosed by TS[7]; the 45-day ones
    # have not, and the tape records exactly what was served.
    assert {s["uid"] for s in tape["signals"]} == {"sig-instant"}
    for entry in tape["signals"]:
        assert entry["knowledge_time"] <= now.isoformat()

    (logged,) = ctx.logs()
    assert logged["note"] == "hello" and logged["value"] == 1.5


def test_the_tape_is_json_shaped(barrier_view: DataView):
    import json

    ctx = build_context(barrier_view, now=TS[7])
    ctx.history("AAA", "close", 5)
    ctx.indicator("AAA", "rsi", n=3)
    ctx.price("AAA")
    tape = ctx.input_tape()
    assert json.loads(json.dumps(tape)) == tape


def test_reset_bar_clears_the_tape_the_intents_and_the_logs(barrier_view: DataView):
    ctx = build_context(barrier_view, now=TS[7])
    ctx.price("AAA")
    ctx.order_target_pct("AAA", 0.2)
    ctx.log(note="x")
    assert ctx.input_tape() and ctx.logs()

    ctx.reset_bar()
    assert ctx.input_tape() == {}
    assert ctx.logs() == []
    assert ctx.drain_intents() == []


def test_capture_off_records_nothing_but_still_serves(barrier_view: DataView):
    ctx = build_context(barrier_view, now=TS[7], capture=False)
    assert ctx.price("AAA") is not None
    assert len(ctx.history("AAA", "close", 5)) == 5
    assert ctx.input_tape() == {}


def test_the_tape_snapshot_is_a_copy(barrier_view: DataView):
    ctx = build_context(barrier_view, now=TS[7])
    ctx.price("AAA")
    tape = ctx.input_tape()
    tape["prices"] = {"AAA": 1.0}
    assert ctx.input_tape()["prices"]["AAA"] != 1.0
