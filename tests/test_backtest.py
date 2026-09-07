"""End-to-end backtest tests, plus known-answer checks on the metrics module.

The runner is tested through a hand-built ``DataView`` rather than the store, so
these tests assert on arithmetic they can predict: which bar a decision lands
on, which bar it fills against, and what the equity curve must therefore be.
Determinism gets its own test because "same commit + same config + same data
version implies identical metrics" is a claim the platform makes out loud.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from lab.backtest import metrics as M
from lab.backtest.runner import (
    BacktestConfig,
    BacktestResult,
    load_universe,
    run_backtest,
)
from lab.engine.events import GateAction, Side, Trade
from lab.registry.journal import DecisionJournal, EventJournal
from lab.registry.runs import RunRegistry
from lab.timeutil import UTC

from tests.conftest import bar_timestamps, make_strategy, make_view

N = 12
TS = bar_timestamps(N)
ENTRY, EXIT = TS[2], TS[6]


def view():
    return make_view(("AAA", "BBB"), N)


def one_round_trip(pct: float = 0.5) -> Callable[[Any], None]:
    """Buy AAA on bar 2, flatten on bar 6. Fills land on bars 3 and 7."""

    def on_bar(ctx: Any) -> None:
        if ctx.now == ENTRY:
            ctx.order_target_pct("AAA", pct, tag="entry", reason="scripted")
        elif ctx.now == EXIT:
            ctx.close("AAA", tag="exit")

    return on_bar


def config(**overrides: Any) -> BacktestConfig:
    base: dict[str, Any] = {
        "strategy": "probe",
        "tickers": ["AAA", "BBB"],
        "cash": 100_000.0,
        "fills": {"mode": "next_open", "slippage_bps": 5.0},
        # Room for a half-book position; the point of these runs is the loop,
        # not the gate, which has its own suite.
        "limits": {"max_position_pct": 0.6, "max_gross_exposure": 1.0},
    }
    base.update(overrides)
    return BacktestConfig(**base)


def run(on_bar=None, *, register: bool = False, journal: bool = False, **cfg: Any):
    strategy = make_strategy(on_bar or one_round_trip(), name="probe")
    return run_backtest(
        config(**cfg), register=register, journal=journal, data=view(), strategy=strategy
    )


# --- the loop -----------------------------------------------------------------


def test_a_decision_is_recorded_for_every_bar():
    result = run()
    assert [d.at for d in result.decisions] == TS
    assert all(d.run_id == result.run_id for d in result.decisions)
    assert all(d.strategy == "probe" for d in result.decisions)


def test_orders_fill_on_the_bar_after_the_decision():
    result = run()
    assert [f.ticker for f in result.fills] == ["AAA", "AAA"]
    assert [f.side for f in result.fills] == [Side.BUY, Side.SELL]
    assert [f.at for f in result.fills] == [TS[3], TS[7]], "next_open means the next bar"

    bars = view().frame("AAA")
    entry_open = float(bars["open"].iloc[3])
    assert result.fills[0].price == pytest.approx(entry_open * 1.0005), "5bps against the buyer"


def test_the_round_trip_reconciles_with_the_equity_curve():
    result = run()
    (trade,) = result.trades

    assert trade.ticker == "AAA"
    assert trade.entry_time == TS[3] and trade.exit_time == TS[7]
    assert trade.exit_reason == "exit"
    assert trade.bars_held == 4

    assert result.equity.iloc[-1] == pytest.approx(100_000.0 + trade.pnl), (
        "flat at the end, so the whole curve is explained by the one round trip"
    )
    # And independently, from the fill ledger alone.
    cash = 100_000.0
    for f in result.fills:
        cash -= f.signed_qty * f.price + f.commission
    assert result.equity.iloc[-1] == pytest.approx(cash)


def test_equity_marks_the_open_position_every_bar():
    result = run(lambda ctx: ctx.order_target_pct("AAA", 0.5) if ctx.now == ENTRY else None)
    closes = view().frame("AAA")["close"]
    qty = result.fills[0].qty
    cash = 100_000.0 - result.fills[0].signed_qty * result.fills[0].price

    for ts in TS[4:]:
        assert result.equity.loc[ts] == pytest.approx(cash + qty * float(closes.loc[ts]))
    assert result.equity.index.tolist() == TS


def test_warmup_is_excluded_from_the_curve_not_just_from_decisions():
    """The curve starts where trading could start.

    It used to keep the warm-up bars, which pinned the equity flat while the
    benchmark compounded beside it and stretched the CAGR denominator over a
    period the strategy had no say in.
    """
    result = run(warmup=4)
    assert [d.at for d in result.decisions] == TS[4:]
    assert result.equity.index.tolist() == TS[4:], "no dead prefix"
    assert len(result.equity) == N - 4
    assert result.metrics["warmup_bars"] == 4
    assert result.metrics["tradeable_bars"] == N - 4
    assert result.equity.iloc[0] == pytest.approx(100_000.0)


def test_a_warmed_up_run_cannot_trade_on_the_skipped_bars():
    result = run(one_round_trip(), warmup=4)
    assert result.trades == [], "the entry bar was inside the warm-up"
    assert result.fills == []
    assert result.equity.nunique() == 1
    assert result.equity.index[0] == TS[4]


def test_strategy_hooks_fire_in_order():
    seen: list[str] = []
    strategy = make_strategy(
        lambda ctx: seen.append("bar"),
        name="probe",
        on_start=lambda ctx: seen.append("start"),
        on_stop=lambda ctx: seen.append("stop"),
    )
    run_backtest(config(), register=False, journal=False, data=view(), strategy=strategy)
    assert seen[0] == "start" and seen[-1] == "stop"
    assert seen.count("bar") == N


def test_on_fill_receives_every_fill():
    strategy = make_strategy(one_round_trip(), name="probe", track_fills=True)
    result = run_backtest(config(), register=False, journal=False, data=view(), strategy=strategy)
    assert [f.id for f in strategy.instance.fills_seen] == [f.id for f in result.fills]


def test_a_run_with_no_bars_is_refused():
    from lab.engine.context import DataView

    with pytest.raises(ValueError, match="no bars"):
        run_backtest(
            config(),
            register=False,
            journal=False,
            data=DataView({"AAA": pd.DataFrame()}),
            strategy=make_strategy(one_round_trip()),
        )


# --- determinism --------------------------------------------------------------


def test_the_same_config_and_data_produce_identical_metrics():
    first = run()
    second = run()

    assert first.metrics == second.metrics, "metrics must be reproducible in full"
    assert first.run_id != second.run_id
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert [t.to_dict() for t in first.trades] == [t.to_dict() for t in second.trades]
    assert [d.at for d in first.decisions] == [d.at for d in second.decisions]
    assert [d.inputs for d in first.decisions] == [d.inputs for d in second.decisions]


def test_a_different_parameter_changes_the_result():
    small = run(one_round_trip(0.1))
    large = run(one_round_trip(0.5))
    assert small.metrics["total_return"] != large.metrics["total_return"]


# --- artifacts and persistence ------------------------------------------------

ARTIFACTS = (
    "metrics.json",
    "equity.parquet",
    "trades.csv",
    "config.json",
    "decisions.jsonl",
    "provenance.json",
)


def test_every_artifact_is_written(paths):
    result = run(register=True, journal=True)
    d = result.artifact_dir

    assert d == paths.runs / result.run_id
    for name in ARTIFACTS:
        assert (d / name).exists(), f"{name} was not written"

    equity = pd.read_parquet(d / "equity.parquet")
    assert len(equity) == N
    assert {"equity", "drawdown", "exposure"} <= set(equity.columns)
    assert len((d / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()) == N
    assert "survivorship" in json.loads((d / "provenance.json").read_text(encoding="utf-8"))[
        "survivorship_note"
    ].lower()


def test_metrics_json_round_trips_without_a_custom_encoder(paths):
    result = run(register=True, journal=True)

    assert json.loads(json.dumps(result.metrics)) == result.metrics
    on_disk = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == result.metrics
    for key, value in result.metrics.items():
        assert not isinstance(value, float) or math.isfinite(value), key


def test_the_run_is_registered_and_finished(paths):
    result = run(register=True, journal=True)
    record = RunRegistry().get(result.run_id)

    assert record is not None
    assert record.status == "ok"
    assert record.kind == "backtest"
    assert record.strategy == "probe"
    assert record.metrics == result.metrics
    assert record.attempt == 1 == result.attempt
    assert record.finished_at is not None
    assert record.start == TS[0] and record.end == TS[-1]
    assert record.run_id.startswith("probe-"), "run ids are sortable and self-describing"


def test_the_journals_receive_the_run(paths):
    result = run(register=True, journal=True)

    journal = DecisionJournal()
    assert journal.count(result.run_id) == N
    rows = journal.list(result.run_id)
    assert [r["at"] for r in rows] == [t.isoformat() for t in TS]

    events = EventJournal().tail(0, limit=500, run_id=result.run_id)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("fill") == 2


def test_gate_verdicts_are_persisted_with_the_decision(paths):
    """A clip has to be visible in the tape, or "why is this position this size"
    is unanswerable after the fact."""
    result = run(
        one_round_trip(0.5),
        register=True,
        journal=True,
        limits={"max_position_pct": 0.05},
    )

    entry = next(d for d in result.decisions if d.at == ENTRY)
    (verdict,) = entry.verdicts
    assert verdict.action is GateAction.CLIP
    assert verdict.rule == "position_cap"
    assert verdict.requested_pct == pytest.approx(0.5)
    assert verdict.approved_pct == pytest.approx(0.05)

    stored = DecisionJournal().get(entry.id)
    assert stored is not None
    assert stored["verdicts"][0]["rule"] == "position_cap"
    assert stored["verdicts"][0]["action"] == "clipped"
    assert "clipped" in stored["summary"]

    blocks = EventJournal().tail(0, limit=500, run_id=result.run_id, kinds=["gate_block"])
    assert any(e["payload"]["rule"] == "position_cap" for e in blocks)
    # The clip is real money: 5% of the book, not 50%.
    assert result.fills[0].qty * result.fills[0].price < 0.06 * 100_000


def test_the_decision_tape_carries_the_inputs_the_strategy_read():
    def on_bar(ctx: Any) -> None:
        ctx.price("AAA")
        ctx.history("AAA", "close", 3)

    result = run(on_bar)
    tape = result.decisions[-1].inputs
    assert tape["prices"]["AAA"] is not None
    assert tape["history"]["AAA.close"]["n"] == 3


# --- the optimistic fill warning ----------------------------------------------


def test_same_close_fills_are_flagged_as_optimistic():
    result = run(fills={"mode": "same_close", "slippage_bps": 0.0})

    assert result.metrics["optimistic_fills"] is True
    assert result.metrics["fill_mode"] == "same_close"
    assert result.warnings and "optimistic" in result.warnings[0].lower()
    assert result.metrics["warnings"] == result.warnings
    assert result.fills[0].at == ENTRY, "same_close fills on the deciding bar"


def test_next_open_fills_are_not_flagged():
    result = run()
    assert result.metrics["optimistic_fills"] is False
    assert result.warnings == []


def test_result_to_json_is_small_and_serializable():
    result = run()
    payload = result.to_json()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["run_id"] == result.run_id
    assert payload["n_decisions"] == N
    assert payload["n_trades"] == 1
    assert "equity" not in payload, "the JSON payload carries numbers, not curves"


# --- config -------------------------------------------------------------------


def test_config_from_yaml_rejects_an_unknown_key(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("strategy: probe\ntickers: [AAA]\ntikcers: [AAA]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        BacktestConfig.from_yaml(path)

    with pytest.raises(FileNotFoundError):
        BacktestConfig.from_yaml(tmp_path / "missing.yaml")


def test_config_normalizes_tickers_dates_and_universe_files(tmp_path: Path):
    universe = tmp_path / "u.txt"
    universe.write_text("aapl\n# a comment\n\nmsft  # trailing\n", encoding="utf-8")
    assert load_universe(universe) == ["AAPL", "MSFT"]

    cfg = BacktestConfig.from_mapping(
        {"strategy": "probe", "universe": str(universe), "start": "2024-01-01", "end": "2024-06-01"}
    )
    assert cfg.tickers == ["AAPL", "MSFT"]
    assert cfg.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert cfg.to_dict()["end"] == "2024-06-01T00:00:00+00:00"
    assert json.loads(json.dumps(cfg.to_dict()))["tickers"] == ["AAPL", "MSFT"]


def test_config_overrides_win_over_the_file(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("strategy: probe\ntickers: [AAA]\ncash: 1000\n", encoding="utf-8")
    cfg = BacktestConfig.from_yaml(path, {"cash": 5_000.0, "notes": None})
    assert cfg.cash == 5_000.0
    assert cfg.notes == "", "a None override is ignored, not applied"


# --- metrics: known answers ---------------------------------------------------


def curve(values: list[float], *, start: str = "2024-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="equity")


#: Returns +2%, -1%, +2%, -1% -- hand-computable in full.
ALTERNATING = curve([100.0, 102.0, 100.98, 102.9996, 101.969604])


def test_sharpe_and_sortino_have_known_answers():
    m = M.compute_metrics(ALTERNATING, [], periods_per_year=4)

    # mean 0.005, stdev(ddof=1) 0.0173205 -> 0.005/0.0173205*sqrt(4)
    assert m["sharpe"] == pytest.approx(0.57735, abs=1e-5)
    # downside deviation sqrt(mean([0,0.01,0,0.01]^2)) = 0.00707107
    assert m["sortino"] == pytest.approx(1.414214, abs=1e-5)
    assert m["volatility"] == pytest.approx(0.0173205 * 2, abs=1e-5)
    assert m["downside_deviation"] == pytest.approx(0.00707107 * 2, abs=1e-5)
    assert m["best_period"] == pytest.approx(0.02, abs=1e-9)
    assert m["worst_period"] == pytest.approx(-0.01, abs=1e-9)
    assert m["periods"] == 5
    assert m["periods_per_year"] == 4


def test_a_flat_curve_scores_zero_rather_than_dividing_by_zero():
    m = M.compute_metrics(curve([100.0] * 10), [])
    assert m["sharpe"] == 0.0
    assert m["sortino"] == 0.0
    assert m["volatility"] == 0.0
    assert m["calmar"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["total_return"] == 0.0


def test_max_drawdown_and_calmar():
    m = M.compute_metrics(ALTERNATING, [], periods_per_year=4)
    assert m["max_drawdown"] == pytest.approx(-0.01, abs=1e-9)
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.01, rel=1e-6)
    assert m["peak_equity"] == pytest.approx(102.9996)
    assert m["final_equity"] == pytest.approx(101.969604)


def test_cagr_over_exactly_four_years():
    idx = pd.DatetimeIndex(["2020-01-01", "2024-01-01"], tz="UTC")  # 1461 days = 4.0 years
    m = M.compute_metrics(pd.Series([100.0, 200.0], index=idx), [])
    assert m["total_return"] == pytest.approx(1.0)
    assert m["cagr"] == pytest.approx(2 ** 0.25 - 1, abs=1e-6)
    assert m["days"] == pytest.approx(1461.0)
    assert m["start"].startswith("2020-01-01")


def test_a_total_loss_is_reported_as_minus_one_hundred_percent():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"], tz="UTC")
    m = M.compute_metrics(pd.Series([100.0, 0.0], index=idx), [])
    assert m["total_return"] == pytest.approx(-1.0)
    assert m["cagr"] == -1.0
    assert m["max_drawdown"] == pytest.approx(-1.0)


def test_drawdown_series_tracks_the_running_peak():
    dd = M.drawdown_series(curve([100.0, 120.0, 60.0, 90.0, 150.0]))
    assert list(dd.round(6)) == [0.0, 0.0, -0.5, -0.25, 0.0]
    assert M.drawdown_series(pd.Series(dtype="float64")).empty


def test_drawdown_duration_counts_calendar_days_underwater():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-10"], tz="UTC")
    # Peak-to-recovery: the peak is Jan 1, recovery is Jan 10, so nine days --
    # not eight. Measuring from the first bar *below* the peak would silently
    # under-report every drawdown by one bar.
    recovered = pd.Series([100.0, 90.0, 95.0, 105.0], index=idx)
    assert M.max_drawdown_duration_days(recovered) == pytest.approx(9.0)

    # An unrecovered drawdown runs to the end of the curve.
    unrecovered = pd.Series([100.0, 90.0, 95.0, 99.0], index=idx)
    assert M.max_drawdown_duration_days(unrecovered) == pytest.approx(9.0)

    assert M.max_drawdown_duration_days(curve([100.0, 101.0, 102.0])) == 0.0


def test_periods_per_year_follows_the_trading_calendar():
    assert M.periods_per_year("1d") == 252
    # A regular session is 6.5 hours, so hourly bars annualize at 252 * 6.5.
    # Truncating to 6 would make the 1h factor the only one that disagrees with
    # the minutes-elapsed arithmetic the other timeframes use.
    assert M.periods_per_year("1h") == round(252 * 390 / 60)
    assert M.periods_per_year("15m") == 252 * 26
    assert M.periods_per_year("5m") == 252 * 78


def trade(pnl: float, *, bars: int = 3, ticker: str = "AAA") -> Trade:
    return Trade(
        ticker=ticker,
        side="long",
        qty=10,
        entry_time=TS[0],
        entry_price=100.0,
        exit_time=TS[1],
        exit_price=100.0 + pnl / 10,
        pnl=pnl,
        pnl_pct=pnl / 1_000.0,
        bars_held=bars,
    )


def test_summarize_trades_has_known_answers():
    s = M.summarize_trades([trade(100.0, bars=2), trade(-50.0, bars=4), trade(25.0, bars=6)])

    assert s["trades"] == 3 and s["wins"] == 2 and s["losses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert s["avg_win"] == pytest.approx(62.5)
    assert s["avg_loss"] == pytest.approx(-50.0)
    assert s["gross_profit"] == pytest.approx(125.0)
    assert s["gross_loss"] == pytest.approx(50.0)
    assert s["profit_factor"] == pytest.approx(2.5)
    assert s["avg_trade_pnl"] == pytest.approx(25.0)
    assert s["best_trade"] == 100.0 and s["worst_trade"] == -50.0
    assert s["avg_bars_held"] == pytest.approx(4.0)
    assert s["expectancy"] == pytest.approx(2 / 3 * 62.5 + 1 / 3 * -50.0, abs=1e-6)


def test_a_breakeven_trade_counts_as_a_loss_not_a_win():
    s = M.summarize_trades([trade(0.0)])
    assert s["wins"] == 0 and s["losses"] == 1 and s["hit_rate"] == 0.0


def test_open_trades_are_excluded_from_the_summary():
    open_trade = trade(500.0)
    open_trade.exit_time = None
    assert M.summarize_trades([open_trade])["trades"] == 0
    assert M.summarize_trades([])["profit_factor"] == 0.0


def test_every_metric_survives_json_dumps_even_with_no_losers():
    """``profit_factor`` is the one value that can go infinite; JSON cannot."""
    m = M.compute_metrics(ALTERNATING, [trade(10.0), trade(20.0)], periods_per_year=4)

    assert m["profit_factor"] == 1e9, "infinity is clamped, not emitted"
    encoded = json.dumps(m)
    assert json.loads(encoded) == m
    for key, value in m.items():
        assert value is None or isinstance(value, (int, float, str, bool, list)), key
        if isinstance(value, float):
            assert math.isfinite(value), key


def test_empty_equity_still_returns_the_full_metric_shape():
    m = M.compute_metrics(pd.Series(dtype="float64"), [])
    assert m["periods"] == 0 and m["start"] is None
    assert json.loads(json.dumps(m)) == m
    assert {"sharpe", "sortino", "max_drawdown", "cagr", "exposure", "turnover"} <= set(m)


def test_exposure_and_turnover_come_from_the_run():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([100.0] * 5, index=idx)
    m = M.compute_metrics(
        equity, [], exposure=pd.Series([0.0, 0.5, 1.0, 0.5, 0.0], index=idx),
        turnover_notional=400.0,
    )
    assert m["exposure"] == pytest.approx(0.4)
    # 400 notional on a 100 book over 4 days, annualized.
    assert m["turnover"] == pytest.approx(4.0 / (4 / 365.25), rel=1e-6)


def test_a_benchmark_adds_relative_metrics():
    # The benchmark has to actually vary: beta is cov/var, so a constant-return
    # benchmark makes it 0/0 and the metric is correctly withheld (see the
    # degenerate case below). Here every equity return is exactly 2x the
    # benchmark's, which pins beta at 2 by construction.
    bench = curve([100.0, 104.0, 101.92, 108.0352, 104.794144])
    equity = curve([100.0, 108.0, 103.68, 116.1216, 109.154304])
    m = M.compute_metrics(equity, [], periods_per_year=252, benchmark=bench)

    # Metrics are rounded to 6dp on the way out so the JSON stays clean, so
    # compare on an absolute tolerance rather than approx's default relative one.
    assert m["benchmark_total_return"] == pytest.approx(0.04794144, abs=1e-6)
    assert m["excess_return"] == pytest.approx(0.09154304 - 0.04794144, abs=1e-6)
    assert m["beta"] == pytest.approx(2.0, abs=1e-6), "every move is twice the benchmark"
    assert "alpha" in m


def test_a_flat_benchmark_withholds_beta_rather_than_inventing_one():
    equity = curve([100.0, 110.0, 121.0, 133.1])
    flat = curve([100.0, 105.0, 110.25, 115.7625])  # constant 5%, zero variance
    m = M.compute_metrics(equity, [], periods_per_year=252, benchmark=flat)

    assert m["benchmark_total_return"] == pytest.approx(0.157625, abs=1e-6)
    assert "beta" not in m, "beta against a zero-variance benchmark is undefined"


def test_equity_frame_pairs_the_curve_with_its_drawdown():
    frame = M.equity_frame(curve([100.0, 120.0, 60.0]))
    assert list(frame.columns) == ["equity", "drawdown"]
    assert frame["drawdown"].iloc[-1] == pytest.approx(-0.5)


def test_is_oos_split_shades_the_out_of_sample_windows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    flags = M.is_oos_split(
        idx, [{"oos_start": "2024-01-03", "oos_end": "2024-01-04"}]
    )
    assert flags == [False, False, True, True, False]
    assert M.is_oos_split(idx, None) == [False] * 5


def test_two_bar_sources_for_one_ticker_is_refused_not_silently_mixed(seeded_store_factory):
    """The bug this pins cost a whole research session.

    The store keeps one row per (source, ticker, timeframe, event_time) so a
    bootstrap feed and a real pull can coexist. A *reader* that ignores the
    source interleaves them into a price series that jumps between two unrelated
    universes -- and the backtest over it produces confident numbers about
    nothing. One real session concluded a strategy beat SPY because mixed bars
    gave SPY a negative Sharpe.
    """
    from lab.engine.context import available_bar_sources, require_one_bar_source
    from lab.store import parquet_io
    from lab.store.schema import Bar

    info = seeded_store_factory(tickers=["AAA"], days=120)
    assert available_bar_sources(["AAA"]) == ["synthetic"]
    # Unambiguous: resolves silently.
    assert require_one_bar_source(["AAA"]) == "synthetic"

    # Now a second source for the same ticker, at wildly different prices.
    rows = parquet_io.read_bars(["AAA"], timeframe="1d", source="synthetic")
    parquet_io.write_bars(
        Bar(
            event_time=r.event_time,
            knowledge_time=r.knowledge_time,
            source="alpaca",
            ticker="AAA",
            timeframe="1d",
            open=float(r.open) * 10,
            high=float(r.high) * 10,
            low=float(r.low) * 10,
            close=float(r.close) * 10,
            volume=float(r.volume),
        )
        for r in rows.head(50).itertuples()
    )

    assert available_bar_sources(["AAA"]) == ["alpaca", "synthetic"]
    with pytest.raises(ValueError) as exc:
        require_one_bar_source(["AAA"])
    message = str(exc.value)
    assert "more than one source" in message
    assert "alpaca" in message and "synthetic" in message
    assert "source:" in message, "say how to fix it"

    # A backtest must refuse rather than trade the interleaved series.
    cfg = BacktestConfig(strategy="strategies/buy_and_hold.py", tickers=["AAA"], timeframe="1d")
    with pytest.raises(ValueError, match="more than one source"):
        run_backtest(cfg, register=False, journal=False)

    # Naming the source resolves it.
    cfg.source = "synthetic"
    result = run_backtest(cfg, register=False, journal=False)
    assert result.metrics["periods"] > 0
    assert result.metrics["ledger_residual"] == 0.0

def test_intraday_reads_drop_extended_hours_bars(paths):
    """Alpaca returns pre-market and after-hours bars by default.

    They are thin enough that any fill model misprices them, there are more per
    day than the annualization factor assumes, and the live runner never fires
    on them -- so a backtest that keeps them is measuring a different instrument
    than the one it would trade.
    """
    import pandas as pd

    from lab.engine.context import DataView, regular_hours_only
    from lab.store import parquet_io
    from lab.store.schema import Bar

    # One session of 15m bars from 08:00 to 17:00 ET, i.e. with wings.
    day = pd.Timestamp("2024-07-02", tz="America/New_York")
    stamps = pd.date_range(day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=17),
                           freq="15min", tz="America/New_York")
    parquet_io.write_bars(
        Bar(event_time=ts.tz_convert("UTC"), knowledge_time=ts.tz_convert("UTC"),
            source="probe", ticker="AAA", timeframe="15m",
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)
        for ts in stamps
    )

    view = DataView.from_store(["AAA"], timeframe="15m", bar_source="probe")
    et = pd.DatetimeIndex(view.frame("AAA").index).tz_convert("America/New_York")
    assert len(et) == 26, "a regular 6.5h session is exactly 26 fifteen-minute bars"
    assert et.min().strftime("%H:%M") == "09:45", "first bar closes after the open"
    assert et.max().strftime("%H:%M") == "16:00", "last bar closes at the bell"

    # Opt out and the wings come back.
    wide = DataView.from_store(["AAA"], timeframe="15m", bar_source="probe",
                               regular_hours=False)
    assert len(wide.frame("AAA")) == len(stamps)

    # Daily bars are untouched by the filter.
    daily = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-07-02 20:00", tz="UTC")]),
    )
    assert len(regular_hours_only(daily)) == 1


def test_every_from_store_call_site_passes_the_bar_source():
    """A guard is only as good as its least-careful caller.

    The ambiguity check lives in ``DataView.from_store``, so a call site that
    omits ``bar_source`` does not bypass it -- it *trips* it, and the run dies
    with a confusing error even though its config named a source. One such site
    (the author loop's timestamp probe) shipped that way and only surfaced when a
    console-launched session failed.
    """
    import ast
    from pathlib import Path as _Path

    from lab.config import get_settings

    root = _Path(get_settings().paths.root) / "lab"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "from_store"):
                continue
            names = {kw.arg for kw in node.keywords}
            if "bar_source" not in names:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"from_store without bar_source: {', '.join(offenders)}"


def test_warmup_bars_are_excluded_from_the_curve_and_the_comparison(seeded_store_factory):
    """Warm-up is not part of the experiment, so it must not be part of the maths.

    The strategy is structurally unable to trade through those bars. Leaving them
    in the curve hands the benchmark a free head start (it compounds while the
    strategy is pinned flat), stretches the CAGR denominator, and pads the return
    series with zeros. On one real run, 211 dead bars gave SPY +61% before the
    strategy was allowed a single order, turning a +217pp edge into +115pp.
    """
    info = seeded_store_factory(tickers=["AAA", "BBB"], days=400)
    warmup = 60

    cfg = BacktestConfig(
        strategy="strategies/buy_and_hold.py",
        tickers=["AAA", "BBB"],
        timeframe="1d",
        source="synthetic",
        benchmark="AAA",
        warmup=warmup,
        cash=100_000.0,
    )
    result = run_backtest(cfg, register=False, journal=False)
    m = result.metrics

    assert m["warmup_bars"] == warmup
    assert m["tradeable_bars"] == len(result.equity)
    assert m["periods"] == len(result.equity)

    # The curve begins where trading could begin, so it never opens with a flat
    # stretch the strategy had no say in.
    assert result.equity.iloc[0] == pytest.approx(100_000.0)
    assert result.equity.iloc[1] != pytest.approx(result.equity.iloc[-1]) or len(result.equity) < 3

    # And the benchmark is measured over exactly that window, not from bar zero.
    from lab.store import parquet_io

    bars = parquet_io.read_bars(["AAA"], timeframe="1d", source="synthetic")
    closes = bars.set_index("event_time")["close"].astype("float64")
    over_curve = closes.reindex(result.equity.index).ffill().dropna()
    expected = float(over_curve.iloc[-1] / over_curve.iloc[0] - 1.0)
    assert m["benchmark_total_return"] == pytest.approx(expected, abs=1e-4)

    # A warm-up that eats the whole range is an error, not an empty curve.
    with pytest.raises(ValueError, match="consumed every one of"):
        run_backtest(
            BacktestConfig(
                strategy="strategies/buy_and_hold.py", tickers=["AAA"], timeframe="1d",
                source="synthetic", warmup=10_000,
            ),
            register=False, journal=False,
        )


def test_no_consumer_reads_bars_without_naming_a_source():
    """Every reader that feeds a chart or a metric must pin its source.

    Three separate call sites shipped without it -- the backtest itself, the
    author loop's timestamp probe, and the run-detail price chart -- and each one
    silently interleaved two unrelated price series. `parquet_io.read_bars` is a
    raw reader and stays unopinionated; the consumers are what must be careful.
    """
    import ast
    from pathlib import Path as _Path

    from lab.config import get_settings

    root = _Path(get_settings().paths.root) / "lab"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "parquet_io.py":
            continue  # the reader itself
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name != "read_bars":
                continue
            if "source" not in {kw.arg for kw in node.keywords}:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"read_bars without source: {', '.join(offenders)}"
