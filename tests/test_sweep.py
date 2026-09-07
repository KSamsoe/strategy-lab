"""Sweep and walk-forward tests, weighted toward the overfitting defenses.

The interesting assertions here are not "the code runs" but "the code refuses to
flatter you": survivors ranked out-of-sample, both halves reported, the attempt
total carried, and a truncated grid that says so. The sweeps below drive the
real event-driven engine against the synthetic bars already in the store, so a
regression in the runner surfaces here too.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

from lab.backtest.runner import BacktestConfig
from lab.backtest.sweep import SweepConfig, expand_grid, fast_screen, run_sweep
from lab.backtest.walkforward import (
    Window,
    make_windows,
    parse_spec,
    run_walk_forward,
    spans_mask,
    stitch,
)
from lab.registry.runs import RunRegistry
from lab.timeutil import to_utc, trading_days

PROJECT = Path(__file__).resolve().parents[1]
STRATEGY = str(PROJECT / "strategies" / "buy_and_hold.py")

TICKERS = ["AAA", "BBB", "CCC"]

#: ``seed_store`` writes bars from 2 Jan 2024 over ``days * 1.5`` calendar days,
#: so the tape below is a fixed ~3-year span of ~775 sessions -- long enough for
#: "4:1" to roll the full five windows with blocks that are still an evaluation
#: rather than noise. Seed 3 is not arbitrary: on that tape the in-sample winner
#: of the rebalance grid is *not* the out-of-sample winner, which is the
#: disagreement the ranking tests exist to catch.
SEED, DAYS = 3, 750
START, END = "2024-01-02", "2027-01-31"
HALF_YEAR, ONE_YEAR, TWO_YEARS = "2024-07-01", "2025-01-01", "2026-01-01"


@pytest.fixture()
def lab_env(seeded_store_factory) -> dict:
    """Deterministic synthetic bars in the isolated store.

    ``conftest.isolated_paths`` is autouse and re-points ``data/`` and ``runs/``
    at a per-test tmpdir, so the developer's real store is invisible and anything
    that reads bars has to seed its own. Per test rather than per module for the
    same reason the registry is isolated at all: a sweep's whole point is the
    attempt counter, and a registry shared with the neighbouring tests would make
    that number unassertable.
    """
    return seeded_store_factory(tickers=TICKERS, days=DAYS, seed=SEED)


def base_config(start: str = START, end: str = END, **kw) -> BacktestConfig:
    return BacktestConfig(strategy=STRATEGY, tickers=TICKERS, start=start, end=end, **kw)


@pytest.fixture()
def wf_sweep(lab_env) -> dict:
    """One real walk-forward sweep, read by every test below that needs one."""
    cfg = SweepConfig(
        base=base_config(),
        grid={"rebalance_days": [0, 10, 40]},
        walk_forward="4:1",
        metric="sharpe",
    )
    return run_walk_forward(cfg)


# --- expand_grid --------------------------------------------------------------


def test_expand_grid_cardinality_and_ordering():
    grid = {"a": [1, 2], "b": ["x", "y", "z"], "c": [True, False]}
    combos = expand_grid(grid)

    assert len(combos) == 2 * 3 * 2
    assert len({tuple(sorted(c.items(), key=str)) for c in combos}) == len(combos)
    assert all(list(c) == ["a", "b", "c"] for c in combos)
    # Last axis varies fastest, so a truncated grid is a coherent prefix rather
    # than an arbitrary scatter through the space.
    assert combos[0] == {"a": 1, "b": "x", "c": True}
    assert combos[1] == {"a": 1, "b": "x", "c": False}
    assert combos[2] == {"a": 1, "b": "y", "c": True}
    assert combos[-1] == {"a": 2, "b": "z", "c": False}


def test_expand_grid_edges():
    assert expand_grid({}) == [{}]
    # A scalar axis is one value, and a string is a value, not three characters.
    assert expand_grid({"n": 5, "mode": "next_open"}) == [{"n": 5, "mode": "next_open"}]
    with pytest.raises(ValueError, match="empty"):
        expand_grid({"n": []})


# --- make_windows -------------------------------------------------------------


def test_parse_spec_forms():
    assert parse_spec("4:1") == (4, 1)
    assert parse_spec("3:1") == (3, 1)
    assert parse_spec("6:2") == (6, 2)
    assert parse_spec(4) == (4, 1)
    assert parse_spec("4") == (4, 1)
    for bad in ("", "4:0", "0:1", "-3:1", "four:one", "4:1:1", True):
        with pytest.raises(ValueError):
            parse_spec(bad)


@pytest.mark.parametrize("spec,train,test", [("4:1", 4, 1), ("3:1", 3, 1), ("6:2", 6, 2), (4, 4, 1)])
def test_make_windows_contiguous_disjoint_and_covering(spec, train, test):
    start, end = to_utc("2018-01-01"), to_utc("2026-01-01")
    windows = make_windows(start, end, spec)
    days = trading_days(start.date(), end.date())

    assert len(windows) >= 1
    assert [w.index for w in windows] == list(range(len(windows)))

    for w in windows:
        assert w.is_start < w.is_end < w.oos_start < w.oos_end
        # No bar may be in both halves of the same window.
        assert w.is_end < w.oos_start

    covered: list = []
    for prev, nxt in zip(windows, windows[1:]):
        assert nxt.oos_start > prev.oos_end  # disjoint
        gap = [d for d in days if prev.oos_end < to_utc(d) + pd.Timedelta(hours=12) < nxt.oos_start]
        assert gap == []  # and contiguous: no trading day falls between them
        assert nxt.is_start > prev.is_start  # the schedule rolls forward

    # Coverage: the first window's training data starts at the range start, the
    # last window's test data ends at the range end, and every trading day in
    # between lands in exactly one OOS span or in the initial training block.
    assert windows[0].is_start.date() == days[0]
    assert windows[-1].oos_end.date() == days[-1]
    for d in days:
        ts = to_utc(d) + pd.Timedelta(hours=12)
        hits = sum(1 for w in windows if w.oos_start <= ts <= w.oos_end)
        assert hits <= 1
        if hits == 0:
            assert windows[0].is_start <= ts <= windows[0].is_end
        covered.append(hits)
    assert sum(covered) > 0

    # The IS:OOS ratio the spec asked for, within one block of rounding.
    w0 = windows[0]
    is_days = len(trading_days(w0.is_start.date(), w0.is_end.date()))
    oos_days = len(trading_days(w0.oos_start.date(), w0.oos_end.date()))
    assert is_days / oos_days == pytest.approx(train / test, rel=0.15)


def test_make_windows_rejects_unusable_specs():
    with pytest.raises(ValueError, match="fewer than one usable window"):
        make_windows("2019-01-01", "2019-01-05", "4:1")
    with pytest.raises(ValueError, match="fewer than one usable window"):
        make_windows("2019-01-01", "2022-01-01", "4:0")
    with pytest.raises(ValueError, match="empty"):
        make_windows("2022-01-01", "2019-01-01", "4:1")
    with pytest.raises(ValueError):
        make_windows("2019-01-01", "2022-01-01", "not-a-spec")
    with pytest.raises(ValueError):
        make_windows("2019-01-01", "2022-01-01", "4:1", n_windows=500)


def test_window_roundtrips_through_dict():
    w = make_windows("2019-01-01", "2022-01-01", "4:1")[0]
    assert Window.from_dict(w.to_dict()) == w


# --- the real sweep -----------------------------------------------------------


def test_sweep_runs_the_real_engine_once_per_combination(wf_sweep):
    rows = wf_sweep["runs"]
    assert len(rows) == 3
    assert wf_sweep["combinations"] == 3
    assert all(r["status"] == "ok" for r in rows), [r["error"] for r in rows]

    run_ids = [r["run_id"] for r in rows]
    assert len(set(run_ids)) == 3

    registered = RunRegistry().list(sweep_id=wf_sweep["sweep_id"], limit=50)
    assert sorted(r.run_id for r in registered) == sorted(run_ids)
    assert {r.sweep_id for r in registered} == {wf_sweep["sweep_id"]}
    assert {r.status for r in registered} == {"ok"}
    # Each registered run carries the params it was actually run with.
    by_id = {r.run_id: r for r in registered}
    for row in rows:
        assert by_id[row["run_id"]].params["rebalance_days"] == row["params"]["rebalance_days"]


def test_sweep_counts_the_attempts(wf_sweep):
    registry = RunRegistry()
    assert wf_sweep["attempts"] == registry.family_attempts(wf_sweep["strategy"])
    assert wf_sweep["attempts"] >= wf_sweep["attempts_before"] + len(wf_sweep["runs"])


def test_every_row_reports_in_and_out_of_sample_side_by_side(wf_sweep):
    for row in wf_sweep["runs"]:
        assert isinstance(row["is_sharpe"], float)
        assert isinstance(row["oos_sharpe"], float)
        for key in ("sharpe", "total_return", "max_drawdown", "cagr", "bars"):
            assert f"is_{key}" in row and f"oos_{key}" in row
        # The two halves partition the run: no bar counted twice, none dropped.
        assert row["is_bars"] + row["oos_bars"] == row["metrics"]["periods"]
        assert row["is_sharpe"] != row["oos_sharpe"]


def test_walk_forward_reports_every_window(wf_sweep):
    windows = wf_sweep["walk_forward"]
    assert len(windows) == len(wf_sweep["windows"]) >= 2
    for entry in windows:
        assert entry["n_combinations"] == 3
        assert entry["selected"] in [r["params"] for r in wf_sweep["runs"]]
        assert isinstance(entry["is_sharpe"], float)
        assert isinstance(entry["oos_sharpe"], float)
        assert entry["oos_bars"] > 0
        # Per-window scores for every combination, both halves.
        assert len(entry["scores"]) == 3
        for s in entry["scores"]:
            assert "is_sharpe" in s and "oos_sharpe" in s

    summary = wf_sweep["walk_forward_summary"]
    assert summary["windows"] == len(windows)
    assert "mean_oos_sharpe" in summary and "degradation" in summary


def test_best_is_selected_out_of_sample(wf_sweep):
    rows = wf_sweep["runs"]
    assert wf_sweep["ranked_on"] == "out_of_sample"
    assert wf_sweep["ranked_on_metric"] == "oos_sharpe"
    assert wf_sweep["validated_out_of_sample"] is True

    best_oos = max(rows, key=lambda r: r["oos_sharpe"])
    best_is = max(rows, key=lambda r: r["is_sharpe"])
    assert wf_sweep["best"]["run_id"] == best_oos["run_id"]
    assert wf_sweep["best"]["score_key"] == "oos_sharpe"
    assert [r["rank"] for r in rows] == [1, 2, 3]  # rows come back ranked

    # The defense only means something if the two disagree on this data, and on
    # these three grid points they do: the in-sample winner is not the pick.
    assert best_is["run_id"] != best_oos["run_id"]
    assert wf_sweep["best"]["run_id"] != best_is["run_id"]


def test_ranking_on_an_in_sample_metric_is_marked_as_unvalidated(lab_env):
    cfg = SweepConfig(
        base=base_config(),
        grid={"rebalance_days": [0, 10, 40]},
        walk_forward="4:1",
        metric="is_sharpe",
    )
    res = run_sweep(cfg)

    assert res["ranked_on"] == "in_sample"
    assert res["ranked_on_metric"] == "is_sharpe"
    assert res["validated_out_of_sample"] is False
    assert any("IN-SAMPLE" in n for n in res["notes"])
    # It still works -- it just cannot be presented as validated.
    assert res["best"]["run_id"] == max(res["runs"], key=lambda r: r["is_sharpe"])["run_id"]
    assert res["best"]["run_id"] != max(res["runs"], key=lambda r: r["oos_sharpe"])["run_id"]
    assert all(isinstance(r["oos_sharpe"], float) for r in res["runs"])


def test_sweep_without_walk_forward_is_in_sample_and_nulls_the_oos_half(lab_env):
    cfg = SweepConfig(base=base_config(START, ONE_YEAR), grid={"rebalance_days": [0, 20]})
    res = run_sweep(cfg)

    assert res["ranked_on"] == "in_sample"
    assert res["validated_out_of_sample"] is False
    assert res["walk_forward"] == [] and res["windows"] == []
    for row in res["runs"]:
        assert row["is_sharpe"] == row["metrics"]["sharpe"]
        # Present and null, not absent: a missing key invites a silent fallback
        # to the in-sample number somewhere downstream.
        assert "oos_sharpe" in row and row["oos_sharpe"] is None


def test_max_runs_truncation_is_recorded_and_logged(lab_env, caplog):
    cfg = SweepConfig(
        base=base_config(START, HALF_YEAR),
        grid={"rebalance_days": [0, 10, 40, 120]},
        max_runs=2,
    )
    with caplog.at_level(logging.WARNING, logger="lab.backtest.sweep"):
        res = run_sweep(cfg)

    assert res["truncated"] is True
    assert res["grid_size"] == 4
    assert res["combinations"] == 2
    assert res["max_runs"] == 2
    assert len(res["runs"]) == 2
    assert res["selection"] == "grid_order"
    assert any("TRUNCATED" in n and "max_runs=2" in n for n in res["notes"])
    assert any("TRUNCATED" in rec.message for rec in caplog.records)


def test_run_walk_forward_refuses_without_a_spec(lab_env):
    cfg = SweepConfig(base=base_config(), grid={"rebalance_days": [0]})
    with pytest.raises(ValueError, match="walk_forward"):
        run_walk_forward(cfg)


def test_sweep_artifact_is_written(wf_sweep):
    path = Path(wf_sweep["artifact_dir"]) / "sweep.json"
    assert path.exists()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sweep_id"] == wf_sweep["sweep_id"]
    assert payload["ranked_on"] == "out_of_sample"


# --- workers ------------------------------------------------------------------


def test_unpicklable_payload_degrades_to_serial(lab_env):
    cfg = SweepConfig(
        base=base_config(START, HALF_YEAR, params={"unpicklable": lambda x: x}),
        grid={"rebalance_days": [0, 20]},
    )
    res = run_sweep(cfg, workers=2)

    assert [r["status"] for r in res["runs"]] == ["ok", "ok"]
    assert any("will not pickle" in n and "serially" in n for n in res["notes"])


def _spawn_safe() -> bool:
    """multiprocessing spawn re-imports ``__main__`` in the child; skip the pool
    test when that would not be a plain importable module."""
    main = sys.modules.get("__main__")
    if getattr(main, "__spec__", None) is not None:
        return True
    return str(getattr(main, "__file__", "") or "").endswith(".py")


@pytest.mark.skipif(not _spawn_safe(), reason="__main__ cannot be re-imported by a spawned worker")
def test_process_pool_path_produces_the_same_shape_as_serial(lab_env):
    cfg = SweepConfig(
        base=base_config(START, HALF_YEAR), grid={"rebalance_days": [0, 20]}
    )
    parallel = run_sweep(cfg, workers=2)
    serial = run_sweep(cfg, workers=1)

    assert not any("serially" in n for n in parallel["notes"])
    got = {tuple(r["params"].items()): r["metrics"]["sharpe"] for r in parallel["runs"]}
    want = {tuple(r["params"].items()): r["metrics"]["sharpe"] for r in serial["runs"]}
    assert got == want  # same engine, same numbers, whoever ran it


# --- fast_screen --------------------------------------------------------------


def test_fast_screen_works_without_vectorbt(lab_env):
    pytest.importorskip("pandas")
    try:
        import vectorbt  # noqa: F401

        pytest.skip("vectorbt is installed; this test covers the numpy fallback")
    except ImportError:
        pass

    cfg = SweepConfig(
        base=base_config(START, TWO_YEARS),
        grid={"lookback": [21, 63], "top_n": [1, 2]},
    )
    df = fast_screen(cfg)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert df.attrs["engine"] == "numpy"
    assert df.attrs["discriminates"] is True
    # The frame states its own status on every row, not just in the docstring.
    assert (df["confirmed"] == False).all()  # noqa: E712 - column-wise comparison
    assert df["caveat"].str.contains("event-driven engine").all()
    assert "COARSE SCREEN ONLY" in df.attrs["caveat"]
    assert list(df["screen_rank"]) == [1, 2, 3, 4]
    assert df["screen_sharpe"].is_monotonic_decreasing
    assert df["screen_sharpe"].nunique() > 1  # the knobs actually moved something


def test_fast_screen_says_when_it_cannot_see_the_grid(lab_env):
    cfg = SweepConfig(
        base=base_config(START, ONE_YEAR), grid={"atr_stop_mult": [2.0, 3.0]}
    )
    df = fast_screen(cfg)
    assert df.attrs["discriminates"] is False
    assert df.attrs["ignored_knobs"] == ["atr_stop_mult"]
    assert df["screen_sharpe"].nunique() == 1


# --- segment arithmetic -------------------------------------------------------


def test_stitch_does_not_earn_the_return_across_an_excluded_gap():
    idx = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 121.0, 1000.0, 1100.0, 1210.0], index=idx)
    spans = [(idx[0].to_pydatetime(), idx[2].to_pydatetime()),
             (idx[3].to_pydatetime(), idx[5].to_pydatetime())]

    curve = stitch(equity, spans_mask(idx, spans))

    assert len(curve) == 6
    # Four 10% steps, and the 121 -> 1000 jump that happened in the excluded
    # stretch is not one of them.
    assert float(curve.iloc[-1]) == pytest.approx(100.0 * 1.1**4)
    assert float(curve.iloc[3]) == pytest.approx(121.0)


def test_split_metrics_partitions_the_curve(wf_sweep):
    row = wf_sweep["runs"][0]
    assert row["is_bars"] > 0 and row["oos_bars"] > 0
    # exposure and turnover are not restated per segment rather than reported
    # as a confident zero.
    assert "is_exposure" not in row and "oos_turnover" not in row
