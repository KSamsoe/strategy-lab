"""Tests for the indicator layer.

The causality tests are the point of this file. They walk ``INDICATORS`` rather
than a hand-written list, so an indicator added later cannot join the registry
without proving that its past does not move when the future arrives.
"""

from __future__ import annotations

import math
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from lab.config import reset_settings_cache
from lab.indicators import cache, fetched
from lab.indicators.computed import INDICATORS, available, compute

UTC = timezone.utc


# --- fixtures ----------------------------------------------------------------


def make_bars(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Deterministic GBM-ish OHLCV bars on a business-day UTC index."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    wick = np.abs(rng.normal(0.0, 0.006, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(100_000, 5_000_000, n).astype("float64")
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC", name="event_time")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture()
def lab_env(tmp_path, monkeypatch):
    """Point the lab's data root at a tmp dir for cache tests."""
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(tmp_path / "runs"))
    reset_settings_cache()
    cache.reset_stats()
    yield tmp_path
    reset_settings_cache()


ALL_INDICATORS = sorted(INDICATORS)


# --- causality ---------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_indicator_is_causal(name):
    """Appending future bars must not move a single past value, bit for bit."""
    bars = make_bars(300)
    cut = 200
    past_only = compute(name, bars.iloc[:cut])
    with_future = compute(name, bars)

    if isinstance(with_future, pd.DataFrame):
        pd.testing.assert_frame_equal(with_future.iloc[:cut], past_only, check_exact=True)
    else:
        pd.testing.assert_series_equal(with_future.iloc[:cut], past_only, check_exact=True)


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_warmup_is_nan_and_never_backfilled(name):
    bars = make_bars(300)
    out = compute(name, bars)
    frame = out.to_frame() if isinstance(out, pd.Series) else out
    assert list(frame.index) == list(bars.index)

    for col in frame.columns:
        s = frame[col]
        first = s.first_valid_index()
        assert first is not None, f"{name}.{col} produced nothing"
        assert first > bars.index[0], f"{name}.{col} has no warm-up; something is peeking"
        assert s.loc[:first].iloc[:-1].isna().all(), f"{name}.{col} back-filled its warm-up"
        assert not s.loc[first:].isna().any(), f"{name}.{col} has holes after warm-up"


@pytest.mark.parametrize("name", ALL_INDICATORS)
def test_indicator_is_deterministic(name):
    bars = make_bars(120)
    a, b = compute(name, bars), compute(name, bars)
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    else:
        pd.testing.assert_series_equal(a, b, check_exact=True)


def test_registry_covers_the_contract():
    required = {
        "sma", "ema", "wma", "rsi", "atr", "bbands", "macd", "roc", "momentum",
        "zscore", "rolling_vol", "donchian", "returns", "vwap", "adx", "stoch",
        "max_drawdown", "slope",
    }
    assert required <= set(available())


def test_frame_shaped_indicators_have_the_documented_columns():
    bars = make_bars(80)
    assert list(compute("bbands", bars).columns) == ["lower", "mid", "upper"]
    assert list(compute("macd", bars).columns) == ["macd", "signal", "hist"]
    assert list(compute("donchian", bars).columns) == ["upper", "lower"]


# --- correctness spot checks -------------------------------------------------


def frame_from_closes(closes) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.ones(len(c))}, index=idx
    )


def test_sma_matches_hand_computed():
    df = frame_from_closes([10, 11, 12, 13, 14])
    got = compute("sma", df, n=3)
    assert got.iloc[:2].isna().all()
    assert list(got.iloc[2:]) == [11.0, 12.0, 13.0]


def test_ema_matches_hand_computed_recursion():
    # alpha = 2/(3+1) = 0.5, seeded at the first observation, masked for 2 rows.
    df = frame_from_closes([10, 11, 12, 13, 14])
    got = compute("ema", df, n=3)
    assert got.iloc[:2].isna().all()
    assert got.iloc[2] == pytest.approx(11.25)
    assert got.iloc[3] == pytest.approx(12.125)
    assert got.iloc[4] == pytest.approx(13.0625)


def test_wma_matches_hand_computed():
    df = frame_from_closes([10, 11, 12])
    got = compute("wma", df, n=3)
    assert got.iloc[2] == pytest.approx((10 * 1 + 11 * 2 + 12 * 3) / 6.0)


def test_zscore_matches_hand_computed():
    df = frame_from_closes([10, 11, 12, 13, 14])
    got = compute("zscore", df, n=3)
    assert got.iloc[:2].isna().all()
    # every 3-bar window is a straight line: z = 1 / sqrt(2/3)
    expected = 1.0 / math.sqrt(2.0 / 3.0)
    assert got.iloc[2] == pytest.approx(expected)
    assert got.iloc[4] == pytest.approx(expected)


def wilder_reference(values: list[float], n: int) -> list[float]:
    """Independent Wilder smoother, written out longhand for the spot checks."""
    out = [float("nan")] * len(values)
    start = next((i for i, v in enumerate(values) if not math.isnan(v)), None)
    if start is None or start + n - 1 >= len(values):
        return out
    seed = start + n - 1
    acc = sum(values[start : seed + 1]) / n
    out[seed] = acc
    for i in range(seed + 1, len(values)):
        acc = (acc * (n - 1) + values[i]) / n
        out[i] = acc
    return out


def test_rsi_matches_an_independent_wilder_implementation():
    # Wilder's worked example, the series every TA text reprints.
    closes = [
        44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
        45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
        46.0328, 46.4116, 46.2222, 45.6439,
    ]
    df = frame_from_closes(closes)
    got = compute("rsi", df, n=14)

    deltas = [float("nan")] + [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if (not math.isnan(d) and d > 0) else (float("nan") if math.isnan(d) else 0.0) for d in deltas]
    losses = [-d if (not math.isnan(d) and d < 0) else (float("nan") if math.isnan(d) else 0.0) for d in deltas]
    avg_gain = wilder_reference(gains, 14)
    avg_loss = wilder_reference(losses, 14)
    expected = [
        float("nan") if math.isnan(g) else 100.0 - 100.0 / (1.0 + g / l)
        for g, l in zip(avg_gain, avg_loss)
    ]

    assert got.iloc[:14].isna().all()
    for i in range(14, len(closes)):
        assert got.iloc[i] == pytest.approx(expected[i], rel=1e-12)
    # ...and against the published readings for that series.
    assert got.iloc[14:].round(2).to_list() == [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]


def test_rsi_pins_at_100_with_no_losses_and_50_when_flat():
    rising = compute("rsi", frame_from_closes(list(range(10, 30))), n=14)
    assert rising.iloc[-1] == pytest.approx(100.0)
    flat = compute("rsi", frame_from_closes([50.0] * 20), n=14)
    assert flat.iloc[-1] == pytest.approx(50.0)


def test_atr_matches_hand_computed():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [9.5, 10.0, 11.0],
            "high": [10.0, 11.0, 12.0],
            "low": [9.0, 9.5, 10.5],
            "close": [9.5, 10.5, 11.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    # TR = [1.0, 1.5, 1.5]; ATR(2) seeds at the mean of the first two, then Wilder.
    got = compute("atr", df, n=2)
    assert math.isnan(got.iloc[0])
    assert got.iloc[1] == pytest.approx(1.25)
    assert got.iloc[2] == pytest.approx(1.375)


def test_returns_and_roc_differ_by_a_factor_of_100():
    df = frame_from_closes([100.0, 110.0, 121.0])
    assert compute("returns", df, n=1).iloc[1] == pytest.approx(0.10)
    assert compute("roc", df, n=1).iloc[1] == pytest.approx(10.0)
    assert compute("returns", df, n=1, log=True).iloc[1] == pytest.approx(math.log(1.1))
    assert compute("momentum", df, n=1).iloc[1] == pytest.approx(10.0)


def test_donchian_includes_the_current_bar():
    bars = make_bars(60)
    ch = compute("donchian", bars, n=10)
    assert ch["upper"].iloc[-1] == pytest.approx(bars["high"].iloc[-10:].max())
    assert ch["lower"].iloc[-1] == pytest.approx(bars["low"].iloc[-10:].min())


def test_max_drawdown_is_negative_and_bounded():
    dd = compute("max_drawdown", make_bars(200), n=20).dropna()
    assert (dd <= 0.0).all()
    assert (dd > -1.0).all()


def test_rolling_vol_annualizes_by_sqrt_periods():
    bars = make_bars(120)
    raw = compute("rolling_vol", bars, n=20, annualize=False)
    ann = compute("rolling_vol", bars, n=20, annualize=True)
    assert (ann.dropna() / raw.dropna()).round(9).unique().tolist() == [
        round(math.sqrt(252), 9)
    ]


# --- input validation --------------------------------------------------------


def test_compute_rejects_unknown_indicator_and_params():
    bars = make_bars(30)
    with pytest.raises(ValueError, match="unknown indicator"):
        compute("sharpe_ratio", bars)
    with pytest.raises(ValueError, match="no parameter"):
        compute("sma", bars, window=5)
    with pytest.raises(ValueError, match="must be >= 1"):
        compute("sma", bars, n=0)
    with pytest.raises(ValueError, match="shorter than slow"):
        compute("macd", bars, fast=26, slow=12)


def test_compute_rejects_unsorted_or_incomplete_frames():
    bars = make_bars(30)
    with pytest.raises(ValueError, match="sorted ascending"):
        compute("sma", bars.iloc[::-1])
    with pytest.raises(ValueError, match="no column"):
        compute("atr", bars.drop(columns=["high"]))
    with pytest.raises(ValueError, match="expected a DataFrame"):
        compute("sma", bars["close"])


# --- fetched: knowledge_time alignment ---------------------------------------


def bar_index(days: int = 7) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")


def events_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp(et, tz="UTC"),
                "knowledge_time": pd.Timestamp(kt, tz="UTC"),
                "source": "govgreed",
                "ticker": "AAPL",
                "score": score,
            }
            for et, kt, score in rows
        ]
    )


def test_align_places_events_by_knowledge_time_not_event_time():
    idx = bar_index()
    # Traded Jan 1, disclosed Jan 4: invisible for three bars.
    events = events_frame([("2024-01-01", "2024-01-04", 1.0)])
    out = fetched.align(events, idx, ffill_limit=0)

    assert out.iloc[:3].isna().all(), "an event leaked onto bars before its disclosure"
    assert out.iloc[3] == pytest.approx(1.0)
    assert out.iloc[4:].isna().all()


def test_align_rounds_forward_to_the_next_bar():
    idx = bar_index()
    # Known mid-session on Jan 2; the Jan 2 bar's timestamp already passed.
    events = events_frame([("2023-12-20", "2024-01-02 12:00", 2.0)])
    out = fetched.align(events, idx, ffill_limit=0)
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)


def test_align_drops_events_known_after_the_last_bar():
    idx = bar_index()
    events = events_frame([("2024-01-01", "2024-01-09", 9.0)])
    assert fetched.align(events, idx).isna().all()


def test_align_forward_fills_within_limit():
    idx = bar_index()
    events = events_frame(
        [("2023-12-01", "2024-01-03", 2.0), ("2023-12-15", "2024-01-04", 1.0)]
    )

    none_ = fetched.align(events, idx, ffill_limit=0).to_list()
    assert none_[2] == 2.0 and none_[3] == 1.0 and math.isnan(none_[4])

    one = fetched.align(events, idx, ffill_limit=1).to_list()
    assert one[4] == 1.0 and math.isnan(one[5])

    unlimited = fetched.align(events, idx, ffill_limit=None).to_list()
    assert unlimited[4] == unlimited[5] == unlimited[6] == 1.0


def test_align_aggregates_same_bar_events():
    idx = bar_index()
    events = events_frame(
        [
            ("2024-01-01", "2024-01-03", 1.0),
            ("2024-01-02", "2024-01-03", 3.0),
        ]
    )
    assert fetched.align(events, idx, agg="mean", ffill_limit=0).iloc[2] == pytest.approx(2.0)
    assert fetched.align(events, idx, agg="last", ffill_limit=0).iloc[2] == pytest.approx(3.0)
    assert fetched.align(events, idx, agg="first", ffill_limit=0).iloc[2] == pytest.approx(1.0)
    assert fetched.align(events, idx, agg="count", ffill_limit=0).iloc[2] == pytest.approx(2.0)


def test_align_handles_empty_and_bad_input():
    idx = bar_index()
    assert fetched.align(pd.DataFrame(), idx).isna().all()

    events = events_frame([("2024-01-01", "2024-01-03", 1.0)])
    with pytest.raises(ValueError, match="knowledge_time"):
        fetched.align(events.drop(columns=["knowledge_time"]), idx)
    with pytest.raises(ValueError, match="no column"):
        fetched.align(events, idx, field="conviction")
    with pytest.raises(ValueError, match="unknown agg"):
        fetched.align(events, idx, agg="median_ish")
    with pytest.raises(ValueError, match="sorted ascending"):
        fetched.align(events, idx[::-1])
    with pytest.raises(ValueError, match="ffill_limit"):
        fetched.align(events, idx, ffill_limit=-1)


def test_align_localizes_a_naive_index_as_utc():
    naive = pd.date_range("2024-01-01", periods=7, freq="D")
    events = events_frame([("2024-01-01", "2024-01-03", 5.0)])
    out = fetched.align(events, naive, ffill_limit=0)
    assert str(out.index.tz) == "UTC"
    assert out.iloc[2] == pytest.approx(5.0)


def test_series_reads_the_store_and_applies_the_barrier(monkeypatch):
    """``series`` is exercised against a stub store: the real parquet_io is a
    parallel module, and this pins the call shape we depend on."""
    captured: dict = {}

    def read_signals(source=None, *, tickers=None, start=None, end=None, as_of=None, kind=None):
        captured.update(source=source, tickers=tickers, as_of=as_of, kind=kind)
        return pd.DataFrame(
            [
                {
                    "event_time": pd.Timestamp("2024-01-01", tz="UTC"),
                    "knowledge_time": pd.Timestamp("2024-01-04", tz="UTC"),
                    "source": "govgreed",
                    "ticker": "AAPL",
                    "score": 7.0,
                },
                {   # a different ticker the store should have filtered out
                    "event_time": pd.Timestamp("2024-01-01", tz="UTC"),
                    "knowledge_time": pd.Timestamp("2024-01-02", tz="UTC"),
                    "source": "govgreed",
                    "ticker": "MSFT",
                    "score": -99.0,
                },
                {   # known after the cutoff; the local barrier must drop it
                    "event_time": pd.Timestamp("2024-01-01", tz="UTC"),
                    "knowledge_time": pd.Timestamp("2024-01-06", tz="UTC"),
                    "source": "govgreed",
                    "ticker": "AAPL",
                    "score": 42.0,
                },
            ]
        )

    stub = types.ModuleType("lab.store.parquet_io")
    stub.read_signals = read_signals
    import lab.store

    monkeypatch.setitem(sys.modules, "lab.store.parquet_io", stub)
    monkeypatch.setattr(lab.store, "parquet_io", stub, raising=False)

    out = fetched.series(
        "govgreed",
        "aapl",
        bar_index(),
        as_of=datetime(2024, 1, 5, tzinfo=UTC),
        kind="composite",
        ffill_limit=0,
    )

    assert captured["tickers"] == ["AAPL"]
    assert captured["as_of"] == datetime(2024, 1, 5, tzinfo=UTC)
    assert captured["kind"] == "composite"
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx(7.0)
    assert out.iloc[4:].isna().all()

    with pytest.raises(ValueError, match="does not accept"):
        fetched.series("govgreed", "AAPL", bar_index(), sector="tech")


def test_series_against_the_real_store(lab_env):
    """Integration across the store boundary: a disclosure written with a late
    knowledge_time must stay invisible until the bar it was disclosed on."""
    parquet_io = pytest.importorskip("lab.store.parquet_io")
    from lab.store.schema import Event

    parquet_io.write_events(
        [
            Event(
                event_time=datetime(2024, 1, 1, tzinfo=UTC),
                knowledge_time=datetime(2024, 1, 5, tzinfo=UTC),
                source="govgreed",
                ticker="AAPL",
                kind="composite",
                uid="composite:AAPL:2024-01-01:a",
                score=0.8,
            ),
            Event(  # another ticker's score must not bleed into AAPL's series
                event_time=datetime(2024, 1, 1, tzinfo=UTC),
                knowledge_time=datetime(2024, 1, 2, tzinfo=UTC),
                source="govgreed",
                ticker="MSFT",
                kind="composite",
                uid="composite:MSFT:2024-01-01:b",
                score=-9.0,
            ),
        ]
    )

    out = fetched.series("govgreed", "aapl", bar_index(), ffill_limit=1)
    assert out.iloc[:4].isna().all(), "a disclosure leaked onto bars before it existed"
    assert out.iloc[4] == pytest.approx(0.8)
    assert out.iloc[5] == pytest.approx(0.8)
    assert out.iloc[6:].isna().all()

    # ...and as_of clamps it back out of view entirely.
    early = fetched.series(
        "govgreed", "AAPL", bar_index(), as_of=datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert early.isna().all()


# --- cache -------------------------------------------------------------------


def test_cache_key_is_order_independent_and_version_sensitive(lab_env):
    a = cache.cache_key("AAPL", "sma", {"n": 20, "field": "close"}, "v1")
    b = cache.cache_key("AAPL", "sma", {"field": "close", "n": 20}, "v1")
    assert a == b
    assert a.startswith("AAPL_sma_")

    assert cache.cache_key("AAPL", "sma", {"n": 20}, "v1") != a
    assert cache.cache_key("AAPL", "sma", {"n": 20, "field": "close"}, "v2") != a
    assert cache.cache_key("MSFT", "sma", {"n": 20, "field": "close"}, "v1") != a
    assert cache.cache_key("AAPL", "ema", {"n": 20, "field": "close"}, "v1") != a
    # numpy scalars from a YAML grid must land on the same key as python ints
    assert cache.cache_key("AAPL", "sma", {"n": np.int64(20), "field": "close"}, "v1") == \
        cache.cache_key("AAPL", "sma", {"n": 20, "field": "close"}, "v1")

    with pytest.raises(ValueError, match="data_version"):
        cache.cache_key("AAPL", "sma", {"n": 20}, "")


def test_cache_round_trips_series_and_frames(lab_env):
    bars = make_bars(60)
    s = compute("sma", bars, n=10)
    key = cache.cache_key("AAPL", "sma", {"n": 10}, "v1")
    assert cache.get(key) is None

    cache.put(key, s)
    back = cache.get(key)
    assert isinstance(back, pd.Series)
    pd.testing.assert_series_equal(back, s, check_exact=True)

    frame = compute("bbands", bars, n=10)
    fkey = cache.cache_key("AAPL", "bbands", {"n": 10}, "v1")
    cache.put(fkey, frame)
    fback = cache.get(fkey)
    assert isinstance(fback, pd.DataFrame)
    pd.testing.assert_frame_equal(fback, frame, check_exact=True)


def test_cached_indicator_hits_misses_and_busts_on_data_version(lab_env):
    bars = make_bars(80)
    cache.reset_stats()

    first = cache.cached_indicator("AAPL", "sma", bars, "v1", n=10)
    assert cache.stats()["misses"] == 1 and cache.stats()["hits"] == 0

    second = cache.cached_indicator("AAPL", "sma", bars, "v1", n=10)
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1
    pd.testing.assert_series_equal(first, second, check_exact=True)
    pd.testing.assert_series_equal(first, compute("sma", bars, n=10), check_exact=True)

    # Restated data ⇒ new version ⇒ the old entry is unreachable, not reused.
    cache.cached_indicator("AAPL", "sma", bars, "v2", n=10)
    assert cache.stats()["misses"] == 2
    assert cache.stats()["entries"] == 2

    # ...and a changed param is likewise a different entry.
    cache.cached_indicator("AAPL", "sma", bars, "v2", n=20)
    assert cache.stats()["entries"] == 3


def test_cached_indicator_serves_a_stale_version_only_under_its_own_key(lab_env):
    bars = make_bars(80)
    v1 = cache.cached_indicator("AAPL", "sma", bars, "v1", n=10)
    # Same key, different underlying bars: the cache is keyed on the version, so
    # this is exactly the situation data_version exists to make impossible.
    restated = bars * 2.0
    v2 = cache.cached_indicator("AAPL", "sma", restated, "v2", n=10)
    assert not np.allclose(v1.dropna().to_numpy(), v2.dropna().to_numpy())


def test_cache_stats_and_clear(lab_env):
    bars = make_bars(40)
    cache.reset_stats()
    cache.cached_indicator("AAPL", "sma", bars, "v1", n=5)
    cache.cached_indicator("MSFT", "sma", bars, "v1", n=5)

    st = cache.stats()
    assert st["entries"] == 2 and st["bytes"] > 0

    assert cache.clear("AAPL") == 1
    assert cache.stats()["entries"] == 1
    assert cache.clear() == 1
    assert cache.stats()["entries"] == 0


def test_cache_rejects_bad_values_and_keys(lab_env):
    with pytest.raises(ValueError, match="Series or DataFrame"):
        cache.put(cache.cache_key("AAPL", "sma", {}, "v1"), [1, 2, 3])
    with pytest.raises(ValueError, match="bad cache key"):
        cache.get("../../etc/passwd")


def test_corrupt_cache_entry_is_a_miss_not_a_crash(lab_env):
    bars = make_bars(40)
    key = cache.cache_key("AAPL", "sma", {"n": 5}, "v1")
    cache.cached_indicator("AAPL", "sma", bars, "v1", n=5)

    path = cache._path(key)
    path.write_bytes(b"not parquet")
    cache.reset_stats()
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1
    assert not path.exists()
