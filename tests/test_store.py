"""Store tests, weighted toward the two properties everything else trusts:
``as_of`` really hides future knowledge, and a re-pull restates instead of
duplicating."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from lab.config import get_settings, reset_settings_cache
from lab.store import duck, parquet_io
from lab.store.schema import BARS, BARS_SCHEMA, SIGNALS, SIGNALS_SCHEMA, Bar, Event, empty_frame
from lab.timeutil import UTC


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(tmp_path / "runs"))
    reset_settings_cache()
    yield
    reset_settings_cache()


def _ts(day: int, *, month: int = 1, year: int = 2024, hour: int = 14) -> datetime:
    return datetime(year, month, day, hour, 30, tzinfo=UTC)


def _bar(ticker: str, day: int, *, close: float = 100.0, source: str = "synthetic",
         timeframe: str = "1d", **kw) -> Bar:
    et = _ts(day)
    return Bar(
        event_time=et,
        knowledge_time=kw.pop("knowledge_time", et + timedelta(days=1)),
        source=source,
        ticker=ticker,
        timeframe=timeframe,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1_000.0,
        **kw,
    )


def _event(ticker: str, day: int, *, lag_days: int = 42, uid: str | None = None,
           score: float = 0.5, source: str = "govgreed", kind: str = "congress_trade") -> Event:
    et = _ts(day, month=3)
    return Event(
        event_time=et,
        knowledge_time=et + timedelta(days=lag_days),
        source=source,
        ticker=ticker,
        kind=kind,
        uid=uid or f"{kind}:{ticker}:{et.date()}",
        direction="buy",
        tier="A",
        score=score,
    )


# --- round trips -------------------------------------------------------------


def test_bars_round_trip_preserves_schema_and_order():
    written = parquet_io.write_bars(
        [_bar("aapl", 3, close=101), _bar("MSFT", 2, close=201), _bar("AAPL", 2, close=100)]
    )
    assert written == 3

    df = parquet_io.read_bars()
    assert list(df.columns) == BARS_SCHEMA.names
    assert len(df) == 3
    assert df["event_time"].is_monotonic_increasing
    # Tickers are upper-cased on the way in, never on the way out.
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    assert df.loc[df["ticker"] == "AAPL", "close"].tolist() == [100.0, 101.0]
    assert df["year"].tolist() == [2024, 2024, 2024]


def test_partition_layout_is_hive_by_source_ticker_year():
    parquet_io.write_bars([_bar("AAPL", 2), _bar("AAPL", 2, source="yfinance")])
    root = parquet_io.table_path(BARS)
    assert (root / "source=synthetic" / "ticker=AAPL" / "year=2024" / "part-0.parquet").exists()
    assert (root / "source=yfinance" / "ticker=AAPL" / "year=2024" / "part-0.parquet").exists()
    # Partition keys live in the path only, so the file cannot contradict them.
    import pyarrow.parquet as pq

    cols = pq.ParquetFile(root / "source=synthetic/ticker=AAPL/year=2024/part-0.parquet").schema.names
    assert "source" not in cols and "ticker" not in cols and "year" not in cols


def test_events_round_trip():
    ev = _event("AAPL", 1, score=0.75)
    assert parquet_io.write_events([ev]) == 1

    df = parquet_io.read_signals(as_of=ev.knowledge_time)
    assert list(df.columns) == SIGNALS_SCHEMA.names
    assert len(df) == 1
    row = df.iloc[0]
    assert row["uid"] == ev.uid and row["score"] == 0.75 and row["tier"] == "A"
    assert row["knowledge_time"] - row["event_time"] == timedelta(days=42)


def test_filters_narrow_by_ticker_source_timeframe_and_window():
    parquet_io.write_bars(
        [_bar(t, d, timeframe=tf)
         for t in ("AAPL", "MSFT", "NVDA") for d in (2, 3, 4) for tf in ("1d", "1h")]
    )
    late = _ts(9)  # far enough out that knowledge_time never bites
    assert set(parquet_io.read_bars("AAPL", as_of=late)["ticker"]) == {"AAPL"}
    assert set(parquet_io.read_bars(["AAPL", "NVDA"], as_of=late)["ticker"]) == {"AAPL", "NVDA"}
    assert set(parquet_io.read_bars(as_of=late)["timeframe"]) == {"1d"}
    assert set(parquet_io.read_bars(as_of=late, timeframe="1h")["timeframe"]) == {"1h"}
    assert parquet_io.read_bars(as_of=late, source="nope").empty

    windowed = parquet_io.read_bars(as_of=late, start=_ts(3), end=_ts(3))
    assert len(windowed) == 3 and set(windowed["event_time"]) == {_ts(3)}


def test_reads_push_filters_down_instead_of_scanning_the_store(monkeypatch):
    """A three-ticker question must not open thirty tickers' worth of files."""
    parquet_io.write_bars([_bar(t, d) for t in ("AAPL", "MSFT", "NVDA", "TSLA") for d in (2, 3)])
    real = parquet_io._dataset
    seen: dict[str, object] = {}

    class _Spy:
        def __init__(self, inner):
            self._inner = inner

        def to_table(self, **kw):
            seen["filter"] = kw.get("filter")
            seen["fragments"] = len(list(self._inner.get_fragments(filter=kw.get("filter"))))
            return self._inner.to_table(**kw)

    monkeypatch.setattr(parquet_io, "_dataset", lambda t: _Spy(real(t)))
    df = parquet_io.read_bars("AAPL", as_of=_ts(9))

    assert len(df) == 2
    expr = str(seen["filter"])
    assert "knowledge_time" in expr and "ticker" in expr
    assert seen["fragments"] == 1  # four tickers on disk, one partition opened


def test_awkward_tickers_and_year_boundaries():
    parquet_io.write_bars(
        [
            _bar("BRK.B", 2),
            Bar(event_time=_ts(30, month=12, year=2023), knowledge_time=_ts(31, month=12, year=2023),
                source="synthetic", ticker="BRK.B", timeframe="1d",
                open=1, high=2, low=0.5, close=1.5, volume=10),
        ]
    )
    root = parquet_io.table_path(BARS) / "source=synthetic" / "ticker=BRK.B"
    assert {p.name for p in root.iterdir()} == {"year=2023", "year=2024"}

    late = _ts(9, month=6)
    assert len(parquet_io.read_bars("brk.b", as_of=late)) == 2
    only_2024 = parquet_io.read_bars("BRK.B", as_of=late, start=_ts(1))
    assert only_2024["year"].tolist() == [2024]
    assert duck.query("SELECT DISTINCT year FROM bars ORDER BY year")["year"].tolist() == [2023, 2024]


# --- the look-ahead barrier --------------------------------------------------


def test_as_of_hides_rows_whose_knowledge_time_is_in_the_future():
    ev = _event("AAPL", 15, lag_days=42)
    parquet_io.write_events([ev])

    # A congressional trade filed six weeks late: readable at disclosure, not before.
    assert parquet_io.read_signals(as_of=ev.event_time).empty
    assert parquet_io.read_signals(as_of=ev.knowledge_time - timedelta(seconds=1)).empty
    assert len(parquet_io.read_signals(as_of=ev.knowledge_time)) == 1
    assert len(parquet_io.read_signals(as_of=ev.knowledge_time + timedelta(days=365))) == 1
    # No as_of at all is the raw store, which is why the engine always passes one.
    assert len(parquet_io.read_signals()) == 1


def test_as_of_wins_over_the_event_time_window():
    """A window that brackets event_time must not resurrect a row we could not
    have known -- as_of is applied first, not as one clause among equals."""
    ev = _event("AAPL", 15, lag_days=42)
    parquet_io.write_events([ev])

    df = parquet_io.read_signals(
        tickers=["AAPL"],
        start=ev.event_time - timedelta(days=1),
        end=ev.event_time + timedelta(days=1),
        as_of=ev.event_time + timedelta(days=1),
        kind=ev.kind,
    )
    assert df.empty


def test_as_of_hides_bars_too():
    bars = [_bar("AAPL", 2), _bar("AAPL", 3), _bar("AAPL", 4)]
    parquet_io.write_bars(bars)
    # knowledge_time is event_time + 1 day: at bar 3's open, only bars 1-2 are known.
    df = parquet_io.read_bars("AAPL", as_of=_ts(3))
    assert df["event_time"].tolist() == [_ts(2)]


# --- dedupe ------------------------------------------------------------------


def test_bar_rewrite_restates_rather_than_duplicates():
    parquet_io.write_bars([_bar("AAPL", 2, close=100.0)])
    parquet_io.write_bars([_bar("AAPL", 2, close=123.5)])

    df = parquet_io.read_bars("AAPL", as_of=_ts(9))
    assert len(df) == 1
    assert df.iloc[0]["close"] == 123.5


def test_bar_dedupe_key_separates_timeframe_and_source():
    parquet_io.write_bars(
        [
            _bar("AAPL", 2, close=1.0),
            _bar("AAPL", 2, close=2.0, timeframe="1h"),
            _bar("AAPL", 2, close=3.0, source="yfinance"),
        ]
    )
    assert len(parquet_io.read_bars("AAPL", as_of=_ts(9), timeframe="1d")) == 2
    assert len(parquet_io.read_bars("AAPL", as_of=_ts(9), timeframe="1h")) == 1


def test_event_rewrite_restates_on_uid():
    first = _event("AAPL", 1, uid="u-1", score=0.10)
    second = _event("AAPL", 1, uid="u-1", score=0.90)
    parquet_io.write_events([first])
    parquet_io.write_events([second])

    df = parquet_io.read_signals(as_of=second.knowledge_time)
    assert len(df) == 1 and df.iloc[0]["score"] == 0.90


def test_duplicates_inside_one_batch_collapse():
    parquet_io.write_bars([_bar("AAPL", 2, close=1.0), _bar("AAPL", 2, close=9.0)])
    df = parquet_io.read_bars("AAPL", as_of=_ts(9))
    assert len(df) == 1 and df.iloc[0]["close"] == 9.0


def test_dedupe_false_appends_a_new_part_file():
    parquet_io.write_bars([_bar("AAPL", 2, close=1.0)])
    parquet_io.write_bars([_bar("AAPL", 2, close=2.0)], dedupe=False)
    part = parquet_io.table_path(BARS) / "source=synthetic" / "ticker=AAPL" / "year=2024"
    assert sorted(p.name for p in part.glob("*.parquet")) == ["part-0.parquet", "part-1.parquet"]
    assert len(parquet_io.read_bars("AAPL", as_of=_ts(9))) == 2


# --- provenance --------------------------------------------------------------


def test_data_version_is_stable_and_moves_when_data_does():
    parquet_io.write_bars([_bar("AAPL", 2), _bar("MSFT", 2)])
    v1 = parquet_io.data_version(BARS)
    assert len(v1) == 16 and int(v1, 16) >= 0
    assert parquet_io.data_version(BARS) == v1

    parquet_io.write_bars([_bar("AAPL", 3)])
    v2 = parquet_io.data_version(BARS)
    assert v2 != v1

    # Scoping to untouched tickers keeps the old answer; scoping to the touched
    # one does not. That is what makes a per-run stamp meaningful.
    assert parquet_io.data_version(BARS, tickers=["MSFT"]) != parquet_io.data_version(BARS, tickers=["AAPL"])
    assert parquet_io.data_version(BARS, source="synthetic") == v2
    assert parquet_io.data_version(SIGNALS) == parquet_io.data_version(SIGNALS)


def test_coverage_summarizes_both_tables():
    parquet_io.write_bars([_bar("AAPL", 2), _bar("AAPL", 3), _bar("MSFT", 2)])
    cov = parquet_io.coverage(BARS)
    assert list(cov.columns) == ["source", "ticker", "timeframe", "rows", "first", "last"]
    aapl = cov[cov["ticker"] == "AAPL"].iloc[0]
    assert aapl["rows"] == 2 and aapl["first"] == _ts(2) and aapl["last"] == _ts(3)

    parquet_io.write_events([_event("AAPL", 1)])
    sig_cov = parquet_io.coverage(SIGNALS)
    assert sig_cov.iloc[0]["timeframe"] == "congress_trade"  # sub-type slot carries `kind`


# --- empty store -------------------------------------------------------------


def test_empty_store_reads_return_typed_empty_frames():
    for table, reader in ((BARS, parquet_io.read_bars), (SIGNALS, parquet_io.read_signals)):
        df = reader()
        expected = empty_frame(table)
        assert df.empty
        assert list(df.columns) == list(expected.columns)
        assert df.dtypes.equals(expected.dtypes)

    assert parquet_io.read_bars(["AAPL"], as_of=_ts(2), start=_ts(1)).empty
    assert parquet_io.coverage(BARS).empty
    assert len(parquet_io.data_version(BARS)) == 16
    assert not get_settings().paths.parquet.joinpath(BARS).exists()


def test_reads_that_match_nothing_keep_the_schema():
    parquet_io.write_bars([_bar("AAPL", 2)])
    df = parquet_io.read_bars("NVDA", as_of=_ts(9))
    assert df.empty and df.dtypes.equals(empty_frame(BARS).dtypes)


def test_duck_tolerates_an_empty_store():
    assert duck.query("SELECT count(*) AS n FROM bars").iloc[0]["n"] == 0
    assert duck.query("SELECT count(*) AS n FROM signals").iloc[0]["n"] == 0
    desc = duck.describe()
    assert set(desc["table"]) == {BARS, SIGNALS}
    assert desc["rows"].tolist() == [0, 0]


# --- duckdb ------------------------------------------------------------------


def test_duck_query_round_trip():
    parquet_io.write_bars(
        [_bar("AAPL", 2, close=100), _bar("AAPL", 3, close=110), _bar("MSFT", 2, close=200)]
    )
    parquet_io.write_events([_event("AAPL", 1, score=0.4)])

    agg = duck.query(
        "SELECT ticker, count(*) AS n, max(close) AS hi FROM bars GROUP BY ticker ORDER BY ticker"
    )
    assert agg["ticker"].tolist() == ["AAPL", "MSFT"]
    assert agg["n"].tolist() == [2, 1]
    assert agg["hi"].tolist() == [110.0, 200.0]

    one = duck.query("SELECT close FROM bars WHERE ticker = ? AND event_time = ?", ["MSFT", _ts(2)])
    assert one["close"].tolist() == [200.0]

    named = duck.query("SELECT count(*) AS n FROM signals WHERE source = $src", {"src": "govgreed"})
    assert named.iloc[0]["n"] == 1

    # Hive columns are reconstructed from the path, typed, and in schema order.
    cols = duck.query("SELECT * FROM bars LIMIT 1").columns.tolist()
    assert cols == BARS_SCHEMA.names
    assert duck.query("SELECT year FROM bars LIMIT 1").iloc[0]["year"] == 2024

    desc = duck.describe()
    assert desc.set_index("table").loc[BARS, "rows"] == 3


def test_duck_can_apply_the_barrier_itself():
    ev = _event("AAPL", 15, lag_days=42)
    parquet_io.write_events([ev])
    hidden = duck.query(
        "SELECT count(*) AS n FROM signals WHERE knowledge_time <= $t", {"t": ev.event_time}
    )
    assert hidden.iloc[0]["n"] == 0


# --- input validation --------------------------------------------------------


def test_knowledge_time_before_event_time_is_rejected():
    et = _ts(2)
    df = pd.DataFrame(
        [{"event_time": et, "knowledge_time": et - timedelta(days=1), "source": "synthetic",
          "ticker": "AAPL", "timeframe": "1d", "open": 1.0, "high": 1.0, "low": 1.0,
          "close": 1.0, "volume": 1.0}]
    )
    with pytest.raises(ValueError, match="precedes event_time"):
        parquet_io.write_frame(BARS, df)


def test_bad_input_raises_value_error():
    with pytest.raises(ValueError, match="unknown table"):
        parquet_io.table_path("nope")
    with pytest.raises(ValueError, match="missing required column"):
        parquet_io.write_frame(BARS, pd.DataFrame([{"event_time": _ts(2), "ticker": "AAPL"}]))
    with pytest.raises(ValueError):
        parquet_io.read_bars([])
    assert parquet_io.write_bars([]) == 0
    assert parquet_io.write_frame(BARS, empty_frame(BARS)) == 0
