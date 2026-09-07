"""The lab's memory: the run registry, the two journals, and the wall between them.

Most of what is pinned here is not "does the SQL work" but the properties other
modules are entitled to assume. Three matter more than the rest.

*The attempt counter is evidence.* ``attempts()`` and ``family_attempts()`` are
the only mechanism that makes "someone ran four hundred variations and showed me
the good one" visible after the fact, so they are tested for exact values, not
for monotonicity. An off-by-one here is not a cosmetic bug; it is a number in a
report that quietly understates how much searching happened.

*``seq`` is a resumable cursor.* The console reconnects and asks for everything
after the last row it saw. That contract only holds if ``seq`` is strictly
increasing, never reused, and survives the connection being dropped -- which is
exactly what a runner crash or a WebSocket drop looks like from here.

*The two databases are separate on purpose.* A test that passes because both
classes happened to open the same file would hide the day one of them stops
doing so, so the separation is asserted directly rather than assumed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from lab.engine.events import (
    Decision,
    EventKind,
    GateAction,
    GateVerdict,
    Intent,
    event,
)
from lab.registry import db
from lab.registry.journal import DecisionJournal, EventJournal
from lab.registry.runs import (
    RUN_KINDS,
    RunRegistry,
    config_hash,
    git_commit,
    run_artifact_dir,
)
from lab.timeutil import utcnow

TABLES = {
    "runs",
    "run_metrics",
    "decisions",
    "orders",
    "fills",
    "trades",
    "events",
    "agent_calls",
    "quota_log",
    "live_strategies",
}


@pytest.fixture()
def registry() -> RunRegistry:
    return RunRegistry()


@pytest.fixture()
def decisions() -> DecisionJournal:
    return DecisionJournal()


@pytest.fixture()
def events() -> EventJournal:
    return EventJournal()


def make_decision(
    *,
    id: str = "dec-1",
    run_id: str = "run-1",
    at: Any = None,
    ticker: str = "AAA",
) -> Decision:
    """A decision carrying something non-trivial in every JSON column."""
    return Decision(
        id=id,
        run_id=run_id,
        strategy="momo",
        at=at if at is not None else utcnow(),
        inputs={
            "prices": {ticker: 101.25},
            "signals": [{"ticker": ticker, "score": 0.8, "knowledge_time": "2024-01-02T21:00:00+00:00"}],
            "indicators": {"sma_50": 99.5, "warm": True, "missing": None},
        },
        intents=[Intent(ticker.lower(), 0.25, tag="entry", reason="rank 1")],
        verdicts=[
            GateVerdict(ticker, GateAction.CLIP, 0.25, 0.05, rule="position_cap", detail="capped")
        ],
        order_ids=["o-1", "o-2"],
        portfolio={"cash": 90_000.0, "equity": 100_500.5, "positions": {ticker: 12.0}},
        logs=[{"event": "allocate", "n": 2}],
        agent={"model": "test", "input_tokens": 11},
        duration_ms=3.5,
    )


# --- schema and plumbing ------------------------------------------------------


def test_init_db_is_idempotent_and_creates_every_table(registry: RunRegistry) -> None:
    con = db.connect()
    record = registry.create(strategy="momo", params={"lookback": 20})

    db.init_db(con)
    db.init_db(con)

    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names
    # Re-running the schema must not truncate anything.
    assert registry.get(record.run_id) is not None


def test_registry_and_journal_are_separate_databases(
    registry: RunRegistry, events: EventJournal
) -> None:
    assert db.registry_path() != db.journal_path()

    registry.create(strategy="momo")
    events.append(event(EventKind.LOG, "test", at=utcnow(), message="hello"))

    reg_con = db.connect(db.registry_path())
    jou_con = db.connect(db.journal_path())
    assert reg_con is not jou_con
    # Both files carry the whole schema, so "wrong database" shows up as an
    # empty table rather than an OperationalError -- assert on the counts.
    assert reg_con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert reg_con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert jou_con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert jou_con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_connect_caches_one_connection_per_path() -> None:
    assert db.connect() is db.connect()
    assert db.connect(db.journal_path()) is db.connect(db.journal_path())


def test_transaction_rolls_back_on_error() -> None:
    con = db.connect()
    with pytest.raises(RuntimeError):
        with db.transaction(con):
            con.execute(
                "INSERT INTO runs (run_id, strategy, created_at) VALUES ('x','s','2024-01-01')"
            )
            raise RuntimeError("boom")
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


# --- runs: the round trip -----------------------------------------------------


def test_create_finish_get_list_round_trip(registry: RunRegistry) -> None:
    start = utcnow() - timedelta(days=30)
    record = registry.create(
        strategy="momo",
        kind="backtest",
        params={"lookback": 126, "top_n": 4},
        config={"tickers": ["AAA", "BBB"], "cash": 100_000},
        data_version="deadbeefdeadbeef",
        start=start,
        end=utcnow(),
        notes="round trip",
    )
    assert record.status == "running"
    assert record.finished_at is None
    assert record.run_id.startswith("momo-")

    finished = registry.finish(
        record.run_id, metrics={"sharpe": 1.25, "trades": 7, "note": "ok", "nan": float("nan")}
    )
    assert finished.status == "ok"
    assert finished.finished_at is not None
    assert finished.metrics["sharpe"] == 1.25

    got = registry.get(record.run_id)
    assert got is not None
    # Every column that went in comes back out with its type intact.
    assert got.params == {"lookback": 126, "top_n": 4}
    assert got.config["tickers"] == ["AAA", "BBB"]
    assert got.data_version == "deadbeefdeadbeef"
    assert got.notes == "round trip"
    assert got.metrics["trades"] == 7
    assert got.start is not None and abs((got.start - start).total_seconds()) < 1
    assert got.created_at.tzinfo is not None

    listed = registry.list(strategy="momo")
    assert [r.run_id for r in listed] == [record.run_id]
    assert registry.get("no-such-run") is None
    assert registry.list(strategy="other") == []


def test_finish_projects_metrics_into_run_metrics(registry: RunRegistry) -> None:
    record = registry.create(strategy="momo")
    registry.finish(record.run_id, metrics={"sharpe": 1.5, "optimistic_fills": True, "start": "x"})

    rows = {
        r["name"]: (r["value"], r["text_value"])
        for r in db.connect().execute(
            "SELECT name, value, text_value FROM run_metrics WHERE run_id = ?", (record.run_id,)
        )
    }
    # Numbers land in `value` so `ORDER BY sharpe` stays pure SQL; strings do not.
    assert rows["sharpe"] == (1.5, None)
    assert rows["optimistic_fills"][0] == 1.0
    assert rows["start"] == (None, "x")


def test_finish_replaces_rather_than_accumulates_metrics(registry: RunRegistry) -> None:
    record = registry.create(strategy="momo", metrics={"sharpe": 9.9, "stale": 1})
    registry.finish(record.run_id, metrics={"sharpe": 0.5})

    names = [
        r[0]
        for r in db.connect().execute(
            "SELECT name FROM run_metrics WHERE run_id = ?", (record.run_id,)
        )
    ]
    assert names == ["sharpe"]
    assert registry.get(record.run_id).metrics == {"sharpe": 0.5}


def test_list_filters_order_and_paginate(registry: RunRegistry) -> None:
    made = [
        registry.create(strategy="momo", kind="backtest", origin="human"),
        registry.create(strategy="momo", kind="sweep", origin="agent-loop", sweep_id="s1"),
        registry.create(strategy="other", kind="backtest", origin="human"),
    ]
    ids = [r.run_id for r in made]

    assert [r.run_id for r in registry.list()] == list(reversed(ids))
    assert [r.run_id for r in registry.list(kind="sweep")] == [ids[1]]
    assert [r.run_id for r in registry.list(origin="agent-loop")] == [ids[1]]
    assert [r.run_id for r in registry.list(sweep_id="s1")] == [ids[1]]
    assert [r.run_id for r in registry.list(limit=1)] == [ids[2]]
    assert [r.run_id for r in registry.list(limit=1, offset=1)] == [ids[1]]
    assert [r.run_id for r in registry.list(order_by="created_at asc")][0] == ids[0]


def test_list_rejects_an_unsafe_order_by(registry: RunRegistry) -> None:
    # The one place caller text reaches the SQL text; it is allowlisted.
    with pytest.raises(ValueError):
        registry.list(order_by="created_at; DROP TABLE runs")
    with pytest.raises(ValueError):
        registry.list(order_by="metrics")


def test_delete_is_reported_and_not_repeated(registry: RunRegistry) -> None:
    record = registry.create(strategy="momo")
    assert registry.delete(record.run_id) is True
    assert registry.delete(record.run_id) is False
    assert registry.get(record.run_id) is None


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"strategy": ""}, "strategy"),
        ({"strategy": "momo", "kind": "nonsense"}, "kind"),
        ({"strategy": "momo", "bogus_field": 1}, "unknown"),
    ],
)
def test_create_rejects_bad_input(registry: RunRegistry, fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        registry.create(**fields)


def test_finish_on_an_unknown_run_raises(registry: RunRegistry) -> None:
    with pytest.raises(ValueError, match="no such run"):
        registry.finish("not-a-run", metrics={})


def test_run_kinds_are_the_documented_four() -> None:
    assert RUN_KINDS == {"backtest", "paper", "live", "sweep"}


# --- the attempt counter ------------------------------------------------------


def test_attempt_increments_for_the_same_strategy_and_config(registry: RunRegistry) -> None:
    params = {"lookback": 126, "top_n": 4}
    first, second, third = (registry.create(strategy="momo", params=params) for _ in range(3))

    assert [first.attempt, second.attempt, third.attempt] == [1, 2, 3]
    # Same config every time, so all three share one hash and one family.
    assert first.config_hash == second.config_hash == third.config_hash
    assert registry.attempts("momo", first.config_hash) == 3
    assert registry.family_attempts("momo") == 3
    # The count is durable, not a property of the in-memory record.
    assert registry.get(third.run_id).attempt == 3


def test_attempt_resets_per_config_while_family_attempts_accumulates(
    registry: RunRegistry,
) -> None:
    a1 = registry.create(strategy="momo", params={"lookback": 126})
    a2 = registry.create(strategy="momo", params={"lookback": 126})
    b1 = registry.create(strategy="momo", params={"lookback": 200})
    b2 = registry.create(strategy="momo", params={"lookback": 200})
    c1 = registry.create(strategy="momo", params={"lookback": 250})

    assert a1.config_hash != b1.config_hash != c1.config_hash
    assert [a1.attempt, a2.attempt] == [1, 2]
    assert [b1.attempt, b2.attempt] == [1, 2]
    assert c1.attempt == 1

    assert registry.attempts("momo", a1.config_hash) == 2
    assert registry.attempts("momo", b1.config_hash) == 2
    assert registry.attempts("momo", c1.config_hash) == 1
    # Five variations were tried; that is the number the report has to show.
    assert registry.family_attempts("momo") == 5
    assert registry.attempts("momo") == 5


def test_attempts_are_scoped_to_one_strategy(registry: RunRegistry) -> None:
    shared = {"lookback": 126}
    mine = registry.create(strategy="momo", params=shared)
    theirs = registry.create(strategy="meanrev", params=shared)

    assert mine.config_hash == theirs.config_hash  # same knobs, different strategy
    assert mine.attempt == theirs.attempt == 1
    assert registry.family_attempts("momo") == 1
    assert registry.family_attempts("meanrev") == 1
    assert registry.family_attempts("never-run") == 0


def test_attempt_groups_by_config_when_one_is_given(registry: RunRegistry) -> None:
    # An explicit `config` wins over `params` for grouping: two runs of the same
    # config file with different resolved params are still the same experiment.
    cfg = {"tickers": ["AAA"], "cash": 100_000}
    one = registry.create(strategy="momo", config=cfg, params={"lookback": 1})
    two = registry.create(strategy="momo", config=dict(reversed(list(cfg.items()))), params={"lookback": 2})

    assert one.config_hash == two.config_hash
    assert [one.attempt, two.attempt] == [1, 2]


def test_an_explicit_config_hash_is_honoured(registry: RunRegistry) -> None:
    first = registry.create(strategy="momo", config_hash="fixed", params={"a": 1})
    second = registry.create(strategy="momo", config_hash="fixed", params={"b": 2})
    assert [first.attempt, second.attempt] == [1, 2]


def test_config_hash_is_stable_and_order_independent() -> None:
    left = config_hash({"a": 1, "b": [1, 2], "c": {"d": True}})
    right = config_hash({"c": {"d": True}, "b": [1, 2], "a": 1})
    assert left == right
    assert len(left) == 16
    # 1 and 1.0 are the same knob; a different value is a different experiment.
    assert config_hash({"a": 1}) == config_hash({"a": 1.0})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert config_hash({}) == config_hash({})
    with pytest.raises(ValueError):
        config_hash(["not", "a", "mapping"])  # type: ignore[arg-type]


# --- compare ------------------------------------------------------------------


def test_compare_returns_one_row_per_run_in_the_order_asked(registry: RunRegistry) -> None:
    first = registry.create(strategy="momo", params={"lookback": 126, "top_n": 4})
    second = registry.create(strategy="momo", params={"lookback": 200, "top_n": 4})
    third = registry.create(strategy="momo", params={"lookback": 250, "top_n": 4})
    registry.finish(first.run_id, metrics={"sharpe": 1.0, "trades": 4})
    registry.finish(second.run_id, metrics={"sharpe": 0.5, "trades": 9})
    registry.finish(third.run_id, metrics={"sharpe": -0.2})

    ids = [third.run_id, first.run_id, second.run_id]
    frame = registry.compare(ids)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert list(frame["run_id"]) == ids
    # Params that moved are shown; the one that did not is dropped as noise.
    assert "param.lookback" in frame.columns
    assert "param.top_n" not in frame.columns
    assert list(frame["param.lookback"]) == [250, 126, 200]
    # The metric union, with a hole where a run never reported one.
    assert list(frame["sharpe"]) == [-0.2, 1.0, 0.5]
    assert pd.isna(frame["trades"].iloc[0])
    assert set(("strategy", "attempt", "config_hash", "data_version")) <= set(frame.columns)


def test_compare_of_one_run_keeps_its_params(registry: RunRegistry) -> None:
    only = registry.create(strategy="momo", params={"lookback": 126})
    frame = registry.compare([only.run_id])
    assert len(frame) == 1
    assert frame["param.lookback"].iloc[0] == 126


def test_compare_rejects_an_unknown_run(registry: RunRegistry) -> None:
    known = registry.create(strategy="momo")
    with pytest.raises(ValueError, match="unknown run_id"):
        registry.compare([known.run_id, "ghost"])
    assert registry.compare([]).empty


# --- provenance ---------------------------------------------------------------


def test_git_commit_returns_none_or_a_string_and_never_raises() -> None:
    got = git_commit()
    assert got is None or (isinstance(got, str) and got and "\n" not in got)


def test_run_artifact_dir_is_created_under_the_isolated_runs_dir(paths) -> None:
    path = run_artifact_dir("momo-20240102T000000-abc123")
    assert path.is_dir()
    assert path.parent == paths.runs
    assert run_artifact_dir("momo-20240102T000000-abc123") == path  # idempotent


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", "a\\b", "why?"])
def test_run_artifact_dir_refuses_a_path_traversing_run_id(bad: str) -> None:
    with pytest.raises(ValueError):
        run_artifact_dir(bad)


# --- decision journal ---------------------------------------------------------


def test_decision_round_trips_every_json_column(decisions: DecisionJournal) -> None:
    original = make_decision()
    decisions.append(original)

    got = decisions.get("dec-1")
    assert got is not None
    assert got["run_id"] == "run-1"
    assert got["strategy"] == "momo"
    # Nested structure, floats, bools and nulls all survive the JSON columns.
    assert got["inputs"] == original.inputs
    assert got["intents"][0]["ticker"] == "AAA"  # Intent upper-cases on init
    assert got["intents"][0]["target_pct"] == 0.25
    assert got["intents"][0]["tag"] == "entry"
    assert got["verdicts"][0]["action"] == GateAction.CLIP.value
    assert got["verdicts"][0]["rule"] == "position_cap"
    assert got["order_ids"] == ["o-1", "o-2"]
    assert got["portfolio"]["positions"] == {"AAA": 12.0}
    assert got["logs"] == [{"event": "allocate", "n": 2}]
    assert got["agent"] == {"model": "test", "input_tokens": 11}
    assert got["duration_ms"] == 3.5
    # The collapsed line the tape renders is stored, not recomputed on read.
    assert got["summary"] == original.summary()
    assert "clipped" in got["summary"]
    assert decisions.get("no-such-decision") is None


def test_decision_at_is_stored_as_iso_utc(decisions: DecisionJournal) -> None:
    at = utcnow()
    decisions.append(make_decision(at=at))
    stored = decisions.get("dec-1")["at"]
    assert isinstance(stored, str)
    assert pd.Timestamp(stored).tz is not None


def test_decision_list_orders_filters_and_paginates(decisions: DecisionJournal) -> None:
    base = utcnow()
    rows = [
        make_decision(id="d3", at=base + timedelta(minutes=2), ticker="CCC"),
        make_decision(id="d1", at=base, ticker="AAA"),
        make_decision(id="d2", at=base + timedelta(minutes=1), ticker="BBB"),
        make_decision(id="other", run_id="run-2", at=base, ticker="AAA"),
    ]
    assert decisions.bulk_append(rows) == 4

    listed = decisions.list("run-1")
    assert [d["id"] for d in listed] == ["d1", "d2", "d3"]  # oldest first
    assert decisions.count("run-1") == 3
    assert decisions.count("run-2") == 1
    assert decisions.count("missing") == 0

    assert [d["id"] for d in decisions.list("run-1", limit=2)] == ["d1", "d2"]
    assert [d["id"] for d in decisions.list("run-1", limit=2, offset=2)] == ["d3"]
    assert [d["id"] for d in decisions.list("run-1", start=base + timedelta(minutes=1))] == [
        "d2",
        "d3",
    ]
    assert [d["id"] for d in decisions.list("run-1", end=base)] == ["d1"]


def test_decision_ticker_filter_matches_whole_tickers_only(
    decisions: DecisionJournal,
) -> None:
    decisions.append(make_decision(id="d1", ticker="AAA"))
    decisions.append(make_decision(id="d2", ticker="AA"))

    assert [d["id"] for d in decisions.list("run-1", ticker="AAA")] == ["d1"]
    assert [d["id"] for d in decisions.list("run-1", ticker="AA")] == ["d2"]
    assert [d["id"] for d in decisions.list("run-1", ticker="aaa")] == ["d1"]
    assert decisions.list("run-1", ticker="ZZZ") == []


def test_appending_the_same_decision_twice_restates_it(decisions: DecisionJournal) -> None:
    # A crashed runner replaying its tail must not duplicate the tape.
    decisions.append(make_decision())
    decisions.append(make_decision())
    assert decisions.count("run-1") == 1


def test_bulk_append_of_nothing_is_free(decisions: DecisionJournal) -> None:
    assert decisions.bulk_append([]) == 0


def test_decision_journal_rejects_a_non_decision(decisions: DecisionJournal) -> None:
    with pytest.raises(ValueError):
        decisions.append(object())  # type: ignore[arg-type]


# --- event journal ------------------------------------------------------------


def log_event(message: str, **kw: Any):
    return event(EventKind.LOG, kw.pop("source", "test"), at=utcnow(), message=message, **kw)


def test_event_seq_is_strictly_increasing_and_assigned_back(events: EventJournal) -> None:
    written = [log_event(f"m{i}") for i in range(5)]
    seqs = [events.append(ev) for ev in written]

    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 5
    assert all(a < b for a, b in zip(seqs, seqs[1:]))
    assert [ev.seq for ev in written] == seqs  # the caller's object learns its seq
    assert events.latest_seq() == seqs[-1]
    assert events.count() == 5


def test_appending_a_duplicate_id_returns_the_existing_seq(events: EventJournal) -> None:
    ev = log_event("once")
    first = events.append(ev)
    second = events.append(ev)
    assert first == second
    assert events.count() == 1


def test_tail_resumes_from_a_cursor_across_a_reconnect(events: EventJournal) -> None:
    for i in range(3):
        events.append(log_event(f"before-{i}"))

    seen = events.tail(0)
    assert [r["message"] for r in seen] == ["before-0", "before-1", "before-2"]
    cursor = seen[-1]["seq"]

    # The console drops its connection here and comes back with a fresh handle.
    db.close_all()
    reconnected = EventJournal()
    assert reconnected.tail(cursor) == []

    after = [reconnected.append(log_event(f"after-{i}")) for i in range(2)]
    resumed = reconnected.tail(cursor)
    assert [r["seq"] for r in resumed] == after
    assert [r["message"] for r in resumed] == ["after-0", "after-1"]
    # Nothing was replayed and nothing was skipped.
    assert min(after) > cursor
    assert len(reconnected.tail(0)) == 5


def test_seq_is_never_reused_after_a_purge(events: EventJournal) -> None:
    high = events.append(log_event("doomed"))
    events.con.execute("DELETE FROM events")
    assert events.latest_seq() == 0  # MAX over an empty table

    fresh = events.append(log_event("new"))
    # AUTOINCREMENT: a cursor still holding `high` must not be handed old rows.
    assert fresh > high


def test_tail_filters_and_limits(events: EventJournal) -> None:
    events.append(log_event("a", run_id="r1", strategy="momo"))
    events.append(log_event("b", run_id="r2", strategy="momo"))
    events.append(event(EventKind.FILL, "broker", at=utcnow(), message="c", run_id="r1"))

    assert [r["message"] for r in events.tail(0, run_id="r1")] == ["a", "c"]
    assert [r["message"] for r in events.tail(0, strategy="momo")] == ["a", "b"]
    assert [r["message"] for r in events.tail(0, kinds=[EventKind.FILL])] == ["c"]
    assert [r["message"] for r in events.tail(0, kinds=["log"])] == ["a", "b"]
    assert [r["message"] for r in events.tail(0, limit=1)] == ["a"]


def test_event_payload_round_trips(events: EventJournal) -> None:
    events.append(
        event(
            EventKind.GATE_BLOCK,
            "gate",
            at=utcnow(),
            message="blocked",
            ticker="AAA",
            payload={"rule": "position_cap", "requested": 0.25, "nested": {"ok": False}},
        )
    )
    row = events.tail(0)[0]
    assert row["kind"] == "gate_block"
    assert row["ticker"] == "AAA"
    assert row["payload"] == {"rule": "position_cap", "requested": 0.25, "nested": {"ok": False}}


def test_event_journal_rejects_a_non_event(events: EventJournal) -> None:
    with pytest.raises(ValueError):
        events.append(object())  # type: ignore[arg-type]


# --- heartbeats ---------------------------------------------------------------


def test_heartbeat_reports_a_fresh_age_and_its_meta(events: EventJournal) -> None:
    events.heartbeat("runner", meta={"pid": 4242, "bar": "2024-01-02"})

    beats = events.last_heartbeats()
    assert set(beats) == {"runner"}
    assert beats["runner"]["meta"] == {"pid": 4242, "bar": "2024-01-02"}
    assert 0 <= beats["runner"]["age_s"] < 60
    assert beats["runner"]["seq"] == events.latest_seq()


def test_heartbeat_age_is_measured_from_the_stored_timestamp(events: EventJournal) -> None:
    # Staleness has to be computed at read time, not assumed: a source that
    # stopped beating an hour ago must read as an hour old, not as absent.
    stale = event(
        EventKind.HEARTBEAT,
        "broker",
        at=utcnow() - timedelta(seconds=3600),
        message="heartbeat",
    )
    events.append(stale)
    events.heartbeat("runner")

    beats = events.last_heartbeats()
    assert set(beats) == {"broker", "runner"}
    assert beats["broker"]["age_s"] >= 3599
    assert beats["runner"]["age_s"] < 60


def test_last_heartbeat_per_source_wins(events: EventJournal) -> None:
    events.heartbeat("runner", meta={"n": 1})
    events.heartbeat("runner", meta={"n": 2})
    events.heartbeat("runner", meta={"n": 3})

    beats = events.last_heartbeats()
    assert list(beats) == ["runner"]
    assert beats["runner"]["meta"] == {"n": 3}
    assert beats["runner"]["seq"] == events.latest_seq()


def test_heartbeat_needs_a_source(events: EventJournal) -> None:
    with pytest.raises(ValueError):
        events.heartbeat("")


def test_heartbeats_of_an_empty_log_are_empty(events: EventJournal) -> None:
    assert events.last_heartbeats() == {}
    assert events.latest_seq() == 0
    assert events.tail(0) == []


# --- the journals share one database, not one table ---------------------------


def test_both_journals_write_to_the_journal_database(
    decisions: DecisionJournal, events: EventJournal
) -> None:
    decisions.append(make_decision())
    events.append(log_event("x"))

    con = db.connect(db.journal_path())
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    # And the tape is readable as raw JSON by anything that speaks sqlite.
    raw = con.execute("SELECT inputs FROM decisions WHERE id = 'dec-1'").fetchone()[0]
    assert json.loads(raw)["prices"] == {"AAA": 101.25}


def test_a_journal_can_be_handed_an_explicit_connection() -> None:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(db.SCHEMA_SQL)

    journal = EventJournal(con)
    assert journal.con is con
    journal.append(log_event("in memory"))
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    # The on-disk journal is untouched.
    assert EventJournal().count() == 0
