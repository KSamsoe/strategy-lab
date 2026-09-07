"""Tests for the console's read layer.

The suite runs against a *real* short backtest written into a throwaway data
directory: synthetic bars go into a temp store, ``run_backtest`` produces real
artifacts, a real registry row and a real decision journal, and the API is asked
to render them. Mocking the store here would test the mocks -- the shapes this
layer has to get right (parquet columns, journal JSON, gate verdicts) only exist
once something has actually run.

Two tests carry more weight than the rest: ``test_no_route_can_increase_exposure``
and ``test_api_never_imports_a_broker``. They are the executable form of the
console's defining property.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from lab.config import Paths, get_settings, reset_settings_cache
from lab.engine.events import EventKind, event
from lab.registry import db as registry_db
from lab.timeutil import UTC, utcnow

START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2022, 6, 30, tzinfo=UTC)
UNIVERSE = ["AAPL", "MSFT", "SPY", "NVDA"]


# --- environment ---------------------------------------------------------------


@pytest.fixture(scope="module")
def lab_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point the whole lab at a temp directory for this module only.

    Module scope, not session scope: a session fixture would still be active
    while the other test files run and would silently move their store too.
    """
    root = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setenv("LAB_DATA_DIR", str(root / "data"))
    mp.setenv("LAB_RUNS_DIR", str(root / "runs"))
    mp.setenv("LAB_KILL_FILE", str(root / "KILL"))
    mp.delenv("LAB_UI_TOKEN", raising=False)
    reset_settings_cache()
    registry_db.close_all()
    try:
        yield root
    finally:
        mp.undo()
        reset_settings_cache()
        registry_db.close_all()


@pytest.fixture(autouse=True)
def isolated_paths(lab_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Paths]:
    """Widen conftest's isolation from per-test to per-module.

    Overrides the autouse ``isolated_paths`` in ``tests/conftest.py``, which is
    function-scoped and re-points ``LAB_DATA_DIR``/``LAB_RUNS_DIR`` at a fresh
    tmpdir for every test. Being function-scoped it is set up *after* ``lab_env``
    and wins, so the module-scoped backtest the whole file renders would land in
    a directory no request can see and every route would answer against an empty
    registry. Same guarantee -- the developer's real store is never reachable --
    one directory wider, which is the granularity a shared fixture corpus needs.

    Re-asserting the env each test also keeps a test that flips a var (token,
    kill switch) from leaking into the next one, and the cache reset on both
    edges keeps a stale ``Settings`` from outliving the change.
    """
    monkeypatch.setenv("LAB_DATA_DIR", str(lab_env / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(lab_env / "runs"))
    monkeypatch.setenv("LAB_KILL_FILE", str(lab_env / "KILL"))
    monkeypatch.delenv("LAB_KILL_SWITCH", raising=False)
    monkeypatch.delenv("LAB_UI_TOKEN", raising=False)
    reset_settings_cache()
    try:
        # No `close_all()` on either edge: the path is unchanged between tests,
        # so dropping the cached sqlite handles would only re-open them.
        yield get_settings().paths
    finally:
        reset_settings_cache()


@pytest.fixture(scope="module")
def runs(lab_env: Path) -> dict[str, Any]:
    """One momentum run (trades, clipped verdicts, hundreds of decisions) and one
    buy-and-hold run, so list filters and compare have something to chew on."""
    from lab.adapters.synthetic import SyntheticAdapter
    from lab.backtest.runner import BacktestConfig, run_backtest
    from lab.store import parquet_io

    parquet_io.write_bars(SyntheticAdapter().fetch_bars(UNIVERSE, "1d", START, END))
    strategies = get_settings().paths.strategies

    momo = run_backtest(
        BacktestConfig(
            strategy=str(strategies / "momo.py"),
            tickers=UNIVERSE,
            start=START,
            end=END,
            warmup=30,
            params={
                "lookback": 20,
                "top_n": 2,
                "trend_ma": 20,
                "min_history": 25,
                "atr_len": 5,
                "exit_rank_buffer": 1,
            },
            limits={"max_position_pct": 0.30, "max_positions": 3},
            notes="api fixture",
        )
    )
    hold = run_backtest(
        BacktestConfig(
            strategy=str(strategies / "buy_and_hold.py"),
            tickers=UNIVERSE[:2],
            start=START,
            end=END,
            limits={"max_position_pct": 0.30, "max_positions": 3},
        )
    )
    return {"momo": momo, "hold": hold}


@pytest.fixture(scope="module")
def client(lab_env: Path, runs: dict[str, Any]) -> Iterator[TestClient]:
    from lab.api.app import create_app

    with TestClient(create_app(console_dist_dir=lab_env / "not-built")) as c:
        yield c


def _journal_con() -> sqlite3.Connection:
    return registry_db.connect(registry_db.journal_path())


# --- health ---------------------------------------------------------------------


def test_health_is_open_and_reports_state(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["auth_required"] is False
    assert body["kill_switch"]["engaged"] is False
    assert body["runs"] >= 2
    assert body["latest_seq"] >= 1
    assert body["console_built"] is False  # this app was built without a dist


# --- runs -----------------------------------------------------------------------


def test_runs_list_and_detail(client: TestClient, runs: dict[str, Any]) -> None:
    listing = client.get("/api/runs", params={"limit": 10}).json()
    ids = [r["run_id"] for r in listing["runs"]]
    assert runs["momo"].run_id in ids and runs["hold"].run_id in ids
    assert listing["count"] == len(listing["runs"])

    only = client.get("/api/runs", params={"strategy": "buy_and_hold"}).json()
    assert {r["strategy"] for r in only["runs"]} == {"buy_and_hold"}

    detail = client.get(f"/api/runs/{runs['momo'].run_id}").json()
    assert detail["strategy"] == "momo"
    assert detail["status"] == "ok"
    assert detail["attempt"] >= 1
    assert detail["config_hash"] and detail["data_version"]
    assert detail["metrics"]["sharpe"] == pytest.approx(runs["momo"].metrics["sharpe"])
    assert detail["params"]["lookback"] == 20
    assert detail["decisions"] > 0
    names = {a["name"] for a in detail["artifacts"]}
    assert {"equity.parquet", "trades.csv", "metrics.json", "config.json"} <= names
    # The optimistic-fill flag has to survive the trip to the UI, loudly.
    assert detail["optimistic_fills"] is False


def test_bad_order_by_is_a_400(client: TestClient) -> None:
    assert client.get("/api/runs", params={"order_by": "sharpe; drop"}).status_code == 400


def test_unknown_run_is_404_everywhere(client: TestClient) -> None:
    for suffix in ("", "/equity", "/trades", "/decisions", "/bars"):
        r = client.get(f"/api/runs/does-not-exist{suffix}")
        assert r.status_code == 404, suffix
        assert "does-not-exist" in r.json()["detail"]


def test_equity_is_parallel_arrays(client: TestClient, runs: dict[str, Any]) -> None:
    body = client.get(f"/api/runs/{runs['momo'].run_id}/equity").json()
    n = body["n"]
    assert n == len(runs["momo"].equity)
    assert len(body["t"]) == len(body["equity"]) == len(body["drawdown"]) == n
    assert len(body["is_oos"]) == n and not any(body["is_oos"])  # no walk-forward split
    assert body["t"] == sorted(body["t"])
    assert body["t"][0].endswith("+00:00")
    assert max(body["drawdown"]) <= 0.0
    assert body["equity"][0] == pytest.approx(100_000.0)
    # NaN is not JSON; the payload must never contain a bare NaN token.
    assert "NaN" not in client.get(f"/api/runs/{runs['momo'].run_id}/equity").text


def test_trades_and_bars(client: TestClient, runs: dict[str, Any]) -> None:
    run_id = runs["momo"].run_id
    trades = client.get(f"/api/runs/{run_id}/trades").json()
    assert trades["count"] == len(trades["trades"]) == len(runs["momo"].trades)
    assert {t["ticker"] for t in trades["trades"]} <= set(UNIVERSE)

    bars = client.get(f"/api/runs/{run_id}/bars", params={"ticker": "aapl"}).json()
    assert bars["ticker"] == "AAPL" and bars["n"] > 200
    assert len(bars["t"]) == len(bars["close"]) == bars["n"]
    assert all(h >= l for h, l in zip(bars["high"], bars["low"]))

    default = client.get(f"/api/runs/{run_id}/bars").json()
    assert default["ticker"] == UNIVERSE[0]
    assert client.get(f"/api/runs/{run_id}/bars", params={"ticker": "TSLA"}).status_code == 404


def test_decision_tape_paginates_and_filters(client: TestClient, runs: dict[str, Any]) -> None:
    run_id = runs["momo"].run_id
    page = client.get(f"/api/runs/{run_id}/decisions", params={"limit": 5}).json()
    assert page["count"] == 5 and page["has_more"] is True
    assert page["total"] == len(runs["momo"].decisions)

    row = next(r for r in page["rows"] if r["n_intents"])
    assert row["summary"]  # the collapsed one-liner
    assert row["detail"]["intents"] and row["detail"]["intents"][0]["ticker"] in UNIVERSE
    assert row["detail"]["verdicts"][0]["action"] in {"pass", "clipped", "blocked"}
    assert "portfolio" in row["detail"] and "inputs" in row["detail"]

    second = client.get(
        f"/api/runs/{run_id}/decisions", params={"limit": 5, "offset": 5}
    ).json()
    assert {r["id"] for r in second["rows"]}.isdisjoint({r["id"] for r in page["rows"]})

    by_ticker = client.get(
        f"/api/runs/{run_id}/decisions", params={"ticker": "nvda", "limit": 50}
    ).json()
    assert by_ticker["count"] > 0
    assert all("NVDA" in r["tickers"] for r in by_ticker["rows"])

    mid = page["rows"][-1]["at"]
    later = client.get(f"/api/runs/{run_id}/decisions", params={"start": mid}).json()
    assert all(r["at"] >= mid for r in later["rows"])

    # A clipped verdict is the whole reason the tape exists: position_cap at 30%.
    clipped = client.get(f"/api/runs/{run_id}/decisions", params={"limit": 2000}).json()
    rules = {
        v["rule"]
        for r in clipped["rows"]
        for v in r["detail"]["verdicts"]
        if v["action"] != "pass"
    }
    assert "position_cap" in rules


def test_compare(client: TestClient, runs: dict[str, Any]) -> None:
    ids = f"{runs['momo'].run_id},{runs['hold'].run_id}"
    body = client.get("/api/runs/compare", params={"ids": ids}).json()
    assert body["ids"] == ids.split(",")
    assert [r["run_id"] for r in body["rows"]] == ids.split(",")
    assert "sharpe" in body["columns"]
    assert client.get("/api/runs/compare", params={"ids": "nope"}).status_code == 404


def test_strategies(client: TestClient) -> None:
    body = client.get("/api/strategies").json()
    found = {s["name"]: s for s in body["strategies"]}
    assert {"momo", "buy_and_hold"} <= set(found)
    assert found["momo"]["loadable"] is True
    assert found["momo"]["params"]["lookback"] == 126


# --- sweeps and the agent's paper trail -----------------------------------------


@pytest.fixture(scope="module")
def sweep(lab_env: Path, runs: dict[str, Any]) -> str:
    from lab.registry.runs import RunRegistry

    registry = RunRegistry()
    for i, sharpe in enumerate([0.4, 1.7, 0.9]):
        rec = registry.create(
            strategy="momo", kind="backtest", sweep_id="sw-1", params={"lookback": 10 * i}
        )
        registry.finish(rec.run_id, metrics={"sharpe": sharpe, "oos_sharpe": sharpe / 2})
    return "sw-1"


def test_sweep_detail_defaults_to_the_oos_metric(client: TestClient, sweep: str) -> None:
    # No sweep.json here, so this exercises the registry fallback: the grid and
    # the walk-forward split are unrecoverable, but the rows still are.
    body = client.get(f"/api/sweeps/{sweep}").json()
    assert body["count"] == 3
    assert body["metric"] == "oos_sharpe"  # never in-sample by default
    assert body["best"]["is_metrics"]["sharpe"] == pytest.approx(1.7)
    assert body["attempts"] >= 3  # the overfitting tell, front and centre
    assert client.get("/api/sweeps/nope").status_code == 404


def test_sweep_detail_prefers_the_artifact_so_the_honesty_flags_survive(
    client: TestClient, lab_env: Path
) -> None:
    """A reconstruction from registry rows loses exactly the parts worth keeping.

    grid, ranked_on and the truncation record only exist in the sweep's own
    artifact, and they are what stop the console presenting a capped, in-sample
    ranking as if it were validation.
    """
    import json

    from lab.registry.runs import RunRegistry, run_artifact_dir

    registry = RunRegistry()
    rec = registry.create(
        strategy="momo", kind="backtest", sweep_id="sw-art", params={"lookback": 20}
    )
    registry.finish(rec.run_id, metrics={"sharpe": 1.1})

    artifact = {
        "sweep_id": "sw-art",
        "strategy": "momo",
        "metric": "sharpe",
        "ranked_on": "in_sample",
        "grid": {"lookback": [20, 40], "top_n": [3, 4]},
        "grid_size": 4,
        "combinations": 2,
        "truncated": True,
        "attempts": 7,
        "runs": [
            {
                "run_id": rec.run_id,
                "params": {"lookback": 20},
                "status": "ok",
                "score": 1.1,
                "is_sharpe": 1.1,
                "oos_sharpe": 0.4,
            }
        ],
        "best": {"run_id": rec.run_id, "params": {"lookback": 20}, "score": 1.1,
                 "is_sharpe": 1.1, "oos_sharpe": 0.4},
        "walk_forward": [
            {"index": 0, "is_start": "2024-01-01T00:00:00+00:00",
             "is_end": "2024-06-01T00:00:00+00:00",
             "oos_start": "2024-06-01T00:00:00+00:00",
             "oos_end": "2024-08-01T00:00:00+00:00",
             "is_sharpe": 1.4, "oos_sharpe": 0.4, "is_bars": 100},
        ],
    }
    d = run_artifact_dir("sw-art")
    (d / "sweep.json").write_text(json.dumps(artifact), encoding="utf-8")

    body = client.get("/api/sweeps/sw-art").json()
    assert body["grid"] == {"lookback": [20, 40], "top_n": [3, 4]}
    assert body["ranked_on"] == "in_sample", "must not be presented as validated"
    assert body["truncated"] == {"applied": True, "requested": 4, "ran": 2}
    assert body["attempts"] == 7
    assert body["runs"][0]["is_metrics"]["sharpe"] == pytest.approx(1.1)
    assert body["runs"][0]["oos_metrics"]["sharpe"] == pytest.approx(0.4)

    window = body["walk_forward"][0]
    assert window["oos_start"] == "2024-06-01T00:00:00+00:00"
    assert window["is_metrics"] == {"sharpe": 1.4, "bars": 100}
    assert window["oos_metrics"] == {"sharpe": 0.4}
    # The boundary fields share the is_/oos_ prefix and must not leak in as
    # metrics named "start" and "end".
    assert "start" not in window["is_metrics"]


@pytest.fixture(scope="module")
def agent_runs(lab_env: Path, runs: dict[str, Any]) -> str:
    from lab.registry.runs import RunRegistry

    registry = RunRegistry()
    parent: str | None = None
    for i, sharpe in enumerate([0.2, 0.8, 0.5]):
        rec = registry.create(
            strategy="agent_momo",
            kind="backtest",
            origin="agent-loop",
            parent_run_id=parent,
            params={"lookback": 20 + i},
            notes=f"iteration {i}",
        )
        registry.finish(rec.run_id, metrics={"oos_sharpe": sharpe})
        parent = rec.run_id
        with registry_db.transaction(registry_db.connect()) as con:
            con.execute(
                "INSERT INTO agent_calls (id, run_id, at, strategy, model, tool, prompt,"
                " response, input_tokens, output_tokens, cost_usd, latency_ms, ok, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"call-{i}", rec.run_id, utcnow().isoformat(), "agent_momo",
                    "claude-sonnet-5", "propose_targets", "prompt text", "response text",
                    1000, 200, 0.01, 850.0, 1, "",
                ),
            )
    return "agent_momo"


def test_agent_lineage_and_calls(client: TestClient, agent_runs: str) -> None:
    lineage = client.get(f"/api/agent/lineage/{agent_runs}").json()
    assert [s["n"] for s in lineage["steps"]] == [1, 2, 3]
    assert lineage["metric"] == "oos_sharpe"
    assert lineage["steps"][1]["parent_run_id"] == lineage["steps"][0]["run_id"]
    assert lineage["best_run_id"] == lineage["steps"][1]["run_id"]
    assert lineage["attempts"] == 3
    assert client.get("/api/agent/lineage/nobody").status_code == 404

    calls = client.get("/api/agent/calls").json()
    assert calls["count"] == 3
    assert calls["total_cost_usd"] == pytest.approx(0.03)
    assert calls["total_tokens"] == 3600
    one = client.get(
        "/api/agent/calls", params={"run_id": lineage["steps"][0]["run_id"]}
    ).json()
    assert one["count"] == 1 and one["calls"][0]["model"] == "claude-sonnet-5"


# --- live ------------------------------------------------------------------------


@pytest.fixture
def live_row(lab_env: Path) -> Iterator[str]:
    name = "momo_live"
    with registry_db.transaction(_journal_con()) as con:
        con.execute(
            "INSERT OR REPLACE INTO live_strategies (strategy, run_id, kind, status, config,"
            " host, started_at, updated_at, paused) VALUES (?,?,?,?,?,?,?,?,0)",
            (name, "r1", "paper", "running", json.dumps({"tickers": ["AAPL"]}),
             "localhost", utcnow().isoformat(), utcnow().isoformat()),
        )
    yield name
    with registry_db.transaction(_journal_con()) as con:
        con.execute("DELETE FROM live_strategies WHERE strategy = ?", (name,))


def test_live_strategies(client: TestClient, live_row: str) -> None:
    body = client.get("/api/live/strategies").json()
    row = next(s for s in body["strategies"] if s["strategy"] == live_row)
    assert row["paused"] is False and row["status"] == "running"
    assert row["config"] == {"tickers": ["AAPL"]}


def test_live_events_paginate_by_seq(client: TestClient, runs: dict[str, Any]) -> None:
    first = client.get("/api/live/events", params={"since": 0, "limit": 5}).json()
    assert first["count"] == 5
    seqs = [e["seq"] for e in first["events"]]
    assert seqs == sorted(seqs)
    assert first["latest_seq"] >= max(seqs)

    resumed = client.get("/api/live/events", params={"since": seqs[-1], "limit": 5}).json()
    assert min(e["seq"] for e in resumed["events"]) > seqs[-1]

    fills = client.get(
        "/api/live/events", params={"since": 0, "limit": 5, "kinds": "fill"}
    ).json()
    assert {e["kind"] for e in fills["events"]} == {"fill"}


def test_live_health_computes_age_at_request_time(client: TestClient, live_row: str) -> None:
    from lab.registry.journal import EventJournal

    journal = EventJournal()
    journal.heartbeat("runner", meta={"next_fire": "16:00"})
    journal.append(
        event(
            EventKind.HEARTBEAT, "stale_adapter",
            at=utcnow() - timedelta(minutes=30), message="heartbeat", payload={},
        )
    )

    body = client.get("/api/live/health").json()
    beats = {b["source"]: b for b in body["heartbeats"]}
    assert beats["runner"]["age_s"] < 60 and beats["runner"]["stale"] is False
    assert beats["runner"]["meta"] == {"next_fire": "16:00"}
    assert beats["stale_adapter"]["age_s"] > 1000 and beats["stale_adapter"]["stale"] is True
    assert {a["name"] for a in body["adapters"]} >= {"synthetic", "govgreed"}
    assert body["quota"]["source"] == "govgreed"
    assert body["breaker"]["tripped"] is False
    assert body["kill_switch"]["engaged"] is False
    assert body["strategies"] >= 1

    tight = client.get("/api/live/health", params={"stale_after_s": 0.001}).json()
    assert all(b["stale"] for b in tight["heartbeats"])


def test_live_health_reads_the_breaker_off_the_journal(client: TestClient) -> None:
    from lab.registry.journal import EventJournal

    EventJournal().append(
        event(
            EventKind.BREAKER, "gate", at=utcnow(), strategy="momo_live",
            message="daily loss limit", payload={"tripped": True, "reason": "daily loss 3.1%"},
        )
    )
    breaker = client.get("/api/live/health").json()["breaker"]
    assert breaker["tripped"] is True
    assert breaker["reason"] == "daily loss 3.1%"
    assert breaker["strategy"] == "momo_live"
    assert breaker["age_s"] >= 0
    # Re-arming is a CLI action, and it is only believed once it is journaled.
    EventJournal().append(
        event(EventKind.BREAKER, "cli", at=utcnow(), message="re-armed",
              payload={"tripped": False, "reason": ""})
    )
    assert client.get("/api/live/health").json()["breaker"]["tripped"] is False


# --- the asymmetric controls ------------------------------------------------------


def _control_events(action: str) -> list[dict[str, Any]]:
    from lab.registry.journal import EventJournal

    return [
        e
        for e in EventJournal().tail(0, limit=5000, kinds=[EventKind.LOG])
        if (e["payload"] or {}).get("control") == action
    ]


def test_pause_is_applied_and_journaled(client: TestClient, live_row: str) -> None:
    before = len(_control_events("pause"))
    body = client.post(
        "/api/control/pause", json={"strategy": live_row, "reason": "eyeballing it"}
    ).json()
    assert body["ok"] is True and body["applied"] is True and body["seq"] > 0

    row = _journal_con().execute(
        "SELECT paused FROM live_strategies WHERE strategy = ?", (live_row,)
    ).fetchone()
    assert row["paused"] == 1

    journaled = _control_events("pause")
    assert len(journaled) == before + 1
    assert journaled[-1]["payload"]["reason"] == "eyeballing it"
    assert journaled[-1]["source"] == "console"

    # Idempotent: pausing a paused strategy is not an error and stays journaled.
    again = client.post("/api/control/pause", json={"strategy": live_row}).json()
    assert again["applied"] is False and "already paused" in again["message"]
    assert len(_control_events("pause")) == before + 2

    assert client.post("/api/control/pause", json={"strategy": "ghost"}).status_code == 404
    assert client.post("/api/control/pause", json={}).status_code == 422


def test_cancel_orders_is_a_journaled_request_not_a_broker_call(
    client: TestClient, live_row: str
) -> None:
    before = len(_control_events("cancel_orders"))
    body = client.post("/api/control/cancel_orders", json={"strategy": live_row}).json()
    assert body["ok"] is True
    # False on purpose: only the process holding the broker session may cancel.
    assert body["applied"] is False
    assert len(_control_events("cancel_orders")) == before + 1
    assert client.post("/api/control/cancel_orders", json={"strategy": "ghost"}).status_code == 404


def test_kill_engages_the_switch_and_is_journaled(client: TestClient) -> None:
    kill_file = get_settings().kill_file
    try:
        body = client.post("/api/control/kill", json={"reason": "smells wrong"}).json()
        assert body["applied"] is True and body["action"] == "kill"
        assert kill_file.exists()
        engaged, reason = get_settings().kill_switch_engaged()
        assert engaged and reason
        assert client.get("/api/health").json()["kill_switch"]["engaged"] is True
        assert _control_events("kill")[-1]["payload"]["reason"] == "smells wrong"
    finally:
        # Releasing is a CLI action; the test does it by hand so the rest of the
        # module does not run under a tripped kill switch.
        kill_file.unlink(missing_ok=True)
        reset_settings_cache()
    assert client.get("/api/health").json()["kill_switch"]["engaged"] is False


# --- the defining property ---------------------------------------------------------

#: The complete mutation surface.
#:
#: The rule is about *market* exposure: nothing here may cause an order. Three of
#: these only ever reduce what is running. The fourth, research/start, is the one
#: deliberate exception -- a research session runs backtests and cannot reach a
#: broker, so it cannot move exposure at all, but it does spend model budget and
#: execute model-authored Python, which is why the server keeps it behind an
#: explicit opt-in rather than folding it in with the others.
ALLOWED_MUTATIONS = {
    ("POST", "/api/control/pause"),
    ("POST", "/api/control/cancel_orders"),
    ("POST", "/api/control/kill"),
    ("POST", "/api/control/research/start"),
    ("POST", "/api/control/research/stop"),
}

#: Paths exempt from the word check below, because their verb is about a research
#: session rather than a position. Listed one by one: a prefix exemption would let
#: the next route in that namespace inherit it silently.
WORD_CHECK_EXEMPT = {"/api/control/research/start", "/api/control/research/stop"}

#: Verbs that would move the system the other way. None of them may appear in a
#: path at all -- not as a route, not as a "convenience" alias.
FORBIDDEN_IN_PATH = (
    "start", "resume", "launch", "restart", "release", "arm", "enable", "raise",
    "increase", "submit", "buy", "sell", "allocate", "rebalance", "execute",
    "override", "unpause", "unkill", "reset",
)


def _iter_routes(router: Any) -> Iterator[Any]:
    """FastAPI keeps included routers as one entry rather than flattening them,
    so the walk has to recurse or the audit below inspects three routes."""
    for route in getattr(router, "routes", []) or []:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _iter_routes(inner)
        else:
            yield route


def test_no_route_can_increase_exposure(client: TestClient) -> None:
    mutations: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for route in _iter_routes(client.app):
        path = getattr(route, "path", None)
        if not path:
            continue
        paths.add(path)
        for method in getattr(route, "methods", None) or set():
            if method in {"POST", "PUT", "PATCH", "DELETE"}:
                mutations.add((method, path))

    assert mutations == ALLOWED_MUTATIONS

    for path in paths:
        if path in WORD_CHECK_EXEMPT:
            continue
        lowered = path.lower()
        for word in FORBIDDEN_IN_PATH:
            assert word not in lowered, f"{path} looks like it can increase exposure"

    # And the schema agrees: no verb beyond GET outside the three controls.
    schema = client.get("/api/openapi.json").json()
    for path, operations in schema["paths"].items():
        for method in operations:
            assert method.upper() in {"GET", "HEAD", "OPTIONS"} or (
                method.upper(),
                path,
            ) in ALLOWED_MUTATIONS


def test_the_research_exception_cannot_reach_a_broker_or_a_limit() -> None:
    """The exception is narrow, and these are the two ways it could stop being.

    A research session is allowed because it cannot cause an order. That holds
    only while (a) the launch request cannot compose a backtest config -- naming
    `limits` in a form is raising a limit from the UI wearing a different hat --
    and (b) nothing in the route can start a *strategy* rather than a session.
    """
    from lab.api.models import ResearchStartRequest

    fields = set(ResearchStartRequest.model_fields)
    for forbidden in ("limits", "fills", "tickers", "universe", "cash", "start", "end", "broker"):
        assert forbidden not in fields, f"the launch form must not be able to set {forbidden!r}"
    assert "config" in fields, "it names an existing config instead"

    # The broker half of the invariant is already covered, and better, by
    # test_api_never_imports_a_broker: it walks the whole lab/api package with
    # ast rather than grepping text, so it cannot be fooled by a docstring that
    # merely mentions the word.


def test_starting_research_is_refused_without_the_opt_in(client: TestClient) -> None:
    body = {"config": "momo.yaml", "brief": "beat the benchmark"}
    resp = client.post("/api/control/research/start", json=body)
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert "LAB_UI_ALLOW_RESEARCH" in detail, "say exactly how to enable it"

    options = client.get("/api/control/research/options").json()
    assert options["enabled"] is False and options["reason"]


def test_stopping_research_never_needs_the_opt_in(client: TestClient, tmp_path) -> None:
    """Stopping only ever reduces what is running, so it is always available."""
    session = get_settings().paths.runs / "research-stoppable"
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text("{}", encoding="utf-8")

    resp = client.post(
        "/api/control/research/stop", json={"session_id": "research-stoppable"}
    )
    assert resp.status_code == 200
    assert (session / "STOP").exists(), "a sentinel file, so it reaches a session we did not start"

    assert client.post(
        "/api/control/research/stop", json={"session_id": "../escape"}
    ).status_code in (400, 404)


def test_api_never_imports_a_broker() -> None:
    """The UI reads journals. If it could reach a broker, a console bug could
    become a trading incident, and a runner crash could be blamed on the UI."""
    package = Path(get_settings().paths.root) / "lab" / "api"
    banned = ("broker", "alpaca", "alpaca_trade_api", "ib_insync", "lab.live.runner")
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                modules.append(base)
                modules.extend(f"{base}.{a.name}" for a in node.names)
        for module in modules:
            assert not any(b in module for b in banned), f"{path.name} imports {module}"


def test_contract_routes_all_exist(client: TestClient) -> None:
    """docs/CONTRACTS.md, lab/api section -- every documented route, verbatim."""
    schema = client.get("/api/openapi.json").json()["paths"]
    expected = [
        ("get", "/api/health"),
        ("get", "/api/runs"),
        ("get", "/api/runs/{run_id}"),
        ("get", "/api/runs/{run_id}/equity"),
        ("get", "/api/runs/{run_id}/trades"),
        ("get", "/api/runs/{run_id}/decisions"),
        ("get", "/api/runs/{run_id}/bars"),
        ("get", "/api/runs/compare"),
        ("get", "/api/sweeps/{sweep_id}"),
        ("get", "/api/strategies"),
        ("get", "/api/live/strategies"),
        ("get", "/api/live/events"),
        ("get", "/api/live/health"),
        ("get", "/api/agent/lineage/{strategy}"),
        ("get", "/api/agent/calls"),
        ("post", "/api/control/pause"),
        ("post", "/api/control/cancel_orders"),
        ("post", "/api/control/kill"),
    ]
    for method, path in expected:
        assert path in schema and method in schema[path], f"{method.upper()} {path} missing"
    ws = {r.path for r in _iter_routes(client.app) if type(r).__name__ == "APIWebSocketRoute"}
    assert "/api/ws/events" in ws


def test_unknown_api_path_is_404_not_the_spa(client: TestClient) -> None:
    r = client.get("/api/nonsense")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


# --- auth ---------------------------------------------------------------------------


def test_open_access_when_no_token_is_configured(client: TestClient) -> None:
    assert get_settings().ui_token is None
    for path in ("/api/health", "/api/runs", "/api/live/health", "/api/strategies"):
        assert client.get(path).status_code == 200, path


def test_bearer_token_is_enforced_when_set(
    lab_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lab.api.app import create_app

    monkeypatch.setenv("LAB_UI_TOKEN", "s3cret")
    reset_settings_cache()
    with TestClient(create_app(console_dist_dir=lab_env / "not-built")) as guarded:
        assert guarded.get("/api/health").status_code == 200  # always open
        assert guarded.get("/api/health").json()["auth_required"] is True

        for path in ("/api/runs", "/api/live/health", "/api/strategies", "/api/agent/calls"):
            assert guarded.get(path).status_code == 401, path
        assert guarded.post("/api/control/kill", json={"reason": "x"}).status_code == 401
        assert guarded.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401

        ok = {"Authorization": "Bearer s3cret"}
        assert guarded.get("/api/runs", headers=ok).status_code == 200
        # A bare value works too, because that is what people curl.
        assert guarded.get("/api/runs", headers={"Authorization": "s3cret"}).status_code == 200

        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with guarded.websocket_connect("/api/ws/events"):
                pass
        with guarded.websocket_connect("/api/ws/events?token=s3cret") as ws:
            assert ws.receive_json()["type"] == "hello"

    assert not get_settings().kill_switch_engaged()[0]


# --- the SPA -----------------------------------------------------------------------


def test_placeholder_page_when_the_console_is_not_built(client: TestClient) -> None:
    body = client.get("/").text
    assert "npm install" in body and "npm run build" in body
    assert client.get("/runs/whatever").status_code == 200  # deep link still answers


def test_serves_the_built_spa_when_it_exists(lab_env: Path) -> None:
    from lab.api.app import create_app

    dist = lab_env / "dist"
    (dist / "assets").mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<title>console</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("export const x = 1;", encoding="utf-8")

    with TestClient(create_app(console_dist_dir=dist)) as spa:
        assert spa.get("/").text == "<title>console</title>"
        assert "export const x" in spa.get("/assets/app.js").text
        # Client-side routes fall back to the shell rather than 404ing.
        assert spa.get("/runs/abc/decisions").text == "<title>console</title>"
        assert spa.get("/api/health").status_code == 200
        assert spa.get("/api/nope").status_code == 404


# --- the websocket -------------------------------------------------------------------


def test_ws_streams_new_events_and_resumes_from_a_cursor(client: TestClient) -> None:
    from lab.registry.journal import EventJournal

    journal = EventJournal()
    with client.websocket_connect("/api/ws/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        base = hello["since"]
        assert base == hello["latest_seq"]  # no `since` means "start at the head"

        seq = journal.append(
            event(EventKind.LOG, "test", at=utcnow(), message="after-connect", ticker="AAPL")
        )
        frame = ws.receive_json()
        assert frame["type"] == "event"
        assert frame["seq"] == seq
        assert frame["event"]["message"] == "after-connect"
        assert frame["event"]["ticker"] == "AAPL"

    # Reconnect with the cursor from before that event: it must be replayed, not
    # lost. This is the entire point of a seq cursor.
    with client.websocket_connect(f"/api/ws/events?since={base}") as ws:
        assert ws.receive_json()["since"] == base
        replayed = ws.receive_json()
        assert replayed["seq"] == seq
        assert replayed["event"]["message"] == "after-connect"

    # ...and a cursor at the head yields nothing old, only what lands next.
    with client.websocket_connect(f"/api/ws/events?since={seq}") as ws:
        assert ws.receive_json()["type"] == "hello"
        nxt = journal.append(event(EventKind.LOG, "test", at=utcnow(), message="later"))
        frame = ws.receive_json()
        assert frame["seq"] == nxt and frame["event"]["message"] == "later"


def test_ws_filters_by_kind(client: TestClient) -> None:
    from lab.registry.journal import EventJournal

    journal = EventJournal()
    with client.websocket_connect("/api/ws/events?kinds=fill") as ws:
        ws.receive_json()
        journal.append(event(EventKind.LOG, "test", at=utcnow(), message="ignored"))
        journal.append(event(EventKind.FILL, "broker", at=utcnow(), message="AAPL +1 @ 1.00"))
        frame = ws.receive_json()
        assert frame["event"]["kind"] == "fill"
        assert frame["event"]["message"] == "AAPL +1 @ 1.00"
