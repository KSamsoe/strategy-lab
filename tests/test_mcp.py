"""The MCP layer: does every tool register, run, and carry its caveats?

The point of this server is not that it exposes the lab -- the CLI already did,
with ``--json`` on every command. The point is that a result cannot be read
without the thing that would embarrass it, because five research sessions each
reached a confident wrong conclusion from a number whose caveat was computable at
the moment it was handed over. These tests pin that property, not the plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from lab.mcp import (
    _common,
    tools_core,
    tools_experiment,
    tools_explore,
    tools_findings,
    tools_jobs,
    tools_live,
    tools_validate,
)

MCP_DIR = Path(_common.__file__).parent


@pytest.fixture()
def cfg_path(tmp_path: Path, seeded_store: dict[str, Any]) -> str:
    path = tmp_path / "bt.yaml"
    path.write_text(
        yaml.safe_dump({
            "strategy": "strategies/buy_and_hold.py",
            "tickers": list(seeded_store["tickers"]),
            "timeframe": "1d",
            "cash": 100000,
            "warmup": 30,
            "benchmark": seeded_store["tickers"][0],
        }),
        encoding="utf-8",
    )
    return str(path)


# --- registration -------------------------------------------------------------


def test_every_tool_registers_with_a_schema_and_a_description():
    import anyio

    from lab.mcp.server import server

    tools = anyio.run(server.list_tools)
    assert len(tools) >= 20
    for t in tools:
        assert t.description and len(t.description) > 40, f"{t.name} needs a real description"
        assert t.input_schema is not None, f"{t.name} has no input schema"


def test_the_server_advertises_how_to_work():
    """Instructions are the only place a caller learns the order of operations."""
    from lab.mcp.server import server

    text = server.instructions or ""
    for token in ("data_coverage", "signal_scan", "backtest", "warnings", "one_sigma"):
        assert token in text, f"instructions should mention {token}"


# --- the property that matters ------------------------------------------------


def test_a_backtest_result_carries_its_own_caveats(cfg_path):
    out = tools_core.backtest(cfg_path, register=False)

    for key in ("windows", "market", "vs_market", "score_one_sigma", "gate", "warnings"):
        assert key in out, f"a result without {key} can be misread"
    for window in ("full", "in_sample", "oos"):
        assert window in out["windows"]
    assert json.dumps(out), "every result must be JSON-serialisable for the wire"


def test_a_cash_heavy_result_is_flagged_as_de_levering():
    """The failure that once produced a 1.44 score on a half-invested book."""
    flagged = _common.warnings_for({
        "windows": {"full": {"trades": 10}, "oos": {"exposure": 0.22}},
        "vs_market": {}, "gate": {}, "score": 1.4, "score_one_sigma": 0.3,
        "market": {"oos": {"sharpe": 0.8}},
    })
    assert any("de-levering" in w for w in flagged)


def test_a_clipped_result_says_the_gate_wrote_it():
    """81% clipping was once cited as evidence a pick was robust."""
    flagged = _common.warnings_for({
        "windows": {"full": {"trades": 10}, "oos": {"exposure": 0.9}},
        "vs_market": {}, "score": 1.0, "score_one_sigma": 0.3,
        "market": {"oos": {"sharpe": 0.5}},
        "gate": {"share_clipped_or_blocked": 0.81,
                 "top_rules": [{"rule": "position_cap"}]},
    })
    assert any("tuning the wrong thing" in w for w in flagged)


def test_a_score_inside_its_error_bar_says_so():
    flagged = _common.warnings_for({
        "windows": {"full": {"trades": 10}, "oos": {"exposure": 0.9}},
        "vs_market": {}, "gate": {},
        "score": 1.05, "score_one_sigma": 0.49,
        "market": {"oos": {"sharpe": 1.20}},
    })
    assert any("within one standard error" in w for w in flagged)


def test_winning_full_period_while_losing_the_holdout_is_named():
    """momo beat the market by 217 points and was written off on 11 months."""
    flagged = _common.warnings_for({
        "windows": {"full": {"trades": 10}, "oos": {"exposure": 0.9}},
        "gate": {}, "score": 0.6, "score_one_sigma": 1.0,
        "market": {"oos": {"sharpe": 0.85}},
        "vs_market": {"full_return": 2.17, "oos_return": -0.04},
    })
    assert any("out of favour rather than broken" in w for w in flagged)


def test_a_run_with_no_trades_is_not_a_result():
    flagged = _common.warnings_for({
        "windows": {"full": {"trades": 0}, "oos": {"exposure": 0.0}},
        "vs_market": {}, "gate": {}, "market": {},
    })
    assert any("no trades" in w for w in flagged)


# --- exploration: the capability the research loop never had -------------------


def test_signal_scan_reports_correlations_with_their_sample_size(cfg_path):
    out = tools_explore.signal_scan(cfg_path, lookbacks=[5, 21], forwards=[5])
    assert out["cross_sectional_persistence"], "should measure at least one horizon"
    for row in out["cross_sectional_persistence"]:
        assert "mean_rank_corr" in row and "n_dates" in row, (
            "a correlation without its sample size invites reading noise as signal"
        )
    assert out["notes"]


def test_conditional_returns_buckets_forward_returns(cfg_path):
    out = tools_explore.conditional_returns(cfg_path, condition="vol", buckets=3)
    assert out["buckets"]
    for row in out["buckets"]:
        assert {"bucket", "mean_forward_return", "std", "n"} <= set(row)


def test_correlation_matrix_reports_effective_breadth(cfg_path):
    """A universe of names that all move together is a universe of one."""
    out = tools_explore.correlation_matrix(cfg_path)
    assert out["effective_independent_bets"] >= 1
    assert out["effective_independent_bets"] <= out["tickers"]


# --- experiments --------------------------------------------------------------


def test_ablate_compares_named_cases_against_one_error_bar(cfg_path):
    out = tools_experiment.ablate(cfg_path, cases={"a": {}, "b": {"cash_buffer": 0.10}})
    assert len(out["results"]) == 2
    assert out["best_case"] in {"a", "b"}
    assert any("sigma" in n for n in out["notes"])


def test_compare_universes_holds_everything_but_the_tickers(cfg_path, seeded_store):
    """The experiment that showed a champion's edge was its ticker list."""
    names = list(seeded_store["tickers"])
    out = tools_experiment.compare_universes(
        cfg_path, universes={"all": names, "half": names[:2]}
    )
    assert len(out["results"]) == 2
    sizes = {r["case"]: r["universe_size"] for r in out["results"]}
    assert sizes["all"] == len(names) and sizes["half"] == 2


# --- validation ---------------------------------------------------------------


def test_deflated_sharpe_punishes_more_trials():
    """The best of many tries is biased upward even when none has an edge."""
    few = tools_validate.deflated_sharpe(1.5, bars=1000, trials=1)
    many = tools_validate.deflated_sharpe(1.5, bars=1000, trials=100)
    assert many["expected_max_sharpe_from_luck"] > few["expected_max_sharpe_from_luck"]
    assert many["deflated_probability"] < few["deflated_probability"]


def test_bootstrap_reports_an_interval_not_just_a_number(cfg_path):
    out = tools_core.backtest(cfg_path, register=True)
    boot = tools_validate.bootstrap(out["run_id"], samples=120, block_bars=5)
    assert boot["sharpe"]["p05"] <= boot["sharpe"]["p50"] <= boot["sharpe"]["p95"]
    assert boot["sharpe"]["std_error"] > 0
    assert any("honest" in n for n in boot["notes"])


def test_bootstrap_vs_benchmark_pairs_the_resamples(cfg_path):
    """The paired test answers the question the unpaired one only looks like it does.

    Strategy and benchmark live through the same crashes, so resampling them
    independently gives each a wide interval and the two overlap almost whatever
    the difference between them is. On a real 21-year run the unpaired interval
    was [0.572, 1.231] and comfortably contained the benchmark's 0.656 -- which
    reads as "no evidence" -- while the paired difference was [+0.057, +0.428]
    with P(strategy <= benchmark) = 1.4%.
    """
    out = tools_core.backtest(cfg_path, register=True)
    paired = tools_validate.bootstrap_vs_benchmark(
        out["run_id"], cfg_path, samples=120, block_bars=5
    )
    diff = paired["difference"]
    assert diff["p05"] <= diff["mean"] <= diff["p95"]
    assert 0.0 <= paired["prob_no_better_than_benchmark"] <= 1.0
    # The observed difference is arithmetic, not a resample: it must be exactly
    # the gap between the two Sharpes it reports.
    obs = paired["observed"]
    assert obs["difference"] == pytest.approx(
        obs["strategy_sharpe"] - obs["benchmark_sharpe"], abs=1e-9
    )
    assert any("same" in n or "paired" in n for n in paired["notes"])


def test_neighbourhood_reports_coverage_it_did_not_have(cfg_path):
    """A HOLDS from half a check has to say it only did half.

    This tool used to fabricate an ``Experiment``, a ``ResearchConfig`` and a
    ``ResearchSession`` to reach the research loop's copy of this arithmetic. It
    now runs its own backtests, so the thing worth pinning is that the verdict
    still carries what the budget could not cover.
    """
    out = tools_core.backtest(cfg_path, register=True, params={"top_n": 2, "lookback": 10})
    nb = tools_core.neighbourhood(out["run_id"], cfg_path, budget=1)
    assert nb["checked"] == 1
    assert nb["coverage"]["complete"] is False
    cov = nb["coverage"]
    assert cov["runs_for_full_coverage"] > cov["runs_spent"]
    assert cov["one_direction_only"] or cov["not_tested"]
    assert "PARTIAL COVERAGE" in nb["verdict"]


# --- layering -----------------------------------------------------------------


def test_the_toolset_imports_no_private_names_from_other_packages():
    """The primary interface must not reach into the layer it demoted.

    It did: nine underscore-prefixed imports from ``lab.agent``, the loop this
    server exists to replace, including the perturbation plan and every window
    metric. That is not a style complaint -- ``CLIPPING_CONFOUNDS`` was defined
    twice with the two copies agreeing, which is the state a constant is in
    immediately before it stops agreeing. Anything both layers need lives in
    ``lab.analysis`` now, and this test is what keeps it there.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            # Inside lab.mcp, a private name is this package's own business.
            if node.module.startswith("lab.mcp"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    offenders.append(f"{path.name}: from {node.module} import {alias.name}")
    assert not offenders, "private cross-package imports: " + "; ".join(offenders)


def test_the_toolset_does_not_import_the_research_loop():
    """Its whole premise is that the agent comes from outside.

    ``lab.agent.research`` carries the provider abstraction, the budget meter and
    the prompt cache -- about 4,600 lines of orchestration a desktop agent already
    provides. A tool that imports it has quietly made this server depend on the
    thing it replaces, and the dependency will not announce itself.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = getattr(node, "module", None) if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if (module or "").startswith("lab.agent.research") or any(
                n.startswith("lab.agent.research") for n in names
            ):
                offenders.append(path.name)
    assert not offenders, f"MCP tools importing the research loop: {sorted(set(offenders))}"


# --- the ledger ---------------------------------------------------------------


def test_findings_are_recorded_searched_and_kept_when_refuted():
    """A refuted claim stays, marked, with the reason -- it is a finding too."""
    f = tools_findings.findings_record(
        "momentum horizon", "signal only at ~126 days", tags=["Momentum"],
        run_ids=["r1"], evidence={"ic_126": 0.04},
    )
    assert f["status"] == "open" and f["tags"] == ["momentum"]
    assert tools_findings.findings_search(query="126")["count"] == 1
    assert tools_findings.findings_search(run_id="r1")["count"] == 1
    assert tools_findings.findings_search(tags=["momentum"])["count"] == 1
    assert tools_findings.findings_search(query="nothing like this")["count"] == 0

    u = tools_findings.findings_update(f["id"], status="refuted", note="broad universe disagreed")
    assert u["status"] == "refuted" and u["history"][-1]["note"] == "broad universe disagreed"
    # Still findable: refuted is not deleted.
    assert tools_findings.findings_search(query="126")["count"] == 1

    g = tools_findings.findings_record("momentum horizon, revised", "63-126 days", supersedes=f["id"])
    assert tools_findings.findings_search(query="momentum horizon")["count"] == 1
    assert tools_findings.findings_search(query="momentum horizon", include_superseded=True)["count"] == 2
    assert g["supersedes"] == f["id"]


def test_findings_resource_renders_the_ledger():
    from lab.registry.findings import Findings

    tools_findings.findings_record("t", "a claim", tags=["x"])
    digest = Findings().digest()
    assert "# Findings" in digest and "a claim" in digest and "[open]" in digest


# --- trials come from the registry ----------------------------------------------


def test_deflated_sharpe_counts_trials_from_the_registry(cfg_path):
    """The caller under-counted every time; the registry cannot.

    Three backtests over the same holdout under two strategy names is three
    trials, whatever the surviving one is called, and passing a smaller number
    does not lower it.
    """
    a = tools_core.backtest(cfg_path, register=True)
    tools_core.backtest(cfg_path, register=True, params={"lookback": 10})
    tools_core.backtest(cfg_path, register=True, strategy="strategies/momo.py")
    out = tools_validate.deflated_sharpe(run_id=a["run_id"], trials=1)
    assert out["trials_source"] == "registry"
    assert out["registered_runs_same_holdout"] == 3
    assert out["trials"] == 3
    # Raising is allowed; that is what the argument is for.
    assert tools_validate.deflated_sharpe(run_id=a["run_id"], trials=40)["trials"] == 40


# --- walk-forward ---------------------------------------------------------------


def test_walk_forward_scores_every_fold_and_says_what_it_did_not_test(cfg_path):
    out = tools_validate.walk_forward(cfg_path, spec="2:1")
    assert out["summary"]["windows"] >= 2
    assert len(out["windows"]) == out["summary"]["windows"]
    for w in out["windows"]:
        assert "oos_sharpe" in w and w["oos_bars"] > 0
        assert "benchmark_oos_return" in w
    assert any("consistency check" in n for n in out["notes"])


def test_walk_forward_with_a_grid_reports_where_the_pick_landed(cfg_path):
    out = tools_validate.walk_forward(
        cfg_path, spec="2:1", strategy="strategies/momo.py",
        grid={"lookback": [10, 20]},
    )
    assert out["grid_size"] == 2
    assert all(w["selected"] is not None for w in out["windows"])
    assert all(w["oos_rank_of_selected"] in (1, 2) for w in out["windows"])


# --- authoring without a filesystem ---------------------------------------------


def test_strategy_write_validates_before_it_writes(monkeypatch, tmp_path):
    from lab.config import reset_settings_cache

    lib = tmp_path / "strategies"
    lib.mkdir()
    monkeypatch.setenv("LAB_STRATEGIES_DIR", str(lib))
    reset_settings_cache()

    bad = tools_core.strategy_write("probe", "def broken(:\n  pass")
    assert bad["ok"] is False and "SyntaxError" in bad["error"]
    assert not (lib / "probe.py").exists()

    api = tools_core.strategy_api()
    ok = tools_core.strategy_write("probe", api["skeleton"])
    assert ok["ok"] is True and (lib / "probe.py").exists()
    assert ok["params"] == {"lookback": 126, "top_n": 5, "gross": 0.95}

    again = tools_core.strategy_write("probe", api["skeleton"])
    assert again["ok"] is False and "overwrite" in again["refused"]
    assert tools_core.strategy_write("probe", api["skeleton"], overwrite=True)["overwrote"] is True
    assert tools_core.strategy_read("probe")["source"] == api["skeleton"]

    for name in ("_private", "../escape", "a/b"):
        with pytest.raises(ValueError):
            tools_core.strategy_write(name, api["skeleton"])


def test_strategy_api_carries_the_contract_and_the_skeleton_loads():
    from pathlib import Path as _P
    import tempfile

    from lab.engine.loader import load_strategy

    api = tools_core.strategy_api()
    assert "def order_target_pct" in api["context_protocol"]
    assert "sma" in api["indicators"] and "rsi" in api["indicators"]
    with tempfile.TemporaryDirectory() as d:
        f = _P(d) / "skel.py"
        f.write_text(api["skeleton"], encoding="utf-8")
        assert load_strategy(f).name == "my_strategy"


# --- paper trading against its backtest -------------------------------------------


def _fake_paper_run(cfg_path, seeded_store, sessions=6, equity_path=None):
    """A paper run the way the daily runner leaves one: a registry row per
    session with one journaled decision each, equity carried across days."""
    import yaml
    from datetime import timedelta

    from lab.engine.events import Decision, new_id
    from lab.engine.loader import load_strategy
    from lab.registry.journal import DecisionJournal
    from lab.registry.runs import RunRegistry, config_hash
    from lab.store import parquet_io

    raw = yaml.safe_load(_P(cfg_path).read_text(encoding="utf-8"))
    name = load_strategy(_P(raw["strategy"])).name
    bars = parquet_io.read_bars(raw["tickers"], timeframe="1d")
    # The synthetic adapter emits weekend bars; the trading clock does not,
    # so a paper run only ever steps on weekdays.
    days = [d for d in sorted(bars["event_time"].unique()) if d.weekday() < 5][-sessions:]
    live_cfg = {
        "strategy": raw["strategy"], "tickers": raw["tickers"], "timeframe": "1d",
        "cash": raw["cash"], "params": {}, "lookback_days": 120, "paper": True,
        "source": None,
    }
    reg, dj = RunRegistry(), DecisionJournal()
    equity = float(raw["cash"])
    run_ids = []
    for i, day in enumerate(days):
        rec = reg.create(
            strategy=name, kind="paper", config_hash=config_hash(live_cfg),
            config=live_cfg, params={}, origin="human",
        )
        run_ids.append(rec.run_id)
        equity = equity_path[i] if equity_path else equity * (1 + 0.001 * (i % 3 - 1))
        # The daily runner decides at 09:35 ET on the bar's own session.
        at = day.to_pydatetime().replace(hour=13, minute=35, second=0, microsecond=0)
        dj.append(Decision(
            id=new_id("d_"), run_id=rec.run_id, strategy=name, at=at,
            portfolio={"equity": equity, "positions": [{"ticker": t} for t in raw["tickers"]]},
            order_ids=["o1"] if i == 0 else [],
        ))
        reg.finish(rec.run_id, metrics={"final_equity": equity})
    return name, run_ids


from pathlib import Path as _P  # noqa: E402


def test_live_vs_backtest_stitches_daily_runs_and_refuses_to_judge_early(cfg_path, seeded_store):
    name, run_ids = _fake_paper_run(cfg_path, seeded_store, sessions=6)
    out = tools_live.live_vs_backtest(strategy=name)
    assert out["runs_stitched"] == len(run_ids)
    assert out["sessions_compared"] >= 5
    assert out["verdict"] == "too short to judge"
    assert any("sessions compared" in w for w in out["warnings"])
    assert out["positions"]["overlap"] == 1.0  # buy-and-hold holds the universe both ways
    assert "session_return_correlation" in out["agreement"]
    # A single run id works too, and says it has one point.
    one = tools_live.live_vs_backtest(run_id=run_ids[0])
    assert one["steps"] == 1


def test_live_vs_backtest_flags_a_reset_book(cfg_path, seeded_store):
    cash = 100000.0
    name, _ = _fake_paper_run(
        cfg_path, seeded_store, sessions=5, equity_path=[cash, 100200.0, cash, 100100.0, 100300.0]
    )
    out = tools_live.live_vs_backtest(strategy=name, min_steps=3)
    assert any("starting cash" in w for w in out["warnings"])


def test_live_status_reads_the_journal_not_the_runner():
    out = tools_live.live_status()
    assert out["count"] == 0 and "kill_switch" in out


# --- jobs ------------------------------------------------------------------------


def test_a_backtest_runs_as_a_job_and_its_result_survives(cfg_path):
    import time

    from lab.mcp import jobs

    q = jobs.JobQueue(cap=1)
    a = q.submit("backtest", {"config": cfg_path, "register": False}, note="first")
    b = q.submit("backtest", {"config": cfg_path, "register": False}, note="second")
    assert a["status"] == "running"
    assert b["status"] == "queued" and b["queue_position"] == 1

    c = q.cancel(b["job_id"])
    assert c["status"] == "cancelled"

    deadline = time.time() + 240
    while q.status(a["job_id"])["status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(1)
    rec = q.status(a["job_id"])
    assert rec["status"] == "done", rec.get("error")
    res = q.result(a["job_id"])
    assert res["result"]["strategy"] == "buy_and_hold.py"
    assert "warnings" in res["result"]

    # A fresh queue object (a restarted server) still finds the result on disk.
    fresh = jobs.JobQueue(cap=1)
    assert fresh.result(a["job_id"])["status"] == "done"
    names = {j["job_id"]: j["status"] for j in fresh.list()}
    assert names[a["job_id"]] == "done" and names[b["job_id"]] == "cancelled"


def test_only_compute_tools_may_run_as_jobs():
    from lab.mcp import jobs

    with pytest.raises(ValueError):
        jobs.JobQueue(cap=1).submit("promote", {"run_id": "x"})
    with pytest.raises(ValueError):
        jobs.JobQueue(cap=1).submit("os.system", {})
    assert set(jobs.JOBABLE) <= {fn.__name__ for fn in _server().TOOLS}


# --- prompts and resources -------------------------------------------------------


def _server():
    from lab.mcp import server

    return server


def test_prompts_only_name_tools_that_exist():
    """A checklist that names a tool this server does not have is worse than none."""
    import inspect
    import re

    import anyio

    srv = _server().server
    tools = dict(srv._tool_manager._tools)

    async def render(name, args):
        r = await srv.get_prompt(name, args)
        return r.messages[0].content.text

    for name, args in (
        ("audit_run", {"run_id": "r1", "config": "cfg/x.yaml"}),
        ("start_research", {"config": "cfg/x.yaml", "brief": "b"}),
    ):
        text = anyio.run(render, name, args)
        called = set(re.findall(r"`([a-z_]+)\(", text))
        assert called and called <= set(tools), sorted(called - set(tools))
        assert "findings" in text
        # Every keyword a prompt shows being passed must be one the tool takes.
        # The first live audit failed at step 6 because the prompt handed
        # `permutation_test` a `run_id` it did not accept.
        for m in re.finditer(r"`([a-z_]+)\(([^`]*)\)`", text):
            fn = tools[m.group(1)].fn
            params = set(inspect.signature(fn).parameters)
            for kw in re.findall(r"(?:^|[,\s(])([a-z_]+)=", m.group(2)):
                assert kw in params, f"{m.group(1)} has no argument {kw!r}"
        for m in re.finditer(r"job_start\(\"([a-z_]+)\",\s*\{([^}]*)\}", text):
            fn = tools[m.group(1)].fn
            params = set(inspect.signature(fn).parameters)
            for kw in re.findall(r"\"([a-z_]+)\":", m.group(2)):
                assert kw in params, f"{m.group(1)} has no argument {kw!r}"


def test_resources_render():
    import json

    import anyio

    srv = _server().server

    async def go():
        uris = {str(r.uri) for r in await srv.list_resources()}
        assert {"lab://findings", "lab://strategy-api", "lab://docs/mcp"} <= uris
        api = await srv.read_resource("lab://strategy-api")
        body = json.loads(list(api)[0].content)
        assert "skeleton" in body
        led = await srv.read_resource("lab://findings")
        assert "# Findings" in list(led)[0].content

    anyio.run(go)


def test_every_registered_tool_has_a_docstring_that_says_what_the_answer_means():
    for fn in _server().TOOLS:
        assert fn.__doc__ and len(fn.__doc__.strip()) > 20, fn.__name__
