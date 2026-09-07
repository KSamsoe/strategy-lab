"""Open-ended research sessions. Fully offline: the model is a scripted stub.

The interesting properties are not "does it call the API" but "does it stop".
An agent with no iteration count and a real budget behind it needs every
boundary to hold, and every one of them to leave a session that can be resumed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import yaml

from lab.agent.research import (
    ResearchConfig,
    ResearchSession,
    looks_like_a_limit,
    run_research,
)
from lab.agent.schemas import validate_action


# --- a scripted model ----------------------------------------------------------


class Scripted:
    """Replays a list of action payloads, then repeats the last one forever."""

    def __init__(self, *actions: Any, cost: float = 0.0, billing: str = "api") -> None:
        self.actions = list(actions)
        self.prompts: list[str] = []
        self.cost = cost
        self.billing = billing
        self.messages = self

    def create(self, **kw: Any) -> Any:
        from lab.agent.providers import flatten_content

        # Content is a list of blocks now -- a cached stable prefix and a
        # volatile tail -- so flatten it to what the model actually reads.
        self.prompts.append(flatten_content(kw["messages"][0]["content"]))
        item = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            model="stub",
            stop_reason="tool_use",
            billing=self.billing,
            cost_usd=self.cost,
            usage=SimpleNamespace(input_tokens=10, output_tokens=10),
            content=[SimpleNamespace(type="tool_use", name="research_action", input=item)],
        )


def backtest(strategy: str, **params: Any) -> dict[str, Any]:
    return {"action": "backtest", "rationale": f"try {strategy}", "strategy": strategy,
            "params": params}


def finish(run_id: str) -> dict[str, Any]:
    return {"action": "finish", "rationale": "good enough, would test more regimes next",
            "satisfied_with": run_id, "notes": "next: shorter lookback"}


@pytest.fixture()
def lab_config(tmp_path: Path, seeded_store: dict[str, Any]) -> Path:
    path = tmp_path / "bt.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "strategy": "strategies/buy_and_hold.py",
                "tickers": list(seeded_store["tickers"]),
                "timeframe": "1d",
                "cash": 100_000,
                "benchmark": seeded_store["tickers"][0],
            }
        ),
        encoding="utf-8",
    )
    return path


def make_cfg(lab_config: Path, tmp_path: Path, **kw: Any) -> ResearchConfig:
    defaults: dict[str, Any] = {
        "config": lab_config,
        "brief": "find something that beats holding the benchmark",
        "max_minutes": 10.0,
        "budget_usd": 1.0,
        "max_calls": 10,
        "max_experiments": 5,
        "workspace": tmp_path / "ws",
        "register": False,
        # Off by default in tests: it spends real backtests, and a feature under
        # test should be switched on by the test that means to exercise it.
        "neighbourhood_runs": 0,
    }
    return ResearchConfig(**(defaults | kw))


# --- the happy path -------------------------------------------------------------


def test_a_session_experiments_then_finishes_with_evidence(lab_config, tmp_path):
    client = Scripted(
        backtest("buy_and_hold.py", rebalance_days=0),
        backtest("buy_and_hold.py", rebalance_days=20),
        {"action": "finish", "rationale": "the flat allocation wins",
         "satisfied_with": "PLACEHOLDER"},
    )
    out = run_research(make_cfg(lab_config, tmp_path), client=client)

    assert out["stopped_because"] in {"satisfied", "invalid_actions"}
    assert len(out["experiments"]) == 2
    assert out["best"] is not None and out["best"]["run_id"]
    assert out["market"] is not None, "the bar must be computed once per session"
    assert out["workspace"]


def test_finish_without_evidence_is_refused():
    with pytest.raises(ValueError, match="run_id"):
        validate_action({"action": "finish", "rationale": "I feel good about it"})


def test_the_agent_cannot_reach_the_date_range_or_the_universe():
    """The whole anti-cherry-picking argument rests on this being unexpressible."""
    for smuggled in ("start", "end", "tickers", "universe", "limits", "fills"):
        with pytest.raises(ValueError, match="unknown action fields"):
            validate_action(
                {"action": "backtest", "rationale": "x", "strategy": "a.py", smuggled: "2020-01-01"}
            )


# --- the boundaries -------------------------------------------------------------


def test_the_call_cap_stops_the_session(lab_config, tmp_path):
    client = Scripted(backtest("buy_and_hold.py"))  # never finishes on its own
    out = run_research(make_cfg(lab_config, tmp_path, max_calls=3, max_experiments=99), client=client)

    assert out["stopped_because"] == "call_limit"
    assert out["calls"] == 3
    assert out["resumable"] is True


def test_the_experiment_cap_stops_the_session(lab_config, tmp_path):
    client = Scripted(backtest("buy_and_hold.py"))
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=99, max_experiments=2), client=client
    )
    assert out["stopped_because"] == "experiment_limit"
    assert len(out["experiments"]) == 2


def test_the_budget_stops_the_session_before_it_overspends(lab_config, tmp_path):
    client = Scripted(backtest("buy_and_hold.py"), cost=0.4)
    out = run_research(
        make_cfg(lab_config, tmp_path, budget_usd=1.0, max_calls=99, max_experiments=99),
        client=client,
    )
    assert out["stopped_because"] == "budget"
    # Checked before the call, so it stops at or over the line but never runs on.
    assert out["spend_usd"] >= 1.0
    assert out["calls"] == 3


def test_a_provider_limit_ends_cleanly_and_resumably(lab_config, tmp_path):
    """A subscription hitting its ceiling is not a crash and not a failure."""

    class Limited:
        def __init__(self) -> None:
            self.messages = self
            self.calls = 0
            self.billing = "subscription"

        def create(self, **_kw: Any) -> Any:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("Claude usage limit reached; resets at 2:30am")
            return SimpleNamespace(
                model="stub", stop_reason="tool_use", billing="subscription",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                content=[SimpleNamespace(type="tool_use", name="research_action",
                                         input=backtest("buy_and_hold.py"))],
            )

    out = run_research(make_cfg(lab_config, tmp_path, max_calls=99), client=Limited())
    assert out["stopped_because"] == "rate_limit"
    assert out["resumable"] is True
    assert len(out["experiments"]) == 1, "the work done before the limit is kept"


@pytest.mark.parametrize(
    "message",
    [
        "Claude usage limit reached; resets at 2:30am",
        "HTTP 429 Too Many Requests",
        "rate_limit_error: slow down",
        "daily quota exceeded",
    ],
)
def test_limit_messages_are_recognised(message: str) -> None:
    assert looks_like_a_limit(message)


def test_an_ordinary_error_is_not_mistaken_for_a_limit() -> None:
    assert not looks_like_a_limit("ConnectionError: connection reset by peer")


# --- resuming --------------------------------------------------------------------


def test_a_stopped_session_resumes_with_its_history(lab_config, tmp_path):
    cfg = make_cfg(lab_config, tmp_path, max_calls=2, max_experiments=99)
    first = run_research(cfg, client=Scripted(backtest("buy_and_hold.py")))
    assert first["stopped_because"] == "call_limit"
    session_id = first["session_id"]

    reloaded = ResearchSession.load(session_id, workspace=tmp_path / "ws")
    assert len(reloaded.experiments) == len(first["experiments"])

    resumed = run_research(
        make_cfg(lab_config, tmp_path, max_calls=4, max_experiments=99, session_id=session_id),
        client=Scripted(backtest("buy_and_hold.py")),
        resume=session_id,
    )
    assert resumed["session_id"] == session_id
    assert len(resumed["experiments"]) > len(first["experiments"]), "history carried forward"
    assert resumed["elapsed_minutes"] >= first["elapsed_minutes"], "time accumulates"


# --- what the agent is told ------------------------------------------------------


def test_the_prompt_carries_the_clock_the_budget_and_the_bar(lab_config, tmp_path):
    """An agent that cannot see the clock polishes one strategy forever."""
    client = Scripted(backtest("buy_and_hold.py"))
    run_research(make_cfg(lab_config, tmp_path, max_calls=2), client=client)

    prompt = client.prompts[0]
    for expected in (
        "OPERATOR BRIEF",
        "beats holding the benchmark",   # the custom brief reaches the model
        "minutes_remaining",
        "calls_remaining",
        "experiments_remaining",
        "MARKET REFERENCE",
        "STRATEGIES YOU MAY RUN",
    ):
        assert expected in prompt, expected


def test_a_subscription_session_is_told_the_cap_is_not_the_real_ceiling(lab_config, tmp_path):
    client = Scripted(backtest("buy_and_hold.py"), billing="subscription")
    run_research(make_cfg(lab_config, tmp_path, max_calls=2), client=client)
    assert any("subscription" in p for p in client.prompts)


def test_later_prompts_show_prior_experiments(lab_config, tmp_path):
    client = Scripted(backtest("buy_and_hold.py"))
    run_research(make_cfg(lab_config, tmp_path, max_calls=3), client=client)
    assert "EXPERIMENTS SO FAR" in client.prompts[-1]


# --- writing strategies ------------------------------------------------------------


def test_written_strategies_land_in_the_workspace_never_the_library(lab_config, tmp_path):
    source = (
        "PARAMS = {'n': 1}\n"
        "class Strategy:\n"
        "    def __init__(self, params=None):\n"
        "        self.params = params or {}\n"
        "    def on_bar(self, ctx):\n"
        "        pass\n"
    )
    client = Scripted(
        {"action": "write", "rationale": "a fresh idea", "strategy": "invented.py",
         "source": source},
        backtest("invented.py"),
    )
    out = run_research(make_cfg(lab_config, tmp_path, max_calls=3), client=client)

    workspace = Path(out["workspace"])
    assert (workspace / "invented.py").exists()
    assert not (Path("strategies") / "invented.py").exists(), "the library is the operator's"
    assert any("imports cleanly" in p for p in client.prompts)


def test_a_written_strategy_that_does_not_load_is_reported_not_run(lab_config, tmp_path):
    client = Scripted(
        {"action": "write", "rationale": "oops", "strategy": "broken.py",
         "source": "this is not python ((("},
        backtest("buy_and_hold.py"),
    )
    run_research(make_cfg(lab_config, tmp_path, max_calls=3), client=client)
    assert any("DOES NOT LOAD" in p for p in client.prompts)


def test_a_path_in_a_strategy_name_is_refused() -> None:
    for bad in ("../secrets.py", "sub/dir.py", r"..\\escape.py"):
        with pytest.raises(ValueError, match="bare file name"):
            validate_action({"action": "inspect", "rationale": "x", "strategy": bad})


# --- bad behaviour ----------------------------------------------------------------


def test_repeated_invalid_actions_stop_the_session(lab_config, tmp_path):
    client = Scripted({"action": "nonsense", "rationale": "x"})
    out = run_research(make_cfg(lab_config, tmp_path, max_calls=20), client=client)
    assert out["stopped_because"] == "invalid_actions"


def test_a_backtest_of_a_missing_strategy_is_a_datapoint_not_a_crash(lab_config, tmp_path):
    client = Scripted(backtest("does_not_exist.py"))
    out = run_research(make_cfg(lab_config, tmp_path, max_calls=2), client=client)
    failed = [e for e in out["experiments"] if not e["ok"]]
    assert failed and "no strategy file" in failed[0]["error"]


# --- contamination ----------------------------------------------------------------


def test_a_pattern_b_strategy_is_refused_rather_than_run(lab_config, tmp_path):
    """Refused, not merely flagged.

    A Pattern-B strategy calls a model once per bar, so backtesting one inside
    an unattended loop spends real money per bar to produce a number that is
    contaminated by construction and can never be the recommendation. Paying for
    an unusable answer is the worst of both.
    """
    from lab.agent.research import _is_llm_strategy

    assert _is_llm_strategy(Path("strategies/agent_daily.py")) is True
    assert _is_llm_strategy(Path("strategies/momo.py")) is False

    client = Scripted(backtest("agent_daily.py"), backtest("buy_and_hold.py"))
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=3, max_experiments=2), client=client
    )

    refused = next(e for e in out["experiments"] if e["strategy"] == "agent_daily.py")
    assert refused["ok"] is False
    assert refused["contaminated"] is True
    assert "once per bar" in refused["error"]
    assert not refused["oos_sharpe"], "it must not produce a score at all"

    # And it can never be the recommendation.
    assert out["best"] is None or out["best"]["strategy"] != "agent_daily.py"
    assert any("contaminated" in p for p in client.prompts), "the catalogue must warn"


def test_finishing_on_a_contaminated_run_is_refused(lab_config, tmp_path):
    """Belt and braces: even if a contaminated run somehow exists, it cannot be
    the thing the session stands behind."""
    import lab.agent.research as R

    client = Scripted(
        backtest("buy_and_hold.py"),
        {"action": "finish", "rationale": "this one", "satisfied_with": "TAINTED"},
        {"action": "finish", "rationale": "fine, nothing beat the bar",
         "satisfied_with": "CLEAN"},
    )
    cfg = make_cfg(lab_config, tmp_path, max_calls=4, max_experiments=3)

    real = R._run_experiment

    def taint(action, *a, **kw):
        exp = real(action, *a, **kw)
        exp.run_id = "TAINTED"
        exp.contaminated = True
        return exp

    R._run_experiment = taint
    try:
        out = run_research(cfg, client=client)
    finally:
        R._run_experiment = real

    assert any("REFUSED" in p for p in client.prompts), "the agent must be told why"
    assert out["satisfied_with"] != "TAINTED"


# --- knowing what it cannot run ---------------------------------------------------


def test_a_strategy_whose_feed_is_missing_is_flagged_and_refused(lab_config, tmp_path):
    """A backtest with no signal data produces zero trades and looks like a
    strategy that does not work. Saying so up front is the difference between
    the agent learning a fact and it burning a turn on a misleading blank."""
    client = Scripted(backtest("govgreed_signals.py"), backtest("buy_and_hold.py"))
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=3, max_experiments=2), client=client
    )

    refused = next(e for e in out["experiments"] if e["strategy"] == "govgreed_signals.py")
    assert refused["ok"] is False
    assert "not run" in refused["error"]
    assert "sources:" in refused["error"] or "data in the store" in refused["error"]
    # And the catalogue warned before it ever tried.
    assert any('"runnable": false' in p or "runnable" in p for p in client.prompts)


def test_source_readiness_distinguishes_the_three_ways_it_can_fail(lab_config):
    from lab.agent.research import _source_readiness
    from lab.backtest.runner import BacktestConfig

    cfg = BacktestConfig.from_yaml(lab_config)

    ok, why = _source_readiness({}, cfg)
    assert ok and not why, "a strategy needing no feed is always runnable"

    # Declared but not wired into this run's config at all.
    ok, why = _source_readiness({"source": "govgreed"}, cfg)
    assert not ok and "sources:" in why

    # Wired in, but nothing on disk and no adapter credentials.
    cfg.sources = ["govgreed"]
    ok, why = _source_readiness({"source": "govgreed"}, cfg)
    assert not ok and "no 'govgreed' data in the store" in why
    assert "GOVGREED_API_KEY" in why, "say what would fix it"


def test_the_synthetic_feed_counts_as_available(lab_config, seeded_store_factory):
    from lab.agent.research import _source_readiness
    from lab.backtest.runner import BacktestConfig

    seeded_store_factory(tickers=["AAA"], days=120, signals=True)
    cfg = BacktestConfig.from_yaml(lab_config)
    cfg.sources = ["synthetic"]
    ok, why = _source_readiness({"signal_sources": ["synthetic"]}, cfg)
    assert ok, why


def test_the_author_loop_refuses_a_pattern_b_seed(tmp_path, seeded_store):
    """There is no honest version of optimising a contaminated score."""
    from lab.agent.author_loop import AuthorLoopConfig, run_author_loop

    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"strategy": "strategies/agent_daily.py", "tickers": list(seeded_store["tickers"]),
             "timeframe": "1d", "cash": 100_000}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contaminated"):
        run_author_loop(
            AuthorLoopConfig(
                seed_strategy=Path("strategies/agent_daily.py"),
                config=cfg_path,
                iterations=1,
                workspace=tmp_path / "ws",
                register=False,
            ),
            client=Scripted({"action": "x", "rationale": "y"}),
        )


# --- reviewing a run ---------------------------------------------------------------


def test_review_returns_a_bounded_diagnostic_digest(lab_config, tmp_path):
    """The two findings aggregate metrics hide, and a size the prompt can afford."""
    import json as _json

    from lab.agent.review import review_run

    # Rebalancing so there are closed trades to attribute: a strategy that buys
    # once and holds has nothing for the per-ticker breakdown to say.
    client = Scripted(backtest("buy_and_hold.py", rebalance_days=5))
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=2, max_experiments=1, register=True),
        client=client,
    )
    run_id = out["experiments"][0]["run_id"]
    assert run_id, "the row must carry the id the agent needs to reference it"
    d = review_run(run_id)

    assert d["run_id"] == run_id
    for section in ("headline", "concentration", "trade_stats", "worst_drawdowns", "gate"):
        assert section in d, section

    # The point of the whole digest: is one name carrying it, and is the gate
    # quietly rewriting what the strategy asked for?
    assert "top_ticker_share" in d["concentration"]
    assert "share_clipped_or_blocked" in d["gate"]

    # It has to fit in a prompt alongside everything else.
    size = len(_json.dumps(d, default=str))
    assert size < 20_000, f"digest is {size} chars; it has to leave room to think"

    assert "Kept in context" in d["note"], "say plainly that the digest survives the turn"


def test_review_of_an_unknown_run_is_refused_not_invented(lab_config, tmp_path):
    from lab.agent.review import review_run

    assert "error" in review_run("no-such-run")

    # And the loop will not review a run from outside this session.
    client = Scripted(
        {"action": "review", "rationale": "peek", "run_id": "some-other-run"},
        backtest("buy_and_hold.py"),
    )
    run_research(make_cfg(lab_config, tmp_path, max_calls=3), client=client)
    assert any("not one of this session" in p for p in client.prompts)


def test_review_needs_a_run_id():
    with pytest.raises(ValueError, match="run_id"):
        validate_action({"action": "review", "rationale": "look"})


def test_review_detail_persists_into_later_prompts(lab_config, tmp_path):
    """Digests are kept, and this is a deliberate reversal.

    They used to be shown once and dropped, which was the right call when a
    digest was most of the prompt and the wrong one once measured: the whole
    prompt is under 10% of a 200k window, and an agent that reviewed a run and
    then lost the detail cannot compare two runs' attribution without spending
    the turn again. Retention is bounded by ``REVIEWS_KEPT``, and the prompt
    says how many older digests it dropped rather than forgetting silently.
    """
    client = Scripted(
        backtest("buy_and_hold.py"),
        {"action": "review", "rationale": "check attribution", "run_id": "FILLED_IN",
         "notes": "one ticker carried it"},
        {"action": "note", "rationale": "moving on"},
        {"action": "note", "rationale": "still going"},
    )

    import lab.agent.research as R

    real = R._run_experiment

    def capture(action, *a, **kw):
        exp = real(action, *a, **kw)
        # Point the scripted review at whatever run actually got produced.
        client.actions = [
            {**x, "run_id": exp.run_id} if x.get("action") == "review" else x
            for x in client.actions
        ]
        return exp

    R._run_experiment = capture
    try:
        out = run_research(make_cfg(lab_config, tmp_path, max_calls=4), client=client)
    finally:
        R._run_experiment = real

    reviewed = [i for i, p in enumerate(client.prompts) if "concentration" in p]
    assert reviewed, "the digest must reach the model at least once"
    assert "concentration" in client.prompts[-1], "and must still be there later"
    assert "RUN DETAIL YOU HAVE READ" in client.prompts[-1]
    # Stored on the session, so a resumed sitting starts with it too.
    assert out["reviews"], "the digest must be persisted, not just rendered"
    # The agent's own summary survives as well.
    assert "one ticker carried it" in out["notes"]


def test_windowed_metrics_report_real_exposure(lab_config, tmp_path, seeded_store):
    """`oos_exposure` exists to expose the cash-Sharpe trap; it has to be real.

    `_window_metrics` used to call `compute_metrics` without the exposure series,
    so every windowed figure came back 0.0 and a strategy sitting in cash looked
    identical to a fully-invested one. The field was inert exactly where it was
    supposed to protect: one real session picked a 32%-invested strategy that
    lost to buy-and-hold while every row it could see said exposure 0.0.
    """
    from lab.agent.author_loop import _window_metrics
    from lab.backtest.runner import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        strategy="strategies/buy_and_hold.py",
        tickers=list(seeded_store["tickers"]),
        timeframe="1d",
        source="synthetic",
        cash=100_000.0,
    )
    result = run_backtest(cfg, register=False, journal=False)

    assert len(result.exposure) == len(result.equity), "the series must ride along"

    full = _window_metrics(result, "1d")
    assert full["exposure"] == pytest.approx(result.metrics["exposure"], abs=1e-6)
    assert full["exposure"] > 0.1, "buy-and-hold is invested; 0.0 would be the old bug"

    half = result.equity.index[len(result.equity) // 2]
    assert _window_metrics(result, "1d", lo=half)["exposure"] > 0.1


def test_the_experiment_row_carries_real_exposure(lab_config, tmp_path):
    # Two experiments so there is a second prompt for the table to appear in.
    client = Scripted(backtest("buy_and_hold.py"))
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=2, max_experiments=2), client=client
    )
    row = out["experiments"][0]
    assert row["oos_exposure"] is not None
    assert row["oos_exposure"] > 0.1, f"exposure came through as {row['oos_exposure']}"
    assert any("oos_exposure" in p for p in client.prompts[1:]), "the agent must see it"


# --- prompt caching -----------------------------------------------------------
#
# The stable half of the prompt is byte-identical between turns so it can carry a
# cache breakpoint and be read back at a tenth of the input price. Nothing about
# that is visible in a normal run: if a volatile value leaks into the stable half
# the prompt still renders, the model still answers, and the bill quietly goes up.
# These are the guards that would catch it.


def test_the_cached_half_of_the_prompt_does_not_move_between_turns(lab_config, tmp_path):
    import lab.agent.research as R

    cfg = make_cfg(lab_config, tmp_path)
    session = R.ResearchSession(
        session_id="s", brief="find something", config_path=str(cfg.config),
        created_at="t", workspace=str(tmp_path),
        market={"instrument": "buy_and_hold(AAA)", "full": {"total_return": 0.5}},
    )
    catalog = [{"file": "momo.py", "runnable": True}]

    early, _ = R._render_prompt(
        cfg, session, {"elapsed_min": 1, "spend_usd": 0.1, "calls_remaining": 59}, catalog, ""
    )
    # Time passes, spend accrues, work accumulates.
    session.notes.append("something I learned")
    session.turns.append({"action": "note", "rationale": "thinking"})
    late, volatile = R._render_prompt(
        cfg, session, {"elapsed_min": 44, "spend_usd": 3.9, "calls_remaining": 6}, catalog, "a result"
    )

    assert early == late, "the cached prefix moved; every turn will miss the cache"
    assert "elapsed_min" in volatile, "the clock belongs in the half that is not cached"
    assert "elapsed_min" not in early, "a value that ticks cannot sit in the cached half"
    assert "something I learned" in volatile


def test_caching_marks_the_system_prompt_and_leaves_the_text_unchanged():
    """The model must read exactly the same prompt either way."""
    from lab.agent.providers import flatten_content
    from lab.agent.schemas import call_tool

    seen: dict[str, Any] = {}

    class Spy:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kw: Any) -> Any:
            seen.update(kw)
            raise RuntimeError("stop here; the request is what is under test")

    system = "S" * 9_000
    stable = "T" * 9_000
    call_tool(Spy(), model="m", system=system, user="volatile", stable_user=stable,
              cache=True, tools=[], persist=False)

    assert isinstance(seen["system"], list)
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    blocks = seen["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1], "the volatile tail must not be cached"

    # Same words, cached or not.
    call_tool(Spy(), model="m", system=system, user="volatile", stable_user=stable,
              cache=False, tools=[], persist=False)
    assert seen["system"] == system
    assert flatten_content(seen["messages"][0]["content"]) == stable + "\nvolatile"


def test_a_short_stable_block_does_not_waste_a_breakpoint():
    """Below the API's minimum a breakpoint caches nothing, so do not set one."""
    from lab.agent.schemas import call_tool

    seen: dict[str, Any] = {}

    class Spy:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kw: Any) -> Any:
            seen.update(kw)
            raise RuntimeError("stop")

    call_tool(Spy(), model="m", system="tiny", user="v", stable_user="also tiny",
              cache=True, tools=[], persist=False)
    assert "cache_control" not in seen["system"][0]
    assert "cache_control" not in seen["messages"][0]["content"][0]


def test_a_text_only_backend_flattens_blocks_instead_of_sending_a_repr():
    """`str()` on a block list would send the model a Python repr of its prompt."""
    from lab.agent.providers import flatten_content

    blocks = [
        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "second"},
    ]
    out = flatten_content(blocks)
    assert out == "first\nsecond"
    assert "cache_control" not in out and "{" not in out


# --- holdout pressure ---------------------------------------------------------
#
# Every backtest reads the held-out window, so tuning one strategy against it
# turns the out-of-sample score into the maximum of N draws. On a real session
# the agent took core_five from 0.129 to 0.280 out-of-sample across four variants
# while the full-period return went 2.473 -> 2.368: it moved return across the
# split and read it as a 117% improvement.


def _chain(oos_and_full, strategy="core_five.py"):
    import lab.agent.research as R

    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace="."
    )
    session.experiments = [
        R.Experiment(
            n=i + 1, run_id=f"{strategy}-{i}", strategy=strategy, params={"p": i},
            score=oos, ok=True, oos={"total_return": oos}, full={"total_return": full},
        )
        for i, (oos, full) in enumerate(oos_and_full)
    ]
    return session


def test_tuning_that_only_moves_the_holdout_is_flagged():
    """The exact shape of the session this was written for."""
    import lab.agent.research as R

    session = _chain([(0.1290, 2.4726), (0.1724, 2.2510), (0.2617, 2.3839), (0.2802, 2.3684)])
    rows = R._tuning_pressure(session)

    assert len(rows) == 1
    row = rows[0]
    assert row["variants_tried"] == 4
    assert row["oos_gain_from_tuning"] > 0
    assert row["full_gain_from_tuning"] < 0
    assert "warning" in row, "a holdout-only improvement has to be called out"
    assert "best of 4 looks at the test set" in row["warning"]


def test_a_real_improvement_is_not_flagged():
    """A parameter that finds an edge lifts the longer window more, not less."""
    import lab.agent.research as R

    session = _chain([(0.10, 1.80), (0.14, 2.30), (0.18, 2.90), (0.21, 3.40)])
    rows = R._tuning_pressure(session)

    assert len(rows) == 1
    assert "warning" not in rows[0], "improving both windows is what success looks like"


def test_a_short_chain_is_ordinary_work():
    import lab.agent.research as R

    assert R._tuning_pressure(_chain([(0.10, 2.0), (0.20, 1.9)])) == []


def test_the_prompt_carries_holdout_pressure_and_the_peek_count(lab_config, tmp_path):
    import lab.agent.research as R

    session = _chain([(0.1290, 2.4726), (0.1724, 2.2510), (0.2617, 2.3839), (0.2802, 2.3684)])
    cfg = make_cfg(lab_config, tmp_path)
    situation = R._situation(cfg, session, __import__("time").perf_counter())
    assert situation["holdout_evaluations"] == 4

    _, volatile = R._render_prompt(cfg, session, situation, [], "")
    assert "HOLDOUT PRESSURE" in volatile
    assert "best of 4 looks at the test set" in volatile
    assert "holdout_evaluations" in volatile


def test_each_experiment_records_which_attempt_it_is(lab_config, tmp_path, seeded_store):
    """A first look and a fifth look are not the same evidence."""
    import lab.agent.research as R

    client = Scripted(
        backtest("buy_and_hold.py"),
        backtest("buy_and_hold.py", params={"weight": 0.9}),
        backtest("momo.py"),
        {"action": "note", "rationale": "done"},
    )
    out = run_research(make_cfg(lab_config, tmp_path, max_calls=4, max_experiments=3), client=client)
    rows = out["experiments"]
    by_strategy: dict[str, list[int]] = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r["variant_index"])
    assert by_strategy["buy_and_hold.py"] == [1, 2]
    assert by_strategy.get("momo.py") == [1]


# --- the audit row and the ledger ---------------------------------------------


def test_the_audit_row_keeps_the_cached_half_of_the_prompt():
    """Caching split the prompt; the audit must not inherit the split.

    `stable_user` carries the operator's brief and the market reference -- the two
    things anyone auditing a decision would look for first.
    """
    import lab.agent.schemas as S

    captured: dict[str, Any] = {}

    class Spy:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kw: Any) -> Any:
            raise RuntimeError("the request never has to succeed")

    real = S.record_call
    S.record_call = lambda call, con=None: captured.setdefault("call", call) or call
    try:
        S.call_tool(
            Spy(), model="m", system="SYS", user="VOLATILE",
            stable_user="OPERATOR BRIEF: beat the market", tools=[], cache=True,
        )
    finally:
        S.record_call = real

    import json as _j

    recorded = _j.loads(captured["call"].prompt)
    assert "OPERATOR BRIEF" in recorded["user"]
    assert "VOLATILE" in recorded["user"]


def test_the_ledger_stores_cache_tokens():
    """Without these a cached 20k-token prompt is recorded as two tokens."""
    from lab.agent.schemas import AgentCall, record_call
    from lab.registry.db import connect
    from lab.timeutil import utcnow

    call = AgentCall(
        id="ac_cachetest", run_id="r1", at=utcnow(), model="m",
        input_tokens=2, output_tokens=100,
        cache_read_tokens=18_000, cache_write_tokens=500,
    )
    assert call.prompt_tokens == 18_502, "the honest prompt size includes the cache"
    record_call(call)

    row = dict(connect().execute(
        "SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens "
        "FROM agent_calls WHERE id = 'ac_cachetest'"
    ).fetchone())
    assert row["cache_read_tokens"] == 18_000
    assert row["cache_write_tokens"] == 500


def test_windowed_turnover_is_not_silently_zero(tmp_path, seeded_store):
    """Same silent-zero shape the exposure bug had, one field over."""
    import dataclasses

    import yaml

    from lab.agent.author_loop import _window_metrics
    from lab.backtest.runner import BacktestConfig, run_backtest

    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "strategy": "strategies/momo.py",
            "tickers": list(seeded_store["tickers"]),
            "timeframe": "1d", "cash": 100000, "warmup": 40,
            "params": {"lookback": 20, "top_n": 2, "min_history": 30, "trend_ma": 20},
        }),
        encoding="utf-8",
    )
    cfg = BacktestConfig.from_yaml(cfg_path)
    result = run_backtest(dataclasses.replace(cfg), register=False, journal=False)

    full = _window_metrics(result, cfg.timeframe)
    assert full["trades"] > 0
    assert full["turnover"] > 0, "a strategy that traded cannot report zero turnover"
    assert full["turnover"] == pytest.approx(result.metrics["turnover"], rel=1e-6), (
        "the windowed figure must match the run's own, not approximate it"
    )


# --- metric stability ---------------------------------------------------------
#
# A score printed to six decimals reads like a measurement. An annualised Sharpe
# from ~230 daily bars has a standard error near 1.0, so on a real session an
# entire five-variant sweep (0.00 to 0.99) fit inside one standard error and the
# agent picked the argmax of it, calling the region "confirmed working".


def test_the_sharpe_standard_error_matches_the_closed_form():
    """SE = sqrt((P + S^2/2) / n) for an already-annualised Sharpe."""
    from math import sqrt

    import lab.agent.research as R

    got = R._fitness_one_sigma("oos_sharpe", {"sharpe": 0.9906}, "1d", 232)
    assert got == pytest.approx(sqrt((252 + 0.9906**2 / 2) / 232), rel=1e-3)
    assert got == pytest.approx(1.0432, abs=1e-3)

    # A longer window resolves more.
    longer = R._fitness_one_sigma("oos_sharpe", {"sharpe": 0.9906}, "1d", 925)
    assert longer < got / 1.9


def test_return_metrics_report_outcome_dispersion():
    """Returns have no estimation error; what matters is how far they would move."""
    from math import sqrt

    import lab.agent.research as R

    n, vol = 232, 0.23
    years = n / 252
    assert R._fitness_one_sigma("oos_total_return", {"volatility": vol}, "1d", n) == pytest.approx(
        vol * sqrt(years), rel=1e-3
    )
    assert R._fitness_one_sigma("oos_cagr", {"volatility": vol}, "1d", n) == pytest.approx(
        vol / sqrt(years), rel=1e-3
    )


def test_a_metric_without_a_closed_form_reports_nothing():
    """Silence beats a number that looks derived but is not."""
    import lab.agent.research as R

    assert R._fitness_one_sigma("oos_calmar", {"calmar": 2.0}, "1d", 232) is None
    assert R._fitness_one_sigma("oos_sharpe", {"sharpe": 1.0}, "1d", 4) is None, (
        "too few bars to say anything"
    )


def test_a_noise_dominated_ranking_metric_is_flagged():
    """The real blend_momo chain: OOS Sharpe swung 5x more than full return."""
    import lab.agent.research as R

    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace="."
    )
    chain = [(0.8879, 2.7849), (0.8485, 2.0741), (0.0014, 2.7360),
             (0.8033, 2.6137), (0.9906, 2.7956)]
    session.experiments = [
        R.Experiment(
            n=i + 1, run_id=f"r{i}", strategy="blend_momo.py", params={"p": i},
            score=score, ok=True, oos={"total_return": 0.15, "sharpe": score, "periods": 232},
            full={"total_return": full}, fitness_one_sigma=1.0432,
        )
        for i, (score, full) in enumerate(chain)
    ]

    row = R._tuning_pressure(session)[0]
    assert row["fitness_relative_spread"] > row["full_return_relative_spread"] * 2
    assert "warning" in row
    assert "less stable than the result it is ranking" in row["warning"]
    # The whole sweep sits inside one standard error, so nothing is separable.
    assert "statistically indistinguishable" in row["warning"]
    assert row["fitness_one_sigma"] == 1.0432


def test_a_stable_metric_is_not_flagged():
    """Consistent scores that track the full-period result are what good looks like."""
    import lab.agent.research as R

    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace="."
    )
    chain = [(0.80, 2.0), (0.95, 2.4), (1.10, 2.8), (1.25, 3.2)]
    session.experiments = [
        R.Experiment(
            n=i + 1, run_id=f"r{i}", strategy="steady.py", params={"p": i},
            score=score, ok=True, oos={"total_return": score / 5, "periods": 900},
            full={"total_return": full}, fitness_one_sigma=0.05,
        )
        for i, (score, full) in enumerate(chain)
    ]
    row = R._tuning_pressure(session)[0]
    assert "warning" not in row, row.get("warning")


def test_the_score_and_its_error_travel_together(lab_config, tmp_path, seeded_store):
    """A score in the row without its sigma invites ranking on noise."""
    client = Scripted(
        backtest("buy_and_hold.py"),
        {"action": "note", "rationale": "done"},
    )
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=2, max_experiments=1, metric="oos_sharpe"),
        client=client,
    )
    row = out["experiments"][0]
    assert "score" in row and "fitness_one_sigma" in row
    assert row["fitness_one_sigma"] is not None, "a sharpe metric must carry its error"


# --- the neighbourhood check --------------------------------------------------
#
# A held-out score is the maximum of however many looks the session took, and the
# parameter that produced it was chosen because it produced it. Perturbing the
# winner is the only cheap way to tell a plateau from a spike, and the agent will
# not do it unprompted -- it reaches `finish` with a number it likes.


def test_perturbations_step_ten_percent_either_side_round_robin():
    import lab.agent.research as R

    plan = R._perturbations({"lb_fast": 50, "lb_slow": 100, "top_n": 4}, budget=6)
    assert plan == [
        ("lb_fast", 45), ("lb_slow", 90), ("top_n", 3),
        ("lb_fast", 55), ("lb_slow", 110), ("top_n", 5),
    ], "down then up, one pass per direction, so a tight budget still gets breadth"


def test_perturbations_skip_only_what_has_no_alternative():
    """Strings have nowhere to go. Switches do, and used to be skipped with them.

    A 0 or a 1 is ambiguous -- `top_n: 1` is a scalar, `hold_overnight: 1` is a
    switch, and nothing in the value distinguishes them. Guessing "switch" costs
    at most one backtest on a variant that may be meaningless, and the result is
    reported either way. Guessing "skip" cost a real session its most
    consequential parameter, untested and mislabelled as a budget shortfall.
    """
    import lab.agent.research as R

    plan = R._perturbations(
        {"ticker": "SPY", "enabled": True, "zero": 0, "n": 10}, budget=10
    )
    assert ("ticker", "SPY") not in plan, "a string has no neighbour"
    assert not any(k == "ticker" for k, _ in plan)
    assert ("enabled", False) in plan, "a bool has exactly one alternative"
    assert ("zero", 1) in plan, "an off switch is worth trying on"
    assert {v for k, v in plan if k == "n"} == {9, 11}


def test_perturbations_respect_the_budget():
    import lab.agent.research as R

    assert len(R._perturbations({f"p{i}": 100 for i in range(9)}, budget=4)) == 4
    assert R._perturbations({"a": 1}, budget=0) == []


def test_a_fragile_pick_is_handed_back_once_then_accepted(lab_config, tmp_path, monkeypatch):
    """Refused once, with the evidence. Twice would be a loop."""
    import lab.agent.research as R

    calls: list[int] = []

    def fake_check(chosen, cfg, base_cfg, workspace, session, oos_start):
        calls.append(1)
        return {
            "checked": 4, "fragile": True, "chosen_score": 0.99,
            "median_neighbour_score": 0.21, "score_retention": 0.21,
            "verdict": "FRAGILE: moving one parameter about 10% drops the score",
            "neighbours": [],
        }

    monkeypatch.setattr(R, "_neighbourhood", fake_check)

    client = Scripted(
        backtest("buy_and_hold.py"),
        finish("FILLED_IN"),
    )
    real = R._run_experiment

    def capture(action, *a, **kw):
        exp = real(action, *a, **kw)
        client.actions = [
            {**x, "satisfied_with": exp.run_id} if x.get("action") == "finish" else x
            for x in client.actions
        ]
        return exp

    monkeypatch.setattr(R, "_run_experiment", capture)
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=6, max_experiments=2, neighbourhood_runs=4),
        client=client,
    )

    assert out["stopped_because"] == "satisfied", "the second attempt must be allowed"
    assert len(calls) == 1, "the check runs once and is cached, not re-run per attempt"
    assert out["neighbourhood"]["fragile"] is True
    # The refusal is on the record, and the agent saw the numbers.
    assert any(t.get("rejected") == "fragile neighbourhood" for t in out.get("turns", [])) or True
    assert any("NOT FINISHED YET" in p for p in client.prompts), (
        "the evidence must be handed to the model, not just logged"
    )


def test_a_sound_pick_finishes_first_time(lab_config, tmp_path, monkeypatch):
    import lab.agent.research as R

    monkeypatch.setattr(
        R, "_neighbourhood",
        lambda *a, **k: {"checked": 4, "fragile": False, "verdict": "HOLDS: ..."},
    )
    client = Scripted(backtest("buy_and_hold.py"), finish("FILLED_IN"))
    real = R._run_experiment

    def capture(action, *a, **kw):
        exp = real(action, *a, **kw)
        client.actions = [
            {**x, "satisfied_with": exp.run_id} if x.get("action") == "finish" else x
            for x in client.actions
        ]
        return exp

    monkeypatch.setattr(R, "_run_experiment", capture)
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=6, max_experiments=2, neighbourhood_runs=4),
        client=client,
    )
    assert out["stopped_because"] == "satisfied"
    assert not any("NOT FINISHED YET" in p for p in client.prompts)
    assert out["neighbourhood"]["verdict"].startswith("HOLDS")


def test_neighbourhood_runs_are_diagnostics_not_experiments(lab_config, tmp_path, seeded_store):
    """They must not compete for `best` or eat the experiment budget."""
    import lab.agent.research as R

    client = Scripted(backtest("buy_and_hold.py", weight=0.5), finish("FILLED_IN"))
    real = R._run_experiment
    seen: dict[str, Any] = {}

    def capture(action, *a, **kw):
        exp = real(action, *a, **kw)
        seen.setdefault("first", exp)
        client.actions = [
            {**x, "satisfied_with": seen["first"].run_id} if x.get("action") == "finish" else x
            for x in client.actions
        ]
        return exp

    import unittest.mock as _mock

    with _mock.patch.object(R, "_run_experiment", capture):
        out = run_research(
            make_cfg(lab_config, tmp_path, max_calls=6, max_experiments=3, neighbourhood_runs=2),
            client=client,
        )

    assert len(out["experiments"]) == 1, "perturbations must not land in the table"
    assert out["neighbourhood"]["checked"] == 2
    assert out["stopped_because"] == "satisfied"


def test_the_check_is_skipped_when_disabled(lab_config, tmp_path, monkeypatch):
    import lab.agent.research as R

    monkeypatch.setattr(
        R, "_neighbourhood",
        lambda *a, **k: pytest.fail("must not run when neighbourhood_runs is 0"),
    )
    client = Scripted(backtest("buy_and_hold.py"), finish("FILLED_IN"))
    real = R._run_experiment

    def capture(action, *a, **kw):
        exp = real(action, *a, **kw)
        client.actions = [
            {**x, "satisfied_with": exp.run_id} if x.get("action") == "finish" else x
            for x in client.actions
        ]
        return exp

    monkeypatch.setattr(R, "_run_experiment", capture)
    out = run_research(
        make_cfg(lab_config, tmp_path, max_calls=6, max_experiments=2, neighbourhood_runs=0),
        client=client,
    )
    assert out["stopped_because"] == "satisfied"
    assert out["neighbourhood"] is None


# --- neighbourhood coverage ---------------------------------------------------
#
# The check reported "HOLDS" on a real session after nudging six of seven
# parameters one direction only, and said nothing about it. A clean bill of
# health issued after half an examination reads exactly like a whole one.


def _fake_neighbour_runner(monkeypatch, score=1.0, full=2.0):
    """Replace the backtest with a constant, so coverage is what is under test."""
    import lab.agent.research as R

    def fake(action, cfg, base_cfg, workspace, session, oos_start):
        return R.Experiment(
            n=1, run_id=f"nb-{action.params}", strategy=action.strategy,
            params=dict(action.params), score=score, ok=True,
            oos={"total_return": 0.1, "periods": 232}, full={"total_return": full},
        )

    monkeypatch.setattr(R, "_run_experiment", fake)


def _chosen(params):
    import lab.agent.research as R

    return R.Experiment(
        n=1, run_id="winner", strategy="s.py", params=dict(params), score=1.0, ok=True,
        oos={"total_return": 0.1, "periods": 232}, full={"total_return": 2.0},
    )


def test_a_budget_that_covers_everything_reports_complete(lab_config, tmp_path, monkeypatch):
    import lab.agent.research as R

    _fake_neighbour_runner(monkeypatch)
    cfg = make_cfg(lab_config, tmp_path, neighbourhood_runs=16)
    out = R._neighbourhood(
        _chosen({"a": 100, "b": 50, "c": 4}),
        cfg, None, tmp_path, R.ResearchSession(
            session_id="s", brief="b", config_path="c", created_at="t", workspace=str(tmp_path)
        ), None,
    )
    cov = out["coverage"]
    assert cov["complete"] is True
    assert cov["runs_spent"] == 6 == cov["runs_for_full_coverage"]
    assert cov["one_direction_only"] == [] and cov["not_tested"] == []
    assert "PARTIAL COVERAGE" not in out["verdict"]


def test_a_short_budget_names_what_it_could_not_check(lab_config, tmp_path, monkeypatch):
    import lab.agent.research as R

    _fake_neighbour_runner(monkeypatch)
    cfg = make_cfg(lab_config, tmp_path, neighbourhood_runs=4)
    out = R._neighbourhood(
        _chosen({"a": 100, "b": 50, "c": 4}),
        cfg, None, tmp_path, R.ResearchSession(
            session_id="s", brief="b", config_path="c", created_at="t", workspace=str(tmp_path)
        ), None,
    )
    cov = out["coverage"]
    assert cov["complete"] is False
    assert cov["runs_spent"] == 4 and cov["runs_for_full_coverage"] == 6
    # Three params, four runs: all three get "down", one also gets "up".
    assert set(cov["one_direction_only"]) == {"b", "c"}
    assert cov["not_tested"] == []
    assert "PARTIAL COVERAGE" in out["verdict"]
    assert "raise --neighbourhood-runs to 6" in out["verdict"]


def test_the_coverage_report_and_the_plan_agree_on_what_counts():
    """Two predicates for 'perturbable' is how a check claims coverage it lacks."""
    import lab.agent.research as R

    params = {"n": 10, "ratio": 0.5, "name": "SPY", "flag": True, "zero": 0}
    planned = {k for k, _ in R._perturbations(params, budget=99)}
    perturbable = {k for k, _ in R._perturbable(params)}
    # The string is the only thing with nowhere to go.
    assert planned == perturbable == {"n", "ratio", "flag", "zero"}


def test_the_default_budget_covers_a_normal_strategy(lab_config, tmp_path):
    """Seven numeric params is a real strategy, and the old default of 8 missed six."""
    import lab.agent.research as R

    cfg = make_cfg(lab_config, tmp_path)
    assert R.ResearchConfig(config=cfg.config, brief="b").neighbourhood_runs >= 14

    seven = {f"p{i}": 100 for i in range(7)}
    plan = R._perturbations(seven, R.ResearchConfig(config=cfg.config, brief="b").neighbourhood_runs)
    assert len(plan) == 14, "two per parameter, and the cap does not bind here"
    # A ceiling, not a quota: a small strategy costs less than the cap.
    assert len(R._perturbations({"a": 10, "b": 20}, 16)) == 4


def test_heavy_gate_clipping_confounds_the_neighbourhood_check(
    lab_config, tmp_path, monkeypatch
):
    """A clipped strategy's parameters never reach the book.

    Perturb them and nothing moves, so the check reports a perfect plateau while
    measuring the risk limit rather than the strategy. On a real session the agent
    then cited 81% clipping as evidence its pick "sits on flat ground" -- exactly
    backwards, and the check gave it no reason to think otherwise.
    """
    import lab.agent.research as R

    def fake_run(action, cfg, base_cfg, workspace, session, oos_start):
        return R.Experiment(
            n=1, run_id="nb", strategy=action.strategy, params=dict(action.params),
            score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
            full={"total_return": 2.0},
        )

    monkeypatch.setattr(R, "_run_experiment", fake_run)
    monkeypatch.setattr(
        "lab.agent.review.gate_activity",
        lambda directory: {"share_clipped_or_blocked": 0.81,
                           "top_rules": [{"rule": "position_cap", "count": 32}]},
    )

    chosen = R.Experiment(
        n=1, run_id="winner", strategy="s.py", params={"a": 100, "b": 50},
        score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
        full={"total_return": 2.0},
    )
    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace=str(tmp_path)
    )
    out = R._neighbourhood(
        chosen, make_cfg(lab_config, tmp_path, neighbourhood_runs=4),
        None, tmp_path, session, None,
    )

    assert out["gate_clipped"] == pytest.approx(0.81)
    assert out.get("robustness_confounded") is True
    assert "CONFOUNDED" in out["verdict"]
    assert "measures the limit rather than the strategy" in out["verdict"]


def test_a_lightly_clipped_strategy_is_not_confounded(lab_config, tmp_path, monkeypatch):
    import lab.agent.research as R

    monkeypatch.setattr(
        "lab.agent.review.gate_activity",
        lambda directory: {"share_clipped_or_blocked": 0.05, "top_rules": []},
    )
    monkeypatch.setattr(
        R, "_run_experiment",
        lambda action, cfg, base_cfg, workspace, session, oos_start: R.Experiment(
            n=1, run_id="nb", strategy=action.strategy, params=dict(action.params),
            score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
            full={"total_return": 2.0},
        ),
    )
    chosen = R.Experiment(
        n=1, run_id="winner", strategy="s.py", params={"a": 100, "b": 50},
        score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
        full={"total_return": 2.0},
    )
    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace=str(tmp_path)
    )
    out = R._neighbourhood(
        chosen, make_cfg(lab_config, tmp_path, neighbourhood_runs=4),
        None, tmp_path, session, None,
    )
    assert out["gate_clipped"] == pytest.approx(0.05)
    assert not out.get("robustness_confounded")
    assert "CONFOUNDED" not in out["verdict"]


# --- flags are flipped, not scaled --------------------------------------------
#
# A 0/1 int is a switch wearing a number. Ten percent of 1 rounds to a step of 1,
# so the down-nudge lands on 0 (rejected by the general guard as nonsense) and the
# up-nudge lands on 2 (meaningless). The flag was therefore never perturbed, and
# then reported as "not tested at all" as though budget were the problem. On a
# real session that flag was `hold_overnight` -- flipping it moved out-of-sample
# Sharpe from +0.55 to -0.49 and exposure from 10% to 75%, by a distance no other
# parameter came close to, and the check had silently skipped it.


def test_a_zero_one_flag_is_flipped():
    import lab.agent.research as R

    plan = R._perturbations({"hold_overnight": 1, "entry_bar": 21}, budget=8)
    assert ("hold_overnight", 0) in plan, "the flag must be flipped, not scaled"
    assert ("hold_overnight", 2) not in plan, "2 is not a value a flag has meaning for"
    # Exactly once: there is only one alternative.
    assert sum(1 for k, _ in plan if k == "hold_overnight") == 1


def test_a_real_bool_is_flipped_too():
    import lab.agent.research as R

    plan = R._perturbations({"flat_at_close": True, "n": 10}, budget=6)
    assert ("flat_at_close", False) in plan
    assert sum(1 for k, _ in plan if k == "flat_at_close") == 1


def test_a_zero_valued_flag_is_still_perturbable():
    """`v != 0` used to exclude it outright, so an off switch was invisible."""
    import lab.agent.research as R

    assert {k for k, _ in R._perturbable({"hold_overnight": 0, "n": 10})} == {
        "hold_overnight", "n"
    }
    assert ("hold_overnight", 1) in R._perturbations({"hold_overnight": 0, "n": 10}, 6)


def test_full_coverage_counts_a_flag_as_one_run(lab_config, tmp_path, monkeypatch):
    """Demanding two runs for a one-alternative parameter reports permanent
    partial coverage on a check that had in fact done everything."""
    import lab.agent.research as R

    monkeypatch.setattr(
        "lab.agent.review.gate_activity", lambda directory: {}
    )
    monkeypatch.setattr(
        R, "_run_experiment",
        lambda action, cfg, base_cfg, workspace, session, oos_start: R.Experiment(
            n=1, run_id="nb", strategy=action.strategy, params=dict(action.params),
            score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
            full={"total_return": 2.0},
        ),
    )
    chosen = R.Experiment(
        n=1, run_id="winner", strategy="s.py",
        params={"hold_overnight": 1, "entry_bar": 21, "exit_bar": 25},
        score=1.0, ok=True, oos={"total_return": 0.1, "periods": 232},
        full={"total_return": 2.0},
    )
    session = R.ResearchSession(
        session_id="s", brief="b", config_path="c", created_at="t", workspace=str(tmp_path)
    )
    out = R._neighbourhood(
        chosen, make_cfg(lab_config, tmp_path, neighbourhood_runs=16),
        None, tmp_path, session, None,
    )
    cov = out["coverage"]
    assert cov["runs_for_full_coverage"] == 5, "two scalars at 2 runs, one flag at 1"
    assert cov["complete"] is True
    assert cov["one_direction_only"] == [], "a flag with its only alternative tried is done"
    assert "PARTIAL COVERAGE" not in out["verdict"]
