"""The command line, tested as the agent API it also is.

One contract dominates: **with ``--json``, stdout carries exactly one JSON
object and nothing else.** A human reads a table once; an agent loop parses
stdout thousands of times, and a single stray ``print`` anywhere in the call
graph turns a research loop into an unattributable mystery. So the purity tests
here do not scan stdout for a JSON-looking line -- they ``json.loads`` the whole
buffer and assert it holds exactly one newline, which is the only formulation
that fails when something else writes a byte.

Exit codes are the other half of that API: an agent branches on them before it
reads a word of the payload, so 0/1/2/3 are pinned per command rather than
inferred from the constants.

Everything runs against the autouse tmpdir store, so a test needing bars seeds
its own over a short window. The CLI surface is wide and this file gets run on
every edit, so no test here may cost more than a fraction of a second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from typer.testing import CliRunner

from lab import cli
from lab.cli import app
from lab.config import get_settings, reset_settings_cache
from lab.registry import db
from lab.registry.journal import EventJournal
from lab.timeutil import utcnow

#: ~30 sessions: enough for a strategy to allocate, fill and be marked, and
#: short enough that a backtest through the CLI costs milliseconds.
WINDOW = ("--start", "2024-02-01", "--end", "2024-03-15")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def seeded(seeded_store_factory: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    return seeded_store_factory(tickers=["AAA", "BBB"], days=90)


@pytest.fixture()
def strategy_path() -> Path:
    return get_settings().paths.strategies / "buy_and_hold.py"


@pytest.fixture()
def config_file(tmp_path: Path, strategy_path: Path) -> Path:
    """A backtest config whose limits are loose enough not to clip.

    The default gate caps a position at 5%, which would flatten every parameter
    difference into the same clipped weight and let an override test pass for
    entirely the wrong reason.
    """
    path = tmp_path / "bh.yaml"
    path.write_text(
        "strategy: {strategy}\n"
        "tickers: [AAA, BBB]\n"
        "timeframe: 1d\n"
        "cash: 100000\n"
        "params:\n"
        "  cash_buffer: 0.02\n"
        "limits:\n"
        "  max_position_pct: 0.6\n"
        "  max_positions: 5\n"
        "  min_order_notional: 1\n"
        "fills:\n"
        "  mode: next_open\n"
        "  slippage_bps: 0\n".format(strategy=json.dumps(str(strategy_path))),
        encoding="utf-8",
    )
    return path


def payload_of(result: Any) -> dict[str, Any]:
    """Parse the *entire* stdout buffer. Anything extra is a contract breach."""
    assert result.stdout, f"nothing on stdout; stderr was:\n{result.stderr}"
    assert result.stdout.count("\n") == 1, f"more than one line on stdout: {result.stdout!r}"
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict), f"expected one object, got {type(parsed).__name__}"
    return parsed


def backtest(runner: CliRunner, config: Path, *extra: str) -> dict[str, Any]:
    result = runner.invoke(app, ["backtest", "--config", str(config), *WINDOW, *extra, "--json"])
    assert result.exit_code == 0, result.stderr
    return payload_of(result)


# --- the stdout contract ------------------------------------------------------


JSON_COMMANDS: dict[str, Sequence[str]] = {
    "adapters": ("adapters", "--json"),
    "backtest": ("backtest", "--config", "{config}", *WINDOW, "--json"),
    "data version": ("data", "version", "--json"),
    "runs list": ("runs", "list", "--json"),
    "strategies": ("strategies", "--json"),
}


@pytest.mark.parametrize("name", sorted(JSON_COMMANDS))
def test_json_stdout_is_exactly_one_object(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any], name: str
) -> None:
    args = [a.format(config=config_file) for a in JSON_COMMANDS[name]]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.stderr
    assert payload_of(result)["ok"] is True
    assert "Traceback" not in result.stderr


def test_root_level_json_flag_works_too(runner: CliRunner) -> None:
    # `lab --json adapters` and `lab adapters --json` are one request; an agent
    # should not have to remember which side of the verb the flag lives on.
    before = runner.invoke(app, ["--json", "adapters"])
    after = runner.invoke(app, ["adapters", "--json"])

    assert before.exit_code == after.exit_code == 0
    assert payload_of(before) == payload_of(after)


def test_without_json_the_human_output_is_not_json(runner: CliRunner) -> None:
    result = runner.invoke(app, ["adapters"])

    assert result.exit_code == 0
    assert result.stdout.strip()
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_json_mode_pushes_a_stray_print_onto_stderr(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Purity has to be a property of the CLI, not a promise from its callees.

    ``--json`` hands stdout to the payload alone for the duration of the work,
    so a progress bar, a deprecation notice or somebody's leftover debug print
    in a module the CLI merely calls cannot corrupt the parse. Simulate one.
    """
    import lab.adapters.base as base

    real = base.list_adapters

    def noisy() -> Any:
        print("chatty module writing to stdout")
        return real()

    monkeypatch.setattr(base, "list_adapters", noisy)

    result = runner.invoke(app, ["adapters", "--json"])

    assert result.exit_code == 0
    assert payload_of(result)["ok"] is True
    assert "chatty module" in result.stderr


def test_error_payloads_are_json_too(runner: CliRunner) -> None:
    # A failure an agent cannot parse is as useless as a corrupted success.
    result = runner.invoke(app, ["runs", "show", "ghost", "--json"])

    assert result.exit_code == 1
    parsed = payload_of(result)
    assert parsed["ok"] is False
    assert "ghost" in parsed["error"]
    assert parsed["run_id"] == "ghost"


def test_backtest_payload_carries_the_provenance_an_agent_needs(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    parsed = backtest(runner, config_file)

    assert parsed["run_id"]
    assert parsed["data_version"] == seeded["data_version"]
    assert parsed["attempt"] == 1
    assert parsed["n_decisions"] > 0
    assert parsed["metrics"]["final_equity"] > 0
    assert parsed["report"] is None  # not rendered unless asked for

    listed = payload_of(runner.invoke(app, ["runs", "list", "--json"]))
    assert [r["run_id"] for r in listed["runs"]] == [parsed["run_id"]]
    assert listed["runs"][0]["attempt"] == 1


def test_repeated_backtests_expose_the_attempt_count(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    # The overfitting tell has to survive the trip through the CLI.
    assert [backtest(runner, config_file)["attempt"] for _ in range(3)] == [1, 2, 3]


def test_runs_compare_json_has_one_row_per_run(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    first = backtest(runner, config_file)
    second = backtest(runner, config_file, "-p", "cash_buffer=0.4")

    parsed = payload_of(
        runner.invoke(app, ["runs", "compare", first["run_id"], second["run_id"], "--json"])
    )
    assert [r["run_id"] for r in parsed["rows"]] == [first["run_id"], second["run_id"]]
    assert "param.cash_buffer" in parsed["columns"]


# --- exit codes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "args, code, why",
    [
        (["adapters", "--json"], 0, "a command that works"),
        (["runs", "show", "ghost", "--json"], 1, "a well-formed request that fails"),
        (["pull", "--source", "nope", "--json"], 2, "an argument that names nothing"),
        (["pull", "--source", "synthetic", "--kind", "sideways"], 2, "an out-of-range choice"),
        (["report"], 2, "a missing required argument"),
        (["backtest", "-p", "lookback", "--json"], 2, "--param without a value"),
        (["runs", "delete", "some-run", "--json"], 2, "a destructive call without --yes"),
    ],
)
def test_exit_codes(runner: CliRunner, args: list[str], code: int, why: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == code, f"{why}: {result.stdout}{result.stderr}"


def test_usage_errors_keep_stdout_empty(runner: CliRunner) -> None:
    # click renders its usage box with rich; none of it may reach stdout.
    result = runner.invoke(app, ["report"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Missing argument" in result.stderr


# --- the kill switch ----------------------------------------------------------


def test_kill_engage_and_release_round_trip(runner: CliRunner) -> None:
    kill_file = Path(get_settings().kill_file)
    assert not kill_file.exists()

    engaged = payload_of(runner.invoke(app, ["kill", "--reason", "smoke test", "--json"]))
    assert engaged["ok"] is True
    assert engaged["engaged"] is True
    assert engaged["changed"] is True
    assert engaged["released"] is False
    assert Path(engaged["kill_file"]) == kill_file
    assert kill_file.exists()

    released = payload_of(runner.invoke(app, ["kill", "--release", "--json"]))
    assert released["engaged"] is False
    assert released["changed"] is True
    assert released["released"] is True
    assert not kill_file.exists()

    # Close the loop through the status command rather than the file system.
    assert payload_of(runner.invoke(app, ["status", "--json"]))["kill_switch"]["engaged"] is False


def test_kill_is_safe_to_run_twice(runner: CliRunner) -> None:
    """Idempotent in both directions.

    A panic stop is run by whoever gets there first, sometimes twice, often by a
    script that cannot see the outcome of the previous attempt. The second run
    has to be a quiet no-op, not an error.
    """
    first = payload_of(runner.invoke(app, ["kill", "--json"]))
    second = runner.invoke(app, ["kill", "--json"])

    assert second.exit_code == 0
    assert first["changed"] is True
    twice = payload_of(second)
    assert twice["changed"] is False
    assert twice["engaged"] is True
    assert Path(get_settings().kill_file).exists()

    payload_of(runner.invoke(app, ["kill", "--release", "--json"]))
    again = runner.invoke(app, ["kill", "--release", "--json"])

    assert again.exit_code == 0
    parsed = payload_of(again)
    assert parsed["engaged"] is False
    assert parsed["changed"] is False
    assert not Path(get_settings().kill_file).exists()


def test_engaged_kill_switch_refuses_with_exit_3(runner: CliRunner, strategy_path: Path) -> None:
    payload_of(runner.invoke(app, ["kill", "--reason", "halt", "--json"]))

    result = runner.invoke(app, ["paper", "start", str(strategy_path), "--json"])

    assert result.exit_code == 3
    parsed = payload_of(result)
    assert parsed["ok"] is False
    assert parsed["kill_switch"] is True
    assert "refusing to start a paper runner" in parsed["error"]
    assert "lab kill --release" in parsed["error"]


def test_release_cannot_clear_the_env_flag_and_says_so(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting "released" while the switch is still on would be the single
    most dangerous lie this CLI could tell, so it fails loudly instead."""
    monkeypatch.setenv("LAB_KILL_SWITCH", "1")
    reset_settings_cache()

    result = runner.invoke(app, ["kill", "--release", "--json"])

    assert result.exit_code == 1
    parsed = payload_of(result)
    assert parsed["engaged"] is True
    assert parsed["env_flag"] is True
    assert "LAB_KILL_SWITCH" in result.stderr


def test_kill_flips_are_journaled(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """An undocumented stop at 09:31 is indistinguishable from a crash at 09:31.

    This drives the built-in sentinel shim, which is the path taken when
    ``lab.live.killswitch`` is absent; the real module journals its own flips.
    """
    monkeypatch.setattr(cli, "_kill_switch", lambda: cli._SentinelKill)

    result = runner.invoke(app, ["kill", "--reason", "audit", "--json"])

    assert result.exit_code == 0
    assert payload_of(result)["engaged"] is True
    breakers = EventJournal().tail(0, kinds=["breaker"])
    assert [e["message"] for e in breakers] == ["kill switch engaged"]
    assert breakers[0]["payload"] == {"reason": "audit", "engaged": True}
    assert "could not journal" not in result.stderr


# --- parameter overrides ------------------------------------------------------


def test_param_overrides_keep_their_yaml_types(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    # `rebalance_days=5` must reach the strategy as an int, not as "5"; that is
    # the entire reason the value is parsed as a YAML scalar.
    params = backtest(
        runner,
        config_file,
        "-p", "rebalance_days=5",
        "-p", "cash_buffer=0.25",
        "--param", "enabled=true",
        "--param", "label=alpha",
    )["params"]

    assert params["rebalance_days"] == 5 and isinstance(params["rebalance_days"], int)
    assert params["cash_buffer"] == 0.25
    assert params["enabled"] is True
    assert params["label"] == "alpha"


def test_param_overrides_merge_into_the_config_rather_than_replacing_it(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    parsed = backtest(runner, config_file, "-p", "rebalance_days=7")
    # -p tunes one knob; the config's other parameters survive untouched.
    assert parsed["params"] == {"cash_buffer": 0.02, "rebalance_days": 7}


def test_param_overrides_actually_reach_the_strategy(
    runner: CliRunner, config_file: Path, seeded: dict[str, Any]
) -> None:
    """A payload echoing a param proves plumbing, not effect.

    ``buy_and_hold`` sizes each position at ``(1 - cash_buffer) / n``, so a
    different buffer must produce a different equity curve. If it does not, the
    override stopped somewhere between the flag and ``ctx.params``.
    """
    base = backtest(runner, config_file)
    tweaked = backtest(runner, config_file, "-p", "cash_buffer=0.5")

    assert base["params"]["cash_buffer"] == 0.02
    assert tweaked["params"]["cash_buffer"] == 0.5
    assert base["metrics"]["final_equity"] != tweaked["metrics"]["final_equity"]
    assert base["metrics"]["exposure"] > tweaked["metrics"]["exposure"]


# --- clean failures -----------------------------------------------------------


def test_a_missing_strategy_file_is_an_error_not_a_traceback(runner: CliRunner) -> None:
    result = runner.invoke(app, ["backtest", "strategies/not_a_strategy.py", "--json"])

    assert result.exit_code == 1
    parsed = payload_of(result)
    assert parsed["ok"] is False
    assert "not_a_strategy.py" in parsed["error"]
    # A traceback is the one output an agent loop can do nothing with.
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_an_unavailable_adapter_explains_itself(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Credentials may well exist in the developer's environment; take them away
    # so the unavailable branch is the one under test.
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    reset_settings_cache()

    result = runner.invoke(app, ["pull", "--source", "alpaca", "--tickers", "AAA", "--json"])

    assert result.exit_code == 1
    parsed = payload_of(result)
    assert parsed["ok"] is False
    assert parsed["available"] is False
    assert parsed["source"] == "alpaca"
    assert "unavailable" in parsed["error"]
    assert "Traceback" not in result.stderr


def test_an_unknown_adapter_is_bad_usage_and_names_the_real_ones(runner: CliRunner) -> None:
    result = runner.invoke(app, ["pull", "--source", "nope", "--json"])

    assert result.exit_code == 2
    parsed = payload_of(result)
    assert "nope" in parsed["error"]
    assert "synthetic" in parsed["error"]


def test_a_ticker_spec_that_looks_like_a_path_must_exist(runner: CliRunner) -> None:
    # Guessing by shape would backtest a universe of one ticker called
    # "CFG/NOPE.TXT"; a path that does not resolve is bad usage instead.
    result = runner.invoke(
        app, ["pull", "--source", "synthetic", "--tickers", "cfg/nope.txt", "--json"]
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "universe" in result.stderr


def test_pull_from_the_synthetic_adapter_needs_no_credentials(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        ["pull", "--source", "synthetic", "--tickers", "AAA,BBB",
         "--since", "2024-02-01", "--until", "2024-03-15", "--json"],
    )

    assert result.exit_code == 0, result.stderr
    parsed = payload_of(result)
    assert parsed["tickers"] == ["AAA", "BBB"]
    assert parsed["rows_written"] > 0
    assert parsed["data_version"]


# --- control actions ----------------------------------------------------------


def test_paper_stop_pauses_and_journals(runner: CliRunner) -> None:
    con = db.connect(db.journal_path())
    with db.transaction(con):
        con.execute(
            "INSERT INTO live_strategies (strategy, kind, status, pid, host, paused, updated_at) "
            "VALUES ('momo','paper','running',4242,'testhost',0,?)",
            (utcnow().isoformat(),),
        )

    stopped = runner.invoke(app, ["paper", "stop", "momo", "--json"])
    parsed = payload_of(stopped)

    assert parsed["paused"] is True
    assert parsed["already_paused"] is False
    assert parsed["pid"] == 4242
    assert con.execute("SELECT paused FROM live_strategies WHERE strategy='momo'").fetchone()[0] == 1

    # Pausing a strategy is a control action from outside its process; it has to
    # leave a trace, or a stop is indistinguishable from a crash after the fact.
    logs = EventJournal().tail(0, kinds=["log"])
    assert [e["payload"].get("control") for e in logs] == ["stop"]
    assert logs[0]["strategy"] == "momo"
    assert "could not journal" not in stopped.stderr

    again = payload_of(runner.invoke(app, ["paper", "stop", "momo", "--json"]))
    assert again["already_paused"] is True


def test_paper_stop_of_an_unknown_strategy_names_the_known_ones(runner: CliRunner) -> None:
    con = db.connect(db.journal_path())
    with db.transaction(con):
        con.execute("INSERT INTO live_strategies (strategy) VALUES ('momo')")

    result = runner.invoke(app, ["paper", "stop", "ghost", "--json"])

    assert result.exit_code == 1
    parsed = payload_of(result)
    assert "ghost" in parsed["error"]
    assert "momo" in parsed["error"]


def test_status_reports_the_whole_surface(runner: CliRunner, seeded: dict[str, Any]) -> None:
    parsed = payload_of(runner.invoke(app, ["status", "--json"]))

    assert {"kill_switch", "paths", "live", "heartbeats", "events", "runs", "adapters"} <= set(
        parsed
    )
    assert parsed["kill_switch"]["engaged"] is False
    assert any(a["name"] == "synthetic" for a in parsed["adapters"])



def test_agent_author_wires_the_loop_config_correctly(
    runner: CliRunner, tmp_path: Path, monkeypatch, seeded_store
) -> None:
    """The CLI once passed the *grid path* into ``grid_or_freeform`` -- a mode
    flag that only accepts "grid"/"freeform" -- and never passed a backtest
    config at all, so ``lab agent author`` could not run under any arguments.
    Both are cheap to break again, so pin the translation itself."""
    import yaml

    from lab.agent import author_loop

    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "strategy": "strategies/momo.py",
                "tickers": list(seeded_store["tickers"]),
                "timeframe": "1d",
                "cash": 100000,
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run(cfg: Any, **_kw: Any) -> dict[str, Any]:
        captured["cfg"] = cfg
        return {"lineage": [], "best": None, "spend_usd": 0.0, "billing": "api"}

    monkeypatch.setattr(author_loop, "run_author_loop", fake_run)

    result = runner.invoke(
        app,
        [
            "agent", "author",
            "--seed", "strategies/momo.py",
            "--config", str(cfg_path),
            "--objective", "raise oos sharpe",
            "--oos-split", "3:1",
            "-n", "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output

    cfg = captured["cfg"]
    assert cfg.grid_or_freeform == "grid", "default mode is params-only"
    assert Path(cfg.config) == cfg_path, "the backtest config must reach the loop"
    assert cfg.objective == "raise oos sharpe"
    assert cfg.oos_split == "3:1"
    assert cfg.iterations == 2

    result = runner.invoke(
        app,
        ["agent", "author", "--seed", "strategies/momo.py", "--config", str(cfg_path),
         "--freeform", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert captured["cfg"].grid_or_freeform == "freeform"


def test_agent_author_requires_a_backtest_config(runner: CliRunner) -> None:
    """Without one the loop raises deep inside; fail at the boundary instead."""
    result = runner.invoke(app, ["agent", "author", "--seed", "strategies/momo.py"])
    assert result.exit_code == 2
    assert "--config" in result.output
