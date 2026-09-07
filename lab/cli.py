"""The command line, which is also the agent API (design doc §9).

One contract dominates every decision in this file: **with ``--json``, stdout
carries exactly one JSON object and nothing else.** Logs, progress bars,
warnings and every rich table go to stderr. A human reads the pretty output
once; an agent loop parses stdout thousands of times, and a single stray print
anywhere in the call graph turns a research loop into a mystery. So the JSON
path does not merely *avoid* printing — it redirects ``sys.stdout`` to stderr
for the duration of the work and writes the payload to the real stdout
afterwards, which makes purity a property of the CLI rather than a promise
extracted from every module it calls.

Two other rules shape the layout. Heavy modules are imported inside command
bodies, so ``lab --help`` stays instant and a module that is broken (or not
built yet) cannot take unrelated commands down with it. And exit codes are
load-bearing: 0 ok, 1 error, 2 bad usage, 3 gate/kill refusal — an agent
branches on them before it reads a word of the payload.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, NoReturn, Optional, Sequence

import click
import typer

OK, ERROR, USAGE, REFUSED = 0, 1, 2, 3

app = typer.Typer(
    name="lab",
    help=(
        "Strategy lab: pull data, backtest, sweep, paper-trade, and inspect runs.\n\n"
        "Every command takes --json and then emits exactly one JSON object on "
        "stdout (logs go to stderr), so an agent loop can parse it. "
        "Exit codes: 0 ok, 1 error, 2 bad usage, 3 gate/kill refusal."
    ),
    no_args_is_help=True,
    add_completion=False,
    # Rich tracebacks print a wall of frames over the caller's terminal and tell
    # an agent nothing it can act on; every command reports its own failures.
    pretty_exceptions_enable=False,
)

runs_app = typer.Typer(help="Query the run registry.", no_args_is_help=True)
paper_app = typer.Typer(help="Paper-trading runner control.", no_args_is_help=True)
data_app = typer.Typer(help="Inspect the point-in-time store.", no_args_is_help=True)
govgreed_app = typer.Typer(help="GovGreed alt-data source.", no_args_is_help=True)
agent_app = typer.Typer(help="Agent-authored strategy loop.", no_args_is_help=True)
strategy_app = typer.Typer(help="The strategy library.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")
app.add_typer(paper_app, name="paper")
app.add_typer(data_app, name="data")
app.add_typer(govgreed_app, name="govgreed")
app.add_typer(agent_app, name="agent")
app.add_typer(strategy_app, name="strategy")


# --- output discipline --------------------------------------------------------

#: Set by the root callback so `lab --json backtest ...` works as well as
#: `lab backtest --json`; commands OR it with their own flag.
_STATE: dict[str, Any] = {"json": False, "verbose": False}

#: The real stdout while it is redirected. A stack, not a slot, so nesting is
#: harmless.
_REAL_STDOUT: list[Any] = []


def _want_json(local: bool) -> bool:
    return bool(local or _STATE.get("json"))


@contextmanager
def _json_stdout(active: bool) -> Iterator[None]:
    """Under ``--json``, hand stdout to the payload alone.

    Anything printed by a module the CLI merely calls -- a progress bar, a
    deprecation notice, somebody's leftover debug print -- lands on stderr for
    the duration. Cheap insurance against a corrupted parse that would be
    almost impossible to attribute later.
    """
    if not active:
        yield
        return
    import contextlib

    _REAL_STDOUT.append(sys.stdout)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        _REAL_STDOUT.pop()


def _clean(value: Any) -> Any:
    """Coerce anything into JSON-safe primitives.

    NaN and infinity become ``null``: they are legal in Python's json dialect
    but not in RFC 8259, and a payload that only *this* parser can read is not
    an API. Numpy scalars, dataclasses, Paths and enums all show up in metrics
    and configs, so they are handled here rather than at thirty call sites.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _clean(value.value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _clean(to_dict())
        except Exception:  # a to_dict that needs arguments is not ours to call
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _clean(dataclasses.asdict(value))
    item = getattr(value, "item", None)  # numpy / pandas scalars
    if callable(item):
        try:
            return _clean(item())
        except Exception:
            pass
    return str(value)


def _print_json(payload: Any) -> None:
    stream = _REAL_STDOUT[-1] if _REAL_STDOUT else sys.stdout
    stream.write(json.dumps(_clean(payload), allow_nan=False, ensure_ascii=False) + "\n")
    stream.flush()


def _console(*, err: bool = False):
    from rich.console import Console

    # Built per call: rich resolves `file` lazily, but the width probe does not,
    # and the CLI is exercised under captured streams as well as a terminal.
    return Console(stderr=err, highlight=False)


def _note(message: str) -> None:
    """Human-facing chatter. Always stderr, json mode or not."""
    _console(err=True).print(message)


def _fail(message: str, *, json_out: bool, code: int = ERROR, **fields: Any) -> NoReturn:
    if json_out:
        _print_json({"ok": False, "error": message, **fields})
    else:
        _console(err=True).print(f"[bold red]error[/bold red] {message}")
    raise typer.Exit(code)


def _passthrough() -> tuple[type[BaseException], ...]:
    """Control-flow exceptions the guard must not touch.

    typer vendors its own copy of click, so ``typer.Exit`` is *not*
    ``click.exceptions.Exit``; catching only the latter turns every clean exit
    (including the ones ``_fail`` raises) into a second error payload. Collect
    both families rather than betting on which one a given typer ships.
    """
    found: set[type[BaseException]] = {
        typer.Exit, typer.Abort, click.exceptions.Exit, click.exceptions.Abort,
        click.ClickException,
    }
    found.update(b for b in typer.BadParameter.__mro__ if b.__name__ == "ClickException")
    return tuple(found)


_PASSTHROUGH = _passthrough()


@contextmanager
def _guard(json_out: bool) -> Iterator[None]:
    """Turn an unexpected exception into the documented failure shape.

    A traceback is the one output an agent loop cannot do anything with, and a
    half-written payload is worse than none, so every command body runs in
    here and every payload is emitted at the end.
    """
    try:
        yield
    except _PASSTHROUGH:
        raise
    except KeyboardInterrupt:
        _fail("interrupted", json_out=json_out, code=ERROR)
    except Exception as exc:
        if _STATE.get("verbose"):
            import traceback

            traceback.print_exc(file=sys.stderr)
        _fail(f"{type(exc).__name__}: {exc}", json_out=json_out, code=ERROR)


def _module(path: str, *, what: str, json_out: bool):
    """Import a sibling module, or explain that it is not built yet.

    The modules below the CLI land in parallel; a missing one is a state to
    report, not a stack trace, and it must not stop the commands that do work.
    """
    import importlib

    try:
        return importlib.import_module(path)
    except ImportError as exc:
        _fail(
            f"{what} is not available: {exc}",
            json_out=json_out,
            code=ERROR,
            missing=path,
        )


def _settings():
    from lab.config import get_settings

    return get_settings()


# --- shared parsing -----------------------------------------------------------


def _parse_ts(raw: str | None, *, what: str) -> datetime | None:
    if raw is None or str(raw).strip() == "":
        return None
    import pandas as pd

    from lab.timeutil import to_utc

    try:
        return to_utc(pd.Timestamp(raw))
    except Exception as exc:
        raise typer.BadParameter(f"cannot parse {what} {raw!r}: {exc}")


def _resolve_tickers(spec: str | None, *, required: bool = True) -> list[str]:
    """``--tickers`` is either a universe file or a comma list, by existence.

    Guessing by shape would silently turn a mistyped path into a ticker called
    ``CFG/UNIVERSE.TXT`` and backtest an empty universe, so a value that looks
    like a path but does not exist is an error instead.
    """
    from lab.backtest.runner import load_universe

    if spec is None or not str(spec).strip():
        default = _settings().paths.cfg / "universe.txt"
        if default.exists():
            return load_universe(default)
        if required:
            raise typer.BadParameter("no --tickers given and no cfg/universe.txt to fall back on")
        return []
    text = str(spec).strip()
    path = Path(text)
    if path.exists():
        return load_universe(path)
    if any(sep in text for sep in ("/", "\\")) or text.lower().endswith((".txt", ".csv", ".yaml")):
        raise typer.BadParameter(f"no universe file at {text}")
    out = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
    if not out:
        raise typer.BadParameter(f"no tickers in {text!r}")
    return out


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    """``-p lookback=126`` — values are YAML scalars so types survive.

    This is what lets an agent sweep one knob without writing a config file,
    and YAML rather than ``str`` because ``top_n=4`` must reach the strategy as
    an int, not as ``"4"``.
    """
    import yaml

    out: dict[str, Any] = {}
    for raw in items or []:
        key, sep, value = str(raw).partition("=")
        key = key.strip()
        if not sep or not key:
            raise typer.BadParameter(f"--param expects key=value, got {raw!r}")
        try:
            out[key] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"--param {key}: {value!r} is not a YAML scalar ({exc})")
    return out


def _build_config(
    *,
    strategy: str | None,
    config: Path | None,
    params: Mapping[str, Any],
    tickers: str | None = None,
    start: str | None = None,
    end: str | None = None,
    cash: float | None = None,
    timeframe: str | None = None,
    notes: str | None = None,
    origin: str | None = None,
):
    from lab.backtest.runner import BacktestConfig

    overrides: dict[str, Any] = {
        "strategy": strategy,
        "start": _parse_ts(start, what="--start"),
        "end": _parse_ts(end, what="--end"),
        "cash": cash,
        "timeframe": timeframe,
        "notes": notes,
        "origin": origin,
    }
    if tickers:
        overrides["tickers"] = _resolve_tickers(tickers)
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if config is not None:
        cfg = BacktestConfig.from_yaml(config, overrides)
    else:
        if not strategy:
            raise typer.BadParameter("give a strategy path, or --config, or both")
        base = {"tickers": overrides.pop("tickers", None) or _resolve_tickers(None)}
        cfg = BacktestConfig.from_mapping(base | overrides)
    # Merged rather than replaced: -p tunes one knob, it does not discard the
    # config's other parameters.
    cfg.params = {**cfg.params, **dict(params)}
    return cfg


def _kill_switch():
    """``lab.live.killswitch`` when it exists, an identical shim when it does not.

    The shim drives the same sentinel file as the real module, so engaging here
    and releasing there is one switch and not two.
    """
    try:
        from lab.live import killswitch  # type: ignore[attr-defined]

        if all(hasattr(killswitch, n) for n in ("engaged", "engage", "release")):
            return killswitch
    except ImportError:
        pass
    return _SentinelKill


class _SentinelKill:
    @staticmethod
    def engaged() -> tuple[bool, str | None]:
        return _settings().kill_switch_engaged()

    @staticmethod
    def engage(reason: str = "") -> Path:
        from lab.timeutil import utcnow

        path = _settings().kill_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"at": utcnow().isoformat(), "reason": reason}) + "\n", encoding="utf-8"
        )
        return path

    @staticmethod
    def release() -> bool:
        """Returns whether trading is now *permitted* — same contract as the
        real module, which cannot unset an env flag it does not own."""
        path = _settings().kill_file
        path.unlink(missing_ok=True)
        return not _settings().kill_switch_engaged()[0]


def _refuse_if_killed(json_out: bool, *, action: str) -> None:
    engaged, reason = _kill_switch().engaged()
    if engaged:
        _fail(
            f"kill switch engaged ({reason}); refusing to {action}. "
            "Release it with `lab kill --release` once you know why it is on.",
            json_out=json_out,
            code=REFUSED,
            kill_switch=True,
            reason=reason,
        )


def _table(title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    from rich.table import Table

    table = Table(title=title, title_justify="left", header_style="bold")
    for col in columns:
        table.add_column(str(col), overflow="fold")
    for row in rows:
        table.add_row(*["" if c is None else str(c) for c in row])
    _console().print(table)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
    return "" if value is None else str(value)


# --- root ---------------------------------------------------------------------


@app.callback()
def _root(
    json_out: bool = typer.Option(
        False, "--json", "-j", help="Emit one JSON object on stdout; everything else to stderr."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging on stderr."),
) -> None:
    import logging

    _STATE["json"] = json_out
    _STATE["verbose"] = verbose
    # force=True re-points handlers at the *current* sys.stderr, which matters
    # under a test runner that swaps the streams per invocation.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


# --- data in ------------------------------------------------------------------


@app.command()
def pull(
    source: str = typer.Option(..., "--source", "-s", help="Adapter: synthetic|yfinance|alpaca|govgreed."),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t", help="Comma list or universe file."),
    tf: str = typer.Option("1d", "--tf", "--timeframe", help="Bar timeframe."),
    since: Optional[str] = typer.Option(None, "--since", help="Start (default 2018-01-01)."),
    until: Optional[str] = typer.Option(None, "--until", help="End (default now)."),
    kind: str = typer.Option("auto", "--kind", help="bars | signals | auto."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Fetch from an adapter and write it into the point-in-time store."""
    json_out = _want_json(json_out)
    if kind not in {"auto", "bars", "signals"}:
        raise typer.BadParameter("--kind must be bars, signals or auto")
    start = _parse_ts(since, what="--since") or _parse_ts("2018-01-01", what="--since")
    end = _parse_ts(until, what="--until")

    with _guard(json_out), _json_stdout(json_out):
        from lab.adapters.base import AdapterError, get_adapter
        from lab.store import parquet_io as pio
        from lab.store.schema import BARS, SIGNALS
        from lab.timeutil import utcnow

        end = end or utcnow()
        try:
            adapter = get_adapter(source)
        except ValueError as exc:
            _fail(str(exc), json_out=json_out, code=USAGE)
        ok, reason = adapter.available()
        if not ok:
            _fail(
                f"adapter {adapter.name!r} is unavailable: {reason}",
                json_out=json_out,
                available=False,
                source=adapter.name,
            )

        provides = set(adapter.provides)
        want = kind if kind != "auto" else ("bars" if "bars" in provides else "signals")
        if want not in provides:
            _fail(
                f"adapter {adapter.name!r} provides {sorted(provides) or 'nothing'}, not {want!r}",
                json_out=json_out,
                code=USAGE,
            )

        src = str(getattr(adapter, "source", "") or adapter.name)
        rows: dict[str, int] = {}
        universe: list[str] = []
        try:
            if want == "bars":
                universe = _resolve_tickers(tickers)
                records = []
                for bar in adapter.fetch_bars(universe, tf, start, end):
                    records.append(bar)
                    rows[bar.ticker] = rows.get(bar.ticker, 0) + 1
                written = pio.write_bars(records)
                table = BARS
                version = pio.data_version(BARS, tickers=universe, source=src)
            else:
                universe = _resolve_tickers(tickers, required=False)
                query: dict[str, Any] = {"tickers": universe} if universe else {}
                records = []
                for ev in adapter.fetch_signals(start=start, end=end, **query):
                    records.append(ev)
                    rows[ev.ticker or "-"] = rows.get(ev.ticker or "-", 0) + 1
                written = pio.write_events(records)
                table = SIGNALS
                version = pio.data_version(SIGNALS, source=src)
        except AdapterError as exc:
            _fail(f"{adapter.name}: {exc}", json_out=json_out, source=adapter.name)

        payload = {
            "ok": True,
            "source": src,
            "adapter": adapter.name,
            "kind": want,
            "table": table,
            "timeframe": tf if want == "bars" else None,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tickers": sorted(rows),
            "rows_fetched": dict(sorted(rows.items())),
            "rows_written": int(written),
            "data_version": version,
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        f"pull {src} · {want} · {len(rows)} tickers",
        ["ticker", "rows"],
        sorted(payload["rows_fetched"].items()),
    )
    _console().print(
        f"wrote [bold]{written}[/bold] rows to [cyan]{table}[/cyan] · "
        f"data_version [bold]{version}[/bold]"
    )


# --- research -----------------------------------------------------------------


@app.command()
def backtest(
    strategy: Optional[str] = typer.Argument(None, help="Path to the strategy module."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML backtest config."),
    param: Optional[list[str]] = typer.Option(
        None, "--param", "-p", help="key=value parameter override (YAML scalar). Repeatable."
    ),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    cash: Optional[float] = typer.Option(None, "--cash"),
    timeframe: Optional[str] = typer.Option(None, "--tf", "--timeframe"),
    notes: Optional[str] = typer.Option(None, "--notes"),
    origin: Optional[str] = typer.Option(None, "--origin", help="human | agent-loop."),
    no_register: bool = typer.Option(False, "--no-register", help="Do not record the run."),
    progress: bool = typer.Option(False, "--progress", help="Progress bar on stderr."),
    report: bool = typer.Option(False, "--report", help="Render the HTML report afterwards."),
    open_report: bool = typer.Option(False, "--open", help="Render and open the report."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run one backtest. With --json this emits BacktestResult.to_json()."""
    json_out = _want_json(json_out)
    overrides = _parse_params(param)

    with _guard(json_out), _json_stdout(json_out):
        from lab.backtest.runner import run_backtest

        cfg = _build_config(
            strategy=strategy,
            config=config,
            params=overrides,
            tickers=tickers,
            start=start,
            end=end,
            cash=cash,
            timeframe=timeframe,
            notes=notes,
            origin=origin,
        )
        result = run_backtest(
            cfg,
            register=not no_register,
            # The runner's progress writer ends with a newline on stdout; under
            # --json that newline is the one byte that must not exist.
            progress=progress and not json_out,
        )
        payload = result.to_json()
        payload["ok"] = True
        payload["report"] = None
        if report or open_report:
            payload["report"] = _try_report(result, open_browser=open_report)

    if json_out:
        _print_json(payload)
        return
    _render_metrics(result)
    if payload["report"]:
        _console().print(f"report [cyan]{payload['report']}[/cyan]")
    else:
        _console().print(f"report with [bold]lab report {result.run_id}[/bold]")


def _try_report(result: Any, *, open_browser: bool) -> str | None:
    """Best-effort HTML render: a missing reporter must not fail a good run."""
    import importlib

    try:
        report_mod = importlib.import_module("lab.backtest.report")
    except ImportError as exc:
        _note(f"[yellow]warning[/yellow] report renderer unavailable: {exc}")
        return None
    try:
        return str(report_mod.render_report(result, open_browser=open_browser))
    except Exception as exc:
        _note(f"[yellow]warning[/yellow] report render failed: {type(exc).__name__}: {exc}")
        return None


#: The numbers worth a terminal row. The full set stays in metrics.json.
_HEADLINE = (
    "start", "end", "days", "total_return", "cagr", "sharpe", "sortino", "calmar",
    "volatility", "max_drawdown", "max_drawdown_duration_days", "exposure", "turnover",
    "trades", "hit_rate", "profit_factor", "avg_trade_pnl", "final_equity",
)


def _render_metrics(result: Any) -> None:
    metrics = result.metrics
    _table(
        f"{result.config.get('strategy')} · {result.run_id}",
        ["metric", "value"],
        [(k, _fmt(metrics.get(k))) for k in _HEADLINE if k in metrics],
    )
    if metrics.get("optimistic_fills"):
        _note("[bold yellow]optimistic fills[/bold yellow] — same-bar-close model; treat as an upper bound")
    if metrics.get("contaminated"):
        _note("[bold yellow]contaminated[/bold yellow] — LLM strategy over historical data; not evidence")
    for warning in result.warnings:
        _note(f"[yellow]warning[/yellow] {warning}")
    _console().print(f"artifacts [cyan]{result.artifact_dir}[/cyan]")


@app.command()
def sweep(
    strategy: Optional[str] = typer.Argument(None, help="Strategy path; overrides the base config."),
    grid: Path = typer.Option(..., "--grid", "-g", help="Grid YAML."),
    base: Optional[Path] = typer.Option(None, "--config", "-c", help="Base backtest config."),
    walk_forward: Optional[str] = typer.Option(None, "--walk-forward", "-w", help='e.g. "4:1".'),
    metric: Optional[str] = typer.Option(None, "--metric", "-m"),
    max_runs: Optional[int] = typer.Option(None, "--max-runs"),
    workers: int = typer.Option(1, "--workers"),
    fast: bool = typer.Option(False, "--fast", help="Coarse vectorized screen first."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Expand a parameter grid, score out-of-sample, rank the survivors."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        sweep_mod = _module("lab.backtest.sweep", what="the sweep runner", json_out=json_out)
        cfg = sweep_mod.SweepConfig.from_yaml(grid, base=base)
        if strategy:
            cfg.base.strategy = strategy
        if walk_forward:
            cfg.walk_forward = walk_forward
        if metric:
            cfg.metric = metric
        if max_runs is not None:
            cfg.max_runs = max_runs
        if fast:
            cfg.fast = True
        result = sweep_mod.run_sweep(cfg, workers=workers, progress=not json_out)
        payload = {"ok": True} | _clean(result)

    if json_out:
        _print_json(payload)
        return
    rows = payload.get("runs") or []
    _table(
        f"sweep {payload.get('sweep_id')} · {len(rows)} runs · metric {cfg.metric}",
        ["run_id", "params", cfg.metric],
        [
            (r.get("run_id"), json.dumps(r.get("params", {}), sort_keys=True), _fmt(_score(r, cfg.metric)))
            for r in rows[:40]
        ],
    )
    _console().print(f"best [bold]{json.dumps(payload.get('best'), default=str)}[/bold]")


def _score(row: Mapping[str, Any], metric: str) -> Any:
    if metric in row:
        return row[metric]
    metrics = row.get("metrics") or {}
    return metrics.get(metric) if isinstance(metrics, dict) else None


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run id from `lab runs list`."),
    open_browser: bool = typer.Option(False, "--open", help="Open it in a browser."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the HTML here instead."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Render a run's self-contained HTML report."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        report_mod = _module("lab.backtest.report", what="the report renderer", json_out=json_out)
        path = report_mod.render_report(run_id, out=out, open_browser=open_browser)
        payload = {"ok": True, "run_id": run_id, "report": str(path)}

    if json_out:
        _print_json(payload)
        return
    _console().print(f"report [cyan]{path}[/cyan]")


# --- strategy library ---------------------------------------------------------


@strategy_app.command("promote")
def strategy_promote(
    run_id: str = typer.Argument(..., help="The run whose code you want. From `lab runs list`."),
    name: Optional[str] = typer.Option(
        None, "--name", help="Promote under a different filename (no .py)."
    ),
    no_config: bool = typer.Option(
        False, "--no-config", help="Do not write a cfg/<name>.yaml alongside it."
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing strategy or config of the same name."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Say what would happen and change nothing."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Copy the strategy that produced a run into the library.

    Keyed on the run, not on a file path. A research session rewrites its
    workspace files in place, so the code sitting where a run says its strategy
    lived is often a later variant -- on a real session the recommended run
    scored 0.636 while the file left under that name scored 0.510. This copies
    the source archived inside the run itself, verified against the hash taken
    when it executed, and writes a config carrying the params that run resolved.
    """
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        promote = _module("lab.registry.promote", what="the promoter", json_out=json_out)
        try:
            done = promote.promote_run(
                run_id,
                name=name,
                write_config=not no_config,
                force=force,
                dry_run=dry_run,
            )
        except promote.PromotionError as exc:
            _fail(str(exc), json_out=json_out, code=REFUSED, run_id=run_id)
        payload = {"ok": True, **done.to_dict()}

    if json_out:
        _print_json(payload)
        return

    console = _console()
    verb = "would promote" if done.dry_run else "promoted"
    console.print(f"{verb} [bold]{done.strategy}[/bold] from [cyan]{done.run_id}[/cyan]")
    console.print(f"  strategy [cyan]{done.strategy_path}[/cyan]"
                  + ("  [yellow](replaced)[/yellow]" if done.overwrote else ""))
    if done.config_path:
        console.print(f"  config   [cyan]{done.config_path}[/cyan]")
    if done.params:
        console.print(f"  params   {json.dumps(done.params, default=str)}")
    if not done.dry_run:
        cfg_hint = done.config_path or "<your config>"
        console.print(f"\nverify it reproduces: [dim]lab backtest {done.strategy_path} "
                      f"--config {cfg_hint}[/dim]")


# --- registry -----------------------------------------------------------------


@runs_app.command("list")
def runs_list(
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s"),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="backtest|paper|live|sweep."),
    origin: Optional[str] = typer.Option(None, "--origin", help="human|agent-loop."),
    sweep_id: Optional[str] = typer.Option(None, "--sweep-id"),
    limit: int = typer.Option(20, "--limit", "-n"),
    offset: int = typer.Option(0, "--offset"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Most recent runs first."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry.runs import RunRegistry

        records = RunRegistry().list(
            strategy=strategy, kind=kind, origin=origin, sweep_id=sweep_id,
            limit=limit, offset=offset,
        )
        payload = {
            "ok": True,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "runs": [r.to_dict() for r in records],
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        f"runs ({len(records)})",
        ["run_id", "strategy", "kind", "status", "created", "sharpe", "return", "trades"],
        [
            (
                r.run_id, r.strategy, r.kind, r.status,
                r.created_at.strftime("%Y-%m-%d %H:%M"),
                _fmt(r.metrics.get("sharpe")),
                _fmt(r.metrics.get("total_return")),
                r.metrics.get("trades"),
            )
            for r in records
        ],
    )


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Everything the registry knows about one run."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry.runs import RunRegistry

        record = RunRegistry().get(run_id)
        if record is None:
            _fail(f"no run {run_id!r}", json_out=json_out, run_id=run_id)
        payload = {"ok": True} | record.to_dict()
        payload["artifact_dir"] = str(record.artifact_dir)

    if json_out:
        _print_json(payload)
        return
    meta = [
        (k, payload.get(k))
        for k in ("strategy", "kind", "status", "origin", "attempt", "created_at",
                  "finished_at", "git_commit", "config_hash", "data_version",
                  "sweep_id", "parent_run_id", "notes", "error", "artifact_dir")
    ]
    _table(record.run_id, ["field", "value"], meta)
    if record.params:
        _table("params", ["key", "value"], sorted((k, _fmt(v)) for k, v in record.params.items()))
    if record.metrics:
        _table(
            "metrics",
            ["metric", "value"],
            [(k, _fmt(record.metrics.get(k))) for k in _HEADLINE if k in record.metrics],
        )


@runs_app.command("compare")
def runs_compare(
    run_ids: list[str] = typer.Argument(..., help="Two or more run ids."),
    all_metrics: bool = typer.Option(False, "--all", help="Every metric, not just the headline set."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Side-by-side metrics. Params that are identical across runs are dropped."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry.runs import RunRegistry

        try:
            frame = RunRegistry().compare(run_ids)
        except ValueError as exc:
            _fail(str(exc), json_out=json_out)
        # A metric can share a name with a meta column (`attempt` does), and
        # pandas refuses to serialize duplicate labels; keep the first.
        frame = frame.loc[:, ~frame.columns.duplicated()]
        rows = json.loads(frame.to_json(orient="records", date_format="iso") or "[]")
        payload = {
            "ok": True,
            "run_ids": list(run_ids),
            "columns": [str(c) for c in frame.columns],
            "rows": rows,
        }

    if json_out:
        _print_json(payload)
        return
    # Transposed on purpose: with three runs the interesting axis is the metric,
    # and a wide table of 70 columns is unreadable in a terminal.
    fields = payload["columns"] if all_metrics else [
        c for c in payload["columns"]
        if c.startswith("param.")
        or c in {"strategy", "kind", "status", "origin", "attempt", "created_at", "data_version"}
        or c in _HEADLINE
    ]
    ids = [r.get("run_id", "?") for r in rows]
    _table(
        "compare",
        ["field", *ids],
        [(f, *[_fmt(r.get(f)) for r in rows]) for f in fields],
    )


@runs_app.command("rerun")
def runs_rerun(
    run_id: Optional[str] = typer.Argument(None, help="Run to reproduce."),
    stale: bool = typer.Option(
        False,
        "--stale",
        help=(
            "Reproduce every run whose trade ledger predates the ledger fix "
            "(no ledger_residual recorded, and open positions at the end). "
            "Runs whose strategy file is gone are skipped and listed."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Re-run a stored run from its own recorded config, as a NEW run.

    Backtest artifacts are immutable -- that is the reproducibility promise, and
    it is why the console shows an old run exactly as it was computed, engine
    bug and all. When the engine changes, the way to see corrected numbers is to
    produce a new run, not to rewrite history. The original is left untouched so
    the two can be compared.
    """
    json_out = _want_json(json_out)
    if bool(run_id) == bool(stale):
        raise typer.BadParameter("give exactly one of a run id or --stale")

    if stale:
        _rerun_stale(json_out)
        return

    with _guard(json_out), _json_stdout(json_out):
        from lab.backtest.runner import BacktestConfig, run_backtest
        from lab.registry.runs import RunRegistry, run_artifact_dir

        record = RunRegistry().get(run_id)
        if record is None:
            _fail(f"no such run: {run_id}", json_out=json_out, code=ERROR, run_id=run_id)

        stored = dict(record.config or {})
        if not stored:
            saved = run_artifact_dir(run_id) / "config.json"
            if saved.exists():
                stored = json.loads(saved.read_text(encoding="utf-8"))
        if not stored:
            _fail(
                f"run {run_id} has no stored config to reproduce",
                json_out=json_out, code=ERROR, run_id=run_id,
            )

        # These are outputs of the original run, not inputs to a new one.
        for key in ("resolved_params", "strategy_hash"):
            stored.pop(key, None)

        strategy = Path(str(stored.get("strategy", "")))
        if not strategy.exists():
            _fail(
                f"the strategy file this run used is gone: {strategy}. "
                f"Agent-authored variants live in the loop workspace and are not "
                f"kept once it is cleaned up.",
                json_out=json_out, code=ERROR, run_id=run_id,
            )

        result = run_backtest(BacktestConfig.from_mapping(stored))
        before, after = record.metrics or {}, result.metrics
        payload = {
            "ok": True,
            "source_run_id": run_id,
            "run_id": result.run_id,
            "before": {k: before.get(k) for k in ("total_return", "trades", "hit_rate")},
            "after": {k: after.get(k) for k in ("total_return", "trades", "hit_rate")},
            "ledger_residual": after.get("ledger_residual"),
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        f"rerun {run_id} -> {payload['run_id']}",
        ["metric", "before", "after"],
        [
            (k, _fmt(payload["before"].get(k)), _fmt(payload["after"].get(k)))
            for k in ("total_return", "trades", "hit_rate")
        ],
    )


def _stale_runs() -> list[str]:
    """Runs whose trade ledger predates the fix, newest first.

    The tell is the *absence* of ``ledger_residual`` combined with open
    positions at the end: a run that finished flat booked every leg anyway, so
    re-running it would only burn CPU to reproduce the same numbers.
    """
    from lab.registry.runs import RunRegistry

    out: list[str] = []
    for record in RunRegistry().list(limit=10_000):
        metrics = record.metrics or {}
        if "ledger_residual" not in metrics and metrics.get("open_positions"):
            out.append(record.run_id)
    return out


def _rerun_stale(json_out: bool) -> None:
    rows: list[tuple[str, str, str]] = []
    with _guard(json_out), _json_stdout(json_out):
        from lab.backtest.runner import BacktestConfig, run_backtest
        from lab.registry.runs import RunRegistry

        registry = RunRegistry()
        done: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        for rid in _stale_runs():
            record = registry.get(rid)
            stored = dict((record.config if record else None) or {})
            for key in ("resolved_params", "strategy_hash"):
                stored.pop(key, None)
            strategy = Path(str(stored.get("strategy", "")))
            if not stored or not strategy.exists():
                reason = "no stored config" if not stored else f"strategy gone: {strategy.name}"
                skipped.append({"run_id": rid, "reason": reason})
                rows.append((rid, "skipped", reason))
                continue
            try:
                result = run_backtest(BacktestConfig.from_mapping(stored))
            except Exception as exc:  # one bad run must not stop the sweep
                skipped.append({"run_id": rid, "reason": f"{type(exc).__name__}: {exc}"})
                rows.append((rid, "failed", str(exc)[:60]))
                continue
            before = (record.metrics or {}).get("trades") if record else None
            done.append(
                {"source_run_id": rid, "run_id": result.run_id,
                 "trades_before": before, "trades_after": result.metrics.get("trades")}
            )
            rows.append((rid, result.run_id, f"trades {before} -> {result.metrics.get('trades')}"))

        payload = {"ok": True, "rerun": done, "skipped": skipped,
                   "counts": {"rerun": len(done), "skipped": len(skipped)}}

    if json_out:
        _print_json(payload)
        return
    _table(f"rerun stale · {len(rows)} run(s)", ["source", "new run", "note"], rows)


@runs_app.command("delete")
def runs_delete(
    run_id: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Remove a run from the registry. Artifacts on disk are left alone."""
    json_out = _want_json(json_out)
    if not yes:
        if json_out or not sys.stdin.isatty():
            _fail("refusing to delete without --yes", json_out=json_out, code=USAGE, run_id=run_id)
        if not typer.confirm(f"delete run {run_id}?"):
            raise typer.Exit(OK)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry.runs import RunRegistry, run_artifact_dir

        deleted = RunRegistry().delete(run_id)
        payload = {
            "ok": True,
            "run_id": run_id,
            "deleted": bool(deleted),
            "artifact_dir": str(run_artifact_dir(run_id)),
        }

    if json_out:
        _print_json(payload)
        return
    _console().print(
        f"{'deleted' if deleted else 'no such run'} [bold]{run_id}[/bold]; "
        f"artifacts remain at [cyan]{payload['artifact_dir']}[/cyan]"
    )


# --- live ---------------------------------------------------------------------


@paper_app.command("start")
def paper_start(
    strategy: Optional[str] = typer.Argument(None),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    param: Optional[list[str]] = typer.Option(None, "--param", "-p"),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t"),
    broker: Optional[str] = typer.Option(None, "--broker", help="alpaca|sim."),
    at_time: Optional[str] = typer.Option(None, "--at", help='Decision time, e.g. "09:35".'),
    once: bool = typer.Option(
        False, "--once",
        help=(
            "Decide once for right now and exit, instead of sleeping until the "
            "next fire time. The mode a scheduled task wants."
        ),
    ),
    adopt: bool = typer.Option(
        False, "--adopt",
        help=(
            "Take the broker's positions as truth when they disagree with the "
            "stored book. Say this only after looking at both."
        ),
    ),
    settle_seconds: Optional[int] = typer.Option(
        None, "--settle-seconds",
        help=(
            "With --once, how long to wait for the orders just sent to fill "
            "before saving the book and exiting (default 90). 0 skips the wait, "
            "at the cost of saving intent rather than outcome."
        ),
    ),
    max_stale_sessions: Optional[int] = typer.Option(
        None, "--max-stale-sessions",
        help=(
            "Refuse to decide when the newest bar is older than this many trading "
            "sessions (default 2). 0 disables the guard."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Start the paper runner. Reconciles before it trades.

    Runs in the foreground until interrupted, deciding once per session at
    `--at`. With `--once` it decides for the current moment and exits, which is
    what a daily scheduled task should call -- see `lab paper schedule`.

    The book (positions and cash) is saved after every step and restored on the
    next start, so a reboot is not an event: the runner picks up what it held and
    reconciliation asks only whether anything changed while it was down.
    """
    json_out = _want_json(json_out)
    overrides = _parse_params(param)
    _refuse_if_killed(json_out, action="start a paper runner")

    with _guard(json_out), _json_stdout(json_out):
        runner_mod = _module("lab.live.runner", what="the live runner", json_out=json_out)
        cfg = _live_config(
            runner_mod, strategy=strategy, config=config, params=overrides,
            tickers=tickers, broker=broker, at_time=at_time, json_out=json_out,
        )
        if adopt:
            cfg.adopt = True
        if max_stale_sessions is not None:
            cfg.max_stale_sessions = int(max_stale_sessions)
        if settle_seconds is not None:
            cfg.settle_seconds = int(settle_seconds)
        runner = runner_mod.LiveRunner(cfg)
        try:
            runner.start(once=once)
        except KeyboardInterrupt:
            runner.stop()
        payload = {"ok": not runner.blocked} | _clean(runner.status())

    # A blocked run used to exit 0, so a scheduled task reported success while the
    # bot sat there having decided nothing. Refusing to trade is a correct
    # outcome, but it is not a successful one, and the scheduler is the only
    # thing watching at 09:35.
    if runner.blocked:
        _fail(
            runner.block_reason or "refused to trade",
            json_out=json_out, code=REFUSED, **{k: payload[k] for k in ("strategy", "run_id") if k in payload},
        )

    if json_out:
        _print_json(payload)
        return
    _table("paper runner", ["field", "value"], sorted(payload.items()))


def _live_config(runner_mod, *, strategy, config, params, tickers, broker, at_time, json_out):
    """Build a LiveConfig from a backtest-shaped YAML.

    A live config is a backtest config minus the fill model and plus broker
    wiring, so the same file drives both; unknown keys are dropped rather than
    rejected, which is what makes `--config cfg/momo.yaml` work for paper.
    """
    live_cls = getattr(runner_mod, "LiveConfig", None)
    if live_cls is None:
        _fail("lab.live.runner has no LiveConfig", json_out=json_out)
    raw: dict[str, Any] = {}
    if config is not None:
        import yaml

        if not Path(config).exists():
            _fail(f"no config at {config}", json_out=json_out, code=USAGE)
        raw = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    if strategy:
        raw["strategy"] = strategy
    if tickers:
        raw["tickers"] = _resolve_tickers(tickers)
    elif isinstance(raw.get("tickers"), str):
        raw["tickers"] = _resolve_tickers(raw["tickers"])
    if broker:
        raw["broker"] = broker
    if at_time:
        raw["at_time"] = at_time
    if params:
        raw["params"] = {**(raw.get("params") or {}), **params}
    if not raw.get("strategy"):
        raise typer.BadParameter("give a strategy path or a --config that names one")

    if hasattr(live_cls, "from_mapping"):
        return live_cls.from_mapping(raw)
    known = {f.name for f in dataclasses.fields(live_cls)}
    return live_cls(**{k: v for k, v in raw.items() if k in known})


@paper_app.command("schedule")
def paper_schedule(
    strategy: str = typer.Argument(..., help="Strategy file, e.g. strategies/blend_tilt.py"),
    config: Path = typer.Option(..., "--config", "-c"),
    at_time: str = typer.Option("09:35", "--at", help='Eastern decision time, e.g. "09:35".'),
    name: Optional[str] = typer.Option(None, "--name", help="Task name. Defaults to lab-<strategy>."),
    source: str = typer.Option("alpaca", "--source", help="Data source to pull before deciding."),
    broker: str = typer.Option("alpaca", "--broker"),
    install: bool = typer.Option(
        False, "--install",
        help="Actually register the Windows scheduled task. Without this, only writes the script.",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Write a daily pull-then-decide script, and optionally schedule it.

    A daily strategy does not need a process running all day. It needs one
    decision per session, and the two things that must happen in order: refresh
    the bars, then decide. This writes that as a script and, with `--install`,
    registers it with Windows Task Scheduler.

    One decision per process is what makes a reboot a non-event -- the book lives
    on disk between runs, so there is no in-memory state for a restart to lose.

    The task is written but NOT installed unless you pass `--install`, because
    registering a scheduled task changes your machine and should be a decision
    rather than a side effect. The `schtasks` line is printed either way.

    Note the time is *Eastern*; Task Scheduler fires in local time, so the
    printed command converts it for you and says what it assumed.
    """
    import subprocess
    from datetime import datetime as _dt
    from datetime import timedelta

    from lab.timeutil import ET

    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        strategy_path = Path(strategy).resolve()
        config_path = Path(config).resolve()
        if not strategy_path.exists():
            _fail(f"no strategy at {strategy_path}", json_out=json_out, code=USAGE)
        if not config_path.exists():
            _fail(f"no config at {config_path}", json_out=json_out, code=USAGE)

        task = name or f"lab-{strategy_path.stem}"
        root = Path.cwd().resolve()
        script = root / "scripts" / f"{task}.cmd"
        python = Path(sys.executable).resolve()

        # Resolve the universe here rather than pointing `pull` at the config:
        # --tickers takes a comma list or a universe file, and handing it a
        # backtest config would fail at 09:35 on a morning nobody is watching.
        from lab.backtest.runner import BacktestConfig

        try:
            bt = BacktestConfig.from_yaml(config_path)
        except Exception as exc:  # noqa: BLE001
            _fail(f"could not read {config_path}: {exc}", json_out=json_out, code=USAGE)
        universe = ",".join(bt.tickers)
        if not universe:
            _fail(f"{config_path} names no tickers", json_out=json_out, code=USAGE)
        timeframe = bt.timeframe or "1d"
        pull_source = source or bt.source or "alpaca"
        # A floor, not a window: the pull is incremental, so this only bounds how
        # far back it looks on a store that is already populated. Generous enough
        # that a task idle over a long holiday still overlaps what it has.
        since = (_dt.now(ET).date() - timedelta(days=120)).isoformat()

        hh, _, mm = at_time.partition(":")
        et_today = _dt.now(ET).replace(hour=int(hh), minute=int(mm or 0), second=0, microsecond=0)
        local = et_today.astimezone()
        local_hhmm = local.strftime("%H:%M")

        body = f"""@echo off
REM Generated by `lab paper schedule`. One session's decision, in order.
REM Editing this by hand is fine; re-running the command overwrites it.
cd /d "{root}"

REM Refresh bars first. The runner reads the parquet store, not a live feed,
REM so skipping this means deciding on yesterday's prices -- which the runner
REM now refuses to do rather than doing quietly.
"{python}" -m lab.cli pull --source {pull_source} --tickers "{universe}" ^
  --tf {timeframe} --since {since}
if errorlevel 1 (
  echo pull failed; not trading on stale data
  exit /b 1
)

REM One decision for right now, then exit.
"{python}" -m lab.cli paper start "{strategy_path}" --config "{config_path}" ^
  --broker {broker} --once
exit /b %errorlevel%
"""
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(body, encoding="utf-8")

        # PowerShell rather than schtasks, for one setting: StartWhenAvailable.
        # `schtasks /Create` cannot express it, and without it a machine that is
        # off or asleep at the trigger simply skips the day -- the decision is not
        # late, it never happens. With it, a missed run fires once when the machine
        # next comes up, and because the wrapper pulls before deciding, a late
        # decision is made on fresh data rather than on the morning's.
        #
        # Deliberately NOT waking the machine: that wants stored credentials and a
        # wake timer, and a trading bot that powers on your computer at dawn is a
        # bigger promise than a daily rebalance needs to make.
        days = "Monday,Tuesday,Wednesday,Thursday,Friday"
        ps_lines = [
            f"""$A = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c \"{script}\"'""",
            f"$T = New-ScheduledTaskTrigger -Weekly -DaysOfWeek {days} -At {local_hhmm}",
            "$S = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries"
            " -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)",
            f"Register-ScheduledTask -TaskName '{task}' -Action $A -Trigger $T -Settings $S"
            f" -Description 'strategy-lab daily paper decision' -Force",
        ]

        installed = False
        if install:
            proc = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "; ".join(ps_lines) + " | Out-Null",
                ],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                _fail(
                    f"Register-ScheduledTask failed: "
                    f"{(proc.stderr or proc.stdout).strip()[:300]}",
                    json_out=json_out,
                )
            installed = True

        payload = {
            "ok": True, "task": task, "script": str(script), "installed": installed,
            "at_eastern": at_time, "at_local": local_hhmm,
            "start_when_available": True,
            "command": "\n".join(ps_lines),
        }

    if json_out:
        _print_json(payload)
        return

    console = _console()
    console.print(f"wrote [cyan]{script}[/cyan]")
    console.print(
        f"  fires [bold]{local_hhmm}[/bold] local = {at_time} Eastern, Mon-Fri"
    )
    if installed:
        console.print(f"  registered as scheduled task [bold]{task}[/bold]")
        console.print(
            "  a run missed while the machine was off fires when it next wakes "
            "[dim](StartWhenAvailable)[/dim]"
        )
        console.print(
            f"  remove it with: [dim]Unregister-ScheduledTask -TaskName {task} "
            f"-Confirm:$false[/dim]"
        )
    else:
        console.print("\n  not installed. To register it, in PowerShell:")
        for line in payload["command"].splitlines():
            console.print(f"  [dim]{line}[/dim]")
    console.print(
        "\n  The script pulls bars, then decides once. A run missed because the "
        "machine was off fires when it next comes up, on freshly pulled data. "
        "Market holidays need no handling -- the runner finds no new session."
    )


@paper_app.command("stop")
def paper_stop(
    strategy: str = typer.Argument(..., help="Strategy name as `lab status` shows it."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Stop a running paper strategy from acting, from outside its process.

    The runner re-reads its pause flag from ``live_strategies`` every bar, which
    is the only channel that reaches a process this CLI does not own. It stops
    deciding and stops sending orders; the process itself stays up, heartbeating,
    so the console can still see it. Interrupt the runner to end the process, or
    `lab kill` to stop every strategy at once.
    """
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry import db
        from lab.timeutil import utcnow

        con = db.connect(db.journal_path())
        row = con.execute(
            "SELECT * FROM live_strategies WHERE strategy = ?", (strategy,)
        ).fetchone()
        if row is None:
            known = [
                r["strategy"]
                for r in con.execute("SELECT strategy FROM live_strategies").fetchall()
            ]
            _fail(
                f"no live strategy named {strategy!r}"
                + (f"; known: {', '.join(known)}" if known else ""),
                json_out=json_out,
                strategy=strategy,
            )
        already = bool(row["paused"])
        with db.transaction(con):
            con.execute(
                "UPDATE live_strategies SET paused = 1, updated_at = ? WHERE strategy = ?",
                (utcnow().isoformat(), strategy),
            )
        _journal_control(strategy, "stop", f"paper stop requested for {strategy}")
        payload = {
            "ok": True,
            "strategy": strategy,
            "paused": True,
            "already_paused": already,
            "status": row["status"],
            "pid": row["pid"],
            "host": row["host"],
            "note": "runner stops acting at its next bar; the process keeps running",
        }

    if json_out:
        _print_json(payload)
        return
    _console().print(
        f"[bold]{strategy}[/bold] paused (pid {payload['pid']} on {payload['host']}) — "
        f"{payload['note']}"
    )


def _journal_control(strategy: str, action: str, message: str) -> None:
    """An undocumented stop at 09:31 is indistinguishable from a crash at 09:31."""
    try:
        from lab.engine.events import EventKind, event
        from lab.registry.journal import EventJournal
        from lab.timeutil import utcnow

        EventJournal().append(
            event(
                EventKind.LOG, "cli", at=utcnow(), strategy=strategy, message=message,
                payload={"control": action, "actor": "cli"},
            )
        )
    except Exception as exc:  # noqa: BLE001 - the control action still stands
        _note(f"[yellow]warning[/yellow] could not journal the control action: {exc}")


@app.command()
def status(
    limit: int = typer.Option(10, "--limit", "-n", help="Recent journal events to include."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Kill switch, live strategies, heartbeats, adapters and recent runs."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        settings = _settings()
        engaged, reason = _kill_switch().engaged()
        payload: dict[str, Any] = {
            "ok": True,
            "kill_switch": {
                "engaged": engaged,
                "reason": reason,
                "file": str(settings.kill_file),
                "env_flag": settings.kill_switch_env,
            },
            "paths": {"data": str(settings.paths.data), "runs": str(settings.paths.runs)},
        }
        payload["live"] = _live_rows()
        payload["heartbeats"] = _heartbeats()
        payload["events"] = _recent_events(limit)
        payload["runs"] = _recent_runs(5)
        payload["adapters"] = _adapter_rows()
        payload["agent"] = _agent_backend()

    if json_out:
        _print_json(payload)
        return
    kill = payload["kill_switch"]
    _console().print(
        f"kill switch [bold red]ENGAGED[/bold red] — {kill['reason']}"
        if kill["engaged"]
        else "kill switch [green]clear[/green]"
    )
    agent = payload["agent"]
    # Which backend the agent layer would spend on is a standing question, and
    # "subscription" vs "api" is the half of the answer that costs money.
    _console().print(
        f"agent provider [bold]{agent['active']}[/bold] · {agent['model']} · "
        f"billing {agent['billing']}"
        if agent["active"]
        else f"agent provider [yellow]none usable[/yellow] — {_first_line(agent['reason'])}"
    )
    _table(
        "live strategies",
        ["strategy", "kind", "status", "paused", "pid", "updated"],
        [
            (r.get("strategy"), r.get("kind"), r.get("status"), r.get("paused"),
             r.get("pid"), r.get("updated_at"))
            for r in payload["live"]
        ],
    )
    _table(
        "heartbeats",
        ["source", "age_s", "at"],
        [(k, _fmt(v.get("age_s")), v.get("at")) for k, v in payload["heartbeats"].items()],
    )
    _table(
        "adapters",
        ["name", "available", "reason"],
        [(a["name"], a["available"], a["reason"]) for a in payload["adapters"]],
    )
    _table(
        "recent runs",
        ["run_id", "strategy", "kind", "status"],
        [(r["run_id"], r["strategy"], r["kind"], r["status"]) for r in payload["runs"]],
    )


def _live_rows() -> list[dict[str, Any]]:
    # live_strategies lives in the *journal* database, next to the events it is
    # read alongside; the runner and the console both write it there.
    from lab.registry import db

    con = db.connect(db.journal_path())
    rows = con.execute("SELECT * FROM live_strategies ORDER BY strategy").fetchall()
    return [dict(r) for r in rows]


def _heartbeats() -> dict[str, Any]:
    from lab.registry.journal import EventJournal

    return _clean(EventJournal().last_heartbeats())


def _recent_events(limit: int) -> list[dict[str, Any]]:
    from lab.registry.journal import EventJournal

    journal = EventJournal()
    latest = journal.latest_seq()
    return _clean(journal.tail(since_seq=max(latest - max(limit, 0), 0), limit=max(limit, 0)))


def _recent_runs(limit: int) -> list[dict[str, Any]]:
    from lab.registry.runs import RunRegistry

    return [r.to_dict() for r in RunRegistry().list(limit=limit)]


def _adapter_rows() -> list[dict[str, Any]]:
    from lab.adapters.base import list_adapters

    return [info.to_dict() for info in list_adapters()]


@app.command()
def kill(
    release: bool = typer.Option(False, "--release", help="Disengage instead of engaging."),
    reason: str = typer.Option("", "--reason", help="Why, for the journal and the sentinel file."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Engage (or release) the kill switch. Safe to run twice."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        switch = _kill_switch()
        settings = _settings()
        before, _ = switch.engaged()
        if release:
            switch.release()
        else:
            switch.engage(reason)
        # `changed` is a state diff, not the return value of release(): that
        # reports whether trading is now permitted, which is a different
        # question when LAB_KILL_SWITCH is set in the environment.
        engaged, now_reason = switch.engaged()
        payload = {
            "ok": True,
            "engaged": engaged,
            "changed": before != engaged,
            "released": bool(release),
            "reason": now_reason if engaged else (reason or None),
            "kill_file": str(settings.kill_file),
            "env_flag": settings.kill_switch_env,
        }
        if switch is _SentinelKill:  # the real module journals its own flips
            _journal_kill(engaged, reason or (now_reason or ""))

    if json_out:
        _print_json(payload)
    elif engaged:
        _console().print(
            f"[bold red]kill switch ENGAGED[/bold red] — {payload['reason']}\n"
            f"sentinel [cyan]{payload['kill_file']}[/cyan]"
        )
    else:
        _console().print("[green]kill switch released[/green]")

    # Releasing cannot clear an env flag; saying "released" when the switch is
    # still on would be the single most dangerous lie this CLI could tell.
    if release and engaged:
        _note(
            "[bold yellow]still engaged[/bold yellow] — LAB_KILL_SWITCH is set in the "
            "environment; unset it and re-run"
        )
        raise typer.Exit(ERROR)


def _journal_kill(engaged: bool, reason: str) -> None:
    """Best effort: the switch must work even if the journal cannot be written."""
    try:
        from lab.engine.events import EventKind, event
        from lab.registry.journal import EventJournal
        from lab.timeutil import utcnow

        EventJournal().append(
            event(
                EventKind.BREAKER,
                "cli",
                at=utcnow(),
                message=("kill switch engaged" if engaged else "kill switch released"),
                payload={"reason": reason, "engaged": engaged},
            )
        )
    except Exception as exc:  # noqa: BLE001 - never block a panic stop
        _note(f"[yellow]warning[/yellow] could not journal the kill switch: {exc}")


# --- store introspection ------------------------------------------------------


@data_app.command("coverage")
def data_coverage(
    table: str = typer.Option("bars", "--table", help="bars | signals."),
    raw: bool = typer.Option(False, "--raw", help="Also summarize the raw snapshot log."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """What the store actually holds, per source and ticker."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.store import parquet_io as pio

        frame = pio.coverage(table)
        rows = json.loads(frame.to_json(orient="records", date_format="iso") or "[]")
        payload: dict[str, Any] = {
            "ok": True,
            "table": table,
            "rows": rows,
            "totals": {
                "partitions": len(rows),
                "rows": int(frame["rows"].sum()) if len(frame) else 0,
                "tickers": int(frame["ticker"].nunique()) if len(frame) else 0,
                "sources": sorted(frame["source"].unique().tolist()) if len(frame) else [],
            },
            "data_version": pio.data_version(table),
        }
        if raw:
            from lab.store import raw as raw_store

            payload["raw"] = _clean(raw_store.summary())

    if json_out:
        _print_json(payload)
        return
    _table(
        f"{table} coverage · {payload['totals']['rows']:,} rows",
        ["source", "ticker", "timeframe", "rows", "first", "last"],
        [(r["source"], r["ticker"], r["timeframe"], r["rows"], r["first"], r["last"]) for r in rows],
    )
    if raw:
        _table(
            "raw snapshots",
            ["source", "endpoint", "files", "bytes", "first_day", "last_day"],
            [
                (r["source"], r["endpoint"], r["files"], r["bytes"], r["first_day"], r["last_day"])
                for r in payload["raw"]
            ],
        )
    _console().print(f"data_version [bold]{payload['data_version']}[/bold]")


@data_app.command("version")
def data_version_cmd(
    table: str = typer.Option("bars", "--table", help="bars | signals."),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t"),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """The digest stamped on every run that reads this slice of the store."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.store import parquet_io as pio

        universe = _resolve_tickers(tickers, required=False) if tickers else None
        payload = {
            "ok": True,
            "table": table,
            "tickers": universe,
            "source": source,
            "data_version": pio.data_version(table, tickers=universe, source=source),
        }

    if json_out:
        _print_json(payload)
        return
    _console().print(payload["data_version"])


@app.command()
def adapters(json_out: bool = typer.Option(False, "--json")) -> None:
    """Which data sources can run right now, and why not."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        payload = {"ok": True, "adapters": _adapter_rows()}

    if json_out:
        _print_json(payload)
        return
    _table(
        "adapters",
        ["name", "provides", "available", "reason", "quota"],
        [
            (
                a["name"], ",".join(a["provides"]),
                "[green]yes[/green]" if a["available"] else "[red]no[/red]",
                a["reason"],
                f"{a['quota_used']}/{a['quota_limit']}" if a["quota_limit"] is not None else "",
            )
            for a in payload["adapters"]
        ],
    )


@app.command()
def strategies(
    directory: Optional[Path] = typer.Option(None, "--dir", help="Defaults to strategies/."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Every loadable strategy and its declared parameters."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.engine.loader import discover

        found = discover(directory)
        payload = {"ok": True, "count": len(found), "strategies": _clean(found)}

    if json_out:
        _print_json(payload)
        return
    _table(
        f"strategies ({len(found)})",
        ["name", "path", "params", "doc"],
        [
            (
                s["name"], s["path"],
                json.dumps(s.get("params", {}), sort_keys=True) if s.get("loadable")
                else f"[red]{s.get('error', '')}[/red]",
                s.get("doc", ""),
            )
            for s in payload["strategies"]
        ],
    )


# --- govgreed -----------------------------------------------------------------


@govgreed_app.command("pull")
def govgreed_pull(
    top_n_enrich: int = typer.Option(5, "--enrich", help="Per-ticker lookups after the list calls."),
    write: bool = typer.Option(True, "--write/--no-write", help="Persist normalized events."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """One daily snapshot, inside the call budget. Raw responses are kept verbatim."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.adapters.base import AdapterError, get_adapter
        from lab.store import parquet_io as pio
        from lab.store.schema import SIGNALS

        adapter = get_adapter("govgreed")
        ok, reason = adapter.available()
        if not ok:
            _fail(f"govgreed is unavailable: {reason}", json_out=json_out, available=False)
        try:
            events = adapter.daily_pull(top_n_enrich=top_n_enrich)
        except AdapterError as exc:
            _fail(f"govgreed: {exc}", json_out=json_out, last_pull=_clean(adapter.last_pull))
        written = pio.write_events(events) if (write and events) else 0
        payload = {
            "ok": True,
            "events": len(events),
            "rows_written": int(written),
            "data_version": pio.data_version(SIGNALS, source="govgreed"),
            "last_pull": _clean(adapter.last_pull),
        }

    if json_out:
        _print_json(payload)
        return
    _console().print(
        f"pulled [bold]{payload['events']}[/bold] events, wrote {payload['rows_written']} rows"
    )
    _table(
        "steps",
        ["step", "ok", "events"],
        [(s.get("step"), s.get("ok"), s.get("events")) for s in payload["last_pull"].get("steps", [])],
    )


@govgreed_app.command("status")
def govgreed_status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Quota *and* how much snapshot history has accumulated.

    The second number is the one that matters: the vendor sells no historical
    backfill on the free tier, so the raw snapshot log **is** the backtest
    dataset for this source, and a config pointed at it is worthless until that
    log has depth.
    """
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.adapters.base import get_adapter
        from lab.store import parquet_io as pio
        from lab.store import raw as raw_store
        from lab.store.schema import SIGNALS

        adapter = get_adapter("govgreed")
        info = adapter.info()
        snapshots = [s for s in raw_store.summary("govgreed")]
        days = sorted({s["first_day"] for s in snapshots} | {s["last_day"] for s in snapshots})
        history = {
            "endpoints": _clean(snapshots),
            "files": sum(int(s["files"]) for s in snapshots),
            "bytes": sum(int(s["bytes"]) for s in snapshots),
            "first_day": days[0] if days else None,
            "last_day": days[-1] if days else None,
            "days_with_snapshots": _snapshot_days(),
        }
        frame = pio.coverage(SIGNALS)
        frame = frame[frame["source"] == "govgreed"] if len(frame) else frame
        store = {
            "rows": int(frame["rows"].sum()) if len(frame) else 0,
            "tickers": int(frame["ticker"].nunique()) if len(frame) else 0,
            "first": _clean(frame["first"].min()) if len(frame) else None,
            "last": _clean(frame["last"].max()) if len(frame) else None,
            "data_version": pio.data_version(SIGNALS, source="govgreed"),
        }
        payload = {
            "ok": True,
            "available": info.available,
            "reason": info.reason,
            "quota": _clean(info.detail.get("quota", {})),
            "quota_used": info.quota_used,
            "quota_limit": info.quota_limit,
            "quota_tier": info.quota_tier,
            "base_url": info.detail.get("base_url"),
            "last_pull": _clean(info.detail.get("last_pull", {})),
            "raw_history": history,
            "store": store,
            "note": (
                "No vendor backfill on the free tier: the raw snapshot log is the only "
                "signal history that exists, so a govgreed backtest is exactly as long as "
                "`days_with_snapshots`."
            ),
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        "govgreed",
        ["field", "value"],
        [
            ("available", payload["available"]),
            ("reason", payload["reason"]),
            ("quota", f"{payload['quota_used']}/{payload['quota_limit']} ({payload['quota_tier']})"),
            ("base_url", payload["base_url"]),
            ("snapshot days", history["days_with_snapshots"]),
            ("snapshot files", history["files"]),
            ("snapshot bytes", history["bytes"]),
            ("first day", history["first_day"]),
            ("last day", history["last_day"]),
            ("signal rows in store", store["rows"]),
            ("signal tickers", store["tickers"]),
        ],
    )
    _note(f"[dim]{payload['note']}[/dim]")


def _snapshot_days() -> int:
    root = _settings().paths.raw / "govgreed"
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir())


# --- agent --------------------------------------------------------------------


def _first_line(text: Any) -> str:
    """`resolve()` reports every rejected backend at once; a table cell wants one."""
    lines = str(text or "").splitlines()
    return lines[0].strip() if lines else ""


def _agent_backend() -> dict[str, Any]:
    """Which model backend would actually serve a call, without making one.

    Best effort by construction: `lab status` and `lab agent status` must still
    answer when the agent extra is not installed or nothing is configured, so a
    missing backend is a field in the payload rather than a failed command.
    """
    settings = _settings()
    out: dict[str, Any] = {
        "requested": settings.agent_provider,
        "active": None,
        "billing": None,
        "model": settings.agent_model,
        "forces_tools": None,
        "reason": "",
    }
    try:
        from lab.agent import providers as pmod

        provider = pmod.resolve()
    except Exception as exc:  # noqa: BLE001 - includes ProviderUnavailable and ImportError
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update(
        active=provider.name,
        billing=provider.billing,
        model=provider.model,
        forces_tools=provider.forces_tools,
    )
    return out


@agent_app.command("author")
def agent_author(
    seed: Path = typer.Option(..., "--seed", "-s", help="Strategy the loop starts from."),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Backtest config every iteration is scored against (universe, dates, fills, limits).",
    ),
    freeform: bool = typer.Option(
        False,
        "--freeform/--grid",
        help=(
            "grid (default): the model may only propose new PARAMETER values. "
            "freeform: it may also rewrite the strategy source. Freeform explores "
            "further and overfits faster."
        ),
    ),
    objective: str = typer.Option(
        "", "--objective", "-o", help="One-line brief handed to the model each iteration."
    ),
    iterations: int = typer.Option(10, "--iterations", "-n"),
    metric: str = typer.Option(
        "oos_sharpe", "--metric", "-m", help="Fitness function. Keep it out-of-sample."
    ),
    oos_split: str = typer.Option(
        "4:1", "--oos-split", help="Walk-forward train:test blocks used to score each iteration."
    ),
    periods: int = typer.Option(
        4,
        "--periods",
        help=(
            "Break each run into N contiguous sub-periods so a strategy that only "
            "works in one market regime is visible. Free (it slices an equity curve "
            "that already exists). Pair with --metric worst_sharpe to make "
            "consistency the fitness function. 0 disables."
        ),
    ),
    budget: float = typer.Option(5.0, "--budget", help="Hard ceiling for model spend."),
    model: Optional[str] = typer.Option(None, "--model"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Model backend: anthropic|claude_code|openai. Default: configured."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Let the model propose parameter variations; the gate and the registry judge them."""
    json_out = _want_json(json_out)
    _refuse_if_killed(json_out, action="run the author loop")
    if provider:
        # Scoped to this process: the loop resolves its backend through settings,
        # so overriding the env is the honest way to make the flag win.
        os.environ["LAB_AGENT_PROVIDER"] = provider
        from lab.config import reset_settings_cache

        reset_settings_cache()

    with _guard(json_out), _json_stdout(json_out):
        loop_mod = _module("lab.agent.author_loop", what="the author loop", json_out=json_out)
        cfg = loop_mod.AuthorLoopConfig(
            seed_strategy=seed,
            config=config,
            grid_or_freeform="freeform" if freeform else "grid",
            objective=objective,
            iterations=iterations,
            metric=metric,
            oos_split=oos_split,
            robustness_periods=periods,
            budget_usd=budget,
            model=model,
        )
        payload = {"ok": True} | _clean(loop_mod.run_author_loop(cfg))

    if json_out:
        _print_json(payload)
        return
    lineage = payload.get("lineage") or []
    _table(
        f"author loop · {len(lineage)} iterations",
        ["n", "run_id", "params", metric, "rationale"],
        [
            (
                it.get("n"), it.get("run_id"),
                json.dumps(it.get("params", {}), sort_keys=True),
                _fmt(_score(it, metric)),
                (it.get("rationale") or "")[:80],
            )
            for it in lineage
        ],
    )
    _console().print(f"best [bold]{json.dumps(payload.get('best'), default=str)}[/bold]")


@agent_app.command("research")
def agent_research(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Backtest config every experiment is scored against. The agent cannot change it.",
    ),
    brief: str = typer.Option(
        "", "--brief", "-b",
        help="What to prioritise and when to be satisfied. Your instructions, in plain English.",
    ),
    brief_file: Optional[Path] = typer.Option(
        None, "--brief-file", help="Read the brief from a file instead."
    ),
    minutes: float = typer.Option(
        30.0, "--minutes", help="Wall-clock ceiling for THIS sitting. The session resumes."
    ),
    budget: float = typer.Option(5.0, "--budget", help="Ceiling on model spend."),
    max_calls: int = typer.Option(
        60, "--max-calls",
        help=(
            "Hard cap on model calls. On a Claude subscription nothing reports "
            "remaining quota, so this is the honest proxy for 'do not burn my week'."
        ),
    ),
    max_experiments: int = typer.Option(40, "--max-experiments"),
    metric: str = typer.Option("oos_sharpe", "--metric", "-m"),
    periods: int = typer.Option(4, "--periods"),
    oos_split: str = typer.Option(
        "4:1", "--oos-split",
        help=(
            "In-sample:out-of-sample split, by bar count. Worth widening for "
            "intraday: a metric's error bar tracks the holdout's calendar SPAN, "
            "not its bar count, so 20x more bars over a shorter period resolves "
            "less, not more."
        ),
    ),
    neighbourhood_runs: int = typer.Option(
        16, "--neighbourhood-runs",
        help=(
            "Ceiling on backtests spent perturbing the winning parameters before "
            "the session may finish, ~10% either side of each numeric value. Never "
            "exceeds two per parameter, so a small strategy costs less. Below that "
            "the check reports which parameters it could only nudge one way. Costs "
            "wall clock, not model budget. 0 disables it."
        ),
    ),
    strategies_dir: Optional[Path] = typer.Option(
        None, "--strategies", help="Folder the agent may run. Defaults to strategies/."
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Continue a session by id, with its history intact."
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        help="Name the new session, so a launcher knows the id before it starts.",
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Open-ended research: no seed, no iteration count, stops when satisfied.

    Hands the agent the whole strategy folder and a brief, and lets it choose what
    to try. It sees the clock, the budget and its own past results every turn, and
    has to name a run_id when it claims to be done. It cannot touch the date range,
    the universe, the fill model or the risk limits.

    The session is a file: a sitting cut short by time, budget or a subscription
    limit resumes later with everything it learned.
    """
    json_out = _want_json(json_out)
    _refuse_if_killed(json_out, action="run a research session")

    if brief_file:
        if not brief_file.exists():
            _fail(f"no brief at {brief_file}", json_out=json_out, code=USAGE)
        brief = brief_file.read_text(encoding="utf-8")
    if provider:
        os.environ["LAB_AGENT_PROVIDER"] = provider
        from lab.config import reset_settings_cache

        reset_settings_cache()

    with _guard(json_out), _json_stdout(json_out):
        mod = _module("lab.agent.research", what="the research loop", json_out=json_out)
        cfg = mod.ResearchConfig(
            config=config,
            brief=brief,
            max_minutes=minutes,
            budget_usd=budget,
            max_calls=max_calls,
            max_experiments=max_experiments,
            metric=metric,
            oos_split=oos_split,
            robustness_periods=periods,
            neighbourhood_runs=neighbourhood_runs,
            strategies_dir=strategies_dir,
            model=model,
            session_id=resume or session_id,
        )
        payload = {"ok": True} | _clean(mod.run_research(cfg, resume=resume))

    if json_out:
        _print_json(payload)
        return

    experiments = payload.get("experiments") or []
    _table(
        f"research {payload.get('session_id')} · {payload.get('stopped_because')}",
        ["n", "strategy", "params", "oos sharpe", "worst period"],
        [
            (
                e.get("n"), e.get("strategy"),
                json.dumps(e.get("params", {}), sort_keys=True)[:40],
                _fmt(e.get("oos_sharpe")), _fmt(e.get("worst_period_sharpe")),
            )
            for e in experiments
        ],
    )
    console = _console()
    market = (payload.get("market") or {}).get("oos") or {}
    if market:
        console.print(f"market reference (oos sharpe): [bold]{_fmt(market.get('sharpe'))}[/bold]")
    best = payload.get("best")
    if best:
        console.print(f"best: [bold]{best.get('run_id')}[/bold] {best.get('strategy')} "
                      f"score {_fmt(best.get('score'))}")
    if payload.get("verdict"):
        console.print(f"verdict: {payload['verdict']}")
    if payload.get("next_steps"):
        console.print(f"next: {payload['next_steps']}")
    console.print(
        f"{payload.get('calls')} calls · {_fmt(payload.get('elapsed_minutes'))} min · "
        f"spend {_fmt(payload.get('spend_usd'))} ({payload.get('billing')})"
    )
    if payload.get("resumable"):
        console.print(
            f"resume with: [bold]lab agent research -c {config} "
            f"--resume {payload.get('session_id')}[/bold]"
        )


@agent_app.command("providers")
def agent_providers(
    probe: bool = typer.Option(
        False,
        "--probe",
        help=(
            "SPENDS MONEY: makes one tiny real call to prove the wiring works "
            "(a few API tokens, or one request against your Claude subscription). "
            "Without this flag the command only reads configuration and costs nothing."
        ),
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Probe this backend instead of the configured one: anthropic|claude_code|openai.",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Which model backends can run right now, and why not.

    Free by default — it inspects keys, binaries and base URLs and calls nobody.
    `--probe` is the paid question, and it is a genuinely different one:
    *configured* and *working* are separate states, and only the first can be
    established without spending anything. One probe is one call, so the cost is
    a token or two on an API key, or a single request on a subscription.
    """
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        pmod = _module("lab.agent.providers", what="the provider registry", json_out=json_out)
        settings = _settings()

        if provider is not None and pmod.normalize(provider) not in pmod.REGISTRY:
            known = ", ".join(sorted(pmod.REGISTRY) + sorted(pmod.ALIASES))
            _fail(
                f"unknown provider {provider!r}; known: {known}",
                json_out=json_out,
                code=USAGE,
                provider=provider,
            )

        active, active_reason = None, ""
        try:
            active = pmod.resolve().name
        except Exception as exc:  # noqa: BLE001 - "nothing is usable" is a report, not a crash
            active_reason = str(exc)

        payload: dict[str, Any] = {
            "ok": True,
            "configured": pmod.normalize(settings.agent_provider),
            "active": active,
            "active_reason": active_reason,
            "model": settings.agent_model,
            "auto_order": list(pmod.AUTO_ORDER),
            "providers": [info.to_dict() for info in pmod.list_providers()],
            "probed": bool(probe),
            "probe": None,
        }
        if probe:
            try:
                result = _clean(pmod.probe(provider))
            except Exception as exc:  # noqa: BLE001 - keep the documented probe shape
                # probe() swallows failures *of the call*; the ones that escape
                # come from picking a provider at all. Reporting them in the
                # same shape means a caller parses one payload, not two.
                result = {
                    "provider": pmod.normalize(provider or settings.agent_provider),
                    "model": settings.agent_model,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            payload["probe"] = result
            payload["ok"] = bool(result.get("ok"))

    if json_out:
        _print_json(payload)
        if not payload["ok"]:
            raise typer.Exit(ERROR)
        return

    _table(
        "agent providers",
        ["name", "model", "available", "billing", "forces tools", "base url", "reason"],
        [
            (
                f"{p['name']} [bold]*[/bold]" if p["name"] == payload["active"] else p["name"],
                p["model"],
                "[green]yes[/green]" if p["available"] else "[red]no[/red]",
                p["billing"],
                # A provider that cannot force a schema is not broken, but the
                # caller is relying on a prompt rather than a guarantee.
                "yes" if p["forces_tools"] else "[yellow]prompted[/yellow]",
                p["base_url"] or "",
                p["reason"],
            )
            for p in payload["providers"]
        ],
    )
    if payload["active"]:
        _console().print(
            f"active [bold]{payload['active']}[/bold] "
            f"(LAB_AGENT_PROVIDER={payload['configured']}, auto order: "
            f"{' > '.join(payload['auto_order'])})"
        )
    else:
        _console().print(f"[bold red]no usable provider[/bold red]\n{payload['active_reason']}")

    result = payload["probe"]
    if result is None:
        _note(
            "[dim]configuration only — nothing was called. Add [bold]--probe[/bold] to spend "
            "one token proving it actually works.[/dim]"
        )
        return
    _table(
        "probe",
        ["field", "value"],
        [
            ("provider", result.get("provider")),
            ("model", result.get("model")),
            ("ok", "[green]yes[/green]" if result.get("ok") else "[red]no[/red]"),
            ("billing", result.get("billing")),
            ("tokens", f"{result.get('input_tokens', 0)} in / {result.get('output_tokens', 0)} out"),
            ("cost_usd", "-" if result.get("cost_usd") is None else _fmt(result.get("cost_usd"))),
            ("latency_ms", _fmt(result.get("latency_ms"))),
            ("reply", (result.get("text") or "")[:120]),
            ("error", result.get("error")),
        ],
    )
    if not payload["ok"]:
        raise typer.Exit(ERROR)


@agent_app.command("status")
def agent_status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Model spend, token counts and how many runs the agent has authored."""
    json_out = _want_json(json_out)

    with _guard(json_out), _json_stdout(json_out):
        from lab.registry import db

        settings = _settings()
        con = db.connect()
        db.init_db(con)
        totals = con.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) tin, "
            "COALESCE(SUM(output_tokens),0) tout, COALESCE(SUM(cost_usd),0) cost, "
            "COALESCE(SUM(ok),0) okc, MAX(at) last_at FROM agent_calls"
        ).fetchone()
        by_model = [
            dict(r)
            for r in con.execute(
                "SELECT model, COUNT(*) calls, COALESCE(SUM(cost_usd),0) cost "
                "FROM agent_calls GROUP BY model ORDER BY cost DESC"
            ).fetchall()
        ]
        agent_runs = con.execute(
            "SELECT COUNT(*) n FROM runs WHERE origin != 'human'"
        ).fetchone()["n"]
        backend = _agent_backend()
        payload = {
            "ok": True,
            # Not just the Anthropic key any more: `claude_code` needs none at
            # all, so "is a backend usable" is the question worth answering.
            "configured": bool(backend["active"]),
            "provider": backend["active"],
            "billing": backend["billing"],
            "provider_reason": backend["reason"],
            "model": settings.agent_model,
            "calls": int(totals["n"]),
            "calls_ok": int(totals["okc"]),
            "input_tokens": int(totals["tin"]),
            "output_tokens": int(totals["tout"]),
            "cost_usd": round(float(totals["cost"]), 6),
            "last_call_at": totals["last_at"],
            "by_model": _clean(by_model),
            "agent_runs": int(agent_runs),
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        "agent",
        ["field", "value"],
        [
            (
                "provider",
                f"{payload['provider']} ({payload['billing']})"
                if payload["configured"]
                else f"[red]none usable[/red] — {_first_line(payload['provider_reason'])}",
            ),
            ("model", payload["model"]),
            ("calls", f"{payload['calls_ok']}/{payload['calls']} ok"),
            ("tokens", f"{payload['input_tokens']} in / {payload['output_tokens']} out"),
            ("cost", f"${payload['cost_usd']:.4f}"),
            ("last call", payload["last_call_at"]),
            ("agent-authored runs", payload["agent_runs"]),
        ],
    )


# --- console ------------------------------------------------------------------


@app.command()
def ui(
    port: int = typer.Option(8787, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not launch a browser."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on source changes."),
    allow_research: bool = typer.Option(
        False,
        "--allow-research",
        help=(
            "Let the console START research sessions. Off by default: research cannot "
            "touch a broker, but it spends model budget and executes model-authored "
            "Python, so it is opt-in rather than folded in with pause/cancel/kill. "
            "Stopping a session never needs this."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Serve the console on 127.0.0.1 and open it. Blocks until interrupted."""
    json_out = _want_json(json_out)
    url = f"http://{host}:{port}/"

    if allow_research:
        # The flag and LAB_UI_ALLOW_RESEARCH are the same switch, not two: the
        # API reads it through settings either way.
        os.environ["LAB_UI_ALLOW_RESEARCH"] = "1"
        from lab.config import reset_settings_cache

        reset_settings_cache()

    with _guard(json_out), _json_stdout(json_out):
        settings = _settings()
        # Off-loopback means the console is reachable by anything on the LAN,
        # and its control routes can stop a live strategy. A token is the
        # minimum price of admission.
        if host not in {"127.0.0.1", "localhost", "::1"} and not settings.ui_token:
            _fail(
                f"refusing to bind {host} without LAB_UI_TOKEN; the console exposes control routes",
                json_out=json_out,
                code=USAGE,
            )
        api = _module("lab.api.app", what="the console API", json_out=json_out)
        if getattr(api, "app", None) is None:
            _fail("lab.api.app has no `app`", json_out=json_out)
        payload = {"ok": True, "url": url, "host": host, "port": port, "reload": reload}

    if json_out:
        _print_json(payload)
    else:
        _console().print(f"console on [cyan]{url}[/cyan] — ctrl-c to stop")
    if not no_open:
        import webbrowser

        webbrowser.open(url)
    _serve(api, host=host, port=port, reload=reload)


def _serve(api, *, host: str, port: int, reload: bool) -> None:
    import copy

    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    # uvicorn's access log goes to stdout by default, which would corrupt the
    # JSON payload of a `lab ui --json` that an agent is reading.
    log_config = copy.deepcopy(LOGGING_CONFIG)
    for handler in log_config.get("handlers", {}).values():
        if handler.get("stream") == "ext://sys.stdout":
            handler["stream"] = "ext://sys.stderr"
    target = "lab.api.app:app" if reload else api.app
    uvicorn.run(target, host=host, port=port, reload=reload, log_config=log_config)


# --- the zero-credential proof ------------------------------------------------


@app.command()
def demo(
    start: str = typer.Option("2022-01-01", "--start"),
    end: str = typer.Option("2024-01-01", "--end"),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t"),
    seed: int = typer.Option(7, "--seed", help="Synthetic generator seed."),
    no_seed_data: bool = typer.Option(False, "--no-seed", help="Use the store as it stands."),
    open_reports: bool = typer.Option(False, "--open", help="Open the reports in a browser."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """End-to-end proof with no credentials: seed, backtest twice, render reports.

    This is what a new user should run first. It needs no keys and no network:
    the synthetic adapter generates a deterministic tape plus alt-data events
    with a realistic disclosure lag, so the numbers describe the generator and
    nothing else — but the plumbing they exercise is the real thing.
    """
    json_out = _want_json(json_out)
    start_ts = _parse_ts(start, what="--start")
    end_ts = _parse_ts(end, what="--end")

    with _guard(json_out), _json_stdout(json_out):
        from lab.adapters.base import get_adapter
        from lab.backtest.runner import run_backtest
        from lab.store import parquet_io as pio
        from lab.store.schema import BARS, SIGNALS

        universe = _resolve_tickers(tickers)
        # The benchmark is seeded alongside the universe so every result can carry
        # its market reference -- a demo whose numbers arrived without their
        # caveat would be demonstrating the wrong thing.
        seed_names = universe + [t for t in ("SPY",) if t not in universe]
        seeded: dict[str, Any] = {"skipped": bool(no_seed_data)}
        if not no_seed_data:
            adapter = get_adapter("synthetic", seed=seed, tickers=seed_names)
            # Warm-up: the momentum config needs 210 bars of history before its
            # first decision, so seed earlier than the backtest window starts.
            lo = start_ts - timedelta(days=420)
            _note(f"[dim]seeding synthetic bars {lo.date()} → {end_ts.date()}[/dim]")
            bars = pio.write_bars(list(adapter.fetch_bars(seed_names, "1d", lo, end_ts)))
            events = pio.write_events(list(adapter.fetch_signals(start=lo, end=end_ts)))
            seeded = {
                "skipped": False,
                "tickers": len(universe),
                "bars": int(bars),
                "signals": int(events),
                "seed": seed,
            }
        seeded["bars_version"] = pio.data_version(BARS, tickers=universe)
        seeded["signals_version"] = pio.data_version(SIGNALS)

        cfg_dir = _settings().paths.cfg
        results: list[dict[str, Any]] = []
        for label, config_path in (
            ("momentum", cfg_dir / "momo.yaml"),
            ("govgreed (synthetic)", cfg_dir / "govgreed_synthetic.yaml"),
        ):
            if not config_path.exists():
                results.append({"name": label, "ok": False, "error": f"no config at {config_path}"})
                continue
            _note(f"[dim]backtesting {label}…[/dim]")
            cfg = _build_config(
                strategy=None, config=config_path, params={},
                tickers=",".join(universe), start=start, end=end,
            )
            # Pin the source: the demo seeds synthetic bars, and a store that
            # also holds a real pull is ambiguous by design rather than by
            # accident. Without this the demo would refuse to run on exactly the
            # machines that have gone furthest.
            cfg.source = "synthetic"
            result = run_backtest(cfg)
            results.append(
                {
                    "name": label,
                    "ok": True,
                    "run_id": result.run_id,
                    "strategy": result.config.get("strategy"),
                    "metrics": _clean(result.metrics),
                    "artifact_dir": str(result.artifact_dir),
                    "report": _try_report(result, open_browser=open_reports),
                }
            )

        payload = {
            "ok": all(r.get("ok") for r in results),
            "seeded": seeded,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "runs": results,
            "reports": [r["report"] for r in results if r.get("report")],
            "caveat": (
                "Synthetic data. These numbers describe the generator, not any market."
            ),
            "next": [
                "lab runs list",
                "cp .mcp.json.example .mcp.json  # then ask your agent to run the audit_run prompt on the momentum run with cfg/demo.yaml",
                "lab ui",
            ],
        }

    if json_out:
        _print_json(payload)
        return
    _table(
        "demo",
        ["run", "run_id", "return", "sharpe", "trades", "report"],
        [
            (
                r["name"], r.get("run_id", "-"),
                _fmt((r.get("metrics") or {}).get("total_return")),
                _fmt((r.get("metrics") or {}).get("sharpe")),
                (r.get("metrics") or {}).get("trades"),
                r.get("report") or "-",
            )
            for r in results
        ],
    )
    _console().print(f"[dim]{payload['caveat']}[/dim]")
    _console().print(
        "next: [bold]lab runs list[/bold] · [bold]cp .mcp.json.example .mcp.json[/bold] "
        "and ask your agent to run the [bold]audit_run[/bold] prompt on the momentum run "
        "with cfg/demo.yaml · [bold]lab ui[/bold]"
    )


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
