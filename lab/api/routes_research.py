"""Starting and stopping research sessions from the console.

This is the one place the console does something other than read, pause or stop,
so the line it sits on is worth stating precisely.

The console's rule is that it can only make the system **safer, never riskier**,
and the risk it names is *market exposure*. A research session cannot touch a
broker, place an order, or move a position; it runs backtests. So it does not
cross that line.

It does two other things though, and neither is nothing: it spends model budget,
and in freeform mode it writes Python and imports it. A compromised browser tab
that could begin one unasked would be able to burn a subscription and cause
arbitrary generated code to execute locally. That is why starting is **off by
default** and needs ``LAB_UI_ALLOW_RESEARCH=1`` (or ``lab ui --allow-research``),
rather than being folded into the existing controls.

Two further limits keep the surface narrow:

* **The console cannot author a backtest config.** It picks one that already
  exists in ``cfg/`` by name. Letting a form compose ``limits:`` would be
  "raise a limit from the UI" wearing a different hat.
* **Every resource bound is mandatory and clamped.** A session started here
  always has a wall clock, a spend ceiling and a call cap.

Stopping is unconditional, needs no opt-in, and is available even when starting
is disabled -- it only ever reduces what is running.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from lab.api.models import (
    ResearchLaunchOptions,
    ResearchStartRequest,
    ResearchStartResult,
    ResearchStopResult,
)
from lab.config import get_settings
from lab.timeutil import utcnow

router = APIRouter(prefix="/api", tags=["control"])

#: Hard ceilings the console cannot exceed whatever the form says. A misclick
#: should cost minutes and cents, not an afternoon and a subscription.
MAX_MINUTES = 180.0
MAX_BUDGET = 25.0
MAX_CALLS = 400
MAX_EXPERIMENTS = 200
MAX_BRIEF_CHARS = 8_000


def _session_dir(session_id: str) -> Path:
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail=f"unsafe session id: {session_id!r}")
    return get_settings().paths.runs / session_id


def _record(kind: str, message: str, **payload: Any) -> tuple[int | None, datetime]:
    at = utcnow()
    try:
        from lab.engine.events import EventKind, event
        from lab.registry.journal import EventJournal

        seq = EventJournal().append(
            event(EventKind(kind), "console", at=at, strategy="research",
                  message=message, payload=payload)
        )
        return seq, at
    except Exception:
        return None, at


@router.get("/control/research/options", response_model=ResearchLaunchOptions)
def research_options() -> ResearchLaunchOptions:
    """What the launch form may offer: enabled?, configs, providers, ceilings."""
    settings = get_settings()
    configs = sorted(p.name for p in settings.paths.cfg.glob("*.yaml"))

    providers: list[dict[str, Any]] = []
    try:
        from lab.agent.providers import list_providers

        providers = [p.to_dict() for p in list_providers()]
    except Exception:
        providers = []

    engaged, reason = settings.kill_switch_engaged()
    return ResearchLaunchOptions(
        enabled=bool(settings.ui_allow_research),
        reason=(
            ""
            if settings.ui_allow_research
            else "starting research from the console is off; set LAB_UI_ALLOW_RESEARCH=1 "
                 "or run `lab ui --allow-research`"
        ),
        configs=configs,
        providers=providers,
        default_model=settings.agent_model,
        default_provider=settings.agent_provider,
        kill_switch_engaged=engaged,
        kill_switch_reason=reason,
        limits={
            "max_minutes": MAX_MINUTES,
            "max_budget_usd": MAX_BUDGET,
            "max_calls": MAX_CALLS,
            "max_experiments": MAX_EXPERIMENTS,
        },
    )


@router.post("/control/research/start", response_model=ResearchStartResult)
def start_research(body: ResearchStartRequest) -> ResearchStartResult:
    settings = get_settings()
    if not settings.ui_allow_research:
        raise HTTPException(
            status_code=403,
            detail=(
                "starting research from the console is disabled. It spends model "
                "budget and can execute model-authored code, so it is opt-in: set "
                "LAB_UI_ALLOW_RESEARCH=1 or run `lab ui --allow-research`."
            ),
        )
    engaged, reason = settings.kill_switch_engaged()
    if engaged:
        raise HTTPException(status_code=409, detail=f"kill switch is engaged: {reason}")

    # The console picks an existing config; it cannot author one. Resolved by
    # name inside cfg/ so a path cannot point somewhere else.
    name = Path(body.config).name
    config = settings.paths.cfg / name
    if not config.exists():
        raise HTTPException(status_code=400, detail=f"no config named {name!r} in cfg/")

    brief = (body.brief or "").strip()[:MAX_BRIEF_CHARS]
    if not brief:
        raise HTTPException(
            status_code=400,
            detail="a brief is required: it is what the session optimises for and "
                   "what tells it when to stop",
        )

    minutes = min(max(float(body.minutes), 1.0), MAX_MINUTES)
    budget = min(max(float(body.budget_usd), 0.01), MAX_BUDGET)
    calls = min(max(int(body.max_calls), 1), MAX_CALLS)
    experiments = min(max(int(body.max_experiments), 1), MAX_EXPERIMENTS)

    session_id = f"research-{utcnow().strftime('%Y%m%dT%H%M%S')}-console"
    directory = _session_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable, "-m", "lab.cli", "agent", "research",
        "--config", str(config),
        "--brief", brief,
        "--minutes", str(minutes),
        "--budget", str(budget),
        "--max-calls", str(calls),
        "--max-experiments", str(experiments),
        "--metric", body.metric or "oos_sharpe",
        "--periods", str(int(body.periods)),
        "--neighbourhood-runs", str(int(body.neighbourhood_runs)),
        "--session-id", session_id,
        "--json",
    ]
    if body.model:
        argv += ["--model", body.model]
    if body.provider:
        argv += ["--provider", body.provider]

    # A detached subprocess running the ordinary CLI, not an in-process thread:
    # the session then fails, resumes and is inspected exactly the same way
    # whether a human or the console started it, and an API restart does not
    # take the work with it.
    log = (directory / "console.log").open("ab")
    creationflags = 0
    if sys.platform == "win32":  # pragma: no cover - platform branch
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    try:
        proc = subprocess.Popen(
            argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            cwd=str(settings.paths.root), creationflags=creationflags,
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not start research: {exc}") from exc

    seq, at = _record(
        "run_start",
        f"research session started from the console: {brief[:120]}",
        session_id=session_id, config=name, minutes=minutes, budget_usd=budget,
        max_calls=calls, model=body.model, provider=body.provider,
    )
    return ResearchStartResult(
        ok=True, session_id=session_id, pid=proc.pid, config=name,
        minutes=minutes, budget_usd=budget, max_calls=calls,
        max_experiments=experiments, seq=seq, at=at,
        message=f"started; it will stop on its own or at {minutes:g} minutes",
    )


@router.post("/control/research/stop", response_model=ResearchStopResult)
def stop_research(body: ResearchStartRequest | dict[str, Any] | None = None,
                  session_id: str = "") -> ResearchStopResult:
    """Ask a running session to stop at its next turn.

    Always available -- no opt-in, and it works with the kill switch engaged,
    because it only ever reduces what is running. A sentinel file rather than a
    signal, so it reaches a session the API did not start.
    """
    sid = session_id or (getattr(body, "session_id", None) if body else None) or (
        body.get("session_id") if isinstance(body, dict) else None
    )
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    directory = _session_dir(str(sid))
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"no research session: {sid}")

    sentinel = directory / "STOP"
    sentinel.write_text(f"stopped from the console at {utcnow().isoformat()}\n", encoding="utf-8")
    seq, at = _record("log", f"research session asked to stop: {sid}", session_id=str(sid))
    return ResearchStopResult(
        ok=True, session_id=str(sid), seq=seq, at=at,
        message="it will stop after the current turn; the session stays resumable",
    )
