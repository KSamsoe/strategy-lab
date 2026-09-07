"""The live half: fleet state, the event tape, the freshness strip -- and the
only three mutations the console is allowed to perform.

Everything read here comes out of the journals. The API never asks a broker for
positions and never pokes the runner's process: a console bug therefore cannot
crash a live run, and when the runner *has* crashed these routes keep answering,
which is exactly when you need them most.

**The asymmetry.** Pause, cancel-open-orders and the kill switch exist because
each of them can only reduce exposure. Start, resume-after-a-breaker and
raise-a-limit are absent by construction, not by oversight: they stay CLI
actions behind the manual checklist, so a stolen laptop or a compromised browser
tab cannot make the book bigger. Adding a route here that can increase exposure
breaks the property the whole surface is designed around --
``tests/test_api.py::test_no_route_can_increase_exposure`` enforces it.

Two of the three controls are *requests*, not actions. Pausing flips a flag the
runner reads and journals the request; cancelling orders can only be journaled,
because the process holding the broker session is the one thing allowed to talk
to the broker. Only the kill switch acts directly, since its whole point is to
work when nothing else does -- it is a file on disk that every component checks.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from lab.api.models import (
    AdapterState,
    BreakerState,
    ControlResult,
    EventPage,
    EventRow,
    Heartbeat,
    KillRequest,
    KillSwitchState,
    LiveHealth,
    LiveStrategy,
    LiveStrategyList,
    QuotaState,
    StrategyControlRequest,
)
from lab.config import get_settings
from lab.engine.events import EventKind, JournalEvent, event
from lab.registry.db import connect, journal_path, transaction
from lab.registry.journal import EventJournal
from lab.timeutil import to_utc, utcnow

router = APIRouter(prefix="/api", tags=["live"])

#: Source recorded on every event the console originates, so the tape can show
#: "a human clicked this" apart from "the runner did this".
CONSOLE_SOURCE = "console"

#: Default staleness threshold for the health strip. The journal deliberately
#: refuses to define one (a 2s broker heartbeat and a 30s runner heartbeat are
#: not comparable), so the caller may override it per request.
DEFAULT_STALE_AFTER_S = 120.0


def _journal_con() -> sqlite3.Connection:
    return connect(journal_path())


def _loads(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback


# --- read ----------------------------------------------------------------------


@router.get("/live/strategies", response_model=LiveStrategyList)
def live_strategies() -> LiveStrategyList:
    rows = _journal_con().execute("SELECT * FROM live_strategies ORDER BY strategy").fetchall()
    items = [_live_strategy(r) for r in rows]
    return LiveStrategyList(count=len(items), strategies=items)


def _live_strategy(row: sqlite3.Row) -> LiveStrategy:
    r = dict(row)
    return LiveStrategy(
        strategy=r["strategy"],
        run_id=r.get("run_id"),
        kind=r.get("kind") or "paper",
        status=r.get("status") or "stopped",
        paused=bool(r.get("paused") or 0),
        pid=r.get("pid"),
        host=r.get("host") or "",
        started_at=r.get("started_at") or None,
        updated_at=r.get("updated_at") or None,
        next_fire=r.get("next_fire") or None,
        notes=r.get("notes") or "",
        config=_loads(r.get("config"), {}),
    )


@router.get("/live/events", response_model=EventPage)
def live_events(
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    run_id: str | None = None,
    strategy: str | None = None,
    kinds: str | None = Query(default=None, description="comma-separated event kinds"),
) -> EventPage:
    journal = EventJournal()
    wanted = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    rows = journal.tail(since, limit=limit, run_id=run_id, strategy=strategy, kinds=wanted)
    return EventPage(
        since=since,
        # The head of the log, not of this page: the client needs it to know
        # whether it is caught up before it opens the WebSocket at that cursor.
        latest_seq=journal.latest_seq(),
        count=len(rows),
        events=[EventRow.model_validate(r) for r in rows],
    )


@router.get("/live/health", response_model=LiveHealth)
def live_health(
    stale_after_s: float = Query(default=DEFAULT_STALE_AFTER_S, gt=0),
) -> LiveHealth:
    now = utcnow()
    journal = EventJournal()

    beats: list[Heartbeat] = []
    for source, hb in sorted(journal.last_heartbeats().items()):
        # last_heartbeats() already measures age at call time; recomputing here
        # would be the same number twice. Staleness is a display decision, so it
        # is applied here rather than baked into the journal.
        age = float(hb["age_s"])
        beats.append(
            Heartbeat(
                source=source,
                at=hb["at"],
                age_s=round(age, 3),
                stale=age > stale_after_s,
                seq=int(hb.get("seq") or 0),
                meta=hb.get("meta") or {},
            )
        )

    engaged, reason = get_settings().kill_switch_engaged()
    strategies = int(
        _journal_con().execute("SELECT COUNT(*) FROM live_strategies").fetchone()[0]
    )
    return LiveHealth(
        now=now,
        stale_after_s=stale_after_s,
        adapters=_adapter_states(),
        heartbeats=beats,
        quota=_quota_state(now),
        breaker=_breaker_state(journal, now),
        kill_switch=KillSwitchState(engaged=engaged, reason=reason),
        strategies=strategies,
        latest_seq=journal.latest_seq(),
    )


def _adapter_states() -> list[AdapterState]:
    """Adapter availability, never adapter *use*: ``info()`` is documented as
    credential- and cache-only, so rendering the health strip cannot spend a
    GovGreed call or hit the network."""
    try:
        from lab.adapters.base import list_adapters
    except Exception as exc:  # a broken adapter package must not 500 the strip
        return [AdapterState(name="adapters", available=False, reason=f"{type(exc).__name__}: {exc}")]
    out: list[AdapterState] = []
    for info in list_adapters():
        try:
            out.append(AdapterState.model_validate(info.to_dict()))
        except Exception as exc:
            out.append(AdapterState(name=getattr(info, "name", "?"), available=False, reason=str(exc)))
    return out


def _quota_state(now: datetime) -> QuotaState:
    """Quota comes from the log first and the adapter second.

    The log is the honest source: it is what the client actually observed in the
    ``X-RateLimit-*`` headers on its last call, and it survives the process that
    made the call. A freshly constructed adapter has made zero calls and would
    report an empty quota, which on a health strip reads as "plenty left".
    """
    row = (
        connect()
        .execute(
            "SELECT * FROM quota_log WHERE source = ? ORDER BY at DESC, id DESC LIMIT 1",
            ("govgreed",),
        )
        .fetchone()
    )
    if row is not None:
        r = dict(row)
        at = to_utc(r["at"])
        return QuotaState(
            source=r["source"],
            used=r.get("used"),
            limit=r.get("quota_limit"),
            remaining=r.get("remaining"),
            tier=r.get("tier"),
            reset_at=r.get("reset_at") or None,
            at=at,
            age_s=round((now - at).total_seconds(), 3),
            origin="quota_log",
        )

    try:
        from lab.adapters.base import get_adapter

        info = get_adapter("govgreed").info()
    except Exception:
        return QuotaState(origin="unknown")
    used, limit = info.quota_used, info.quota_limit
    remaining = None if (used is None or limit is None) else max(limit - used, 0)
    return QuotaState(
        used=used,
        limit=limit,
        remaining=remaining,
        tier=info.quota_tier,
        at=info.last_call,
        age_s=None if info.last_call is None else round((now - to_utc(info.last_call)).total_seconds(), 3),
        origin="adapter",
    )


def _breaker_state(journal: EventJournal, now: datetime) -> BreakerState:
    """Derived from the last ``breaker`` event.

    Convention, so the live runner and this reader agree: the payload carries
    ``{"tripped": bool, "reason": str}``. A breaker event *without* ``tripped``
    is read as a trip, because the fail-safe direction for a missing field is
    "assume the brake is on" -- and re-arming is a CLI action that must be
    journaled explicitly to show up here as armed again.
    """
    # tail() reads forward from a cursor; the newest row wants seq DESC, which
    # the ix_events_kind index is built for.
    row = journal.con.execute(
        "SELECT * FROM events WHERE kind = ? ORDER BY seq DESC LIMIT 1",
        (EventKind.BREAKER.value,),
    ).fetchone()
    if row is None:
        return BreakerState(tripped=False)
    payload = _loads(row["payload"], {}) or {}
    at = to_utc(row["at"])
    return BreakerState(
        tripped=bool(payload.get("tripped", True)),
        reason=str(payload.get("reason") or row["message"] or ""),
        strategy=row["strategy"],
        at=at,
        age_s=round((now - at).total_seconds(), 3),
    )


# --- the three safer-only controls ---------------------------------------------


def _record(
    action: str,
    *,
    message: str,
    strategy: str | None = None,
    payload: dict[str, Any],
) -> tuple[int, datetime]:
    """Journal a control action. Every one of them is written down, always:
    an undocumented pause at 09:31 is indistinguishable from a crash at 09:31."""
    at = utcnow()
    ev: JournalEvent = event(
        EventKind.LOG,
        CONSOLE_SOURCE,
        at=at,
        strategy=strategy,
        message=message,
        payload={"control": action, "actor": CONSOLE_SOURCE} | payload,
    )
    return EventJournal().append(ev), at


def _live_row_or_404(strategy: str) -> sqlite3.Row:
    row = _journal_con().execute(
        "SELECT * FROM live_strategies WHERE strategy = ?", (strategy,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no live strategy named {strategy!r}")
    return row


@router.post("/control/pause", response_model=ControlResult, tags=["control"])
def pause_strategy(body: StrategyControlRequest) -> ControlResult:
    """Stop the strategy from acting further. Resuming is deliberately not an
    API operation -- see the module docstring."""
    row = _live_row_or_404(body.strategy)
    already = bool(row["paused"])
    with transaction(_journal_con()) as con:
        con.execute(
            "UPDATE live_strategies SET paused = 1, updated_at = ? WHERE strategy = ?",
            (utcnow().isoformat(), body.strategy),
        )
    seq, at = _record(
        "pause",
        message=f"pause {body.strategy}",
        strategy=body.strategy,
        payload={"reason": body.reason, "was_paused": already},
    )
    return ControlResult(
        ok=True,
        action="pause",
        target=body.strategy,
        applied=not already,
        seq=seq,
        at=at,
        message="already paused" if already else "paused; resume is a CLI action",
    )


@router.post("/control/cancel_orders", response_model=ControlResult, tags=["control"])
def cancel_orders(body: StrategyControlRequest) -> ControlResult:
    """Ask the runner to cancel its open orders.

    ``applied`` is false on purpose: the API cannot and must not reach the
    broker, so this writes a request the runner picks up off its own journal
    tail. Cancelling can only reduce exposure, which is why it is allowed here
    at all.
    """
    _live_row_or_404(body.strategy)
    seq, at = _record(
        "cancel_orders",
        message=f"cancel open orders for {body.strategy}",
        strategy=body.strategy,
        payload={"reason": body.reason},
    )
    return ControlResult(
        ok=True,
        action="cancel_orders",
        target=body.strategy,
        applied=False,
        seq=seq,
        at=at,
        message="cancellation requested; the runner owns the broker session",
    )


@router.post("/control/kill", response_model=ControlResult, tags=["control"])
def kill(body: KillRequest) -> ControlResult:
    """Engage the global kill switch: a sentinel file every component checks.

    A file rather than a signal or an RPC because it must work when the runner
    is wedged, when the API is the only thing still up, and across restarts.
    Releasing it is a CLI action.
    """
    reason = body.reason.strip() or "engaged from the console"
    try:
        path = _engage(reason)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not engage kill switch: {exc}") from exc
    seq, at = _record("kill", message=f"kill switch engaged: {reason}", payload={"reason": reason, "path": str(path)})
    return ControlResult(
        ok=True,
        action="kill",
        target=None,
        applied=True,
        seq=seq,
        at=at,
        message=f"kill switch engaged at {path}; release is a CLI action",
    )


def _engage(reason: str) -> Any:
    """Prefer ``lab.live.killswitch`` when it is installed; fall back to writing
    the sentinel ourselves. The fallback exists because the kill switch is the
    one control that must not depend on another module being importable."""
    try:
        from lab.live import killswitch
    except Exception:
        killswitch = None  # type: ignore[assignment]
    if killswitch is not None and hasattr(killswitch, "engage"):
        return killswitch.engage(reason)

    path = get_settings().kill_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"reason": reason, "at": utcnow().isoformat(), "by": CONSOLE_SOURCE}, indent=2),
        encoding="utf-8",
    )
    return path
