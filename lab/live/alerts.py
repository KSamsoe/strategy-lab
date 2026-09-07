"""Alerts: one call that fans out to the log, the event journal and a webhook.

The journal is the system of record and the webhook is *just another consumer*
of it -- same payload, same vocabulary, no second event schema to keep in sync.
That ordering matters when something goes wrong at 09:35: the journal write
happens first, so a dead webhook endpoint costs a notification, never a record.

Every failure in here is swallowed and logged. An alert that raises would take
down the runner it was trying to tell you about, which is strictly worse than no
alert at all -- so the only thing this module is allowed to do on failure is
complain to the log.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping

from lab.config import get_settings
from lab.engine.events import EventKind, event
from lab.timeutil import to_utc, utcnow

log = logging.getLogger(__name__)

#: Alert kind -> journal event kind. Unknown kinds land on ``LOG`` rather than
#: being dropped, so a new alert site is never silently invisible.
KINDS: dict[str, EventKind] = {
    "fill": EventKind.FILL,
    "order": EventKind.ORDER,
    "reject": EventKind.REJECT,
    "gate_block": EventKind.GATE_BLOCK,
    "breaker": EventKind.BREAKER,
    "kill_switch": EventKind.BREAKER,
    "reconcile": EventKind.RECONCILE,
    "heartbeat": EventKind.HEARTBEAT,
    "error": EventKind.ERROR,
    "log": EventKind.LOG,
}

#: Log level per alert kind. Fills are routine; rejections and breakers are not.
_LEVELS: dict[str, int] = {
    "error": logging.ERROR,
    "reject": logging.ERROR,
    "breaker": logging.ERROR,
    "kill_switch": logging.WARNING,
    "gate_block": logging.WARNING,
    "reconcile": logging.WARNING,
}

#: Field names ``alert`` lifts out of ``**fields`` onto the event itself. Kept
#: as kwargs rather than explicit parameters so the contract signature stays
#: ``alert(kind, message, **fields)``.
_RESERVED = ("run_id", "strategy", "ticker", "at", "source", "journal", "webhook", "timeout")

#: Webhook timeout. Short on purpose: the runner is blocked while this posts,
#: and a slow notifier must never delay an order.
DEFAULT_TIMEOUT = 5.0


def webhook_url() -> str | None:
    return get_settings().alert_webhook


def alert(kind: str, message: str, **fields: Any) -> None:
    """Log, journal and (if configured) POST one alert.

    Reserved kwargs: ``run_id``, ``strategy``, ``ticker``, ``at``, ``source``,
    plus ``journal``/``webhook``/``timeout`` for injection in tests. Everything
    else becomes the event payload.
    """
    kind = str(kind or "log").strip().lower()
    opts = {name: fields.pop(name, None) for name in _RESERVED}
    at = to_utc(opts["at"]) if opts["at"] is not None else utcnow()
    source = str(opts["source"] or "alerts")
    payload = {k: _plain(v) for k, v in fields.items()}

    log.log(_LEVELS.get(kind, logging.INFO), "[%s] %s %s", kind, message, payload or "")

    ev = event(
        KINDS.get(kind, EventKind.LOG),
        source,
        at=at,
        message=str(message),
        run_id=opts["run_id"],
        strategy=opts["strategy"],
        ticker=opts["ticker"],
        payload=payload | {"alert_kind": kind},
    )
    _journal(ev, opts["journal"])
    _post(
        {
            "kind": kind,
            "message": str(message),
            "at": at.isoformat(),
            "source": source,
            "run_id": opts["run_id"],
            "strategy": opts["strategy"],
            "ticker": opts["ticker"],
            "fields": payload,
        },
        url=opts["webhook"],
        timeout=opts["timeout"],
    )


# --- typed shorthands ---------------------------------------------------------


def alert_fill(fill: Any, **fields: Any) -> None:
    side = getattr(getattr(fill, "side", None), "value", "")
    sign = "+" if side == "buy" else "-"
    fields.setdefault("source", "broker")
    fields.setdefault("ticker", fill.ticker)
    fields.setdefault("at", getattr(fill, "at", None))
    fields.setdefault("fill", _plain(fill))
    alert("fill", f"{fill.ticker} {sign}{fill.qty:g} @ {fill.price:.2f}", **fields)


def alert_order(order: Any, **fields: Any) -> None:
    side = getattr(getattr(order, "side", None), "value", "")
    fields.setdefault("source", "broker")
    fields.setdefault("ticker", order.ticker)
    fields.setdefault("order", _plain(order))
    alert("order", f"{side} {order.qty:g} {order.ticker}", **fields)


def alert_reject(order: Any, reason: str, **fields: Any) -> None:
    fields.setdefault("source", "broker")
    fields.setdefault("ticker", getattr(order, "ticker", None))
    fields.setdefault("order", _plain(order))
    fields.setdefault("reason", str(reason))
    alert("reject", f"rejected {getattr(order, 'ticker', '?')}: {reason}", **fields)


def alert_gate_block(verdict: Any, **fields: Any) -> None:
    action = getattr(getattr(verdict, "action", None), "value", "blocked")
    fields.setdefault("source", "gate")
    fields.setdefault("ticker", verdict.ticker)
    fields.setdefault("verdict", _plain(verdict))
    fields.setdefault("rule", verdict.rule)
    alert("gate_block", f"{action} {verdict.ticker} - {verdict.rule}: {verdict.detail}", **fields)


def alert_breaker(reason: str, **fields: Any) -> None:
    fields.setdefault("source", "gate")
    fields.setdefault("reason", str(reason))
    alert("breaker", f"circuit breaker tripped: {reason}", **fields)


def alert_reconcile(result: Any, **fields: Any) -> None:
    fields.setdefault("source", "runner")
    fields.setdefault("result", _plain(result))
    alert("reconcile", getattr(result, "message", "") or "reconciliation", **fields)


def alert_error(exc: BaseException | str, **fields: Any) -> None:
    if isinstance(exc, BaseException):
        message = f"{type(exc).__name__}: {exc}"
        fields.setdefault("exc_type", type(exc).__name__)
    else:
        message = str(exc)
    fields.setdefault("source", "runner")
    alert("error", message, **fields)


# --- sinks --------------------------------------------------------------------


def _journal(ev: Any, journal: Any = None) -> None:
    try:
        if journal is None:
            from lab.registry.journal import EventJournal

            journal = EventJournal()
        journal.append(ev)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        log.warning("alert journal write failed: %s", exc)


def _post(body: Mapping[str, Any], *, url: str | None = None, timeout: float | None = None) -> None:
    target = url or webhook_url()
    if not target:
        return
    try:
        import httpx

        httpx.post(target, json=dict(body), timeout=float(timeout or DEFAULT_TIMEOUT))
    except Exception as exc:  # noqa: BLE001 - see module docstring
        log.warning("alert webhook POST to %s failed: %s", target, exc)


def _plain(value: Any) -> Any:
    """Best-effort JSON-able form. Dataclass-ish lab objects expose ``to_dict``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return to_utc(value).isoformat()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return str(value)
