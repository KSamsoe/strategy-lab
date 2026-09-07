"""Startup reconciliation: the broker is the source of truth, and until the
runner agrees with it, the runner does not trade.

Crash-and-resume is a designed-for path, not an exception. A process that dies
between "submit" and "record the fill" wakes up believing something false about
the book, and the only safe move is to compare against the account itself and
refuse to act on a disagreement. Two failure modes this specifically prevents:

1. **Adopting a position it does not recognise.** A share lot the runner never
   opened -- a manual trade, another strategy, a partial fill it missed -- must
   not silently become collateral for new sizing.
2. **Re-firing orders it already placed.** An open order at the broker with no
   local record is a diff, not noise; the runner stops rather than duplicating.

``acknowledge=True`` is the human override. It says "I have looked, the broker
is right", and it *adopts* the broker's state into the local book rather than
merely muting the warning -- because clearing the alarm while leaving the
accounting wrong would be worse than the alarm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from lab.engine.events import Fill, Order, Position, Side, new_id
from lab.engine.portfolio import Portfolio
from lab.timeutil import to_utc, utcnow

log = logging.getLogger(__name__)

#: Share tolerance. Fractional-share brokers round at the 6th decimal; anything
#: above this is a real disagreement, not float noise.
QTY_EPS = 1e-6

#: Diff kinds, so the console and the alert consumers can switch on a stable
#: string instead of parsing the message.
DIFF_KINDS = (
    "broker_unreachable",
    "unknown_position",
    "missing_position",
    "qty_mismatch",
    "unknown_open_order",
)


@dataclass
class ReconcileResult:
    ok: bool
    diffs: list[dict[str, Any]]
    broker_positions: dict[str, dict[str, float]]
    local_positions: dict[str, dict[str, float]]
    broker_orders: list[Order]
    message: str
    acknowledged: bool = False
    adopted: list[dict[str, Any]] = field(default_factory=list)
    at: datetime = field(default_factory=utcnow)

    @property
    def blocking(self) -> bool:
        """Whether the runner must refuse to trade."""
        return not self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "at": to_utc(self.at).isoformat(),
            "acknowledged": self.acknowledged,
            "n_diffs": len(self.diffs),
            "diffs": self.diffs,
            "broker_positions": self.broker_positions,
            "local_positions": self.local_positions,
            "broker_orders": [o.to_dict() if hasattr(o, "to_dict") else str(o) for o in self.broker_orders],
            "adopted": self.adopted,
            "message": self.message,
        }


def reconcile(
    broker: Any,
    portfolio: Portfolio,
    *,
    acknowledge: bool = False,
    known_order_ids: Iterable[str] | None = None,
    adopt: bool = True,
    at: datetime | None = None,
) -> ReconcileResult:
    """Diff broker state against local state.

    ``known_order_ids`` is whatever the runner recognises as its own -- order
    ids, broker order ids and idem keys, mixed -- so an open order at the broker
    that matches none of them is flagged. Passing ``None`` means "the runner
    knows of no orders", which is the correct assumption on a cold start.
    """
    when = to_utc(at) if at is not None else utcnow()
    known = {str(k) for k in (known_order_ids or ()) if k}
    diffs: list[dict[str, Any]] = []

    broker_positions, err = _safe(broker.positions, "positions")
    if err is not None:
        diffs.append({"kind": "broker_unreachable", "call": "positions", "detail": err})
        broker_positions = {}
    broker_orders, err = _safe(broker.open_orders, "open_orders")
    if err is not None:
        diffs.append({"kind": "broker_unreachable", "call": "open_orders", "detail": err})
        broker_orders = []

    remote = {t.upper(): _pos_row(p) for t, p in dict(broker_positions or {}).items()}
    local = {t.upper(): _pos_row(p) for t, p in dict(portfolio.positions or {}).items()}
    remote = {t: r for t, r in remote.items() if abs(r["qty"]) > QTY_EPS}
    local = {t: r for t, r in local.items() if abs(r["qty"]) > QTY_EPS}

    for ticker in sorted(set(remote) | set(local)):
        r = remote.get(ticker, {}).get("qty", 0.0)
        l = local.get(ticker, {}).get("qty", 0.0)
        if abs(r - l) <= QTY_EPS:
            continue
        if abs(l) <= QTY_EPS:
            kind, detail = "unknown_position", f"broker holds {r:g} {ticker} the runner does not know about"
        elif abs(r) <= QTY_EPS:
            kind, detail = "missing_position", f"runner thinks it holds {l:g} {ticker}; broker holds none"
        else:
            kind, detail = "qty_mismatch", f"{ticker}: broker {r:g} vs local {l:g}"
        diffs.append(
            {"kind": kind, "ticker": ticker, "broker_qty": r, "local_qty": l, "detail": detail}
        )

    for order in list(broker_orders or []):
        ids = {
            str(getattr(order, "id", "") or ""),
            str(getattr(order, "broker_order_id", "") or ""),
            str(getattr(order, "idem_key", "") or ""),
        }
        if ids & known:
            continue
        diffs.append(
            {
                "kind": "unknown_open_order",
                "ticker": str(getattr(order, "ticker", "") or ""),
                "order_id": str(getattr(order, "id", "") or ""),
                "broker_order_id": getattr(order, "broker_order_id", None),
                "detail": f"open order at the broker with no local record: {ids - {''}}",
            }
        )

    unreachable = any(d["kind"] == "broker_unreachable" for d in diffs)
    adopted: list[dict[str, Any]] = []
    # An unreachable broker can never be acknowledged away: we did not see the
    # account, so there is nothing to agree with.
    acknowledged = bool(acknowledge and diffs and not unreachable)
    if acknowledged and adopt:
        adopted = adopt_broker_state(portfolio, broker, remote, at=when)

    ok = (not diffs) or acknowledged
    if not diffs:
        message = f"reconciled: {len(remote)} broker position(s), {len(broker_orders or [])} open order(s), 0 diffs"
    elif acknowledged:
        message = f"acknowledged {len(diffs)} diff(s); adopted broker state as truth"
    else:
        message = (
            f"REFUSING TO TRADE: {len(diffs)} reconciliation diff(s) - "
            + "; ".join(d["detail"] for d in diffs[:4])
        )
    (log.info if ok else log.error)(message)

    return ReconcileResult(
        ok=ok,
        diffs=diffs,
        broker_positions=remote,
        local_positions=local,
        broker_orders=list(broker_orders or []),
        message=message,
        acknowledged=acknowledged,
        adopted=adopted,
        at=when,
    )


def adopt_broker_state(
    portfolio: Portfolio,
    broker: Any,
    remote: Mapping[str, Mapping[str, float]] | None = None,
    *,
    at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Rewrite the local book to match the broker. Returns what changed.

    Share deltas go through ``apply_fill`` so the position, its basis and the
    round-trip ledger stay internally consistent; cash is then taken verbatim
    from the account, because the synthetic fills debited a local cash balance
    that was already wrong. ``Portfolio`` deliberately exposes no cash setter --
    nothing in a backtest may rewrite the book -- and adopting a broker's
    balance is the single legitimate exception, which is why it lives here and
    nowhere else.
    """
    when = to_utc(at) if at is not None else utcnow()
    if remote is None:
        positions, err = _safe(broker.positions, "positions")
        if err is not None:
            return []
        remote = {t.upper(): _pos_row(p) for t, p in dict(positions or {}).items()}

    changed: list[dict[str, Any]] = []
    tickers = set(remote) | {t.upper() for t in portfolio.positions}
    for ticker in sorted(tickers):
        target = float(remote.get(ticker, {}).get("qty", 0.0))
        current = float(portfolio.position(ticker).qty)
        delta = target - current
        if abs(delta) <= QTY_EPS:
            continue
        row = remote.get(ticker, {})
        price = float(row.get("avg_price") or row.get("last_price") or portfolio.position(ticker).avg_price or 0.0)
        portfolio.apply_fill(
            Fill(
                id=new_id("f_"),
                order_id="reconcile",
                ticker=ticker,
                side=Side.BUY if delta > 0 else Side.SELL,
                qty=abs(delta),
                price=price,
                at=when,
                tag="reconcile",
            )
        )
        changed.append({"ticker": ticker, "from_qty": current, "to_qty": target, "price": price})

    account, err = _safe(broker.account, "account")
    if err is None and isinstance(account, Mapping) and account.get("cash") is not None:
        try:
            cash = float(account["cash"])
        except (TypeError, ValueError):
            cash = None
        if cash is not None and abs(cash - portfolio.cash) > 1e-6:
            changed.append({"ticker": None, "cash_from": portfolio.cash, "cash_to": cash})
            portfolio._cash = cash  # noqa: SLF001 - see docstring
    return changed


def _safe(fn: Any, label: str) -> tuple[Any, str | None]:
    """Call a broker method, turning a failure into state rather than a raise.

    A broker outage must produce a *blocking diff*, not a traceback: the runner
    still needs to journal, heartbeat and report why it will not trade.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.error("broker.%s() failed during reconciliation: %s", label, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _pos_row(pos: Any) -> dict[str, float]:
    if isinstance(pos, Mapping):
        get = pos.get
    else:
        get = lambda k, d=None: getattr(pos, k, d)  # noqa: E731
    qty = float(get("qty", 0.0) or 0.0)
    return {
        "qty": qty,
        "avg_price": float(get("avg_price", 0.0) or 0.0),
        "last_price": float(get("last_price", 0.0) or 0.0),
    }


def positions_of(broker: Any) -> dict[str, Position]:
    """Broker positions as ``Position`` objects, tolerating dict-shaped rows."""
    out: dict[str, Position] = {}
    for ticker, pos in dict(broker.positions() or {}).items():
        if isinstance(pos, Position):
            out[ticker.upper()] = pos
            continue
        row = _pos_row(pos)
        out[ticker.upper()] = Position(
            ticker=ticker.upper(),
            qty=row["qty"],
            avg_price=row["avg_price"],
            last_price=row["last_price"] or row["avg_price"],
        )
    return out
