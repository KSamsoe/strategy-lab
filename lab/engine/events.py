"""Engine value types: what a strategy emits, what the gate says about it, what
the broker does with it, and the decision record that ties the three together.

These types are the shared vocabulary of the backtester, the live runner, the
journals, the API and the console. One event schema, two sources -- so every UI
screen is a renderer over these objects and cannot tell replay from reality.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from lab.timeutil import to_utc


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class GateAction(str, Enum):
    PASS = "pass"
    CLIP = "clipped"
    BLOCK = "blocked"


class EventKind(str, Enum):
    DECISION = "decision"
    ORDER = "order"
    FILL = "fill"
    REJECT = "reject"
    GATE_BLOCK = "gate_block"
    BREAKER = "breaker"
    RECONCILE = "reconcile"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    LOG = "log"
    RUN_START = "run_start"
    RUN_END = "run_end"


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}{raw}" if prefix else raw


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_utc(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, sort_keys=True)


# --- what a strategy emits ---------------------------------------------------


@dataclass(slots=True)
class Intent(_Serializable):
    """A strategy's *desire*, never an order.

    Strategies express target weights; translating a target into a delta, into
    shares, into an order -- and every safety check on the way -- happens
    outside strategy code. That separation is what lets the same file run in
    the backtester and against a live broker unchanged.
    """

    ticker: str
    target_pct: float
    tag: str = ""
    reason: str = ""
    limit_price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.target_pct = float(self.target_pct)


# --- what the gate says about it ---------------------------------------------


@dataclass(slots=True)
class GateVerdict(_Serializable):
    """Per-intent ruling from the risk gate: pass, clipped, or blocked, plus the
    rule that fired. Always recorded, even on pass, so the decision journal can
    answer "why is this position this size" without guessing.
    """

    ticker: str
    action: GateAction
    requested_pct: float
    approved_pct: float
    rule: str | None = None
    detail: str = ""

    @property
    def blocked(self) -> bool:
        return self.action is GateAction.BLOCK

    @classmethod
    def passed(cls, intent: "Intent") -> "GateVerdict":
        return cls(
            ticker=intent.ticker,
            action=GateAction.PASS,
            requested_pct=intent.target_pct,
            approved_pct=intent.target_pct,
        )


# --- what the broker does with it --------------------------------------------


@dataclass(slots=True)
class Order(_Serializable):
    """An order the engine actually intends to place.

    ``idem_key`` is ``(strategy, session_date, ticker, side)``; brokers must
    treat a repeat of the same key as a no-op so a crashed-and-rerun job cannot
    double-fire.
    """

    id: str
    ticker: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    created_at: datetime | None = None
    strategy: str = ""
    tag: str = ""
    idem_key: str = ""
    status: OrderStatus = OrderStatus.NEW
    broker_order_id: str | None = None
    decision_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.side = Side(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        self.qty = float(self.qty)

    @staticmethod
    def make_idem_key(strategy: str, session: Any, ticker: str, side: Side | str) -> str:
        side_s = side.value if isinstance(side, Side) else str(side)
        return f"{strategy}|{session}|{ticker.upper()}|{side_s}"


@dataclass(slots=True)
class Fill(_Serializable):
    id: str
    order_id: str
    ticker: str
    side: Side
    qty: float
    price: float
    at: datetime
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = ""
    tag: str = ""
    decision_id: str | None = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.side = Side(self.side)
        self.qty = float(self.qty)
        self.price = float(self.price)

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side is Side.BUY else -self.qty

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(slots=True)
class Position(_Serializable):
    ticker: str
    qty: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    opened_at: datetime | None = None
    realized_pnl: float = 0.0
    tag: str = ""

    @property
    def market_value(self) -> float:
        return self.qty * self.last_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pct(self) -> float:
        basis = abs(self.cost_basis)
        return 0.0 if basis == 0 else self.unrealized_pnl / basis

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-9


@dataclass(slots=True)
class Trade(_Serializable):
    """A round trip, assembled from fills when a position returns to flat."""

    ticker: str
    side: str
    qty: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    commission: float = 0.0
    tag: str = ""
    exit_reason: str = ""


# --- the decision record ------------------------------------------------------


@dataclass(slots=True)
class Decision(_Serializable):
    """One ``on_bar`` call, captured whole.

    The *input tape* is every value the ``Context`` actually served -- history
    slices summarized, indicator values verbatim, signals delivered with their
    ``knowledge_time``. Because the Context mediates all data access, capture is
    a wrapper, not a strategy change. This record is what turns "why does this
    position exist" from an investigation into a lookup.
    """

    id: str
    run_id: str
    strategy: str
    at: datetime
    inputs: dict[str, Any] = field(default_factory=dict)
    intents: list[Intent] = field(default_factory=list)
    verdicts: list[GateVerdict] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)
    portfolio: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    agent: dict[str, Any] | None = None
    duration_ms: float = 0.0

    def summary(self) -> str:
        """The collapsed one-line form the decision tape renders."""
        n_sig = len(self.inputs.get("signals", []) or [])
        bits: list[str] = []
        if n_sig:
            plural = "s" if n_sig != 1 else ""
            bits.append(f"saw {n_sig} signal{plural}")
        for intent in self.intents:
            bits.append(f"intent {intent.ticker} {intent.target_pct:+.1%}")
        for v in self.verdicts:
            if v.action is GateAction.CLIP:
                bits.append(f"gate: clipped to {v.approved_pct:.1%} ({v.rule})")
            elif v.action is GateAction.BLOCK:
                bits.append(f"gate: blocked ({v.rule})")
        if self.order_ids:
            bits.append(f"{len(self.order_ids)} order(s)")
        return " · ".join(bits) if bits else "no action"


@dataclass(slots=True)
class JournalEvent(_Serializable):
    """The append-only unit of the event journal, and the WebSocket payload.

    Backtest and live emit the *same* events; the only difference is which
    clock stamped them. That is what lets one set of UI components render
    replay and reality indistinguishably.
    """

    id: str
    at: datetime
    kind: EventKind
    source: str
    run_id: str | None = None
    strategy: str | None = None
    ticker: str | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int | None = None

    def __post_init__(self) -> None:
        self.kind = EventKind(self.kind)


def event(
    kind: EventKind | str,
    source: str,
    *,
    at: datetime,
    message: str = "",
    run_id: str | None = None,
    strategy: str | None = None,
    ticker: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> JournalEvent:
    return JournalEvent(
        id=new_id("e_"),
        at=to_utc(at),
        kind=EventKind(kind),
        source=source,
        run_id=run_id,
        strategy=strategy,
        ticker=ticker,
        message=message,
        payload=dict(payload or {}),
    )
