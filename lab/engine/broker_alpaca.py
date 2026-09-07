"""The Alpaca broker adapter: the same ``Broker`` surface as ``SimBroker``.

Swapping this in for the sim broker (and a wall clock for the sim clock) is the
whole of "the strategy now runs live", which is why every method here maps onto
the lab's own ``Order``/``Fill``/``Position`` types rather than leaking
``alpaca-py`` objects upward.

**The paper endpoint is the default and it is the only path this repo tests.**
Pointing at real capital requires two explicit, deliberate acts -- passing
``allow_live=True`` *and* setting ``LAB_ALLOW_LIVE_TRADING=1`` in the
environment -- because promotion to live money is a human checklist decision,
not something a config typo should be able to do. Without both, a non-paper
broker constructs fine but reports itself unavailable and refuses every
mutating call.

Two shapes of failure are state rather than exceptions: ``alpaca-py`` may be
absent and the API keys may be absent. Either way construction succeeds and
``available()`` returns ``(False, reason)``, so ``lab status`` prints a table
instead of a traceback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from datetime import datetime
from typing import Any, Iterable

from lab.config import get_settings
from lab.engine.events import Fill, Order, OrderStatus, OrderType, Position, Side, new_id
from lab.timeutil import to_utc, utcnow

log = logging.getLogger(__name__)

#: The environment half of the live-trading opt-in.
LIVE_ENV_FLAG = "LAB_ALLOW_LIVE_TRADING"

#: Alpaca's order states mapped onto ours. Anything unlisted is treated as
#: working, which is the conservative reading: an unknown state is not a
#: terminal state, so the runner keeps watching it.
_STATUS = {
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIAL,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "done_for_day": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
}

#: Substrings Alpaca uses when a client_order_id is already taken. Matching on
#: text is unlovely, but the API returns a generic 422 and the alternative --
#: treating a duplicate as a failure -- would break idempotent resubmission,
#: which is the property the whole crash-and-resume design rests on.
_DUPLICATE_MARKERS = ("client_order_id", "duplicate", "already exists")


def client_order_id(idem_key: str) -> str:
    """Hash ``(strategy, session, ticker, side)`` into Alpaca's id charset.

    Deterministic, so the *same* logical order submitted twice -- by a retry, or
    by a runner that crashed and resumed -- collides at the broker instead of
    double-filling. 32 hex chars leaves room under Alpaca's 128-char limit and
    cannot contain a character the API rejects.
    """
    key = str(idem_key or "").strip()
    if not key:
        raise ValueError("idem_key is empty; refusing to submit a non-idempotent order")
    return "lab-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class AlpacaBroker:
    """Live/paper order routing through ``alpaca-py``'s trading API."""

    name = "alpaca"

    def __init__(
        self,
        *,
        paper: bool = True,
        key_id: str | None = None,
        secret_key: str | None = None,
        allow_live: bool = False,
    ) -> None:
        self.paper = bool(paper)
        self.allow_live = bool(allow_live)
        self._key_id = key_id
        self._secret_key = secret_key
        self._client: Any = None
        self._orders: dict[str, Order] = {}
        self._seen_idem: dict[str, str] = {}
        self._seen_fills: set[str] = set()
        self.calls_made = 0
        self.last_call: datetime | None = None

    # --- availability -------------------------------------------------------

    def credentials(self) -> tuple[str | None, str | None]:
        settings = get_settings()
        return (
            self._key_id or settings.alpaca_key_id,
            self._secret_key or settings.alpaca_secret_key,
        )

    def live_opt_in(self) -> tuple[bool, str]:
        """``(permitted, reason)`` for non-paper trading. Paper is always fine."""
        if self.paper:
            return True, ""
        if not self.allow_live:
            return False, "live trading needs AlpacaBroker(paper=False, allow_live=True)"
        if str(os.getenv(LIVE_ENV_FLAG, "")).strip().lower() not in {"1", "true", "yes", "on"}:
            return False, f"live trading needs {LIVE_ENV_FLAG}=1 in the environment"
        return True, ""

    def available(self) -> tuple[bool, str]:
        try:
            found = importlib.util.find_spec("alpaca") is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return False, "alpaca-py not installed (pip install 'strategy-lab[data]')"
        # The live opt-in is reported ahead of the credential checks: pointing
        # at real money without opting in is the more urgent thing to say.
        ok, reason = self.live_opt_in()
        if not ok:
            return False, reason
        key_id, secret = self.credentials()
        if not key_id:
            return False, "no ALPACA_API_KEY_ID"
        if not secret:
            return False, "no ALPACA_API_SECRET_KEY"
        return True, ""

    def require_available(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"alpaca broker unavailable: {reason}")

    def client(self) -> Any:
        if self._client is None:
            self.require_available()
            from alpaca.trading.client import TradingClient  # noqa: PLC0415

            key_id, secret = self.credentials()
            self._client = TradingClient(key_id, secret, paper=self.paper)
        return self._client

    def _called(self) -> None:
        self.calls_made += 1
        self.last_call = utcnow()

    # --- Broker -------------------------------------------------------------

    def submit(self, order: Order) -> Order:
        """Place an order; a duplicate ``idem_key`` returns the existing one.

        Idempotency is enforced twice on purpose -- locally, so a resubmit
        inside one process costs no API call, and at Alpaca via
        ``client_order_id``, so it survives the process dying.
        """
        if not order.idem_key:
            raise ValueError(f"order {order.id} has no idem_key; refusing to submit")
        prior = self._seen_idem.get(order.idem_key)
        if prior is not None:
            return self._orders[prior]

        coid = client_order_id(order.idem_key)
        try:
            raw = self._place(order, coid)
        except Exception as exc:  # noqa: BLE001 - duplicate detection, see below
            if not _looks_duplicate(exc):
                raise
            log.warning("duplicate client_order_id %s; adopting the existing order", coid)
            raw = self._by_client_id(coid)
            if raw is None:
                raise

        placed = self._to_order(raw, template=order)
        self._orders[placed.id] = placed
        self._seen_idem[order.idem_key] = placed.id
        return placed

    def _place(self, order: Order, coid: str) -> Any:
        from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: PLC0415
        from alpaca.trading.requests import (  # noqa: PLC0415
            LimitOrderRequest,
            MarketOrderRequest,
        )

        side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL
        qty = order.qty if order.qty % 1 else int(order.qty)
        common = {
            "symbol": order.ticker,
            "qty": qty,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": coid,
        }
        if order.order_type is OrderType.LIMIT and order.limit_price:
            request = LimitOrderRequest(limit_price=float(order.limit_price), **common)
        else:
            request = MarketOrderRequest(**common)
        raw = self.client().submit_order(order_data=request)
        self._called()
        return raw

    def _by_client_id(self, coid: str) -> Any:
        try:
            raw = self.client().get_order_by_client_id(coid)
            self._called()
            return raw
        except Exception as exc:  # noqa: BLE001
            log.error("could not fetch existing order %s: %s", coid, exc)
            return None

    def cancel(self, order_id: str) -> bool:
        target = self._orders.get(order_id)
        broker_id = (target.broker_order_id if target else None) or order_id
        try:
            self.client().cancel_order_by_id(broker_id)
            self._called()
        except Exception as exc:  # noqa: BLE001 - an already-gone order is not an error
            log.warning("cancel %s failed: %s", broker_id, exc)
            return False
        if target is not None:
            target.status = OrderStatus.CANCELED
        return True

    def cancel_all(self, ticker: str | None = None) -> int:
        n = 0
        for order in self.open_orders():
            if ticker and order.ticker != ticker.upper():
                continue
            n += int(self.cancel(order.id))
        return n

    def open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
        from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

        raws = self.client().get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        self._called()
        out = [self._to_order(r) for r in raws or []]
        for o in out:
            self._orders.setdefault(o.id, o)
        return out

    def positions(self) -> dict[str, Position]:
        raws = self.client().get_all_positions()
        self._called()
        out: dict[str, Position] = {}
        for r in raws or []:
            ticker = str(getattr(r, "symbol", "")).upper()
            if not ticker:
                continue
            qty = _float(getattr(r, "qty", 0.0))
            if str(getattr(getattr(r, "side", ""), "value", getattr(r, "side", ""))).lower() == "short":
                qty = -abs(qty)
            out[ticker] = Position(
                ticker=ticker,
                qty=qty,
                avg_price=_float(getattr(r, "avg_entry_price", 0.0)),
                last_price=_float(getattr(r, "current_price", 0.0)) or _float(getattr(r, "avg_entry_price", 0.0)),
                realized_pnl=0.0,
            )
        return out

    def account(self) -> dict[str, Any]:
        a = self.client().get_account()
        self._called()
        return {
            "cash": _float(getattr(a, "cash", 0.0)),
            "equity": _float(getattr(a, "equity", 0.0)),
            "buying_power": _float(getattr(a, "buying_power", 0.0)),
            "broker": self.name,
            "paper": self.paper,
            "status": str(getattr(a, "status", "") or ""),
            "account_number": str(getattr(a, "account_number", "") or ""),
            "pattern_day_trader": bool(getattr(a, "pattern_day_trader", False)),
            "daytrade_count": int(getattr(a, "daytrade_count", 0) or 0),
        }

    def clock(self) -> dict[str, Any]:
        c = self.client().get_clock()
        self._called()
        return {
            "timestamp": _iso(getattr(c, "timestamp", None)),
            "is_open": bool(getattr(c, "is_open", False)),
            "next_open": _iso(getattr(c, "next_open", None)),
            "next_close": _iso(getattr(c, "next_close", None)),
        }

    # --- fills --------------------------------------------------------------

    def poll_fills(self, since: datetime | None = None) -> list[Fill]:
        """Fills that landed since ``since``, as lab ``Fill`` records.

        Derived from closed orders rather than the activities feed so the free
        tier is enough. Fill ids are deterministic in
        ``(broker order id, filled qty)``, which is what lets the runner drop a
        fill it has already booked when two polls overlap.
        """
        from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
        from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

        kwargs: dict[str, Any] = {"status": QueryOrderStatus.CLOSED, "limit": 500}
        if since is not None:
            kwargs["after"] = to_utc(since)
        raws = self.client().get_orders(filter=GetOrdersRequest(**kwargs))
        self._called()

        out: list[Fill] = []
        for raw in raws or []:
            qty = _float(getattr(raw, "filled_qty", 0.0))
            price = _float(getattr(raw, "filled_avg_price", 0.0))
            if qty <= 0 or price <= 0:
                continue
            broker_id = str(getattr(raw, "id", "") or "")
            fill_id = f"af_{broker_id}_{qty:g}"
            if fill_id in self._seen_fills:
                continue
            self._seen_fills.add(fill_id)
            order = self._to_order(raw)
            self._orders.setdefault(order.id, order)
            out.append(
                Fill(
                    id=fill_id,
                    order_id=order.id,
                    ticker=order.ticker,
                    side=order.side,
                    qty=qty,
                    price=price,
                    at=to_utc(getattr(raw, "filled_at", None) or utcnow()),
                    strategy=order.strategy,
                    tag=order.tag,
                    decision_id=order.decision_id,
                )
            )
        return out

    # --- mapping ------------------------------------------------------------

    def _to_order(self, raw: Any, template: Order | None = None) -> Order:
        broker_id = str(getattr(raw, "id", "") or "")
        coid = str(getattr(raw, "client_order_id", "") or "")
        side_raw = str(getattr(getattr(raw, "side", ""), "value", getattr(raw, "side", ""))).lower()
        status_raw = str(getattr(getattr(raw, "status", ""), "value", getattr(raw, "status", ""))).lower()
        ticker = str(getattr(raw, "symbol", "") or (template.ticker if template else "")).upper()

        order = Order(
            id=(template.id if template else None) or (broker_id or new_id("o_")),
            ticker=ticker,
            side=Side(side_raw) if side_raw in {"buy", "sell"} else (template.side if template else Side.BUY),
            qty=_float(getattr(raw, "qty", 0.0)) or (template.qty if template else 0.0),
            order_type=(template.order_type if template else OrderType.MARKET),
            limit_price=(template.limit_price if template else None),
            created_at=to_utc(getattr(raw, "created_at", None) or utcnow()),
            strategy=(template.strategy if template else ""),
            tag=(template.tag if template else ""),
            idem_key=(template.idem_key if template else coid),
            status=_STATUS.get(status_raw, OrderStatus.SUBMITTED),
            broker_order_id=broker_id or None,
            decision_id=(template.decision_id if template else None),
            reason=(template.reason if template else ""),
        )
        return order

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        ok, reason = self.available()
        mode = "paper" if self.paper else "LIVE"
        return f"<AlpacaBroker {mode} available={ok}{'' if ok else f' ({reason})'}>"


def _looks_duplicate(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _DUPLICATE_MARKERS)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return to_utc(value).isoformat()
    except Exception:  # noqa: BLE001 - a clock field we cannot parse is not fatal
        return str(value)


def open_order_ids(orders: Iterable[Order]) -> set[str]:
    """Every id an order is known by, for reconciliation matching."""
    out: set[str] = set()
    for o in orders:
        out.update({o.id, o.broker_order_id or "", o.idem_key or ""})
    out.discard("")
    return out
