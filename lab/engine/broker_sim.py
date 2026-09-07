"""The simulated broker.

Deliberately shaped like a real one: orders are *submitted*, sit in a queue, and
get filled later by :meth:`process`. That is not ceremony -- it is what makes
next-bar-open fills the natural default and same-bar fills the explicit opt-in,
and it is why the same runner loop works against Alpaca with only the broker
swapped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from lab.backtest.fills import FillModel
from lab.engine.events import Fill, Order, OrderStatus, Position
from lab.engine.portfolio import Portfolio
from lab.timeutil import to_utc


class SimBroker:
    name = "sim"

    def __init__(
        self,
        portfolio: Portfolio,
        fill_model: FillModel | None = None,
        *,
        name: str = "sim",
    ) -> None:
        self.portfolio = portfolio
        self.fill_model = fill_model or FillModel()
        self.name = name
        self._pending: list[Order] = []
        self._all_orders: dict[str, Order] = {}
        self._seen_idem: dict[str, str] = {}
        self._fills: list[Fill] = []

    # --- Broker ------------------------------------------------------------

    def submit(self, order: Order) -> Order:
        """Queue an order. Repeated ``idem_key`` is a no-op, exactly as a real
        broker adapter must behave so a crashed-and-rerun job cannot double-fire.
        """
        if order.idem_key:
            prior_id = self._seen_idem.get(order.idem_key)
            if prior_id is not None:
                return self._all_orders[prior_id]
            self._seen_idem[order.idem_key] = order.id
        order.status = OrderStatus.SUBMITTED
        order.created_at = order.created_at or None
        self._all_orders[order.id] = order
        self._pending.append(order)
        return order

    def cancel(self, order_id: str) -> bool:
        for i, o in enumerate(self._pending):
            if o.id == order_id:
                o.status = OrderStatus.CANCELED
                self._pending.pop(i)
                return True
        return False

    def cancel_all(self, ticker: str | None = None) -> int:
        targets = [o for o in self._pending if ticker is None or o.ticker == ticker.upper()]
        for o in targets:
            self.cancel(o.id)
        return len(targets)

    def open_orders(self) -> list[Order]:
        return list(self._pending)

    def positions(self) -> dict[str, Position]:
        return dict(self.portfolio.positions)

    def account(self) -> dict[str, Any]:
        return {
            "cash": self.portfolio.cash,
            "equity": self.portfolio.equity,
            "buying_power": max(self.portfolio.cash, 0.0),
            "broker": self.name,
            "paper": True,
        }

    # --- simulation --------------------------------------------------------

    def process(self, ts: datetime, bars: Mapping[str, Mapping[str, Any]]) -> list[Fill]:
        """Fill whatever the model says is fillable at ``ts``.

        An order for a ticker with no bar at this timestamp stays pending rather
        than being silently dropped -- a halted or untraded name should delay a
        fill, not vanish it.
        """
        ts = to_utc(ts)
        filled: list[Fill] = []
        still_pending: list[Order] = []

        for order in self._pending:
            bar = bars.get(order.ticker)
            if bar is None:
                still_pending.append(order)
                continue
            fill = self.fill_model.fill(order, bar, ts)
            if fill is None:
                still_pending.append(order)
                continue
            self.portfolio.apply_fill(fill)
            if abs(fill.qty - order.qty) < 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL
                order.qty -= fill.qty
                still_pending.append(order)
            self._fills.append(fill)
            filled.append(fill)

        self._pending = still_pending
        return filled

    def expire_pending(self, reason: str = "expired") -> list[Order]:
        """Drop day orders that never filled. Called at session end."""
        expired = self._pending
        for o in expired:
            o.status = OrderStatus.EXPIRED
            o.reason = reason
        self._pending = []
        return expired

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def orders(self) -> list[Order]:
        return list(self._all_orders.values())
