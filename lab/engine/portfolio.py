"""Portfolio accounting.

Small, boring, and worth getting exactly right: every metric in every report is
downstream of these numbers. Average-cost basis, commissions charged to realized
P&L, and a round trip closed the moment a position returns to flat -- including
the case where a single fill crosses through zero and opens the other side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from lab.engine.events import Fill, Position, Side, Trade
from lab.timeutil import to_utc

#: Below this many shares a position is flat. Guards float dust from repeated
#: fractional rebalances accumulating into a phantom position.
EPS = 1e-9


class Portfolio:
    """Mutable book of record for one run."""

    def __init__(self, cash: float = 100_000.0) -> None:
        self.starting_cash = float(cash)
        self._cash = float(cash)
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._open_trade: dict[str, Trade] = {}
        self._entry_bar: dict[str, int] = {}
        self._bars_seen = 0
        self.total_commission = 0.0
        self.turnover_notional = 0.0

    # --- PortfolioView -----------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return {t: p for t, p in self._positions.items() if not p.is_flat}

    @property
    def equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    @property
    def gross_exposure(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        return sum(abs(p.market_value) for p in self._positions.values()) / eq

    @property
    def net_exposure(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        return sum(p.market_value for p in self._positions.values()) / eq

    def position(self, ticker: str) -> Position:
        t = ticker.upper()
        return self._positions.get(t) or Position(ticker=t)

    def weight(self, ticker: str) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        return self.position(ticker).market_value / eq

    # --- marking -----------------------------------------------------------

    def mark(self, prices: Mapping[str, float]) -> None:
        """Update last prices. Unpriced positions keep their previous mark."""
        for ticker, price in prices.items():
            pos = self._positions.get(ticker.upper())
            if pos is not None and price is not None and price == price:  # NaN check
                pos.last_price = float(price)

    def tick_bar(self) -> None:
        self._bars_seen += 1

    # --- fills -------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> Trade | None:
        """Apply a fill and return a round trip if one closed.

        A fill that crosses zero closes the old trade at the crossing price and
        opens a new one for the residual, which is the only honest way to book a
        reversal in a single order.
        """
        ticker = fill.ticker
        pos = self._positions.setdefault(ticker, Position(ticker=ticker))
        qty_delta = fill.signed_qty
        price = fill.price

        self._cash -= qty_delta * price
        self._cash -= fill.commission
        self.total_commission += fill.commission
        self.turnover_notional += abs(fill.notional)
        pos.last_price = price

        closed: Trade | None = None
        old_qty = pos.qty
        new_qty = old_qty + qty_delta

        if abs(old_qty) < EPS:
            # Opening from flat.
            pos.qty = new_qty
            pos.avg_price = price
            pos.opened_at = fill.at
            pos.tag = fill.tag or pos.tag
            self._open_trade[ticker] = self._new_trade(fill, new_qty)
        elif (old_qty > 0) == (qty_delta > 0):
            # Adding to the same side: weighted-average the basis.
            total_cost = pos.avg_price * old_qty + price * qty_delta
            pos.qty = new_qty
            pos.avg_price = total_cost / new_qty if abs(new_qty) > EPS else price
            trade = self._open_trade.get(ticker)
            if trade is not None:
                trade.qty = abs(new_qty)
                trade.entry_price = pos.avg_price
                trade.commission += fill.commission
        else:
            # Reducing, closing, or reversing.
            closing_qty = min(abs(qty_delta), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realized = closing_qty * (price - pos.avg_price) * direction
            pos.realized_pnl += realized

            # EVERY reducing fill is booked, not only the one that flattens the
            # position. A scale-out is a real exit with realized money behind
            # it; leaving it out of the ledger until the last share is sold is
            # what let a rebalancing strategy return +35% while its trade list
            # showed a single +4% round trip. The sum of ledger P&L now equals
            # total realized P&L by construction -- see `reconcile()`.
            closed = self._book_exit(ticker, fill, realized, closing_qty, abs(old_qty))

            if abs(new_qty) < EPS:
                pos.qty = 0.0
                pos.avg_price = 0.0
                pos.opened_at = None
                self._open_trade.pop(ticker, None)
                self._entry_bar.pop(ticker, None)
            elif (new_qty > 0) == (old_qty > 0):
                pos.qty = new_qty  # partial reduction, basis unchanged
            else:
                # Crossed through zero: the close is booked above; open the
                # residual on the other side.
                pos.qty = new_qty
                pos.avg_price = price
                pos.opened_at = fill.at
                self._open_trade[ticker] = self._new_trade(fill, new_qty)

        if closed is not None:
            self._trades.append(closed)
        if abs(pos.qty) < EPS:
            pos.qty = 0.0
        return closed

    def _new_trade(self, fill: Fill, qty: float) -> Trade:
        self._entry_bar[fill.ticker] = self._bars_seen
        return Trade(
            ticker=fill.ticker,
            side="long" if qty > 0 else "short",
            qty=abs(qty),
            entry_time=to_utc(fill.at),
            entry_price=fill.price,
            commission=fill.commission,
            tag=fill.tag,
        )

    def _book_exit(
        self,
        ticker: str,
        fill: Fill,
        realized: float,
        closing_qty: float,
        open_qty: float,
    ) -> Trade | None:
        """Book one exit leg against the open lot.

        The entry commission is split pro-rata by the fraction of the lot being
        closed, so three trims of a commissioned entry charge that entry once
        between them rather than three times.
        """
        lot = self._open_trade.get(ticker)
        if lot is None:
            return None

        share = (closing_qty / open_qty) if open_qty else 1.0
        entry_commission = lot.commission * share
        lot.commission -= entry_commission
        lot.qty = max(open_qty - closing_qty, 0.0)

        commission = entry_commission + fill.commission
        pnl = realized - commission
        basis = abs(lot.entry_price * closing_qty)
        return Trade(
            ticker=ticker,
            side=lot.side,
            qty=closing_qty,
            entry_time=lot.entry_time,
            entry_price=lot.entry_price,
            exit_time=to_utc(fill.at),
            exit_price=fill.price,
            pnl=pnl,
            pnl_pct=pnl / basis if basis else 0.0,
            bars_held=max(0, self._bars_seen - self._entry_bar.get(ticker, self._bars_seen)),
            commission=commission,
            tag=lot.tag,
            exit_reason=fill.tag or "",
        )

    # --- reconciliation ----------------------------------------------------

    def realized_pnl(self) -> float:
        """Realized P&L across every position, open or closed."""
        return sum(p.realized_pnl for p in self._positions.values())

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def reconcile(self) -> dict[str, float]:
        """Prove the equity curve and the trade ledger tell the same story.

        ``residual`` is money the book made that no ledger row explains. It must
        be zero to rounding; anything else means a fill moved cash without
        producing a trade, and every trade-level metric downstream is then
        describing a different run than the equity curve is.
        """
        ledger = sum(t.pnl for t in self._trades)
        equity_gain = self.equity - self.starting_cash
        unrealized = self.unrealized_pnl()
        open_commission = sum(t.commission for t in self._open_trade.values())
        residual = equity_gain - ledger - unrealized + open_commission
        # Snap float noise to a true zero. A residual of -1e-11 displayed as
        # "-0.000" reads as a real discrepancy someone will go hunting for.
        if abs(residual) < 1e-6:
            residual = 0.0
        return {
            "equity_gain": round(equity_gain, 6),
            "ledger_pnl": round(ledger, 6),
            "unrealized_pnl": round(unrealized, 6),
            "open_commission": round(open_commission, 6),
            "residual": round(residual, 6),
        }

    # --- sizing ------------------------------------------------------------

    def target_to_delta_shares(
        self, ticker: str, target_pct: float, price: float, *, allow_fractional: bool = False
    ) -> float:
        """Shares to trade to move ``ticker`` to ``target_pct`` of equity.

        Sized against current equity, which is the honest reading of "5% of the
        book" and keeps sizing stable as the book grows or shrinks.
        """
        if price is None or price <= 0:
            return 0.0
        target_value = self.equity * float(target_pct)
        current_value = self.position(ticker).qty * price
        delta_shares = (target_value - current_value) / price
        if not allow_fractional:
            delta_shares = float(int(delta_shares))
        return 0.0 if abs(delta_shares) < EPS else delta_shares

    # --- reporting ---------------------------------------------------------

    def trades(self) -> list[Trade]:
        return list(self._trades)

    def open_trades(self) -> list[Trade]:
        return list(self._open_trade.values())

    def snapshot(self, at: datetime | None = None) -> dict[str, Any]:
        return {
            "at": to_utc(at).isoformat() if at else None,
            "cash": round(self._cash, 6),
            "equity": round(self.equity, 6),
            "gross_exposure": round(self.gross_exposure, 6),
            "net_exposure": round(self.net_exposure, 6),
            "n_positions": len(self.positions),
            "positions": [
                {
                    "ticker": p.ticker,
                    "qty": p.qty,
                    "avg_price": round(p.avg_price, 6),
                    "last_price": round(p.last_price, 6),
                    "market_value": round(p.market_value, 6),
                    "unrealized_pnl": round(p.unrealized_pnl, 6),
                    "unrealized_pct": round(p.unrealized_pct, 6),
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                    "tag": p.tag,
                }
                for p in self.positions.values()
            ],
        }


class ReadOnlyPortfolio:
    """What a strategy gets. Mutation is not merely discouraged, it is absent."""

    __slots__ = ("_p",)

    def __init__(self, portfolio: Portfolio) -> None:
        self._p = portfolio

    @property
    def cash(self) -> float:
        return self._p.cash

    @property
    def equity(self) -> float:
        return self._p.equity

    @property
    def positions(self) -> Mapping[str, Position]:
        return dict(self._p.positions)

    @property
    def gross_exposure(self) -> float:
        return self._p.gross_exposure

    @property
    def net_exposure(self) -> float:
        return self._p.net_exposure

    @property
    def starting_cash(self) -> float:
        return self._p.starting_cash

    def position(self, ticker: str) -> Position:
        return self._p.position(ticker)

    def weight(self, ticker: str) -> float:
        return self._p.weight(ticker)

    def snapshot(self, at: datetime | None = None) -> dict[str, Any]:
        return self._p.snapshot(at)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PortfolioView equity={self.equity:,.2f} positions={len(self.positions)}>"
