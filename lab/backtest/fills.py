"""The fill model: how an order becomes a price.

Default is next-bar-open, because a strategy that decides on a bar cannot also
trade inside it. ``same_close`` exists -- sometimes you genuinely want to model a
market-on-close decision -- but it is optimistic in the common case and is
flagged loudly wherever run metadata is displayed, rather than being quietly
allowed to inflate a Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from lab.engine.events import Fill, Order, OrderType, Side, new_id

MODES = ("next_open", "next_close", "same_close")


@dataclass
class FillModel:
    mode: str = "next_open"
    slippage_bps: float = 5.0
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    partial_fill_volume_pct: float | None = None
    allow_fractional: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown fill mode {self.mode!r}; expected one of {MODES}")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")

    @property
    def optimistic(self) -> bool:
        """True when the model lets a decision trade at a price it helped set."""
        return self.mode == "same_close"

    @property
    def price_field(self) -> str:
        return "open" if self.mode == "next_open" else "close"

    @property
    def same_bar(self) -> bool:
        return self.mode == "same_close"

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> "FillModel":
        if not m:
            return cls()
        known = {f: m[f] for f in cls.__dataclass_fields__ if f in m}
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__} | {
            "optimistic": self.optimistic
        }

    # --- pricing -----------------------------------------------------------

    def fill_price(self, side: Side, bar: Mapping[str, Any]) -> float:
        """Reference price plus slippage, always against the trader."""
        base = bar.get(self.price_field)
        if base is None or base != base:  # missing or NaN
            base = bar.get("close")
        if base is None or base != base:
            raise ValueError("bar has no usable price")
        base = float(base)
        drift = base * (self.slippage_bps / 10_000.0)
        price = base + drift if side is Side.BUY else base - drift
        # Slippage must never push a fill outside the bar it happened in.
        high, low = bar.get("high"), bar.get("low")
        if high is not None and high == high:
            price = min(price, float(high))
        if low is not None and low == low:
            price = max(price, float(low))
        return max(price, 1e-6)

    def commission(self, qty: float, price: float) -> float:
        c = self.commission_per_order + self.commission_per_share * abs(qty)
        if self.min_commission and abs(qty) > 0:
            c = max(c, self.min_commission)
        return round(c, 6)

    def cap_qty(self, qty: float, bar: Mapping[str, Any]) -> float:
        """Clip to a share of the bar's volume, so a backtest cannot pretend to
        buy a day's entire float without moving the price."""
        if self.partial_fill_volume_pct is None:
            return qty
        volume = bar.get("volume")
        if volume is None or volume != volume or volume <= 0:
            return qty
        cap = float(volume) * float(self.partial_fill_volume_pct)
        capped = min(abs(qty), cap)
        if not self.allow_fractional:
            capped = float(int(capped))
        return capped * (1.0 if qty > 0 else -1.0)

    # --- the fill ----------------------------------------------------------

    def fill(self, order: Order, bar: Mapping[str, Any], ts: datetime) -> Fill | None:
        """Attempt to fill ``order`` against ``bar``. ``None`` means no fill."""
        if bar is None:
            return None
        qty = order.qty
        if qty <= 0:
            return None

        price = self.fill_price(order.side, bar)

        if order.order_type is OrderType.LIMIT and order.limit_price is not None:
            limit = float(order.limit_price)
            low, high = bar.get("low"), bar.get("high")
            if order.side is Side.BUY:
                if low is None or low != low or float(low) > limit:
                    return None
                price = min(price, limit)
            else:
                if high is None or high != high or float(high) < limit:
                    return None
                price = max(price, limit)

        signed = qty if order.side is Side.BUY else -qty
        signed = self.cap_qty(signed, bar)
        qty = abs(signed)
        if qty <= 0:
            return None

        reference = bar.get(self.price_field)
        reference = float(reference) if reference is not None and reference == reference else price
        return Fill(
            id=new_id("f_"),
            order_id=order.id,
            ticker=order.ticker,
            side=order.side,
            qty=qty,
            price=price,
            at=ts,
            commission=self.commission(qty, price),
            slippage=abs(price - reference) * qty,
            strategy=order.strategy,
            tag=order.tag,
            decision_id=order.decision_id,
        )
