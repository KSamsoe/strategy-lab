"""Equal-weight buy and hold.

The baseline every other strategy has to beat. Keep it in the registry and
compare against it constantly -- a Sharpe of 1.1 means nothing until you know
what holding the same names would have paid.
"""

from __future__ import annotations

from lab.engine.protocols import Context

NAME = "buy_and_hold"

PARAMS = {
    "rebalance_days": 0,  # 0 = never rebalance after the initial allocation
    "cash_buffer": 0.02,  # leave a little cash so a rebalance can never bounce
}


class BuyAndHold:
    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(PARAMS) | dict(params or {})
        self._allocated = False
        self._bars_since_rebalance = 0

    def on_bar(self, ctx: Context) -> None:
        universe = [t for t in ctx.universe if ctx.price(t) is not None]
        if not universe:
            return

        rebalance_days = int(ctx.params.get("rebalance_days", 0) or 0)
        self._bars_since_rebalance += 1
        due = rebalance_days and self._bars_since_rebalance >= rebalance_days
        if self._allocated and not due:
            return

        buffer = float(ctx.params.get("cash_buffer", 0.02))
        weight = (1.0 - buffer) / len(universe)
        for ticker in universe:
            ctx.order_target_pct(ticker, weight, tag="hold", reason="equal weight")

        ctx.log(event="allocate", n=len(universe), weight=round(weight, 4))
        self._allocated = True
        self._bars_since_rebalance = 0


STRATEGY = BuyAndHold()
