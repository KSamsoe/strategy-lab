"""Cross-sectional momentum with a trend filter and an ATR stop.

Rank the universe by trailing return, hold the top N while they are above their
long moving average, size equally, and exit on rank decay, trend break, or a
volatility-scaled stop. Nothing original -- which is the point. It exercises
history, indicators, ranking, position management and exits, so it is the
natural first end-to-end strategy and the natural sweep subject.

Every number lives in ``PARAMS`` so a sweep or an agent can vary it without
touching this file.
"""

from __future__ import annotations

import pandas as pd

from lab.engine.protocols import Context

NAME = "momo"

PARAMS = {
    "lookback": 126,        # ~6 months of trailing return for the ranking
    "top_n": 4,             # how many names to hold
    "trend_ma": 200,        # only hold names above this moving average
    "exit_rank_buffer": 2,  # hold until rank slips past top_n + this
    "atr_len": 14,
    "atr_stop_mult": 3.0,   # exit when price falls this many ATRs below the peak
    "min_history": 210,     # refuse to rank a name we barely know
    "target_pct": 0.0,      # 0 = equal weight across the held names
}


class Momo:
    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(PARAMS) | dict(params or {})
        # Peak close since entry, per ticker -- the anchor for the trailing stop.
        self._peak: dict[str, float] = {}

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _trailing_return(ctx: Context, ticker: str, lookback: int) -> float | None:
        closes = ctx.history(ticker, "close", lookback + 1)
        if len(closes) < lookback + 1:
            return None
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        if first <= 0:
            return None
        return last / first - 1.0

    @staticmethod
    def _last(series: pd.Series) -> float | None:
        s = series.dropna()
        return float(s.iloc[-1]) if len(s) else None

    # --- the decision ------------------------------------------------------

    def on_bar(self, ctx: Context) -> None:
        p = ctx.params
        lookback = int(p["lookback"])
        top_n = int(p["top_n"])
        trend_ma = int(p["trend_ma"])
        min_history = int(p["min_history"])

        scores: dict[str, float] = {}
        trending: set[str] = set()

        for ticker in ctx.universe:
            price = ctx.price(ticker)
            if price is None:
                continue
            if len(ctx.history(ticker, "close", min_history)) < min_history:
                continue
            ret = self._trailing_return(ctx, ticker, lookback)
            if ret is None:
                continue
            scores[ticker] = ret
            ma = self._last(ctx.indicator(ticker, "sma", n=trend_ma, lookback=trend_ma * 2))
            if ma is not None and price > ma:
                trending.add(ticker)

        if not scores:
            return

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        rank_of = {t: i for i, t in enumerate(ranked)}
        wanted = [t for t in ranked[:top_n] if t in trending]

        held = set(ctx.portfolio.positions)
        buffer = int(p["exit_rank_buffer"])
        atr_mult = float(p["atr_stop_mult"])

        # --- exits first: they free capacity the entries below may need -----
        for ticker in sorted(held):
            price = ctx.price(ticker)
            if price is None:
                continue
            self._peak[ticker] = max(self._peak.get(ticker, price), price)

            reason = None
            if ticker not in trending:
                reason = "trend break"
            elif rank_of.get(ticker, 10**6) >= top_n + buffer:
                reason = f"rank decay ({rank_of.get(ticker)})"
            else:
                atr = self._last(
                    ctx.indicator(ticker, "atr", n=int(p["atr_len"]), lookback=int(p["atr_len"]) * 4)
                )
                if atr and price <= self._peak[ticker] - atr_mult * atr:
                    reason = f"atr stop ({atr_mult}x)"

            if reason:
                ctx.close(ticker, tag="exit", reason=reason)
                self._peak.pop(ticker, None)
                wanted = [t for t in wanted if t != ticker]

        if not wanted:
            return

        explicit = float(p.get("target_pct") or 0.0)
        weight = explicit if explicit > 0 else (1.0 / max(top_n, 1))

        for ticker in wanted:
            if ticker not in held:
                self._peak[ticker] = ctx.price(ticker) or 0.0
            ctx.order_target_pct(
                ticker, weight, tag="entry" if ticker not in held else "hold",
                reason=f"rank {rank_of[ticker] + 1} · {lookback}d return {scores[ticker]:+.1%}",
            )

        ctx.log(
            event="rank",
            top=[{"t": t, "ret": round(scores[t], 4), "trend": t in trending} for t in ranked[:top_n]],
            holding=len(wanted),
        )


STRATEGY = Momo()
