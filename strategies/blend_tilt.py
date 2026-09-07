NAME = "blend_tilt"

PARAMS = {
    "lb_fast": 63,
    "lb_slow": 126,
    "top_n": 4,
    "core_weight": 0.30,
    "tilt_weight": 0.68,
    "rebalance_every": 5,
    "trend_ma": 200,
}


class Strategy:
    def __init__(self, params=None):
        self.params = dict(PARAMS) | dict(params or {})
        self.state = {"bar": -1}

    def on_bar(self, ctx):
        p = ctx.params
        self.state["bar"] += 1
        if self.state["bar"] % int(p["rebalance_every"]) != 0:
            return

        lb_fast = int(p["lb_fast"])
        lb_slow = int(p["lb_slow"])
        need = max(lb_fast, lb_slow) + 1
        scores = {}
        for t in ctx.universe:
            if t == "SPY":
                continue
            closes = ctx.history(t, "close", need)
            if len(closes) < need:
                continue
            last = float(closes.iloc[-1])
            base_f = float(closes.iloc[-1 - lb_fast])
            base_s = float(closes.iloc[-1 - lb_slow])
            if base_f <= 0 or base_s <= 0:
                continue
            mom_f = last / base_f - 1.0
            mom_s = last / base_s - 1.0
            sma = ctx.indicator(t, "sma", n=int(p["trend_ma"])).dropna()
            if not len(sma) or last <= float(sma.iloc[-1]):
                continue
            blend = 0.5 * mom_f + 0.5 * mom_s
            if blend <= 0:
                continue
            scores[t] = blend

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        wanted = ranked[: int(p["top_n"])]

        for t in list(ctx.portfolio.positions):
            if t != "SPY" and t not in wanted:
                ctx.close(t, reason="fell out of tilt book")

        slot = float(p["tilt_weight"]) / max(1, int(p["top_n"]))
        for i, t in enumerate(wanted):
            ctx.order_target_pct(t, slot, tag="tilt", reason=f"blend rank {i+1}")

        unused = float(p["tilt_weight"]) - slot * len(wanted)
        spy_w = float(p["core_weight"]) + unused
        if "SPY" in ctx.universe:
            ctx.order_target_pct("SPY", spy_w, tag="core", reason="SPY core + idle tilt capacity")
