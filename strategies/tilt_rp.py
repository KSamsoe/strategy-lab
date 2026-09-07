"""SPY core plus a momentum tilt, with the tilt sized by inverse volatility.

Built from what the data actually shows rather than from what usually works.
Three measurements on this universe, 2020-2026, drove every choice here:

**Only the 126-day horizon predicts anything.** Cross-sectional rank correlation
between trailing and forward returns is +0.033 at a 126-day lookback and
indistinguishable from zero at 5, 21 and 63 days -- slightly negative at 21.
So ``w_slow`` defaults to 0.75: the fast leg is along for confirmation, not for
signal.

**The 200-day trend gate is not obviously helping.** SPY below its 200-day
average returned +21.1% annualised over this window against +14.4% above it.
The gate improves Sharpe (1.13 vs 0.92) by avoiding volatility, but it does so by
sitting out the recoveries that produced the returns. ``use_trend_gate`` is a
flag rather than an assumption so the question can be answered instead of
inherited.

**Volatility is the one thing that is genuinely predictable, and market timing is
the wrong use for it.** Trailing 21-day realised vol correlates +0.49 with the
next 21 days -- by far the strongest relationship in the data. But scaling market
exposure by it barely pays: targeting 12% vol on SPY lifted Sharpe from 0.99 to
1.02 while costing 4.5 points of annual return, because high-volatility periods
here also had *higher* returns. So volatility is used where the prediction is
worth something -- equalising risk across the names already selected -- and not
to decide how much to own overall. ``invvol_strength`` at 0 is plain equal
weighting, which makes the choice an ablation rather than a belief.
"""

NAME = "tilt_rp"

PARAMS = {
    # Momentum ranking. `w_slow` is the weight on the 126-day leg, which is the
    # only horizon that measured as predictive.
    "lb_fast": 63,
    "lb_slow": 126,
    "w_slow": 0.75,
    "top_n": 4,
    # Book construction.
    "core_weight": 0.30,
    "tilt_weight": 0.68,
    "rebalance_every": 5,
    # The two questions this file exists to answer. Both default to the answer
    # the measurements suggest; set them to the other value to ablate.
    "use_trend_gate": 0,      # 1 restores blend_tilt's price > 200d SMA filter
    "trend_ma": 200,
    "invvol_strength": 1.0,   # 0 = equal weight, 1 = full inverse-vol
    "vol_window": 21,
    # A floor on any single slot's share of the tilt, so inverse-vol sizing
    # cannot quietly turn a 4-name book into a 1-name book when one name goes
    # quiet. Without it the sizing rule and the concentration risk are the same
    # dial.
    "min_slot_frac": 0.5,
    "core_ticker": "SPY",
}


class Strategy:
    def __init__(self, params=None):
        self.params = dict(PARAMS) | dict(params or {})
        self.state = {"bar": -1}

    def on_bar(self, ctx):
        p = ctx.params
        self.state["bar"] += 1
        if self.state["bar"] % max(1, int(p["rebalance_every"])) != 0:
            return

        core = str(p["core_ticker"]).upper()
        lb_fast, lb_slow = int(p["lb_fast"]), int(p["lb_slow"])
        need = max(lb_fast, lb_slow) + 1
        w_slow = float(p["w_slow"])
        gate = bool(int(p["use_trend_gate"]))

        scores, vols = {}, {}
        for t in ctx.universe:
            if t == core:
                continue
            closes = ctx.history(t, "close", need)
            if len(closes) < need:
                continue
            last = float(closes.iloc[-1])
            base_f = float(closes.iloc[-1 - lb_fast])
            base_s = float(closes.iloc[-1 - lb_slow])
            if base_f <= 0 or base_s <= 0 or last <= 0:
                continue

            if gate:
                sma = ctx.indicator(t, "sma", n=int(p["trend_ma"])).dropna()
                if not len(sma) or last <= float(sma.iloc[-1]):
                    continue

            blend = (1.0 - w_slow) * (last / base_f - 1.0) + w_slow * (last / base_s - 1.0)
            if blend <= 0:
                continue
            scores[t] = blend

            # Realised vol for sizing. Falls back to a flat weight rather than
            # dropping the name: a missing vol estimate is a reason to size it
            # like everything else, not a reason to refuse a ranked signal.
            series = ctx.indicator(t, "rolling_vol", n=int(p["vol_window"])).dropna()
            v = float(series.iloc[-1]) if len(series) else 0.0
            vols[t] = v if v > 1e-6 else 0.0

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        wanted = ranked[: max(1, int(p["top_n"]))]

        for t in list(ctx.portfolio.positions):
            if t != core and t not in wanted:
                ctx.close(t, reason="fell out of the tilt book")

        weights = self._slot_weights(wanted, vols, p)
        tilt_total = float(p["tilt_weight"])
        used = 0.0
        for i, t in enumerate(wanted):
            w = tilt_total * weights[t]
            used += w
            ctx.order_target_pct(t, w, tag="tilt", reason=f"rank {i + 1}, w {w:.3f}")

        # Whatever the tilt did not use goes to the core rather than to cash: an
        # unfilled slot is a missing signal, not a reason to be out of the market.
        if core in ctx.universe:
            ctx.order_target_pct(
                core, float(p["core_weight"]) + (tilt_total - used),
                tag="core", reason="core plus idle tilt capacity",
            )

    @staticmethod
    def _slot_weights(wanted, vols, p):
        """Shares of the tilt sleeve, summing to 1 over the names chosen.

        Inverse volatility, pulled toward equal weight by ``invvol_strength`` and
        floored by ``min_slot_frac`` so no single name can dominate the sleeve
        just because it went quiet.
        """
        n = len(wanted)
        if n == 0:
            return {}
        equal = 1.0 / n
        strength = max(0.0, min(1.0, float(p["invvol_strength"])))
        usable = [vols.get(t, 0.0) for t in wanted]
        if strength <= 0 or not any(v > 0 for v in usable):
            return {t: equal for t in wanted}

        # A name with no vol estimate gets the median, so it is neither favoured
        # nor punished for the gap.
        known = sorted(v for v in usable if v > 0)
        fallback = known[len(known) // 2]
        inv = {t: 1.0 / (vols.get(t) or fallback) for t in wanted}
        total = sum(inv.values())
        raw = {t: inv[t] / total for t in wanted}

        blended = {t: strength * raw[t] + (1.0 - strength) * equal for t in wanted}
        floor = equal * max(0.0, min(1.0, float(p["min_slot_frac"])))
        clipped = {t: max(floor, w) for t, w in blended.items()}
        scale = sum(clipped.values())
        return {t: w / scale for t, w in clipped.items()}
