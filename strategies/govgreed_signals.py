"""GovGreed composite-signal strategy (v1 of the standalone bot, ported).

This is the strategy from ``docs/govgreed-bot-design-v1.md`` §6, expressed as an
ordinary lab strategy. The standalone bot's scraper, scheduler, sizing code and
SQLite store all dissolve into platform concerns; what is left is the actual
hypothesis, which is the only part that was ever strategy-specific.

Entry: the ticker appears in ``/signals/top`` at tier A or better with the fresh
flag set, **and** is corroborated by either a herd signal at threshold with a
matching direction, or an insider score above a floor.

Exit: whichever comes first of a time stop, a fixed stop-loss, or the signal
flipping direction on a later pull.

Two honesty notes that belong in the code rather than in anyone's head. First,
the vendor's headline win-rate claims are unaudited marketing, congressional
disclosure is delayed by design, and the same feed sells to competing products,
so any edge is shared and possibly crowded. Second -- and this is what the
platform enforces rather than merely advises -- the context serves signals by
``knowledge_time``, so this strategy sees a congressional trade on its disclosure
date, never on its trade date. Backtests that get this wrong look spectacular
and mean nothing.

Sizing and concentration are NOT handled here. Position caps, the max-concurrent
limit and per-sector limits belong to the risk gate, which applies the same rules
to every strategy in the lab.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from lab.engine.protocols import Context
from lab.timeutil import session_date

NAME = "govgreed_signals"

#: Tier ordering, best first. Anything unrecognized sorts worst.
TIER_RANK = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}

PARAMS = {
    "source": "govgreed",
    # Which Event.kind carries each of the three inputs. Parameterized rather
    # than hardcoded so the same rules can be pointed at a stand-in source --
    # the synthetic adapter, or a future vendor -- without editing the logic.
    "kind_signal": "signal",
    "kind_herd": "herd",
    "kind_insider": "insider",
    "min_tier": "A",            # composite tier floor for a candidate
    "require_fresh": True,      # the vendor's per-signal freshness flag
    "herd_min_tier": "B",       # corroboration path 1
    "insider_min_score": 6.0,   # corroboration path 2
    "signal_lookback_days": 5,  # how stale a pull may be and still count
    "target_pct": 0.05,         # <= 5% of equity per position (gate clips too)
    "max_new_per_day": 3,       # pace entries; the gate caps the book, not the flow
    "time_stop_days": 30,       # trading days
    "stop_loss_pct": 0.12,
    "take_profit_pct": 0.0,     # 0 disables
    "cooldown_days": 5,         # do not re-enter straight after an exit
    "exit_on_flip": True,
}


def _tier_ok(tier: str | None, floor: str) -> bool:
    if not tier:
        return False
    return TIER_RANK.get(str(tier).upper(), 99) <= TIER_RANK.get(floor.upper(), 99)


def _is_buy(direction: str | None) -> bool:
    return str(direction or "").strip().upper() in {"BUY", "LONG", "BULLISH", "ACCUMULATE"}


def _is_sell(direction: str | None) -> bool:
    return str(direction or "").strip().upper() in {"SELL", "SHORT", "BEARISH", "DISTRIBUTE"}


class GovGreedSignals:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(PARAMS) | dict(params or {})
        self._entry_bar: dict[str, int] = {}
        self._entry_price: dict[str, float] = {}
        self._cooldown_until: dict[str, date] = {}
        self._bar = 0

    # --- signal assembly ----------------------------------------------------

    def _pull(self, ctx: Context, kind: str) -> dict[str, Any]:
        """Latest event per ticker for one signal kind, within the lookback."""
        events = ctx.signals(
            ctx.params["source"],
            kind=kind,
            lookback_days=float(ctx.params["signal_lookback_days"]),
        )
        latest: dict[str, Any] = {}
        for e in events:  # already ordered by knowledge_time
            latest[e.ticker] = e
        return latest

    def _candidates(self, ctx: Context) -> dict[str, dict[str, Any]]:
        p = ctx.params
        composites = self._pull(ctx, str(p["kind_signal"]))
        herd = self._pull(ctx, str(p["kind_herd"]))
        insiders = self._pull(ctx, str(p["kind_insider"]))

        out: dict[str, dict[str, Any]] = {}
        for ticker, sig in composites.items():
            if not _tier_ok(sig.tier, str(p["min_tier"])):
                continue
            if p["require_fresh"] and sig.fresh is False:
                continue
            if not _is_buy(sig.direction):
                continue

            h = herd.get(ticker)
            herd_ok = bool(
                h and _tier_ok(h.tier, str(p["herd_min_tier"])) and _is_buy(h.direction)
            )
            i = insiders.get(ticker)
            insider_ok = bool(
                i and i.score is not None and float(i.score) >= float(p["insider_min_score"])
            )
            if not (herd_ok or insider_ok):
                continue

            reasons = []
            if herd_ok:
                reasons.append(f"herd {h.tier}")
            if insider_ok:
                reasons.append(f"insider {float(i.score):.1f}")
            out[ticker] = {
                "signal": sig,
                "score": float(sig.score or 0.0),
                "reason": f"tier {sig.tier} + " + " + ".join(reasons),
            }
        return out

    # --- the decision -------------------------------------------------------

    def on_bar(self, ctx: Context) -> None:
        self._bar += 1
        p = ctx.params
        today = session_date(ctx.now)

        candidates = self._candidates(ctx)
        held = dict(ctx.portfolio.positions)

        # Direction flips are the one exit that needs the full unfiltered feed:
        # a SELL signal on a name we hold matters even if it fails the entry bar.
        flipped: set[str] = set()
        if p["exit_on_flip"] and held:
            for e in ctx.signals(
                p["source"], kind=str(p["kind_signal"]),
                lookback_days=float(p["signal_lookback_days"]),
            ):
                if e.ticker in held and _is_sell(e.direction):
                    flipped.add(e.ticker)

        # --- exits ----------------------------------------------------------
        for ticker, pos in sorted(held.items()):
            price = ctx.price(ticker)
            if price is None:
                continue
            entry = self._entry_price.get(ticker, pos.avg_price or price)
            change = (price / entry - 1.0) if entry else 0.0
            bars_held = self._bar - self._entry_bar.get(ticker, self._bar)

            reason = None
            if ticker in flipped:
                reason = "signal flipped to SELL"
            elif float(p["stop_loss_pct"]) > 0 and change <= -float(p["stop_loss_pct"]):
                reason = f"stop loss {change:+.1%}"
            elif float(p["take_profit_pct"]) > 0 and change >= float(p["take_profit_pct"]):
                reason = f"take profit {change:+.1%}"
            elif int(p["time_stop_days"]) > 0 and bars_held >= int(p["time_stop_days"]):
                reason = f"time stop ({bars_held} bars)"

            if reason:
                ctx.close(ticker, tag="exit", reason=reason)
                self._entry_bar.pop(ticker, None)
                self._entry_price.pop(ticker, None)
                cooldown = int(p["cooldown_days"])
                if cooldown > 0:
                    self._cooldown_until[ticker] = today + _days(cooldown)
                candidates.pop(ticker, None)

        # --- entries ----------------------------------------------------------
        ranked = sorted(candidates, key=lambda t: candidates[t]["score"], reverse=True)
        opened = 0
        for ticker in ranked:
            if opened >= int(p["max_new_per_day"]):
                break
            if ticker in held:
                continue
            until = self._cooldown_until.get(ticker)
            if until and today < until:
                ctx.log(event="cooldown_skip", ticker=ticker, until=until.isoformat())
                continue
            if ctx.price(ticker) is None:
                continue

            info = candidates[ticker]
            ctx.order_target_pct(
                ticker, float(p["target_pct"]), tag="entry", reason=info["reason"],
                tier=info["signal"].tier,
                knowledge_time=info["signal"].knowledge_time.isoformat(),
                disclosure_lag_days=round(
                    (info["signal"].knowledge_time - info["signal"].event_time).days, 1
                ),
            )
            self._entry_bar[ticker] = self._bar
            self._entry_price[ticker] = ctx.price(ticker) or 0.0
            opened += 1

        if candidates or held:
            ctx.log(
                event="scan",
                candidates=len(candidates),
                held=len(held),
                opened=opened,
                top=[{"t": t, "s": round(candidates[t]["score"], 2)} for t in ranked[:5]],
            )


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)


STRATEGY = GovGreedSignals()
