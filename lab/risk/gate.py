"""The risk gate: the deterministic wall between *any* strategy's intents and
the broker.

The gate is strategy-independent on purpose. A human-written momentum file and
an LLM picking tickers hit exactly the same rules in exactly the same order, so
"the agent proposes, the gate disposes" is a structural property rather than a
convention someone has to remember. Model output is untrusted input and is
validated like any other input.

Two invariants are worth stating before the code, because getting either wrong
is how a risk system becomes the risk:

1. **An exit is never blocked by a capacity rule.** A target of ``0`` -- or any
   reduction of an existing position -- is subject only to the kill switch. A
   tripped breaker stops new and increasing exposure while leaving the exit path
   open, because a breaker that traps the book is worse than no breaker.
2. **``approved_pct`` on a block is the position's *current* weight, not zero.**
   Callers turn ``approved_pct`` into a delta; a zero here would liquidate the
   very position the gate just declined to change.

Rules fire in the fixed order named by ``RULES`` and stop at the first block.
A cap that can be partially honoured clips (``approved_pct`` becomes the largest
allowed value); one that cannot -- denylist, cooldown, a new position past
``max_positions``, an order under ``min_order_notional`` -- blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from lab.config import get_settings
from lab.engine.events import GateAction, GateVerdict, Intent, Order, Side
from lab.engine.protocols import PortfolioView
from lab.timeutil import session_date, to_utc, utcnow

#: Rule names, in evaluation order. These strings are a contract: they land in
#: ``GateVerdict.rule``, the decision journal, alerts and the console filter, so
#: they are matched on downstream and must not drift.
RULES: tuple[str, ...] = (
    "kill_switch",
    "breaker",
    "denylist",
    "allowlist",
    "cooldown",
    "position_cap",
    "position_notional",
    "max_positions",
    "sector_positions",
    "sector_pct",
    "gross_exposure",
    "max_orders_per_day",
    "min_notional",
)

#: Prefix marking a breaker trip the *daily* loss rule caused. A new session
#: clears those (the loss budget is per-day); a manual ``trip()`` latches until
#: a human calls ``reset()``.
DAILY_LOSS_TRIP = "daily_loss"

# Weights and dollar amounts are compared with a tolerance so float noise in
# equity arithmetic cannot manufacture a phantom clip.
_EPS = 1e-9


def _number(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive(name: str, value: Any) -> float:
    out = _number(name, value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def _non_negative(name: str, value: Any) -> float:
    out = _number(name, value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def _count(name: str, value: Any) -> int:
    out = _number(name, value)
    if out != int(out):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(out)


def _tickers(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of tickers, got {value!r}")
    out: list[str] = []
    for item in value:
        sym = str(item).strip().upper()
        if not sym:
            raise ValueError(f"{name} contains an empty ticker")
        if sym not in out:
            out.append(sym)
    return out


@dataclass
class RiskLimits:
    """Declarative limits. Validated on construction so a typo in a YAML config
    fails at load time rather than at the first order of a live session."""

    max_position_pct: float = 0.05
    max_positions: int = 8
    max_sector_positions: int = 2
    max_sector_pct: float = 0.25
    max_gross_exposure: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_orders_per_day: int = 20
    min_order_notional: float = 50.0
    allowlist: list[str] = field(default_factory=list)
    denylist: list[str] = field(default_factory=list)
    cooldown_days: int = 0
    max_position_notional: float | None = None

    def __post_init__(self) -> None:
        self.max_position_pct = _positive("max_position_pct", self.max_position_pct)
        self.max_positions = _count("max_positions", self.max_positions)
        self.max_sector_positions = _count("max_sector_positions", self.max_sector_positions)
        self.max_sector_pct = _positive("max_sector_pct", self.max_sector_pct)
        self.max_gross_exposure = _positive("max_gross_exposure", self.max_gross_exposure)
        self.max_daily_loss_pct = _positive("max_daily_loss_pct", self.max_daily_loss_pct)
        if self.max_daily_loss_pct > 1.0:
            raise ValueError("max_daily_loss_pct is a fraction of equity; > 1.0 can never trip")
        self.max_orders_per_day = _count("max_orders_per_day", self.max_orders_per_day)
        self.min_order_notional = _non_negative("min_order_notional", self.min_order_notional)
        self.allowlist = _tickers("allowlist", self.allowlist)
        self.denylist = _tickers("denylist", self.denylist)
        self.cooldown_days = _count("cooldown_days", self.cooldown_days)
        if self.max_position_notional is not None:
            self.max_position_notional = _positive(
                "max_position_notional", self.max_position_notional
            )

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> "RiskLimits":
        if m is None:
            return cls()
        if not isinstance(m, Mapping):
            raise ValueError(f"risk limits must be a mapping, got {type(m).__name__}")
        known = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(map(str, m)) - known)
        if unknown:
            # Silently ignoring a mistyped key would silently widen a limit.
            raise ValueError(
                f"unknown risk limit key(s): {', '.join(unknown)}; "
                f"known keys: {', '.join(sorted(known))}"
            )
        # An explicit null means "leave at the default", which is the only sane
        # reading of a commented-out-then-blanked YAML line.
        return cls(**{str(k): v for k, v in m.items() if v is not None})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskLimits":
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"no such limits file: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{p} must contain a mapping of limits")
        body = raw.get("limits", raw)
        if not isinstance(body, Mapping):
            raise ValueError(f"{p}: 'limits' must be a mapping")
        # Per-strategy override blocks live alongside the defaults; layering them
        # is the caller's job (it knows which strategy is running), so they are
        # not an unknown key here.
        return cls.from_mapping({k: v for k, v in body.items() if k != "strategies"})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclass_fields(self)}


@dataclass
class GateState:
    """Mutable per-session state. Small and picklable so a live runner can
    persist it across a crash and resume with the day's budget intact."""

    day: date | None = None
    orders_today: int = 0
    day_start_equity: float = 0.0
    breaker_tripped: bool = False
    breaker_reason: str = ""
    #: ticker -> session date of the most recent reduction, for the cooldown.
    exited: dict[str, date] = field(default_factory=dict)


class RiskGate:
    def __init__(
        self,
        limits: RiskLimits,
        *,
        sectors: Mapping[str, str] | None = None,
        state: GateState | None = None,
    ) -> None:
        if not isinstance(limits, RiskLimits):
            raise ValueError("limits must be a RiskLimits instance")
        if state is not None and not isinstance(state, GateState):
            raise ValueError("state must be a GateState instance")
        self.limits = limits
        self.sectors: dict[str, str] = {}
        for ticker, sector in (sectors or {}).items():
            name = "" if sector is None else str(sector).strip()
            if name:
                self.sectors[str(ticker).strip().upper()] = name
        self.state = state if state is not None else GateState()
        # Order ids already charged against today's budget. A resubmit after a
        # timeout must not eat the budget twice; not part of GateState because
        # it is a de-dup cache, not accounting.
        self._counted: set[str] = set()

    # --- session / breaker ---------------------------------------------------

    def roll_day(self, now: datetime, equity: float) -> None:
        """Start a new session: zero the order budget, re-arm the loss breaker.

        Idempotent within a session date, so calling it every bar is free.

        A state whose ``day`` is ``None`` is *armed*, not rolled: an unstamped
        state is either brand new or restored from a crash mid-session, and
        handing a restored state a fresh order budget is exactly the failure
        that lets a wedged runner double its daily order count.
        """
        session = session_date(to_utc(now))
        eq = _positive("equity", equity)
        if self.state.day == session:
            return
        rolled = self.state.day is not None
        self.state.day = session
        if rolled:
            self.state.orders_today = 0
            self._counted.clear()
        if rolled or self.state.day_start_equity <= 0:
            self.state.day_start_equity = eq
        if rolled and self.state.breaker_tripped:
            if self.state.breaker_reason.startswith(DAILY_LOSS_TRIP):
                # The daily-loss budget is per-session; a manual trip is not.
                self.state.breaker_tripped = False
                self.state.breaker_reason = ""
        if rolled:
            keep = self.limits.cooldown_days
            self.state.exited = {
                t: d for t, d in self.state.exited.items() if (session - d).days <= keep
            }

    def trip(self, reason: str) -> None:
        self.state.breaker_tripped = True
        self.state.breaker_reason = str(reason).strip() or "manual trip"

    def reset(self) -> None:
        """Clear the breaker and the whole session state.

        Deliberately manual: resuming after a trip is a human decision behind
        the promotion checklist, so nothing in the automated path calls this.
        """
        self.state.day = None
        self.state.orders_today = 0
        self.state.day_start_equity = 0.0
        self.state.breaker_tripped = False
        self.state.breaker_reason = ""
        self.state.exited.clear()
        self._counted.clear()

    @property
    def tripped(self) -> bool:
        return self.state.breaker_tripped

    def _check_breaker(self, equity: float) -> None:
        base = self.state.day_start_equity
        if self.state.breaker_tripped or base <= 0:
            return
        drawdown = (equity - base) / base
        if drawdown <= -self.limits.max_daily_loss_pct:
            self.state.breaker_tripped = True
            self.state.breaker_reason = (
                f"{DAILY_LOSS_TRIP}: equity {drawdown:+.2%} on the session "
                f"(limit {-self.limits.max_daily_loss_pct:.2%})"
            )

    # --- order accounting ----------------------------------------------------

    def note_orders(self, orders: Sequence[Order]) -> None:
        """Charge orders against the day's budget, after they are built.

        Separate from ``evaluate`` so that evaluating is side-effect free and a
        caller that decides not to submit does not burn budget.
        """
        for order in orders or ():
            ticker = str(getattr(order, "ticker", "") or "").strip().upper()
            if not ticker:
                raise ValueError(f"order {order!r} has no ticker")
            qty = _number("order qty", getattr(order, "qty", 0.0))
            if abs(qty) <= _EPS:
                continue
            oid = str(getattr(order, "id", "") or "")
            if oid:
                if oid in self._counted:
                    continue
                self._counted.add(oid)
            self.state.orders_today += 1
            side = getattr(order, "side", None)
            side_s = side.value if isinstance(side, Side) else str(side or "").lower()
            if side_s == Side.SELL.value:
                # Any reduction starts the cooldown clock: the rule exists to stop
                # churn, and "sold then immediately re-bought" is the churn.
                created = getattr(order, "created_at", None)
                self.note_exit(ticker, created)

    def note_exit(self, ticker: str, when: datetime | None = None) -> None:
        """Record a reduction/close for the cooldown rule."""
        sym = str(ticker).strip().upper()
        if not sym:
            raise ValueError("ticker is empty")
        if when is not None:
            day = session_date(to_utc(when))
        else:
            day = self.state.day or session_date(utcnow())
        self.state.exited[sym] = day

    # --- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        intents: Sequence[Intent],
        *,
        portfolio: PortfolioView,
        now: datetime,
        prices: Mapping[str, float],
    ) -> list[GateVerdict]:
        """One verdict per intent, in the order given.

        Pure with respect to state except the breaker (which must be able to
        trip mid-evaluation) and the session roll that arms it. Evaluating the
        same inputs twice therefore yields the same verdicts.
        """
        if intents is None:
            raise ValueError("intents must be a sequence, got None")
        for attr in ("equity", "positions", "weight"):
            if not hasattr(portfolio, attr):
                raise ValueError(f"portfolio is not a PortfolioView: missing {attr!r}")

        ts = to_utc(now)
        session = session_date(ts)
        equity = _number("portfolio equity", portfolio.equity)
        if equity <= 0:
            raise ValueError("portfolio equity must be > 0 to size weight targets")
        px = self._clean_prices(prices)

        # Arm the session here too: a runner that forgets roll_day would
        # otherwise run all day with a silently disabled loss breaker.
        self.roll_day(ts, equity)
        self._check_breaker(equity)

        engaged, kill_reason = get_settings().kill_switch_engaged()
        weights = self._projected_weights(portfolio, px, equity)
        budget = max(self.limits.max_orders_per_day - self.state.orders_today, 0)

        verdicts: list[GateVerdict] = []
        for intent in intents:
            ticker, target = _read_intent(intent)
            current = weights.get(ticker, 0.0)
            verdict = self._evaluate_one(
                ticker=ticker,
                target=target,
                current=current,
                weights=weights,
                equity=equity,
                session=session,
                budget=budget,
                killed=engaged,
                kill_reason=kill_reason,
            )
            # Later intents in the same batch see earlier approvals, so a basket
            # cannot slip past a portfolio-level cap one name at a time.
            weights[ticker] = verdict.approved_pct
            if not verdict.blocked and abs(verdict.approved_pct - current) * equity > _EPS:
                budget = max(budget - 1, 0)
            verdicts.append(verdict)
        return verdicts

    def sector_of(self, ticker: str) -> str | None:
        return self.sectors.get(str(ticker).strip().upper())

    # --- internals -----------------------------------------------------------

    @staticmethod
    def _clean_prices(prices: Mapping[str, float] | None) -> dict[str, float]:
        if prices is None:
            return {}
        if not isinstance(prices, Mapping):
            raise ValueError(f"prices must be a mapping, got {type(prices).__name__}")
        out: dict[str, float] = {}
        for ticker, value in prices.items():
            sym = str(ticker).strip().upper()
            price = _number(f"price for {sym}", value)
            if price <= 0:
                raise ValueError(f"price for {sym} must be > 0, got {price!r}")
            out[sym] = price
        return out

    @staticmethod
    def _projected_weights(
        portfolio: PortfolioView, prices: Mapping[str, float], equity: float
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for key, position in (portfolio.positions or {}).items():
            ticker = str(key).strip().upper()
            weight = _number(f"weight for {ticker}", portfolio.weight(key))
            qty = _number(f"qty for {ticker}", getattr(position, "qty", 0.0) or 0.0)
            if abs(weight) <= _EPS and abs(qty) > _EPS and ticker in prices:
                # An unmarked position still occupies a slot and eats capacity.
                weight = qty * prices[ticker] / equity
            weights[ticker] = weight
        return weights

    @staticmethod
    def _is_exit(target: float, current: float) -> bool:
        if abs(target) <= _EPS:
            return True
        if abs(current) <= _EPS:
            return False
        same_sign = (target > 0) == (current > 0)
        return same_sign and abs(target) < abs(current) - _EPS

    def _evaluate_one(
        self,
        *,
        ticker: str,
        target: float,
        current: float,
        weights: Mapping[str, float],
        equity: float,
        session: date,
        budget: int,
        killed: bool,
        kill_reason: str | None,
    ) -> GateVerdict:
        limits = self.limits

        def block(rule: str, detail: str) -> GateVerdict:
            return GateVerdict(
                ticker=ticker,
                action=GateAction.BLOCK,
                requested_pct=target,
                approved_pct=current,  # hold, never liquidate
                rule=rule,
                detail=detail,
            )

        if killed:
            return block("kill_switch", kill_reason or "kill switch engaged")

        if self._is_exit(target, current):
            return GateVerdict(
                ticker=ticker,
                action=GateAction.PASS,
                requested_pct=target,
                approved_pct=target,
                rule=None,
                detail="exit: capacity rules do not apply",
            )

        if self.state.breaker_tripped:
            return block("breaker", self.state.breaker_reason or "circuit breaker tripped")
        if ticker in limits.denylist:
            return block("denylist", f"{ticker} is on the denylist")
        if limits.allowlist and ticker not in limits.allowlist:
            return block("allowlist", f"{ticker} is not on the {len(limits.allowlist)}-name allowlist")
        last_exit = self.state.exited.get(ticker)
        if last_exit is not None and limits.cooldown_days > 0:
            elapsed = (session - last_exit).days
            if elapsed < limits.cooldown_days:
                return block(
                    "cooldown",
                    f"reduced {ticker} {elapsed}d ago; cooldown is {limits.cooldown_days}d",
                )

        sign = -1.0 if target < 0 else 1.0
        magnitude = abs(target)
        clipped_by: str | None = None
        detail = ""

        def clip(allowed: float, rule: str, why: Callable[[float], str]) -> None:
            nonlocal magnitude, clipped_by, detail
            capped = max(allowed, 0.0)
            if capped < magnitude - _EPS:
                magnitude = capped
                clipped_by = rule
                detail = why(capped)

        clip(
            limits.max_position_pct,
            "position_cap",
            lambda cap: f"{abs(target):.2%} over per-position cap {cap:.2%}",
        )
        if limits.max_position_notional is not None:
            clip(
                limits.max_position_notional / equity,
                "position_notional",
                lambda cap: (
                    f"capped at ${limits.max_position_notional:,.0f} of "
                    f"${equity:,.0f} equity ({cap:.2%})"
                ),
            )

        is_new = abs(current) <= _EPS
        opens_slot = is_new and magnitude > _EPS
        if opens_slot:
            held = sum(1 for t, w in weights.items() if t != ticker and abs(w) > _EPS)
            if held >= limits.max_positions:
                return block(
                    "max_positions",
                    f"{held} open position(s) at cap {limits.max_positions}",
                )

        sector = self.sectors.get(ticker)
        if sector:
            # A ticker with no known sector is exempt rather than lumped into a
            # bucket it does not belong to.
            peers = [
                t
                for t, w in weights.items()
                if t != ticker and abs(w) > _EPS and self.sectors.get(t) == sector
            ]
            if opens_slot and len(peers) >= limits.max_sector_positions:
                return block(
                    "sector_positions",
                    f"{len(peers)} position(s) in {sector} at cap {limits.max_sector_positions}",
                )
            used = sum(abs(weights[t]) for t in peers)
            clip(
                limits.max_sector_pct - used,
                "sector_pct",
                lambda room: (
                    f"{sector} at {used:.2%} of {limits.max_sector_pct:.2%}; "
                    f"room for {room:.2%}"
                ),
            )

        gross_others = sum(abs(w) for t, w in weights.items() if t != ticker)
        clip(
            limits.max_gross_exposure - gross_others,
            "gross_exposure",
            lambda room: (
                f"gross {gross_others:.2%} of {limits.max_gross_exposure:.2%}; "
                f"room for {room:.2%}"
            ),
        )

        approved = sign * magnitude
        notional = abs(approved - current) * equity
        was_clipped = clipped_by is not None
        # A no-op hold (target already met, nothing clipped) needs no order, so
        # the order-budget and minimum-size rules have nothing to say about it.
        if notional > _EPS or was_clipped:
            if notional > _EPS and budget <= 0:
                return block(
                    "max_orders_per_day",
                    f"{self.state.orders_today} order(s) already today, cap {limits.max_orders_per_day}",
                )
            if notional < limits.min_order_notional - _EPS:
                tail = f" after the {clipped_by} clip" if was_clipped else ""
                return block(
                    "min_notional",
                    f"${notional:,.2f} order{tail} is under the "
                    f"${limits.min_order_notional:,.2f} minimum",
                )

        return GateVerdict(
            ticker=ticker,
            action=GateAction.CLIP if was_clipped else GateAction.PASS,
            requested_pct=target,
            approved_pct=approved,
            rule=clipped_by,
            detail=detail,
        )


def _read_intent(intent: Intent) -> tuple[str, float]:
    """Untrusted-input boundary: an agent-authored intent gets the same scrutiny
    as a hand-written one."""
    ticker = getattr(intent, "ticker", None)
    target = getattr(intent, "target_pct", None)
    if ticker is None or target is None:
        raise ValueError(f"intent {intent!r} must expose ticker and target_pct")
    sym = str(ticker).strip().upper()
    if not sym:
        raise ValueError("intent ticker is empty")
    return sym, _number(f"{sym} target_pct", target)
