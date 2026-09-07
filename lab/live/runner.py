"""The live/paper runner: the backtester's loop, with a wall clock and a real broker.

The platform's central promise is that one strategy file runs unchanged in both
places. That promise is only as good as this file: the per-step sequence below
mirrors ``lab.backtest.runner.run_backtest`` bar for bar -- poll fills, dispatch
``on_fill``, mark, roll the session, decide, gate, size, submit, journal -- and
it reuses the *same* ``EngineContext``, ``RiskGate``, ``Portfolio`` and
``DataView``, plus the backtester's own order-building function. Any divergence
here silently decouples paper results from backtest results, which would destroy
the one property the whole lab is built to keep.

Three operational rules, none of them optional:

**Reconcile first.** Startup fetches broker positions and open orders, diffs
them against local state, and refuses to trade until they agree or a human
acknowledges. See ``reconcile.py``.

**Idempotent orders.** ``idem_key`` is ``(strategy, session_date, ticker,
side)``, checked against the persisted order table *and* enforced by the broker
adapter, so a runner that crashed after submitting cannot re-fire on resume.

**Kill switch before every batch.** Env flag or sentinel file, re-checked
between building an order batch and sending it, so a panic-stop lands even
mid-decision. Under the switch, orders that *reduce* an existing position still
go through -- a stop that traps the book is worse than no stop.

**Promotion to live capital is not automated anywhere in this module.** The
paper endpoint is the default and the only tested path; going live requires
``paper: false`` plus the explicit two-part opt-in in
``lab.engine.broker_alpaca`` (``allow_live=True`` and ``LAB_ALLOW_LIVE_TRADING``
in the environment). It is a human checklist decision -- minimum paper
duration, out-of-sample metrics, gate-trip review, sizing sanity -- every time.
"""

from __future__ import annotations

import logging
import os
import socket
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lab.backtest.runner import MIN_SHARES, _build_orders, load_universe
from lab.engine.clock import WallClock
from lab.engine.context import DataView, EngineContext
from lab.engine.events import (
    Decision,
    EventKind,
    Fill,
    GateAction,
    GateVerdict,
    Order,
    OrderStatus,
    Side,
    Trade,
    event,
    new_id,
)
from lab.engine.loader import LoadedStrategy, load_strategy
from lab.engine.portfolio import Portfolio, ReadOnlyPortfolio
from lab.live import alerts, killswitch
from lab.live.reconcile import ReconcileResult, reconcile
from lab.registry.db import connect, journal_path, transaction
from lab.registry.journal import DecisionJournal, EventJournal
from lab.risk.gate import RiskGate, RiskLimits
from lab.timeutil import session_date, to_utc, trading_days, utcnow

log = logging.getLogger(__name__)

#: Heartbeat sources. Fixed strings: the console's freshness strip keys off them.
HEARTBEAT_SOURCES = ("runner", "data", "broker")

#: Backtest-only config keys a live config accepts and ignores. One YAML file
#: has to drive both runners -- that is the promotion path -- so a fill model or
#: a benchmark in the file is not a typo, it is simply not live's business.
BACKTEST_ONLY_KEYS = frozenset(
    {"fills", "benchmark", "start", "end", "seed", "windows", "sweep_id", "contaminated"}
)


@dataclass
class LiveConfig:
    """A backtest config's live twin. Same YAML, different half of the keys."""

    strategy: str
    tickers: list[str] = field(default_factory=list)
    timeframe: str = "1d"
    broker: str = "alpaca"
    paper: bool = True
    at_time: str = "09:35"
    cash: float = 100_000.0
    params: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    #: Which adapter's bars to trade. Not backtest-only: the live runner reads
    #: the same store, and trading a series interleaved from two feeds would be
    #: worse live than in a backtest, not better.
    source: str | None = None
    regular_hours: bool = True
    poll_seconds: int = 30
    lookback_days: int = 400
    warmup: int = 0
    origin: str = "human"
    notes: str = ""
    strict: bool = True
    allow_fractional: bool = False
    allow_live: bool = False
    calendar: bool = True
    #: Refuse to decide when the newest bar is more than this many *trading
    #: sessions* old. Counted in sessions rather than hours because a Friday bar
    #: is three days old on Monday morning and perfectly fresh. Staleness used to
    #: be reported in the heartbeat and enforced nowhere, so a runner whose data
    #: pull had been failing for a week went on trading against week-old prices
    #: without complaint. 0 disables the guard.
    #: Seconds a one-shot run waits for the orders it just sent to fill before
    #: saving the book and exiting. Without it the saved book is always
    #: pre-fill: the process submits, exits, the fills land at the broker a
    #: second later, and tomorrow's startup sees a book that disagrees with the
    #: account and refuses to trade. A daily bot would then need a human every
    #: single morning, which is the failure this whole path exists to remove.
    settle_seconds: int = 90
    max_stale_sessions: int = 2
    #: Adopt the broker's positions into a disagreeing local book at startup
    #: instead of refusing to trade. The operator's "I have looked, the broker is
    #: right" -- never a default.
    adopt: bool = False
    parent_run_id: str | None = None
    ignored_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tickers = [str(t).upper() for t in self.tickers]
        self.sectors = {str(k).upper(): str(v) for k, v in (self.sectors or {}).items()}
        self.poll_seconds = int(self.poll_seconds)
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")
        if self.cash <= 0:
            raise ValueError("cash must be > 0")
        if not self.paper and not self.allow_live:
            # Not an error yet -- the broker adapter is the enforcement point --
            # but the config that asks for real money should say so out loud.
            log.warning(
                "config requests non-paper trading; this needs allow_live plus "
                "LAB_ALLOW_LIVE_TRADING and stays a manual human decision"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LiveConfig":
        data = dict(raw)
        tickers = data.get("tickers") or data.get("universe") or []
        if isinstance(tickers, str):
            tickers = load_universe(tickers)
        data.pop("universe", None)
        data["tickers"] = list(tickers)
        known = set(cls.__dataclass_fields__)
        ignored = sorted(k for k in data if k in BACKTEST_ONLY_KEYS)
        unknown = sorted(set(data) - known - BACKTEST_ONLY_KEYS)
        if unknown:
            raise ValueError(
                f"unknown live config keys: {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(known))}"
            )
        cfg = cls(**{k: v for k, v in data.items() if k in known and k != "ignored_keys"})
        cfg.ignored_keys = ignored
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Mapping[str, Any] | None = None) -> "LiveConfig":
        import yaml

        raw: dict[str, Any] = {}
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"no config at {p}")
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveRunner:
    """One strategy, one broker, one wall clock."""

    def __init__(
        self,
        cfg: LiveConfig,
        *,
        broker: Any | None = None,
        data: DataView | None = None,
        clock: Any | None = None,
        strategy: LoadedStrategy | None = None,
        register: bool = True,
        events: EventJournal | None = None,
        decisions: DecisionJournal | None = None,
        max_consecutive_errors: int = 5,
    ) -> None:
        self.cfg = cfg
        self.register = bool(register)
        self.max_consecutive_errors = int(max_consecutive_errors)

        self.strategy = strategy or load_strategy(cfg.strategy, cfg.params)
        self.params = self.strategy.params
        self.portfolio = Portfolio(cfg.cash)
        self.view = ReadOnlyPortfolio(self.portfolio)
        limits = RiskLimits.from_mapping(cfg.limits) if cfg.limits else RiskLimits()
        self.gate = RiskGate(limits, sectors=cfg.sectors)
        self.broker = broker if broker is not None else _make_broker(cfg, self.portfolio)
        self.clock = clock if clock is not None else WallClock(
            cfg.timeframe, cfg.at_time, calendar=cfg.calendar
        )
        self.events = events if events is not None else EventJournal()
        self.decisions = decisions if decisions is not None else DecisionJournal()

        self._fixed_data = data
        self._universe: list[str] = list(cfg.tickers) or (data.tickers() if data else [])
        self._on_start = self.strategy.hook("on_start")
        self._on_bar = self.strategy.hook("on_bar")
        self._on_fill = self.strategy.hook("on_fill")
        self._on_stop = self.strategy.hook("on_stop")

        self.run_id: str | None = None
        self.blocked = False
        self.block_reason = ""
        self.paused = False
        self.last_reconcile: ReconcileResult | None = None
        self.started_at: datetime | None = None
        self.stopped_at: datetime | None = None
        self.last_step_at: datetime | None = None
        self.steps = 0

        self._stopped = True
        self._hooks_started = False
        self._ctx: EngineContext | None = None
        self._last_now: datetime | None = None
        self._last_view: DataView | None = None
        self._last_bar_at: datetime | None = None
        self.settle_seconds = int(getattr(cfg, "settle_seconds", 90) or 0)
        self._last_fill_poll: datetime | None = None
        self._seen_fills: set[str] = set()
        self._idem_by_session: dict[date, set[str]] = {}
        self._orders: dict[str, Order] = {}
        self._errors = 0
        self._breaker_alerted = False

    # --- lifecycle ----------------------------------------------------------

    def start(self, *, max_steps: int | None = None, once: bool = False) -> None:
        """Reconcile, then loop. Returns when the clock stops or a step budget
        is exhausted; refuses to enter the loop at all if reconciliation fails.

        ``once`` decides for the current moment and returns instead of entering
        the clock loop. That is the mode a scheduled task wants: the clock exists
        to wait until the next fire time, and a task that has *already* been woken
        at 09:35 would otherwise sleep until 09:35 tomorrow -- ``_next_daily_fire``
        treats a candidate at or before now as belonging to the next session. One
        decision per process also means a reboot costs nothing: there is no
        long-lived state to lose, because the book is on disk between runs.
        """
        self.started_at = utcnow()
        self._ensure_run()
        self._journal(
            EventKind.RUN_START,
            "runner",
            f"{self.strategy.name} -> {getattr(self.broker, 'name', '?')} "
            f"({'paper' if self.cfg.paper else 'LIVE'}), {len(self._universe)} tickers",
            payload={"config": self.cfg.to_dict()},
        )

        # Before reconciling, not after: see restore_book's docstring.
        self.restore_book()

        result = self.reconcile(acknowledge=bool(self.cfg.adopt))
        if not result.ok:
            self._block(result.message)
            self._upsert_live_row("blocked")
            return

        # The book -- restored or adopted -- is the state as of *now*, which means
        # every fill that produced it has already been counted. Without this
        # watermark the first poll asks for the broker's entire fill history and
        # applies it on top, so an adopted 1-share position immediately becomes 2
        # and the next exit order is rejected for selling more than exists.
        #
        # A fill that landed between the last save and this startup is therefore
        # not picked up here -- it is caught by reconciliation as a qty_mismatch,
        # which is the check that exists for exactly that gap. The book is the
        # fast path; the broker is the truth.
        self._last_fill_poll = utcnow()

        self._stopped = False
        self.blocked = False
        self._upsert_live_row("running")

        if once:
            try:
                self.step(utcnow())
            except Exception as exc:  # noqa: BLE001 - report it, do not traceback
                self._errors += 1
                alerts.alert_error(
                    exc, run_id=self.run_id, strategy=self.strategy.name,
                    at=utcnow(), consecutive=self._errors, journal=self.events,
                )
            else:
                # Settle before saving, or the book records intent rather than
                # outcome and tomorrow's startup refuses on a diff we caused.
                self.settle()
                self.sync_book_from_broker()
                self._save_book()
            self.stop("single step complete", cancel_open=False)
            return

        n = 0
        for ts in self.clock:
            if self._stopped:
                break
            self.heartbeat()
            try:
                self.step(ts)
            except Exception as exc:  # noqa: BLE001 - one bad bar must not kill the runner
                self._errors += 1
                alerts.alert_error(
                    exc, run_id=self.run_id, strategy=self.strategy.name, at=ts,
                    consecutive=self._errors, journal=self.events,
                )
                if self._errors >= self.max_consecutive_errors:
                    self.stop(f"{self._errors} consecutive step errors")
                    break
            else:
                self._errors = 0
                self._save_book()
            n += 1
            if max_steps is not None and n >= max_steps:
                break
        if not self._stopped:
            self.stop("clock exhausted")

    def stop(self, reason: str = "", *, cancel_open: bool = True) -> None:
        """Shut down. Working orders are cancelled by default: an order with no
        runner watching it is an unsupervised position."""
        if self._stopped and self.stopped_at is not None:
            return
        self._stopped = True
        self.stopped_at = utcnow()
        stop_clock = getattr(self.clock, "stop", None)
        if callable(stop_clock):
            stop_clock()

        cancelled = 0
        if cancel_open:
            try:
                for order in self.broker.open_orders() or []:
                    cancelled += int(bool(self.broker.cancel(order.id)))
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                log.warning("cancel-on-stop failed: %s", exc)

        if self._on_stop is not None and self._ctx is not None:
            try:
                self._on_stop(self._ctx)
            except Exception as exc:  # noqa: BLE001
                alerts.alert_error(exc, run_id=self.run_id, strategy=self.strategy.name,
                                   hook="on_stop", journal=self.events)

        self._journal(
            EventKind.RUN_END,
            "runner",
            f"stopped: {reason or 'requested'} ({self.steps} steps, {cancelled} order(s) cancelled)",
            payload={"reason": reason, "steps": self.steps, "cancelled": cancelled},
        )
        self._upsert_live_row("stopped")
        self._finish_run(reason)

    # --- the loop body ------------------------------------------------------

    def stale_sessions(self, now: datetime) -> int | None:
        """Trading sessions between the newest bar and ``now``.

        Sessions, not hours: a Friday close is three calendar days old on Monday
        morning and completely fresh, while a Tuesday bar on Thursday is two
        sessions stale and means the data pull has been failing.
        """
        if self._last_bar_at is None:
            return None
        last = session_date(self._last_bar_at)
        today = session_date(now)
        if today <= last:
            return 0
        # trading_days is inclusive of both ends; the bar's own session does not
        # count against it, so a bar from today or yesterday scores 0 or 1.
        return max(0, len(trading_days(last, today)) - 1)

    def _refuse_if_stale(self, now: datetime) -> str:
        """Empty when the data is fresh enough to act on, else why it is not."""
        limit = int(getattr(self.cfg, "max_stale_sessions", 0) or 0)
        if limit <= 0:
            return ""
        age = self.stale_sessions(now)
        if age is None:
            return (
                "no bars at all for this universe -- the store is empty for the "
                "configured source and lookback. Run `lab pull` before trading."
            )
        if age > limit:
            return (
                f"newest bar is {age} trading session(s) old (limit {limit}); "
                f"last bar {self._last_bar_at.isoformat() if self._last_bar_at else '?'}. "
                f"Deciding on stale prices is worse than not deciding -- run "
                f"`lab pull` and start again."
            )
        return ""

    def step(self, now: datetime) -> Decision | None:
        """One bar: the backtester's sequence, against a live broker.

        Returns the ``Decision`` when the strategy got to decide, and ``None``
        when it did not (blocked, paused, no ``on_bar``, or a dead book).
        """
        now = to_utc(now)
        self.steps += 1
        self.last_step_at = now
        if self.blocked:
            self._journal(EventKind.RECONCILE, "runner",
                          f"step skipped, trading blocked: {self.block_reason}")
            return None
        self._ensure_run()

        view = self._data_view(now)
        # After _data_view, which is what learns the newest bar's timestamp.
        # Blocking rather than warning: staleness was reported in the heartbeat
        # and enforced nowhere, which is indistinguishable from fresh data right
        # up until the day it costs money.
        stale = self._refuse_if_stale(now)
        if stale:
            self._block(stale)
            self._upsert_live_row("blocked")
            return None

        ctx = self._new_context(view)
        # Fills are dispatched with the context still on the *previous* bar,
        # exactly as the backtest loop leaves it; the first step has no previous
        # bar, so it starts on `now` like the backtester's pre-loop set_now.
        ctx.set_now(self._last_now or now)
        if not self._hooks_started:
            self._hooks_started = True
            if self._on_start is not None:
                self._on_start(ctx)
                ctx.reset_bar()

        fills = self._poll_fills(now, view)
        self._dispatch_fills(fills, ctx)

        prices = self._prices(view, now)
        self.portfolio.mark(prices)
        self.portfolio.tick_bar()

        equity = self.portfolio.equity
        if equity <= 0:
            self._block(f"portfolio equity is {equity:,.2f}; refusing to trade")
            return None
        self.gate.roll_day(now, equity)
        self._note_breaker()

        self.paused = self._is_paused()
        if self.paused or self._on_bar is None:
            self._ctx, self._last_now = ctx, now
            return None

        t0 = _time.perf_counter()
        ctx.reset_bar()
        ctx.set_now(now)
        self._ctx, self._last_now = ctx, now

        self._on_bar(ctx)
        intents = ctx.drain_intents()
        verdicts = self.gate.evaluate(intents, portfolio=self.view, now=now, prices=prices)
        self._note_breaker()

        orders = _build_orders(
            verdicts,
            {i.ticker: i for i in intents},
            self.portfolio,
            prices,
            strategy=self.strategy.name,
            session=session_date(now),
            decision_id="pending",
            allow_fractional=self.cfg.allow_fractional,
            now=now,
        )
        decision = Decision(
            id=new_id("d_"),
            run_id=self.run_id or "",
            strategy=self.strategy.name,
            at=now,
            inputs=ctx.input_tape(),
            intents=intents,
            verdicts=verdicts,
            portfolio=self.view.snapshot(now),
            logs=ctx.logs(),
            duration_ms=round((_time.perf_counter() - t0) * 1000, 3),
        )
        for o in orders:
            o.decision_id = decision.id
        submitted = self.submit_orders(orders, now=now)
        decision.order_ids = [o.id for o in submitted]

        self._persist_decision(decision, verdicts, now)
        return decision

    def submit_orders(self, orders: Sequence[Order], *, now: datetime | None = None) -> list[Order]:
        """The order batch gate: idempotency, then the kill switch, then send.

        The kill switch is re-checked *here* rather than only inside the risk
        gate, because a panic-stop that lands between "decide" and "send" must
        still catch this batch.
        """
        when = to_utc(now) if now is not None else utcnow()
        session = session_date(when)
        seen = self._submitted_idem_keys(session)
        engaged, kill_reason = killswitch.engaged()

        sent: list[Order] = []
        for order in orders:
            if order.idem_key and order.idem_key in seen:
                self._journal(
                    EventKind.ORDER, "runner",
                    f"skipped duplicate {order.ticker} {order.side.value}: idem_key already used today",
                    ticker=order.ticker, payload={"idem_key": order.idem_key, "order": order.to_dict()},
                )
                continue
            if engaged:
                allowed = self._kill_switch_allows(order)
                if not allowed:
                    order.status = OrderStatus.REJECTED
                    order.reason = f"kill_switch: {kill_reason}"
                    alerts.alert_reject(order, f"kill switch engaged ({kill_reason})",
                                        run_id=self.run_id, strategy=self.strategy.name,
                                        source="runner", journal=self.events)
                    continue
            try:
                placed = self.broker.submit(order)
            except Exception as exc:  # noqa: BLE001 - a refused order is data, not a crash
                order.status = OrderStatus.REJECTED
                order.reason = f"{type(exc).__name__}: {exc}"
                alerts.alert_reject(order, order.reason, run_id=self.run_id,
                                    strategy=self.strategy.name, journal=self.events)
                continue
            placed.status = placed.status if placed.status is not OrderStatus.NEW else OrderStatus.SUBMITTED
            self._orders[placed.id] = placed
            if placed.idem_key:
                seen.add(placed.idem_key)
            sent.append(placed)

        if sent:
            self.gate.note_orders(sent)
            self._persist_orders(sent)
            for order in sent:
                alerts.alert_order(order, run_id=self.run_id, strategy=self.strategy.name,
                                   journal=self.events)
        return sent

    # --- reconciliation -----------------------------------------------------

    def reconcile(self, *, acknowledge: bool = False) -> ReconcileResult:
        result = reconcile(
            self.broker,
            self.portfolio,
            acknowledge=acknowledge,
            known_order_ids=self._known_order_ids(),
        )
        self.last_reconcile = result
        self._journal(
            EventKind.RECONCILE, "runner", result.message, payload=result.to_dict()
        )
        if result.diffs:
            alerts.alert_reconcile(result, run_id=self.run_id, strategy=self.strategy.name,
                                   journal=self.events)
        if result.ok:
            self.blocked = False
            self.block_reason = ""
        return result

    def acknowledge(self, note: str = "") -> ReconcileResult:
        """Adopt the broker's state as truth and resume trading.

        Deliberately a separate, explicitly-called method: acknowledging a
        reconciliation diff is a human decision, and nothing in the automated
        path may call it.
        """
        result = self.reconcile(acknowledge=True)
        if result.ok:
            self.blocked = False
            self.block_reason = ""
            self._upsert_live_row("running" if not self._stopped else "stopped")
            self._journal(EventKind.RECONCILE, "runner",
                          f"diffs acknowledged by operator{f': {note}' if note else ''}",
                          payload={"note": note, "adopted": result.adopted})
        return result

    # --- control ------------------------------------------------------------

    def pause(self, reason: str = "") -> None:
        self.paused = True
        self._set_paused(True)
        self._journal(EventKind.LOG, "runner", f"paused{f': {reason}' if reason else ''}")

    def resume(self) -> None:
        self.paused = False
        self._set_paused(False)
        self._journal(EventKind.LOG, "runner", "resumed")

    def cancel_open_orders(self, ticker: str | None = None) -> int:
        n = 0
        for order in self.broker.open_orders() or []:
            if ticker and order.ticker != ticker.upper():
                continue
            n += int(bool(self.broker.cancel(order.id)))
        self._journal(EventKind.ORDER, "runner", f"cancelled {n} open order(s)",
                      ticker=(ticker.upper() if ticker else None))
        return n

    # --- observability ------------------------------------------------------

    def heartbeat(self) -> None:
        """One beat per source, per poll, with the staleness metadata the
        console's freshness strip renders. Never raises."""
        now = utcnow()
        try:
            self.events.heartbeat("runner", meta={
                "strategy": self.strategy.name,
                "run_id": self.run_id,
                "state": self.state,
                "session": str(session_date(now)),
                "poll_seconds": self.cfg.poll_seconds,
                "steps": self.steps,
                "last_step": self.last_step_at.isoformat() if self.last_step_at else None,
                "step_age_s": self._age(self.last_step_at, now),
                "next_fire": self._next_fire_iso(now),
                "pid": os.getpid(),
            })
            self.events.heartbeat("data", meta={
                "timeframe": self.cfg.timeframe,
                "tickers": len(self._universe),
                "last_bar": self._last_bar_at.isoformat() if self._last_bar_at else None,
                "bar_age_s": self._age(self._last_bar_at, now),
                "sources": list(self.cfg.sources),
            "source": self.cfg.source,
                "fixed_view": self._fixed_data is not None,
            })
            self.events.heartbeat("broker", meta=self._broker_meta())
        except Exception as exc:  # noqa: BLE001 - a missed heartbeat must not stop trading
            log.warning("heartbeat failed: %s", exc)

    @property
    def state(self) -> str:
        if self.blocked:
            return "blocked"
        if self._stopped:
            return "stopped"
        return "paused" if self.paused else "running"

    def status(self) -> dict[str, Any]:
        engaged, kill_reason = killswitch.engaged()
        try:
            heartbeats = self.events.last_heartbeats()
        except Exception:  # noqa: BLE001
            heartbeats = {}
        return {
            "strategy": self.strategy.name,
            "run_id": self.run_id,
            "kind": "paper" if self.cfg.paper else "live",
            "state": self.state,
            "paper": self.cfg.paper,
            "broker": getattr(self.broker, "name", "?"),
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "paused": self.paused,
            "kill_switch": {"engaged": engaged, "reason": kill_reason},
            "breaker": {"tripped": self.gate.tripped, "reason": self.gate.state.breaker_reason},
            "orders_today": self.gate.state.orders_today,
            "session": str(self.gate.state.day) if self.gate.state.day else None,
            "equity": round(self.portfolio.equity, 2),
            "cash": round(self.portfolio.cash, 2),
            "positions": self.view.snapshot(self.last_step_at)["positions"],
            "steps": self.steps,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_step": self.last_step_at.isoformat() if self.last_step_at else None,
            "next_fire": self._next_fire_iso(utcnow()),
            "reconcile": self.last_reconcile.to_dict() if self.last_reconcile else None,
            "heartbeats": heartbeats,
            "universe": list(self._universe),
            "poll_seconds": self.cfg.poll_seconds,
        }

    # --- internals: data & context -----------------------------------------

    def _data_view(self, now: datetime) -> DataView:
        if self._fixed_data is not None:
            view = self._fixed_data
        else:
            view = DataView.from_store(
                self._universe,
                timeframe=self.cfg.timeframe,
                start=now - timedelta(days=self.cfg.lookback_days),
                end=now,
                sources=self.cfg.sources,
            bar_source=self.cfg.source,
            regular_hours=self.cfg.regular_hours,
                # Without `as_of` the store hands back the raw table, future
                # included; this is the look-ahead barrier in live mode.
                as_of=now,
            )
        self._last_view = view
        latest: datetime | None = None
        for ticker in self._universe or view.tickers():
            frame = view.bars(ticker, 1, now)
            if len(frame):
                stamp = to_utc(frame.index[-1])
                latest = stamp if latest is None or stamp > latest else latest
        self._last_bar_at = latest
        return view

    def _new_context(self, view: DataView) -> EngineContext:
        return EngineContext(
            run_id=self.run_id or "pending",
            strategy=self.strategy.name,
            params=self.params,
            universe=self._universe or view.tickers(),
            portfolio=self.view,
            data=view,
            timeframe=self.cfg.timeframe,
            strict=self.cfg.strict,
        )

    def _prices(self, view: DataView, now: datetime) -> dict[str, float]:
        """Latest *knowable* close per ticker.

        Not ``bars_at(now)``: a live runner firing at 09:35 must not mark
        against a bar whose knowledge_time is this afternoon's close.
        """
        out: dict[str, float] = {}
        for ticker in set(self._universe) | set(self.portfolio.positions):
            price = view.price(ticker, now)
            if price is not None and price > 0:
                out[ticker.upper()] = float(price)
        return out

    # --- internals: fills ---------------------------------------------------

    def _poll_fills(self, now: datetime, view: DataView) -> list[Fill]:
        raw: Iterable[Fill] = ()
        applied = False
        poll = getattr(self.broker, "poll_fills", None)
        process = getattr(self.broker, "process", None)
        if callable(poll):
            raw = poll(since=self._last_fill_poll) or ()
        elif callable(process):
            # A sim broker in a paper loop: it owns the fill model and applies
            # fills to the portfolio it was constructed with.
            raw = process(now, view.bars_at(now)) or ()
            applied = getattr(self.broker, "portfolio", None) is self.portfolio
        self._last_fill_poll = now

        fresh: list[Fill] = []
        for fill in raw:
            if fill.id in self._seen_fills:
                continue
            self._seen_fills.add(fill.id)
            if not applied:
                trade = self.portfolio.apply_fill(fill)
                if trade is not None:
                    self._persist_trade(trade)
            fresh.append(fill)
        return fresh

    def _dispatch_fills(self, fills: Sequence[Fill], ctx: EngineContext) -> None:
        for fill in fills:
            if self._on_fill is not None:
                try:
                    self._on_fill(ctx, fill)
                except Exception as exc:  # noqa: BLE001 - a bad hook must not lose the fill
                    alerts.alert_error(exc, run_id=self.run_id, strategy=self.strategy.name,
                                       hook="on_fill", ticker=fill.ticker, journal=self.events)
            alerts.alert_fill(fill, run_id=self.run_id, strategy=self.strategy.name,
                              journal=self.events)
        if fills:
            self._persist_fills(fills)

    # --- internals: safety --------------------------------------------------

    def _kill_switch_allows(self, order: Order) -> bool:
        """Under the kill switch, only exposure-reducing orders survive.

        The quantity is clamped to the position: a sell larger than the holding
        would flip long to short, which is new exposure wearing an exit's
        clothes.
        """
        qty = self.portfolio.position(order.ticker).qty
        if qty > MIN_SHARES and order.side is Side.SELL:
            allowed = min(order.qty, qty)
        elif qty < -MIN_SHARES and order.side is Side.BUY:
            allowed = min(order.qty, -qty)
        else:
            return False
        if allowed < order.qty:
            order.qty = allowed
            order.reason = (order.reason + " | " if order.reason else "") + "clamped by kill switch"
        return allowed > MIN_SHARES

    def _note_breaker(self) -> None:
        if self.gate.tripped and not self._breaker_alerted:
            self._breaker_alerted = True
            reason = self.gate.state.breaker_reason or "circuit breaker tripped"
            alerts.alert_breaker(reason, run_id=self.run_id, strategy=self.strategy.name,
                                 journal=self.events)
        elif not self.gate.tripped:
            self._breaker_alerted = False

    def _block(self, reason: str) -> None:
        self.blocked = True
        self.block_reason = reason
        log.error("trading blocked: %s", reason)
        alerts.alert_error(reason, run_id=self.run_id, strategy=self.strategy.name,
                           blocked=True, journal=self.events)

    def _known_order_ids(self) -> set[str]:
        out: set[str] = set()
        for order in self._orders.values():
            out.update({order.id, order.broker_order_id or "", order.idem_key or ""})
        out.update(self._submitted_idem_keys(session_date(utcnow())))
        out.update(self._persisted_order_ids())
        out.discard("")
        return out

    def _submitted_idem_keys(self, session: date) -> set[str]:
        cached = self._idem_by_session.get(session)
        if cached is None:
            cached = self._load_idem_keys(session)
            self._idem_by_session[session] = cached
        return cached

    def _load_idem_keys(self, session: date) -> set[str]:
        """Today's already-placed keys, from the order table.

        This is the half of idempotency that survives the process dying. A
        rejected order is excluded so a transient refusal can be retried; a
        cancelled one is not, because "we placed it and then pulled it" is still
        a decision that was acted on today.
        """
        prefix = f"{self.strategy.name}|{session}|"
        if not self.register:
            return set()
        try:
            pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            rows = connect().execute(
                "SELECT idem_key FROM orders WHERE idem_key LIKE ? ESCAPE '\\' "
                "AND status != 'rejected'",
                (pattern,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 - an unreadable registry is not a licence to double-fire
            log.error("could not read prior orders: %s", exc)
            self._block(f"cannot read the order table to de-duplicate: {exc}")
            return set()
        return {str(r[0]) for r in rows if str(r[0]).startswith(prefix)}

    def _persisted_order_ids(self) -> set[str]:
        if not self.register or not self.run_id:
            return set()
        try:
            rows = connect().execute(
                "SELECT id, broker_order_id, idem_key FROM orders WHERE strategy = ?",
                (self.strategy.name,),
            ).fetchall()
        except Exception:  # noqa: BLE001
            return set()
        out: set[str] = set()
        for r in rows:
            out.update({str(r[0] or ""), str(r[1] or ""), str(r[2] or "")})
        out.discard("")
        return out

    # --- internals: persistence --------------------------------------------

    def _ensure_run(self) -> str:
        if self.run_id:
            return self.run_id
        if not self.register:
            self.run_id = new_id("live_")
            return self.run_id
        from lab.registry.runs import RunRegistry, config_hash, git_commit

        cfg = self.cfg.to_dict()
        record = RunRegistry().create(
            strategy=self.strategy.name,
            kind="paper" if self.cfg.paper else "live",
            git_commit=git_commit(),
            config_hash=config_hash(cfg),
            data_version=_data_version(self._universe),
            params=self.params,
            config=cfg | {"strategy_hash": self.strategy.source_hash},
            start=utcnow(),
            origin=self.cfg.origin,
            notes=self.cfg.notes,
            parent_run_id=self.cfg.parent_run_id,
        )
        self.run_id = record.run_id
        return self.run_id

    def _finish_run(self, reason: str) -> None:
        if not (self.register and self.run_id):
            return
        try:
            from lab.registry.runs import RunRegistry

            RunRegistry().finish(
                self.run_id,
                metrics={
                    "final_equity": round(self.portfolio.equity, 6),
                    "cash": round(self.portfolio.cash, 6),
                    "open_positions": len(self.portfolio.positions),
                    "trades": len(self.portfolio.trades()),
                    "steps": self.steps,
                    "commission": round(self.portfolio.total_commission, 6),
                    "stop_reason": reason,
                },
                status="ok" if not self.blocked else "blocked",
                error=self.block_reason,
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not mask the shutdown
            log.warning("could not finish run record: %s", exc)

    def _persist_decision(self, decision: Decision, verdicts: Sequence[GateVerdict], now: datetime) -> None:
        try:
            self.decisions.append(decision)
        except Exception as exc:  # noqa: BLE001
            log.error("decision journal write failed: %s", exc)
        self._journal(EventKind.DECISION, "runner", decision.summary(),
                      at=now, payload={"decision_id": decision.id,
                                       "n_intents": len(decision.intents),
                                       "n_orders": len(decision.order_ids)})
        for v in verdicts:
            if v.action is GateAction.PASS:
                continue
            alerts.alert_gate_block(v, run_id=self.run_id, strategy=self.strategy.name,
                                    at=now, journal=self.events)

    def _persist_orders(self, orders: Sequence[Order]) -> None:
        if not (self.register and orders):
            return
        rows = [
            (o.id, self.run_id, o.decision_id, o.strategy, o.ticker, o.side.value, o.qty,
             o.order_type.value, o.limit_price,
             to_utc(o.created_at).isoformat() if o.created_at else None,
             o.status.value, o.broker_order_id, o.idem_key, o.tag, o.reason)
            for o in orders
        ]
        self._write(
            """
            INSERT OR REPLACE INTO orders
                (id, run_id, decision_id, strategy, ticker, side, qty, order_type,
                 limit_price, created_at, status, broker_order_id, idem_key, tag, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )

    def _persist_fills(self, fills: Sequence[Fill]) -> None:
        if not (self.register and fills):
            return
        rows = [
            (f.id, f.order_id, self.run_id, f.decision_id, f.strategy or self.strategy.name,
             f.ticker, f.side.value, f.qty, f.price, to_utc(f.at).isoformat(),
             f.commission, f.slippage, f.tag)
            for f in fills
        ]
        self._write(
            """
            INSERT OR REPLACE INTO fills
                (id, order_id, run_id, decision_id, strategy, ticker, side, qty, price,
                 at, commission, slippage, tag)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )

    def _persist_trade(self, trade: Trade) -> None:
        if not self.register:
            return
        self._write(
            """
            INSERT INTO trades
                (run_id, ticker, side, qty, entry_time, entry_price, exit_time, exit_price,
                 pnl, pnl_pct, bars_held, commission, tag, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [(self.run_id, trade.ticker, trade.side, trade.qty,
              to_utc(trade.entry_time).isoformat() if trade.entry_time else None,
              trade.entry_price,
              to_utc(trade.exit_time).isoformat() if trade.exit_time else None,
              trade.exit_price, trade.pnl, trade.pnl_pct, trade.bars_held,
              trade.commission, trade.tag, trade.exit_reason)],
        )

    @staticmethod
    def _write(sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
        try:
            con = connect()
            with transaction(con):
                con.executemany(sql, list(rows))
        except Exception as exc:  # noqa: BLE001 - the journal is the record of last resort
            log.error("registry write failed: %s", exc)

    def _journal(
        self,
        kind: EventKind,
        source: str,
        message: str,
        *,
        at: datetime | None = None,
        ticker: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.events.append(
                event(kind, source, at=at or utcnow(), run_id=self.run_id,
                      strategy=self.strategy.name, ticker=ticker, message=message,
                      payload=dict(payload or {}))
            )
        except Exception as exc:  # noqa: BLE001
            log.error("event journal write failed: %s", exc)

    # --- internals: live_strategies row ------------------------------------

    # --- the book across restarts -------------------------------------------

    def _book_snapshot(self) -> str:
        """Positions and cash as of now.

        Derived state is normally worth distrusting, but this snapshot is checked
        against the broker on the very next startup -- which is exactly what
        reconciliation is for. Drift shows up as a diff rather than as a silent
        wrong number.
        """
        return _dumps({
            "cash": self.portfolio.cash,
            "positions": [
                {
                    "ticker": t,
                    "qty": pos.qty,
                    "avg_price": pos.avg_price,
                    "last_price": pos.last_price,
                }
                for t, pos in self.portfolio.positions.items()
                if abs(pos.qty) > 0
            ],
            "saved_at": utcnow().isoformat(),
            "run_id": self.run_id,
        })

    def restore_book(self) -> dict[str, Any] | None:
        """Load the last saved book into the portfolio, before reconciling.

        Order matters. Reconciling an *empty* book against a live account makes
        every position look like a stranger, so a restart used to mean "refuse to
        trade until a human intervenes". Restoring first turns the comparison
        into the question actually worth asking: did anything change while this
        process was down?

        Positions are re-established through ``apply_fill`` -- the same route
        ``reconcile`` uses to adopt broker state -- so average-cost accounting
        stays consistent instead of acquiring a second, subtly different path.
        """
        if not self.register:
            return None
        try:
            con = connect(journal_path())
            row = con.execute(
                "SELECT book FROM live_strategies WHERE strategy = ?",
                (self.strategy.name,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001 - no book is a cold start, not a crash
            log.warning("could not read the stored book: %s", exc)
            return None

        raw = (dict(row).get("book") if row else "") or ""
        if not raw:
            return None
        try:
            book = _loads(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("stored book is unreadable, starting cold: %s", exc)
            return None

        when = utcnow()
        restored = 0
        for entry in book.get("positions") or []:
            try:
                ticker = str(entry["ticker"]).upper()
                qty = float(entry.get("qty") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if abs(qty) <= 0:
                continue
            price = float(entry.get("avg_price") or entry.get("last_price") or 0.0)
            self.portfolio.apply_fill(
                Fill(
                    id=new_id("f_"), order_id="restore", ticker=ticker,
                    side=Side.BUY if qty > 0 else Side.SELL, qty=abs(qty),
                    price=price, at=when, tag="restore",
                )
            )
            restored += 1

        # Set last, so the synthesized fills' cash effects do not survive: the
        # saved cash is the truth, not whatever re-buying the book would imply.
        cash = book.get("cash")
        if isinstance(cash, (int, float)):
            self.portfolio._cash = float(cash)  # noqa: SLF001 - see reconcile.py

        self._journal(
            EventKind.RECONCILE, "runner",
            f"restored {restored} position(s) from the book saved at {book.get('saved_at')}",
            payload={"restored": restored, "saved_at": book.get("saved_at"),
                     "previous_run_id": book.get("run_id")},
        )
        return book

    def _save_book(self) -> None:
        if not self.register:
            return
        try:
            con = connect(journal_path())
            with transaction(con):
                con.execute(
                    "UPDATE live_strategies SET book = ?, updated_at = ? WHERE strategy = ?",
                    (self._book_snapshot(), utcnow().isoformat(), self.strategy.name),
                )
        except Exception as exc:  # noqa: BLE001 - losing a snapshot is not worth a crash
            log.warning("could not save the book: %s", exc)

    def settle(self, seconds: int | None = None) -> int:
        """Wait for the orders just sent to resolve. Returns fills applied.

        Market orders clear in seconds while the market is open; outside hours
        they sit until the next open and this simply times out, which is correct.
        Either way the book saved afterwards reflects reality rather than intent.
        """
        budget = int(self.settle_seconds if seconds is None else seconds)
        if budget <= 0:
            return 0
        applied = 0
        deadline = _time.monotonic() + budget
        while _time.monotonic() < deadline:
            try:
                working = list(self.broker.open_orders() or [])
            except Exception as exc:  # noqa: BLE001 - a dead broker ends the wait
                log.warning("could not poll open orders while settling: %s", exc)
                break
            fills = self._poll_fills(utcnow(), self._last_view) if self._last_view else []
            applied += len(fills)
            if not working:
                break
            _time.sleep(min(2.0, max(0.5, self.cfg.poll_seconds / 10.0)))
        if applied:
            self._journal(
                EventKind.FILL, "runner",
                f"settled {applied} fill(s) before exit",
                payload={"fills": applied},
            )
        return applied

    def sync_book_from_broker(self) -> int:
        """Overwrite local positions with the broker's, without ceremony.

        Called at exit, where "the broker is right" is not a judgement call: we
        are recording what is there, not deciding whether to trust it. Doing this
        at *startup* would be adopting silently, which is the thing reconciliation
        refuses to do -- the asymmetry is deliberate.
        """
        try:
            remote = dict(self.broker.positions() or {})
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read broker positions at exit: %s", exc)
            return 0
        changed = 0
        when = utcnow()
        for ticker in sorted(set(remote) | set(self.portfolio.positions)):
            target = float(getattr(remote.get(ticker), "qty", 0.0) or 0.0)
            current = float(self.portfolio.position(ticker).qty)
            delta = target - current
            if abs(delta) <= 1e-6:
                continue
            price = float(
                getattr(remote.get(ticker), "avg_price", 0.0)
                or self.portfolio.position(ticker).avg_price
                or 0.0
            )
            self.portfolio.apply_fill(
                Fill(id=new_id("f_"), order_id="settle", ticker=ticker,
                     side=Side.BUY if delta > 0 else Side.SELL, qty=abs(delta),
                     price=price, at=when, tag="settle")
            )
            changed += 1

        # Cash too, or the book drifts from the account by every dollar of P&L
        # that happened outside its view -- commissions, dividends, a fill at a
        # price the local model did not predict. Positions matching while cash
        # does not is still a wrong book, and equity is what sizes the next
        # session's orders.
        try:
            account = self.broker.account() or {}
            cash = account.get("cash")
            if cash is not None and abs(float(cash) - self.portfolio.cash) > 1e-6:
                self.portfolio._cash = float(cash)  # noqa: SLF001 - see reconcile.py
                changed += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read broker cash at exit: %s", exc)
        return changed

    def _upsert_live_row(self, status: str) -> None:
        if not self.register:
            return
        now = utcnow().isoformat()
        try:
            con = connect(journal_path())
            with transaction(con):
                con.execute(
                    """
                    INSERT INTO live_strategies
                        (strategy, run_id, kind, status, config, pid, host, started_at,
                         updated_at, paused, next_fire, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(strategy) DO UPDATE SET
                        run_id=excluded.run_id, kind=excluded.kind, status=excluded.status,
                        config=excluded.config, pid=excluded.pid, host=excluded.host,
                        started_at=excluded.started_at, updated_at=excluded.updated_at,
                        next_fire=excluded.next_fire, notes=excluded.notes
                    """,
                    (
                        self.strategy.name, self.run_id, "paper" if self.cfg.paper else "live",
                        status, _dumps(self.cfg.to_dict()), os.getpid(), socket.gethostname(),
                        self.started_at.isoformat() if self.started_at else None, now,
                        int(self.paused), self._next_fire_iso(utcnow()), self.cfg.notes,
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - a status row is not worth a crash
            log.warning("live_strategies upsert failed: %s", exc)

    def _set_paused(self, paused: bool) -> None:
        if not self.register:
            return
        try:
            con = connect(journal_path())
            with transaction(con):
                con.execute(
                    "UPDATE live_strategies SET paused = ?, updated_at = ? WHERE strategy = ?",
                    (int(paused), utcnow().isoformat(), self.strategy.name),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("pause flag write failed: %s", exc)

    def _is_paused(self) -> bool:
        """Pause is read from the table, not from memory, so the console's
        confirm-gated pause button reaches a runner it does not share a process
        with."""
        if not self.register:
            return self.paused
        try:
            row = connect(journal_path()).execute(
                "SELECT paused FROM live_strategies WHERE strategy = ?", (self.strategy.name,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return self.paused
        return bool(row[0]) if row is not None else self.paused

    # --- internals: small helpers ------------------------------------------

    def _broker_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "broker": getattr(self.broker, "name", "?"),
            "paper": self.cfg.paper,
        }
        available = getattr(self.broker, "available", None)
        if callable(available):
            ok, reason = available()
            meta |= {"available": bool(ok), "reason": reason}
        t0 = _time.perf_counter()
        try:
            account = self.broker.account() or {}
            meta |= {
                "connected": True,
                "latency_ms": round((_time.perf_counter() - t0) * 1000, 2),
                "cash": account.get("cash"),
                "equity": account.get("equity"),
            }
        except Exception as exc:  # noqa: BLE001 - a broker outage is metadata, not a crash
            meta |= {"connected": False, "error": f"{type(exc).__name__}: {exc}"}
        return meta

    def _next_fire_iso(self, now: datetime) -> str | None:
        next_fire = getattr(self.clock, "next_fire", None)
        if not callable(next_fire):
            return None
        try:
            return to_utc(next_fire(now)).isoformat()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _age(ts: datetime | None, now: datetime) -> float | None:
        return None if ts is None else round((now - to_utc(ts)).total_seconds(), 3)


def _make_broker(cfg: LiveConfig, portfolio: Portfolio) -> Any:
    """Resolve ``cfg.broker``. ``sim`` exists so the whole live path -- clock,
    reconciliation, journals, alerts -- can be exercised with no credentials."""
    name = str(cfg.broker or "alpaca").strip().lower()
    if name in {"sim", "paper_sim", "fake"}:
        from lab.backtest.fills import FillModel
        from lab.engine.broker_sim import SimBroker

        return SimBroker(portfolio, FillModel(mode="same_close"), name="sim")
    if name == "alpaca":
        from lab.engine.broker_alpaca import AlpacaBroker

        return AlpacaBroker(paper=cfg.paper, allow_live=cfg.allow_live)
    raise ValueError(f"unknown broker {cfg.broker!r}; expected 'alpaca' or 'sim'")


def _data_version(tickers: Sequence[str]) -> str:
    try:
        from lab.store import parquet_io

        return parquet_io.data_version(tickers=list(tickers))
    except Exception:  # noqa: BLE001 - a live run over an injected view has no partitions
        return "live"


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, default=str, sort_keys=True)


def _loads(raw: str) -> dict[str, Any]:
    import json

    value = json.loads(raw)
    return value if isinstance(value, dict) else {}
