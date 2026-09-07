"""Live-runner tests. Fully offline: a fake broker, an injected DataView, and a
tmpdir for the registry, the journals and the kill file.

The load-bearing case is the last one: the same strategy, the same bars and the
same limits must produce the *same decision* whether the backtester or the live
runner is driving. If that assertion ever fails, paper results have stopped
being comparable to backtest results and the platform's central promise is gone.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pytest

from lab.backtest.runner import BacktestConfig, run_backtest
from lab.config import get_settings, reset_settings_cache
from lab.engine.broker_alpaca import AlpacaBroker, client_order_id
from lab.engine.context import DataView
from lab.engine.events import (
    Fill,
    GateAction,
    Order,
    OrderStatus,
    Position,
    Side,
    new_id,
)
from lab.live import alerts, killswitch
from lab.live.reconcile import reconcile
from lab.live.runner import LiveConfig, LiveRunner
from lab.registry.db import close_all
from lab.registry.journal import EventJournal
from lab.timeutil import UTC, session_date

START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)  # 09:30 ET, a Tuesday
TICKERS = ("AAPL", "MSFT")


# --- harness -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Own data dir, own kill file, no webhook, no stray env kill switch."""
    close_all()
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("LAB_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.delenv("LAB_KILL_SWITCH", raising=False)
    monkeypatch.delenv("LAB_ALERT_WEBHOOK", raising=False)
    monkeypatch.delenv("LAB_ALLOW_LIVE_TRADING", raising=False)
    reset_settings_cache()
    yield
    close_all()
    reset_settings_cache()


class FakeBroker:
    """A Broker with no network: submits into a list, fills on command.

    Lives in the tests on purpose -- shipping a fake broker in the library is
    how a fake broker ends up wired into something real.
    """

    name = "fake"

    def __init__(
        self,
        positions: Mapping[str, Position] | None = None,
        open_orders: Sequence[Order] | None = None,
        cash: float = 100_000.0,
        unreachable: bool = False,
    ) -> None:
        self._positions = dict(positions or {})
        self._open = list(open_orders or [])
        self._cash = float(cash)
        self.unreachable = unreachable
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []
        self.fail_submit = False
        self._by_idem: dict[str, Order] = {}
        self._queued_fills: list[Fill] = []

    def submit(self, order: Order) -> Order:
        prior = self._by_idem.get(order.idem_key)
        if order.idem_key and prior is not None:
            return prior
        if self.fail_submit:
            raise RuntimeError("broker refused the order")
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"b_{order.id}"
        self.submitted.append(order)
        self._open.append(order)
        if order.idem_key:
            self._by_idem[order.idem_key] = order
        return order

    def cancel(self, order_id: str) -> bool:
        for i, o in enumerate(self._open):
            if o.id == order_id:
                o.status = OrderStatus.CANCELED
                self._open.pop(i)
                self.cancelled.append(order_id)
                return True
        return False

    def open_orders(self) -> list[Order]:
        if self.unreachable:
            raise ConnectionError("broker is down")
        return list(self._open)

    def positions(self) -> dict[str, Position]:
        if self.unreachable:
            raise ConnectionError("broker is down")
        return dict(self._positions)

    def account(self) -> dict[str, Any]:
        if self.unreachable:
            raise ConnectionError("broker is down")
        equity = self._cash + sum(p.market_value for p in self._positions.values())
        return {"cash": self._cash, "equity": equity, "broker": self.name, "paper": True}

    def poll_fills(self, since: datetime | None = None) -> list[Fill]:
        out, self._queued_fills = self._queued_fills, []
        return out

    # -- test helpers --
    def queue_fill(self, order: Order, price: float, at: datetime) -> Fill:
        fill = Fill(
            id=f"f_{order.id}",
            order_id=order.id,
            ticker=order.ticker,
            side=order.side,
            qty=order.qty,
            price=price,
            at=at,
            strategy=order.strategy,
            tag=order.tag,
        )
        self._queued_fills.append(fill)
        return fill


class NoClock:
    """A clock that never fires: lets ``start()`` be tested without a loop."""

    def __iter__(self):
        return iter(())

    def next_fire(self, after: datetime | None = None) -> datetime:
        return START

    def stop(self) -> None:
        pass


def make_view(n: int = 40) -> DataView:
    """AAPL trends up, MSFT trends down. knowledge_time == event_time."""
    idx = pd.DatetimeIndex([START + timedelta(days=i) for i in range(n)])
    frames: dict[str, pd.DataFrame] = {}
    for k, ticker in enumerate(TICKERS):
        base, slope = 100.0 + 50 * k, (1.0 if k == 0 else -0.5)
        close = [base + slope * i for i in range(n)]
        frames[ticker] = pd.DataFrame(
            {
                "open": close,
                "high": [c + 1 for c in close],
                "low": [c - 1 for c in close],
                "close": close,
                "volume": [1_000_000.0] * n,
                "knowledge_time": idx,
            },
            index=idx,
        )
    return DataView(frames)


MOMO_SRC = '''
"""Above-its-own-average momentum. Small enough to reason about by hand."""
from __future__ import annotations

PARAMS = {"n": 3, "target": 0.10}


class Strategy:
    def __init__(self, params):
        self.params = dict(params)

    def on_bar(self, ctx):
        n = int(ctx.params["n"])
        for ticker in ctx.universe:
            hist = ctx.history(ticker, "close", n + 1)
            price = ctx.price(ticker)
            if len(hist) < n + 1 or price is None:
                continue
            if price > float(hist.tail(n).mean()):
                ctx.order_target_pct(ticker, float(ctx.params["target"]), tag="entry")
            else:
                ctx.close(ticker)
'''

TARGETS_SRC = '''
"""Emits whatever `params["targets"]` says. A fixture, not a strategy."""
from __future__ import annotations

PARAMS = {"targets": {}}


class Strategy:
    def __init__(self, params):
        self.params = dict(params)

    def on_bar(self, ctx):
        for ticker, pct in (ctx.params.get("targets") or {}).items():
            ctx.order_target_pct(ticker, float(pct), tag="test")
'''


def write_strategy(tmp_path: Path, name: str, source: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


LIMITS = {"max_position_pct": 0.20, "max_positions": 4, "min_order_notional": 10.0}


def live_cfg(strategy: str, **over: Any) -> LiveConfig:
    base: dict[str, Any] = {
        "strategy": strategy,
        "tickers": list(TICKERS),
        "broker": "sim",
        "cash": 100_000.0,
        "limits": dict(LIMITS),
    }
    base.update(over)
    return LiveConfig(**base)


def make_runner(cfg: LiveConfig, broker: FakeBroker | None = None, **kw: Any) -> LiveRunner:
    kw.setdefault("register", False)
    kw.setdefault("clock", NoClock())
    kw.setdefault("data", make_view())
    return LiveRunner(cfg, broker=broker or FakeBroker(), **kw)


def held(portfolio, ticker: str, qty: float, price: float, at: datetime) -> None:
    portfolio.apply_fill(
        Fill(id=new_id("f_"), order_id="seed", ticker=ticker, side=Side.BUY,
             qty=qty, price=price, at=at)
    )


# --- kill switch --------------------------------------------------------------


def test_kill_switch_file_engages_and_releases():
    assert killswitch.engaged() == (False, None)
    path = killswitch.engage("testing")
    assert path.exists()
    on, reason = killswitch.engaged()
    assert on and "kill file" in reason
    assert killswitch.describe()["detail"]["reason"] == "testing"
    assert killswitch.release() is True
    assert killswitch.engaged() == (False, None)
    assert killswitch.release() is True  # releasing twice is not an error


def test_kill_switch_env_flag_cannot_be_released_by_deleting_a_file(monkeypatch):
    monkeypatch.setenv("LAB_KILL_SWITCH", "1")
    reset_settings_cache()
    on, reason = killswitch.engaged()
    assert on and "env flag" in reason
    # The file is gone but the switch is not, and release() must say so.
    assert killswitch.release() is False


# --- alerts -------------------------------------------------------------------


def test_alert_journals_without_a_webhook():
    journal = EventJournal()
    alerts.alert("error", "boom", run_id="r1", strategy="s", detail=7)
    rows = journal.tail(0, kinds=["error"])
    assert rows and rows[-1]["message"] == "boom"
    assert rows[-1]["payload"]["detail"] == 7
    assert rows[-1]["payload"]["alert_kind"] == "error"


def test_alert_swallows_a_failing_webhook(monkeypatch):
    monkeypatch.setenv("LAB_ALERT_WEBHOOK", "http://127.0.0.1:1/hook")
    reset_settings_cache()
    import httpx

    posted: list[str] = []

    def boom(*args: Any, **kwargs: Any):
        posted.append("tried")
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "post", boom)
    journal = EventJournal()
    alerts.alert("fill", "AAPL +1 @ 100")  # must not raise
    assert posted == ["tried"]
    assert journal.tail(0, kinds=["fill"])  # journal still got it


def test_alert_helpers_do_not_raise_on_a_dead_journal():
    class DeadJournal:
        def append(self, ev):
            raise RuntimeError("disk on fire")

    alerts.alert("error", "still fine", journal=DeadJournal())


# --- reconciliation -----------------------------------------------------------


def test_reconcile_clean_when_broker_and_runner_agree(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    result = reconcile(FakeBroker(), runner.portfolio)
    assert result.ok and result.diffs == [] and "0 diffs" in result.message


def test_reconcile_blocks_on_an_unknown_position_then_proceeds_once_acknowledged(tmp_path):
    strategy = write_strategy(tmp_path, "targets", TARGETS_SRC)
    cfg = live_cfg(strategy, params={"targets": {"MSFT": 0.10}})
    broker = FakeBroker(
        positions={"AAPL": Position(ticker="AAPL", qty=10.0, avg_price=100.0, last_price=100.0)}
    )
    runner = make_runner(cfg, broker)

    runner.start()
    assert runner.blocked is True
    assert runner.state == "blocked"
    assert [d["kind"] for d in runner.last_reconcile.diffs] == ["unknown_position"]
    assert broker.submitted == []

    # A blocked runner refuses to trade even if stepped directly.
    assert runner.step(START + timedelta(days=10)) is None
    assert broker.submitted == []

    result = runner.acknowledge("checked the account by hand")
    assert result.ok and result.acknowledged
    assert runner.blocked is False
    assert runner.portfolio.position("AAPL").qty == 10.0
    assert runner.portfolio.cash == pytest.approx(100_000.0)  # taken from the account

    decision = runner.step(START + timedelta(days=10))
    assert decision is not None and broker.submitted


def test_reconcile_flags_an_open_order_the_runner_does_not_know(tmp_path):
    stray = Order(id="o_stray", ticker="AAPL", side=Side.BUY, qty=1.0, idem_key="x|y|AAPL|buy")
    result = reconcile(FakeBroker(open_orders=[stray]), make_runner(
        live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC))).portfolio)
    assert not result.ok
    assert [d["kind"] for d in result.diffs] == ["unknown_open_order"]

    known = reconcile(FakeBroker(open_orders=[stray]), make_runner(
        live_cfg(write_strategy(tmp_path, "t2", TARGETS_SRC))).portfolio,
        known_order_ids={"o_stray"})
    assert known.ok


def test_reconcile_refuses_to_acknowledge_an_unreachable_broker(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    result = reconcile(FakeBroker(unreachable=True), runner.portfolio, acknowledge=True)
    assert not result.ok and not result.acknowledged
    assert {d["kind"] for d in result.diffs} == {"broker_unreachable"}


def test_reconcile_detects_a_position_the_broker_lost(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    held(runner.portfolio, "AAPL", 5.0, 100.0, START)
    result = reconcile(FakeBroker(), runner.portfolio)
    assert [d["kind"] for d in result.diffs] == ["missing_position"]
    result = reconcile(FakeBroker(), runner.portfolio, acknowledge=True)
    assert result.ok and runner.portfolio.position("AAPL").qty == 0.0


# --- idempotency / crash-and-resume -------------------------------------------


def test_a_resumed_runner_does_not_refire_an_order_with_an_existing_idem_key(tmp_path):
    strategy = write_strategy(tmp_path, "targets", TARGETS_SRC)
    cfg = live_cfg(strategy, params={"targets": {"AAPL": 0.10}})
    ts = START + timedelta(days=10)
    view = make_view()

    first = LiveRunner(cfg, broker=FakeBroker(), data=view, clock=NoClock(), register=True)
    decision = first.step(ts)
    assert decision is not None and len(decision.order_ids) == 1
    placed = first.broker.submitted[0]
    assert placed.idem_key == Order.make_idem_key(
        first.strategy.name, session_date(ts), "AAPL", Side.BUY
    )

    # The process dies here. A fresh runner, fresh broker, fresh book, same bar.
    resumed = LiveRunner(cfg, broker=FakeBroker(), data=view, clock=NoClock(), register=True)
    again = resumed.step(ts)
    assert again is not None
    assert again.order_ids == []
    assert resumed.broker.submitted == []

    skips = [
        r for r in EventJournal().tail(0, kinds=["order"])
        if "duplicate" in r["message"]
    ]
    assert skips, "the skip must be journaled, not silent"


def test_the_broker_adapter_is_idempotent_too(tmp_path):
    """Belt and braces: the runner's DB check and the broker both de-dupe."""
    broker = FakeBroker()
    order = Order(id="o1", ticker="AAPL", side=Side.BUY, qty=3.0, idem_key="s|2024-01-02|AAPL|buy")
    twin = Order(id="o2", ticker="AAPL", side=Side.BUY, qty=3.0, idem_key="s|2024-01-02|AAPL|buy")
    assert broker.submit(order) is broker.submit(twin)
    assert len(broker.submitted) == 1


# --- kill switch in the order path --------------------------------------------


def _entry_and_exit(runner: LiveRunner, ts: datetime) -> tuple[Order, Order]:
    session = session_date(ts)
    name = runner.strategy.name
    entry = Order(
        id=new_id("o_"), ticker="MSFT", side=Side.BUY, qty=5.0, created_at=ts,
        strategy=name, tag="entry",
        idem_key=Order.make_idem_key(name, session, "MSFT", Side.BUY),
    )
    exit_ = Order(
        id=new_id("o_"), ticker="AAPL", side=Side.SELL, qty=10.0, created_at=ts,
        strategy=name, tag="exit",
        idem_key=Order.make_idem_key(name, session, "AAPL", Side.SELL),
    )
    return entry, exit_


def test_kill_file_blocks_entries_but_lets_exits_through(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    held(runner.portfolio, "AAPL", 10.0, 100.0, START)
    killswitch.engage("panic")

    entry, exit_ = _entry_and_exit(runner, START)
    sent = runner.submit_orders([entry, exit_], now=START)

    assert [o.ticker for o in sent] == ["AAPL"]
    assert entry.status is OrderStatus.REJECTED and "kill_switch" in entry.reason
    rejects = EventJournal().tail(0, kinds=["reject"])
    assert rejects and rejects[-1]["ticker"] == "MSFT"


def test_env_kill_switch_blocks_entries_but_lets_exits_through(tmp_path, monkeypatch):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    held(runner.portfolio, "AAPL", 10.0, 100.0, START)
    monkeypatch.setenv("LAB_KILL_SWITCH", "on")
    reset_settings_cache()

    entry, exit_ = _entry_and_exit(runner, START)
    sent = runner.submit_orders([entry, exit_], now=START)
    assert [o.ticker for o in sent] == ["AAPL"] and sent[0].qty == 10.0


def test_kill_switch_clamps_an_exit_that_would_flip_the_position(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    held(runner.portfolio, "AAPL", 10.0, 100.0, START)
    killswitch.engage("panic")

    oversized = Order(
        id=new_id("o_"), ticker="AAPL", side=Side.SELL, qty=25.0, created_at=START,
        strategy=runner.strategy.name, tag="exit",
        idem_key=Order.make_idem_key(runner.strategy.name, session_date(START), "AAPL", Side.SELL),
    )
    (sent,) = runner.submit_orders([oversized], now=START)
    assert sent.qty == 10.0 and "clamped" in sent.reason


# --- breaker ------------------------------------------------------------------


def test_a_tripped_breaker_stops_entries_but_not_exits(tmp_path):
    strategy = write_strategy(tmp_path, "targets", TARGETS_SRC)
    cfg = live_cfg(strategy, params={"targets": {"AAPL": 0.0, "MSFT": 0.10}})
    runner = make_runner(cfg)
    ts = START + timedelta(days=10)
    held(runner.portfolio, "AAPL", 10.0, 110.0, ts)
    runner.gate.trip("manual trip for the test")

    decision = runner.step(ts)
    assert decision is not None
    by_ticker = {v.ticker: v for v in decision.verdicts}
    assert by_ticker["MSFT"].action is GateAction.BLOCK
    assert by_ticker["MSFT"].rule == "breaker"
    assert by_ticker["AAPL"].action is GateAction.PASS
    assert [(o.ticker, o.side) for o in runner.broker.submitted] == [("AAPL", Side.SELL)]

    breaker_events = EventJournal().tail(0, kinds=["breaker"])
    assert breaker_events and "manual trip" in breaker_events[-1]["message"]


# --- heartbeats ---------------------------------------------------------------


def test_heartbeats_land_in_the_journal_with_sane_ages(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    runner.step(START + timedelta(days=10))
    runner.heartbeat()

    beats = EventJournal().last_heartbeats()
    assert set(beats) == {"runner", "data", "broker"}
    for source, beat in beats.items():
        assert 0.0 <= beat["age_s"] < 60.0, source
    assert beats["runner"]["meta"]["strategy"] == runner.strategy.name
    assert beats["data"]["meta"]["tickers"] == len(TICKERS)
    assert beats["data"]["meta"]["bar_age_s"] is not None
    assert beats["broker"]["meta"]["connected"] is True


def test_a_broker_outage_shows_up_in_the_heartbeat_not_as_a_crash(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)),
                         FakeBroker(unreachable=True))
    runner.heartbeat()
    beat = EventJournal().last_heartbeats()["broker"]
    assert beat["meta"]["connected"] is False and "ConnectionError" in beat["meta"]["error"]


# --- fills, status, control ---------------------------------------------------


def test_a_polled_fill_updates_the_book_and_alerts(tmp_path):
    strategy = write_strategy(tmp_path, "targets", TARGETS_SRC)
    runner = make_runner(live_cfg(strategy, params={"targets": {}}))
    ts = START + timedelta(days=10)
    order = Order(id="o_x", ticker="AAPL", side=Side.BUY, qty=4.0, idem_key="k")
    runner.broker.queue_fill(order, price=110.0, at=ts)

    runner.step(ts)
    assert runner.portfolio.position("AAPL").qty == 4.0
    fills = EventJournal().tail(0, kinds=["fill"])
    assert fills and fills[-1]["ticker"] == "AAPL"

    # A re-reported fill must not be booked twice.
    runner.broker._queued_fills.append(
        Fill(id="f_o_x", order_id="o_x", ticker="AAPL", side=Side.BUY, qty=4.0,
             price=110.0, at=ts)
    )
    runner.step(ts + timedelta(days=1))
    assert runner.portfolio.position("AAPL").qty == 4.0


def test_status_reports_the_safety_state(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    killswitch.engage("inspection")
    status = runner.status()
    assert status["kill_switch"]["engaged"] is True
    assert status["state"] == "stopped"
    assert status["kind"] == "paper"
    assert status["breaker"] == {"tripped": False, "reason": ""}
    assert status["universe"] == list(TICKERS)


def test_stop_cancels_working_orders(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    order = Order(id="o_1", ticker="AAPL", side=Side.BUY, qty=1.0, idem_key="a|b|AAPL|buy")
    runner.broker.submit(order)
    runner.stop("done")
    assert runner.broker.cancelled == ["o_1"]
    assert runner.state == "stopped"


def test_a_broker_rejection_is_journaled_not_raised(tmp_path):
    runner = make_runner(live_cfg(write_strategy(tmp_path, "t", TARGETS_SRC)))
    runner.broker.fail_submit = True
    order = Order(id="o_1", ticker="AAPL", side=Side.BUY, qty=1.0,
                  idem_key="a|b|AAPL|buy", strategy=runner.strategy.name)
    assert runner.submit_orders([order], now=START) == []
    assert order.status is OrderStatus.REJECTED
    assert EventJournal().tail(0, kinds=["reject"])


# --- config -------------------------------------------------------------------


def test_live_config_reads_a_backtest_yaml_and_ignores_its_backtest_half():
    cfg = LiveConfig.from_yaml(get_settings().paths.cfg / "momo.yaml")
    assert cfg.timeframe == "1d" and cfg.paper is True and cfg.broker == "alpaca"
    assert cfg.tickers and cfg.tickers[0].isupper()
    assert "fills" in cfg.ignored_keys and "benchmark" in cfg.ignored_keys


def test_live_config_rejects_a_typo():
    with pytest.raises(ValueError, match="unknown live config keys"):
        LiveConfig.from_mapping({"strategy": "x.py", "pol_seconds": 5})


# --- alpaca adapter (offline) --------------------------------------------------


def test_alpaca_broker_constructs_without_credentials():
    broker = AlpacaBroker()
    ok, reason = broker.available()
    assert ok is False and "ALPACA_API_KEY" in reason
    assert broker.paper is True


def test_alpaca_live_needs_a_two_part_opt_in(monkeypatch):
    ok, reason = AlpacaBroker(paper=False).available()
    assert ok is False and "allow_live=True" in reason

    ok, reason = AlpacaBroker(paper=False, allow_live=True).available()
    assert ok is False and "LAB_ALLOW_LIVE_TRADING" in reason

    monkeypatch.setenv("LAB_ALLOW_LIVE_TRADING", "1")
    ok, reason = AlpacaBroker(paper=False, allow_live=True).available()
    assert ok is False and "ALPACA_API_KEY_ID" in reason  # now only creds are missing


def test_client_order_id_is_deterministic_and_safe():
    key = "momo|2024-01-02|AAPL|buy"
    coid = client_order_id(key)
    assert coid == client_order_id(key)
    assert coid != client_order_id("momo|2024-01-02|AAPL|sell")
    assert len(coid) <= 128 and coid.replace("-", "").isalnum()
    with pytest.raises(ValueError):
        client_order_id("")


def test_alpaca_mutating_calls_refuse_without_availability():
    broker = AlpacaBroker()
    order = Order(id="o", ticker="AAPL", side=Side.BUY, qty=1.0, idem_key="a|b|AAPL|buy")
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.submit(order)


# --- the promise ---------------------------------------------------------------


def test_one_step_matches_the_backtester_on_the_same_inputs(tmp_path):
    """Same strategy, same bars, same limits, flat book: same decision.

    This is the assertion the live runner exists to keep. It compares intents,
    gate verdicts and the resulting orders -- everything except the ids and the
    wall-clock timings, which are the only things allowed to differ.
    """
    strategy = write_strategy(tmp_path, "momo_test", MOMO_SRC)
    view = make_view()
    params = {"n": 3, "target": 0.10}

    bt = run_backtest(
        BacktestConfig(
            strategy=strategy, tickers=list(TICKERS), cash=100_000.0,
            params=params, limits=dict(LIMITS),
        ),
        register=False,
        journal=False,
        data=view,
    )
    expected = next(d for d in bt.decisions if d.order_ids)
    ts = expected.at
    bt_orders = sorted(
        (o for o in bt.orders if o.decision_id == expected.id),
        key=lambda o: o.ticker,
    )

    runner = LiveRunner(
        live_cfg(strategy, params=params),
        broker=FakeBroker(),
        data=view,
        clock=NoClock(),
        register=False,
    )
    got = runner.step(ts)
    assert got is not None

    assert [(i.ticker, i.target_pct) for i in got.intents] == [
        (i.ticker, i.target_pct) for i in expected.intents
    ]
    assert [(v.ticker, v.action, v.approved_pct, v.rule) for v in got.verdicts] == [
        (v.ticker, v.action, v.approved_pct, v.rule) for v in expected.verdicts
    ]
    live_orders = sorted(runner.broker.submitted, key=lambda o: o.ticker)
    assert [(o.ticker, o.side, o.qty, o.idem_key) for o in live_orders] == [
        (o.ticker, o.side, o.qty, o.idem_key) for o in bt_orders
    ]
    assert got.portfolio["equity"] == pytest.approx(
        expected.portfolio["equity"], rel=1e-12
    )


# --- surviving a restart ------------------------------------------------------
#
# A daily bot on a personal machine gets rebooted. Before this, the runner came
# back with an empty book, reconciliation read every broker position as a
# stranger, and it refused to trade until a human intervened -- with no command
# to intervene with.


def test_the_book_survives_a_restart_and_reconciles_clean(tmp_path):
    """The whole point: a restart is a diff against known state, not amnesia."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker(positions={"AAPL": Position(ticker="AAPL", qty=7, avg_price=100.0,
                                                    last_price=101.0)})

    first = make_runner(live_cfg(src), broker, register=True)
    held(first.portfolio, "AAPL", 7, 100.0, START)
    first._upsert_live_row("running")
    first._save_book()

    # A new process: nothing in memory, everything on disk.
    second = make_runner(live_cfg(src), broker, register=True)
    assert second.portfolio.positions == {}, "a fresh runner starts empty"

    book = second.restore_book()
    assert book is not None
    assert second.portfolio.position("AAPL").qty == pytest.approx(7.0)

    result = second.reconcile()
    assert result.ok, result.message
    assert not result.diffs, "a restored book must agree with the broker it came from"


def test_without_the_stored_book_a_restart_still_refuses(tmp_path):
    """The old behaviour, kept for the case where there is genuinely no book."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker(positions={"AAPL": Position(ticker="AAPL", qty=7, avg_price=100.0,
                                                    last_price=101.0)})
    runner = make_runner(live_cfg(src), broker, register=True)

    assert runner.restore_book() is None, "no row, no book"
    result = runner.reconcile()
    assert not result.ok
    assert any(d["kind"] == "unknown_position" for d in result.diffs)


def test_adopt_takes_the_broker_as_truth_at_startup(tmp_path):
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker(positions={"AAPL": Position(ticker="AAPL", qty=7, avg_price=100.0,
                                                    last_price=101.0)})
    runner = make_runner(live_cfg(src, adopt=True), broker, register=True)
    runner.start()

    assert not runner.blocked, runner.block_reason
    assert runner.portfolio.position("AAPL").qty == pytest.approx(7.0)


def test_startup_does_not_replay_fills_the_book_already_contains(tmp_path):
    """The bug this nearly shipped with.

    `poll(since=None)` returns the broker's whole fill history. Applied on top of
    a book that already reflects those fills, an adopted 1-share position becomes
    2 and the next exit is rejected for selling more than exists. Seen live.
    """
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker(positions={"AAPL": Position(ticker="AAPL", qty=5, avg_price=100.0,
                                                    last_price=101.0)})
    # The fill that created that position, still sitting in the broker's history.
    broker._queued_fills = [
        Fill(id="f_history", order_id="old", ticker="AAPL", side=Side.BUY,
             qty=5, price=100.0, at=START)
    ]

    runner = make_runner(live_cfg(src, adopt=True), broker, register=True)
    runner.start()

    assert runner.portfolio.position("AAPL").qty == pytest.approx(5.0), (
        "the historical fill was applied on top of the adopted position"
    )


# --- the staleness guard ------------------------------------------------------


def test_stale_bars_block_the_decision(tmp_path):
    """Reported in the heartbeat and enforced nowhere is the same as fresh."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    runner = make_runner(live_cfg(src, max_stale_sessions=2), register=True)

    # The view ends 2024-02-10; decide well past it so the newest bar really is
    # several sessions stale rather than merely not the latest.
    runner.start()
    assert not runner.blocked
    runner.step(START + timedelta(days=90))

    assert runner.blocked
    assert "trading session(s) old" in runner.block_reason


def test_fresh_bars_do_not_block(tmp_path):
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    runner = make_runner(live_cfg(src, max_stale_sessions=2), register=True)
    runner.start()
    runner.step(runner._data_view(START).timestamps()[-1])
    assert not runner.blocked, runner.block_reason


def test_the_guard_can_be_switched_off(tmp_path):
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    runner = make_runner(live_cfg(src, max_stale_sessions=0), register=True)
    runner.start()
    runner.step(START + timedelta(days=365))
    assert not runner.blocked, "0 disables the guard"


# --- one-shot mode ------------------------------------------------------------


def test_once_takes_exactly_one_step_and_stops(tmp_path):
    """What a scheduled task calls: decide for now, save, exit."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    runner = make_runner(live_cfg(src, max_stale_sessions=0), register=True)
    runner.start(once=True)

    assert runner.steps == 1
    assert runner.stopped_at is not None
    # And the book is on disk for the next invocation to pick up.
    from lab.registry.db import connect, journal_path

    row = connect(journal_path()).execute(
        "SELECT book FROM live_strategies WHERE strategy = ?", (runner.strategy.name,)
    ).fetchone()
    assert dict(row)["book"], "one-shot mode must leave a book behind"


# --- settling before exit -----------------------------------------------------
#
# One-shot mode submits and exits, so fills land at the broker a moment after the
# process is gone. Saving the book at that instant records *intent*: tomorrow's
# startup then reads a book that disagrees with the account and refuses to trade,
# which would need a human every single morning -- the exact failure the stored
# book exists to remove. Seen live: the first scheduled run refused because the
# previous run's sell had filled after it saved.


def test_settle_applies_fills_that_land_after_the_step(tmp_path):
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker()
    runner = make_runner(
        live_cfg(src, max_stale_sessions=0, settle_seconds=5,
                 params={"targets": {"MSFT": 0.10}}),
        broker, register=True,
    )
    runner.start()
    runner.step(START)

    sent = list(broker.submitted)
    assert sent, "the strategy must have sent something to settle"
    for order in sent:
        broker.queue_fill(order, price=100.0, at=START)
    broker._open = []

    applied = runner.settle()
    assert applied == len(sent), "every queued fill must be picked up before exit"


def test_the_book_saved_at_exit_matches_the_broker_not_the_local_guess(tmp_path):
    """Positions *and* cash. A book that agrees on shares but not on money is
    still wrong, and equity is what sizes tomorrow's orders."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker(
        positions={"AAPL": Position(ticker="AAPL", qty=9, avg_price=100.0, last_price=101.0)},
        cash=12_345.67,
    )
    runner = make_runner(live_cfg(src), broker, register=True)

    # A local book that disagrees on both axes.
    held(runner.portfolio, "AAPL", 2, 100.0, START)
    assert runner.portfolio.position("AAPL").qty == pytest.approx(2.0)

    runner.sync_book_from_broker()

    assert runner.portfolio.position("AAPL").qty == pytest.approx(9.0)
    assert runner.portfolio.cash == pytest.approx(12_345.67)


def test_one_shot_leaves_a_book_that_reconciles_next_time(tmp_path):
    """End to end: decide, settle, save -- and the next start finds no diffs."""
    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    broker = FakeBroker()
    first = make_runner(
        live_cfg(src, max_stale_sessions=0, settle_seconds=1,
                 params={"targets": {"MSFT": 0.10}}),
        broker, register=True,
    )
    first.start(once=True)

    # Whatever the broker ended up holding is what tomorrow must agree with.
    second = make_runner(live_cfg(src, max_stale_sessions=0), broker, register=True)
    second.restore_book()
    result = second.reconcile()

    assert result.ok, result.message
    assert not result.diffs


def test_a_blocked_run_exits_refused_not_zero(tmp_path, monkeypatch):
    """A scheduled task is the only thing watching at 09:35.

    Refusing to trade is a correct outcome but not a successful one, and exiting
    0 told Task Scheduler the morning had gone fine when the bot did nothing.
    """
    import json as _json

    from typer.testing import CliRunner

    from lab.cli import app

    src = write_strategy(tmp_path, "keeper", TARGETS_SRC)
    cfg_path = tmp_path / "live.yaml"
    cfg_path.write_text(
        _json.dumps({"strategy": src, "tickers": list(TICKERS), "broker": "sim"}),
        encoding="utf-8",
    )

    # A broker holding something the runner has never heard of: reconcile refuses.
    import lab.live.runner as R

    real = R.LiveRunner.reconcile

    def blocked(self, *, acknowledge: bool = False):
        out = real(self, acknowledge=acknowledge)
        self._block("forced block for the test")
        return out

    monkeypatch.setattr(R.LiveRunner, "reconcile", blocked)

    result = CliRunner().invoke(
        app, ["paper", "start", src, "--config", str(cfg_path), "--broker", "sim",
              "--once", "--json"],
    )
    assert result.exit_code == 3, f"expected REFUSED, got {result.exit_code}"
