"""Clock, broker, fill model and strategy loader.

The properties worth pinning here are the ones a wrong answer makes invisible:
a fill price that drifts outside the bar it happened in, an order that vanishes
instead of waiting, a repeated submission that double-fires, a wall clock that
fires on Christmas.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

from lab.backtest.fills import MODES, FillModel
from lab.engine.broker_sim import SimBroker
from lab.engine.clock import ManualClock, SimClock, WallClock
from lab.engine.events import Order, OrderStatus, OrderType, Side, new_id
from lab.engine.loader import discover, load_strategy, resolve_path
from lab.engine.portfolio import Portfolio
from lab.timeutil import (
    ET,
    UTC,
    is_trading_day,
    session_close_utc,
    session_open_utc,
    to_et,
)

from tests.conftest import bar_timestamps

TS = bar_timestamps(6)
BAR = {"open": 100.0, "high": 104.0, "low": 96.0, "close": 102.0, "volume": 10_000.0}


def order(
    ticker: str = "AAA",
    side: Side = Side.BUY,
    qty: float = 10.0,
    *,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    idem_key: str = "",
    strategy: str = "probe",
) -> Order:
    return Order(
        id=new_id("o_"),
        ticker=ticker,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        created_at=TS[0],
        strategy=strategy,
        idem_key=idem_key,
    )


# --- SimClock -----------------------------------------------------------------


def test_simclock_walks_its_timestamps_in_order():
    clock = SimClock(TS)
    assert len(clock) == len(TS)
    assert clock.index == -1
    assert clock.now == TS[0], "before the first tick, now is the first bar"

    seen = []
    for i, ts in enumerate(clock):
        seen.append(ts)
        assert clock.index == i
        assert clock.now == ts
    assert seen == TS
    assert clock.now == TS[-1]


def test_simclock_rejects_a_decreasing_tape():
    with pytest.raises(ValueError, match="non-decreasing"):
        SimClock([TS[2], TS[1]])
    SimClock([TS[1], TS[1], TS[2]])  # a repeat is not a reversal


def test_simclock_normalizes_naive_timestamps_to_utc():
    naive = [datetime(2024, 1, 2, 21, 0), datetime(2024, 1, 3, 21, 0)]
    clock = SimClock(naive)
    assert all(t.tzinfo is UTC for t in clock)


def test_simclock_advance_and_peek():
    clock = SimClock(TS[:3])
    assert clock.peek(0) is None, "nothing consumed yet"
    assert clock.advance() == TS[0]
    assert clock.peek() == TS[1]
    assert clock.advance() == TS[1]
    assert clock.advance() == TS[2]
    assert clock.advance() is None
    assert clock.peek() is None
    assert clock.now == TS[2]


def test_manual_clock_is_driven_by_hand():
    clock = ManualClock(TS[0])
    assert clock.now == TS[0]
    assert clock.set(TS[3]) == TS[3]
    assert list(clock) == [TS[3]]


# --- WallClock ----------------------------------------------------------------


def test_wallclock_daily_fire_skips_the_weekend():
    clock = WallClock("1d", at_time=time(9, 35))
    friday_afternoon = datetime(2024, 3, 8, 15, 0, tzinfo=UTC)
    fire = clock.next_fire(friday_afternoon)

    assert to_et(fire).date() == date(2024, 3, 11), "Monday, not Saturday"
    assert to_et(fire).time() == time(9, 35)
    # 13:35Z, because US DST began on the 10th -- the ET wall time is the anchor.
    assert fire == datetime(2024, 3, 11, 13, 35, tzinfo=UTC)


def test_wallclock_daily_fire_skips_a_market_holiday():
    clock = WallClock("1d", at_time="09:35")
    christmas_eve_pm = datetime(2024, 12, 24, 20, 0, tzinfo=UTC)
    fire = clock.next_fire(christmas_eve_pm)

    assert to_et(fire).date() == date(2024, 12, 26), "the 25th is a holiday"
    assert to_et(fire).time() == time(9, 35)


def test_wallclock_daily_fire_is_always_a_future_session_at_the_declared_time():
    clock = WallClock("1d", at_time=time(9, 35))
    ref = datetime(2024, 6, 27, 3, 0, tzinfo=UTC)
    for _ in range(40):
        fire = clock.next_fire(ref)
        assert fire > ref
        assert is_trading_day(to_et(fire).date())
        assert to_et(fire).time() == time(9, 35)
        ref = fire


def test_wallclock_without_a_calendar_fires_every_day():
    clock = WallClock("1d", at_time=time(9, 35), calendar=False)
    saturday = clock.next_fire(datetime(2024, 3, 9, 20, 0, tzinfo=UTC))
    assert to_et(saturday).date() == date(2024, 3, 10), "a Sunday, because no calendar"


def test_wallclock_intraday_fires_stay_inside_regular_hours():
    clock = WallClock("15m")
    ref = datetime(2024, 3, 5, 12, 0, tzinfo=UTC)  # 07:00 ET, pre-market
    for _ in range(60):
        fire = clock.next_fire(ref)
        session = to_et(fire).date()
        assert is_trading_day(session)
        assert session_open_utc(session) < fire <= session_close_utc(session)
        # Bars are stamped at their close, so every fire is a whole number of
        # intervals after the open.
        offset = (fire - session_open_utc(session)).total_seconds()
        assert offset % (15 * 60) == 0
        assert fire > ref
        ref = fire


def test_wallclock_intraday_rolls_to_the_next_session_after_the_close():
    clock = WallClock("1h")
    after_close = datetime(2024, 3, 5, 21, 30, tzinfo=UTC)  # 16:30 ET
    fire = clock.next_fire(after_close)
    assert to_et(fire).date() == date(2024, 3, 6)
    assert to_et(fire).time() == time(10, 30)


def test_wallclock_iteration_is_interruptible():
    clock = WallClock("1d")
    clock.stop()
    assert list(clock) == [], "stop() must unwind the loop rather than sleep"


# --- FillModel: pricing -------------------------------------------------------


def test_fill_model_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown fill mode"):
        FillModel(mode="magic")
    with pytest.raises(ValueError, match="slippage_bps"):
        FillModel(slippage_bps=-1)


@pytest.mark.parametrize(
    ("mode", "field", "optimistic"),
    [("next_open", "open", False), ("next_close", "close", False), ("same_close", "close", True)],
)
def test_fill_mode_selects_its_reference_price(mode: str, field: str, optimistic: bool):
    model = FillModel(mode=mode, slippage_bps=0.0)
    assert model.price_field == field
    assert model.optimistic is optimistic
    assert model.same_bar is (mode == "same_close")
    assert model.fill_price(Side.BUY, BAR) == pytest.approx(BAR[field])
    assert set(MODES) == {"next_open", "next_close", "same_close"}


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_slippage_always_works_against_the_trader(side: Side):
    model = FillModel(mode="next_open", slippage_bps=25.0)
    price = model.fill_price(side, BAR)
    base = BAR["open"]
    if side is Side.BUY:
        assert price > base
    else:
        assert price < base
    assert price == pytest.approx(base * (1 + (1 if side is Side.BUY else -1) * 0.0025))


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
@pytest.mark.parametrize("bps", [0.0, 5.0, 500.0, 50_000.0])
def test_slippage_never_escapes_the_bar(side: Side, bps: float):
    model = FillModel(mode="next_open", slippage_bps=bps)
    for bar in (
        BAR,
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        {"open": 99.0, "high": 130.0, "low": 20.0, "close": 25.0, "volume": 1.0},
    ):
        price = model.fill_price(side, bar)
        assert bar["low"] <= price <= bar["high"]
        assert price > 0


def test_fill_price_falls_back_to_close_and_then_refuses():
    model = FillModel(mode="next_open", slippage_bps=0.0)
    assert model.fill_price(Side.BUY, {"close": 50.0, "high": 60.0, "low": 40.0}) == 50.0
    assert model.fill_price(
        Side.BUY, {"open": float("nan"), "close": 50.0, "high": 60.0, "low": 40.0}
    ) == 50.0
    with pytest.raises(ValueError, match="no usable price"):
        model.fill_price(Side.BUY, {"volume": 1.0})


# --- FillModel: limits, caps, commission --------------------------------------


def test_a_buy_limit_fills_only_when_the_bar_trades_down_to_it():
    model = FillModel(mode="next_open", slippage_bps=0.0)
    o = order(side=Side.BUY, order_type=OrderType.LIMIT, limit_price=97.0)

    assert model.fill(o, BAR, TS[1]) is not None, "bar low 96 <= limit 97"
    assert model.fill(o, dict(BAR, low=98.0), TS[1]) is None, "never traded down to 97"


def test_a_sell_limit_fills_only_when_the_bar_trades_up_to_it():
    model = FillModel(mode="next_open", slippage_bps=0.0)
    o = order(side=Side.SELL, order_type=OrderType.LIMIT, limit_price=103.0)

    assert model.fill(o, BAR, TS[1]) is not None, "bar high 104 >= limit 103"
    assert model.fill(o, dict(BAR, high=102.0), TS[1]) is None


def test_a_limit_fill_is_never_worse_than_the_limit():
    model = FillModel(mode="next_open", slippage_bps=100.0)  # 1% of 100 = 101
    buy_fill = model.fill(
        order(side=Side.BUY, order_type=OrderType.LIMIT, limit_price=100.5), BAR, TS[1]
    )
    assert buy_fill is not None and buy_fill.price == pytest.approx(100.5)

    sell_fill = model.fill(
        order(side=Side.SELL, order_type=OrderType.LIMIT, limit_price=99.5), BAR, TS[1]
    )
    assert sell_fill is not None and sell_fill.price == pytest.approx(99.5)


def test_partial_fill_caps_quantity_at_a_share_of_bar_volume():
    model = FillModel(mode="next_open", slippage_bps=0.0, partial_fill_volume_pct=0.01)
    bar = dict(BAR, volume=1_000.0)  # cap = 10 shares

    assert model.fill(order(qty=4), bar, TS[1]).qty == 4
    assert model.fill(order(qty=250), bar, TS[1]).qty == 10
    assert model.fill(order(side=Side.SELL, qty=250), bar, TS[1]).qty == 10


def test_the_volume_cap_respects_fractional_settings_and_missing_volume():
    capped = FillModel(mode="next_open", partial_fill_volume_pct=0.001)  # cap 10.5 of 10,500
    bar = dict(BAR, volume=10_500.0)
    assert capped.cap_qty(100.0, bar) == 10.0, "whole shares by default"

    frac = FillModel(mode="next_open", partial_fill_volume_pct=0.001, allow_fractional=True)
    assert frac.cap_qty(100.0, bar) == pytest.approx(10.5)

    assert capped.cap_qty(100.0, dict(BAR, volume=0.0)) == 100.0
    assert capped.cap_qty(100.0, {"close": 1.0}) == 100.0
    assert FillModel().cap_qty(100.0, bar) == 100.0, "no cap configured"


def test_a_cap_that_rounds_to_zero_produces_no_fill():
    model = FillModel(mode="next_open", partial_fill_volume_pct=0.0001)  # cap 0.1 share
    assert model.fill(order(qty=50), dict(BAR, volume=1_000.0), TS[1]) is None


def test_commission_is_per_order_plus_per_share_with_a_floor():
    model = FillModel(commission_per_order=1.0, commission_per_share=0.005, min_commission=2.0)
    assert model.commission(100, 10.0) == pytest.approx(1.5 + 0.5)  # floored to 2.0
    assert model.commission(1_000, 10.0) == pytest.approx(6.0)
    assert model.commission(0, 10.0) == pytest.approx(1.0), "the floor needs a quantity"
    assert FillModel().commission(100, 10.0) == 0.0


def test_fill_records_slippage_against_the_reference_price():
    model = FillModel(mode="next_open", slippage_bps=50.0, commission_per_share=0.01)
    f = model.fill(order(qty=10), BAR, TS[1])
    assert f is not None
    assert f.price == pytest.approx(100.5)
    assert f.slippage == pytest.approx(0.5 * 10)
    assert f.commission == pytest.approx(0.1)
    assert f.at == TS[1]
    assert f.side is Side.BUY


def test_fill_refuses_impossible_orders():
    model = FillModel()
    assert model.fill(order(qty=0), BAR, TS[1]) is None
    assert model.fill(order(qty=-5), BAR, TS[1]) is None
    assert model.fill(order(), None, TS[1]) is None


def test_fill_model_from_mapping_ignores_unknown_keys():
    model = FillModel.from_mapping({"mode": "next_close", "slippage_bps": 1.0, "nonsense": 7})
    assert model.mode == "next_close" and model.slippage_bps == 1.0
    assert FillModel.from_mapping(None).mode == "next_open"
    assert FillModel.from_mapping({}).to_dict()["optimistic"] is False


# --- SimBroker ----------------------------------------------------------------


def broker(**fill_kwargs: Any) -> tuple[SimBroker, Portfolio]:
    portfolio = Portfolio(100_000.0)
    model = FillModel(mode="next_open", slippage_bps=0.0, **fill_kwargs)
    return SimBroker(portfolio, model), portfolio


def test_submit_is_idempotent_on_idem_key():
    b, _ = broker()
    key = Order.make_idem_key("probe", date(2024, 1, 2), "AAA", Side.BUY)
    first = b.submit(order(idem_key=key))
    again = b.submit(order(idem_key=key, qty=999))

    assert again is first, "a repeat must return the original order"
    assert len(b.open_orders()) == 1
    assert b.open_orders()[0].qty == 10, "the duplicate's quantity is discarded"
    assert first.status is OrderStatus.SUBMITTED


def test_orders_without_an_idem_key_are_not_deduplicated():
    b, _ = broker()
    b.submit(order())
    b.submit(order())
    assert len(b.open_orders()) == 2


def test_idempotency_survives_a_fill_and_a_resubmit():
    b, _ = broker()
    key = Order.make_idem_key("probe", date(2024, 1, 2), "AAA", Side.BUY)
    first = b.submit(order(idem_key=key))
    b.process(TS[1], {"AAA": BAR})
    assert first.status is OrderStatus.FILLED

    replay = b.submit(order(idem_key=key))
    assert replay is first
    assert b.open_orders() == [], "the replay must not queue a second order"


def test_an_order_with_no_bar_stays_pending_instead_of_vanishing():
    b, portfolio = broker()
    o = b.submit(order(ticker="HALTED"))

    assert b.process(TS[1], {"AAA": BAR}) == []
    assert b.open_orders() == [o], "a halted name delays a fill, it does not cancel it"
    assert portfolio.positions == {}

    fills = b.process(TS[2], {"HALTED": BAR})
    assert [f.ticker for f in fills] == ["HALTED"]
    assert b.open_orders() == []


def test_process_applies_fills_to_the_portfolio():
    b, portfolio = broker()
    b.submit(order(qty=10))
    (fill,) = b.process(TS[1], {"AAA": BAR})

    assert portfolio.position("AAA").qty == 10
    assert portfolio.cash == pytest.approx(100_000.0 - 10 * fill.price)
    assert b.fills == [fill]
    assert b.account()["equity"] == pytest.approx(portfolio.equity)
    assert b.positions()["AAA"].qty == 10


def test_a_partially_filled_order_stays_pending_for_the_remainder():
    b, portfolio = broker(partial_fill_volume_pct=0.01)
    o = b.submit(order(qty=25))
    bar = dict(BAR, volume=1_000.0)  # cap = 10 shares per bar

    first = b.process(TS[1], {"AAA": bar})
    assert first[0].qty == 10
    assert o.status is OrderStatus.PARTIAL
    assert o.qty == 15
    assert b.open_orders() == [o]

    b.process(TS[2], {"AAA": bar})
    b.process(TS[3], {"AAA": bar})
    assert b.open_orders() == []
    assert o.status is OrderStatus.FILLED
    assert portfolio.position("AAA").qty == 25


def test_cancel_and_expire_clear_the_queue():
    b, _ = broker()
    a = b.submit(order(ticker="AAA"))
    c = b.submit(order(ticker="BBB"))

    assert b.cancel(a.id) is True
    assert a.status is OrderStatus.CANCELED
    assert b.cancel(a.id) is False, "cancelling twice is not an error, just a no-op"
    assert b.open_orders() == [c]

    expired = b.expire_pending("session over")
    assert expired == [c]
    assert c.status is OrderStatus.EXPIRED and c.reason == "session over"
    assert b.open_orders() == []
    assert {o.id for o in b.orders} == {a.id, c.id}, "the audit trail keeps both"


def test_cancel_all_can_be_scoped_to_a_ticker():
    b, _ = broker()
    b.submit(order(ticker="AAA"))
    b.submit(order(ticker="AAA"))
    b.submit(order(ticker="BBB"))

    assert b.cancel_all("aaa") == 2
    assert [o.ticker for o in b.open_orders()] == ["BBB"]
    assert b.cancel_all() == 1
    assert b.open_orders() == []


# --- the strategy loader ------------------------------------------------------

_ALL_FORMS = '''
PARAMS = {"n": 3, "shared": "module"}

class Other:
    def on_bar(self, ctx): pass

class Strategy:
    def on_bar(self, ctx): pass

def build(params):
    obj = Other()
    obj.origin = "build"
    return obj

class Named:
    def __init__(self, params=None):
        self.params = dict(params or {})
        self.origin = "STRATEGY"
    def on_bar(self, ctx): pass

STRATEGY = Named()
'''


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_loader_prefers_an_explicit_strategy_object(tmp_path: Path):
    loaded = load_strategy(write(tmp_path, "s_all", _ALL_FORMS))
    assert loaded.instance.origin == "STRATEGY"
    assert loaded.name == "s_all"


def test_loader_falls_back_to_build(tmp_path: Path):
    body = _ALL_FORMS.replace("STRATEGY = Named()", "")
    loaded = load_strategy(write(tmp_path, "s_build", body), {"n": 9})
    assert loaded.instance.origin == "build"


def test_loader_then_falls_back_to_a_class_named_strategy(tmp_path: Path):
    body = _ALL_FORMS.replace("STRATEGY = Named()", "").replace(
        "def build(params):\n    obj = Other()\n    obj.origin = \"build\"\n    return obj\n", ""
    )
    loaded = load_strategy(write(tmp_path, "s_class", body))
    assert type(loaded.instance).__name__ == "Strategy"


def test_loader_finally_takes_the_single_on_bar_class(tmp_path: Path):
    body = 'class OnlyOne:\n    def on_bar(self, ctx): pass\n'
    loaded = load_strategy(write(tmp_path, "s_single", body))
    assert type(loaded.instance).__name__ == "OnlyOne"


def test_loader_refuses_an_ambiguous_module(tmp_path: Path):
    body = (
        "class Alpha:\n    def on_bar(self, ctx): pass\n\n"
        "class Beta:\n    def on_bar(self, ctx): pass\n"
    )
    with pytest.raises(ValueError, match="several strategy classes"):
        load_strategy(write(tmp_path, "s_ambiguous", body))


def test_loader_reports_a_module_with_no_strategy(tmp_path: Path):
    with pytest.raises(ValueError, match="exposes no strategy"):
        load_strategy(write(tmp_path, "s_empty", "X = 1\n"))


def test_loader_rejects_a_strategy_without_a_callable_on_bar(tmp_path: Path):
    body = "class Broken:\n    on_bar = 42\n\nSTRATEGY = Broken()\n"
    with pytest.raises(ValueError, match="no callable on_bar"):
        load_strategy(write(tmp_path, "s_broken", body))


def test_caller_params_win_over_module_defaults(tmp_path: Path):
    body = (
        'PARAMS = {"n": 3, "shared": "module"}\n\n'
        "class S:\n"
        "    def __init__(self, params=None):\n        self.params = dict(params or {})\n"
        "    def on_bar(self, ctx): pass\n"
    )
    loaded = load_strategy(write(tmp_path, "s_params", body), {"shared": "caller", "extra": 1})
    assert loaded.params == {"n": 3, "shared": "caller", "extra": 1}
    assert loaded.instance.params == loaded.params


def test_loaded_strategy_carries_provenance(tmp_path: Path):
    path = write(tmp_path, "s_prov", '"""A docstring."""\n\nclass S:\n    def on_bar(self, c): pass\n')
    loaded = load_strategy(path)

    assert loaded.path == path.resolve()
    assert loaded.doc == "A docstring."
    assert len(loaded.source_hash) == 16
    assert load_strategy(path).source_hash == loaded.source_hash, "hash is content-addressed"
    assert loaded.hook("on_bar") is not None
    assert loaded.hook("on_fill") is None
    assert loaded.to_dict()["name"] == "s_prov"


def test_loader_hooks_are_only_reported_when_defined(tmp_path: Path):
    body = (
        "class S:\n"
        "    def on_start(self, ctx): pass\n"
        "    def on_bar(self, ctx): pass\n"
        "    def on_fill(self, ctx, fill): pass\n"
    )
    loaded = load_strategy(write(tmp_path, "s_hooks", body))
    assert loaded.hook("on_start") is not None
    assert loaded.hook("on_fill") is not None
    assert loaded.hook("on_stop") is None


def test_resolve_path_accepts_a_bare_strategy_name():
    resolved = resolve_path("buy_and_hold")
    assert resolved.name == "buy_and_hold.py" and resolved.is_absolute()
    assert resolve_path("buy_and_hold.py") == resolved
    with pytest.raises(FileNotFoundError):
        resolve_path("no_such_strategy_anywhere")


def test_discover_reports_loadable_and_broken_files(tmp_path: Path):
    write(tmp_path, "good", "class S:\n    def on_bar(self, ctx): pass\n")
    write(tmp_path, "bad", "raise RuntimeError('boom')\n")
    write(tmp_path, "_hidden", "class S:\n    def on_bar(self, ctx): pass\n")

    found = {d["name"]: d for d in discover(tmp_path)}
    assert set(found) == {"good", "bad"}, "underscore-prefixed files are skipped"
    assert found["good"]["loadable"] is True
    assert found["bad"]["loadable"] is False
    assert "RuntimeError" in found["bad"]["error"]
    assert discover(tmp_path / "nowhere") == []
