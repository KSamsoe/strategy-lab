"""Portfolio accounting tests.

Every metric in every report is downstream of these numbers, so the tests here
are about *identities* rather than examples: cash equals the fill ledger, equity
equals cash plus marks, a round trip closes exactly when the position returns to
flat. The subtle case -- one fill crossing through zero, closing a long and
opening a short in the same trade -- gets its own section.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pytest

from lab.engine.events import Fill, Side, Trade, new_id
from lab.engine.portfolio import EPS, Portfolio, ReadOnlyPortfolio
from lab.timeutil import UTC

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def fill(
    ticker: str,
    side: Side,
    qty: float,
    price: float,
    *,
    commission: float = 0.0,
    at: datetime | None = None,
    tag: str = "",
) -> Fill:
    return Fill(
        id=new_id("f_"),
        order_id=new_id("o_"),
        ticker=ticker,
        side=side,
        qty=qty,
        price=price,
        at=at or T0,
        commission=commission,
        tag=tag,
    )


def buy(ticker: str, qty: float, price: float, **kw) -> Fill:
    return fill(ticker, Side.BUY, qty, price, **kw)


def sell(ticker: str, qty: float, price: float, **kw) -> Fill:
    return fill(ticker, Side.SELL, qty, price, **kw)


def ledger_cash(start: float, fills: Iterable[Fill]) -> float:
    """Cash implied by the fills alone -- the independent check on ``_cash``."""
    cash = start
    for f in fills:
        cash -= f.signed_qty * f.price
        cash -= f.commission
    return cash


# --- basis --------------------------------------------------------------------


def test_average_cost_basis_across_adds():
    p = Portfolio(100_000.0)
    p.apply_fill(buy("AAA", 100, 10.0))
    p.apply_fill(buy("AAA", 300, 14.0))

    pos = p.position("AAA")
    assert pos.qty == 400
    # (100*10 + 300*14) / 400
    assert pos.avg_price == pytest.approx(13.0)
    assert pos.cost_basis == pytest.approx(5_200.0)


def test_partial_reduction_leaves_basis_unchanged():
    p = Portfolio(100_000.0)
    p.apply_fill(buy("AAA", 400, 13.0))
    closed = p.apply_fill(sell("AAA", 150, 20.0))

    pos = p.position("AAA")
    assert closed is not None, "a scale-out is a booked exit, not an invisible one"
    assert closed.qty == 150
    assert closed.pnl == pytest.approx(150 * (20.0 - 13.0))
    assert pos.qty == 250
    assert pos.avg_price == pytest.approx(13.0), "basis must not drift on a reduction"
    assert pos.realized_pnl == pytest.approx(150 * (20.0 - 13.0))
    assert p.reconcile()["residual"] == pytest.approx(0.0)


def test_adding_to_a_short_averages_the_basis():
    p = Portfolio(100_000.0)
    p.apply_fill(sell("AAA", 100, 20.0))
    p.apply_fill(sell("AAA", 100, 30.0))

    pos = p.position("AAA")
    assert pos.qty == -200
    assert pos.avg_price == pytest.approx(25.0)


# --- round trips --------------------------------------------------------------


def test_round_trip_closes_at_flat_with_commissions_charged():
    p = Portfolio(100_000.0)
    p.apply_fill(buy("AAA", 100, 10.0, commission=1.0))
    closed = p.apply_fill(sell("AAA", 100, 12.0, commission=1.5, tag="exit"))

    assert isinstance(closed, Trade)
    assert closed.side == "long"
    assert closed.qty == 100
    assert closed.entry_price == pytest.approx(10.0)
    assert closed.exit_price == pytest.approx(12.0)
    assert closed.commission == pytest.approx(2.5), "both legs' commissions"
    assert closed.pnl == pytest.approx(200.0 - 2.5)
    assert closed.pnl_pct == pytest.approx((200.0 - 2.5) / 1_000.0)
    assert closed.exit_reason == "exit"
    assert p.position("AAA").is_flat
    assert p.trades() == [closed]


def test_short_round_trip_profits_when_price_falls():
    p = Portfolio(100_000.0)
    p.apply_fill(sell("AAA", 50, 20.0))
    assert p.position("AAA").qty == -50
    assert p.position("AAA").market_value == pytest.approx(-1_000.0)

    closed = p.apply_fill(buy("AAA", 50, 15.0))
    assert closed is not None
    assert closed.side == "short"
    assert closed.pnl == pytest.approx(250.0)
    assert p.cash == pytest.approx(100_000.0 + 250.0)


def test_fill_crossing_through_zero_closes_one_trip_and_opens_the_opposite():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 100, 10.0, commission=1.0))
    closed = p.apply_fill(sell("AAA", 150, 12.0, commission=1.5))

    assert closed is not None, "crossing zero must book the old round trip"
    assert closed.side == "long"
    assert closed.qty == 100, "only the crossed quantity is closed"
    assert closed.exit_price == pytest.approx(12.0)

    pos = p.position("AAA")
    assert pos.qty == -50, "the residual opens the other side"
    assert pos.avg_price == pytest.approx(12.0), "residual is based at the crossing price"
    assert pos.opened_at == T0
    assert len(p.open_trades()) == 1
    assert p.open_trades()[0].side == "short"
    assert p.open_trades()[0].entry_price == pytest.approx(12.0)

    final = p.apply_fill(buy("AAA", 50, 11.0, commission=0.5))
    assert final is not None and final.side == "short"
    assert final.pnl == pytest.approx(50.0 * (12.0 - 11.0) - 2.0)
    assert [t.side for t in p.trades()] == ["long", "short"]
    assert p.position("AAA").is_flat


def test_crossing_fill_commission_lands_on_both_legs_but_cash_pays_it_once():
    """Characterization: the crossing fill's commission is charged to the trade
    it closes *and* carried into the trade it opens, so the ledger's summed P&L
    understates cash P&L by exactly that commission. Cash is the authority."""
    fills = [
        buy("AAA", 100, 10.0, commission=1.0),
        sell("AAA", 150, 12.0, commission=1.5),
        buy("AAA", 50, 11.0, commission=0.5),
    ]
    p = Portfolio(10_000.0)
    for f in fills:
        p.apply_fill(f)

    assert p.cash == pytest.approx(ledger_cash(10_000.0, fills))
    cash_pnl = p.cash - 10_000.0
    ledger_pnl = sum(t.pnl for t in p.trades())
    assert cash_pnl - ledger_pnl == pytest.approx(1.5)
    assert p.total_commission == pytest.approx(3.0)


def test_scaling_out_books_every_leg_and_the_ledger_reconciles():
    """A trade reduced in stages books one ledger row per leg.

    The bug this pins is the one that made a +35% backtest report a single +4%
    trade: realized money from a trim went into cash but produced no ledger row,
    so hit_rate, profit_factor, avg_win and expectancy were all computed off an
    unrepresentative fraction of the run. The invariant that catches it is
    `reconcile()["residual"] == 0` -- every dollar the book made is explained by
    a trade or by an open position, with nothing left over.
    """
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 100, 10.0, commission=1.0))

    first = p.apply_fill(sell("AAA", 60, 20.0, commission=1.0))
    assert first is not None, "the trim is a real exit and must be booked now"
    assert first.qty == 60
    # Entry commission is split pro-rata, so it is charged once across the legs.
    assert first.pnl == pytest.approx(60 * 10.0 - (0.6 + 1.0))

    second = p.apply_fill(sell("AAA", 40, 10.0, commission=1.0))
    assert second is not None and second.qty == 40
    assert second.pnl == pytest.approx(40 * 0.0 - (0.4 + 1.0))

    trades = p.trades()
    assert len(trades) == 2, "one row per exit leg"
    assert sum(t.pnl for t in trades) == pytest.approx(600.0 - 3.0)
    assert sum(t.pnl for t in trades) == pytest.approx(p.cash - 10_000.0)
    assert sum(t.commission for t in trades) == pytest.approx(3.0)
    assert p.position("AAA").realized_pnl == pytest.approx(600.0)
    assert p.reconcile()["residual"] == pytest.approx(0.0)


def test_an_open_scaled_out_position_still_reconciles():
    """The reported case: profit taken by trims, position never flat."""
    p = Portfolio(100_000.0)
    p.apply_fill(buy("AAA", 100, 100.0))
    p.mark({"AAA": 150.0})
    p.apply_fill(sell("AAA", 50, 150.0))
    p.mark({"AAA": 150.0})

    r = p.reconcile()
    assert r["equity_gain"] == pytest.approx(5_000.0)
    assert r["ledger_pnl"] == pytest.approx(2_500.0), "the trim is visible"
    assert r["unrealized_pnl"] == pytest.approx(2_500.0), "the rest is still open"
    assert r["residual"] == pytest.approx(0.0), "nothing unexplained"


def test_commission_on_an_add_is_charged_to_the_round_trip():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 100, 10.0, commission=1.0))
    p.apply_fill(buy("AAA", 100, 10.0, commission=1.0))
    closed = p.apply_fill(sell("AAA", 200, 10.0, commission=1.0))

    assert closed is not None
    assert closed.commission == pytest.approx(3.0)
    assert closed.pnl == pytest.approx(-3.0)
    assert closed.pnl == pytest.approx(p.cash - 10_000.0)


def test_bars_held_is_populated_on_close():
    p = Portfolio(100_000.0)
    p.tick_bar()
    p.apply_fill(buy("AAA", 10, 10.0))
    for _ in range(4):
        p.tick_bar()
    closed = p.apply_fill(sell("AAA", 10, 11.0))

    assert closed is not None
    assert closed.bars_held == 4
    # An add mid-trade must not reset the entry bar.
    p.apply_fill(buy("BBB", 10, 10.0))
    p.tick_bar()
    p.apply_fill(buy("BBB", 10, 10.0))
    p.tick_bar()
    closed_b = p.apply_fill(sell("BBB", 20, 10.0))
    assert closed_b is not None and closed_b.bars_held == 2


# --- cash, equity, exposure ---------------------------------------------------


def test_cash_reconciles_with_the_trade_ledger_exactly():
    fills = [
        buy("AAA", 100, 10.0, commission=1.0),
        buy("BBB", 40, 25.0, commission=0.5),
        sell("AAA", 60, 11.0, commission=0.75),
        sell("BBB", 40, 24.0, commission=0.5),
        sell("AAA", 40, 9.0, commission=0.25),
    ]
    p = Portfolio(50_000.0)
    for f in fills:
        p.apply_fill(f)
        assert p.cash == pytest.approx(ledger_cash(50_000.0, fills[: fills.index(f) + 1]))

    assert p.cash == pytest.approx(ledger_cash(50_000.0, fills))
    assert p.total_commission == pytest.approx(3.0)
    assert p.turnover_notional == pytest.approx(
        sum(abs(f.notional) for f in fills)
    )
    # Flat at the end, so cash P&L is exactly the ledger's P&L (no crossings).
    assert not p.positions
    assert p.cash - 50_000.0 == pytest.approx(sum(t.pnl for t in p.trades()))


def test_equity_is_cash_plus_market_value_at_every_step():
    p = Portfolio(20_000.0)
    steps = [
        (buy("AAA", 100, 10.0, commission=1.0), {"AAA": 10.5}),
        (buy("BBB", 50, 40.0, commission=1.0), {"AAA": 11.0, "BBB": 39.0}),
        (sell("AAA", 100, 11.0, commission=1.0), {"BBB": 41.0}),
    ]
    for f, marks in steps:
        p.apply_fill(f)
        p.mark(marks)
        expected = p.cash + sum(pos.qty * pos.last_price for pos in p.positions.values())
        assert p.equity == pytest.approx(expected)


def test_mark_ignores_unknown_and_nan_prices():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 10, 10.0))
    p.mark({"AAA": 12.0})
    before = p.equity
    p.mark({"ZZZ": 99.0, "AAA": float("nan")})
    assert p.equity == pytest.approx(before), "a NaN mark must not wipe the position"
    assert p.position("AAA").last_price == pytest.approx(12.0)


def test_weight_and_gross_exposure_account_for_shorts():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 100, 20.0))     # +2,000
    p.apply_fill(sell("BBB", 100, 10.0))    # -1,000
    p.mark({"AAA": 20.0, "BBB": 10.0})

    assert p.equity == pytest.approx(10_000.0)
    assert p.weight("AAA") == pytest.approx(0.2)
    assert p.weight("BBB") == pytest.approx(-0.1)
    assert p.weight("ZZZ") == 0.0
    assert p.gross_exposure == pytest.approx(0.3)
    assert p.net_exposure == pytest.approx(0.1)


def test_flat_positions_disappear_from_the_book_but_not_the_ledger():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 10, 10.0))
    p.apply_fill(sell("AAA", 10, 10.0))
    assert "AAA" not in p.positions
    assert p.position("AAA").qty == 0.0
    assert len(p.trades()) == 1


def test_dust_below_eps_is_swept_to_flat():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 1.0, 10.0))
    p.apply_fill(sell("AAA", 1.0 - EPS / 10, 10.0))
    assert p.position("AAA").qty == 0.0
    assert not p.positions


# --- sizing -------------------------------------------------------------------


def test_target_to_delta_shares_integer_and_fractional_paths():
    p = Portfolio(10_000.0)
    # 10% of 10,000 = 1,000 -> 33.33 shares at 30.
    assert p.target_to_delta_shares("AAA", 0.10, 30.0) == 33.0
    assert p.target_to_delta_shares("AAA", 0.10, 30.0, allow_fractional=True) == pytest.approx(
        1_000.0 / 30.0
    )


def test_target_to_delta_shares_is_relative_to_the_current_position():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 50, 20.0))   # 1,000 of a 10,000 book
    p.mark({"AAA": 20.0})
    assert p.equity == pytest.approx(10_000.0)

    assert p.target_to_delta_shares("AAA", 0.20, 20.0) == 50.0    # up to 2,000
    assert p.target_to_delta_shares("AAA", 0.05, 20.0) == -25.0   # down to 500
    assert p.target_to_delta_shares("AAA", 0.10, 20.0) == 0.0     # already there
    assert p.target_to_delta_shares("AAA", -0.10, 20.0) == -100.0 # flip to short


def test_target_to_delta_shares_refuses_unusable_prices():
    p = Portfolio(10_000.0)
    assert p.target_to_delta_shares("AAA", 0.5, 0.0) == 0.0
    assert p.target_to_delta_shares("AAA", 0.5, -1.0) == 0.0
    assert p.target_to_delta_shares("AAA", 0.5, None) == 0.0
    # Integer truncation of a sub-share delta is a no-trade, not a rounding up.
    assert p.target_to_delta_shares("AAA", 0.00001, 1_000.0) == 0.0


# --- the read-only view -------------------------------------------------------

READ_ONLY_SURFACE = {
    "cash",
    "equity",
    "positions",
    "gross_exposure",
    "net_exposure",
    "starting_cash",
    "position",
    "weight",
    "snapshot",
}

MUTATORS = (
    "apply_fill",
    "mark",
    "tick_bar",
    "target_to_delta_shares",
    "trades",
    "open_trades",
    "total_commission",
    "turnover_notional",
)


def test_readonly_portfolio_exposes_no_mutator():
    p = Portfolio(10_000.0)
    ro = ReadOnlyPortfolio(p)

    assert {n for n in dir(ro) if not n.startswith("_")} == READ_ONLY_SURFACE
    for name in MUTATORS:
        assert not hasattr(ro, name), f"{name} leaked into the read-only view"


@pytest.mark.parametrize("attr", ["cash", "equity", "positions", "anything_new"])
def test_readonly_portfolio_rejects_assignment(attr):
    ro = ReadOnlyPortfolio(Portfolio(10_000.0))
    with pytest.raises(AttributeError):
        setattr(ro, attr, 1.0)


def test_readonly_portfolio_hands_out_a_copy_of_the_book():
    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 10, 10.0))
    ro = ReadOnlyPortfolio(p)

    snapshot = ro.positions
    snapshot.pop("AAA")
    assert "AAA" in ro.positions, "mutating the returned mapping must not touch the book"


def test_readonly_portfolio_tracks_the_live_portfolio():
    p = Portfolio(10_000.0)
    ro = ReadOnlyPortfolio(p)
    assert ro.equity == pytest.approx(10_000.0)

    p.apply_fill(buy("AAA", 100, 10.0))
    p.mark({"AAA": 12.0})
    assert ro.cash == pytest.approx(9_000.0)
    assert ro.equity == pytest.approx(10_200.0)
    assert ro.weight("AAA") == pytest.approx(1_200.0 / 10_200.0)
    assert ro.gross_exposure == pytest.approx(1_200.0 / 10_200.0)
    assert ro.starting_cash == pytest.approx(10_000.0)


def test_snapshot_is_json_shaped():
    import json

    p = Portfolio(10_000.0)
    p.apply_fill(buy("AAA", 10, 10.0, at=T0))
    p.mark({"AAA": 11.0})
    snap = ReadOnlyPortfolio(p).snapshot(T0 + timedelta(days=1))

    assert json.loads(json.dumps(snap)) == snap
    assert snap["n_positions"] == 1
    assert snap["positions"][0]["ticker"] == "AAA"
    assert snap["equity"] == pytest.approx(p.equity)
