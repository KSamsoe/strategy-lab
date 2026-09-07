"""Risk-gate tests.

This is the module most worth over-testing: every rule name is exercised, and
the two properties that matter most -- exits are never trapped, and blocks hold
rather than liquidate -- are asserted against every capacity rule in turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import pytest

from lab.config import reset_settings_cache
from lab.engine.events import GateAction, Intent, Order, OrderType, Position, Side, new_id
from lab.risk.gate import RULES, GateState, RiskGate, RiskLimits
from lab.timeutil import UTC

NOW = datetime(2024, 3, 5, 14, 35, tzinfo=UTC)  # 09:35 ET, a Tuesday
DAY2 = NOW + timedelta(days=1)
DAY3 = NOW + timedelta(days=2)
DAY4 = NOW + timedelta(days=3)
PRICES = {t: 100.0 for t in ("AAPL", "MSFT", "NVDA", "XOM", "XYZ", "SPY")}
CFG_LIMITS = Path(__file__).resolve().parent.parent / "cfg" / "limits.yaml"


class FakePortfolio:
    """Minimal PortfolioView: cash plus (qty, mark) holdings."""

    def __init__(
        self, cash: float = 100_000.0, holdings: Mapping[str, tuple[float, float]] | None = None
    ) -> None:
        self._cash = float(cash)
        self._pos = {
            t.upper(): Position(ticker=t.upper(), qty=float(q), avg_price=float(p), last_price=float(p))
            for t, (q, p) in (holdings or {}).items()
        }

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._pos.values())

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._pos)

    def position(self, ticker: str) -> Position:
        return self._pos.get(ticker.upper(), Position(ticker=ticker.upper()))

    def weight(self, ticker: str) -> float:
        pos = self._pos.get(ticker.upper())
        return 0.0 if pos is None else pos.market_value / self.equity

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self._pos.values()) / self.equity


def portfolio(equity: float = 100_000.0, weights: Mapping[str, float] | None = None) -> FakePortfolio:
    weights = weights or {}
    holdings = {t.upper(): (w * equity / 100.0, 100.0) for t, w in weights.items()}
    return FakePortfolio(equity - sum(w * equity for w in weights.values()), holdings)


def gate(state: GateState | None = None, sectors: Mapping[str, str] | None = None, **limits) -> RiskGate:
    return RiskGate(RiskLimits(**limits), sectors=sectors, state=state)


def verdict(g: RiskGate, *intents: Intent, port: FakePortfolio | None = None, now: datetime = NOW):
    out = g.evaluate(list(intents), portfolio=port or portfolio(), now=now, prices=PRICES)
    return out[0] if len(intents) == 1 else out


def order(ticker: str, side: Side = Side.BUY, qty: float = 10.0, at: datetime | None = NOW) -> Order:
    return Order(
        id=new_id("o_"),
        ticker=ticker,
        side=side,
        qty=qty,
        order_type=OrderType.MARKET,
        created_at=at,
    )


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Every test gets its own data dir and kill file: a stray sentinel in the
    developer's working tree must not silently pass the kill-switch tests."""
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.delenv("LAB_KILL_SWITCH", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


# --- limits ------------------------------------------------------------------


def test_defaults_match_contract():
    lim = RiskLimits()
    assert (lim.max_position_pct, lim.max_positions, lim.max_sector_positions) == (0.05, 8, 2)
    assert (lim.max_sector_pct, lim.max_gross_exposure, lim.max_daily_loss_pct) == (0.25, 1.0, 0.03)
    assert (lim.max_orders_per_day, lim.min_order_notional) == (20, 50.0)
    assert lim.allowlist == [] and lim.denylist == []
    assert lim.cooldown_days == 0 and lim.max_position_notional is None


def test_shipped_yaml_matches_the_dataclass_defaults():
    assert RiskLimits.from_yaml(CFG_LIMITS) == RiskLimits()


def test_yaml_accepts_a_nested_limits_block_and_ignores_strategy_overrides(tmp_path):
    p = tmp_path / "l.yaml"
    p.write_text(
        "limits:\n  max_positions: 3\nstrategies:\n  momo:\n    max_positions: 99\n",
        encoding="utf-8",
    )
    assert RiskLimits.from_yaml(p).max_positions == 3


def test_from_mapping_rejects_typos_and_bad_values():
    with pytest.raises(ValueError, match="unknown risk limit key"):
        RiskLimits.from_mapping({"max_position_pc": 0.5})
    with pytest.raises(ValueError):
        RiskLimits(max_position_pct=0)
    with pytest.raises(ValueError):
        RiskLimits(max_positions=-1)
    with pytest.raises(ValueError):
        RiskLimits(max_positions=2.5)
    with pytest.raises(ValueError):
        RiskLimits(max_daily_loss_pct=1.5)
    with pytest.raises(ValueError):
        RiskLimits(max_position_notional=0)
    with pytest.raises(ValueError):
        RiskLimits(denylist="AAPL")
    with pytest.raises(ValueError):
        RiskLimits.from_yaml("no/such/file.yaml")


def test_from_mapping_normalizes_and_treats_null_as_default():
    lim = RiskLimits.from_mapping({"allowlist": ["aapl", " msft "], "max_positions": None})
    assert lim.allowlist == ["AAPL", "MSFT"]
    assert lim.max_positions == 8


def test_rule_names_and_order_are_the_contract():
    assert RULES == (
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


# --- the happy path ----------------------------------------------------------


def test_plain_intent_passes_untouched():
    v = verdict(gate(), Intent("AAPL", 0.04))
    assert v.action is GateAction.PASS and v.rule is None
    assert v.approved_pct == pytest.approx(0.04)


def test_one_verdict_per_intent_in_order():
    g = gate()
    vs = g.evaluate(
        [Intent("AAPL", 0.01), Intent("MSFT", 0.01), Intent("NVDA", 0.01)],
        portfolio=portfolio(),
        now=NOW,
        prices=PRICES,
    )
    assert [v.ticker for v in vs] == ["AAPL", "MSFT", "NVDA"]


def test_holding_the_current_target_is_a_no_op_pass_not_a_min_notional_block():
    v = verdict(gate(), Intent("AAPL", 0.05), port=portfolio(weights={"AAPL": 0.05}))
    assert v.action is GateAction.PASS and v.rule is None


# --- clip rules --------------------------------------------------------------


def test_position_cap_clips():
    v = verdict(gate(max_position_pct=0.05), Intent("AAPL", 0.30))
    assert v.action is GateAction.CLIP and v.rule == "position_cap"
    assert v.approved_pct == pytest.approx(0.05)


def test_position_cap_clips_shorts_by_magnitude():
    v = verdict(gate(max_position_pct=0.05), Intent("AAPL", -0.30))
    assert v.rule == "position_cap" and v.approved_pct == pytest.approx(-0.05)


def test_position_notional_clips():
    v = verdict(gate(max_position_notional=2_000.0), Intent("AAPL", 0.04))
    assert v.action is GateAction.CLIP and v.rule == "position_notional"
    assert v.approved_pct == pytest.approx(0.02)


def test_sector_pct_clips():
    g = gate(max_sector_pct=0.10, sectors={"AAPL": "Tech", "MSFT": "Tech"})
    v = verdict(g, Intent("MSFT", 0.05), port=portfolio(weights={"AAPL": 0.07}))
    assert v.action is GateAction.CLIP and v.rule == "sector_pct"
    assert v.approved_pct == pytest.approx(0.03)


def test_gross_exposure_clips():
    g = gate(max_gross_exposure=0.10)
    v = verdict(g, Intent("MSFT", 0.04), port=portfolio(weights={"AAPL": 0.08}))
    assert v.action is GateAction.CLIP and v.rule == "gross_exposure"
    assert v.approved_pct == pytest.approx(0.02)


def test_batch_shares_one_capacity_budget():
    # Three names sized individually inside the cap must not collectively bust it.
    g = gate(max_gross_exposure=0.10)
    vs = g.evaluate(
        [Intent("AAPL", 0.04), Intent("MSFT", 0.04), Intent("NVDA", 0.04)],
        portfolio=portfolio(),
        now=NOW,
        prices=PRICES,
    )
    assert [v.action for v in vs] == [GateAction.PASS, GateAction.PASS, GateAction.CLIP]
    assert vs[2].rule == "gross_exposure"
    assert sum(abs(v.approved_pct) for v in vs) == pytest.approx(0.10)


# --- block rules -------------------------------------------------------------


def test_denylist_blocks():
    v = verdict(gate(denylist=["aapl"]), Intent("AAPL", 0.04))
    assert v.action is GateAction.BLOCK and v.rule == "denylist"


def test_allowlist_blocks_everything_outside_it():
    g = gate(allowlist=["MSFT"])
    assert verdict(g, Intent("AAPL", 0.04)).rule == "allowlist"
    assert verdict(g, Intent("MSFT", 0.04)).action is GateAction.PASS


def test_max_positions_blocks_a_new_name_but_not_an_add():
    g = gate(max_positions=2)
    port = portfolio(weights={"AAPL": 0.04, "MSFT": 0.04})
    assert verdict(g, Intent("NVDA", 0.04), port=port).rule == "max_positions"
    assert verdict(g, Intent("AAPL", 0.05), port=port).action is GateAction.PASS


def test_unmarked_position_still_occupies_a_slot():
    # Price fallback: a position the portfolio has not marked would otherwise
    # look flat and hand out a free slot.
    g = gate(max_positions=1)
    port = FakePortfolio(100_000.0, {"AAPL": (50.0, 0.0)})
    assert verdict(g, Intent("MSFT", 0.04), port=port).rule == "max_positions"


def test_sector_positions_blocks_the_third_name_in_a_sector():
    g = gate(max_sector_positions=2, sectors={"AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech"})
    port = portfolio(weights={"AAPL": 0.04, "MSFT": 0.04})
    assert verdict(g, Intent("NVDA", 0.03), port=port).rule == "sector_positions"


def test_unknown_sector_is_exempt_and_does_not_crash():
    g = gate(max_sector_pct=0.10, max_sector_positions=1, sectors={"AAPL": "Tech"})
    port = portfolio(weights={"AAPL": 0.09})
    v = verdict(g, Intent("XYZ", 0.05), port=port)
    assert v.action is GateAction.PASS and g.sector_of("XYZ") is None


def test_max_orders_per_day_blocks_once_the_budget_is_spent():
    g = gate(max_orders_per_day=1)
    g.note_orders([order("MSFT")])
    v = verdict(g, Intent("AAPL", 0.04))
    assert v.action is GateAction.BLOCK and v.rule == "max_orders_per_day"


def test_order_budget_is_shared_across_a_batch():
    g = gate(max_orders_per_day=2)
    vs = g.evaluate(
        [Intent("AAPL", 0.04), Intent("MSFT", 0.04), Intent("NVDA", 0.04)],
        portfolio=portfolio(),
        now=NOW,
        prices=PRICES,
    )
    assert [v.rule for v in vs] == [None, None, "max_orders_per_day"]


def test_note_orders_is_idempotent_per_order_id():
    g = gate()
    o = order("AAPL")
    g.note_orders([o, o])
    assert g.state.orders_today == 1


def test_min_notional_blocks_dust():
    v = verdict(gate(min_order_notional=500.0), Intent("AAPL", 0.001))
    assert v.action is GateAction.BLOCK and v.rule == "min_notional"


def test_clipping_below_the_minimum_blocks_instead():
    g = gate(max_position_pct=0.0004, min_order_notional=50.0)
    v = verdict(g, Intent("AAPL", 0.05))
    assert v.action is GateAction.BLOCK and v.rule == "min_notional"
    assert "position_cap" in v.detail
    assert v.approved_pct == 0.0


def test_a_sector_with_no_room_left_blocks_rather_than_approving_nothing():
    g = gate(max_sector_pct=0.05, sectors={"AAPL": "Tech", "MSFT": "Tech"})
    v = verdict(g, Intent("MSFT", 0.04), port=portfolio(weights={"AAPL": 0.05}))
    assert v.action is GateAction.BLOCK and v.rule == "min_notional"
    assert "sector_pct" in v.detail


def test_cooldown_blocks_re_entry_then_expires():
    g = gate(cooldown_days=3)
    g.note_orders([order("AAPL", side=Side.SELL, at=NOW)])
    assert verdict(g, Intent("AAPL", 0.04), now=DAY2).rule == "cooldown"
    assert verdict(g, Intent("AAPL", 0.04), now=DAY4).action is GateAction.PASS


def test_cooldown_off_by_default():
    g = gate()
    g.note_orders([order("AAPL", side=Side.SELL, at=NOW)])
    assert verdict(g, Intent("AAPL", 0.04), now=DAY2).action is GateAction.PASS


# --- blocks hold, they do not liquidate --------------------------------------


def test_block_approves_the_current_weight_not_zero():
    g = gate(denylist=["AAPL"])
    v = verdict(g, Intent("AAPL", 0.20), port=portfolio(weights={"AAPL": 0.05}))
    assert v.action is GateAction.BLOCK
    assert v.approved_pct == pytest.approx(0.05)  # a 0.0 here would liquidate


# --- the exit property -------------------------------------------------------

EXIT_HOSTILE = {
    "denylist": dict(denylist=["AAPL"]),
    "max_positions": dict(max_positions=0),
    "gross_exposure": dict(max_gross_exposure=0.001),
    "position_cap": dict(max_position_pct=0.0001),
    "position_notional": dict(max_position_notional=1.0),
    "min_notional": dict(min_order_notional=1e9),
    "max_orders_per_day": dict(max_orders_per_day=0),
    "allowlist": dict(allowlist=["MSFT"]),
    "cooldown": dict(cooldown_days=30),
    "sector_pct": dict(max_sector_pct=0.0001),
}


@pytest.mark.parametrize("name,limits", sorted(EXIT_HOSTILE.items()))
@pytest.mark.parametrize("target", [0.0, 0.02])
def test_exit_is_never_blocked_by_a_capacity_rule(name, limits, target):
    g = gate(sectors={"AAPL": "Tech"}, **limits)
    g.note_exit("AAPL", NOW)
    port = portfolio(weights={"AAPL": 0.05})
    v = verdict(g, Intent("AAPL", target), port=port, now=DAY2)
    assert v.action is GateAction.PASS, f"{name} trapped an exit"
    assert v.approved_pct == pytest.approx(target)


def test_reducing_a_short_is_an_exit():
    g = gate(denylist=["AAPL"])
    port = portfolio(weights={"AAPL": -0.05})
    assert verdict(g, Intent("AAPL", -0.02), port=port).action is GateAction.PASS
    assert verdict(g, Intent("AAPL", -0.08), port=port).rule == "denylist"


def test_an_exit_still_spends_order_budget():
    # It becomes a real order, so a later entry in the same batch must see it.
    g = gate(max_orders_per_day=1)
    vs = g.evaluate(
        [Intent("AAPL", 0.0), Intent("MSFT", 0.04)],
        portfolio=portfolio(weights={"AAPL": 0.05}),
        now=NOW,
        prices=PRICES,
    )
    assert vs[0].action is GateAction.PASS
    assert vs[1].rule == "max_orders_per_day"


def test_a_sign_flip_is_not_an_exit():
    # +5% to -3% opens a short; treat it as new exposure, not a reduction.
    g = gate(denylist=["AAPL"])
    v = verdict(g, Intent("AAPL", -0.03), port=portfolio(weights={"AAPL": 0.05}))
    assert v.action is GateAction.BLOCK and v.rule == "denylist"


def test_increasing_an_existing_position_is_not_an_exit():
    g = gate(denylist=["AAPL"])
    v = verdict(g, Intent("AAPL", 0.06), port=portfolio(weights={"AAPL": 0.05}))
    assert v.rule == "denylist"


# --- breaker -----------------------------------------------------------------


def test_breaker_trips_at_the_daily_loss_limit_and_spares_exits():
    g = gate(max_daily_loss_pct=0.03)
    g.roll_day(NOW, 100_000.0)
    port = portfolio(equity=96_000.0, weights={"AAPL": 0.05})
    vs = g.evaluate(
        [Intent("MSFT", 0.04), Intent("AAPL", 0.0)], portfolio=port, now=NOW, prices=PRICES
    )
    assert g.tripped and g.state.breaker_reason.startswith("daily_loss")
    assert vs[0].action is GateAction.BLOCK and vs[0].rule == "breaker"
    assert vs[1].action is GateAction.PASS  # the book is never trapped


def test_breaker_does_not_trip_just_inside_the_limit():
    g = gate(max_daily_loss_pct=0.03)
    g.roll_day(NOW, 100_000.0)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=97_100.0))
    assert not g.tripped


def test_breaker_trips_exactly_at_the_limit():
    g = gate(max_daily_loss_pct=0.03)
    g.roll_day(NOW, 100_000.0)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=97_000.0))
    assert g.tripped


def test_a_new_session_clears_an_auto_trip_but_not_a_manual_one():
    g = gate(max_daily_loss_pct=0.03)
    g.roll_day(NOW, 100_000.0)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=90_000.0))
    assert g.tripped
    g.roll_day(DAY2, 90_000.0)
    assert not g.tripped
    g.trip("hand-pulled after a reconcile diff")
    g.roll_day(DAY3, 90_000.0)
    assert g.tripped  # resuming after a manual trip is a human decision
    g.reset()
    assert not g.tripped and g.state.day is None


def test_evaluate_arms_the_breaker_even_if_roll_day_was_never_called():
    g = gate(max_daily_loss_pct=0.03)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=100_000.0))
    assert g.state.day_start_equity == pytest.approx(100_000.0)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=90_000.0))
    assert g.tripped


def test_roll_day_is_idempotent_within_a_session():
    g = gate()
    g.roll_day(NOW, 100_000.0)
    g.note_orders([order("AAPL")])
    g.roll_day(NOW + timedelta(hours=3), 80_000.0)
    assert g.state.orders_today == 1
    assert g.state.day_start_equity == pytest.approx(100_000.0)
    g.roll_day(DAY2, 80_000.0)
    assert g.state.orders_today == 0 and g.state.day_start_equity == pytest.approx(80_000.0)


# --- kill switch -------------------------------------------------------------


def test_kill_switch_env_flag_blocks_everything_including_exits(monkeypatch):
    monkeypatch.setenv("LAB_KILL_SWITCH", "1")
    reset_settings_cache()
    g = gate()
    vs = g.evaluate(
        [Intent("MSFT", 0.04), Intent("AAPL", 0.0)],
        portfolio=portfolio(weights={"AAPL": 0.05}),
        now=NOW,
        prices=PRICES,
    )
    assert all(v.action is GateAction.BLOCK and v.rule == "kill_switch" for v in vs)
    assert vs[1].approved_pct == pytest.approx(0.05)


def test_kill_switch_sentinel_file_blocks(tmp_path):
    (tmp_path / "KILL").write_text("panic", encoding="utf-8")
    v = verdict(gate(), Intent("MSFT", 0.04))
    assert v.rule == "kill_switch" and "KILL" in v.detail


# --- determinism and input validation ----------------------------------------


def test_evaluate_is_deterministic():
    g = gate(max_position_pct=0.05, max_gross_exposure=0.08, denylist=["XOM"])
    port = portfolio(weights={"AAPL": 0.02})
    intents = [Intent("AAPL", 0.30), Intent("XOM", 0.04), Intent("MSFT", 0.10), Intent("NVDA", 0.0)]
    first = [v.to_dict() for v in g.evaluate(intents, portfolio=port, now=NOW, prices=PRICES)]
    second = [v.to_dict() for v in g.evaluate(intents, portfolio=port, now=NOW, prices=PRICES)]
    assert first == second
    assert [v["rule"] for v in first] == ["position_cap", "denylist", "gross_exposure", None]


def test_evaluate_does_not_mutate_the_intents():
    g = gate(max_position_pct=0.01)
    intent = Intent("AAPL", 0.30)
    verdict(g, intent)
    assert intent.target_pct == pytest.approx(0.30)


def test_bad_input_raises_value_error():
    g = gate()
    with pytest.raises(ValueError):
        verdict(g, Intent("AAPL", float("nan")))
    with pytest.raises(ValueError):
        verdict(g, Intent("AAPL", float("inf")))
    with pytest.raises(ValueError):
        g.evaluate([Intent("AAPL", 0.01)], portfolio=portfolio(), now=NOW, prices={"AAPL": -1.0})
    with pytest.raises(ValueError):
        g.evaluate([Intent("AAPL", 0.01)], portfolio=FakePortfolio(0.0), now=NOW, prices=PRICES)
    with pytest.raises(ValueError):
        g.evaluate([Intent("AAPL", 0.01)], portfolio=object(), now=NOW, prices=PRICES)
    with pytest.raises(ValueError):
        RiskGate(limits={"max_positions": 3})
    with pytest.raises(ValueError):
        g.note_orders([Order(id="x", ticker="", side=Side.BUY, qty=1.0)])


def test_state_can_be_handed_in_and_survives_the_gate():
    state = GateState(orders_today=3)
    g = gate(state=state, max_orders_per_day=3)
    assert verdict(g, Intent("AAPL", 0.04)).rule == "max_orders_per_day"
    assert g.state is state


def test_resuming_mid_session_keeps_the_budget_and_the_loss_baseline():
    # Crash-and-resume is a designed-for path: a restored state must not be
    # handed a fresh order budget or a re-based loss breaker.
    state = GateState(day=None, orders_today=7, day_start_equity=100_000.0)
    g = gate(state=state, max_daily_loss_pct=0.03)
    g.roll_day(NOW, 90_000.0)
    assert state.orders_today == 7
    assert state.day_start_equity == pytest.approx(100_000.0)
    verdict(g, Intent("MSFT", 0.04), port=portfolio(equity=90_000.0))
    assert g.tripped


def test_note_orders_without_a_timestamp_uses_the_current_session():
    g = gate(cooldown_days=2)
    g.roll_day(NOW, 100_000.0)
    g.note_orders([order("AAPL", side=Side.SELL, at=None)])
    assert g.state.exited["AAPL"] == NOW.date()


# --- coverage of the rule table ----------------------------------------------


def test_every_rule_name_can_fire():
    fired: set[str] = set()

    def fire(g: RiskGate, intent: Intent, port: FakePortfolio | None = None, now: datetime = NOW):
        v = verdict(g, intent, port=port, now=now)
        if v.rule:
            fired.add(v.rule)

    held = portfolio(weights={"AAPL": 0.04, "MSFT": 0.04})
    tech = {"AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech"}

    fire(gate(denylist=["NVDA"]), Intent("NVDA", 0.04))
    fire(gate(allowlist=["AAPL"]), Intent("NVDA", 0.04))
    fire(gate(max_position_pct=0.01), Intent("NVDA", 0.04))
    fire(gate(max_position_notional=500.0), Intent("NVDA", 0.04))
    fire(gate(max_positions=2), Intent("NVDA", 0.04), port=held)
    fire(gate(max_sector_positions=2, sectors=tech), Intent("NVDA", 0.04), port=held)
    fire(gate(max_sector_pct=0.10, max_sector_positions=3, sectors=tech), Intent("NVDA", 0.05), port=held)
    fire(gate(max_gross_exposure=0.10), Intent("NVDA", 0.05), port=held)
    fire(gate(min_order_notional=1_000.0), Intent("NVDA", 0.001))

    g_orders = gate(max_orders_per_day=0)
    fire(g_orders, Intent("NVDA", 0.04))

    g_cool = gate(cooldown_days=5)
    g_cool.note_exit("NVDA", NOW)
    fire(g_cool, Intent("NVDA", 0.04), now=DAY2)

    g_breaker = gate()
    g_breaker.trip("test")
    fire(g_breaker, Intent("NVDA", 0.04))

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LAB_KILL_SWITCH", "1")
        reset_settings_cache()
        fire(gate(), Intent("NVDA", 0.04))
    reset_settings_cache()

    assert fired == set(RULES)
