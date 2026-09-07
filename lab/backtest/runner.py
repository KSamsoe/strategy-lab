"""The event-driven backtest loop.

Event-driven rather than vectorized, on purpose: path-dependent logic, position
management, and agent strategies cannot be expressed honestly in vectorized
form. The coarse-screening speed path lives in ``sweep.py``; anything that
survives it gets validated here before it is believed.

The loop below is also the *same* loop the live runner walks. Only the clock and
the broker differ, which is the property the whole platform is built to keep.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from lab.backtest import metrics as M
from lab.backtest.fills import FillModel
from lab.engine.broker_sim import SimBroker
from lab.engine.clock import SimClock
from lab.engine.context import DataView, EngineContext
from lab.engine.events import (
    Decision,
    EventKind,
    Fill,
    GateAction,
    GateVerdict,
    Intent,
    Order,
    OrderType,
    Side,
    Trade,
    event,
    new_id,
)
from lab.engine.loader import LoadedStrategy, load_strategy
from lab.engine.portfolio import Portfolio, ReadOnlyPortfolio
from lab.risk.gate import RiskGate, RiskLimits
from lab.timeutil import session_date, to_utc, utcnow

MIN_SHARES = 1e-9


@dataclass
class BacktestConfig:
    strategy: str
    tickers: list[str] = field(default_factory=list)
    timeframe: str = "1d"
    start: datetime | None = None
    end: datetime | None = None
    cash: float = 100_000.0
    params: dict[str, Any] = field(default_factory=dict)
    fills: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    #: Which adapter's bars to trade. Required when the store holds bars for
    #: these tickers from more than one source; mixing them is meaningless.
    source: str | None = None
    #: Intraday only. Drops pre-market and after-hours bars, which are thin
    #: enough to be mispriced by any fill model and numerous enough to skew
    #: annualization. The live runner fires only in regular hours, so keeping
    #: them would also make backtest and live disagree about what a bar is.
    regular_hours: bool = True
    warmup: int = 0
    sectors: dict[str, str] = field(default_factory=dict)
    benchmark: str | None = None
    origin: str = "human"
    notes: str = ""
    seed: int | None = None
    strict: bool = True
    allow_fractional: bool = False
    windows: list[dict[str, Any]] = field(default_factory=list)
    sweep_id: str | None = None
    parent_run_id: str | None = None
    contaminated: bool = False

    def __post_init__(self) -> None:
        self.tickers = [t.upper() for t in self.tickers]
        self.sectors = {k.upper(): v for k, v in (self.sectors or {}).items()}
        if self.start is not None:
            self.start = to_utc(self.start)
        if self.end is not None:
            self.end = to_utc(self.end)

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: Mapping[str, Any] | None = None
    ) -> "BacktestConfig":
        import yaml

        raw: dict[str, Any] = {}
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"no config at {p}")
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BacktestConfig":
        data = dict(raw)
        # `universe` is an accepted alias for `tickers`; consume it here so the
        # unknown-key check below does not reject the very spelling we support.
        tickers = data.pop("universe", None) or data.get("tickers") or []
        if isinstance(tickers, str):
            tickers = load_universe(tickers)
        data["tickers"] = list(tickers)
        for key in ("start", "end"):
            if data.get(key):
                data[key] = to_utc(pd.Timestamp(data[key]))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        # Unknown keys are almost always a typo in a hand-written YAML; failing
        # loudly beats silently backtesting something other than what was meant.
        if unknown:
            raise ValueError(
                f"unknown config keys: {', '.join(sorted(unknown))}; "
                f"known keys are {', '.join(sorted(known))}"
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = self.start.isoformat() if self.start else None
        d["end"] = self.end.isoformat() if self.end else None
        return d


def load_universe(path: str | Path) -> list[str]:
    """One ticker per line; ``#`` comments and blanks ignored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no universe file at {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.upper())
    return out


@dataclass
class BacktestResult:
    run_id: str
    metrics: dict[str, Any]
    equity: pd.Series
    trades: list[Trade]
    decisions: list[Decision]
    orders: list[Order]
    fills: list[Fill]
    config: dict[str, Any]
    data_version: str
    artifact_dir: Path
    #: Gross exposure per bar, aligned to ``equity``. Carried rather than left in
    #: the artifact because every windowed metric needs it: without the series,
    #: ``compute_metrics`` reports exposure 0.0, and a cash-heavy strategy then
    #: looks identical to a fully-invested one to anything reading a sub-window.
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    #: The exact strategy text this run executed. Archived because ``strategy_hash``
    #: alone proves a file *changed* without being able to say what it was: an
    #: agent rewrites its workspace file in place across turns, so the path a run
    #: records can hold entirely different code an hour later, and the run that
    #: produced the result you want to ship becomes unreproducible.
    strategy_source: str = ""
    warnings: list[str] = field(default_factory=list)
    attempt: int = 0
    duration_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """The ``--json`` payload. Deliberately small: an agent reads this
        thousands of times, so it carries the numbers and pointers, not the
        curves."""
        return {
            "run_id": self.run_id,
            "strategy": self.config.get("strategy"),
            "metrics": self.metrics,
            "config_keys": sorted(self.config),
            "params": self.config.get("params", {}),
            "data_version": self.data_version,
            "attempt": self.attempt,
            "n_decisions": len(self.decisions),
            "n_orders": len(self.orders),
            "n_fills": len(self.fills),
            "n_trades": len(self.trades),
            "warnings": self.warnings,
            "artifact_dir": str(self.artifact_dir),
            "duration_s": round(self.duration_s, 3),
        }


def _build_orders(
    verdicts: Sequence[GateVerdict],
    intents: Mapping[str, Intent],
    portfolio: Portfolio,
    prices: Mapping[str, float],
    *,
    strategy: str,
    session,
    decision_id: str,
    allow_fractional: bool,
    now: datetime,
) -> list[Order]:
    """Turn approved target weights into share deltas.

    This is the only place a percentage becomes an order, which is what keeps
    sizing arithmetic out of strategy code and out of the gate.
    """
    orders: list[Order] = []
    for v in verdicts:
        if v.action is GateAction.BLOCK:
            continue
        price = prices.get(v.ticker)
        if price is None or price <= 0:
            continue
        delta = portfolio.target_to_delta_shares(
            v.ticker, v.approved_pct, price, allow_fractional=allow_fractional
        )
        if abs(delta) < MIN_SHARES:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        intent = intents.get(v.ticker)
        orders.append(
            Order(
                id=new_id("o_"),
                ticker=v.ticker,
                side=side,
                qty=abs(delta),
                order_type=OrderType.LIMIT if (intent and intent.limit_price) else OrderType.MARKET,
                limit_price=intent.limit_price if intent else None,
                created_at=now,
                strategy=strategy,
                tag=(intent.tag if intent else "") or ("exit" if v.approved_pct == 0 else "entry"),
                idem_key=Order.make_idem_key(strategy, session, v.ticker, side),
                decision_id=decision_id,
                reason=(intent.reason if intent else "") or v.detail,
            )
        )
    return orders


def run_backtest(
    config: BacktestConfig,
    *,
    register: bool = True,
    journal: bool = True,
    progress: bool = False,
    data: DataView | None = None,
    strategy: LoadedStrategy | None = None,
) -> BacktestResult:
    started = _time.perf_counter()
    warnings: list[str] = []

    loaded = strategy or load_strategy(config.strategy, config.params)
    params = loaded.params

    if data is None:
        if not config.tickers:
            raise ValueError("backtest config has no tickers")
        data = DataView.from_store(
            config.tickers,
            timeframe=config.timeframe,
            start=config.start,
            end=config.end,
            sources=config.sources,
            bar_source=config.source,
            regular_hours=config.regular_hours,
        )

    timestamps = data.timestamps()
    if not timestamps:
        raise ValueError(
            "no bars for this universe/timeframe/date range -- run `lab pull` first"
        )

    fill_model = FillModel.from_mapping(config.fills)
    if config.allow_fractional:
        fill_model.allow_fractional = True
    if fill_model.optimistic:
        warnings.append(
            "same-bar-close fills are enabled: decisions trade at a price they "
            "helped set, which flatters results. Treat metrics as optimistic."
        )

    limits = RiskLimits.from_mapping(config.limits) if config.limits else RiskLimits()
    gate = RiskGate(limits, sectors=config.sectors)

    portfolio = Portfolio(config.cash)
    view = ReadOnlyPortfolio(portfolio)
    broker = SimBroker(portfolio, fill_model)

    data_version = _data_version_for(config)
    ctx = EngineContext(
        run_id="pending",
        strategy=loaded.name,
        params=params,
        universe=config.tickers or data.tickers(),
        portfolio=view,
        data=data,
        timeframe=config.timeframe,
        strict=config.strict,
        indicator_data_version=data_version,
    )

    registry, run = None, None
    run_id = new_id("r_")
    attempt = 0
    if register:
        registry, run = _register_run(config, loaded, data_version, timestamps)
        run_id = run.run_id
        attempt = run.attempt
    ctx.run_id = run_id
    artifact_dir = _artifact_dir(run_id)

    decisions: list[Decision] = []
    equity_rows: list[tuple[datetime, float, float]] = []
    all_fills: list[Fill] = []
    journal_events = []

    on_start = loaded.hook("on_start")
    on_bar = loaded.hook("on_bar")
    on_fill = loaded.hook("on_fill")
    on_stop = loaded.hook("on_stop")

    clock = SimClock(timestamps)
    ctx.set_now(timestamps[0])
    if on_start:
        on_start(ctx)
        ctx.reset_bar()

    if journal:
        journal_events.append(
            event(
                EventKind.RUN_START, "backtest", at=timestamps[0], run_id=run_id,
                strategy=loaded.name,
                message=f"{loaded.name} · {len(config.tickers)} tickers · {len(timestamps)} bars",
                payload={"config": config.to_dict(), "data_version": data_version},
            )
        )

    error: str | None = None
    try:
        for i, ts in enumerate(clock):
            bars_now = data.bars_at(ts)
            closes = {t: b["close"] for t, b in bars_now.items() if b.get("close") == b.get("close")}

            # Orders queued on earlier bars fill against this bar, unless the
            # model is same-bar-close, in which case they fill after the decision.
            if not fill_model.same_bar:
                fills = broker.process(ts, bars_now)
                all_fills.extend(fills)
                _dispatch_fills(fills, ctx, on_fill, journal, journal_events, run_id, loaded.name)

            portfolio.mark(closes)
            portfolio.tick_bar()
            gate.roll_day(ts, portfolio.equity)

            decision: Decision | None = None
            if i >= config.warmup and on_bar is not None:
                t0 = _time.perf_counter()
                ctx.reset_bar()
                ctx.set_now(ts)
                on_bar(ctx)
                intents = ctx.drain_intents()

                verdicts = gate.evaluate(
                    intents, portfolio=view, now=ts, prices=closes
                )
                by_ticker = {i_.ticker: i_ for i_ in intents}
                orders = _build_orders(
                    verdicts, by_ticker, portfolio, closes,
                    strategy=loaded.name, session=session_date(ts),
                    decision_id="pending", allow_fractional=fill_model.allow_fractional,
                    now=ts,
                )
                decision = Decision(
                    id=new_id("d_"),
                    run_id=run_id,
                    strategy=loaded.name,
                    at=ts,
                    inputs=ctx.input_tape(),
                    intents=intents,
                    verdicts=verdicts,
                    portfolio=view.snapshot(ts),
                    logs=ctx.logs(),
                    duration_ms=round((_time.perf_counter() - t0) * 1000, 3),
                )
                for o in orders:
                    o.decision_id = decision.id
                    broker.submit(o)
                gate.note_orders(orders)
                decision.order_ids = [o.id for o in orders]
                decisions.append(decision)

                if journal:
                    for v in verdicts:
                        if v.action is not GateAction.PASS:
                            journal_events.append(
                                event(
                                    EventKind.GATE_BLOCK, "gate", at=ts, run_id=run_id,
                                    strategy=loaded.name, ticker=v.ticker,
                                    message=f"{v.action.value} {v.ticker} — {v.rule}",
                                    payload=v.to_dict(),
                                )
                            )

            if fill_model.same_bar:
                fills = broker.process(ts, bars_now)
                all_fills.extend(fills)
                _dispatch_fills(fills, ctx, on_fill, journal, journal_events, run_id, loaded.name)
                portfolio.mark(closes)

            # Warm-up bars are excluded from the curve, not merely from the
            # decisions. They exist so indicators have history; the strategy is
            # structurally unable to trade through them, so counting them makes
            # every comparison wrong in the same direction. The benchmark is
            # reindexed onto this curve, so including them hands it a free head
            # start -- on one real run, 211 dead bars gave SPY +61% before the
            # strategy was allowed its first order. They also stretch the
            # denominator of CAGR and pad the return series with zeros.
            if i >= config.warmup:
                equity_rows.append((ts, portfolio.equity, portfolio.gross_exposure))

            if progress and i % 250 == 0:
                _progress(i, len(timestamps), portfolio.equity)

        if on_stop:
            ctx.set_now(timestamps[-1])
            on_stop(ctx)
    except Exception as exc:  # a failed run is still a run; record it
        error = f"{type(exc).__name__}: {exc}"
        if register and registry is not None:
            registry.finish(run_id, metrics={}, status="error", error=error)
        raise
    finally:
        if progress:
            print()

    if not equity_rows:
        raise ValueError(
            f"warmup ({config.warmup}) consumed every one of the {len(timestamps)} bars, "
            f"so the strategy was never allowed to trade. Shorten the warm-up or widen "
            f"the date range."
        )
    equity = pd.Series(
        [e for _, e, _ in equity_rows],
        index=pd.DatetimeIndex([t for t, _, _ in equity_rows], name="event_time"),
        name="equity",
    )
    exposure = pd.Series([x for _, _, x in equity_rows], index=equity.index)

    benchmark = _benchmark_series(config) if config.benchmark else None
    computed = M.compute_metrics(
        equity,
        portfolio.trades(),
        timeframe=config.timeframe,
        benchmark=benchmark,
        exposure=exposure,
        turnover_notional=portfolio.turnover_notional,
    )
    # Does the trade ledger explain the equity curve? A non-zero residual means
    # money moved that no trade accounts for, and every trade-level metric below
    # is then describing a different run than `total_return` is. Surfaced rather
    # than asserted: a run that already happened is still worth reading, but it
    # must say so.
    reconciliation = portfolio.reconcile()
    if abs(reconciliation["residual"]) > max(1e-6, abs(config.cash) * 1e-9):
        msg = (
            f"trade ledger does not explain the equity curve: "
            f"{reconciliation['residual']:,.2f} of P&L has no trade behind it. "
            f"Trade-level metrics (hit rate, profit factor, avg win/loss) describe "
            f"only the {reconciliation['ledger_pnl']:,.2f} that does."
        )
        warnings.append(msg)
        import logging

        logging.getLogger(__name__).warning(msg)

    computed |= {
        "commission": round(portfolio.total_commission, 6),
        "open_positions": len(portfolio.positions),
        "warmup_bars": int(config.warmup),
        "tradeable_bars": len(equity_rows),
        "realized_pnl": round(portfolio.realized_pnl(), 6),
        "unrealized_pnl": round(portfolio.unrealized_pnl(), 6),
        "ledger_residual": reconciliation["residual"],
        "fill_mode": fill_model.mode,
        "optimistic_fills": fill_model.optimistic,
        "contaminated": bool(config.contaminated),
        "attempt": attempt,
        "warnings": warnings,
    }

    result = BacktestResult(
        run_id=run_id,
        metrics=computed,
        equity=equity,
        trades=portfolio.trades(),
        decisions=decisions,
        orders=broker.orders,
        fills=all_fills,
        config=config.to_dict() | {"resolved_params": params, "strategy_hash": loaded.source_hash},
        data_version=data_version,
        artifact_dir=artifact_dir,
        exposure=exposure,
        strategy_source=loaded.source,
        warnings=warnings,
        attempt=attempt,
        duration_s=_time.perf_counter() - started,
    )

    _write_artifacts(result, exposure)

    if journal:
        journal_events.append(
            event(
                EventKind.RUN_END, "backtest", at=timestamps[-1], run_id=run_id,
                strategy=loaded.name,
                message=f"{len(result.trades)} trades · {computed.get('total_return', 0):.2%}",
                payload={"metrics": computed},
            )
        )
        _persist_journal(decisions, journal_events)

    if register and registry is not None:
        registry.finish(run_id, metrics=computed, status="ok")

    return result


def _dispatch_fills(fills, ctx, on_fill, journal, journal_events, run_id, strategy) -> None:
    for f in fills:
        if on_fill is not None:
            on_fill(ctx, f)
        if journal:
            journal_events.append(
                event(
                    EventKind.FILL, "broker", at=f.at, run_id=run_id, strategy=strategy,
                    ticker=f.ticker,
                    message=f"{f.ticker} {'+' if f.side is Side.BUY else '-'}{f.qty:g} @ {f.price:.2f}",
                    payload=f.to_dict(),
                )
            )


def _progress(i: int, n: int, equity: float) -> None:
    import sys

    pct = (i + 1) / n
    bar = "#" * int(pct * 28)
    sys.stderr.write(f"\r  [{bar:<28}] {pct:5.1%}  equity {equity:>12,.0f}")
    sys.stderr.flush()


def _data_version_for(config: BacktestConfig) -> str:
    try:
        from lab.store import parquet_io

        return parquet_io.data_version(tickers=config.tickers)
    except Exception:
        # A run over an injected DataView (tests, synthetic demo) has no store
        # partitions behind it; that is legitimate, not an error.
        return "unversioned"


def _benchmark_series(config: BacktestConfig) -> pd.Series | None:
    try:
        from lab.store import parquet_io

        df = parquet_io.read_bars(
            [config.benchmark], timeframe=config.timeframe,
            start=config.start, end=config.end, source=config.source,
        )
        if len(df) == 0:
            return None
        return df.set_index("event_time")["close"].rename(config.benchmark)
    except Exception:
        return None


def artifact_dir(run_id: str) -> Path:
    from lab.config import get_settings

    d = get_settings().paths.runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


#: Where a run's artifacts live. Public because the MCP toolset, the promoter
#: and the review digest all need it, and three modules reaching for one
#: underscore name is a naming problem rather than an encapsulation one.
_artifact_dir = artifact_dir


def _register_run(config: BacktestConfig, loaded: LoadedStrategy, data_version: str, timestamps):
    from lab.registry.runs import RunRegistry, config_hash, git_commit

    registry = RunRegistry()
    cfg = config.to_dict()
    run = registry.create(
        strategy=loaded.name,
        kind="backtest",
        git_commit=git_commit(),
        config_hash=config_hash(cfg),
        data_version=data_version,
        params=loaded.params,
        config=cfg,
        start=timestamps[0],
        end=timestamps[-1],
        origin=config.origin,
        notes=config.notes,
        sweep_id=config.sweep_id,
        parent_run_id=config.parent_run_id,
    )
    return registry, run


def _persist_journal(decisions: Iterable[Decision], events: Iterable[Any]) -> None:
    from lab.registry.journal import DecisionJournal, EventJournal

    dj = DecisionJournal()
    dj.bulk_append(decisions)
    ej = EventJournal()
    for ev in events:
        ej.append(ev)


def _write_artifacts(result: BacktestResult, exposure: pd.Series) -> None:
    d = result.artifact_dir
    d.mkdir(parents=True, exist_ok=True)

    (d / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
    )
    (d / "config.json").write_text(
        json.dumps(result.config, indent=2, default=str), encoding="utf-8"
    )

    # The strategy itself, verbatim. `lab strategy promote` copies this rather
    # than whatever now sits at the recorded path, which is the only way to
    # promote the code that actually produced a result.
    if result.strategy_source:
        (d / "strategy.py").write_text(result.strategy_source, encoding="utf-8")

    eq = M.equity_frame(result.equity)
    eq["exposure"] = exposure.reindex(eq.index)
    eq.reset_index().rename(columns={"index": "event_time"}).to_parquet(
        d / "equity.parquet", index=False
    )

    trades = pd.DataFrame([t.to_dict() for t in result.trades])
    trades.to_csv(d / "trades.csv", index=False)

    with (d / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for dec in result.decisions:
            fh.write(json.dumps(dec.to_dict(), default=str) + "\n")

    (d / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "data_version": result.data_version,
                "warnings": result.warnings,
                "written_at": utcnow().isoformat(),
                "survivorship_note": (
                    "Universe is today's ticker list, not a point-in-time universe. "
                    "Survivorship bias is mitigated, not solved."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
