"""Controlled comparisons: isolate what is actually doing the work.

Each of these was hand-rolled at least once during a real investigation, and one
of them settled the most important question this lab has asked. ``blend_tilt``
beat SPY by 157 points on the universe it was found on and lost to it by 5 on a
broader one, same dates, same parameters, same data source -- so its edge was the
ticker list rather than the method. That took three scripts to establish and is
one call here.

A backtest answers "how did this do". These answer "why", which is the question
that decides whether a result survives contact with a different market.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from lab.mcp._common import evaluate, jsonable, load_config, resolve_strategy, split_point


def _row(ev: dict[str, Any], label: str) -> dict[str, Any]:
    """One comparable line from a full evaluation."""
    full, oos = ev["windows"]["full"], ev["windows"]["oos"]
    return {
        "case": label,
        "run_id": ev["run_id"],
        "score": ev.get("score"),
        "full_return": full.get("total_return"),
        "full_sharpe": full.get("sharpe"),
        "max_drawdown": full.get("max_drawdown"),
        "oos_sharpe": oos.get("sharpe"),
        "exposure": full.get("exposure"),
        "trades": full.get("trades"),
        "vs_market_full_return": ev.get("vs_market", {}).get("full_return"),
        "gate_clipped": (ev.get("gate") or {}).get("share_clipped_or_blocked"),
        "warnings": ev.get("warnings"),
    }


def _run_case(cfg, strategy, params, oos, stamps, metric, periods, label):
    from lab.backtest.runner import run_backtest

    case_cfg = replace(cfg, strategy=strategy, params=dict(params or {}))
    result = run_backtest(case_cfg, register=False, journal=False)
    ev = evaluate(result, case_cfg, oos_start=oos, stamps=stamps,
                  periods=periods, metric=metric)
    return _row(ev, label), ev


def ablate(
    config: str,
    cases: dict[str, dict[str, Any]],
    strategy: str | None = None,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
    periods: int = 4,
) -> dict[str, Any]:
    """Run named parameter variants together and compare them on one table.

    `cases` maps a label to a parameter dict, e.g.
    `{"gate on": {"use_gate": 1}, "gate off": {"use_gate": 0}}`. Change one thing
    per case and the table tells you what that thing was worth; change several
    and it tells you nothing you can attribute.

    Include a case that reproduces the strategy you are comparing against. If it
    does not match its known numbers, the ablation is measuring your
    reimplementation rather than your change, and everything after that is noise.
    """
    cfg = load_config(config)
    path = resolve_strategy(strategy, cfg)
    oos, stamps = split_point(cfg, oos_split)

    rows = []
    for label, params in cases.items():
        row, _ = _run_case(cfg, path, params, oos, stamps, metric, periods, label)
        rows.append(row)

    best = max(rows, key=lambda r: (r["score"] if r["score"] is not None else float("-inf")))
    spread = [r["score"] for r in rows if r["score"] is not None]
    notes = []
    if len(spread) > 1:
        notes.append(
            f"score spread across cases is {max(spread) - min(spread):.4f}; compare "
            "that against `score_one_sigma` from a single backtest before believing "
            "any ordering"
        )
    return jsonable({
        "strategy": path.name, "cases": len(rows), "metric": metric,
        "results": rows, "best_case": best["case"], "notes": notes,
    })


def compare_universes(
    config: str,
    universes: dict[str, list[str]],
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
) -> dict[str, Any]:
    """Same strategy, same window, same parameters -- only the ticker list changes.

    The single most informative experiment available here, because a strategy's
    apparent edge is very often its universe. Momentum ranking inside a basket of
    megacap technology names picks the one that went up the most; run the same
    code on a broad universe and the edge can vanish entirely.

    `universes` maps a label to a ticker list. Each is run against the config's
    dates, source, fills and risk limits, so the ticker list is the only variable.
    Note that a benchmark that is not in a universe is still valid -- the market
    reference is computed independently.
    """
    cfg = load_config(config, params=params)
    path = resolve_strategy(strategy, cfg)

    rows = []
    for label, tickers in universes.items():
        names = [t.strip().upper() for t in tickers if str(t).strip()]
        case_cfg = replace(cfg, tickers=names, strategy=path)
        oos, stamps = split_point(case_cfg, oos_split)
        from lab.analysis import market_reference
        market = market_reference(case_cfg, oos, stamps[0] if stamps else None)
        from lab.backtest.runner import run_backtest
        result = run_backtest(case_cfg, register=False, journal=False)
        ev = evaluate(result, case_cfg, oos_start=oos, stamps=stamps,
                      metric=metric, market=market)
        row = _row(ev, label)
        row["universe_size"] = len(names)
        rows.append(row)

    notes = []
    excess = [r.get("vs_market_full_return") for r in rows if r.get("vs_market_full_return") is not None]
    if len(excess) > 1 and max(excess) > 0 > min(excess):
        notes.append(
            "this strategy beats the benchmark on one universe and loses on another "
            "with everything else held constant -- its edge is the ticker list, not "
            "the method, and it will not transfer"
        )
    return jsonable({
        "strategy": path.name, "params": dict(params or {}),
        "results": rows, "notes": notes,
    })


def regimes(
    config: str,
    windows: dict[str, list[str]],
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    metric: str = "oos_sharpe",
) -> dict[str, Any]:
    """Run the same strategy across named date ranges: does it need one era?

    `windows` maps a label to `[start, end]` ISO dates, e.g.
    `{"crisis": ["2005-01-01", "2011-12-31"], "recent": ["2021-01-01", "2026-08-27"]}`.

    Each window gets its own benchmark over its own bars, so the comparison is
    like for like. A strategy that beats its benchmark in every era is a different
    proposition from one that made everything in a single regime, and the pooled
    number cannot tell them apart.
    """
    from datetime import datetime, timezone

    from lab.analysis import market_reference
    from lab.backtest.runner import run_backtest

    base = load_config(config, params=params)
    path = resolve_strategy(strategy, base)
    iso = lambda s: datetime.fromisoformat(str(s)).replace(tzinfo=timezone.utc)

    rows = []
    for label, span in windows.items():
        start, end = span[0], span[1]
        cfg = replace(base, strategy=path, start=iso(start), end=iso(end))
        try:
            oos, stamps = split_point(cfg, "4:1")
        except ValueError as exc:
            rows.append({"case": label, "error": str(exc)})
            continue
        market = market_reference(cfg, oos, stamps[0] if stamps else None)
        result = run_backtest(cfg, register=False, journal=False)
        ev = evaluate(result, cfg, oos_start=oos, stamps=stamps, metric=metric, market=market)
        row = _row(ev, label)
        row |= {
            "window": [start, end], "bars": len(stamps),
            "benchmark_return": (market or {}).get("full", {}).get("total_return"),
            "benchmark_sharpe": (market or {}).get("full", {}).get("sharpe"),
        }
        rows.append(row)

    beat = [r for r in rows if isinstance(r.get("vs_market_full_return"), (int, float))
            and r["vs_market_full_return"] > 0]
    notes = [
        f"beats its benchmark in {len(beat)} of {len([r for r in rows if 'error' not in r])} "
        "windows"
    ]
    return jsonable({"strategy": path.name, "results": rows, "notes": notes})


def cost_sensitivity(
    config: str,
    slippage_bps: list[float] | None = None,
    commission_per_share: list[float] | None = None,
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
) -> dict[str, Any]:
    """How much trading cost does this strategy survive?

    A robustness dimension almost nobody tests and the one that kills intraday
    ideas outright: on 15-minute bars a momentum book paid $3,001 in commission
    across 2,833 trades and turned a market-beating signal into a losing
    strategy. The question is not whether an edge exists at zero cost but at
    what cost it disappears, and the gradient is more informative than any single
    assumption.

    Reports the score at each cost level and the point where the strategy stops
    beating its benchmark.
    """
    from lab.backtest.runner import run_backtest

    cfg = load_config(config, params=params)
    path = resolve_strategy(strategy, cfg)
    oos, stamps = split_point(cfg, oos_split)
    slips = [float(x) for x in (slippage_bps or [0, 2, 5, 10, 20])]
    comms = [float(x) for x in (commission_per_share or [0.0])]

    rows = []
    for slip in slips:
        for comm in comms:
            fills = dict(cfg.fills or {}) | {
                "slippage_bps": slip, "commission_per_share": comm
            }
            case = replace(cfg, strategy=path, fills=fills)
            result = run_backtest(case, register=False, journal=False)
            ev = evaluate(result, case, oos_start=oos, stamps=stamps, metric=metric)
            row = _row(ev, f"{slip:g}bps + ${comm:g}/share")
            row |= {"slippage_bps": slip, "commission_per_share": comm,
                    "commission_paid": (result.metrics or {}).get("commission"),
                    "turnover": (result.metrics or {}).get("turnover")}
            rows.append(row)

    beaten = [r for r in rows if isinstance(r.get("vs_market_full_return"), (int, float))
              and r["vs_market_full_return"] <= 0]
    notes = []
    if beaten:
        first = min(beaten, key=lambda r: r["slippage_bps"])
        notes.append(
            f"stops beating the benchmark at {first['slippage_bps']:g}bps "
            f"(+${first['commission_per_share']:g}/share)"
        )
    else:
        notes.append("beats the benchmark at every cost level tested -- widen the grid")
    if rows and rows[0].get("turnover"):
        notes.append(
            f"turnover is {rows[0]['turnover']:.1f}; every round trip has to pay for "
            "itself, and a high number here is what makes cost assumptions decisive"
        )
    return jsonable({"strategy": path.name, "results": rows, "notes": notes})
