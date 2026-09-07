"""What a run actually *did*, compressed to fit in a prompt.

A research agent that only ever sees `sharpe: 1.006` is reasoning about a
strategy it has never watched behave. It cannot tell a broad edge from one lucky
ticker, a steady curve from one that made everything in six weeks, or a
well-behaved book from one the risk gate spent the whole run clipping.

The constraint is context. A run has thousands of decisions and hundreds of
trades, and dumping them would crowd out the reasoning they are supposed to
support. So this returns *derived facts* rather than rows: attribution instead of
a trade list, drawdown episodes instead of an equity curve, gate rule counts
instead of verdicts. Everything is bounded, and anything dropped is reported --
a digest that silently truncates is a digest that lies by omission.

The single most valuable number here is ``concentration.top_ticker_share``. One
name carrying a whole backtest is the most common way a result turns out not to
be one, and it is invisible in every aggregate metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

#: Rows kept per section. These were once tuned to land the digest around 2k
#: tokens, which sounded prudent and was not: four sample trades out of 1,769 is
#: an anecdote, and the agent drew wrong conclusions from the gaps.
#:
#: The aggregate sections are now effectively uncapped, because they are cheap --
#: every ticker in the universe instead of eight costs ~390 tokens, and the month
#: table and drawdown list are smaller still. ``SAMPLE_TRADES`` is the one that
#: is not: it yields best *and* worst, so each unit is two trade rows at ~119
#: tokens apiece, and it alone decides whether a digest is 2k tokens or 4.5k.
#:
#: Eight is where the measured session cost lands next to a default budget: a
#: 60-call session on Opus runs ~$4.9 of input at these settings against ~$1.9
#: before. Raising it further is a budget decision, not a context-window one --
#: the whole prompt is still under 10% of a 200k window.
TOP_TICKERS = 25
WORST_DRAWDOWNS = 12
SAMPLE_TRADES = 8
MAX_MONTHS = 240
MAX_GATE_RULES = 40


def _num(v: Any, digits: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, digits)


def _artifact_dir(run_id: str) -> Path:
    from lab.config import get_settings

    return get_settings().paths.runs / run_id


def _by_ticker(trades: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """P&L attribution, and how concentrated it is.

    A strategy whose entire result is one ticker has not been shown to work; it
    has been shown to have held something that went up. That is worth knowing
    before anyone tunes its parameters for another hour.
    """
    if trades.empty or "ticker" not in trades:
        return [], {}

    grouped = trades.groupby("ticker")["pnl"]
    agg = pd.DataFrame(
        {
            "trades": grouped.count(),
            "pnl": grouped.sum(),
            "win_rate": trades.groupby("ticker")["pnl"].apply(lambda s: (s > 0).mean()),
        }
    ).sort_values("pnl", ascending=False)

    total = float(agg["pnl"].sum())
    winners = agg[agg["pnl"] > 0]["pnl"]
    gross_win = float(winners.sum()) if len(winners) else 0.0

    rows = [
        {
            "ticker": str(t),
            "trades": int(r["trades"]),
            "pnl": _num(r["pnl"], 2),
            "share_of_gross_profit": _num(r["pnl"] / gross_win, 3) if gross_win > 0 and r["pnl"] > 0 else None,
            "win_rate": _num(r["win_rate"], 3),
        }
        for t, r in agg.head(TOP_TICKERS).iterrows()
    ]
    # The worst few matter as much as the best: a strategy can be a good idea
    # bleeding out through two names.
    tail = [
        {
            "ticker": str(t),
            "trades": int(r["trades"]),
            "pnl": _num(r["pnl"], 2),
            "win_rate": _num(r["win_rate"], 3),
        }
        for t, r in agg.tail(3).iterrows()
        if r["pnl"] < 0
    ]

    top = float(agg["pnl"].iloc[0]) if len(agg) else 0.0
    concentration = {
        "tickers_traded": int(len(agg)),
        "tickers_profitable": int((agg["pnl"] > 0).sum()),
        "top_ticker": str(agg.index[0]) if len(agg) else None,
        "top_ticker_share": _num(top / gross_win, 3) if gross_win > 0 else None,
        "top3_share": _num(float(agg["pnl"].head(3).sum()) / gross_win, 3) if gross_win > 0 else None,
        "total_pnl": _num(total, 2),
    }
    return rows + ([{"__worst__": tail}] if tail else []), concentration


def _drawdowns(equity: pd.Series, limit: int = WORST_DRAWDOWNS) -> list[dict[str, Any]]:
    """The worst peak-to-recovery episodes, with dates.

    "-29% max drawdown" says nothing about whether that was one bad week or two
    years underwater, and those are different strategies to live with.
    """
    if equity is None or len(equity) < 3:
        return []
    peak = equity.cummax()
    under = equity < peak
    episodes: list[dict[str, Any]] = []
    start: Any = None
    peak_ts: Any = equity.index[0]
    for ts, is_under in under.items():
        if is_under and start is None:
            start = peak_ts
        elif not is_under:
            if start is not None:
                seg = equity.loc[start:ts]
                episodes.append(
                    {
                        "from": str(pd.Timestamp(start).date()),
                        "to": str(pd.Timestamp(ts).date()),
                        "depth": _num(float(seg.min() / seg.iloc[0] - 1.0), 4),
                        "days": int((pd.Timestamp(ts) - pd.Timestamp(start)).days),
                    }
                )
                start = None
            peak_ts = ts
    if start is not None:
        seg = equity.loc[start:]
        episodes.append(
            {
                "from": str(pd.Timestamp(start).date()),
                "to": "still underwater",
                "depth": _num(float(seg.min() / seg.iloc[0] - 1.0), 4),
                "days": int((equity.index[-1] - pd.Timestamp(start)).days),
            }
        )
    episodes.sort(key=lambda d: d["depth"] or 0.0)
    return episodes[:limit]


def _monthly(equity: pd.Series) -> dict[str, Any]:
    if equity is None or len(equity) < 2:
        return {}
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return {}
    kept = monthly.tail(MAX_MONTHS)
    return {
        "returns": {str(k.date())[:7]: _num(v, 4) for k, v in kept.items()},
        "positive_months": int((monthly > 0).sum()),
        "total_months": int(len(monthly)),
        "best": _num(float(monthly.max()), 4),
        "worst": _num(float(monthly.min()), 4),
        "truncated_to_last": MAX_MONTHS if len(monthly) > MAX_MONTHS else None,
    }


def gate_activity(directory: Path) -> dict[str, Any]:
    """How often the gate intervened, and which rule.

    A strategy whose intents are mostly clipped is not the strategy that was
    written -- it is that strategy filtered through a limit, and tuning its
    parameters is tuning the wrong thing.
    """
    path = directory / "decisions.jsonl"
    if not path.exists():
        return {}
    rules: dict[str, int] = {}
    actions = {"pass": 0, "clipped": 0, "blocked": 0}
    decisions = intents = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                decisions += 1
                intents += len(d.get("intents") or [])
                for v in d.get("verdicts") or []:
                    act = str(v.get("action") or "pass")
                    actions[act] = actions.get(act, 0) + 1
                    if act != "pass" and v.get("rule"):
                        rules[str(v["rule"])] = rules.get(str(v["rule"]), 0) + 1
    except (OSError, ValueError):
        return {}

    total = sum(actions.values()) or 1
    ranked = sorted(rules.items(), key=lambda kv: kv[1], reverse=True)[:MAX_GATE_RULES]
    return {
        "decisions": decisions,
        "intents": intents,
        "verdicts": actions,
        "share_clipped_or_blocked": _num((actions.get("clipped", 0) + actions.get("blocked", 0)) / total, 3),
        "top_rules": [{"rule": r, "count": c} for r, c in ranked],
    }


def review_run(run_id: str) -> dict[str, Any]:
    """A bounded, diagnostic digest of one finished run."""
    from lab.registry.runs import RunRegistry

    record = RunRegistry().get(run_id)
    directory = _artifact_dir(run_id)
    if record is None and not directory.is_dir():
        return {"error": f"no such run: {run_id}"}

    metrics: Mapping[str, Any] = (record.metrics if record else {}) or {}
    if not metrics and (directory / "metrics.json").exists():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))

    trades = pd.DataFrame()
    tpath = directory / "trades.csv"
    if tpath.exists():
        try:
            trades = pd.read_csv(tpath)
        except (OSError, ValueError):
            trades = pd.DataFrame()

    equity = pd.Series(dtype="float64")
    epath = directory / "equity.parquet"
    if epath.exists():
        try:
            frame = pd.read_parquet(epath)
            col = "event_time" if "event_time" in frame.columns else frame.columns[0]
            equity = frame.set_index(col)["equity"].astype("float64")
            equity.index = pd.DatetimeIndex(equity.index)
        except (OSError, ValueError, KeyError):
            equity = pd.Series(dtype="float64")

    per_ticker, concentration = _by_ticker(trades)

    trade_stats: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    if not trades.empty and "pnl" in trades:
        pnl = trades["pnl"].astype("float64")
        held = trades["bars_held"].astype("float64") if "bars_held" in trades else pd.Series(dtype="float64")
        trade_stats = {
            "count": int(len(pnl)),
            "win_rate": _num((pnl > 0).mean(), 3),
            "avg_win": _num(pnl[pnl > 0].mean(), 2),
            "avg_loss": _num(pnl[pnl <= 0].mean(), 2),
            "best": _num(pnl.max(), 2),
            "worst": _num(pnl.min(), 2),
            "median_bars_held": _num(held.median(), 1) if len(held) else None,
            "max_bars_held": _num(held.max(), 0) if len(held) else None,
        }
        cols = [c for c in ("ticker", "entry_time", "exit_time", "pnl", "pnl_pct", "bars_held", "exit_reason") if c in trades]
        ordered = trades.sort_values("pnl", ascending=False)
        samples = {
            "best": ordered.head(SAMPLE_TRADES)[cols].to_dict(orient="records"),
            "worst": ordered.tail(SAMPLE_TRADES)[cols].to_dict(orient="records"),
        }

    digest: dict[str, Any] = {
        "run_id": run_id,
        "strategy": (record.strategy if record else metrics.get("strategy")),
        "params": dict(record.params) if record else {},
        "headline": {
            k: metrics.get(k)
            for k in (
                "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                "max_drawdown_duration_days", "trades", "hit_rate", "profit_factor",
                "exposure", "turnover", "commission", "open_positions",
                "benchmark_total_return", "excess_return", "alpha", "beta",
            )
            if metrics.get(k) is not None
        },
        "concentration": concentration,
        "by_ticker": per_ticker,
        "trade_stats": trade_stats,
        "sample_trades": samples,
        "worst_drawdowns": _drawdowns(equity),
        "monthly": _monthly(equity),
        "gate": gate_activity(directory),
        "note": (
            "Derived facts, not raw rows. Kept in context for the rest of the "
            "session, so you can compare this run against another you reviewed "
            "without fetching either again."
        ),
    }
    return digest


#: Kept so existing callers and their tests are untouched.
_gate_activity = gate_activity
