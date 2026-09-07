"""Paper trading against its own backtest: the luck filter that runs on money.

Every validation tool in this server works on history. This one works on what
the strategy actually did once it was handed a broker: it re-runs the backtest
over exactly the window the paper run has traded, from a flat book on the same
first session, and puts the two equity curves side by side. The gap between
them is fills, timing and data freshness -- the things a backtest assumes away
-- and it is the number that decides whether a construction edge measured on
history survives execution.

Promotion to real capital is never automated here. This tool tells you whether
the paper run is behaving like its backtest; it does not tell you to trade it.
"""

from __future__ import annotations

from typing import Any

from lab.mcp._common import jsonable

#: Paper steps below which the comparison is reported but not believed.
MIN_LIVE_STEPS = 20


def live_status(strategy: str | None = None) -> dict[str, Any]:
    """What is running, paused or stopped on a broker right now.

    Reads the journal, not the runner, so it works after the process has died
    -- which is exactly when you want to know what state it was in.
    """
    from lab.registry.db import connect, journal_path

    con = connect(journal_path())
    sql = "SELECT strategy, run_id, kind, status, pid, host, started_at, updated_at, paused, next_fire, notes FROM live_strategies"
    args: list[Any] = []
    if strategy:
        sql += " WHERE strategy = ?"
        args.append(strategy)
    rows = [dict(zip(
        ("strategy", "run_id", "kind", "status", "pid", "host", "started_at", "updated_at",
         "paused", "next_fire", "notes"), r,
    )) for r in con.execute(sql, args).fetchall()]
    from lab.live import killswitch

    return jsonable({"count": len(rows), "strategies": rows, "kill_switch": killswitch.describe()})


def live_vs_backtest(
    strategy: str | None = None,
    run_id: str | None = None,
    since: str | None = None,
    min_steps: int = MIN_LIVE_STEPS,
) -> dict[str, Any]:
    """Compare a strategy's paper (or live) trading against a backtest of the same window.

    Give `strategy` to stitch together every paper run of that strategy -- the
    daily one-shot runner registers one run per session, so a month of paper
    trading is twenty-odd run ids with one decision each -- or `run_id` for a
    single long-running run. `since` (ISO date) drops early test sessions.

    The backtest is rebuilt from the live config the newest run recorded --
    same strategy, params, universe, limits and data source -- and started flat
    on the first paper session, so the only things that can differ are the ones
    a backtest cannot model: real fills, decision timing against bar timing,
    and whatever data the live runner did not have yet.

    Reports both curves, their per-session return correlation, tracking error,
    the return gap, and whether the two books hold the same names today. Below
    `min_steps` sessions the comparison is shown but flagged as too short to
    mean anything.
    """
    import math
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    import numpy as np
    import pandas as pd

    from lab.analysis import tradeable_timestamps, window_metrics
    from lab.backtest.metrics import periods_per_year
    from lab.backtest.runner import BacktestConfig, run_backtest
    from lab.registry.journal import DecisionJournal
    from lab.registry.runs import RunRegistry

    reg = RunRegistry()
    if run_id:
        record = reg.get(run_id)
        if record is None:
            raise ValueError(f"no such run: {run_id}")
        if record.kind not in ("paper", "live"):
            raise ValueError(f"{run_id} is a {record.kind} run; this compares paper/live runs")
        records = [record]
    elif strategy:
        records = sorted(
            (r for r in reg.list(strategy=strategy, limit=5000) if r.kind in ("paper", "live")),
            key=lambda r: r.created_at,
        )
        if not records:
            raise ValueError(f"no paper or live runs of {strategy!r}")
        record = records[-1]  # newest config is the one the book is running
    else:
        raise ValueError("give strategy or run_id")

    dj = DecisionJournal()
    steps: list[dict[str, Any]] = []
    for r in records:
        steps.extend(dj.list(r.run_id, limit=100_000))
    if not steps:
        return {"strategy": record.strategy, "runs": len(records), "steps": 0,
                "verdict": "no decisions journaled yet"}

    def _at(s: dict[str, Any]) -> datetime:
        v = s.get("at")
        return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))

    steps.sort(key=_at)
    if since:
        cutoff = datetime.fromisoformat(str(since))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        steps = [s for s in steps if _at(s) >= cutoff]
    # One point per session: the last decision of the day is the book that
    # carried overnight.
    by_session: dict[Any, dict[str, Any]] = {}
    for s in steps:
        by_session[_at(s).date()] = s
    steps = list(by_session.values())

    live = pd.Series(
        {_at(s): float((s.get("portfolio") or {}).get("equity") or float("nan")) for s in steps}
    ).dropna().sort_index()
    live.index = pd.to_datetime(live.index, utc=True)
    if len(live) < 2:
        return {"strategy": record.strategy, "runs": len(records), "steps": len(live),
                "verdict": "fewer than two sessions with an equity point"}
    live_start, live_end = live.index[0], live.index[-1]
    run_id = record.run_id

    # Rebuild the backtest from the live config: the keys BacktestConfig knows.
    lc = dict(record.config or {})
    keys = {f for f in BacktestConfig.__dataclass_fields__}
    raw = {k: v for k, v in lc.items() if k in keys and k not in ("start", "end", "warmup", "origin")}
    raw["origin"] = "audit"
    lookback = int(lc.get("lookback_days") or 400)
    cfg = BacktestConfig(**raw)
    cfg = replace(
        cfg,
        start=(live_start - timedelta(days=lookback)).to_pydatetime(),
        end=(live_end + timedelta(days=1)).to_pydatetime(),
        warmup=0,
    )
    # Start trading on the paper run's first session, flat, like the paper run did.
    stamps = tradeable_timestamps(cfg, None)
    before = sum(1 for t in stamps if pd.Timestamp(t).tz_convert("UTC").normalize() < live_start.normalize())
    cfg = replace(cfg, warmup=before)
    result = run_backtest(cfg, register=False, journal=False)
    tf = cfg.timeframe
    bt = result.equity.astype("float64")
    bt.index = pd.to_datetime(bt.index, utc=True)
    bt = bt[bt.index.normalize() >= live_start.normalize()]

    # Align by session: the paper run steps at 09:35, the backtest marks at the
    # close. Comparing per-session returns is the like-for-like view.
    l_day = live.groupby(live.index.normalize()).last()
    b_day = bt.groupby(bt.index.normalize()).last()
    both = pd.concat([l_day.rename("live"), b_day.rename("backtest")], axis=1).dropna()
    rets = both.pct_change().dropna()
    ppy = int(periods_per_year(tf))

    out: dict[str, Any] = {
        "strategy": record.strategy,
        "kind": record.kind,
        "runs_stitched": len(records),
        "newest_run_id": run_id,
        "window": [live_start.isoformat(), live_end.isoformat()],
        "steps": int(len(live)),
        "sessions_compared": int(len(both)),
        "live": {
            "total_return": round(float(live.iloc[-1] / live.iloc[0] - 1.0), 6),
            "start_equity": round(float(live.iloc[0]), 2),
            "end_equity": round(float(live.iloc[-1]), 2),
            "orders_submitted": int(sum(len(s.get("order_ids") or []) for s in steps)),
        },
        "backtest": {
            k: v for k, v in window_metrics(result, tf, lo=live_start).items()
            if k in ("total_return", "sharpe", "max_drawdown", "exposure", "trades")
        } | {"orders_submitted": int(len(result.orders))},
        "warnings": [],
    }
    if len(rets) >= 3:
        corr = float(rets["live"].corr(rets["backtest"]))
        diff = rets["live"] - rets["backtest"]
        te = float(diff.std() * math.sqrt(ppy)) if len(diff) > 1 else float("nan")
        gap = (both["live"] / both["live"].iloc[0]) - (both["backtest"] / both["backtest"].iloc[0])
        out["agreement"] = {
            "session_return_correlation": round(corr, 4) if corr == corr else None,
            "tracking_error_annualised": round(te, 4) if te == te else None,
            "return_gap": round(float(gap.iloc[-1]), 6),
            "max_abs_gap": round(float(gap.abs().max()), 6),
            "mean_session_diff_bps": round(float(diff.mean() * 1e4), 2),
        }
    # Do the two books hold the same names now?
    live_pos = {p["ticker"] for p in ((steps[-1].get("portfolio") or {}).get("positions") or [])}
    bt_pos: set[str] = set()
    if result.decisions:
        bt_pos = {p["ticker"] for p in (result.decisions[-1].portfolio.get("positions") or [])}
    union = live_pos | bt_pos
    out["positions"] = {
        "live": sorted(live_pos), "backtest": sorted(bt_pos),
        "overlap": round(len(live_pos & bt_pos) / len(union), 3) if union else None,
        "only_live": sorted(live_pos - bt_pos), "only_backtest": sorted(bt_pos - live_pos),
    }

    # The caveats.
    starting_cash = float(lc.get("cash") or 0)
    resets = [
        ts.date().isoformat() for ts, v in live.iloc[1:].items()
        if starting_cash and abs(v - starting_cash) < 1e-6
    ]
    if resets:
        out["warnings"].append(
            f"equity returned exactly to starting cash on {', '.join(resets)} -- a fresh "
            "book, not a trade; pass `since` past the last reset to compare one "
            "continuous run"
        )
    recorded = lc.get("strategy_hash")
    if recorded and getattr(result, "strategy_hash", None) and recorded != result.strategy_hash:
        out["warnings"].append(
            "the strategy file has changed since the paper run started -- the backtest "
            "ran today's code, the paper run is running the old code, and the two are "
            "not the same strategy"
        )
    if len(both) < int(min_steps):
        out["warnings"].append(
            f"only {len(both)} sessions compared; below {int(min_steps)} the agreement "
            "numbers describe noise, not the strategy"
        )
    agree = out.get("agreement") or {}
    corr = agree.get("session_return_correlation")
    if isinstance(corr, (int, float)) and corr < 0.5 and len(both) >= int(min_steps):
        out["warnings"].append(
            f"session-return correlation is {corr:.2f} -- the paper book is not doing "
            "what the backtest did; look at fill prices, decision timing and data "
            "freshness before trusting either number"
        )
    ov = out["positions"]["overlap"]
    if isinstance(ov, (int, float)) and ov < 0.5 and union:
        out["warnings"].append(
            f"the two books share only {ov:.0%} of their names -- a divergence in what "
            "was held, not just at what price"
        )
    out["verdict"] = (
        "too short to judge" if len(both) < int(min_steps) else
        "paper run tracks its backtest" if not out["warnings"] else
        "paper run diverges from its backtest -- see warnings"
    )
    return jsonable(out)
