"""Performance metrics.

Every number here is computed from the equity curve and the trade ledger, both
of which come out of the same portfolio accounting the live runner uses. That is
deliberate: a paper account's Sharpe and a backtest's Sharpe are produced by the
same function, so comparing them is meaningful rather than approximate.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lab.engine.events import Trade
from lab.timeutil import parse_timeframe

TRADING_DAYS = 252
TRADING_MINUTES_PER_DAY = 390


def periods_per_year(timeframe: str = "1d") -> int:
    """Annualization factor for a cadence, on the trading calendar."""
    delta = parse_timeframe(timeframe)
    if delta.days >= 1:
        return max(1, int(round(TRADING_DAYS / delta.days)))
    minutes = delta.total_seconds() / 60.0
    return max(1, int(round(TRADING_DAYS * TRADING_MINUTES_PER_DAY / minutes)))


#: Bound here because ``compute_metrics`` takes a ``periods_per_year`` keyword
#: that would otherwise shadow the function of the same name.
_periods_per_year = periods_per_year


def _clean(equity: pd.Series) -> pd.Series:
    if equity is None or len(equity) == 0:
        return pd.Series(dtype="float64")
    s = pd.Series(equity).astype("float64").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    return s[~s.index.duplicated(keep="last")].sort_index()


def drawdown_series(equity: pd.Series) -> pd.Series:
    s = _clean(equity)
    if s.empty:
        return s
    peak = s.cummax()
    return (s / peak) - 1.0


def max_drawdown_duration_days(equity: pd.Series) -> float:
    """Longest peak-to-recovery stretch, in calendar days.

    Measured from the **peak itself**, not from the first bar below it, which is
    the convention pyfolio and quantstats use and the one a trader means by "I
    was underwater for four months". An unrecovered drawdown runs to the end of
    the curve.

    Depth gets all the attention, but duration is what actually breaks someone's
    willingness to keep running a strategy.
    """
    s = _clean(equity)
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return 0.0
    peak = s.cummax()
    underwater = s < peak
    longest = 0.0
    peak_ts: pd.Timestamp = s.index[0]
    start: pd.Timestamp | None = None
    for ts, is_under in underwater.items():
        if is_under:
            if start is None:
                start = peak_ts  # the high we fell from, not the first low
        else:
            if start is not None:
                longest = max(longest, (ts - start).total_seconds() / 86400.0)
                start = None
            peak_ts = ts
    if start is not None:
        longest = max(longest, (s.index[-1] - start).total_seconds() / 86400.0)
    return round(longest, 2)


def _annualized_return(s: pd.Series, ppy: int) -> float:
    if len(s) < 2 or s.iloc[0] <= 0:
        return 0.0
    total = s.iloc[-1] / s.iloc[0]
    if total <= 0:
        return -1.0
    if isinstance(s.index, pd.DatetimeIndex):
        years = (s.index[-1] - s.index[0]).total_seconds() / (365.25 * 86400.0)
    else:
        years = len(s) / ppy
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def summarize_trades(trades: Sequence[Trade]) -> dict[str, Any]:
    closed = [t for t in trades if t.exit_time is not None]
    if not closed:
        return {
            "trades": 0, "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "avg_trade_pnl": 0.0, "best_trade": 0.0,
            "worst_trade": 0.0, "avg_bars_held": 0.0, "wins": 0, "losses": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "expectancy": 0.0,
        }
    pnls = np.array([t.pnl for t in closed], dtype="float64")
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    hit_rate = float(wins.size / pnls.size)
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    return {
        "trades": int(pnls.size),
        "wins": int(wins.size),
        "losses": int(losses.size),
        "hit_rate": round(hit_rate, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        ),
        "avg_trade_pnl": round(float(pnls.mean()), 6),
        "best_trade": round(float(pnls.max()), 6),
        "worst_trade": round(float(pnls.min()), 6),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in closed])), 3),
        "expectancy": round(hit_rate * avg_win + (1 - hit_rate) * avg_loss, 6),
    }


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade] = (),
    *,
    periods_per_year_: int | None = None,
    periods_per_year: int | None = None,
    timeframe: str = "1d",
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
    exposure: pd.Series | None = None,
    turnover_notional: float = 0.0,
) -> dict[str, Any]:
    """The metrics JSON every run emits.

    Returns plain floats and strings only -- the dict is written straight to
    ``metrics.json`` and read by the registry, the report, the console, and an
    iterating agent, so it must survive ``json.dumps`` without a custom encoder.
    """
    ppy = periods_per_year or periods_per_year_ or _periods_per_year(timeframe)
    s = _clean(equity)

    out: dict[str, Any] = {
        "start": None, "end": None, "days": 0.0, "periods": int(len(s)),
        "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "volatility": 0.0, "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0.0, "exposure": 0.0, "turnover": 0.0,
        "final_equity": 0.0, "peak_equity": 0.0, "starting_equity": 0.0,
        "best_period": 0.0, "worst_period": 0.0, "periods_per_year": int(ppy),
    }
    out |= summarize_trades(trades)

    if s.empty:
        return out

    start_eq, final_eq = float(s.iloc[0]), float(s.iloc[-1])
    out["starting_equity"] = round(start_eq, 6)
    out["final_equity"] = round(final_eq, 6)
    out["peak_equity"] = round(float(s.max()), 6)
    out["total_return"] = round(final_eq / start_eq - 1.0, 6) if start_eq else 0.0

    if isinstance(s.index, pd.DatetimeIndex):
        out["start"] = s.index[0].isoformat()
        out["end"] = s.index[-1].isoformat()
        out["days"] = round((s.index[-1] - s.index[0]).total_seconds() / 86400.0, 2)

    rets = s.pct_change().dropna()
    out["cagr"] = round(_annualized_return(s, ppy), 6)

    if len(rets) > 1:
        vol = float(rets.std(ddof=1))
        ann_vol = vol * math.sqrt(ppy)
        out["volatility"] = round(ann_vol, 6)
        rf_per_period = risk_free / ppy
        excess = rets - rf_per_period
        out["sharpe"] = round(
            float(excess.mean() / vol * math.sqrt(ppy)) if vol > 0 else 0.0, 6
        )
        # Downside deviation proper -- sqrt(mean(min(excess, 0)^2)) -- not the
        # sample stdev of the negative subset, which is undefined for a single
        # losing period and understates risk for a few.
        below = excess.clip(upper=0.0)
        dstd = float(np.sqrt((below**2).mean()))
        out["sortino"] = round(
            float(excess.mean() / dstd * math.sqrt(ppy)) if dstd > 0 else 0.0, 6
        )
        out["downside_deviation"] = round(dstd * math.sqrt(ppy), 6)
        out["best_period"] = round(float(rets.max()), 6)
        out["worst_period"] = round(float(rets.min()), 6)

    dd = drawdown_series(s)
    max_dd = float(dd.min()) if len(dd) else 0.0
    out["max_drawdown"] = round(max_dd, 6)
    out["max_drawdown_duration_days"] = max_drawdown_duration_days(s)
    out["calmar"] = round(out["cagr"] / abs(max_dd), 6) if max_dd < 0 else 0.0

    if exposure is not None and len(exposure):
        out["exposure"] = round(float(pd.Series(exposure).astype("float64").mean()), 6)

    avg_equity = float(s.mean())
    if avg_equity > 0 and turnover_notional > 0:
        years = max(out["days"] / 365.25, 1e-9)
        out["turnover"] = round(turnover_notional / avg_equity / years, 6)

    if benchmark is not None:
        b = _clean(benchmark)
        if len(b) > 1:
            b = b.reindex(s.index).ffill().dropna()
            if len(b) > 1 and float(b.iloc[0]) > 0:
                out["benchmark_total_return"] = round(float(b.iloc[-1] / b.iloc[0] - 1.0), 6)
                out["benchmark_cagr"] = round(_annualized_return(b, ppy), 6)
                out["excess_return"] = round(
                    out["total_return"] - out["benchmark_total_return"], 6
                )
                brets = b.pct_change().dropna()
                aligned = pd.concat([rets, brets], axis=1, join="inner").dropna()
                if len(aligned) > 2 and float(aligned.iloc[:, 1].var()) > 0:
                    cov = float(aligned.cov().iloc[0, 1])
                    out["beta"] = round(cov / float(aligned.iloc[:, 1].var()), 6)
                    out["alpha"] = round(
                        out["cagr"] - out["beta"] * out.get("benchmark_cagr", 0.0), 6
                    )

    # json.dumps chokes on inf; profit_factor is the one place it can appear.
    for k, v in list(out.items()):
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None if math.isnan(v) else 1e9
    return out


def equity_frame(equity: pd.Series) -> pd.DataFrame:
    """Equity plus drawdown, the shape the report and the API serve."""
    s = _clean(equity)
    return pd.DataFrame({"equity": s, "drawdown": drawdown_series(s)})


def _align_tz(value: Any, tz: Any) -> pd.Timestamp:
    """Put a window bound in the index's timezone.

    Window bounds arrive as aware datetimes from ``walkforward`` but as ISO
    strings once a run's ``config.json`` has been round-tripped, and equity
    indexes are always aware UTC -- comparing the two raises rather than
    mis-shading, so normalize before comparing.
    """
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_convert(None) if ts.tzinfo is not None else ts
    return ts.tz_localize("UTC").tz_convert(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def is_oos_split(
    index: pd.DatetimeIndex, windows: Sequence[Mapping[str, Any]] | None
) -> list[bool]:
    """Per-timestamp out-of-sample flags, for the shaded report/console charts."""
    flags = [False] * len(index)
    if not windows:
        return flags
    tz = getattr(index, "tz", None)
    for w in windows:
        start, end = w.get("oos_start"), w.get("oos_end")
        if start is None or end is None:
            continue
        lo, hi = _align_tz(start, tz), _align_tz(end, tz)
        for i, ts in enumerate(index):
            if lo <= ts <= hi:
                flags[i] = True
    return flags
