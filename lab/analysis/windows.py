"""Windowed metrics, benchmarks and error bars -- the arithmetic of a result.

Pure functions over an equity curve and a config. Nothing here knows about
models, prompts or sessions; these lived in ``lab/agent`` only because that is
where they were first needed, which left the MCP layer importing nine private
names from the layer it was built to replace.

Two of these encode fixes that cost real money to find:

``tradeable_timestamps`` drops warmup bars. Anything derived from the bar list
has to, or the in-sample/out-of-sample split lands at a different date than the
declared ratio implies *and* the benchmark gets measured from the first bar of
data while the strategy is measured from the first bar it could trade. On the
shipped momo config that was a 210-bar, ~107-point head start handed to
buy-and-hold.

``window_metrics`` carries exposure and turnover into the slice. Omitting them
does not lose a column -- ``compute_metrics`` reports 0.0, and a strategy sitting
in cash then reads as fully invested, and one with 1,428 trades reads as one that
never traded.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from lab.timeutil import to_utc

#: Metric keys carried through a windowed slice. ``periods`` is the bar count and
#: is what turns a score into a score with an error bar.
LINEAGE_METRICS: tuple[str, ...] = (
    "sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown",
    "volatility", "trades", "hit_rate", "profit_factor", "exposure", "turnover",
    "final_equity", "periods",
)

def tradeable_timestamps(base_cfg: Any, data: Any) -> list[datetime]:
    """Bars the strategy can actually trade on -- warmup excluded.

    The warmup bars exist so indicators are warm before the first decision; the
    strategy is deliberately inert across them and the runner keeps them out of
    the equity curve. Anything derived from the bar list has to drop them too,
    or it silently compares two different windows:

    * the in-sample/out-of-sample split lands at a different date than the
      declared ratio implies, and
    * the benchmark gets measured from the first bar of *data* while the
      strategy is measured from the first bar it could *trade* -- on the shipped
      momo config that is a 210-bar, ~107pp head start handed to buy-and-hold.

    Both failures flatter the benchmark, so a strategy that beat the market can
    read as one that lost to it.
    """
    if data is not None:
        stamps = [to_utc(ts) for ts in data.timestamps()]
    else:
        from lab.engine.context import DataView

        view = DataView.from_store(
            base_cfg.tickers,
            timeframe=base_cfg.timeframe,
            start=base_cfg.start,
            end=base_cfg.end,
            sources=base_cfg.sources,
            bar_source=getattr(base_cfg, "source", None),
            regular_hours=getattr(base_cfg, "regular_hours", True),
        )
        stamps = [to_utc(ts) for ts in view.timestamps()]

    warmup = int(getattr(base_cfg, "warmup", 0) or 0)
    # Never hand back an empty list: a warmup longer than the data is a broken
    # config, but the split needs two bars to say anything at all.
    return stamps[warmup:] if 0 < warmup < len(stamps) else stamps


def split_point(timestamps: Sequence[datetime], spec: str) -> tuple[datetime, datetime]:
    """Parse an ``IS:OOS`` ratio (``"4:1"``) into the boundary timestamps."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", str(spec))
    if not m:
        raise ValueError(f"oos_split must look like '4:1', got {spec!r}")
    is_part, oos_part = float(m.group(1)), float(m.group(2))
    if is_part <= 0 or oos_part <= 0:
        raise ValueError(f"oos_split parts must be > 0, got {spec!r}")
    n = len(timestamps)
    cut = int(n * is_part / (is_part + oos_part))
    cut = min(max(cut, 1), n - 1)
    return timestamps[cut - 1], timestamps[cut]


def market_reference(
    base_cfg: Any,
    oos_start: datetime | None,
    tradeable_start: datetime | None = None,
) -> dict[str, Any] | None:
    """What simply holding the benchmark would have done, split the same way.

    The loop's own baseline is the *unmodified seed*, which answers "is this
    proposal better than where we started" and nothing else. A lineage can climb
    steadily and still be worse than owning SPY and going outside, and neither
    the model nor the operator would see it. So the benchmark is carried as a
    fixed reference row the model is told to clear.

    Deliberately costless -- no slippage, no commission, no gate. That flatters
    the benchmark slightly, which is the right direction for a bar you are
    trying to beat.
    """
    ticker = getattr(base_cfg, "benchmark", None)
    if not ticker:
        return None
    try:
        from lab.backtest import metrics as M
        from lab.store import parquet_io

        bars = parquet_io.read_bars(
            [str(ticker)],
            timeframe=base_cfg.timeframe,
            start=base_cfg.start,
            end=base_cfg.end,
            source=getattr(base_cfg, "source", None),
        )
        if bars is None or len(bars) == 0:
            return None
        closes = bars.set_index("event_time")["close"].astype("float64").sort_index().dropna()
        # Start where the strategy starts. Normalising from an earlier bar would
        # credit the benchmark with the warmup period, which the strategy spent
        # inert and out of the market.
        if tradeable_start is not None:
            closes = closes[closes.index >= pd.Timestamp(to_utc(tradeable_start))]
        if len(closes) < 2 or float(closes.iloc[0]) <= 0:
            return None

        timeframe = getattr(base_cfg, "timeframe", "1d")
        equity = closes / float(closes.iloc[0]) * float(base_cfg.cash)
        keep = ("sharpe", "total_return", "max_drawdown")

        full = M.compute_metrics(equity, [], timeframe=timeframe)
        if oos_start is not None:
            cut = pd.Timestamp(to_utc(oos_start))
            # Same boundary convention as _window_metrics: the split bar belongs
            # to out-of-sample, so the two halves cannot both claim it.
            ins = M.compute_metrics(equity[equity.index < cut], [], timeframe=timeframe)
            oos = M.compute_metrics(equity[equity.index >= cut], [], timeframe=timeframe)
        else:
            ins, oos = full, {}
    except Exception as exc:
        # A missing or unreadable benchmark is a thinner prompt, never a failed
        # loop -- but say so, because a silently absent bar is one nobody clears.
        log.warning("no market reference for %s: %s", ticker, exc)
        return None

    return {
        "instrument": f"buy_and_hold({str(ticker).upper()})",
        "window": {
            "start": closes.index[0].date().isoformat(),
            "end": closes.index[-1].date().isoformat(),
            "bars": int(len(closes)),
            "oos_bars": int((closes.index >= pd.Timestamp(to_utc(oos_start))).sum())
            if oos_start is not None
            else None,
        },
        "oos": {k: oos.get(k) for k in keep},
        "in_sample": {k: ins.get(k) for k in keep},
        # The whole tradeable period, not just the held-out slice. Without it the
        # only comparison on offer is the last ~20% of the range, and a strategy
        # that quadrupled the benchmark over four years can be written off on
        # eleven months of it.
        "full": {k: full.get(k) for k in keep},
    }


def window_metrics(
    result: Any,
    timeframe: str,
    *,
    lo: datetime | None = None,
    hi: datetime | None = None,
    inclusive_hi: bool = True,
) -> dict[str, Any]:
    from lab.backtest import metrics as M

    equity = result.equity
    if lo is not None:
        equity = equity[equity.index >= pd.Timestamp(to_utc(lo))]
    if hi is not None:
        cut = pd.Timestamp(to_utc(hi))
        equity = equity[equity.index <= cut] if inclusive_hi else equity[equity.index < cut]

    # Slice exposure onto the same window. Omitting it does not merely lose a
    # column -- compute_metrics reports exposure 0.0, and every consumer then
    # reads a strategy sitting in cash as one that is fully invested.
    exposure = getattr(result, "exposure", None)
    if exposure is not None and len(exposure):
        exposure = exposure.reindex(equity.index).dropna()
    else:
        exposure = None

    trades = []
    for t in result.trades:
        when = t.exit_time or t.entry_time
        if when is None:
            continue
        when = to_utc(when)
        if lo is not None and when < to_utc(lo):
            continue
        if hi is not None:
            edge = to_utc(hi)
            if when > edge or (not inclusive_hi and when >= edge):
                continue
        trades.append(t)

    # Turnover accumulates per fill across the whole run, so a window that does
    # not re-sum it reports 0.0 -- the same silent-zero shape the exposure bug
    # had and just as misleading: a strategy with 1,428 trades reads as one that
    # never traded. Summed from fills, not trades, so it matches the runner's own
    # definition rather than approximating it with round trips.
    turnover_notional = 0.0
    for fill in getattr(result, "fills", None) or []:
        when = getattr(fill, "at", None)
        if when is None:
            continue
        when = to_utc(when)
        if lo is not None and when < to_utc(lo):
            continue
        if hi is not None:
            edge = to_utc(hi)
            if when > edge or (not inclusive_hi and when >= edge):
                continue
        turnover_notional += abs(fill.notional)

    computed = M.compute_metrics(
        equity, trades, timeframe=timeframe, exposure=exposure,
        turnover_notional=turnover_notional,
    )
    return {k: computed.get(k) for k in LINEAGE_METRICS}


def period_breakdown(result: Any, timeframe: str, n_periods: int) -> list[dict[str, Any]]:
    """Split the run into contiguous equal blocks and score each one.

    This is the harness asking "does it work in more than one market", and it
    matters that the *harness* asks. Letting the agent choose its own test
    window would be cherry-picking with extra steps: whoever picks the period
    and is scored on it will find the period that flatters them. So the split is
    fixed, the agent sees the per-period result, and the only thing it can
    change is the strategy.
    """
    if n_periods < 2:
        return []
    equity = getattr(result, "equity", None)
    if equity is None or len(equity) < n_periods * 2:
        return []

    index = equity.index
    edges = [int(round(i * len(index) / n_periods)) for i in range(n_periods + 1)]
    out: list[dict[str, Any]] = []
    for i in range(n_periods):
        lo_i, hi_i = edges[i], edges[i + 1]
        if hi_i - lo_i < 2:
            continue
        lo, hi = index[lo_i], index[hi_i - 1]
        block = window_metrics(result, timeframe, lo=lo, hi=hi)
        out.append(
            {
                "period": i + 1,
                "start": to_utc(lo).date().isoformat(),
                "end": to_utc(hi).date().isoformat(),
                "sharpe": block.get("sharpe"),
                "total_return": block.get("total_return"),
                "max_drawdown": block.get("max_drawdown"),
                "trades": block.get("trades"),
            }
        )
    return out


def fitness_one_sigma(metric: str, windows: Mapping[str, Any], timeframe: str,
                       n_periods: int) -> float | None:
    """One-sigma sampling error on the fitness metric, from the window's length.

    A score quoted to four decimals reads like a measurement. On a held-out
    window of 232 daily bars an annualised Sharpe carries a standard error of
    about 1.04 -- so 0.99 and 0.04 are the same number, and a whole parameter
    sweep can rank on nothing. Reporting the error alongside the score is the
    difference between "this one is better" and "these are indistinguishable".

    Sharpe: SE = sqrt((P + S^2/2) / n), the usual Lo (2002) result rearranged for
    an already-annualised figure. Return metrics have no estimation error in the
    same sense -- the number is what happened -- so what is reported is the
    dispersion of the outcome, sigma*sqrt(T) for a total return and
    sigma/sqrt(T) for an annualised one: how far this result would move if the
    same process were sampled again.
    """
    from math import sqrt

    from lab.backtest.metrics import periods_per_year

    base = str(metric).split("_", 1)[-1] if "_" in str(metric) else str(metric)
    n = int(n_periods or 0)
    if n < 8:
        return None
    ppy = max(1, int(periods_per_year(timeframe)))

    if base in ("sharpe", "sortino"):
        score = windows.get(base)
        if not isinstance(score, (int, float)):
            return None
        return round(sqrt((ppy + float(score) ** 2 / 2.0) / n), 4)

    if base in ("total_return", "cagr"):
        vol = windows.get("volatility")
        if not isinstance(vol, (int, float)) or float(vol) <= 0:
            return None
        years = n / ppy
        if years <= 0:
            return None
        return round(float(vol) * sqrt(years) if base == "total_return"
                     else float(vol) / sqrt(years), 4)

    # calmar and anything else: the error is drawdown-driven and not worth a
    # closed form nobody can check. Silence beats a number that looks derived.
    return None
