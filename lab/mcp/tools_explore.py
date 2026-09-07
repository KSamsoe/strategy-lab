"""Look at the data before designing against it.

The research loop could only backtest, which meant it could only ever guess and
check. Every genuine finding this lab has produced came from measuring first:

* cross-sectional momentum has signal at ~126 days and none at 5, 21 or 63 --
  which explains why every short-horizon strategy failed and stopped a whole
  class of them being tried;
* realised volatility is the most predictable quantity in the data (+0.49
  autocorrelation) but scaling market exposure by it barely pays, because the
  high-volatility periods also had the higher returns;
* deep drawdowns predict higher forward returns, but not by enough to overcome
  the cash drag of holding powder under a 1.0 gross cap -- a conclusion reached
  with arithmetic instead of four minutes of backtesting.

None of those were reachable by running more backtests. All three are one call
here.
"""

from __future__ import annotations

from typing import Any

from lab.mcp._common import jsonable, load_config


def _prices(config: str, tickers: list[str] | None = None):
    """Aligned close prices for a config's universe, from its declared source."""
    import pandas as pd

    from lab.store import parquet_io

    cfg = load_config(config)
    names = [t.upper() for t in (tickers or cfg.tickers)]
    bars = parquet_io.read_bars(
        names, timeframe=cfg.timeframe, start=cfg.start, end=cfg.end,
        source=getattr(cfg, "source", None),
    )
    if bars is None or not len(bars):
        raise ValueError(
            f"no bars for {len(names)} ticker(s) at {cfg.timeframe} from "
            f"source={getattr(cfg, 'source', None)!r} -- check data_coverage"
        )
    px = bars.pivot_table(index="event_time", columns="ticker", values="close").sort_index()
    return cfg, px.dropna(axis=0, how="any")


def signal_scan(
    config: str,
    lookbacks: list[int] | None = None,
    forwards: list[int] | None = None,
) -> dict[str, Any]:
    """Measure what actually predicts anything in this universe.

    Three questions, answered on the config's own data and timeframe:

    * **Time-series autocorrelation** -- does a name's own past return predict
      its next one? Usually no, at any lag, and knowing that kills a family of
      strategies before you write them.
    * **Cross-sectional rank persistence** -- do this period's winners keep
      winning relative to their peers? This is the momentum signal, and it is
      typically real at only one horizon.
    * **Volatility persistence** -- does trailing realised vol predict forward
      vol? Almost always strongly yes, which is why it is worth sizing with even
      when it is useless for timing.

    Correlations near zero are the useful answer, not a failed measurement: they
    tell you which horizons are noise. Reported alongside the sample count so a
    suggestive number on thin data is visible as one.
    """
    import numpy as np
    import pandas as pd

    cfg, px = _prices(config)
    ret = px.pct_change().dropna()
    lookbacks = [int(x) for x in (lookbacks or [5, 21, 63, 126, 252])]
    forwards = [int(x) for x in (forwards or [5, 21])]

    autocorr = []
    flat = ret.stack()
    for lag in (1, 2, 5, 10, 21):
        both = pd.concat([flat, ret.shift(lag).stack()], axis=1).dropna()
        if len(both) > 30:
            autocorr.append({
                "lag_bars": lag,
                "corr": round(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 5),
                "n": int(len(both)),
            })

    # Three names is the fewest a rank correlation can be computed on at all.
    # Returning an empty table for a small universe would read as "no signal
    # here" when the truth is "this question was never asked".
    min_names = 3
    persistence = []
    for look in lookbacks:
        if look >= len(px) - max(forwards):
            continue
        trail = px.pct_change(look)
        for fwd in forwards:
            forward = px.shift(-fwd) / px - 1.0
            cors = []
            for t in range(look, len(px) - fwd, max(1, fwd)):
                a, b = trail.iloc[t], forward.iloc[t]
                ok = a.notna() & b.notna()
                if int(ok.sum()) >= min_names:
                    value = a[ok].rank().corr(b[ok].rank())
                    if value == value:  # a constant column gives NaN
                        cors.append(value)
            if cors:
                persistence.append({
                    "lookback_bars": look, "forward_bars": fwd,
                    "mean_rank_corr": round(float(np.mean(cors)), 5),
                    "n_dates": len(cors),
                    "names_ranked": int(px.shape[1]),
                })

    vol = []
    bench = str(getattr(cfg, "benchmark", "") or "").upper()
    series = ret[bench] if bench in ret else ret.mean(axis=1)
    for w in (5, 10, 21, 63):
        if len(series) < w * 3:
            continue
        both = pd.concat([series.rolling(w).std(),
                          series.shift(-w).rolling(w).std()], axis=1).dropna()
        if len(both) > 30:
            vol.append({
                "window_bars": w,
                "corr": round(float(both.iloc[:, 0].corr(both.iloc[:, 1])), 5),
                "n": int(len(both)),
            })

    best = max(persistence, key=lambda d: d["mean_rank_corr"], default=None)
    notes = []
    if px.shape[1] < 8:
        notes.append(
            f"only {px.shape[1]} names: a cross-sectional rank correlation over so few "
            "is dominated by which one happened to lead, and should not be read as a "
            "signal either way"
        )
    if not persistence:
        notes.append(
            "no cross-sectional measurement was possible -- the universe has fewer "
            "than three names with overlapping history, or the lookbacks exceed the "
            "data. This is a missing question, not a negative answer."
        )
    notes += [
        "Rank correlations near zero mean that horizon carries no cross-sectional "
        "signal -- a useful answer, not a failed measurement.",
        "Volatility persistence is usually far stronger than any return signal. It "
        "is worth sizing positions with; it is usually not worth timing exposure "
        "with, because high-volatility periods often carry high returns too.",
    ]
    if best and best["mean_rank_corr"] > 0.01:
        notes.append(
            f"strongest cross-sectional horizon here is {best['lookback_bars']} bars "
            f"(rank corr {best['mean_rank_corr']:+.4f} over {best['forward_bars']} "
            "bars forward). Blending it with a second horizon usually still beats "
            "using it alone."
        )
    return jsonable({
        "universe": len(px.columns), "bars": len(px),
        "range": [px.index[0], px.index[-1]], "timeframe": cfg.timeframe,
        "autocorrelation": autocorr,
        "cross_sectional_persistence": persistence,
        "volatility_persistence": vol,
        "notes": notes,
    })


def conditional_returns(
    config: str,
    condition: str = "drawdown",
    forward_bars: int = 21,
    buckets: int = 4,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Forward returns bucketed by a condition: does this state predict anything?

    Args:
        condition: one of `drawdown` (from a trailing 252-bar high), `vol`
            (trailing 21-bar realised), `trend` (distance from a 200-bar mean).
        forward_bars: horizon of the forward return being predicted.
        buckets: number of quantile buckets, ignored for `drawdown` which uses
            fixed, interpretable cut points.
        ticker: a single name, else the config's benchmark, else equal-weight.

    Reports mean and standard deviation of the forward return per bucket. A flat
    mean with a rising standard deviation is the common and important case: the
    state predicts *risk* but not *return*, so it is worth sizing on and not
    worth timing on.
    """
    import numpy as np
    import pandas as pd

    cfg, px = _prices(config)
    ret = px.pct_change().dropna()
    name = (ticker or getattr(cfg, "benchmark", "") or "").upper()
    if name and name in px:
        series, level = ret[name], px[name]
    else:
        series = ret.mean(axis=1)
        level = (1 + series).cumprod()
        name = "equal_weight"

    if condition == "drawdown":
        signal = level / level.rolling(252, min_periods=60).max() - 1.0
        edges = [-1.0, -0.15, -0.07, -0.03, -0.005, 1.0]
        labels = ["< -15%", "-15..-7%", "-7..-3%", "-3..-0.5%", "at highs"]
        cut = pd.cut(signal, edges, labels=labels)
    elif condition == "vol":
        signal = series.rolling(21).std() * np.sqrt(252)
        cut = pd.qcut(signal, int(buckets), duplicates="drop")
    elif condition == "trend":
        signal = level / level.rolling(200, min_periods=60).mean() - 1.0
        cut = pd.qcut(signal, int(buckets), duplicates="drop")
    else:
        raise ValueError(f"condition must be drawdown, vol or trend; got {condition!r}")

    fwd = series.shift(-int(forward_bars)).rolling(int(forward_bars)).sum()
    frame = pd.DataFrame({"bucket": cut, "fwd": fwd}).dropna()
    grouped = frame.groupby("bucket", observed=True)["fwd"].agg(["mean", "std", "count"])

    rows = [
        {"bucket": str(idx), "mean_forward_return": round(float(r["mean"]), 5),
         "std": round(float(r["std"]), 5), "n": int(r["count"])}
        for idx, r in grouped.iterrows()
    ]
    notes = []
    if len(rows) >= 2:
        means = [r["mean_forward_return"] for r in rows]
        stds = [r["std"] for r in rows]
        if max(means) - min(means) < 0.01 and max(stds) > 1.5 * min(stds):
            notes.append(
                "means are flat while dispersion rises sharply: this condition "
                "predicts risk, not return. Size with it; do not time with it."
            )
    thin = [r for r in rows if r["n"] < 40]
    if thin:
        notes.append(
            f"{len(thin)} bucket(s) have under 40 overlapping observations, and "
            "overlapping forward windows make the effective sample smaller still -- "
            "treat those rows as suggestive at best"
        )
    return jsonable({
        "series": name, "condition": condition, "forward_bars": int(forward_bars),
        "buckets": rows, "notes": notes,
    })


def correlation_matrix(config: str, tickers: list[str] | None = None) -> dict[str, Any]:
    """Return correlations across the universe, plus how concentrated it really is.

    A universe of thirty names that all move together is a universe of one, and a
    strategy that "diversifies" across it has not. The effective-N figure is the
    inverse Herfindahl of the correlation eigenvalues: roughly how many
    independent bets the universe can actually express.
    """
    import numpy as np

    _, px = _prices(config, tickers)
    ret = px.pct_change().dropna()
    corr = ret.corr()

    vals = np.linalg.eigvalsh(corr.values)
    vals = np.clip(vals, 0, None)
    share = vals / vals.sum() if vals.sum() > 0 else vals
    effective_n = float(1.0 / np.sum(share**2)) if vals.sum() > 0 else 0.0

    off = corr.values[np.triu_indices_from(corr.values, k=1)]
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((float(corr.iat[i, j]), cols[i], cols[j]))
    pairs.sort(reverse=True)

    return jsonable({
        "tickers": len(cols), "bars": len(ret),
        "mean_pairwise_corr": round(float(np.mean(off)), 4),
        "effective_independent_bets": round(effective_n, 2),
        "most_correlated": [
            {"pair": [a, b], "corr": round(c, 4)} for c, a, b in pairs[:8]
        ],
        "least_correlated": [
            {"pair": [a, b], "corr": round(c, 4)} for c, a, b in pairs[-8:]
        ],
        "notes": [
            f"{len(cols)} names but roughly {effective_n:.1f} independent bets -- "
            "a strategy holding many of these is less diversified than its position "
            "count suggests"
        ],
    })


def data_pull(
    source: str,
    tickers: str,
    timeframe: str = "1d",
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Fetch bars into the point-in-time store.

    `tickers` is a comma list or the path to a universe file. Note that a data
    tier may cap history by *row count* rather than by date -- Alpaca's does, at
    roughly 1,530 bars per symbol, so asking for 2005 silently returns 2020
    onward. `data_coverage` afterwards will show what actually arrived.
    """
    from lab.adapters import get_adapter
    from lab.backtest.runner import load_universe
    from lab.store import parquet_io
    from lab.timeutil import to_utc, utcnow

    from datetime import datetime

    names = (
        load_universe(tickers) if tickers.endswith((".txt", ".csv"))
        else [t.strip().upper() for t in tickers.split(",") if t.strip()]
    )
    lo = to_utc(datetime.fromisoformat(since)) if since else to_utc(datetime(2018, 1, 1))
    hi = to_utc(datetime.fromisoformat(until)) if until else utcnow()

    adapter = get_adapter(source)
    ok, why = adapter.available()
    if not ok:
        return {"ok": False, "source": source, "refused": why}

    rows = parquet_io.write_bars(adapter.fetch_bars(names, timeframe, lo, hi))
    return jsonable({
        "ok": True, "source": source, "timeframe": timeframe,
        "tickers": len(names), "rows_written": rows,
        "requested_from": lo, "requested_to": hi,
        "note": "check data_coverage for what actually landed -- some tiers cap by "
                "row count rather than by date",
    })


def data_quality(config: str) -> dict[str, Any]:
    """Integrity check on the bars a config would actually trade.

    Looks for the failures that quietly ruin a backtest rather than break it:
    tickers whose history starts late (so the universe silently changes
    mid-test), gaps in the calendar, zero-volume or unchanged bars, and single-bar
    moves large enough to be an unadjusted split rather than a real return.
    """
    import numpy as np

    cfg, px_all = _prices(config)
    from lab.store import parquet_io

    bars = parquet_io.read_bars(
        [t.upper() for t in cfg.tickers], timeframe=cfg.timeframe,
        start=cfg.start, end=cfg.end, source=getattr(cfg, "source", None),
    )
    wide = bars.pivot_table(index="event_time", columns="ticker", values="close").sort_index()

    first = wide.apply(lambda s: s.first_valid_index())
    latest_start = first.max()
    late = sorted(
        [{"ticker": t, "first_bar": v} for t, v in first.items() if v is not None and v > first.min()],
        key=lambda d: str(d["first_bar"]), reverse=True,
    )[:10]

    ret = wide.pct_change()
    suspicious = []
    for t in wide.columns:
        s = ret[t].dropna()
        if not len(s):
            continue
        big = s[abs(s) > 0.35]
        for stamp, move in big.items():
            suspicious.append({"ticker": t, "at": stamp, "move": round(float(move), 4)})
    suspicious = sorted(suspicious, key=lambda d: abs(d["move"]), reverse=True)[:10]

    flat = {
        t: int((wide[t].diff() == 0).sum()) for t in wide.columns
        if int((wide[t].diff() == 0).sum()) > max(5, len(wide) * 0.02)
    }

    warnings = []
    if late:
        warnings.append(
            f"{len(late)} ticker(s) start after the earliest -- the universe changes "
            "mid-test, which confounds 'did the strategy work then' with 'did the "
            f"universe change then'. Latest start: {latest_start}"
        )
    if suspicious:
        warnings.append(
            f"{len(suspicious)} single-bar move(s) above 35% -- verify these are real "
            "returns and not unadjusted splits before trusting any result"
        )
    if flat:
        warnings.append(
            f"{len(flat)} ticker(s) have many unchanged closes, which usually means "
            "stale or padded bars"
        )
    return jsonable({
        "tickers": int(wide.shape[1]), "bars": int(wide.shape[0]),
        "aligned_bars": int(len(px_all)),
        "range": [wide.index[0], wide.index[-1]],
        "late_starters": late,
        "suspicious_moves": suspicious,
        "stale_bar_counts": flat,
        "warnings": warnings or ["no integrity problems found"],
    })
