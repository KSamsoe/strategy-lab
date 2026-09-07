"""Is it luck? The question a single holdout usually cannot answer.

Every session this lab has run hit the same wall. Sharpe standard errors came out
between 0.49 and 1.18, which is wider than the gaps being ranked on -- one sweep
had its entire five-variant spread (0.00 to 0.99) sitting inside a single
standard error, and the agent picked the maximum of it and called the region
"confirmed working". The closed-form error bar is reported everywhere in this
toolset for exactly that reason, but it is an approximation resting on
assumptions daily returns do not honour.

These tools replace the approximation with measurement:

* ``bootstrap`` resamples the strategy's own returns in blocks, giving an
  empirical distribution rather than a formula.
* ``permutation_test`` destroys the signal while keeping the machinery, and asks
  whether the real thing beats its own randomised twin. This is the direct test
  for luck and nothing else here substitutes for it.
* ``deflated_sharpe`` corrects a Sharpe for how many things were tried before it
  was chosen, which is the honest version of the tuning-pressure heuristic.

All three parallelise across cores. ``workers`` defaults to something modest;
raise it on a machine that can take it -- backtests are CPU-bound and embarrassingly
parallel, so throughput scales close to linearly with cores.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from lab.mcp._common import jsonable, load_config, resolve_strategy, split_point


def _equity_returns(run_id: str):
    """Per-bar returns of a finished run, from its stored equity curve."""
    import pandas as pd

    from lab.backtest.runner import artifact_dir

    frame = pd.read_parquet(artifact_dir(run_id) / "equity.parquet")
    col = "event_time" if "event_time" in frame else frame.columns[0]
    equity = frame.set_index(col)["equity"].astype("float64")
    equity.index = pd.to_datetime(equity.index, utc=True)
    return equity.pct_change().dropna()


def _sharpe(returns, ppy: int) -> float:
    sd = float(returns.std())
    return float(returns.mean()) / sd * math.sqrt(ppy) if sd > 0 else 0.0


def bootstrap_vs_benchmark(
    run_id: str,
    config: str,
    timeframe: str = "1d",
    samples: int = 2000,
    block_bars: int = 21,
    seed: int = 0,
) -> dict[str, Any]:
    """Is this run's Sharpe better than the benchmark's? The paired test.

    Comparing a one-sample interval against a benchmark's point estimate is the
    wrong test and it is badly conservative. A strategy and its benchmark live
    through the same crashes, so their errors are strongly correlated and the
    uncertainty of the *difference* is far smaller than the uncertainty of
    either. Measured on a real 21-year run: the unpaired 90% interval was
    [0.572, 1.231] and comfortably contained the benchmark's 0.656, which looks
    like "no evidence"; the paired difference came out [+0.057, +0.428] with
    P(strategy <= benchmark) = 1.4%.

    Resamples both series with the *same* block indices, so every draw compares
    them over identical market conditions.
    """
    import numpy as np
    import pandas as pd

    from lab.backtest.metrics import periods_per_year
    from lab.mcp._common import load_config
    from lab.store import parquet_io

    cfg = load_config(config)
    bench_name = str(getattr(cfg, "benchmark", "") or "").upper()
    if not bench_name:
        raise ValueError(f"{config} names no `benchmark:`; nothing to compare against")

    strat = _equity_returns(run_id)
    bars = parquet_io.read_bars(
        [bench_name], timeframe=cfg.timeframe, start=cfg.start, end=cfg.end,
        source=getattr(cfg, "source", None),
    )
    closes = bars.set_index("event_time")["close"].astype("float64").sort_index()
    closes.index = pd.to_datetime(closes.index, utc=True)
    bench = closes.pct_change().dropna()

    both = pd.concat([strat.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
    if len(both) < block_bars * 4:
        raise ValueError(f"only {len(both)} overlapping bars; too few to compare")

    ppy = int(periods_per_year(timeframe))
    sharpe = lambda a: a.mean() / a.std() * math.sqrt(ppy) if a.std() > 0 else 0.0
    obs_s, obs_b = sharpe(both["s"].to_numpy()), sharpe(both["b"].to_numpy())

    values = both.to_numpy()
    n, block = len(values), max(1, int(block_bars))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(samples))
    for i in range(int(samples)):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        drawn = np.concatenate([values[s:s + block] for s in starts])[:n]
        diffs[i] = sharpe(drawn[:, 0]) - sharpe(drawn[:, 1])

    worse = float((diffs <= 0).mean())
    return jsonable({
        "run_id": run_id, "benchmark": bench_name, "paired_bars": n,
        "samples": int(samples), "block_bars": block,
        "observed": {
            "strategy_sharpe": round(obs_s, 4),
            "benchmark_sharpe": round(obs_b, 4),
            "difference": round(obs_s - obs_b, 4),
        },
        "difference": {
            "mean": round(float(diffs.mean()), 4),
            "std_error": round(float(diffs.std(ddof=1)), 4),
            "p05": round(float(np.percentile(diffs, 5)), 4),
            "p95": round(float(np.percentile(diffs, 95)), 4),
        },
        "prob_no_better_than_benchmark": round(worse, 4),
        "notes": [
            "paired: both series are resampled with the same block indices, so every "
            "draw compares them over identical market conditions",
            "this says the strategy's risk-adjusted return beats the benchmark's. It "
            "does not say the strategy's SIGNAL is doing the work -- construction, "
            "diversification and sizing can account for all of it. Use "
            "permutation_test to separate those.",
        ],
    })


def bootstrap(
    run_id: str,
    timeframe: str = "1d",
    samples: int = 1000,
    block_bars: int = 21,
    seed: int = 0,
) -> dict[str, Any]:
    """Empirical distribution of this run's Sharpe and return, by block bootstrap.

    Resamples the strategy's own bar returns in contiguous blocks, which
    preserves the volatility clustering and autocorrelation that an
    observation-by-observation bootstrap destroys. `block_bars` should be long
    enough to carry that structure -- roughly a month of bars is a reasonable
    default.

    The interval this produces is the honest one. Where the closed-form error bar
    assumes IID normal returns, this assumes only that the blocks are
    exchangeable, and on real data it is usually wider.
    """
    import numpy as np

    from lab.backtest.metrics import periods_per_year

    returns = _equity_returns(run_id)
    if len(returns) < block_bars * 4:
        raise ValueError(f"only {len(returns)} bars; too few to bootstrap in blocks")

    ppy = int(periods_per_year(timeframe))
    observed = _sharpe(returns, ppy)
    observed_total = float((1 + returns).prod() - 1)

    values = returns.to_numpy()
    n, block = len(values), max(1, int(block_bars))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n - block + 1, size=(int(samples), n_blocks))

    sharpes = np.empty(int(samples))
    totals = np.empty(int(samples))
    for i in range(int(samples)):
        drawn = np.concatenate([values[s:s + block] for s in starts[i]])[:n]
        sd = drawn.std()
        sharpes[i] = drawn.mean() / sd * math.sqrt(ppy) if sd > 0 else 0.0
        totals[i] = np.prod(1 + drawn) - 1

    pct = lambda a, q: round(float(np.percentile(a, q)), 4)
    return jsonable({
        "run_id": run_id, "samples": int(samples), "block_bars": block,
        "bars": n,
        "observed": {"sharpe": round(observed, 4), "total_return": round(observed_total, 4)},
        "sharpe": {
            "mean": round(float(sharpes.mean()), 4),
            "std_error": round(float(sharpes.std(ddof=1)), 4),
            "p05": pct(sharpes, 5), "p50": pct(sharpes, 50), "p95": pct(sharpes, 95),
            "prob_below_zero": round(float((sharpes <= 0).mean()), 4),
        },
        "total_return": {
            "p05": pct(totals, 5), "p50": pct(totals, 50), "p95": pct(totals, 95),
        },
        "notes": [
            "the bootstrap standard error is the honest one; the closed-form figure "
            "reported elsewhere assumes IID normal returns and is usually narrower "
            "than reality",
            "this measures the uncertainty of THIS strategy's returns. It cannot tell "
            "you whether the strategy was selected by data mining -- use "
            "permutation_test and deflated_sharpe for that",
        ],
    })


def permutation_test(
    config: str,
    run_id: str | None = None,
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    samples: int = 50,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
    seed: int = 0,
    workers: int = 4,
) -> dict[str, Any]:
    """Does this strategy beat its own randomised twin? The direct test for luck.

    Re-runs the strategy against shuffled price histories -- each ticker's returns
    are permuted in blocks, destroying cross-sectional and serial signal while
    preserving each name's own return and volatility distribution. Any strategy
    still "works" on such data only by chance, so the fraction of shuffles that
    match or beat the real run is an empirical p-value.

    Give `run_id` to test the exact strategy and parameters a run executed.
    This is expensive: it costs `samples` full backtests. It is also the only
    thing here that distinguishes a real edge from a well-fitted one, and on a
    many-core machine it is the obvious thing to spend cores on.

    A p-value near 0.5 means the strategy is doing nothing the shuffled data
    cannot do. Below ~0.05 is evidence of real structure -- though it still says
    nothing about whether that structure survives out of sample or costs.
    """
    import numpy as np

    from lab.backtest.runner import run_backtest
    from lab.mcp._common import evaluate

    if run_id:
        # An audit is about a run: take the strategy and params that run
        # actually executed, not whatever file is on disk under that name.
        from lab.mcp.tools_core import _strategy_path_for
        from lab.registry.runs import RunRegistry

        record = RunRegistry().get(run_id)
        if record is None:
            raise ValueError(f"no such run: {run_id}")
        strategy = str(_strategy_path_for(run_id, record))
        params = dict(record.params or {}) if params is None else params
    cfg = load_config(config, params=params)
    path = resolve_strategy(strategy, cfg)
    cfg = replace(cfg, strategy=path)
    oos, stamps = split_point(cfg, oos_split)

    real = run_backtest(cfg, register=False, journal=False)
    observed = evaluate(real, cfg, oos_start=oos, stamps=stamps, metric=metric)
    target = observed.get("score")
    if target is None:
        raise ValueError(f"metric {metric!r} produced no score on the real run")

    from lab.store import parquet_io

    bars = parquet_io.read_bars(
        [t.upper() for t in cfg.tickers], timeframe=cfg.timeframe,
        start=cfg.start, end=cfg.end, source=getattr(cfg, "source", None),
    )
    scores = _shuffled_scores(
        bars, cfg, oos, stamps, metric, int(samples), int(seed), int(workers)
    )
    if not scores:
        raise RuntimeError("every shuffled run failed; nothing to compare against")

    arr = np.array(scores, dtype="float64")
    # +1 in numerator and denominator: the observed run is itself one draw from
    # the null, so a p-value of exactly zero is not a claim the data can support.
    p = float((np.sum(arr >= target) + 1) / (len(arr) + 1))
    exposure = (observed.get("windows", {}).get("full") or {}).get("exposure")
    notes = [
        f"{len(arr)} of {samples} shuffles completed",
        "THE NULL: returns are resampled in blocks with every ticker reordered by the "
            "SAME block permutation, so each name keeps its own drift and volatility and "
            "the cross-sectional correlation survives; what is destroyed is which name "
            "leads on a given day. So this asks whether the SELECTION AND TIMING add "
            "anything beyond holding this universe in this construction -- not whether the strategy beats cash. A "
        "long-biased strategy inherits the drift and will score well on this null no "
        "matter how little its signal contributes, which is exactly why a high "
        "p-value here is informative.",
    ]
    if isinstance(exposure, (int, float)) and exposure > 0.8:
        notes.append(
            f"this strategy is {exposure:.0%} invested, so most of its return is "
            "market exposure the null also has. Read the p-value as a verdict on the "
            "signal, not on the strategy's profitability."
        )
    if len(arr) < 30:
        notes.append(
            f"only {len(arr)} draws -- far too few to place a p-value. Treat this as a "
            "smoke test and re-run with samples>=200 before concluding anything."
        )
    elif p > 0.2:
        notes.append(
            f"p={p:.3f}: the signal does little that randomised data cannot do. The "
            "result is consistent with luck, or with the strategy being mostly beta."
        )
    elif p <= 0.05:
        notes.append(
            f"p={p:.3f}: the observed score is hard to reach by chance on this data. "
            "That is evidence of real structure, not evidence it will persist."
        )
    return jsonable({
        "strategy": path.name, "metric": metric,
        "observed_score": target,
        "null": {
            "n": len(arr), "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(arr.max()), 4),
        },
        "p_value": round(p, 4),
        "notes": notes,
    })


def _shuffled_scores(bars, cfg, oos, stamps, metric, samples, seed, workers):
    """Score the strategy on `samples` block-shuffled versions of the data."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    payload = (bars, cfg, oos, stamps, metric)
    if workers <= 1:
        return [s for s in (_one_shuffle(payload, seed + i) for i in range(samples))
                if s is not None]

    out: list[float] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_shuffle, payload, seed + i) for i in range(samples)]
        for f in as_completed(futures):
            try:
                value = f.result()
            except Exception:  # noqa: BLE001 - a failed shuffle is not a failed test
                continue
            if value is not None:
                out.append(value)
    return out


def _one_shuffle(payload, seed, block: int = 21):
    """One shuffled backtest. Module-level so it survives pickling to a worker.

    **Every ticker is reordered by the same block permutation.** That is the whole
    correctness of this null. Shuffling each name independently also destroys the
    cross-sectional correlation, and on a broad universe that is catastrophic:
    measured on 36 names, independent shuffling took mean pairwise correlation
    from 0.41 to 0.0003 and equal-weight volatility from 18.6% to 4.9%, lifting
    the null's Sharpe to 3.1. Every diversified strategy then loses to the null by
    construction, and the p-value reports the diversification the null was handed
    rather than anything about the strategy.

    Sharing one permutation preserves contemporaneous co-movement -- names still
    crash together -- while scrambling the order of blocks, which is what destroys
    trend and cross-sectional momentum at horizons longer than a block. Measured
    on the same data, the synchronised null reproduces the real correlation (0.44
    against 0.41) and volatility (20.1% against 18.6%).
    """
    import numpy as np
    import pandas as pd

    from lab.backtest.runner import run_backtest
    from lab.engine.context import DataView
    from lab.mcp._common import evaluate

    bars, cfg, oos, stamps, metric = payload
    rng = np.random.default_rng(seed)

    wide = bars.pivot_table(index="event_time", columns="ticker", values="close").sort_index()
    n = len(wide)
    if n < block * 4:
        return None

    # One permutation of block starts, applied to every column alike.
    n_blocks = int(np.ceil((n - 1) / block))
    starts = rng.integers(0, max(1, n - 1 - block), size=n_blocks)
    order = np.concatenate([np.arange(s, s + block) for s in starts])[: n - 1]

    rets = wide.pct_change().to_numpy()[1:]          # (n-1, tickers)
    shuffled = rets[order]
    first = wide.to_numpy()[0]
    paths = first * np.cumprod(np.vstack([np.ones_like(first), 1.0 + shuffled]), axis=0)
    rebuilt = pd.DataFrame(paths, index=wide.index, columns=wide.columns)

    by_ticker: dict[str, Any] = {}
    for ticker, grp in bars.groupby("ticker"):
        frame = grp.set_index("event_time").sort_index().copy()
        name = str(ticker).upper()
        if name not in rebuilt:
            by_ticker[name] = frame
            continue
        new = rebuilt[name].reindex(frame.index)
        old = frame["close"].to_numpy(dtype="float64")
        # A ticker whose history does not span the aligned window keeps its own
        # bars rather than being silently truncated into a shorter universe.
        if new.isna().any() or (old <= 0).any():
            by_ticker[name] = frame
            continue
        scale = new.to_numpy() / old
        frame["close"] = new.to_numpy()
        for col in ("open", "high", "low", "vwap"):
            if col in frame:
                frame[col] = frame[col].to_numpy(dtype="float64") * scale
        by_ticker[name] = frame

    view = DataView(by_ticker, timeframe=getattr(cfg, "timeframe", "1d"))
    result = run_backtest(cfg, data=view, register=False, journal=False)
    ev = evaluate(result, cfg, oos_start=oos, stamps=stamps, metric=metric)
    return ev.get("score")


def deflated_sharpe(
    sharpe: float | None = None,
    bars: int | None = None,
    trials: int | None = None,
    run_id: str | None = None,
    timeframe: str = "1d",
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, Any]:
    """Correct a Sharpe ratio for how many strategies were tried to find it.

    The best of N attempts is biased upward even when none of them has an edge,
    which is why a session that tuned eight variants and kept the top one is not
    comparable to one that tested a single idea. This is the Bailey and Lopez de
    Prado deflated Sharpe: it computes the Sharpe you would expect the best of
    `trials` random strategies to post, and asks whether the observed one clears
    it.

    Give a `run_id` and the trial count is read from the registry -- every
    registered run of the same strategy, whatever its config -- along with the
    Sharpe, bar count and timeframe. That is the honest count: the tool used to
    ask the caller how many things were tried, and the caller, being the one
    who tried them, under-counted every time. Pass `trials` explicitly only to
    *raise* the number (unregistered runs, other tools); it is never lowered
    below what the registry has seen.
    """
    from statistics import NormalDist

    from lab.backtest.metrics import periods_per_year

    counted: dict[str, Any] = {}
    if run_id:
        from lab.registry.runs import RunRegistry

        reg = RunRegistry()
        record = reg.get(run_id)
        if record is None:
            raise ValueError(f"no such run: {run_id}")
        family = reg.family_attempts(record.strategy)
        same_cfg = reg.attempts(record.strategy, record.config_hash)
        # Every registered backtest that read the same holdout, whatever it
        # called itself. A research session that authored nine strategies and
        # kept one took nine looks at the test set, and the survivor's own name
        # appears on two of them.
        same_holdout = _same_holdout_runs(reg, record)
        counted = {
            "strategy": record.strategy,
            "registered_runs_of_strategy": family,
            "registered_runs_same_config": same_cfg,
            "registered_runs_same_holdout": same_holdout,
        }
        if sharpe is None:
            sharpe = (record.metrics or {}).get("sharpe")
        if bars is None:
            bars = (record.metrics or {}).get("periods") or len(_equity_returns(run_id)) + 1
        timeframe = str((record.config or {}).get("timeframe") or timeframe)
        trials = max(int(trials or 0), int(family), int(same_holdout))
    if sharpe is None or bars is None:
        raise ValueError("give sharpe and bars, or a run_id to read them from")
    if trials is None:
        raise ValueError("give trials, or a run_id so they can be counted from the registry")

    nd = NormalDist()
    n, t = int(bars), max(1, int(trials))
    ppy = int(periods_per_year(timeframe))

    # Expected maximum of `t` independent standard normals.
    euler = 0.5772156649
    if t > 1:
        e_max = (1 - euler) * nd.inv_cdf(1 - 1 / t) + euler * nd.inv_cdf(1 - 1 / (t * math.e))
    else:
        e_max = 0.0

    sr_per = float(sharpe) / math.sqrt(ppy)          # de-annualise
    sr_std = math.sqrt((1 - skew * sr_per + (kurtosis - 1) / 4 * sr_per**2) / max(1, n - 1))
    threshold = e_max * sr_std
    z = (sr_per - threshold) / sr_std if sr_std > 0 else 0.0
    p = nd.cdf(z)

    return jsonable({
        "observed_sharpe": round(float(sharpe), 4),
        "trials": t, "bars": n,
        "trials_source": "registry" if counted else "caller",
        **counted,
        "expected_max_sharpe_from_luck": round(threshold * math.sqrt(ppy), 4),
        "deflated_probability": round(p, 4),
        "verdict": (
            "clears the luck threshold" if p >= 0.95 else
            "indistinguishable from the best of this many random tries" if p < 0.5 else
            "above the luck threshold but not decisively"
        ),
        "notes": [
            f"the best of {t} strategies with no edge would be expected to post an "
            f"annualised Sharpe near {threshold * math.sqrt(ppy):.3f} on {n} bars",
            "count every configuration evaluated against the holdout, including "
            "discarded ones -- undercounting trials inflates this number",
        ],
    })


def _same_holdout_runs(reg, record) -> int:
    """Registered backtests over the same universe, dates and timeframe."""
    cfg = record.config or {}
    key = (
        tuple(sorted(str(t).upper() for t in (cfg.get("tickers") or []))),
        str(cfg.get("start") or "")[:10], str(cfg.get("end") or "")[:10],
        str(cfg.get("timeframe") or "1d"), str(cfg.get("source") or ""),
    )
    n = 0
    for r in reg.list(kind="backtest", limit=100_000):
        c = r.config or {}
        k = (
            tuple(sorted(str(t).upper() for t in (c.get("tickers") or []))),
            str(c.get("start") or "")[:10], str(c.get("end") or "")[:10],
            str(c.get("timeframe") or "1d"), str(c.get("source") or ""),
        )
        n += int(k == key)
    return n


def walk_forward(
    config: str,
    spec: str = "4:1",
    grid: dict[str, list[Any]] | None = None,
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    metric: str = "sharpe",
    max_runs: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Roll a train/test schedule across the span: does it hold in every fold?

    `spec` fixes the in-sample:out-of-sample ratio; the span is cut into equal
    blocks of *bars* and five windows slide across it, each choosing parameters
    on its own in-sample block and being scored on the next, unseen one. The OOS
    blocks are contiguous and disjoint, so their returns pool into one honest
    out-of-sample curve.

    With a `grid` this is real walk-forward selection: each window picks the
    grid point that won in-sample and reports how that pick did out-of-sample,
    and `oos_rank_of_selected` says whether the in-sample winner was anywhere
    near the out-of-sample winner. Without a grid it is a consistency check on
    a single parameter set -- weaker, but it is still five separate holdouts
    instead of one, which is the thing a single 4:1 split cannot give you.

    Every window gets the benchmark over its own out-of-sample bars, so the
    comparison is like for like.
    """
    from lab.backtest.sweep import SweepConfig, run_sweep
    from lab.store import parquet_io

    cfg = load_config(config, params=params)
    cfg = replace(cfg, strategy=resolve_strategy(strategy, cfg))
    scfg = SweepConfig(base=cfg, grid=dict(grid or {}), walk_forward=str(spec),
                       metric=metric, max_runs=max_runs)
    res = run_sweep(scfg, workers=int(workers), progress=False, journal=False)

    table = res.get("walk_forward") or []
    summary = res.get("walk_forward_summary") or {}

    # Benchmark over each window's own OOS bars.
    bench: dict[int, float | None] = {}
    if cfg.benchmark:
        try:
            bars = parquet_io.read_bars(
                [cfg.benchmark], timeframe=cfg.timeframe, source=cfg.source,
                start=cfg.start, end=cfg.end,
            )
            closes = bars.set_index("event_time")["close"].astype("float64").sort_index()
            for w in table:
                lo, hi = w.get("oos_start"), w.get("oos_end")
                seg = closes[(closes.index >= str(lo)) & (closes.index <= str(hi))]
                bench[int(w["index"])] = (
                    round(float(seg.iloc[-1] / seg.iloc[0] - 1.0), 6) if len(seg) > 1 else None
                )
        except Exception:  # noqa: BLE001 - a missing benchmark is a note, not a failure
            bench = {}

    rows = []
    for w in table:
        rows.append({
            "window": w.get("index"),
            "is": [w.get("is_start"), w.get("is_end")],
            "oos": [w.get("oos_start"), w.get("oos_end")],
            "selected": w.get("selected"),
            f"is_{metric}": w.get(f"is_{metric}"),
            f"oos_{metric}": w.get(f"oos_{metric}"),
            "oos_bars": w.get("oos_bars"),
            "oos_rank_of_selected": w.get("oos_rank_of_selected"),
            f"best_oos_{metric}": w.get(f"best_oos_{metric}"),
            "benchmark_oos_return": bench.get(int(w.get("index", -1))),
            "combinations": w.get("n_combinations"),
        })

    notes = []
    n = summary.get("windows") or 0
    pos = summary.get("positive_oos_windows")
    if n:
        notes.append(f"{metric} positive out-of-sample in {pos} of {n} windows")
    if isinstance(summary.get("degradation"), (int, float)):
        notes.append(
            f"in-sample to out-of-sample degradation of {summary['degradation']:.3f} on "
            f"{metric} -- the part of the in-sample number that was fitting"
        )
    if grid:
        ranks = [r["oos_rank_of_selected"] for r in rows if isinstance(r.get("oos_rank_of_selected"), int)]
        if ranks and max(ranks) > 1:
            notes.append(
                f"the in-sample winner ranked {ranks} out-of-sample across windows "
                f"(1 = it was also the out-of-sample winner); ranks far from 1 mean the "
                "grid was fitting each window rather than finding a parameter"
            )
    else:
        notes.append(
            "no grid: this is a consistency check on one parameter set, not a "
            "selection test -- pass a grid to see whether in-sample choice survives"
        )
    if res.get("truncated"):
        notes.append(f"grid truncated to {res.get('combinations')} of {res.get('grid_size')} points")
    return jsonable({
        "strategy": res.get("strategy"), "spec": spec, "metric": metric,
        "sweep_id": res.get("sweep_id"), "grid_size": res.get("grid_size"),
        "summary": summary, "windows": rows, "best": (res.get("best") or {}).get("params"),
        "registered_runs_of_strategy": res.get("attempts"),
        "notes": notes, "errors": res.get("errors") or [],
    })
