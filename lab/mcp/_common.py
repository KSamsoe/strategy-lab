"""Shared plumbing for the MCP tools, and the guards every result carries.

The design rule here is the lesson of every audit this lab has been through: a
number handed over without its caveat gets used without it. Five research
sessions each produced a confident wrong conclusion, and in every case the
correcting fact was already computable at the moment the result was returned --
the benchmark measured over a different window, the exposure that made a Sharpe
look good, the risk gate quietly rewriting the strategy, the error bar wider than
the difference being ranked on.

So ``evaluate()`` returns the metrics *and* the things that would embarrass them,
and ``warnings_for()`` states the embarrassing ones in words. A caller that reads
only the headline still cannot claim it was not told.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Metrics worth returning from any window. Deliberately includes `periods` (the
#: bar count) and `exposure`, because those are what turn a score into a score
#: with an error bar and a score with a caveat.
WINDOW_KEYS = (
    "total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
    "volatility", "exposure", "turnover", "trades", "hit_rate", "profit_factor",
    "periods",
)


def jsonable(value: Any) -> Any:
    """numpy/pandas scalars and timestamps into things ``json`` will accept."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    for attr in ("item", "isoformat"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return jsonable(fn())
            except Exception:  # noqa: BLE001 - fall through to str
                break
    return str(value)


def load_config(config: str, params: Mapping[str, Any] | None = None, **over: Any):
    """A ``BacktestConfig`` from a YAML path, with overrides applied."""
    from lab.backtest.runner import BacktestConfig

    cfg = BacktestConfig.from_yaml(Path(config))
    changes: dict[str, Any] = {k: v for k, v in over.items() if v is not None}
    if params is not None:
        changes["params"] = dict(params)
    return replace(cfg, **changes) if changes else cfg


def resolve_strategy(strategy: str | None, cfg: Any) -> Path:
    """A strategy path from a bare name, a library name, or an explicit path."""
    if not strategy:
        return Path(cfg.strategy)
    p = Path(strategy)
    if p.exists():
        return p
    from lab.config import get_settings

    name = p.name if p.suffix == ".py" else f"{p.name}.py"
    candidate = Path(get_settings().paths.strategies) / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no strategy {strategy!r}; give a path, or a name in "
        f"{get_settings().paths.strategies}"
    )


def gate_for(run_id: str) -> dict[str, Any]:
    """How much of what the strategy asked for actually reached the book."""
    try:
        from lab.agent.review import gate_activity
        from lab.backtest.runner import artifact_dir

        return dict(gate_activity(artifact_dir(run_id)) or {})
    except Exception:  # noqa: BLE001 - a missing journal is not a failed backtest
        return {}


def split_point(cfg: Any, oos_split: str = "4:1") -> tuple[Any, list[Any]]:
    """``(oos_start, tradeable_timestamps)`` -- warmup already excluded."""
    from lab.analysis import split_point as split_bars
    from lab.analysis import tradeable_timestamps

    stamps = tradeable_timestamps(cfg, None)
    if len(stamps) < 4:
        raise ValueError(
            f"only {len(stamps)} tradeable bars for this config; pull data first"
        )
    _, oos = split_bars(stamps, oos_split)
    return oos, stamps


def evaluate(
    result: Any,
    cfg: Any,
    *,
    oos_start: Any,
    stamps: Sequence[Any],
    periods: int = 4,
    metric: str = "oos_sharpe",
    market: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything worth knowing about one finished run, caveats included."""
    from lab.analysis import (
        fitness_one_sigma,
        market_reference,
        period_breakdown,
        window_metrics,
    )

    tf = getattr(cfg, "timeframe", "1d")
    windows = {
        "full": window_metrics(result, tf),
        "in_sample": window_metrics(result, tf, hi=oos_start, inclusive_hi=False),
        "oos": window_metrics(result, tf, lo=oos_start),
    }
    if market is None:
        market = market_reference(cfg, oos_start, stamps[0] if stamps else None)

    by_period = period_breakdown(result, tf, int(periods))
    sharpes = [p.get("sharpe") for p in by_period if isinstance(p.get("sharpe"), (int, float))]

    base = metric.split("_", 1)[-1] if "_" in metric else metric
    key = "in_sample" if metric.startswith("is_") else "full" if metric.startswith("full_") else "oos"
    window = windows.get(key) or {}
    score = window.get(base)
    sigma = fitness_one_sigma(metric, window, tf, int(window.get("periods") or 0))

    out: dict[str, Any] = {
        "run_id": result.run_id,
        "strategy": Path(str(cfg.strategy)).name,
        "params": dict(getattr(cfg, "params", {}) or {}),
        "metric": metric,
        "score": score,
        "score_one_sigma": sigma,
        "windows": {k: {m: v.get(m) for m in WINDOW_KEYS} for k, v in windows.items()},
        "market": market,
        "by_period": by_period,
        "worst_period_sharpe": min(sharpes) if sharpes else None,
        "gate": gate_for(result.run_id),
        "ledger_residual": (result.metrics or {}).get("ledger_residual"),
    }
    out["vs_market"] = _vs_market(windows, market)
    out["warnings"] = warnings_for(out)
    return jsonable(out)


def _vs_market(windows: Mapping[str, Any], market: Mapping[str, Any] | None) -> dict[str, Any]:
    """Excess return per window, each side measured over the same bars."""
    out: dict[str, Any] = {}
    for key in ("full", "oos", "in_sample"):
        mine = (windows.get(key) or {}).get("total_return")
        theirs = ((market or {}).get(key) or {}).get("total_return")
        if isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
            out[f"{key}_return"] = round(mine - theirs, 6)
        mine_s = (windows.get(key) or {}).get("sharpe")
        theirs_s = ((market or {}).get(key) or {}).get("sharpe")
        if isinstance(mine_s, (int, float)) and isinstance(theirs_s, (int, float)):
            out[f"{key}_sharpe"] = round(mine_s - theirs_s, 6)
    return out


# Thresholds live in lab.analysis so the research loop and this layer cannot
# drift apart -- they were duplicated, and two copies that happen to agree is the
# state a constant is in immediately before it stops agreeing.
from lab.analysis import CLIPPING_CONFOUNDS, CONCENTRATION, LOW_EXPOSURE  # noqa: E402


def warnings_for(evaluated: Mapping[str, Any]) -> list[str]:
    """The caveats, in words, attached to the numbers they qualify."""
    out: list[str] = []
    full = evaluated.get("windows", {}).get("full") or {}
    oos = evaluated.get("windows", {}).get("oos") or {}
    vs = evaluated.get("vs_market") or {}

    clipped = (evaluated.get("gate") or {}).get("share_clipped_or_blocked")
    if isinstance(clipped, (int, float)) and clipped >= CLIPPING_CONFOUNDS:
        rules = ", ".join(
            str(r.get("rule")) for r in ((evaluated.get("gate") or {}).get("top_rules") or [])[:2]
        )
        out.append(
            f"gate clipped or blocked {clipped:.0%} of intents"
            + (f" (mostly {rules})" if rules else "")
            + " -- what was backtested is the gate's version of this strategy, and "
            "tuning its parameters is tuning the wrong thing"
        )

    expo = oos.get("exposure")
    if isinstance(expo, (int, float)) and expo < LOW_EXPOSURE:
        out.append(
            f"out-of-sample exposure is {expo:.0%} -- a ratio earned while mostly in "
            "cash is de-levering, not skill; compare on return as well"
        )

    score, sigma = evaluated.get("score"), evaluated.get("score_one_sigma")
    market_score = ((evaluated.get("market") or {}).get("oos") or {}).get("sharpe")
    if all(isinstance(v, (int, float)) for v in (score, sigma, market_score)):
        if abs(score - market_score) < sigma:
            out.append(
                f"score {score:.3f} is within one standard error ({sigma:.3f}) of the "
                f"benchmark's {market_score:.3f} -- the holdout cannot separate them, "
                "so decide on full-period behaviour, drawdown and consistency"
            )

    fr, orr = vs.get("full_return"), vs.get("oos_return")
    if isinstance(fr, (int, float)) and isinstance(orr, (int, float)) and fr > 0 > orr:
        out.append(
            f"beats the benchmark by {fr:+.2f} over the full period but trails by "
            f"{orr:+.2f} out of sample -- likely out of favour rather than broken, "
            "but say which you think it is"
        )

    resid = evaluated.get("ledger_residual")
    if isinstance(resid, (int, float)) and abs(resid) > 0.01:
        out.append(f"trade ledger does not reconcile (residual {resid:.2f})")

    if not (full.get("trades") or 0):
        out.append("no trades were taken -- this measures nothing")
    return out


def concentration_warning(review: Mapping[str, Any]) -> list[str]:
    """Attribution caveats, for tools that have a review digest to hand."""
    out: list[str] = []
    top = (review.get("concentration") or {}).get("top_ticker_share")
    if isinstance(top, (int, float)) and top >= CONCENTRATION:
        name = (review.get("concentration") or {}).get("top_ticker")
        out.append(
            f"{name} carries {top:.0%} of gross profit -- one name carrying the result "
            "is the most common way a backtest turns out not to be one"
        )
    return out


def chunk(items: Iterable[Any], size: int) -> list[list[Any]]:
    out, cur = [], []
    for x in items:
        cur.append(x)
        if len(cur) >= size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out
