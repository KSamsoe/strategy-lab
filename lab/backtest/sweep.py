"""Parameter sweeps: the grid runner, and the honesty rules around it.

Four rules are wired in rather than offered (design doc §6):

1. **Out-of-sample is the fitness function.** With walk-forward on, survivors are
   ranked on the pooled OOS metric. Ranking on an in-sample number still works,
   because sometimes that is the question, but the result is stamped
   ``ranked_on="in_sample"`` so no downstream view can present it as validated.
2. **Report both, always.** Every row carries ``is_*`` and ``oos_*`` metrics side
   by side, with ``oos_*`` present-but-null when nothing was held out.
3. **Count the attempts.** The result carries the registry's attempt total for
   the strategy family, before and after. An agent is a tireless overfitter; the
   lab cannot stop it, it can refuse to hide the number of tries.
4. **No silent caps.** A grid clipped by ``max_runs`` says so in the result dict
   *and* on the log. Silent truncation reads as "we tried everything".

Every combination goes through the real event-driven runner. ``fast_screen`` is
the coarse path, and its output is a shortlist, never a result.
"""

from __future__ import annotations

import itertools
import json
import logging
import pickle
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lab.backtest.runner import BacktestConfig, run_backtest
from lab.backtest.walkforward import Window, make_windows, split_metrics, window_metrics
from lab.engine.events import new_id
from lab.engine.loader import load_strategy
from lab.timeutil import to_utc, utcnow

_LOG = logging.getLogger(__name__)

#: Metrics where a smaller number is the better one. ``max_drawdown`` is not
#: here: it is stored as a negative fraction, so plain maximization is correct.
LOWER_IS_BETTER = frozenset(
    {"volatility", "downside_deviation", "max_drawdown_duration_days", "turnover", "beta"}
)

#: Knobs the built-in coarse screener understands. Anything else in the grid is
#: invisible to it, which the returned frame states outright.
_SCREEN_KNOBS = ("lookback", "top_n", "trend_ma", "rebalance_days")

_SCREEN_CAVEAT = (
    "COARSE SCREEN ONLY -- vectorized proxy, not a backtest: no risk gate, no "
    "fills, no commissions, no position management, one-bar signal shift instead "
    "of the engine's knowledge-time barrier. Every survivor must be re-run "
    "through the event-driven engine (run_sweep) before it is believed."
)


@dataclass
class SweepConfig:
    base: BacktestConfig
    grid: dict[str, list[Any]] = field(default_factory=dict)
    walk_forward: str | None = None
    metric: str = "sharpe"
    max_runs: int | None = None
    fast: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base, BacktestConfig):
            raise ValueError("SweepConfig.base must be a BacktestConfig")
        self.grid = {str(k): _axis(k, v) for k, v in dict(self.grid or {}).items()}
        if self.max_runs is not None:
            self.max_runs = int(self.max_runs)
            if self.max_runs < 1:
                raise ValueError(f"max_runs must be >= 1, got {self.max_runs}")
        if self.walk_forward is not None:
            self.walk_forward = str(self.walk_forward)
        if not str(self.metric or "").strip():
            raise ValueError("metric must be a non-empty metric name, e.g. 'sharpe'")

    @classmethod
    def from_yaml(
        cls, path: str | Path, base: BacktestConfig | str | Path | None = None
    ) -> "SweepConfig":
        """Load a grid file. ``base:`` inside it may be a path or an inline
        config mapping; an explicit ``base`` argument wins over both."""
        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no sweep config at {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{p} does not contain a mapping")
        raw = dict(raw)

        declared = raw.pop("base", None)
        resolved = base if base is not None else declared
        if resolved is None:
            raise ValueError(
                f"{p} has no `base:` config and none was passed; a grid needs a "
                f"backtest config to vary"
            )
        if isinstance(resolved, BacktestConfig):
            base_cfg = resolved
        elif isinstance(resolved, Mapping):
            base_cfg = BacktestConfig.from_mapping(resolved)
        else:
            base_cfg = BacktestConfig.from_yaml(_resolve_base_path(resolved, p))

        known = {f for f in cls.__dataclass_fields__} - {"base"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown sweep config keys: {', '.join(sorted(unknown))}; "
                f"known keys are base, {', '.join(sorted(known))}"
            )
        return cls(base=base_cfg, **{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "grid": self.grid,
            "walk_forward": self.walk_forward,
            "metric": self.metric,
            "max_runs": self.max_runs,
            "fast": self.fast,
        }


def _resolve_base_path(value: str | Path, grid_path: Path) -> Path:
    """A grid file naming ``cfg/momo.yaml`` should work from anywhere, so try
    the grid's own directory before the process working directory."""
    candidate = Path(value)
    options = [candidate]
    if not candidate.is_absolute():
        options = [grid_path.parent / candidate, candidate]
        try:
            from lab.config import get_settings

            options.append(get_settings().paths.cfg / candidate.name)
            options.append(get_settings().paths.root / candidate)
        except Exception:  # config is frozen and cheap, but never fail the load here
            pass
    for opt in options:
        if opt.exists():
            return opt
    raise FileNotFoundError(f"no base config at {value!r} (tried {[str(o) for o in options]})")


def _axis(key: str, values: Any) -> list[Any]:
    if isinstance(values, (list, tuple, set)):
        axis = list(values)
    elif isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        axis = [values]
    else:
        axis = list(values)
    if not axis:
        raise ValueError(f"grid axis {key!r} is empty; drop the key or give it values")
    return axis


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the axes, last key varying fastest.

    An empty grid expands to one empty combination -- "sweep of nothing" is the
    base config run once, which is what a caller with an empty grid means.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    axes = [_axis(k, grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


# --- the worker ---------------------------------------------------------------


def _run_combo(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One grid point, start to finish. Module-level and dict-in/dict-out so the
    serial path and the process-pool path are literally the same code."""
    cfg: BacktestConfig = payload["config"]
    params: dict[str, Any] = dict(payload["params"])
    windows: list[Window] = list(payload["windows"])

    row: dict[str, Any] = {
        "params": params,
        "run_id": None,
        "status": "ok",
        "error": "",
        "attempt": 0,
        "duration_s": 0.0,
        "metrics": {},
        "windows": [],
    }
    try:
        res = run_backtest(
            cfg, register=payload.get("register", True), journal=payload.get("journal", False)
        )
    except Exception as exc:  # one bad grid point must not sink the sweep
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["run_id"] = res.run_id
    row["attempt"] = res.attempt
    row["duration_s"] = round(res.duration_s, 3)
    row["metrics"] = res.metrics
    row["warnings"] = list(res.warnings)

    tf = cfg.timeframe
    if windows:
        is_m, oos_m = split_metrics(res.equity, res.trades, windows, timeframe=tf)
        row["windows"] = window_metrics(res.equity, res.trades, windows, timeframe=tf)
    else:
        # Nothing was held out, so the whole run is in-sample and the oos_* half
        # of the row is null rather than absent -- a missing key invites a
        # downstream `.get(k, is_value)` fallback, a null does not.
        is_m, oos_m = dict(res.metrics), {}
    row |= _prefixed(is_m, "is_")
    row |= _prefixed(oos_m, "oos_", null_keys=is_m if not oos_m else None)
    return row


def _prefixed(
    m: Mapping[str, Any], prefix: str, *, null_keys: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if null_keys is not None:
        return {f"{prefix}{k}": None for k, v in null_keys.items() if _scalar(v)}
    return {f"{prefix}{k}": v for k, v in m.items() if _scalar(v)}


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (int, float, str, bool))


# --- the sweep ----------------------------------------------------------------


def run_sweep(
    cfg: SweepConfig,
    *,
    workers: int = 1,
    progress: bool = False,
    journal: bool = False,
) -> dict[str, Any]:
    """Run every grid point through the event-driven engine and rank the result.

    ``workers=1`` is the default and the reference path: a sweep that is fast and
    wrong is worthless, so the pool exists only to run the same ``_run_combo``
    more times at once, and degrades to serial the moment it cannot.

    ``journal=False`` because a grid of runs would bury the decision journal
    under tens of thousands of rows; each run still writes its own
    ``runs/<run_id>/decisions.jsonl``, so nothing is lost.
    """
    started = _time.perf_counter()
    base = cfg.base
    notes: list[str] = []

    combos_all = expand_grid(cfg.grid)
    grid_size = len(combos_all)
    combos, selection = _select(cfg, combos_all, notes)

    loaded = load_strategy(base.strategy, base.params)
    strategy_name = loaded.name

    windows: list[Window] = []
    span: tuple[datetime, datetime] | None = None
    if cfg.walk_forward:
        span = _resolve_span(base)
        windows = make_windows(span[0], span[1], cfg.walk_forward, timeframe=base.timeframe)

    sweep_id = _new_sweep_id(strategy_name)
    from lab.registry.runs import RunRegistry

    registry = RunRegistry()
    attempts_before = registry.family_attempts(strategy_name)

    payloads = [
        {
            "config": _combo_config(base, params, sweep_id=sweep_id, windows=windows, span=span),
            "params": params,
            "windows": windows,
            "register": True,
            "journal": journal,
        }
        for params in combos
    ]

    rows = _execute(payloads, workers=workers, progress=progress, notes=notes)

    metric, ranked_on, score_key = _ranking(cfg.metric, bool(windows), notes)
    _rank_rows(rows, score_key, metric)
    ok_rows = [r for r in rows if r["status"] == "ok" and r.get("score") is not None]
    best = ok_rows[0] if ok_rows else None

    wf_table = _walk_forward_table(rows, windows, metric)
    result: dict[str, Any] = {
        "sweep_id": sweep_id,
        "strategy": strategy_name,
        "runs": rows,
        "best": best,
        "grid": dict(cfg.grid),
        "attempts": registry.family_attempts(strategy_name),
        "walk_forward": wf_table,
        # --- provenance and the honesty flags -------------------------------
        "attempts_before": attempts_before,
        "grid_size": grid_size,
        "combinations": len(combos),
        "truncated": len(combos) < grid_size,
        "max_runs": cfg.max_runs,
        "selection": selection,
        "metric": metric,
        "ranked_on": ranked_on,
        "ranked_on_metric": score_key,
        "validated_out_of_sample": ranked_on == "out_of_sample",
        "walk_forward_spec": cfg.walk_forward,
        "walk_forward_summary": _walk_forward_summary(wf_table, metric),
        "windows": [w.to_dict() for w in windows],
        "start": span[0].isoformat() if span else None,
        "end": span[1].isoformat() if span else None,
        "workers": workers,
        "errors": [
            {"params": r["params"], "error": r["error"]} for r in rows if r["status"] == "error"
        ],
        "notes": notes,
        "created_at": utcnow().isoformat(),
        "duration_s": round(_time.perf_counter() - started, 3),
    }
    result["artifact_dir"] = str(_write_sweep_artifact(sweep_id, result))
    return result


def _select(
    cfg: SweepConfig, combos: list[dict[str, Any]], notes: list[str]
) -> tuple[list[dict[str, Any]], str]:
    """Apply ``max_runs``, loudly. Returns the combinations to run and how they
    were chosen."""
    if cfg.max_runs is None or len(combos) <= cfg.max_runs:
        return combos, "full_grid"

    keep = cfg.max_runs
    selection = "grid_order"
    chosen = combos[:keep]
    if cfg.fast:
        try:
            screened = fast_screen(cfg)
            order = [row["params"] for row in screened.to_dict("records")]
            chosen = [p for p in order if p in combos][:keep]
            selection = "fast_screen"
        except Exception as exc:
            notes.append(
                f"fast=True but the coarse screen failed ({type(exc).__name__}: {exc}); "
                f"fell back to grid order for the {keep} combinations run."
            )
            _LOG.warning("fast_screen failed, using grid order: %s", exc)
            chosen = combos[:keep]

    msg = (
        f"GRID TRUNCATED: {len(combos)} combinations requested, {len(chosen)} run "
        f"(max_runs={cfg.max_runs}, selection={selection}). "
        f"{len(combos) - len(chosen)} combinations were never tested -- this sweep "
        f"does not say they are worse, only that nobody looked."
    )
    _LOG.warning(msg)
    notes.append(msg)
    return chosen, selection


def _ranking(metric: str, has_windows: bool, notes: list[str]) -> tuple[str, str, str]:
    """Resolve ``(base_metric, ranked_on, score_key)``.

    An explicit ``is_``/``oos_`` prefix on ``cfg.metric`` is honoured; a bare name
    means "the validated one if there is one".
    """
    m = str(metric).strip()
    explicit = True
    if m.startswith("oos_"):
        base, ranked_on = m[4:], "out_of_sample"
    elif m.startswith("is_"):
        base, ranked_on = m[3:], "in_sample"
    else:
        explicit = False
        base, ranked_on = m, ("out_of_sample" if has_windows else "in_sample")

    if ranked_on == "out_of_sample" and not has_windows:
        ranked_on = "in_sample"
        notes.append(
            f"ranked on the IN-SAMPLE {base}: {metric!r} was asked for but no "
            f"walk_forward spec was given, so nothing was held out."
        )
    elif ranked_on == "in_sample":
        why = (
            f"{metric!r} was asked for explicitly"
            if explicit
            else "no walk_forward spec was given, so nothing was held out"
        )
        notes.append(
            f"ranked on the IN-SAMPLE {base}: {why}"
            + (" even though walk-forward windows exist" if explicit and has_windows else "")
            + ". These numbers are not validated out of sample."
        )
    return base, ranked_on, f"{'oos' if ranked_on == 'out_of_sample' else 'is'}_{base}"


def _rank_rows(rows: list[dict[str, Any]], score_key: str, metric: str) -> None:
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    for r in rows:
        raw = r.get(score_key)
        r["score"] = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        r["score_key"] = score_key
    rows.sort(key=lambda r: (r["score"] is None, -(sign * (r["score"] or 0.0))))
    for i, r in enumerate(rows):
        r["rank"] = i + 1 if r["score"] is not None else None


def _combo_config(
    base: BacktestConfig,
    params: Mapping[str, Any],
    *,
    sweep_id: str,
    windows: Sequence[Window],
    span: tuple[datetime, datetime] | None,
) -> BacktestConfig:
    cfg = BacktestConfig.from_mapping(base.to_dict() | {"start": None, "end": None})
    cfg.params = dict(base.params) | dict(params)
    cfg.start = span[0] if span else base.start
    cfg.end = span[1] if span else base.end
    cfg.sweep_id = sweep_id
    cfg.windows = [w.to_dict() for w in windows]
    return cfg


def _resolve_span(base: BacktestConfig) -> tuple[datetime, datetime]:
    if base.start is not None and base.end is not None:
        return base.start, base.end
    from lab.store import parquet_io

    df = parquet_io.read_bars(
        base.tickers or None,
        timeframe=base.timeframe,
        start=base.start,
        end=base.end,
        source=(base.sources[0] if base.sources else None),
    )
    if len(df) == 0:
        raise ValueError(
            "walk-forward needs a date range: the config has no start/end and the "
            "store has no bars for this universe -- run `lab pull` first"
        )
    return to_utc(df["event_time"].min()), to_utc(df["event_time"].max())


def _new_sweep_id(strategy: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in strategy).strip("_").lower()[:24] or "sweep"
    return f"{slug}-sweep-{utcnow().strftime('%Y%m%dT%H%M%S')}-{new_id()[:6]}"


def _write_sweep_artifact(sweep_id: str, result: Mapping[str, Any]) -> Path:
    from lab.registry.runs import run_artifact_dir

    d = run_artifact_dir(sweep_id)
    (d / "sweep.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return d


# --- execution ----------------------------------------------------------------


def _execute(
    payloads: Sequence[Mapping[str, Any]], *, workers: int, progress: bool, notes: list[str]
) -> list[dict[str, Any]]:
    n = len(payloads)
    results: list[dict[str, Any] | None] = [None] * n

    if workers > 1:
        try:
            pickle.dumps(list(payloads))
        except Exception as exc:
            msg = (
                f"workers={workers} requested but the sweep payload will not pickle "
                f"({type(exc).__name__}: {exc}); ran serially instead."
            )
            _LOG.warning(msg)
            notes.append(msg)
            workers = 1

    if workers > 1:
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_combo, p): i for i, p in enumerate(payloads)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as exc:
                        results[i] = _error_row(payloads[i], exc)
                    done += 1
                    if progress:
                        _progress(done, n, results[i])
        except Exception as exc:
            msg = (
                f"process pool failed after {done}/{n} combinations "
                f"({type(exc).__name__}: {exc}); the rest ran serially."
            )
            _LOG.warning(msg)
            notes.append(msg)

    remaining = [i for i, r in enumerate(results) if r is None]
    for done, i in enumerate(remaining, start=1):
        try:
            results[i] = _run_combo(payloads[i])
        except Exception as exc:  # _run_combo already traps; this is belt and braces
            results[i] = _error_row(payloads[i], exc)
        if progress:
            _progress(done + (n - len(remaining)), n, results[i])
    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    return [r for r in results if r is not None]


def _error_row(payload: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "params": dict(payload["params"]),
        "run_id": None,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "attempt": 0,
        "duration_s": 0.0,
        "metrics": {},
        "windows": [],
    }


def _progress(done: int, total: int, row: Mapping[str, Any] | None) -> None:
    tail = ""
    if row:
        tail = f"  {row.get('status', '')} {row.get('run_id') or ''}"
    sys.stderr.write(f"\r  sweep {done}/{total}{tail:<48}")
    sys.stderr.flush()


# --- walk-forward assembly ----------------------------------------------------


def _walk_forward_table(
    rows: Sequence[Mapping[str, Any]], windows: Sequence[Window], metric: str
) -> list[dict[str, Any]]:
    """Per-window evidence: what the in-sample choice was, and what it then did
    out of sample. The gap between those two columns is the overfitting."""
    if not windows:
        return []
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    scored = [r for r in rows if r["status"] == "ok" and r.get("windows")]
    table: list[dict[str, Any]] = []

    for w in windows:
        entries: list[dict[str, Any]] = []
        for r in scored:
            per = next((x for x in r["windows"] if x["index"] == w.index), None)
            if per is None:
                continue
            entries.append(
                {
                    "params": r["params"],
                    "run_id": r["run_id"],
                    f"is_{metric}": per["is"].get(metric),
                    f"oos_{metric}": per["oos"].get(metric),
                    "is_bars": per["is"].get("bars", 0),
                    "oos_bars": per["oos"].get("bars", 0),
                }
            )
        entry = w.to_dict() | {"n_combinations": len(entries), "scores": entries}
        usable = [e for e in entries if isinstance(e[f"is_{metric}"], (int, float))]
        if usable:
            pick = max(usable, key=lambda e: sign * float(e[f"is_{metric}"]))
            by_oos = sorted(
                (e for e in usable if isinstance(e[f"oos_{metric}"], (int, float))),
                key=lambda e: -sign * float(e[f"oos_{metric}"]),
            )
            entry |= {
                "selected": pick["params"],
                "selected_run_id": pick["run_id"],
                f"is_{metric}": pick[f"is_{metric}"],
                f"oos_{metric}": pick[f"oos_{metric}"],
                "is_bars": pick["is_bars"],
                "oos_bars": pick["oos_bars"],
                # Where the in-sample winner actually landed once the data was
                # unseen. A rank far from 1 means the grid fitted the window.
                "oos_rank_of_selected": (
                    next(
                        (i + 1 for i, e in enumerate(by_oos) if e["run_id"] == pick["run_id"]),
                        None,
                    )
                ),
                f"best_oos_{metric}": (by_oos[0][f"oos_{metric}"] if by_oos else None),
            }
        table.append(entry)
    return table


def _walk_forward_summary(table: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    if not table:
        return {}
    picks_is = [w.get(f"is_{metric}") for w in table]
    picks_oos = [w.get(f"oos_{metric}") for w in table]
    num_is = [float(v) for v in picks_is if isinstance(v, (int, float))]
    num_oos = [float(v) for v in picks_oos if isinstance(v, (int, float))]
    out: dict[str, Any] = {
        "windows": len(table),
        "metric": metric,
        f"mean_is_{metric}": round(float(np.mean(num_is)), 6) if num_is else None,
        f"mean_oos_{metric}": round(float(np.mean(num_oos)), 6) if num_oos else None,
        "positive_oos_windows": sum(1 for v in num_oos if v > 0),
    }
    if num_is and num_oos:
        # The honest headline: what the in-sample selection kept once it was
        # asked to perform on data it had never been chosen against.
        out["degradation"] = round(float(np.mean(num_is) - np.mean(num_oos)), 6)
        out["consistency"] = round(sum(1 for v in num_oos if v > 0) / len(num_oos), 4)
    return out


# --- the coarse screen --------------------------------------------------------


def fast_screen(cfg: SweepConfig) -> pd.DataFrame:
    """Rank the whole grid with a vectorized proxy, in seconds.

    THIS IS NOT A BACKTEST AND ITS OUTPUT IS NOT A RESULT. It approximates a
    long-only equal-weight top-N rotation straight off the close matrix: no risk
    gate, no fill model, no commissions, no position management, and a one-bar
    signal shift standing in for the engine's knowledge-time barrier. Its job is
    to turn a 5,000-point grid into a shortlist. Every survivor is re-run through
    the event-driven engine (``run_sweep``) before it is believed -- the returned
    frame says so on every row, in ``confirmed`` and ``caveat``.

    Uses ``vectorbt`` when it is importable and a built-in numpy screener when it
    is not, which it usually is not. The columns are identical either way.
    """
    combos = expand_grid(cfg.grid)
    closes = _screen_closes(cfg.base)
    knobs = [k for k in _SCREEN_KNOBS if k in cfg.grid]

    engine, scores = _screen_with_vectorbt(closes, combos, cfg.base)
    if scores is None:
        engine, scores = "numpy", [_screen_numpy(closes, cfg.base.params | c) for c in combos]

    rows: list[dict[str, Any]] = []
    for params, s in zip(combos, scores):
        rows.append(
            {"params": params}
            | {f"param.{k}": v for k, v in params.items()}
            | s
            | {"engine": engine, "confirmed": False, "caveat": _SCREEN_CAVEAT}
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("screen_sharpe", ascending=False, kind="stable").reset_index(drop=True)
        df.insert(0, "screen_rank", np.arange(1, len(df) + 1))
    df.attrs |= {
        "caveat": _SCREEN_CAVEAT,
        "engine": engine,
        "confirmed": False,
        "tickers": list(closes.columns),
        "bars": int(len(closes)),
        # A grid whose axes the screener cannot see produces identical scores;
        # saying so beats letting someone read tie-broken noise as a ranking.
        "discriminates": bool(knobs),
        "screened_knobs": knobs,
        "ignored_knobs": [k for k in cfg.grid if k not in knobs],
        "created_at": utcnow().isoformat(),
    }
    if not knobs and cfg.grid:
        _LOG.warning(
            "fast_screen understands none of the grid axes %s; every row scores the "
            "same and the ranking is meaningless",
            sorted(cfg.grid),
        )
    return df


def _screen_closes(base: BacktestConfig) -> pd.DataFrame:
    from lab.store import parquet_io

    df = parquet_io.read_bars(
        base.tickers or None,
        timeframe=base.timeframe,
        start=base.start,
        end=base.end,
        source=(base.sources[0] if base.sources else None),
    )
    if len(df) == 0:
        raise ValueError("fast_screen found no bars for this universe -- run `lab pull` first")
    wide = df.pivot_table(index="event_time", columns="ticker", values="close", aggfunc="last")
    return wide.sort_index().astype("float64")


def _screen_with_vectorbt(
    closes: pd.DataFrame, combos: Sequence[Mapping[str, Any]], base: BacktestConfig
) -> tuple[str, list[dict[str, Any]] | None]:
    """vectorbt if it is installed; ``(engine, None)`` to hand back to numpy."""
    try:
        import vectorbt as vbt  # noqa: F401  (lazy: optional dependency)
    except Exception as exc:
        _LOG.debug("vectorbt unavailable (%s); using the numpy screener", exc)
        return "numpy", None

    out: list[dict[str, Any]] = []
    try:
        for combo in combos:
            weights = _screen_weights(closes, dict(base.params) | dict(combo))
            pf = vbt.Portfolio.from_orders(
                close=closes,
                size=weights.shift(1).fillna(0.0),
                size_type="targetpercent",
                group_by=True,
                cash_sharing=True,
                init_cash=float(base.cash),
                freq=base.timeframe,
            )
            rets = pf.returns()
            out.append(_screen_stats(rets, base.timeframe))
    except Exception as exc:
        _LOG.warning("vectorbt screen failed (%s); falling back to the numpy screener", exc)
        return "numpy", None
    return "vectorbt", out


def _screen_weights(closes: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Target weights of the proxy rule: hold the top-N momentum names that are
    above their trend filter, equally weighted."""
    lookback = max(int(params.get("lookback", 63) or 63), 1)
    top_n = max(int(params.get("top_n", 3) or 3), 1)
    trend_ma = int(params.get("trend_ma", 0) or 0)
    rebalance = int(params.get("rebalance_days", 0) or 0)

    mom = closes.pct_change(lookback, fill_method=None)
    picked = mom.rank(axis=1, ascending=False, method="first") <= top_n
    picked &= mom.notna()
    if trend_ma > 1:
        picked &= closes > closes.rolling(trend_ma, min_periods=trend_ma).mean()

    counts = picked.sum(axis=1)
    weights = picked.astype("float64").div(counts.where(counts > 0), axis=0).fillna(0.0)
    if rebalance > 1:
        # Hold the last rebalance's weights in between, the coarse analogue of
        # the engine's "only trade when the strategy says so".
        due = (np.arange(len(weights)) % rebalance == 0)[:, None]
        weights = weights.where(np.broadcast_to(due, weights.shape)).ffill().fillna(0.0)
    return weights


def _screen_numpy(closes: pd.DataFrame, params: Mapping[str, Any]) -> dict[str, Any]:
    weights = _screen_weights(closes, params)
    # shift(1): decide on bar t's close, earn bar t -> t+1. The single line that
    # keeps the screen from being pure look-ahead.
    rets = (weights.shift(1) * closes.pct_change(fill_method=None)).sum(axis=1)
    return _screen_stats(rets, "1d")


def _screen_stats(rets: pd.Series, timeframe: str) -> dict[str, Any]:
    from lab.backtest.metrics import periods_per_year

    r = pd.Series(rets).astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ppy = periods_per_year(timeframe if timeframe else "1d")
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    curve = (1.0 + r).cumprod()
    dd = float((curve / curve.cummax() - 1.0).min()) if len(curve) else 0.0
    return {
        "screen_sharpe": round(float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0, 6),
        "screen_total_return": round(float(curve.iloc[-1] - 1.0) if len(curve) else 0.0, 6),
        "screen_max_drawdown": round(dd, 6),
        "screen_bars": int(len(r)),
    }
