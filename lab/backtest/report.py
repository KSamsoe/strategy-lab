"""The self-contained HTML backtest report (design doc §6).

One file, no network. Inline CSS, hand-built inline SVG, no CDN, no JavaScript,
no webfont fetch -- because a report's whole job is to survive being emailed,
archived, and opened off a disk in five years, and anything fetched at render
time is something that can 404 or silently change what the reader sees. The
console supersedes this for interactive work and links here for archival.

Palette and type follow §7 of the console design doc, so a report and the
console read as the same instrument: colour is reserved for meaning (P&L
direction, warning state, selection) and never spent on decoration.

Two things the charts must do beyond plotting: shade out-of-sample spans so OOS
performance cannot be skimmed past, and carry the standing caveats -- optimistic
fills, training-data contamination, survivorship bias -- next to the numbers
they qualify rather than in a footnote nobody opens.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd

from lab.backtest import metrics as M

if TYPE_CHECKING:  # runner imports nothing from here; keep it that way
    from lab.backtest.runner import BacktestResult

#: Beyond this the ledger stops being readable and starts being a data dump;
#: trades.csv next to the report is the complete record.
MAX_LEDGER_ROWS = 400

#: At most this many plotted vertices per series. Bucketed min/max, so a spike
#: is never smoothed away by the thinning.
MAX_CHART_POINTS = 1400

#: These strings are asserted on by tests and read by the console, so they are
#: module constants rather than inline literals.
OPTIMISTIC_WARNING = (
    "OPTIMISTIC FILLS — this run filled at the same bar's close, so a "
    "decision traded at a price it helped set. Every number below is flattered. "
    "Treat them as an upper bound, not a result."
)
CONTAMINATION_WARNING = (
    "TRAINING-DATA CONTAMINATION — this strategy consults a language model "
    "whose training data covers the backtest window, so the model may be "
    "recalling the period rather than reasoning about it. This is not an "
    "out-of-sample result; real evaluation is forward-only on paper."
)
SURVIVORSHIP_NOTE = (
    "Survivorship bias — the universe is today's ticker list, not a "
    "point-in-time universe. Names that were delisted, acquired, or went to zero "
    "never appear, which biases returns upward. A point-in-time universe is a "
    "known v2 item and this note stands until it lands."
)

_DERIVED_CONFIG_KEYS = ("resolved_params", "strategy_hash")

_PCT_KEYS = frozenset(
    {
        "total_return", "cagr", "hit_rate", "exposure", "max_drawdown",
        "volatility", "downside_deviation", "excess_return", "alpha",
        "benchmark_total_return", "benchmark_cagr", "best_period",
        "worst_period",
    }
)
_MONEY_KEYS = frozenset(
    {
        "final_equity", "peak_equity", "starting_equity", "avg_win", "avg_loss",
        "avg_trade_pnl", "best_trade", "worst_trade", "gross_profit",
        "gross_loss", "commission", "expectancy",
    }
)
_INT_KEYS = frozenset(
    {"trades", "wins", "losses", "periods", "periods_per_year", "attempt", "open_positions"}
)
#: Keys whose sign carries P&L meaning and therefore earn a colour.
_SIGNED_KEYS = frozenset(
    {
        "total_return", "cagr", "excess_return", "alpha", "max_drawdown",
        "avg_win", "avg_loss", "avg_trade_pnl", "best_trade", "worst_trade",
        "expectancy", "best_period", "worst_period", "sharpe", "sortino",
        "calmar", "benchmark_total_return", "benchmark_cagr",
    }
)

_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "returns",
        (
            "total_return", "cagr", "excess_return", "benchmark_total_return",
            "benchmark_cagr", "alpha", "beta", "starting_equity", "final_equity",
            "peak_equity", "best_period", "worst_period",
        ),
    ),
    (
        "risk",
        (
            "sharpe", "sortino", "calmar", "volatility", "downside_deviation",
            "max_drawdown", "max_drawdown_duration_days", "exposure", "turnover",
        ),
    ),
    (
        "trades",
        (
            "trades", "wins", "losses", "hit_rate", "profit_factor", "expectancy",
            "avg_trade_pnl", "avg_win", "avg_loss", "best_trade", "worst_trade",
            "avg_bars_held", "gross_profit", "gross_loss", "open_positions",
        ),
    ),
    ("costs", ("commission", "fill_mode", "optimistic_fills")),
    (
        "run",
        ("start", "end", "days", "periods", "periods_per_year", "attempt", "contaminated"),
    ),
)

_LEDGER_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("ticker", "ticker", "l"),
    ("side", "side", "l"),
    ("qty", "qty", "r"),
    ("entry_time", "entry", "l"),
    ("entry_price", "entry px", "r"),
    ("exit_time", "exit", "l"),
    ("exit_price", "exit px", "r"),
    ("bars_held", "bars", "r"),
    ("pnl", "p&l", "r"),
    ("pnl_pct", "p&l %", "r"),
    ("commission", "comm", "r"),
    ("exit_reason", "reason", "l"),
)

# --- chart geometry, in viewBox units; both charts share the x band ----------
_W = 1000.0
_EQ_H = 300.0
_DD_H = 152.0
_PAD_L = 78.0
_PAD_R = 16.0
_PAD_T = 12.0
_EQ_PAD_B = 12.0
_DD_PAD_B = 28.0


# --- public API ---------------------------------------------------------------


def report_path(run_id: str) -> Path:
    """``runs/<run_id>/report.html`` -- the archival location the console links
    to, so it must be derivable from a run_id alone."""
    return _run_dir(run_id) / "report.html"


def render_report(
    result: "BacktestResult | str",
    *,
    out: Path | None = None,
    open_browser: bool = False,
) -> Path:
    """Render one run to a self-contained HTML file and return its path.

    ``result`` is either a live ``BacktestResult`` or a ``run_id``; the run_id
    path reads ``runs/<run_id>/`` artifacts only, which is what makes a report
    reproducible from an archived directory long after the process that made it
    is gone.
    """
    data = _load(result)
    target = _resolve_out(data.run_id, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_document(data), encoding="utf-8")
    if open_browser:
        import webbrowser

        webbrowser.open(target.resolve().as_uri())
    return target


# --- loading ------------------------------------------------------------------


@dataclass
class _Report:
    run_id: str
    strategy: str
    metrics: dict[str, Any]
    equity: pd.DataFrame
    trades: list[dict[str, Any]]
    config: dict[str, Any]
    data_version: str
    git_commit: str
    config_hash: str
    attempt: int
    family_attempts: int
    benchmark: pd.Series | None = None
    warnings: list[str] = field(default_factory=list)
    n_trades_total: int = 0


def _run_dir(run_id: str) -> Path:
    from lab.config import get_settings

    if not run_id or any(c in run_id for c in '/\\:*?"<>|'):
        raise ValueError(f"unsafe run_id for a directory name: {run_id!r}")
    return get_settings().paths.runs / run_id


def _resolve_out(run_id: str, out: Path | None) -> Path:
    if out is None:
        return report_path(run_id)
    p = Path(out)
    return (p / "report.html") if p.is_dir() else p


def _load(result: "BacktestResult | str") -> _Report:
    if isinstance(result, (str, Path)):
        return _from_artifacts(str(result))
    if not hasattr(result, "run_id") or not hasattr(result, "metrics"):
        raise ValueError(
            f"render_report expects a BacktestResult or a run_id string, "
            f"got {type(result).__name__}"
        )
    return _from_result(result)


def _from_result(result: "BacktestResult") -> _Report:
    config = dict(result.config or {})
    metrics = dict(result.metrics or {})
    equity = M.equity_frame(result.equity)
    trades = [t.to_dict() for t in (result.trades or [])]
    git, chash, attempt, family = _provenance(result.run_id, config, getattr(result, "attempt", 0))
    return _Report(
        run_id=result.run_id,
        strategy=_strategy_name(config, metrics),
        metrics=metrics,
        equity=equity,
        trades=trades,
        config=config,
        data_version=result.data_version or "unversioned",
        git_commit=git,
        config_hash=chash,
        attempt=attempt,
        family_attempts=family,
        benchmark=_benchmark(config, equity.index),
        warnings=list(getattr(result, "warnings", []) or []),
        n_trades_total=len(trades),
    )


def _from_artifacts(run_id: str) -> _Report:
    d = _run_dir(run_id)
    if not (d / "metrics.json").exists():
        raise ValueError(f"no run artifacts at {d}; nothing to report on")

    metrics = _read_json(d / "metrics.json", {})
    config = _read_json(d / "config.json", {})
    provenance = _read_json(d / "provenance.json", {})

    equity = pd.DataFrame(columns=["equity", "drawdown"])
    eq_path = d / "equity.parquet"
    if eq_path.exists():
        raw = pd.read_parquet(eq_path)
        if "event_time" in raw.columns:
            raw = raw.set_index("event_time")
        if "drawdown" not in raw.columns and "equity" in raw.columns:
            raw["drawdown"] = M.drawdown_series(raw["equity"])
        equity = raw

    trades: list[dict[str, Any]] = []
    tr_path = d / "trades.csv"
    if tr_path.exists() and tr_path.stat().st_size > 0:
        tdf = pd.read_csv(tr_path)
        trades = [
            {k: (None if v != v else v) for k, v in row.items()}  # NaN -> None
            for row in tdf.to_dict(orient="records")
        ]

    git, chash, attempt, family = _provenance(run_id, config, int(metrics.get("attempt", 0) or 0))
    return _Report(
        run_id=run_id,
        strategy=_strategy_name(config, metrics),
        metrics=metrics,
        equity=equity,
        trades=trades,
        config=config,
        data_version=provenance.get("data_version") or "unversioned",
        git_commit=git,
        config_hash=chash,
        attempt=attempt,
        family_attempts=family,
        benchmark=_benchmark(config, equity.index),
        warnings=list(provenance.get("warnings") or metrics.get("warnings") or []),
        n_trades_total=len(trades),
    )


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _strategy_name(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    raw = str(config.get("strategy") or metrics.get("strategy") or "strategy")
    return Path(raw).stem if raw.endswith(".py") else raw


def _provenance(
    run_id: str, config: Mapping[str, Any], fallback_attempt: int
) -> tuple[str, str, int, int]:
    """Registry first, recomputation second. A missing or locked registry
    degrades the header, it never fails the report."""
    try:
        from lab.registry.runs import RunRegistry, config_hash, git_commit
    except Exception:  # pragma: no cover - registry is a hard dep in practice
        return "", "", fallback_attempt, 0

    record = None
    registry = None
    try:
        registry = RunRegistry()
        record = registry.get(run_id)
    except Exception:
        record = None

    if record is not None:
        family = 0
        try:
            family = registry.family_attempts(record.strategy)
        except Exception:
            family = 0
        return (
            record.git_commit or "",
            record.config_hash or "",
            int(record.attempt or 0),
            family,
        )

    try:
        chash = config_hash({k: v for k, v in config.items() if k not in _DERIVED_CONFIG_KEYS})
    except Exception:
        chash = ""
    try:
        commit = git_commit() or ""
    except Exception:
        commit = ""
    return commit, chash, fallback_attempt, 0


def _benchmark(config: Mapping[str, Any], index: pd.Index) -> pd.Series | None:
    """The benchmark close series, rebased at render time.

    Read without an ``as_of`` barrier on purpose: this line is a display
    reference drawn after the fact, never an input to a decision, so the
    knowledge-time barrier that governs the engine does not apply here.
    """
    ticker = config.get("benchmark")
    if not ticker or len(index) == 0 or not isinstance(index, pd.DatetimeIndex):
        return None
    try:
        from lab.store import parquet_io

        df = parquet_io.read_bars(
            [str(ticker).upper()],
            timeframe=str(config.get("timeframe") or "1d"),
            start=index[0].to_pydatetime(),
            end=index[-1].to_pydatetime(),
            # The run's own source: unioning two feeds would draw a benchmark
            # line out of two unrelated price series.
            source=config.get("source"),
        )
    except Exception:
        return None
    if df is None or len(df) == 0 or "close" not in df.columns:
        return None
    s = df.set_index("event_time")["close"].astype("float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(index.union(s.index)).ffill().reindex(index).dropna()
    return s.rename(str(ticker).upper()) if len(s) > 1 else None


# --- formatting ----------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _esc_text(value: Any) -> str:
    """Escape for a text node, where quotes are not special.

    The standing caveats are module constants that operators and tests read back
    out of the rendered document; turning the apostrophe in ``bar's`` into
    ``&#x27;`` would make the document's copy of a warning differ from the
    warning, for no safety gain outside an attribute value.
    """
    return html.escape("" if value is None else str(value), quote=False)


def _fmt_value(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value) if value else "none"
    if isinstance(value, str):
        return _fmt_timestamp(value) if key in ("start", "end") else value
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return str(value)

    v = float(value)
    if key in _PCT_KEYS:
        return f"{v * 100:,.2f}%"
    if key in _MONEY_KEYS:
        return f"{v:,.2f}"
    if key in _INT_KEYS or (isinstance(value, int) and not isinstance(value, bool)):
        return f"{int(v):,d}"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:,.3f}"


def _fmt_timestamp(raw: Any) -> str:
    if raw in (None, "", "nan"):
        return "—"
    try:
        ts = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return str(raw)
    if ts is pd.NaT or pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d")


def _fmt_param(value: Any) -> str:
    """Parameters are inputs, not measurements: show them exactly as tuned,
    without the fixed decimal places the metrics tables use for column
    alignment, which would round 5e-4 to 0.000."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,d}"
    if isinstance(value, float):
        return f"{value:,.6g}" if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_param(v) for v in value) if value else "none"
    return "—" if value is None else str(value)


def _sign_class(key: str, value: Any) -> str:
    if key not in _SIGNED_KEYS or not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    if value > 0:
        return " gain"
    if value < 0:
        return " loss"
    return ""


def _pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:,.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 2) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{f:,.{digits}f}" if math.isfinite(f) else "—"


# --- chart primitives -----------------------------------------------------------


def _downsample(values: Sequence[float], limit: int) -> list[int]:
    """Indices to plot: per bucket keep the min and the max, in original order,
    so thinning a long curve cannot erase the extremes that matter."""
    n = len(values)
    if n <= limit:
        return list(range(n))
    buckets = max(limit // 2, 2)
    keep: set[int] = {0, n - 1}
    for b in range(buckets):
        lo = int(b * n / buckets)
        hi = max(lo + 1, int((b + 1) * n / buckets))
        window = range(lo, min(hi, n))
        lo_i = min(window, key=lambda i: values[i])
        hi_i = max(window, key=lambda i: values[i])
        keep.add(lo_i)
        keep.add(hi_i)
    return sorted(keep)


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0]
    # A degenerate range is a real case -- a strategy that never traded has a flat
    # equity curve -- and widening it would print gridlines at values the series
    # never reaches. One label at the level is the only honest axis. The relative
    # tolerance also keeps float noise from driving `step` below the ULP of `lo`,
    # where `floor(lo / step) * step` stops advancing and the loop below runs
    # ~1e17 times.
    if lo <= hi <= lo + abs(lo) * 1e-12 + 1e-12:
        return [round(lo, 10)]
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0) * 0.01
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag * 10
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            step = mult * mag
            break
    ticks: list[float] = []
    v = math.floor(lo / step) * step
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks or [lo, hi]


def _x_of(i: int, n: int) -> float:
    span = _W - _PAD_L - _PAD_R
    if n <= 1:
        return _PAD_L + span / 2.0
    return _PAD_L + span * (i / (n - 1))


def _x_ticks(index: pd.Index, count: int = 6) -> list[tuple[int, str]]:
    n = len(index)
    if n == 0:
        return []
    count = max(2, min(count, n))
    try:
        span_days = (index[-1] - index[0]).days
    except Exception:
        span_days = 0
    fmt = "%Y-%m-%d" if span_days <= 420 else "%Y-%m"
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for k in range(count):
        i = int(round(k * (n - 1) / (count - 1)))
        if i in seen:
            continue
        seen.add(i)
        try:
            label = pd.Timestamp(index[i]).strftime(fmt)
        except Exception:
            label = str(index[i])
        out.append((i, label))
    return out


def _oos_spans(index: pd.Index, windows: Sequence[Mapping[str, Any]] | None) -> list[tuple[int, int]]:
    """Contiguous out-of-sample index ranges, derived from walk-forward windows.

    Normalizes the window bounds onto the equity index's timezone first --
    walk-forward configs round-trip through JSON as naive strings, and comparing
    those against a tz-aware index raises rather than silently mis-shading.
    """
    if not windows or len(index) == 0:
        return []
    tz = getattr(index, "tz", None)
    normalized: list[dict[str, Any]] = []
    for w in windows:
        lo, hi = w.get("oos_start"), w.get("oos_end")
        if lo is None or hi is None:
            continue
        try:
            lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        except (TypeError, ValueError):
            continue
        if tz is not None:
            lo_ts = lo_ts.tz_localize(tz) if lo_ts.tzinfo is None else lo_ts.tz_convert(tz)
            hi_ts = hi_ts.tz_localize(tz) if hi_ts.tzinfo is None else hi_ts.tz_convert(tz)
        elif lo_ts.tzinfo is not None:
            lo_ts, hi_ts = lo_ts.tz_localize(None), hi_ts.tz_localize(None)
        normalized.append({"oos_start": lo_ts, "oos_end": hi_ts})

    flags = M.is_oos_split(index, normalized)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans


def _oos_rects(spans: Sequence[tuple[int, int]], n: int, top: float, height: float,
               *, label: bool) -> str:
    """The single most important thing the chart does: make OOS unmissable."""
    parts: list[str] = []
    for k, (i0, i1) in enumerate(spans):
        x0 = _x_of(i0, n)
        x1 = _x_of(i1, n)
        width = max(x1 - x0, 1.5)
        parts.append(
            f'<rect class="oos" x="{x0:.1f}" y="{top:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}"/>'
        )
        parts.append(
            f'<line class="oos-edge" x1="{x0:.1f}" y1="{top:.1f}" x2="{x0:.1f}" '
            f'y2="{top + height:.1f}"/>'
        )
        if label and width > 26:
            parts.append(
                f'<text class="oos-label" x="{x0 + 4:.1f}" y="{top + 11:.1f}">'
                f"OOS{'' if len(spans) == 1 else f' {k + 1}'}</text>"
            )
    return "".join(parts)


def _polyline(xs: Sequence[float], ys: Sequence[float], css: str) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline class="{css}" points="{pts}"/>'


def _equity_svg(data: _Report, spans: Sequence[tuple[int, int]]) -> str:
    eq = data.equity["equity"].astype("float64")
    n = len(eq)
    if n == 0:
        return '<p class="empty">no equity curve recorded for this run</p>'

    values = eq.tolist()
    series: list[tuple[str, list[float]]] = [("line-equity", values)]
    bench = data.benchmark
    bench_vals: list[float] | None = None
    if bench is not None and len(bench) > 1 and float(bench.iloc[0]) > 0:
        scale = values[0] / float(bench.iloc[0])
        bench_vals = [float(v) * scale for v in bench.reindex(eq.index).ffill().tolist()]
        series.append(("line-bench", bench_vals))

    pool = [v for _, vals in series for v in vals if v == v]
    lo, hi = min(pool), max(pool)
    pad = (hi - lo) * 0.06 or (abs(hi) * 0.01 or 1.0)
    lo, hi = lo - pad, hi + pad
    ticks = _nice_ticks(lo, hi, 6)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    top, bottom = _PAD_T, _EQ_H - _EQ_PAD_B
    span = hi - lo or 1.0

    def y_of(v: float) -> float:
        return bottom - (v - lo) / span * (bottom - top)

    parts: list[str] = []
    parts.append(_oos_rects(spans, n, top, bottom - top, label=True))
    for t in ticks:
        y = y_of(t)
        parts.append(f'<line class="gline" x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ytick" x="{_PAD_L - 8}" y="{y + 3.5:.1f}">{_num(t, 0)}</text>')
    for i, _ in _x_ticks(eq.index):
        x = _x_of(i, n)
        parts.append(f'<line class="gline-v" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom:.1f}"/>')

    start_y = y_of(values[0])
    parts.append(
        f'<line class="baseline" x1="{_PAD_L}" y1="{start_y:.1f}" x2="{_W - _PAD_R}" '
        f'y2="{start_y:.1f}"/>'
    )

    up = values[-1] >= values[0]
    for css, vals in series:
        keep = _downsample(vals, MAX_CHART_POINTS)
        xs = [_x_of(i, n) for i in keep]
        ys = [y_of(vals[i]) for i in keep]
        klass = css if css != "line-equity" else f"line-equity {'up' if up else 'down'}"
        parts.append(_polyline(xs, ys, klass))

    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {_EQ_H:.0f}" '
        f'role="img" aria-label="equity curve">{"".join(parts)}</svg>'
    )


def _drawdown_svg(data: _Report, spans: Sequence[tuple[int, int]]) -> str:
    if "drawdown" in data.equity.columns:
        dd = data.equity["drawdown"].astype("float64")
    else:
        dd = M.drawdown_series(data.equity.get("equity", pd.Series(dtype="float64")))
    n = len(dd)
    if n == 0:
        return ""

    values = [float(v) if v == v else 0.0 for v in dd.tolist()]
    worst = min(min(values), -0.005)
    top, bottom = _PAD_T, _DD_H - _DD_PAD_B
    height = bottom - top

    def y_of(v: float) -> float:
        return top + (v / worst) * height if worst else top

    parts: list[str] = []
    parts.append(_oos_rects(spans, n, top, height, label=False))
    for t in _nice_ticks(worst, 0.0, 3):
        if t > 0:
            continue
        y = y_of(t)
        parts.append(f'<line class="gline" x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ytick" x="{_PAD_L - 8}" y="{y + 3.5:.1f}">{t * 100:,.1f}%</text>')
    # Depth is the number readers look for, so label the floor explicitly rather
    # than leaving it to be inferred from the nearest round gridline.
    parts.append(
        f'<text class="ytick worst" x="{_PAD_L - 8}" y="{bottom:.1f}">{worst * 100:,.1f}%</text>'
    )

    keep = _downsample(values, MAX_CHART_POINTS)
    xs = [_x_of(i, n) for i in keep]
    ys = [y_of(values[i]) for i in keep]
    area = " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    parts.append(
        f'<path class="dd-area" d="M{xs[0]:.1f},{top:.1f} {area} L{xs[-1]:.1f},{top:.1f} Z"/>'
    )
    parts.append(_polyline(xs, ys, "line-dd"))

    for i, label in _x_ticks(dd.index):
        x = _x_of(i, n)
        parts.append(f'<line class="gline-v" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom:.1f}"/>')
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        parts.append(
            f'<text class="xtick" x="{x:.1f}" y="{_DD_H - 10:.1f}" text-anchor="{anchor}">'
            f"{_esc(label)}</text>"
        )

    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {_DD_H:.0f}" '
        f'role="img" aria-label="drawdown">{"".join(parts)}</svg>'
    )


# --- document sections ------------------------------------------------------------


def _kv(label: str, value: str, *, mono: bool = True) -> str:
    css = "v mono" if mono else "v"
    return f'<div class="kv"><span class="k">{_esc(label)}</span><span class="{css}">{value}</span></div>'


def _header(data: _Report) -> str:
    m = data.metrics
    start = _fmt_timestamp(m.get("start"))
    end = _fmt_timestamp(m.get("end"))
    attempts = f"#{data.attempt}"
    if data.family_attempts:
        attempts += f" of {data.family_attempts} for {_esc(data.strategy)}"
    items = [
        _kv("range", f"{_esc(start)} → {_esc(end)}"),
        _kv("bars", f"{int(m.get('periods', len(data.equity)) or 0):,d}"),
        _kv("git", _esc(data.git_commit or "unknown")),
        _kv("config hash", _esc(data.config_hash or "unknown")),
        _kv("data version", _esc(data.data_version)),
        _kv("attempt", attempts),
    ]
    return (
        '<header class="head">'
        f'<h1>{_esc(data.strategy)}<span class="sep">·</span>'
        f'<span class="runid mono">{_esc(data.run_id)}</span></h1>'
        f'<div class="meta">{"".join(items)}</div>'
        "</header>"
    )


def _kpis(data: _Report) -> str:
    m = data.metrics
    cells: list[tuple[str, str, str]] = [
        ("total return", _pct(m.get("total_return")), _sign_class("total_return", m.get("total_return"))),
        ("cagr", _pct(m.get("cagr")), _sign_class("cagr", m.get("cagr"))),
        ("sharpe", _num(m.get("sharpe"), 2), _sign_class("sharpe", m.get("sharpe"))),
        ("max dd", _pct(m.get("max_drawdown")), _sign_class("max_drawdown", m.get("max_drawdown"))),
        ("trades", f"{int(m.get('trades', 0) or 0):,d}", ""),
        ("hit rate", _pct(m.get("hit_rate"), 1), ""),
    ]
    if "benchmark_total_return" in m:
        cells.append(
            (
                "vs benchmark",
                _pct(m.get("excess_return")),
                _sign_class("excess_return", m.get("excess_return")),
            )
        )
    body = "".join(
        f'<div class="kpi"><span class="k">{_esc(label)}</span>'
        f'<span class="v num{cls}">{value}</span></div>'
        for label, value, cls in cells
    )
    return f'<section class="kpis">{body}</section>'


def _banners(data: _Report) -> str:
    m = data.metrics
    out: list[str] = []
    if m.get("optimistic_fills"):
        out.append(f'<div class="banner warn">{_esc_text(OPTIMISTIC_WARNING)}</div>')
    if m.get("contaminated"):
        out.append(f'<div class="banner warn">{_esc_text(CONTAMINATION_WARNING)}</div>')
    for w in data.warnings:
        text = str(w)
        if m.get("optimistic_fills") and "same-bar-close" in text:
            continue  # already said, louder, above
        out.append(f'<div class="banner note">{_esc_text(text)}</div>')
    return "".join(out)


def _charts(data: _Report) -> str:
    spans = _oos_spans(data.equity.index, data.config.get("windows"))
    equity = _equity_svg(data, spans)
    drawdown = _drawdown_svg(data, spans)

    legend = ['<span class="lg"><i class="sw sw-equity"></i>equity</span>']
    if data.benchmark is not None:
        legend.append(
            f'<span class="lg"><i class="sw sw-bench"></i>{_esc(data.benchmark.name)} (rebased)</span>'
        )
    legend.append('<span class="lg"><i class="sw sw-dd"></i>drawdown</span>')
    if spans:
        legend.append(
            f'<span class="lg"><i class="sw sw-oos"></i>out-of-sample '
            f"({len(spans)} window{'s' if len(spans) != 1 else ''})</span>"
        )
    else:
        legend.append('<span class="lg muted">no walk-forward windows in this config</span>')

    return (
        '<section class="panel chart-panel">'
        '<div class="panel-head"><span class="title">equity &amp; drawdown</span>'
        f'<span class="legend">{"".join(legend)}</span></div>'
        f"{equity}{drawdown}"
        "</section>"
    )


def _metrics_tables(data: _Report) -> str:
    m = dict(data.metrics)
    shown: set[str] = set()
    panels: list[str] = []

    for title, keys in _GROUPS:
        rows = [k for k in keys if k in m]
        if not rows:
            continue
        shown.update(rows)
        panels.append(_metrics_panel(title, [(k, m[k]) for k in rows]))

    leftover = [k for k in m if k not in shown and k != "warnings"]
    if leftover:
        panels.append(_metrics_panel("other", [(k, m[k]) for k in sorted(leftover)]))

    return f'<div class="grid">{"".join(panels)}</div>'


def _metrics_panel(title: str, rows: Sequence[tuple[str, Any]]) -> str:
    body = "".join(
        f'<tr><td class="key">{_esc(k.replace("_", " "))}</td>'
        f'<td class="num{_sign_class(k, v)}">{_esc(_fmt_value(k, v))}</td></tr>'
        for k, v in rows
    )
    return (
        f'<section class="panel"><div class="panel-head"><span class="title">{_esc(title)}</span></div>'
        f'<table class="metrics"><tbody>{body}</tbody></table></section>'
    )


def _ledger(data: _Report) -> str:
    total = data.n_trades_total
    rows = data.trades[:MAX_LEDGER_ROWS]
    head = "".join(
        '<th class="num">' + _esc(label) + "</th>"
        if align == "r"
        else "<th>" + _esc(label) + "</th>"
        for _, label, align in _LEDGER_COLUMNS
    )

    if not rows:
        body = (
            f'<tr><td class="empty" colspan="{len(_LEDGER_COLUMNS)}">'
            "no closed round trips in this run</td></tr>"
        )
    else:
        cells: list[str] = []
        for t in rows:
            tds: list[str] = []
            for key, _, align in _LEDGER_COLUMNS:
                v = t.get(key)
                if key in ("entry_time", "exit_time"):
                    text = _fmt_timestamp(v) if v not in (None, "") else "open"
                elif key in ("pnl_pct",):
                    text = _pct(v)
                elif key in ("pnl", "entry_price", "exit_price", "qty", "commission"):
                    text = _num(v)
                elif key == "bars_held":
                    text = _num(v, 0)
                else:
                    text = "" if v in (None, "") else str(v)
                cls = "num" if align == "r" else ""
                if key in ("pnl", "pnl_pct") and isinstance(v, (int, float)):
                    cls += " gain" if v > 0 else (" loss" if v < 0 else "")
                cls = cls.strip()
                open_tag = f'<td class="{cls}">' if cls else "<td>"
                tds.append(f"{open_tag}{_esc(text)}</td>")
            cells.append(f"<tr>{''.join(tds)}</tr>")
        body = "".join(cells)

    note = ""
    if total > len(rows):
        note = (
            f'<span class="truncated">showing {len(rows):,d} of {total:,d} trades — '
            f"the full ledger is trades.csv next to this file</span>"
        )
    return (
        '<section class="panel"><div class="panel-head">'
        f'<span class="title">trade ledger</span><span class="count">{total:,d} round trips</span>'
        f"{note}</div>"
        f'<div class="scroll"><table class="ledger"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></section>"
    )


def _footer(data: _Report) -> str:
    fills = dict(data.config.get("fills") or {})
    mode = str(data.metrics.get("fill_mode") or fills.get("mode") or "next_open")
    bits = [f"mode {mode}"]
    if fills.get("slippage_bps") is not None:
        bits.append(f"slippage {_num(fills['slippage_bps'], 1)} bps")
    for key, label in (
        ("commission_per_order", "commission/order"),
        ("commission_per_share", "commission/share"),
        ("min_commission", "min commission"),
        ("partial_fill_volume_pct", "volume cap"),
    ):
        if fills.get(key):
            bits.append(f"{label} {_num(fills[key], 4)}")
    if fills.get("allow_fractional") or data.config.get("allow_fractional"):
        bits.append("fractional shares")

    caveats = [f'<li class="caveat">{_esc_text(SURVIVORSHIP_NOTE)}</li>']
    if data.metrics.get("optimistic_fills"):
        caveats.insert(0, f'<li class="caveat loud">{_esc_text(OPTIMISTIC_WARNING)}</li>')
    if data.metrics.get("contaminated"):
        caveats.insert(0, f'<li class="caveat loud">{_esc_text(CONTAMINATION_WARNING)}</li>')

    prov = "".join(
        [
            _kv("run", _esc(data.run_id)),
            _kv("git", _esc(data.git_commit or "unknown")),
            _kv("config hash", _esc(data.config_hash or "unknown")),
            _kv("data version", _esc(data.data_version)),
            _kv("fill model", _esc(" · ".join(bits))),
            _kv("universe", f"{len(data.config.get('tickers') or []):,d} tickers", mono=True),
        ]
    )
    return (
        '<footer class="prov"><div class="panel-head"><span class="title">provenance</span></div>'
        f'<div class="meta">{prov}</div>'
        f'<ul class="caveats">{"".join(caveats)}</ul></footer>'
    )


def _params_panel(data: _Report) -> str:
    params = data.config.get("resolved_params") or data.config.get("params") or {}
    if not isinstance(params, Mapping) or not params:
        return ""
    rows = "".join(
        f'<tr><td class="key">{_esc(k)}</td><td class="num">{_esc(_fmt_param(v))}</td></tr>'
        for k, v in sorted(params.items(), key=lambda kv: str(kv[0]))
    )
    return (
        '<section class="panel"><div class="panel-head"><span class="title">parameters</span></div>'
        f'<table class="metrics"><tbody>{rows}</tbody></table></section>'
    )


def _document(data: _Report) -> str:
    title = f"{data.strategy} · {data.run_id}"
    params = _params_panel(data)
    params = f'<div class="grid">{params}</div>\n' if params else ""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>\n{_CSS}\n</style>\n</head>\n<body>\n"
        '<main class="wrap">\n'
        f"{_header(data)}\n{_banners(data)}\n{_kpis(data)}\n{_charts(data)}\n"
        f"{_metrics_tables(data)}\n{params}"
        f"{_ledger(data)}\n{_footer(data)}\n"
        "</main>\n</body>\n</html>\n"
    )


# --- style (§7 of the console design doc) -------------------------------------
# Inline and hand-written: a stylesheet that has to be fetched is a stylesheet
# that can fail, and this file is meant to render off a disk with no network.

_CSS = """
:root{
  --graphite:#14171C; --gunmetal:#1C2128; --slate:#2B323C;
  --chalk:#E9ECEF; --ash:#98A2AE;
  --moss:#6FBF73; --ember:#E0654F; --amber:#D9A441; --teal:#45B2A1;
  --sans:"IBM Plex Sans","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--graphite); color:var(--chalk);
  font:400 13px/1.45 var(--sans);
  -webkit-print-color-adjust:exact; print-color-adjust:exact;
}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
.mono,.num{font-family:var(--mono);font-variant-numeric:tabular-nums lining-nums;
  font-feature-settings:"tnum" 1,"lnum" 1}
.num{text-align:right;white-space:nowrap}
.gain{color:var(--moss)} .loss{color:var(--ember)} .muted{color:var(--ash)}

h1{font-size:19px;font-weight:600;margin:0 0 10px;letter-spacing:-.01em}
h1 .sep{color:var(--slate);margin:0 8px}
h1 .runid{font-size:14px;font-weight:400;color:var(--ash)}
.head{border-bottom:1px solid var(--slate);padding-bottom:14px;margin-bottom:16px}
.meta{display:flex;flex-wrap:wrap;gap:6px 26px}
.kv{display:flex;flex-direction:column;gap:2px}
.k{font-size:9.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--ash)}
.v{font-size:12px;color:var(--chalk)}

.banner{border-left:3px solid var(--amber);background:var(--gunmetal);
  padding:10px 14px;margin:0 0 10px;font-size:12.5px;color:var(--chalk)}
.banner.warn{color:var(--amber);font-weight:600;letter-spacing:.005em}
.banner.note{border-left-color:var(--slate);color:var(--ash);font-weight:400}

.kpis{display:flex;flex-wrap:wrap;gap:1px;background:var(--slate);
  border:1px solid var(--slate);margin:14px 0 18px}
.kpi{flex:1 1 132px;background:var(--gunmetal);padding:9px 14px;
  display:flex;flex-direction:column;gap:3px}
.kpi .v{font-size:19px;line-height:1.1}

.panel{background:var(--gunmetal);border:1px solid var(--slate);margin:0 0 14px}
.panel-head{display:flex;align-items:baseline;gap:14px;padding:8px 14px;
  border-bottom:1px solid var(--slate)}
.panel-head .title{font-size:10px;text-transform:uppercase;letter-spacing:.11em;color:var(--ash)}
.panel-head .count,.panel-head .truncated{font-size:11px;color:var(--ash);margin-left:auto}
.panel-head .truncated{color:var(--amber)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;
  align-items:start}
.grid .panel{margin:0}

.chart-panel .chart{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:var(--ash);margin-left:auto}
.lg{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:3px;display:inline-block}
.sw-equity{background:var(--moss)} .sw-bench{background:var(--ash)}
.sw-dd{background:var(--ember)} .sw-oos{background:var(--teal);height:9px;opacity:.55}

.gline{stroke:var(--slate);stroke-width:1;shape-rendering:crispEdges}
.gline-v{stroke:var(--slate);stroke-width:1;opacity:.45;shape-rendering:crispEdges}
.baseline{stroke:var(--ash);stroke-width:1;stroke-dasharray:3 4;opacity:.6}
.line-equity{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke;
  stroke-linejoin:round;stroke:var(--chalk)}
.line-equity.up{stroke:var(--moss)} .line-equity.down{stroke:var(--ember)}
.line-bench{fill:none;stroke:var(--ash);stroke-width:1.1;stroke-dasharray:4 3;
  vector-effect:non-scaling-stroke;opacity:.7}
.line-dd{fill:none;stroke:var(--ember);stroke-width:1.3;vector-effect:non-scaling-stroke}
.dd-area{fill:var(--ember);opacity:.16}
.oos{fill:var(--teal);opacity:.11}
.oos-edge{stroke:var(--teal);stroke-width:1;opacity:.5;shape-rendering:crispEdges}
.oos-label{fill:var(--teal);font:600 9px var(--mono);letter-spacing:.08em}
.ytick{fill:var(--ash);font:400 10px var(--mono);text-anchor:end}
.ytick.worst{fill:var(--ember)}
.xtick{fill:var(--ash);font:400 10px var(--mono)}

table{border-collapse:collapse;width:100%}
th{font-size:9.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--ash);
  font-weight:500;text-align:left;padding:7px 14px;border-bottom:1px solid var(--slate);
  white-space:nowrap;position:sticky;top:0;background:var(--gunmetal)}
th.num{text-align:right}
td{padding:5px 14px;border-bottom:1px solid rgba(43,50,60,.55);font-size:12px;
  white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.key{color:var(--ash)}
td.empty{color:var(--ash);text-align:center;padding:18px}
.metrics td{padding:4px 14px}
.ledger td{font-family:var(--mono);font-variant-numeric:tabular-nums lining-nums}
.scroll{max-height:560px;overflow:auto}

.prov{border-top:1px solid var(--slate);margin-top:22px;padding-top:14px}
.prov .meta{margin-bottom:12px}
.caveats{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.caveat{border-left:3px solid var(--slate);padding:7px 12px;font-size:12px;color:var(--ash);
  background:var(--gunmetal)}
.caveat.loud{border-left-color:var(--amber);color:var(--amber);font-weight:600}
p.empty{color:var(--ash);padding:22px 14px;margin:0;text-align:center}
"""


__all__ = [
    "CONTAMINATION_WARNING",
    "MAX_LEDGER_ROWS",
    "OPTIMISTIC_WARNING",
    "SURVIVORSHIP_NOTE",
    "render_report",
    "report_path",
]
