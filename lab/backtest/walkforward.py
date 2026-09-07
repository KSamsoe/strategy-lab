"""Walk-forward splitting: the defense that makes a sweep mean something.

A grid search scored on the same data it was fitted to is a random-number
generator with good manners (design doc §6). This module cuts the span into
equal blocks and rolls a train/test schedule across them, so every bar that
scores a parameter set was unseen when that set was chosen. The OOS spans of
consecutive windows are contiguous and never overlap, which is what lets their
returns be pooled into a single out-of-sample curve -- pooled by ``stitch``,
which rebases each span onto the running level so only returns earned *inside*
one span ever compound.

Splitting happens in *bar* space, not calendar space: blocks are equal counts of
trading days (or intraday bars), so a window spanning a thin summer does not
quietly get fewer observations than one spanning a busy autumn.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lab.backtest import metrics as M
from lab.engine.events import Trade
from lab.timeutil import UTC, is_intraday, parse_timeframe, to_utc, trading_days

if TYPE_CHECKING:  # pragma: no cover - typing only, sweep imports this module
    from lab.backtest.sweep import SweepConfig

_LOG = logging.getLogger(__name__)

#: How many rolls a bare ratio produces. Five OOS periods is enough to see
#: whether an edge is consistent without burying the report in windows.
DEFAULT_WINDOWS = 5

#: Blocks below this many bars stop being an evaluation and start being noise,
#: so the default window count is reduced rather than the blocks shrunk.
MIN_BLOCK_UNITS = 21

_SPEC_RE = re.compile(r"^\s*(\d+)\s*[:/x]\s*(\d+)\s*$", re.IGNORECASE)

#: Metrics the segment split cannot honestly restate. ``exposure`` and
#: ``turnover`` are computed by the runner from series the segment does not
#: carry; reporting them as 0.0 would be a lie with a decimal point on it.
_SEGMENT_DROP = frozenset({"exposure", "turnover"})


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "is_start": self.is_start.isoformat(),
            "is_end": self.is_end.isoformat(),
            "oos_start": self.oos_start.isoformat(),
            "oos_end": self.oos_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Window":
        return cls(
            index=int(d.get("index", 0)),
            is_start=to_utc(d["is_start"]),
            is_end=to_utc(d["is_end"]),
            oos_start=to_utc(d["oos_start"]),
            oos_end=to_utc(d["oos_end"]),
        )

    @property
    def is_span(self) -> tuple[datetime, datetime]:
        return (self.is_start, self.is_end)

    @property
    def oos_span(self) -> tuple[datetime, datetime]:
        return (self.oos_start, self.oos_end)

    @property
    def span(self) -> tuple[datetime, datetime]:
        return (self.is_start, self.oos_end)


def parse_spec(spec: str | int) -> tuple[int, int]:
    """``"4:1"`` -> ``(4, 1)``. A bare int is *n* train blocks against 1 test."""
    if isinstance(spec, bool):  # bool is an int subclass; a flag is not a spec
        raise ValueError(f"bad walk-forward spec {spec!r}; expected e.g. '4:1'")
    if isinstance(spec, (int, float)):
        if float(spec) != int(spec):
            raise ValueError(f"bad walk-forward spec {spec!r}; expected e.g. '4:1'")
        train, test = int(spec), 1
    else:
        text = str(spec).strip()
        m = _SPEC_RE.match(text)
        if m:
            train, test = int(m.group(1)), int(m.group(2))
        elif text.isdigit():
            train, test = int(text), 1
        else:
            raise ValueError(
                f"bad walk-forward spec {spec!r}; expected 'train:test' like '4:1', "
                f"'3:1', '6:2', or a plain integer"
            )
    if train < 1 or test < 1:
        raise ValueError(
            f"walk-forward spec {spec!r} yields fewer than one usable window: "
            f"train={train}, test={test}, both must be >= 1"
        )
    return train, test


def _unit_bounds(
    start: datetime, end: datetime, timeframe: str
) -> list[tuple[datetime, datetime]]:
    """Half-open bar slots as inclusive ``(lo, hi)`` pairs.

    Daily units are whole UTC days rather than bar timestamps because a stored
    daily bar is stamped at the session open (14:30Z); a boundary at midnight
    would drop the very bar it was meant to include.
    """
    if is_intraday(timeframe):
        delta = parse_timeframe(timeframe)
        idx = pd.date_range(start=start, end=end, freq=delta, tz=UTC)
        eps = timedelta(microseconds=1)
        return [(ts.to_pydatetime(), (ts + delta - eps).to_pydatetime()) for ts in idx]
    day = timedelta(days=1)
    eps = timedelta(microseconds=1)
    out: list[tuple[datetime, datetime]] = []
    for d in trading_days(start.date(), end.date()):
        lo = datetime(d.year, d.month, d.day, tzinfo=UTC)
        out.append((lo, lo + day - eps))
    return out


def make_windows(
    start: datetime | date | str,
    end: datetime | date | str,
    spec: str | int,
    *,
    timeframe: str = "1d",
    n_windows: int | None = None,
    min_block_units: int = MIN_BLOCK_UNITS,
) -> list[Window]:
    """Rolling train/test windows tiling ``[start, end]``.

    ``spec`` fixes the IS:OOS *ratio*, not the block count -- with ``"4:1"`` and
    *k* windows the span is cut into ``4 + k`` equal blocks, window *k* trains on
    four consecutive blocks and tests on the next one, then slides by one test
    block. Because the block size is derived from *k*, coverage is exact for any
    *k*: the first window's IS starts at ``start``, the last window's OOS ends at
    ``end``, and the OOS spans in between are contiguous and disjoint.
    """
    lo, hi = to_utc(start), to_utc(end)
    if lo >= hi:
        raise ValueError(f"walk-forward range is empty: start={lo.isoformat()} >= end={hi.isoformat()}")

    train, test = parse_spec(spec)
    units = _unit_bounds(lo, hi, timeframe)
    n = len(units)
    if n < train + test:
        raise ValueError(
            f"walk-forward spec {spec!r} yields fewer than one usable window over "
            f"{lo.date()}..{hi.date()}: {n} {timeframe} bars cannot fill "
            f"{train + test} blocks"
        )

    # One ceiling for both paths. An explicit n_windows that clears
    # `train + test*k <= n` can still cut the span into 1-bar "out-of-sample
    # periods", which is not a smaller evaluation but a different, useless one --
    # so the MIN_BLOCK_UNITS floor binds whether the caller asked for k or not.
    block_budget = max(train + test, n // max(int(min_block_units), 1))
    k_max = max(1, (block_budget - train) // test)

    if n_windows is not None:
        k = int(n_windows)
        if k < 1 or k > k_max:
            raise ValueError(
                f"n_windows={n_windows} does not fit {lo.date()}..{hi.date()} at "
                f"{spec!r}: {n} {timeframe} bars support at most {k_max} window(s) "
                f"with blocks of >= {min_block_units} bars; lower min_block_units "
                f"to overrule the floor"
            )
    else:
        # Prefer DEFAULT_WINDOWS rolls, but back off rather than cut blocks
        # below MIN_BLOCK_UNITS -- a 3-bar "out-of-sample period" measures noise.
        k = max(1, min(DEFAULT_WINDOWS, k_max))
        if k < DEFAULT_WINDOWS:
            _LOG.debug(
                "%s bars only support %d walk-forward window(s) at %s with >= %d-bar blocks",
                n, k, spec, min_block_units,
            )

    n_blocks = train + test * k
    # Integer edges keep blocks within one bar of each other and guarantee
    # edges[-1] == n, i.e. the last OOS bar is the last bar of the range.
    edges = [(j * n) // n_blocks for j in range(n_blocks + 1)]

    windows: list[Window] = []
    for i in range(k):
        a = edges[i * test]
        b = edges[i * test + train]
        c = edges[i * test + train + test]
        windows.append(
            Window(
                index=i,
                is_start=units[a][0],
                is_end=units[b - 1][1],
                oos_start=units[b][0],
                oos_end=units[c - 1][1],
            )
        )
    return windows


# --- scoring a run against its windows ---------------------------------------


def spans_mask(index: pd.DatetimeIndex, spans: Sequence[tuple[datetime, datetime]]) -> np.ndarray:
    """Segment labels for the timestamps falling inside any span (inclusive).

    ``0`` means "outside every span"; otherwise the value is the 1-based
    position of the *first* span containing the timestamp. The label, not just
    the in/out bit, is what ``stitch`` needs: two spans can be adjacent in bar
    space (walk-forward OOS blocks always are) and a plain boolean mask would
    lose the seam between them.
    """
    labels = np.zeros(len(index), dtype=np.int64)
    if len(index) == 0:
        return labels
    for k, (lo, hi) in enumerate(spans, start=1):
        inside = (index >= pd.Timestamp(lo)) & (index <= pd.Timestamp(hi))
        labels = np.where((labels == 0) & inside, k, labels)
    return labels


def _segment_labels(mask: np.ndarray | None, n: int) -> np.ndarray | None:
    """Normalize a boolean mask or a label array to ``int64`` labels.

    A boolean mask collapses to a single label, so its only segment breaks are
    the gaps -- which is exactly what it can express.
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 1 or len(arr) != n:
        return None
    return arr.astype(np.int64, copy=False)


def stitch(equity: pd.Series, mask: np.ndarray) -> pd.Series:
    """Equity restricted to ``mask``, with each segment rebased onto the last.

    Only returns whose *both* endpoints sit in the **same** segment are earned.
    The first bar of every segment after the first carries the running stitched
    level forward unchanged, because the move into it happened over an excluded
    stretch: the strategy was either being fitted there (in-sample) or running
    under a different parameter set (the previous walk-forward roll). Compounding
    straight through would credit the out-of-sample curve with a return it never
    produced out of sample -- and since walk-forward OOS blocks are contiguous by
    construction, that leak is invisible to any check based on gaps alone.
    """
    s = pd.Series(equity).astype("float64").dropna().sort_index()
    labels = _segment_labels(mask, len(s))
    if s.empty or labels is None or not labels.any():
        return pd.Series(dtype="float64")

    values = s.to_numpy()
    idx = s.index
    out_ts: list[Any] = []
    out_eq: list[float] = []
    level: float | None = None
    prev_label = 0
    for i in range(len(s)):
        label = int(labels[i])
        if label == 0:
            prev_label = 0
            continue
        if level is None:
            level = float(values[i])
        elif label == prev_label:
            prior = float(values[i - 1])
            level *= (float(values[i]) / prior) if prior else 1.0
        # else: first bar of a new segment -- rebase, i.e. leave the running
        # level where the previous segment left it.
        out_ts.append(idx[i])
        out_eq.append(level)
        prev_label = label

    return pd.Series(out_eq, index=pd.DatetimeIndex(out_ts, name=idx.name), name="equity")


def _trades_in(
    trades: Iterable[Trade], spans: Sequence[tuple[datetime, datetime]]
) -> list[Trade]:
    """Round trips *closed* inside the spans; still-open ones count where they
    were entered, since that is the only timestamp they have."""
    picked: list[Trade] = []
    for t in trades:
        when = t.exit_time or t.entry_time
        if when is None:
            continue
        w = to_utc(when)
        if any(lo <= w <= hi for lo, hi in spans):
            picked.append(t)
    return picked


def segment_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    spans: Sequence[tuple[datetime, datetime]],
    *,
    timeframe: str = "1d",
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics for one slice of a run, computed by the same function the whole
    run uses so an OOS Sharpe and a full-sample Sharpe are comparable."""
    s = pd.Series(equity).astype("float64").dropna().sort_index()
    if mask is None:
        mask = spans_mask(s.index, spans)
    curve = stitch(s, mask)
    out = M.compute_metrics(curve, _trades_in(trades, spans), timeframe=timeframe)
    for key in _SEGMENT_DROP:
        out.pop(key, None)
    out["bars"] = int(len(curve))
    return out


def split_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    windows: Sequence[Window],
    *,
    timeframe: str = "1d",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pooled ``(in_sample, out_of_sample)`` metrics for a run.

    A bar that is out-of-sample for *any* window is out-of-sample, full stop --
    later windows train on earlier windows' test blocks, and letting a bar be
    counted as in-sample there would launder it back into the optimistic half.
    """
    s = pd.Series(equity).astype("float64").dropna().sort_index()
    is_spans = [w.is_span for w in windows]
    oos_spans = [w.oos_span for w in windows]
    oos_mask = spans_mask(s.index, oos_spans)
    # Zeroing rather than masking keeps the labels: a training block cut in two
    # by a later window's OOS block must not compound across the removed middle.
    is_mask = np.where(oos_mask > 0, 0, spans_mask(s.index, is_spans))
    return (
        segment_metrics(s, trades, is_spans, timeframe=timeframe, mask=is_mask),
        segment_metrics(s, trades, oos_spans, timeframe=timeframe, mask=oos_mask),
    )


def window_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    windows: Sequence[Window],
    *,
    timeframe: str = "1d",
) -> list[dict[str, Any]]:
    """Per-window ``{"is": {...}, "oos": {...}}`` metrics for one run."""
    s = pd.Series(equity).astype("float64").dropna().sort_index()
    out: list[dict[str, Any]] = []
    for w in windows:
        out.append(
            {
                "index": w.index,
                "is": segment_metrics(s, trades, [w.is_span], timeframe=timeframe),
                "oos": segment_metrics(s, trades, [w.oos_span], timeframe=timeframe),
            }
        )
    return out


def run_walk_forward(cfg: "SweepConfig", *, progress: bool = False) -> dict[str, Any]:
    """Walk-forward the grid in ``cfg`` and rank the survivors out-of-sample.

    Thin on purpose: the sweep runner already walks the grid through the real
    engine and splits every run against ``cfg.walk_forward``. Duplicating that
    here would give the lab two answers to the same question.
    """
    # Lazy: sweep.py imports this module for make_windows, so a module-level
    # import would close the cycle.
    from lab.backtest.sweep import run_sweep

    if not cfg.walk_forward:
        raise ValueError(
            "run_walk_forward needs cfg.walk_forward set (e.g. '4:1'); "
            "use run_sweep for an unvalidated in-sample grid"
        )
    return run_sweep(cfg, progress=progress)
