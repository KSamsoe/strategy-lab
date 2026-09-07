"""Computed indicators: pure functions from an OHLCV frame to a series.

Two properties are load-bearing here.

**Causality.** The value at row *i* is a function of rows ``<= i`` and nothing
else. Every warm-up row is ``NaN`` and is never back-filled, because a
back-filled warm-up is look-ahead wearing a hat: it hands the strategy a number
on a bar where the number could not have existed. ``tests/test_indicators.py``
proves this mechanically for every registered name by recomputing over a longer
frame and demanding a bit-identical prefix, so a new indicator cannot join the
registry without satisfying it.

**No pandas-ta.** The design doc names it; we deviate deliberately. It is
unmaintained against numpy 2.x and drags a heavy import into a hot path for a
few dozen lines of arithmetic we can own outright. Everything below is plain
pandas/numpy.

Indicators are registered by name so sweeps, the cache, and ``ctx.indicator``
can address them as data rather than as imports.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

#: name -> indicator function. The single source of truth for what exists;
#: iterate it rather than maintaining a parallel list anywhere.
INDICATORS: dict[str, Callable[..., pd.Series | pd.DataFrame]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: add an indicator to the registry under ``name``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("indicator name must be a non-empty string")
    if key in INDICATORS:
        raise ValueError(f"indicator {key!r} is already registered")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        INDICATORS[key] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(INDICATORS)


def compute(name: str, df: pd.DataFrame, **params: Any) -> pd.Series | pd.DataFrame:
    """Run indicator ``name`` over the OHLCV frame ``df``.

    Unknown parameters are a ``ValueError`` rather than a silent no-op: in a
    parameter sweep a typo that is ignored produces a grid of identical runs
    and a very confident wrong conclusion.
    """
    key = str(name).strip().lower()
    fn = INDICATORS.get(key)
    if fn is None:
        raise ValueError(f"unknown indicator {name!r}; have {available()}")
    frame = _guard(df)
    unknown = set(params) - _param_names(fn)
    if unknown:
        raise ValueError(
            f"indicator {key!r} takes no parameter(s) {sorted(unknown)}; "
            f"accepts {sorted(_param_names(fn))}"
        )
    return fn(frame, **params)


# --- input handling ----------------------------------------------------------


@lru_cache(maxsize=None)
def _param_names(fn: Callable[..., Any]) -> frozenset[str]:
    sig = inspect.signature(fn)
    return frozenset(list(sig.parameters)[1:])  # skip the frame


def _guard(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"expected a DataFrame of bars, got {type(df).__name__}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "bar index must be sorted ascending; causality is meaningless on an "
            "unsorted frame"
        )
    return df


def _col(df: pd.DataFrame, field: str) -> pd.Series:
    """Fetch one column as float64, matching case-insensitively."""
    if field in df.columns:
        s = df[field]
    else:
        lookup = {str(c).lower(): c for c in df.columns}
        col = lookup.get(str(field).lower())
        if col is None:
            raise ValueError(
                f"frame has no column {field!r}; columns are {[str(c) for c in df.columns]}"
            )
        s = df[col]
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _need(df: pd.DataFrame, *fields: str) -> tuple[pd.Series, ...]:
    return tuple(_col(df, f) for f in fields)


def _win(n: Any, *, name: str = "n", minimum: int = 1) -> int:
    """Validate a lookback length. Sweeps hand these in from YAML, so they
    arrive as whatever the parser felt like producing."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        if isinstance(n, (float, np.floating)) and float(n).is_integer():
            n = int(n)
        else:
            raise ValueError(f"{name} must be an integer, got {n!r}")
    n = int(n)
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {n}")
    return n


def _num(x: Any, *, name: str, minimum: float | None = None) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


# --- shared math -------------------------------------------------------------


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing: SMA seed at the first full window, then recursive.

    Written out instead of reached for via ``ewm`` because Wilder seeds with an
    SMA while pandas seeds with the first observation. The two disagree over the
    whole warm-up and forever after by a decaying amount, and RSI/ATR/ADX
    published values assume Wilder's seed.
    """
    v = s.to_numpy(dtype="float64", copy=False)
    out = np.full(v.shape[0], np.nan)
    finite = np.flatnonzero(np.isfinite(v))
    if finite.size == 0:
        return pd.Series(out, index=s.index, dtype="float64")
    start = int(finite[0])
    seed = start + n - 1
    if seed >= v.shape[0]:
        return pd.Series(out, index=s.index, dtype="float64")
    acc = float(v[start : seed + 1].mean())
    out[seed] = acc
    for i in range(seed + 1, v.shape[0]):
        acc = (acc * (n - 1) + v[i]) / n
        out[i] = acc
    return pd.Series(out, index=s.index, dtype="float64")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    # skipna on the row-wise max makes the first bar fall back to high-low,
    # which is the conventional seed and keeps ATR's warm-up at n rows.
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return tr.max(axis=1, skipna=True).astype("float64")


def _frame(index: pd.Index, **cols: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(cols, index=index)
    return out.astype("float64")


# --- trend / average ---------------------------------------------------------


@register("sma")
def sma(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """Simple moving average."""
    n = _win(n)
    s = _col(df, field)
    return s.rolling(n, min_periods=n).mean().rename("sma")


@register("ema")
def ema(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """Exponential moving average, alpha = 2/(n+1).

    Seeded at the first observation (``adjust=False``) and masked until ``n``
    rows exist, so the reported warm-up matches the other windowed indicators
    even though the recursion itself has no warm-up.
    """
    n = _win(n)
    s = _col(df, field)
    return s.ewm(span=n, adjust=False, min_periods=n).mean().rename("ema")


@register("wma")
def wma(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """Linearly weighted moving average; weight i for the i-th oldest bar."""
    n = _win(n)
    s = _col(df, field)
    w = np.arange(1, n + 1, dtype="float64")
    denom = float(w.sum())
    return (
        s.rolling(n, min_periods=n)
        .apply(lambda x: float(np.dot(x, w)) / denom, raw=True)
        .rename("wma")
    )


@register("slope")
def slope(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """OLS slope of ``field`` against bar number over a trailing window.

    Units are price per bar, so it is comparable across tickers only after
    normalizing by price.
    """
    n = _win(n, minimum=2)
    s = _col(df, field)
    x = np.arange(n, dtype="float64")
    xc = x - x.mean()
    denom = float(xc @ xc)
    return (
        s.rolling(n, min_periods=n)
        .apply(lambda w: float(xc @ w) / denom, raw=True)
        .rename("slope")
    )


# --- oscillators -------------------------------------------------------------


@register("rsi")
def rsi(df: pd.DataFrame, n: int = 14, field: str = "close") -> pd.Series:
    """Wilder's RSI.

    A window with no losses pins at 100; a perfectly flat window has no
    information either way and reports the neutral 50 rather than 100.
    """
    n = _win(n)
    s = _col(df, field)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain, avg_loss = _rma(gain, n), _rma(loss, n)
    rs = avg_gain / avg_loss.mask(avg_loss == 0.0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return out.where(avg_gain.notna() & avg_loss.notna()).rename("rsi")


@register("stoch")
def stoch(df: pd.DataFrame, n: int = 14, d: int = 3) -> pd.DataFrame:
    """Stochastic oscillator: fast %K and its ``d``-bar average %D.

    A flat window (high == low across all n bars) has an undefined position in
    range; reported as the neutral 50 so one dead window does not punch a hole
    through the middle of the series.
    """
    n, d = _win(n), _win(d, name="d")
    high, low, close = _need(df, "high", "low", "close")
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    rng = hh - ll
    k = 100.0 * (close - ll) / rng.mask(rng == 0.0)
    k = k.where(rng != 0.0, 50.0)
    dd = k.rolling(d, min_periods=d).mean()
    return _frame(df.index, k=k, d=dd)


@register("adx")
def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX. Warm-up is 2n-1 rows: n for the DI pair, n more to smooth
    DX into ADX."""
    n = _win(n)
    high, low, close = _need(df, "high", "low", "close")
    up = high.diff()
    down = -low.diff()
    plus = up.where((up > down) & (up > 0.0), 0.0).mask(up.isna() | down.isna())
    minus = down.where((down > up) & (down > 0.0), 0.0).mask(up.isna() | down.isna())
    atr_ = _rma(_true_range(high, low, close), n)
    denom = atr_.mask(atr_ == 0.0)
    plus_di = 100.0 * _rma(plus, n) / denom
    minus_di = 100.0 * _rma(minus, n) / denom
    di_sum = (plus_di + minus_di).mask(lambda x: x == 0.0)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return _rma(dx, n).rename("adx")


# --- volatility / range ------------------------------------------------------


@register("atr")
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range, Wilder-smoothed."""
    n = _win(n)
    high, low, close = _need(df, "high", "low", "close")
    return _rma(_true_range(high, low, close), n).rename("atr")


@register("bbands")
def bbands(df: pd.DataFrame, n: int = 20, k: float = 2.0, field: str = "close") -> pd.DataFrame:
    """Bollinger bands. Population standard deviation (ddof=0), as Bollinger
    defined them; ddof=1 shifts the bands by a few basis points at small n."""
    n = _win(n, minimum=2)
    k = _num(k, name="k", minimum=0.0)
    s = _col(df, field)
    mid = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return _frame(df.index, lower=mid - k * sd, mid=mid, upper=mid + k * sd)


@register("donchian")
def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Donchian channel over the last ``n`` bars, **inclusive of the current
    bar**. A breakout rule wants ``.shift(1)`` on top of this, otherwise the
    channel touches price by construction on every new extreme."""
    n = _win(n)
    high, low = _need(df, "high", "low")
    return _frame(
        df.index,
        upper=high.rolling(n, min_periods=n).max(),
        lower=low.rolling(n, min_periods=n).min(),
    )


@register("rolling_vol")
def rolling_vol(
    df: pd.DataFrame,
    n: int = 20,
    annualize: bool = True,
    periods: int = 252,
    field: str = "close",
) -> pd.Series:
    """Standard deviation of simple returns, optionally annualized.

    ``periods`` is the bar count per year; pass 252*390 for minute bars. Sample
    std (ddof=1) because this is an estimate of a population we cannot see.
    """
    n = _win(n, minimum=2)
    periods = _win(periods, name="periods")
    s = _col(df, field)
    vol = s.pct_change().rolling(n, min_periods=n).std(ddof=1)
    if annualize:
        vol = vol * float(np.sqrt(periods))
    return vol.rename("rolling_vol")


@register("max_drawdown")
def max_drawdown(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """Worst peak-to-trough decline inside the trailing ``n`` bars, as a
    negative fraction. Only the window is considered, so this is a local
    roughness measure, not the run's drawdown."""
    n = _win(n, minimum=2)
    s = _col(df, field)

    def _mdd(w: np.ndarray) -> float:
        if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            return float("nan")
        return float(np.min(w / np.maximum.accumulate(w) - 1.0))

    return s.rolling(n, min_periods=n).apply(_mdd, raw=True).rename("max_drawdown")


# --- momentum / returns ------------------------------------------------------


@register("returns")
def returns(df: pd.DataFrame, n: int = 1, log: bool = False, field: str = "close") -> pd.Series:
    """``n``-bar return as a fraction (or log return when ``log``)."""
    n = _win(n)
    s = _col(df, field)
    prev = s.shift(n)
    if log:
        ratio = (s / prev.mask(prev <= 0.0)).mask(s <= 0.0)
        out = pd.Series(np.log(ratio.to_numpy(dtype="float64")), index=s.index)
    else:
        out = s / prev.mask(prev == 0.0) - 1.0
    return out.astype("float64").rename("returns")


@register("roc")
def roc(df: pd.DataFrame, n: int = 10, field: str = "close") -> pd.Series:
    """Rate of change in **percent**. ``returns`` is the same thing as a
    fraction; both exist because strategy code reads better with one or the
    other and silently mixing them costs a factor of 100."""
    n = _win(n)
    s = _col(df, field)
    prev = s.shift(n)
    return (100.0 * (s / prev.mask(prev == 0.0) - 1.0)).rename("roc")


@register("momentum")
def momentum(df: pd.DataFrame, n: int = 10, field: str = "close") -> pd.Series:
    """Absolute price change over ``n`` bars, in price units."""
    n = _win(n)
    s = _col(df, field)
    return (s - s.shift(n)).rename("momentum")


@register("zscore")
def zscore(df: pd.DataFrame, n: int = 20, field: str = "close") -> pd.Series:
    """Rolling z-score against the trailing window's own mean and population
    std. Degenerate (zero-variance) windows yield NaN rather than infinity."""
    n = _win(n, minimum=2)
    s = _col(df, field)
    mean = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return ((s - mean) / sd.mask(sd == 0.0)).rename("zscore")


# --- volume ------------------------------------------------------------------


@register("vwap")
def vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Rolling volume-weighted average of the typical price over ``n`` bars.

    Deliberately not the session-anchored VWAP a trader would quote: this is an
    indicator over a fixed lookback, so it behaves the same on daily bars as on
    minute bars and needs no session boundaries.
    """
    n = _win(n)
    high, low, close, volume = _need(df, "high", "low", "close", "volume")
    typical = (high + low + close) / 3.0
    num = (typical * volume).rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum()
    return (num / den.mask(den == 0.0)).rename("vwap")


# --- composites --------------------------------------------------------------


@register("macd")
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    field: str = "close",
) -> pd.DataFrame:
    """MACD line, its signal EMA, and the histogram between them."""
    fast = _win(fast, name="fast")
    slow = _win(slow, name="slow")
    signal = _win(signal, name="signal")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    s = _col(df, field)
    line = (
        s.ewm(span=fast, adjust=False, min_periods=slow).mean()
        - s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    )
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return _frame(df.index, macd=line, signal=sig, hist=line - sig)


def default_params(name: str) -> dict[str, Any]:
    """The registered defaults for ``name``. Used by the causality test to walk
    the registry, and handy for the console's parameter forms."""
    key = str(name).strip().lower()
    fn = INDICATORS.get(key)
    if fn is None:
        raise ValueError(f"unknown indicator {name!r}; have {available()}")
    sig = inspect.signature(fn)
    return {
        p.name: p.default
        for p in list(sig.parameters.values())[1:]
        if p.default is not inspect.Parameter.empty
    }


def required_columns(name: str) -> frozenset[str]:
    """Best-effort declaration of which OHLCV columns ``name`` reads."""
    key = str(name).strip().lower()
    if key not in INDICATORS:
        raise ValueError(f"unknown indicator {name!r}; have {available()}")
    if key in {"adx", "atr", "stoch"}:
        return frozenset({"high", "low", "close"})
    if key == "donchian":
        return frozenset({"high", "low"})
    if key == "vwap":
        return frozenset({"high", "low", "close", "volume"})
    return frozenset({"close"})


__all__: Iterable[str] = [
    "INDICATORS",
    "available",
    "compute",
    "default_params",
    "register",
    "required_columns",
]
