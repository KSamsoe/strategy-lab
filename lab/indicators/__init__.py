"""The indicator layer: pure functions from stored data to a series.

Two flavors, one timeline. :mod:`lab.indicators.computed` derives series from
OHLCV bars; :mod:`lab.indicators.fetched` places already-scored alt-data onto
the bar index by ``knowledge_time``. Both are causal by construction, and
:mod:`lab.indicators.cache` memoizes results on
``(ticker, indicator, params, data_version)`` so sweeps never recompute what has
not changed.

Nothing here maintains streaming state. Live mode recomputes over a rolling
lookback each bar: mathematically identical to an incremental update,
dramatically simpler, and cheap at daily-to-minute cadence.
"""

from __future__ import annotations

from lab.indicators.cache import cache_key, cached_indicator
from lab.indicators.computed import (
    INDICATORS,
    available,
    compute,
    default_params,
    register,
    required_columns,
)
from lab.indicators.fetched import align, series

__all__ = [
    "INDICATORS",
    "align",
    "available",
    "cache_key",
    "cached_indicator",
    "compute",
    "default_params",
    "register",
    "required_columns",
    "series",
]
