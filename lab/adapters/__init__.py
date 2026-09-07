"""Market-data and alt-data adapters.

Importing this package is cheap and always succeeds: every optional third-party
dependency (``yfinance``, ``alpaca-py``, ``httpx`` clients) is imported inside
the function that needs it, so a machine with none of them installed can still
run ``lab adapters`` and be told exactly what is missing.

``govgreed`` is deliberately absent from these re-exports -- it is resolved
through :func:`lab.adapters.base.get_adapter`, which tolerates an adapter module
that is missing or broken instead of taking the whole registry down with it.
"""

from __future__ import annotations

from lab.adapters.alpaca import AlpacaAdapter
from lab.adapters.base import (
    REGISTRY,
    AdapterError,
    AdapterInfo,
    AdapterUnavailable,
    BaseAdapter,
    QuotaExceeded,
    get_adapter,
    list_adapters,
)
from lab.adapters.synthetic import SyntheticAdapter
from lab.adapters.yfinance import YFinanceAdapter

__all__ = [
    "REGISTRY",
    "AdapterError",
    "AdapterInfo",
    "AdapterUnavailable",
    "AlpacaAdapter",
    "BaseAdapter",
    "QuotaExceeded",
    "SyntheticAdapter",
    "YFinanceAdapter",
    "get_adapter",
    "list_adapters",
]
