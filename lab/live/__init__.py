"""Live and paper execution: the backtester's loop against a real broker.

Import order below is load-bearing -- ``runner`` imports ``alerts`` and
``killswitch`` from this package, so they must be bound first.

Paper is the default and the only tested path; promotion to live capital is a
manual human decision and is not automated anywhere in here.
"""

from __future__ import annotations

from lab.live import killswitch  # noqa: I001 - see module docstring
from lab.live import alerts
from lab.live.reconcile import ReconcileResult, reconcile
from lab.live.runner import LiveConfig, LiveRunner

__all__ = [
    "LiveConfig",
    "LiveRunner",
    "ReconcileResult",
    "alerts",
    "killswitch",
    "reconcile",
]
