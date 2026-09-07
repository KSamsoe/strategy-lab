"""Adapter contract and registry.

An adapter's whole job is to turn some external source into ``Bar`` or ``Event``
records carrying both timestamps, and to be honest about whether it can run at
all. Missing credentials are a *state*, not an exception: ``available()``
returns ``(False, reason)`` so `lab adapters` can print a useful table instead of
a stack trace.

Upgrades (Polygon, Databento, another alt-data vendor) slot in here as new
adapters without touching anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Sequence

from lab.store.schema import Bar, Event


class AdapterError(RuntimeError):
    """Anything an adapter can fail at that the caller might handle."""


class AdapterUnavailable(AdapterError):
    """The adapter cannot run: no credentials, missing optional dependency."""


class QuotaExceeded(AdapterError):
    """A hard vendor quota was hit. Callers should stop, not retry."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class AdapterInfo:
    name: str
    provides: frozenset[str]
    available: bool
    reason: str = ""
    quota_used: int | None = None
    quota_limit: int | None = None
    quota_tier: str | None = None
    last_call: datetime | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provides": sorted(self.provides),
            "available": self.available,
            "reason": self.reason,
            "quota_used": self.quota_used,
            "quota_limit": self.quota_limit,
            "quota_tier": self.quota_tier,
            "last_call": self.last_call.isoformat() if self.last_call else None,
            "detail": self.detail,
        }


class BaseAdapter:
    """Subclass, set ``name``/``provides``, implement what you provide."""

    name: str = ""
    provides: frozenset[str] = frozenset()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self) -> AdapterInfo:
        ok, reason = self.available()
        return AdapterInfo(
            name=self.name, provides=self.provides, available=ok, reason=reason
        )

    def fetch_bars(
        self,
        tickers: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        raise NotImplementedError(f"{self.name} does not provide bars")

    def fetch_signals(
        self, start: datetime | None = None, end: datetime | None = None, **query: Any
    ) -> Iterator[Event]:
        raise NotImplementedError(f"{self.name} does not provide signals")

    # -- convenience ---------------------------------------------------------
    def require_available(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise AdapterUnavailable(f"{self.name}: {reason}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} provides={sorted(self.provides)}>"


#: Populated lazily by :func:`_load_registry` so that importing this module does
#: not drag in optional third-party packages.
REGISTRY: dict[str, type[BaseAdapter]] = {}

_KNOWN = {
    "synthetic": ("lab.adapters.synthetic", "SyntheticAdapter"),
    "yfinance": ("lab.adapters.yfinance", "YFinanceAdapter"),
    "alpaca": ("lab.adapters.alpaca", "AlpacaAdapter"),
    "govgreed": ("lab.adapters.govgreed", "GovGreedAdapter"),
}


def _load_registry() -> dict[str, type[BaseAdapter]]:
    import importlib

    for key, (module_name, class_name) in _KNOWN.items():
        if key in REGISTRY:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:  # a broken optional adapter must not break the rest
            continue
        cls = getattr(module, class_name, None)
        if isinstance(cls, type):
            REGISTRY[key] = cls
    return REGISTRY


def get_adapter(name: str, **kwargs: Any) -> BaseAdapter:
    registry = _load_registry()
    key = name.strip().lower()
    if key not in registry:
        known = ", ".join(sorted(registry)) or "none loaded"
        raise ValueError(f"unknown adapter {name!r}; known: {known}")
    return registry[key](**kwargs)


def list_adapters() -> list[AdapterInfo]:
    infos: list[AdapterInfo] = []
    for key, cls in sorted(_load_registry().items()):
        try:
            infos.append(cls().info())
        except Exception as exc:  # never let one bad adapter hide the others
            infos.append(
                AdapterInfo(
                    name=key,
                    provides=getattr(cls, "provides", frozenset()),
                    available=False,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return infos
