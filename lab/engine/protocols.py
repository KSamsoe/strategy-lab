"""The strategy interface, and the adapter/broker/clock seams around it.

Three rules give this interface its value:

1. The ``Context`` object is the **only** way a strategy touches data, and it
   structurally refuses to serve anything past ``ctx.now``. Look-ahead becomes a
   bug the engine can catch rather than a discipline the author must maintain.
2. Strategies emit *intents* (target percentages), never raw orders. Translation
   into orders, and every safety check, happens outside strategy code.
3. ``params`` is a plain declared mapping, which is what makes sweeps,
   config-driven iteration, and agent-authored variation trivial.

Everything here is a ``Protocol``: strategies are duck-typed, so a strategy file
is just a module exposing a class (or ``STRATEGY``/``build()``) with these
methods. No base class to inherit, nothing to register.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from lab.engine.events import Fill, Intent, Order, Position
from lab.store.schema import Event


class LookAheadError(RuntimeError):
    """Raised when something asks the context for data it could not have known.

    This is deliberately loud. A silent clamp would let a subtly wrong strategy
    produce a plausibly wrong backtest, which is worse than a crash.
    """


@runtime_checkable
class PortfolioView(Protocol):
    """Read-only portfolio state as a strategy sees it."""

    @property
    def cash(self) -> float: ...

    @property
    def equity(self) -> float: ...

    @property
    def positions(self) -> Mapping[str, Position]: ...

    def position(self, ticker: str) -> Position: ...

    def weight(self, ticker: str) -> float:
        """Current market value of ``ticker`` as a fraction of equity."""
        ...

    @property
    def gross_exposure(self) -> float: ...


@runtime_checkable
class Context(Protocol):
    """Everything a strategy is allowed to see and do.

    All read methods serve only rows with ``knowledge_time <= ctx.now``.
    """

    #: Declared strategy parameters. Plain data, so sweeps can vary them.
    params: Mapping[str, Any]
    #: Current engine time (UTC, aware). Sim clock or wall clock; same type.
    now: datetime
    #: The trading session ``now`` belongs to, in Eastern terms.
    session: date
    #: The tickers this run is allowed to touch.
    universe: Sequence[str]
    #: Read-only portfolio state.
    portfolio: PortfolioView

    def history(
        self, ticker: str, field: str = "close", n: int = 100, timeframe: str | None = None
    ) -> pd.Series:
        """The last ``n`` values of ``field`` for ``ticker``, oldest first.

        Indexed by ``event_time``. Never includes a bar whose
        ``knowledge_time`` is after ``now``.
        """
        ...

    def bars(self, ticker: str, n: int = 100, timeframe: str | None = None) -> pd.DataFrame:
        """OHLCV frame for ``ticker``: the same slice ``history`` serves."""
        ...

    def price(self, ticker: str) -> float | None:
        """Latest known close for ``ticker``, or ``None`` if we have none yet."""
        ...

    def indicator(self, ticker: str, name: str, **params: Any) -> pd.Series:
        """A computed or fetched indicator series, aligned to the timeline."""
        ...

    def signals(self, source: str, **query: Any) -> list[Event]:
        """Alt-data events from ``source``, subject to the same time rule.

        Filtered by ``knowledge_time <= now``, which is the whole point: a
        congressional trade is invisible until its disclosure date, not its
        trade date.
        """
        ...

    def order_target_pct(
        self, ticker: str, pct: float, tag: str = "", reason: str = "", **meta: Any
    ) -> None:
        """Express a target weight. This records an intent; it places nothing."""
        ...

    def close(self, ticker: str, tag: str = "", reason: str = "") -> None:
        """Shorthand for ``order_target_pct(ticker, 0.0)``."""
        ...

    def log(self, **fields: Any) -> None:
        """Structured log line, captured into the decision journal."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """A strategy. Every hook is optional except ``on_bar``."""

    def on_start(self, ctx: Context) -> None: ...

    def on_bar(self, ctx: Context) -> None:
        """The decision point. Called once per bar, per strategy."""
        ...

    def on_fill(self, ctx: Context, fill: Fill) -> None: ...

    def on_stop(self, ctx: Context) -> None: ...


@runtime_checkable
class Clock(Protocol):
    """Sim clock or wall clock. The engine does not know which it has."""

    @property
    def now(self) -> datetime: ...

    def __iter__(self) -> Iterable[datetime]: ...


@runtime_checkable
class Broker(Protocol):
    """Sim broker or a real one. Same surface either way."""

    name: str

    def submit(self, order: Order) -> Order:
        """Place an order. Must be a no-op for a repeated ``idem_key``."""
        ...

    def cancel(self, order_id: str) -> bool: ...

    def open_orders(self) -> list[Order]: ...

    def positions(self) -> dict[str, Position]: ...

    def account(self) -> dict[str, Any]:
        """At minimum ``{"cash": float, "equity": float}``."""
        ...


@runtime_checkable
class DataAdapter(Protocol):
    """A source of bars, alt-data events, or both.

    Adapters normalize into ``Bar``/``Event`` records carrying both timestamps
    and hand them to the store. They persist raw responses verbatim first and
    normalize second, so the original stays re-parseable when schemas drift.
    """

    name: str
    #: What this adapter can produce: subset of {"bars", "signals"}.
    provides: frozenset[str]

    def available(self) -> tuple[bool, str]:
        """``(usable, reason)`` -- e.g. missing credentials is not an error."""
        ...

    def fetch_bars(
        self,
        tickers: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Iterable[Any]:
        """Yield ``lab.store.schema.Bar`` records."""
        ...

    def fetch_signals(
        self, start: datetime | None = None, end: datetime | None = None, **query: Any
    ) -> Iterable[Event]:
        """Yield ``lab.store.schema.Event`` records."""
        ...
