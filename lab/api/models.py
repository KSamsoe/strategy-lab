"""The typed JSON contract. Every route declares one of these as its
``response_model``, which is what makes ``/api/openapi.json`` a real schema the
console's TypeScript client can be generated from instead of a guess.

Two shape rules worth stating, because both are load-bearing:

* Time series are **parallel arrays**, not arrays of objects. An equity curve is
  2,000 points and a decision tape can be tens of thousands; ``{t:[], equity:[]}``
  is roughly a third of the bytes of ``[{t,equity},...]`` and it is the column
  layout the charting library wants anyway.
* Anything with an age -- heartbeats, quota, breaker -- carries ``age_s``
  computed at request time. Freshness is displayed, never assumed, so the wire
  format never lets the UI show a timestamp without its staleness.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- health -------------------------------------------------------------------


class KillSwitchState(BaseModel):
    engaged: bool
    reason: str | None = None


class Health(BaseModel):
    """Unauthenticated liveness probe. Degrades to ``ok=false`` with a reason
    rather than a 500: a health endpoint that cannot answer when the thing it
    reports on is broken is useless."""

    ok: bool
    version: str
    now: datetime
    auth_required: bool
    console_built: bool
    kill_switch: KillSwitchState
    runs: int = 0
    events: int = 0
    latest_seq: int = 0
    detail: str = ""


# --- runs ---------------------------------------------------------------------


class RunSummary(BaseModel):
    run_id: str
    strategy: str
    kind: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    start: datetime | None = None
    end: datetime | None = None
    origin: str = "human"
    attempt: int = 0
    git_commit: str | None = None
    config_hash: str = ""
    data_version: str = ""
    sweep_id: str | None = None
    parent_run_id: str | None = None
    notes: str = ""
    error: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    name: str
    bytes: int
    modified: datetime


class RunDetail(RunSummary):
    params: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str
    artifacts: list[Artifact] = Field(default_factory=list)
    decisions: int = 0
    #: Mirrored out of ``metrics`` so the UI cannot render a run without the
    #: caveats attached: same-bar fills are optimistic, and an LLM strategy
    #: backtested over its own training window is contaminated.
    optimistic_fills: bool = False
    contaminated: bool = False
    warnings: list[str] = Field(default_factory=list)
    #: This run predates the trade-ledger fix and held open positions at the
    #: end, so its trade-level stats counted only round trips that returned
    #: exactly to flat. Its equity curve and return are unaffected.
    stale_ledger: bool = False
    #: Non-zero means P&L the trade ledger cannot account for.
    ledger_residual: float | None = None


class RunList(BaseModel):
    count: int
    limit: int
    offset: int
    runs: list[RunSummary]


class EquitySeries(BaseModel):
    """Columns, not points -- see the module docstring."""

    run_id: str
    n: int
    #: ISO-8601 UTC, matching every other timestamp on the wire.
    t: list[str]
    #: ``null`` rather than NaN: bare NaN is not JSON and breaks ``JSON.parse``.
    equity: list[float | None]
    drawdown: list[float | None]
    is_oos: list[bool]
    exposure: list[float | None] | None = None
    #: The run's benchmark, rebased onto the same starting equity so the two
    #: lines are readable on one axis. Absent when the run named no benchmark
    #: or its bars are not in the store.
    benchmark: list[float | None] | None = None


class TradeRow(BaseModel):
    ticker: str
    side: str = ""
    qty: float = 0.0
    entry_time: datetime | None = None
    entry_price: float | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    commission: float = 0.0
    tag: str = ""
    exit_reason: str = ""


class TradeList(BaseModel):
    run_id: str
    count: int
    trades: list[TradeRow]


class BarSeries(BaseModel):
    run_id: str
    ticker: str
    timeframe: str
    n: int
    t: list[str]
    open: list[float | None]
    high: list[float | None]
    low: list[float | None]
    close: list[float | None]
    volume: list[float | None]


# --- the decision tape ---------------------------------------------------------


class IntentOut(BaseModel):
    """``extra='allow'`` on the tape's nested records: the journal stores
    whatever the engine emitted, and dropping an unrecognized field on the floor
    would silently amputate the forensics this screen exists for."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    target_pct: float
    tag: str = ""
    reason: str = ""
    limit_price: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class VerdictOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    action: str
    requested_pct: float
    approved_pct: float
    rule: str | None = None
    detail: str = ""


class DecisionDetail(BaseModel):
    """The expanded half of a tape row: everything the Context served, what the
    strategy wanted, what the gate said about each want, and what was sent."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    intents: list[IntentOut] = Field(default_factory=list)
    verdicts: list[VerdictOut] = Field(default_factory=list)
    order_ids: list[str] = Field(default_factory=list)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    logs: list[dict[str, Any]] = Field(default_factory=list)
    agent: dict[str, Any] | None = None
    duration_ms: float = 0.0


class DecisionRow(BaseModel):
    id: str
    run_id: str
    strategy: str = ""
    at: datetime
    #: The collapsed one-liner the tape renders before you expand it.
    summary: str = ""
    tickers: list[str] = Field(default_factory=list)
    n_intents: int = 0
    n_orders: int = 0
    blocked: bool = False
    clipped: bool = False
    detail: DecisionDetail


class DecisionPage(BaseModel):
    run_id: str
    #: Decisions in the run, unfiltered -- the denominator for "row 812 of 3,441".
    total: int
    count: int
    limit: int
    offset: int
    has_more: bool
    rows: list[DecisionRow]


class CompareTable(BaseModel):
    """Column-oriented on purpose: the compare grid highlights the best cell per
    column, and params that are identical across the set are dropped upstream so
    the two knobs that actually moved are not buried."""

    ids: list[str]
    columns: list[str]
    rows: list[dict[str, Any]]


class Truncation(BaseModel):
    """How much of the grid actually ran.

    Served even when nothing was capped, because a silent truncation reads as
    "we tried everything" when we did not.
    """

    applied: bool = False
    requested: int = 0
    ran: int = 0


class SweepRunRow(BaseModel):
    """One grid point, with its in-sample and out-of-sample halves side by side.

    ``run_sweep`` stores these flattened (``is_sharpe``, ``oos_sharpe``); they
    are re-nested here so a caller can compare the two halves without knowing
    the prefix convention.
    """

    run_id: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    score: float | None = None
    is_metrics: dict[str, Any] = Field(default_factory=dict)
    oos_metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class WalkForwardWindow(BaseModel):
    index: int = 0
    is_start: str | None = None
    is_end: str | None = None
    oos_start: str | None = None
    oos_end: str | None = None
    is_metrics: dict[str, Any] = Field(default_factory=dict)
    oos_metrics: dict[str, Any] = Field(default_factory=dict)


class SweepDetail(BaseModel):
    """Everything the sweep screen needs to be honest about a grid search.

    ``ranked_on`` and ``attempts`` are not decoration: ranking on an in-sample
    metric is not validation, and the attempt count is what makes "someone ran
    400 variations and kept one" impossible to miss.
    """

    sweep_id: str
    strategy: str = ""
    metric: str
    count: int
    attempts: int
    ranked_on: str = "in_sample"
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    truncated: Truncation | None = None
    best: SweepRunRow | None = None
    runs: list[SweepRunRow] = Field(default_factory=list)
    walk_forward: list[WalkForwardWindow] = Field(default_factory=list)


class StrategyInfo(BaseModel):
    name: str
    path: str
    loadable: bool = False
    doc: str = ""
    source_hash: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class StrategyList(BaseModel):
    count: int
    strategies: list[StrategyInfo]


# --- agent --------------------------------------------------------------------


class AgentCall(BaseModel):
    id: str
    run_id: str | None = None
    at: datetime
    strategy: str = ""
    model: str = ""
    tool: str = ""
    prompt: str = ""
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ok: bool = True
    error: str = ""


class AgentCallList(BaseModel):
    count: int
    total_cost_usd: float
    total_tokens: int
    calls: list[AgentCall]


class LineageStep(BaseModel):
    n: int
    run_id: str
    created_at: datetime
    parent_run_id: str | None = None
    attempt: int = 0
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ResearchLaunchOptions(BaseModel):
    """What the console's launch form may offer, and the ceilings it cannot exceed."""

    enabled: bool = False
    reason: str = ""
    configs: list[str] = Field(default_factory=list)
    providers: list[dict[str, Any]] = Field(default_factory=list)
    default_model: str = ""
    default_provider: str = "auto"
    kill_switch_engaged: bool = False
    kill_switch_reason: str | None = None
    limits: dict[str, Any] = Field(default_factory=dict)


class ResearchStartRequest(BaseModel):
    """Deliberately cannot express a backtest config -- only which existing one
    to use. Composing `limits` here would be raising a limit from the UI."""

    config: str
    brief: str
    minutes: float = 30.0
    budget_usd: float = 5.0
    max_calls: int = 60
    max_experiments: int = 40
    metric: str = "oos_sharpe"
    periods: int = 4
    #: Ceiling on backtests spent perturbing the winning parameters before the
    #: session may finish -- never more than two per numeric parameter. Wall
    #: clock, not model budget. 0 disables the check.
    neighbourhood_runs: int = 16
    model: str | None = None
    provider: str | None = None
    session_id: str | None = None


class ResearchStartResult(BaseModel):
    ok: bool
    session_id: str
    pid: int | None = None
    config: str = ""
    minutes: float = 0.0
    budget_usd: float = 0.0
    max_calls: int = 0
    max_experiments: int = 0
    seq: int | None = None
    at: datetime | None = None
    message: str = ""


class ResearchStopResult(BaseModel):
    ok: bool
    session_id: str
    seq: int | None = None
    at: datetime | None = None
    message: str = ""


class ResearchSummary(BaseModel):
    session_id: str
    status: str = "running"
    stopped_because: str = ""
    brief: str = ""
    created_at: str = ""
    updated_at: str = ""
    elapsed_minutes: float = 0.0
    calls: int = 0
    spend_usd: float = 0.0
    billing: str = "api"
    experiments: int = 0
    best_run_id: str | None = None
    best_score: float | None = None
    satisfied_with: str | None = None


class ResearchDetail(ResearchSummary):
    """The whole session: what it tried, what it found, and why it said so."""

    verdict: str = ""
    next_steps: str = ""
    notes: list[str] = Field(default_factory=list)
    market: dict[str, Any] | None = None
    #: One row per backtest, newest last.
    runs: list[dict[str, Any]] = Field(default_factory=list)
    #: One row per turn, including rejected actions -- the reasoning trail.
    turns: list[dict[str, Any]] = Field(default_factory=list)
    workspace: str = ""
    resumable: bool = True


class ResearchList(BaseModel):
    count: int
    sessions: list[ResearchSummary]


class Lineage(BaseModel):
    """The "is the agent converging or just churning" payload: a metric
    trajectory plus the attempt count it took to get there."""

    strategy: str
    metric: str
    count: int
    attempts: int
    best_run_id: str | None = None
    steps: list[LineageStep]


# --- live ---------------------------------------------------------------------


class LiveStrategy(BaseModel):
    strategy: str
    run_id: str | None = None
    kind: str = "paper"
    status: str = "stopped"
    paused: bool = False
    pid: int | None = None
    host: str = ""
    started_at: datetime | None = None
    updated_at: datetime | None = None
    next_fire: datetime | None = None
    notes: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class LiveStrategyList(BaseModel):
    count: int
    strategies: list[LiveStrategy]


class EventRow(BaseModel):
    seq: int
    id: str
    at: datetime
    kind: str
    source: str
    run_id: str | None = None
    strategy: str | None = None
    ticker: str | None = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class EventPage(BaseModel):
    since: int
    latest_seq: int
    count: int
    events: list[EventRow]


class AdapterState(BaseModel):
    name: str
    provides: list[str] = Field(default_factory=list)
    available: bool = False
    reason: str = ""
    quota_used: int | None = None
    quota_limit: int | None = None
    quota_tier: str | None = None
    last_call: datetime | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Heartbeat(BaseModel):
    source: str
    at: datetime
    age_s: float
    stale: bool
    seq: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class QuotaState(BaseModel):
    source: str = "govgreed"
    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    tier: str | None = None
    reset_at: datetime | None = None
    at: datetime | None = None
    age_s: float | None = None
    #: Where the numbers came from: the quota log, the adapter, or nowhere yet.
    origin: Literal["quota_log", "adapter", "unknown"] = "unknown"


class BreakerState(BaseModel):
    tripped: bool = False
    reason: str = ""
    strategy: str | None = None
    at: datetime | None = None
    age_s: float | None = None


class LiveHealth(BaseModel):
    now: datetime
    stale_after_s: float
    adapters: list[AdapterState]
    heartbeats: list[Heartbeat]
    quota: QuotaState
    breaker: BreakerState
    kill_switch: KillSwitchState
    strategies: int = 0
    latest_seq: int = 0


# --- controls (the safer-only half) --------------------------------------------


class StrategyControlRequest(BaseModel):
    strategy: str = Field(min_length=1)
    reason: str = ""


class KillRequest(BaseModel):
    reason: str = ""


class ControlResult(BaseModel):
    """``applied`` is what the API itself changed; ``seq`` is the journal entry
    the runner will act on. Cancelling orders is a *request* recorded in the
    journal, because the process holding the broker session is the only thing
    allowed to talk to the broker."""

    ok: bool
    action: Literal["pause", "cancel_orders", "kill"]
    target: str | None = None
    applied: bool
    seq: int
    at: datetime
    message: str = ""


# --- websocket frames ----------------------------------------------------------


class WSHello(BaseModel):
    """First frame. Tells the client which cursor the server actually resumed
    from, so a reconnect can detect a gap instead of silently losing events."""

    type: Literal["hello"] = "hello"
    since: int
    latest_seq: int
    at: datetime


class WSEvent(BaseModel):
    type: Literal["event"] = "event"
    seq: int
    event: EventRow
