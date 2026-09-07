# Module contracts

The binding interface list for the lab. Every module below must expose exactly
these names with these signatures; anything else is free. Written down because
the modules are built in parallel and must compose on first import.

Already implemented and **frozen** — read them, do not change them:

- `lab/config.py` — `get_settings() -> Settings`, `Settings.paths` (`Paths` with
  `root data parquet raw runs cache cfg strategies registry_db journal_db`),
  `Settings.kill_switch_engaged() -> (bool, str|None)`, `reset_settings_cache()`
- `lab/timeutil.py` — `UTC ET utcnow to_utc to_et session_date is_trading_day
  trading_days next_trading_day session_open_utc session_close_utc
  parse_timeframe is_intraday market_holidays`
- `lab/store/schema.py` — `Bar Event BARS SIGNALS BARS_SCHEMA SIGNALS_SCHEMA
  SCHEMAS PARTITION_KEYS empty_frame`; both records expose `.as_row()`
- `lab/engine/events.py` — `Side OrderType OrderStatus GateAction EventKind
  Intent GateVerdict Order Fill Position Trade Decision JournalEvent event()
  new_id()`
- `lab/engine/protocols.py` — `Context Strategy Clock Broker DataAdapter
  PortfolioView LookAheadError`

## Conventions

- Python 3.12, `from __future__ import annotations`, full type hints.
- All datetimes are **aware UTC**. Use `lab.timeutil.to_utc` at every boundary.
- Tickers are upper-case everywhere.
- No module may import from a layer above it. Dependency order:
  `config/timeutil` → `store` → `indicators` → `engine` → `risk` → `backtest`
  → `live`/`agent`/`api`/`cli`.
- Optional third-party imports (`yfinance`, `alpaca`, `anthropic`, `vectorbt`)
  are imported **lazily inside functions**, never at module import time. A
  missing optional dep must degrade to a clear message, never an ImportError at
  startup.
- Public functions raise `ValueError` for bad input, not asserts.

---

## `lab/store/parquet_io.py`

```python
def write_bars(bars: Iterable[Bar], *, dedupe: bool = True) -> int
def write_events(events: Iterable[Event], *, dedupe: bool = True) -> int
def write_frame(table: str, df: pd.DataFrame, *, dedupe: bool = True) -> int
def read_bars(tickers: Sequence[str] | str | None = None, *, timeframe: str = "1d",
              start: datetime | None = None, end: datetime | None = None,
              as_of: datetime | None = None, source: str | None = None) -> pd.DataFrame
def read_signals(source: str | None = None, *, tickers: Sequence[str] | None = None,
                 start: datetime | None = None, end: datetime | None = None,
                 as_of: datetime | None = None, kind: str | None = None) -> pd.DataFrame
def data_version(table: str = BARS, *, tickers: Sequence[str] | None = None,
                 source: str | None = None) -> str
def coverage(table: str = BARS) -> pd.DataFrame   # source,ticker,timeframe,rows,first,last
def table_path(table: str) -> Path
```

- Layout: `data/parquet/<table>/source=<s>/ticker=<T>/year=<Y>/part-<n>.parquet`.
- `as_of` filters `knowledge_time <= as_of`. **This is the look-ahead barrier at
  the storage layer** and must be applied before any other filtering.
- `start`/`end` filter `event_time`.
- Dedupe key: bars `(source,ticker,timeframe,event_time)`, events
  `(source,uid)` — last write wins, so re-pulls restate rather than duplicate.
- `data_version` is a stable short hex digest (16 chars) of the contributing
  partition files' `(relative path, size, mtime_ns, row count)`. Same data ⇒
  same string, on any machine, in any order.
- Returned frames are sorted by `event_time` and carry the schema's columns.

## `lab/store/duck.py`

```python
def connect() -> duckdb.DuckDBPyConnection   # views `bars`, `signals` registered
def query(sql: str, params: Sequence | Mapping | None = None) -> pd.DataFrame
def register_views(con) -> None
def describe() -> pd.DataFrame
```

Views point at the parquet globs via `read_parquet(..., hive_partitioning=1)`
and tolerate an empty store (return zero rows, never raise).

## `lab/store/raw.py`

```python
def save(source: str, endpoint: str, payload: Any, *, request_id: str | None = None,
         fetched_at: datetime | None = None, params: Mapping | None = None) -> Path
def load(source: str, endpoint: str | None = None, *, since: datetime | None = None) -> list[dict]
def iter_raw(source: str, endpoint: str | None = None) -> Iterator[dict]
```

Verbatim persistence, before normalization, always. One gzipped JSON line per
response under `data/raw/<source>/<YYYY-MM-DD>/<endpoint>.jsonl.gz`, each line
`{"fetched_at","endpoint","request_id","params","payload"}`. This snapshot *is*
the alt-data backtest dataset, since vendor historical backfill is not available
on the free tier.

## `lab/indicators/computed.py`

```python
INDICATORS: dict[str, Callable[..., pd.Series | pd.DataFrame]]
def register(name: str) -> Callable      # decorator
def compute(name: str, df: pd.DataFrame, **params) -> pd.Series | pd.DataFrame
def available() -> list[str]
```

Pure pandas/numpy — no `pandas-ta` (dependency is unreliable on numpy 2.x).
Required indicators, all taking an OHLCV frame (`open high low close volume`,
indexed by `event_time`) plus params:

`sma(n)` `ema(n)` `wma(n)` `rsi(n=14)` `atr(n=14)` `bbands(n=20,k=2)` (frame:
`lower mid upper`) `macd(fast=12,slow=26,signal=9)` (frame: `macd signal hist`)
`roc(n)` `momentum(n)` `zscore(n)` `rolling_vol(n,annualize=True)`
`donchian(n)` (frame: `upper lower`) `returns(n=1,log=False)` `vwap(n)`
`adx(n=14)` `stoch(n=14,d=3)` `max_drawdown(n)` `slope(n)`

Every indicator must be **causal**: value at index *i* uses only rows ≤ *i*.
Leading warm-up rows are `NaN`, never back-filled. Add a test proving causality
(shifting future rows must not change past values).

## `lab/indicators/fetched.py`

```python
def align(events: pd.DataFrame, index: pd.DatetimeIndex, *, field: str = "score",
          agg: str = "last", ffill_limit: int | None = None) -> pd.Series
def series(source: str, ticker: str, index: pd.DatetimeIndex, *, field: str = "score",
           as_of: datetime | None = None, **query) -> pd.Series
```

Alt-data arrives already scored; this layer only places it on the bar timeline
**by `knowledge_time`** (never `event_time`) and forward-fills within
`ffill_limit` bars.

## `lab/indicators/cache.py`

```python
def cache_key(ticker: str, indicator: str, params: Mapping, data_version: str) -> str
def get(key: str) -> pd.Series | pd.DataFrame | None
def put(key: str, value: pd.Series | pd.DataFrame) -> None
def cached_indicator(ticker: str, name: str, df: pd.DataFrame, data_version: str,
                     **params) -> pd.Series | pd.DataFrame
def clear(prefix: str | None = None) -> int
def stats() -> dict   # {"entries": int, "bytes": int, "hits": int, "misses": int}
```

Parquet files under `data/cache/`. Key is a hex digest so it is filesystem-safe.

## `lab/adapters/base.py`

```python
class AdapterError(RuntimeError): ...
class AdapterUnavailable(AdapterError): ...
class QuotaExceeded(AdapterError): ...

@dataclass
class AdapterInfo:
    name: str; provides: frozenset[str]; available: bool; reason: str = ""
    quota_used: int | None = None; quota_limit: int | None = None
    last_call: datetime | None = None

class BaseAdapter:
    name: str = ""
    provides: frozenset[str] = frozenset()
    def available(self) -> tuple[bool, str]: ...
    def info(self) -> AdapterInfo: ...
    def fetch_bars(self, tickers, timeframe, start, end) -> Iterator[Bar]: ...
    def fetch_signals(self, start=None, end=None, **query) -> Iterator[Event]: ...

def get_adapter(name: str) -> BaseAdapter
def list_adapters() -> list[AdapterInfo]
REGISTRY: dict[str, type[BaseAdapter]]
```

Unimplemented `fetch_*` raises `NotImplementedError`. `get_adapter` resolves
`"alpaca" | "yfinance" | "govgreed" | "synthetic"`.

## `lab/adapters/yfinance.py` · `alpaca.py` · `synthetic.py`

Each exposes one `BaseAdapter` subclass (`YFinanceAdapter`, `AlpacaAdapter`,
`SyntheticAdapter`) registered in `lab.adapters.base.REGISTRY`.

- **yfinance** — daily/intraday bars, convenience bootstrap only. Set
  `knowledge_time = event_time + one bar` (you know a bar once it closes).
  `adjusted=True`.
- **alpaca** — `alpaca-py` lazily imported; historical bars via
  `StockHistoricalDataClient`, keys from settings. `available()` returns
  `(False, "no ALPACA_API_KEY_ID")` rather than raising.
- **synthetic** — deterministic seeded geometric-Brownian bars plus synthetic
  alt-data events with a realistic disclosure lag. No network, no credentials.
  This is what tests and the offline demo run on, so it must be exactly
  reproducible from `(seed, ticker, start, end, timeframe)`.

## `lab/adapters/govgreed.py`

The official GovGreed API (**not** the old web scraper — that approach is dead).
Base URL `https://www.govgreed.com/api/v1`, key from `GOVGREED_API_KEY` as a
Bearer token. See `docs/govgreed-bot-design-v1.md`.

```python
@dataclass
class Quota:
    used: int | None; limit: int | None; remaining: int | None
    tier: str | None; reset_at: datetime | None

class GovGreedError(AdapterError):
    def __init__(self, status: int, code: str | None, title: str, detail: str,
                 request_id: str | None): ...
class GovGreedAuthError(GovGreedError): ...      # 401/403 — halt, never retry
class GovGreedQuotaError(QuotaExceeded): ...     # 429 DAILY_QUOTA_EXCEEDED
class GovGreedBurstError(GovGreedError): ...     # 429 BURST_LIMIT_EXCEEDED

class GovGreedClient:
    def __init__(self, api_key=None, base_url=None, *, min_interval: float = 0.5,
                 max_retries: int = 3, timeout: float = 20.0, session=None): ...
    def get(self, path: str, **params) -> tuple[Any, dict]   # (data, meta)
    # typed endpoint helpers, each persisting the raw response first:
    def status(self) -> dict
    def me(self) -> dict
    def usage(self) -> dict
    def atlas(self) -> dict
    def signals_top(self, *, tier="A", fresh=True, limit=25) -> list[dict]
    def herd_signals(self, *, days=30) -> list[dict]
    def predictions_top(self, *, tier="A", status="ACTIVE") -> list[dict]
    def insider_signal(self, ticker: str) -> dict
    def bill_timeline(self, bill: str) -> dict
    def sector_positioning(self, sector: str) -> dict
    @property
    def quota(self) -> Quota
    @property
    def calls_made(self) -> int

class GovGreedAdapter(BaseAdapter):   # provides={"signals"}
    def fetch_signals(self, start=None, end=None, **query) -> Iterator[Event]: ...
    def daily_pull(self, *, top_n_enrich: int = 5) -> list[Event]: ...
    def replay(self, start=None, end=None) -> Iterator[Event]: ...
```

Hard requirements:

- Unwrap the `{data, meta}` envelope; record `meta.request_id` on **every** call
  and attach it to every `Event` produced from that response.
- Read quota from `X-RateLimit-*` headers **and** `meta.quota`; never hardcode
  limits (docs disagree between 20/day and 250/day — the code must not care).
- Self-throttle to ≈2 req/sec (`min_interval`), so `BURST_LIMIT_EXCEEDED` never
  fires; if it does, honour `Retry-After`.
- Errors are RFC 7807 `application/problem+json`. Map by documented code:
  `DAILY_QUOTA_EXCEEDED` → abort the run cleanly, record progress, alert, do
  **not** consume the retry buffer; `BURST_LIMIT_EXCEEDED` → sleep `Retry-After`
  and retry; `401/403` → `GovGreedAuthError`, halt immediately, no retries;
  `5xx` → exponential backoff, 3 attempts, then give up with the `request_id` in
  the message.
- Parse defensively: unknown fields are ignored, missing required fields log a
  warning and skip the record rather than crashing the run.
- `raw.save(...)` the verbatim response **before** parsing, always.
- Two-timestamp mapping: `event_time` = the underlying trade/transaction date
  when present (`trade_date`, `transaction_date`, `filed_at`, else the disclosure
  date); `knowledge_time` = when *we fetched it* (or the API's disclosure
  timestamp if it is later). Never let `knowledge_time < event_time`.
- `Event.uid` must be stable across pulls, e.g.
  `f"{kind}:{ticker}:{event_date}:{politician_or_source_hash}"`.
- `replay()` reads `lab/store/raw.py` snapshots and re-normalizes them, so the
  accumulated daily snapshots become a backtestable history.

## `lab/engine/clock.py`

```python
class SimClock:
    def __init__(self, timestamps: Sequence[datetime]): ...
    now: datetime            # property
    index: int               # property
    def __iter__(self) -> Iterator[datetime]
    def advance(self) -> datetime | None
class WallClock:
    def __init__(self, timeframe: str = "1d", at_time: time | None = None,
                 calendar: bool = True): ...
    now: datetime
    def __iter__(self) -> Iterator[datetime]   # blocks until each next bar close
    def next_fire(self) -> datetime
```

## `lab/engine/portfolio.py`

```python
class Portfolio:                          # implements PortfolioView
    def __init__(self, cash: float): ...
    cash equity positions gross_exposure   # properties
    def position(self, ticker: str) -> Position
    def weight(self, ticker: str) -> float
    def mark(self, prices: Mapping[str, float]) -> None
    def apply_fill(self, fill: Fill) -> Trade | None   # returns a closed round trip
    def snapshot(self) -> dict
    def trades(self) -> list[Trade]
    def target_to_delta_shares(self, ticker: str, target_pct: float,
                               price: float) -> float
class ReadOnlyPortfolio:  # wraps Portfolio, mutation raises
```

Accounting rules: average-cost basis; a fill that crosses through zero closes
the old round trip and opens a new one; commissions reduce cash and are charged
to realized P&L; `equity = cash + sum(market_value)`.

## `lab/engine/context.py`

```python
class EngineContext:      # implements Context
    def __init__(self, *, run_id: str, strategy: str, params: Mapping,
                 universe: Sequence[str], portfolio: PortfolioView,
                 data: DataView, timeframe: str = "1d",
                 strict: bool = True, capture: bool = True): ...
    def set_now(self, ts: datetime) -> None
    def drain_intents(self) -> list[Intent]
    def input_tape(self) -> dict     # everything served this bar, for the journal
    def reset_bar(self) -> None
class DataView:
    """Bar/signal access bound to a run. Holds preloaded frames in memory for
    backtests, delegates to the store in live mode."""
    def __init__(self, bars: Mapping[str, pd.DataFrame], signals: pd.DataFrame | None = None,
                 timeframe: str = "1d"): ...
    def history(self, ticker, field, n, upto: datetime) -> pd.Series
    def bars(self, ticker, n, upto: datetime) -> pd.DataFrame
    def signals(self, source, upto: datetime, **query) -> list[Event]
    def price(self, ticker, upto: datetime) -> float | None
    def timestamps(self) -> list[datetime]
    @classmethod
    def from_store(cls, tickers, timeframe, start, end, *, sources=None) -> "DataView"
```

The look-ahead barrier lives here and it is the most important code in the repo.
`DataView` filters on `knowledge_time <= upto` on **every** call. `strict=True`
makes a request for a future timestamp raise `LookAheadError`. The context
records every value it serves into `input_tape()`, which becomes the decision
journal's input tape — capture is a wrapper, not a strategy change.

## `lab/engine/broker_sim.py`

```python
class SimBroker:                      # implements Broker
    def __init__(self, portfolio: Portfolio, fill_model: FillModel,
                 *, name: str = "sim"): ...
    def submit(self, order: Order) -> Order      # queued, not filled
    def process(self, ts: datetime, bar_data: Mapping[str, pd.Series]) -> list[Fill]
    def cancel(self, order_id) -> bool
    def open_orders(self) -> list[Order]
    def positions(self) -> dict[str, Position]
    def account(self) -> dict
```

`submit` is idempotent on `idem_key`. `process` is what the runner calls once
per bar to fill whatever the fill model says is fillable at that timestamp.

## `lab/engine/loader.py`

```python
@dataclass
class LoadedStrategy:
    name: str; instance: Any; module: ModuleType; path: Path
    params: dict; source: str; source_hash: str
def load_strategy(path: str | Path, params: Mapping | None = None) -> LoadedStrategy
def discover(directory: Path | None = None) -> list[dict]
```

Resolution order inside the module: a `STRATEGY` object, a `build(params)`
factory, a class named `Strategy`, then the single class defining `on_bar`.
Merge `module.PARAMS` defaults under the passed `params`.

## `lab/engine/broker_alpaca.py`

```python
class AlpacaBroker:      # implements Broker
    def __init__(self, *, paper: bool = True, key_id=None, secret_key=None): ...
    def submit / cancel / open_orders / positions / account
    def available(self) -> tuple[bool, str]
    def clock(self) -> dict
```

Lazy `alpaca-py` import. Idempotency via `client_order_id = idem_key` hashed to
Alpaca's allowed charset; a duplicate submission is caught and returned as the
existing order rather than raising. **Paper endpoint is the default.**

## `lab/risk/gate.py`

```python
@dataclass
class RiskLimits:
    max_position_pct: float = 0.05
    max_positions: int = 8
    max_sector_positions: int = 2
    max_sector_pct: float = 0.25
    max_gross_exposure: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_orders_per_day: int = 20
    min_order_notional: float = 50.0
    allowlist: list[str] = field(default_factory=list)
    denylist: list[str] = field(default_factory=list)
    cooldown_days: int = 0
    max_position_notional: float | None = None
    @classmethod
    def from_yaml(cls, path) -> "RiskLimits"
    @classmethod
    def from_mapping(cls, m) -> "RiskLimits"

@dataclass
class GateState:
    day: date | None = None; orders_today: int = 0
    day_start_equity: float = 0.0; breaker_tripped: bool = False
    breaker_reason: str = ""; exited: dict[str, date] = field(default_factory=dict)

class RiskGate:
    def __init__(self, limits: RiskLimits, *, sectors: Mapping[str, str] | None = None,
                 state: GateState | None = None): ...
    def evaluate(self, intents: Sequence[Intent], *, portfolio: PortfolioView,
                 now: datetime, prices: Mapping[str, float]) -> list[GateVerdict]
    def note_orders(self, orders: Sequence[Order]) -> None
    def roll_day(self, now: datetime, equity: float) -> None
    def trip(self, reason: str) -> None
    def reset(self) -> None
    @property
    def tripped(self) -> bool
```

Deterministic and strategy-independent: the *same* gate sits between any
strategy — human-written or agent-driven — and the broker. Rule names that must
appear verbatim in `GateVerdict.rule`: `kill_switch`, `breaker`, `denylist`,
`allowlist`, `cooldown`, `position_cap`, `position_notional`, `max_positions`,
`sector_positions`, `sector_pct`, `gross_exposure`, `max_orders_per_day`,
`min_notional`. Evaluate in that order. Exits (target 0, or any reduction of an
existing position) are **never blocked** by capacity rules — only by
`kill_switch`. A tripped breaker stops new/increasing exposure while leaving
exit logic running.

## `lab/registry/db.py`

```python
def connect(path: Path | None = None) -> sqlite3.Connection   # WAL, row factory
def init_db(con=None) -> None
SCHEMA_SQL: str
def transaction(con) -> ContextManager
```

Tables: `runs`, `run_metrics`, `decisions`, `orders`, `fills`, `trades`,
`events`, `agent_calls`, `quota_log`, `live_strategies`.

## `lab/registry/runs.py`

```python
@dataclass
class RunRecord:
    run_id: str; strategy: str; kind: str            # backtest|paper|live|sweep
    status: str; created_at: datetime; finished_at: datetime | None
    git_commit: str | None; config_hash: str; data_version: str
    params: dict; config: dict; metrics: dict
    start: datetime | None; end: datetime | None
    origin: str = "human"                            # human|agent-loop
    parent_run_id: str | None = None; sweep_id: str | None = None
    notes: str = ""; attempt: int = 0; error: str = ""

class RunRegistry:
    def __init__(self, con=None): ...
    def create(self, **fields) -> RunRecord
    def finish(self, run_id, *, metrics: Mapping, status: str = "ok", error: str = "") -> RunRecord
    def get(self, run_id) -> RunRecord | None
    def list(self, *, strategy=None, kind=None, origin=None, sweep_id=None,
             limit=100, offset=0, order_by="created_at desc") -> list[RunRecord]
    def compare(self, run_ids: Sequence[str]) -> pd.DataFrame
    def attempts(self, strategy: str, config_hash: str | None = None) -> int
    def family_attempts(self, strategy: str) -> int
    def delete(self, run_id) -> bool

def git_commit() -> str | None
def config_hash(config: Mapping) -> str
def run_artifact_dir(run_id: str) -> Path      # runs/<run_id>/
```

The attempt counter is the overfitting tell: it makes "someone quietly ran 400
variations and cherry-picked one" impossible to hide. Increment on `create`.

## `lab/registry/journal.py`

```python
class DecisionJournal:
    def __init__(self, con=None): ...
    def append(self, decision: Decision) -> None
    def bulk_append(self, decisions: Iterable[Decision]) -> int
    def list(self, run_id: str, *, start=None, end=None, limit=1000,
             offset=0, ticker=None) -> list[dict]
    def get(self, decision_id: str) -> dict | None
    def count(self, run_id: str) -> int

class EventJournal:
    def __init__(self, con=None): ...
    def append(self, ev: JournalEvent) -> int        # returns assigned seq
    def tail(self, since_seq: int = 0, *, limit: int = 500, run_id=None,
             strategy=None, kinds=None) -> list[dict]
    def latest_seq(self) -> int
    def subscribe(self) -> Iterator[dict]            # polling generator for the WS
    def heartbeat(self, source: str, *, meta: Mapping | None = None) -> None
    def last_heartbeats(self) -> dict[str, dict]     # source -> {at, age_s, meta}
```

Append-only. The UI never touches the broker or the runner — it reads journals,
so a runner crash cannot be caused by the UI and the UI still works as forensics
when the runner is down.

## `lab/backtest/fills.py`

```python
@dataclass
class FillModel:
    mode: str = "next_open"        # next_open | same_close | next_close
    slippage_bps: float = 5.0
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    partial_fill_volume_pct: float | None = None   # cap qty at % of bar volume
    allow_fractional: bool = False
    @property
    def optimistic(self) -> bool   # True for same_close
    @classmethod
    def from_mapping(cls, m) -> "FillModel"
    def fill_price(self, side: Side, bar: Mapping) -> float
    def commission(self, qty: float, price: float) -> float
    def fill(self, order: Order, bar: Mapping, ts: datetime) -> Fill | None
```

`same_close` must be reported loudly as optimistic wherever run metadata is
shown.

## `lab/backtest/metrics.py`

```python
def compute_metrics(equity: pd.Series, trades: Sequence[Trade], *,
                    periods_per_year: int = 252, risk_free: float = 0.0,
                    benchmark: pd.Series | None = None) -> dict
def drawdown_series(equity: pd.Series) -> pd.Series
def periods_per_year(timeframe: str) -> int
def summarize_trades(trades: Sequence[Trade]) -> dict
```

`compute_metrics` returns at least: `start end days total_return cagr sharpe
sortino calmar volatility max_drawdown max_drawdown_duration_days exposure
turnover trades hit_rate avg_win avg_loss profit_factor avg_trade_pnl
best_trade worst_trade avg_bars_held final_equity peak_equity`. Every value is a
plain float/int/str — the dict is written straight to JSON.

## `lab/backtest/runner.py`

```python
@dataclass
class BacktestConfig:
    strategy: str; tickers: list[str]; timeframe: str = "1d"
    start: datetime | None = None; end: datetime | None = None
    cash: float = 100_000.0; params: dict = field(default_factory=dict)
    fills: dict = field(default_factory=dict); limits: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list); warmup: int = 0
    sectors: dict = field(default_factory=dict); benchmark: str | None = None
    origin: str = "human"; notes: str = ""; seed: int | None = None
    @classmethod
    def from_yaml(cls, path, overrides: Mapping | None = None) -> "BacktestConfig"
    def to_dict(self) -> dict

@dataclass
class BacktestResult:
    run_id: str; metrics: dict; equity: pd.Series; trades: list[Trade]
    decisions: list[Decision]; orders: list[Order]; fills: list[Fill]
    config: dict; data_version: str; artifact_dir: Path
    def to_json(self) -> dict        # the `--json` CLI payload

def run_backtest(config: BacktestConfig, *, register: bool = True,
                 journal: bool = True, progress: bool = False) -> BacktestResult
```

Event loop per bar: mark portfolio → `broker.process` fills from the *previous*
bar's orders → `on_fill` hooks → `ctx.set_now` → `strategy.on_bar` → drain
intents → `gate.evaluate` → build orders → `broker.submit` → persist the
`Decision`. Writes `runs/<run_id>/{metrics.json,equity.parquet,trades.csv,
config.json,decisions.jsonl}` and registers the run. Same commit + config + data
version must produce byte-identical metrics.

## `lab/backtest/report.py`

```python
def render_report(result: BacktestResult | str, *, out: Path | None = None,
                  open_browser: bool = False) -> Path
def report_path(run_id: str) -> Path
```

One self-contained HTML file — inline CSS, inline SVG charts, no CDN. Equity +
drawdown, trade ledger, metrics table, and a provenance footer carrying git
commit, config hash, `data_version`, fill model (with the loud optimistic flag)
and the standing survivorship-bias caveat until a point-in-time universe lands.
Use the same palette as the console (§7 of the console doc).

## `lab/backtest/sweep.py` + `walkforward.py`

```python
# sweep.py
@dataclass
class SweepConfig:
    base: BacktestConfig; grid: dict[str, list]; walk_forward: str | None = None
    metric: str = "sharpe"; max_runs: int | None = None; fast: bool = False
    @classmethod
    def from_yaml(cls, path, base=None) -> "SweepConfig"
def expand_grid(grid: Mapping[str, Sequence]) -> list[dict]
def run_sweep(cfg: SweepConfig, *, workers: int = 1, progress: bool = False) -> dict
def fast_screen(cfg: SweepConfig) -> pd.DataFrame   # vectorized coarse screen

# walkforward.py
@dataclass
class Window:
    index: int; is_start: datetime; is_end: datetime; oos_start: datetime; oos_end: datetime
def make_windows(start, end, spec: str, *, timeframe="1d") -> list[Window]  # "4:1"
def run_walk_forward(cfg: SweepConfig, *, progress=False) -> dict
```

`run_sweep` returns `{"sweep_id","runs":[...],"best","grid","attempts",
"walk_forward":[...]}`. Out-of-sample score is the fitness function, **never**
in-sample — an agent is a tireless overfitter and the sweep runner is where that
gets contained. `fast_screen` tries `vectorbt` if importable and otherwise uses
a built-in numpy screener; either way its output is a *coarse screen* and every
survivor is re-run through the event-driven engine before it is believed.

## `lab/live/*`

```python
# alerts.py
def alert(kind: str, message: str, **fields) -> None      # webhook + journal + log
def alert_fill(fill) / alert_gate_block(v) / alert_breaker(reason) / alert_error(exc)

# killswitch.py
def engaged() -> tuple[bool, str | None]
def engage(reason: str = "") -> Path
def release() -> bool

# reconcile.py
@dataclass
class ReconcileResult:
    ok: bool; diffs: list[dict]; broker_positions: dict; local_positions: dict
    broker_orders: list; message: str
def reconcile(broker, portfolio, *, acknowledge: bool = False) -> ReconcileResult

# runner.py
@dataclass
class LiveConfig(BacktestConfig-like):
    strategy: str; tickers: list[str]; timeframe: str = "1d"
    broker: str = "alpaca"; paper: bool = True; at_time: str = "09:35"
    params/limits/sectors/sources; poll_seconds: int = 30
class LiveRunner:
    def __init__(self, cfg: LiveConfig): ...
    def start(self) -> None       # reconcile-first, then loop
    def step(self, now: datetime) -> Decision | None
    def stop(self) -> None
    def status(self) -> dict
```

Startup is reconcile-first and treats the broker as the source of truth: fetch
positions and open orders, diff against local state, and **refuse to trade**
until discrepancies are resolved or explicitly acknowledged. Crash-and-resume is
a designed-for path. Every event goes to the `EventJournal`; heartbeats every
`poll_seconds` from runner, data adapter, and broker connection.

## `lab/agent/*`

```python
# schemas.py
TARGETS_TOOL: dict          # Anthropic tool-use schema: list of {ticker, target_pct, rationale}
STRATEGY_PROPOSAL_TOOL: dict
def validate_targets(payload: Mapping, *, universe: Sequence[str],
                     max_positions: int) -> list[Intent]

# in_loop_strategy.py
class AgentStrategy:        # implements Strategy
    def __init__(self, params: Mapping): ...
    def on_bar(self, ctx: Context) -> None
    def build_bundle(self, ctx: Context) -> dict
    def call_model(self, bundle: dict) -> dict

# author_loop.py
@dataclass
class AuthorLoopConfig:
    seed_strategy: Path; grid_or_freeform: str; iterations: int = 10
    metric: str = "oos_sharpe"; budget_usd: float = 5.0; model: str | None = None
@dataclass
class Iteration:
    n: int; run_id: str; params: dict; diff: str; metrics: dict; rationale: str
def run_author_loop(cfg: AuthorLoopConfig) -> dict      # {"lineage":[Iteration], "best": ...}
```

Pattern B rules, non-negotiable: the agent **proposes**, the risk gate
**disposes** — model output is untrusted input validated like any other, with no
broker access of its own. Any external text the agent reads is untrusted too and
must land on the schema wall. Historical backtests of an LLM strategy are
contaminated by training data, so `AgentStrategy` runs must be tagged
`contaminated: true` in their metrics and the report must say so; real
evaluation is forward-only on paper. Every prompt and response is persisted to
`agent_calls` with token counts and cost.

## `lab/agent/providers.py`

```python
class ProviderUnavailable(RuntimeError): ...          # reason a human can act on

@dataclass
class Block:        type: str; text: str = ""; name: str = ""; input: dict = {}
@dataclass
class Usage:        input_tokens: int; output_tokens: int
                    cache_read_input_tokens: int; cache_creation_input_tokens: int
@dataclass
class Response:     model: str; content: list[Block]; stop_reason: str; usage: Usage
                    cost_usd: float | None; billing: str; raw: dict
@dataclass
class ProviderInfo: name: str; model: str; available: bool; reason: str
                    billing: str; base_url: str | None; forces_tools: bool
                    detail: dict
                    def to_dict(self) -> dict

class BaseProvider:
    name: str; billing: str
    def __init__(self, *, model: str | None = None, timeout: float = 120.0): ...
    @property
    def messages(self): ...                # .create(**request) -> Response
    @property
    def forces_tools(self) -> bool
    def available(self) -> tuple[bool, str]
    def info(self) -> ProviderInfo
    def create(self, **request) -> Response
    def require(self) -> BaseProvider      # raises ProviderUnavailable

class AnthropicProvider(BaseProvider):     # name "anthropic",   billing "api"
class ClaudeCodeProvider(BaseProvider):    # name "claude_code", billing "subscription"
class OpenAICompatProvider(BaseProvider):  # name "openai",      billing "api"

REGISTRY: dict[str, type[BaseProvider]]    # the three canonical names
ALIASES: dict[str, str]                    # claude|api|cli|openrouter|ollama|...
AUTO_ORDER: tuple[str, ...]                # ("anthropic", "claude_code", "openai")

def normalize(name: str | None) -> str                    # alias -> canonical
def build(name: str | None = None, **kwargs) -> BaseProvider
def resolve(name: str | None = None, **kwargs) -> BaseProvider   # honours "auto"
def list_providers(**kwargs) -> list[ProviderInfo]
def probe(name: str | None = None, *, prompt: str = ...) -> dict  # ONE real call
```

**The binding rule: every provider adapts to the Anthropic Messages
request/response shape, so `schemas.call_tool` stays the single audited
chokepoint.** That one function writes the `agent_calls` ledger, extracts the
tool block and prices the exchange; a provider's job is translation and nothing
else, so adding a fourth backend can never add a fourth place where a model call
escapes the ledger.

Two consequences worth stating. `forces_tools` is `False` for `claude_code`
alone — it cannot force a schema, so the schema is prompted and the JSON parsed
back out; tolerable only because `validate_targets` treats model output as
untrusted and the gate re-checks it. And `Response.cost_usd` is set only when a
backend reports real money (OpenRouter's usage accounting, the Claude Code
tally); `call_tool` prefers it over `estimate_cost`, while `billing` says whether
those dollars left an account or are notional subscription equivalents.

`probe()` is the only function here that spends anything. `list_providers()` and
`info()` read configuration and call nobody.

## `lab/cli.py`

Typer app, `main()` entry point. Every command takes `--json` and emits a single
JSON object on stdout with **nothing else on stdout** (logs go to stderr), so an
agent loop can parse it.

```
lab pull --source alpaca --tickers cfg/universe.txt --tf 1d --since 2018-01-01
lab backtest strategies/momo.py --config cfg/momo.yaml [--json]
lab sweep strategies/momo.py --grid cfg/momo_grid.yaml --walk-forward 4:1 [--json]
lab report <run_id> [--open]
lab runs list|show|compare|delete
lab paper start|stop strategies/momo.py --config cfg/momo.yaml
lab status | lab kill [--release]
lab data coverage|version
lab adapters
lab govgreed pull|status
lab agent author|status
lab agent providers [--probe [--provider anthropic|claude_code|openai]]
lab ui [--port 8787] [--no-open]
lab strategies
```

Exit codes: 0 ok, 1 error, 2 bad usage, 3 gate/kill refusal.

`lab agent providers` lists the model backends with their model id,
availability, billing mode, whether they can force a tool schema, base URL and
the reason any of them cannot run. It is free: configuration only, no call.
`--probe` makes exactly one tiny real call through
`lab.agent.providers.probe()` — spending an API token, or one request against a
Claude subscription — and exits 1 if it fails. `lab status` and `lab agent
status` both carry the resolved backend (`agent.active`, `agent.billing`) so the
question "what is about to be spent, and on whose account" has an answer that
does not require running anything.

## `lab/api/*`

FastAPI app bound to `127.0.0.1`, optional bearer token from `LAB_UI_TOKEN`.
REST mirrors the CLI JSON contract, so anything the UI can show a script can
also fetch — the console is a view over the same contract, not a second API.

```
GET  /api/health
GET  /api/runs?strategy=&kind=&origin=&limit=&offset=
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/equity          -> {t:[], equity:[], drawdown:[], is_oos:[]}
GET  /api/runs/{run_id}/trades
GET  /api/runs/{run_id}/decisions?start=&end=&limit=&offset=&ticker=
GET  /api/runs/{run_id}/bars?ticker=&tf=
GET  /api/runs/compare?ids=a,b,c
GET  /api/sweeps/{sweep_id}
GET  /api/strategies
GET  /api/live/strategies
GET  /api/live/events?since=&limit=
GET  /api/live/health                   -> adapters, heartbeats, quota, breaker
GET  /api/agent/lineage/{strategy}
GET  /api/agent/calls?run_id=
POST /api/control/pause                 {strategy}
POST /api/control/cancel_orders         {strategy}
POST /api/control/kill                  {reason}
WS   /api/ws/events?since=
```

**Asymmetric controls — the console can only make the system safer, never
riskier.** Pause, cancel open orders, and the kill switch are available,
confirm-gated and journaled. Starting a live strategy, raising a limit, and
resuming after a breaker are **not** API operations at all; they stay CLI
actions behind the manual checklist. Any route that would increase exposure must
not exist.
