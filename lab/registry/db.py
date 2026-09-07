"""SQLite plumbing shared by the run registry and the two journals.

Two databases, deliberately: ``registry.sqlite`` holds the experiment record
(runs, metrics, orders, fills, trades, agent calls, quota) and
``journal.sqlite`` holds the append-only streams the console reads (decisions,
events, live strategies). Splitting them means the UI tailing the journal at
10Hz never contends with a sweep hammering the registry, and a corrupt journal
never costs the run history. ``SCHEMA_SQL`` is applied to both -- one idempotent
script is cheaper to reason about than two, and the unused tables cost a few
hundred bytes.

Datetimes are stored as ISO-8601 UTC ``TEXT`` with ``detect_types`` left off:
lexicographic order equals chronological order for that format, so SQL can sort
and range-filter timestamps without a converter, and every read parses
explicitly rather than depending on sqlite3's global converter registry.

Concurrency: one connection per (thread, database path), cached on a
``threading.local``. WAL lets many readers run alongside one writer, and
``busy_timeout`` absorbs the overlap. Connections are opened with
``check_same_thread=False`` so a caller may hand one to a worker thread
(``RunRegistry(con=...)``); a process-wide re-entrant lock serializes writes so
two threads sharing one connection cannot interleave transactions.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from lab.config import get_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    strategy      TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'backtest',
    status        TEXT NOT NULL DEFAULT 'running',
    created_at    TEXT NOT NULL,
    finished_at   TEXT,
    git_commit    TEXT,
    config_hash   TEXT NOT NULL DEFAULT '',
    data_version  TEXT NOT NULL DEFAULT '',
    params        TEXT NOT NULL DEFAULT '{}',
    config        TEXT NOT NULL DEFAULT '{}',
    metrics       TEXT NOT NULL DEFAULT '{}',
    start_ts      TEXT,
    end_ts        TEXT,
    origin        TEXT NOT NULL DEFAULT 'human',
    parent_run_id TEXT,
    sweep_id      TEXT,
    notes         TEXT NOT NULL DEFAULT '',
    attempt       INTEGER NOT NULL DEFAULT 0,
    error         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_runs_family  ON runs(strategy, config_hash);
CREATE INDEX IF NOT EXISTS ix_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_runs_sweep   ON runs(sweep_id);
CREATE INDEX IF NOT EXISTS ix_runs_origin  ON runs(origin, kind);

-- Metrics live twice: verbatim JSON on `runs` for round-tripping, and one row
-- per scalar here so `ORDER BY sharpe` and cross-run queries stay pure SQL.
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    value      REAL,
    text_value TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE INDEX IF NOT EXISTS ix_run_metrics_name ON run_metrics(name, value);

CREATE TABLE IF NOT EXISTS decisions (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    strategy    TEXT NOT NULL DEFAULT '',
    at          TEXT NOT NULL,
    inputs      TEXT NOT NULL DEFAULT '{}',
    intents     TEXT NOT NULL DEFAULT '[]',
    verdicts    TEXT NOT NULL DEFAULT '[]',
    order_ids   TEXT NOT NULL DEFAULT '[]',
    portfolio   TEXT NOT NULL DEFAULT '{}',
    logs        TEXT NOT NULL DEFAULT '[]',
    agent       TEXT,
    duration_ms REAL NOT NULL DEFAULT 0,
    summary     TEXT NOT NULL DEFAULT '',
    -- ',AAPL,MSFT,' -- denormalized from intents/verdicts so the tape can filter
    -- by ticker without parsing every JSON blob in the run.
    tickers     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_decisions_run ON decisions(run_id, at);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    run_id          TEXT,
    decision_id     TEXT,
    strategy        TEXT NOT NULL DEFAULT '',
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    order_type      TEXT NOT NULL DEFAULT 'market',
    limit_price     REAL,
    created_at      TEXT,
    status          TEXT NOT NULL DEFAULT 'new',
    broker_order_id TEXT,
    idem_key        TEXT,
    tag             TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_orders_run  ON orders(run_id, created_at);
CREATE INDEX IF NOT EXISTS ix_orders_idem ON orders(idem_key);

CREATE TABLE IF NOT EXISTS fills (
    id          TEXT PRIMARY KEY,
    order_id    TEXT,
    run_id      TEXT,
    decision_id TEXT,
    strategy    TEXT NOT NULL DEFAULT '',
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    at          TEXT NOT NULL,
    commission  REAL NOT NULL DEFAULT 0,
    slippage    REAL NOT NULL DEFAULT 0,
    tag         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_fills_run ON fills(run_id, at);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL DEFAULT '',
    qty         REAL NOT NULL DEFAULT 0,
    entry_time  TEXT,
    entry_price REAL,
    exit_time   TEXT,
    exit_price  REAL,
    pnl         REAL NOT NULL DEFAULT 0,
    pnl_pct     REAL NOT NULL DEFAULT 0,
    bars_held   INTEGER NOT NULL DEFAULT 0,
    commission  REAL NOT NULL DEFAULT 0,
    tag         TEXT NOT NULL DEFAULT '',
    exit_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_trades_run ON trades(run_id, entry_time);

-- `seq` is the resumable cursor: AUTOINCREMENT so an id is never reused even
-- after a purge, which is what makes `tail(since_seq)` safe for a WebSocket
-- that reconnects.
CREATE TABLE IF NOT EXISTS events (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    id       TEXT NOT NULL UNIQUE,
    at       TEXT NOT NULL,
    kind     TEXT NOT NULL,
    source   TEXT NOT NULL,
    run_id   TEXT,
    strategy TEXT,
    ticker   TEXT,
    message  TEXT NOT NULL DEFAULT '',
    payload  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind, seq DESC);
CREATE INDEX IF NOT EXISTS ix_events_run  ON events(run_id, seq DESC);
CREATE INDEX IF NOT EXISTS ix_events_strat ON events(strategy, seq DESC);

CREATE TABLE IF NOT EXISTS agent_calls (
    id            TEXT PRIMARY KEY,
    run_id        TEXT,
    at            TEXT NOT NULL,
    strategy      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    tool          TEXT NOT NULL DEFAULT '',
    prompt        TEXT NOT NULL DEFAULT '',
    response      TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    -- The API reports input_tokens net of the cache, so these are not a detail:
    -- without them a 20k-token prompt served from cache is recorded as 2 tokens.
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    latency_ms    REAL NOT NULL DEFAULT 0,
    ok            INTEGER NOT NULL DEFAULT 1,
    error         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_agent_calls_run ON agent_calls(run_id, at);

CREATE TABLE IF NOT EXISTS quota_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    source      TEXT NOT NULL,
    endpoint    TEXT NOT NULL DEFAULT '',
    used        INTEGER,
    quota_limit INTEGER,
    remaining   INTEGER,
    tier        TEXT,
    reset_at    TEXT,
    request_id  TEXT,
    status      INTEGER
);
CREATE INDEX IF NOT EXISTS ix_quota_log_source ON quota_log(source, at DESC);

-- What was concluded, as distinct from what was run. See lab/registry/findings.py.
CREATE TABLE IF NOT EXISTS findings (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title      TEXT NOT NULL,
    claim      TEXT NOT NULL,
    evidence   TEXT NOT NULL DEFAULT '{}',
    run_ids    TEXT NOT NULL DEFAULT '[]',
    tags       TEXT NOT NULL DEFAULT '[]',
    status     TEXT NOT NULL DEFAULT 'open',
    supersedes TEXT,
    author     TEXT NOT NULL DEFAULT '',
    history    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_findings_status ON findings(status, created_at DESC);

CREATE TABLE IF NOT EXISTS live_strategies (
    strategy   TEXT PRIMARY KEY,
    run_id     TEXT,
    kind       TEXT NOT NULL DEFAULT 'paper',
    status     TEXT NOT NULL DEFAULT 'stopped',
    config     TEXT NOT NULL DEFAULT '{}',
    pid        INTEGER,
    host       TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    updated_at TEXT,
    paused     INTEGER NOT NULL DEFAULT 0,
    next_fire  TEXT,
    notes      TEXT NOT NULL DEFAULT '',
    -- Positions and cash as of the last completed step. Without it a restarted
    -- runner starts from an empty book, and reconciliation then reads every
    -- broker position as a stranger and refuses to trade. With it, a diff means
    -- something actually changed while the process was down.
    book       TEXT NOT NULL DEFAULT ''
);
"""

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
)

_local = threading.local()
_write_lock = threading.RLock()
_init_lock = threading.Lock()
_initialized: set[str] = set()


def registry_path() -> Path:
    return get_settings().paths.registry_db


def journal_path() -> Path:
    return get_settings().paths.journal_db


def _cache() -> dict[str, sqlite3.Connection]:
    cache = getattr(_local, "conns", None)
    if cache is None:
        cache = {}
        _local.conns = cache
    return cache


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (or reuse) the connection this thread holds for ``path``.

    Defaults to the registry database; the journal classes pass the journal
    path. The schema is created on first touch, so nothing in the lab needs an
    explicit migrate step before its first write.
    """
    target = Path(path) if path is not None else registry_path()
    target = target.expanduser()
    key = str(target)
    cache = _cache()
    con = cache.get(key)
    if con is not None:
        return con

    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(key, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        con.execute(pragma)
    cache[key] = con

    with _init_lock:
        fresh = key not in _initialized
    if fresh:
        init_db(con)
        with _init_lock:
            _initialized.add(key)
    return con


#: Columns added after a table shipped. ``CREATE TABLE IF NOT EXISTS`` does not
#: add a column to a database that already exists, so without this a new field is
#: present on a fresh install and absent everywhere else -- and the tests, which
#: run against a clean tmpdir, pass either way.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("agent_calls", "cache_read_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("agent_calls", "cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("live_strategies", "book", "TEXT NOT NULL DEFAULT ''"),
)


def init_db(con: sqlite3.Connection | None = None) -> None:
    """Idempotent: every statement is ``IF NOT EXISTS``, so calling it on a
    populated database is a no-op rather than a data loss."""
    con = con if con is not None else connect()
    with _write_lock:
        con.executescript(SCHEMA_SQL)
        for table, column, decl in _ADDED_COLUMNS:
            existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` .. ``COMMIT``, re-entrant and write-serialized.

    IMMEDIATE because the read-then-write patterns here (the attempt counter,
    most of all) must not lose a race to another writer between the SELECT and
    the INSERT. Nesting is a no-op so callers can compose freely.
    """
    if con.in_transaction:
        yield con
        return
    with _write_lock:
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")


def close_all() -> None:
    """Drop this thread's cached connections. Tests point the settings at a
    fresh tmpdir between cases; without this they would keep writing to the
    previous one."""
    cache = _cache()
    for con in cache.values():
        try:
            con.close()
        except sqlite3.Error:
            pass
    cache.clear()
    with _init_lock:
        _initialized.clear()
