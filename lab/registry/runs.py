"""The run registry: every experiment the lab has ever run, with the provenance
needed to reproduce it and the attempt count needed to distrust it.

Two jobs. First, attribution -- a run row carries ``git_commit``,
``config_hash`` and ``data_version``, which is what makes "same commit + same
config + same data version implies identical metrics" a checkable claim instead
of a hope, and what makes a restated vendor dataset show up as a changed
``data_version`` rather than as mysteriously drifting results.

Second, the attempt counter. ``attempts()`` counts runs for one
(strategy, config_hash) pair and ``family_attempts()`` counts every run of the
strategy whatever the config. Both increment on ``create()``, before a single
metric exists, so the count cannot be trimmed after the fact by only recording
the runs that looked good. The lab cannot stop anyone -- human or agent -- from
trying four hundred variations and keeping one; it can refuse to let that be
invisible.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from lab.config import get_settings
from lab.registry.db import connect, transaction
from lab.timeutil import to_utc, utcnow

#: The run kinds the rest of the lab agrees on (design doc §6, §7).
RUN_KINDS = frozenset({"backtest", "paper", "live", "sweep"})

_ORDER_COLUMNS = frozenset(
    {
        "created_at",
        "finished_at",
        "strategy",
        "kind",
        "status",
        "origin",
        "attempt",
        "run_id",
        "config_hash",
        "sweep_id",
    }
)
_ORDER_BY_RE = re.compile(r"^\s*(\w+)(?:\s+(asc|desc))?\s*$", re.IGNORECASE)

_META_COLUMNS = (
    "run_id",
    "strategy",
    "kind",
    "status",
    "origin",
    "attempt",
    "created_at",
    "finished_at",
    "git_commit",
    "config_hash",
    "data_version",
    "sweep_id",
    "notes",
)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    strategy: str
    kind: str = "backtest"
    status: str = "running"
    created_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    git_commit: str | None = None
    config_hash: str = ""
    data_version: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None
    origin: str = "human"
    parent_run_id: str | None = None
    sweep_id: str | None = None
    notes: str = ""
    attempt: int = 0
    error: str = ""

    @property
    def artifact_dir(self) -> Path:
        return run_artifact_dir(self.run_id)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: timestamps as ISO-8601 UTC, everything else plain. This
        is the payload shape ``lab runs show --json`` and ``/api/runs`` emit."""
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "kind": self.kind,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "finished_at": _iso(self.finished_at),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "params": self.params,
            "config": self.config,
            "metrics": self.metrics,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "origin": self.origin,
            "parent_run_id": self.parent_run_id,
            "sweep_id": self.sweep_id,
            "notes": self.notes,
            "attempt": self.attempt,
            "error": self.error,
        }


def _iso(ts: datetime | None) -> str | None:
    return to_utc(ts).isoformat() if ts is not None else None


def _parse_ts(raw: Any) -> datetime | None:
    return to_utc(raw) if raw else None


def _loads(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def run_from_row(row: sqlite3.Row | Mapping[str, Any]) -> RunRecord:
    """Row -> record. Exported because the API layer reads `runs` directly for
    joins and should not re-invent the JSON/timestamp decoding."""
    r = dict(row)
    return RunRecord(
        run_id=r["run_id"],
        strategy=r["strategy"],
        kind=r["kind"],
        status=r["status"],
        created_at=to_utc(r["created_at"]),
        finished_at=_parse_ts(r.get("finished_at")),
        git_commit=r.get("git_commit"),
        config_hash=r.get("config_hash") or "",
        data_version=r.get("data_version") or "",
        params=_loads(r.get("params"), {}),
        config=_loads(r.get("config"), {}),
        metrics=_loads(r.get("metrics"), {}),
        start=_parse_ts(r.get("start_ts")),
        end=_parse_ts(r.get("end_ts")),
        origin=r.get("origin") or "human",
        parent_run_id=r.get("parent_run_id"),
        sweep_id=r.get("sweep_id"),
        notes=r.get("notes") or "",
        attempt=int(r.get("attempt") or 0),
        error=r.get("error") or "",
    )


class RunRegistry:
    """Reads and writes the ``runs`` table. Cheap to construct; hold one per
    component rather than threading a connection through call sites."""

    def __init__(self, con: sqlite3.Connection | None = None) -> None:
        self._con = con if con is not None else connect()

    @property
    def con(self) -> sqlite3.Connection:
        return self._con

    # --- write -------------------------------------------------------------

    def create(self, **fields: Any) -> RunRecord:
        """Register a run at its *start*, not its end.

        ``attempt`` is assigned here, inside the same IMMEDIATE transaction as
        the insert, so parallel sweep workers cannot both claim attempt N.
        """
        unknown = set(fields) - {f for f in RunRecord.__slots__}
        if unknown:
            raise ValueError(f"unknown run field(s): {sorted(unknown)}")

        strategy = str(fields.get("strategy") or "").strip()
        if not strategy:
            raise ValueError("create() needs a non-empty strategy")

        kind = str(fields.get("kind") or "backtest")
        if kind not in RUN_KINDS:
            raise ValueError(f"unknown run kind {kind!r}; expected one of {sorted(RUN_KINDS)}")

        origin = str(fields.get("origin") or "human").strip()
        if not origin:
            raise ValueError("origin must be a non-empty string")

        created_at = to_utc(fields.get("created_at") or utcnow())
        params = dict(fields.get("params") or {})
        config = dict(fields.get("config") or {})
        # A run with no config file is still a distinct experiment: fall back to
        # params so the attempt counter groups sweeps of the same grid point.
        chash = fields.get("config_hash") or config_hash(config or params)

        record = RunRecord(
            run_id=str(fields.get("run_id") or _new_run_id(created_at, strategy)),
            strategy=strategy,
            kind=kind,
            status=str(fields.get("status") or "running"),
            created_at=created_at,
            finished_at=_parse_ts(fields.get("finished_at")),
            git_commit=fields["git_commit"] if "git_commit" in fields else git_commit(),
            config_hash=str(chash),
            data_version=str(fields.get("data_version") or ""),
            params=params,
            config=config,
            metrics=dict(fields.get("metrics") or {}),
            start=_parse_ts(fields.get("start")),
            end=_parse_ts(fields.get("end")),
            origin=origin,
            parent_run_id=fields.get("parent_run_id"),
            sweep_id=fields.get("sweep_id"),
            notes=str(fields.get("notes") or ""),
            error=str(fields.get("error") or ""),
        )

        with transaction(self._con) as con:
            (prior,) = con.execute(
                "SELECT COUNT(*) FROM runs WHERE strategy = ? AND config_hash = ?",
                (record.strategy, record.config_hash),
            ).fetchone()
            record.attempt = int(prior) + 1
            con.execute(
                """
                INSERT INTO runs (run_id, strategy, kind, status, created_at, finished_at,
                                  git_commit, config_hash, data_version, params, config,
                                  metrics, start_ts, end_ts, origin, parent_run_id,
                                  sweep_id, notes, attempt, error)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.run_id,
                    record.strategy,
                    record.kind,
                    record.status,
                    _iso(record.created_at),
                    _iso(record.finished_at),
                    record.git_commit,
                    record.config_hash,
                    record.data_version,
                    _dumps(record.params),
                    _dumps(record.config),
                    _dumps(record.metrics),
                    _iso(record.start),
                    _iso(record.end),
                    record.origin,
                    record.parent_run_id,
                    record.sweep_id,
                    record.notes,
                    record.attempt,
                    record.error,
                ),
            )
            self._write_metrics(con, record.run_id, record.metrics)
        return record

    def finish(
        self,
        run_id: str,
        *,
        metrics: Mapping[str, Any],
        status: str = "ok",
        error: str = "",
    ) -> RunRecord:
        metrics = dict(metrics or {})
        finished = utcnow()
        with transaction(self._con) as con:
            cur = con.execute(
                """
                UPDATE runs SET metrics = ?, status = ?, error = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (_dumps(metrics), status, error, _iso(finished), run_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"no such run: {run_id!r}")
            con.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))
            self._write_metrics(con, run_id, metrics)
        record = self.get(run_id)
        if record is None:  # deleted between the UPDATE and the read-back
            raise ValueError(f"no such run: {run_id!r}")
        return record

    def delete(self, run_id: str) -> bool:
        """Drops the row and its metrics. Journal entries and the artifact
        directory are left alone -- forensics outlive the index."""
        with transaction(self._con) as con:
            cur = con.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0

    @staticmethod
    def _write_metrics(con: sqlite3.Connection, run_id: str, metrics: Mapping[str, Any]) -> None:
        rows = []
        for name, value in metrics.items():
            if isinstance(value, bool):
                rows.append((run_id, str(name), float(value), str(value)))
            elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                num = float(value)
                rows.append((run_id, str(name), None if pd.isna(num) else num, None))
            elif value is None:
                rows.append((run_id, str(name), None, None))
            else:
                rows.append((run_id, str(name), None, str(value)))
        if rows:
            con.executemany(
                "INSERT OR REPLACE INTO run_metrics (run_id, name, value, text_value) "
                "VALUES (?,?,?,?)",
                rows,
            )

    # --- read --------------------------------------------------------------

    def get(self, run_id: str) -> RunRecord | None:
        row = self._con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return run_from_row(row) if row is not None else None

    def list(
        self,
        *,
        strategy: str | None = None,
        kind: str | None = None,
        origin: str | None = None,
        sweep_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at desc",
    ) -> list[RunRecord]:
        where: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("strategy", strategy),
            ("kind", kind),
            ("origin", origin),
            ("sweep_id", sweep_id),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # rowid breaks created_at ties: runs registered inside the same
        # microsecond still list in insertion order.
        sql = (
            f"SELECT * FROM runs {clause} ORDER BY {_order_clause(order_by)}, rowid DESC "
            "LIMIT ? OFFSET ?"
        )
        args.extend([max(int(limit), 0), max(int(offset), 0)])
        return [run_from_row(r) for r in self._con.execute(sql, args).fetchall()]

    def compare(self, run_ids: Sequence[str]) -> pd.DataFrame:
        """One row per requested run, in the order asked for.

        Columns: fixed provenance metadata, then the params that actually
        *differ* across the set (prefixed ``param.``), then the union of metric
        keys. Identical params are dropped because a comparison table that
        repeats the same twelve values in every row hides the two that moved.
        """
        ids = [str(r) for r in run_ids]
        if not ids:
            return pd.DataFrame(columns=list(_META_COLUMNS))
        records: list[RunRecord] = []
        missing: list[str] = []
        for rid in ids:
            rec = self.get(rid)
            if rec is None:
                missing.append(rid)
            else:
                records.append(rec)
        if missing:
            raise ValueError(f"unknown run_id(s): {missing}")

        param_keys = sorted({k for rec in records for k in rec.params})
        varying = [
            k
            for k in param_keys
            if len({_dumps(rec.params.get(k)) for rec in records}) > 1 or len(records) == 1
        ]
        # A metric may share a name with a provenance column -- `attempt` is
        # copied into metrics.json so a bare metrics blob is self-describing.
        # Emitting both would put the same label on two columns, and pandas
        # silently drops one, so provenance wins and the duplicate is skipped.
        metric_keys = sorted(
            {k for rec in records for k in rec.metrics} - set(_META_COLUMNS)
        )

        rows: list[dict[str, Any]] = []
        for rec in records:
            row: dict[str, Any] = {
                "run_id": rec.run_id,
                "strategy": rec.strategy,
                "kind": rec.kind,
                "status": rec.status,
                "origin": rec.origin,
                "attempt": rec.attempt,
                "created_at": _iso(rec.created_at),
                "finished_at": _iso(rec.finished_at),
                "git_commit": rec.git_commit,
                "config_hash": rec.config_hash,
                "data_version": rec.data_version,
                "sweep_id": rec.sweep_id,
                "notes": rec.notes,
            }
            for k in varying:
                row[f"param.{k}"] = rec.params.get(k)
            for k in metric_keys:
                row[k] = rec.metrics.get(k)
            rows.append(row)

        columns = list(_META_COLUMNS) + [f"param.{k}" for k in varying] + metric_keys
        return pd.DataFrame(rows, columns=columns)

    # --- the attempt counter -----------------------------------------------

    def attempts(self, strategy: str, config_hash: str | None = None) -> int:
        """Runs recorded for this strategy, narrowed to one config when given."""
        if config_hash is None:
            row = self._con.execute(
                "SELECT COUNT(*) FROM runs WHERE strategy = ?", (strategy,)
            ).fetchone()
        else:
            row = self._con.execute(
                "SELECT COUNT(*) FROM runs WHERE strategy = ? AND config_hash = ?",
                (strategy, config_hash),
            ).fetchone()
        return int(row[0])

    def family_attempts(self, strategy: str) -> int:
        """Every run of the strategy regardless of config -- the number that
        answers "how many variations were tried before this one was shown to
        me"."""
        return self.attempts(strategy)


def _order_clause(order_by: str) -> str:
    """Allowlist the ORDER BY; it is the one place user text reaches the SQL."""
    m = _ORDER_BY_RE.match(order_by or "")
    if not m:
        raise ValueError(f"bad order_by {order_by!r}; expected '<column> [asc|desc]'")
    column, direction = m.group(1), (m.group(2) or "asc").lower()
    if column not in _ORDER_COLUMNS:
        raise ValueError(f"cannot order by {column!r}; allowed: {sorted(_ORDER_COLUMNS)}")
    return f"{column} {direction.upper()}"


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in text.strip()).strip("_").lower()
    return (out[:24] or "run")


def _new_run_id(created: datetime, strategy: str) -> str:
    """Sortable and self-describing: ``runs/`` is browsed by humans, and a
    directory listing that sorts chronologically beats one full of opaque uuids.
    """
    import uuid

    return f"{_slug(strategy)}-{to_utc(created).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def git_commit() -> str | None:
    """Short HEAD, or None outside a repo.

    Never raises: the lab must be usable in a plain directory, and losing
    commit attribution is a degraded run record, not a failed one.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(get_settings().paths.root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _canonical(value: Any) -> Any:
    """Reduce a config to a form where equivalent configs compare equal:
    key order does not matter, tuples and sets are lists, 1 and 1.0 are the
    same knob, and anything exotic degrades to its string form."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_dumps(_canonical(v)) for v in value))
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, Decimal)):
        return float(value) if isinstance(value, Decimal) else value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, datetime):
        return to_utc(value).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable 16-hex digest of a config mapping, independent of key order.

    Stable across processes and machines (no ``hash()``, no id()), because it
    is half of the reproducibility claim: the same digest must mean the same
    experiment tomorrow, on another checkout.
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError(f"config_hash expects a mapping, got {type(config).__name__}")
    blob = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_artifact_dir(run_id: str) -> Path:
    """``runs/<run_id>/`` -- created on demand; the backtest runner writes
    metrics.json, equity.parquet, trades.csv, config.json, decisions.jsonl here."""
    if not run_id or any(c in run_id for c in '/\\:*?"<>|'):
        raise ValueError(f"unsafe run_id for a directory name: {run_id!r}")
    path = get_settings().paths.runs / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_metrics(con: sqlite3.Connection, run_id: str, metrics: Mapping[str, Any]) -> None:
    """Metric upsert without touching the run row; used by walk-forward, which
    scores a run after the fact."""
    with transaction(con):
        RunRegistry._write_metrics(con, run_id, dict(metrics))


def iter_runs(con: sqlite3.Connection | None = None, **filters: Any) -> Iterable[RunRecord]:
    return RunRegistry(con).list(**filters)
