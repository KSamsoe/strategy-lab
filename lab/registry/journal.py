"""The two journals: why a position exists, and what the system is doing.

The **decision journal** stores one row per ``on_bar``: the input tape the
Context actually served, the intents the strategy emitted, the gate's verdict on
each, and the resulting order ids. Because the Context mediates every read,
capture is a wrapper rather than a strategy change -- and "why does this
position exist" becomes a lookup keyed by (run_id, timestamp) instead of an
investigation.

The **event journal** is append-only and monotonically sequenced. ``seq`` is the
resumable cursor: the WebSocket and ``/api/live/events?since=`` both replay from
it, so a dropped connection costs a reconnect and not a gap. Alert webhooks are
just another consumer of the same stream. Nothing in the UI path touches the
runner or the broker -- the console reads these tables, which is why a UI bug
cannot crash a live run and why the console still works as forensics after the
runner has died.

Heartbeats are ordinary events (``kind='heartbeat'``) rather than a mutable
status table, so the freshness strip is derived from the same append-only log as
everything else. ``last_heartbeats()`` computes ``age_s`` at call time: staleness
is displayed, never assumed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping, Sequence

from lab.engine.events import Decision, EventKind, JournalEvent, event
from lab.registry.db import connect, journal_path, transaction
from lab.timeutil import to_utc, utcnow

#: How long ``subscribe`` waits between polls. Short enough that the console
#: feels live, long enough that an idle tail costs nothing.
POLL_SECONDS = 0.25


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _loads(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _ticker_index(*collections: Iterable[Mapping[str, Any]]) -> str:
    """``,AAPL,MSFT,`` -- delimiters on both ends so a LIKE filter cannot match
    a substring of a longer ticker."""
    seen: list[str] = []
    for items in collections:
        for item in items or ():
            t = str(item.get("ticker") or "").upper()
            if t and t not in seen:
                seen.append(t)
    return "," + ",".join(seen) + "," if seen else ""


class DecisionJournal:
    def __init__(self, con: sqlite3.Connection | None = None) -> None:
        self._con = con if con is not None else connect(journal_path())

    @property
    def con(self) -> sqlite3.Connection:
        return self._con

    def append(self, decision: Decision) -> None:
        with transaction(self._con) as con:
            con.execute(*_decision_insert(decision))

    def bulk_append(self, decisions: Iterable[Decision]) -> int:
        """One transaction for the whole batch -- a backtest writes thousands of
        these and per-row commits dominate the run time otherwise."""
        rows = [_decision_insert(d)[1] for d in decisions]
        if not rows:
            return 0
        with transaction(self._con) as con:
            con.executemany(_DECISION_SQL, rows)
        return len(rows)

    def list(
        self,
        run_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
        offset: int = 0,
        ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        where = ["run_id = ?"]
        args: list[Any] = [run_id]
        if start is not None:
            where.append("at >= ?")
            args.append(to_utc(start).isoformat())
        if end is not None:
            where.append("at <= ?")
            args.append(to_utc(end).isoformat())
        if ticker:
            where.append("tickers LIKE ?")
            args.append(f"%,{ticker.upper()},%")
        args.extend([max(int(limit), 0), max(int(offset), 0)])
        sql = (
            f"SELECT * FROM decisions WHERE {' AND '.join(where)} "
            "ORDER BY at ASC, rowid ASC LIMIT ? OFFSET ?"
        )
        return [_decision_from_row(r) for r in self._con.execute(sql, args).fetchall()]

    def get(self, decision_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT * FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        return _decision_from_row(row) if row is not None else None

    def count(self, run_id: str) -> int:
        row = self._con.execute(
            "SELECT COUNT(*) FROM decisions WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])


_DECISION_SQL = """
INSERT OR REPLACE INTO decisions
    (id, run_id, strategy, at, inputs, intents, verdicts, order_ids, portfolio,
     logs, agent, duration_ms, summary, tickers)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def _decision_insert(decision: Decision) -> tuple[str, tuple[Any, ...]]:
    if not hasattr(decision, "to_dict"):
        raise ValueError(f"expected a Decision, got {type(decision).__name__}")
    d = decision.to_dict()
    for required in ("id", "run_id", "at"):
        if not d.get(required):
            raise ValueError(f"decision is missing {required!r}")
    intents = d.get("intents") or []
    verdicts = d.get("verdicts") or []
    summary = decision.summary() if hasattr(decision, "summary") else ""
    return _DECISION_SQL, (
        d["id"],
        d["run_id"],
        d.get("strategy") or "",
        to_utc(d["at"]).isoformat(),
        _dumps(d.get("inputs") or {}),
        _dumps(intents),
        _dumps(verdicts),
        _dumps(d.get("order_ids") or []),
        _dumps(d.get("portfolio") or {}),
        _dumps(d.get("logs") or []),
        _dumps(d["agent"]) if d.get("agent") is not None else None,
        float(d.get("duration_ms") or 0.0),
        summary,
        _ticker_index(intents, verdicts),
    )


def _decision_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Mirrors ``Decision.to_dict()`` plus ``summary``. Timestamps stay ISO-8601
    strings so the API can hand the dict straight to the JSON encoder."""
    r = dict(row)
    return {
        "id": r["id"],
        "run_id": r["run_id"],
        "strategy": r["strategy"],
        "at": r["at"],
        "inputs": _loads(r["inputs"], {}),
        "intents": _loads(r["intents"], []),
        "verdicts": _loads(r["verdicts"], []),
        "order_ids": _loads(r["order_ids"], []),
        "portfolio": _loads(r["portfolio"], {}),
        "logs": _loads(r["logs"], []),
        "agent": _loads(r["agent"], None),
        "duration_ms": float(r["duration_ms"] or 0.0),
        "summary": r["summary"],
    }


class EventJournal:
    def __init__(self, con: sqlite3.Connection | None = None) -> None:
        self._con = con if con is not None else connect(journal_path())

    @property
    def con(self) -> sqlite3.Connection:
        return self._con

    def append(self, ev: JournalEvent) -> int:
        """Assign and return ``seq``. Re-appending an event with an id already
        in the log returns the existing seq instead of raising: replaying a
        crashed runner's tail must be safe."""
        if not hasattr(ev, "to_dict"):
            raise ValueError(f"expected a JournalEvent, got {type(ev).__name__}")
        d = ev.to_dict()
        if not d.get("id"):
            raise ValueError("event is missing an id")
        row = (
            d["id"],
            to_utc(d["at"]).isoformat(),
            str(d["kind"]),
            d.get("source") or "",
            d.get("run_id"),
            d.get("strategy"),
            (d.get("ticker") or None),
            d.get("message") or "",
            _dumps(d.get("payload") or {}),
        )
        with transaction(self._con) as con:
            cur = con.execute(
                """
                INSERT OR IGNORE INTO events
                    (id, at, kind, source, run_id, strategy, ticker, message, payload)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            if cur.rowcount:
                seq = int(cur.lastrowid)
            else:
                seq = int(
                    con.execute("SELECT seq FROM events WHERE id = ?", (d["id"],)).fetchone()[0]
                )
        ev.seq = seq
        return seq

    def bulk_append(self, events: Iterable[JournalEvent]) -> int:
        n = 0
        with transaction(self._con):
            for ev in events:
                self.append(ev)
                n += 1
        return n

    def tail(
        self,
        since_seq: int = 0,
        *,
        limit: int = 500,
        run_id: str | None = None,
        strategy: str | None = None,
        kinds: Sequence[str | EventKind] | None = None,
    ) -> list[dict[str, Any]]:
        """Rows with ``seq > since_seq``, oldest first. Ascending because the
        caller's cursor is the last seq it saw, and it needs the *next* ones."""
        where = ["seq > ?"]
        args: list[Any] = [int(since_seq)]
        if run_id is not None:
            where.append("run_id = ?")
            args.append(run_id)
        if strategy is not None:
            where.append("strategy = ?")
            args.append(strategy)
        if kinds:
            wanted = [k.value if isinstance(k, EventKind) else str(k) for k in kinds]
            where.append(f"kind IN ({','.join('?' * len(wanted))})")
            args.extend(wanted)
        args.append(max(int(limit), 0))
        sql = f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY seq ASC LIMIT ?"
        return [_event_from_row(r) for r in self._con.execute(sql, args).fetchall()]

    def latest_seq(self) -> int:
        row = self._con.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
        return int(row[0])

    def get(self, event_id: str) -> dict[str, Any] | None:
        row = self._con.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _event_from_row(row) if row is not None else None

    def count(self) -> int:
        return int(self._con.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def subscribe(
        self,
        *,
        since_seq: int | None = None,
        poll: float = POLL_SECONDS,
        limit: int = 500,
        run_id: str | None = None,
        strategy: str | None = None,
        kinds: Sequence[str | EventKind] | None = None,
        idle_timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Poll the log and yield rows as they land.

        Polling, not triggers: SQLite has no server-side notify, and a 250ms
        poll of an indexed integer cursor is cheaper than any workaround. The
        default cursor is *now*, so a new subscriber gets the live tail and not
        the whole history; pass ``since_seq`` to resume (that is what the
        WebSocket's ``?since=`` does).

        ``GeneratorExit`` -- ``gen.close()``, or a WebSocket client
        disconnecting -- unwinds the loop immediately.
        """
        if poll <= 0:
            raise ValueError("poll must be > 0 seconds")
        cursor = self.latest_seq() if since_seq is None else int(since_seq)
        idle = 0.0
        try:
            while True:
                batch = self.tail(
                    cursor, limit=limit, run_id=run_id, strategy=strategy, kinds=kinds
                )
                if batch:
                    idle = 0.0
                    for row in batch:
                        cursor = max(cursor, int(row["seq"]))
                        yield row
                    continue
                if idle_timeout is not None and idle >= idle_timeout:
                    return
                time.sleep(poll)
                idle += poll
        except GeneratorExit:
            return

    # --- heartbeats ---------------------------------------------------------

    def heartbeat(self, source: str, *, meta: Mapping[str, Any] | None = None) -> None:
        if not source:
            raise ValueError("heartbeat needs a source")
        self.append(
            event(
                EventKind.HEARTBEAT,
                source,
                at=utcnow(),
                message="heartbeat",
                payload=dict(meta or {}),
            )
        )

    def last_heartbeats(self) -> dict[str, dict[str, Any]]:
        """``source -> {at, age_s, meta, seq}``, with ``age_s`` measured now.

        The freshness strip renders whatever this returns; deciding what counts
        as stale is the caller's business, because a 30s daily-runner heartbeat
        and a 2s broker heartbeat have very different thresholds.
        """
        rows = self._con.execute(
            """
            SELECT source, at, payload, MAX(seq) AS seq
            FROM events WHERE kind = ? GROUP BY source
            """,
            (EventKind.HEARTBEAT.value,),
        ).fetchall()
        now = utcnow()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            at = to_utc(r["at"])
            out[r["source"]] = {
                "at": r["at"],
                "age_s": (now - at).total_seconds(),
                "meta": _loads(r["payload"], {}),
                "seq": int(r["seq"]),
            }
        return out


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    r = dict(row)
    return {
        "seq": int(r["seq"]),
        "id": r["id"],
        "at": r["at"],
        "kind": r["kind"],
        "source": r["source"],
        "run_id": r["run_id"],
        "strategy": r["strategy"],
        "ticker": r["ticker"],
        "message": r["message"],
        "payload": _loads(r["payload"], {}),
    }
