"""The findings ledger: what this lab has concluded, and on what evidence.

The run registry records what was *run*. Nothing recorded what was *learned*.
"The edge was the ticker list, not the method" and "cross-sectional momentum
only has signal at about 126 days" were the two most valuable things any
session here produced, and both lived in one agent's context window and a
docstring. The next session started from zero and was free to rediscover them
at full price, or to contradict them without knowing it.

A finding is a claim with its evidence attached: the run ids it rests on, the
numbers, the tags, and a status that can move from ``open`` to ``confirmed``
or ``refuted`` as later work bears on it. A finding that turns out to be wrong
is not deleted -- it is marked refuted with the reason, because "we believed X
until Y showed otherwise" is itself a finding.

Stored in the registry database next to the runs it cites.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from lab.engine.events import new_id
from lab.registry.db import connect, transaction
from lab.timeutil import to_utc, utcnow

STATUSES = ("open", "confirmed", "refuted", "superseded")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _loads(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _row(r: sqlite3.Row | Sequence[Any]) -> dict[str, Any]:
    keys = ("id", "created_at", "updated_at", "title", "claim", "evidence", "run_ids",
            "tags", "status", "supersedes", "author", "history")
    d = dict(zip(keys, r))
    d["evidence"] = _loads(d["evidence"], {})
    d["run_ids"] = _loads(d["run_ids"], [])
    d["tags"] = _loads(d["tags"], [])
    d["history"] = _loads(d["history"], [])
    return d


class Findings:
    def __init__(self, con: sqlite3.Connection | None = None) -> None:
        self._con = con if con is not None else connect()

    def record(
        self,
        title: str,
        claim: str,
        *,
        run_ids: Iterable[str] = (),
        tags: Iterable[str] = (),
        evidence: Mapping[str, Any] | None = None,
        status: str = "open",
        supersedes: str | None = None,
        author: str = "",
    ) -> dict[str, Any]:
        title, claim = str(title).strip(), str(claim).strip()
        if not title or not claim:
            raise ValueError("a finding needs both a title and a claim")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if supersedes and self.get(supersedes) is None:
            raise ValueError(f"cannot supersede {supersedes}: no such finding")
        fid = new_id("f_")
        now = utcnow().isoformat()
        with transaction(self._con) as con:
            con.execute(
                "INSERT INTO findings (id, created_at, updated_at, title, claim, evidence, "
                "run_ids, tags, status, supersedes, author, history) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fid, now, now, title, claim, _dumps(dict(evidence or {})),
                    _dumps(sorted({str(r) for r in run_ids if r})),
                    _dumps(sorted({str(t).strip().lower() for t in tags if str(t).strip()})),
                    status, supersedes, author, _dumps([]),
                ),
            )
            if supersedes:
                con.execute(
                    "UPDATE findings SET status = 'superseded', updated_at = ? WHERE id = ?",
                    (now, supersedes),
                )
        return self.get(fid)  # type: ignore[return-value]

    def update(self, finding_id: str, *, status: str | None = None, note: str = "",
               run_ids: Iterable[str] = (), evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cur = self.get(finding_id)
        if cur is None:
            raise ValueError(f"no such finding: {finding_id}")
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        now = utcnow().isoformat()
        history = list(cur["history"]) + [{
            "at": now, "from": cur["status"], "to": status or cur["status"], "note": note,
        }]
        merged_runs = sorted(set(cur["run_ids"]) | {str(r) for r in run_ids if r})
        merged_ev = dict(cur["evidence"]) | dict(evidence or {})
        with transaction(self._con) as con:
            con.execute(
                "UPDATE findings SET status = ?, updated_at = ?, run_ids = ?, evidence = ?, "
                "history = ? WHERE id = ?",
                (status or cur["status"], now, _dumps(merged_runs), _dumps(merged_ev),
                 _dumps(history), finding_id),
            )
        return self.get(finding_id)  # type: ignore[return-value]

    def get(self, finding_id: str) -> dict[str, Any] | None:
        r = self._con.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        return _row(r) if r is not None else None

    def search(
        self,
        query: str | None = None,
        *,
        tags: Iterable[str] = (),
        run_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where, args = [], []
        if query:
            where.append("(title LIKE ? OR claim LIKE ?)")
            args += [f"%{query}%", f"%{query}%"]
        if run_id:
            where.append("run_ids LIKE ?")
            args.append(f'%"{run_id}"%')
        for t in tags:
            where.append("tags LIKE ?")
            args.append(f'%"{str(t).strip().lower()}"%')
        if status:
            where.append("status = ?")
            args.append(status)
        elif not include_superseded:
            where.append("status != 'superseded'")
        sql = "SELECT * FROM findings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, int(limit)))
        return [_row(r) for r in self._con.execute(sql, args).fetchall()]

    def digest(self) -> str:
        """Every live finding as markdown -- what a new session should read first."""
        rows = self.search(limit=500)
        if not rows:
            return "# Findings\n\nNothing recorded yet.\n"
        out = ["# Findings", "", "Newest first. Status in brackets; refuted claims are kept "
               "because knowing what was believed and why it fell is a finding too.", ""]
        for r in rows:
            tags = ", ".join(r["tags"]) if r["tags"] else ""
            out.append(f"## [{r['status']}] {r['title']}  `{r['id']}`")
            out.append("")
            out.append(r["claim"].strip())
            meta = []
            if r["run_ids"]:
                meta.append("runs: " + ", ".join(r["run_ids"]))
            if tags:
                meta.append("tags: " + tags)
            if r["evidence"]:
                meta.append("evidence: " + _dumps(r["evidence"]))
            if meta:
                out.append("")
                out.extend(f"- {m}" for m in meta)
            if r["history"]:
                last = r["history"][-1]
                out.append(f"- last change: {last.get('from')} -> {last.get('to')}"
                           + (f" ({last.get('note')})" if last.get("note") else ""))
            out.append("")
        return "\n".join(out)
