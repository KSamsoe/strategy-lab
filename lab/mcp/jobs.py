"""Long-running tool calls as background jobs.

Every tool in this server is a synchronous call, which caps a piece of work at
whatever the client's tool timeout happens to be. That is fine for a backtest
and fatal for a 1,000-sample permutation test that takes two hours on 36 cores:
the client gives up, the connection closes, and the work either dies with it or
finishes into a void. Without this module the homelab is a faster laptop.

A job is one tool call run in a separate, *detached* process -- launched with
``subprocess`` rather than ``multiprocessing`` so that the server does not wait
on it at shutdown and a server restart does not take it down. The child writes
its result to a file the moment it finishes, so a finished job survives the client
disconnecting, the server being restarted, and the agent that asked for it
forgetting that it did. Records live under ``data/jobs/<job_id>.json`` and are
plain enough to read by hand.

Jobs queue rather than reject: a cap on concurrent jobs (``LAB_MCP_MAX_JOBS``)
keeps a batch of permutation tests from oversubscribing the machine, and the
ones past the cap wait their turn.

What this is not: a distributed scheduler. One server, one machine, one queue.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

#: Tools that may run as jobs: everything that does real compute. `backtest` is
#: here because an intraday backtest on years of 15-minute bars is a job too.
JOBABLE = (
    "backtest",
    "neighbourhood",
    "ablate",
    "compare_universes",
    "regimes",
    "cost_sensitivity",
    "bootstrap",
    "bootstrap_vs_benchmark",
    "permutation_test",
    "walk_forward",
    "live_vs_backtest",
    "signal_scan",
    "conditional_returns",
)

TERMINAL = ("done", "failed", "cancelled", "lost")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jobs_dir() -> Path:
    from lab.config import get_settings

    d = get_settings().paths.data / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def max_jobs() -> int:
    raw = os.getenv("LAB_MCP_MAX_JOBS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, (os.cpu_count() or 2) // 4)


def _resolve_tool(name: str) -> Callable[..., Any]:
    """The tool function by name, from the same modules the server registers."""
    if name not in JOBABLE:
        raise ValueError(f"{name!r} cannot run as a job; jobable tools: {', '.join(JOBABLE)}")
    from lab.mcp import tools_core, tools_experiment, tools_explore, tools_live, tools_validate

    for mod in (tools_core, tools_experiment, tools_explore, tools_validate, tools_live):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise ValueError(f"no tool named {name!r}")


def _child(record_path: str) -> None:
    """Runs in the worker process. Writes the result file, then the record."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    rec_path = Path(record_path)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    res_path = rec_path.with_name(rec_path.stem + ".result.json")
    try:
        fn = _resolve_tool(rec["tool"])
        result = fn(**rec["args"])
        res_path.write_text(json.dumps(result, default=str), encoding="utf-8")
        _patch(rec_path, status="done", finished_at=_now(), result_path=str(res_path))
    except BaseException as exc:  # noqa: BLE001 - the record must say what happened
        _patch(
            rec_path,
            status="failed",
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-4000:],
        )


def _patch(path: Path, **fields: Any) -> dict[str, Any]:
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rec.update(fields)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return rec


class JobQueue:
    """In-process scheduler: a pending list, a running dict, a cap."""

    def __init__(self, cap: int | None = None) -> None:
        self.cap = int(cap or max_jobs())
        self._lock = threading.RLock()
        self._pending: list[str] = []
        self._running: dict[str, subprocess.Popen] = {}

    # --- submit ---------------------------------------------------------------

    def submit(self, tool: str, args: Mapping[str, Any] | None = None, note: str = "") -> dict[str, Any]:
        _resolve_tool(tool)  # fail now, not in the child
        from lab.engine.events import new_id

        job_id = new_id("job_")
        path = jobs_dir() / f"{job_id}.json"
        rec = {
            "job_id": job_id,
            "tool": tool,
            "args": dict(args or {}),
            "note": note,
            "status": "queued",
            "submitted_at": _now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "result_path": None,
            "error": None,
        }
        path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        with self._lock:
            self._pending.append(job_id)
            self._pump()
        return self.status(job_id)

    def _pump(self) -> None:
        """Start queued jobs while there is room. Caller holds the lock."""
        self._reap()
        while self._pending and len(self._running) < self.cap:
            job_id = self._pending.pop(0)
            path = jobs_dir() / f"{job_id}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("status") != "queued":
                continue
            log = open(jobs_dir() / f"{job_id}.log", "ab")  # noqa: SIM115 - handed to the child
            flags = 0
            if os.name == "nt":
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                [sys.executable, "-m", "lab.mcp.jobs", "--run", str(path)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=flags, start_new_session=(os.name != "nt"),
                cwd=str(_project_root()),
            )
            log.close()
            _patch(path, status="running", started_at=_now(), pid=proc.pid)
            self._running[job_id] = proc

    def _reap(self) -> None:
        """Forget finished processes; mark ones that died without a record."""
        for job_id, proc in list(self._running.items()):
            if proc.poll() is None:
                continue
            del self._running[job_id]
            path = jobs_dir() / f"{job_id}.json"
            rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if rec.get("status") == "running":
                _patch(
                    path, status="failed", finished_at=_now(),
                    error=f"worker exited with code {proc.returncode} before writing a result"
                          f" -- see {jobs_dir() / (job_id + '.log')}",
                )

    # --- read -----------------------------------------------------------------

    def status(self, job_id: str) -> dict[str, Any]:
        path = jobs_dir() / f"{job_id}.json"
        if not path.exists():
            raise ValueError(f"no such job: {job_id}")
        with self._lock:
            self._pump()
        rec = json.loads(path.read_text(encoding="utf-8"))
        # A job marked running whose process this server does not own was
        # started by a previous server. If its pid is gone, it is not coming back.
        if rec.get("status") == "running" and job_id not in self._running:
            if not _pid_alive(rec.get("pid")):
                rec = _patch(
                    path, status="lost", finished_at=_now(),
                    error="the server that started this job is gone and so is the worker",
                )
        if rec.get("status") == "queued":
            with self._lock:
                rec["queue_position"] = (
                    self._pending.index(job_id) + 1 if job_id in self._pending else None
                )
        return rec

    def result(self, job_id: str) -> dict[str, Any]:
        rec = self.status(job_id)
        if rec["status"] != "done":
            return {"job_id": job_id, "status": rec["status"], "error": rec.get("error"),
                    "note": "no result yet" if rec["status"] in ("queued", "running") else None}
        result = json.loads(Path(rec["result_path"]).read_text(encoding="utf-8"))
        return {"job_id": job_id, "status": "done", "tool": rec["tool"], "args": rec["args"],
                "result": result}

    def list(self, limit: int = 25, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._pump()
        out = []
        for path in sorted(jobs_dir().glob("job_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path.name.endswith(".result.json"):
                continue
            try:
                rec = self.status(path.stem)
            except Exception:  # noqa: BLE001 - a half-written record is not a crash
                continue
            if status and rec.get("status") != status:
                continue
            out.append({k: rec.get(k) for k in (
                "job_id", "tool", "status", "note", "submitted_at", "started_at",
                "finished_at", "error", "queue_position",
            )})
            if len(out) >= limit:
                break
        return out

    def cancel(self, job_id: str) -> dict[str, Any]:
        path = jobs_dir() / f"{job_id}.json"
        if not path.exists():
            raise ValueError(f"no such job: {job_id}")
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
                return _patch(path, status="cancelled", finished_at=_now())
            proc = self._running.pop(job_id, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            rec = _patch(path, status="cancelled", finished_at=_now())
            with self._lock:
                self._pump()
            return rec
        rec = self.status(job_id)
        if rec.get("status") == "running" and _pid_alive(rec.get("pid")):
            _kill_pid(int(rec["pid"]))
            return _patch(path, status="cancelled", finished_at=_now())
        return rec


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal

        os.kill(pid, signal.SIGTERM)


def _project_root():
    from lab.config import get_settings

    return get_settings().paths.root


def _pid_alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(int(pid))
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


_queue: JobQueue | None = None


def queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m lab.mcp.jobs")
    ap.add_argument("--run", required=True, help="path to the job record to execute")
    _child(ap.parse_args().run)
