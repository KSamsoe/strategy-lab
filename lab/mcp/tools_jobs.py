"""Submit, poll and collect background jobs.

The pattern for anything that takes longer than a client will wait: `job_start`
returns immediately with a job id; `job_status` says queued / running / done;
`job_result` hands back exactly what the synchronous tool would have returned.
The result is written to disk the moment the worker finishes, so a job outlives
the conversation that started it -- ask `jobs_list` on your next session and
the answer is still there.
"""

from __future__ import annotations

from typing import Any

from lab.mcp._common import jsonable
from lab.mcp.jobs import JOBABLE, queue


def job_start(tool: str, args: dict[str, Any], note: str = "") -> dict[str, Any]:
    """Run a tool in the background and return a job id immediately.

    `tool` is the name of any compute-heavy tool -- `permutation_test`,
    `walk_forward`, `bootstrap`, `ablate`, `compare_universes`, `backtest` and
    the rest -- and `args` is exactly the dict you would have passed to it. Use
    this for anything that would take more than a minute or two: a synchronous
    call that outlives the client's timeout is lost work, and this is not.

    `note` is free text kept with the job so a list of twelve permutation tests
    can be told apart later.
    """
    return jsonable(queue().submit(tool, args, note=note))


def job_status(job_id: str) -> dict[str, Any]:
    """Where a job is: queued (with its position), running, done, failed, cancelled, lost.

    `lost` means the server that started it was restarted and the worker is
    gone; resubmit. A `failed` record carries the error and the tail of the
    traceback.
    """
    return jsonable(queue().status(job_id))


def job_result(job_id: str) -> dict[str, Any]:
    """The finished job's result -- the same dict the synchronous tool returns.

    For a job that is not done yet this says so rather than blocking; poll
    `job_status` and come back.
    """
    return jsonable(queue().result(job_id))


def jobs_list(limit: int = 25, status: str | None = None) -> dict[str, Any]:
    """Recent jobs, newest first, optionally filtered by status.

    Results persist across sessions, so a job started yesterday by another agent
    shows up here with its result intact.
    """
    rows = queue().list(limit=int(limit), status=status)
    return jsonable({
        "count": len(rows), "concurrency": queue().cap, "jobable_tools": list(JOBABLE),
        "jobs": rows,
    })


def job_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a queued job, or terminate a running one."""
    return jsonable(queue().cancel(job_id))
