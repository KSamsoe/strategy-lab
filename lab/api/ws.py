"""``WS /api/ws/events`` -- the event journal, tailed by monotonic ``seq``.

The cursor is the whole design. A client reconnects with the last ``seq`` it
rendered and gets exactly the events it missed, so a dropped WebSocket costs a
reconnect rather than a hole in the tape. ``seq`` is an AUTOINCREMENT primary
key, so an id is never reused even after a purge -- which is what makes
resuming safe rather than approximately safe.

Idle cost is one indexed ``seq > ?`` query every 250ms and nothing else: the
loop awaits a sleep instead of spinning, and it does its SQLite reads in the
threadpool so a slow disk cannot stall the event loop that is serving the rest
of the console. Two tasks run per connection -- a pump that writes and a drain
that watches for the close frame -- and whichever finishes first cancels the
other, so a client that vanishes mid-idle does not leak a poller forever.

Those two live in an **anyio** task group rather than bare ``asyncio`` tasks.
The distinction is not stylistic: an ASGI handler already runs inside the
server's cancel scope, and tasks spawned with ``asyncio.create_task`` sit
outside it. On disconnect the scope then cancels this coroutine while it is
still awaiting cleanup for children the scope cannot see, the second
cancellation does not match the count the scope expects to absorb, and it
escapes as a bare ``CancelledError`` at the ASGI boundary -- intermittently, on
whichever connection loses the race. A task group makes the children part of the
scope, so a disconnect unwinds in one pass.
"""

from __future__ import annotations

import functools
from typing import Any

import anyio
from fastapi import APIRouter, Query, WebSocket
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect, WebSocketState

from lab.api import bearer_token, token_ok
from lab.api.models import EventRow, WSEvent, WSHello
from lab.registry.journal import POLL_SECONDS, EventJournal
from lab.timeutil import utcnow

router = APIRouter(tags=["live"])

#: Rows per poll. Caps the burst a client gets when it resumes from far behind,
#: so a week-old cursor streams in chunks instead of one multi-megabyte frame.
BATCH = 500

_CLOSE_POLICY = 1008  # RFC 6455 policy violation


@router.websocket("/api/ws/events")
async def ws_events(
    websocket: WebSocket,
    since: int | None = Query(default=None, ge=0),
    limit: int = Query(default=BATCH, ge=1, le=2000),
    run_id: str | None = None,
    strategy: str | None = None,
    kinds: str | None = None,
    token: str | None = Query(default=None),
) -> None:
    """Stream events with ``seq > since``. Omitting ``since`` starts at the head
    (the live tail); ``since=0`` replays the log from the beginning."""
    # A browser's WebSocket constructor cannot set headers, so the token is also
    # accepted as a query parameter. It never leaves this machine's loopback
    # interface unless the operator deliberately exposed the port, which is the
    # same case that made them set a token in the first place.
    presented = token or bearer_token(websocket.headers.get("authorization"))
    if not token_ok(presented):
        await websocket.close(code=_CLOSE_POLICY, reason="invalid token")
        return

    await websocket.accept()
    journal = EventJournal()
    latest = await run_in_threadpool(journal.latest_seq)
    cursor = latest if since is None else int(since)

    await websocket.send_json(
        WSHello(since=cursor, latest_seq=latest, at=utcnow()).model_dump(mode="json")
    )

    filters: dict[str, Any] = {
        "limit": limit,
        "run_id": run_id,
        "strategy": strategy,
        "kinds": [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None,
    }
    async with anyio.create_task_group() as tg:
        tg.start_soon(_pump, websocket, journal, cursor, filters, tg.cancel_scope)
        tg.start_soon(_drain, websocket, tg.cancel_scope)
    # Reached only once both children are done, and never while cancelled: the
    # group's scope absorbs its own cancellation on the way out.
    if websocket.client_state is WebSocketState.CONNECTED:
        await websocket.close()


#: Everything a dead peer can look like from either side of the socket.
#: ``RuntimeError`` is starlette's "send after close" -- the socket went away
#: between the poll and the send, which is the same news as a disconnect.
_GONE = (WebSocketDisconnect, RuntimeError, anyio.ClosedResourceError, anyio.BrokenResourceError)


async def _pump(
    websocket: WebSocket,
    journal: EventJournal,
    cursor: int,
    filters: dict[str, Any],
    scope: anyio.CancelScope,
) -> None:
    try:
        while True:
            rows = await run_in_threadpool(functools.partial(journal.tail, cursor, **filters))
            for row in rows:
                # Advances only over rows the filter actually returned, which is
                # what keeps the cursor honest: a `kinds=` subscriber must not
                # skip past an event it was never shown.
                cursor = max(cursor, int(row["seq"]))
                frame = WSEvent(seq=cursor, event=EventRow.model_validate(row))
                await websocket.send_json(frame.model_dump(mode="json"))
            if not rows:
                await anyio.sleep(POLL_SECONDS)
    except _GONE:
        return
    finally:
        # Whichever child finishes first ends the connection; cancelling an
        # already-cancelled scope is a no-op, so both may do it.
        scope.cancel()


async def _drain(websocket: WebSocket, scope: anyio.CancelScope) -> None:
    """Read and discard client frames. Its real job is to notice the close
    frame while the pump is idle, so a dead connection is reclaimed at once
    instead of at the next event -- which for a quiet strategy could be hours."""
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except _GONE:
        return
    finally:
        scope.cancel()
