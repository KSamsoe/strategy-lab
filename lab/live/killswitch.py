"""The kill switch: two independent ways to stop trading, neither needing a deploy.

An env flag (``LAB_KILL_SWITCH``) covers "restart the process with trading off";
a sentinel file covers the panic case, where a human -- or the console, or a
cron watchdog -- must stop a *running* process from the outside, immediately,
with nothing more than a file write. Both are read through
:meth:`lab.config.Settings.kill_switch_engaged` so the runner, the gate and the
CLI can never disagree about whether the switch is on.

The asymmetry is deliberate: engaging is cheap and always possible, releasing is
not. :func:`release` can delete the sentinel file, but it cannot unset an env
var in a process it does not own, so it reports whether the switch is *actually*
clear rather than whether it deleted something.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

from lab.config import get_settings
from lab.timeutil import utcnow

log = logging.getLogger(__name__)

#: The env var :func:`lab.config.get_settings` reads. Named here so callers can
#: mention it in an error message without hardcoding the string again.
ENV_FLAG = "LAB_KILL_SWITCH"


def kill_file() -> Path:
    """Where the sentinel lives (``LAB_KILL_FILE``, else ``<data>/KILL``)."""
    return get_settings().kill_file


def engaged() -> tuple[bool, str | None]:
    """``(engaged, reason)``. Checked before every order batch."""
    return get_settings().kill_switch_engaged()


def engage(reason: str = "") -> Path:
    """Write the sentinel file and return its path.

    The payload is informational -- the file's *existence* is the switch, so a
    zero-byte ``touch`` from a shell works exactly as well and is the documented
    panic path.
    """
    path = kill_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": str(reason or "manual kill"),
        "at": utcnow().isoformat(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.warning("kill switch ENGAGED: %s (%s)", payload["reason"], path)
    _journal("kill_switch engaged", payload)
    return path


def release() -> bool:
    """Remove the sentinel and report whether trading is now permitted.

    Returns ``False`` when the env flag is still set: the file is gone but the
    switch is not, and a caller that reads this as "released" would restart into
    a runner that silently refuses every order.
    """
    path = kill_file()
    try:
        path.unlink()
        removed = True
    except FileNotFoundError:
        removed = False
    except OSError as exc:  # a locked/permission-denied sentinel is still ON
        log.error("could not remove kill file %s: %s", path, exc)
        return False

    still_on, reason = engaged()
    log.warning(
        "kill switch release: file %s, now %s",
        "removed" if removed else "absent",
        reason if still_on else "clear",
    )
    _journal(
        "kill_switch released" if not still_on else "kill_switch still engaged",
        {"file_removed": removed, "path": str(path), "still_engaged": still_on, "reason": reason},
    )
    return not still_on


def describe() -> dict[str, Any]:
    """Status payload for ``lab status`` and the console's kill indicator."""
    on, reason = engaged()
    path = kill_file()
    detail: dict[str, Any] = {}
    if path.exists():
        try:
            detail = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A hand-touched empty sentinel is valid and common; it just has
            # nothing to say beyond existing.
            detail = {}
    return {
        "engaged": on,
        "reason": reason,
        "env_flag": ENV_FLAG,
        "env_set": get_settings().kill_switch_env,
        "file": str(path),
        "file_exists": path.exists(),
        "detail": detail,
    }


def _journal(message: str, payload: dict[str, Any]) -> None:
    """Record the flip in the event journal, best-effort.

    Imported lazily and swallowed on failure: a kill switch that cannot engage
    because the journal is unwritable would be the worst possible failure mode.
    """
    try:
        from lab.live import alerts

        alerts.alert("kill_switch", message, **payload)
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("kill switch journal write failed: %s", exc)
