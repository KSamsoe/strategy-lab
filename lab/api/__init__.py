"""The console's read layer, and the auth policy every route in it shares.

The API is a *reader*. It talks to the registry, the two journals and the run
artifacts -- never to a broker, never to a live runner's process. That is why a
console bug cannot crash a live run, and why the console still answers "what did
it do this morning" after the runner has died.

The mutation surface is deliberately asymmetric: pause, cancel open orders and
the kill switch exist; start, resume and raise-a-limit do not exist as routes at
all. Anything that could *increase* exposure stays a CLI action behind the
manual checklist, so a stolen laptop, a misclick or a compromised browser tab
cannot make the system riskier than it already is.

Auth lives here rather than in ``app.py`` because ``ws.py`` needs the same
check and importing it from ``app`` would close an import cycle (app imports the
routers, the routers would import app). ``create_app`` is re-exported lazily via
``__getattr__`` for the same reason: importing ``lab.api`` must not drag in the
route modules.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import Header, HTTPException, status

from lab.config import get_settings

__all__ = [
    "app",
    "create_app",
    "serve",
    "auth_required",
    "bearer_token",
    "token_ok",
    "require_token",
]

_LAZY = {"app", "create_app", "serve"}


def auth_required() -> bool:
    """A token is optional: this is a localhost single-user tool, and forcing a
    secret on ``lab ui`` would be ceremony. Setting ``LAB_UI_TOKEN`` turns the
    lock on -- which is what the "check it from my laptop on the LAN" case
    needs."""
    return bool(get_settings().ui_token)


def bearer_token(header_value: str | None) -> str | None:
    """``Authorization: Bearer <t>`` -> ``<t>``. A bare value is accepted too,
    because ``curl -H "Authorization: $LAB_UI_TOKEN"`` is what people type."""
    if not header_value:
        return None
    value = header_value.strip()
    scheme, _, rest = value.partition(" ")
    if scheme.lower() == "bearer":
        return rest.strip() or None
    return value or None


def token_ok(presented: str | None) -> bool:
    expected = get_settings().ui_token
    if not expected:
        return True
    if not presented:
        return False
    # Constant time: the token is short and a local attacker can make a lot of
    # requests, so a naive == leaks it one byte at a time.
    return secrets.compare_digest(str(presented), str(expected))


def require_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency guarding every ``/api`` route except ``/api/health``.

    Attached once per router at ``include_router`` time rather than copied into
    each handler -- an auth check you have to remember to write is an auth check
    that eventually goes missing from one route.
    """
    if token_ok(bearer_token(authorization)):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing or invalid bearer token (LAB_UI_TOKEN is set)",
        headers={"WWW-Authenticate": "Bearer"},
    )


def __getattr__(name: str) -> Any:  # PEP 562: keep `lab.api` import-cheap
    if name in _LAZY:
        from lab.api import app as _app_module

        return getattr(_app_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
