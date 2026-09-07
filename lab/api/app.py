"""The FastAPI application: wiring, auth, CORS, and serving the console SPA.

Bound to ``127.0.0.1`` by ``lab ui``. There is no multi-user story and no
deployment story -- the security model is "the port is on loopback", with an
optional bearer token for the one case that leaves it (checking the fleet from a
laptop on the same LAN).

``/api/openapi.json`` is the contract itself: the CLI's ``--json`` payloads and
these responses are the same shapes, so the console is a view over the API a
script would use, not a second API written for a UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lab.api import auth_required, require_token
from lab.api.models import Health, KillSwitchState
from lab.config import get_settings
from lab.timeutil import utcnow

#: Vite's dev server, both spellings of localhost. Nothing else: a wildcard here
#: would let any page the operator happens to have open drive the control routes.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

TITLE = "strategy-lab console API"


def version() -> str:
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        return _version("strategy-lab")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0+unknown"


def console_dist(override: str | Path | None = None) -> Path:
    """Where the built SPA lives. Overridable by env so a dev build elsewhere
    can be served without moving files."""
    if override:
        return Path(override)
    env = os.getenv("LAB_CONSOLE_DIST")
    if env:
        return Path(env)
    return get_settings().paths.root / "console" / "dist"


def create_app(*, console_dist_dir: str | Path | None = None) -> FastAPI:
    from lab.api import routes_live, routes_research, routes_runs, ws

    app = FastAPI(
        title=TITLE,
        version=version(),
        summary="Read layer for the strategy-lab console. Reads journals, never brokers.",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_exception_handler(ValueError, _value_error_handler)

    @app.get("/api/health", response_model=Health, tags=["health"])
    def health() -> Health:
        """Unauthenticated on purpose -- it is how you find out *whether* the
        thing is up, including when it is up but its database is not."""
        settings = get_settings()
        engaged, reason = settings.kill_switch_engaged()
        ok, detail = True, ""
        runs = events = latest = 0
        try:
            from lab.registry.db import connect
            from lab.registry.journal import EventJournal

            runs = int(connect().execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            journal = EventJournal()
            events, latest = journal.count(), journal.latest_seq()
        except Exception as exc:  # a broken store is a degraded answer, not a 500
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        return Health(
            ok=ok,
            version=version(),
            now=utcnow(),
            auth_required=auth_required(),
            console_built=(console_dist(console_dist_dir) / "index.html").exists(),
            kill_switch=KillSwitchState(engaged=engaged, reason=reason),
            runs=runs,
            events=events,
            latest_seq=latest,
            detail=detail,
        )

    guard = [Depends(require_token)]
    app.include_router(routes_runs.router, dependencies=guard)
    app.include_router(routes_live.router, dependencies=guard)
    app.include_router(routes_research.router, dependencies=guard)
    # The WebSocket authenticates inside its handler: an HTTP dependency cannot
    # answer a failed handshake with a 1008 close frame.
    app.include_router(ws.router)

    _mount_console(app, console_dist(console_dist_dir))
    return app


def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """The lab's convention is ``ValueError`` for bad input (contracts §
    conventions), which is a 400 here rather than an unhandled 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _mount_console(app: FastAPI, dist: Path) -> None:
    """Serve the SPA if it is built, a build hint if it is not.

    Never fatal: the API is useful on its own (``curl``, an agent loop, the
    OpenAPI schema), so a missing frontend must not stop ``lab ui`` from coming
    up. Registered last so the catch-all cannot shadow an API route.
    """
    index = dist / "index.html"
    assets = dist / "assets"
    if index.exists() and assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> Any:
        if full_path.startswith("api/"):
            # Without this an unknown /api path would render the SPA with a 200
            # and the frontend would parse HTML as JSON.
            raise HTTPException(status_code=404, detail=f"no such endpoint: /{full_path}")
        if not index.exists():
            return HTMLResponse(_placeholder(dist), status_code=200)
        candidate = (dist / full_path).resolve()
        root = dist.resolve()
        # `..%2f` in a URL is normalized by most clients but not all; check.
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        # Deep links (/runs/abc) are client-side routes: hand back the shell and
        # let the router sort it out.
        return FileResponse(index)


def _placeholder(dist: Path) -> str:
    """Console palette (console doc §7), inline, no CDN -- this page has to work
    on a machine with no network."""
    return f"""<!doctype html>
<meta charset="utf-8">
<title>strategy-lab · console not built</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         background:#14171C; color:#E9ECEF;
         font:15px/1.6 "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; }}
  main {{ max-width:44rem; padding:2rem 2.5rem; background:#1C2128;
         border:1px solid #2B323C; border-radius:10px; }}
  h1 {{ font-size:1.05rem; letter-spacing:.14em; text-transform:uppercase;
       color:#98A2AE; margin:0 0 1rem; }}
  code, pre {{ font-family:"IBM Plex Mono", ui-monospace, monospace; }}
  pre {{ background:#14171C; border:1px solid #2B323C; border-radius:6px;
        padding:.85rem 1rem; overflow-x:auto; }}
  a {{ color:#45B2A1; }}
  p.dim {{ color:#98A2AE; }}
</style>
<main>
  <h1>strategy-lab · console</h1>
  <p>The API is up. The frontend is not built yet.</p>
  <pre>cd console
npm install &amp;&amp; npm run build</pre>
  <p class="dim">Expected at <code>{dist}</code> (override with <code>LAB_CONSOLE_DIST</code>).</p>
  <p>Meanwhile: <a href="/api/health">/api/health</a> ·
     <a href="/api/runs">/api/runs</a> ·
     <a href="/api/openapi.json">/api/openapi.json</a> ·
     <a href="/api/docs">/api/docs</a></p>
</main>
"""


def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    open_browser: bool = False,
    log_level: str = "info",
) -> None:
    """Run uvicorn. ``lab ui`` calls this."""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"} and not auth_required():
        # Off loopback the only thing between the fleet controls and the network
        # is the token, so refuse rather than quietly publish the kill switch.
        raise ValueError(
            f"refusing to bind {host} without LAB_UI_TOKEN set; "
            "the control routes would be reachable from the network"
        )
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


#: Module-level instance for ``uvicorn lab.api.app:app``. Tests build their own
#: with ``create_app()`` so a monkeypatched environment is actually picked up.
app = create_app()
