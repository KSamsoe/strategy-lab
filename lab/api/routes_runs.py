"""Everything the console reads about *experiments*: the registry, the run
artifacts, the decision tape, sweeps, strategies and the agent's paper trail.

All of it is read-only. The registry and the two journals are the source of
truth; nothing here re-runs a backtest, imports the engine's broker, or asks a
runner for its state. Worst case this router can be slow -- it cannot be
dangerous.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from lab.api.models import (
    AgentCall,
    AgentCallList,
    Artifact,
    BarSeries,
    CompareTable,
    DecisionDetail,
    DecisionPage,
    DecisionRow,
    EquitySeries,
    Lineage,
    LineageStep,
    RunDetail,
    RunList,
    ResearchDetail,
    ResearchList,
    ResearchSummary,
    RunSummary,
    StrategyInfo,
    StrategyList,
    SweepDetail,
    SweepRunRow,
    TradeList,
    TradeRow,
    Truncation,
    WalkForwardWindow,
)
from lab.config import get_settings
from lab.registry.db import connect
from lab.registry.journal import DecisionJournal
from lab.registry.runs import RunRecord, RunRegistry, run_artifact_dir
from lab.timeutil import UTC, to_utc

router = APIRouter(prefix="/api", tags=["runs"])

#: Preference order when the caller does not name a metric. Out-of-sample first,
#: always: an in-sample default would make the console complicit in the same
#: cherry-picking the sweep runner exists to contain.
_DEFAULT_METRICS = ("oos_sharpe", "sharpe")

_UNSAFE_PATH_CHARS = frozenset('/\\:*?"<>|')


def _registry() -> RunRegistry:
    return RunRegistry()


def _artifact_dir(run_id: str) -> Path:
    """``runs/<run_id>/`` **without** creating it -- a GET must not mkdir. The
    id is checked even though it came from the registry, because a path segment
    interpolated from a URL is exactly where traversal gets in."""
    if not run_id or any(c in run_id for c in _UNSAFE_PATH_CHARS) or ".." in run_id:
        raise HTTPException(status_code=400, detail=f"unsafe run_id: {run_id!r}")
    return get_settings().paths.runs / run_id


def _run_or_404(run_id: str) -> RunRecord:
    record = _registry().get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return record


def _summary(record: RunRecord) -> RunSummary:
    return RunSummary.model_validate(record.to_dict())


def _iso_list(index: pd.DatetimeIndex) -> list[str]:
    return [to_utc(ts).isoformat() for ts in index]


def _floats(series: pd.Series) -> list[float | None]:
    """NaN becomes ``null``. Python's json encoder happily emits bare ``NaN``,
    which is not JSON and blows up ``JSON.parse`` in the browser."""
    return [None if pd.isna(v) else float(v) for v in series]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows with NaN replaced by real ``None`` -- ``where(..., None)`` on a float
    column silently keeps NaN, so the object cast has to come first."""
    if frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


# --- runs ----------------------------------------------------------------------


@router.get("/runs", response_model=RunList, summary="List registered runs")
def list_runs(
    strategy: str | None = None,
    kind: str | None = None,
    origin: str | None = None,
    sweep_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    order_by: str = "created_at desc",
) -> RunList:
    try:
        records = _registry().list(
            strategy=strategy,
            kind=kind,
            origin=origin,
            sweep_id=sweep_id,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )
    except ValueError as exc:  # bad order_by is caller error, not a server fault
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RunList(
        count=len(records),
        limit=limit,
        offset=offset,
        runs=[_summary(r) for r in records],
    )


@router.get("/runs/compare", response_model=CompareTable, summary="Compare runs")
def compare_runs(ids: str = Query(description="comma-separated run ids")) -> CompareTable:
    # Declared before /runs/{run_id}: FastAPI matches in declaration order, and
    # the parameterized route would otherwise swallow the literal "compare".
    wanted: list[str] = []
    for raw in ids.split(","):
        rid = raw.strip()
        if rid and rid not in wanted:
            wanted.append(rid)
    if not wanted:
        raise HTTPException(status_code=400, detail="ids must name at least one run")
    try:
        frame = _registry().compare(wanted)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    frame = _dedupe_columns(frame)
    return CompareTable(
        ids=wanted,
        columns=[str(c) for c in frame.columns],
        rows=_records(frame),
    )


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first of any repeated label.

    ``RunRegistry.compare`` lays down fixed provenance columns and then the
    union of every metric key, and the runner records ``attempt`` as both -- so
    the frame arrives with two ``attempt`` columns. ``to_dict(orient="records")``
    keeps one and warns, which would leave ``columns`` advertising a header the
    rows cannot fill and the compare grid drawing a permanently blank column.
    """
    if frame.columns.is_unique:
        return frame
    return frame.loc[:, ~frame.columns.duplicated()]


@router.get("/runs/{run_id}", response_model=RunDetail, summary="One run, with provenance")
def get_run(run_id: str) -> RunDetail:
    record = _run_or_404(run_id)
    directory = _artifact_dir(run_id)
    artifacts: list[Artifact] = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file():
                stat = path.stat()
                artifacts.append(
                    Artifact(
                        name=path.name,
                        bytes=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    )
                )
    metrics = record.metrics or {}
    payload = record.to_dict() | {
        "artifact_dir": str(directory),
        "artifacts": [a.model_dump() for a in artifacts],
        "decisions": DecisionJournal().count(run_id),
        "optimistic_fills": bool(metrics.get("optimistic_fills", False)),
        "contaminated": bool(metrics.get("contaminated", False)),
        "warnings": list(metrics.get("warnings") or []),
        # Runs written before the ledger fix carry no residual at all. That
        # absence is the tell, and it only mattered when something was still
        # open at the end -- a run that finished flat booked everything anyway.
        "stale_ledger": (
            "ledger_residual" not in metrics and bool(metrics.get("open_positions"))
        ),
        "ledger_residual": metrics.get("ledger_residual"),
    }
    return RunDetail.model_validate(payload)


@router.get(
    "/runs/{run_id}/equity",
    response_model=EquitySeries,
    summary="Equity, drawdown and the out-of-sample mask, as parallel arrays",
)
def get_equity(run_id: str) -> EquitySeries:
    record = _run_or_404(run_id)
    path = _artifact_dir(run_id) / "equity.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no equity.parquet for run {run_id}; artifacts may have been pruned",
        )
    frame = pd.read_parquet(path)
    if "event_time" in frame.columns:
        frame = frame.set_index("event_time")
    index = pd.DatetimeIndex(frame.index)

    from lab.backtest import metrics as M

    flags = M.is_oos_split(index, _normalized_windows(record.config))
    exposure = _floats(frame["exposure"]) if "exposure" in frame.columns else None
    equity = _floats(frame["equity"])
    return EquitySeries(
        run_id=run_id,
        n=len(frame),
        t=_iso_list(index),
        equity=equity,
        drawdown=_floats(frame["drawdown"]) if "drawdown" in frame.columns else [],
        is_oos=list(flags),
        exposure=exposure,
        benchmark=_benchmark_curve(record, index, equity),
    )


def _benchmark_curve(
    record: RunRecord, index: pd.DatetimeIndex, equity: Sequence[float | None]
) -> list[float | None] | None:
    """The run's benchmark, rebased onto the same starting equity.

    Rebasing rather than shipping raw prices is the whole point: a $600 index
    against a $100k book is not a comparison anyone can read off a chart. This
    is best-effort -- a pruned store or a benchmark that was never pulled just
    means no overlay, never a failed request.
    """
    ticker = (record.config or {}).get("benchmark")
    if not ticker or not len(index):
        return None
    start_equity = next((v for v in equity if v is not None), None)
    if not start_equity:
        return None
    try:
        from lab.store import parquet_io

        bars = parquet_io.read_bars(
            [str(ticker)],
            timeframe=str((record.config or {}).get("timeframe") or "1d"),
            start=to_utc(index[0]),
            end=to_utc(index[-1]),
            source=(record.config or {}).get("source"),
        )
    except Exception:
        return None
    if bars is None or len(bars) == 0:
        return None

    closes = (
        bars.set_index("event_time")["close"]
        .sort_index()
        .reindex(index, method="ffill")
        .astype("float64")
    )
    base = next((v for v in closes if pd.notna(v)), None)
    if not base:
        return None
    return _floats(closes / float(base) * float(start_equity))


def _normalized_windows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk-forward windows round-trip through JSON as naive-looking ISO strings;
    ``is_oos_split`` compares them against a tz-aware index, and naive-vs-aware
    comparison raises. Re-attach UTC here rather than shading the whole chart
    in-sample because of a timezone."""
    windows = config.get("windows") or []
    out: list[dict[str, Any]] = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        item = dict(w)
        for key in ("is_start", "is_end", "oos_start", "oos_end"):
            if item.get(key):
                item[key] = to_utc(pd.Timestamp(item[key]))
        out.append(item)
    return out


@router.get("/runs/{run_id}/trades", response_model=TradeList, summary="Closed round trips")
def get_trades(run_id: str) -> TradeList:
    _run_or_404(run_id)
    path = _artifact_dir(run_id) / "trades.csv"
    rows: list[TradeRow] = []
    if path.exists():
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()  # a run with no round trips writes an empty file
        rows = [TradeRow.model_validate(r) for r in _records(frame)]
    return TradeList(run_id=run_id, count=len(rows), trades=rows)


@router.get(
    "/runs/{run_id}/decisions",
    response_model=DecisionPage,
    summary="The decision tape: collapsed summary plus expandable detail",
)
def get_decisions(
    run_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    ticker: str | None = None,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> DecisionPage:
    _run_or_404(run_id)
    journal = DecisionJournal()
    # limit+1 probes for a next page without a second COUNT over the filter.
    rows = journal.list(
        run_id,
        start=to_utc(start) if start else None,
        end=to_utc(end) if end else None,
        ticker=ticker.upper() if ticker else None,
        limit=limit + 1,
        offset=offset,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return DecisionPage(
        run_id=run_id,
        total=journal.count(run_id),
        count=len(rows),
        limit=limit,
        offset=offset,
        has_more=has_more,
        rows=[_decision_row(r) for r in rows],
    )


def _decision_row(row: dict[str, Any]) -> DecisionRow:
    intents = row.get("intents") or []
    verdicts = row.get("verdicts") or []
    tickers: list[str] = []
    for item in (*intents, *verdicts):
        t = str(item.get("ticker") or "").upper()
        if t and t not in tickers:
            tickers.append(t)
    actions = {str(v.get("action") or "") for v in verdicts}
    return DecisionRow(
        id=row["id"],
        run_id=row["run_id"],
        strategy=row.get("strategy") or "",
        at=row["at"],
        summary=row.get("summary") or "",
        tickers=tickers,
        n_intents=len(intents),
        n_orders=len(row.get("order_ids") or []),
        blocked="blocked" in actions,
        clipped="clipped" in actions,
        detail=DecisionDetail.model_validate(
            {
                "inputs": row.get("inputs") or {},
                "intents": intents,
                "verdicts": verdicts,
                "order_ids": row.get("order_ids") or [],
                "portfolio": row.get("portfolio") or {},
                "logs": row.get("logs") or [],
                "agent": row.get("agent"),
                "duration_ms": row.get("duration_ms") or 0.0,
            }
        ),
    )


@router.get("/runs/{run_id}/bars", response_model=BarSeries, summary="Price bars for a run's chart")
def get_bars(run_id: str, ticker: str | None = None, tf: str | None = None) -> BarSeries:
    record = _run_or_404(run_id)
    universe = [str(t).upper() for t in (record.config.get("tickers") or [])]
    symbol = (ticker or (universe[0] if universe else "")).upper()
    if not symbol:
        raise HTTPException(status_code=400, detail=f"run {run_id} has no universe; pass ?ticker=")
    if universe and symbol not in universe:
        raise HTTPException(
            status_code=404, detail=f"{symbol} is not in run {run_id}'s universe"
        )
    timeframe = tf or str(record.config.get("timeframe") or "1d")

    from lab.store import parquet_io

    # No `as_of`: this series is the realized market path the chart draws trade
    # markers on, not an input the strategy was allowed to see. The look-ahead
    # barrier belongs on the decision tape's input tape, which is captured at
    # decision time and served verbatim.
    frame = parquet_io.read_bars(
        [symbol], timeframe=timeframe, start=record.start, end=record.end,
        source=(record.config or {}).get("source"),
    )
    if len(frame) == 0:
        return BarSeries(
            run_id=run_id, ticker=symbol, timeframe=timeframe, n=0,
            t=[], open=[], high=[], low=[], close=[], volume=[],
        )
    index = pd.DatetimeIndex(frame["event_time"])
    return BarSeries(
        run_id=run_id,
        ticker=symbol,
        timeframe=timeframe,
        n=len(frame),
        t=_iso_list(index),
        open=_floats(frame["open"]),
        high=_floats(frame["high"]),
        low=_floats(frame["low"]),
        close=_floats(frame["close"]),
        volume=_floats(frame["volume"]),
    )


# --- sweeps --------------------------------------------------------------------


def _unprefix(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix):]: v for k, v in row.items() if k.startswith(prefix)}


def _sweep_row(row: Mapping[str, Any]) -> SweepRunRow:
    return SweepRunRow(
        run_id=str(row.get("run_id") or ""),
        params=dict(row.get("params") or {}),
        status=str(row.get("status") or "ok"),
        score=row.get("score"),
        is_metrics=_unprefix(row, "is_"),
        oos_metrics=_unprefix(row, "oos_"),
        error=str(row.get("error") or ""),
    )


def _sweep_from_artifact(sweep_id: str, metric: str | None) -> SweepDetail | None:
    """Serve the sweep's own artifact when it exists.

    ``run_sweep`` writes the whole result -- grid, walk-forward table, the
    ranked-on flag and the truncation record -- to ``runs/<id>/sweep.json``.
    None of that survives a reconstruction from registry rows alone, and the
    honesty flags are exactly the parts worth not losing, so the artifact wins
    whenever it is present.
    """
    import json

    path = run_artifact_dir(sweep_id) / "sweep.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    rows = [_sweep_row(r) for r in blob.get("runs") or []]
    best_raw = blob.get("best")
    grid_size = int(blob.get("grid_size") or 0)
    ran = int(blob.get("combinations") or len(rows))
    return SweepDetail(
        sweep_id=sweep_id,
        strategy=str(blob.get("strategy") or ""),
        metric=metric or str(blob.get("metric") or ""),
        count=len(rows),
        attempts=int(blob.get("attempts") or 0),
        ranked_on=str(blob.get("ranked_on") or "in_sample"),
        grid={k: list(v) for k, v in (blob.get("grid") or {}).items()},
        truncated=Truncation(
            applied=bool(blob.get("truncated")), requested=grid_size, ran=ran
        ),
        best=_sweep_row(best_raw) if isinstance(best_raw, Mapping) else None,
        runs=rows,
        walk_forward=[
            _wf_window(w, i) for i, w in enumerate(blob.get("walk_forward") or [])
        ],
    )


#: Window boundary fields share the is_/oos_ prefixes with the metrics, so they
#: have to be held out of the un-prefixing or they arrive as bogus metrics named
#: "start" and "end".
_WF_BOUNDS = frozenset({"is_start", "is_end", "oos_start", "oos_end"})


def _wf_window(w: Mapping[str, Any], i: int) -> WalkForwardWindow:
    body = {k: v for k, v in w.items() if k not in _WF_BOUNDS}
    return WalkForwardWindow(
        index=int(w.get("index") or i),
        is_start=w.get("is_start"),
        is_end=w.get("is_end"),
        oos_start=w.get("oos_start"),
        oos_end=w.get("oos_end"),
        is_metrics=_unprefix(body, "is_"),
        oos_metrics=_unprefix(body, "oos_"),
    )


@router.get("/sweeps/{sweep_id}", response_model=SweepDetail, tags=["sweeps"])
def get_sweep(sweep_id: str, metric: str | None = None) -> SweepDetail:
    from_artifact = _sweep_from_artifact(sweep_id, metric)
    if from_artifact is not None:
        return from_artifact

    # Fallback: a sweep whose artifact is gone is still legible from the
    # registry, minus the grid and the walk-forward split.
    registry = _registry()
    runs = registry.list(sweep_id=sweep_id, limit=1000, order_by="created_at asc")
    if not runs:
        raise HTTPException(status_code=404, detail=f"no such sweep: {sweep_id}")
    chosen = metric or _pick_metric(runs)
    best = _best_by(runs, chosen)
    strategies = {r.strategy for r in runs}
    attempts = sum(registry.family_attempts(s) for s in strategies)
    rows = [
        SweepRunRow(
            run_id=r.run_id,
            params=dict(r.params),
            status=r.status,
            score=r.metrics.get(chosen),
            is_metrics=dict(r.metrics),
        )
        for r in runs
    ]
    best_row = next((row for row in rows if best and row.run_id == best.run_id), None)
    return SweepDetail(
        sweep_id=sweep_id,
        strategy=next(iter(strategies), ""),
        metric=chosen,
        count=len(runs),
        attempts=attempts,
        best=best_row,
        runs=rows,
    )


def _pick_metric(runs: Sequence[RunRecord]) -> str:
    for name in _DEFAULT_METRICS:
        if any(name in (r.metrics or {}) for r in runs):
            return name
    return _DEFAULT_METRICS[-1]


def _best_by(runs: Sequence[RunRecord], metric: str) -> RunRecord | None:
    scored = [
        (float(r.metrics[metric]), r)
        for r in runs
        if isinstance(r.metrics.get(metric), (int, float))
        and not isinstance(r.metrics.get(metric), bool)
        and not pd.isna(r.metrics.get(metric))
    ]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]


# --- strategies ----------------------------------------------------------------


@router.get("/strategies", response_model=StrategyList, tags=["strategies"])
def list_strategies() -> StrategyList:
    from lab.engine.loader import discover

    found = [StrategyInfo.model_validate(entry) for entry in discover()]
    return StrategyList(count=len(found), strategies=found)


# --- agent ---------------------------------------------------------------------


@router.get("/agent/lineage/{strategy}", response_model=Lineage, tags=["agent"])
def get_lineage(strategy: str, metric: str | None = None) -> Lineage:
    registry = _registry()
    every = registry.list(strategy=strategy, limit=1000, order_by="created_at asc")
    if not every:
        raise HTTPException(status_code=404, detail=f"no runs for strategy: {strategy}")
    chain = [r for r in every if r.origin != "human"]
    chosen = metric or _pick_metric(chain or every)
    best = _best_by(chain, chosen)
    return Lineage(
        strategy=strategy,
        metric=chosen,
        count=len(chain),
        # The whole family, not just the agent's slice: "you tried 400 things"
        # is the number this screen exists to make impossible to miss.
        attempts=registry.family_attempts(strategy),
        best_run_id=best.run_id if best is not None else None,
        steps=[
            LineageStep(
                n=i,
                run_id=r.run_id,
                created_at=r.created_at,
                parent_run_id=r.parent_run_id,
                attempt=r.attempt,
                params=r.params,
                metrics=r.metrics,
                notes=r.notes,
            )
            for i, r in enumerate(chain, start=1)
        ],
    )


def _research_dirs() -> list[Path]:
    root = get_settings().paths.runs
    if not root.exists():
        return []
    return sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "session.json").exists()),
        key=lambda d: d.name,
        reverse=True,
    )


def _load_session(directory: Path) -> dict[str, Any] | None:
    import json

    try:
        return json.loads((directory / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _research_summary(blob: Mapping[str, Any]) -> ResearchSummary:
    experiments = blob.get("experiments") or []
    scored = [
        e
        for e in experiments
        if e.get("ok") and not e.get("contaminated") and isinstance(e.get("score"), (int, float))
    ]
    best = max(scored, key=lambda e: e["score"]) if scored else None
    return ResearchSummary(
        session_id=str(blob.get("session_id") or ""),
        status=str(blob.get("status") or "running"),
        stopped_because=str(blob.get("stopped_because") or ""),
        brief=str(blob.get("brief") or ""),
        created_at=str(blob.get("created_at") or ""),
        updated_at=str(blob.get("updated_at") or ""),
        elapsed_minutes=round(float(blob.get("elapsed_s") or 0.0) / 60.0, 2),
        calls=int(blob.get("calls") or 0),
        spend_usd=round(float(blob.get("spend_usd") or 0.0), 6),
        billing=str(blob.get("billing") or "api"),
        experiments=len(experiments),
        best_run_id=(best or {}).get("run_id"),
        best_score=(best or {}).get("score"),
        satisfied_with=blob.get("satisfied_with"),
    )


@router.get("/agent/research", response_model=ResearchList, tags=["agent"])
def list_research() -> ResearchList:
    """Every research session on disk, newest first."""
    sessions: list[ResearchSummary] = []
    for directory in _research_dirs():
        blob = _load_session(directory)
        if blob:
            sessions.append(_research_summary(blob))
    return ResearchList(count=len(sessions), sessions=sessions)


@router.get("/agent/research/{session_id}", response_model=ResearchDetail, tags=["agent"])
def get_research(session_id: str) -> ResearchDetail:
    """One session, including the per-turn reasoning trail.

    Read from the session file rather than the registry: the file is written
    after every turn, so a session still running is readable *while* it runs.
    That is the whole point -- watching is the feature.
    """
    directory = _artifact_dir(session_id)
    blob = _load_session(directory) if directory.is_dir() else None
    if blob is None:
        raise HTTPException(status_code=404, detail=f"no research session: {session_id}")

    experiments = blob.get("experiments") or []
    rows: list[dict[str, Any]] = []
    for e in experiments:
        oos = e.get("oos") or {}
        ins = e.get("in_sample") or {}
        rows.append(
            {
                "n": e.get("n"),
                "run_id": e.get("run_id"),
                "strategy": e.get("strategy"),
                "params": e.get("params") or {},
                "score": e.get("score"),
                "oos_sharpe": oos.get("sharpe"),
                "oos_return": oos.get("total_return"),
                "is_sharpe": ins.get("sharpe"),
                "worst_period_sharpe": e.get("worst_period_sharpe"),
                "trades": oos.get("trades"),
                "periods": e.get("periods") or [],
                "rationale": e.get("rationale") or "",
                "ok": bool(e.get("ok", True)),
                "error": e.get("error") or "",
                "contaminated": bool(e.get("contaminated")),
            }
        )

    summary = _research_summary(blob)
    return ResearchDetail(
        **summary.model_dump(),
        verdict=str(blob.get("verdict") or ""),
        next_steps=str(blob.get("next_steps") or ""),
        notes=list(blob.get("notes") or []),
        market=blob.get("market"),
        runs=rows,
        turns=list(blob.get("turns") or []),
        workspace=str(blob.get("workspace") or ""),
        resumable=str(blob.get("stopped_because") or "") != "satisfied",
    )


@router.get("/agent/calls", response_model=AgentCallList, tags=["agent"])
def list_agent_calls(
    run_id: str | None = None,
    strategy: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> AgentCallList:
    where: list[str] = []
    args: list[Any] = []
    for column, value in (("run_id", run_id), ("strategy", strategy)):
        if value is not None:
            where.append(f"{column} = ?")
            args.append(value)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    args.extend([limit, offset])
    rows = connect().execute(
        f"SELECT * FROM agent_calls {clause} ORDER BY at ASC, rowid ASC LIMIT ? OFFSET ?", args
    ).fetchall()
    calls = [_agent_call(r) for r in rows]
    return AgentCallList(
        count=len(calls),
        total_cost_usd=round(sum(c.cost_usd for c in calls), 6),
        total_tokens=sum(c.input_tokens + c.output_tokens for c in calls),
        calls=calls,
    )


def _agent_call(row: sqlite3.Row) -> AgentCall:
    r = dict(row)
    return AgentCall(
        id=r["id"],
        run_id=r.get("run_id"),
        at=r["at"],
        strategy=r.get("strategy") or "",
        model=r.get("model") or "",
        tool=r.get("tool") or "",
        prompt=r.get("prompt") or "",
        response=r.get("response") or "",
        input_tokens=int(r.get("input_tokens") or 0),
        output_tokens=int(r.get("output_tokens") or 0),
        cost_usd=float(r.get("cost_usd") or 0.0),
        latency_ms=float(r.get("latency_ms") or 0.0),
        ok=bool(r.get("ok", 1)),
        error=r.get("error") or "",
    )
