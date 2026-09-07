"""The core verbs: run a backtest, read what it did, look it up, ship it.

These wrap machinery that already existed behind ``lab``'s CLI. What is new is
that every result arrives with its caveats attached rather than in a separate
document nobody opens -- see ``_common.warnings_for``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lab.mcp._common import (
    concentration_warning,
    evaluate,
    gate_for,
    jsonable,
    load_config,
    resolve_strategy,
    split_point,
)


def backtest(
    config: str,
    strategy: str | None = None,
    params: dict[str, Any] | None = None,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
    periods: int = 4,
    register: bool = True,
) -> dict[str, Any]:
    """Run one backtest and return its metrics, its benchmark and its caveats.

    Args:
        config: path to a backtest YAML -- universe, dates, fills, risk limits.
        strategy: strategy file or library name. Defaults to the config's.
        params: parameter overrides. Omitted keys keep the strategy's own.
        metric: which score to headline, e.g. oos_sharpe, oos_cagr, worst_sharpe.
        oos_split: in-sample:out-of-sample ratio, by tradeable bar count.
        periods: number of contiguous sub-periods to score separately.
        register: record the run so it can be reviewed and promoted later.

    Returns a dict whose `windows` covers full/in_sample/oos, `market` is the
    benchmark measured over the *same* bars, `score_one_sigma` is the sampling
    error on the headline score, `gate` is how much the risk limits rewrote, and
    `warnings` states in words anything that would embarrass the headline.
    """
    from dataclasses import replace

    from lab.backtest.runner import run_backtest

    cfg = load_config(config, params=params)
    cfg = replace(cfg, strategy=resolve_strategy(strategy, cfg))
    oos, stamps = split_point(cfg, oos_split)
    result = run_backtest(cfg, register=register, journal=False)
    return evaluate(
        result, cfg, oos_start=oos, stamps=stamps, periods=periods, metric=metric
    )


def review(run_id: str) -> dict[str, Any]:
    """A bounded diagnostic digest of a finished run.

    Per-ticker P&L attribution, trade distribution, the worst drawdown episodes
    with dates, monthly returns and risk-gate activity by rule. Aggregate metrics
    hide the two things that most often turn a result into a non-result: one
    ticker carrying the whole P&L, and a limit quietly clipping most of what the
    strategy tried to do. Both are in here.
    """
    from lab.agent.review import review_run

    digest = dict(review_run(run_id) or {})
    digest["warnings"] = concentration_warning(digest)
    gate = digest.get("gate") or {}
    clipped = gate.get("share_clipped_or_blocked")
    if isinstance(clipped, (int, float)) and clipped >= 0.5:
        digest["warnings"].append(
            f"gate clipped or blocked {clipped:.0%} of intents -- this run measures "
            "the risk limits more than the strategy"
        )
    return jsonable(digest)


def neighbourhood(
    run_id: str,
    config: str,
    budget: int = 16,
    metric: str = "oos_sharpe",
    oos_split: str = "4:1",
) -> dict[str, Any]:
    """Re-run a finished strategy at nearby parameters: plateau, or spike?

    Perturbs each numeric parameter about 10% either side (flags are flipped
    instead), and reports what the score does. A result that depends on the exact
    value you happened to land on is a property of this sample, not of the idea.

    `budget` is a ceiling, never exceeding two runs per parameter -- one for a
    flag. Below full coverage the answer says which parameters it could only nudge
    one way, and if the risk gate overrode most of this strategy's intents it says
    the check is measuring the limit rather than the strategy.
    """
    from dataclasses import replace

    from lab.analysis import neighbourhood_report, perturbations
    from lab.backtest.runner import run_backtest
    from lab.registry.runs import RunRegistry

    record = RunRegistry().get(run_id)
    if record is None:
        raise ValueError(f"no such run: {run_id}")

    cfg = load_config(config)
    oos, stamps = split_point(cfg, oos_split)

    # `record.strategy` is the strategy's NAME ("momo"), not its file, so it will
    # not resolve. Prefer the path the run itself recorded, and fall back to the
    # source archived inside the run -- which is the only thing that still matches
    # for a strategy a research session wrote into a workspace and then edited.
    source = _strategy_path_for(run_id, record)
    base = _evaluate_existing(run_id, record, cfg, oos, stamps, metric)
    params = dict(record.params or {})

    tried: list[dict[str, Any]] = []
    for key, value in perturbations(params, max(0, int(budget))):
        case = replace(cfg, strategy=source, params=params | {key: value})
        entry: dict[str, Any] = {"param": key, "value": value, "ok": False,
                                 "score": None, "full_return": None, "error": None}
        try:
            # register=False: these are diagnostics about a run that already
            # exists, and they must not appear in the registry as results.
            result = run_backtest(case, register=False, journal=False)
        except Exception as exc:  # noqa: BLE001 - one bad neighbour is a data point
            entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            ev = evaluate(result, case, oos_start=oos, stamps=stamps, metric=metric)
            entry |= {
                "ok": True,
                "score": ev.get("score"),
                "full_return": ev["windows"]["full"].get("total_return"),
            }
        tried.append(entry)

    out = neighbourhood_report(
        params,
        base.get("score"),
        base["windows"]["full"].get("total_return"),
        tried,
        gate=gate_for(run_id),
    )
    out["strategy"] = source.name
    return jsonable(out)


def _strategy_path_for(run_id: str, record: Any) -> Path:
    """The file to perturb: the run's own path, else its archived source."""
    from lab.backtest.runner import artifact_dir

    recorded = (record.config or {}).get("strategy")
    if recorded and Path(recorded).exists():
        return Path(recorded)

    archived = artifact_dir(run_id) / "strategy.py"
    if archived.exists():
        return archived

    from lab.config import get_settings

    name = str(record.strategy or "")
    candidate = Path(get_settings().paths.strategies) / (
        name if name.endswith(".py") else f"{name}.py"
    )
    if candidate.exists():
        return candidate
    raise ValueError(
        f"cannot locate the strategy for {run_id}: its recorded path is gone and it "
        f"has no archived source (runs from before source archiving carry only a hash)"
    )


def _evaluate_existing(run_id, record, cfg, oos, stamps, metric):
    """Re-derive window metrics for a run from its stored equity curve."""
    import pandas as pd

    from lab.backtest.runner import artifact_dir

    directory = artifact_dir(run_id)
    frame = pd.read_parquet(directory / "equity.parquet")
    stamp_col = "event_time" if "event_time" in frame else frame.columns[0]
    equity = frame.set_index(stamp_col)["equity"]
    equity.index = pd.to_datetime(equity.index, utc=True)

    class _Stub:
        pass

    stub = _Stub()
    stub.equity = equity
    stub.trades = []
    stub.fills = []
    stub.exposure = (
        frame.set_index(stamp_col)["exposure"] if "exposure" in frame else pd.Series(dtype="float64")
    )
    if len(stub.exposure):
        stub.exposure.index = pd.to_datetime(stub.exposure.index, utc=True)
    stub.run_id = run_id
    stub.metrics = dict(record.metrics or {})
    return evaluate(stub, cfg, oos_start=oos, stamps=stamps, metric=metric)


def runs_list(
    strategy: str | None = None,
    kind: str | None = None,
    origin: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Recent runs from the registry, newest first."""
    from lab.registry.runs import RunRegistry

    rows = RunRegistry().list(strategy=strategy, kind=kind, origin=origin, limit=int(limit))
    return jsonable({
        "count": len(rows),
        "runs": [
            {
                "run_id": r.run_id, "strategy": r.strategy, "kind": r.kind,
                "created_at": r.created_at, "origin": r.origin,
                "params": r.params,
                "metrics": {
                    k: (r.metrics or {}).get(k)
                    for k in ("total_return", "sharpe", "max_drawdown", "exposure", "trades")
                },
            }
            for r in rows
        ],
    })


def runs_compare(run_ids: list[str]) -> dict[str, Any]:
    """Metrics for several runs side by side, with identical params dropped."""
    from lab.registry.runs import RunRegistry

    reg = RunRegistry()
    records = [reg.get(r) for r in run_ids]
    missing = [r for r, rec in zip(run_ids, records) if rec is None]
    if missing:
        raise ValueError(f"no such run(s): {', '.join(missing)}")

    keys = ("total_return", "cagr", "sharpe", "sortino", "max_drawdown",
            "exposure", "turnover", "trades")
    all_params = [dict(r.params or {}) for r in records]
    shared = {
        k for k in set().union(*[set(p) for p in all_params]) if all_params
        and all(k in p for p in all_params) and len({repr(p[k]) for p in all_params}) == 1
    }
    return jsonable({
        "runs": [
            {
                "run_id": r.run_id, "strategy": r.strategy,
                "params": {k: v for k, v in (r.params or {}).items() if k not in shared},
                "metrics": {k: (r.metrics or {}).get(k) for k in keys},
            }
            for r in records
        ],
        "shared_params": {k: all_params[0][k] for k in sorted(shared)} if all_params else {},
    })


def strategies_list() -> dict[str, Any]:
    """Every loadable strategy in the library, with its declared parameters."""
    from lab.config import get_settings
    from lab.engine.loader import load_strategy

    out = []
    root = Path(get_settings().paths.strategies)
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        entry: dict[str, Any] = {"file": path.name, "path": str(path)}
        try:
            loaded = load_strategy(path)
            entry |= {"name": loaded.name, "params": loaded.params,
                      "doc": (loaded.doc or "").strip().splitlines()[:1]}
        except Exception as exc:  # noqa: BLE001 - a broken file is worth listing
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out.append(entry)
    return jsonable({"count": len(out), "strategies": out})


def validate_strategy(path: str) -> dict[str, Any]:
    """Check a strategy file imports and expose what the engine sees in it.

    Cheaper than discovering the same thing through a failed backtest, which
    costs a run and tells you less.
    """
    from lab.engine.loader import load_strategy

    try:
        loaded = load_strategy(Path(path))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return jsonable({
        "ok": True, "name": loaded.name, "params": loaded.params,
        "source_hash": loaded.source_hash,
        "hooks": [h for h in ("on_start", "on_bar", "on_fill", "on_stop")
                  if loaded.hook(h) is not None],
    })


def data_coverage() -> dict[str, Any]:
    """What is in the point-in-time store, by source, timeframe and ticker.

    Worth reading before a backtest: a config naming a source with no bars fails
    in a way that looks like a strategy that does not trade.
    """
    from lab.store import parquet_io

    frame = parquet_io.coverage()
    if not len(frame):
        return {"groups": [], "note": "the store is empty -- run data_pull first"}

    groups = []
    for (source, timeframe), part in frame.groupby(["source", "timeframe"], sort=True):
        groups.append({
            "source": str(source),
            "timeframe": str(timeframe),
            "tickers": int(part["ticker"].nunique()),
            "rows": int(part["rows"].sum()),
            "first": part["first"].min(),
            "last": part["last"].max(),
            # Named so a caller can spot the trap that cost this lab 52 runs: the
            # same tickers present under two sources, silently unioned into one
            # price series built from two unrelated universes.
            "sample_tickers": sorted(part["ticker"].unique().tolist())[:8],
        })
    overlap = {
        t: sorted(s) for t, s in
        frame.groupby("ticker")["source"].agg(lambda s: set(s)).items() if len(s) > 1
    }
    out: dict[str, Any] = {"groups": groups}
    if overlap:
        out["multi_source_tickers"] = overlap
        out["warnings"] = [
            f"{len(overlap)} ticker(s) have bars from more than one source; a config "
            "must name `source:` explicitly or the backtest will refuse to guess"
        ]
    return jsonable(out)


def promote(
    run_id: str,
    name: str | None = None,
    write_config: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy the exact code that produced a run into the strategy library.

    Keyed on the run, never on a file path: a research session rewrites its
    workspace files in place, so the file sitting where a run says its strategy
    lived is often a later, different variant. This copies the source archived
    inside the run itself, verified against the hash taken when it executed, and
    writes a config carrying the parameters that run actually resolved.
    """
    from lab.registry.promote import PromotionError, promote_run

    try:
        done = promote_run(
            run_id, name=name, write_config=write_config, force=force, dry_run=dry_run
        )
    except PromotionError as exc:
        return {"ok": False, "refused": str(exc)}
    return jsonable({"ok": True, **done.to_dict()})


def market_reference(config: str, oos_split: str = "4:1") -> dict[str, Any]:
    """What simply holding the benchmark would have done, over the same bars.

    Costless, and measured from the first *tradeable* bar rather than the first
    bar of data -- warmup bars the strategy sat out are excluded from both sides,
    which is the difference between a fair comparison and a 107-point head start.
    """
    from lab.analysis import market_reference

    cfg = load_config(config)
    oos, stamps = split_point(cfg, oos_split)
    return jsonable(market_reference(cfg, oos, stamps[0] if stamps else None))


# --- authoring without a filesystem -------------------------------------------


def _strategy_target(name: str) -> Path:
    from lab.config import get_settings

    stem = str(name).strip().removesuffix(".py")
    if not stem or stem.startswith("_") or "/" in stem or "\\" in stem or ".." in stem:
        raise ValueError(f"{name!r} is not a usable strategy name")
    return Path(get_settings().paths.strategies) / f"{stem}.py"


def _promoted_from(target: Path) -> str | None:
    """The run a library strategy was promoted from, if its config says so."""
    from lab.config import get_settings

    cfg = Path(get_settings().paths.cfg) / f"{target.stem}.yaml"
    if not cfg.exists():
        return None
    head = cfg.read_text(encoding="utf-8").splitlines()[:1]
    if head and head[0].startswith("# Promoted from run "):
        return head[0].removeprefix("# Promoted from run ").rstrip(".").strip()
    return None


def strategy_read(name: str) -> dict[str, Any]:
    """The source of a library strategy, for a client that cannot open files."""
    target = _strategy_target(name)
    if not target.exists():
        raise ValueError(f"no strategy named {target.stem!r} in the library")
    return jsonable({
        "name": target.stem, "path": str(target),
        "promoted_from": _promoted_from(target),
        "source": target.read_text(encoding="utf-8"),
    })


def strategy_write(name: str, source: str, overwrite: bool = False) -> dict[str, Any]:
    """Save a strategy into the library, validated first.

    For clients that cannot write files. The source is loaded by the engine
    before anything is written, so a file that would not import never lands.
    Refuses to overwrite an existing strategy unless `overwrite` is set, and
    when the existing one was *promoted* from a run it says which run, because
    overwriting it breaks the link between the library and the result it was
    promoted for -- promote the new run instead.
    """
    import tempfile

    from lab.engine.loader import load_strategy

    target = _strategy_target(name)
    if target.exists() and not overwrite:
        promoted = _promoted_from(target)
        return {
            "ok": False,
            "refused": f"{target.name} already exists"
            + (f" and was promoted from run {promoted}; use promote on the new run "
               "or pass overwrite=true knowingly" if promoted else
               "; pass overwrite=true to replace it, or choose another name"),
        }

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / target.name
        probe.write_text(source, encoding="utf-8")
        try:
            loaded = load_strategy(probe)
        except Exception as exc:  # noqa: BLE001 - the whole point is to report this
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(source, encoding="utf-8")
    return jsonable({
        "ok": True, "path": str(target), "name": loaded.name, "params": loaded.params,
        "source_hash": loaded.source_hash, "overwrote": existed,
        "hooks": [h for h in ("on_start", "on_bar", "on_fill", "on_stop") if loaded.hook(h) is not None],
    })


def strategy_api() -> dict[str, Any]:
    """The contract a strategy file has to meet, with a skeleton and the indicators.

    Also served as the `lab://strategy-api` resource. Read this before
    `strategy_write`: the engine finds a class (or `STRATEGY` / `build()`) in the
    module, calls `on_bar(ctx)` once per bar, and everything the strategy may
    see or do goes through `ctx`. Every read is filtered to `knowledge_time <=
    ctx.now` -- that is the look-ahead barrier, and it is structural.
    """
    import inspect

    from lab.engine import protocols
    from lab.indicators import computed

    def _src(obj: Any) -> str:
        try:
            return inspect.getsource(obj)
        except OSError:
            return ""

    indicators = {}
    for name in computed.available():
        try:
            indicators[name] = computed.default_params(name)
        except Exception:  # noqa: BLE001
            indicators[name] = {}

    return jsonable({
        "conventions": [
            "module-level NAME (str) and PARAMS (dict of plain values) -- every tunable "
            "number lives in PARAMS so sweeps and ablations can vary it without editing code",
            "a class whose __init__(self, params) merges PARAMS | params into self.params",
            "on_bar(self, ctx) is called once per bar after warmup; on_start/on_fill/on_stop are optional",
            "express positions with ctx.order_target_pct(ticker, pct) -- a target weight, "
            "not an order; the risk gate may clip it, and the backtest reports how often it did",
            "size to fit inside the config's limits (max_position_pct, max_positions, "
            "max_sector_pct) -- what the gate rewrites is what gets measured",
            "never reach past ctx for data; ctx.history/bars/indicator/signals are the "
            "look-ahead barrier",
            "return None from on_bar when a name lacks history rather than guessing",
        ],
        "context_protocol": _src(protocols.Context),
        "strategy_protocol": _src(protocols.Strategy),
        "indicators": indicators,
        "skeleton": SKELETON,
    })


SKELETON = '''"""One-line description of the idea.

Say what it holds, when it exits, and which PARAMS matter most.
"""

from __future__ import annotations

from lab.engine.protocols import Context

NAME = "my_strategy"

PARAMS = {
    "lookback": 126,   # bars of trailing return for the ranking
    "top_n": 5,        # names to hold
    "gross": 0.95,     # total target weight
}


class MyStrategy:
    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(PARAMS) | dict(params or {})

    def on_bar(self, ctx: Context) -> None:
        p = self.params
        lookback, top_n = int(p["lookback"]), int(p["top_n"])

        scores: dict[str, float] = {}
        for t in ctx.universe:
            closes = ctx.history(t, "close", lookback + 1)
            if len(closes) < lookback + 1 or float(closes.iloc[0]) <= 0:
                continue
            scores[t] = float(closes.iloc[-1]) / float(closes.iloc[0]) - 1.0
        if not scores:
            return

        held = set(ctx.portfolio.positions)
        want = [t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]
        weight = float(p["gross"]) / max(1, len(want))
        for t in held - set(want):
            ctx.close(t, reason="rank decay")
        for t in want:
            ctx.order_target_pct(t, weight, reason=f"rank score {scores[t]:.3f}")
'''
