"""Pattern A: the agent as researcher. The primary pattern, and the safe one.

The loop proposes parameters (or a whole replacement strategy file), runs a
*real* backtest through ``lab.backtest.runner``, reads the metrics, and revises.
What ships out of it is an ordinary deterministic strategy artifact -- a `.py`
file and a params dict, reviewable by a human, with no LLM anywhere in the
runtime path. That is the entire point: most of the value of "agentic trading"
with none of the runtime risk.

Two containments make it more than a random search with a large bill:

**Fitness is out-of-sample, always.** Each iteration runs once over the full
range and the equity curve is split at the walk-forward boundary; the score the
agent is ranked on, and the score it is shown first, comes from the OOS segment
only. An agent is a tireless overfitter -- give it in-sample Sharpe and it will
find a beautiful curve that predicts nothing. In-sample numbers are shown next
to the OOS ones purely so the *gap* is visible, and the prompt says so.

**Every iteration is recorded.** An ``Iteration`` carries the run id, the params,
a unified diff against the previous artifact, both metric sets, and the agent's
own rationale. The lineage that falls out is what lets the console draw the
trajectory and answer the only question that matters about a research loop: is
this converging, or just churning? Runs are registered with
``origin="agent-loop"`` and ``parent_run_id`` chained, so the same lineage is
reconstructible from the registry alone.

Budget is enforced before each call, not after, and the loop stops cleanly --
with everything it has learned persisted -- rather than raising.
"""

from __future__ import annotations

import difflib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from lab.agent.schemas import (
    ModelClient,
    call_tool,
    get_client,
    link_call,
    sanitize_text,
    strategy_proposal_tool,
    validate_proposal,
)
from lab.timeutil import to_utc, utcnow

log = logging.getLogger(__name__)

from lab.analysis import (  # noqa: E402  -- re-exported below for callers
    LINEAGE_METRICS,
    market_reference,
    period_breakdown,
    split_point,
    tradeable_timestamps,
    window_metrics,
)

# --- moved to lab.analysis ------------------------------------------------------
#
# These are pure arithmetic over an equity curve and were only ever here because
# this is where they were first needed. They now live in ``lab.analysis`` so the
# MCP toolset can use them without importing private names from the orchestration
# layer it replaces. The aliases keep this module's own call sites and its tests
# working unchanged.
_tradeable_timestamps = tradeable_timestamps
_split_point = split_point
_market_reference = market_reference
_window_metrics = window_metrics
_period_breakdown = period_breakdown


#: Metrics carried into the lineage record. The full set lives on the run in the
#: registry; this is what a human (or the next prompt) actually reads.

SYSTEM_PROMPT = """You are a quantitative researcher iterating on one trading strategy inside an
automated lab. Each turn you propose the next variation to backtest; the lab runs it and
returns the metrics.

You are ranked on the OUT-OF-SAMPLE score only. In-sample numbers are shown beside it so
you can see the gap between them, which is the honest measure of how much of a result is
fitting. A variation whose in-sample Sharpe rises while its out-of-sample Sharpe falls is
a worse strategy than the one before it, not a better one, and proposing more of the same
direction wastes the budget.

Work like a researcher, not a search:
- Change one thing at a time, and say in the rationale what hypothesis the change tests.
- Read the lineage before proposing. If the last three variations moved the OOS score by
  noise, the parameter you are turning is not the one that matters — change direction or
  stop.
- Prefer robustness to peaks. A parameter value that only works in a narrow neighbourhood
  is a fitting artifact; say so when you see one.
- Read `by_period` before you read the pooled score. A strategy that earns everything in
  one sub-period and loses in the rest is fitted to a market regime, not to an edge, and
  its pooled Sharpe is a rounding artifact of which years you happened to test. Prefer a
  variation that is positive in most periods over one with a higher average and one
  catastrophic period. You cannot change the periods or the date range -- only the
  strategy.
- Every run is counted, permanently, whatever its result. You cannot quietly try four
  hundred variations and present one. Set stop=true when further variation is not
  justified — stopping early is a good outcome, not a failure.

Reply only by calling the propose_strategy tool.
"""


@dataclass
class AuthorLoopConfig:
    """What to iterate on, how far, and how much it may cost.

    The first six fields are the module contract; the rest have defaults so the
    loop can be driven from a YAML config or a CLI without changing that shape.
    """

    seed_strategy: Path
    grid_or_freeform: str = "grid"
    iterations: int = 10
    metric: str = "oos_sharpe"
    budget_usd: float = 5.0
    model: str | None = None
    # --- how the backtests are set up ------------------------------------
    config: Path | None = None
    oos_split: str = "4:1"
    #: Contiguous sub-periods the run is additionally broken down over, so a
    #: strategy that only works in one market regime is visible as such. Pure
    #: slicing of an equity curve that already exists -- it costs no extra
    #: backtest, so it is on by default. 0 disables it.
    robustness_periods: int = 4
    max_tokens: int = 4_096
    objective: str = ""
    workspace: Path | None = None
    register: bool = True
    journal: bool = False

    def __post_init__(self) -> None:
        self.seed_strategy = Path(self.seed_strategy)
        if self.config is not None:
            self.config = Path(self.config)
        if self.grid_or_freeform not in {"grid", "freeform"}:
            raise ValueError(
                f"grid_or_freeform must be 'grid' or 'freeform', got {self.grid_or_freeform!r}"
            )
        if int(self.iterations) < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if float(self.budget_usd) <= 0:
            raise ValueError(f"budget_usd must be > 0, got {self.budget_usd}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AuthorLoopConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown author-loop keys: {', '.join(sorted(unknown))}")
        return cls(**raw)


@dataclass
class Iteration:
    """One proposal, tested. The unit the console draws lineage from."""

    n: int
    run_id: str
    params: dict[str, Any]
    diff: str
    metrics: dict[str, Any]
    rationale: str
    score: float = float("nan")
    cost_usd: float = 0.0
    call_id: str = ""
    source_path: str = ""
    parent_run_id: str | None = None
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = None if self.score != self.score else self.score
        return d


def run_author_loop(
    cfg: AuthorLoopConfig,
    *,
    client: ModelClient | None = None,
    base: Any = None,
    data: Any = None,
) -> dict[str, Any]:
    """Drive ``cfg.iterations`` propose-backtest-revise cycles and return the lineage.

    ``client`` accepts any object with ``messages.create`` -- the real Anthropic
    client, or the offline stub the tests inject. ``base`` and ``data`` let a
    caller supply an already-built ``BacktestConfig`` and a preloaded
    ``DataView`` instead of reading YAML and the parquet store.
    """
    from lab.backtest.runner import BacktestConfig
    from lab.engine.loader import load_strategy, resolve_path

    seed_path = resolve_path(cfg.seed_strategy)

    from lab.agent.research import _is_llm_strategy

    if _is_llm_strategy(seed_path):
        # Optimising a Pattern-B strategy against historical data optimises the
        # model's recall of that period. Every score the loop would rank on is
        # contaminated, so there is no honest version of this loop to run.
        raise ValueError(
            f"{seed_path.name} puts a model inside the trading loop, so every backtest "
            f"of it is contaminated by that model's training data. There is nothing for "
            f"an author loop to optimise against. Pattern-B strategies are evaluated "
            f"forward, on paper, by a human."
        )

    seed_source = seed_path.read_text(encoding="utf-8")
    seed_loaded = load_strategy(seed_path)
    strategy_name = seed_loaded.name

    if base is not None:
        base_cfg: Any = base
    elif cfg.config is not None:
        base_cfg = BacktestConfig.from_yaml(cfg.config)
    else:
        raise ValueError(
            "author loop needs a backtest config: set AuthorLoopConfig.config or "
            "pass base=BacktestConfig(...)"
        )
    base_cfg = replace(base_cfg, strategy=str(seed_path), origin="agent-loop")

    model = cfg.model or _default_model()
    loop_id = _loop_id(strategy_name)
    workspace = Path(cfg.workspace) if cfg.workspace else _loop_dir(loop_id)
    workspace.mkdir(parents=True, exist_ok=True)

    if client is None:
        # A clear message, not an ImportError, and not a half-run loop.
        client = get_client()

    timestamps = _tradeable_timestamps(base_cfg, data)
    if len(timestamps) < 4:
        raise ValueError("author loop needs at least 4 bars of data to split IS/OOS")
    is_end, oos_start = _split_point(timestamps, cfg.oos_split)
    windows = [
        {
            "index": 0,
            "is_start": timestamps[0].isoformat(),
            "is_end": is_end.isoformat(),
            "oos_start": oos_start.isoformat(),
            "oos_end": timestamps[-1].isoformat(),
        }
    ]

    lineage: list[Iteration] = []
    spend = 0.0
    billing = "api"
    stopped = "completed"
    consecutive_failures = 0
    prev_params = dict(seed_loaded.params)
    prev_source = seed_source
    #: The file the *next* iteration builds on. In freeform mode the chain
    #: advances to each newly written variant; in grid mode it stays the seed.
    prev_path = seed_path
    prev_run_id: str | None = None

    baseline = _evaluate(
        n=0,
        base_cfg=base_cfg,
        strategy_path=seed_path,
        params=prev_params,
        diff="",
        rationale="seed baseline — the score every proposal must beat",
        metric=cfg.metric,
        oos_start=oos_start,
        windows=windows,
        data=data,
        parent_run_id=None,
        register=cfg.register,
        journal=cfg.journal,
        periods=int(cfg.robustness_periods),
    )
    if baseline.ok:
        prev_run_id = baseline.run_id

    market = _market_reference(base_cfg, oos_start, timestamps[0] if timestamps else None)

    for n in range(1, int(cfg.iterations) + 1):
        if spend >= float(cfg.budget_usd):
            stopped = "budget_exhausted"
            log.warning(
                "author loop stopping: spent $%.4f of $%.4f budget", spend, cfg.budget_usd
            )
            break

        system, user = _render_prompt(
            cfg, strategy_name, prev_source, prev_params, baseline, lineage, market
        )
        call = call_tool(
            client,
            model=model,
            system=system,
            user=user,
            tools=[strategy_proposal_tool(sorted(prev_params))],
            force_tool="propose_strategy",
            max_tokens=int(cfg.max_tokens),
            run_id=loop_id,
            strategy=strategy_name,
        )
        spend += call.cost_usd
        # What the spend figure *means* follows the backend: real money on an
        # API key, a notional API-equivalent against a Claude subscription.
        # A budget reported without that label misleads either way.
        billing = call.billing or billing

        if not call.ok:
            consecutive_failures += 1
            lineage.append(
                Iteration(
                    n=n, run_id="", params=dict(prev_params), diff="", metrics={},
                    rationale="", cost_usd=call.cost_usd, call_id=call.call_id,
                    ok=False, error=call.error,
                )
            )
            if consecutive_failures >= 2:
                stopped = "model_unavailable"
                break
            continue
        consecutive_failures = 0

        try:
            params, rationale, source, stop = validate_proposal(
                call.input, allowed_params=sorted(prev_params)
            )
        except ValueError as exc:
            # Off-schema proposals are dropped like any other malformed model
            # output; the loop keeps its budget and asks again next turn.
            lineage.append(
                Iteration(
                    n=n, run_id="", params={}, diff="", metrics={}, rationale="",
                    cost_usd=call.cost_usd, call_id=call.call_id, ok=False,
                    error=f"invalid proposal: {exc}",
                )
            )
            continue

        merged = dict(prev_params) | params
        # Carry the chain forward. Defaulting to the seed here would silently
        # revert the *code* whenever a freeform iteration proposed parameters
        # only, while the recorded diff still claimed the previous iteration's
        # source -- the run and its audit trail would disagree about what ran.
        strategy_path = prev_path
        new_source = prev_source
        if source and cfg.grid_or_freeform == "freeform":
            strategy_path = workspace / f"{seed_path.stem}_i{n}.py"
            strategy_path.write_text(source, encoding="utf-8")
            new_source = source

        diff = _diff(prev_source, new_source, prev_params, merged, n)
        iteration = _evaluate(
            n=n,
            base_cfg=base_cfg,
            strategy_path=strategy_path,
            params=merged,
            diff=diff,
            rationale=rationale,
            metric=cfg.metric,
            oos_start=oos_start,
            windows=windows,
            data=data,
            parent_run_id=prev_run_id,
            register=cfg.register,
            journal=cfg.journal,
            periods=int(cfg.robustness_periods),
        )
        iteration.cost_usd = call.cost_usd
        iteration.call_id = call.call_id
        lineage.append(iteration)

        if iteration.ok and iteration.run_id:
            # Re-home the audit row onto the run it produced, so
            # /api/agent/calls?run_id=<run> answers "why was this tried".
            link_call(call.call_id, iteration.run_id)
            prev_run_id = iteration.run_id
            prev_params = merged
            prev_source = new_source
            prev_path = strategy_path

        if stop:
            stopped = "agent_stopped"
            break

    scored = [it for it in [baseline, *lineage] if it.ok and it.score == it.score]
    best = max(scored, key=lambda it: it.score) if scored else None

    out = {
        "loop_id": loop_id,
        "strategy": strategy_name,
        "seed_strategy": str(seed_path),
        "mode": cfg.grid_or_freeform,
        "model": model,
        "metric": cfg.metric,
        "windows": windows,
        "baseline": baseline.to_dict(),
        "market": market,
        "lineage": [it.to_dict() for it in lineage],
        "trajectory": [None if it.score != it.score else it.score for it in lineage],
        "best": best.to_dict() if best is not None else None,
        "iterations_run": len(lineage),
        "spend_usd": round(spend, 6),
        "billing": billing,
        "budget_usd": float(cfg.budget_usd),
        "stopped": stopped,
        "workspace": str(workspace),
        "finished_at": utcnow().isoformat(),
    }
    (workspace / "lineage.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    return out


# --- evaluation ---------------------------------------------------------------


def _evaluate(
    *,
    n: int,
    base_cfg: Any,
    strategy_path: Path,
    params: Mapping[str, Any],
    diff: str,
    rationale: str,
    metric: str,
    oos_start: datetime,
    windows: Sequence[Mapping[str, Any]],
    data: Any,
    parent_run_id: str | None,
    register: bool,
    journal: bool,
    periods: int = 0,
) -> Iteration:
    """Run one variation and score it on the out-of-sample segment.

    One backtest, not two: the run is continuous over the whole range and the
    equity curve is cut at ``oos_start``. Splitting the *curve* rather than the
    *run* keeps positions carried across the boundary honest -- an IS entry that
    is still open pays for itself in the OOS segment, exactly as it would live.
    """
    from lab.backtest.runner import run_backtest

    cfg = replace(
        base_cfg,
        strategy=str(strategy_path),
        params=dict(params),
        origin="agent-loop",
        parent_run_id=parent_run_id,
        notes=sanitize_text(rationale, limit=400),
        windows=[dict(w) for w in windows],
    )
    try:
        result = run_backtest(cfg, data=data, register=register, journal=journal)
    except Exception as exc:  # a variation that crashes is a data point, not a stop
        log.warning("author loop iteration %s failed: %s", n, exc)
        return Iteration(
            n=n, run_id="", params=dict(params), diff=diff, metrics={},
            rationale=rationale, ok=False, error=f"{type(exc).__name__}: {exc}",
            parent_run_id=parent_run_id, source_path=str(strategy_path),
        )

    timeframe = getattr(cfg, "timeframe", "1d")
    is_metrics = _window_metrics(result, timeframe, hi=oos_start, inclusive_hi=False)
    oos_metrics = _window_metrics(result, timeframe, lo=oos_start)
    metrics = {
        "is": is_metrics,
        "oos": oos_metrics,
        "full": {k: result.metrics.get(k) for k in LINEAGE_METRICS},
        "periods": _period_breakdown(result, timeframe, periods),
        "attempt": result.metrics.get("attempt", 0),
    }
    score = _score(metrics, metric)
    metrics["score"] = None if score != score else score
    return Iteration(
        n=n,
        run_id=result.run_id,
        params=dict(params),
        diff=diff,
        metrics=metrics,
        rationale=rationale,
        score=score,
        parent_run_id=parent_run_id,
        source_path=str(strategy_path),
    )






def _worst(periods: Sequence[Mapping[str, Any]], name: str) -> float:
    values = [p.get(name) for p in periods]
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    return min(numeric) if numeric else float("nan")


def _score(metrics: Mapping[str, Any], metric: str) -> float:
    """Read the fitness value. ``oos_`` prefixed names read the OOS block; a bare
    name reads it too. Reaching in-sample numbers requires saying ``is_`` out
    loud, which is the point -- the default cannot silently be the wrong one."""
    name = str(metric)
    # `worst_x` scores the weakest sub-period rather than the pooled number. A
    # strategy that makes all its money in one regime and gives it back in the
    # others reads as excellent pooled and correctly reads as bad here.
    if name.startswith("worst_"):
        return _worst(metrics.get("periods") or [], name[6:])
    block = "oos"
    if name.startswith("oos_"):
        name = name[4:]
    elif name.startswith("is_"):
        block = "is"
        name = name[3:]
    value = (metrics.get(block) or {}).get(name)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if out == out else float("nan")


# --- windows ------------------------------------------------------------------






# --- prompt -------------------------------------------------------------------


def _render_prompt(
    cfg: AuthorLoopConfig,
    strategy_name: str,
    source: str,
    params: Mapping[str, Any],
    baseline: Iteration,
    lineage: Sequence[Iteration],
    market: dict[str, Any] | None = None,
) -> tuple[str, str]:
    rows = [_lineage_row(baseline)] + [_lineage_row(it) for it in lineage]
    mode = (
        "You may propose parameter values only; the strategy file is fixed."
        if cfg.grid_or_freeform == "grid"
        else "You may propose parameter values and, when the idea needs it, a full "
        "replacement for the strategy source."
    )
    parts = [
        f"Strategy: {strategy_name}. Fitness metric: {cfg.metric} (out-of-sample).",
        mode,
        f"Iterations used: {len(lineage)} of {cfg.iterations}.",
        "",
        "Current strategy source, the file under edit:",
        "<strategy_source>",
        source,
        "</strategy_source>",
        "",
        "Current parameters:",
        json.dumps(dict(params), indent=1, sort_keys=True, default=str),
        "",
        "Lineage so far — out-of-sample first, in-sample beside it so you can see the gap:",
        json.dumps(rows, indent=1, default=str),
    ]
    if market:
        parts += [
            "",
            "Market reference — simply holding the benchmark over the same windows, "
            "with no costs. Beating the seed is not the bar; a proposal that cannot "
            "beat this out-of-sample is not worth shipping:",
            json.dumps(market, indent=1, default=str),
        ]
    if cfg.objective:
        parts += ["", f"Operator objective: {sanitize_text(cfg.objective, limit=400)}"]
    parts += ["", "Propose the next variation."]
    return SYSTEM_PROMPT, "\n".join(parts)




def _lineage_row(it: Iteration) -> dict[str, Any]:
    oos = it.metrics.get("oos") or {}
    ins = it.metrics.get("is") or {}
    return {
        "n": it.n,
        "params": it.params,
        "rationale": it.rationale,
        "ok": it.ok,
        "error": it.error,
        "oos": {k: oos.get(k) for k in ("sharpe", "total_return", "max_drawdown", "trades")},
        "in_sample": {k: ins.get(k) for k in ("sharpe", "total_return", "max_drawdown", "trades")},
        # Per-regime, so a strategy that only works in one market is visible as
        # such rather than averaged into a respectable pooled number.
        "by_period": it.metrics.get("periods") or [],
    }


# --- bookkeeping --------------------------------------------------------------


def _diff(
    prev_source: str,
    new_source: str,
    prev_params: Mapping[str, Any],
    new_params: Mapping[str, Any],
    n: int,
) -> str:
    """Unified diff of whatever actually changed -- source if it moved, else params."""
    if new_source != prev_source:
        a, b = prev_source.splitlines(), new_source.splitlines()
        label_a, label_b = f"iteration_{n - 1}.py", f"iteration_{n}.py"
    else:
        a = json.dumps(dict(prev_params), indent=1, sort_keys=True, default=str).splitlines()
        b = json.dumps(dict(new_params), indent=1, sort_keys=True, default=str).splitlines()
        label_a, label_b = f"params@{n - 1}", f"params@{n}"
    return "\n".join(difflib.unified_diff(a, b, fromfile=label_a, tofile=label_b, lineterm=""))


def _default_model() -> str:
    from lab.config import get_settings

    return get_settings().agent_model


def _loop_id(strategy: str) -> str:
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{strategy}-loop-{stamp}-{random.randint(100000, 999999)}"


def _loop_dir(loop_id: str) -> Path:
    from lab.config import get_settings

    return get_settings().paths.runs / loop_id
