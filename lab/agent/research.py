"""Open-ended research: a session, not a lineage.

``author_loop`` answers "make this strategy better" -- one seed, one chain, a
fixed number of turns. This answers the looser question an actual researcher
starts with: *here is a folder of strategies and a brief; go find something
worth running, and tell me when you are done.*

Four things change, and each one is the reason a separate module exists rather
than a flag on the old loop:

**No seed.** The agent is handed the whole strategy folder and picks what to
work on, including writing new files, including abandoning a line of attack.

**No iteration count.** It runs until it says it is satisfied, or until a
resource boundary stops it. Which means it has to be *told* where it stands --
every turn carries elapsed time, spend against budget, calls made, and how many
experiments it has already run. An agent that cannot see the clock will polish
one strategy forever.

**Satisfaction is a claim it has to defend.** ``finish`` requires naming the
run_id it stands behind. Stopping without evidence is not a conclusion, and
"I have improved things" with nothing to point at is the failure mode this
guards against.

**It is resumable.** The session is a file. A run that stops because the clock
ran out or a subscription hit its limit picks up later with its history intact,
which is what makes the subscription backend usable for work of this shape.

What does *not* change is every containment from the rest of the platform. Dates
and universe come from the operator's config and the schema cannot express a
change to them. The risk gate applies to every backtest. Fitness is out-of-sample.
Written strategies land in the session workspace, never in ``strategies/``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.agent.schemas import (
    RESEARCH_ACTION_TOOL,
    ModelClient,
    ResearchAction,
    call_tool,
    get_client,
    link_call,
    sanitize_text,
    validate_action,
)
from lab.timeutil import utcnow

log = logging.getLogger(__name__)

from lab.analysis import (  # noqa: E402
    TUNING_CHAIN_ALERT,
    fitness_one_sigma,
    flip,
    is_flag,
    neighbourhood_report,
    perturbable,
    perturbations,
)

# Moved to lab.analysis; alias kept so this module and its tests are unchanged.
_fitness_one_sigma = fitness_one_sigma


SYSTEM_PROMPT = """You are a quantitative researcher with a strategy folder, a backtest engine and a
finite amount of time. Each turn you take exactly one action and see the result.

You are not tuning one strategy. You are looking for something worth running. Abandoning a
line of attack is a legitimate and often correct move.

How to work:
- Read the brief first. It is the operator's actual goal and it outranks your instincts
  about what makes a nice strategy.
- Establish a baseline before you optimise. You cannot tell whether a change helped if you
  never measured the thing you changed it from.
- Beat the market reference or say plainly that you did not. A strategy that loses to
  holding the benchmark is not a result, however good its Sharpe looks next to its own
  earlier versions.
- Read `by_period` before the pooled score. Something that earns everything in one
  sub-period is fitted to a regime, not to an edge.
- Every row carries BOTH windows. `oos_*` is the held-out slice and answers "does this
  generalise"; `full_*` is the whole tradeable period and answers "what did it actually
  do". The out-of-sample window is often only a year or so, which is short enough to be
  mostly noise, so a strategy that beat the benchmark by a mile over the full period and
  trailed it over the last few months is probably out of favour, not broken -- say which
  you think it is rather than discarding it on the short window alone. Rank on the fitness
  metric, but do not write off a run whose `full_return_vs_market` is strongly positive
  without saying why.
- Every backtest reads the out-of-sample window, so the window stops being
  out-of-sample as you go. Tune one strategy five ways and keep the best score and
  you have the maximum of five draws, not an estimate -- the number is optimistic by
  roughly the spread between them, and `HOLDOUT PRESSURE` shows you that spread. If
  `oos_spread` is much larger than `full_spread`, the parameter is not finding an
  edge, it is moving return across the split; prefer the middle of a range that
  works over the single best value, and say in your verdict how many variants you
  tried before picking. A strategy you tested once and a strategy you tuned eight
  times are not comparable on out-of-sample score, and the second needs the bigger
  margin to be believed.
- Every score carries `fitness_one_sigma`: the sampling error of that metric on a
  window that short. Two scores closer together than one sigma are the same score,
  however many decimal places they are printed to -- an annualised Sharpe from ~230
  daily bars has a standard error near 1.0, which is wider than most sweeps. When the
  spread across your variants is inside one sigma, stop ranking on it and decide on
  full-period behaviour, exposure and per-period consistency instead, and say in the
  verdict that the score could not separate them.
- Sharpe alone will mislead you, and in one specific direction: a strategy that sits in
  cash has almost no volatility, so it can post a fine Sharpe while making almost no
  money. Every row carries `oos_exposure` and `oos_return_vs_market` for exactly this.
  A high Sharpe on 20% exposure that returns less than the benchmark is not a better
  strategy than a lower Sharpe that actually compounds -- it is a worse one wearing a
  better ratio. Before you finish, check that your pick beats the market reference on
  RETURN as well as on the fitness metric, and if it does not, say so plainly in the
  verdict rather than leaning on the ratio.
- Heavy gate clipping is a CONFOUND, never evidence of robustness. If most of a
  strategy's intents are clipped, what was backtested is the gate's version of it,
  its parameters barely reach the book, and a neighbourhood check that finds it
  insensitive to them is measuring the limit rather than the strategy. Do not argue
  that a pick "sits on flat ground" when `gate.share_clipped_or_blocked` is high --
  the flatness is the limit's. Either size the strategy to fit inside the limits or
  say plainly that the result belongs to the gate.
- `review` a run before you conclude anything about it. Aggregate metrics hide the two
  things that most often turn a result into a non-result: one ticker carrying the whole
  P&L, and a risk limit quietly clipping most of what the strategy tried to do. Check
  `concentration.top_ticker_share` and `gate.share_clipped_or_blocked` in particular.
  Reviewing costs a turn and no backtest, and it is usually worth it before a round of
  tuning rather than after.
- Review digests STAY in your context for the rest of the session, so you can compare
  two runs' attribution without paying for either again. Only the most recent few are
  kept; anything you want to outlive that belongs in `notes`.
- Watch the clock and the budget. They are shown every turn. Spending your last third of a
  session on the fourth decimal place of one parameter is a bad trade; so is stopping with
  half the budget unused because the first thing you tried looked fine.
- You cannot change the date range, the universe, the fill model or the risk limits. If a
  strategy only works when you choose its test period, it does not work.
- The catalogue marks strategies you cannot run: `runnable: false` means its data feed is
  missing or not configured for this run, and `contaminated: true` means it puts a model in
  the trading loop and is human-backtest-only. Do not spend turns on either -- they are
  listed so you know they exist, not so you can try them.

Stopping:
- `finish` when the evidence supports a recommendation, and name the run_id. Say what you
  would do next if you had longer -- the session is resumable and that note is what the
  next sitting starts from.
- If the run you name is the best of a tuning chain, say how many variants you tried and
  whether neighbouring values held up. Picking the single highest score off a sweep you
  ran against the holdout is the most common way to end a session with a number that will
  not survive contact with new data.
- When you finish, the winning parameters are automatically re-run about 10% either
  side of every numeric value. If the score collapses, the finish is handed back to you
  once with the numbers -- so prefer a value you expect to survive that, and do not
  treat a lone peak as a result.
- If you are running low on time or budget, finish deliberately at a defensible point
  rather than being cut off mid-experiment.
- Finishing early with an honest "nothing here beat buy-and-hold" is a real result and a
  better outcome than a fitted curve.

Reply only by calling the research_action tool.
"""

#: Appended to the system prompt so it never costs a turn to learn the API, and so
#: it sits in the cached prefix rather than being resent with every situation
#: update. Deliberately a complete, runnable example rather than a description: the
#: failure mode it prevents is a written strategy that does not import, which costs
#: a write turn *and* a backtest turn to discover.
STRATEGY_API = """
WRITING A STRATEGY — the whole API. You do not need to read an existing file for this.

A strategy is a plain module. No base class, no registration.

```python
NAME = "my_strategy"

PARAMS = {"lookback": 126, "top_n": 4}   # declared so they can be varied per backtest

class Strategy:
    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(PARAMS) | dict(params or {})
        self.state = {}                   # persists across bars within a run

    def on_bar(self, ctx) -> None:        # called once per bar; the decision point
        p = ctx.params
        scores = {}
        for t in ctx.universe:
            closes = ctx.history(t, "close", p["lookback"] + 1)
            if len(closes) < p["lookback"] + 1:
                continue                  # not enough history yet -- skip, never guess
            scores[t] = float(closes.iloc[-1]) / float(closes.iloc[0]) - 1.0

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        wanted = ranked[: int(p["top_n"])]

        for t in list(ctx.portfolio.positions):      # exits first: they free capacity
            if t not in wanted:
                ctx.close(t, reason="fell out of the top N")
        for t in wanted:
            ctx.order_target_pct(t, 1.0 / len(wanted), tag="entry", reason=f"rank {ranked.index(t)+1}")
```

CONTEXT API — the only way to touch data. Everything is filtered to what was
knowable at ctx.now, so look-ahead is impossible by construction.

  ctx.history(ticker, field, n) -> pd.Series   last n values, oldest first ("close","open","high","low","volume")
  ctx.bars(ticker, n)           -> pd.DataFrame OHLCV
  ctx.price(ticker)             -> float | None    latest close, None if unknown
  ctx.indicator(ticker, name, **params) -> pd.Series
  ctx.signals(source, **query)  -> list[Event]     alt-data, only if the config declares `sources`
  ctx.portfolio.positions       -> dict[str, Position]  (.qty .avg_price .last_price .unrealized_pct)
  ctx.portfolio.equity / .cash / .weight(ticker) / .gross_exposure
  ctx.order_target_pct(ticker, pct, tag="", reason="")   pct is a FRACTION of equity
  ctx.close(ticker, reason="")                            same as target 0
  ctx.log(**fields)             structured, lands in the decision journal
  ctx.now / ctx.session / ctx.universe / ctx.params

INDICATORS available to ctx.indicator (all causal, warm-up rows are NaN):
  sma ema wma rsi atr bbands macd roc momentum zscore rolling_vol donchian
  returns vwap adx stoch max_drawdown slope
  e.g. ctx.indicator(t, "sma", n=200) ; ctx.indicator(t, "rsi", n=14)
  bbands/macd/donchian/stoch return a DataFrame -- pass column="upper" to pick one.
  Take the last value defensively:  s = ctx.indicator(...).dropna(); v = float(s.iloc[-1]) if len(s) else None

RULES THAT WILL BITE YOU
  * on_bar runs EVERY bar. Guard on your own state if you only want to act sometimes.
  * order_target_pct sets a TARGET WEIGHT, not a delta. Calling it twice with 0.1
    leaves 10%, not 20%. Omitting a held ticker leaves it untouched -- exit explicitly.
  * A ticker not in ctx.universe raises. Iterate ctx.universe, never a hardcoded list.
  * Sizing and caps are NOT yours: the risk gate clips targets afterwards. Do not
    reimplement position limits; assume your target may come back smaller.
  * Return early when history is short. Indexing an empty Series raises and kills the run.
  * Keep it deterministic. No randomness, no wall-clock, no file or network access.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip() + "\n" + STRATEGY_API

#: Turns kept verbatim in the prompt. Older ones survive as rows in the
#: experiment table -- the numbers are what matter later, not the prose.
VERBATIM_TURNS = 16

#: Review digests carried in the prompt. A digest is ~2k tokens, so unlike the
#: caps inside one this is a real bound: at four retained a 60-call session costs
#: about $4.9 of input on Opus, against $1.9 when digests were dropped after the
#: turn that fetched them. It keeps the *most recent* four and the prompt says how
#: many it dropped, because a context that silently forgets is worse than one that
#: admits what it lost.
REVIEWS_KEPT = 4

#: Phrases that mean a provider stopped us rather than failed. Worth detecting
#: precisely: a rate limit should end the session cleanly and resumably, while a
#: genuine error should be retried or recorded as a failure.
_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota",
    "429",
    "too many requests",
    "overloaded",
    "resets at",
)


def looks_like_a_limit(message: str) -> bool:
    low = (message or "").lower()
    return any(marker in low for marker in _LIMIT_MARKERS)


def _emit(
    session_id: str,
    kind: str,
    message: str,
    **payload: Any,
) -> None:
    """Append one line to the event journal.

    The journal is what the console tails over its WebSocket, so this is the
    entire mechanism by which a session is watchable while it runs. Failures are
    swallowed: a research loop must not die because a log write did.
    """
    try:
        from lab.engine.events import EventKind, event
        from lab.registry.journal import EventJournal

        EventJournal().append(
            event(
                EventKind(kind),
                "research",
                at=utcnow(),
                run_id=session_id,
                strategy="research",
                message=message,
                payload=payload,
            )
        )
    except Exception:  # pragma: no cover - logging must never be load-bearing
        log.debug("could not journal research event", exc_info=True)


@dataclass
class ResearchConfig:
    """What the session may touch, and what stops it."""

    config: Path
    brief: str = ""
    #: Wall-clock ceiling for this sitting. The session is resumable, so this is
    #: "how long am I willing to leave it running", not "how long is the work".
    max_minutes: float = 30.0
    budget_usd: float = 5.0
    #: A hard cap on model calls. On a subscription there is no API that reports
    #: remaining quota, so this is the honest proxy for "do not burn my week".
    max_calls: int = 60
    max_experiments: int = 40
    model: str | None = None
    strategies_dir: Path | None = None
    oos_split: str = "4:1"
    robustness_periods: int = 6
    #: Backtests spent perturbing the winning parameters before a session is
    #: allowed to finish. These are the runs the agent will not do for itself:
    #: it has every incentive to stop at a good number and none to go looking for
    #: the cliff next to it. 0 disables the check.
    #:
    #: A *ceiling*, not a quota -- the plan never exceeds two runs per numeric
    #: parameter, so a four-parameter strategy costs eight regardless. 16 covers
    #: eight parameters both ways, which is a normal strategy; the previous
    #: default of 8 silently left six of seven parameters nudged one way only on
    #: a real session, and reported HOLDS.
    neighbourhood_runs: int = 16
    metric: str = "oos_sharpe"
    max_tokens: int = 4_096
    session_id: str | None = None
    workspace: Path | None = None
    register: bool = True

    def __post_init__(self) -> None:
        self.config = Path(self.config)
        if self.strategies_dir is not None:
            self.strategies_dir = Path(self.strategies_dir)
        if float(self.max_minutes) <= 0:
            raise ValueError("max_minutes must be > 0")
        if float(self.budget_usd) <= 0:
            raise ValueError("budget_usd must be > 0")
        if int(self.max_calls) < 1:
            raise ValueError("max_calls must be >= 1")


@dataclass
class Experiment:
    """One backtest the agent chose to run."""

    n: int
    run_id: str
    strategy: str
    params: dict[str, Any]
    score: float | None
    oos: dict[str, Any] = field(default_factory=dict)
    in_sample: dict[str, Any] = field(default_factory=dict)
    periods: list[dict[str, Any]] = field(default_factory=list)
    worst_period_sharpe: float | None = None
    #: Share of the out-of-sample window actually invested. Without it a strategy
    #: that sits in cash looks like a good one: low exposure means low volatility
    #: means a flattering Sharpe, on almost no return.
    oos_exposure: float | None = None
    #: Out-of-sample return minus the market reference's. The single number that
    #: says whether this was worth doing instead of buying the benchmark.
    oos_return_vs_market: float | None = None
    #: The whole tradeable period. The out-of-sample slice answers "does it
    #: generalise"; this answers "what did it actually do", and judging on the
    #: first alone throws away four fifths of the evidence.
    full: dict[str, Any] = field(default_factory=dict)
    full_return_vs_market: float | None = None
    #: Which attempt at *this strategy* this is, 1-based. Every backtest reads the
    #: held-out window, so the fifth variant's out-of-sample score is the best of
    #: five looks at the test set, not an independent estimate of anything.
    variant_index: int = 1
    #: One-sigma sampling error on ``score``, given how short its window is. Two
    #: scores closer together than this are the same score.
    fitness_one_sigma: float | None = None
    rationale: str = ""
    ok: bool = True
    error: str = ""
    #: A Pattern-B run: the model's training data covers the test window, so this
    #: measures plumbing, never edge. Excluded from `best` and from `finish`.
    contaminated: bool = False

    def row(self) -> dict[str, Any]:
        """The compact form that survives in the prompt after it scrolls off."""
        return {
            "n": self.n,
            # The agent needs this to review or finish on a run. Without it,
            # ids are only visible for the handful of turns kept verbatim, so
            # an older experiment becomes unreferenceable once it scrolls off.
            "run_id": self.run_id,
            "strategy": self.strategy,
            "params": self.params,
            "oos_sharpe": (self.oos or {}).get("sharpe"),
            "oos_return": (self.oos or {}).get("total_return"),
            "oos_exposure": self.oos_exposure,
            "oos_return_vs_market": self.oos_return_vs_market,
            "full_return": (self.full or {}).get("total_return"),
            "full_sharpe": (self.full or {}).get("sharpe"),
            "full_return_vs_market": self.full_return_vs_market,
            "variant_index": self.variant_index,
            # The score and its error travel together or not at all: a score alone
            # invites ranking on differences the window cannot resolve.
            "score": self.score,
            "fitness_one_sigma": self.fitness_one_sigma,
            "is_sharpe": (self.in_sample or {}).get("sharpe"),
            "worst_period_sharpe": self.worst_period_sharpe,
            "trades": (self.oos or {}).get("trades"),
            "ok": self.ok,
            "error": self.error,
            "contaminated": self.contaminated,
        }


@dataclass
class ResearchSession:
    """Everything a later sitting needs to carry on."""

    session_id: str
    brief: str
    config_path: str
    created_at: str
    updated_at: str = ""
    elapsed_s: float = 0.0
    calls: int = 0
    spend_usd: float = 0.0
    billing: str = "api"
    experiments: list[Experiment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    #: Review digests already read, kept rather than dropped after the turn that
    #: fetched them. Culling made sense when a digest was most of the prompt; it
    #: is not, and an agent that reviewed a run and then lost the detail cannot
    #: compare two runs' attribution without spending the turn again.
    reviews: list[dict[str, Any]] = field(default_factory=list)
    #: Result of perturbing the winning parameters. Kept on the session so the
    #: operator sees the same fragility evidence the agent was made to answer for.
    neighbourhood: dict[str, Any] | None = None
    #: A fragile pick is refused exactly once. Twice would be a loop, and the
    #: agent has already been shown the evidence by then -- if it wants to stand
    #: behind the run anyway it may, on the record.
    neighbourhood_challenged: bool = False
    status: str = "running"
    stopped_because: str = ""
    verdict: str = ""
    satisfied_with: str | None = None
    next_steps: str = ""
    market: dict[str, Any] | None = None
    workspace: str = ""

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["experiments"] = [asdict(e) for e in self.experiments]
        return d

    def save(self) -> Path:
        path = Path(self.workspace) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utcnow().isoformat()
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str, *, workspace: Path | str | None = None) -> "ResearchSession":
        """Reload a session by id, or from an explicit workspace.

        Sessions normally live in ``runs/<session_id>/``, but a caller may pin a
        workspace elsewhere; resuming has to find it either way, so both are
        tried before giving up.
        """
        candidates = [Path(workspace) / "session.json"] if workspace else []
        candidates.append(_session_dir(session_id) / "session.json")
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            looked = " or ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"no research session at {looked}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        experiments = [Experiment(**e) for e in raw.pop("experiments", [])]
        known = {f for f in cls.__dataclass_fields__}
        session = cls(**{k: v for k, v in raw.items() if k in known})
        session.experiments = experiments
        return session

    # --- reading -----------------------------------------------------------

    @property
    def best(self) -> Experiment | None:
        scored = [
            e
            for e in self.experiments
            if e.ok and not e.contaminated and e.score is not None and e.score == e.score
        ]
        return max(scored, key=lambda e: e.score or float("-inf")) if scored else None


def _session_dir(session_id: str) -> Path:
    from lab.config import get_settings

    return get_settings().paths.runs / session_id


def _new_session_id() -> str:
    return f"research-{utcnow().strftime('%Y%m%dT%H%M%S')}-{int(time.perf_counter() * 1e6) % 1_000_000:06d}"


# --- the catalogue the agent chooses from --------------------------------------


def _catalog(
    cfg: ResearchConfig, workspace: Path, session: ResearchSession, base_cfg: Any
) -> list[dict[str, Any]]:
    """Every strategy the agent may run, with its best result so far.

    Attaching results to the catalogue rather than making the agent cross-
    reference the experiment table is the difference between "here are eight
    files" and "here is what you already know about each of them".
    """
    from lab.config import get_settings
    from lab.engine.loader import load_strategy

    roots = [cfg.strategies_dir or get_settings().paths.strategies, workspace]
    best_by_strategy: dict[str, float] = {}
    for e in session.experiments:
        if e.ok and e.score is not None and e.score == e.score:
            prior = best_by_strategy.get(e.strategy)
            if prior is None or e.score > prior:
                best_by_strategy[e.strategy] = e.score

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root or not Path(root).exists():
            continue
        for path in sorted(Path(root).glob("*.py")):
            if path.name.startswith("_") or path.name in seen:
                continue
            seen.add(path.name)
            entry: dict[str, Any] = {
                "file": path.name,
                "origin": "workspace" if root == workspace else "library",
            }
            try:
                loaded = load_strategy(path)
                doc = (loaded.doc or "").strip().splitlines()
                entry["summary"] = doc[0] if doc else ""
                entry["params"] = loaded.params
            except Exception as exc:
                entry["error"] = f"does not load: {type(exc).__name__}: {exc}"
            runnable, why = _source_readiness(entry.get("params") or {}, base_cfg)
            if not runnable:
                entry["runnable"] = False
                entry["unavailable"] = why
            if _is_llm_strategy(path):
                entry["contaminated"] = True
                entry["warning"] = (
                    "puts a model inside the trading loop; any historical backtest of it "
                    "is contaminated by the model's training data and cannot be your "
                    "recommendation. Run it only as a plumbing check."
                )
            if path.name in best_by_strategy:
                entry["your_best_score"] = round(best_by_strategy[path.name], 4)
            out.append(entry)
    return out


def _is_llm_strategy(path: Path) -> bool:
    """Does this file put a model inside the trading loop?

    It matters here more than anywhere: a Pattern-B strategy backtested over
    historical data is contaminated by the model's own training set, so its
    numbers measure recall, not edge. A research loop that ranked one as its
    best find would be recommending exactly the thing the platform spends the
    rest of its effort refusing to let anyone believe.
    """
    try:
        from lab.agent.in_loop_strategy import AgentStrategy
        from lab.engine.loader import load_strategy

        loaded = load_strategy(path)
        if isinstance(loaded.instance, AgentStrategy):
            return True
        return type(loaded.instance).__name__ == "AgentStrategy"
    except Exception:
        # Fall back to the text: a file we cannot import might still be one.
        try:
            return "in_loop_strategy" in path.read_text(encoding="utf-8")
        except OSError:
            return False


def _declared_sources(params: Mapping[str, Any]) -> list[str]:
    """Alt-data sources a strategy says it needs.

    Read off the declared params rather than by importing and inspecting: a
    strategy names its feed in ``PARAMS`` precisely so it can be varied from a
    config, which makes it equally readable from outside.
    """
    found: list[str] = []
    for key, value in (params or {}).items():
        if "source" not in key.lower():
            continue
        if isinstance(value, str) and value:
            found.append(value)
        elif isinstance(value, (list, tuple, set)):
            found.extend(str(v) for v in value if v)
    return sorted(set(found))


def _source_readiness(params: Mapping[str, Any], base_cfg: Any) -> tuple[bool, str]:
    """Can this strategy see the data it needs, in *this* run config?

    Three ways it cannot, and they need different fixes, so they get different
    messages. Reporting them up front is the difference between the agent
    learning "govgreed is unavailable" and it burning a turn on a backtest that
    silently produces zero trades and looks like a strategy that does not work.
    """
    needed = _declared_sources(params)
    if not needed:
        return True, ""

    configured = {str(x) for x in (getattr(base_cfg, "sources", None) or [])}
    problems: list[str] = []
    for source in needed:
        if source not in configured:
            problems.append(
                f"the run config does not list {source!r} under `sources:`, so "
                f"ctx.signals() would return nothing"
            )
            continue
        try:
            from lab.store import parquet_io

            rows = len(parquet_io.read_signals(source))
        except Exception:
            rows = 0
        if rows:
            continue
        reason = f"no {source!r} data in the store"
        try:
            from lab.adapters.base import get_adapter

            ok, why = get_adapter(source).available()
            if not ok:
                reason += f" and the adapter cannot fetch any ({why})"
        except Exception:
            reason += " and there is no adapter by that name"
        problems.append(reason)

    return (not problems), "; ".join(problems)


def _resolve_strategy(name: str, cfg: ResearchConfig, workspace: Path) -> Path | None:
    """Workspace first: an edited variant shadows the library file it came from."""
    from lab.config import get_settings

    for root in (workspace, cfg.strategies_dir or get_settings().paths.strategies):
        if not root:
            continue
        candidate = Path(root) / name
        if candidate.exists():
            return candidate
    return None


# --- situational awareness ------------------------------------------------------


def _situation(cfg: ResearchConfig, session: ResearchSession, started: float) -> dict[str, Any]:
    elapsed = session.elapsed_s + (time.perf_counter() - started)
    limit_s = float(cfg.max_minutes) * 60.0
    return {
        "elapsed_minutes": round(elapsed / 60.0, 1),
        "minutes_remaining": round(max(0.0, limit_s - elapsed) / 60.0, 1),
        "calls_used": session.calls,
        "calls_remaining": max(0, int(cfg.max_calls) - session.calls),
        "spend": round(session.spend_usd, 4),
        "budget": float(cfg.budget_usd),
        "billing": session.billing,
        "experiments_run": len(session.experiments),
        # Named separately from experiments_run because it means something
        # different: this is how many times the held-out window has been read.
        "holdout_evaluations": sum(1 for e in session.experiments if e.ok),
        "experiments_remaining": max(0, int(cfg.max_experiments) - len(session.experiments)),
    }


#: A tuning chain shorter than this is ordinary work, not a warning sign.




def _tuning_pressure(session: "ResearchSession") -> list[dict[str, Any]]:
    """How hard each strategy has been tuned against the held-out window.

    Every backtest is a look at the test set. Tune one strategy five ways, keep
    the best out-of-sample score, and that score is the maximum of five draws --
    biased upward by roughly the spread of the noise, and no longer an estimate
    of anything.

    The tell is direction, not spread. A parameter change that finds a real edge
    lifts both windows, and lifts the full period *more* -- it covers four times
    as long, so the same annualised improvement compounds further. A change that
    lifts the held-out slice while the full period stays flat or falls has not
    found anything; it has moved return across the split boundary.

    That is exactly what happened on the session this was written for. Tuning one
    parameter took the out-of-sample return from 0.129 to 0.280 while the full
    period went 2.473 to 2.368 -- a 117% "improvement" that cost 4% of the actual
    result, because in-sample return fell by more than out-of-sample gained.
    """
    chains: dict[str, list[Experiment]] = {}
    for e in session.experiments:
        if e.ok and e.score is not None:
            chains.setdefault(e.strategy, []).append(e)

    def _ret(exp: Experiment, window: str) -> float | None:
        v = (getattr(exp, window, None) or {}).get("total_return")
        return float(v) if isinstance(v, (int, float)) else None

    def _spread(values: Sequence[float]) -> tuple[float, float]:
        """Absolute spread, and spread relative to the mean magnitude."""
        lo, hi = min(values), max(values)
        mean = sum(abs(v) for v in values) / len(values)
        return hi - lo, ((hi - lo) / mean if mean > 1e-9 else float("inf"))

    out: list[dict[str, Any]] = []
    for name, runs in chains.items():
        usable = [r for r in runs if _ret(r, "oos") is not None and _ret(r, "full") is not None]
        if len(usable) < TUNING_CHAIN_ALERT:
            continue
        first = usable[0]
        # Argmax of the fitness metric, because that is the one the agent ranks on
        # and therefore the one it will end up standing behind.
        best = max(usable, key=lambda r: r.score if r.score is not None else float("-inf"))
        oos_gain = (_ret(best, "oos") or 0.0) - (_ret(first, "oos") or 0.0)
        full_gain = (_ret(best, "full") or 0.0) - (_ret(first, "full") or 0.0)

        row: dict[str, Any] = {
            "strategy": name,
            "variants_tried": len(usable),
            "best_variant": best.run_id,
            "best_fitness": best.score,
            "fitness_one_sigma": best.fitness_one_sigma,
            "best_oos_return": round(_ret(best, "oos") or 0.0, 4),
            "full_return_of_that_variant": round(_ret(best, "full") or 0.0, 4),
            "oos_gain_from_tuning": round(oos_gain, 4),
            "full_gain_from_tuning": round(full_gain, 4),
        }

        warnings: list[str] = []
        # The full period is four-ish times longer, so a real improvement shows up
        # *larger* there. Requiring only half is deliberately generous.
        if oos_gain > 0.02 and full_gain < oos_gain * 0.5:
            warnings.append(
                f"tuning lifted the held-out window by {oos_gain:+.4f} while the full "
                f"period moved {full_gain:+.4f}. The full period is the longer window, "
                f"so a real improvement shows up larger there, not smaller -- this is "
                f"return moving across the split, not an edge. Treat "
                f"{row['best_oos_return']} as the best of {len(usable)} looks at the "
                f"test set rather than as expected performance."
            )

        scores = [r.score for r in usable if r.score is not None]
        fulls = [_ret(r, "full") for r in usable]
        fulls = [v for v in fulls if v is not None]
        if len(scores) >= TUNING_CHAIN_ALERT and len(fulls) >= TUNING_CHAIN_ALERT:
            fit_abs, fit_rel = _spread(scores)
            _, full_rel = _spread(fulls)
            row["fitness_spread"] = round(fit_abs, 4)
            row["fitness_relative_spread"] = round(fit_rel, 3)
            row["full_return_relative_spread"] = round(full_rel, 3)

            # The ranking metric jumping around far more than the thing it is
            # supposed to rank means the ranking is mostly noise.
            if full_rel > 1e-9 and fit_rel > full_rel * 2.0:
                warnings.append(
                    f"the fitness metric swings {fit_rel:.2f}x its own mean across these "
                    f"{len(scores)} variants while full-period return swings only "
                    f"{full_rel:.2f}x -- the metric is {fit_rel / full_rel:.1f} times less "
                    f"stable than the result it is ranking, so most of the ordering is "
                    f"noise rather than signal."
                )
            # If the whole sweep fits inside one standard error, no ordering exists.
            sigma = best.fitness_one_sigma
            if isinstance(sigma, (int, float)) and sigma > 0 and fit_abs <= float(sigma):
                warnings.append(
                    f"the entire spread across these variants ({fit_abs:.4f}) is smaller "
                    f"than one standard error of the metric itself ({float(sigma):.4f}), "
                    f"which is what a window this short can resolve. These variants are "
                    f"statistically indistinguishable -- choosing between them on score "
                    f"is choosing on noise. Decide on full-period behaviour, exposure and "
                    f"per-period consistency instead."
                )

        if warnings:
            row["warning"] = " ".join(warnings)
        out.append(row)
    return out


def _render_prompt(
    cfg: ResearchConfig, session: ResearchSession, situation: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]], last_result: str,
) -> tuple[str, str]:
    """Render the turn as ``(stable, volatile)``.

    Split rather than returned as one string because the stable half is byte
    identical between calls in a sitting and can therefore carry a cache
    breakpoint, while everything that grows or ticks sits behind it at ordinary
    price. Only genuinely invariant material goes in the first half: the brief,
    the market reference, the billing note. The catalogue, the experiment table
    and the digests all change as the session runs, and marking a breakpoint
    after them would pay the 1.25x cache-write premium every turn for an entry
    that is never read back.

    The situation block moved to the end for the same reason it reads better
    there -- the clock immediately before "take one action" -- but it used to sit
    second, where a value that changes every single turn invalidated everything
    behind it.
    """
    stable: list[str] = []
    volatile: list[str] = []

    if session.brief:
        stable += [
            "OPERATOR BRIEF — this is the goal, and it outranks your own taste:",
            "<brief>",
            sanitize_text(session.brief, limit=4_000),
            "</brief>",
            "",
        ]
    if session.billing == "subscription":
        stable += [
            "You are running against a Claude subscription. There is no API that "
            "reports remaining quota, so `calls_remaining` is the operator's cap, "
            "not the real ceiling — if a limit is hit the session ends cleanly and "
            "resumes later, but finishing deliberately is better than being cut off.",
            "",
        ]
    if session.market:
        stable += [
            "MARKET REFERENCE — holding the benchmark over the same window, costless. "
            "This is the bar, not the seed:",
            json.dumps(session.market, indent=1),
            "",
        ]

    volatile += [
        "STRATEGIES YOU MAY RUN (workspace files shadow library files of the same name):",
        json.dumps(list(catalog), indent=1, default=str),
        "",
    ]
    if session.experiments:
        volatile += [
            f"EXPERIMENTS SO FAR ({len(session.experiments)}), both windows on every row:",
            json.dumps([e.row() for e in session.experiments], indent=1, default=str),
            "",
        ]
    if session.reviews:
        kept = session.reviews[-REVIEWS_KEPT:]
        dropped = len(session.reviews) - len(kept)
        header = f"RUN DETAIL YOU HAVE READ ({len(kept)})"
        if dropped:
            header += f", plus {dropped} older digest(s) no longer shown"
        volatile += [header + ":", json.dumps(kept, indent=1, default=str), ""]
    pressure = _tuning_pressure(session)
    if pressure:
        volatile += [
            "HOLDOUT PRESSURE — every backtest is a look at the held-out window:",
            json.dumps(pressure, indent=1, default=str),
            "",
        ]
    if session.notes:
        volatile += ["YOUR NOTES:", *(f"- {n}" for n in session.notes), ""]

    recent = session.turns[-VERBATIM_TURNS:]
    if recent:
        volatile += [
            "YOUR LAST FEW TURNS:",
            json.dumps(recent, indent=1, default=str),
            "",
        ]

    best = session.best
    if best is not None:
        volatile += [f"BEST SO FAR: {best.run_id} ({best.strategy}) score {best.score}", ""]
    volatile += ["WHERE YOU STAND:", json.dumps(dict(situation), indent=1), ""]
    if last_result:
        volatile += ["RESULT OF YOUR LAST ACTION:", last_result, ""]

    volatile.append("Take one action.")
    return "\n".join(stable), "\n".join(volatile)


# --- executing an action ---------------------------------------------------------


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _excess(total_return: Any, session: "ResearchSession", window: str = "oos") -> float | None:
    """This run's return minus the benchmark's, over the same window."""
    mine = _round(total_return)
    bar = _round(((session.market or {}).get(window) or {}).get("total_return"))
    return None if mine is None or bar is None else round(mine - bar, 4)


def _run_experiment(
    action: ResearchAction,
    cfg: ResearchConfig,
    base_cfg: Any,
    workspace: Path,
    session: ResearchSession,
    oos_start: datetime,
) -> Experiment:
    from lab.agent.author_loop import _period_breakdown, _score, _window_metrics
    from lab.backtest.runner import run_backtest

    n = len(session.experiments) + 1
    path = _resolve_strategy(str(action.strategy), cfg, workspace)
    if path is None:
        return Experiment(
            n=n, run_id="", strategy=str(action.strategy), params=dict(action.params),
            score=None, ok=False, rationale=action.rationale,
            error=f"no strategy file named {action.strategy!r}",
        )

    from lab.engine.loader import load_strategy

    try:
        declared = load_strategy(path).params
    except Exception:
        declared = {}
    ready, why = _source_readiness(declared, base_cfg)
    if not ready:
        return Experiment(
            n=n, run_id="", strategy=path.name, params=dict(action.params),
            score=None, ok=False, rationale=action.rationale,
            error=(
                f"not run: {why}. A backtest would produce zero trades and look "
                f"like a strategy that does not work. Pick one whose data you have."
            ),
        )

    contaminated = _is_llm_strategy(path)
    if contaminated:
        # Refused, not merely flagged. A Pattern-B strategy calls a model on
        # every bar, so backtesting one inside an unattended loop spends real
        # money per bar to produce a number that is contaminated by
        # construction and can never be the recommendation. Paying for an
        # unusable answer is the worst of both.
        return Experiment(
            n=n, run_id="", strategy=path.name, params=dict(action.params),
            score=None, ok=False, rationale=action.rationale, contaminated=True,
            error=(
                f"{path.name} puts a model inside the trading loop. Backtesting it "
                f"would call that model once per bar, and the result would be "
                f"contaminated by the model's training data either way. Not run. "
                f"Pick a deterministic strategy."
            ),
        )
    run_cfg = replace(
        base_cfg,
        strategy=str(path),
        params=dict(action.params),
        origin="agent-research",
        notes=f"research {session.session_id} turn {n}",
        contaminated=bool(getattr(base_cfg, "contaminated", False) or contaminated),
    )
    try:
        result = run_backtest(run_cfg, register=cfg.register, journal=False)
    except Exception as exc:
        return Experiment(
            n=n, run_id="", strategy=path.name, params=dict(action.params),
            score=None, ok=False, rationale=action.rationale,
            error=f"{type(exc).__name__}: {exc}", contaminated=contaminated,
        )

    timeframe = getattr(run_cfg, "timeframe", "1d")
    is_metrics = _window_metrics(result, timeframe, hi=oos_start, inclusive_hi=False)
    oos_metrics = _window_metrics(result, timeframe, lo=oos_start)
    full_metrics = _window_metrics(result, timeframe)
    periods = _period_breakdown(result, timeframe, int(cfg.robustness_periods))
    metrics = {"is": is_metrics, "oos": oos_metrics, "full": full_metrics, "periods": periods}
    score = _score(metrics, cfg.metric)
    worst = [p.get("sharpe") for p in periods if isinstance(p.get("sharpe"), (int, float))]

    return Experiment(
        n=n,
        run_id=result.run_id,
        strategy=path.name,
        params=dict(action.params),
        score=None if score != score else round(score, 6),
        oos=oos_metrics,
        in_sample=is_metrics,
        periods=periods,
        worst_period_sharpe=round(min(worst), 4) if worst else None,
        oos_exposure=_round(oos_metrics.get("exposure")),
        oos_return_vs_market=_excess(oos_metrics.get("total_return"), session),
        full=full_metrics,
        full_return_vs_market=_excess(full_metrics.get("total_return"), session, "full"),
        variant_index=sum(1 for e in session.experiments if e.strategy == path.name) + 1,
        fitness_one_sigma=_one_sigma_for(cfg, metrics, timeframe),
        rationale=action.rationale,
        contaminated=contaminated,
    )


def _one_sigma_for(cfg: "ResearchConfig", metrics: Mapping[str, Any],
                   timeframe: str) -> float | None:
    """Route the metric to the window it is actually measured over."""
    metric = str(cfg.metric)
    if metric.startswith("worst_"):
        # A sub-period, so a quarter or a sixth of the bars -- and correspondingly
        # noisier than the pooled figure it is often compared against.
        full = metrics.get("full") or {}
        n = int(full.get("periods") or 0) // max(1, int(cfg.robustness_periods))
        return _fitness_one_sigma(metric, full, timeframe, n)
    key = "is" if metric.startswith("is_") else "full" if metric.startswith("full_") else "oos"
    window = metrics.get(key) or {}
    return _fitness_one_sigma(metric, window, timeframe, int(window.get("periods") or 0))


#: A neighbour scoring below this fraction of the chosen score is treated as a
#: cliff rather than a slope. Generous on purpose: the point is to catch a pick
#: perched on a spike, not to demand a flat landscape.

#: Above this share of intents clipped or blocked, a neighbourhood result stops
#: being evidence about the strategy. Half is generous: at that point as many of
#: the strategy's decisions are the gate's as are its own.


#: The perturbation plan moved to lab.analysis so that this loop and the MCP
#: toolset ask the same question with the same steps -- two implementations of
#: "about ten percent either side" is how a robustness check comes to mean one
#: thing in a research session and another in a review. Aliased under the old
#: private names: renaming a check is not a change to what it checks.
_is_flag = is_flag
_perturbable = perturbable
_flip = flip
_perturbations = perturbations


def _neighbourhood(
    chosen: Experiment, cfg: "ResearchConfig", base_cfg: Any, workspace: Path,
    session: "ResearchSession", oos_start: datetime,
) -> dict[str, Any]:
    """Re-run the winning strategy at nearby parameters and see if it survives.

    A held-out score is the maximum of however many looks the session took at the
    test set, and the parameter that produced it was chosen because it produced
    it. Neither fact is visible from the number. Perturbing the winner is the
    cheapest way to tell a plateau from a spike, and it costs backtests rather
    than model calls -- wall clock, not budget.
    """
    budget = max(0, int(cfg.neighbourhood_runs))
    plan = perturbations(chosen.params, budget)
    if not plan:
        return {"checked": 0, "note": "no numeric parameters to perturb"}

    tried: list[dict[str, Any]] = []
    for key, value in plan:
        action = ResearchAction(
            action="backtest",
            rationale=f"neighbourhood check: {key}={value}",
            strategy=chosen.strategy,
            params=dict(chosen.params) | {key: value},
        )
        # register=False: these are diagnostics, not experiments the agent chose,
        # and they must not compete for `best` or inflate the experiment count.
        run = _run_experiment(
            action, replace(cfg, register=False), base_cfg, workspace, session, oos_start
        )
        tried.append({
            "param": key,
            "value": value,
            "score": run.score,
            "full_return": (run.full or {}).get("total_return"),
            "ok": run.ok,
            "error": run.error,
        })
        _emit(
            session.session_id, "log",
            f"neighbourhood {key}={value} -> {run.score}",
            turn=len(session.turns) + 1,
        )

    # A perturbation the gate overrides cannot move the score, so a heavily
    # clipped strategy is guaranteed to look robust and the flatness belongs to
    # the limit rather than the strategy. Read here, judged in the report.
    gate: dict[str, Any] = {}
    try:
        from lab.agent.review import gate_activity
        from lab.backtest.runner import artifact_dir

        gate = dict(gate_activity(artifact_dir(chosen.run_id)) or {})
    except Exception as exc:  # noqa: BLE001 - no gate data is not a failed check
        log.warning("could not read gate activity for the neighbourhood check: %s", exc)

    return neighbourhood_report(
        chosen.params,
        chosen.score,
        (chosen.full or {}).get("total_return"),
        tried,
        gate=gate,
    )


def _write_strategy(action: ResearchAction, workspace: Path) -> str:
    """Persist a model-authored strategy into the session workspace only.

    Never into ``strategies/``. The library is the operator's; a research
    session gets a scratch directory and everything it writes stays reviewable
    and disposable.
    """
    name = str(action.strategy)
    if not name.endswith(".py"):
        name = f"{name}.py"
    path = workspace / name
    path.write_text(str(action.source or ""), encoding="utf-8")

    from lab.engine.loader import load_strategy

    try:
        load_strategy(path)
    except Exception as exc:
        return (
            f"wrote {name} ({len(action.source or '')} chars) but it DOES NOT LOAD: "
            f"{type(exc).__name__}: {exc}. Fix it before backtesting."
        )
    return f"wrote {name} ({len(action.source or '')} chars) and it imports cleanly."


def _inspect(action: ResearchAction, cfg: ResearchConfig, workspace: Path) -> str:
    path = _resolve_strategy(str(action.strategy), cfg, workspace)
    if path is None:
        return f"no strategy file named {action.strategy!r}"
    source = path.read_text(encoding="utf-8")
    if len(source) > 20_000:
        source = source[:20_000] + "\n# ... truncated ...\n"
    return f"--- {path.name} ---\n{source}"


# --- the loop ---------------------------------------------------------------------


def run_research(
    cfg: ResearchConfig,
    *,
    client: ModelClient | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Run one sitting of an open-ended research session."""
    from lab.agent.author_loop import (
        _market_reference,
        _split_point,
        _tradeable_timestamps,
    )
    from lab.backtest.runner import BacktestConfig

    started = time.perf_counter()
    base_cfg = BacktestConfig.from_yaml(cfg.config)
    timestamps = _tradeable_timestamps(base_cfg, None)
    if len(timestamps) < 4:
        raise ValueError("research needs at least 4 bars of data; run `lab pull` first")
    _, oos_start = _split_point(timestamps, cfg.oos_split)

    if resume:
        session = ResearchSession.load(resume, workspace=cfg.workspace)
        session.status = "running"
        session.stopped_because = ""
    else:
        session_id = cfg.session_id or _new_session_id()
        workspace = Path(cfg.workspace) if cfg.workspace else _session_dir(session_id)
        workspace.mkdir(parents=True, exist_ok=True)
        session = ResearchSession(
            session_id=session_id,
            brief=cfg.brief,
            config_path=str(cfg.config),
            created_at=utcnow().isoformat(),
            workspace=str(workspace),
            market=_market_reference(
                base_cfg, oos_start, timestamps[0] if timestamps else None
            ),
        )

    workspace = Path(session.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    _emit(
        session.session_id,
        "run_start",
        f"research {'resumed' if resume else 'started'}"
        + (f" — {session.brief[:120]}" if session.brief else ""),
        brief=session.brief,
        resumed=bool(resume),
        experiments_so_far=len(session.experiments),
    )
    if client is None:
        client = get_client()
    session.billing = getattr(client, "billing", "api")
    model = cfg.model or getattr(client, "model", None) or "unknown"

    last_result = ""
    consecutive_failures = 0

    while True:
        situation = _situation(cfg, session, started)
        stop = _stop_reason(cfg, session, situation)
        if stop:
            session.stopped_because = stop
            break

        stable, volatile = _render_prompt(
            cfg, session, situation, _catalog(cfg, workspace, session, base_cfg), last_result
        )
        call = call_tool(
            client,
            model=model,
            system=SYSTEM_PROMPT,
            user=volatile,
            stable_user=stable,
            cache=True,
            tools=[RESEARCH_ACTION_TOOL],
            force_tool="research_action",
            max_tokens=int(cfg.max_tokens),
            run_id=session.session_id,
            strategy="research",
        )
        session.calls += 1
        session.spend_usd += call.cost_usd
        session.billing = call.billing or session.billing

        if not call.ok:
            if looks_like_a_limit(call.error):
                # Not a failure: the provider stopped us. End cleanly so the
                # session can be picked up when the window resets.
                session.stopped_because = "rate_limit"
                session.next_steps = session.next_steps or (
                    "stopped on a provider limit mid-session; resume to continue"
                )
                break
            consecutive_failures += 1
            last_result = f"your last call failed: {call.error}"
            if consecutive_failures >= 3:
                session.stopped_because = "model_errors"
                break
            continue

        try:
            action = validate_action(call.input)
        except ValueError as exc:
            consecutive_failures += 1
            last_result = f"your action was rejected: {exc}. Try again."
            session.turns.append({"rejected": str(exc)})
            if consecutive_failures >= 3:
                session.stopped_because = "invalid_actions"
                break
            continue

        consecutive_failures = 0
        turn: dict[str, Any] = {"action": action.action, "rationale": action.rationale}
        _emit(
            session.session_id,
            "log",
            f"{action.action}"
            + (f" {action.strategy}" if action.strategy else "")
            + f" — {action.rationale}",
            turn=len(session.turns) + 1,
            action=action.action,
            strategy=action.strategy,
            params=action.params,
            rationale=action.rationale,
            elapsed_minutes=situation["elapsed_minutes"],
            calls_used=situation["calls_used"],
            spend=situation["spend"],
        )
        if action.notes:
            session.notes.append(action.notes)

        if action.action == "finish":
            chosen = next(
                (e for e in session.experiments if e.run_id == action.satisfied_with), None
            )
            if chosen is not None and chosen.contaminated:
                # Refused rather than accepted-with-a-caveat: this is the one
                # conclusion the platform must never let a session reach.
                last_result = (
                    f"REFUSED: {action.satisfied_with} is a contaminated Pattern-B run "
                    f"({chosen.strategy} puts a model in the trading loop), so it cannot be "
                    f"your recommendation. Pick a deterministic strategy, or finish by "
                    f"saying nothing here beat the benchmark."
                )
                session.turns.append({"rejected": last_result})
                continue

            # Perturb the winning parameters before letting the session close.
            # The agent will not do this for itself: it arrives at `finish` with a
            # number it likes and no incentive to go looking for the cliff beside
            # it. Costs backtests, not model calls.
            if chosen is not None and chosen.ok and int(cfg.neighbourhood_runs) > 0:
                if session.neighbourhood is None:
                    _emit(
                        session.session_id, "log",
                        f"checking the neighbourhood of {action.satisfied_with} "
                        f"({cfg.neighbourhood_runs} runs)",
                        turn=len(session.turns) + 1,
                    )
                    session.neighbourhood = _neighbourhood(
                        chosen, cfg, base_cfg, workspace, session, oos_start
                    )
                    session.save()

                fragile = bool((session.neighbourhood or {}).get("fragile"))
                room = situation["calls_remaining"] > 2 and situation["minutes_remaining"] > 2
                if fragile and not session.neighbourhood_challenged and room:
                    # Refused once, with the evidence. Not twice -- by then the
                    # agent has seen it, and standing behind the run anyway is a
                    # judgement it is allowed to make on the record.
                    session.neighbourhood_challenged = True
                    last_result = (
                        "NOT FINISHED YET — your pick does not survive its own "
                        "neighbourhood:\n"
                        + json.dumps(session.neighbourhood, indent=1, default=str)
                        + "\n\nEither pick a parameter set from the middle of the range "
                        "that holds, or finish again and say plainly in your verdict "
                        "that the result is sensitive to the exact values."
                    )
                    session.turns.append({"rejected": "fragile neighbourhood"})
                    continue

            session.satisfied_with = action.satisfied_with
            session.verdict = action.rationale
            session.next_steps = action.notes or ""
            session.stopped_because = "satisfied"
            session.turns.append(turn | {"satisfied_with": action.satisfied_with})
            break

        if action.action == "backtest":
            experiment = _run_experiment(action, cfg, base_cfg, workspace, session, oos_start)
            session.experiments.append(experiment)
            if experiment.ok and experiment.run_id and call.call_id:
                link_call(call.call_id, experiment.run_id)
            last_result = json.dumps(experiment.row(), indent=1, default=str)
            turn |= {"strategy": experiment.strategy, "run_id": experiment.run_id,
                     "score": experiment.score}
            _emit(
                session.session_id,
                "log" if experiment.ok else "error",
                (
                    f"{experiment.strategy} oos sharpe "
                    f"{(experiment.oos or {}).get('sharpe')}"
                    if experiment.ok
                    else f"{experiment.strategy} failed: {experiment.error[:120]}"
                ),
                turn=len(session.turns) + 1,
                result=experiment.row(),
            )
        elif action.action == "write":
            last_result = _write_strategy(action, workspace)
            turn |= {"strategy": action.strategy, "result": last_result}
        elif action.action == "review":
            from lab.agent.review import review_run

            known = {e.run_id for e in session.experiments if e.run_id}
            if action.run_id not in known:
                last_result = (
                    f"{action.run_id} is not one of this session's runs. "
                    f"Available: {', '.join(sorted(known)) or 'none yet'}"
                )
            else:
                digest = review_run(str(action.run_id))
                # Kept on the session, not just echoed once: the next turn can
                # still see it, and so can the operator reading session.json.
                # A digest that failed to build is shown but not stored -- there
                # is nothing in it worth carrying for the rest of the session.
                if not digest.get("error"):
                    session.reviews = [
                        r for r in session.reviews if r.get("run_id") != action.run_id
                    ] + [digest]
                last_result = json.dumps(digest, indent=1, default=str)
            turn |= {"run_id": action.run_id}
        elif action.action == "inspect":
            last_result = _inspect(action, cfg, workspace)
            turn |= {"strategy": action.strategy}
        else:  # note
            last_result = "noted."

        session.turns.append(turn)
        session.save()

    session.elapsed_s += time.perf_counter() - started
    session.status = "finished" if session.stopped_because == "satisfied" else "stopped"
    _emit(
        session.session_id,
        "run_end",
        f"research {session.status}: {session.stopped_because}"
        + (f" — {session.verdict[:120]}" if session.verdict else ""),
        stopped_because=session.stopped_because,
        satisfied_with=session.satisfied_with,
        experiments=len(session.experiments),
        calls=session.calls,
    )
    best = session.best
    session.save()

    return {
        "session_id": session.session_id,
        "status": session.status,
        "stopped_because": session.stopped_because,
        "brief": session.brief,
        "elapsed_minutes": round(session.elapsed_s / 60.0, 2),
        "calls": session.calls,
        "spend_usd": round(session.spend_usd, 6),
        "billing": session.billing,
        "experiments": [e.row() for e in session.experiments],
        "best": None if best is None else {"run_id": best.run_id, "strategy": best.strategy,
                                           "params": best.params, "score": best.score},
        "satisfied_with": session.satisfied_with,
        "verdict": session.verdict,
        "next_steps": session.next_steps,
        "market": session.market,
        "notes": session.notes,
        #: Digests the agent read. Carried out of the session so the operator can
        #: see the same attribution the agent was reasoning from, rather than
        #: taking its summary on trust.
        "reviews": session.reviews,
        #: What happened when the winning parameters were perturbed. The operator
        #: should see this whether or not the agent chose to mention it.
        "neighbourhood": session.neighbourhood,
        "workspace": session.workspace,
        "resumable": session.stopped_because not in {"satisfied"},
    }


def _stop_reason(
    cfg: ResearchConfig, session: ResearchSession, situation: Mapping[str, Any]
) -> str:
    """Resource boundaries, checked before spending anything.

    Checked *before* the call rather than after, so the session never overshoots
    a budget it was given and never ends mid-experiment with a half-recorded
    result.
    """
    if situation["minutes_remaining"] <= 0:
        return "time_limit"
    if session.spend_usd >= float(cfg.budget_usd):
        return "budget"
    if session.calls >= int(cfg.max_calls):
        return "call_limit"
    if len(session.experiments) >= int(cfg.max_experiments):
        return "experiment_limit"
    # A sentinel file rather than a signal: the session may have been launched by
    # the console, the CLI, or a scheduler, and a file is the one channel all
    # three share. Checked between turns so it stops cleanly and resumably
    # instead of tearing down mid-experiment.
    if (Path(session.workspace) / "STOP").exists():
        return "stopped_by_operator"
    return ""
