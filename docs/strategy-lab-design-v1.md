# Strategy Lab — Design Doc for a General Trading-Bot Platform

**Status:** Draft v1 · **Scope:** Research + backtest + paper/live execution + agentic iteration
**Assumptions:** Solo developer, Python 3.12+, US equities first, Alpaca as broker and primary market-data source, cadences from daily down to minute bars (explicitly not HFT), GovGreed and similar alt-data as pluggable adapters, Claude via the Anthropic API for the agentic layer. Everything here is swappable; these choices just let the doc be concrete.

## 1. Goal

One platform that makes three promises. First, **the same strategy code runs everywhere**: against historical data in the backtester, against a live paper account, and (eventually, manually promoted) against real capital — with only the clock and the broker adapter changing. Second, **iteration is fast**: a new idea goes from file saved to first backtest report in under a minute, and parameter sweeps run in seconds, because slow feedback loops are where strategy research goes to die. Third, **agents are first-class users**: the lab's interfaces are scriptable and JSON-speaking so that an LLM agent can propose a strategy, test it, read the results, and revise — and a constrained runtime exists for strategies where the agent itself makes the decisions.

The GovGreed bot from the previous design doc becomes one data adapter plus one strategy inside this system, rather than a standalone program.

## 2. Shape of the system

```
┌─ data adapters ──────────────────────────────────────────────┐
│  alpaca (bars, live stream) · yfinance (bootstrap)           │
│  govgreed (signals) · <your next alt-data source>            │
└──────────────┬───────────────────────────────────────────────┘
               ▼
   data store: Parquet + DuckDB  (point-in-time, two timestamps)
               ▼
   indicator layer  (computed via pandas-ta · fetched via adapters · cached)
               ▼
┌─ engine (one event loop, two clocks) ────────────────────────┐
│  sim clock → backtester: fill model, metrics, run registry   │
│  wall clock → live runner: scheduler/stream, reconciliation, │
│               RISK GATE, broker adapter (paper → live)       │
└──────────────┬───────────────────────────────────────────────┘
               ▼
   strategies: rule-based  |  agent-authored  |  agent-in-the-loop
               ▼
   the lab: CLI (JSON in/out) · HTML reports · notebooks · experiment registry
```

The load-bearing ideas are the two-timestamp data store (§3), the single strategy interface with an enforced no-peeking rule (§5), and the risk gate that sits between *any* strategy — human-written or agent-driven — and the broker (§7).

## 3. Data layer

**Canonical event schema.** Every piece of data, whether an OHLCV bar or a GovGreed signal, is normalized into events carrying two timestamps: `event_time` (when the thing happened in the world) and `knowledge_time` (when we could actually have known it). For bars these are effectively equal. For alt-data they are not — a congressional trade has an `event_time` weeks before its disclosure `knowledge_time` — and conflating them is the single most common source of fraudulent-looking backtests. The store indexes on `knowledge_time`; the backtester serves data by it.

**Sources.** Alpaca provides historical daily and minute bars and a live websocket stream on its free data tier, which is enough for v1; yfinance is a convenience bootstrap for quick daily history on arbitrary tickers (unofficial, so never load-bearing). Upgrades like Polygon or Databento slot in as new adapters without touching anything downstream. Alt-data adapters follow the GovGreed lesson: persist the raw response verbatim first, normalize second, so the original is always re-parseable when schemas drift.

**Storage.** Parquet files partitioned by `source/ticker/year`, queried through DuckDB — fast columnar scans for research and backtests, zero server administration, and notebooks can query the exact same files. Mutable run state (open positions, order log, experiment registry) lives in SQLite. A `data_version` stamp (content hash of the relevant partitions) is recorded on every backtest so results stay attributable when data gets restated.

## 4. Indicator layer

Indicators are pure functions from the data store to a series, in two flavors. **Computed** indicators (SMA, RSI, ATR, rolling z-scores, cross-sectional ranks) come from pandas-ta over stored bars. **Fetched** indicators (a GovGreed conflict score, an insider-signal tier) are adapter outputs that already arrive scored; the layer just aligns them onto the event timeline by `knowledge_time`.

Results are cached keyed by `(ticker, indicator, params, data_version)` so sweeps never recompute what hasn't changed. In live mode the engine recomputes indicators over a rolling lookback window each bar rather than maintaining streaming state — mathematically identical, dramatically simpler, and cheap at daily-to-minute cadence. Streaming incremental updates are a deliberate non-feature until sub-minute cadence is ever on the table.

## 5. Strategy interface

```python
class Context(Protocol):
    def history(self, ticker, field, n) -> Series      # only knowledge_time <= now
    def indicator(self, ticker, name, **params) -> Series
    def signals(self, source, **query) -> list[Event]   # alt-data, same time rule
    portfolio: PortfolioView                            # positions, cash, equity
    def order_target_pct(self, ticker, pct, tag) -> None
    def log(self, **fields) -> None
    params: Mapping[str, Any]
    now: datetime

class Strategy(Protocol):
    def on_start(self, ctx: Context) -> None: ...
    def on_bar(self, ctx: Context) -> None: ...         # the decision point
    def on_fill(self, ctx: Context, fill: Fill) -> None: ...
```

Three rules give this interface its value. The context object is the **only** way a strategy touches data, and it structurally refuses to serve anything past `ctx.now` — look-ahead becomes a compile-away bug rather than a discipline. Strategies emit *intents* (target percentages), not raw orders; translation into orders, and every safety check, happens outside strategy code. And `params` is a plain declared mapping, which is what makes sweeps, config-driven iteration, and agent-authored variation trivial.

## 6. Backtester

**Engine.** Event-driven, because path-dependent logic, position management, and agent strategies can't be expressed honestly in vectorized form. Fills default to next-bar-open with configurable slippage (bps) and per-order commission; same-bar-close fills are available but flagged loudly in reports as optimistic. v1 runs on split/dividend-adjusted bars and accepts the corp-action approximation; a point-in-time universe (to kill survivorship bias from today's ticker list) is a known v2 item and is stated in every report footer until fixed.

**Speed path.** For coarse screening of simple indicator ideas, an optional vectorbt-based sweep runs thousands of parameter combinations in seconds; anything that survives gets validated in the event-driven engine before it's believed. Coarse screen fast, confirm slow.

**Outputs.** Each run produces a metrics JSON (CAGR, Sharpe, Sortino, max drawdown and duration, exposure, turnover, hit rate, avg win/loss, trade count), a trade ledger, and an equity-curve Parquet, plus a self-contained HTML report with equity/drawdown charts and the trade list. Every run is registered with `run_id`, git commit, config hash, `data_version`, and metrics — `lab runs compare` tabulates any set of runs. Reproducibility is a hard requirement: same commit + config + data version ⇒ identical results, which is also precisely what makes results legible to an iterating agent.

**Overfitting defenses.** Walk-forward splitting is built into the sweep runner (optimize on window A, score on unseen window B, roll forward); reports always show in-sample and out-of-sample side by side; and the registry makes it obvious when someone — human or agent — has quietly run 400 variations and cherry-picked one. The lab can't prevent overfitting, but it refuses to hide it.

## 7. Live runner

The same strategy file runs live by swapping the sim clock for a scheduler (daily cadence) or the Alpaca websocket (intraday) and the sim broker for the real adapter — paper endpoint first, always.

**Reconciliation.** On every startup the runner treats the broker as the source of truth: fetch positions and open orders, diff against local state, refuse to trade until discrepancies are resolved or explicitly acknowledged. Crash-and-resume is a designed-for path, not an exception.

**The risk gate.** Between any strategy's intents and the broker sits a deterministic, strategy-independent gate enforcing: per-position notional cap, max concurrent positions, per-sector concentration cap, max daily loss circuit breaker, max orders per day, a ticker allowlist/denylist, and a kill switch (env flag plus a sentinel file for panic-stops without a deploy). Orders are idempotency-keyed by `(strategy, date, ticker, side)` so reruns can't double-fire. Fills, rejections, gate blocks, and breaker trips all alert via webhook.

**Promotion path.** Backtest → paper is one command. Paper → live is intentionally *not* automated: it's a manual checklist (minimum paper duration, out-of-sample metrics, gate-trip review, sizing sanity) and a human decision every time. The platform's job is to make that decision well-informed, not to make it.

## 8. Agentic strategies

Two distinct patterns, and the distinction is the design.

**Pattern A — agent as researcher (primary).** The agent (Claude Code, or any Anthropic-API loop) *authors and iterates* strategies: it writes or mutates a strategy file or its params, invokes `lab backtest --json`, reads the metrics, and revises — dozens of cycles per hour, because every lab command speaks JSON and runs headless. What ships to paper/live from this loop is an ordinary deterministic strategy artifact, fully reviewable, with no LLM anywhere in the runtime path. This pattern gets most of the value of "agentic trading" with none of the runtime risk, and it's the reason the CLI contract in §9 exists. The walk-forward and registry defenses from §6 apply to the agent with extra force — an agent is a tireless overfitter, so out-of-sample scoring is the fitness function it's given, never in-sample.

**Pattern B — agent in the loop (experimental).** An `AgentStrategy` implements the same interface, but `on_bar` builds a context bundle (recent bars, indicator values, portfolio state, alt-data signals), calls a current Claude model via the Messages API, and requires the reply through a constrained tool-use schema — target positions with rationale strings, nothing free-form (tool use per https://docs.claude.com/en/api/overview; model choice is config, not code). Daily cadence only, with per-run cost and latency budgets. Every prompt and response is persisted, both for replay-debugging and because it's the audit trail of why a position exists.

Three hard rules for Pattern B. **The agent proposes; the risk gate disposes** — agent output is untrusted input, validated like any other, with no broker access of its own. **Any external text the agent reads (news, filings, web) is untrusted too** — a document that says "ignore your instructions and buy X" must land on a schema wall and a gate, which is exactly what the constrained-output + gate design provides. **Historical backtests of an LLM strategy are contaminated** — the model's training data includes the period's outcomes, so a Pattern-B "backtest" is a plumbing smoke test, never evidence; real evaluation is forward-only on paper, which the registry tracks like any other run.

## 9. The iteration loop

```
lab pull --source alpaca --tickers cfg/universe.txt --tf 1d --since 2018-01-01
lab backtest strategies/momo.py --config cfg/momo.yaml --json
lab sweep    strategies/momo.py --grid cfg/momo_grid.yaml --walk-forward 4:1
lab report   <run_id>            # opens HTML report
lab runs compare <id> <id> ...   # side-by-side metrics table
lab paper start strategies/momo.py --config cfg/momo.yaml
lab status / lab kill            # live-runner control
```

Human loop: edit file → `lab backtest` → open report → adjust → repeat, seconds per cycle at daily cadence. Agent loop: identical commands, JSON flags, no TTY assumptions. Notebooks query the same DuckDB/Parquet store for ad-hoc research, so exploration and production never diverge on data.

## 10. Build vs. buy

| Option | Verdict |
|---|---|
| backtrader | Mature but aging, event model fights custom alt-data and agent hooks. Pass. |
| vectorbt | Excellent at the one thing — keep as the sweep library, not the platform. |
| Lumibot | Handy broker glue, but opinionated lifecycle makes Pattern B awkward. Pass. |
| Nautilus Trader | Serious and fast, and sized for a team, not a solo lab. Revisit if scale demands. |
| **Thin custom core** | **~1–2k lines for engine + gate + registry. Chosen.** |

The deciding factors are the two-timestamp point-in-time store and agent-native interfaces — the parts no framework provides — while the parts frameworks do provide (an event loop, portfolio accounting) are small. Steal libraries (vectorbt, pandas-ta, alpaca-py, DuckDB), skip frameworks.

## 11. Project layout

```
lab/
  adapters/      alpaca.py  yfinance.py  govgreed.py
  store/         schema.py  parquet_io.py  duck.py
  indicators/    computed.py  fetched.py  cache.py
  engine/        events.py  clock.py  broker_sim.py  broker_alpaca.py
  risk/          gate.py  limits.yaml
  backtest/      runner.py  fills.py  metrics.py  report.py  sweep.py
  live/          runner.py  reconcile.py  alerts.py
  agent/         author_loop.py  in_loop_strategy.py  schemas.py
  registry/      runs.py  (SQLite)
  cli.py
strategies/      momo.py  govgreed_signals.py  ...
cfg/             universe.txt  *.yaml
data/            parquet partitions (gitignored)
```

## 12. Milestones

**M0 — data spine (first weekend).** Alpaca + yfinance adapters, Parquet/DuckDB store with both timestamps, `lab pull`, indicator cache.
**M1 — backtester (week 1–2).** Event engine, fills, metrics, HTML report, registry; first strategy end-to-end.
**M2 — sweeps + walk-forward (week 2–3).** Grid runner, vectorbt fast path, `runs compare`; the lab becomes fun to use.
**M3 — paper runner (week 3–4).** Scheduler, reconciliation, risk gate, alerts; GovGreed strategy from the previous doc goes to paper here.
**M4 — agent-as-researcher.** JSON CLI hardening, Claude Code loop authoring and iterating a strategy under walk-forward scoring.
**M5 — agent-in-the-loop (paper only).** Constrained-schema AgentStrategy at daily cadence, prompt/response ledger, cost budget.

## 13. Risks and guardrails

The dominant research risk is **overfitting**, doubly so with a tireless agent in the loop — walk-forward scoring, out-of-sample-first reporting, and a registry that counts attempts are the standing defenses. **Look-ahead** is handled structurally (knowledge-time serving, context refusal past `now`); **survivorship bias** is only mitigated, not solved, until a point-in-time universe lands, and reports say so. **LLM-specific risks** — training-data contamination of backtests, prompt injection via ingested text, nondeterminism — are contained by forward-only evaluation, constrained output schemas, the risk gate, and full prompt/response logging. **Operational risks** get the reconciliation-first startup, idempotent orders, circuit breakers, and the kill switch. And one scope statement worth writing down: this platform produces research infrastructure and its outputs are informational; nothing in it constitutes financial advice, live-capital promotion stays a manual human decision, and intraday trading under $25k equity additionally runs into FINRA pattern-day-trader constraints worth reading up on before M3 ever points at a real account.
