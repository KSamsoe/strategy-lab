# Strategy Lab — the user guide

Everything you need to go from a clean checkout to a strategy running on paper.
Read [`agents.md`](agents.md) for the agentic layer and
[`CONTRACTS.md`](CONTRACTS.md) if you are changing the code rather than using it.

**Contents**

1. [Setup](#1-setup)
2. [First run, no credentials](#2-first-run-no-credentials)
3. [Getting real data](#3-getting-real-data)
4. [The loop](#4-the-loop)
5. [Writing a strategy](#5-writing-a-strategy)
6. [Configs](#6-configs)
7. [Reading results](#7-reading-results)
8. [Sweeps and walk-forward](#8-sweeps-and-walk-forward)
9. [The risk gate](#9-the-risk-gate)
10. [The console](#10-the-console)
11. [Paper trading](#11-paper-trading)
12. [GovGreed](#12-govgreed)
13. [Command reference](#13-command-reference)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Setup

Python 3.12+. From the repo root:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[data,agent,dev]"
```

On macOS/Linux the interpreter is `.venv/bin/python`. Everything below writes
`lab` for the console script; `python -m lab.cli` is identical if the script is
not on your PATH.

Copy `.env.example` to `.env` and fill in only what you need — every credential
is optional and the lab tells you what it cannot do without one:

```bash
lab adapters
```

```
alpaca      unavailable   no ALPACA_API_KEY_ID       provides bars
govgreed    unavailable   no GOVGREED_API_KEY        provides signals
synthetic   available     ready                      provides bars,signals
yfinance    available     ready                      provides bars
```

The console needs a build once:

```bash
cd console && npm install && npm run build
```

## 2. First run, no credentials

```bash
lab demo
```

Seeds synthetic data, runs two backtests, renders two HTML reports, prints where
they landed. ~16 seconds. This is the proof the whole path works before you
introduce a single API key.

> The numbers are meaningless — the synthetic adapter generates near-random
> walks. Momentum *losing* on it is the correct result. It exists to exercise
> the plumbing and to make the look-ahead barrier observable.

## 3. Getting real data

```bash
lab pull --source alpaca --tickers cfg/universe.txt --tf 1d --since 2018-01-01
lab data coverage
lab data version
```

`yfinance` needs no key and is fine for a quick look; it is unofficial, so never
build anything load-bearing on it. `alpaca` needs `ALPACA_API_KEY_ID` and
`ALPACA_API_SECRET_KEY` (the free data tier is enough).

Everything lands in `data/parquet/<table>/source=…/ticker=…/year=…/`, and every
row carries **two timestamps**:

- `event_time` — when the thing happened
- `knowledge_time` — when you could have known it

For bars these match. For alt-data they do not, and that gap is the single most
important thing in the system. `lab data version` prints the content digest
stamped on every run that reads this slice, so a result stays attributable when
data gets restated.

## 4. The loop

```bash
lab backtest strategies/momo.py --config cfg/momo.yaml     # run it
lab report <run_id> --open                                 # look at it
lab runs list                                              # what have I tried
lab runs compare <id> <id>                                 # side by side
```

Add `--json` to any command and stdout carries exactly one JSON object and
nothing else — logs go to stderr. That is what makes the whole CLI scriptable
and is the contract the agent layer depends on.

Override a single parameter without editing a file:

```bash
lab backtest strategies/momo.py --config cfg/momo.yaml -p lookback=200 -p top_n=6
```

## 5. Writing a strategy

A strategy is a plain Python module. No base class, no registration.

```python
NAME = "my_strategy"

PARAMS = {"lookback": 50, "top_n": 3}      # declared, so sweeps can vary them

class Strategy:
    def __init__(self, params=None):
        self.params = params or {}

    def on_bar(self, ctx):                  # the decision point
        for ticker in ctx.universe:
            closes = ctx.history(ticker, "close", ctx.params["lookback"])
            if len(closes) < ctx.params["lookback"]:
                continue
            if closes.iloc[-1] > closes.mean():
                ctx.order_target_pct(ticker, 0.1, tag="entry", reason="above mean")
            else:
                ctx.close(ticker, reason="below mean")
```

The loader looks for, in order: a `STRATEGY` object, a `build(params)` factory, a
class named `Strategy`, then the single class defining `on_bar`.

### What `ctx` gives you

| Call | Returns |
|---|---|
| `ctx.history(ticker, field, n)` | Series of the last `n` values, oldest first |
| `ctx.bars(ticker, n)` | OHLCV frame for the same slice |
| `ctx.price(ticker)` | Latest known close, or `None` |
| `ctx.indicator(ticker, name, **params)` | A computed indicator series |
| `ctx.signals(source, **query)` | Alt-data events, filtered by `knowledge_time` |
| `ctx.portfolio` | `cash`, `equity`, `positions`, `weight(t)` |
| `ctx.order_target_pct(t, pct, tag, reason)` | Express a target weight |
| `ctx.close(t, reason)` | Shorthand for target 0 |
| `ctx.log(**fields)` | Structured log into the decision journal |
| `ctx.now`, `ctx.session`, `ctx.universe`, `ctx.params` | Context |

Three rules make this worth using:

**You cannot see the future.** Every read filters on `knowledge_time <= ctx.now`.
Not by convention — structurally, because there is no other door to the data.
Indicators are computed once over the full series and sliced, which is sound
*only* because every indicator is causal, so the engine proves that per
indicator on first use and raises `LookAheadError` if one peeks.

**You emit intents, not orders.** `order_target_pct` records a desire. Turning a
percentage into shares, and every safety check, happens outside your code. That
is why the same file runs in the backtester and against a live broker unchanged.

**`params` is plain data.** Which is what makes sweeps and agent-authored
variation a config change rather than an edit.

Available indicators: `sma ema wma rsi atr bbands macd roc momentum zscore
rolling_vol donchian returns vwap adx stoch max_drawdown slope`.

```bash
lab strategies      # every loadable strategy and its declared params
```

## 6. Configs

A run is defined by a YAML file. Unknown keys are rejected loudly — a typo
should not silently backtest something other than what you meant.

```yaml
strategy: strategies/momo.py
tickers: cfg/universe.txt      # a path reads a universe file; a list also works
timeframe: 1d
start: 2018-01-01
end: 2026-01-01
cash: 100000
benchmark: SPY                 # gives you alpha/beta and the console overlay
warmup: 210                    # bars to skip before the first decision
sources: [govgreed]            # alt-data ctx.signals() may read. Omit and it sees nothing.

params:                        # overrides the strategy's PARAMS
  lookback: 126
  top_n: 4

fills:
  mode: next_open              # next_open | next_close | same_close
  slippage_bps: 5
  commission_per_order: 0.0
  partial_fill_volume_pct: 0.05

limits:                        # the risk gate, see §9
  max_position_pct: 0.30
  max_positions: 5
```

**`warmup` matters.** Without it the opening trades are made on half-formed
moving averages. Set it to at least your longest indicator window.

**`same_close` is optimistic.** It lets a decision trade at a price its own
signal helped set. It is available, and it is flagged loudly in every report and
in the console when you use it.

## 7. Reading results

```bash
lab report <run_id> --open
```

One self-contained HTML file — no CDN, no network. Equity and drawdown with
out-of-sample spans shaded, the trade ledger, the metrics table, and a
provenance footer carrying git commit, config hash, `data_version` and the fill
model.

Every run writes `runs/<run_id>/`:

| File | What |
|---|---|
| `metrics.json` | Every metric, JSON-safe |
| `equity.parquet` | Equity, drawdown, exposure per bar |
| `trades.csv` | The trade ledger |
| `decisions.jsonl` | One record per `on_bar`: the full input tape |
| `config.json`, `provenance.json` | What produced this |
| `report.html` | The rendered report |

### Metrics worth understanding

`total_return`, `cagr`, `sharpe`, `sortino`, `calmar`, `max_drawdown`,
`max_drawdown_duration_days` (peak to recovery), `exposure`, `turnover`,
`trades`, `hit_rate`, `profit_factor`, `avg_win`/`avg_loss`.

Three that exist to keep the rest honest:

- **`ledger_residual`** — equity gain minus trade P&L minus open unrealized. It
  must be zero. Anything else means money moved that no trade explains, and a
  loud warning is attached to the run.
- **`optimistic_fills`** — same-bar-close fills were used.
- **`contaminated`** — an LLM strategy was backtested inside its own training
  window. Not evidence.

Artifacts are **immutable**. If the engine changes, the way to see corrected
numbers is a new run:

```bash
lab runs rerun <run_id>      # reproduce from its own stored config
lab runs rerun --stale       # every run predating the trade-ledger fix
```

## 8. Sweeps and walk-forward

```bash
lab sweep strategies/momo.py --grid cfg/momo_grid.yaml --walk-forward 4:1
```

```yaml
# cfg/momo_grid.yaml
base: cfg/momo.yaml
grid:
  lookback: [63, 126, 189, 252]
  top_n: [3, 4, 6]
metric: sharpe
walk_forward: "4:1"      # train on 4 blocks, test on the next, roll forward
max_runs: 200
```

**Out-of-sample is the fitness function.** Ranking on an in-sample metric still
works but the result is marked `ranked_on: in_sample` so nothing downstream can
present it as validation. Every row reports in-sample and out-of-sample side by
side, so the *gap* is visible — that gap is the honest measure of how much of a
result is fitting.

The registry counts attempts per strategy family. That number is deliberately
hard to miss: it is what makes "someone quietly ran 400 variations and
cherry-picked one" impossible to hide.

A capped grid records the truncation rather than silently reporting the subset.

## 9. The risk gate

One deterministic gate sits between *any* strategy — yours or an agent's — and
the broker. Configure it under `limits:`.

| Rule | Controls |
|---|---|
| `max_position_pct` | Largest single position, as a fraction of equity |
| `max_position_notional` | …or as a dollar cap |
| `max_positions` | Concurrent positions |
| `max_sector_positions`, `max_sector_pct` | Concentration, using the `sectors:` map |
| `max_gross_exposure` | Total book size |
| `max_daily_loss_pct` | Circuit breaker |
| `max_orders_per_day` | Order-rate cap |
| `min_order_notional` | Refuses dust |
| `allowlist` / `denylist` | Tickers |
| `cooldown_days` | Blocks re-entry after an exit |

Two behaviours worth knowing:

**An exit is never blocked by a capacity rule.** Only the kill switch can stop
one. A tripped breaker stops new and increasing exposure while letting the book
close — the alternative traps you in a losing position.

**A cap that can be partially honoured clips; one that cannot, blocks.** Either
way the verdict names the rule that fired, and it lands in the decision journal.

The kill switch is an env flag *or* a sentinel file, so a panic stop needs no
deploy:

```bash
lab kill                 # engage. Safe to run twice.
lab kill --release
lab status
```

## 10. The console

```bash
lab ui
```

FastAPI on `127.0.0.1:8787` serving a React SPA. It answers *what is my bot
doing, and why* — identically for a live paper account and a backtest from last
night, because both emit the same events.

| Screen | For |
|---|---|
| Fleet | Is everything okay: health, staleness, quota, strategy cards, live tape |
| Runs | The registry, filterable, with origin tags and the attempt counter |
| Run detail | Equity, price chart with trade markers, ledger, **decision tape** |
| Compare | Normalized curves, best-per-column metrics, drawdown multiples |
| Sweep | Parameter heatmap (OOS by default), walk-forward bars, attempts |
| Agent | Research sessions, author-loop lineage, Pattern-B ledger |

The **decision tape** is the reason it exists. One row per decision, collapsed to
a terse line:

```
06:30 · saw 3 signals · intent NVDA +4.0% · gate: clipped to 3.1% (position_cap) · filled
```

Expanded, it shows the entire input tape — every indicator value, every signal
with **both** its timestamps and the lag between them, the portfolio state, the
per-intent gate verdicts, and the resulting orders. Time-locked to the charts
and the ledger: click a trade and everything moves to it.

The UI can only make the system **safer**. Pause, cancel orders and kill are
available and journaled. Starting a strategy, raising a limit, or resuming after
a breaker are not routes at all — they stay CLI actions behind a manual
checklist.

Development: `cd console && npm run dev` proxies `/api` to the lab. After
changing console code you must `npm run build` and restart `lab ui` — the server
serves a build artifact, it does not rebuild.

## 11. Paper trading

```bash
lab paper start strategies/momo.py --config cfg/momo.yaml
lab status
lab paper stop momo
```

The same strategy file, with the sim clock swapped for a scheduler and the sim
broker for Alpaca's paper endpoint.

**It reconciles first.** On startup it fetches broker positions and open orders,
diffs against local state, and refuses to trade until they agree or you
acknowledge. Crash-and-resume is designed for, not an accident.

### Running it daily, and surviving a reboot

A daily strategy does not need a process running all day. It needs one decision
per session, and two things in the right order — refresh the bars, then decide:

```bash
lab paper schedule strategies/blend_tilt.py --config cfg/blend_tilt.yaml --at 09:35
```

That writes `scripts/lab-<strategy>.cmd` and prints the `schtasks` line to
register it; add `--install` to register it for you. `--at` is **Eastern** and
the printed command converts to your local time. The generated script resolves
the universe from your config, pulls, and then calls:

```bash
lab paper start strategies/blend_tilt.py --config cfg/blend_tilt.yaml --broker alpaca --once
```

`--once` decides for the current moment and exits. Without it the runner sleeps
until the next fire time — which for a task woken *at* 09:35 means sleeping until
09:35 tomorrow, since `_next_daily_fire` treats a candidate at or before now as
belonging to the next session.

**The book survives.** Positions and cash are saved to `live_strategies.book`
after every step and restored before reconciliation on the next start. That
ordering is the whole trick: reconciling an *empty* book against a live account
makes every position look like a stranger, which is why a reboot used to mean
"refuse to trade". Restored first, the comparison asks the question worth asking
— did anything change while I was down? A clean restart now reads:

```
reconciled: 1 broker position(s), 1 open order(s), 0 diffs
```

When they genuinely disagree — a manual trade, a fill that landed while the
process was down — it still refuses, and `--adopt` is the override that says "I
have looked, the broker is right". It adopts rather than mutes.

**Missed runs catch up.** The task is registered with `StartWhenAvailable`, so a
machine that was off or asleep at 09:35 runs the decision once when it next comes
up — on freshly pulled data, because the wrapper pulls before it decides. It does
*not* wake the machine: that needs stored credentials and a wake timer, and a
trading bot that powers on your computer at dawn is a bigger promise than a daily
rebalance needs to make.

**It settles before it exits.** A one-shot run submits orders and would otherwise
save the book a moment before those orders fill — recording intent, not outcome.
The next morning's startup would then find a book that disagrees with the account
and refuse, needing a human every single day. So `--settle-seconds` (default 90)
waits for the orders to resolve, and the book written at exit is taken from the
broker's own positions and cash rather than from the local guess. Positions
matching while cash does not is still a wrong book, and equity is what sizes the
next session's orders.

**A blocked run exits 3, not 0.** Refusing to trade is a correct outcome but not
a successful one, and the scheduler is the only thing watching at 09:35.

**Stale data is refused, not just reported.** `--max-stale-sessions` (default 2)
blocks the decision when the newest bar is too old, counted in *trading sessions*
so a Friday bar is fresh on Monday morning. Bar age was previously reported in
the heartbeat and enforced nowhere, which is indistinguishable from fresh data
right up until the day it costs money. `0` disables the guard.

Orders are idempotency-keyed on `(strategy, session_date, ticker, side)`, so a
rerun cannot double-fire. Fills, rejections, gate blocks, breaker trips and
reconciliation diffs all alert via webhook and land in the event journal.

**Paper → live is deliberately not automated.** It is a manual checklist and a
human decision: minimum paper duration, out-of-sample metrics, gate-trip review,
sizing sanity. Live trading additionally requires `allow_live=True` *and*
`LAB_ALLOW_LIVE_TRADING` in the environment.

## 12. GovGreed

The adapter talks to the official REST API (`/api/v1`, bearer token from
`GOVGREED_API_KEY`).

```bash
lab govgreed pull      # one daily snapshot, inside the call budget
lab govgreed status    # quota, and how much history you have accumulated
```

Two things shape it. The free tier is a small, moving call budget, so the client
reads `X-RateLimit-*` headers and the `meta.quota` envelope rather than
hardcoding a number, self-throttles to ~2 req/sec, and degrades gracefully when
quota runs out mid-run. And **historical backfill is not available**, so every
response is persisted verbatim before it is parsed — those daily snapshots are
the only signal history that will ever exist for backtesting.

Treat the numbers sceptically: the vendor's published win rates are unaudited,
disclosure is delayed by design, and the same feed sells to competitors. The
forward paper log is the only performance number worth believing.

## 13. Command reference

```
lab demo                                  zero-credential end-to-end proof
lab pull --source S --tickers F --tf 1d --since D
lab backtest <strategy> --config C [-p k=v] [--json]
lab sweep <strategy> --grid G [--walk-forward 4:1] [--json]
lab report <run_id> [--open]
lab runs list|show|compare|rerun|delete
lab runs rerun --stale                    reproduce runs predating the ledger fix
lab data coverage|version
lab adapters                              what can fetch, and why not
lab strategies                            what is loadable, and its params
lab status                                kill switch, live strategies, heartbeats
lab kill [--release]
lab paper start [--once] [--adopt] [--max-stale-sessions N]
lab paper schedule <strategy> -c C --at 09:35 [--install]   daily task, Windows
lab paper stop <strategy>
lab govgreed pull|status
lab strategy promote <run_id>             copy a run's exact code into strategies/
lab agent research ... --neighbourhood-runs N   perturb the winner before finishing
lab agent research -c C --brief "..."     open-ended session (see agents.md)
lab agent author -s S -c C                iterate one strategy
lab agent providers [--probe]             which model backends work
lab agent status
lab ui [--port 8787] [--no-open]
```

Exit codes: `0` ok, `1` error, `2` bad usage, `3` gate or kill-switch refusal.

## 14. Troubleshooting

**"no bars for this universe/timeframe/date range"** — the store is empty for
what you asked. `lab data coverage` shows what you actually have.

**A strategy makes no trades.** Usually `warmup` is shorter than the longest
indicator window, or the config's `sources:` does not list the feed the strategy
reads, so `ctx.signals()` returns nothing. `lab strategies` shows declared
params; the decision tape shows what the context actually served.

**`LookAheadError`** — an indicator is reading forward. This is the engine
catching a real bug, not a false positive.

**The console shows old numbers.** Backtest artifacts are immutable by design.
`lab ui` serves a *built* SPA, so console changes need `npm run build` first, and
Python changes need the server restarted. Neither regenerates stored run data —
that is `lab runs rerun`.

**Tests are slow or spend model quota.** They should not: the suite is hermetic,
scrubbing credentials and pinning the agent backend to one with no key. If a
test reaches the network, that is a bug worth reporting.

**A run shows a huge return but almost no trades.** Check `ledger_residual` and
`open_positions`. Runs predating the trade-ledger fix counted only round trips
that returned exactly to flat; the console flags them and `lab runs rerun`
regenerates them.
