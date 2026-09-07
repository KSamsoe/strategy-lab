# Strategy Lab

A quantitative research lab that a desktop agent — Claude Code, Cursor, Cherry
Studio, OpenClaw, anything that speaks MCP — can drive end to end: pull data,
measure it, write a strategy, backtest it, and then try to prove the result is
luck. Every number the lab hands back arrives with the fact that would embarrass
it: the benchmark over the *same* bars, the exposure, how much the risk gate
rewrote, the error bar on the score, and a `warnings` list that says in words
what the headline leaves out.

It exists because five research sessions on an earlier version of this platform
each reached a confident wrong conclusion, and in every case the correcting fact
was computable at the moment the result was returned. So now it is returned.

> **Not financial advice, and not a strategy.** This is research infrastructure.
> Its own conclusion about the strategies found with it is that none has an
> edge that survives the permutation null as a *signal* — the ones that beat the
> index do so through construction (diversification and sizing), and the lab
> tells you that unprompted. Promotion to real capital is a manual human
> decision the platform deliberately does not automate.

## Five minutes, no keys

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[data,mcp,dev]"   # Windows
# . .venv/bin/activate && pip install -e ".[data,mcp,dev]"        # macOS / Linux
lab demo
```

`lab demo` seeds a deterministic synthetic tape (bars, a benchmark, and alt-data
events with a realistic disclosure lag), runs two backtests, and prints where
the reports landed. The numbers describe the generator, not a market; the
plumbing they exercise is the real thing.

Then hand the lab to your agent:

```bash
cp .mcp.json.example .mcp.json     # Claude Code picks this up on next launch
python scripts/seed_findings.py    # the ledger's method lessons, so the agent starts informed
```

and ask it to run the **`audit_run`** prompt on the momentum run with
`cfg/demo.yaml`. It will read the ledger, review attribution and gate activity,
compute the benchmark over the same bars, perturb the parameters, run a paired
bootstrap and a permutation null, deflate the Sharpe by the number of trials the
registry has seen, and write down what it concluded — and each step's answer
comes with what that answer means.

Real data needs keys in `.env` (copy `.env.example`); `yfinance` works without any:

```bash
lab pull --source yfinance --tickers cfg/universe.txt --tf 1d --since 2015-01-01
lab backtest strategies/momo.py --config cfg/momo.yaml
```

## What the agent gets

37 tools, 3 resources and 2 prompts, documented in [`docs/mcp.md`](docs/mcp.md).
The shape of the toolset is the argument:

| Verb | Tools | Why it is here |
|---|---|---|
| **Measure first** | `data_coverage` `data_quality` `signal_scan` `conditional_returns` `correlation_matrix` `data_pull` | The most valuable findings came from measuring the data, not from more backtests. |
| **Run** | `backtest` `review` `market_reference` `runs_list` `runs_compare` | Every result carries its benchmark, exposure, gate interference and error bar. |
| **Author** | `strategy_api` `strategy_write` `strategy_read` `validate_strategy` `strategies_list` | Source is loaded through the engine before it is written; clients without a filesystem can still author. |
| **Isolate** | `ablate` `compare_universes` `regimes` `cost_sensitivity` | A strategy's edge is very often its ticker list. This is the call that finds out. |
| **Is it luck?** | `walk_forward` `bootstrap_vs_benchmark` `bootstrap` `permutation_test` `deflated_sharpe` `neighbourhood` | Paired against the benchmark; a null that keeps the construction and destroys the signal; trials counted from the registry, never from memory. |
| **On money** | `live_status` `live_vs_backtest` | A paper book compared to a backtest of the same window, session by session. |
| **Remember** | `findings_record` `findings_search` `findings_update` | What was *learned*, with evidence and status, so the next session starts informed instead of rediscovering a negative result at full price. |
| **Long work** | `job_start` `job_status` `job_result` `jobs_list` `job_cancel` | Detached workers whose results outlive the client, the server and the conversation. |
| **Ship** | `promote` | Copies the *archived* source that produced a run, verified against its hash — never the file that happens to be on disk. |

`lab-mcp --http --host 0.0.0.0 --port 8765` serves the same toolset over
streamable HTTP for an agent on another machine.

## What makes a result trustworthy here

- **Two timestamps.** Every datum has `event_time` (when it happened) and
  `knowledge_time` (when you could have known it). The store filters on the
  second before anything else. A congressional trade is invisible until its
  disclosure date. This is a property of the storage layer, not of any strategy.
- **The gate reports what it did.** A deterministic risk gate sits between the
  strategy's intents and the book, and every result says what share of intents
  it clipped or blocked — because a heavily clipped strategy is guaranteed to
  look robust to parameter changes, and the flatness belongs to the limit.
- **The benchmark shares the window.** From the first *tradeable* bar, warmup
  excluded, full period and out-of-sample both. A benchmark measured over a
  different window than the strategy is the platform's most-repeated bug class,
  and it is now computed in one place.
- **Every score has an error bar.** A holdout of ~230 daily bars carries a
  Sharpe standard error near 1.0; the toolset says so next to the number, and
  the neighbourhood, bootstrap and permutation tools replace the approximation
  with measurement.
- **Trials are counted, not recalled.** `deflated_sharpe` reads the registry:
  every backtest over the same holdout, under any strategy name.
- **Runs archive their source.** `promote` ships the code that produced a run,
  hash-verified, not whatever an agent has since rewritten into that file.
- **The thresholds live in one file.** `lab/analysis/thresholds.py` — each
  constant with the story of the mistake it encodes.

## Limits, stated plainly

- Daily-bar US equities is the well-trodden path. Intraday bars work and are
  tested, but the finding there was that costs kill the ideas before the
  signal does.
- Universes are today's lists: survivorship bias is real and every report
  footer says so. A point-in-time universe is not built.
- Data sources: `yfinance` (no keys, reaches 2005), Alpaca (keys, ~1,500 bars
  per symbol by row count), a GovGreed alt-data adapter, and a synthetic
  adapter for tests and the demo.
- One live broker (Alpaca), paper by default, and a two-part opt-in for real
  money that the platform itself never exercises.
- Jobs run on the machine the server runs on. One server, one queue.
- The strategies in `strategies/` are examples and negative results, not
  products. Read the findings ledger before drawing conclusions from them.

## Layout

```
lab/
  store/      Parquet + DuckDB, two timestamps, verbatim raw snapshots
  engine/     events · clock · context (the look-ahead barrier) · brokers
  risk/       the gate
  backtest/   runner · fills · metrics · report · sweep · walk-forward
  analysis/   windowed metrics, the benchmark, error bars, the neighbourhood check, thresholds
  mcp/        the toolset: server · tools_* · jobs · prompts
  registry/   runs, decisions, events, findings (SQLite)
  live/       paper/live runner · reconcile · alerts · kill switch
  adapters/   yfinance · alpaca · govgreed · synthetic
  indicators/ pure pandas/numpy, causal by construction, cached
  agent/      the earlier self-driving research loop (see below)
  api/        FastAPI read layer behind the console
  cli.py      `lab` — every command speaks --json
strategies/   examples: momo · buy_and_hold · blend_tilt · core20_vt · tilt_rp · govgreed_signals
cfg/          configs, universes, research briefs
console/      React + TypeScript console (decision tape, live status)
docs/         mcp.md · guide.md · CONTRACTS.md · the design docs
```

## The earlier agent loop

`lab/agent` is a self-driving research loop that makes its own model calls —
provider abstraction, prompt caching, budget metering, resume, a call ledger.
It still works, and it is still the right tool for an unattended overnight
session, because it runs while nobody is watching. It is no longer the primary
interface: a desktop agent already has a model, a context window and a harness,
and about 4,600 lines of this package re-implemented that. The MCP toolset is
the same engine offered to the agent you already have.
[`docs/agents.md`](docs/agents.md) covers it.

## Tests

```bash
./.venv/Scripts/python.exe -m pip install -e ".[data,mcp,agent,dev]"   # `agent` for the loop's own tests
./.venv/Scripts/python.exe -m pytest -q          # ~930 tests, about two minutes
cd console && npm run test && npm run typecheck
```

The suite includes guards that fail the build if a prompt names a tool or an
argument the server does not have, if the MCP layer imports a private name
from another package, or if a benchmark is computed over a different window
than the strategy it is compared to.

## License

MIT. See [`LICENSE`](LICENSE).
