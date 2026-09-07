# Strategy Lab

A backtesting and research toolkit that an AI coding agent can operate through
[MCP](https://modelcontextprotocol.io). Works with Claude Code, Cursor, Cherry
Studio, OpenClaw, or any other MCP client.

The agent can pull data, measure it, write a strategy, backtest it, and then
run the statistical checks that tell you whether the result is real or luck.
Every result includes the context needed to judge it: the benchmark over the
same bars, the exposure, how much the risk limits changed the strategy, the
error bar on the score, and a list of plain-English warnings.

> **Not financial advice.** This is research infrastructure, not a trading
> strategy. The example strategies included here are mostly negative results.
> Nothing in the platform promotes a strategy to real money automatically.

## Quick start (no API keys needed)

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[data,mcp,dev]"   # Windows
# . .venv/bin/activate && pip install -e ".[data,mcp,dev]"        # macOS / Linux
lab demo
```

`lab demo` generates synthetic price data, runs two backtests, and writes HTML
reports. The numbers are meaningless (it's synthetic data), but it proves the
install works.

To connect your agent:

```bash
cp .mcp.json.example .mcp.json     # Claude Code reads this on launch
python scripts/seed_findings.py    # loads the findings ledger with some method notes
```

Then ask the agent to run the `audit_run` prompt on the momentum run using
`cfg/demo.yaml`. It will walk through the full validation checklist and record
what it found.

For real data, `yfinance` needs no keys:

```bash
lab pull --source yfinance --tickers cfg/universe.txt --tf 1d --since 2015-01-01
lab backtest strategies/momo.py --config cfg/momo.yaml
```

Alpaca and other sources need keys in `.env` (copy `.env.example`).

## What the agent gets

37 tools, 3 resources and 2 prompts. Full reference in [`docs/mcp.md`](docs/mcp.md).

| Group | Tools | Purpose |
|---|---|---|
| **Explore** | `data_coverage` `data_quality` `signal_scan` `conditional_returns` `correlation_matrix` `data_pull` | Measure the data before writing a strategy against it. |
| **Run** | `backtest` `review` `market_reference` `runs_list` `runs_compare` | Backtest and inspect. Every result includes benchmark, exposure, gate activity and error bar. |
| **Author** | `strategy_api` `strategy_write` `strategy_read` `validate_strategy` `strategies_list` | Write and validate strategy files, including from clients that can't touch the filesystem. |
| **Isolate** | `ablate` `compare_universes` `regimes` `cost_sensitivity` | Find out what is actually producing the result. Often it's the ticker list. |
| **Validate** | `walk_forward` `bootstrap_vs_benchmark` `bootstrap` `permutation_test` `deflated_sharpe` `neighbourhood` | Statistical checks for luck, overfitting and parameter fragility. |
| **Paper trading** | `live_status` `live_vs_backtest` | Compare a paper-trading book against a backtest of the same period. |
| **Findings** | `findings_record` `findings_search` `findings_update` | A persistent ledger of conclusions, so the next session doesn't repeat work. |
| **Jobs** | `job_start` `job_status` `job_result` `jobs_list` `job_cancel` | Run slow tools in the background. Results survive disconnects and restarts. |
| **Ship** | `promote` | Copy the exact source code that produced a run into the strategy library. |

To serve the toolset to an agent on another machine:

```bash
lab-mcp --http --host 0.0.0.0 --port 8765
```

## Design principles

**Two timestamps on every record.** Each data point has an `event_time` (when
it happened) and a `knowledge_time` (when you could have known about it). The
store filters on `knowledge_time` before anything else, so a strategy can never
see data from the future. This applies to alt-data too: a congressional trade
is invisible until its disclosure date.

**The risk gate reports what it did.** A rules-based gate sits between the
strategy and the portfolio. Every result reports what fraction of orders it
clipped or blocked. This matters because a heavily clipped strategy looks
robust to parameter changes for the wrong reason.

**Benchmarks use the same window.** The benchmark is measured from the first
tradeable bar, with warmup excluded, over both the full period and the
out-of-sample window. Mismatched windows were the most common bug in earlier
versions of this project, so the calculation now lives in one place.

**Every score has an error bar.** Around 230 daily bars gives a Sharpe ratio a
standard error near 1.0. The toolset reports the error bar next to the number,
and the bootstrap and permutation tools measure it directly.

**Trial counts come from the registry.** `deflated_sharpe` counts every
backtest run against the same data, under any strategy name. Asking the user
how many things they tried produced undercounts every time.

**Runs archive their source.** `promote` copies the code that actually produced
a run, verified by hash. It does not copy whatever file currently sits at that
path.

**Thresholds live in one file.** See `lab/analysis/thresholds.py`. Each
constant has a comment explaining the mistake it prevents.

## Limitations

- Best tested on daily-bar US equities. Intraday bars work, but transaction
  costs tend to kill intraday strategies before anything else does.
- Universes are current ticker lists, so backtests carry survivorship bias.
  There is no point-in-time universe.
- Data sources: `yfinance` (free, back to 2005), Alpaca (keys required, roughly
  1,500 bars per symbol), a GovGreed alt-data adapter, and a synthetic adapter
  for tests and demos.
- One live broker (Alpaca). Paper trading by default. Real money requires a
  two-step manual opt-in.
- Background jobs run on the same machine as the server. There is no
  distributed execution.
- The included strategies are examples and negative results, not products.

## Project layout

```
lab/
  store/      Parquet + DuckDB storage, two timestamps per record
  engine/     event loop, clock, strategy context, broker adapters
  risk/       the risk gate
  backtest/   runner, fills, metrics, reports, sweeps, walk-forward
  analysis/   windowed metrics, benchmark, error bars, neighbourhood check, thresholds
  mcp/        the MCP server and tools
  registry/   SQLite: runs, decisions, events, findings
  live/       paper/live runner, reconciliation, alerts, kill switch
  adapters/   yfinance, alpaca, govgreed, synthetic
  indicators/ pandas/numpy indicators, cached
  agent/      earlier self-driving research loop (see below)
  api/        FastAPI backend for the console
  cli.py      the `lab` command; every subcommand supports --json
strategies/   example strategies
cfg/          configs, universe files, research briefs
console/      React + TypeScript web console
docs/         mcp.md, guide.md, CONTRACTS.md, design docs
```

## About `lab/agent`

Before the MCP toolset, this project had its own research loop that called
LLM APIs directly, with its own provider abstraction, prompt caching, budget
tracking and resume logic. That code still works and is still useful for
unattended overnight runs. But a desktop agent already has all of that
infrastructure, so the MCP server is now the primary interface. See
[`docs/agents.md`](docs/agents.md) for the old loop.

## Running the tests

```bash
./.venv/Scripts/python.exe -m pip install -e ".[data,mcp,agent,dev]"
./.venv/Scripts/python.exe -m pytest -q          # ~930 tests, about two minutes
cd console && npm run test && npm run typecheck
```

The suite includes guard tests that fail if a prompt references a tool or
argument that doesn't exist, if the MCP layer imports private names from other
packages, or if a benchmark is computed over a different window than the
strategy.

## License

MIT. See [`LICENSE`](LICENSE).
