# The MCP toolset

The lab exposed as a set of tools for an AI agent, instead of a platform that
makes its own LLM calls.

```bash
lab-mcp                      # stdio transport; or: python -m lab.mcp.server
```

Copy `.mcp.json.example` to `.mcp.json` in the repo root to register it with
Claude Code. After a restart, `strategy-lab` appears with 37 tools, 3 resources
and 2 prompts.

## Why this exists

The earlier version of this project had its own research loop (`lab/agent/`,
about 5,600 lines): provider abstraction across three LLM backends, prompt
caching, budget tracking, rate-limit handling, resume, a call ledger. A desktop
agent already has all of that. What it doesn't have is the deterministic part of
this project: the point-in-time data store, the look-ahead barrier, the risk
gate, and the run registry that archives each run's exact source. So the MCP
server exposes that part and leaves the LLM to the client.

There was also a capability gap. The old loop could only run backtests. The
useful findings in this project all came from measuring the data first:

- cross-sectional momentum has signal at about 126 days and none at 5, 21 or 63;
- realised volatility is the most predictable quantity in the data (+0.49 rank
  correlation), but timing exposure with it barely pays;
- the best-performing strategy's entire edge came from its ticker list. Same
  code, same dates, same data source, different universe: +157pp over SPY
  became −5pp.

None of those could be found by running more backtests. Each is now a single
tool call.

The old loop still works. `lab agent research` runs unattended for hours and
handles rate limits by stopping cleanly and resuming later. Use it for batch
work. Use the MCP toolset when you're working interactively.

## The main design rule

Every result includes the information needed to judge it.

In earlier versions, five research sessions each reached a confident wrong
conclusion. In every case the fact that would have corrected it was already
computable when the result was returned; it just wasn't included. So `backtest`
now returns the benchmark over the same bars, the exposure, how much the risk
gate changed the strategy, the standard error on the score, and a `warnings`
list in plain English:

```
WARN: score 0.598 is within one standard error (1.043) of the benchmark's 0.853
      -- the holdout cannot separate them, so decide on full-period behaviour,
      drawdown and consistency
WARN: beats the benchmark by +2.17 over the full period but trails by -0.04 out
      of sample -- likely out of favour rather than broken, but say which
```

Each of those warnings previously took a manual audit to discover. Putting the
check in the tool means it runs no matter who or what is driving.

## The tools

### Core

`backtest`, `review`, `neighbourhood`, `runs_list`, `runs_compare`,
`strategies_list`, `validate_strategy`, `data_coverage`, `market_reference`,
`promote`.

`data_coverage` flags tickers that have bars from more than one source. Mixing
two feeds into one price series once invalidated 52 runs. `promote` copies the
source code archived inside the run, verified against the hash recorded at
execution time. It does not copy the file at the recorded path, because research
sessions rewrite their working files and that file is often a later variant.

### Exploration

`signal_scan`, `conditional_returns`, `correlation_matrix`, `data_pull`,
`data_quality`.

These measure the data before a strategy is written. `correlation_matrix`
reports the number of effective independent bets: the 21-name daily universe is
really about 6.4, which is most of why a "diversified" momentum book behaved
like a concentrated one. `conditional_returns` detects the case where a
condition predicts dispersion but not mean return, and says so: use that
condition for sizing, not for timing.

### Experiments

`ablate`, `compare_universes`, `regimes`, `cost_sensitivity`.

`compare_universes` holds everything constant except the ticker list. On `momo`:

```
tech 6         +185.0%   vs SPY +111pp
defensive 6     +56.2%   vs SPY  -18pp
NOTE: beats the benchmark on one universe and loses on another with everything
      else held constant -- its edge is the ticker list, not the method
```

`cost_sensitivity` sweeps slippage and commission and reports where the strategy
stops beating the benchmark. Intraday strategies usually fail here.

### Validation

`bootstrap`, `bootstrap_vs_benchmark`, `permutation_test`, `deflated_sharpe`,
`walk_forward`, `neighbourhood`.

A single holdout can't usually tell you whether a result is luck. Every session
so far has produced Sharpe standard errors between 0.49 and 1.18, which is wider
than the differences being ranked on. These tools measure the uncertainty
directly instead of assuming a formula.

- **`bootstrap`** resamples the strategy's own returns in blocks and reports an
  empirical interval.
- **`bootstrap_vs_benchmark`** resamples the strategy and the benchmark with the
  same block indices. Use this for "is it better than the index". An unpaired
  comparison understates the evidence because both series go through the same
  crashes, so their intervals overlap regardless of the true difference. Example
  from `core20_vt`: the unpaired interval [0.572, 1.231] contained SPY's 0.656
  and looked like no evidence; the paired difference was [+0.057, +0.428] with
  P(strategy ≤ SPY) = 1.2%.
- **`permutation_test`** re-runs the strategy on block-shuffled prices. Every
  ticker is reordered with the same block permutation, so each name keeps its
  own return distribution and cross-sectional correlation survives. What gets
  destroyed is which name leads on any given day. The question this answers is
  whether the strategy's selection and timing add anything beyond holding the
  same names with the same construction. A long-only strategy inherits the
  market's drift under this null, so a high p-value is meaningful: it says the
  signal isn't doing the work.
- **`deflated_sharpe`** adjusts a Sharpe ratio for the number of configurations
  tried. Given a `run_id`, it counts trials from the registry: every registered
  run of that strategy, plus every registered backtest over the same universe,
  dates and timeframe under any strategy name. It only accepts a caller-supplied
  `trials` value that is higher than the registry count. On `core20_vt` the
  registry count was 11; only 2 of those carried the strategy's name.
- **`walk_forward`** rolls a train/test schedule across the data: five windows,
  each choosing parameters on its own in-sample block and scored on the next
  block. Each window gets its own benchmark. With a `grid`, this is true
  walk-forward selection and `oos_rank_of_selected` shows how close the
  in-sample winner came to the out-of-sample winner. Without a grid, it's a
  consistency check on one parameter set across five holdouts.
- **`neighbourhood`** perturbs each parameter about 10% in both directions and
  reports whether the score holds. The `coverage` field says which parameters
  were only tested in one direction. `most_sensitive` names the parameter that
  costs the most.

**A note on the permutation null.** Shuffling each ticker independently is the
obvious implementation and it's wrong. Measured on the broad universe, it
dropped mean pairwise correlation from 0.41 to 0.0003 and equal-weight
volatility from 18.6% to 4.9%. The shuffled portfolio diversified away its own
risk and scored Sharpe 3.1, every shuffle beat the real run, and the p-value
came out as 1.000. Synchronised blocks keep correlation at 0.39 and volatility
at 17.3%, close to the real 0.41 and 18.6%.

`permutation_test` takes a `workers` argument. Backtests are CPU-bound and
independent, so throughput scales with cores: 4 samples on 4 workers took 49s
against about 130s serially. A 1,000-sample test on the intraday config would
take about 56 hours on one core and under two on 36.

### Authoring

`strategy_api`, `strategy_write`, `strategy_read`.

For clients that can't write files directly. `strategy_api` (also available as
the `lab://strategy-api` resource) returns the `Context` and `Strategy`
protocols, the conventions, the available indicators with defaults, and a
skeleton that loads. `strategy_write` loads the source through the engine before
saving it, so a file that won't import is never written. It refuses to overwrite
a promoted strategy unless told to.

### Paper trading

`live_status`, `live_vs_backtest`.

`live_vs_backtest(strategy=...)` collects every paper run of a strategy (the
daily runner registers one run per session), rebuilds the backtest from the live
config, starts it from a flat book on the first paper session, and reports
per-session return correlation, tracking error, the return gap, and whether the
two books hold the same names. Below 20 sessions it shows the numbers but flags
them as noise. This tells you whether the paper book is behaving like its
backtest. It doesn't recommend trading it.

### Findings

`findings_record`, `findings_search`, `findings_update`, and the
`lab://findings` resource.

The run registry records what was run, not what was learned. Before this
existed, conclusions like "the edge was the ticker list" lived in one agent's
context window, and the next session could rediscover them or contradict them
without knowing. A finding is a claim with attached run ids, numbers, tags, and
a status (`open`, `confirmed`, `refuted`, `superseded`). Refuted findings are
kept with the reason. `scripts/seed_findings.py` loads the method-level lessons
into a fresh ledger.

### Jobs

`job_start`, `job_status`, `job_result`, `jobs_list`, `job_cancel`.

Every tool is a synchronous call, so a slow one is limited by the client's
timeout. `job_start(tool, args)` runs any compute-heavy tool in a detached
process and returns a job id immediately. The worker writes its result to
`data/jobs/` when it finishes, so results survive client disconnects and server
restarts. Jobs beyond the concurrency cap (`LAB_MCP_MAX_JOBS`, default cores/4)
wait in a queue. If the server that started a job has died, the job shows as
`lost` rather than running forever.

## Remote clients

```bash
lab-mcp --http --host 0.0.0.0 --port 8765
```

Serves the same toolset over streamable HTTP at `http://<host>:8765/mcp`. Point
the client at that URL. There is no authentication, so only expose it on a
network you control.

## Prompts

- **`audit_run(run_id, config)`** walks the full validation checklist: findings,
  review, benchmark, neighbourhood, paired bootstrap, permutation test (as a
  job), deflated Sharpe, paper-vs-backtest if applicable, and then records a
  finding. Each step explains what its result means.
- **`start_research(config, brief)`** starts a research session in the right
  order: findings first, then data coverage and quality, then signal
  measurement, and only then a strategy.

A test checks that every tool and argument named in a prompt exists on the
server.

## Suggested order of work

0. `findings_search` (or read `lab://findings`) to see what's already known.
1. `data_coverage`. A config that names a source with no bars fails in a way
   that looks like a strategy that doesn't trade.
2. `signal_scan` / `conditional_returns` to measure before designing.
3. `strategy_api` / `strategy_write` if the client can't write files.
4. `backtest`. Read `warnings` on every result.
5. `review` for attribution and gate activity.
6. `ablate` / `compare_universes` / `cost_sensitivity` / `regimes` to isolate
   what's producing the result.
7. `walk_forward` / `bootstrap_vs_benchmark` / `permutation_test` to test for
   luck. Run slow ones through `job_start`.
8. `neighbourhood`, then `promote`. `live_vs_backtest` once it has paper traded.
9. `findings_record`, especially for negative results.

## Known limits

- `bootstrap` measures the uncertainty of a run's own returns. It says nothing
  about data mining; use `permutation_test` and `deflated_sharpe` for that. For
  comparisons against an index, use `bootstrap_vs_benchmark`.
- `permutation_test` keeps the portfolio construction (position count,
  weighting, vol target) and destroys only the signal. If the null still beats
  the index, the construction is doing the work. On `core20_vt` the null
  averaged Sharpe 0.85 against SPY's 0.656, and removing the signal cost only
  0.07 (p = 0.43).
- `signal_scan` on fewer than about 8 names reports that the sample is too
  small. A rank correlation over three tickers is dominated by whichever one led.
- `neighbourhood` needs a run whose strategy file still exists or whose source
  was archived. Older runs that only recorded a hash can't be perturbed.
- Jobs run on the same machine as the server. There is no distribution across
  hosts.
- `live_vs_backtest` compares equity at the paper run's decision time (09:35 ET)
  against the backtest's close-to-close curve, aligned by session. Intraday
  timing differences show up as tracking error.
