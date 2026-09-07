# The MCP toolset

The lab as a set of tools a desktop agent drives, rather than a platform that
makes its own model calls.

```bash
lab-mcp                      # stdio; or: python -m lab.mcp.server
```

`.mcp.json` in the repo root registers it for Claude Code. Restart the client and
`strategy-lab` appears with 37 tools, 3 resources and 2 prompts.

## Why this exists

`lab/agent/` is ~5,600 lines of LLM orchestration — provider abstraction across
three backends, prompt caching, budget metering, rate-limit detection, resume, a
call ledger, forced tool schemas. A desktop agent harness already provides every
one of those. Meanwhile the *deterministic* half — the point-in-time store, the
look-ahead barrier, the risk gate, the run registry that archives each run's
exact source — is the part nothing else has.

There is also a capability argument. The research loop could only backtest. Every
genuine finding this lab has produced came from measuring first:

- cross-sectional momentum has signal at ~126 days and none at 5, 21 or 63;
- realised volatility is the most predictable quantity in the data (+0.49) but
  timing exposure with it barely pays;
- the reigning champion's entire edge was its ticker list — same code, same
  dates, same source, different universe, and +157pp becomes −5pp.

None were reachable by running more backtests. Each is now one call.

**The old loop is untouched.** `lab agent research` still runs unattended for
hours, which a toolset cannot, and it handles a subscription rate limit by
stopping cleanly and resuming with its history intact. Use it for batch work. Use
this when you want to think.

## The design rule

Every number arrives with the thing that would embarrass it.

Five research sessions each reached a confident wrong conclusion, and in every
case the correcting fact was computable at the moment the result was handed over.
So `backtest` returns the benchmark measured over the *same* bars, the exposure,
the gate's interference, the sampling error on the score, and a `warnings` list
in plain words:

```
WARN: score 0.598 is within one standard error (1.043) of the benchmark's 0.853
      -- the holdout cannot separate them, so decide on full-period behaviour,
      drawdown and consistency
WARN: beats the benchmark by +2.17 over the full period but trails by -0.04 out
      of sample -- likely out of favour rather than broken, but say which
```

Both of those took a full audit round to establish by hand. Guards that live in
the tool fire whoever is driving; guards that live in a prompt fire only when the
prompt is read.

## The tools

**Core** — `backtest`, `review`, `neighbourhood`, `runs_list`, `runs_compare`,
`strategies_list`, `validate_strategy`, `data_coverage`, `market_reference`,
`promote`.

`data_coverage` flags tickers carrying bars from more than one source — the trap
that once invalidated 52 runs by unioning two feeds into one price series.
`promote` copies the source *archived inside the run*, verified against the hash
taken when it executed, because a session rewrites its workspace files in place
and the file at the recorded path is often a later, worse variant.

**Exploration** — `signal_scan`, `conditional_returns`, `correlation_matrix`,
`data_pull`, `data_quality`.

The capability the loop never had. `correlation_matrix` reports *effective
independent bets*: the 21-name daily universe is really about 6.4, which is most
of why a "diversified" momentum book was a concentrated one. `conditional_returns`
detects the flat-mean/rising-dispersion case and says so — the state predicts
risk, not return, so size with it and do not time with it.

**Experiments** — `ablate`, `compare_universes`, `regimes`, `cost_sensitivity`.

`compare_universes` holds everything constant but the ticker list. On `momo`:

```
tech 6         +185.0%   vs SPY +111pp
defensive 6     +56.2%   vs SPY  -18pp
NOTE: beats the benchmark on one universe and loses on another with everything
      else held constant -- its edge is the ticker list, not the method
```

`cost_sensitivity` sweeps slippage and commission and reports where the edge
dies. Intraday strategies die there and nobody had measured the gradient.

**Validation** — `bootstrap`, `bootstrap_vs_benchmark`, `permutation_test`,
`deflated_sharpe`.

The luck problem, which a single holdout usually cannot answer. Every session hit
Sharpe standard errors between 0.49 and 1.18 — wider than the gaps being ranked
on.

- `bootstrap` resamples the strategy's own returns in blocks for an *empirical*
  interval rather than a closed form assuming IID normal returns.
- `bootstrap_vs_benchmark` resamples the strategy **and** the benchmark with the
  same block indices. Use this one for "is it better than the index", because the
  unpaired interval systematically understates the evidence: both sides live
  through the same crashes, so each gets a wide interval and the two overlap
  almost whatever the difference between them is. On `core20_vt` the unpaired
  interval was [0.572, 1.231] and comfortably contained SPY's 0.656 — which reads
  as no evidence — while the paired difference was [+0.057, +0.428] with
  P(strategy ≤ SPY) = 1.2%.
- `permutation_test` re-runs the strategy on block-shuffled prices. Each name
  keeps its own drift and volatility, and **every ticker is reordered by the same
  block permutation** so cross-sectional correlation survives; what is destroyed
  is which name leads on a given day. So this asks whether **selection and timing
  add anything beyond holding the universe in this construction** — not whether
  the strategy beats cash. A long-biased strategy inherits the drift and scores
  well on this null however little its signal contributes, which is exactly what
  makes a high p-value informative.
- `deflated_sharpe` corrects a Sharpe for how many configurations were evaluated.
  Count every backtest against the holdout, including discarded ones.

Shuffling each ticker independently is the tempting mistake and it invalidates
the test: measured on the broad universe it took mean pairwise correlation from
0.41 to 0.0003 and equal-weight volatility from 18.6% to 4.9%, so the null
portfolio diversified away its own risk and scored Sharpe 3.1. Every shuffle beat
the real run and the p-value came back 1.000, which looks like a damning result
and is an artefact of the null. Synchronised blocks hold correlation at 0.39 and
volatility at 17.3% against the real 0.41 and 18.6%.

`permutation_test` takes `workers` and parallelises across processes: 4 samples on
4 workers ran in 49s against ~130s serially. Backtests are CPU-bound and
embarrassingly parallel, so this is the obvious thing to point a many-core machine
at. A 1,000-sample test on the intraday config is ~56 hours serially and under two
on 36 cores — the difference between "we cannot know" and "we know".

**Walk-forward** — `walk_forward`.

The single 4:1 holdout is the weakest thing in the validation set. This rolls a
train/test schedule across the span — five windows, each choosing parameters on
its own in-sample block and being scored on the next, unseen one — with the
benchmark over each window's own out-of-sample bars. With a `grid` it is real
walk-forward selection and `oos_rank_of_selected` says whether the in-sample
winner was anywhere near the out-of-sample winner; without one it is a
consistency check on a single parameter set, which is still five holdouts
instead of one. Built on the sweep engine's existing walk-forward split rather
than a second implementation.

**Trial counting** — `deflated_sharpe(run_id=...)`.

The tool used to ask the caller how many configurations were tried, and the
caller, being the one who tried them, under-counted every time. Given a run id
it now reads the count from the registry: every registered run of the strategy,
and every registered backtest over the same universe, dates and timeframe under
any name. A research session that authored nine strategies against one holdout
and kept one took nine looks at the test set. `trials` can only raise the number.
On `core20_vt` the honest count was 11, not the 2 runs carrying its name.

**Authoring without a filesystem** — `strategy_api`, `strategy_write`,
`strategy_read`.

Claude Code can write files; Cherry Studio and OpenClaw may not. `strategy_api`
(also the `lab://strategy-api` resource) returns the `Context` and `Strategy`
protocols verbatim, the conventions, the available indicators with their
defaults, and a skeleton that loads. `strategy_write` loads the source through
the engine *before* writing, so a file that would not import never lands, and
refuses to overwrite a promoted strategy without being told to.

**Paper trading against its backtest** — `live_status`, `live_vs_backtest`.

The luck filter that runs on money. `live_vs_backtest(strategy=...)` stitches
together every paper run of a strategy (the daily runner registers one run per
session), rebuilds the backtest from the live config the newest run recorded,
starts it flat on the first paper session, and reports per-session return
correlation, tracking error, the return gap and whether the two books hold the
same names. Below 20 sessions it shows the numbers and says they are noise.
Promotion to real capital remains a human decision; this says whether the paper
book is behaving like its backtest, not whether to trade it.

**The ledger** — `findings_record`, `findings_search`, `findings_update`; the
`lab://findings` resource.

The run registry records what was run. Nothing recorded what was *learned*.
"The edge was the ticker list" and "momentum only has signal at ~126 days" lived
in one agent's context window and a docstring, and the next session was free to
rediscover them at full price or contradict them without knowing. A finding is
a claim with its run ids, numbers and tags attached, and a status that moves to
`confirmed` or `refuted` as later work bears on it. Refuted claims stay, marked,
with the reason. The ledger was seeded with what this lab knew as of 2026-09-06.

**Long work** — `job_start`, `job_status`, `job_result`, `jobs_list`,
`job_cancel`.

Every tool is a synchronous call, which caps a piece of work at the client's
tool timeout — fine for a backtest, fatal for a 1,000-sample permutation test.
`job_start(tool, args)` runs any compute tool in a separate process and returns
a job id at once; the worker writes its result to `data/jobs/` the moment it
finishes, so a job outlives the client disconnecting, the server restarting and
the agent forgetting it asked. Jobs past the concurrency cap
(`LAB_MCP_MAX_JOBS`, default cores/4) queue rather than fail. A job whose
server died mid-run is reported `lost` on the next look rather than running
forever.

## Serving a remote client

```bash
lab-mcp --http --host 0.0.0.0 --port 8765
```

runs the same server over streamable HTTP at `http://<host>:8765/mcp`, which is
how an agent on a laptop drives the toolset on a box in the rack. Point the
client at that URL instead of a command. The server binds to localhost unless
told otherwise; there is no authentication layer, so keep it on a network you
control.

## Prompts

- `audit_run(run_id, config)` — the audit sequence this lab converged on:
  ledger, review, benchmark over the same bars, neighbourhood, paired
  bootstrap, permutation null as a job, trial-deflated Sharpe, paper-vs-backtest
  if applicable, then record the finding. Each step says what its answer means.
- `start_research(config, brief)` — ledger first, then data coverage and
  quality, then signal measurement, and only then a strategy.

A test pins that every tool a prompt names is one the server registers.

## Suggested order of work

0. `findings_search` (or read `lab://findings`) — what is already known.
1. `data_coverage` — a config naming a source with no bars fails in a way that
   looks like a strategy that does not trade.
2. `signal_scan` / `conditional_returns` — measure before designing.
3. `strategy_api` / `strategy_write` if the client cannot write files.
4. `backtest` — read `warnings` on every result.
5. `review` — attribution and gate activity.
6. `ablate` / `compare_universes` / `cost_sensitivity` / `regimes` — isolate what
   is doing the work.
7. `walk_forward` / `bootstrap_vs_benchmark` / `permutation_test` — is it luck?
   Slow ones through `job_start`.
8. `neighbourhood`, then `promote`; `live_vs_backtest` once it has paper traded.
9. `findings_record` — negative results first.

## Known limits

- `bootstrap` measures the uncertainty of a run's *own* returns. It says nothing
  about whether the strategy was selected by data mining; use `permutation_test`
  and `deflated_sharpe` for that. For a comparison against an index use
  `bootstrap_vs_benchmark` — an unpaired interval that contains the benchmark is
  not evidence of no difference.
- `permutation_test` preserves the portfolio construction — the position count,
  the weighting, the vol target — and destroys only the signal. A null that still
  beats the index is telling you the construction is doing the work, which is a
  finding rather than a failure: on `core20_vt` the null averaged Sharpe 0.85
  against SPY's 0.656, and removing the signal cost only 0.07 (p = 0.43).
- `signal_scan` on fewer than ~8 names says so — a rank correlation over three
  tickers is dominated by whichever one led.
- `neighbourhood` needs a run whose strategy file still exists or whose source was
  archived. Runs from before source archiving carry only a hash and cannot be
  perturbed.
- Jobs run on the machine the server runs on. One server, one queue; there is no
  distribution across hosts.
- `live_vs_backtest` compares equity marked at the paper run's decision time
  (09:35) against the backtest's close-to-close curve; the alignment is by
  session, so intraday timing differences show up as tracking error, which is
  the point.
