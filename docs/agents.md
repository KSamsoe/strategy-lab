# Agent runs — the options, and which ones matter

Two unrelated things share the word "agent" in this repo, and almost every
decision below depends on which one you mean.

|  | **Pattern A — agent as researcher** | **Pattern B — agent in the loop** |
|---|---|---|
| Command | `lab agent author` | a strategy file (`strategies/agent_daily.py`) |
| When the model runs | at research time, offline | inside `on_bar`, every session |
| What ships | an ordinary deterministic strategy | a strategy that calls a model to trade |
| LLM in the runtime path | **no** | **yes** |
| Backtestable | yes, honestly | **no** — contaminated by construction |
| Status | the primary pattern | experimental, paper only |

Pattern A is where nearly all the value is. It gets most of what people mean by
"agentic trading" with none of the runtime risk, because the artifact it
produces is a normal Python file you can read, diff, and run without any model
present. Pattern B exists so the idea can be evaluated honestly rather than
argued about.

---

## 1. Which backend the model comes from

Orthogonal to the pattern — every option below works with any of these. Set it
once in `.env`, or override per command with `--provider`.

| `LAB_AGENT_PROVIDER` | Auth | Structured output | Cost reporting |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | forced tool schema | price-table estimate |
| `claude_code` | **Claude subscription, no key** | prompted schema, JSON parsed back | CLI's own tally, notional |
| `openai` | `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / none for local | forced tool schema | measured, on OpenRouter |
| `auto` *(default)* | first one that can run | — | — |

```bash
lab agent providers           # who can run right now, and why not. Costs nothing.
lab agent providers --probe   # spends one call proving it actually works.
```

`configured` and `working` are different states, which is why `--probe` exists.

**Picking one.** `claude_code` if you have a Claude subscription and no API
budget — it is the cheapest way to run Pattern A hard. `anthropic` if you want
forced tool schemas and the lowest latency. `openai` for OpenRouter's model
menu, or to point at a local Ollama/LM Studio and pay nothing at all.

Three things about `claude_code` that are not obvious:

- It **cannot force a tool schema**. The schema goes into the prompt and the
  JSON is parsed back out. Acceptable only because `validate_targets` already
  treats model output as hostile and the gate re-checks everything; the provider
  reports `forces_tools: false` rather than letting you assume parity.
- Each fresh invocation re-pays for Claude Code's own system prompt — tens of
  thousands of cache-creation tokens — so its per-call cost looks alarming next
  to a bare API call.
- Add a few seconds of process startup per call. Fine for Pattern A, fine for
  Pattern B at daily cadence, wrong for anything faster.

`LAB_AGENT_MODEL` spelling differs per backend: `claude-sonnet-5` for
`anthropic`/`claude_code`, `anthropic/claude-sonnet-4.5` for OpenRouter, a local
model name for Ollama.

---

## 2. Pattern A — `lab agent author`

The loop: propose parameters → run a real walk-forward backtest → read the
out-of-sample score → revise. Every iteration is a registered run with a diff, a
rationale and a lineage entry.

```bash
lab agent author \
  --seed strategies/momo.py \
  --config cfg/momo.yaml \
  --objective "Raise out-of-sample Sharpe. Vary lookback and top_n only." \
  -n 10 --budget 5.00
```

| Flag | Default | What it actually controls |
|---|---|---|
| `--seed` / `-s` | *required* | The strategy to start from. Its `PARAMS` define the search space. |
| `--config` / `-c` | *required* | Backtest config every iteration is scored against: universe, dates, fills, limits. |
| `--grid` / `--freeform` | `--grid` | `grid`: the model may only change **parameter values**. `freeform`: it may rewrite the **strategy source**. |
| `--objective` / `-o` | none | One-line brief handed to the model each iteration. The main steering wheel. |
| `--iterations` / `-n` | `10` | Propose-test-revise cycles. |
| `--metric` / `-m` | `oos_sharpe` | The fitness function. |
| `--oos-split` | `4:1` | Walk-forward train:test blocks used to score each iteration. |
| `--periods` | `4` | Break each run into N contiguous sub-periods so a regime-dependent strategy is visible. |
| `--budget` | `5.0` | Hard ceiling on model spend. The loop stops when hit. |
| `--model` | configured | Override the model id for this run. |
| `--provider` | configured | Override the backend for this run. |
| `--json` | off | One JSON object on stdout, nothing else. |

**`--grid` versus `--freeform` is the consequential choice.** Grid mode searches
a space you defined by writing the strategy, so the worst case is a badly-tuned
version of your idea. Freeform lets the model author new logic — it explores
further and overfits faster, and what it writes needs reading before it goes
anywhere near an account.

### What freeform actually does to files

It writes model-authored Python to disk and imports it, so it is worth being
precise about where and what:

- **Your seed is never modified.** It is read once and never written. Point the
  loop at `strategies/momo.py` and that file is byte-identical afterwards.
- **Each rewrite becomes one new file**, `runs/<loop_id>/<seed>_i<n>.py` — a
  *full replacement* source, not a patch. Nothing lands in `strategies/`; a
  variant you want to keep is yours to copy out deliberately.
- **It is a chain, not a star.** Iteration 2 is shown iteration 1's file as "the
  file under edit", not the seed. The loop walks away from where it started, so
  by iteration 8 the code can share very little with what you wrote.
- **A params-only iteration keeps the previous file.** Freeform does not have to
  rewrite the source every time; when it proposes parameters alone, the chain's
  current file keeps running.
- **Nothing is auto-promoted.** Every variant is an ordinary registered run.
  `lab runs list --origin agent-loop` and the console's lineage view are how you
  find the one worth keeping.

The practical consequence: freeform output is code you have not reviewed,
executing in your process, selected by a metric. That is fine for research and
is exactly why the artifact is deterministic and readable at the end — but read
the diff before the file goes anywhere near an account.

**`--metric` should stay out-of-sample.** It is the containment. An agent is a
tireless overfitter, and pointing its fitness function at an in-sample number
turns the loop into an expensive random-number generator with good manners.

### Can the agent choose its own backtest window?

No, and it should not. The proposal schema is `additionalProperties: false` with
exactly `params`, `rationale`, `source`, `stop` — the date range comes from your
`--config` and the agent cannot reach it.

That is deliberate. Whoever picks the test period *and* is scored on it will find
the period that flatters them; an agent free to choose its own window would
optimise the window, not the strategy, and report a beautiful number about a
market that no longer exists.

The underlying goal — "does this work in more than one market" — is the right
one, so the **harness** asks it instead. `--periods N` splits every run into N
contiguous sub-periods and scores each, and the model sees the breakdown as
`by_period` on every lineage row. It is free: pure slicing of an equity curve
that already exists, no extra backtest. The prompt tells the model to read it
before the pooled score, and that a strategy earning everything in one
sub-period is fitted to a regime rather than to an edge.

To make consistency the fitness function rather than a hint, score on the worst
period:

```bash
lab agent author --seed strategies/momo.py --config cfg/momo.yaml   --periods 4 --metric worst_sharpe
```

`worst_<metric>` takes the minimum across the sub-periods, so a strategy that
makes all its money in one year and gives it back in the others — excellent
pooled, hollow in truth — scores as badly as it deserves.

### What the loop compares against

Only one strategy is ever explored: the seed and its descendants. The loop does
not survey `strategies/`, and it will not try a different idea because yours is
not working. Two reference points sit alongside the lineage:

- **Iteration 0, the seed baseline** — your strategy with its own unmodified
  params. It answers "is this proposal better than where we started", and it
  competes for `best`, so a loop that never improves on the seed correctly
  reports the seed.
- **The market reference** — what simply holding `benchmark:` from your backtest
  config would have done over the same split, costless. It is shown to the model
  every iteration with the instruction that a proposal which cannot beat it
  out-of-sample is not worth shipping, and it comes back in the result as
  `market`. It carries three windows — `in_sample`, `oos` and `full` — plus a
  `window` block naming the exact dates and bar counts each covers.

  **Both the reference and the split start at the first *tradeable* bar**, not the
  first bar of data. Warmup bars are excluded from the strategy's equity curve, so
  measuring the benchmark across them would credit it with a move the strategy sat
  out. On `cfg/momo.yaml` that was a 210-bar, ~107pp head start; correcting it took
  buy-and-hold's full-period return from +181% to +74%.

The second one exists because the first is not a real bar. A lineage can climb
steadily and still lose to owning SPY and going outside — on the synthetic demo
data the momentum seed scores 1.60 OOS Sharpe against buy-and-hold's 2.29, which
is exactly the comparison that used to be invisible. Set `benchmark:` in your
config or you do not get it.

**`--budget` means different things per backend.** Real money on `anthropic` or
OpenRouter; a notional API-equivalent against a Claude subscription. The result
carries `billing` so the number is never presented without that label.

Reading the result:

```bash
lab agent status                # spend, tokens, runs authored
lab runs list --origin agent-loop
lab ui                          # /agent → lineage: is it converging or churning?
```

The lineage chart is the one to watch. A running-best line that flattens while
per-iteration scores stay noisy means the loop is churning, and more iterations
will not help.

---

## 2b. Open-ended research — `lab agent research`

Pattern A with the leash off: no seed, no iteration count. The agent gets the
whole strategy folder and a brief, chooses what to try, and stops when it says
it is satisfied.

```bash
lab agent research   --config cfg/momo.yaml   --brief "Beat buy-and-hold SPY out of sample. I care more about consistency
           across sub-periods than peak Sharpe. Be satisfied once you have
           something positive in 3 of 4 periods, or once you can tell me
           nothing here beats the benchmark."   --minutes 30 --budget 5 --max-calls 60
```

| Flag | Default | Controls |
|---|---|---|
| `--config` / `-c` | *required* | Universe, dates, fills, limits. The agent cannot change any of it. |
| `--brief` / `--brief-file` | none | Your instructions: what to prioritise, when to be satisfied. |
| `--minutes` | `30` | Wall clock for **this sitting**. The session resumes. |
| `--budget` | `5.0` | Model spend ceiling. |
| `--max-calls` | `60` | Hard cap on model calls. |
| `--max-experiments` | `40` | Hard cap on backtests. |
| `--metric` / `--periods` | `oos_sharpe` / `4` | Fitness, and the regime breakdown. |
| `--strategies` | `strategies/` | Folder it may run. |
| `--resume <id>` | — | Continue a session with its history intact. |

Each turn it takes one action — `backtest`, `review`, `write`, `inspect`,
`note`, or `finish` — and sees the result. Every prompt carries elapsed time, spend against
budget, calls remaining, experiments run, the market reference, the strategy
catalogue with its own best result per file, and its prior experiments. An agent
that cannot see the clock polishes one strategy forever.

**It sees both windows, and this matters more than it sounds.** Every experiment
row carries `oos_*` *and* `full_*`. Out-of-sample is the anti-overfitting guard,
but on a 4:1 split it is the last fifth of the range — often under a year, which
is short enough to be mostly noise. Shown alone it reads as the verdict. In a real
session the agent looked at `momo`, saw 9.98% against SPY's 15.2% out-of-sample,
concluded it was worse than buy-and-hold, and set about replacing it. It was
arithmetically right and completely wrong: over the full tradeable period `momo`
returned **+291% against SPY's +74%**. The prompt now tells it to rank on the
fitness metric but to say *why* before discarding a run whose
`full_return_vs_market` is strongly positive — a strategy that wins over four
years and trails over eleven months is usually out of favour, not broken.

**Holdout pressure.** Every backtest reads the out-of-sample window, so the
window stops being out-of-sample as a session runs. Tune one strategy five ways
and keep the best score and you have the maximum of five draws, not an estimate.
Nothing in an out-of-sample split defends against this — the split is what is
being consumed.

So each experiment records `variant_index` (which attempt at that strategy it
is), the clock strip carries `holdout_evaluations`, and once a strategy has three
variants the prompt gains a `HOLDOUT PRESSURE` block comparing what tuning did to
each window:

```
oos_gain_from_tuning:  +0.1513
full_gain_from_tuning: -0.1042
warning: tuning lifted the held-out window by +0.1513 while the full period
         moved -0.1042 ... this is return moving across the split, not an edge.
```

**Every score carries its error bar.** Each experiment reports
`fitness_one_sigma` beside `score`, and the console shows them together as
`score ±1σ`. For a Sharpe-family metric this is the closed form
`SE = sqrt((P + S²/2) / n)`; for return metrics it is the dispersion of the
outcome (`σ√T` for a total return, `σ/√T` annualised); for Calmar there is no
honest closed form, so it reports nothing rather than a number that merely looks
derived.

The magnitudes are sobering. On a 232-bar holdout:

```
oos_sharpe         0.5981  ± 1.0426
oos_total_return   0.1049  ± 0.2148
oos_cagr           0.1149  ± 0.2334
worst_sharpe      -0.2470  ± 1.1478
```

An out-of-sample Sharpe from eleven months of daily bars is barely distinguishable
from zero. This is not a defect in the metric — it is what a window that short can
resolve — but it means a sweep whose scores span less than ±1.0 has produced no
ordering at all. `HOLDOUT PRESSURE` says so outright when the spread across a
tuning chain falls inside one sigma, and again when the fitness metric swings far
more across variants than the full-period result does:

```
the fitness metric swings 1.40x its own mean across these 5 variants while
full-period return swings only 0.28x -- the metric is 5.0 times less stable
than the result it is ranking, so most of the ordering is noise.
```

That is a real session. The agent picked the argmax of a five-variant sweep whose
entire spread (0.99) sat inside one standard error (1.04), and defended it as "the
middle of a confirmed working region" — while a one-bar change to `lb_slow` moved
the score from 0.89 to 0.36. The full-period edge was robust across every
parameter tried; only the ranking between them was noise.

**The direction is the tell, not the size.** The full period is roughly four
times longer, so a real improvement compounds *further* there. A parameter that
lifts the holdout while the full period sits still or falls has not found an
edge — it has moved return across the split boundary. That is a real session:
`core_five` went from 0.129 to 0.280 out-of-sample across four variants while its
full-period return went 2.473 → 2.368, and the agent read a 117% improvement in
something that had got slightly worse. The console shows `variant_index` as a
**try** column, amber from the third attempt.

**The neighbourhood check.** Before a session is allowed to finish, the winning
parameters are automatically re-run about 10% either side of every numeric value
(`--neighbourhood-runs`, default 16, 0 disables). These are diagnostics: they never
enter the experiment table, compete for `best`, or count against
`--max-experiments`, and they cost wall clock rather than model budget.

The number is a **ceiling, not a quota** — the plan never exceeds two runs per
numeric parameter, so a four-parameter strategy costs eight whatever the setting.
When the ceiling *does* bind, the check says so rather than reporting a clean
result on a partial examination:

```
PARTIAL COVERAGE: 8 of 14 runs needed for 7 parameters -- core_weight, lb_slow,
rebalance_every, tilt_weight, top_n, trend_ma nudged one direction only. A cliff
on an untried side would not have been seen, so this is weaker evidence than a
complete check; raise --neighbourhood-runs to 14 to close it.
```

That is a real session: the default was 8, the winning strategy had seven numeric
parameters, and six of them were nudged downward only — while the verdict read
`HOLDS ... 97%` with nothing to distinguish it from a complete check. The default
is now 16 because seven parameters is an ordinary strategy, and the console shows
partial coverage in amber next to the verdict.

It exists because the two diagnostics above can only reason about points the agent
chose to sample. A parameter can look settled across four variants and still sit
next to a cliff nobody ran. The agent will not go looking for that cliff — it
arrives at `finish` with a number it likes.

If the median neighbour keeps less than 60% of the winning score, the finish is
**handed back once** with the numbers attached, and the agent must either re-pick
from the middle of a range that holds or say plainly in its verdict that the result
is value-sensitive. Once only — twice is a loop, and by then it has seen the
evidence.

Run against a real session's winner:

```
       param    value     score  full ret
     lb_fast       47    0.8101    3.1118
     lb_slow      104    0.5810    2.4437
       top_n        3    0.1390    2.4413
    trend_ma      180    0.9524    2.7699
     lb_fast       57    1.0966    3.1945
     lb_slow      128    0.6282    2.4588
```

> HOLDS: nearby parameters keep a median 0.722 against your 0.991 (73%), so the
> pick is on a plateau rather than a spike. Most sensitive parameter: `top_n`
> 4 → 3 costs 86% of the score. Note that `lb_fast=57` scored 1.0966, above the
> value you picked — the point you landed on was not even the local best.
> Full-period return holds at 95% across the same perturbations while the score
> keeps only 73%, so what is fragile is the score, not the strategy.

The verdict names the most sensitive parameter whichever way it goes, because a
plateau on the median can still have one axis it cannot survive. It also says when
a neighbour scored *better* than the pick, and — the line that matters most often —
when full-period return survives perturbations that the fitness score does not.

**Two things weaken a neighbourhood verdict, and both are said out loud.**
Partial coverage (above), and gate clipping: if the risk gate overrode most of a
strategy's intents, its parameters never reached the book, so perturbing them
cannot move the score and the check is measuring the limit rather than the
strategy. Above 50% clipped the verdict carries `CONFOUNDED` and
`robustness_confounded: true`. This is not hypothetical — an intraday session
produced a pick whose intents were **81% clipped** by `position_cap`, the check
reported `HOLDS, 99.7% retention` with one perturbation returning a score
identical to five decimals, and the agent then cited that flatness as evidence
*for* its choice. `_gate_activity`'s docstring had said "tuning its parameters is
tuning the wrong thing" the whole time; the check just never consulted it.

**Flags are flipped, not scaled.** A 0/1 int is a switch wearing a number, and
ten percent of 1 rounds to a step of 1 — so the down-nudge lands on 0 (rejected
as nonsense) and the up-nudge on 2 (meaningless). Such a parameter was silently
never tested, then reported as "not tested at all" as though budget were the
problem. On a real session that flag was `hold_overnight`, and flipping it was
worth more than every other parameter combined:

```
hold_overnight=1   oosSharpe  0.5538   fullSharpe 0.7810   ret 0.589   expo 0.10
hold_overnight=0   oosSharpe -0.4891   fullSharpe 1.1656   ret 1.812   expo 0.75
```

Full coverage counts a flag as one run rather than two, since it has exactly one
alternative — otherwise a complete check reports permanent partial coverage.

**The system prompt is delivered as a file, not on the command line.** This is a
constraint worth knowing before adding guidance to it. `claude` resolves to an
npm shim (`claude.CMD`) on Windows, so the process runs through `cmd.exe`, whose
command-line limit is **8,191 characters** — not the 32,767 `CreateProcess`
allows. The system prompt used to be inlined via `--append-system-prompt`; it
crossed 8,191 when the holdout-pressure guidance was added, and every call
started failing with `The command line is too long`, surfacing as
`stopped_because: model_errors`. It now goes to a temp file passed as
`--append-system-prompt-file`, so the command line is a constant ~208 characters
regardless of prompt size. `test_the_system_prompt_never_reaches_the_command_line`
holds that invariant against a 70KB prompt.

**It gets the engine API in its system prompt.** A complete, runnable strategy
example plus the whole `ctx` surface and the indicator registry ship in the cached
prefix, so writing a strategy does not cost a turn spent `inspect`-ing an existing
file first. The suite compiles that example and runs a backtest with it
(`test_the_prompt_strategy_example_actually_runs`), so it cannot drift out of date
without a test going red.

**Seeing how a run behaved.** Aggregate metrics hide the two things that most
often turn a result into a non-result, so `review <run_id>` returns a bounded
digest of what a run actually did: per-ticker P&L attribution for the whole
universe, trade distribution, the worst drawdown episodes with dates, every
month of returns, and gate activity by rule. About 2,400 tokens, and the last
four digests stay in context rather than being dropped after the turn that
fetched them.

**On how much the agent is shown.** The caps used to be much tighter — eight
tickers, four sample trades out of 1,769, three years of months, and the digest
culled on the next turn. That was set by intuition and never measured. It is
worth knowing the actual shape of the tradeoff before tightening it again:

| | old | now |
|---|---|---|
| digest | ~1,600 tok | ~2,400 tok |
| average prompt over a 60-call session | ~8,000 tok | ~18,200 tok |
| peak prompt | ~6% of a 200k window | ~12% |
| input cost, 60 calls on Opus | $1.91 | $4.94 |

Context window was never the binding constraint — 12% of it is still nothing.
**Budget is.** Input tokens are paid on every call, so the honest reason to cap
anything here is spend, not room to think. `SAMPLE_TRADES` is the dial that
matters: it yields best *and* worst trades, so each unit is two rows at ~119
tokens, and it alone moves a digest between 2k and 4.5k. `REVIEWS_KEPT` is the
other. Both are constants at the top of their modules.

To offset it, the system prompt and tool schemas now carry a cache breakpoint
and are read back at a tenth of the input price. That required splitting the
prompt into a stable half and a volatile one — the clock and the spend used to
sit second in the prompt, where a value that changes every turn invalidated
everything behind it. `test_the_cached_half_of_the_prompt_does_not_move_between_turns`
is what stops that regressing, because a leaked volatile value breaks nothing
visible; it just quietly doubles the bill.

The two numbers worth looking at first:

- `concentration.top_ticker_share` — one name carrying the whole P&L is the most
  common way a backtest turns out not to be a result. A real run of `momo`
  scored a respectable Sharpe with **NVDA alone at 44% of gross profit**, top
  three at 69%.
- `gate.share_clipped_or_blocked` — if most intents are being clipped, the thing
  being measured is not the strategy that was written but that strategy filtered
  through a limit. The same run had **40% of intents clipped or blocked**, mostly
  by `gross_exposure`.

Neither is visible in Sharpe, return or drawdown.

The digest is shown **once** and then falls out of context — it appears under
"result of your last action" and is never written into the turn history. That is
what makes it affordable, and it is why the prompt tells the agent to move
anything it learned into `notes` on the same turn. Manual compaction: read the
detail, write down the finding, let the rows go.

**Stopping.** `finish` requires naming the `run_id` it stands behind —
satisfaction is a claim it has to defend, and "things improved" with nothing to
point at is the failure mode. Otherwise a boundary stops it: `time_limit`,
`budget`, `call_limit`, `experiment_limit`, `rate_limit`, or repeated invalid
actions. Every one of those leaves a resumable session.

**Subscriptions.** Nothing reports remaining quota, so `--max-calls` is the
honest proxy. If a provider limit is hit mid-session the run ends cleanly rather
than crashing, keeps the work already done, and tells you how to resume — which
is what makes `claude_code` usable for work of this shape.

**Starting one from the console.** `lab ui --allow-research` (or
`LAB_UI_ALLOW_RESEARCH=1`) adds a launch form to the Agent → research tab: brief,
config, provider, model, fitness metric, and the four bounds. A running session
gets a **stop** button that ends it after the current turn, leaving it resumable.

The opt-in is deliberate and is the one place the console does something other
than read, pause or stop. The console's rule is that it can only make the system
*safer*, and the risk that names is **market exposure** -- which a research
session cannot create, because it runs backtests and cannot reach a broker. But
it does spend model budget and, in freeform mode, write and execute Python, so a
compromised browser tab should not be able to begin one unasked. Two further
limits keep the surface narrow: the form **picks an existing config** rather than
composing one (letting it set `limits:` would be raising a limit from the UI in
disguise), and every bound is clamped server-side to 180 min / $25 / 400 calls.
Stopping needs no opt-in and works even with the kill switch engaged, because it
only ever reduces what is running.

Sessions launched this way run as a **detached subprocess** of the ordinary CLI,
so they fail, resume and are inspected identically whether a human or the console
started them, and restarting the API does not take the work with it.

**Watching it.** `lab ui` → Agent → *research*. The session file is rewritten
after every turn, so the screen is readable *while* the agent works: the brief,
a live clock/spend/calls strip, best-so-far against the market bar, the
experiment table with each result, and a numbered reasoning trail of every turn
including rejected actions. The loop also writes to the event journal, so
`run_start`, each action with its rationale, each result and the final stop
reason all appear in the fleet screen's live tape as they happen.

**What it cannot do.** Change the date range, universe, fill model or risk
limits — the action schema is `additionalProperties: false` and has no field for
any of them. Write into `strategies/` — authored files go to the session
workspace. Or run a Pattern-B strategy: those call a model once per bar and
produce a contaminated number, so they are refused rather than merely flagged,
excluded from `best`, and refused as a conclusion.

### Promoting what a session found

Authored strategies stay in the session workspace, so getting one into the
library is a deliberate step:

```bash
lab strategy promote <run_id>
```

**It keys on a run, never on a file path, and that distinction is load-bearing.**
The agent rewrites the same workspace filename across turns, so the file sitting
where a run says its strategy lived is frequently a later variant. In one real
session the run the agent recommended scored **0.636** while the file left on
disk under that name was its failed follow-up at **0.510** — nothing in the
filesystem told them apart. Every run therefore archives `strategy.py` next to
its metrics, and promotion copies *that*, verified against the hash recorded when
it executed. A tampered archive is refused, not promoted.

It also writes `cfg/<name>.yaml` carrying the run's `resolved_params`, not the
strategy file's `PARAMS` defaults. Params live in the config and the agent varies
them per experiment, so the file alone reproduces nothing. Pass `--no-config` to
skip it, `--name` to promote under a different filename, `--force` to replace an
existing file, and `--dry-run` to see what would happen. Promoting identical code
twice is a no-op rather than a collision.

Runs from before source archiving carry only a hash. Those are refused with an
explanation rather than falling back to the path, because silently promoting the
wrong code is the failure this exists to prevent. `python scripts/promotable.py
runs/<session_id>` lists which runs in a session can be promoted.

Verify a promotion reproduces the run it came from:

```bash
lab backtest strategies/<name>.py --config cfg/<name>.yaml
```

---

## 3. Pattern B — agent in the loop

A strategy that asks a model for target weights each session. Configured through
`params` in a normal backtest config — see `cfg/agent_daily.yaml`.

| Param | Default | What it controls |
|---|---|---|
| `model` | `null` → `LAB_AGENT_MODEL` | Model choice is config, not code. |
| `max_positions` | `5` | Cap on proposed positions, enforced in the schema. |
| `max_position_pct` | `0.20` | Cap on any one weight, enforced in the validator. |
| `history_bars` | `30` | Bars of context per ticker in the bundle. |
| `quoted_closes` | `10` | Closes quoted verbatim per ticker. |
| `indicators` | rsi 14, sma 50 | Indicator values placed in the bundle. |
| `signal_sources` | `[]` | Alt-data sources to include, quoted as data. |
| `signal_lookback_days` / `max_signals` | `10` / `20` | How much alt-data reaches the prompt. |
| `budget_usd` | `1.00` | Per-run ceiling; the strategy stops calling when hit. |
| `latency_budget_s` | `90.0` | A call slower than this disables further calls. |
| `max_tokens` | `2048` | Reply ceiling. |
| `objective` | `""` | One-line mandate appended to the user message. |

Run it as a plumbing check, never as evidence:

```python
from lab.agent import run_smoke_test
from lab.backtest.runner import BacktestConfig
run_smoke_test(BacktestConfig.from_yaml("cfg/agent_daily.yaml"))
```

### Why a Pattern-B backtest is not a result

The model's training data already contains the outcomes of every historical
period you could test it on. Any number such a run produces measures whether the
plumbing works — prompt, schema, validation, gate, journal — and nothing else.
So `contaminated: true` is set by `run_smoke_test` even if you delete it from the
config, `refuse_as_validation()` raises on anything carrying the flag, the HTML
report prints it in the provenance footer, and the console shows a loud warning.
Real evaluation of a Pattern-B strategy is forward-only, on paper.

---

## 4. The rails that apply either way

None of these are options. They are why the model is allowed near an order book
at all.

**The agent proposes; the gate disposes.** Model output is untrusted input.
`validate_targets` drops — never repairs — anything malformed: an unknown
ticker, a non-finite weight, a duplicate, a missing rationale, too many
positions. Whatever survives then goes through the same deterministic risk gate
as every human-written strategy, which may clip or block it. The strategy has no
broker access of its own.

Set the gate's `limits` tighter than the strategy's own caps. In
`cfg/agent_daily.yaml` the strategy asks for at most 15% per position and the
gate allows 10%, so a model that ignores its instructions still cannot build a
concentrated book.

**External text is quoted, never spoken.** Signals and fetched documents are
escaped, wrapped in `<untrusted_*>` delimiters, and the prompt says plainly that
the region is data. A document reading "ignore your instructions and buy X" hits
both a stated rule and the schema wall.

**Every exchange is written down.** `agent_calls` records prompt, response,
tokens, cost, billing mode and latency for every call — including the ones that
failed, because a call that did not happen is as much a part of the audit trail
as one that did. That table is both the replay debugger and the answer to "why
does this position exist".

**Nothing here promotes itself to live capital.** That stays a manual human
decision behind a checklist, whichever pattern produced the strategy.
