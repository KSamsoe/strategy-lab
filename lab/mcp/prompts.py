"""Prompts: the checklists this lab learned the hard way, as something a client
can invoke by name.

A tool fires whoever is driving; a prompt fires when it is asked for. These
exist because the audits that caught every wrong conclusion here followed the
same sequence each time, and an agent that has the sequence in front of it does
not skip the step that would have embarrassed the headline.
"""

from __future__ import annotations


def audit_run(run_id: str, config: str) -> str:
    """Audit one finished run the way this lab audits runs: benchmark over the
    same bars, attribution, gate interference, parameter plateau, paired
    bootstrap, permutation null, trial-deflated Sharpe -- then record what was
    learned. Every step names the tool and says what its answer means."""
    return f"""\
Audit run `{run_id}` (config `{config}`). Work through every step; do not stop at
the first good number, and do not skip a step because the previous one looked
fine -- the wrong conclusions this lab has reached were each one step short.

1. `findings_search(run_id="{run_id}")` and `findings_search(query=<strategy name>)`.
   Read what is already known about this strategy before forming a view.

2. `review("{run_id}")`. Read `concentration` (one name carrying the result is
   the most common way a backtest turns out not to be one) and `gate`. If
   `gate.share_clipped_or_blocked` is high, what was backtested is the risk
   limits' version of the strategy, and every later step measures the limits.

3. `market_reference("{config}")`. The benchmark over the *same* bars, from the
   first tradeable bar. Compare full-period AND out-of-sample; judging on the
   holdout alone discards most of the evidence, and an unpaired interval that
   contains the benchmark is not evidence of no difference (step 5 is).

4. `neighbourhood("{run_id}", "{config}")`. Plateau or spike? Read `coverage`:
   a HOLDS covering half the parameters is a weaker claim. Read
   `most_sensitive` whichever way the verdict went.

5. `bootstrap_vs_benchmark("{run_id}", "{config}")`. The paired test. Report the
   90% interval on the *difference* and P(strategy <= benchmark). This is the
   answer to "is it better than the index"; `bootstrap` alone is not.

6. `permutation_test` -- as a job: `job_start("permutation_test",
   {{"run_id": "{run_id}", "config": "{config}", "samples": 100, "workers": 8}})`
   and poll `job_status`. The null keeps the construction (names, weighting,
   vol target) and destroys the signal. A high p-value with a null that still
   beats the benchmark means the *construction* is the edge and the signal is
   not doing much -- which is a finding, not a failure. Say which it is.

7. `deflated_sharpe(run_id="{run_id}")`. Trials are counted from the registry;
   do not pass a smaller number from memory.

8. If the strategy is or has been paper traded: `live_status()` then
   `live_vs_backtest(<paper run id>)`.

9. `findings_record(...)`. One claim someone could act on or refute, with the
   numbers in `evidence` and this run id in `run_ids`. If the audit refuted an
   existing finding, `findings_update` it with the reason rather than leaving
   two contradictory claims open.

Then write the verdict in three parts: what the result rests on (construction,
signal, universe, or one name), what would have to be true for it to hold going
forward, and what you did not test.
"""


def start_research(config: str, brief: str = "") -> str:
    """Begin a research pass the way that has actually produced findings here:
    read the ledger, measure the data, then and only then write a strategy."""
    return f"""\
Research brief: {brief or "(none given -- ask what the operator wants to know)"}
Config: `{config}`

Order of work. The most valuable findings this lab has produced came from
measuring the data before writing a strategy against it, and none of them were
reachable by running more backtests.

1. `findings_search()` -- read the ledger. What is already known, and what was
   refuted. Do not re-discover a negative result at full price.
2. `data_coverage()` then `data_quality(...)` for the config's universe. A
   config naming a source with no bars fails in a way that looks like a
   strategy that does not trade.
3. `signal_scan("{config}", ...)` and `conditional_returns(...)`. Which horizons
   carry cross-sectional signal? Which conditions change forward returns? If
   the answer is "none", say so and stop -- that is a finding.
4. Only now: `strategy_api()` for the contract, `strategy_write(...)` for the
   file, `validate_strategy(...)` before the first run.
5. `backtest("{config}", ...)`. Read `warnings` on every result. Size the
   strategy to fit inside the config's limits rather than letting the gate
   rewrite it.
6. `ablate(...)` with one change per case. `compare_universes(...)` before
   believing an edge: a strategy's edge is very often its ticker list.
7. `walk_forward("{config}", ...)` for consistency across folds; then the
   audit sequence (`audit_run` prompt) on the survivor.
8. `findings_record(...)` for every conclusion, negative ones first.

Scores carry `score_one_sigma`; two numbers closer than that are the same
number. Prefer the middle of a working range over the single best value.
"""
