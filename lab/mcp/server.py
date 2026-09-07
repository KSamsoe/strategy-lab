"""The strategy-lab MCP server: the lab as a toolset a desktop agent can drive.

The platform grew an LLM-orchestration layer of its own -- provider abstraction,
prompt caching, budget metering, rate-limit handling, resume, a call ledger --
about 4,600 lines re-implementing what a desktop agent harness already provides.
This module inverts that. The deterministic parts stay (the point-in-time store,
the look-ahead barrier, the risk gate, the run registry, the metrics), and the
agent comes from outside.

The old ``lab agent research`` loop is untouched and still useful for unattended
runs -- it works while nobody is watching, which a toolset cannot. What changes
is which one is primary.

Two design rules, both learned expensively:

**Every number arrives with its caveat.** Five research sessions each reached a
confident wrong conclusion, and in every case the correcting fact was computable
at the moment the result was handed over. So ``backtest`` returns the benchmark
measured over the same bars, the exposure, the gate's interference, the error bar
on the score, and a ``warnings`` list that says in words what would embarrass the
headline. Guards that live in the tool fire whoever is driving; guards that live
in a prompt fire only when the prompt is read.

**Exploration is a first-class verb.** The research loop could only backtest. The
most valuable findings on this platform came from measuring the data first --
that cross-sectional momentum only has signal at ~126 days, that the winning
strategy's edge was entirely its ticker list -- and none of them were reachable
by running more backtests.

Run it with ``lab-mcp`` (stdio) for a client on the same machine, or
``lab-mcp --http --host 0.0.0.0 --port 8765`` on a box in the rack so an agent
elsewhere can drive it. Long work goes through ``job_start``.
"""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer

from lab.mcp import (
    prompts,
    tools_core,
    tools_experiment,
    tools_explore,
    tools_findings,
    tools_jobs,
    tools_live,
    tools_validate,
)

INSTRUCTIONS = """\
strategy-lab: backtesting with a point-in-time data store, a structural
look-ahead barrier, a deterministic risk gate, and a run registry that archives
the exact source of every run.

Start by reading `lab://findings` (or calling `findings_search`): what this lab
already knows, including what it has refuted. The `audit_run` and
`start_research` prompts carry the checklists.

Suggested order of work:

1. `data_coverage` -- know what bars exist before designing anything. A config
   naming a source with no data fails in a way that looks like a strategy that
   does not trade.
2. `signal_scan` / `conditional_returns` -- measure the structure in the data
   before writing a strategy against it. Cheap, and it is where the real
   findings come from.
3. `strategy_api` then `strategy_write` if you cannot write files yourself.
4. `backtest` -- the main verb. Read `warnings` on every result.
5. `review` -- per-ticker attribution and gate activity. Aggregate metrics hide
   the two things that most often turn a result into a non-result.
6. `ablate` / `compare_universes` / `cost_sensitivity` / `regimes` -- isolate
   what is actually doing the work. A strategy's edge is often its universe.
7. `walk_forward` / `bootstrap_vs_benchmark` / `permutation_test` -- is it
   luck? A single holdout usually cannot tell you; these can. Anything slow
   goes through `job_start` and is collected with `job_result`.
8. `neighbourhood` then `promote` -- does it survive its own parameters, and if
   so, ship the archived source rather than whatever file is on disk.
9. `findings_record` -- write down what was learned, negative results first.

Things worth knowing:

* Scores carry `score_one_sigma`. Two numbers closer together than that are the
  same number. An annualised Sharpe from ~230 daily bars has a standard error
  near 1.0, which is wider than most parameter sweeps.
* Out-of-sample and full-period are both reported. The holdout answers "does it
  generalise"; the full period answers "what did it actually do". Judging on the
  first alone discards most of the evidence.
* If `gate.share_clipped_or_blocked` is high, the backtest measured the risk
  limits rather than the strategy, and tuning its parameters is tuning the wrong
  thing.
* Every backtest reads the held-out window, so tuning against it consumes it.
  Prefer the middle of a working range over the single best value.
* `deflated_sharpe(run_id=...)` counts trials from the registry. Do not pass a
  smaller number from memory.
"""

server = MCPServer(
    name="strategy-lab",
    version="0.2.0",
    instructions=INSTRUCTIONS,
)

TOOLS = (
    # --- core: run, read, look up, ship ---------------------------------------
    tools_core.backtest,
    tools_core.review,
    tools_core.neighbourhood,
    tools_core.runs_list,
    tools_core.runs_compare,
    tools_core.strategies_list,
    tools_core.validate_strategy,
    tools_core.strategy_read,
    tools_core.strategy_write,
    tools_core.strategy_api,
    tools_core.data_coverage,
    tools_core.market_reference,
    tools_core.promote,
    # --- exploration: measure before designing --------------------------------
    tools_explore.signal_scan,
    tools_explore.conditional_returns,
    tools_explore.correlation_matrix,
    tools_explore.data_pull,
    tools_explore.data_quality,
    # --- experiments: isolate what does the work ------------------------------
    tools_experiment.ablate,
    tools_experiment.compare_universes,
    tools_experiment.regimes,
    tools_experiment.cost_sensitivity,
    # --- validation: is it luck? ----------------------------------------------
    tools_validate.walk_forward,
    tools_validate.bootstrap,
    tools_validate.bootstrap_vs_benchmark,
    tools_validate.permutation_test,
    tools_validate.deflated_sharpe,
    # --- paper trading against its backtest -----------------------------------
    tools_live.live_status,
    tools_live.live_vs_backtest,
    # --- the ledger -----------------------------------------------------------
    tools_findings.findings_record,
    tools_findings.findings_search,
    tools_findings.findings_update,
    # --- long work ------------------------------------------------------------
    tools_jobs.job_start,
    tools_jobs.job_status,
    tools_jobs.job_result,
    tools_jobs.jobs_list,
    tools_jobs.job_cancel,
)

for fn in TOOLS:
    server.add_tool(fn)


# --- resources: things to read before doing --------------------------------------


@server.resource("lab://findings", name="findings", mime_type="text/markdown",
                 description="Everything this lab has concluded, with evidence and status.")
def findings_resource() -> str:
    from lab.registry.findings import Findings

    return Findings().digest()


@server.resource("lab://strategy-api", name="strategy-api", mime_type="application/json",
                 description="The contract a strategy file must meet, with a skeleton.")
def strategy_api_resource() -> str:
    import json

    return json.dumps(tools_core.strategy_api(), indent=2)


@server.resource("lab://docs/mcp", name="docs", mime_type="text/markdown",
                 description="How this toolset is meant to be used, and why each tool exists.")
def docs_resource() -> str:
    from pathlib import Path

    from lab.config import get_settings

    path = Path(get_settings().paths.root) / "docs" / "mcp.md"
    return path.read_text(encoding="utf-8") if path.exists() else "docs/mcp.md is missing"


# --- prompts: the checklists ----------------------------------------------------

server.prompt(name="audit_run", description=prompts.audit_run.__doc__)(prompts.audit_run)
server.prompt(name="start_research", description=prompts.start_research.__doc__)(
    prompts.start_research
)


def main() -> None:
    ap = argparse.ArgumentParser(prog="lab-mcp", description="strategy-lab MCP server")
    ap.add_argument("--http", action="store_true",
                    help="serve over streamable HTTP instead of stdio (for a remote client)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
