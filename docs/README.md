# Documentation

**Using it**

- [**mcp.md**](mcp.md) — the toolset a desktop agent drives: every tool, why it
  exists, the order of work, the prompts, jobs and the remote transport. This
  is the primary interface; start here.
- [**guide.md**](guide.md) — the user guide. Setup, the loop, writing a
  strategy, configs, sweeps, the risk gate, the console, paper trading, a
  command reference and troubleshooting. Start here.
- [**agents.md**](agents.md) — the agentic layer: which model backend to use
  (including a Claude subscription with no API key), open-ended research
  sessions, the author loop, and Pattern-B strategies. Superseded by the MCP
  toolset as the primary interface; kept for unattended runs.

**Changing it**

- [**CONTRACTS.md**](CONTRACTS.md) — the binding interface list between
  modules. Read before adding to `lab/`.
- [**console/COMPONENTS.md**](../console/COMPONENTS.md) — prop signatures and
  house rules for the frontend.

**Why it is shaped this way**

The three design docs the implementation was built from. They explain intent
rather than API, and the code deliberately follows them:

- [strategy-lab-design-v1.md](strategy-lab-design-v1.md) — the platform: the
  two-timestamp store, the strategy interface, the backtester, the risk gate,
  the two agent patterns.
- [strategy-lab-console-design-v1.md](strategy-lab-console-design-v1.md) — the
  console: one event schema and two sources, the decision tape, and the rule
  that the UI can only make the system safer.
- [govgreed-bot-design-v1.md](govgreed-bot-design-v1.md) — the alt-data source
  that became one adapter plus one strategy.

## The short version

If you read nothing else, these are the ideas everything else follows from.

**Two timestamps.** Every datum carries `event_time` (when it happened) and
`knowledge_time` (when you could have known it). The store indexes on the
second. A congressional trade is invisible until its disclosure date.

**Look-ahead is structural.** A strategy reaches data only through a `Context`
that cannot serve past `ctx.now`, and the engine proves each indicator is causal
before relying on it.

**One risk gate.** The same deterministic rules sit between every strategy —
human or agent — and the broker. Exits are never blocked by a capacity rule.

**Out-of-sample is the fitness function.** Always, and especially for an agent,
which is a tireless overfitter.

**The ledger must explain the curve.** Trade P&L plus open unrealized equals the
equity gain, checked on every run.

**Optimism is labelled.** Same-bar fills, survivorship bias, and LLM training-
window contamination each get a loud flag rather than a footnote.

**The console can only make things safer.** Pause, cancel, kill. Starting a
strategy or raising a limit are CLI actions behind a human checklist.
