"""The agentic layer (design doc §8), in two clearly separated patterns.

**Superseded as the primary interface.** This package makes its own model
calls -- provider abstraction, prompt caching, budget metering, resume, a call
ledger -- and still works, and is still the right tool for an unattended
overnight session because it runs while nobody is watching. The primary way to
drive the lab is now ``lab.mcp``: the same engine as a toolset for a desktop
agent that already has a model and a harness of its own. See docs/mcp.md.

**Pattern A — agent as researcher** (``author_loop``), the primary one. An
Anthropic-API loop authors and iterates strategies against real backtests and
ships an ordinary deterministic artifact: no LLM in the runtime path.

**Pattern B — agent in the loop** (``in_loop_strategy``), experimental. A
strategy whose ``on_bar`` asks a model for target weights through a constrained
tool. The agent proposes; the risk gate disposes. Historical runs of it are
contaminated by the model's training data and are tagged as such.

Both share ``schemas``: the tool definitions, the validator that treats model
output as untrusted input, and the audited call path that writes every prompt,
response, token count and dollar to the registry's ``agent_calls`` table.

Nothing here imports ``anthropic`` at module scope, so ``import lab.agent`` is
safe without the optional dependency; a missing package or key degrades to an
``AgentUnavailable`` with a message that says what to do about it.
"""

from __future__ import annotations

from lab.agent.author_loop import AuthorLoopConfig, Iteration, run_author_loop
from lab.agent.in_loop_strategy import (
    CONTAMINATION_NOTE,
    AgentStrategy,
    is_contaminated,
    mark_contaminated,
    refuse_as_validation,
    run_smoke_test,
)
from lab.agent.schemas import (
    STRATEGY_PROPOSAL_TOOL,
    TARGETS_TOOL,
    AgentCall,
    AgentUnavailable,
    ContaminatedRunError,
    ToolCall,
    call_tool,
    estimate_cost,
    get_client,
    list_calls,
    record_call,
    scrub_untrusted,
    spend,
    strategy_proposal_tool,
    targets_tool,
    validate_proposal,
    validate_targets,
    validate_targets_verbose,
)

__all__ = [
    "AgentCall",
    "AgentStrategy",
    "AgentUnavailable",
    "AuthorLoopConfig",
    "CONTAMINATION_NOTE",
    "ContaminatedRunError",
    "Iteration",
    "STRATEGY_PROPOSAL_TOOL",
    "TARGETS_TOOL",
    "ToolCall",
    "call_tool",
    "estimate_cost",
    "get_client",
    "is_contaminated",
    "list_calls",
    "mark_contaminated",
    "record_call",
    "refuse_as_validation",
    "run_author_loop",
    "run_smoke_test",
    "scrub_untrusted",
    "spend",
    "strategy_proposal_tool",
    "targets_tool",
    "validate_proposal",
    "validate_targets",
    "validate_targets_verbose",
]
