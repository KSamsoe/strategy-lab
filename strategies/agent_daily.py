"""Pattern B in strategy-file form: one model call per session, gate-disposed.

EXPERIMENTAL, and a historical run of this file is **not evidence of anything**
-- the model's training data already contains the outcomes of whatever period
you backtest. Run it through ``lab.agent.in_loop_strategy.run_smoke_test``,
which forces ``contaminated=True`` onto the config so the metrics, the registry
row and the report all say so. Real evaluation is forward-only, on paper.

Everything interesting lives in ``lab.agent.in_loop_strategy``; this file exists
because the engine loads strategies from ``strategies/*.py`` and because the
parameters below -- model, budgets, what goes in the bundle -- are config that
should be editable without touching library code.
"""

from __future__ import annotations

from lab.agent.in_loop_strategy import AgentStrategy

NAME = "agent_daily"

#: True for every run of this file, whatever the config says. Read by the
#: contamination fence and by anything deciding whether a run may be cited.
CONTAMINATED = True

PARAMS = {
    "model": None,              # None -> settings.agent_model (LAB_AGENT_MODEL)
    "max_positions": 4,
    "max_position_pct": 0.15,   # the gate enforces its own cap on top of this
    "history_bars": 40,
    "quoted_closes": 10,
    "indicators": [
        {"name": "rsi", "n": 14},
        {"name": "sma", "n": 50},
        {"name": "rolling_vol", "n": 20},
    ],
    "signal_sources": [],       # e.g. ["govgreed"]; payload text is quoted, never trusted
    "signal_lookback_days": 10,
    "max_signals": 20,
    "budget_usd": 1.00,         # per-run ceiling; the strategy stops calling when hit
    "latency_budget_s": 90.0,
    "max_tokens": 2048,
    "objective": (
        "Hold a small book of the strongest names in the universe. Prefer cash to "
        "a thin thesis."
    ),
}


def build(params: dict | None = None) -> AgentStrategy:
    """Fresh instance per load: the strategy carries per-run spend and session
    state, so a module-level singleton would leak one run's budget into the next."""
    return AgentStrategy(params or PARAMS)
