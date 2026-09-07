"""Analysis primitives shared by every consumer of a backtest.

The arithmetic of judging a result -- windowed metrics, the benchmark measured
over the same bars, sub-period breakdowns, the sampling error on a score, and the
thresholds that decide when a number needs a caveat.

These used to live inside ``lab/agent``, which meant the MCP toolset imported
nine underscore-prefixed internals from the orchestration layer it was built to
replace, and two copies of ``CLIPPING_CONFOUNDS`` drifted apart waiting to
happen. Nothing here knows about models, prompts or sessions.
"""

from lab.analysis.neighbourhood import (
    flip,
    is_flag,
    neighbourhood_report,
    perturbable,
    perturbations,
)
from lab.analysis.thresholds import (
    CLIPPING_CONFOUNDS,
    CONCENTRATION,
    LOW_EXPOSURE,
    MIN_BARS_FOR_ERROR_BAR,
    NEIGHBOUR_RETENTION,
    SENSITIVE_PARAM_COST,
    TUNING_CHAIN_ALERT,
)
from lab.analysis.windows import (
    LINEAGE_METRICS,
    fitness_one_sigma,
    market_reference,
    period_breakdown,
    split_point,
    tradeable_timestamps,
    window_metrics,
)

__all__ = [
    "CLIPPING_CONFOUNDS",
    "CONCENTRATION",
    "LINEAGE_METRICS",
    "LOW_EXPOSURE",
    "MIN_BARS_FOR_ERROR_BAR",
    "NEIGHBOUR_RETENTION",
    "SENSITIVE_PARAM_COST",
    "TUNING_CHAIN_ALERT",
    "fitness_one_sigma",
    "flip",
    "is_flag",
    "market_reference",
    "neighbourhood_report",
    "period_breakdown",
    "perturbable",
    "perturbations",
    "split_point",
    "tradeable_timestamps",
    "window_metrics",
]
