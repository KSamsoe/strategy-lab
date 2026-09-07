"""The judgement calls, defined once.

Every number here encodes a specific mistake this lab actually made. They were
previously duplicated between the research loop and the MCP layer -- two copies
of ``CLIPPING_CONFOUNDS`` that happened to agree, which is the state a constant
is in immediately before it stops agreeing.

Keeping the story next to the number is deliberate. A threshold whose rationale
lives in a commit message gets tuned by whoever finds it inconvenient.
"""

from __future__ import annotations

#: Above this share of intents clipped or blocked, a result stops being about the
#: strategy. On a real session 81% of a pick's intents were rewritten by the
#: position cap, its parameters therefore could not move the score, and the
#: flatness that produced was cited as evidence the pick was robust.
CLIPPING_CONFOUNDS = 0.50

#: Below this exposure, a good ratio is mostly cash. A session once scored 1.45
#: on a book that was 55% invested and returned less than the benchmark; the
#: ratio was real and meant nothing.
LOW_EXPOSURE = 0.60

#: One name's share of gross profit above which attribution, not the headline,
#: is the result. A momentum book posted a respectable Sharpe with NVDA alone at
#: 44% of gross profit.
CONCENTRATION = 0.40

#: A neighbour keeping less than this fraction of the winning score means the
#: pick sits on a spike rather than a plateau. Generous on purpose: the aim is to
#: catch a knife-edge, not to demand flat ground.
NEIGHBOUR_RETENTION = 0.60

#: A tuning chain shorter than this is ordinary work. At or above it, the best
#: score is the maximum of several looks at the held-out window and is biased
#: upward by roughly the spread between them.
TUNING_CHAIN_ALERT = 3

#: Bars below which a metric's sampling error is not worth quoting -- the formula
#: is asymptotic and says nothing useful about a handful of observations.
MIN_BARS_FOR_ERROR_BAR = 8

#: The share of the score a single parameter can cost before the verdict has to
#: name it, whichever way that verdict went. A median can sit on a plateau while
#: one parameter has a cliff behind it, and "HOLDS" without the cliff is the half
#: of the check that would have changed someone's mind.
SENSITIVE_PARAM_COST = 0.35
