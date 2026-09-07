"""Record what was learned, not just what was run.

The run registry is complete and useless for this: it says a backtest happened,
not what it meant. Two sessions rediscovered the same negative result at full
price because the first one's conclusion lived in a context window. Findings
are the ledger the next agent reads before it starts -- also exposed as the
`lab://findings` resource.
"""

from __future__ import annotations

from typing import Any

from lab.mcp._common import jsonable


def findings_record(
    title: str,
    claim: str,
    run_ids: list[str] | None = None,
    tags: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    status: str = "open",
    supersedes: str | None = None,
    author: str = "",
) -> dict[str, Any]:
    """Record a conclusion with the evidence it rests on.

    `claim` should be a sentence someone could act on or refute: "blend_tilt's
    edge is its ticker list -- same params and dates on a 36-name broad universe
    lose to SPY by 5pp" rather than "blend_tilt looked worse". Put the numbers in
    `evidence` and the runs in `run_ids` so the claim can be re-checked when the
    data or the code changes.

    Negative results are the ones most worth recording; they are the ones most
    often paid for twice. `status` starts `open`; `confirmed` when independent
    evidence agrees, `refuted` when it does not. Pass `supersedes` with an older
    finding's id when this one replaces it.
    """
    from lab.registry.findings import Findings

    return jsonable(Findings().record(
        title, claim, run_ids=run_ids or [], tags=tags or [], evidence=evidence,
        status=status, supersedes=supersedes, author=author,
    ))


def findings_search(
    query: str | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
    status: str | None = None,
    include_superseded: bool = False,
    limit: int = 25,
) -> dict[str, Any]:
    """What this lab already knows. Read before designing, and before concluding.

    Free-text `query` matches title and claim; `tags` and `run_id` narrow it.
    A search that comes back empty on a topic is itself information: nobody has
    measured it yet.
    """
    from lab.registry.findings import Findings

    rows = Findings().search(
        query, tags=tags or [], run_id=run_id, status=status,
        include_superseded=include_superseded, limit=int(limit),
    )
    return jsonable({"count": len(rows), "findings": rows})


def findings_update(
    finding_id: str,
    status: str | None = None,
    note: str = "",
    run_ids: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Move a finding to confirmed / refuted, or attach more evidence to it.

    Say why in `note`: the history of a claim is kept, and "refuted: the broad
    universe run had the 40% gross ceiling from the empty limits block" is worth
    more than the status change alone.
    """
    from lab.registry.findings import Findings

    return jsonable(Findings().update(
        finding_id, status=status, note=note, run_ids=run_ids or [], evidence=evidence,
    ))
