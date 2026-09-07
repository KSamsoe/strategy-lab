"""Is this result a plateau or a spike? -- the neighbourhood check, on plain data.

A held-out score is the maximum of however many looks were taken at the test set,
and the parameter that produced it was chosen because it produced it. Neither
fact is visible from the number. Re-running the strategy at nearby parameters is
the cheapest way to tell a plateau from a spike, and it costs backtests rather
than model calls -- wall clock, not budget.

Split into two plain-data pieces so that whoever is driving can run the
backtests themselves: :func:`perturbations` says what to try, and
:func:`neighbourhood_report` turns the results into a verdict. It previously
lived inside the research loop and took an ``Experiment``, a ``ResearchConfig``
and a ``ResearchSession``, which meant the MCP tool had to fabricate all three to
ask a question about a parameter dict and a list of scores.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping, Sequence

from lab.analysis.thresholds import (
    CLIPPING_CONFOUNDS,
    NEIGHBOUR_RETENTION,
    SENSITIVE_PARAM_COST,
)


def is_flag(value: Any) -> bool:
    """A switch wearing a number: 0/1 ints, and actual bools.

    Worth its own case because scaling one is meaningless. Ten percent of 1
    rounds to a step of 1, so the down-nudge lands on 0 -- which the general
    guard rejects as nonsense -- and the up-nudge lands on 2, which is not a
    value the strategy has any meaning for. The net effect was that a flag was
    silently never perturbed and then reported as "not tested", as though more
    budget would have covered it. On a real session that flag was
    ``hold_overnight``: the single most consequential choice the strategy made,
    and the one the operator's brief had asked about by name.
    """
    if isinstance(value, bool):
        return True
    return isinstance(value, int) and value in (0, 1)


def perturbable(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Parameters a perturbation means anything for.

    Factored out so the plan and the coverage report cannot disagree about what
    "everything" was: a report derived from a second, slightly different
    predicate is how a check comes to claim coverage it does not have.
    """
    return [
        (k, v) for k, v in params.items()
        if isinstance(v, (int, float, bool)) and (is_flag(v) or v != 0)
    ]


def flip(value: Any) -> Any:
    """The only other value a flag can take, in its own type."""
    if isinstance(value, bool):
        return not value
    return 0 if int(value) == 1 else 1


def perturbations(params: Mapping[str, Any], budget: int) -> list[tuple[str, Any]]:
    """``(param, value)`` pairs to try: ~10% either side of each numeric param.

    Round-robin over parameters rather than exhausting one before starting the
    next, so a tight budget still touches every parameter at least once. Which
    direction gets dropped is arbitrary and the odds of catching a given cliff
    are the same either way -- what matters is that the caller is told, so
    ``neighbourhood_report`` reports exactly which parameters were nudged only one way.

    Ten percent because that is the scale at which a parameter choice should not
    matter -- if it does, the result is a property of the exact number rather
    than of the idea.
    """
    numeric = perturbable(params)
    out: list[tuple[str, Any]] = []
    for direction in (-1, 1):
        for key, value in numeric:
            if len(out) >= budget:
                return out
            if is_flag(value):
                # One alternative, not two: flip it on the first pass and leave
                # it alone on the second rather than testing the same thing twice.
                if direction == -1:
                    out.append((key, flip(value)))
                continue
            if isinstance(value, int):
                step = max(1, int(round(abs(value) * 0.10)))
                moved: Any = int(value) + direction * step
                if moved <= 0:
                    continue
            else:
                moved = round(float(value) * (1.0 + direction * 0.10), 6)
                if moved <= 0:
                    continue
            out.append((key, moved))
    return out


def neighbourhood_report(
    params: Mapping[str, Any],
    score: float | None,
    full_return: float | None,
    tried: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a set of perturbation results into a verdict, with its own caveats.

    `tried` is one entry per run: ``{"param", "value", "score", "full_return",
    "ok", "error"}``. `gate` is the winning run's gate activity, if it can be
    read -- a perturbation the gate overrides cannot move the score, so a heavily
    clipped strategy is guaranteed to look robust and the flatness is the limit's
    rather than the strategy's. Without it the check can report a clean bill of
    health on a strategy it never actually tested.
    """
    if not tried:
        return {"checked": 0, "note": "no numeric parameters to perturb"}

    scores = [t["score"] for t in tried if t["ok"] and t["score"] is not None]
    fulls = [t["full_return"] for t in tried if t["ok"] and t["full_return"] is not None]

    # What the check did NOT do. A budget short of two runs per parameter leaves
    # some of them nudged one way only, and a verdict that says HOLDS without
    # saying so is a clean bill of health issued after half an examination.
    movable = perturbable(params)
    names = [k for k, _ in movable]
    # A flag has a single alternative, so demanding two runs for it would report
    # permanent partial coverage on a check that had in fact done everything.
    needed = sum(1 if is_flag(v) else 2 for _, v in movable)
    flags = {k for k, v in movable if is_flag(v)}
    directions: dict[str, set[str]] = {}
    for t in tried:
        base = params.get(t["param"])
        if isinstance(base, (int, float)):
            directions.setdefault(t["param"], set()).add(
                "down" if float(t["value"]) < float(base) else "up"
            )
    partial = sorted(
        k for k in names if k not in flags and len(directions.get(k, ())) == 1
    )
    untested = sorted(k for k in names if k not in directions)
    coverage = {
        "parameters": len(names),
        "runs_for_full_coverage": needed,
        "runs_spent": len(tried),
        "complete": not partial and not untested,
        "one_direction_only": partial,
        "not_tested": untested,
    }

    out: dict[str, Any] = {
        "checked": len(tried),
        "coverage": coverage,
        "chosen_score": score,
        "chosen_full_return": full_return,
        "neighbours": tried,
    }
    if not scores:
        out["note"] = "every perturbation failed to run; nothing can be concluded"
        return out

    out["median_neighbour_score"] = round(median(scores), 6)
    out["worst_neighbour_score"] = round(min(scores), 6)

    # How much of what the strategy asked for actually reached the book. A
    # perturbation the gate overrides cannot move the score, so a heavily clipped
    # strategy is guaranteed to look robust -- the flatness is the limit's, not
    # the strategy's. `_gate_activity`'s own docstring has said for a while that
    # "tuning its parameters is tuning the wrong thing"; this is the check that
    # was doing exactly that.
    clipped = (gate or {}).get("share_clipped_or_blocked")
    if isinstance(clipped, (int, float)):
        out["gate_clipped"] = float(clipped)
        out["gate_top_rules"] = (gate or {}).get("top_rules")
    if fulls:
        out["median_neighbour_full_return"] = round(median(fulls), 4)

    base = score
    if isinstance(base, (int, float)) and base > 0:
        retention = median(scores) / base
        out["score_retention"] = round(retention, 3)
        out["fragile"] = bool(retention < NEIGHBOUR_RETENTION)

        # A median hides the cliff. Naming the parameter that costs the most is
        # the actionable half of this check: "top_n 4 -> 3 costs 86% of the score"
        # tells you what the result actually rests on.
        ranked = [t for t in tried if t["ok"] and t["score"] is not None]
        if ranked:
            worst = min(ranked, key=lambda t: t["score"])
            out["most_sensitive"] = {
                "param": worst["param"],
                "from": params.get(worst["param"]),
                "to": worst["value"],
                "score": round(float(worst["score"]), 6),
                "costs": round(1.0 - float(worst["score"]) / base, 3),
            }
            best_nearby = max(ranked, key=lambda t: t["score"])
            if float(best_nearby["score"]) > base:
                out["better_nearby"] = {
                    "param": best_nearby["param"],
                    "to": best_nearby["value"],
                    "score": round(float(best_nearby["score"]), 6),
                }
        if out["fragile"]:
            full_note = ""
            if fulls and isinstance(out.get("chosen_full_return"), (int, float)):
                cf = float(out["chosen_full_return"])
                if cf > 0 and median(fulls) / cf > 0.8:
                    full_note = (
                        " Full-period return holds up across the same perturbations, so "
                        "the strategy is probably fine and it is the parameter choice "
                        "that is not -- pick from the middle of the range that works, or "
                        "stand behind the family rather than this exact point."
                    )
            out["verdict"] = (
                f"FRAGILE: moving one parameter about 10% drops the score to a median "
                f"{out['median_neighbour_score']} against your {base}, "
                f"{retention:.0%} of it. A result that depends on the exact value you "
                f"happened to land on is a property of this sample, not of the "
                f"strategy.{full_note}"
            )
        else:
            out["verdict"] = (
                f"HOLDS: nearby parameters keep a median {out['median_neighbour_score']} "
                f"against your {base} ({retention:.0%}), so the pick is on a plateau "
                f"rather than a spike."
            )

        # Said whichever way the verdict went: a plateau on the median can still
        # have one parameter it cannot survive, and that is worth knowing before
        # anyone trades it.
        sensitive = out.get("most_sensitive")
        if sensitive and float(sensitive["costs"]) >= SENSITIVE_PARAM_COST:
            out["verdict"] += (
                f" Most sensitive parameter: {sensitive['param']} "
                f"{sensitive['from']} -> {sensitive['to']} costs "
                f"{float(sensitive['costs']):.0%} of the score."
            )
        better = out.get("better_nearby")
        if better:
            out["verdict"] += (
                f" Note that {better['param']}={better['to']} scored "
                f"{better['score']}, above the value you picked -- the point you "
                f"landed on was not even the local best."
            )
        if fulls and isinstance(out.get("chosen_full_return"), (int, float)):
            cf = float(out["chosen_full_return"])
            if cf > 0:
                full_ret = median(fulls) / cf
                out["full_return_retention"] = round(full_ret, 3)
                if full_ret > 0.85 and retention < 0.85:
                    out["verdict"] += (
                        f" Full-period return holds at {full_ret:.0%} across the same "
                        f"perturbations while the score keeps only {retention:.0%}, so "
                        f"what is fragile is the score, not the strategy."
                    )

    # Before coverage, because this one can invalidate the result outright rather
    # than merely weaken it.
    clipped = out.get("gate_clipped")
    if isinstance(clipped, (int, float)) and clipped >= CLIPPING_CONFOUNDS:
        rules = ", ".join(
            str(r.get("rule")) for r in (out.get("gate_top_rules") or [])[:2] if r.get("rule")
        )
        out["robustness_confounded"] = True
        out["verdict"] = (out.get("verdict") or "") + (
            f" CONFOUNDED: the risk gate clipped or blocked {clipped:.0%} of this "
            f"strategy's intents"
            + (f" (mostly {rules})" if rules else "")
            + ". Parameters the gate overrides cannot move the score, so this "
            "check measures the limit rather than the strategy and its verdict is "
            "not evidence either way. What was actually backtested is the strategy "
            "filtered through the gate -- if that is not what you meant to test, "
            "raise the limit or size the strategy to fit inside it."
        ).rstrip()

    # Appended last so it is the note the reader ends on: a HOLDS covering half
    # the directions is a weaker claim than a HOLDS covering all of them, and the
    # two must not read the same.
    if not coverage["complete"]:
        gaps = []
        if coverage["one_direction_only"]:
            gaps.append(
                f"{', '.join(coverage['one_direction_only'])} nudged one direction only"
            )
        if coverage["not_tested"]:
            gaps.append(f"{', '.join(coverage['not_tested'])} not tested at all")
        out["verdict"] = (out.get("verdict") or "") + (
            f" PARTIAL COVERAGE: {coverage['runs_spent']} of "
            f"{coverage['runs_for_full_coverage']} runs needed for "
            f"{coverage['parameters']} parameters -- {'; '.join(gaps)}. A cliff on an "
            f"untried side would not have been seen, so this is weaker evidence than a "
            f"complete check; raise --neighbourhood-runs to "
            f"{coverage['runs_for_full_coverage']} to close it."
        ).rstrip()
    return out
