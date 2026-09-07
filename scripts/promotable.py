"""Which runs from a research session can be promoted into the library.

    python scripts/promotable.py runs/research-20260823T142800-console

Promotion keys on a run, not a file, because a session rewrites its workspace
files in place -- so the question is not "is this file still here" but "did this
run archive the code it executed". Runs made before source archiving did not, and
their exact code is unrecoverable; this says which ones those are.

Promote with:  lab strategy promote <run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(workspace: str) -> int:
    ws = Path(workspace)
    session = json.loads((ws / "session.json").read_text(encoding="utf-8"))
    runs_root = REPO / "runs"
    chosen = session.get("satisfied_with")

    print(f"workspace     {ws}")
    print(f"stands behind {chosen or '(nothing)'}\n")
    print(f"{'run_id':<44} {'strategy':<20} {'score':>9}  {'promotable?':<24}")
    print("-" * 102)

    ok = 0
    experiments = session.get("experiments") or []
    for exp in experiments:
        run_id, name = exp.get("run_id") or "", exp.get("strategy") or ""
        directory = runs_root / run_id
        if not (directory / "config.json").exists():
            verdict = "no artifacts"
        elif (directory / "strategy.py").exists():
            verdict = "YES"
            ok += 1
        else:
            verdict = "NO - source not archived"

        score = exp.get("score")
        score_s = "n/a" if score is None else f"{float(score):.4f}"
        mark = " <-- chosen" if run_id == chosen else ""
        print(f"{run_id:<44} {name:<20} {score_s:>9}  {verdict:<24}{mark}")

    print(f"\n{ok} of {len(experiments)} runs carry their source.")

    def promotable(e):
        return (runs_root / str(e.get("run_id")) / "strategy.py").exists()

    def beats_market(e):
        """Positive against the benchmark on BOTH windows.

        Not the top fitness score. Ranking on the score alone recommends exactly
        the candidate this whole check exists to catch: on a real session the
        highest scorer was a low-exposure book that lost to SPY by 18pp over the
        full period and 6pp out of sample. A high ratio on a strategy that makes
        less money than the benchmark is a worse strategy wearing a better number.
        """
        a, b = e.get("oos_return_vs_market"), e.get("full_return_vs_market")
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0

    if ok:
        stood_behind = next(
            (e for e in experiments if e.get("run_id") == chosen and promotable(e)), None
        )
        if stood_behind:
            print(
                f"\nThe run this session stood behind:"
                f"\n  lab strategy promote {stood_behind['run_id']}"
            )

        beat = sorted(
            (e for e in experiments if promotable(e) and beats_market(e)),
            key=lambda e: float(e.get("full_return_vs_market") or 0),
            reverse=True,
        )
        if beat:
            print("\nRuns that beat the benchmark on BOTH windows, best full-period first:")
            for e in beat:
                mark = "  <- chosen" if e.get("run_id") == chosen else ""
                print(
                    f"  {e['run_id']:<44} oos {float(e['oos_return_vs_market']):+.4f}  "
                    f"full {float(e['full_return_vs_market']):+.4f}{mark}"
                )
        else:
            print(
                "\nNothing here beat the benchmark on both windows. That is a result, "
                "not a reason to promote the best of a bad set."
            )

        top = max(
            (e for e in experiments if e.get("score") is not None and promotable(e)),
            key=lambda e: float(e["score"]),
            default=None,
        )
        if top and not beats_market(top):
            print(
                f"\nNote: the highest fitness score ({top['run_id']}, {float(top['score']):.4f}) "
                f"loses to the benchmark on return -- oos "
                f"{float(top.get('oos_return_vs_market') or 0):+.4f}, full "
                f"{float(top.get('full_return_vs_market') or 0):+.4f}. Do not promote it on "
                f"the strength of the ratio."
            )
    if ok < len(experiments):
        print(
            "\nRuns without archived source predate the archiving change. Re-run the\n"
            "strategy to get a promotable run -- the code those runs executed is gone."
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
