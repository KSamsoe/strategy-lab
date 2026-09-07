"""Seed the findings ledger with the method lessons this lab was built around.

Run once on a fresh checkout:

    python scripts/seed_findings.py

These are the platform-level findings -- things that are true of the *method*
rather than of any one strategy or dataset -- so they carry no run ids. The
strategy-specific findings that produced them are described in docs/mcp.md;
yours will accumulate as you work. Re-running is safe: titles already present
are skipped.
"""

from __future__ import annotations

from lab.mcp import tools_findings as F

SEED = [
    dict(
        title="A strategy's edge is very often its ticker list",
        claim=(
            "Before believing an edge, run compare_universes with everything but the "
            "tickers held constant. The case that produced this rule: a momentum tilt "
            "beat SPY by 88 points on the 21 megacap-technology names it was discovered "
            "on and lost to SPY by 5 on a 36-name, 8-sector universe -- same source, "
            "same window, same parameters. No research loop could have found it because "
            "the loop was not allowed to change the universe."
        ),
        tags=["method", "universe", "negative-result"],
        status="confirmed",
    ),
    dict(
        title="Measure the data before writing a strategy against it",
        claim=(
            "The most valuable findings here came from signal_scan and "
            "conditional_returns, not from backtests: which horizons carry cross-sectional "
            "signal, and which conditions change forward returns. A backtest answers "
            "'how did this do'; a measurement answers 'is there anything here at all', "
            "and it costs seconds."
        ),
        tags=["method", "exploration"],
        status="confirmed",
    ),
    dict(
        title="An unpaired bootstrap understates evidence against a correlated benchmark",
        claim=(
            "Strategy and index live through the same crashes, so resampling them "
            "independently gives each a wide interval and the two overlap almost "
            "whatever the true difference is. On a real 21-year run the unpaired 90% "
            "interval [0.57, 1.23] contained the benchmark's 0.66 and read as no "
            "evidence; the paired difference was [+0.06, +0.43] with P(strategy <= "
            "benchmark) = 1.2%. Use bootstrap_vs_benchmark for 'is it better than the index'."
        ),
        tags=["method", "bootstrap", "benchmark"],
        status="confirmed",
    ),
    dict(
        title="A permutation null must reorder every ticker by the same block indices",
        claim=(
            "Shuffling each ticker independently destroys cross-sectional correlation "
            "(0.41 -> 0.0003) and collapses equal-weight volatility (18.6% -> 4.9%), so "
            "the null portfolio diversifies away its own risk and scores Sharpe 3.1. Every "
            "shuffle beat the real run and p came back 1.000 -- an artefact of the null. "
            "Synchronised blocks hold correlation at 0.39 and volatility at 17.3%."
        ),
        tags=["method", "permutation"],
        evidence={"independent": {"corr": 0.0003, "ew_vol": 0.049, "null_sharpe": 3.116},
                  "synchronised": {"corr": 0.3924, "ew_vol": 0.1727},
                  "real": {"corr": 0.4102, "ew_vol": 0.1859}},
        status="confirmed",
    ),
    dict(
        title="A high permutation p-value with a null that still beats the index means construction, not signal",
        claim=(
            "The null keeps the position count, the weighting and the vol target and "
            "destroys only the signal. When the null already beats the benchmark and "
            "removing the signal costs less than the null's spread, the strategy's "
            "advantage is diversification and sizing -- a finding, not a failure, and a "
            "different proposition from a stock-picking edge."
        ),
        tags=["method", "permutation", "construction"],
        status="confirmed",
    ),
    dict(
        title="Count trials from the registry, never from memory",
        claim=(
            "deflated_sharpe asked the caller how many configurations were tried, and the "
            "caller under-counted every time. Every registered backtest over the same "
            "universe, dates and timeframe -- under any strategy name -- is a look at the "
            "same holdout. A session that authored nine strategies and kept one took "
            "nine looks; the survivor's own name appeared on two of them."
        ),
        tags=["method", "deflated-sharpe", "multiple-testing"],
        status="confirmed",
    ),
    dict(
        title="An empty limits block silently caps gross exposure at 40%",
        claim=(
            "With no limits in the config the global defaults (5% per name x 8 positions) "
            "impose a hard 40% gross ceiling. A brief demanding full exposure then produces "
            "books reporting 17% exposure that look like de-levering. State the limits in "
            "the config and in the brief so the strategy sizes to fit; read "
            "gate.share_clipped_or_blocked on every result."
        ),
        tags=["platform", "gate", "limits", "pitfall"],
        evidence={"default_max_position_pct": 0.05, "default_max_positions": 8},
        status="confirmed",
    ),
    dict(
        title="Intraday ideas die on commission before they die on signal",
        claim=(
            "A momentum book on 15-minute bars paid about $3,000 in commission across "
            "2,800 trades and turned a market-beating signal into a losing strategy. The "
            "question for any intraday idea is at what cost the edge disappears, not "
            "whether it exists at zero cost; run cost_sensitivity first."
        ),
        tags=["intraday", "costs", "negative-result"],
        status="confirmed",
    ),
    dict(
        title="Judge on the full period as well as the holdout",
        claim=(
            "A holdout of ~230 daily bars carries a Sharpe standard error near 1.0, wider "
            "than most parameter sweeps. A strategy that beats its benchmark by 200 points "
            "over four years and trails it on the last 20% is out of favour, not broken -- "
            "but shown only the holdout, an agent called it a loss. Every result here "
            "reports both windows and the error bar on the score."
        ),
        tags=["method", "holdout", "error-bar"],
        status="confirmed",
    ),
]


def main() -> None:
    have = {f["title"] for f in F.findings_search(limit=1000, include_superseded=True)["findings"]}
    added = 0
    for item in SEED:
        if item["title"] in have:
            continue
        F.findings_record(author="seed", **item)
        added += 1
    total = F.findings_search(limit=1000)["count"]
    print(f"seeded {added} findings; the ledger holds {total}")


if __name__ == "__main__":
    main()
