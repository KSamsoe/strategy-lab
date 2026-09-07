/**
 * Did it beat buying and holding?
 *
 * The single question every backtest has to answer before any of its other
 * numbers matter, and the one most easily lost in a wall of metrics. A strategy
 * with a respectable Sharpe that trails the index is not a strategy, it is an
 * expensive way to own less of the index — so this states the verdict in words
 * before it shows the arithmetic.
 *
 * The metrics come from the run itself (`compute_metrics` fills `benchmark_*`,
 * `alpha`, `beta` and `excess_return` whenever the config names a benchmark), so
 * this panel is a renderer, not a second opinion.
 */

import type { Metrics } from "../lib/types";
import { MISSING, delta, pct, ratio } from "../lib/format";
import { Badge, Empty, Panel, cx } from "./ui";

export interface VsBenchmarkProps {
  metrics: Metrics;
  /** Ticker the run named as its benchmark, for the label. */
  benchmark?: string | null;
}

/** True when the run actually carries a benchmark comparison. */
export function hasBenchmark(m: Metrics): boolean {
  return typeof m.benchmark_total_return === "number";
}

interface Row {
  label: string;
  strategy: number | null | undefined;
  bench: number | null | undefined;
  format: (v: number | null | undefined) => string;
  /** Lower is better — drawdown and volatility invert the comparison. */
  lowerIsBetter?: boolean;
}

export function VsBenchmark({ metrics: m, benchmark }: VsBenchmarkProps) {
  const label = benchmark ? `buy & hold ${benchmark}` : "buy & hold";

  if (!hasBenchmark(m)) {
    return (
      <Panel title="vs buy & hold">
        <Empty cmd="add `benchmark: SPY` to the config">
          this run named no benchmark, so there is nothing to beat
        </Empty>
      </Panel>
    );
  }

  const rows: Row[] = [
    {
      label: "total return",
      strategy: m.total_return,
      bench: m.benchmark_total_return,
      format: (v) => delta(v),
    },
    { label: "CAGR", strategy: m.cagr, bench: m.benchmark_cagr, format: (v) => delta(v) },
  ];

  const excess = m.excess_return ?? null;
  const won = typeof excess === "number" ? excess > 0 : null;

  return (
    <Panel
      title="vs buy & hold"
      right={
        won === null ? null : (
          <Badge kind={won ? "good" : "bad"}>{won ? "beat the market" : "lost to the market"}</Badge>
        )
      }
    >
      <div className="px-3 py-2 flex flex-col gap-3">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="unit text-left font-medium pb-1">metric</th>
              <th className="unit text-right font-medium pb-1">this run</th>
              <th className="unit text-right font-medium pb-1">{label}</th>
              <th className="unit text-right font-medium pb-1">difference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const a = r.strategy;
              const b = r.bench;
              const diff =
                typeof a === "number" && typeof b === "number" ? a - b : null;
              const better =
                diff === null ? null : r.lowerIsBetter ? diff < 0 : diff > 0;
              return (
                <tr key={r.label} className="border-t border-slate/50">
                  <td className="py-1 text-ash">{r.label}</td>
                  <td className="num py-1 text-chalk">{r.format(a)}</td>
                  <td className="num py-1 text-ash">{r.format(b)}</td>
                  <td
                    className={cx(
                      "num py-1",
                      better === null ? "text-ash-dim" : better ? "text-moss" : "text-ember",
                    )}
                  >
                    {r.format(diff)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
          <span className="text-ash">
            alpha <span className={cx("num", (m.alpha ?? 0) > 0 ? "text-moss" : "text-ember")}>
              {typeof m.alpha === "number" ? delta(m.alpha) : MISSING}
            </span>
            <span className="text-ash-dim"> — return above what its beta explains</span>
          </span>
          <span className="text-ash">
            beta <span className="num text-chalk">{ratio(m.beta ?? null)}</span>
            <span className="text-ash-dim">
              {" "}
              — {typeof m.beta === "number" && m.beta < 1 ? "less" : "more"} market exposure than
              holding it outright
            </span>
          </span>
          <span className="text-ash">
            exposure <span className="num text-chalk">{pct(m.exposure ?? null)}</span>
            <span className="text-ash-dim"> — of the time invested at all</span>
          </span>
        </div>

        {won === false ? (
          <p className="text-xs text-ash-dim m-0">
            Trailing the benchmark is the common outcome and worth taking at face value. Check
            whether the shortfall is the strategy or the cost of being out of the market —
            exposure below 100% caps the upside by construction.
          </p>
        ) : null}
      </div>
    </Panel>
  );
}
