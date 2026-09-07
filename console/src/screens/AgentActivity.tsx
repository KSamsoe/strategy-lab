/**
 * Agent activity — two screens wearing one route, because the two agent
 * patterns are two different questions.
 *
 * Pattern A (author loop) asks "is this thing converging or just churning?",
 * which is a shape question, so it gets a trajectory chart with the running
 * best drawn over the per-iteration value. A flat best-line under a noisy
 * value-line IS churn, visible before you read a single number.
 *
 * Pattern B (model in the runtime path) asks "what did it cost and can I
 * believe any of it?", which is an accounting question, so it gets the ledger
 * in `AgentLedger` — cost against budget, and the contamination warning that
 * makes a Pattern-B backtest legible as the plumbing test it is.
 *
 * Everything the model wrote — rationales, diffs, prompts — is rendered as
 * text. It is untrusted input here exactly as it is at the risk gate.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import type { LineageStep, Metrics } from "../lib/types";
import { count, money, num, ratio, shortId, titleize } from "../lib/format";
import { ResearchView } from "../components/ResearchView";
import { AgentLedger } from "../components/AgentLedger";
import {
  Badge,
  Empty,
  ErrorNote,
  Loading,
  Panel,
  Table,
  Td,
  Th,
  Tr,
  cx,
} from "../components/ui";

// --- pure derivations (exported: the chart is only as trustworthy as these) ---

/**
 * Metrics where a smaller number is the better one. Max drawdown is absent on
 * purpose: it is stored negative, so "higher is better" is already correct for
 * it and special-casing it would invert the comparison.
 */
const LOWER_IS_BETTER = new Set(["volatility", "turnover", "max_drawdown_duration_days"]);

/** Metrics is an index signature of `unknown`; only finite numbers plot. */
export function metricValue(metrics: Metrics, key: string): number | null {
  const v = metrics[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Fitness metric the loop was scored on, inferred from what the steps carry. */
export function pickMetric(steps: LineageStep[]): string {
  for (const k of ["oos_sharpe", "sharpe", "oos_total_return", "total_return"]) {
    if (steps.some((s) => metricValue(s.metrics, k) !== null)) return k;
  }
  return "oos_sharpe";
}

/** Every numeric metric any step reports, so the operator can re-score by eye. */
export function availableMetrics(steps: LineageStep[]): string[] {
  const keys = new Set<string>();
  for (const s of steps) {
    for (const k of Object.keys(s.metrics)) {
      if (metricValue(s.metrics, k) !== null) keys.add(k);
    }
  }
  return [...keys].sort();
}

export interface TrajectoryPoint {
  n: number;
  runId: string;
  /** The step's own score. Null when the iteration errored or skipped scoring. */
  value: number | null;
  /** Best score seen up to and including this step. Carries across nulls. */
  best: number | null;
  isNewBest: boolean;
  cumCost: number;
}

/**
 * Per-iteration value plus the running best.
 *
 * The running best carries forward through nulls: an iteration that failed to
 * produce a metric did not un-discover the best strategy so far, and resetting
 * the line there would draw a cliff that never happened.
 */
export function trajectory(steps: LineageStep[], metric: string): TrajectoryPoint[] {
  const better = LOWER_IS_BETTER.has(metric)
    ? (a: number, b: number) => a < b
    : (a: number, b: number) => a > b;

  let best: number | null = null;
  let cum = 0;
  return steps.map((s) => {
    const value = metricValue(s.metrics, metric);
    cum += typeof s.cost_usd === "number" && Number.isFinite(s.cost_usd) ? s.cost_usd : 0;
    const isNewBest = value !== null && (best === null || better(value, best));
    if (isNewBest) best = value;
    return { n: s.n, runId: s.run_id, value, best, isNewBest, cumCost: cum };
  });
}

export type Convergence = "converging" | "churning" | "flat" | "thin";

export interface ConvergenceRead {
  verdict: Convergence;
  /** New bests after the first one. The first is discovery, not improvement. */
  improvements: number;
  /** Iterations burned since the last new best. The churn number. */
  sinceBest: number;
  /** Range of the per-iteration values: churn is spread without progress. */
  spread: number;
}

export function readConvergence(points: TrajectoryPoint[]): ConvergenceRead {
  const vals = points.map((p) => p.value).filter((v): v is number => v !== null);
  const newBests = points.filter((p) => p.isNewBest);
  const lastBestAt = newBests.length ? points.lastIndexOf(newBests[newBests.length - 1]) : -1;
  const sinceBest = lastBestAt < 0 ? points.length : points.length - 1 - lastBestAt;
  const spread = vals.length ? Math.max(...vals) - Math.min(...vals) : 0;
  const improvements = Math.max(0, newBests.length - 1);

  if (vals.length < 3) return { verdict: "thin", improvements, sinceBest, spread };
  // Everything landed on the same number: not churn, but not progress either.
  if (spread <= 1e-9) return { verdict: "flat", improvements, sinceBest, spread };
  // Half the budget spent without beating the incumbent (min 3) reads as churn.
  const stall = Math.max(3, Math.ceil(points.length / 2));
  if (sinceBest >= stall) return { verdict: "churning", improvements, sinceBest, spread };
  return { verdict: "converging", improvements, sinceBest, spread };
}

// --- diffs -------------------------------------------------------------------

export type DiffLineKind = "add" | "del" | "hunk" | "meta" | "ctx";

export function diffLineKind(line: string): DiffLineKind {
  if (line.startsWith("@@")) return "hunk";
  if (line.startsWith("+++") || line.startsWith("---")) return "meta";
  if (line.startsWith("+")) return "add";
  if (line.startsWith("-")) return "del";
  return "ctx";
}

const DIFF_TONE: Record<DiffLineKind, string> = {
  add: "text-moss",
  del: "text-ember",
  hunk: "text-teal",
  meta: "text-ash-dim",
  ctx: "text-ash",
};

export interface ParamChange {
  key: string;
  before: unknown;
  after: unknown;
  kind: "added" | "removed" | "changed";
}

/**
 * Key-level diff of two param maps. A tuned lookback is a one-line change; a
 * unified diff of two JSON dumps buries it in braces, so params get their own
 * comparison. Structural equality is by JSON, which is exact for the scalars
 * and short lists params actually hold.
 */
export function paramsDiff(
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): ParamChange[] {
  const out: ParamChange[] = [];
  for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
    const hadKey = key in before;
    const hasKey = key in after;
    const a = before[key];
    const b = after[key];
    if (hadKey && hasKey) {
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        out.push({ key, before: a, after: b, kind: "changed" });
      }
    } else if (hasKey) {
      out.push({ key, before: undefined, after: b, kind: "added" });
    } else {
      out.push({ key, before: a, after: undefined, kind: "removed" });
    }
  }
  return out.sort((x, y) => x.key.localeCompare(y.key));
}

/** Param values are mixed scalars; numbers still go through the formatters. */
export function paramText(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? count(v) : num(v, 4);
  if (typeof v === "string") return JSON.stringify(v);
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

function ParamsDiff({ changes }: { changes: ParamChange[] }) {
  if (!changes.length) {
    return <p className="mono text-xs text-ash-dim px-3 py-2">no param change on this step</p>;
  }
  return (
    <div className="scroll-x px-3 py-2">
      <table className="mono text-xs border-collapse">
        <tbody>
          {changes.map((c) => (
            <tr key={c.key} data-kind={c.kind}>
              <td
                className={cx(
                  "pr-2 select-none",
                  c.kind === "added" ? "text-moss" : c.kind === "removed" ? "text-ember" : "text-ash-dim",
                )}
              >
                {c.kind === "added" ? "+" : c.kind === "removed" ? "-" : "~"}
              </td>
              <td className="pr-3 text-ash whitespace-nowrap">{c.key}</td>
              <td className="pr-2 text-ember whitespace-nowrap" data-slot="before">
                {c.kind === "added" ? "" : paramText(c.before)}
              </td>
              <td className="pr-2 text-ash-dim select-none">→</td>
              <td className="text-moss whitespace-nowrap" data-slot="after">
                {c.kind === "removed" ? "" : paramText(c.after)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Unified diff, rendered as text. Model-authored content never becomes markup. */
function FileDiff({ diff }: { diff: string }) {
  const lines = useMemo(() => diff.replace(/\n$/, "").split("\n"), [diff]);
  return (
    <div className="scroll-x px-3 py-2">
      <pre className="mono text-xs leading-[1.45] m-0">
        {lines.map((line, i) => {
          const kind = diffLineKind(line);
          return (
            <div key={i} data-diff={kind} className={DIFF_TONE[kind]}>
              {line === "" ? " " : line}
            </div>
          );
        })}
      </pre>
    </div>
  );
}

// --- trajectory chart --------------------------------------------------------

/**
 * Container width, measured. The chart draws in real pixels rather than a
 * scaled viewBox so a 1px hairline stays 1px and point markers stay round.
 */
function useWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth || 720);
    if (typeof ResizeObserver === "undefined") return; // jsdom, and old Safari
    const ro = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width ?? 0;
      if (next > 0) setW(next);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const PAD = { l: 46, r: 12, t: 12, b: 20 };
const CHART_H = 208;

function Trajectory({
  points,
  metric,
  selected,
  onSelect,
}: {
  points: TrajectoryPoint[];
  metric: string;
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  const [ref, width] = useWidth();
  const iw = Math.max(80, width - PAD.l - PAD.r);
  const ih = CHART_H - PAD.t - PAD.b;

  const scale = useMemo(() => {
    const vals: number[] = [];
    for (const p of points) {
      if (p.value !== null) vals.push(p.value);
      if (p.best !== null) vals.push(p.best);
    }
    const lo = vals.length ? Math.min(...vals) : 0;
    const hi = vals.length ? Math.max(...vals) : 1;
    const span = hi - lo || Math.abs(hi) || 1;
    const min = lo - span * 0.12;
    const max = hi + span * 0.12;
    const x = (i: number) =>
      PAD.l + (points.length <= 1 ? iw / 2 : (i / (points.length - 1)) * iw);
    const y = (v: number) => PAD.t + ih - ((v - min) / (max - min)) * ih;
    return { min, max, x, y };
  }, [points, iw, ih]);

  // Value line breaks at nulls: a failed iteration is a gap, not a straight
  // line through territory the loop never visited.
  const valueSegments = useMemo(() => {
    const segs: string[] = [];
    let cur: string[] = [];
    points.forEach((p, i) => {
      if (p.value === null) {
        if (cur.length > 1) segs.push(cur.join(" "));
        cur = [];
        return;
      }
      cur.push(`${cur.length ? "L" : "M"}${scale.x(i)} ${scale.y(p.value)}`);
    });
    if (cur.length > 1) segs.push(cur.join(" "));
    return segs;
  }, [points, scale]);

  // Best line is a step: it holds flat until something actually beats it, and
  // that flat stretch is the churn signal.
  const bestPath = useMemo(() => {
    const d: string[] = [];
    let prevY: number | null = null;
    points.forEach((p, i) => {
      if (p.best === null) return;
      const x = scale.x(i);
      const y = scale.y(p.best);
      if (prevY === null) d.push(`M${x} ${y}`);
      else d.push(`L${x} ${prevY}`, `L${x} ${y}`);
      prevY = y;
    });
    if (d.length === 1) return "";
    return d.join(" ");
  }, [points, scale]);

  const ticks = [scale.max, (scale.max + scale.min) / 2, scale.min];

  return (
    <div ref={ref} className="w-full">
      <svg
        width={width}
        height={CHART_H}
        role="img"
        aria-label={`${metric} per iteration with running best`}
        className="block"
      >
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD.l}
              x2={PAD.l + iw}
              y1={scale.y(t)}
              y2={scale.y(t)}
              className="text-slate"
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray={i === 1 ? "2 4" : undefined}
            />
            <text
              x={PAD.l - 6}
              y={scale.y(t) + 3}
              textAnchor="end"
              className="mono text-ash-dim"
              fill="currentColor"
              fontSize={10}
            >
              {ratio(t)}
            </text>
          </g>
        ))}

        {valueSegments.map((d, i) => (
          <path
            key={i}
            d={d}
            fill="none"
            className="text-ash-dim"
            stroke="currentColor"
            strokeWidth={1}
          />
        ))}

        {bestPath ? (
          <path
            d={bestPath}
            data-series="best"
            fill="none"
            className="text-moss"
            stroke="currentColor"
            strokeWidth={1.5}
          />
        ) : null}

        {points.map((p, i) =>
          p.value === null ? null : (
            <circle
              key={p.n}
              cx={scale.x(i)}
              cy={scale.y(p.value)}
              r={p.isNewBest ? 3.25 : 2}
              data-point={p.n}
              data-best={p.isNewBest ? "1" : "0"}
              className={p.isNewBest ? "text-moss" : "text-ash"}
              fill="currentColor"
            />
          ),
        )}

        {selected !== null &&
          points.some((p) => p.n === selected) &&
          (() => {
            const i = points.findIndex((p) => p.n === selected);
            return (
              <line
                x1={scale.x(i)}
                x2={scale.x(i)}
                y1={PAD.t}
                y2={PAD.t + ih}
                className="text-teal"
                stroke="currentColor"
                strokeWidth={1}
              />
            );
          })()}

        {points.map((p, i) => {
          const half = points.length <= 1 ? iw / 2 : iw / (points.length - 1) / 2;
          return (
            <rect
              key={`hit-${p.n}`}
              x={scale.x(i) - half}
              y={PAD.t}
              width={half * 2}
              height={ih}
              fill="transparent"
              className="cursor-pointer"
              onClick={() => onSelect(p.n)}
            >
              <title>{`iteration ${p.n} · ${metric} ${ratio(p.value)}`}</title>
            </rect>
          );
        })}
      </svg>
    </div>
  );
}

// --- pattern A ---------------------------------------------------------------

const VERDICT: Record<Convergence, { label: string; kind: "good" | "warn" | "neutral" }> = {
  converging: { label: "converging", kind: "good" },
  churning: { label: "churning", kind: "warn" },
  flat: { label: "no movement", kind: "warn" },
  thin: { label: "too few iterations", kind: "neutral" },
};

function LineageView({ strategy }: { strategy: string }) {
  const q = useQuery({
    // Lineage is a view over finished runs: immutable once the loop stops, and
    // a running loop is re-read by navigating, not by a timer.
    queryKey: ["agent", "lineage", strategy],
    queryFn: () => api.lineage(strategy),
    enabled: Boolean(strategy),
    retry: false,
  });

  const steps = useMemo(() => q.data?.steps ?? [], [q.data]);
  const [metricOverride, setMetricOverride] = useState<string | null>(null);
  const metric = metricOverride ?? pickMetric(steps);
  const points = useMemo(() => trajectory(steps, metric), [steps, metric]);
  const read = useMemo(() => readConvergence(points), [points]);
  const metrics = useMemo(() => availableMetrics(steps), [steps]);

  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => setSelected(null), [strategy]);
  const activeN = selected ?? (steps.length ? steps[steps.length - 1].n : null);
  const step = steps.find((s) => s.n === activeN) ?? null;
  const stepIdx = step ? steps.indexOf(step) : -1;
  const prevParams = stepIdx > 0 ? steps[stepIdx - 1].params : {};
  const changes = step ? paramsDiff(prevParams, step.params) : [];
  const totalCost = points.length ? points[points.length - 1].cumCost : 0;

  if (q.isLoading) return <Loading what="lineage" />;
  if (q.error) return <ErrorNote error={q.error} />;
  if (!steps.length) {
    return (
      <Empty cmd={`lab agent author --seed strategies/${strategy}.py --grid cfg/${strategy}_grid.yaml -n 10`}>
        no author-loop iterations for {strategy}
      </Empty>
    );
  }

  const verdict = VERDICT[read.verdict];

  return (
    <div className="flex flex-col gap-2 p-2 min-h-0">
      <Panel
        title="convergence"
        right={
          <div className="flex items-center gap-3">
            <span className="unit">
              iterations <span className="num text-chalk">{count(steps.length)}</span>
            </span>
            <span className="unit">
              since best <span className="num text-chalk">{count(read.sinceBest)}</span>
            </span>
            <span className="unit">
              spend <span className="num text-chalk">{money(totalCost, 2)}</span>
            </span>
            <Badge kind={verdict.kind} title="new bests after the first, vs iterations since the last one">
              {verdict.label}
            </Badge>
            <select
              value={metric}
              onChange={(e) => setMetricOverride(e.target.value)}
              aria-label="fitness metric"
              className="mono text-xs h-5 bg-graphite text-chalk border border-slate rounded px-1"
            >
              {(metrics.includes(metric) ? metrics : [metric, ...metrics]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        }
      >
        <Trajectory points={points} metric={metric} selected={activeN} onSelect={setSelected} />
        <div className="flex items-center gap-4 px-3 pb-2 unit">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-px bg-ash-dim" aria-hidden /> per iteration
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-moss" aria-hidden /> running best
          </span>
          <span className="text-ash-dim normal-case tracking-normal">
            a flat best-line under a noisy value-line is churn, not search
          </span>
        </div>
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2 min-h-0">
        <Panel title={`iterations · ${titleize(metric)}`} bodyClassName="max-h-[46vh] overflow-auto">
          <Table>
            <thead>
              <tr>
                <Th align="right">n</Th>
                <Th>run</Th>
                <Th align="right">{metric}</Th>
                <Th align="right">best</Th>
                <Th align="right">cum $</Th>
                <Th>rationale</Th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => {
                const s = steps[points.indexOf(p)];
                return (
                  <Tr key={p.n} selected={p.n === activeN} onClick={() => setSelected(p.n)}>
                    <Td align="right">
                      <button
                        type="button"
                        onClick={() => setSelected(p.n)}
                        className="num text-ash hover:text-chalk"
                        aria-label={`iteration ${p.n}`}
                      >
                        {count(p.n)}
                      </button>
                    </Td>
                    <Td mono>
                      <Link
                        to="/runs/$runId"
                        params={{ runId: p.runId }}
                        className="text-teal no-underline hover:underline"
                      >
                        {shortId(p.runId)}
                      </Link>
                    </Td>
                    <Td align="right" mono className={p.isNewBest ? "text-moss" : "text-chalk"}>
                      {ratio(p.value)}
                    </Td>
                    <Td align="right" mono className="text-ash">
                      {ratio(p.best)}
                    </Td>
                    <Td align="right" mono className="text-ash">
                      {money(p.cumCost, 2)}
                    </Td>
                    <Td className="text-ash max-w-[22ch] overflow-hidden text-ellipsis" title={s.rationale}>
                      {s.rationale || "—"}
                    </Td>
                  </Tr>
                );
              })}
            </tbody>
          </Table>
        </Panel>

        <Panel
          title={step ? `step ${step.n} · ${shortId(step.run_id)}` : "step"}
          right={
            step ? (
              <Link
                to="/runs/$runId"
                params={{ runId: step.run_id }}
                className="mono text-xs text-teal no-underline hover:underline"
              >
                open run →
              </Link>
            ) : null
          }
          bodyClassName="max-h-[46vh] overflow-auto"
        >
          {!step ? (
            <Empty>pick an iteration</Empty>
          ) : (
            <div className="flex flex-col divide-y divide-slate">
              <div>
                <h3 className="unit px-3 pt-2">params</h3>
                <ParamsDiff changes={changes} />
              </div>
              {step.diff ? (
                <div>
                  <h3 className="unit px-3 pt-2">strategy file</h3>
                  <FileDiff diff={step.diff} />
                </div>
              ) : null}
              <div className="px-3 py-2">
                <h3 className="unit">rationale</h3>
                {/* Model-authored. Text node, never markup. */}
                <p className="text-xs text-ash leading-relaxed whitespace-pre-wrap m-0 mt-1">
                  {step.rationale || "the model recorded no rationale for this step"}
                </p>
              </div>
              <div className="flex items-center gap-4 px-3 py-2">
                <span className="unit">
                  step cost <span className="num text-chalk">{money(step.cost_usd ?? 0, 4)}</span>
                </span>
                <span className="unit">
                  {metric} <span className="num text-chalk">{ratio(metricValue(step.metrics, metric))}</span>
                </span>
                {step.metrics.contaminated ? <Badge kind="warn">contaminated</Badge> : null}
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

// --- screen ------------------------------------------------------------------

type Pattern = "A" | "B" | "R";

const PATTERN_NOTE: Record<Pattern, string> = {
  R: "open-ended sessions: what it tried, what it concluded, and why. Live while it runs.",
  A: "the model authors and mutates the strategy offline; what ships to paper is an ordinary deterministic artifact with no LLM in the runtime path",
  B: "the model is called inside on_bar; it proposes, the risk gate disposes, and every historical backtest of it is contaminated by training data",
};

export default function AgentActivity() {
  const strategies = useQuery({
    queryKey: ["strategies"],
    queryFn: api.strategies,
    retry: false,
  });

  const [pattern, setPattern] = useState<Pattern>("R");
  const [strategy, setStrategy] = useState<string>("");
  const names = useMemo(
    () => (strategies.data?.strategies ?? []).map((s) => s.name),
    [strategies.data],
  );
  useEffect(() => {
    if (!strategy && names.length) setStrategy(names[0]);
  }, [names, strategy]);

  return (
    <div className="h-full flex flex-col min-h-0">
      <header className="shrink-0 flex items-center gap-3 h-9 px-3 border-b border-slate bg-gunmetal">
        <span className="unit">strategy</span>
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          aria-label="strategy"
          disabled={!names.length}
          className="mono text-xs h-6 bg-graphite text-chalk border border-slate rounded px-1 min-w-40"
        >
          {names.length ? null : <option value="">—</option>}
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-1" role="tablist" aria-label="agent pattern">
          {(["R", "A", "B"] as const).map((p) => (
            <button
              key={p}
              type="button"
              role="tab"
              aria-selected={pattern === p}
              onClick={() => setPattern(p)}
              className={cx(
                "h-6 px-2 rounded border text-xs",
                pattern === p
                  ? "border-teal/50 text-teal bg-teal-wash"
                  : "border-slate text-ash hover:text-chalk",
              )}
            >
              {p === "R" ? "research" : p === "A" ? "A · lineage" : "B · ledger"}
            </button>
          ))}
        </div>

        <p className="text-xs text-ash-dim truncate m-0">{PATTERN_NOTE[pattern]}</p>
      </header>

      <div className="flex-1 min-h-0 overflow-auto">
        {strategies.isLoading ? (
          <Loading what="strategies" />
        ) : strategies.error ? (
          <ErrorNote error={strategies.error} />
        ) : !strategy && pattern !== "R" ? (
          <Empty cmd="lab agent author --seed strategies/momo.py --grid cfg/momo_grid.yaml -n 10">
            no strategies discovered
          </Empty>
        ) : pattern === "R" ? (
          <ResearchView />
        ) : pattern === "A" ? (
          <LineageView strategy={strategy} />
        ) : (
          <AgentLedger strategy={strategy} />
        )}
      </div>
    </div>
  );
}
