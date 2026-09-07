/**
 * Compare — the screen that has to be hard to fool.
 *
 * Two things here are load-bearing and both are about not flattering a run.
 * First, every equity curve is rebased to 100 at *its own* first bar: overlay
 * raw equity and the run with the longest window wins on nothing but elapsed
 * time. Second, "best in column" needs a direction per metric, and the two that
 * read backwards — max drawdown and volatility — are exactly the two a careless
 * `Math.max` would crown. That direction table lives in one place, is exported,
 * and is tested, because getting it wrong recommends the worst strategy quietly.
 */

import { useMemo, useState } from "react";
import { useSearch } from "@tanstack/react-router";
import { useQueries, useQuery } from "@tanstack/react-query";

import { EChart } from "../components/charts/EChart";
import { Badge, Empty, ErrorNote, Loading, Panel, Table, Td, Th, Tr, cx } from "../components/ui";
import { api } from "../lib/api";
import { count, day, pct, ratio, shortId, toneClass } from "../lib/format";
import { toMs } from "../lib/timelock";
import type { EquitySeries, Metrics, RunRecord } from "../lib/types";

// --- the metric direction table ----------------------------------------------

export interface MetricColumn {
  key: string;
  label: string;
  /**
   * Which end of the range wins, applied to `rank(value)`. `null` means the
   * metric is reported but not ranked — calling one exposure "best" is a claim
   * the number does not support.
   */
  dir: "higher" | "lower" | null;
  /**
   * Projection used for ranking. Max drawdown arrives signed (−0.18), so under
   * a naive "lower is better" the *deepest* drawdown would win; magnitude is
   * what actually competes. Robust to backends that report it unsigned too.
   */
  rank?: (v: number) => number;
  fmt: (v: number | null | undefined) => string;
  /** Colour the cell by P&L direction. */
  tone?: boolean;
}

/**
 * Exported because Sweep ranks the same metrics in its results table and a
 * metric's direction must have exactly one definition in the app. The shared
 * layer is frozen, so the screen that owns the concept owns the table.
 */
export const METRIC_COLUMNS: readonly MetricColumn[] = [
  { key: "total_return", label: "return", dir: "higher", fmt: (v) => pct(v, 1), tone: true },
  { key: "cagr", label: "cagr", dir: "higher", fmt: (v) => pct(v, 1), tone: true },
  { key: "sharpe", label: "sharpe", dir: "higher", fmt: ratio },
  { key: "sortino", label: "sortino", dir: "higher", fmt: ratio },
  { key: "calmar", label: "calmar", dir: "higher", fmt: ratio },
  { key: "max_drawdown", label: "max dd", dir: "lower", rank: Math.abs, fmt: (v) => pct(v, 1) },
  { key: "volatility", label: "vol", dir: "lower", fmt: (v) => pct(v, 1) },
  { key: "turnover", label: "turnover", dir: "lower", fmt: ratio },
  { key: "hit_rate", label: "hit rate", dir: "higher", fmt: (v) => pct(v, 0) },
  { key: "profit_factor", label: "pf", dir: "higher", fmt: ratio },
  { key: "exposure", label: "exposure", dir: null, fmt: (v) => pct(v, 0) },
  { key: "trades", label: "trades", dir: null, fmt: count },
];

export const METRIC_BY_KEY: ReadonlyMap<string, MetricColumn> = new Map(
  METRIC_COLUMNS.map((c) => [c.key, c]),
);

/** Metrics carries an index signature, so every read has to be narrowed. */
export function metricValue(m: Metrics | undefined | null, key: string): number | null {
  const v = m?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Indices of the winning cells in one column. Ties all win — a silent
 * tie-break would invent a ranking the data does not contain. A column with
 * fewer than two comparable values highlights nothing: "best of one" is not a
 * comparison, it is decoration.
 */
export function bestIndices(
  values: readonly (number | null | undefined)[],
  col: MetricColumn,
): Set<number> {
  const out = new Set<number>();
  if (!col.dir) return out;
  const rank = col.rank ?? ((v: number) => v);
  const finite = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (finite.length < 2) return out;

  let best: number | null = null;
  values.forEach((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v)) return;
    const r = rank(v);
    const wins = best === null || (col.dir === "higher" ? r > best : r < best);
    if (wins) {
      best = r;
      out.clear();
      out.add(i);
    } else if (r === best) {
      out.add(i);
    }
  });
  return out;
}

/**
 * Colour has to encode good→bad, not big→small: −5% and −40% drawdown are both
 * "low" numbers and only one of them is good news. Every heat scale in the
 * comparison screens runs over this projection, never the raw metric.
 */
export function goodness(col: MetricColumn | undefined, v: number): number {
  if (!col || col.dir !== "lower") return v;
  return -(col.rank ? col.rank(v) : v);
}

// --- chart ink ---------------------------------------------------------------

export interface Ink {
  teal: string;
  chalk: string;
  ash: string;
  ashDim: string;
  moss: string;
  ember: string;
  slate: string;
}

let INK: Ink | null = null;

/**
 * ECharts paints to a canvas and cannot resolve CSS custom properties, so the
 * token layer has to be handed over as literals. Read the live `:root` value
 * where a stylesheet exists — a token edit then still propagates — and fall
 * back to the same numbers written in styles/tokens.css. Cached: the app has
 * one theme, so re-reading computed style per frame buys nothing.
 */
export function ink(): Ink {
  if (INK) return INK;
  const read = (name: string, fallback: string): string => {
    if (typeof document === "undefined") return fallback;
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  };
  INK = {
    teal: read("--color-teal", "#45B2A1"),
    chalk: read("--color-chalk", "#E9ECEF"),
    ash: read("--color-ash", "#98A2AE"),
    ashDim: read("--color-ash-dim", "#6B7480"),
    moss: read("--color-moss", "#6FBF73"),
    ember: read("--color-ember", "#E0654F"),
    slate: read("--color-slate", "#2B323C"),
  };
  return INK;
}

/**
 * Series identity is encoded by luminance and dash pattern, not by hue: a hue
 * on this screen means P&L direction or selection, and eight rainbow lines
 * would spend all of it on "which line is which". Teal is held back for the
 * one run the operator has picked.
 */
const NEUTRAL_CLASS = ["text-chalk", "text-ash", "text-ash-dim"] as const;
const STROKES: { echarts: string | number[]; svg?: string }[] = [
  { echarts: "solid" },
  { echarts: "dashed", svg: "5 3" },
  { echarts: "dotted", svg: "1 3" },
  { echarts: [7, 3, 2, 3], svg: "7 3 2 3" },
  { echarts: [2, 2], svg: "2 2" },
  { echarts: [10, 4], svg: "10 4" },
];

const lumIndex = (i: number) => i % NEUTRAL_CLASS.length;
const strokeIndex = (i: number) => Math.floor(i / NEUTRAL_CLASS.length) % STROKES.length;

// --- normalization -----------------------------------------------------------

/**
 * Rebase to 100 at the run's own first usable point. Without this a 2018–2026
 * run and a 2024–2026 run are compared on how long they were allowed to
 * compound, which is not a property of the strategy.
 */
export function rebase(equity: readonly (number | null)[]): number[] {
  const base = equity.find((v): v is number => v !== null && Number.isFinite(v) && v !== 0);
  if (base === undefined) return equity.map(() => 100);
  // NaN, not 0, for a missing point -- ECharts renders it as a break in the
  // line, which is the honest picture of a gap in the curve.
  return equity.map((v) =>
    v !== null && Number.isFinite(v) ? (100 * v) / base : Number.NaN,
  );
}

/** ECharts copes with 100k points; the operator does not. Keep the tail exact. */
function stride<T>(xs: T[], cap = 1200): T[] {
  if (xs.length <= cap) return xs;
  const k = Math.ceil(xs.length / cap);
  const out = xs.filter((_, i) => i % k === 0);
  const last = xs[xs.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

// --- screen ------------------------------------------------------------------

export default function Compare() {
  const { ids } = useSearch({ from: "/compare" });
  const parsed = useMemo(() => parseIds(ids), [ids]);
  return <CompareView ids={parsed} />;
}

export function parseIds(raw: string | undefined): string[] {
  if (!raw) return [];
  const seen = new Set<string>();
  for (const part of raw.split(",")) {
    const id = part.trim();
    if (id) seen.add(id);
  }
  return [...seen];
}

interface Row {
  id: string;
  run: RunRecord | null;
  metrics: Metrics;
}

export function CompareView({ ids }: { ids: string[] }) {
  const idsKey = ids.join(",");
  const [selected, setSelected] = useState<string | null>(null);
  const [xMode, setXMode] = useState<"date" | "elapsed">("date");
  const [scatterX, setScatterX] = useState("turnover");
  const [scatterY, setScatterY] = useState("sharpe");

  // Backtest artifacts are immutable, so these are fetched once and never
  // polled or invalidated by time.
  const meta = useQuery({
    queryKey: ["runs", "compare", idsKey],
    queryFn: () => api.compare(ids),
    enabled: ids.length > 0,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });

  const curves = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["runs", id, "equity"],
      queryFn: () => api.equity(id),
      staleTime: Number.POSITIVE_INFINITY,
      retry: false,
    })),
  });

  const rows: Row[] = useMemo(() => {
    const byId = new Map((meta.data?.runs ?? []).map((r) => [r.run_id, r]));
    return ids.map((id) => {
      const run = byId.get(id) ?? null;
      return { id, run, metrics: meta.data?.metrics?.[id] ?? run?.metrics ?? {} };
    });
  }, [ids, meta.data]);

  const bestByCol = useMemo(() => {
    const m = new Map<string, Set<number>>();
    for (const col of METRIC_COLUMNS) {
      m.set(
        col.key,
        bestIndices(
          rows.map((r) => metricValue(r.metrics, col.key)),
          col,
        ),
      );
    }
    return m;
  }, [rows]);

  // useQueries hands back a fresh array every render; the fetch stamps are the
  // only stable signal that the underlying series actually changed.
  const curveStamp = curves.map((c) => `${c.dataUpdatedAt}:${c.status}`).join("|");
  const series = useMemo(
    () =>
      ids
        .map((id, i) => ({ id, data: curves[i]?.data as EquitySeries | undefined }))
        .filter((s): s is { id: string; data: EquitySeries } => Boolean(s.data?.equity?.length)),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- curveStamp is the identity of `curves`
    [idsKey, curveStamp],
  );

  const curvesLoading = curves.some((c) => c.isLoading);
  const curvesError = curves.find((c) => c.error)?.error;

  const overlay = useMemo(
    () => overlayOption(series, rows, xMode, selected),
    [series, rows, xMode, selected],
  );

  const ddFloor = useMemo(() => {
    let lo = 0;
    for (const s of series)
      for (const d of s.data.drawdown ?? []) if (d !== null && d < lo) lo = d;
    return lo;
  }, [series]);

  if (ids.length === 0) {
    return (
      <div className="p-3">
        <Panel title="compare">
          <Empty cmd="lab compare 8f3a1c 9c1bd0 --json">
            No runs selected. Multi-select in the run browser, or open
            <code className="mono text-teal px-1">/compare?ids=a,b</code>
          </Empty>
        </Panel>
      </div>
    );
  }

  return (
    <div className="p-3 flex flex-col gap-3">
      <header className="flex items-center gap-3 flex-wrap">
        <h1 className="unit text-chalk">compare</h1>
        <span className="mono text-ash">{count(ids.length)} runs</span>
        {ids.length < 2 ? (
          <span className="text-xs text-ash-dim">
            one run selected — nothing to rank until there are two
          </span>
        ) : null}
        <div className="flex-1" />
        <Legend rows={rows} selected={selected} onSelect={setSelected} />
      </header>

      <Panel
        title="normalized equity · rebased to 100 at each run's own start"
        right={
          <div className="flex items-center gap-1">
            <span className="unit">x</span>
            <button
              type="button"
              onClick={() => setXMode("date")}
              className={cx(
                "h-5 px-1.5 rounded border text-xs",
                xMode === "date"
                  ? "border-teal/50 bg-teal-wash text-teal"
                  : "border-slate text-ash hover:text-chalk",
              )}
            >
              date
            </button>
            <button
              type="button"
              onClick={() => setXMode("elapsed")}
              className={cx(
                "h-5 px-1.5 rounded border text-xs",
                xMode === "elapsed"
                  ? "border-teal/50 bg-teal-wash text-teal"
                  : "border-slate text-ash hover:text-chalk",
              )}
            >
              elapsed
            </button>
          </div>
        }
      >
        {curvesLoading ? (
          <Loading what="loading equity" />
        ) : curvesError && series.length === 0 ? (
          <ErrorNote error={curvesError} />
        ) : series.length === 0 ? (
          <Empty cmd="lab backtest strategies/momo.py --json">
            No equity series for these runs
          </Empty>
        ) : (
          <div data-chart="compare-overlay">
            <EChart option={overlay} height={300} />
          </div>
        )}
      </Panel>

      <Panel title="metrics · best per column">
        {meta.isLoading ? (
          <Loading what="loading metrics" />
        ) : meta.error ? (
          <ErrorNote error={meta.error} />
        ) : rows.length === 0 ? (
          <Empty cmd="lab runs --json">No runs matched those ids</Empty>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>run</Th>
                <Th>period</Th>
                {METRIC_COLUMNS.map((c) => (
                  <Th
                    key={c.key}
                    align="right"
                    className={c.dir ? undefined : "text-ash-dim"}
                  >
                    <span
                      title={
                        c.dir === "lower"
                          ? "lower is better"
                          : c.dir === "higher"
                            ? "higher is better"
                            : "reported, not ranked"
                      }
                    >
                      {c.label}
                      {c.dir === "lower" ? " ↓" : c.dir === "higher" ? " ↑" : ""}
                    </span>
                  </Th>
                ))}
                <Th>flags</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <Tr
                  key={r.id}
                  selected={selected === r.id}
                  onClick={() => setSelected(selected === r.id ? null : r.id)}
                >
                  <Td mono title={r.id}>
                    <span data-run={r.id}>{shortId(r.id)}</span>
                    <span className="ml-2 text-ash">{r.run?.strategy ?? ""}</span>
                  </Td>
                  <Td className="text-ash-dim text-xs" mono>
                    {r.run?.start ? `${day(r.run.start)} → ${day(r.run.end)}` : null}
                  </Td>
                  {METRIC_COLUMNS.map((c) => {
                    const v = metricValue(r.metrics, c.key);
                    const best = bestByCol.get(c.key)?.has(i) ?? false;
                    return (
                      <Td
                        key={c.key}
                        align="right"
                        mono
                        className={cx(best && "bg-teal-wash", c.tone && toneClass(v))}
                      >
                        <span
                          data-col={c.key}
                          data-run={r.id}
                          data-best={best ? "1" : undefined}
                          className={cx(best && "text-teal")}
                          title={best ? "best in column" : undefined}
                        >
                          {c.fmt(v)}
                        </span>
                      </Td>
                    );
                  })}
                  <Td>
                    <span className="flex gap-1">
                      {r.metrics.optimistic_fills ? <Badge kind="warn">fills</Badge> : null}
                      {r.metrics.contaminated ? <Badge kind="warn">contam</Badge> : null}
                      {r.run?.origin === "agent-loop" ? <Badge>agent</Badge> : null}
                    </span>
                  </Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel title="drawdown · shared scale">
        {curvesLoading ? (
          <Loading what="loading drawdown" />
        ) : series.length === 0 ? (
          <Empty cmd="lab backtest strategies/momo.py --json">No drawdown series</Empty>
        ) : (
          <div className="grid gap-px bg-slate/60 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
            {series.map((s, i) => {
              const run = rows.find((r) => r.id === s.id)?.run ?? null;
              return (
                <div key={s.id} className="bg-gunmetal p-1" data-chart="compare-drawdown">
                  <div className="flex items-baseline gap-2 px-1">
                    <span className="mono text-xs text-ash">{shortId(s.id)}</span>
                    <span className="text-xs text-ash-dim">{run?.strategy ?? ""}</span>
                    <span className="flex-1" />
                    <span className="num text-xs text-ember">
                      {pct(metricValue(rows.find((r) => r.id === s.id)?.metrics, "max_drawdown"), 1)}
                    </span>
                  </div>
                  <EChart
                    option={drawdownOption(s.data, ddFloor, selected === s.id, i)}
                    height={96}
                  />
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {rows.length >= 3 ? (
        <Panel
          title="scatter"
          right={
            <div className="flex items-center gap-1">
              <MetricSelect value={scatterX} onChange={setScatterX} label="x" />
              <MetricSelect value={scatterY} onChange={setScatterY} label="y" />
            </div>
          }
        >
          <div data-chart="compare-scatter">
            <EChart
              option={scatterOption(rows, scatterX, scatterY, selected)}
              height={260}
            />
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

// --- pieces ------------------------------------------------------------------

function Legend({
  rows,
  selected,
  onSelect,
}: {
  rows: Row[];
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  return (
    <div className="flex items-center gap-1 flex-wrap" data-testid="compare-legend">
      {rows.map((r, i) => {
        const on = selected === r.id;
        return (
          <button
            key={r.id}
            type="button"
            onClick={() => onSelect(on ? null : r.id)}
            title={r.id}
            className={cx(
              "inline-flex items-center gap-1.5 h-6 px-1.5 rounded border text-xs",
              on ? "border-teal/50 bg-teal-wash text-teal" : "border-slate text-ash hover:text-chalk",
            )}
          >
            <svg
              width="18"
              height="6"
              aria-hidden
              className={on ? "text-teal" : NEUTRAL_CLASS[lumIndex(i)]}
            >
              <line
                x1="0"
                y1="3"
                x2="18"
                y2="3"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeDasharray={on ? undefined : STROKES[strokeIndex(i)].svg}
              />
            </svg>
            <span className="mono">{shortId(r.id, 6)}</span>
            <span className="text-ash-dim">{r.run?.strategy ?? ""}</span>
          </button>
        );
      })}
    </div>
  );
}

function MetricSelect({
  value,
  onChange,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  label: string;
}) {
  return (
    <label className="inline-flex items-center gap-1">
      <span className="unit">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-5 rounded border border-slate bg-graphite text-ash text-xs px-1"
      >
        {METRIC_COLUMNS.map((c) => (
          <option key={c.key} value={c.key}>
            {c.label}
          </option>
        ))}
      </select>
    </label>
  );
}

// --- chart options -----------------------------------------------------------

function overlayOption(
  series: { id: string; data: EquitySeries }[],
  rows: Row[],
  xMode: "date" | "elapsed",
  selected: string | null,
): unknown {
  const c = ink();
  const neutral = [c.chalk, c.ash, c.ashDim];
  return {
    animation: false,
    grid: { left: 48, right: 12, top: 12, bottom: 24 },
    tooltip: { trigger: "axis", axisPointer: { type: "line" } },
    xAxis:
      xMode === "date"
        ? { type: "time" }
        : { type: "value", name: "bars", nameLocation: "end", nameGap: 6 },
    yAxis: { type: "value", scale: true, name: "= 100", nameGap: 8 },
    series: series.map((s) => {
      const i = rows.findIndex((r) => r.id === s.id);
      const idx = i < 0 ? 0 : i;
      const on = selected === s.id;
      const values = rebase(s.data.equity);
      const points = values.map((v, k) =>
        xMode === "date" ? [toMs(s.data.t[k]), v] : [k, v],
      );
      return {
        id: s.id,
        name: shortId(s.id, 6),
        type: "line",
        showSymbol: false,
        data: stride(points),
        z: on ? 5 : 2,
        lineStyle: {
          width: on ? 2 : 1,
          color: on ? c.teal : neutral[lumIndex(idx)],
          type: on ? "solid" : STROKES[strokeIndex(idx)].echarts,
          opacity: selected && !on ? 0.55 : 1,
        },
        itemStyle: { color: on ? c.teal : neutral[lumIndex(idx)] },
      };
    }),
  };
}

function drawdownOption(
  data: EquitySeries,
  floor: number,
  on: boolean,
  idx: number,
): unknown {
  const c = ink();
  const points = data.drawdown.map((v, k) => [toMs(data.t[k]), v]);
  return {
    animation: false,
    grid: { left: 44, right: 8, top: 8, bottom: 18 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "time", axisLabel: { show: idx >= 0 } },
    // Shared floor across every small multiple: independently scaled panes make
    // a −5% drawdown and a −40% one look identical, which is the whole failure
    // mode small multiples exist to prevent.
    yAxis: { type: "value", min: floor, max: 0 },
    series: [
      {
        type: "line",
        showSymbol: false,
        data: points,
        lineStyle: { width: 1, color: c.ember },
        areaStyle: { color: c.ember, opacity: on ? 0.24 : 0.12 },
      },
    ],
  };
}

function scatterOption(
  rows: Row[],
  xKey: string,
  yKey: string,
  selected: string | null,
): unknown {
  const c = ink();
  const xc = METRIC_BY_KEY.get(xKey);
  const yc = METRIC_BY_KEY.get(yKey);
  const points = rows
    .map((r) => ({
      id: r.id,
      x: metricValue(r.metrics, xKey),
      y: metricValue(r.metrics, yKey),
    }))
    .filter((p): p is { id: string; x: number; y: number } => p.x !== null && p.y !== null);

  return {
    animation: false,
    grid: { left: 56, right: 16, top: 16, bottom: 36 },
    tooltip: { trigger: "item" },
    xAxis: { type: "value", scale: true, name: xc?.label ?? xKey, nameLocation: "middle", nameGap: 22 },
    yAxis: { type: "value", scale: true, name: yc?.label ?? yKey, nameGap: 8 },
    series: [
      {
        type: "scatter",
        symbolSize: 9,
        data: points.map((p) => ({
          name: shortId(p.id, 6),
          value: [p.x, p.y],
          itemStyle: { color: selected === p.id ? c.teal : c.ash },
        })),
        label: {
          show: true,
          position: "right",
          color: c.ashDim,
          fontFamily: "IBM Plex Mono",
          fontSize: 10,
          formatter: (p: { name: string }) => p.name,
        },
      },
    ],
  };
}
