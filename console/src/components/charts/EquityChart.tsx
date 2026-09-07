/**
 * Equity and drawdown, with the out-of-sample spans shaded.
 *
 * The shading is the point of the component, not a garnish. A walk-forward run
 * interleaves fitted and unseen periods, and an equity curve drawn without that
 * distinction is the single most flattering lie this console could tell: the
 * eye reads one smooth line and credits the strategy for territory it was
 * fitted on. So `is_oos` is rendered as bands the reader cannot miss, the flag
 * flips as many times as the run says it does, and the header states outright
 * what fraction of the curve was earned on unseen data.
 *
 * `is_oos` is a per-point boolean array, so turning it into spans is where the
 * bug would live -- an off-by-one closes a band one bar early and quietly
 * re-labels a losing OOS week as in-sample. `oosSpans` is therefore a pure
 * function, exported, and tested on its own.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";

import type { EquitySeries } from "../../lib/types";
import { count, pct } from "../../lib/format";
import { nearestIndex, timeLock, toChartTime, toMs } from "../../lib/timelock";
import { Empty, cx } from "../ui";

/**
 * The token palette as literals. Canvas libraries cannot resolve a CSS custom
 * property, so the "never a raw hex" rule breaks here and only here -- one
 * constant per chart file, mirroring `src/styles/tokens.css`. Kept local rather
 * than imported from `EChart.tsx` so a screen that only draws a time series
 * does not pull ECharts into its chunk.
 */
export const EQUITY_INK = {
  slate: "#2B323C",
  slateHi: "#3A424E",
  chalk: "#E9ECEF",
  ash: "#98A2AE",
  ashDim: "#6B7480",
  ember: "#E0654F",
  emberFill: "#E0654F33",
  emberFillFaint: "#E0654F08",
  teal: "#45B2A1",
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

// --- the OOS span computation ------------------------------------------------

/** Inclusive index range into the series arrays. */
export interface Span {
  start: number;
  end: number;
}

/**
 * Maximal runs of `true` in a boolean array, as inclusive index spans.
 *
 * A walk-forward run flips `is_oos` once per window, so this returns many
 * disjoint spans, and the last one is very often open at the right edge --
 * the run ends inside out-of-sample territory. Both of those are the normal
 * case, not an edge case.
 */
export function oosSpans(flags: readonly boolean[] | null | undefined): Span[] {
  const out: Span[] = [];
  if (!flags || flags.length === 0) return out;
  let start = -1;
  for (let i = 0; i < flags.length; i++) {
    if (flags[i]) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      out.push({ start, end: i - 1 });
      start = -1;
    }
  }
  if (start >= 0) out.push({ start, end: flags.length - 1 });
  return out;
}

/** Share of the curve earned on unseen data. Stated in words above the chart. */
export function oosFraction(flags: readonly boolean[] | null | undefined): number {
  if (!flags || flags.length === 0) return 0;
  let n = 0;
  for (const f of flags) if (f) n++;
  return n / flags.length;
}

/** Chart-time (seconds) edges of each span. */
export interface TimeSpan {
  from: number;
  to: number;
}

/**
 * Spans projected onto the time axis. Clamped to the shorter of the two arrays
 * because a truncated payload must shade less, never shade the wrong place.
 */
export function oosTimeSpans(
  times: readonly string[] | null | undefined,
  flags: readonly boolean[] | null | undefined,
): TimeSpan[] {
  if (!times || times.length === 0) return [];
  const n = Math.min(times.length, flags?.length ?? 0);
  if (n === 0) return [];
  return oosSpans(flags?.slice(0, n)).map((s) => ({
    from: toChartTime(times[s.start]),
    to: toChartTime(times[s.end]),
  }));
}

// --- data shaping ------------------------------------------------------------

/**
 * lightweight-charts throws on a duplicated or out-of-order timestamp, and a
 * throw inside a layout effect takes the whole screen down. A malformed
 * artifact should degrade to a shorter line, so non-finite values and
 * non-increasing times are dropped.
 */
function lineData(
  secs: readonly number[],
  values: readonly (number | null)[] | null | undefined,
): LineData<Time>[] {
  if (!values) return [];
  const out: LineData<Time>[] = [];
  let last = Number.NEGATIVE_INFINITY;
  const n = Math.min(secs.length, values.length);
  for (let i = 0; i < n; i++) {
    const time = secs[i];
    const value = values[i];
    // A null is a gap in the curve, never a zero: plotting zero would draw a
    // crash the account never had.
    if (value === null || !Number.isFinite(time) || !Number.isFinite(value) || time <= last)
      continue;
    last = time;
    out.push({ time: time as UTCTimestamp, value });
  }
  return out;
}

// --- component ---------------------------------------------------------------

export interface EquityChartProps {
  scope: string;
  data: EquitySeries;
  height?: number; // default 220
  showDrawdown?: boolean; // default true: stacked drawdown pane
  benchmarkLabel?: string;
}

interface Band {
  key: string;
  left: number;
  width: number;
}

const sameBands = (a: readonly Band[], b: readonly Band[]): boolean =>
  a.length === b.length && a.every((x, i) => x.left === b[i].left && x.width === b[i].width);

export function EquityChart({
  scope,
  data,
  height = 220,
  showDrawdown = true,
  benchmarkLabel = "benchmark",
}: EquityChartProps) {
  const uid = useId();
  const SOURCE = `equity-chart${uid}`;

  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const equityRef = useRef<ISeriesApi<"Line", Time> | null>(null);
  const benchRef = useRef<ISeriesApi<"Line", Time> | null>(null);
  const ddRef = useRef<ISeriesApi<"Area", Time> | null>(null);

  const [bands, setBands] = useState<Band[]>([]);

  /**
   * Read the playhead as two primitives rather than through `useTimeLock`.
   * This component *emits* on pointer move, and subscribing to the whole lock
   * state would re-render the chart on every one of its own hover ticks.
   * `useSyncExternalStore` over a primitive bails out when nothing changed.
   */
  const store = useMemo(() => timeLock(scope), [scope]);
  const t = useSyncExternalStore(store.subscribe, () => store.getState().t);
  const src = useSyncExternalStore(store.subscribe, () => store.getState().source);

  const secs = useMemo(() => (data.t ?? []).map((s) => toChartTime(s)), [data.t]);
  const ms = useMemo(() => (data.t ?? []).map((s) => toMs(s)), [data.t]);
  const spans = useMemo(() => oosTimeSpans(data.t, data.is_oos), [data.t, data.is_oos]);
  const oosShare = useMemo(() => oosFraction(data.is_oos), [data.is_oos]);
  const hasBenchmark = Boolean(data.benchmark && data.benchmark.length > 0);
  const points = Math.min(data.t?.length ?? 0, data.equity?.length ?? 0);

  /**
   * Bands are HTML over the canvas rather than a chart primitive: the shading
   * has to survive pan and zoom, and mapping two timestamps to two x
   * coordinates is the whole job. Recomputed on range change and on resize.
   */
  const recompute = useCallback(() => {
    const chart = chartRef.current;
    const host = hostRef.current;
    if (!chart || !host) return;
    const scale = chart.timeScale();
    const w = host.clientWidth;
    const next: Band[] = [];
    for (const s of spans) {
      const a = scale.timeToCoordinate(s.from as UTCTimestamp);
      const b = scale.timeToCoordinate(s.to as UTCTimestamp);
      if (typeof a !== "number" || typeof b !== "number") continue;
      const left = Math.max(0, Math.min(a, b));
      const right = Math.min(w, Math.max(a, b));
      if (right <= 0 || left >= w) continue;
      // A one-bar span is still a real out-of-sample bar; give it a visible
      // minimum rather than letting it collapse to nothing.
      next.push({ key: `${s.from}-${s.to}`, left, width: Math.max(2, right - left) });
    }
    setBands((prev) => (sameBands(prev, next) ? prev : next));
  }, [spans]);

  const recomputeRef = useRef(recompute);
  recomputeRef.current = recompute;

  /** Latest lock emitters, so the chart subscriptions never need rebinding. */
  const busRef = useRef({ store, SOURCE });
  busRef.current = { store, SOURCE };

  useEffect(() => {
    const host = hostRef.current;
    if (!host || points === 0) return;

    const chart = createChart(host, {
      width: host.clientWidth || 640,
      height,
      autoSize: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: EQUITY_INK.ash,
        fontFamily: EQUITY_INK.mono,
        fontSize: 10,
      },
      grid: {
        vertLines: { color: EQUITY_INK.slate },
        horzLines: { color: EQUITY_INK.slate },
      },
      rightPriceScale: {
        borderColor: EQUITY_INK.slate,
        scaleMargins: showDrawdown ? { top: 0.06, bottom: 0.34 } : { top: 0.08, bottom: 0.06 },
      },
      timeScale: { borderColor: EQUITY_INK.slate, rightOffset: 2, fixLeftEdge: true },
      crosshair: {
        vertLine: { color: EQUITY_INK.teal, labelBackgroundColor: EQUITY_INK.slate },
        horzLine: { color: EQUITY_INK.slateHi, labelBackgroundColor: EQUITY_INK.slate },
      },
    });
    chartRef.current = chart;

    const equity = chart.addLineSeries({
      color: EQUITY_INK.chalk,
      lineWidth: 2,
      priceLineVisible: false,
      crosshairMarkerRadius: 3,
    });
    equityRef.current = equity;

    benchRef.current = chart.addLineSeries({
      color: EQUITY_INK.ashDim,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: benchmarkLabel,
    });

    if (showDrawdown) {
      const dd = chart.addAreaSeries({
        lineColor: EQUITY_INK.ember,
        topColor: EQUITY_INK.emberFillFaint,
        bottomColor: EQUITY_INK.emberFill,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        priceScaleId: "drawdown",
      });
      // Stacked, not overlaid: drawdown owns the bottom third of the frame and
      // its own scale, so a −2% dip cannot masquerade as a −40% one.
      dd.priceScale().applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });
      ddRef.current = dd;
    }

    const onMove = (param: MouseEventParams<Time>) => {
      const time = param.time;
      busRef.current.store.hover(typeof time === "number" ? time * 1000 : null);
    };
    const onClick = (param: MouseEventParams<Time>) => {
      const time = param.time;
      if (typeof time !== "number") return;
      busRef.current.store.seek(time * 1000, busRef.current.SOURCE);
    };
    chart.subscribeCrosshairMove(onMove);
    chart.subscribeClick(onClick);

    const scale = chart.timeScale();
    const onRange = () => recomputeRef.current();
    scale.subscribeVisibleLogicalRangeChange(onRange);

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver((entries) => {
        const w = entries[0]?.contentRect.width ?? 0;
        if (w > 0) chart.resize(w, height);
        recomputeRef.current();
      });
      ro.observe(host);
    }

    return () => {
      ro?.disconnect();
      scale.unsubscribeVisibleLogicalRangeChange(onRange);
      chart.unsubscribeCrosshairMove(onMove);
      chart.unsubscribeClick(onClick);
      chartRef.current = null;
      equityRef.current = null;
      benchRef.current = null;
      ddRef.current = null;
      chart.remove();
    };
    // Rebuilt only on structural changes; data flows through the effect below.
  }, [points === 0, height, showDrawdown, benchmarkLabel]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    equityRef.current?.setData(lineData(secs, data.equity));
    benchRef.current?.setData(lineData(secs, data.benchmark ?? []));
    ddRef.current?.setData(lineData(secs, data.drawdown));
    chart.timeScale().fitContent();
    recomputeRef.current();
  }, [secs, data.equity, data.benchmark, data.drawdown]);

  // An external seek moves the crosshair; our own echo does not, or two synced
  // charts would push the playhead back and forth forever.
  useEffect(() => {
    const chart = chartRef.current;
    const series = equityRef.current;
    if (!chart || !series) return;
    if (src === SOURCE) return;
    if (t === null || ms.length === 0) {
      chart.clearCrosshairPosition();
      return;
    }
    const i = nearestIndex(ms, t);
    const value = data.equity?.[i];
    if (i < 0 || typeof value !== "number" || !Number.isFinite(value)) return;
    chart.setCrosshairPosition(value, secs[i] as UTCTimestamp, series);
  }, [t, src, SOURCE, ms, secs, data.equity]);

  if (points === 0) {
    return (
      <Empty cmd="lab backtest strategies/momo.py --json">
        no equity series for this {scope === "live" ? "session" : "run"}
      </Empty>
    );
  }

  return (
    <div className="flex flex-col">
      <div className="relative" style={{ height }}>
        <div ref={hostRef} className="absolute inset-0" />
        {/* Shading sits over the canvas: it must survive pan and zoom, and the
            wash is translucent so the curve reads straight through it. */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden>
          {bands.map((b) => (
            <div
              key={b.key}
              data-oos-band
              className="absolute inset-y-0 bg-oos-wash border-x border-teal/25"
              style={{ left: b.left, width: b.width }}
            >
              <span className="absolute top-0 left-0 px-1 leading-4 text-micro uppercase tracking-wider text-teal">
                oos
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4 px-2 py-1 border-t border-slate flex-wrap">
        <span className="inline-flex items-center gap-1.5 unit">
          <span className="w-4 h-0.5 bg-chalk" aria-hidden /> equity
        </span>
        {showDrawdown ? (
          <span className="inline-flex items-center gap-1.5 unit">
            <span className="w-4 h-0.5 bg-ember" aria-hidden /> drawdown
          </span>
        ) : null}
        {hasBenchmark ? (
          <span className="inline-flex items-center gap-1.5 unit">
            <span className="w-4 h-0.5 bg-ash-dim" aria-hidden /> {benchmarkLabel}
          </span>
        ) : null}
        <span className="inline-flex items-center gap-1.5 unit">
          <span className="w-4 h-3 bg-oos-wash border border-teal/25" aria-hidden /> out of sample
        </span>
        <span className="flex-1" />
        <span
          data-oos-summary
          className={cx(
            "text-micro uppercase tracking-wider",
            oosShare > 0 ? "text-teal" : "text-amber",
          )}
          title={
            oosShare > 0
              ? "shaded spans were never seen during fitting"
              : "every point on this curve is in-sample — nothing here is evidence of generalisation"
          }
        >
          {oosShare > 0 ? (
            <>
              <span className="num">{pct(oosShare, 0)}</span> out of sample ·{" "}
              <span className="num">{count(spans.length)}</span> span
              {spans.length === 1 ? "" : "s"}
            </>
          ) : (
            "in-sample only"
          )}
        </span>
      </div>
    </div>
  );
}
