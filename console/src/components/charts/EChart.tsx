/**
 * The ECharts wrapper.
 *
 * `lightweight-charts` owns time series; this owns everything that is not one --
 * sweep heatmaps, compare scatters, walk-forward bars. It stays a *wrapper* on
 * purpose: no defaults beyond the theme, no data massaging, no opinion about
 * what is being plotted. The caller hands over an option object and gets back a
 * canvas that resizes and disposes itself.
 *
 * Only the four series types and five components the console actually draws are
 * registered. The `echarts` barrel is about a megabyte; heatmap, scatter, bar
 * and line plus tooltip/grid/visualMap/legend is what the bundle should carry.
 */

import { useEffect, useRef } from "react";
import {
  init,
  registerTheme,
  use,
  type EChartsCoreOption,
  type EChartsType,
} from "echarts/core";
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

import { cx } from "../ui";

/**
 * The token palette as literals.
 *
 * Charting libraries paint to a canvas and cannot resolve a CSS custom
 * property, so the one rule the rest of the app lives by -- never a raw hex in
 * a component -- has to break exactly here. It breaks in exactly one constant,
 * mirroring `src/styles/tokens.css`, which stays the source of truth: if a
 * token moves, this moves with it and nothing else in the file has to change.
 */
export const CHART_INK = {
  graphite: "#14171C",
  gunmetal: "#1C2128",
  slate: "#2B323C",
  slateHi: "#3A424E",
  chalk: "#E9ECEF",
  ash: "#98A2AE",
  ashDim: "#6B7480",
  moss: "#6FBF73",
  ember: "#E0654F",
  amber: "#D9A441",
  teal: "#45B2A1",
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

const axis = {
  axisLine: { show: true, lineStyle: { color: CHART_INK.slate } },
  axisTick: { show: false },
  axisLabel: {
    color: CHART_INK.ash,
    fontFamily: CHART_INK.mono,
    fontSize: 10,
  },
  splitLine: { show: true, lineStyle: { color: CHART_INK.slate, width: 1 } },
  splitArea: { show: false },
  nameTextStyle: { color: CHART_INK.ashDim, fontFamily: CHART_INK.mono, fontSize: 10 },
};

/**
 * The console theme. Transparent background so the panel surface shows through,
 * slate hairlines, ash labels, Plex Mono everywhere a number can appear.
 *
 * The default series colours are a luminance ramp, not a rainbow: on this
 * screen a hue means P&L direction, warning state or selection, so a chart that
 * spends teal on "series 3" has spent the selection colour on decoration. Any
 * chart that genuinely encodes meaning sets its own colours per series.
 */
export const darkTheme = {
  backgroundColor: "transparent",
  color: [CHART_INK.ash, CHART_INK.chalk, CHART_INK.ashDim, CHART_INK.slateHi],
  textStyle: { fontFamily: CHART_INK.mono, color: CHART_INK.ash, fontSize: 11 },
  title: { textStyle: { color: CHART_INK.chalk, fontSize: 12 } },
  grid: { borderColor: CHART_INK.slate, containLabel: false },
  categoryAxis: axis,
  valueAxis: axis,
  timeAxis: axis,
  logAxis: axis,
  legend: {
    textStyle: { color: CHART_INK.ash, fontFamily: CHART_INK.mono, fontSize: 10 },
    inactiveColor: CHART_INK.ashDim,
    itemWidth: 14,
    itemHeight: 2,
  },
  tooltip: {
    backgroundColor: CHART_INK.gunmetal,
    borderColor: CHART_INK.slate,
    borderWidth: 1,
    padding: [4, 8],
    textStyle: { color: CHART_INK.chalk, fontFamily: CHART_INK.mono, fontSize: 11 },
    axisPointer: {
      lineStyle: { color: CHART_INK.slateHi, width: 1 },
      crossStyle: { color: CHART_INK.slateHi, width: 1 },
      label: {
        backgroundColor: CHART_INK.slate,
        color: CHART_INK.chalk,
        fontFamily: CHART_INK.mono,
        fontSize: 10,
      },
    },
  },
  visualMap: {
    textStyle: { color: CHART_INK.ash, fontFamily: CHART_INK.mono, fontSize: 10 },
    borderColor: CHART_INK.slate,
  },
} as const;

export const THEME_NAME = "lab-dark";

/**
 * Registration is module-level and idempotent. `use` de-duplicates internally,
 * but the guard keeps a hot-reload from re-registering the theme under a chart
 * that is already alive.
 */
let registered = false;
function ensureRegistered(): void {
  if (registered) return;
  registered = true;
  use([
    HeatmapChart,
    ScatterChart,
    BarChart,
    LineChart,
    TooltipComponent,
    GridComponent,
    VisualMapComponent,
    LegendComponent,
    CanvasRenderer,
  ]);
  registerTheme(THEME_NAME, darkTheme);
}

export interface EChartProps {
  option: unknown; // echarts EChartsOption
  height?: number;
  onEvent?: Record<string, (params: unknown) => void>;
  className?: string;
}

export function EChart({ option, height = 240, onEvent, className }: EChartProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);
  /** Handlers change every render; the subscription must not. */
  const handlersRef = useRef(onEvent);
  handlersRef.current = onEvent;

  // Init and teardown, once. Everything else applies to the live instance.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    ensureRegistered();
    const chart = init(host, THEME_NAME, { renderer: "canvas" });
    chartRef.current = chart;

    // A panel that changes size must not leave the canvas at its birth width.
    // jsdom and old Safari have no ResizeObserver; the chart simply stays put.
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => chart.resize());
      ro.observe(host);
    }

    return () => {
      ro?.disconnect();
      chartRef.current = null;
      chart.dispose();
    };
  }, []);

  // `notMerge` because callers rebuild the whole option: merging a five-series
  // overlay into a two-series one silently leaves three ghosts on the canvas.
  useEffect(() => {
    chartRef.current?.setOption(option as EChartsCoreOption, { notMerge: true });
  }, [option]);

  const eventNames = Object.keys(onEvent ?? {}).sort().join(",");
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !eventNames) return;
    const names = eventNames.split(",");
    for (const name of names) {
      chart.on(name, (params: unknown) => handlersRef.current?.[name]?.(params));
    }
    return () => {
      for (const name of names) chart.off(name);
    };
  }, [eventNames]);

  useEffect(() => {
    chartRef.current?.resize();
  }, [height]);

  return <div ref={hostRef} className={cx("w-full", className)} style={{ height }} />;
}
