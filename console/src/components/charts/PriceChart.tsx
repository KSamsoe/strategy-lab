/**
 * Candles for one ticker, with the run's own entries and exits marked on them.
 *
 * The markers are the reason this is not a generic price chart: they turn
 * "the strategy is long NVDA" into "it bought here, on this bar, and here is
 * what the bar looked like". Marker colour encodes the trade's P&L, so a column
 * of red arrows down one stretch of the chart reads as a losing regime before a
 * single number is read.
 *
 * Bars are a backtest artifact and therefore immutable: fetched once, never
 * polled, cached forever.
 */

import { useEffect, useId, useMemo, useRef, useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ColorType,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";

import { api } from "../../lib/api";
import type { TradeRow } from "../../lib/types";
import { count, money, signedMoney } from "../../lib/format";
import { nearestIndex, timeLock, toChartTime } from "../../lib/timelock";
import { Empty, ErrorNote, Loading } from "../ui";

/**
 * The token palette as literals -- see the note in `EquityChart.tsx`. Canvas
 * libraries take colours as JS values, so this is the one exported constant in
 * the file that is allowed to hold hex, and it mirrors `styles/tokens.css`.
 */
export const PRICE_INK = {
  slate: "#2B323C",
  slateHi: "#3A424E",
  ash: "#98A2AE",
  moss: "#6FBF73",
  ember: "#E0654F",
  teal: "#45B2A1",
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

/** Marker colour is P&L direction. A flat or open trade stays neutral. */
function pnlColor(pnl: number): string {
  if (!Number.isFinite(pnl) || pnl === 0) return PRICE_INK.ash;
  return pnl > 0 ? PRICE_INK.moss : PRICE_INK.ember;
}

/**
 * Entry and exit markers for one ticker, in ascending time order.
 *
 * lightweight-charts requires sorted markers and will not draw an unsorted
 * list, and trades arrive ordered by entry -- so an exit from an earlier trade
 * can legitimately fall after a later entry. Sorting here is load-bearing, not
 * defensive tidying. An open trade contributes an entry and no exit.
 */
export function tradeMarkers(
  trades: readonly TradeRow[] | undefined,
  ticker: string,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const tr of trades ?? []) {
    if (tr.ticker !== ticker) continue;
    const color = pnlColor(tr.pnl);
    if (tr.entry_time) {
      out.push({
        time: toChartTime(tr.entry_time) as UTCTimestamp,
        position: "belowBar",
        shape: "arrowUp",
        color,
        text: `${tr.side} ${count(tr.qty)} @ ${money(tr.entry_price, 2)}`,
      });
    }
    if (tr.exit_time) {
      out.push({
        time: toChartTime(tr.exit_time) as UTCTimestamp,
        position: "aboveBar",
        shape: "arrowDown",
        color,
        text: `${signedMoney(tr.pnl, 0)} ${tr.exit_reason || ""}`.trim(),
      });
    }
  }
  return out.sort((a, b) => Number(a.time) - Number(b.time));
}

export interface PriceChartProps {
  scope: string;
  runId: string;
  ticker: string;
  trades?: TradeRow[]; // entry/exit markers
  height?: number; // default 260
}

export function PriceChart({ scope, runId, ticker, trades, height = 260 }: PriceChartProps) {
  const uid = useId();
  const SOURCE = `price-chart${uid}`;

  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick", Time> | null>(null);

  const store = useMemo(() => timeLock(scope), [scope]);
  const t = useSyncExternalStore(store.subscribe, () => store.getState().t);
  const src = useSyncExternalStore(store.subscribe, () => store.getState().source);

  const q = useQuery({
    queryKey: ["runs", runId, "bars", ticker],
    queryFn: () => api.bars(runId, ticker),
    enabled: Boolean(runId && ticker),
    // A finished run's bars cannot change. Refetching them is pure heat.
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });

  const bars = q.data;
  const n = Math.min(bars?.t?.length ?? 0, bars?.close?.length ?? 0);

  const candles = useMemo(() => {
    if (!bars || n === 0) return [] as CandlestickData<Time>[];
    const out: CandlestickData<Time>[] = [];
    let last = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      const time = toChartTime(bars.t[i]);
      const close = bars.close[i];
      if (!Number.isFinite(time) || !Number.isFinite(close) || time <= last) continue;
      last = time;
      out.push({
        time: time as UTCTimestamp,
        open: bars.open?.[i] ?? close,
        high: bars.high?.[i] ?? close,
        low: bars.low?.[i] ?? close,
        close,
      });
    }
    return out;
  }, [bars, n]);

  const ms = useMemo(() => candles.map((c) => Number(c.time) * 1000), [candles]);
  const markers = useMemo(() => tradeMarkers(trades, ticker), [trades, ticker]);

  const busRef = useRef({ store, SOURCE });
  busRef.current = { store, SOURCE };

  const hasCandles = candles.length > 0;

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !hasCandles) return;

    const chart = createChart(host, {
      width: host.clientWidth || 640,
      height,
      autoSize: false,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: PRICE_INK.ash,
        fontFamily: PRICE_INK.mono,
        fontSize: 10,
      },
      grid: {
        vertLines: { color: PRICE_INK.slate },
        horzLines: { color: PRICE_INK.slate },
      },
      rightPriceScale: { borderColor: PRICE_INK.slate, scaleMargins: { top: 0.12, bottom: 0.12 } },
      timeScale: { borderColor: PRICE_INK.slate, rightOffset: 2, fixLeftEdge: true },
      crosshair: {
        vertLine: { color: PRICE_INK.teal, labelBackgroundColor: PRICE_INK.slate },
        horzLine: { color: PRICE_INK.slateHi, labelBackgroundColor: PRICE_INK.slate },
      },
    });
    chartRef.current = chart;

    seriesRef.current = chart.addCandlestickSeries({
      upColor: PRICE_INK.moss,
      downColor: PRICE_INK.ember,
      wickUpColor: PRICE_INK.moss,
      wickDownColor: PRICE_INK.ember,
      borderVisible: false,
      priceLineVisible: false,
    });

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

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver((entries) => {
        const w = entries[0]?.contentRect.width ?? 0;
        if (w > 0) chart.resize(w, height);
      });
      ro.observe(host);
    }

    return () => {
      ro?.disconnect();
      chart.unsubscribeCrosshairMove(onMove);
      chart.unsubscribeClick(onClick);
      chartRef.current = null;
      seriesRef.current = null;
      chart.remove();
    };
  }, [hasCandles, height]);

  useEffect(() => {
    if (!chartRef.current || !seriesRef.current) return;
    seriesRef.current.setData(candles);
    seriesRef.current.setMarkers(markers);
    chartRef.current.timeScale().fitContent();
  }, [candles, markers]);

  // Move the crosshair for a seek that came from somewhere else. Our own echo
  // is skipped: the cursor is already there, and re-emitting would let two
  // synced charts trade the playhead back and forth forever.
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    if (src === SOURCE) return;
    if (t === null || ms.length === 0) {
      chart.clearCrosshairPosition();
      return;
    }
    const i = nearestIndex(ms, t);
    if (i < 0) return;
    chart.setCrosshairPosition(candles[i].close, candles[i].time, series);
  }, [t, src, SOURCE, ms, candles]);

  if (!ticker) return <Empty>pick a ticker to see its bars and this run&apos;s trades on them</Empty>;
  if (q.isLoading) return <Loading what={`loading ${ticker} bars`} />;
  if (q.error) return <ErrorNote error={q.error} />;
  if (!hasCandles) {
    return (
      <Empty cmd={`lab data fetch ${ticker} --tf 1d`}>
        no bars stored for {ticker} in this run
      </Empty>
    );
  }

  const entries = markers.filter((m) => m.shape === "arrowUp").length;
  const exits = markers.length - entries;

  return (
    <div className="flex flex-col">
      <div style={{ height }}>
        <div ref={hostRef} className="w-full h-full" />
      </div>
      <div className="flex items-center gap-4 px-2 py-1 border-t border-slate">
        <span className="mono text-chalk">{ticker}</span>
        <span className="text-micro uppercase tracking-wider text-ash">
          <span className="num">{count(candles.length)}</span> bars
        </span>
        <span className="text-micro uppercase tracking-wider text-ash">
          <span aria-hidden>▲</span> <span className="num">{count(entries)}</span> entries
        </span>
        <span className="text-micro uppercase tracking-wider text-ash">
          <span aria-hidden>▼</span> <span className="num">{count(exits)}</span> exits
        </span>
        <span className="flex-1" />
        <span
          className="text-micro uppercase tracking-wider text-ash-dim"
          title="marker colour is the trade's realised P&L"
        >
          marker colour = trade p&amp;l
        </span>
      </div>
    </div>
  );
}
