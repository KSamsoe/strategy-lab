/**
 * Tests for the shared chart and ledger components.
 *
 * jsdom has no canvas, so `lightweight-charts` and `echarts` are mocked at the
 * module boundary. That is not a cop-out: what is worth asserting about these
 * components is not that a pixel got painted but that the *arithmetic between
 * the data and the library* is right — seconds versus milliseconds, where an
 * out-of-sample span starts and stops, which row a sort puts first, whether a
 * seek that came from elsewhere moves the crosshair without being re-emitted.
 * Every one of those is a silent, plausible-looking bug, and every one of them
 * is testable without a GPU.
 *
 * The OOS span helper gets a randomised round-trip on top of its table of
 * cases, because an off-by-one there does not crash: it just re-labels a losing
 * out-of-sample week as in-sample, which is the exact lie the shading exists to
 * prevent.
 */

import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

// --- module mocks ------------------------------------------------------------

vi.mock("lightweight-charts", () => {
  const makeSeries = () => ({
    setData: vi.fn(),
    setMarkers: vi.fn(),
    applyOptions: vi.fn(),
    priceScale: () => ({ applyOptions: vi.fn() }),
  });
  const createChart = vi.fn(() => {
    const crosshair: ((p: { time?: number }) => void)[] = [];
    const clicks: ((p: { time?: number }) => void)[] = [];
    const scale = {
      fitContent: vi.fn(),
      // Any monotone-ish projection will do; the component only has to turn
      // whatever comes back into a band rectangle.
      timeToCoordinate: vi.fn((t: number) => (t / 1e6) % 700),
      subscribeVisibleLogicalRangeChange: vi.fn(),
      unsubscribeVisibleLogicalRangeChange: vi.fn(),
    };
    return {
      __crosshair: crosshair,
      __clicks: clicks,
      __scale: scale,
      addLineSeries: vi.fn(makeSeries),
      addAreaSeries: vi.fn(makeSeries),
      addCandlestickSeries: vi.fn(makeSeries),
      subscribeCrosshairMove: vi.fn((h: (p: { time?: number }) => void) => {
        crosshair.push(h);
      }),
      unsubscribeCrosshairMove: vi.fn(),
      subscribeClick: vi.fn((h: (p: { time?: number }) => void) => {
        clicks.push(h);
      }),
      unsubscribeClick: vi.fn(),
      timeScale: vi.fn(() => scale),
      setCrosshairPosition: vi.fn(),
      clearCrosshairPosition: vi.fn(),
      resize: vi.fn(),
      applyOptions: vi.fn(),
      remove: vi.fn(),
    };
  });
  return { createChart, ColorType: { Solid: "solid", VerticalGradient: "gradient" } };
});

vi.mock("echarts/core", () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
  })),
  use: vi.fn(),
  registerTheme: vi.fn(),
}));
vi.mock("echarts/charts", () => ({
  BarChart: {},
  HeatmapChart: {},
  LineChart: {},
  ScatterChart: {},
}));
vi.mock("echarts/components", () => ({
  GridComponent: {},
  LegendComponent: {},
  TooltipComponent: {},
  VisualMapComponent: {},
}));
vi.mock("echarts/renderers", () => ({ CanvasRenderer: {} }));

vi.mock("../lib/api", () => ({
  api: { bars: vi.fn(), runs: vi.fn(), decisions: vi.fn() },
  ApiError: class ApiError extends Error {},
  setToken: vi.fn(),
  subscribeEvents: vi.fn(() => () => undefined),
}));

import { createChart } from "lightweight-charts";
import { init } from "echarts/core";

import { api } from "../lib/api";
import { disposeTimeLock, timeLock, toChartTime, toMs } from "../lib/timelock";
import type {
  AgentCall,
  DecisionRow,
  EquitySeries,
  RunRecord,
  TradeRow,
} from "../lib/types";

import { EChart } from "../components/charts/EChart";
import {
  EquityChart,
  oosFraction,
  oosSpans,
  oosTimeSpans,
  type Span,
} from "../components/charts/EquityChart";
import { PriceChart, tradeMarkers } from "../components/charts/PriceChart";
import {
  ROW_CAP,
  TradeLedger,
  ledgerRows,
  ledgerTotals,
  sortTrades,
} from "../components/TradeLedger";
import {
  AgentLedger,
  SpendMeter,
  agentEntries,
  pickRun,
  readBudget,
  spendFraction,
  spendLevel,
} from "../components/AgentLedger";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// --- environment stubs -------------------------------------------------------

class StubResizeObserver {
  constructor(_cb: unknown) {}
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = StubResizeObserver;

// jsdom reports every element as zero-width; the band geometry needs a box.
Object.defineProperty(HTMLElement.prototype, "clientWidth", {
  configurable: true,
  get: () => 800,
});

// --- chart stub access -------------------------------------------------------

interface ChartStub {
  __crosshair: ((p: { time?: number }) => void)[];
  __clicks: ((p: { time?: number }) => void)[];
  setCrosshairPosition: Mock;
  clearCrosshairPosition: Mock;
  remove: Mock;
}

function lastChart(): ChartStub {
  const results = vi.mocked(createChart).mock.results;
  expect(results.length).toBeGreaterThan(0);
  return results[results.length - 1].value as unknown as ChartStub;
}

interface EChartStub {
  setOption: Mock;
  dispose: Mock;
  on: Mock;
}

function lastEChart(): EChartStub {
  const results = vi.mocked(init).mock.results;
  expect(results.length).toBeGreaterThan(0);
  return results[results.length - 1].value as unknown as EChartStub;
}

// --- harness -----------------------------------------------------------------

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function mount(node: React.ReactNode): Promise<HTMLDivElement> {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
  });
  await settle();
  return container;
}

async function settle(turns = 6): Promise<void> {
  for (let i = 0; i < turns; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

const text = (el: Element | null): string => (el?.textContent ?? "").replace(/\s+/g, " ").trim();

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.clearAllMocks();
});

// --- fixtures ----------------------------------------------------------------

const DAY = 86_400_000;
const T0 = Date.UTC(2024, 0, 2);

const isoAt = (i: number): string => new Date(T0 + i * DAY).toISOString();

function equitySeries(n: number, oos: (i: number) => boolean = () => false): EquitySeries {
  return {
    run_id: "r-test",
    n,
    t: Array.from({ length: n }, (_, i) => isoAt(i)),
    equity: Array.from({ length: n }, (_, i) => 100_000 + i * 137),
    drawdown: Array.from({ length: n }, (_, i) => -((i % 7) / 100)),
    is_oos: Array.from({ length: n }, (_, i) => oos(i)),
  };
}

const trade = (t: Partial<TradeRow> & { ticker: string }): TradeRow => ({
  side: "buy",
  qty: 100,
  entry_time: isoAt(1),
  entry_price: 220.5,
  exit_time: isoAt(4),
  exit_price: 231.25,
  pnl: 1075,
  pnl_pct: 0.0487,
  bars_held: 3,
  commission: 1.2,
  tag: "",
  exit_reason: "target",
  ...t,
});

const runRecord = (r: Partial<RunRecord> & { run_id: string }): RunRecord => ({
  strategy: "agent_daily",
  kind: "paper",
  status: "ok",
  created_at: "2026-02-01T10:00:00Z",
  finished_at: null,
  git_commit: "a1c2f9e4",
  config_hash: "7d21ab90",
  data_version: "v41",
  params: {},
  config: {},
  metrics: {},
  start: null,
  end: null,
  origin: "human",
  parent_run_id: null,
  sweep_id: null,
  notes: "",
  attempt: 1,
  error: "",
  ...r,
});

const agentCall = (c: Partial<AgentCall> = {}): AgentCall => ({
  model: "claude-sonnet",
  prompt: "positions: none\nsignals: 2",
  response: { targets: [{ ticker: "NVDA", pct: 0.04 }] },
  rationale: "momentum confirmed by two independent sources",
  input_tokens: 1840,
  output_tokens: 210,
  cost_usd: 0.3,
  latency_ms: 1420,
  ...c,
});

const decision = (d: Partial<DecisionRow> & { id: string }): DecisionRow => ({
  run_id: "run-agent-1",
  strategy: "agent_daily",
  at: isoAt(1),
  summary: "1 intent · gate: clipped",
  inputs: { indicators: {}, signals: [], prices: {} },
  intents: [
    { ticker: "NVDA", target_pct: 0.04, tag: "", reason: "momentum", limit_price: null, meta: {} },
  ],
  verdicts: [
    {
      ticker: "NVDA",
      action: "clipped",
      requested_pct: 0.04,
      approved_pct: 0.031,
      rule: "position_cap",
      detail: "clipped to 3.1%",
    },
  ],
  order_ids: [],
  portfolio: {
    at: isoAt(1),
    cash: 50_000,
    equity: 103_412,
    gross_exposure: 0.51,
    net_exposure: 0.51,
    n_positions: 3,
    positions: [],
  },
  logs: [],
  agent: agentCall(),
  duration_ms: 1500,
  ...d,
});

// --- toChartTime -------------------------------------------------------------

describe("toChartTime", () => {
  it("returns seconds, not milliseconds", () => {
    const iso = "2024-01-02T00:00:00Z";
    expect(toChartTime(iso)).toBe(1_704_153_600);
    expect(toMs(iso)).toBe(1_704_153_600_000);
    // The 1000x trap: the two must differ by exactly three orders of magnitude.
    expect(toMs(iso) / toChartTime(iso)).toBe(1000);
  });

  it("floors sub-second precision instead of rounding up past the bar", () => {
    expect(toChartTime(1_704_153_600_999)).toBe(1_704_153_600);
    expect(toChartTime(1_704_153_600_001)).toBe(1_704_153_600);
  });

  it("accepts a number, an ISO string and a Date identically", () => {
    const ms = Date.UTC(2021, 5, 14, 13, 30, 0);
    expect(toChartTime(ms)).toBe(toChartTime(new Date(ms).toISOString()));
    expect(toChartTime(ms)).toBe(toChartTime(new Date(ms)));
  });
});

// --- oosSpans ----------------------------------------------------------------

/** Paint spans back onto a flag array, so a round trip can be asserted. */
function rebuild(n: number, spans: readonly Span[]): boolean[] {
  const out = new Array<boolean>(n).fill(false);
  for (const s of spans) for (let i = s.start; i <= s.end; i++) out[i] = true;
  return out;
}

describe("oosSpans", () => {
  it("returns nothing for empty or absent input", () => {
    expect(oosSpans([])).toEqual([]);
    expect(oosSpans(null)).toEqual([]);
    expect(oosSpans(undefined)).toEqual([]);
  });

  it("returns nothing when the whole run is in-sample", () => {
    expect(oosSpans([false, false, false])).toEqual([]);
  });

  it("covers the whole range when the whole run is out-of-sample", () => {
    expect(oosSpans([true])).toEqual([{ start: 0, end: 0 }]);
    expect(oosSpans([true, true, true])).toEqual([{ start: 0, end: 2 }]);
  });

  it("closes a span that ends at the last point", () => {
    expect(oosSpans([false, true, true])).toEqual([{ start: 1, end: 2 }]);
  });

  it("opens a span that starts at the first point", () => {
    expect(oosSpans([true, true, false, false])).toEqual([{ start: 0, end: 1 }]);
  });

  it("finds every disjoint span, including single points", () => {
    expect(oosSpans([false, true, true, false, true, false, false, true])).toEqual([
      { start: 1, end: 2 },
      { start: 4, end: 4 },
      { start: 7, end: 7 },
    ]);
  });

  it("handles a fully alternating flag array", () => {
    expect(oosSpans([true, false, true, false, true])).toEqual([
      { start: 0, end: 0 },
      { start: 2, end: 2 },
      { start: 4, end: 4 },
    ]);
  });

  it("finds one span per window on a walk-forward shape", () => {
    // Four 10-point windows, each 6 in-sample then 4 out-of-sample.
    const flags = Array.from({ length: 40 }, (_, i) => i % 10 >= 6);
    const spans = oosSpans(flags);
    expect(spans).toHaveLength(4);
    expect(spans[0]).toEqual({ start: 6, end: 9 });
    expect(spans[3]).toEqual({ start: 36, end: 39 });
    expect(rebuild(flags.length, spans)).toEqual(flags);
  });

  it("round-trips any flag array back to itself", () => {
    // Deterministic LCG: a failure has to be reproducible, so no Math.random.
    let seed = 20260222;
    const next = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let trial = 0; trial < 300; trial++) {
      const n = 1 + Math.floor(next() * 60);
      const p = next();
      const flags = Array.from({ length: n }, () => next() < p);
      const spans = oosSpans(flags);
      expect(rebuild(n, spans)).toEqual(flags);
      // Maximal and disjoint: no span touches the next one.
      for (let i = 1; i < spans.length; i++) {
        expect(spans[i].start).toBeGreaterThan(spans[i - 1].end + 1);
      }
      for (const s of spans) expect(s.end).toBeGreaterThanOrEqual(s.start);
    }
  });
});

describe("oosFraction", () => {
  it("is zero for an empty or fully in-sample run", () => {
    expect(oosFraction([])).toBe(0);
    expect(oosFraction(null)).toBe(0);
    expect(oosFraction([false, false])).toBe(0);
  });

  it("counts flags, not spans", () => {
    expect(oosFraction([true, true, false, false])).toBe(0.5);
    expect(oosFraction([true, false, true, false, true, false, true, false])).toBe(0.5);
    expect(oosFraction([true])).toBe(1);
  });
});

describe("oosTimeSpans", () => {
  it("converts span edges to chart seconds", () => {
    const t = [isoAt(0), isoAt(1), isoAt(2), isoAt(3)];
    expect(oosTimeSpans(t, [false, true, true, false])).toEqual([
      { from: toChartTime(t[1]), to: toChartTime(t[2]) },
    ]);
  });

  it("clamps to the shorter array rather than shading the wrong place", () => {
    const t = [isoAt(0), isoAt(1)];
    expect(oosTimeSpans(t, [true, true, true, true])).toEqual([
      { from: toChartTime(t[0]), to: toChartTime(t[1]) },
    ]);
    expect(oosTimeSpans(t, [])).toEqual([]);
    expect(oosTimeSpans([], [true])).toEqual([]);
  });
});

// --- EChart ------------------------------------------------------------------

describe("EChart", () => {
  it("initialises, sets the option and disposes on unmount", async () => {
    const option = { series: [{ type: "bar", data: [1, 2, 3] }] };
    await mount(<EChart option={option} height={120} />);

    expect(vi.mocked(init)).toHaveBeenCalledTimes(1);
    const chart = lastEChart();
    expect(chart.setOption).toHaveBeenCalledWith(option, { notMerge: true });

    act(() => root?.unmount());
    root = null;
    expect(chart.dispose).toHaveBeenCalledTimes(1);
  });

  it("binds the handlers it is given", async () => {
    await mount(<EChart option={{}} onEvent={{ click: () => undefined }} />);
    expect(lastEChart().on).toHaveBeenCalledWith("click", expect.any(Function));
  });
});

// --- EquityChart -------------------------------------------------------------

describe("EquityChart", () => {
  it("says what to run instead of drawing an empty frame", async () => {
    const el = await mount(
      <EquityChart
        scope="eq-empty"
        data={{ run_id: "r-empty", n: 0, t: [], equity: [], drawdown: [], is_oos: [] }}
      />,
    );
    expect(text(el)).toContain("lab backtest strategies/momo.py");
    expect(vi.mocked(createChart)).not.toHaveBeenCalled();
  });

  it("survives a one-point series", async () => {
    const el = await mount(<EquityChart scope="eq-one" data={equitySeries(1, () => true)} />);
    expect(vi.mocked(createChart)).toHaveBeenCalledTimes(1);
    expect(text(el)).toContain("out of sample");
    disposeTimeLock("eq-one");
  });

  it("shades one band per out-of-sample span and states the share", async () => {
    // Two disjoint OOS stretches out of twenty points.
    const data = equitySeries(20, (i) => (i >= 5 && i <= 8) || i >= 16);
    const el = await mount(<EquityChart scope="eq-oos" data={data} />);

    expect(el.querySelectorAll("[data-oos-band]")).toHaveLength(2);
    const summary = text(el.querySelector("[data-oos-summary]"));
    expect(summary).toContain("2 spans");
    expect(summary).toContain("40%"); // 8 of 20 points
    disposeTimeLock("eq-oos");
  });

  it("calls out a curve with no out-of-sample section at all", async () => {
    const el = await mount(<EquityChart scope="eq-is" data={equitySeries(10)} />);
    expect(text(el.querySelector("[data-oos-summary]"))).toContain("in-sample only");
    expect(el.querySelectorAll("[data-oos-band]")).toHaveLength(0);
    disposeTimeLock("eq-is");
  });

  it("seeks the lock on click, in milliseconds, and does not echo its own move", async () => {
    const scope = "eq-lock";
    const store = timeLock(scope);
    await mount(<EquityChart scope={scope} data={equitySeries(12, (i) => i > 8)} />);

    const chart = lastChart();
    const seconds = toChartTime(isoAt(3));
    await act(async () => {
      chart.__clicks[0]({ time: seconds });
    });

    expect(store.getState().t).toBe(seconds * 1000);
    const own = store.getState().source;
    expect(own).toMatch(/^equity-chart/);

    // Our own seek must not push the crosshair back: two synced charts doing
    // that to each other is an infinite loop, not a feature.
    expect(chart.setCrosshairPosition).not.toHaveBeenCalled();

    // A move from anywhere else does move the crosshair.
    await act(async () => {
      store.seek(toMs(isoAt(6)), "trade-ledger");
    });
    expect(chart.setCrosshairPosition).toHaveBeenCalledTimes(1);

    // And a further move attributed to us is ignored again.
    await act(async () => {
      store.seek(toMs(isoAt(7)), own ?? "");
    });
    expect(chart.setCrosshairPosition).toHaveBeenCalledTimes(1);

    disposeTimeLock(scope);
  });

  it("publishes crosshair movement as a hover, in milliseconds", async () => {
    const scope = "eq-hover";
    const store = timeLock(scope);
    await mount(<EquityChart scope={scope} data={equitySeries(8)} />);
    const chart = lastChart();

    const seconds = toChartTime(isoAt(2));
    await act(async () => {
      chart.__crosshair[0]({ time: seconds });
    });
    expect(store.getState().hoverT).toBe(seconds * 1000);

    await act(async () => {
      chart.__crosshair[0]({});
    });
    expect(store.getState().hoverT).toBeNull();
    disposeTimeLock(scope);
  });
});

// --- PriceChart --------------------------------------------------------------

describe("tradeMarkers", () => {
  it("keeps only the requested ticker", () => {
    const ms = tradeMarkers([trade({ ticker: "NVDA" }), trade({ ticker: "AAPL" })], "NVDA");
    expect(ms).toHaveLength(2);
  });

  it("emits an entry with no exit for an open trade", () => {
    const ms = tradeMarkers(
      [trade({ ticker: "NVDA", exit_time: null, exit_price: null, pnl: 0 })],
      "NVDA",
    );
    expect(ms).toHaveLength(1);
    expect(ms[0].shape).toBe("arrowUp");
    expect(ms[0].position).toBe("belowBar");
  });

  it("sorts markers ascending, which the library requires", () => {
    const ms = tradeMarkers(
      [
        trade({ ticker: "NVDA", entry_time: isoAt(1), exit_time: isoAt(9) }),
        trade({ ticker: "NVDA", entry_time: isoAt(3), exit_time: isoAt(4) }),
      ],
      "NVDA",
    );
    const times = ms.map((m) => Number(m.time));
    expect(times).toEqual([...times].sort((a, b) => a - b));
  });

  it("colours the pair by realised P&L, not by direction of the arrow", () => {
    const win = tradeMarkers([trade({ ticker: "X", pnl: 100 })], "X");
    const loss = tradeMarkers([trade({ ticker: "X", pnl: -100 })], "X");
    expect(new Set(win.map((m) => m.color)).size).toBe(1);
    expect(win[0].color).not.toBe(loss[0].color);
  });

  it("copes with no trades at all", () => {
    expect(tradeMarkers(undefined, "NVDA")).toEqual([]);
    expect(tradeMarkers([], "NVDA")).toEqual([]);
  });
});

describe("PriceChart", () => {
  beforeEach(() => {
    vi.mocked(api.bars).mockResolvedValue({
      ticker: "NVDA",
      t: Array.from({ length: 6 }, (_, i) => isoAt(i)),
      open: [10, 11, 12, 13, 14, 15],
      high: [11, 12, 13, 14, 15, 16],
      low: [9, 10, 11, 12, 13, 14],
      close: [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
      volume: [1, 1, 1, 1, 1, 1],
    });
  });

  it("mounts, fetches once and reports the marker counts", async () => {
    const el = await mount(
      <PriceChart
        scope="px-1"
        runId="run-1"
        ticker="NVDA"
        trades={[trade({ ticker: "NVDA" }), trade({ ticker: "NVDA", exit_time: null })]}
      />,
    );
    expect(vi.mocked(api.bars)).toHaveBeenCalledWith("run-1", "NVDA");
    const body = text(el);
    expect(body).toContain("2 entries");
    expect(body).toContain("1 exits");
    disposeTimeLock("px-1");
  });

  it("says what to run when the run stored no bars", async () => {
    vi.mocked(api.bars).mockResolvedValue({
      ticker: "NVDA",
      t: [],
      open: [],
      high: [],
      low: [],
      close: [],
      volume: [],
    });
    const el = await mount(<PriceChart scope="px-2" runId="run-1" ticker="NVDA" />);
    expect(text(el)).toContain("lab data fetch NVDA");
    disposeTimeLock("px-2");
  });

  it("surfaces a read failure instead of an empty frame", async () => {
    vi.mocked(api.bars).mockRejectedValue(new Error("no parquet for NVDA"));
    const el = await mount(<PriceChart scope="px-3" runId="run-1" ticker="NVDA" />);
    expect(text(el)).toContain("no parquet for NVDA");
    disposeTimeLock("px-3");
  });

  it("seeks on click and moves the crosshair only for foreign seeks", async () => {
    const scope = "px-lock";
    const store = timeLock(scope);
    await mount(<PriceChart scope={scope} runId="run-1" ticker="NVDA" />);
    const chart = lastChart();

    const seconds = toChartTime(isoAt(2));
    await act(async () => {
      chart.__clicks[0]({ time: seconds });
    });
    expect(store.getState().t).toBe(seconds * 1000);
    expect(store.getState().source).toMatch(/^price-chart/);
    expect(chart.setCrosshairPosition).not.toHaveBeenCalled();

    await act(async () => {
      store.seek(toMs(isoAt(4)), "decision-tape");
    });
    expect(chart.setCrosshairPosition).toHaveBeenCalledTimes(1);
    disposeTimeLock(scope);
  });
});

// --- TradeLedger -------------------------------------------------------------

const LEDGER: TradeRow[] = [
  trade({ ticker: "NVDA", entry_time: isoAt(1), exit_time: isoAt(4), pnl: 1075, pnl_pct: 0.048 }),
  trade({ ticker: "AAPL", entry_time: isoAt(2), exit_time: isoAt(3), pnl: -240, pnl_pct: -0.011 }),
  trade({
    ticker: "MSFT",
    entry_time: isoAt(3),
    exit_time: null,
    exit_price: null,
    pnl: 0,
    pnl_pct: 0,
    exit_reason: "",
  }),
];

describe("sortTrades", () => {
  it("orders by the requested key in both directions", () => {
    const rows = ledgerRows(LEDGER);
    expect(sortTrades(rows, "pnl", "desc").map((r) => r.trade.ticker)).toEqual([
      "NVDA",
      "MSFT",
      "AAPL",
    ]);
    expect(sortTrades(rows, "pnl", "asc").map((r) => r.trade.ticker)).toEqual([
      "AAPL",
      "MSFT",
      "NVDA",
    ]);
    expect(sortTrades(rows, "ticker", "asc").map((r) => r.trade.ticker)).toEqual([
      "AAPL",
      "MSFT",
      "NVDA",
    ]);
  });

  it("keeps missing values last in both directions", () => {
    const rows = ledgerRows(LEDGER);
    // MSFT is still open: it has no exit price, so it can never win the column.
    expect(sortTrades(rows, "exit_price", "desc")[2].trade.ticker).toBe("MSFT");
    expect(sortTrades(rows, "exit_price", "asc")[2].trade.ticker).toBe("MSFT");
  });

  it("is stable, so a symbol's trades stay chronological", () => {
    const rows = ledgerRows([
      trade({ ticker: "AAPL", entry_time: isoAt(1), pnl: 5 }),
      trade({ ticker: "AAPL", entry_time: isoAt(2), pnl: 5 }),
      trade({ ticker: "AAPL", entry_time: isoAt(3), pnl: 5 }),
    ]);
    expect(sortTrades(rows, "pnl", "desc").map((r) => r.i)).toEqual([0, 1, 2]);
    expect(sortTrades(rows, "ticker", "desc").map((r) => r.i)).toEqual([0, 1, 2]);
  });

  it("does not mutate its input", () => {
    const rows = ledgerRows(LEDGER);
    const before = rows.map((r) => r.id);
    sortTrades(rows, "pnl", "desc");
    expect(rows.map((r) => r.id)).toEqual(before);
  });
});

describe("ledgerTotals", () => {
  it("counts hit rate over closed trades only", () => {
    const totals = ledgerTotals(LEDGER);
    expect(totals.n).toBe(3);
    expect(totals.closed).toBe(2);
    expect(totals.wins).toBe(1);
    expect(totals.hitRate).toBe(0.5);
    expect(totals.pnl).toBe(835);
  });

  it("reports no hit rate rather than zero when nothing has closed", () => {
    expect(ledgerTotals([trade({ ticker: "X", exit_time: null })]).hitRate).toBeNull();
  });
});

describe("TradeLedger", () => {
  it("says what to run when the run closed no trades", async () => {
    const el = await mount(<TradeLedger scope="tl-empty" trades={[]} />);
    expect(text(el)).toContain("lab backtest strategies/momo.py");
  });

  it("re-sorts when a header is clicked and shows the direction", async () => {
    const el = await mount(<TradeLedger scope="tl-sort" trades={LEDGER} />);
    const firstTicker = () => text(el.querySelector("tbody tr td"));
    expect(firstTicker()).toBe("NVDA"); // entry_time ascending

    const pnlHeader = el.querySelectorAll("thead th")[7];
    expect(text(pnlHeader)).toContain("p&l");

    await act(async () => {
      (pnlHeader as HTMLElement).click();
    });
    expect(firstTicker()).toBe("NVDA"); // biggest P&L first
    expect(text(pnlHeader)).toContain("▼");

    await act(async () => {
      (pnlHeader as HTMLElement).click();
    });
    expect(firstTicker()).toBe("AAPL"); // smallest P&L first
    expect(text(pnlHeader)).toContain("▲");
    disposeTimeLock("tl-sort");
  });

  it("selects into the time lock and hands the ticker back to the screen", async () => {
    const scope = "tl-select";
    const store = timeLock(scope);
    const seen: string[] = [];
    const el = await mount(
      <TradeLedger scope={scope} trades={LEDGER} onSelectTicker={(t) => seen.push(t)} />,
    );

    const button = el.querySelector<HTMLButtonElement>("tbody tr button");
    await act(async () => {
      button?.click();
    });

    const state = store.getState();
    expect(state.selection.kind).toBe("trade");
    expect(state.selection.ticker).toBe("NVDA");
    expect(state.selection.id).toBe(`0:NVDA:${isoAt(1)}`);
    expect(state.t).toBe(toMs(isoAt(1)));
    expect(state.source).toBe("trade-ledger");
    expect(seen).toEqual(["NVDA"]);

    // The selected row is marked, so the operator can see where they are.
    expect(el.querySelector("tbody tr")?.className).toContain("bg-teal-wash");
    disposeTimeLock(scope);
  });

  it("caps the rendered rows and says so out loud", async () => {
    const many = Array.from({ length: ROW_CAP + 12 }, (_, i) =>
      trade({ ticker: `T${i}`, entry_time: isoAt(i % 300) }),
    );
    const el = await mount(<TradeLedger scope="tl-cap" trades={many} />);
    expect(el.querySelectorAll("tbody tr")).toHaveLength(ROW_CAP);
    expect(text(el)).toContain(`showing 1,000 of ${(ROW_CAP + 12).toLocaleString("en-US")}`);
    disposeTimeLock("tl-cap");
  });
});

// --- the cost meter ----------------------------------------------------------

describe("spendLevel", () => {
  it("stays quiet while there is room", () => {
    expect(spendLevel(0, 1)).toBe("ok");
    expect(spendLevel(0.5, 1)).toBe("ok");
    expect(spendLevel(0.79, 1)).toBe("ok");
  });

  it("warns from 80% of the budget", () => {
    expect(spendLevel(0.8, 1)).toBe("near");
    expect(spendLevel(0.999, 1)).toBe("near");
    // Exactly at the ceiling is not yet through it.
    expect(spendLevel(1, 1)).toBe("near");
  });

  it("goes ember once past the ceiling", () => {
    expect(spendLevel(1.0001, 1)).toBe("over");
    expect(spendLevel(40, 10)).toBe("over");
  });

  it("has nothing to warn about when no budget was declared", () => {
    expect(spendLevel(1000, null)).toBe("ok");
    expect(spendLevel(1000, undefined)).toBe("ok");
    expect(spendLevel(1000, 0)).toBe("ok");
    expect(spendLevel(Number.NaN, 10)).toBe("ok");
  });
});

describe("spendFraction", () => {
  it("clamps to the bar, so an overrun cannot paint off-panel", () => {
    expect(spendFraction(0.25, 1)).toBe(0.25);
    expect(spendFraction(5, 1)).toBe(1);
    expect(spendFraction(-1, 1)).toBe(0);
    expect(spendFraction(1, null)).toBe(0);
  });
});

describe("readBudget", () => {
  it("takes the first declared ceiling it finds", () => {
    expect(readBudget({ agent_budget_usd: 2.5 })).toBe(2.5);
    expect(readBudget({}, { daily_budget_usd: 4 })).toBe(4);
    expect(readBudget({ budget_usd: 0 }, { max_cost_usd: 7 })).toBe(7);
  });

  it("returns null rather than inventing one", () => {
    expect(readBudget({}, undefined, null)).toBeNull();
    expect(readBudget({ agent_budget_usd: "lots" })).toBeNull();
  });
});

describe("SpendMeter", () => {
  it("crosses from neutral to amber to ember", async () => {
    for (const [spent, level] of [
      [0.2, "ok"],
      [0.85, "near"],
      [1.4, "over"],
    ] as const) {
      const el = await mount(<SpendMeter spent={spent} budget={1} />);
      expect(el.querySelector("[data-spend]")?.getAttribute("data-spend")).toBe(level);
      act(() => root?.unmount());
      container?.remove();
      root = null;
      container = null;
    }
  });

  it("says so when the run declared no budget", async () => {
    const el = await mount(<SpendMeter spent={9} budget={null} />);
    expect(text(el)).toContain("no budget declared");
    expect(el.querySelector("[role=meter]")).toBeNull();
  });
});

// --- AgentLedger -------------------------------------------------------------

describe("agentEntries", () => {
  it("keeps only decisions that called the model and accumulates the spend", () => {
    const entries = agentEntries([
      decision({ id: "d1", agent: agentCall({ cost_usd: 0.1 }) }),
      decision({ id: "d2", agent: null }),
      decision({ id: "d3", agent: agentCall({ cost_usd: 0.25 }) }),
    ]);
    expect(entries.map((e) => e.decision.id)).toEqual(["d1", "d3"]);
    expect(entries[1].cum).toBeCloseTo(0.35, 10);
  });

  it("treats a missing cost as zero rather than poisoning the running total", () => {
    const entries = agentEntries([
      decision({ id: "d1", agent: agentCall({ cost_usd: Number.NaN }) }),
      decision({ id: "d2", agent: agentCall({ cost_usd: 0.5 }) }),
    ]);
    expect(entries[1].cum).toBe(0.5);
  });
});

describe("pickRun", () => {
  it("prefers the money that is moving now", () => {
    const runs = [
      runRecord({ run_id: "bt", kind: "backtest", created_at: "2026-03-01T00:00:00Z" }),
      runRecord({ run_id: "paper", kind: "paper", created_at: "2026-01-01T00:00:00Z" }),
    ];
    expect(pickRun(runs)?.run_id).toBe("paper");
  });

  it("falls back to the newest run and to nothing at all", () => {
    const runs = [
      runRecord({ run_id: "old", kind: "backtest", created_at: "2025-01-01T00:00:00Z" }),
      runRecord({ run_id: "new", kind: "backtest", created_at: "2026-01-01T00:00:00Z" }),
    ];
    expect(pickRun(runs)?.run_id).toBe("new");
    expect(pickRun([])).toBeNull();
  });
});

describe("AgentLedger", () => {
  const RUNS = [
    runRecord({
      run_id: "run-agent-1",
      kind: "paper",
      config: { agent_budget_usd: 1 },
      metrics: { contaminated: true },
    }),
  ];

  beforeEach(() => {
    vi.mocked(api.runs).mockResolvedValue({ runs: RUNS, total: 1 });
  });

  it("lists the calls, warns about contamination and runs the meter to amber", async () => {
    vi.mocked(api.decisions).mockResolvedValue({
      decisions: [
        decision({ id: "d1", at: isoAt(1), agent: agentCall({ cost_usd: 0.3 }) }),
        decision({ id: "d2", at: isoAt(2), agent: agentCall({ cost_usd: 0.3 }) }),
        decision({ id: "d3", at: isoAt(3), agent: agentCall({ cost_usd: 0.3 }) }),
      ],
      total: 3,
    });
    const el = await mount(<AgentLedger strategy="agent_daily" />);
    const body = text(el);

    expect(el.querySelectorAll("tbody tr").length).toBeGreaterThanOrEqual(3);
    expect(body).toContain("training data");
    expect(el.querySelector("[data-spend]")?.getAttribute("data-spend")).toBe("near");
    // Per-target gate verdict, requested against approved.
    expect(body).toContain("position_cap");
    disposeTimeLock("run-agent-1");
  });

  it("turns ember once the calls run past the budget", async () => {
    vi.mocked(api.decisions).mockResolvedValue({
      decisions: [
        decision({ id: "d1", agent: agentCall({ cost_usd: 0.8 }) }),
        decision({ id: "d2", agent: agentCall({ cost_usd: 0.8 }) }),
      ],
      total: 2,
    });
    const el = await mount(<AgentLedger strategy="agent_daily" />);
    expect(el.querySelector("[data-spend]")?.getAttribute("data-spend")).toBe("over");
    disposeTimeLock("run-agent-1");
  });

  it("renders model-authored text as text, never as markup", async () => {
    vi.mocked(api.decisions).mockResolvedValue({
      decisions: [
        decision({
          id: "d1",
          agent: agentCall({ rationale: "<img src=x onerror=alert(1)>bought the dip" }),
        }),
      ],
      total: 1,
    });
    const el = await mount(<AgentLedger strategy="agent_daily" />);
    expect(el.querySelector("img")).toBeNull();
    expect(text(el)).toContain("<img src=x onerror=alert(1)>bought the dip");
    disposeTimeLock("run-agent-1");
  });

  it("says what to run when the strategy has no runs at all", async () => {
    vi.mocked(api.runs).mockResolvedValue({ runs: [], total: 0 });
    const el = await mount(<AgentLedger strategy="agent_daily" />);
    expect(text(el)).toContain("lab paper run agent_daily");
  });

  it("says the run has no model in its path when nothing journalled a call", async () => {
    vi.mocked(api.decisions).mockResolvedValue({
      decisions: [decision({ id: "d1", agent: null })],
      total: 1,
    });
    const el = await mount(<AgentLedger strategy="agent_daily" />);
    expect(text(el)).toContain("no LLM in its runtime path");
    disposeTimeLock("run-agent-1");
  });
});
