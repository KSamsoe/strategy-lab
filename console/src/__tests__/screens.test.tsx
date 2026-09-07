/**
 * Render and behaviour tests for the fleet/live/forensics screens.
 *
 * Same harness as `runs-screens.test.tsx`: React's own root API under `act`,
 * a real in-memory router, and the API module mocked at the boundary. The
 * charts are mocked too — not to dodge canvas, but because what these tests
 * assert about a chart is which *instant and metric* it was handed, and a
 * stub div carries that in an attribute where a real canvas would not.
 *
 * The assertions concentrate on the four claims these screens exist to make:
 * staleness degrades instead of sitting green, the time lock actually moves
 * three panels at once, an in-sample surface is never shown under an
 * out-of-sample label, and the live screen has no control that adds exposure.
 */

import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../lib/api", () => ({
  api: {
    health: vi.fn(),
    runs: vi.fn(),
    run: vi.fn(),
    equity: vi.fn(),
    trades: vi.fn(),
    decisions: vi.fn(),
    bars: vi.fn(),
    compare: vi.fn(),
    sweep: vi.fn(),
    strategies: vi.fn(),
    liveStrategies: vi.fn(),
    liveEvents: vi.fn(),
    liveHealth: vi.fn(),
    lineage: vi.fn(),
    agentCalls: vi.fn(),
    pause: vi.fn(),
    cancelOrders: vi.fn(),
    kill: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
  setToken: vi.fn(),
  subscribeEvents: vi.fn(() => () => undefined),
}));

// Chart stubs. Everything referenced inside a vi.mock factory has to be created
// inside it: the factory runs while the module graph is still being imported,
// before this file's own top-level bindings exist.
vi.mock("../components/charts/EChart", async () => {
  const { createElement } = await import("react");
  const safe = (o: unknown): string =>
    JSON.stringify(o, (_k, v) => (typeof v === "function" ? "fn" : v)) ?? "";
  return {
    EChart: ({ option }: { option: unknown }) =>
      createElement("div", { "data-testid": "echart", "data-option": safe(option) }),
  };
});

vi.mock("../components/charts/EquityChart", async () => {
  const { createElement } = await import("react");
  return {
    EquityChart: ({ scope }: { scope: string }) =>
      createElement("div", { "data-testid": "equity-chart", "data-scope": scope }),
  };
});

vi.mock("../components/charts/PriceChart", async () => {
  const { createElement } = await import("react");
  return {
    PriceChart: ({ scope, ticker, runId }: { scope: string; ticker: string; runId: string }) =>
      createElement("div", {
        "data-testid": "price-chart",
        "data-scope": scope,
        "data-ticker": ticker,
        "data-run": runId,
      }),
  };
});

vi.mock("../components/TradeLedger", async () => {
  const { createElement } = await import("react");
  return {
    TradeLedger: ({ trades }: { trades: { ticker: string }[] }) =>
      createElement(
        "div",
        { "data-testid": "trade-ledger", "data-rows": String(trades.length) },
        `${trades.length} trades`,
      ),
  };
});

import { api } from "../lib/api";
import { timeLock } from "../lib/timelock";
import type {
  DecisionRow,
  EquitySeries,
  JournalEvent,
  LiveHealth,
  LiveStrategy,
  Metrics,
  RunRecord,
  SweepResult,
  TradeRow,
} from "../lib/types";

import Fleet, {
  ageNow,
  adapterHealth,
  quotaHealth,
  sparkPath,
  staleness,
} from "../screens/Fleet";
import RunDetail, { tickersByActivity } from "../screens/RunDetail";
import Sweep, { gridAxes, indexCells, orderValues } from "../screens/Sweep";
import StrategyLive, {
  firstOfToday,
  liveRunIdOf,
  orderStates,
} from "../screens/StrategyLive";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// --- fixtures -----------------------------------------------------------------

const NOW = Date.now();
const iso = (msAgo: number): string => new Date(NOW - msAgo).toISOString();

const RUN_ID = "momo_v3-20260220T093000-8f3a1c";

const metrics = (m: Metrics): Metrics => ({
  cagr: 0.142,
  total_return: 1.84,
  sharpe: 1.31,
  max_drawdown: -0.184,
  trades: 312,
  hit_rate: 0.54,
  exposure: 0.72,
  turnover: 3.1,
  final_equity: 284_000,
  starting_equity: 100_000,
  fill_mode: "same_bar_close",
  ...m,
});

const run = (r: Partial<RunRecord> = {}): RunRecord => ({
  run_id: RUN_ID,
  strategy: "momo_v3",
  kind: "backtest",
  status: "ok",
  created_at: "2026-02-20T09:30:00Z",
  finished_at: "2026-02-20T09:34:00Z",
  git_commit: "a1c2f9e4d1b7",
  config_hash: "7d21ab90cc",
  data_version: "v41",
  params: { lookback: 20, top_n: 5 },
  config: {},
  metrics: metrics({}),
  start: "2018-01-02T00:00:00Z",
  end: "2026-01-30T00:00:00Z",
  origin: "human",
  parent_run_id: null,
  sweep_id: null,
  notes: "",
  attempt: 4,
  error: "",
  ...r,
});

const EQUITY: EquitySeries = {
  run_id: "r-test",
  n: 3,
  t: ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-06T00:00:00Z"],
  equity: [100_000, 101_200, 100_400],
  drawdown: [0, 0, -0.0079],
  is_oos: [false, false, true],
  benchmark: null,
};

const trade = (t: Partial<TradeRow> & { ticker: string }): TradeRow => ({
  side: "long",
  qty: 100,
  entry_time: "2026-01-02T14:31:00Z",
  entry_price: 227.14,
  exit_time: "2026-01-06T20:00:00Z",
  exit_price: 231.02,
  pnl: 388,
  pnl_pct: 0.017,
  bars_held: 3,
  commission: 1.2,
  tag: "momo",
  exit_reason: "signal",
  ...t,
});

const TRADES: TradeRow[] = [
  trade({ ticker: "AAPL" }),
  trade({ ticker: "AAPL", entry_time: "2026-01-08T14:31:00Z" }),
  trade({ ticker: "NVDA", pnl: -120, pnl_pct: -0.009 }),
];

const decision = (d: Partial<DecisionRow> & { id: string; at: string }): DecisionRow => ({
  run_id: RUN_ID,
  strategy: "momo_v3",
  summary: "saw 2 signals · intent NVDA +4.0% · 1 order(s)",
  inputs: {},
  intents: [],
  verdicts: [],
  order_ids: [],
  portfolio: {
    at: d.at,
    cash: 10_000,
    equity: 100_000,
    gross_exposure: 0.9,
    net_exposure: 0.9,
    n_positions: 2,
    positions: [],
  },
  logs: [],
  agent: null,
  duration_ms: 12,
  ...d,
});

const DECISIONS: DecisionRow[] = [
  decision({ id: "d1", at: "2026-01-02T14:30:00Z" }),
  decision({
    id: "d2",
    at: "2026-01-06T14:30:00Z",
    intents: [
      { ticker: "NVDA", target_pct: 0.04, tag: "momo", reason: "breakout", limit_price: null, meta: {} },
    ],
  }),
];

const HEALTH: LiveHealth = {
  adapters: [
    {
      name: "alpaca",
      provides: ["bars", "orders"],
      available: true,
      reason: "",
      quota_used: null,
      quota_limit: null,
      quota_tier: null,
      last_call: iso(2_000),
    },
    {
      name: "govgreed",
      provides: ["signals"],
      available: true,
      reason: "",
      quota_used: 14,
      quota_limit: 20,
      quota_tier: "starter",
      last_call: iso(60_000),
    },
  ],
  heartbeats: [
    { source: "runner", at: iso(2_000), age_s: 2, meta: { period_s: 30 } },
    // Two hours quiet against a 30s beat: this must not read as healthy.
    { source: "broker", at: iso(7_200_000), age_s: 7200, meta: { period_s: 30 } },
  ],
  breaker: { tripped: false, reason: "" },
  kill_switch: { engaged: false, reason: null },
  equity: 103_412,
  day_change_pct: 0.008,
};

const position = (p: Partial<LiveStrategy["positions"][number]> & { ticker: string }) => ({
  qty: 120,
  avg_price: 227.14,
  last_price: 229.4,
  market_value: 27_528,
  unrealized_pnl: 271.2,
  unrealized_pct: 0.0099,
  opened_at: iso(3_600_000),
  tag: "momo",
  ...p,
});

const LIVE: LiveStrategy[] = [
  {
    strategy: "momo_v3",
    mode: "paper",
    status: "running",
    pnl_pct: 0.012,
    equity: 103_412,
    n_positions: 2,
    gate_events_today: 0,
    next_fire: iso(-900_000),
    started_at: iso(14_400_000),
    positions: [position({ ticker: "AAPL" }), position({ ticker: "NVDA" })],
  },
  {
    strategy: "govgreed_sig",
    mode: "paper",
    status: "paused",
    pnl_pct: -0.003,
    equity: 51_000,
    n_positions: 1,
    gate_events_today: 1,
    next_fire: iso(-3_600_000),
    started_at: iso(14_400_000),
    spend_usd: 0.42,
    positions: [],
  },
];

const event = (e: Partial<JournalEvent> & { seq: number }): JournalEvent => ({
  id: `ev-${e.seq}`,
  at: iso(60_000),
  kind: "log",
  source: "runner",
  run_id: "momo_v3-live-0001",
  strategy: "momo_v3",
  ticker: null,
  message: "tick",
  payload: {},
  ...e,
});

const EVENTS: JournalEvent[] = [
  event({ seq: 5, kind: "fill", ticker: "AAPL", message: "+120 @ 227.14", payload: { order_id: "o1" }, at: iso(30_000) }),
  event({ seq: 4, kind: "order", ticker: "NVDA", message: "buy 40 limit", payload: { order_id: "o2" }, at: iso(45_000) }),
  event({ seq: 3, kind: "order", ticker: "AAPL", message: "buy 120 market", payload: { order_id: "o1" }, at: iso(50_000) }),
  event({ seq: 2, kind: "gate_block", ticker: "NVDA", message: "sector cap 2/2", at: iso(70_000) }),
  event({ seq: 1, kind: "run_start", message: "session open", at: iso(3_600_000) }),
];

const sweepRow = (
  lookback: number,
  topN: number,
  isSharpe: number,
  oosSharpe: number,
): SweepResult["runs"][number] => ({
  run_id: `sw-${lookback}-${topN}`,
  params: { lookback, top_n: topN },
  is_metrics: metrics({ sharpe: isSharpe }),
  oos_metrics: metrics({ sharpe: oosSharpe }),
  score: oosSharpe,
});

const SWEEP: SweepResult = {
  sweep_id: "sweep-momo-0012",
  strategy: "momo_v3",
  grid: { lookback: [10, 20, 40], top_n: [3, 5] },
  metric: "sharpe",
  ranked_on: "out_of_sample",
  attempts: 417,
  truncated: null,
  runs: [
    sweepRow(10, 3, 1.9, 0.4),
    sweepRow(20, 3, 2.4, 0.9),
    sweepRow(40, 3, 1.1, 0.2),
    sweepRow(10, 5, 2.0, 0.5),
    sweepRow(20, 5, 2.8, 1.2),
    sweepRow(40, 5, 1.4, 0.3),
  ],
  best: sweepRow(20, 5, 2.8, 1.2),
  walk_forward: [
    {
      index: 1,
      is_start: "2018-01-02T00:00:00Z",
      is_end: "2021-12-31T00:00:00Z",
      oos_start: "2022-01-03T00:00:00Z",
      oos_end: "2022-12-30T00:00:00Z",
      is_metrics: metrics({ sharpe: 2.1 }),
      oos_metrics: metrics({ sharpe: 0.6 }),
    },
    {
      index: 2,
      is_start: "2019-01-02T00:00:00Z",
      is_end: "2022-12-30T00:00:00Z",
      oos_start: "2023-01-03T00:00:00Z",
      oos_end: "2023-12-29T00:00:00Z",
      is_metrics: metrics({ sharpe: 1.8 }),
      oos_metrics: metrics({ sharpe: -0.2 }),
    },
  ],
};

// --- harness -------------------------------------------------------------------

function makeRouter(path: string) {
  const rootRoute = createRootRoute({ component: () => <Outlet /> });
  const children = [
    createRoute({ getParentRoute: () => rootRoute, path: "/", component: Fleet }),
    createRoute({
      getParentRoute: () => rootRoute,
      path: "/runs/$runId",
      component: RunDetail,
    }),
    createRoute({
      getParentRoute: () => rootRoute,
      path: "/sweeps/$sweepId",
      component: Sweep,
    }),
    createRoute({
      getParentRoute: () => rootRoute,
      path: "/live/$strategy",
      component: StrategyLive,
    }),
    createRoute({
      getParentRoute: () => rootRoute,
      path: "/runs",
      component: () => <div>runs</div>,
      validateSearch: (s: Record<string, unknown>) => ({ strategy: (s.strategy as string) || undefined }),
    }),
    createRoute({
      getParentRoute: () => rootRoute,
      path: "/compare",
      component: () => <div>compare</div>,
      validateSearch: (s: Record<string, unknown>) => ({ ids: typeof s.ids === "string" ? s.ids : "" }),
    }),
  ];
  return createRouter({
    routeTree: rootRoute.addChildren(children),
    history: createMemoryHistory({ initialEntries: [path] }),
  });
}

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function mountAt(path: string): Promise<HTMLDivElement> {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  const router = makeRouter(path);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(
      <QueryClientProvider client={qc}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );
  });
  await settle();
  return container;
}

async function settle(turns = 8): Promise<void> {
  for (let i = 0; i < turns; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

const text = (el: Element | null): string => (el?.textContent ?? "").replace(/\s+/g, " ").trim();

/** The class on a StatusDot's coloured dot, found by its label. */
function dotClass(el: Element, label: string): string {
  const spans = Array.from(el.querySelectorAll("span"));
  const hit = spans.find((s) => s.textContent === label && s.className.includes("text-ash"));
  return hit?.parentElement?.firstElementChild?.className ?? "";
}

beforeEach(() => {
  vi.mocked(api.liveHealth).mockResolvedValue(HEALTH);
  vi.mocked(api.liveStrategies).mockResolvedValue({ strategies: LIVE });
  vi.mocked(api.liveEvents).mockResolvedValue({ events: EVENTS, latest_seq: 5 });
  vi.mocked(api.run).mockResolvedValue(run());
  vi.mocked(api.equity).mockResolvedValue(EQUITY);
  vi.mocked(api.trades).mockResolvedValue({ trades: TRADES });
  vi.mocked(api.decisions).mockResolvedValue({ decisions: DECISIONS, total: DECISIONS.length });
  vi.mocked(api.sweep).mockResolvedValue(SWEEP);
  vi.mocked(api.pause).mockResolvedValue({ ok: true, strategy: "momo_v3" });
  vi.mocked(api.cancelOrders).mockResolvedValue({ ok: true, canceled: 1 });
  vi.mocked(api.kill).mockResolvedValue({ ok: true, engaged: true, reason: "test" });
  // jsdom throws on the real ones.
  vi.stubGlobal("confirm", vi.fn(() => true));
  vi.stubGlobal("prompt", vi.fn(() => "test"));
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  timeLock(RUN_ID).reset();
  timeLock("live").reset();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

// --- pure helpers ----------------------------------------------------------------

describe("staleness", () => {
  it("degrades ok → warn → bad as a heartbeat goes quiet", () => {
    expect(staleness(2, 30)).toBe("ok");
    expect(staleness(60, 30)).toBe("ok");
    expect(staleness(61, 30)).toBe("warn");
    expect(staleness(300, 30)).toBe("warn");
    expect(staleness(301, 30)).toBe("bad");
  });

  it("is idle, not healthy, when nothing has been reported", () => {
    expect(staleness(null)).toBe("idle");
    expect(staleness(undefined)).toBe("idle");
    expect(staleness(Number.NaN)).toBe("idle");
  });

  it("judges each source against its own declared period", () => {
    // A six-hour adapter that answered an hour ago is fine; a 30s runner is not.
    expect(staleness(3600, 21_600)).toBe("ok");
    expect(staleness(3600, 30)).toBe("bad");
  });

  it("keeps aging after the last successful poll", () => {
    // The anchor is the server's age; the drift is why a dead poll loop still
    // turns the dot amber instead of freezing it green.
    const fetched = 1_000_000;
    expect(ageNow(2, fetched, fetched)).toBe(2);
    expect(ageNow(2, fetched, fetched + 600_000)).toBe(602);
    expect(staleness(ageNow(2, fetched, fetched + 600_000), 30)).toBe("bad");
    expect(ageNow(null, fetched, fetched)).toBeNull();
  });
});

describe("quota", () => {
  it("goes warn near the cap and bad at it", () => {
    expect(quotaHealth(14, 20)).toBe("ok");
    expect(quotaHealth(16, 20)).toBe("warn");
    expect(quotaHealth(20, 20)).toBe("bad");
    expect(quotaHealth(null, 20)).toBe("idle");
    expect(quotaHealth(1, 0)).toBe("idle");
  });

  it("reports an unavailable adapter as bad regardless of quota", () => {
    expect(
      adapterHealth({
        name: "govgreed",
        provides: ["signals"],
        available: false,
        reason: "401",
        quota_used: 0,
        quota_limit: 20,
        quota_tier: null,
        last_call: null,
      }),
    ).toBe("bad");
  });
});

describe("sparkPath", () => {
  it("draws nothing from a single sample", () => {
    expect(sparkPath([1], 96, 20)).toBe("");
    expect(sparkPath([], 96, 20)).toBe("");
  });

  it("spans the box and inverts y so up is up", () => {
    const d = sparkPath([0, 10], 100, 20);
    expect(d).toBe("M0,20 L100,0");
  });

  it("survives a flat series without dividing by zero", () => {
    expect(sparkPath([5, 5, 5], 100, 20)).not.toContain("NaN");
  });
});

describe("sweep grid shape", () => {
  it("counts only the axes that vary", () => {
    expect(gridAxes({ lookback: [10, 20], top_n: [5], mode: ["a", "b"] }).map((a) => a.key)).toEqual([
      "lookback",
      "mode",
    ]);
  });

  it("puts numeric axes in numeric order", () => {
    expect(orderValues([40, 10, 20])).toEqual([10, 20, 40]);
    expect(orderValues(["b", "a"])).toEqual(["b", "a"]);
  });

  it("indexes cells by their two parameter values", () => {
    const cells = indexCells(SWEEP.runs, "lookback", "top_n");
    expect(cells.get("20|5")?.run_id).toBe("sw-20-5");
    expect(cells.get("99|5")).toBeUndefined();
  });
});

describe("journal derivations", () => {
  it("calls an order open only until a fill or reject for its id appears", () => {
    const states = orderStates(EVENTS);
    expect(states.find((s) => s.event.seq === 3)?.state).toBe("filled");
    expect(states.find((s) => s.event.seq === 4)?.state).toBe("open");
  });

  it("marks an order with no journalled id unknown rather than open", () => {
    const states = orderStates([event({ seq: 9, kind: "order", message: "buy", payload: {} })]);
    expect(states[0].state).toBe("unknown");
  });

  it("finds the session run id and the first event of the day", () => {
    expect(liveRunIdOf(EVENTS, "momo_v3")).toBe("momo_v3-live-0001");
    expect(liveRunIdOf(EVENTS, "nope")).toBeNull();
    expect(firstOfToday(EVENTS, NOW)).not.toBeNull();
  });

  it("orders tickers by how often the run traded them", () => {
    expect(tickersByActivity(TRADES)).toEqual(["AAPL", "NVDA"]);
    expect(tickersByActivity([])).toEqual([]);
  });
});

// --- Fleet -------------------------------------------------------------------

describe("Fleet", () => {
  it("shows equity, the day's change and a card per strategy", async () => {
    const el = await mountAt("/");
    const body = text(el);

    expect(body).toContain("$103,412");
    expect(body).toContain("+0.8%");
    expect(body).toContain("momo_v3");
    expect(body).toContain("govgreed_sig");
    expect(body).toContain("paper");

    const link = el.querySelector<HTMLAnchorElement>("a[href*='/live/momo_v3']");
    expect(link).not.toBeNull();
  });

  it("degrades a quiet heartbeat instead of leaving it green", async () => {
    const el = await mountAt("/");
    expect(dotClass(el, "runner")).toContain("bg-moss");
    // Two hours of silence against a 30-second beat.
    expect(dotClass(el, "broker")).toContain("bg-ember");
    expect(dotClass(el, "broker")).not.toContain("bg-moss");
  });

  it("gives the govgreed quota its own health item with the call count", async () => {
    const el = await mountAt("/");
    expect(text(el)).toContain("14/20");
    expect(text(el)).toContain("calls");
    expect(dotClass(el, "govgreed")).toContain("bg-moss");
  });

  it("turns the quota meter amber as the day's budget runs out", async () => {
    vi.mocked(api.liveHealth).mockResolvedValue({
      ...HEALTH,
      adapters: HEALTH.adapters.map((a) =>
        a.name === "govgreed" ? { ...a, quota_used: 19 } : a,
      ),
    });
    const el = await mountAt("/");
    expect(text(el)).toContain("19/20");
    expect(dotClass(el, "govgreed")).toContain("bg-amber");
  });

  it("says what to run when nothing is running", async () => {
    vi.mocked(api.liveStrategies).mockResolvedValue({ strategies: [] });
    const el = await mountAt("/");
    expect(text(el)).toContain("lab paper start");
  });

  it("keeps the journal readable when the health endpoint is down", async () => {
    vi.mocked(api.liveHealth).mockRejectedValue(new Error("connection refused"));
    const el = await mountAt("/");
    const body = text(el);
    expect(body).toContain("connection refused");
    // The tape is still there: the runner being down is a normal state.
    expect(body).toContain("Event journal");
  });
});

// --- RunDetail ---------------------------------------------------------------

describe("RunDetail", () => {
  it("renders provenance in the header and the archival report link", async () => {
    const el = await mountAt(`/runs/${RUN_ID}`);
    const body = text(el);

    expect(body).toContain("momo_v3");
    expect(body).toContain("a1c2f9e");
    expect(body).toContain("7d21ab90");
    expect(body).toContain("v41");

    const report = el.querySelector<HTMLAnchorElement>("a[href*='report.html']");
    expect(report).not.toBeNull();
    expect(decodeURIComponent(report!.getAttribute("href") ?? "")).toBe(
      `/runs/${RUN_ID}/report.html`,
    );
  });

  it("states the survivorship caveat whether or not anything flagged it", async () => {
    const el = await mountAt(`/runs/${RUN_ID}`);
    expect(text(el)).toContain("Survivorship");
  });

  it("raises both loud claims as banners, not footnotes", async () => {
    vi.mocked(api.run).mockResolvedValue(
      run({ metrics: metrics({ optimistic_fills: true, contaminated: true }) }),
    );
    const el = await mountAt(`/runs/${RUN_ID}`);
    const loud = Array.from(el.querySelectorAll(".border-amber\\/40")).map(text).join(" ");
    expect(loud).toContain("Same-bar-close fills");
    expect(loud).toContain("Training-window contamination");
  });

  it("marks a run without out-of-sample metrics as in-sample only", async () => {
    const el = await mountAt(`/runs/${RUN_ID}`);
    expect(text(el)).toContain("in-sample only");
  });

  it("moves the price chart when the time lock selects a trade", async () => {
    const el = await mountAt(`/runs/${RUN_ID}`);
    const chart = () => el.querySelector("[data-chart='price']");

    // Defaults to the most-traded ticker.
    expect(chart()?.getAttribute("data-ticker")).toBe("AAPL");

    await act(async () => {
      timeLock(RUN_ID).select(
        { kind: "trade", id: "t3", ticker: "NVDA" },
        Date.parse("2026-01-02T14:31:00Z"),
        "ledger",
      );
    });
    await settle(2);

    expect(chart()?.getAttribute("data-ticker")).toBe("NVDA");
    expect(el.querySelector("[data-testid='price-chart']")?.getAttribute("data-scope")).toBe(RUN_ID);
  });

  it("follows a tape row to the ticker its decision was about", async () => {
    const el = await mountAt(`/runs/${RUN_ID}`);
    expect(el.querySelector("[data-chart='price']")?.getAttribute("data-ticker")).toBe("AAPL");

    await act(async () => {
      timeLock(RUN_ID).select(
        { kind: "decision", id: "d2" },
        Date.parse("2026-01-06T14:30:00Z"),
        "decision-tape",
      );
    });
    await settle(2);

    expect(el.querySelector("[data-chart='price']")?.getAttribute("data-ticker")).toBe("NVDA");
  });

  it("says the run took no trades rather than drawing an empty chart", async () => {
    vi.mocked(api.trades).mockResolvedValue({ trades: [] });
    const el = await mountAt(`/runs/${RUN_ID}`);
    expect(el.querySelector("[data-chart='price']")).toBeNull();
    expect(text(el)).toContain("no trades");
  });

  it("surfaces a missing run instead of an empty shell", async () => {
    vi.mocked(api.run).mockRejectedValue(new Error("no such run"));
    const el = await mountAt(`/runs/${RUN_ID}`);
    expect(text(el)).toContain("no such run");
  });
});

// --- Sweep -------------------------------------------------------------------

describe("Sweep", () => {
  it("puts the attempt counter first and names the axes", async () => {
    const el = await mountAt("/sweeps/sweep-momo-0012");
    const body = text(el);
    expect(body).toContain("attempts");
    expect(body).toContain("417");
    expect(body).toContain("lookback × top_n");
  });

  it("colours the heatmap by the out-of-sample metric by default, and says so", async () => {
    const el = await mountAt("/sweeps/sweep-momo-0012");
    const map = el.querySelector("[data-chart='sweep-heatmap']");
    expect(map).not.toBeNull();
    expect(map!.getAttribute("data-source")).toBe("out_of_sample");
    expect(map!.getAttribute("data-metric")).toBe("sharpe");
    expect(text(el)).toContain("OOS sharpe");
    expect(text(el)).not.toContain("IN-SAMPLE sharpe");
  });

  it("labels an in-sample surface unmistakably when the source is switched", async () => {
    const el = await mountAt("/sweeps/sweep-momo-0012");
    const select = el.querySelector<HTMLSelectElement>("select[aria-label='colour source']");
    expect(select).not.toBeNull();

    await act(async () => {
      select!.value = "in_sample";
      select!.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle(2);

    expect(el.querySelector("[data-chart='sweep-heatmap']")?.getAttribute("data-source")).toBe(
      "in_sample",
    );
    expect(text(el)).toContain("IN-SAMPLE sharpe");
    expect(text(el)).toContain("fitted the data they were chosen on");
  });

  it("falls back to in-sample loudly when no out-of-sample metric exists", async () => {
    vi.mocked(api.sweep).mockResolvedValue({
      ...SWEEP,
      runs: SWEEP.runs.map((r) => ({ ...r, oos_metrics: {} })),
      best: null,
    });
    const el = await mountAt("/sweeps/sweep-momo-0012");
    expect(el.querySelector("[data-chart='sweep-heatmap']")?.getAttribute("data-source")).toBe(
      "in_sample",
    );
    expect(text(el)).toContain("No out-of-sample metric was recorded");
  });

  it("surfaces a capped grid instead of implying the whole grid ran", async () => {
    vi.mocked(api.sweep).mockResolvedValue({
      ...SWEEP,
      truncated: { applied: true, requested: 1024, ran: 6 },
    });
    const el = await mountAt("/sweeps/sweep-momo-0012");
    const body = text(el);
    expect(body).toContain("This grid was capped");
    expect(body).toContain("1,024");
    expect(body).toContain("capped 6/1,024");
  });

  it("warns when the winner was ranked in-sample", async () => {
    vi.mocked(api.sweep).mockResolvedValue({ ...SWEEP, ranked_on: "in_sample" });
    const el = await mountAt("/sweeps/sweep-momo-0012");
    expect(text(el)).toContain("Ranked in-sample");
  });

  it("falls back to a table for a one-axis grid", async () => {
    vi.mocked(api.sweep).mockResolvedValue({
      ...SWEEP,
      grid: { lookback: [10, 20, 40], top_n: [5] },
    });
    const el = await mountAt("/sweeps/sweep-momo-0012");
    expect(el.querySelector("[data-chart='sweep-heatmap']")).toBeNull();
    expect(text(el)).toContain("not a surface");
    expect(el.querySelectorAll("tbody tr").length).toBeGreaterThan(0);
  });

  it("shows walk-forward windows with both sides of each split", async () => {
    const el = await mountAt("/sweeps/sweep-momo-0012");
    const body = text(el);
    expect(body).toContain("walk-forward windows");
    expect(el.querySelector("[data-chart='walk-forward']")).not.toBeNull();
    // Both sides of every split, and the decay between them. Asserted on the
    // metrics rather than the dates: `day()` renders in the local zone and the
    // fixture's UTC midnights land on the previous day west of Greenwich.
    expect(body).toContain("2.10");
    expect(body).toContain("0.60");
    expect(body).toContain("1.80");
    expect(body).toContain("-0.20");
  });
});

// --- StrategyLive ------------------------------------------------------------

describe("StrategyLive", () => {
  it("lists positions with their P&L and age", async () => {
    const el = await mountAt("/live/momo_v3");
    const body = text(el);
    expect(body).toContain("AAPL");
    expect(body).toContain("NVDA");
    expect(body).toContain("+$271.20");
    expect(body).toContain("1h");
  });

  it("offers exactly the three risk-reducing controls and nothing that adds exposure", async () => {
    const el = await mountAt("/live/momo_v3");
    const labels = Array.from(el.querySelectorAll("button")).map((b) => text(b).toLowerCase());

    expect(labels.some((l) => l.includes("pause"))).toBe(true);
    expect(labels.some((l) => l.includes("cancel orders"))).toBe(true);
    expect(labels.some((l) => l.includes("kill"))).toBe(true);

    // The asymmetry is the feature: no button may start, resume or loosen.
    expect(labels.filter((l) => /\bstart\b|resume|release|raise|increase/.test(l))).toEqual([]);
    expect(text(el)).toContain("No start, resume or raise-limit control here");
  });

  it("confirms before pausing and calls only the pause route", async () => {
    const el = await mountAt("/live/momo_v3");
    const button = Array.from(el.querySelectorAll("button")).find((b) =>
      text(b).toLowerCase().includes("pause"),
    );
    await act(async () => {
      button!.click();
    });
    await settle(2);

    expect(window.confirm).toHaveBeenCalled();
    expect(vi.mocked(api.pause)).toHaveBeenCalledWith("momo_v3");
    expect(vi.mocked(api.cancelOrders)).not.toHaveBeenCalled();
    expect(vi.mocked(api.kill)).not.toHaveBeenCalled();
  });

  it("does nothing when the confirm is declined", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false));
    const el = await mountAt("/live/momo_v3");
    const button = Array.from(el.querySelectorAll("button")).find((b) =>
      text(b).toLowerCase().includes("cancel orders"),
    );
    await act(async () => {
      button!.click();
    });
    await settle(2);
    expect(vi.mocked(api.cancelOrders)).not.toHaveBeenCalled();
  });

  it("replays today by taking the time lock out of pinned mode", async () => {
    const el = await mountAt("/live/momo_v3");

    // Live: the tape pins the playhead to now.
    expect(timeLock("live").getState().pinned).toBe(true);

    const replay = Array.from(el.querySelectorAll("button")).find((b) =>
      text(b).toLowerCase().includes("replay today"),
    );
    expect(replay).not.toBeUndefined();

    await act(async () => {
      replay!.click();
    });
    await settle(2);

    const state = timeLock("live").getState();
    expect(state.pinned).toBe(false);
    expect(state.t).not.toBeNull();
    expect(text(el)).toContain("replaying");

    // And back again, through the same one piece of state.
    const back = Array.from(el.querySelectorAll("button")).find((b) =>
      text(b).toLowerCase().includes("back to live"),
    );
    await act(async () => {
      back!.click();
    });
    await settle(2);
    expect(timeLock("live").getState().pinned).toBe(true);
  });

  it("derives resting orders and recent gate events from the journal", async () => {
    const el = await mountAt("/live/momo_v3");
    const body = text(el);
    expect(body).toContain("1 resting of 2 seen");
    expect(body).toContain("sector cap 2/2");
    expect(body).toContain("not from the broker's book");
  });

  it("stays useful as forensics when the runner is down", async () => {
    vi.mocked(api.liveStrategies).mockRejectedValue(new Error("runner socket closed"));
    const el = await mountAt("/live/momo_v3");
    const body = text(el);

    expect(body).toContain("The runner is not reporting momo_v3");
    expect(body).toContain("position state is unknown, not flat");
    // The journal is still there, and so is the kill switch.
    expect(body).toContain("Event journal");
    const labels = Array.from(el.querySelectorAll("button")).map((b) => text(b).toLowerCase());
    expect(labels.some((l) => l.includes("kill"))).toBe(true);
  });
});
