/**
 * Render tests for the two run screens.
 *
 * There is no @testing-library in the tree, so these drive React's own root API
 * under `act` and assert against the DOM. The router is real but in-memory: the
 * compare hand-off is a URL, and asserting on a hand-built string instead of the
 * href TanStack actually emits would test nothing.
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
    runs: vi.fn(),
    run: vi.fn(),
    equity: vi.fn(),
    trades: vi.fn(),
    decisions: vi.fn(),
    strategies: vi.fn(),
    bars: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
  setToken: vi.fn(),
  subscribeEvents: vi.fn(() => () => undefined),
}));

import { api } from "../lib/api";
import type { Metrics, RunRecord } from "../lib/types";
import RunBrowser from "../screens/RunBrowser";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// --- fixtures ----------------------------------------------------------------

const metrics = (m: Metrics): Metrics => ({
  cagr: 0.142,
  sharpe: 1.31,
  max_drawdown: -0.184,
  trades: 312,
  ...m,
});

const run = (r: Partial<RunRecord> & { run_id: string }): RunRecord => ({
  strategy: "momo_v3",
  kind: "backtest",
  status: "ok",
  created_at: "2026-02-01T10:00:00Z",
  finished_at: "2026-02-01T10:04:00Z",
  git_commit: "a1c2f9e4d1",
  config_hash: "7d21ab90",
  data_version: "v41",
  params: {},
  config: {},
  metrics: metrics({}),
  start: "2018-01-02T00:00:00Z",
  end: "2026-01-30T00:00:00Z",
  origin: "human",
  parent_run_id: null,
  sweep_id: null,
  notes: "",
  attempt: 1,
  error: "",
  ...r,
});

const REGISTRY: RunRecord[] = [
  run({ run_id: "rb-alpha-0001", strategy: "momo_v3", origin: "human", attempt: 3 }),
  run({
    run_id: "rb-beta-0002",
    strategy: "govgreed_sig",
    origin: "agent-loop",
    attempt: 417,
    metrics: metrics({ cagr: 0.41, sharpe: 2.9, oos_sharpe: 0.2 }),
  }),
  run({ run_id: "rb-gamma-0003", strategy: "momo_v3", origin: "human", attempt: 2, kind: "sweep" }),
];

// --- harness -----------------------------------------------------------------

function makeRouter(path: string) {
  const rootRoute = createRootRoute({ component: () => <Outlet /> });
  const runsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/runs",
    component: RunBrowser,
    validateSearch: (s: Record<string, unknown>) => ({
      strategy: (s.strategy as string) || undefined,
      kind: (s.kind as string) || undefined,
      origin: (s.origin as string) || undefined,
    }),
  });
  const detailRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/runs/$runId",
    component: () => <div>detail</div>,
  });
  const compareRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/compare",
    component: () => <div>compare</div>,
    validateSearch: (s: Record<string, unknown>) => ({
      ids: typeof s.ids === "string" ? s.ids : "",
    }),
  });
  return createRouter({
    routeTree: rootRoute.addChildren([runsRoute, detailRoute, compareRoute]),
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

/** Let queries resolve and React flush; a few turns covers the fetch chain. */
async function settle(turns = 6): Promise<void> {
  for (let i = 0; i < turns; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

const text = (el: Element | null): string => (el?.textContent ?? "").replace(/\s+/g, " ").trim();

beforeEach(() => {
  vi.mocked(api.runs).mockResolvedValue({ runs: REGISTRY, total: REGISTRY.length });
  vi.mocked(api.strategies).mockResolvedValue({ strategies: [] });
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.clearAllMocks();
});

// --- RunBrowser ---------------------------------------------------------------

describe("RunBrowser", () => {
  it("renders a row per run with origin tags and the attempt counter", async () => {
    const el = await mountAt("/runs");
    const body = text(el);

    const rows = el.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(3);

    expect(body).toContain("momo_v3");
    expect(body).toContain("govgreed_sig");

    // Origin is a first-class tag, both values present.
    expect(body).toContain("human");
    expect(body).toContain("agent-loop");

    // The attempt counter, and the warning treatment past the threshold.
    expect(body).toContain("417");
    const loud = el.querySelector("[title*='417 runs recorded']");
    expect(loud).not.toBeNull();
    expect(loud?.className).toContain("text-amber");

    // A three-attempt family stays neutral.
    const quiet = el.querySelector("[title*='3 runs recorded']");
    expect(quiet?.className).toContain("text-ash");
    expect(quiet?.className).not.toContain("amber");
  });

  it("passes the route search params through to the registry query", async () => {
    await mountAt("/runs?strategy=govgreed_sig&kind=backtest&origin=agent-loop");
    expect(vi.mocked(api.runs)).toHaveBeenCalledWith(
      expect.objectContaining({
        strategy: "govgreed_sig",
        kind: "backtest",
        origin: "agent-loop",
      }),
    );
  });

  it("builds a compare link from the multi-selected rows", async () => {
    const el = await mountAt("/runs");

    const boxes = Array.from(
      el.querySelectorAll<HTMLInputElement>("tbody input[type=checkbox]"),
    );
    expect(boxes).toHaveLength(3);

    // One selection is not a comparison.
    await act(async () => {
      boxes[0].click();
    });
    expect(el.querySelector("a[href*='/compare']")).toBeNull();

    await act(async () => {
      boxes[1].click();
    });
    const link = el.querySelector<HTMLAnchorElement>("a[href*='/compare']");
    expect(link).not.toBeNull();
    expect(decodeURIComponent(link!.getAttribute("href") ?? "")).toBe(
      "/compare?ids=rb-alpha-0001,rb-beta-0002",
    );
  });

  it("shows an empty state that names the command to run", async () => {
    vi.mocked(api.runs).mockResolvedValue({ runs: [], total: 0 });
    const el = await mountAt("/runs");
    expect(text(el)).toContain("lab backtest strategies/momo.py");
  });

  it("surfaces a registry read failure instead of an empty table", async () => {
    vi.mocked(api.runs).mockRejectedValue(new Error("registry.sqlite is locked"));
    const el = await mountAt("/runs");
    expect(text(el)).toContain("registry.sqlite is locked");
  });
});
