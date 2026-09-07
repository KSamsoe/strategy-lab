/**
 * Route tree.
 *
 * Code-based rather than file-based: seven screens do not need a codegen step,
 * and having the whole map in one file makes the app's shape legible at a
 * glance. Screens are lazy so the fleet overview -- the screen that is open all
 * day -- does not pay for the charting libraries the run detail needs.
 */

import { lazy } from "react";
import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
} from "@tanstack/react-router";

import { Shell } from "./components/Shell";

const Fleet = lazy(() => import("./screens/Fleet"));
const RunBrowser = lazy(() => import("./screens/RunBrowser"));
const RunDetail = lazy(() => import("./screens/RunDetail"));
const Compare = lazy(() => import("./screens/Compare"));
const Sweep = lazy(() => import("./screens/Sweep"));
const StrategyLive = lazy(() => import("./screens/StrategyLive"));
const AgentActivity = lazy(() => import("./screens/AgentActivity"));

const rootRoute = createRootRoute({
  component: () => (
    <Shell>
      <Outlet />
    </Shell>
  ),
});

const fleetRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Fleet,
});

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

const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: RunDetail,
});

const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: Compare,
  validateSearch: (s: Record<string, unknown>) => ({
    ids: typeof s.ids === "string" ? s.ids : "",
  }),
});

const sweepRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sweeps/$sweepId",
  component: Sweep,
});

const liveRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/live/$strategy",
  component: StrategyLive,
});

const agentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/agent",
  component: AgentActivity,
});

const routeTree = rootRoute.addChildren([
  fleetRoute,
  runsRoute,
  runDetailRoute,
  compareRoute,
  sweepRoute,
  liveRoute,
  agentRoute,
]);

export const router = createRouter({ routeTree, defaultPreload: "intent" });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
