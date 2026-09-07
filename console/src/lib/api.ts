/**
 * The single place the console talks to the lab.
 *
 * Every call is a GET except the three control actions, and those can only ever
 * make the system *safer*: pause, cancel open orders, kill. There is no client
 * function to start a strategy, raise a limit, or resume after a breaker,
 * because the API has no route for them -- those stay CLI actions behind a
 * manual checklist. If you find yourself adding a mutation here, check
 * `docs/strategy-lab-console-design-v1.md` §3 first: the asymmetry is a feature.
 */

import type {
  DecisionRow,
  EquitySeries,
  JournalEvent,
  LineageStep,
  LiveHealth,
  LiveStrategy,
  Metrics,
  ResearchDetail,
  ResearchLaunchOptions,
  ResearchStartRequest,
  ResearchStartResult,
  ResearchSummary,
  RunRecord,
  StrategyInfo,
  SweepResult,
  TradeRow,
} from "./types";

/** The half of a decision the API nests under `detail` to keep list pages light. */
type DecisionDetailPart = Pick<
  DecisionRow,
  "inputs" | "intents" | "verdicts" | "order_ids" | "portfolio" | "logs" | "agent" | "duration_ms"
>;

/** Same-origin when served by `lab ui`; the Vite proxy handles dev. */
const BASE = import.meta.env.VITE_LAB_API ?? "";

/** Optional bearer token for the "check from another machine on my LAN" case. */
let token: string | null =
  typeof localStorage !== "undefined" ? localStorage.getItem("lab_token") : null;

export function setToken(next: string | null): void {
  token = next;
  if (typeof localStorage === "undefined") return;
  if (next) localStorage.setItem("lab_token", next);
  else localStorage.removeItem("lab_token");
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly url: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * FastAPI's `detail` is a plain string for a raised HTTPException but an ARRAY
 * of validation objects for a 422. Coercing that array straight into an Error
 * message yields the useless "[object Object]", which is exactly the moment you
 * most need to read what the server rejected.
 */
function describeDetail(detail: unknown): string | null {
  if (detail === null || detail === undefined) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      if (typeof d === "string") return d;
      const item = d as { loc?: unknown[]; msg?: string; type?: string };
      const where = Array.isArray(item.loc) ? item.loc.join(".") : "";
      const what = item.msg ?? item.type ?? JSON.stringify(d);
      return where ? `${where}: ${what}` : String(what);
    });
    return parts.join("; ") || null;
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function qs(params: Record<string, unknown> = {}): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    usp.set(k, Array.isArray(v) ? v.join(",") : String(v));
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init?.body) headers.set("Content-Type", "application/json");

  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      detail = describeDetail(body.detail) ?? detail;
    } catch {
      /* a non-JSON error body is still an error */
    }
    throw new ApiError(res.status, url, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// --- reads -------------------------------------------------------------------

export const api = {
  health: () => req<{ ok: boolean; version: string }>("/api/health"),

  runs: async (p: {
    strategy?: string;
    kind?: string;
    origin?: string;
    sweep_id?: string;
    limit?: number;
    offset?: number;
  } = {}) => {
    const body = await req<{ count: number; runs: RunRecord[] }>(`/api/runs${qs(p)}`);
    return { runs: body.runs, total: body.count };
  },

  run: (id: string) => req<RunRecord>(`/api/runs/${encodeURIComponent(id)}`),

  equity: (id: string) =>
    req<EquitySeries>(`/api/runs/${encodeURIComponent(id)}/equity`),

  trades: (id: string) =>
    req<{ trades: TradeRow[] }>(`/api/runs/${encodeURIComponent(id)}/trades`),

  /**
   * The API returns each decision as a summary row with the heavy part nested
   * under `detail`, so the list can be paged without shipping every input tape.
   * The tape wants one flat object, so unwrap it here -- this client is the
   * boundary where the wire shape becomes the UI shape.
   */
  decisions: async (
    id: string,
    p: { start?: string; end?: string; ticker?: string; limit?: number; offset?: number } = {},
  ) => {
    const body = await req<{
      total: number;
      rows: (Omit<DecisionRow, keyof DecisionDetailPart> & {
        detail?: DecisionDetailPart;
      })[];
    }>(`/api/runs/${encodeURIComponent(id)}/decisions${qs(p)}`);
    const decisions = (body.rows ?? []).map((row) => {
      const { detail, ...head } = row;
      return { ...head, ...(detail ?? {}) } as DecisionRow;
    });
    return { decisions, total: body.total };
  },

  bars: (id: string, ticker: string, tf = "1d") =>
    req<{
      ticker: string;
      t: string[];
      open: number[];
      high: number[];
      low: number[];
      close: number[];
      volume: number[];
    }>(`/api/runs/${encodeURIComponent(id)}/bars${qs({ ticker, tf })}`),

  /**
   * The API answers with a column-oriented table (one row per run, provenance
   * and metric columns flattened together). The compare grid wants the metrics
   * keyed by run so it can highlight the best cell per column, so pivot here.
   */
  compare: async (ids: string[]) => {
    const body = await req<{
      ids: string[];
      columns: string[];
      rows: Record<string, unknown>[];
    }>(`/api/runs/compare${qs({ ids })}`);
    const metrics: Record<string, Metrics> = {};
    const runs: RunRecord[] = [];
    for (const row of body.rows ?? []) {
      const runId = String(row.run_id ?? "");
      if (!runId) continue;
      metrics[runId] = row as Metrics;
      // The compare table flattens provenance and metrics into one row, so the
      // run record is reconstructed rather than re-fetched -- params and config
      // are not in the table and the grid does not read them.
      runs.push({
        run_id: runId,
        strategy: String(row.strategy ?? ""),
        kind: (row.kind ?? "backtest") as RunRecord["kind"],
        status: (row.status ?? "ok") as RunRecord["status"],
        created_at: String(row.created_at ?? ""),
        finished_at: (row.finished_at as string | null) ?? null,
        git_commit: (row.git_commit as string | null) ?? null,
        config_hash: String(row.config_hash ?? ""),
        data_version: String(row.data_version ?? ""),
        params: {},
        config: {},
        metrics: row as Metrics,
        start: (row.start as string | null) ?? null,
        end: (row.end as string | null) ?? null,
        origin: (row.origin ?? "human") as RunRecord["origin"],
        parent_run_id: (row.parent_run_id as string | null) ?? null,
        sweep_id: (row.sweep_id as string | null) ?? null,
        notes: String(row.notes ?? ""),
        attempt: Number(row.attempt ?? 0),
        error: String(row.error ?? ""),
      });
    }
    return { ids: body.ids, columns: body.columns, rows: body.rows, metrics, runs };
  },

  sweep: (sweepId: string) =>
    req<SweepResult>(`/api/sweeps/${encodeURIComponent(sweepId)}`),

  strategies: () => req<{ strategies: StrategyInfo[] }>("/api/strategies"),

  liveStrategies: () => req<{ strategies: LiveStrategy[] }>("/api/live/strategies"),

  liveEvents: (p: { since?: number; limit?: number } = {}) =>
    req<{ events: JournalEvent[]; latest_seq: number }>(`/api/live/events${qs(p)}`),

  liveHealth: () => req<LiveHealth>("/api/live/health"),

  research: () => req<{ count: number; sessions: ResearchSummary[] }>("/api/agent/research"),

  researchSession: (sessionId: string) =>
    req<ResearchDetail>(`/api/agent/research/${encodeURIComponent(sessionId)}`),

  lineage: (strategy: string) =>
    req<{ strategy: string; steps: LineageStep[] }>(
      `/api/agent/lineage/${encodeURIComponent(strategy)}`,
    ),

  agentCalls: (runId: string) =>
    req<{ calls: unknown[]; total_cost_usd: number }>(
      `/api/agent/calls${qs({ run_id: runId })}`,
    ),

  // --- the only three mutations, all strictly risk-reducing ------------------

  /**
   * Research launch. The one mutation that is not purely risk-reducing, which is
   * why the server keeps it behind an explicit opt-in: a session cannot touch a
   * broker, but it spends budget and can execute model-authored code. `options`
   * reports whether it is allowed and why not.
   */
  researchOptions: () => req<ResearchLaunchOptions>("/api/control/research/options"),

  startResearch: (body: ResearchStartRequest) =>
    req<ResearchStartResult>("/api/control/research/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Always available, opt-in or not: it only ever reduces what is running. */
  stopResearch: (sessionId: string) =>
    req<{ ok: boolean; session_id: string; message: string }>(
      "/api/control/research/stop",
      { method: "POST", body: JSON.stringify({ session_id: sessionId }) },
    ),

  /** Stop a strategy from making new decisions. Journaled. */
  pause: (strategy: string) =>
    req<{ ok: boolean; strategy: string }>("/api/control/pause", {
      method: "POST",
      body: JSON.stringify({ strategy }),
    }),

  /** Cancel resting orders. Cannot open exposure, only remove intent. */
  cancelOrders: (strategy: string) =>
    req<{ ok: boolean; canceled: number }>("/api/control/cancel_orders", {
      method: "POST",
      body: JSON.stringify({ strategy }),
    }),

  /** The panic stop. Engages the sentinel file; releasing it is a CLI action. */
  kill: (reason: string) =>
    req<{ ok: boolean; engaged: boolean; reason: string }>("/api/control/kill", {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),
};

// --- live tail ---------------------------------------------------------------

/**
 * Tail the event journal over a WebSocket, resuming from the last seq seen.
 *
 * The seq cursor is what makes a reconnect lossless: the server replays
 * everything after it rather than dropping the gap. Callers keep a ring buffer
 * so the tape stays O(visible) rather than O(history).
 */
export function subscribeEvents(
  onEvent: (ev: JournalEvent) => void,
  opts: { since?: number; onStatus?: (s: "open" | "closed" | "error") => void } = {},
): () => void {
  let since = opts.since ?? 0;
  let socket: WebSocket | null = null;
  let retry = 0;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const connect = () => {
    if (stopped) return;
    const origin = BASE || window.location.origin;
    const url = new URL("/api/ws/events", origin);
    url.protocol = url.protocol.replace("http", "ws");
    url.searchParams.set("since", String(since));
    if (token) url.searchParams.set("token", token);

    socket = new WebSocket(url.toString());

    socket.onopen = () => {
      retry = 0;
      opts.onStatus?.("open");
    };
    socket.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data as string) as JournalEvent;
        if (typeof ev.seq === "number") since = Math.max(since, ev.seq);
        onEvent(ev);
      } catch {
        /* a malformed frame must not kill the tail */
      }
    };
    socket.onerror = () => opts.onStatus?.("error");
    socket.onclose = () => {
      opts.onStatus?.("closed");
      if (stopped) return;
      // Backoff, capped: the runner being down is a normal state for this app,
      // not an emergency, and the UI keeps working as forensics meanwhile.
      const delay = Math.min(1000 * 2 ** retry++, 15000);
      timer = setTimeout(connect, delay);
    };
  };

  connect();

  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    socket?.close();
  };
}
