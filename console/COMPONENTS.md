# Console component contracts

Binding prop signatures for the shared components, so the screens can be built
in parallel against them. Anything not listed is the implementer's choice.

Already written and **frozen** — read, do not edit:

- `src/lib/types.ts` — the typed JSON contract (mirrors `lab/api/models.py`)
- `src/lib/api.ts` — `api.*` read calls, the three risk-reducing mutations, and
  `subscribeEvents()` for the WebSocket tail
- `src/lib/timelock.ts` — `useTimeLock(scope)`, `toMs`, `toChartTime`,
  `nearestIndex`
- `src/lib/format.ts` — every formatter. **Never** call `toFixed` or
  `Intl.NumberFormat` in a screen; use these, so numbers do not change width
  when they tick.
- `src/components/ui.tsx` — `Panel Stat StatusDot Badge LoudWarning Empty
  ErrorNote Loading Table Th Td Tr Button cx`
- `src/components/Shell.tsx`, `src/router.tsx`, `src/main.tsx`
- `src/styles/tokens.css` — the design-token layer

## House rules

- TypeScript strict. No `any`. No `@ts-ignore`.
- Tailwind utilities over the tokens: `bg-gunmetal`, `text-ash`, `border-slate`,
  `text-moss`, `text-ember`, `text-amber`, `text-teal`, `bg-teal-wash`, …
  **Never** a raw hex or an arbitrary color. If a hue appears it must encode P&L
  direction, warning state, or selection — never decoration.
- Every number, ticker and timestamp gets `className="num"` or `"mono"`
  (tabular figures). Unit and column labels get `"unit"`.
- Data fetching is TanStack Query keyed `["runs", id, …]` / `["live", …]`.
  Backtest artifacts are immutable, so no polling; live views set an explicit
  `refetchInterval`.
- Every panel handles all four states: loading (`<Loading/>`), error
  (`<ErrorNote/>`), empty (`<Empty cmd="lab …">` — say what to RUN to fill it),
  and populated.
- Wide content scrolls in its own `.scroll-x`; the page never scrolls sideways.
- Motion: a state *change* pulses once via `className="pulse"`. Values updating
  in place do not animate. Nothing idles.
- Keyboard focus stays visible; interactive rows are reachable by keyboard.
- Mobile is not a target, but the fleet overview must *survive* a phone screen
  for a status glance.

## Shared components (owner: the `shared-components` agent)

### `src/components/DecisionTape.tsx`

The product's signature object. One row per decision point, collapsed to a terse
line, expandable to the full input tape, time-locked to whatever is above it.

```ts
export interface DecisionTapeProps {
  scope: string;                    // run_id, or "live"
  decisions: DecisionRow[];
  loading?: boolean;
  error?: unknown;
  /** Live mode pins the playhead to the newest row and auto-scrolls. */
  pinned?: boolean;
  /** Render the agent prompt/response block inline (Pattern-B strategies). */
  showAgent?: boolean;
  emptyCmd?: string;
}
export function DecisionTape(props: DecisionTapeProps): JSX.Element
```

Collapsed row: time · `decision.summary` — already computed server-side, e.g.
`saw 3 signals · intent NVDA +4.0% · gate: clipped to 3.1% (position_cap) · 1 order(s)`.
Prefix with a status glyph coloured by the strongest gate action in the row
(pass → ash, clipped → amber, blocked → amber, any fill → moss).

Expanded row shows, in this order:
1. **Input tape** — indicator values verbatim; signals as a small table with
   BOTH `event_time` and `knowledge_time` and the lag between them called out
   (that gap is the reason the component exists); history slices as their
   summarised tails; prices.
2. **Intents** — ticker, target %, tag, reason.
3. **Per-intent gate verdicts** — action, the rule that fired, requested vs
   approved %. A clip shows both numbers.
4. **Orders and fills** — ids, side, qty, price.
5. **Agent block** when `showAgent` and `decision.agent` is set — the prompt, the
   structured reply, rationale, tokens and cost.

Behaviour: clicking a row calls `select({kind:"decision", id}, toMs(at))` on the
time lock; when the lock's `t` moves from elsewhere, the nearest row highlights
and scrolls into view (`nearestIndex` helps). Virtualize or cap at ~500 rendered
rows — a long backtest has thousands and the tape must stay responsive.

### `src/components/charts/EquityChart.tsx`

```ts
export interface EquityChartProps {
  scope: string;
  data: EquitySeries;
  height?: number;              // default 220
  showDrawdown?: boolean;       // default true: stacked drawdown pane
  benchmarkLabel?: string;
}
export function EquityChart(props: EquityChartProps): JSX.Element
```

Equity line plus a stacked drawdown area sharing the x axis. **Shade the
out-of-sample spans** from `data.is_oos` — this is the single most important
thing the chart does, so OOS performance is visually unmissable; use
`--color-oos-wash`. Overlay `data.benchmark` as a dim ash line when present.
Crosshair movement calls `hover(t)`; a click calls `seek(t)`. When the lock's
`t` changes from elsewhere, move the crosshair without re-emitting (check
`source` to avoid a feedback loop). Use `lightweight-charts`; call
`toChartTime()` for every timestamp.

### `src/components/charts/PriceChart.tsx`

```ts
export interface PriceChartProps {
  scope: string;
  runId: string;
  ticker: string;
  trades?: TradeRow[];          // entry/exit markers
  height?: number;              // default 260
}
export function PriceChart(props: PriceChartProps): JSX.Element
```

Candles for one ticker with ▲ entry / ▼ exit markers from `trades`, marker
colour by trade P&L. Fetches via `api.bars(runId, ticker)`. Same time-lock
wiring as EquityChart.

### `src/components/charts/EChart.tsx`

```ts
export interface EChartProps {
  option: unknown;              // echarts EChartsOption
  height?: number;
  onEvent?: Record<string, (params: unknown) => void>;
  className?: string;
}
export function EChart(props: EChartProps): JSX.Element
```

Thin ECharts wrapper: init on mount, `setOption` on change, resize with a
`ResizeObserver`, dispose on unmount. Import from `echarts/core` with only the
needed charts/components registered so the bundle does not carry all of ECharts.
Export a `darkTheme` object matching the token palette (transparent background,
`#2B323C` split lines, `#98A2AE` axis labels, `IBM Plex Mono` for axis text) and
apply it by default.

### `src/components/TradeLedger.tsx`

```ts
export interface TradeLedgerProps {
  scope: string;
  trades: TradeRow[];
  loading?: boolean;
  error?: unknown;
  onSelectTicker?: (ticker: string) => void;
}
export function TradeLedger(props: TradeLedgerProps): JSX.Element
```

Sortable table (click a header to sort; show the ▲/▼ via `Th`'s `sorted` prop).
Columns: ticker, side, qty, entry time/price, exit time/price, P&L, P&L %, bars
held, exit reason. P&L coloured by `toneClass`. Clicking a row calls
`select({kind:"trade", id, ticker}, entryMs)` so the chart and tape jump to that
trade. Truncate past ~1000 rows with a visible note — never silently.

### `src/components/EventTape.tsx` + `src/hooks/useLiveEvents.ts`

```ts
export function useLiveEvents(opts?: { limit?: number; kinds?: EventKind[] }): {
  events: JournalEvent[];       // newest first, ring-buffered
  status: "open" | "closed" | "error";
  latestSeq: number;
}
export interface EventTapeProps {
  events: JournalEvent[];
  status?: "open" | "closed" | "error";
  height?: number;
  onSelectStrategy?: (s: string) => void;
}
export function EventTape(props: EventTapeProps): JSX.Element
```

The hook seeds from `api.liveEvents({since:0})`, then tails via
`subscribeEvents`, keeping a **ring buffer** so the tape stays O(visible) rather
than O(history), and resumes from the last seq on reconnect. New rows get
`className="pulse"` once. Render each event as
`HH:MM:SS  source  KIND  message`, kind coloured by meaning (fill → moss,
gate_block/breaker → amber, error/reject → ember, heartbeat → ash-dim). When the
socket is closed, say so in the header — the runner being down is a normal state
and the console must keep working as forensics.

## Screens

Each screen is a default export from `src/screens/<Name>.tsx`. Stubs exist now
and are meant to be replaced wholesale.

| File | Owner | Design doc |
|---|---|---|
| `Fleet.tsx` | fleet-live | console §4 "Fleet overview" |
| `StrategyLive.tsx` | fleet-live | console §4 "Strategy detail (live)" |
| `RunBrowser.tsx` | runs | console §4 "Run browser" |
| `RunDetail.tsx` | runs | console §4 "Run detail (backtest)" |
| `Compare.tsx` | compare-sweep | console §4 "Compare" |
| `Sweep.tsx` | compare-sweep | console §4 "Sweep" |
| `AgentActivity.tsx` | agent-views | console §4 "Agent activity" |
