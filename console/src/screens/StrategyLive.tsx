/**
 * Strategy detail, live.
 *
 * The same components as the run detail screen, pointed at the "live" time
 * lock instead of a run id — which is the whole architectural bet: backtests
 * get a live-quality inspector for free and live trading gets backtest-quality
 * forensics for free, because neither renderer can tell replay from reality.
 *
 * "Replay today" is where that bet pays off, so it is implemented as *one line
 * of state*: the time lock leaves `pinned` mode. Nothing else changes. The tape
 * stops following now and starts obeying the playhead; the price chart's
 * crosshair follows the same instant it always did. A second read-only replay
 * UI would have been a second thing to keep correct, and the first time the two
 * disagreed the operator would stop trusting both.
 *
 * The control surface is three buttons and they can only reduce exposure.
 * There is no start, no resume, no raise-limit — not because they were missed,
 * but because `lib/api.ts` has no route for them and the console is not allowed
 * to be the thing that increases risk. The screen says so out loud rather than
 * looking incomplete.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api";
import { DecisionTape } from "../components/DecisionTape";
import { EventTape } from "../components/EventTape";
import { PriceChart } from "../components/charts/PriceChart";
import { useLiveEvents } from "../hooks/useLiveEvents";
import { useNow } from "./Fleet";
import type { JournalEvent, LiveStrategy, PositionRow } from "../lib/types";
import { toMs, useTimeLock } from "../lib/timelock";
import {
  MISSING,
  age,
  clock,
  count,
  day,
  delta,
  duration,
  money,
  num,
  shortId,
  signedMoney,
  stamp,
  toneClass,
} from "../lib/format";
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  Loading,
  Panel,
  StatusDot,
  Table,
  Td,
  Th,
  Tr,
  cx,
  type Health,
} from "../components/ui";

/** Live data is polled on an explicit interval; nothing here is immutable. */
const LIVE_MS = 5000;

const SCOPE = "live";

const STATUS_HEALTH: Record<LiveStrategy["status"], Health> = {
  running: "ok",
  paused: "warn",
  blocked: "bad",
  stopped: "idle",
};

// --- journal derivations ------------------------------------------------------

/** The order id an event is about, whatever the runner chose to call it. */
export function orderIdOf(ev: JournalEvent): string | null {
  for (const k of ["order_id", "id", "client_order_id"]) {
    const v = ev.payload?.[k];
    if (typeof v === "string" && v) return v;
  }
  return null;
}

export type OrderState = "open" | "filled" | "rejected" | "unknown";

/**
 * Which orders are still resting, derived from the journal tail.
 *
 * The console reads journals and nothing else, so "open" here means "no fill or
 * reject for this id appears in the window we are holding" — which is not the
 * same as the broker's book. An order whose id the runner never journalled is
 * reported as `unknown` rather than quietly counted as open: a fabricated
 * resting order is a worse error than an unclassified one.
 */
export function orderStates(
  events: readonly JournalEvent[],
): { event: JournalEvent; state: OrderState }[] {
  const resolved = new Map<string, OrderState>();
  for (const e of events) {
    if (e.kind !== "fill" && e.kind !== "reject") continue;
    const id = orderIdOf(e);
    if (id) resolved.set(id, e.kind === "fill" ? "filled" : "rejected");
  }
  return events
    .filter((e) => e.kind === "order")
    .map((event) => {
      const id = orderIdOf(event);
      if (!id) return { event, state: "unknown" as OrderState };
      return { event, state: resolved.get(id) ?? ("open" as OrderState) };
    });
}

/** Earliest event we hold from the current session's calendar day. */
export function firstOfToday(events: readonly JournalEvent[], nowMs: number): number | null {
  const today = day(nowMs);
  let earliest: number | null = null;
  for (const e of events) {
    if (day(e.at) !== today) continue;
    const t = toMs(e.at);
    if (earliest === null || t < earliest) earliest = t;
  }
  return earliest;
}

/** The run id the live session is journalling under, per the event tail. */
export function liveRunIdOf(events: readonly JournalEvent[], strategy: string): string | null {
  for (const e of events) {
    if (e.strategy === strategy && e.run_id) return e.run_id;
  }
  return null;
}

// --- screen -------------------------------------------------------------------

export default function StrategyLive() {
  const { strategy } = useParams({ from: "/live/$strategy" });
  return <StrategyLiveView strategy={strategy} />;
}

export function StrategyLiveView({ strategy }: { strategy: string }) {
  const qc = useQueryClient();
  const lock = useTimeLock(SCOPE);
  const { events, status } = useLiveEvents({ limit: 500 });

  const mine = useMemo(
    () => events.filter((e) => e.strategy === null || e.strategy === strategy),
    [events, strategy],
  );
  const runId = useMemo(() => liveRunIdOf(events, strategy), [events, strategy]);

  const strategiesQ = useQuery({
    queryKey: ["live", "strategies"],
    queryFn: api.liveStrategies,
    refetchInterval: LIVE_MS,
    retry: false,
  });

  const decisionsQ = useQuery({
    queryKey: ["live", "decisions", runId],
    queryFn: () => api.decisions(runId as string, { limit: 500 }),
    enabled: runId !== null,
    refetchInterval: LIVE_MS,
    retry: false,
  });

  // Provenance comes off the registry record for the session's run: the live
  // endpoint reports what the strategy is doing, not what it was built from.
  const runQ = useQuery({
    queryKey: ["runs", runId],
    queryFn: () => api.run(runId as string),
    enabled: runId !== null,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });

  const live = strategiesQ.data?.strategies.find((s) => s.strategy === strategy) ?? null;
  const down = Boolean(strategiesQ.error) || (strategiesQ.isSuccess && live === null);

  const positions = live?.positions ?? [];
  const [ticker, setTicker] = useState<string | null>(null);

  useEffect(() => {
    const names = positions.map((p) => p.ticker);
    setTicker((cur) => (cur && names.includes(cur) ? cur : (names[0] ?? cur)));
    // Positions arrive on a 5s poll; the array identity changes every time, so
    // the name list is the only stable dependency worth watching.
  }, [positions.map((p) => p.ticker).join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (lock.selection.ticker) setTicker(lock.selection.ticker);
  }, [lock.selection.ticker]);

  // The one piece of replay state, and it is not state: the lock is out of
  // pinned mode and pointing somewhere, therefore we are reading history.
  const replaying = lock.t !== null && !lock.pinned;

  const invalidateLive = useCallback(() => {
    void qc.invalidateQueries({ queryKey: ["live"] });
  }, [qc]);

  const pause = useMutation({ mutationFn: () => api.pause(strategy), onSuccess: invalidateLive });
  const cancel = useMutation({
    mutationFn: () => api.cancelOrders(strategy),
    onSuccess: invalidateLive,
  });
  const kill = useMutation({
    mutationFn: (reason: string) => api.kill(reason),
    onSuccess: invalidateLive,
  });

  const orders = useMemo(() => orderStates(mine), [mine]);
  const gates = useMemo(() => mine.filter((e) => e.kind === "gate_block"), [mine]);
  const decisions = decisionsQ.data?.decisions ?? [];
  const showAgent = decisions.some((d) => d.agent !== null);

  return (
    <div className="p-3 flex flex-col gap-3">
      <LiveHeader
        strategy={strategy}
        live={live}
        down={down}
        pending={strategiesQ.isPending}
        error={strategiesQ.error}
        onPause={() => {
          if (
            window.confirm(
              `Pause ${strategy}?\n\nIt stops making new decisions. Open positions are left exactly as they are, and resuming is a CLI action.`,
            )
          ) {
            pause.mutate();
          }
        }}
        onCancel={() => {
          if (
            window.confirm(
              `Cancel every resting order for ${strategy}?\n\nThis removes unfilled intent. It cannot open a position.`,
            )
          ) {
            cancel.mutate();
          }
        }}
        onKill={() => {
          const reason = window.prompt(
            "Engage the kill switch?\n\nThis stops all new orders across every strategy immediately.\nReleasing it requires `lab kill --release` at a terminal.\n\nReason (recorded in the journal):",
            `manual stop from ${strategy} detail`,
          );
          if (reason !== null) kill.mutate(reason || `manual stop from ${strategy} detail`);
        }}
        busy={pause.isPending || cancel.isPending || kill.isPending}
      />

      <ReplayBar
        events={mine}
        replaying={replaying}
        t={lock.t}
        onSeek={(ms) => lock.seek(ms, "replay")}
        onLive={() => lock.pin(Date.now())}
      />

      <Panel
        title="positions"
        right={
          <span className="unit">
            {count(positions.length)} open · {money(live?.equity)} equity
          </span>
        }
      >
        <Positions
          positions={positions}
          selected={ticker}
          down={down}
          pending={strategiesQ.isPending}
          onSelect={(p) => {
            setTicker(p.ticker);
            lock.select(
              { kind: "trade", id: p.ticker, ticker: p.ticker },
              p.opened_at ? toMs(p.opened_at) : undefined,
              "positions",
            );
          }}
        />
      </Panel>

      <Panel
        title="price"
        right={
          positions.length > 0 ? (
            <label className="flex items-center gap-1.5">
              <span className="unit">ticker</span>
              <select
                value={ticker ?? ""}
                onChange={(e) => setTicker(e.target.value)}
                aria-label="ticker for the price chart"
                className="mono h-5 rounded border border-slate bg-graphite px-1 text-xs text-chalk"
              >
                {positions.map((p) => (
                  <option key={p.ticker} value={p.ticker}>
                    {p.ticker}
                  </option>
                ))}
              </select>
            </label>
          ) : null
        }
      >
        {runId === null || ticker === null ? (
          <Empty cmd={`lab paper start strategies/${strategy}.py --config cfg/${strategy}.yaml`}>
            {runId === null
              ? "no live run in the journal tail — nothing to chart against"
              : "no open position to chart"}
          </Empty>
        ) : (
          <div data-chart="price" data-ticker={ticker}>
            <PriceChart scope={SCOPE} runId={runId} ticker={ticker} height={240} />
          </div>
        )}
      </Panel>

      <Panel
        title={
          <span className="flex items-center gap-2">
            decision tape
            {replaying ? (
              <Badge kind="warn" title="the playhead is off now — this is replay">
                replay
              </Badge>
            ) : null}
          </span>
        }
        bodyClassName="h-[380px]"
      >
        {runId === null ? (
          <Empty cmd={`lab paper start strategies/${strategy}.py --config cfg/${strategy}.yaml`}>
            no decisions journalled for this session
          </Empty>
        ) : (
          <DecisionTape
            scope={SCOPE}
            decisions={decisions}
            loading={decisionsQ.isPending}
            error={decisionsQ.error}
            pinned={!replaying}
            showAgent={showAgent}
            emptyCmd={`lab paper start strategies/${strategy}.py --journal`}
          />
        )}
      </Panel>

      <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(380px,1fr))]">
        <Orders orders={orders} />
        <Gates gates={gates} today={live?.gate_events_today ?? null} />
      </div>

      <Provenance strategy={strategy} live={live} runId={runId} run={runQ.data ?? null} />

      <EventTape events={mine} status={status} height={220} />
    </div>
  );
}

// --- header -------------------------------------------------------------------

function LiveHeader({
  strategy,
  live,
  down,
  pending,
  error,
  onPause,
  onCancel,
  onKill,
  busy,
}: {
  strategy: string;
  live: LiveStrategy | null;
  down: boolean;
  pending: boolean;
  error: unknown;
  onPause: () => void;
  onCancel: () => void;
  onKill: () => void;
  busy: boolean;
}) {
  return (
    <header className="bg-gunmetal border border-slate rounded-md">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
        <span className="mono text-lg text-chalk">{strategy}</span>
        {live ? <Badge kind={live.mode === "live" ? "warn" : "neutral"}>{live.mode}</Badge> : null}
        {live ? (
          <StatusDot health={STATUS_HEALTH[live.status] ?? "idle"} label={live.status} />
        ) : (
          <StatusDot health="idle" label={pending ? "reading" : "not reporting"} />
        )}

        <span className="flex items-baseline gap-2">
          <span className="unit">p&amp;l</span>
          <span className={cx("mono text-xl leading-none", toneClass(live?.pnl_pct))}>
            {delta(live?.pnl_pct)}
          </span>
        </span>
        <span className="mono text-ash">{money(live?.equity)}</span>
        <span className="mono text-ash">
          {count(live?.n_positions)} <span className="unit">pos</span>
        </span>
        <span className="mono text-ash-dim" title={live?.next_fire ?? "no schedule"}>
          next {clock(live?.next_fire)}
        </span>

        <span className="flex-1" />

        {/* The only three mutations in the app, all confirm-gated, all strictly
            exposure-reducing. */}
        <Button onClick={onPause} disabled={busy || down} title="stop making new decisions">
          ⏸ pause
        </Button>
        <Button onClick={onCancel} disabled={busy || down} title="cancel resting orders">
          ✕ cancel orders
        </Button>
        <Button kind="danger" onClick={onKill} disabled={busy} title="engage the kill switch">
          ⛔ kill
        </Button>
      </div>

      <p className="px-3 pb-2 text-xs text-ash-dim">
        No start, resume or raise-limit control here: the console can only make the system
        safer. Those are CLI actions behind the manual checklist —{" "}
        <span className="mono text-ash">lab paper start …</span>,{" "}
        <span className="mono text-ash">lab kill --release</span>.
      </p>

      {down ? (
        <div className="px-3 pb-2">
          <p className="text-xs text-ash">
            The runner is not reporting {strategy}. That is a normal state, not an error: this
            page keeps working as forensics over the journal below, and the live figures above
            are blank rather than stale.
          </p>
          {error ? <ErrorNote error={error} /> : null}
        </div>
      ) : null}
    </header>
  );
}

// --- replay -------------------------------------------------------------------

function ReplayBar({
  events,
  replaying,
  t,
  onSeek,
  onLive,
}: {
  events: JournalEvent[];
  replaying: boolean;
  t: number | null;
  onSeek: (ms: number) => void;
  onLive: () => void;
}) {
  const now = useNow(1000);
  const start = firstOfToday(events, now);

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 bg-gunmetal border border-slate rounded-md">
      <span className="unit">playhead</span>
      <span className="mono text-teal w-[9ch]">{t === null ? MISSING : clock(t)}</span>

      {replaying ? (
        <Badge kind="warn">replaying</Badge>
      ) : (
        <Badge kind="accent" title="the playhead is pinned to now">
          live
        </Badge>
      )}

      <input
        type="range"
        min={start ?? now - 3600_000}
        max={now}
        step={1000}
        value={t === null ? now : Math.min(Math.max(t, start ?? now - 3600_000), now)}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="scrub today's timeline"
        disabled={start === null}
        className="flex-1 min-w-[160px] accent-teal"
      />

      <Button
        onClick={() => onSeek(start ?? now)}
        disabled={start === null}
        title={
          start === null
            ? "no events journalled today"
            : `jump to the first event of the day (${clock(start)})`
        }
      >
        ⏮ replay today
      </Button>
      <Button kind="accent" onClick={onLive} disabled={!replaying} title="re-pin the playhead to now">
        ▶ back to live
      </Button>
      <span className="unit text-ash-dim">
        {start === null ? "nothing today" : `from ${clock(start)}`}
      </span>
    </div>
  );
}

// --- positions ----------------------------------------------------------------

function Positions({
  positions,
  selected,
  down,
  pending,
  onSelect,
}: {
  positions: PositionRow[];
  selected: string | null;
  down: boolean;
  pending: boolean;
  onSelect: (p: PositionRow) => void;
}) {
  const now = useNow(1000);

  if (pending) return <Loading what="reading positions" />;
  if (down) {
    return (
      <Empty cmd="lab paper status --json">
        the runner is not reporting — position state is unknown, not flat
      </Empty>
    );
  }
  if (positions.length === 0) {
    return <Empty cmd="lab paper status --json">flat — no open positions</Empty>;
  }

  return (
    <Table>
      <thead>
        <tr>
          <Th>ticker</Th>
          <Th align="right">qty</Th>
          <Th align="right">avg</Th>
          <Th align="right">last</Th>
          <Th align="right">value</Th>
          <Th align="right">unrealized</Th>
          <Th align="right">%</Th>
          <Th align="right">age</Th>
          <Th>tag</Th>
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => (
          <Tr key={p.ticker} selected={selected === p.ticker} onClick={() => onSelect(p)}>
            <Td mono className="text-chalk">
              {p.ticker}
            </Td>
            <Td align="right" mono>
              {num(p.qty, 0)}
            </Td>
            <Td align="right" mono className="text-ash">
              {num(p.avg_price)}
            </Td>
            <Td align="right" mono>
              {num(p.last_price)}
            </Td>
            <Td align="right" mono className="text-ash">
              {money(p.market_value)}
            </Td>
            <Td align="right" mono className={toneClass(p.unrealized_pnl)}>
              {signedMoney(p.unrealized_pnl, 2)}
            </Td>
            <Td align="right" mono className={toneClass(p.unrealized_pct)}>
              {delta(p.unrealized_pct)}
            </Td>
            <Td align="right" mono className="text-ash" title={stamp(p.opened_at)}>
              {p.opened_at === null ? MISSING : age((now - toMs(p.opened_at)) / 1000)}
            </Td>
            <Td className="text-ash-dim">{p.tag || MISSING}</Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  );
}

// --- orders and gates ---------------------------------------------------------

const ORDER_TONE: Record<OrderState, string> = {
  open: "text-teal",
  filled: "text-moss",
  rejected: "text-ember",
  unknown: "text-ash-dim",
};

function Orders({ orders }: { orders: { event: JournalEvent; state: OrderState }[] }) {
  const open = orders.filter((o) => o.state === "open");
  return (
    <Panel
      title="orders"
      right={
        <span className="unit">
          {count(open.length)} resting of {count(orders.length)} seen
        </span>
      }
      bodyClassName="max-h-[240px] overflow-y-auto"
    >
      {orders.length === 0 ? (
        <Empty cmd="lab paper status --json">no orders in the journal window</Empty>
      ) : (
        <>
          <Table>
            <thead>
              <tr>
                <Th>time</Th>
                <Th>ticker</Th>
                <Th>state</Th>
                <Th>order</Th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <Tr key={o.event.seq}>
                  <Td mono className="text-ash-dim">
                    {clock(o.event.at)}
                  </Td>
                  <Td mono className="text-chalk">
                    {o.event.ticker || MISSING}
                  </Td>
                  <Td mono className={ORDER_TONE[o.state]}>
                    {o.state}
                  </Td>
                  <Td mono className="text-ash">
                    {o.event.message}
                  </Td>
                </Tr>
              ))}
            </tbody>
          </Table>
          <p className="px-3 py-1.5 border-t border-slate text-xs text-ash-dim">
            Resting state is derived from the journal window held in this tab, not from the
            broker's book. An order the runner never journalled an id for shows as{" "}
            <span className="mono">unknown</span> rather than being counted as open.
          </p>
        </>
      )}
    </Panel>
  );
}

function Gates({ gates, today }: { gates: JournalEvent[]; today: number | null }) {
  return (
    <Panel
      title="gate events"
      right={
        <span className={cx("unit", (today ?? 0) > 0 && "text-amber")}>
          {today === null ? count(gates.length) : count(today)} today
        </span>
      }
      bodyClassName="max-h-[240px] overflow-y-auto"
    >
      {gates.length === 0 ? (
        <Empty cmd="lab paper status --json">the risk gate has blocked nothing in this window</Empty>
      ) : (
        <Table>
          <thead>
            <tr>
              <Th>time</Th>
              <Th>ticker</Th>
              <Th>reason</Th>
            </tr>
          </thead>
          <tbody>
            {gates.map((g) => (
              <Tr key={g.seq}>
                <Td mono className="text-ash-dim">
                  {clock(g.at)}
                </Td>
                <Td mono className="text-chalk">
                  {g.ticker || MISSING}
                </Td>
                <Td mono className="text-amber">
                  {g.message}
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

// --- provenance ---------------------------------------------------------------

function Provenance({
  strategy,
  live,
  runId,
  run,
}: {
  strategy: string;
  live: LiveStrategy | null;
  runId: string | null;
  run: { git_commit: string | null; config_hash: string; data_version: string; params: Record<string, unknown> } | null;
}) {
  const now = useNow(1000);
  const started = live?.started_at ?? null;
  const uptime = started === null ? null : (now - toMs(started)) / 1000;
  const params = Object.entries(run?.params ?? {});

  return (
    <Panel
      title="provenance"
      right={
        runId ? (
          <Link
            to="/runs/$runId"
            params={{ runId }}
            className="mono text-xs text-teal no-underline hover:underline"
          >
            run {shortId(runId, 10)} ↗
          </Link>
        ) : (
          <span className="unit text-ash-dim">no run id in the journal</span>
        )
      }
    >
      <div className="grid gap-x-6 gap-y-1 px-3 py-2 text-xs [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
        <Field label="strategy" value={strategy} />
        <Field label="mode" value={live?.mode ?? MISSING} />
        <Field label="git commit" value={run?.git_commit ?? MISSING} />
        <Field label="config hash" value={run?.config_hash ?? MISSING} />
        <Field label="data version" value={run?.data_version || MISSING} />
        <Field label="started" value={stamp(started)} />
        <Field label="uptime" value={uptime === null ? MISSING : duration(uptime)} />
      </div>

      {params.length > 0 ? (
        <div className="px-3 pb-2 flex flex-wrap gap-x-4 gap-y-0.5 text-xs">
          <span className="unit">params</span>
          {params.map(([k, v]) => (
            <span key={k} className="mono text-ash">
              {k}=<span className="text-chalk">{String(v)}</span>
            </span>
          ))}
        </div>
      ) : (
        <p className="px-3 pb-2 text-xs text-ash-dim">
          No registry record for this session yet, so params and commit are unknown — shown as
          {" "}
          <span className="mono">{MISSING}</span> rather than guessed from the strategy file.
        </p>
      )}
    </Panel>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-2 min-w-0">
      <span className="unit shrink-0">{label}</span>
      <span className="mono text-ash truncate" title={value}>
        {value}
      </span>
    </span>
  );
}
