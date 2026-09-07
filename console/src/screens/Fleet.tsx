/**
 * Fleet overview — the screen that is open all day.
 *
 * Its job is to be readable in one glance and to be *honest when nothing is
 * happening*, which is the harder half. A dashboard that shows a green dot for
 * a heartbeat that stopped an hour ago is worse than no dashboard: it converts
 * "I do not know" into "everything is fine". So freshness here is computed
 * locally and keeps advancing between polls — the server's `age_s` is the
 * anchor and wall-clock drift since that response is added on top. If the API
 * itself goes away the dots keep aging and go amber, then red, on their own.
 *
 * The GovGreed quota meter sits in the same strip as the connections rather
 * than in a panel somewhere below, because the whole bot is shaped around a
 * 20-call day: "how many calls are left" is a health question, not a statistic.
 */

import { useEffect, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import { EventTape } from "../components/EventTape";
import { useLiveEvents } from "../hooks/useLiveEvents";
import type { AdapterHealth, Heartbeat, LiveStrategy } from "../lib/types";
import { age, clock, count, delta, money, toneClass } from "../lib/format";
import {
  Badge,
  Empty,
  ErrorNote,
  Loading,
  Panel,
  StatusDot,
  cx,
  type Health,
} from "../components/ui";

// --- freshness ----------------------------------------------------------------

/** Default beat interval when a source does not declare its own `period_s`. */
export const HEARTBEAT_PERIOD_S = 30;

/**
 * Staleness, graded against what the source *promised*.
 *
 * Two missed beats is still plausibly a slow poll; ten is a source that has
 * stopped. Both thresholds are relative to the declared period so a 30-second
 * runner heartbeat and a 6-hour daily adapter are not judged by one number.
 */
export function staleness(
  ageSeconds: number | null | undefined,
  periodS: number = HEARTBEAT_PERIOD_S,
): Health {
  if (ageSeconds === null || ageSeconds === undefined || !Number.isFinite(ageSeconds)) {
    return "idle";
  }
  const period = Number.isFinite(periodS) && periodS > 0 ? periodS : HEARTBEAT_PERIOD_S;
  if (ageSeconds <= period * 2) return "ok";
  if (ageSeconds <= period * 10) return "warn";
  return "bad";
}

/**
 * Age *now*, not age at the moment the server answered.
 *
 * Server-side `age_s` is skew-free and therefore the right anchor; adding the
 * time elapsed since that response arrived is what stops a dot freezing green
 * when the poll loop itself dies. Never derived from `at` alone — a clock five
 * minutes off would then report a heartbeat as five minutes stale forever.
 */
export function ageNow(
  serverAgeS: number | null | undefined,
  fetchedAtMs: number,
  nowMs: number,
): number | null {
  if (serverAgeS === null || serverAgeS === undefined || !Number.isFinite(serverAgeS)) {
    return null;
  }
  const drift = fetchedAtMs > 0 ? Math.max(0, (nowMs - fetchedAtMs) / 1000) : 0;
  return Math.max(0, serverAgeS) + drift;
}

/** Declared beat interval, when the source bothered to say. */
export function periodOf(meta: Record<string, unknown> | undefined): number {
  const v = meta?.period_s ?? meta?.poll_seconds;
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : HEARTBEAT_PERIOD_S;
}

/**
 * Quota as a health state. At the cap the bot is not "nearly fine", it is done
 * for the day and every signal it would have read is simply missing.
 */
export function quotaHealth(used: number | null, limit: number | null): Health {
  if (used === null || limit === null || !Number.isFinite(limit) || limit <= 0) return "idle";
  const fraction = used / limit;
  if (fraction >= 1) return "bad";
  if (fraction >= 0.8) return "warn";
  return "ok";
}

/** An adapter that cannot answer is worse news than an adapter that is slow. */
export function adapterHealth(a: AdapterHealth): Health {
  if (!a.available) return "bad";
  const q = quotaHealth(a.quota_used, a.quota_limit);
  return q === "idle" ? "ok" : q;
}

/** Ticking wall clock. Isolated in the strip that needs it so the page does not
 *  re-render once a second just to keep four dots honest. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

// --- sparkline ----------------------------------------------------------------

/** Samples the console has actually observed. Module-scoped so navigating away
 *  and back does not throw the session's trace away. */
const TRACE_CAP = 240;
const EQUITY_TRACE: { t: number; v: number }[] = [];

export function pushSample(
  trace: { t: number; v: number }[],
  t: number,
  v: number,
  cap = TRACE_CAP,
): void {
  if (!Number.isFinite(v)) return;
  const last = trace[trace.length - 1];
  if (last && last.t === t) return;
  trace.push({ t, v });
  if (trace.length > cap) trace.splice(0, trace.length - cap);
}

/**
 * Polyline through the samples, autoscaled to its own range. These are SVG
 * geometry, not figures anyone reads, so they are rounded here rather than run
 * through `format` — nothing in this string is ever shown as a number.
 */
export function sparkPath(values: readonly number[], w: number, h: number): string {
  if (values.length < 2) return "";
  const round = (n: number): number => Math.round(n * 100) / 100;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;
  return values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - lo) / span) * h;
      return `${i === 0 ? "M" : "L"}${round(x)},${round(y)}`;
    })
    .join(" ");
}

function Sparkline({ samples }: { samples: { t: number; v: number }[] }) {
  const values = samples.map((s) => s.v);
  const d = sparkPath(values, 96, 20);
  if (!d) {
    return (
      <span className="unit text-ash-dim" title="one sample so far — a line needs two">
        no trace yet
      </span>
    );
  }
  const rising = values[values.length - 1] >= values[0];
  return (
    <svg
      width="96"
      height="20"
      viewBox="0 0 96 20"
      className={rising ? "text-moss" : "text-ember"}
      // Not the day's curve: it is what this console has watched since it was
      // opened. Saying so beats implying a session-length trace is the day.
      role="img"
    >
      <title>{`equity as observed by this console · ${values.length} samples`}</title>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}

// --- screen -------------------------------------------------------------------

export default function Fleet() {
  const navigate = useNavigate();

  const health = useQuery({
    queryKey: ["live", "health"],
    queryFn: api.liveHealth,
    refetchInterval: 5000,
    retry: false,
  });

  const strategies = useQuery({
    queryKey: ["live", "strategies"],
    queryFn: api.liveStrategies,
    refetchInterval: 5000,
    retry: false,
  });

  const { events, status } = useLiveEvents({ limit: 300 });

  // Record every distinct equity reading the poll loop brings back.
  const stamp = health.dataUpdatedAt;
  const equity = health.data?.equity ?? null;
  useEffect(() => {
    if (equity === null || stamp === 0) return;
    pushSample(EQUITY_TRACE, stamp, equity);
  }, [equity, stamp]);
  const trace = EQUITY_TRACE.slice();

  const rows = strategies.data?.strategies ?? [];
  const breaker = health.data?.breaker;
  const killed = health.data?.kill_switch.engaged ?? false;

  return (
    <div className="p-3 flex flex-col gap-3">
      <Panel bodyClassName="flex flex-col">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-3 py-2">
          <div className="flex items-baseline gap-2">
            <span className="unit">equity</span>
            <span className="mono text-2xl leading-none text-chalk">
              {health.isPending && equity === null ? "…" : money(equity)}
            </span>
          </div>

          <Sparkline samples={trace} />

          <div className="flex items-baseline gap-2">
            <span className="unit">today</span>
            <span className={cx("mono text-lg leading-none", toneClass(health.data?.day_change_pct))}>
              {delta(health.data?.day_change_pct)}
            </span>
          </div>

          <div className="flex-1" />

          {killed ? (
            <Badge kind="bad" title={health.data?.kill_switch.reason ?? undefined}>
              kill switch engaged
            </Badge>
          ) : breaker?.tripped ? (
            <Badge kind="warn" title={breaker.reason}>
              breaker tripped
            </Badge>
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="unit">breaker</span>
              <span className="mono text-ash">{breaker ? "armed" : "—"}</span>
            </span>
          )}
        </div>

        <HealthStrip
          adapters={health.data?.adapters ?? []}
          heartbeats={health.data?.heartbeats ?? []}
          fetchedAt={stamp}
          pending={health.isPending}
          error={health.error}
        />
      </Panel>

      <Panel
        title="strategies"
        right={<span className="unit">{count(rows.length)} registered</span>}
      >
        {strategies.isPending ? (
          <Loading what="reading live strategies" />
        ) : strategies.error ? (
          <>
            {/* The runner being down is normal. Say what is unknown, then get
                out of the way — the event journal below still works. */}
            <p className="px-3 pt-2 text-xs text-ash-dim">
              The runner is not answering. Nothing below is live; the journal is still readable.
            </p>
            <ErrorNote error={strategies.error} />
          </>
        ) : rows.length === 0 ? (
          <Empty cmd="lab paper start strategies/momo.py --config cfg/momo.yaml">
            nothing is running
          </Empty>
        ) : (
          <div className="grid gap-px bg-slate/60 [grid-template-columns:repeat(auto-fill,minmax(280px,1fr))]">
            {rows.map((s) => (
              <StrategyCard key={s.strategy} s={s} />
            ))}
          </div>
        )}
      </Panel>

      <EventTape
        events={events}
        status={status}
        onSelectStrategy={(s) => void navigate({ to: "/live/$strategy", params: { strategy: s } })}
      />
    </div>
  );
}

// --- health strip -------------------------------------------------------------

function HealthStrip({
  adapters,
  heartbeats,
  fetchedAt,
  pending,
  error,
}: {
  adapters: AdapterHealth[];
  heartbeats: Heartbeat[];
  fetchedAt: number;
  pending: boolean;
  error: unknown;
}) {
  const now = useNow(1000);

  // GovGreed is pulled out of the adapter list on purpose: the quota is a
  // first-class health item, not a footnote on a data source.
  const govgreed = adapters.find((a) => a.name === "govgreed") ?? null;
  const others = adapters.filter((a) => a.name !== "govgreed");

  if (error) {
    // The reason goes on screen, not into a tooltip: "why is health blank" is
    // the question the operator is already asking by the time they look here.
    return (
      <div className="border-t border-slate px-3 py-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <StatusDot health="bad" label="health" />
        <span className="text-ash-dim">
          no health report — treat every dot below as unknown, not as green
        </span>
        <span className="mono text-ember">
          {error instanceof Error ? error.message : String(error)}
        </span>
      </div>
    );
  }

  return (
    <div className="border-t border-slate px-3 py-1.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
      {pending && adapters.length === 0 && heartbeats.length === 0 ? (
        <span className="unit text-ash-dim">reading health…</span>
      ) : null}

      {others.map((a) => (
        <StatusDot
          key={a.name}
          health={adapterHealth(a)}
          label={a.name}
          ageSeconds={a.last_call === null ? null : Math.max(0, (now - new Date(a.last_call).getTime()) / 1000)}
          detail={
            a.available
              ? `provides ${a.provides.join(", ") || "—"}`
              : a.reason || "adapter unavailable"
          }
        />
      ))}

      <QuotaMeter adapter={govgreed} />

      {heartbeats.map((hb) => {
        const seconds = ageNow(hb.age_s, fetchedAt, now);
        return (
          <StatusDot
            key={hb.source}
            health={staleness(seconds, periodOf(hb.meta))}
            label={hb.source}
            ageSeconds={seconds}
            detail={`last beat ${clock(hb.at)} — expected every ${age(periodOf(hb.meta))}`}
          />
        );
      })}

      {heartbeats.length === 0 && !pending ? (
        <StatusDot health="idle" label="no heartbeats" detail="nothing has reported in" />
      ) : null}
    </div>
  );
}

/**
 * "govgreed ● 14/20 calls" — the number the whole strategy is budgeted against.
 * The bar is the meter; the hue is the warning state, and it is the only reason
 * this element is allowed a colour at all.
 */
function QuotaMeter({ adapter }: { adapter: AdapterHealth | null }) {
  if (!adapter) {
    return (
      <StatusDot
        health="idle"
        label="govgreed"
        detail="adapter not configured — no signals will be pulled"
      />
    );
  }

  const used = adapter.quota_used;
  const limit = adapter.quota_limit;
  const health = !adapter.available ? "bad" : quotaHealth(used, limit);
  const known = used !== null && limit !== null && limit > 0;
  const fraction = known ? Math.min(1, used / limit) : 0;
  const bar =
    health === "bad" ? "bg-ember" : health === "warn" ? "bg-amber" : "bg-moss";

  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap"
      title={
        adapter.available
          ? `${adapter.quota_tier ?? "quota"} — last call ${clock(adapter.last_call)}`
          : adapter.reason || "govgreed unavailable"
      }
    >
      <span
        className={cx(
          "size-1.5 rounded-full shrink-0",
          health === "bad" ? "bg-ember" : health === "warn" ? "bg-amber" : health === "ok" ? "bg-moss" : "bg-ash-dim",
        )}
        aria-hidden
      />
      <span className="text-ash">govgreed</span>
      <span className="mono text-chalk">
        {known ? `${count(used)}/${count(limit)}` : "—/—"}
      </span>
      <span className="unit">calls</span>
      <span className="w-14 h-1 rounded bg-slate overflow-hidden" aria-hidden>
        <span className={cx("block h-full", bar)} style={{ width: `${fraction * 100}%` }} />
      </span>
    </span>
  );
}

// --- strategy card ------------------------------------------------------------

const STATUS_HEALTH: Record<LiveStrategy["status"], Health> = {
  running: "ok",
  paused: "warn",
  blocked: "bad",
  stopped: "idle",
};

function StrategyCard({ s }: { s: LiveStrategy }) {
  const now = useNow(1000);
  const fires = s.next_fire ? new Date(s.next_fire).getTime() : null;
  const untilS = fires === null ? null : (fires - now) / 1000;

  return (
    <Link
      to="/live/$strategy"
      params={{ strategy: s.strategy }}
      className="bg-gunmetal hover:bg-slate/30 no-underline flex flex-col gap-1.5 px-3 py-2"
    >
      <div className="flex items-center gap-2">
        <span className="mono text-chalk truncate">{s.strategy}</span>
        {/* PAPER vs LIVE is the single most consequential fact on the card, so
            it is the one badge that is allowed to shout. */}
        <Badge kind={s.mode === "live" ? "warn" : "neutral"}>{s.mode}</Badge>
        <span className="flex-1" />
        <StatusDot health={STATUS_HEALTH[s.status] ?? "idle"} label={s.status} />
      </div>

      <div className="flex items-baseline gap-3">
        <span className={cx("mono text-xl leading-none", toneClass(s.pnl_pct))}>
          {delta(s.pnl_pct)}
        </span>
        <span className="mono text-ash">{money(s.equity)}</span>
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
        <span className="mono text-ash">
          {count(s.n_positions)} <span className="unit">pos</span>
        </span>
        <span className={cx("mono", s.gate_events_today > 0 ? "text-amber" : "text-ash")}>
          gate {count(s.gate_events_today)}
        </span>
        {s.spend_usd === undefined ? null : (
          <span className="mono text-ash" title="model spend today">
            {money(s.spend_usd, 2)}
          </span>
        )}
        <span className="flex-1" />
        <span className="mono text-ash-dim" title={s.next_fire ?? "no schedule"}>
          next {clock(s.next_fire)}
          {untilS !== null && untilS > 0 ? (
            <span className="text-ash-dim"> · in {age(untilS)}</span>
          ) : null}
        </span>
      </div>
    </Link>
  );
}
