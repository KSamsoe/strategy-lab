/**
 * The live event tail, rendered.
 *
 * A closed socket is a NORMAL state here, not an error. The console reads
 * journals; the runner being down means there is nothing new to read, not that
 * the tool is broken -- so the header says so in a sentence and the tape below
 * keeps working as forensics. An error banner or a forever-spinner would both
 * be lies about what is wrong.
 */

import { useEffect, useMemo, useRef } from "react";

import { clock, count } from "../lib/format";
import type { EventKind, JournalEvent } from "../lib/types";
import { Empty, Panel, StatusDot, cx, type Health } from "./ui";

export interface EventTapeProps {
  events: JournalEvent[];
  status?: "open" | "closed" | "error";
  height?: number;
  onSelectStrategy?: (s: string) => void;
}

/**
 * Kind is the row's only hue. Message weight (chalk vs ash) carries urgency
 * instead of a second color, so a busy tape does not turn into a paint chart.
 */
const NEUTRAL = { kind: "text-ash", msg: "text-ash" } as const;

const STYLE: Record<EventKind, { kind: string; msg: string }> = {
  fill: { kind: "text-moss", msg: "text-chalk" },
  reject: { kind: "text-ember", msg: "text-chalk" },
  error: { kind: "text-ember", msg: "text-chalk" },
  gate_block: { kind: "text-amber", msg: "text-chalk" },
  breaker: { kind: "text-amber", msg: "text-chalk" },
  reconcile: { kind: "text-teal", msg: "text-ash" },
  heartbeat: { kind: "text-ash-dim", msg: "text-ash-dim" },
  log: { kind: "text-ash-dim", msg: "text-ash" },
  decision: NEUTRAL,
  order: NEUTRAL,
  run_start: NEUTRAL,
  run_end: NEUTRAL,
};

const CONN: Record<NonNullable<EventTapeProps["status"]>, { health: Health; label: string }> = {
  open: { health: "ok", label: "tailing" },
  closed: { health: "idle", label: "socket closed" },
  error: { health: "warn", label: "socket dropped" },
};

/** Fixed columns so the tape reads as a ledger, not as ragged log output. */
const ROW =
  "grid grid-cols-[8ch_11ch_11ch_minmax(0,1fr)] gap-x-3 items-center px-3 w-full text-left text-xs";

export function EventTape({
  events,
  status = "closed",
  height = 320,
  onSelectStrategy,
}: EventTapeProps) {
  // Which seqs are new *since the last commit*. Seeding 500 rows on mount must
  // not light the whole tape up, so the first paint counts as history: `prev`
  // is null then and nothing pulses.
  const prev = useRef<Set<number> | null>(null);
  const fresh = useMemo(() => {
    const before = prev.current;
    if (!before) return null;
    const now = new Set<number>();
    for (const e of events) if (!before.has(e.seq)) now.add(e.seq);
    return now;
  }, [events]);

  // Written after commit, never during render: under StrictMode's double render
  // a render-phase write would eat the pulse on the second pass.
  useEffect(() => {
    prev.current = new Set(events.map((e) => e.seq));
  }, [events]);

  const conn = CONN[status];

  return (
    <Panel
      title="Event journal"
      right={
        <span className="flex items-center gap-3">
          <span className="unit">{count(events.length)} events</span>
          <StatusDot
            health={conn.health}
            label={conn.label}
            detail={
              status === "open"
                ? "tailing the event journal over the WebSocket"
                : "the console reads the journal, not the runner"
            }
          />
        </span>
      }
    >
      {status === "open" ? null : (
        <p className="px-3 py-1.5 border-b border-slate text-xs text-ash-dim">
          {status === "error"
            ? "Event socket dropped; retrying. "
            : "Event socket closed — the runner is not connected. "}
          Everything below is the journal as it stands.
        </p>
      )}

      <div
        role="log"
        aria-live="off"
        aria-label="Event journal tail"
        className="scroll-x overflow-y-auto"
        style={{ height }}
      >
        {events.length === 0 ? (
          <Empty cmd="lab paper start strategies/momo.py --config cfg/momo.yaml">
            {status === "open"
              ? "socket is up — no events journaled yet"
              : "no events in the journal"}
          </Empty>
        ) : (
          <ol className="min-w-max">
            {events.map((e) => (
              <TapeRow
                key={e.seq}
                event={e}
                pulse={fresh?.has(e.seq) ?? false}
                onSelectStrategy={onSelectStrategy}
              />
            ))}
          </ol>
        )}
      </div>
    </Panel>
  );
}

function TapeRow({
  event,
  pulse,
  onSelectStrategy,
}: {
  event: JournalEvent;
  pulse: boolean;
  onSelectStrategy?: (s: string) => void;
}) {
  // Kinds come off the wire; a contract drift must dim a row, not crash a tape.
  const style = STYLE[event.kind] ?? NEUTRAL;
  const strategy = event.strategy;
  const select = strategy && onSelectStrategy ? () => onSelectStrategy(strategy) : null;

  const cells = (
    <>
      <span className="mono text-ash-dim">{clock(event.at)}</span>
      <span className="mono text-ash truncate" title={event.source}>
        {event.source}
      </span>
      <span className={cx("mono uppercase truncate", style.kind)} data-kind={event.kind}>
        {event.kind}
      </span>
      <span className={cx("mono truncate", style.msg)}>
        {event.ticker ? <span className="text-chalk">{event.ticker} </span> : null}
        {event.message}
      </span>
    </>
  );

  const title = [strategy, event.ticker, event.message].filter(Boolean).join(" · ");

  return (
    <li data-seq={event.seq} className={cx(!select && ROW, pulse && !select && "pulse")}>
      {select ? (
        <button
          type="button"
          onClick={select}
          title={title}
          style={{ height: "var(--row-h)" }}
          className={cx(ROW, pulse && "pulse", "hover:bg-slate/40 cursor-pointer")}
        >
          {cells}
        </button>
      ) : (
        <span className="contents" title={title}>
          {cells}
        </span>
      )}
    </li>
  );
}
