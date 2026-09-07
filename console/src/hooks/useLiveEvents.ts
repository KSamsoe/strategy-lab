/**
 * The live event tail.
 *
 * The console never touches the broker or the runner -- it reads journals. So
 * this hook has exactly two inputs (a REST seed and a WebSocket tail of the
 * same append-only journal) and one invariant: the `seq` cursor. Seq is why the
 * server exposes it at all; losing it means the socket resumes from the wrong
 * place and the tape has a silent gap, which is the one failure a forensics
 * tool cannot have.
 *
 * The buffer is a ring so the tape stays O(visible) rather than O(history). A
 * runner that has been up for a week must not put a week of rows in the DOM.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api, subscribeEvents } from "../lib/api";
import type { EventKind, JournalEvent } from "../lib/types";

/** ~16 screens of tape at 32px rows: enough to scroll back, cheap to hold. */
const DEFAULT_LIMIT = 500;

/** Slow path when the socket will not open. Long, because it is a fallback. */
const FALLBACK_POLL_MS = 15_000;

export type ConnStatus = "open" | "closed" | "error";

export interface UseLiveEventsOptions {
  limit?: number;
  kinds?: EventKind[];
}

export interface LiveEventsResult {
  /** Newest first, ring-buffered to `limit`. */
  events: JournalEvent[];
  status: ConnStatus;
  /** Highest journal seq observed, filtered-out kinds included. */
  latestSeq: number;
}

/**
 * Merge into a newest-first ring, deduped by seq.
 *
 * The dedupe is load-bearing, not defensive: the seed and the socket overlap by
 * construction whenever an event lands between the REST response and the
 * socket's `since`, and a doubled fill line on a trading tape reads as a
 * doubled fill.
 */
function ring(prev: JournalEvent[], incoming: JournalEvent[], limit: number): JournalEvent[] {
  if (incoming.length === 0) return prev;

  const seen = new Set<number>();
  for (const e of prev) seen.add(e.seq);

  const fresh: JournalEvent[] = [];
  for (const e of incoming) {
    if (seen.has(e.seq)) continue;
    seen.add(e.seq);
    fresh.push(e);
  }
  if (fresh.length === 0) return prev;

  // Sort the union rather than unshifting: a replayed frame arriving after a
  // live one must land in journal order, not on top of it.
  const next = [...fresh, ...prev].sort((a, b) => b.seq - a.seq);
  return next.length > limit ? next.slice(0, limit) : next;
}

export function useLiveEvents(opts: UseLiveEventsOptions = {}): LiveEventsResult {
  const limit = opts.limit ?? DEFAULT_LIMIT;

  // A string key so an inline `kinds={["fill"]}` at the call site does not tear
  // the socket down on every render. No kinds (or an empty list) means all.
  const kindKey = opts.kinds?.join(",") ?? "";
  const accepted = useMemo(
    () => (kindKey === "" ? null : new Set<string>(kindKey.split(","))),
    [kindKey],
  );

  const [events, setEvents] = useState<JournalEvent[]>([]);
  const [status, setStatus] = useState<ConnStatus>("closed");
  const [latestSeq, setLatestSeq] = useState(0);

  // The resume cursor lives in a ref because it is read synchronously when the
  // socket connects, and because putting it in the effect's deps would
  // re-subscribe on every single event.
  const cursor = useRef(0);
  const note = useCallback((seq: number) => {
    if (!Number.isFinite(seq) || seq <= cursor.current) return;
    cursor.current = seq;
    setLatestSeq(seq);
  }, []);

  // Instance-scoped cache key: this query's `since` comes from a cursor this
  // hook owns, so two tapes sharing one cache entry would hand each other a
  // seed that starts past their own history.
  const instance = useId();

  const seed = useQuery({
    queryKey: ["live", "events", instance, limit],
    queryFn: () => api.liveEvents({ since: cursor.current, limit }),
    retry: false,
    gcTime: 0,
    // The socket is the live path, so nothing polls while it is open. If it
    // never opens -- blocked WS, runner restart loop -- REST takes over slowly
    // rather than letting the tape go blind without saying so.
    refetchInterval: status === "open" ? false : FALLBACK_POLL_MS,
  });

  const seeded = seed.data;
  useEffect(() => {
    if (!seeded) return;
    for (const e of seeded.events) note(e.seq);
    note(seeded.latest_seq);
    const rows = accepted ? seeded.events.filter((e) => accepted.has(e.kind)) : seeded.events;
    setEvents((prev) => ring(prev, rows, limit));
  }, [seeded, accepted, limit, note]);

  // Subscribe only once the seed has settled, so the socket resumes from a real
  // cursor instead of replaying the whole journal from zero. An errored seed
  // still releases the gate: a dead REST layer must not also cost us the tail.
  const settled = seed.isSuccess || seed.isError;
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (settled) setReady(true);
  }, [settled]);

  useEffect(() => {
    if (!ready) return;
    return subscribeEvents(
      (e) => {
        // The cursor advances on every event, including kinds this view filters
        // out: it is a position in the journal, not a position in the view, and
        // skipping a filtered seq makes the server replay it on every reconnect.
        note(e.seq);
        if (accepted && !accepted.has(e.kind)) return;
        setEvents((prev) => ring(prev, [e], limit));
      },
      { since: cursor.current, onStatus: setStatus },
    );
  }, [ready, accepted, limit, note]);

  return { events, status, latestSeq };
}
