/**
 * The time-lock bus: chart <-> tape <-> ledger synchronization.
 *
 * Everything on a detail screen is looking at the same instant. Click a trade
 * row and the chart pans and the tape scrolls; scrub the tape and the chart
 * crosshair follows; hover a candle and the ledger highlights the trade that
 * spans it. That is one shared value -- a playhead timestamp -- plus a
 * selection, scoped to `run_id` (or the literal `"live"`).
 *
 * It is deliberately a tiny external store rather than React state or a
 * dependency: every screen uses it, it updates on pointer-move, and it must not
 * re-render the world. Components subscribe through `useSyncExternalStore` and
 * read only the slice they care about.
 *
 * Live mode is the same object with the playhead pinned to now, which is the
 * whole trick behind "backtests get a live-quality inspector for free, live
 * trading gets backtest-quality forensics for free."
 */

import { useCallback, useMemo, useSyncExternalStore } from "react";

export type Scope = string; // a run_id, or "live"

export type SelectionKind = "trade" | "decision" | "order" | "fill" | null;

export interface Selection {
  kind: SelectionKind;
  id: string | null;
  ticker: string | null;
}

export interface TimeLockState {
  /** Playhead, epoch ms. Null means "no instant selected yet". */
  t: number | null;
  /** Transient hover instant. Never persisted; drives crosshairs only. */
  hoverT: number | null;
  selection: Selection;
  /** Which component last moved the playhead, so it can skip its own echo. */
  source: string | null;
  /** True in live mode: the playhead follows now instead of being scrubbed. */
  pinned: boolean;
}

const EMPTY_SELECTION: Selection = { kind: null, id: null, ticker: null };

const INITIAL: TimeLockState = {
  t: null,
  hoverT: null,
  selection: EMPTY_SELECTION,
  source: null,
  pinned: false,
};

class TimeLockStore {
  private state: TimeLockState = INITIAL;
  private listeners = new Set<() => void>();

  getState = (): TimeLockState => this.state;

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };

  private set(patch: Partial<TimeLockState>): void {
    const next = { ...this.state, ...patch };
    // Identity-stable no-op guard: pointer-move fires this constantly and an
    // unchanged snapshot must not wake every subscriber.
    if (
      next.t === this.state.t &&
      next.hoverT === this.state.hoverT &&
      next.pinned === this.state.pinned &&
      next.source === this.state.source &&
      next.selection.kind === this.state.selection.kind &&
      next.selection.id === this.state.selection.id &&
      next.selection.ticker === this.state.selection.ticker
    ) {
      return;
    }
    this.state = next;
    for (const fn of this.listeners) fn();
  }

  seek(t: number | null, source = "unknown"): void {
    this.set({ t, source, pinned: false });
  }

  hover(t: number | null): void {
    this.set({ hoverT: t });
  }

  select(sel: Partial<Selection>, t?: number | null, source = "unknown"): void {
    this.set({
      selection: { ...EMPTY_SELECTION, ...sel },
      ...(t === undefined ? {} : { t }),
      source,
    });
  }

  clearSelection(): void {
    this.set({ selection: EMPTY_SELECTION });
  }

  /** Live mode: follow now. Any manual seek unpins. */
  pin(t: number): void {
    this.set({ t, pinned: true, source: "live" });
  }

  reset(): void {
    this.state = INITIAL;
    for (const fn of this.listeners) fn();
  }
}

const stores = new Map<Scope, TimeLockStore>();

export function timeLock(scope: Scope): TimeLockStore {
  let s = stores.get(scope);
  if (!s) {
    s = new TimeLockStore();
    stores.set(scope, s);
  }
  return s;
}

/** Drop a scope's store. Call when a run detail screen unmounts for good. */
export function disposeTimeLock(scope: Scope): void {
  stores.delete(scope);
}

export interface TimeLockApi extends TimeLockState {
  seek: (t: number | null, source?: string) => void;
  hover: (t: number | null) => void;
  select: (sel: Partial<Selection>, t?: number | null, source?: string) => void;
  clearSelection: () => void;
  pin: (t: number) => void;
}

export function useTimeLock(scope: Scope): TimeLockApi {
  const store = useMemo(() => timeLock(scope), [scope]);
  const state = useSyncExternalStore(store.subscribe, store.getState, store.getState);

  const seek = useCallback((t: number | null, source?: string) => store.seek(t, source), [store]);
  const hover = useCallback((t: number | null) => store.hover(t), [store]);
  const select = useCallback(
    (sel: Partial<Selection>, t?: number | null, source?: string) => store.select(sel, t, source),
    [store],
  );
  const clearSelection = useCallback(() => store.clearSelection(), [store]);
  const pin = useCallback((t: number) => store.pin(t), [store]);

  return { ...state, seek, hover, select, clearSelection, pin };
}

// --- time helpers shared by every chart and table ----------------------------

export const toMs = (iso: string | number | Date): number =>
  typeof iso === "number" ? iso : new Date(iso).getTime();

/** lightweight-charts wants seconds, not milliseconds. One place to get it right. */
export const toChartTime = (iso: string | number | Date): number =>
  Math.floor(toMs(iso) / 1000);

/**
 * Index of the row whose timestamp is nearest `t`. Used to move a chart, a tape
 * and a ledger to "the same place" when their rows do not line up one-to-one.
 */
export function nearestIndex(times: number[], t: number | null): number {
  if (t === null || times.length === 0) return -1;
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) <= Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}
