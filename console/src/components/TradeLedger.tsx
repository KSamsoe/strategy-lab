/**
 * The trade ledger.
 *
 * Every row is one round trip the run actually took, and clicking it moves the
 * whole screen: the price chart pans to that entry, the decision tape scrolls
 * to the bar that produced it. That hand-off is the component's job -- it is
 * the shortest path from "this trade looks wrong" to the input tape that caused
 * it.
 *
 * Two details are deliberate. Sorting keeps nulls last in *both* directions,
 * because an open trade has no exit price and letting `null` win "highest exit"
 * would invent a fill that never happened. And the row cap is stated on screen:
 * a ledger that silently drops rows past a thousand is a ledger you cannot
 * reconcile against the broker.
 */

import { useCallback, useMemo, useState, useSyncExternalStore } from "react";

import type { TradeRow } from "../lib/types";
import { clock, count, day, delta, money, pct, shares, signedMoney, toneClass } from "../lib/format";
import { timeLock, toMs } from "../lib/timelock";
import { Badge, Empty, ErrorNote, Loading, Table, Td, Th, Tr, cx } from "./ui";

/** Our own name on the bus, so a selection we emitted is not echoed back. */
const SOURCE = "trade-ledger";

/** Past this the DOM stops being worth it. Never silent -- see the note strip. */
export const ROW_CAP = 1000;

export type TradeSortKey =
  | "ticker"
  | "side"
  | "qty"
  | "entry_time"
  | "entry_price"
  | "exit_time"
  | "exit_price"
  | "pnl"
  | "pnl_pct"
  | "bars_held"
  | "exit_reason";

export type SortDir = "asc" | "desc";

/** A trade plus its position in the source array, which is its stable id. */
export interface LedgerRow {
  id: string;
  i: number;
  trade: TradeRow;
}

/**
 * `TradeRow` carries no id, and the ledger sorts, so a row's identity cannot be
 * its rendered position. Source index plus ticker plus entry survives sorting
 * and stays readable in a selection payload.
 */
export function tradeId(t: TradeRow, i: number): string {
  return `${i}:${t.ticker}:${t.entry_time}`;
}

export function ledgerRows(trades: readonly TradeRow[]): LedgerRow[] {
  return trades.map((trade, i) => ({ id: tradeId(trade, i), i, trade }));
}

/** Null for anything missing, so the comparator can push it to the bottom. */
function keyValue(t: TradeRow, key: TradeSortKey): string | number | null {
  switch (key) {
    case "ticker":
      return t.ticker || null;
    case "side":
      return t.side || null;
    case "exit_reason":
      return t.exit_reason || null;
    case "entry_time":
      return t.entry_time ? toMs(t.entry_time) : null;
    case "exit_time":
      return t.exit_time ? toMs(t.exit_time) : null;
    default: {
      const v = t[key];
      return typeof v === "number" && Number.isFinite(v) ? v : null;
    }
  }
}

/**
 * Sort, stably, with missing values pinned to the bottom in both directions.
 *
 * Stability matters more than it looks: sorting by ticker has to leave a
 * symbol's trades in chronological order, or the ledger stops reading as a
 * history of that position.
 */
export function sortTrades(
  rows: readonly LedgerRow[],
  key: TradeSortKey,
  dir: SortDir,
): LedgerRow[] {
  return [...rows].sort((a, b) => {
    const av = keyValue(a.trade, key);
    const bv = keyValue(b.trade, key);
    if (av === null && bv === null) return a.i - b.i;
    if (av === null) return 1;
    if (bv === null) return -1;
    const d =
      typeof av === "string" && typeof bv === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
    if (d === 0) return a.i - b.i;
    return dir === "asc" ? d : -d;
  });
}

export interface LedgerTotals {
  n: number;
  closed: number;
  wins: number;
  hitRate: number | null;
  pnl: number;
}

/** Footer arithmetic, over every trade -- not just the rows that fit. */
export function ledgerTotals(trades: readonly TradeRow[]): LedgerTotals {
  let closed = 0;
  let wins = 0;
  let pnl = 0;
  for (const t of trades) {
    if (Number.isFinite(t.pnl)) pnl += t.pnl;
    if (t.exit_time) {
      closed++;
      if (t.pnl > 0) wins++;
    }
  }
  return { n: trades.length, closed, wins, hitRate: closed ? wins / closed : null, pnl };
}

interface Col {
  key: TradeSortKey;
  label: string;
  align?: "left" | "right";
}

const COLS: readonly Col[] = [
  { key: "ticker", label: "ticker" },
  { key: "side", label: "side" },
  { key: "qty", label: "qty", align: "right" },
  { key: "entry_time", label: "entry" },
  { key: "entry_price", label: "entry px", align: "right" },
  { key: "exit_time", label: "exit" },
  { key: "exit_price", label: "exit px", align: "right" },
  { key: "pnl", label: "p&l", align: "right" },
  { key: "pnl_pct", label: "p&l %", align: "right" },
  { key: "bars_held", label: "bars", align: "right" },
  { key: "exit_reason", label: "exit reason" },
];

export interface TradeLedgerProps {
  scope: string;
  trades: TradeRow[];
  loading?: boolean;
  error?: unknown;
  onSelectTicker?: (ticker: string) => void;
}

export function TradeLedger({ scope, trades, loading, error, onSelectTicker }: TradeLedgerProps) {
  const [key, setKey] = useState<TradeSortKey>("entry_time");
  const [dir, setDir] = useState<SortDir>("asc");

  /**
   * Primitive slices of the lock rather than `useTimeLock`. The charts emit
   * `hover` on every pointer move; a thousand-row table must not re-render for
   * a crosshair it does not draw.
   */
  const store = useMemo(() => timeLock(scope), [scope]);
  const t = useSyncExternalStore(store.subscribe, () => store.getState().t);
  const selection = useSyncExternalStore(store.subscribe, () => store.getState().selection);

  const rows = useMemo(() => ledgerRows(trades), [trades]);
  const sorted = useMemo(() => sortTrades(rows, key, dir), [rows, key, dir]);
  const shown = sorted.length > ROW_CAP ? sorted.slice(0, ROW_CAP) : sorted;
  const totals = useMemo(() => ledgerTotals(trades), [trades]);

  const onHeader = useCallback(
    (k: TradeSortKey) => {
      if (k === key) setDir((d) => (d === "asc" ? "desc" : "asc"));
      else {
        setKey(k);
        // Money and time want their most interesting end first: biggest P&L,
        // most recent trade. Names read better forwards.
        setDir(k === "ticker" || k === "side" || k === "exit_reason" ? "asc" : "desc");
      }
    },
    [key],
  );

  const onRow = useCallback(
    (row: LedgerRow) => {
      store.select(
        { kind: "trade", id: row.id, ticker: row.trade.ticker },
        toMs(row.trade.entry_time),
        SOURCE,
      );
      onSelectTicker?.(row.trade.ticker);
    },
    [store, onSelectTicker],
  );

  const activeId = selection.kind === "trade" ? selection.id : null;

  if (error) return <ErrorNote error={error} />;
  if (loading) return <Loading what="reading trades" />;
  if (trades.length === 0) {
    return (
      <Empty cmd="lab backtest strategies/momo.py --json">
        this run closed no trades — the strategy either never fired or the gate blocked everything
      </Empty>
    );
  }

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="flex items-center gap-3 px-2 h-6 shrink-0 border-b border-slate">
        <span className="text-micro uppercase tracking-wider text-ash">
          <span className="num text-chalk">{count(totals.n)}</span> trades
        </span>
        <span className="text-micro uppercase tracking-wider text-ash">
          hit rate <span className="num text-chalk">{pct(totals.hitRate, 0)}</span>
        </span>
        <span className="text-micro uppercase tracking-wider text-ash">
          net <span className={cx("num", toneClass(totals.pnl))}>{signedMoney(totals.pnl, 0)}</span>
        </span>
        <span className="flex-1" />
        {sorted.length > ROW_CAP ? (
          <Badge kind="warn" title="the table is capped, not filtered — sort to bring rows into view">
            showing {count(ROW_CAP)} of {count(sorted.length)}
          </Badge>
        ) : null}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <Table>
          <thead>
            <tr>
              {COLS.map((c) => (
                <Th
                  key={c.key}
                  align={c.align}
                  onClick={() => onHeader(c.key)}
                  sorted={key === c.key ? dir : null}
                >
                  {c.label}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row) => {
              const tr = row.trade;
              const entryMs = tr.entry_time ? toMs(tr.entry_time) : null;
              const exitMs = tr.exit_time ? toMs(tr.exit_time) : null;
              // The playhead sitting inside a trade's life is worth marking:
              // it is the answer to "which position was open at that instant".
              const spans =
                t !== null && entryMs !== null && t >= entryMs && (exitMs === null || t <= exitMs);
              return (
                <Tr
                  key={row.id}
                  selected={row.id === activeId}
                  onClick={() => onRow(row)}
                  className={cx(spans && row.id !== activeId && "bg-slate/25")}
                >
                  <Td mono>
                    {/* Focusable so the ledger is reachable without a mouse.
                        The click stops here: letting it bubble to the row would
                        run the selection twice and fire `onSelectTicker` twice
                        with it. */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRow(row);
                      }}
                      className="mono text-chalk hover:text-teal"
                      aria-label={`trade ${tr.ticker} entered ${tr.entry_time}`}
                    >
                      {tr.ticker}
                    </button>
                  </Td>
                  <Td className={tr.side === "sell" || tr.side === "short" ? "text-ash" : "text-chalk"}>
                    {tr.side}
                  </Td>
                  <Td align="right" mono>
                    {shares(tr.qty)}
                  </Td>
                  <Td mono className="text-ash" title={tr.entry_time}>
                    {day(tr.entry_time)} <span className="text-ash-dim">{clock(tr.entry_time)}</span>
                  </Td>
                  <Td align="right" mono>
                    {money(tr.entry_price, 2)}
                  </Td>
                  <Td mono className="text-ash" title={tr.exit_time ?? undefined}>
                    {tr.exit_time ? (
                      <>
                        {day(tr.exit_time)} <span className="text-ash-dim">{clock(tr.exit_time)}</span>
                      </>
                    ) : (
                      <Badge kind="accent">open</Badge>
                    )}
                  </Td>
                  <Td align="right" mono>
                    {money(tr.exit_price, 2)}
                  </Td>
                  <Td align="right" mono className={toneClass(tr.pnl)}>
                    {signedMoney(tr.pnl, 0)}
                  </Td>
                  <Td align="right" mono className={toneClass(tr.pnl_pct)}>
                    {delta(tr.pnl_pct, 2)}
                  </Td>
                  <Td align="right" mono className="text-ash">
                    {count(tr.bars_held)}
                  </Td>
                  <Td className="text-ash-dim">{tr.exit_reason || null}</Td>
                </Tr>
              );
            })}
          </tbody>
        </Table>
      </div>
    </div>
  );
}
