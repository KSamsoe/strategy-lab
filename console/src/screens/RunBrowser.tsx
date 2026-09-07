/**
 * The registry, as a table you can interrogate.
 *
 * The column that carries the most weight is the one nobody asks for: attempts.
 * A Sharpe of 1.9 means something very different as the first thing you tried
 * than as the four-hundredth, and the registry is the only place that number
 * survives. So it is a first-class column, and it goes amber once a family has
 * been searched hard enough that the top of the distribution is mostly luck.
 */

import { useCallback, useMemo, useState } from "react";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api";
import type { Metrics, RunKind, RunOrigin, RunRecord } from "../lib/types";
import { count, day, pct, ratio, shortId, toneClass } from "../lib/format";
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
} from "../components/ui";

const KINDS: RunKind[] = ["backtest", "paper", "live", "sweep"];
const ORIGINS: RunOrigin[] = ["human", "agent-loop"];

/** One page is generous; past this the honest move is to narrow the filter. */
const PAGE = 500;

/**
 * Thresholds for the attempt counter. 25 is roughly where a 5% significance
 * test stops meaning anything at all; 100 is where the best result in the
 * family is, on priors, noise wearing a good number.
 */
const ATTEMPT_WARN = 25;
const ATTEMPT_LOUD = 100;

type SortKey = "created_at" | "strategy" | "kind" | "cagr" | "sharpe" | "dd" | "trades" | "attempts";
type Dir = "asc" | "desc";

/** Numeric columns want the biggest value first; label columns want A first. */
const DEFAULT_DIR: Record<SortKey, Dir> = {
  created_at: "desc",
  strategy: "asc",
  kind: "asc",
  cagr: "desc",
  sharpe: "desc",
  dd: "desc",
  trades: "desc",
  attempts: "desc",
};

const hasOos = (m: Metrics): boolean =>
  m.oos_sharpe !== undefined || m.oos_total_return !== undefined;

/** Drawdown is a loss whichever sign the engine stored it with. */
const asLoss = (v: number | undefined): number | null =>
  v === undefined || !Number.isFinite(v) ? null : -Math.abs(v);

function sortValue(r: RunRecord, key: SortKey, attempts: Map<string, number>): number | string {
  switch (key) {
    case "created_at":
      return r.created_at;
    case "strategy":
      return r.strategy;
    case "kind":
      return r.kind;
    case "cagr":
      return r.metrics.cagr ?? Number.NEGATIVE_INFINITY;
    case "sharpe":
      return r.metrics.sharpe ?? Number.NEGATIVE_INFINITY;
    case "dd":
      return Math.abs(r.metrics.max_drawdown ?? 0);
    case "trades":
      return r.metrics.trades ?? 0;
    case "attempts":
      return attempts.get(r.strategy) ?? 0;
  }
}

/**
 * Attempts per *strategy* family, not per config hash.
 *
 * Keying on config_hash would count every sweep cell once and report "1", which
 * is precisely the reassuring lie this column exists to prevent: the thing that
 * was searched 400 times is the strategy, across 400 configs.
 */
function attemptsByFamily(runs: RunRecord[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of runs) {
    // The registry's own ordinal is authoritative and survives pagination;
    // the visible row count is the floor when older records lack one.
    const ordinal = Number.isFinite(r.attempt) ? r.attempt : 0;
    m.set(r.strategy, Math.max(m.get(r.strategy) ?? 0, ordinal));
  }
  const seen = new Map<string, number>();
  for (const r of runs) seen.set(r.strategy, (seen.get(r.strategy) ?? 0) + 1);
  for (const [k, n] of seen) m.set(k, Math.max(m.get(k) ?? 0, n));
  return m;
}

function AttemptCell({ n, strategy }: { n: number; strategy: string }) {
  const loud = n >= ATTEMPT_LOUD;
  const warn = n >= ATTEMPT_WARN;
  return (
    <span
      title={`${count(n)} runs recorded for the ${strategy} family${
        warn ? " — treat the best result as a sample maximum, not a finding" : ""
      }`}
      className={cx(
        "mono",
        warn
          ? "text-amber border border-amber/40 bg-amber-wash rounded px-1 py-0.5"
          : "text-ash",
        loud && "font-semibold",
      )}
    >
      {count(n)}
    </span>
  );
}

export default function RunBrowser() {
  const search = useSearch({ from: "/runs" });
  const navigate = useNavigate({ from: "/runs" });
  const qc = useQueryClient();

  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [dir, setDir] = useState<Dir>("desc");
  const [selected, setSelected] = useState<string[]>([]);

  const filters = {
    strategy: search.strategy,
    kind: search.kind,
    origin: search.origin,
  };

  // The registry is not a live view and a finished run never changes, so this
  // does not poll; the refresh button is the explicit way to pick up new rows.
  const runsQ = useQuery({
    queryKey: ["runs", "list", filters],
    queryFn: () => api.runs({ ...filters, limit: PAGE }),
  });

  // Only for the filter datalist — a missing endpoint must not break the table.
  const strategiesQ = useQuery({
    queryKey: ["runs", "strategies"],
    queryFn: api.strategies,
    retry: false,
  });

  const runs = useMemo(() => runsQ.data?.runs ?? [], [runsQ.data]);
  const attempts = useMemo(() => attemptsByFamily(runs), [runs]);

  const rows = useMemo(() => {
    const out = [...runs];
    out.sort((a, b) => {
      const av = sortValue(a, sortKey, attempts);
      const bv = sortValue(b, sortKey, attempts);
      const c =
        typeof av === "string" || typeof bv === "string"
          ? String(av).localeCompare(String(bv))
          : av - bv;
      return dir === "asc" ? c : -c;
    });
    return out;
  }, [runs, sortKey, dir, attempts]);

  const setFilter = useCallback(
    (patch: Partial<typeof filters>) => {
      setSelected([]);
      void navigate({ search: (prev) => ({ ...prev, ...patch }), replace: true });
    },
    [navigate],
  );

  const toggle = useCallback((id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  const sortBy = useCallback(
    (key: SortKey) => {
      if (key === sortKey) setDir((d) => (d === "asc" ? "desc" : "asc"));
      else {
        setSortKey(key);
        setDir(DEFAULT_DIR[key]);
      }
    },
    [sortKey],
  );

  const sortedAs = (key: SortKey): Dir | null => (key === sortKey ? dir : null);

  const allShown = rows.length > 0 && selected.length === rows.length;
  const hot = [...attempts.values()].filter((n) => n >= ATTEMPT_WARN).length;
  const filtered = Boolean(filters.strategy || filters.kind || filters.origin);
  const total = runsQ.data?.total ?? rows.length;

  return (
    <div className="h-full min-h-0 p-3">
      <Panel
        className="h-full"
        bodyClassName="flex flex-col min-h-0"
        title={
          <span className="flex items-center gap-2">
            run registry
            <span className="mono text-ash-dim normal-case tracking-normal">
              {count(rows.length)}
              {total > rows.length ? ` of ${count(total)}` : ""}
            </span>
            {hot > 0 ? (
              <span className="text-amber normal-case tracking-normal">
                · {count(hot)} famil{hot === 1 ? "y" : "ies"} past {ATTEMPT_WARN} attempts
              </span>
            ) : null}
          </span>
        }
        right={
          <div className="flex items-center gap-2">
            <Button
              onClick={() => void qc.invalidateQueries({ queryKey: ["runs", "list"] })}
              title="Re-read the registry"
            >
              refresh
            </Button>
            {selected.length >= 2 ? (
              <Link
                to="/compare"
                search={{ ids: selected.join(",") }}
                className={cx(
                  "inline-flex items-center h-6 px-2 rounded border text-xs no-underline",
                  "border-teal/50 text-teal hover:bg-teal-wash",
                )}
              >
                compare {count(selected.length)} selected
              </Link>
            ) : (
              <Button disabled title="Tick two or more rows to compare them">
                compare selected
              </Button>
            )}
          </div>
        }
      >
        <div className="shrink-0 flex flex-wrap items-center gap-3 px-3 py-1.5 border-b border-slate">
          <label className="flex items-center gap-1.5">
            <span className="unit">strategy</span>
            <input
              value={filters.strategy ?? ""}
              onChange={(e) => setFilter({ strategy: e.target.value })}
              list="run-strategies"
              placeholder="all"
              aria-label="filter by strategy"
              className="mono h-6 w-40 bg-graphite border border-slate rounded px-1.5 text-xs text-chalk placeholder:text-ash-dim"
            />
          </label>
          <datalist id="run-strategies">
            {(strategiesQ.data?.strategies ?? []).map((s) => (
              <option key={s.name} value={s.name} />
            ))}
          </datalist>

          <label className="flex items-center gap-1.5">
            <span className="unit">kind</span>
            <select
              value={filters.kind ?? ""}
              onChange={(e) => setFilter({ kind: e.target.value })}
              aria-label="filter by kind"
              className="mono h-6 bg-graphite border border-slate rounded px-1 text-xs text-chalk"
            >
              <option value="">all</option>
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-1.5">
            <span className="unit">origin</span>
            <select
              value={filters.origin ?? ""}
              onChange={(e) => setFilter({ origin: e.target.value })}
              aria-label="filter by origin"
              className="mono h-6 bg-graphite border border-slate rounded px-1 text-xs text-chalk"
            >
              <option value="">all</option>
              {ORIGINS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>

          {filtered ? (
            <Button
              onClick={() => setFilter({ strategy: "", kind: "", origin: "" })}
              title="Clear every filter"
            >
              clear
            </Button>
          ) : null}

          {total > rows.length ? (
            <span className="unit text-amber">
              showing first {count(PAGE)} — narrow the filter to see the rest
            </span>
          ) : null}
        </div>

        <div className="min-h-0 flex-1">
          {runsQ.isPending ? (
            <Loading what="reading registry" />
          ) : runsQ.error ? (
            <ErrorNote error={runsQ.error} />
          ) : rows.length === 0 ? (
            <Empty cmd="lab backtest strategies/momo.py">
              {filtered ? "no runs match these filters" : "the registry is empty"}
            </Empty>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th className="w-6">
                    <input
                      type="checkbox"
                      checked={allShown}
                      onChange={() => setSelected(allShown ? [] : rows.map((r) => r.run_id))}
                      aria-label="select all runs"
                      className="accent-teal align-middle"
                    />
                  </Th>
                  <Th>run</Th>
                  <Th onClick={() => sortBy("strategy")} sorted={sortedAs("strategy")}>
                    strategy
                  </Th>
                  <Th onClick={() => sortBy("kind")} sorted={sortedAs("kind")}>
                    kind
                  </Th>
                  <Th>origin</Th>
                  <Th onClick={() => sortBy("created_at")} sorted={sortedAs("created_at")}>
                    range
                  </Th>
                  <Th align="right" onClick={() => sortBy("cagr")} sorted={sortedAs("cagr")}>
                    cagr
                  </Th>
                  <Th align="right" onClick={() => sortBy("sharpe")} sorted={sortedAs("sharpe")}>
                    sharpe
                  </Th>
                  <Th align="right" onClick={() => sortBy("dd")} sorted={sortedAs("dd")}>
                    max dd
                  </Th>
                  <Th align="right" onClick={() => sortBy("trades")} sorted={sortedAs("trades")}>
                    trades
                  </Th>
                  <Th align="center">sample</Th>
                  <Th
                    align="right"
                    onClick={() => sortBy("attempts")}
                    sorted={sortedAs("attempts")}
                    className="text-amber"
                  >
                    attempts
                  </Th>
                  <Th>status</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const m = r.metrics;
                  const dd = asLoss(m.max_drawdown);
                  const n = attempts.get(r.strategy) ?? 0;
                  const isSelected = selected.includes(r.run_id);
                  return (
                    <Tr
                      key={r.run_id}
                      selected={isSelected}
                      onClick={() => void navigate({ to: "/runs/$runId", params: { runId: r.run_id } })}
                    >
                      <Td>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onClick={(e) => e.stopPropagation()}
                          onChange={() => toggle(r.run_id)}
                          aria-label={`select ${r.run_id}`}
                          className="accent-teal align-middle"
                        />
                      </Td>
                      <Td mono title={r.run_id}>
                        {/* A link, not just a row handler: the row has to be
                            reachable and openable from the keyboard. */}
                        <Link
                          to="/runs/$runId"
                          params={{ runId: r.run_id }}
                          onClick={(e) => e.stopPropagation()}
                          className="text-teal no-underline hover:underline"
                        >
                          {shortId(r.run_id)}
                        </Link>
                      </Td>
                      <Td mono className="text-chalk">
                        {r.strategy}
                      </Td>
                      <Td>
                        <Badge>{r.kind}</Badge>
                      </Td>
                      <Td>
                        {/* Deliberately uncoloured: origin is not P&L, warning
                            or selection, and the palette is only for those. */}
                        <Badge title={r.origin === "agent-loop" ? "authored by an agent loop" : "authored by hand"}>
                          {r.origin}
                        </Badge>
                      </Td>
                      <Td mono className="text-ash">
                        {day(r.start)} → {day(r.end)}
                      </Td>
                      <Td align="right" mono className={toneClass(m.cagr)}>
                        {pct(m.cagr)}
                      </Td>
                      <Td align="right" mono>
                        {ratio(m.sharpe)}
                      </Td>
                      <Td align="right" mono className={toneClass(dd)}>
                        {pct(dd)}
                      </Td>
                      <Td align="right" mono>
                        {count(m.trades)}
                      </Td>
                      <Td align="center">
                        {hasOos(m) ? (
                          <Badge kind="good" title="walk-forward run with out-of-sample metrics">
                            oos
                          </Badge>
                        ) : (
                          <Badge title="in-sample only — fitted and measured on the same data">
                            is
                          </Badge>
                        )}
                      </Td>
                      <Td align="right">
                        <AttemptCell n={n} strategy={r.strategy} />
                      </Td>
                      <Td>
                        <StatusDot
                          health={r.status === "ok" ? "ok" : r.status === "error" ? "bad" : "idle"}
                          label={r.status}
                          detail={r.error || undefined}
                        />
                      </Td>
                    </Tr>
                  );
                })}
              </tbody>
            </Table>
          )}
        </div>
      </Panel>
    </div>
  );
}
