/**
 * Run detail — the forensics screen.
 *
 * Everything here hangs off one idea: the four panels below the header are not
 * four views of a run, they are one view of one instant. The time lock is the
 * instant. Click a trade and the equity crosshair, the price chart and the tape
 * all move to it; scrub the tape and the ledger row under the playhead lights
 * up. That is the whole reason to open this screen instead of the static HTML
 * report, so the wiring is explicit and deliberate rather than emergent:
 *
 *   ledger row  → select({kind:"trade", …, ticker}) → this screen switches the
 *                 price chart to that ticker, charts and tape follow `t`
 *   tape row    → select({kind:"decision", …})      → if the decision names one
 *                 ticker, the price chart switches to it too
 *   chart click → seek(t)                            → ledger and tape follow
 *
 * The other load-bearing thing is the two loud claims. `optimistic_fills` and
 * `contaminated` are not caveats, they are the difference between a result and
 * a number, so they render as banners above the metrics rather than as a line
 * in the provenance footer that nobody reaches.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import { DecisionTape } from "../components/DecisionTape";
import { EquityChart } from "../components/charts/EquityChart";
import { PriceChart } from "../components/charts/PriceChart";
import { TradeLedger } from "../components/TradeLedger";
import { VsBenchmark } from "../components/VsBenchmark";
import type { Metrics, RunRecord, TradeRow } from "../lib/types";
import { useTimeLock } from "../lib/timelock";
import {
  MISSING,
  count,
  day,
  money,
  pct,
  ratio,
  shortId,
  stamp,
  titleize,
} from "../lib/format";
import {
  Badge,
  Empty,
  ErrorNote,
  Loading,
  LoudWarning,
  Panel,
  Stat,
  cx,
} from "../components/ui";

/** The API's own ceiling (`le=2000` on the decisions route). Each row carries
 *  its full input tape, so a page is megabytes, not kilobytes -- asking for more
 *  than the route allows just earns a 422. Anything past this is reported as
 *  "first N of M" rather than silently dropped. */
const DECISION_LIMIT = 2000;

export function hasOos(m: Metrics): boolean {
  return m.oos_sharpe !== undefined || m.oos_total_return !== undefined;
}

/** Drawdown reads as a loss whichever sign the engine happened to store. */
export function asLoss(v: number | undefined): number | null {
  return v === undefined || !Number.isFinite(v) ? null : -Math.abs(v);
}

/** Tickers the run actually traded, most-traded first: the selector's order is
 *  a claim about where to look, so it should follow activity, not the alphabet. */
export function tickersByActivity(trades: readonly TradeRow[]): string[] {
  const n = new Map<string, number>();
  for (const t of trades) n.set(t.ticker, (n.get(t.ticker) ?? 0) + 1);
  return [...n.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([k]) => k);
}

export default function RunDetail() {
  const { runId } = useParams({ from: "/runs/$runId" });
  return <RunDetailView runId={runId} />;
}

export function RunDetailView({ runId }: { runId: string }) {
  // A finished run is an immutable artifact. Nothing here polls, and nothing
  // here needs to: if the numbers changed, the run id would have.
  const immutable = { staleTime: Number.POSITIVE_INFINITY, retry: false } as const;

  const runQ = useQuery({ queryKey: ["runs", runId], queryFn: () => api.run(runId), ...immutable });
  const equityQ = useQuery({
    queryKey: ["runs", runId, "equity"],
    queryFn: () => api.equity(runId),
    ...immutable,
  });
  const tradesQ = useQuery({
    queryKey: ["runs", runId, "trades"],
    queryFn: () => api.trades(runId),
    ...immutable,
  });
  const decisionsQ = useQuery({
    queryKey: ["runs", runId, "decisions", DECISION_LIMIT],
    queryFn: () => api.decisions(runId, { limit: DECISION_LIMIT }),
    ...immutable,
  });

  const { selection } = useTimeLock(runId);

  const trades = useMemo(() => tradesQ.data?.trades ?? [], [tradesQ.data]);
  const decisions = useMemo(() => decisionsQ.data?.decisions ?? [], [decisionsQ.data]);
  const tickers = useMemo(() => tickersByActivity(trades), [trades]);

  const [ticker, setTicker] = useState<string | null>(null);

  // The lock is the authority on "what are we looking at". A trade selection
  // names its ticker outright; a decision selection names one only when the
  // decision itself was about exactly one, because switching the chart on a
  // three-ticker rebalance would be an arbitrary pick dressed as an answer.
  useEffect(() => {
    if (selection.ticker) {
      setTicker(selection.ticker);
      return;
    }
    if (selection.kind !== "decision" || !selection.id) return;
    const d = decisions.find((x) => x.id === selection.id);
    if (!d) return;
    const named = [...new Set(d.intents.map((i) => i.ticker))];
    if (named.length === 1) setTicker(named[0]);
  }, [selection.kind, selection.id, selection.ticker, decisions]);

  // Default to the busiest ticker once the ledger lands, and never leave a
  // stale name selected if the ledger changes underneath.
  useEffect(() => {
    setTicker((cur) => (cur && tickers.includes(cur) ? cur : (tickers[0] ?? null)));
  }, [tickers]);

  // NOTE: no `disposeTimeLock` on unmount. Under StrictMode the mount/unmount/
  // remount cycle would drop the store while a live component still holds a
  // reference to it, and two halves of this screen would then be locked to
  // different instants. A handful of small stores is the cheaper bug.

  const run = runQ.data ?? null;
  const metrics: Metrics = run?.metrics ?? {};
  const showAgent = decisions.some((d) => d.agent !== null);

  if (runQ.isPending) return <Loading what="reading run" />;
  if (runQ.error) {
    return (
      <div className="p-3">
        <Panel title="run">
          <ErrorNote error={runQ.error} />
          <Empty cmd="lab runs --json">no run with id {shortId(runId, 12)}</Empty>
        </Panel>
      </div>
    );
  }

  return (
    <div className="p-3 flex flex-col gap-3">
      <RunHeader runId={runId} run={run} />

      {metrics.optimistic_fills ? (
        <LoudWarning>
          <strong className="text-chalk">Same-bar-close fills.</strong> Orders in this run
          executed at the close of the bar the decision was made on, so every trade got a
          price its own signal helped set. The edge below is flattered by an amount this
          backtest cannot measure — re-run with a next-open fill model before believing any
          of these numbers.{" "}
          <span className="mono text-ash">fill_mode = {metrics.fill_mode ?? "same_bar_close"}</span>
        </LoudWarning>
      ) : null}

      {metrics.contaminated ? (
        <LoudWarning>
          <strong className="text-chalk">Training-window contamination.</strong> An LLM
          strategy was backtested over a period inside its own training data. The model may
          be recalling what happened rather than inferring it, which means this run is a
          plumbing smoke test — evidence that the wiring works, not that the strategy does.
          Score it on a window that post-dates the model cutoff.
        </LoudWarning>
      ) : null}

      {run?.stale_ledger ? (
        <LoudWarning>
          <strong className="text-chalk">Trade stats undercount.</strong> This run predates
          the trade-ledger fix and finished holding open positions, so only round trips that
          returned exactly to flat were counted — a trim that took profit and left the
          position open produced no trade row. Trade count, hit rate, profit factor and
          avg win/loss are all computed off that fraction. The equity curve, return, Sharpe
          and drawdown are unaffected. Re-run it to get the full ledger:{" "}
          <code className="mono text-teal">lab runs rerun {runId}</code>
        </LoudWarning>
      ) : null}

      {typeof run?.ledger_residual === "number" && Math.abs(run.ledger_residual) > 0 ? (
        <LoudWarning>
          <strong className="text-chalk">Ledger does not explain the curve.</strong>{" "}
          {money(run.ledger_residual)} of P&amp;L has no trade behind it. Trade-level metrics
          describe only the part that does.
        </LoudWarning>
      ) : null}

      <p className="text-micro text-ash-dim m-0 px-1 uppercase tracking-wider">
        figures below cover the FULL backtest period. A research session ranks on the
        out-of-sample slice only, so its table will show smaller numbers than these.
      </p>

      <MetricTiles metrics={metrics} />

      <VsBenchmark
        metrics={metrics}
        benchmark={(run?.config?.benchmark as string | undefined) ?? null}
      />

      <Panel
        title={
          <span className="flex items-center gap-2">
            equity and drawdown
            <span className="normal-case tracking-normal text-ash-dim">
              out-of-sample spans shaded
            </span>
          </span>
        }
      >
        {equityQ.isPending ? (
          <Loading what="loading equity" />
        ) : equityQ.error ? (
          <ErrorNote error={equityQ.error} />
        ) : !equityQ.data?.equity?.length ? (
          <Empty cmd={`lab backtest --run ${shortId(runId, 12)} --json`}>
            no equity series was written for this run
          </Empty>
        ) : (
          <div data-chart="equity">
            <EquityChart scope={runId} data={equityQ.data} height={240} showDrawdown />
          </div>
        )}
      </Panel>

      <Panel
        title="price with trade markers"
        right={
          tickers.length > 0 ? (
            <label className="flex items-center gap-1.5">
              <span className="unit">ticker</span>
              <select
                value={ticker ?? ""}
                onChange={(e) => setTicker(e.target.value)}
                aria-label="ticker for the price chart"
                className="mono h-5 rounded border border-slate bg-graphite px-1 text-xs text-chalk"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
          ) : null
        }
      >
        {tradesQ.isPending ? (
          <Loading what="loading trades" />
        ) : ticker === null ? (
          <Empty cmd={`lab backtest --run ${shortId(runId, 12)} --json`}>
            this run took no trades, so there is nothing to mark on a price chart
          </Empty>
        ) : (
          <div data-chart="price" data-ticker={ticker}>
            <PriceChart
              scope={runId}
              runId={runId}
              ticker={ticker}
              trades={trades.filter((t) => t.ticker === ticker)}
              height={260}
            />
          </div>
        )}
      </Panel>

      <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(420px,1fr))]">
        <Panel
          title="trade ledger"
          right={<span className="unit">{count(trades.length)} trades</span>}
          bodyClassName="max-h-[420px] overflow-y-auto"
        >
          <TradeLedger
            scope={runId}
            trades={trades}
            loading={tradesQ.isPending}
            error={tradesQ.error}
            onSelectTicker={setTicker}
          />
        </Panel>

        <Panel
          title={
            <span className="flex items-center gap-2">
              decision tape
              {decisionsQ.data && decisionsQ.data.total > decisions.length ? (
                <span className="text-amber normal-case tracking-normal">
                  first {count(decisions.length)} of {count(decisionsQ.data.total)}
                </span>
              ) : null}
            </span>
          }
          bodyClassName="h-[420px]"
        >
          <DecisionTape
            scope={runId}
            decisions={decisions}
            loading={decisionsQ.isPending}
            error={decisionsQ.error}
            showAgent={showAgent}
            emptyCmd="lab backtest strategies/momo.py --journal"
          />
        </Panel>
      </div>

      <Provenance runId={runId} run={run} metrics={metrics} />
    </div>
  );
}

// --- header -------------------------------------------------------------------

function RunHeader({ runId, run }: { runId: string; run: RunRecord | null }) {
  const oos = run ? hasOos(run.metrics) : false;
  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 bg-gunmetal border border-slate rounded-md">
      <span className="flex items-baseline gap-2">
        <span className="unit">run</span>
        <span className="mono text-lg text-chalk" title={runId}>
          {shortId(runId, 12)}
        </span>
      </span>

      <span className="mono text-chalk">{run?.strategy ?? MISSING}</span>
      {run ? <Badge>{run.kind}</Badge> : null}
      {run ? <Badge title={`authored by ${run.origin}`}>{run.origin}</Badge> : null}

      <span className="mono text-ash">
        {day(run?.start)} → {day(run?.end)}
      </span>

      <span className="flex items-center gap-1.5" title="git commit of the strategy tree">
        <span className="unit">git</span>
        <span className="mono text-ash">{shortId(run?.git_commit, 7)}</span>
      </span>
      <span className="flex items-center gap-1.5" title="hash of the resolved config">
        <span className="unit">config</span>
        <span className="mono text-ash">{shortId(run?.config_hash, 8)}</span>
      </span>
      <span className="flex items-center gap-1.5" title="dataset version this run read">
        <span className="unit">data</span>
        <span className="mono text-ash">{run?.data_version || MISSING}</span>
      </span>

      {run?.sweep_id ? (
        <Link
          to="/sweeps/$sweepId"
          params={{ sweepId: run.sweep_id }}
          className="mono text-xs text-teal no-underline hover:underline"
          title="this run is one cell of a parameter sweep"
        >
          sweep {shortId(run.sweep_id, 8)}
        </Link>
      ) : null}

      <span className="flex-1" />

      {oos ? (
        <Badge kind="good" title="walk-forward run: out-of-sample metrics were recorded">
          oos ✓
        </Badge>
      ) : (
        <Badge kind="warn" title="in-sample only — fitted and measured on the same data">
          in-sample only
        </Badge>
      )}

      {run?.status === "error" ? (
        <Badge kind="bad" title={run.error}>
          errored
        </Badge>
      ) : null}
    </header>
  );
}

// --- metrics ------------------------------------------------------------------

function MetricTiles({ metrics }: { metrics: Metrics }) {
  const dd = asLoss(metrics.max_drawdown);
  return (
    <div className="grid gap-px bg-slate/60 border border-slate rounded-md overflow-hidden [grid-template-columns:repeat(auto-fit,minmax(140px,1fr))]">
      <div className="bg-gunmetal">
        <Stat label="cagr" value={pct(metrics.cagr)} tone={metrics.cagr ?? null} />
      </div>
      <div className="bg-gunmetal">
        <Stat label="total return" value={pct(metrics.total_return)} tone={metrics.total_return ?? null} />
      </div>
      <div className="bg-gunmetal">
        <Stat label="sharpe" value={ratio(metrics.sharpe)} sub={oosSub(metrics)} />
      </div>
      <div className="bg-gunmetal">
        <Stat label="max dd" value={pct(dd)} tone={dd} sub={ddSub(metrics)} />
      </div>
      <div className="bg-gunmetal">
        <Stat label="trades" value={count(metrics.trades)} sub={hitSub(metrics)} />
      </div>
      <div className="bg-gunmetal">
        <Stat label="exposure" value={pct(metrics.exposure, 0)} sub={`turnover ${ratio(metrics.turnover)}`} />
      </div>
      <div className="bg-gunmetal">
        <Stat
          label="final equity"
          value={money(metrics.final_equity)}
          sub={`from ${money(metrics.starting_equity)}`}
        />
      </div>
    </div>
  );
}

/** The out-of-sample number belongs next to the in-sample one it contradicts. */
function oosSub(m: Metrics): string {
  if (m.oos_sharpe === undefined) return "in-sample";
  return `oos ${ratio(m.oos_sharpe)}`;
}

function ddSub(m: Metrics): string {
  const d = m.max_drawdown_duration_days;
  return d === undefined ? "" : `${count(d)}d underwater`;
}

function hitSub(m: Metrics): string {
  if (m.hit_rate === undefined) return "";
  return `${pct(m.hit_rate, 0)} hit · pf ${ratio(m.profit_factor)}`;
}

// --- provenance ---------------------------------------------------------------

function Provenance({
  runId,
  run,
  metrics,
}: {
  runId: string;
  run: RunRecord | null;
  metrics: Metrics;
}) {
  const params = Object.entries(run?.params ?? {});
  const warnings = metrics.warnings ?? [];

  return (
    <Panel
      title="provenance"
      right={
        // The static report is the archival artifact: portable, emailable, and
        // still readable when this console is not running.
        <a
          href={`/runs/${encodeURIComponent(runId)}/report.html`}
          target="_blank"
          rel="noreferrer"
          title={`the archival static report at runs/${runId}/report.html`}
          className="mono text-xs text-teal no-underline hover:underline"
        >
          static report ↗
        </a>
      }
    >
      <div className="grid gap-x-6 gap-y-1 px-3 py-2 text-xs [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
        <Field label="run id" value={runId} />
        <Field label="git commit" value={run?.git_commit ?? MISSING} />
        <Field label="config hash" value={run?.config_hash ?? MISSING} />
        <Field label="data version" value={run?.data_version || MISSING} />
        <Field label="fill model" value={metrics.fill_mode ?? "unstated"} />
        <Field label="started" value={stamp(run?.created_at)} />
        <Field label="finished" value={stamp(run?.finished_at)} />
        <Field label="attempt" value={count(run?.attempt)} />
        <Field label="parent run" value={run?.parent_run_id ?? MISSING} />
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
      ) : null}

      {warnings.length > 0 ? (
        <ul className="px-3 pb-2 text-xs text-amber list-disc list-inside">
          {warnings.map((w) => (
            <li key={w}>{titleize(w)}</li>
          ))}
        </ul>
      ) : null}

      {/*
        Always present, never conditional. A universe assembled from today's
        listings has already dropped every company that failed, and no flag in
        the metrics can tell you whether this particular run got away with it.
      */}
      <p className="px-3 py-2 border-t border-slate text-xs text-ash-dim">
        <span className="text-ash">Survivorship:</span> this run traded the tickers the
        universe file names today. Companies delisted, acquired or bankrupted inside the
        window are absent unless the dataset is explicitly point-in-time, and their absence
        makes every return above look better than it was.
      </p>
    </Panel>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <span className={cx("flex items-baseline gap-2 min-w-0")}>
      <span className="unit shrink-0">{label}</span>
      <span className="mono text-ash truncate" title={value}>
        {value}
      </span>
    </span>
  );
}
