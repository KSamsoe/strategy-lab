/**
 * Watching an open-ended research session.
 *
 * The session file is rewritten after every turn, so this is readable *while*
 * the agent works — which is the whole point. It polls rather than tailing the
 * WebSocket because the interesting object is the session's accumulated state
 * (experiments, best-so-far, the clock), not the event stream; the global event
 * tape on the fleet screen already carries the live line-by-line.
 *
 * The reasoning trail is the reason to open this instead of reading the run
 * list: a research session's value is as much in "why did it abandon momentum"
 * as in which run scored highest.
 */

import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import { StartResearch, StopSessionButton } from "./StartResearch";
import type { ResearchDetail, ResearchSummary } from "../lib/types";
import { MISSING, count, duration, money, pct, ratio, shortId, stamp } from "../lib/format";
import {
  Badge,
  Empty,
  ErrorNote,
  Loading,
  LoudWarning,
  Panel,
  Stat,
  Table,
  Td,
  Th,
  Tr,
  cx,
} from "./ui";

/** A running session is polled; a finished one is immutable. */
const LIVE_MS = 4000;

function statusKind(s: ResearchSummary): "good" | "warn" | "accent" | "neutral" {
  if (s.status === "running") return "accent";
  if (s.stopped_because === "satisfied") return "good";
  if (s.stopped_because === "rate_limit" || s.stopped_because === "budget") return "warn";
  return "neutral";
}

export function ResearchView() {
  const [selected, setSelected] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["agent", "research"],
    queryFn: api.research,
    refetchInterval: LIVE_MS,
    retry: false,
  });

  const sessions = list.data?.sessions ?? [];
  const sessionId = selected ?? sessions[0]?.session_id ?? null;

  if (list.isLoading) return <Loading what="research sessions" />;
  if (list.error) return <ErrorNote error={list.error} />;
  if (!sessions.length) {
    return (
      <div className="flex flex-col gap-3 p-3">
        <StartResearch onStarted={setSelected} />
        <Empty cmd='lab agent research -c cfg/momo.yaml --brief "beat buy-and-hold"'>
          no research sessions yet
        </Empty>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <StartResearch onStarted={setSelected} />
      <Panel title="sessions" right={<span className="unit">{count(sessions.length)}</span>}>
        <Table>
          <thead>
            <tr>
              <Th>session</Th>
              <Th>status</Th>
              <Th align="right">experiments</Th>
              <Th align="right">calls</Th>
              <Th align="right">elapsed</Th>
              <Th align="right">spend</Th>
              <Th align="right">best</Th>
              <Th>brief</Th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <Tr
                key={s.session_id}
                selected={s.session_id === sessionId}
                onClick={() => setSelected(s.session_id)}
              >
                <Td mono>{shortId(s.session_id, 24)}</Td>
                <Td>
                  <Badge kind={statusKind(s)}>
                    {s.status === "running" ? "running" : s.stopped_because || s.status}
                  </Badge>
                </Td>
                <Td align="right" mono>
                  {count(s.experiments)}
                </Td>
                <Td align="right" mono>
                  {count(s.calls)}
                </Td>
                <Td align="right" mono>
                  {duration(s.elapsed_minutes * 60)}
                </Td>
                <Td align="right" mono title={s.billing}>
                  {money(s.spend_usd, 2)}
                </Td>
                <Td align="right" mono>
                  {ratio(s.best_score ?? null)}
                </Td>
                <Td className="max-w-[28rem] truncate text-ash" title={s.brief}>
                  {s.brief || MISSING}
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      </Panel>

      {sessionId ? <SessionDetail sessionId={sessionId} /> : null}
    </div>
  );
}

function SessionDetail({ sessionId }: { sessionId: string }) {
  const q = useQuery({
    queryKey: ["agent", "research", sessionId],
    queryFn: () => api.researchSession(sessionId),
    // Poll while it is running; stop once it has stopped.
    refetchInterval: (query) =>
      (query.state.data as ResearchDetail | undefined)?.status === "running" ? LIVE_MS : false,
    retry: false,
  });

  if (q.isLoading) return <Loading what="session" />;
  if (q.error) return <ErrorNote error={q.error} />;
  const s = q.data;
  if (!s) return null;

  const bar = s.market?.oos?.sharpe ?? null;
  const beat = s.best_score !== null && bar !== null && (s.best_score ?? 0) > bar;

  return (
    <>
      {s.brief ? (
        <Panel title="brief">
          <p className="px-3 py-2 text-sm text-chalk whitespace-pre-wrap m-0">{s.brief}</p>
        </Panel>
      ) : null}

      <div className="grid grid-cols-2 md:grid-cols-6 gap-px bg-slate border border-slate rounded-md overflow-hidden">
        <div className="bg-gunmetal">
          <Stat label="status" value={s.status === "running" ? "running" : s.stopped_because} />
        </div>
        <div className="bg-gunmetal">
          <Stat label="experiments" value={count(s.experiments)} />
        </div>
        <div className="bg-gunmetal">
          <Stat label="calls" value={count(s.calls)} />
        </div>
        <div className="bg-gunmetal">
          <Stat label="elapsed" value={duration(s.elapsed_minutes * 60)} />
        </div>
        <div className="bg-gunmetal">
          <Stat label={`spend · ${s.billing}`} value={money(s.spend_usd, 2)} />
        </div>
        <div className="bg-gunmetal">
          <Stat
            label="best vs market"
            value={ratio(s.best_score ?? null)}
            sub={bar === null ? undefined : `bar ${ratio(bar)}`}
            tone={bar === null ? undefined : beat ? 1 : -1}
          />
        </div>
      </div>

      {s.status === "running" ? (
        <div className="flex items-center gap-3 px-1">
          <StopSessionButton sessionId={s.session_id} />
          <span className="text-xs text-ash-dim">
            stops after the current turn and stays resumable
          </span>
        </div>
      ) : null}

      {s.stopped_because === "rate_limit" ? (
        <LoudWarning>
          <strong className="text-chalk">Stopped on a provider limit.</strong> The work so far is
          kept. Resume when the window resets:{" "}
          <code className="mono text-teal">lab agent research -c &lt;config&gt; --resume {s.session_id}</code>
        </LoudWarning>
      ) : null}

      {s.verdict ? (
        <Panel title="verdict">
          <div className="px-3 py-2 flex flex-col gap-2">
            <p className="text-sm text-chalk m-0">{s.verdict}</p>
            {s.satisfied_with ? (
              <p className="text-xs text-ash m-0">
                stands behind{" "}
                <Link
                  to="/runs/$runId"
                  params={{ runId: s.satisfied_with }}
                  className="mono text-teal no-underline"
                >
                  {shortId(s.satisfied_with, 24)} ↗
                </Link>
              </p>
            ) : null}
            {s.next_steps ? (
              <p className="text-xs text-ash-dim m-0">next: {s.next_steps}</p>
            ) : null}
            {/* Shown next to the verdict, not buried below the experiment table:
                whether the winning parameters survive being nudged is part of
                the claim, and the agent does not always volunteer it. */}
            {s.neighbourhood?.verdict ? (
              <p
                className={cx(
                  "text-xs m-0 pt-1 border-t border-slate",
                  s.neighbourhood.fragile ? "text-ember" : "text-moss",
                )}
                title={`${s.neighbourhood.checked} perturbation(s), ~10% either side of each numeric parameter`}
              >
                neighbourhood: {s.neighbourhood.verdict}
              </p>
            ) : null}
            {/* Amber even when the verdict is HOLDS: a clean result covering half
                the directions is a weaker claim than one covering all of them,
                and the two must not look the same on the page. */}
            {s.neighbourhood?.coverage && !s.neighbourhood.coverage.complete ? (
              <p className="text-micro text-amber m-0">
                partial coverage — {s.neighbourhood.coverage.runs_spent} of{" "}
                {s.neighbourhood.coverage.runs_for_full_coverage} runs for{" "}
                {s.neighbourhood.coverage.parameters} parameters; a cliff on an untried side
                would not have been seen
              </p>
            ) : null}
          </div>
        </Panel>
      ) : null}

      <Panel
        title="experiments"
        right={
          <span className="unit">
            all figures OUT-OF-SAMPLE · a run's own page shows the full period
          </span>
        }
      >
        {s.runs.length === 0 ? (
          <Empty>nothing tried yet</Empty>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th align="right">#</Th>
                <Th>strategy</Th>
                <Th align="right">try</Th>
                <Th align="right">score ±1σ</Th>
                <Th>params</Th>
                <Th align="right">oos sharpe</Th>
                <Th align="right">oos return</Th>
                <Th align="right">vs market</Th>
                <Th align="right">full vs mkt</Th>
                <Th align="right">exposure</Th>
                <Th align="right">is sharpe</Th>
                <Th align="right">worst period</Th>
                <Th align="right">trades</Th>
                <Th>why</Th>
              </tr>
            </thead>
            <tbody>
              {s.runs.map((r) => (
                <Tr key={`${r.n}-${r.run_id || "x"}`}>
                  <Td align="right" mono>
                    {r.n}
                  </Td>
                  <Td mono>
                    {r.run_id ? (
                      <Link
                        to="/runs/$runId"
                        params={{ runId: r.run_id }}
                        className="text-teal no-underline"
                      >
                        {r.strategy}
                      </Link>
                    ) : (
                      r.strategy
                    )}
                    {r.contaminated ? (
                      <span className="ml-1">
                        <Badge kind="warn" title="Pattern-B: contaminated, not run">
                          llm
                        </Badge>
                      </span>
                    ) : null}
                  </Td>
                  {/* Every backtest reads the holdout, so the 4th variant's OOS
                      score is the best of four looks at the test set. Amber past
                      three so a tuning chain is visible without reading params. */}
                  <Td
                    align="right"
                    mono
                    className={cx((r.variant_index ?? 1) >= 3 && "text-amber")}
                    title={
                      (r.variant_index ?? 1) >= 3
                        ? `attempt ${r.variant_index} at this strategy — its out-of-sample score is the best of ${r.variant_index} looks at the held-out window, not an independent estimate`
                        : "which attempt at this strategy this is"
                    }
                  >
                    {r.variant_index ?? MISSING}
                  </Td>
                  {/* The fitness score with the error bar the window can actually
                      support. A Sharpe from ~230 daily bars carries roughly ±1.0,
                      which is wider than most sweeps — so a score smaller than its
                      own sigma is dimmed, because ranking on it is ranking on noise. */}
                  <Td align="right" mono>
                    {r.score === null || r.score === undefined ? (
                      MISSING
                    ) : (
                      <span
                        className={cx(
                          r.fitness_one_sigma != null &&
                            Math.abs(r.score) < r.fitness_one_sigma &&
                            "text-amber",
                        )}
                        title={
                          r.fitness_one_sigma != null
                            ? `one-sigma sampling error on this metric is ±${r.fitness_one_sigma.toFixed(3)} — scores closer together than that are indistinguishable`
                            : "no closed-form error for this metric"
                        }
                      >
                        {ratio(r.score)}
                        {r.fitness_one_sigma != null ? (
                          <span className="text-ash-dim"> ±{ratio(r.fitness_one_sigma)}</span>
                        ) : null}
                      </span>
                    )}
                  </Td>
                  <Td className="max-w-[18rem] truncate text-ash-dim" mono title={JSON.stringify(r.params)}>
                    {JSON.stringify(r.params)}
                  </Td>
                  <Td align="right" mono className={cx(!r.ok && "text-ash-dim")}>
                    {ratio(r.oos_sharpe ?? null)}
                  </Td>
                  <Td align="right" mono className="text-ash">
                    {pct(r.oos_return ?? null)}
                  </Td>
                  <Td
                    align="right"
                    mono
                    className={cx(
                      (r.oos_return_vs_market ?? 0) > 0 ? "text-moss" : "text-ember",
                    )}
                    title="out-of-sample return minus the benchmark's, same window"
                  >
                    {r.oos_return_vs_market === null ? MISSING : pct(r.oos_return_vs_market)}
                  </Td>
                  {/* The OOS slice is the last fifth of the range. Shown alone it
                      reads as the verdict, which is how a strategy that beat the
                      benchmark by 217pp over four years got written off on eleven
                      months of it. Both windows, side by side. */}
                  <Td
                    align="right"
                    mono
                    className={cx(
                      (r.full_return_vs_market ?? 0) > 0 ? "text-moss" : "text-ember",
                    )}
                    title="whole tradeable period (warmup excluded) minus the benchmark's, same window"
                  >
                    {r.full_return_vs_market === null ? MISSING : pct(r.full_return_vs_market)}
                  </Td>
                  <Td
                    align="right"
                    mono
                    className={cx((r.oos_exposure ?? 1) < 0.4 && "text-amber")}
                    title="share of the window invested — low exposure flatters Sharpe"
                  >
                    {pct(r.oos_exposure ?? null)}
                  </Td>
                  <Td align="right" mono className="text-ash">
                    {ratio(r.is_sharpe ?? null)}
                  </Td>
                  <Td
                    align="right"
                    mono
                    className={cx((r.worst_period_sharpe ?? 0) < 0 && "text-ember")}
                  >
                    {ratio(r.worst_period_sharpe ?? null)}
                  </Td>
                  <Td align="right" mono>
                    {count(r.trades ?? null)}
                  </Td>
                  <Td
                    className={cx("max-w-[26rem] truncate", r.ok ? "text-ash" : "text-ember")}
                    title={r.error || r.rationale}
                  >
                    {r.error || r.rationale}
                  </Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel
        title="reasoning"
        right={<span className="unit">{count(s.turns.length)} turns</span>}
      >
        {s.turns.length === 0 ? (
          <Empty>no turns yet</Empty>
        ) : (
          <ol className="m-0 p-0 list-none">
            {s.turns.map((t, i) => {
              const rejected = typeof t.rejected === "string";
              const action = String(t.action ?? (rejected ? "rejected" : "?"));
              return (
                <li
                  key={i}
                  className="flex gap-2 px-3 py-1.5 border-b border-slate/50 last:border-0"
                >
                  <span className="mono text-ash-dim w-6 shrink-0 text-right">{i + 1}</span>
                  <span className="shrink-0">
                    <Badge kind={rejected ? "bad" : action === "finish" ? "good" : "neutral"}>
                      {action}
                    </Badge>
                  </span>
                  <span className="text-sm text-chalk min-w-0">
                    {rejected ? String(t.rejected) : String(t.rationale ?? "")}
                    {t.strategy ? (
                      <span className="mono text-ash-dim"> · {String(t.strategy)}</span>
                    ) : null}
                    {typeof t.score === "number" ? (
                      <span className="mono text-ash"> · score {ratio(t.score)}</span>
                    ) : null}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </Panel>

      {s.notes.length ? (
        <Panel title="notes it kept">
          <ul className="m-0 px-6 py-2 text-sm text-ash">
            {s.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </Panel>
      ) : null}

      <footer className="text-xs text-ash-dim mono flex flex-wrap gap-x-4 gap-y-1 px-1">
        <span>session {s.session_id}</span>
        <span>started {stamp(s.created_at)}</span>
        <span>updated {stamp(s.updated_at)}</span>
        {s.market?.oos ? (
          <span>
            market bar: sharpe {ratio(s.market.oos.sharpe ?? null)} · return{" "}
            {pct(s.market.oos.total_return ?? null)}
          </span>
        ) : null}
        {s.workspace ? <span className="truncate">workspace {s.workspace}</span> : null}
      </footer>
    </>
  );
}
