/**
 * The Pattern-B ledger: what the model was told, what it said, what the gate
 * did about it, and what it cost.
 *
 * Pattern B puts an LLM inside `on_bar`, which changes the question the console
 * has to answer. For a deterministic strategy the question is "why did it do
 * that"; here it splits in two. *Can I believe this?* — a historical backtest
 * of a model trained past the test window is contaminated, and that has to be
 * said out loud, not inferred. And *what is it costing me?* — a per-bar API
 * call is a per-bar invoice, so cumulative spend sits against its budget in the
 * header, amber as it closes on the ceiling and ember once through it.
 *
 * Every byte the model produced — prompt, reply, rationale — is rendered as a
 * text node. It is untrusted input here for exactly the reason it is untrusted
 * at the risk gate: the model proposes, the system disposes, and neither the
 * gate nor the DOM takes its word for anything.
 */

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import type { AgentCall, DecisionRow, GateVerdict, RunRecord, SignalInput } from "../lib/types";
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
  pct,
  shortId,
  stamp,
} from "../lib/format";
import { timeLock, toMs } from "../lib/timelock";
import {
  Badge,
  Empty,
  ErrorNote,
  Loading,
  LoudWarning,
  Panel,
  Table,
  Td,
  Th,
  Tr,
  cx,
} from "./ui";

const SOURCE = "agent-ledger";

/** Where a run declares its cost ceiling. First key that carries a number wins. */
const BUDGET_KEYS = [
  "agent_budget_usd",
  "budget_usd",
  "daily_budget_usd",
  "cost_budget_usd",
  "max_cost_usd",
] as const;

/** Fraction of budget at which the meter starts warning rather than reporting. */
export const NEAR_BUDGET = 0.8;

export type SpendLevel = "ok" | "near" | "over";

/**
 * A budget nobody declared is not a budget of zero — with no ceiling there is
 * nothing to be near or past, so the meter reports and stays quiet.
 */
export function spendLevel(spent: number, budget: number | null | undefined): SpendLevel {
  if (budget === null || budget === undefined || !Number.isFinite(budget) || budget <= 0) return "ok";
  if (!Number.isFinite(spent)) return "ok";
  if (spent > budget) return "over";
  if (spent >= budget * NEAR_BUDGET) return "near";
  return "ok";
}

/** Bar width, 0..1. Clamped: a 300%-of-budget run must not paint off-panel. */
export function spendFraction(spent: number, budget: number | null | undefined): number {
  if (budget === null || budget === undefined || !Number.isFinite(budget) || budget <= 0) return 0;
  if (!Number.isFinite(spent) || spent <= 0) return 0;
  return Math.min(1, spent / budget);
}

export function readBudget(...sources: (Record<string, unknown> | undefined | null)[]): number | null {
  for (const src of sources) {
    if (!src) continue;
    for (const k of BUDGET_KEYS) {
      const v = src[k];
      if (typeof v === "number" && Number.isFinite(v) && v > 0) return v;
    }
  }
  return null;
}

/** Only decisions that actually called the model belong in this ledger. */
export interface AgentEntry {
  decision: DecisionRow;
  call: AgentCall;
  /** Spend up to and including this call. */
  cum: number;
}

export function agentEntries(decisions: readonly DecisionRow[]): AgentEntry[] {
  let cum = 0;
  const out: AgentEntry[] = [];
  for (const decision of decisions) {
    const call = decision.agent;
    if (!call) continue;
    cum += Number.isFinite(call.cost_usd) ? call.cost_usd : 0;
    out.push({ decision, call, cum });
  }
  return out;
}

/**
 * Which run's calls to show by default. A paper or live run is the one the
 * operator is paying for right now; a backtest of a Pattern-B strategy is a
 * plumbing test, so it only wins if nothing is running.
 */
export function pickRun(runs: readonly RunRecord[]): RunRecord | null {
  if (runs.length === 0) return null;
  const byNewest = [...runs].sort((a, b) => toMs(b.created_at) - toMs(a.created_at));
  return byNewest.find((r) => r.kind === "live" || r.kind === "paper") ?? byNewest[0];
}

/** Model output is data. Never markup, never a template — just characters. */
export function asText(v: unknown): string {
  if (v === null || v === undefined) return MISSING;
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

// --- spend meter -------------------------------------------------------------

const BAR: Record<SpendLevel, string> = {
  ok: "bg-slate-hi",
  near: "bg-amber",
  over: "bg-ember",
};

const SPEND_TEXT: Record<SpendLevel, string> = {
  ok: "text-chalk",
  near: "text-amber",
  over: "text-ember",
};

export function SpendMeter({ spent, budget }: { spent: number; budget: number | null }) {
  const level = spendLevel(spent, budget);
  const frac = spendFraction(spent, budget);
  return (
    <span data-spend={level} className="inline-flex items-center gap-2">
      <span className="text-micro uppercase tracking-wider text-ash">spend</span>
      <span className={cx("num", SPEND_TEXT[level])}>{money(spent, 4)}</span>
      {budget === null ? (
        <span
          className="text-micro uppercase tracking-wider text-ash-dim"
          title="no budget in this run's config, so there is no ceiling to warn about"
        >
          no budget declared
        </span>
      ) : (
        <>
          <span className="text-ash-dim" aria-hidden>
            /
          </span>
          <span className="num text-ash">{money(budget, 2)}</span>
          <span
            role="meter"
            aria-label="spend against budget"
            aria-valuemin={0}
            aria-valuemax={budget}
            aria-valuenow={spent}
            className="relative block h-1.5 w-24 rounded-sm bg-slate overflow-hidden"
          >
            <span
              data-bar={level}
              className={cx("absolute inset-y-0 left-0", BAR[level])}
              style={{ width: `${frac * 100}%` }}
            />
          </span>
          {level === "over" ? (
            <Badge kind="bad" title="this run has spent more than its declared budget">
              over budget
            </Badge>
          ) : level === "near" ? (
            <Badge kind="warn" title={`past ${Math.round(NEAR_BUDGET * 100)}% of the declared budget`}>
              near budget
            </Badge>
          ) : null}
        </>
      )}
    </span>
  );
}

// --- screen ------------------------------------------------------------------

export interface AgentLedgerProps {
  strategy: string;
  /** Pin the ledger to one run; otherwise the newest paper/live run wins. */
  runId?: string;
  /** Cost ceiling, when the caller knows one the run config does not carry. */
  budgetUsd?: number;
}

export function AgentLedger({ strategy, runId, budgetUsd }: AgentLedgerProps) {
  const runsQ = useQuery({
    queryKey: ["runs", "agent-ledger", strategy],
    queryFn: () => api.runs({ strategy, limit: 50 }),
    enabled: Boolean(strategy),
    retry: false,
  });

  const runs = useMemo(() => runsQ.data?.runs ?? [], [runsQ.data]);
  const [chosen, setChosen] = useState<string | null>(null);
  useEffect(() => setChosen(null), [strategy]);

  const run = useMemo(() => {
    const wanted = runId ?? chosen;
    return (wanted ? runs.find((r) => r.run_id === wanted) : null) ?? pickRun(runs);
  }, [runs, runId, chosen]);

  const live = run?.kind === "live" || run?.kind === "paper";

  const callsQ = useQuery({
    queryKey: ["runs", run?.run_id ?? "none", "decisions", "agent"],
    queryFn: () => api.decisions(run?.run_id ?? "", { limit: 500 }),
    enabled: Boolean(run?.run_id),
    // A finished backtest is immutable; a paper session is still writing rows.
    staleTime: live ? 0 : Number.POSITIVE_INFINITY,
    refetchInterval: live ? 5000 : false,
    retry: false,
  });

  const entries = useMemo(() => agentEntries(callsQ.data?.decisions ?? []), [callsQ.data]);
  const spent = entries.length ? entries[entries.length - 1].cum : 0;
  const budget = budgetUsd ?? readBudget(run?.config, run?.params);

  const scope = run?.run_id ?? strategy;
  const store = useMemo(() => timeLock(scope), [scope]);
  const selection = useSyncExternalStore(store.subscribe, () => store.getState().selection);

  const [openId, setOpenId] = useState<string | null>(null);
  useEffect(() => setOpenId(null), [scope]);

  const activeId =
    (selection.kind === "decision" && selection.id) || openId || entries[entries.length - 1]?.decision.id || null;
  const active = entries.find((e) => e.decision.id === activeId) ?? null;

  if (runsQ.isLoading) return <Loading what="runs" />;
  if (runsQ.error) return <ErrorNote error={runsQ.error} />;
  if (!run) {
    return (
      <Empty cmd={`lab paper run ${strategy}`}>no runs recorded for {strategy}</Empty>
    );
  }

  const select = (e: AgentEntry) => {
    setOpenId(e.decision.id);
    store.select({ kind: "decision", id: e.decision.id }, toMs(e.decision.at), SOURCE);
  };

  return (
    <div className="flex flex-col gap-2 p-2 min-h-0">
      <Panel
        title="agent calls"
        right={
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-micro uppercase tracking-wider text-ash">
              calls <span className="num text-chalk">{count(entries.length)}</span>
            </span>
            <span className="text-micro uppercase tracking-wider text-ash">
              avg <span className="num text-chalk">{money(entries.length ? spent / entries.length : 0, 4)}</span>
            </span>
            <SpendMeter spent={spent} budget={budget} />
            <select
              value={run.run_id}
              onChange={(e) => setChosen(e.target.value)}
              aria-label="run"
              disabled={Boolean(runId)}
              className="mono text-xs h-5 bg-graphite text-chalk border border-slate rounded px-1"
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {shortId(r.run_id)} · {r.kind} · {day(r.created_at)}
                </option>
              ))}
            </select>
          </div>
        }
        bodyClassName="p-2"
      >
        {run.metrics.contaminated ? (
          <LoudWarning>
            this run backtests a model over a window inside its own training data — the numbers
            below measure the plumbing, not the edge
          </LoudWarning>
        ) : null}
        {run.metrics.optimistic_fills ? (
          <LoudWarning>
            same-bar fills were enabled: a decision could trade at the price it set
          </LoudWarning>
        ) : null}
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-2 min-h-0">
        <Panel title="calls" bodyClassName="max-h-[52vh] overflow-auto">
          {callsQ.isLoading ? (
            <Loading what="reading decision journal" />
          ) : callsQ.error ? (
            <ErrorNote error={callsQ.error} />
          ) : entries.length === 0 ? (
            <Empty cmd={`lab paper run ${strategy}`}>
              no model calls journalled for {shortId(run.run_id)} — this run has no LLM in its
              runtime path
            </Empty>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>at</Th>
                  <Th>model</Th>
                  <Th align="right">in</Th>
                  <Th align="right">out</Th>
                  <Th align="right">cost</Th>
                  <Th align="right">cum</Th>
                  <Th align="right">lat</Th>
                  <Th>gate</Th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <Tr
                    key={e.decision.id}
                    selected={e.decision.id === activeId}
                    onClick={() => select(e)}
                  >
                    <Td mono>
                      {/* Stops here: bubbling to the row would select twice. */}
                      <button
                        type="button"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          select(e);
                        }}
                        className="mono text-chalk hover:text-teal"
                        aria-label={`call at ${e.decision.at}`}
                      >
                        {clock(e.decision.at)}
                      </button>
                    </Td>
                    <Td mono className="text-ash">
                      {e.call.model}
                    </Td>
                    <Td align="right" mono className="text-ash">
                      {count(e.call.input_tokens)}
                    </Td>
                    <Td align="right" mono className="text-ash">
                      {count(e.call.output_tokens)}
                    </Td>
                    <Td align="right" mono>
                      {money(e.call.cost_usd, 4)}
                    </Td>
                    <Td align="right" mono className="text-ash">
                      {money(e.cum, 3)}
                    </Td>
                    <Td align="right" mono className="text-ash-dim">
                      {duration(e.call.latency_ms / 1000)}
                    </Td>
                    <Td>
                      <GateGlyphs verdicts={e.decision.verdicts} />
                    </Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          )}
        </Panel>

        <Panel
          title={active ? `call · ${stamp(active.decision.at)}` : "call"}
          bodyClassName="max-h-[52vh] overflow-auto"
        >
          {!active ? <Empty>pick a call</Empty> : <CallDetail entry={active} />}
        </Panel>
      </div>
    </div>
  );
}

// --- one call ----------------------------------------------------------------

function GateGlyphs({ verdicts }: { verdicts: GateVerdict[] }) {
  if (verdicts.length === 0) return <span className="text-ash-dim">—</span>;
  const blocked = verdicts.filter((v) => v.action === "blocked").length;
  const clipped = verdicts.filter((v) => v.action === "clipped").length;
  const passed = verdicts.length - blocked - clipped;
  return (
    <span className="flex items-center gap-1">
      {passed ? <Badge>{`${passed} pass`}</Badge> : null}
      {clipped ? <Badge kind="warn">{`${clipped} clipped`}</Badge> : null}
      {blocked ? <Badge kind="warn">{`${blocked} blocked`}</Badge> : null}
    </span>
  );
}

function Section({
  label,
  right,
  children,
}: {
  label: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="border-t border-slate first:border-t-0">
      <div className="flex items-baseline justify-between gap-2 px-3 pt-2">
        <h3 className="text-micro uppercase tracking-wider text-ash">{label}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

/** Model-authored text. A `<pre>` with a string child — never `innerHTML`. */
function ModelText({ value, max = 320 }: { value: unknown; max?: number }) {
  const text = asText(value);
  return (
    <pre
      className="mono text-xs leading-[1.45] text-ash whitespace-pre-wrap break-words m-0 px-3 py-2 overflow-auto"
      style={{ maxHeight: max }}
    >
      {text}
    </pre>
  );
}

function lagSeconds(s: SignalInput): number {
  return (toMs(s.knowledge_time) - toMs(s.event_time)) / 1000;
}

function CallDetail({ entry }: { entry: AgentEntry }) {
  const { decision, call } = entry;
  const inputs = decision.inputs ?? {};
  const indicators = Object.entries(inputs.indicators ?? {});
  const prices = Object.entries(inputs.prices ?? {});
  const signals = inputs.signals ?? [];
  const verdictFor = (ticker: string): GateVerdict | undefined =>
    decision.verdicts.find((v) => v.ticker === ticker);

  return (
    <div className="flex flex-col">
      <Section
        label="cost"
        right={
          <span className="flex items-baseline gap-3 text-xs">
            <span className="mono text-ash">{call.model}</span>
            <span className="text-micro uppercase tracking-wider text-ash">
              in <span className="num text-chalk">{count(call.input_tokens)}</span>
            </span>
            <span className="text-micro uppercase tracking-wider text-ash">
              out <span className="num text-chalk">{count(call.output_tokens)}</span>
            </span>
            <span className="text-micro uppercase tracking-wider text-ash">
              cost <span className="num text-chalk">{money(call.cost_usd, 4)}</span>
            </span>
            <span className="text-micro uppercase tracking-wider text-ash">
              latency <span className="num text-chalk">{duration(call.latency_ms / 1000)}</span>
            </span>
          </span>
        }
      >
        <p className="px-3 pb-2 pt-1 text-xs text-ash-dim m-0">
          cumulative through this call <span className="num text-ash">{money(entry.cum, 4)}</span>
        </p>
      </Section>

      <Section label="context bundle · what the model was handed">
        <div className="px-3 py-2 flex flex-col gap-2">
          {indicators.length ? (
            <div className="scroll-x">
              <table className="mono text-xs border-collapse">
                <tbody>
                  {indicators.map(([name, ind]) => (
                    <tr key={name}>
                      <td className="pr-3 text-ash whitespace-nowrap">{name}</td>
                      <td className="pr-3 text-ash-dim whitespace-nowrap">
                        {Object.values(ind.params).length
                          ? `(${Object.values(ind.params).map(String).join(",")})`
                          : ""}
                      </td>
                      <td className="num text-chalk">{num(ind.value, 4)}</td>
                      <td className="pl-3 text-ash-dim">n={count(ind.n)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {prices.length ? (
            <p className="mono text-xs text-ash m-0">
              {prices.map(([tk, p]) => `${tk} ${money(p, 2)}`).join("   ")}
            </p>
          ) : null}

          {signals.length ? (
            <div className="scroll-x">
              <table className="mono text-xs border-collapse w-full">
                <thead>
                  <tr>
                    <Th>source</Th>
                    <Th>ticker</Th>
                    <Th>event_time</Th>
                    <Th>knowledge_time</Th>
                    <Th align="right">lag</Th>
                  </tr>
                </thead>
                <tbody>
                  {signals.map((s) => {
                    const lag = lagSeconds(s);
                    return (
                      <tr key={s.uid}>
                        <Td className="text-ash">{s.source}</Td>
                        <Td mono>{s.ticker}</Td>
                        <Td mono className="text-ash-dim">
                          {stamp(s.event_time)}
                        </Td>
                        <Td mono className="text-ash-dim">
                          {stamp(s.knowledge_time)}
                        </Td>
                        <Td align="right" mono className={lag < 0 ? "text-ember" : "text-ash"}>
                          {lag < 0 ? `−${age(-lag)}` : age(lag)}
                        </Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}

          <p className="text-xs text-ash-dim m-0">
            portfolio <span className="num text-ash">{money(decision.portfolio.equity)}</span> ·{" "}
            <span className="num text-ash">{count(decision.portfolio.n_positions)}</span> positions ·
            gross <span className="num text-ash">{pct(decision.portfolio.gross_exposure, 0)}</span>
          </p>
        </div>
      </Section>

      <Section label="prompt">
        <ModelText value={call.prompt} max={220} />
      </Section>

      <Section label="structured reply">
        <ModelText value={call.response} max={260} />
      </Section>

      <Section label="rationale">
        {/* Text node. The model's words are quoted, never executed. */}
        <p className="text-xs text-ash leading-relaxed whitespace-pre-wrap m-0 px-3 py-2">
          {call.rationale || "the model recorded no rationale for this call"}
        </p>
      </Section>

      <Section
        label="proposed targets · gate verdict"
        right={
          <span className="text-micro uppercase tracking-wider text-ash-dim">
            the model proposes, the gate disposes
          </span>
        }
      >
        {decision.intents.length === 0 ? (
          <p className="text-xs text-ash-dim px-3 py-2 m-0">the model proposed no position change</p>
        ) : (
          <Table className="text-xs">
            <thead>
              <tr>
                <Th>ticker</Th>
                <Th align="right">requested</Th>
                <Th align="right">approved</Th>
                <Th>action</Th>
                <Th>rule</Th>
                <Th>reason</Th>
              </tr>
            </thead>
            <tbody>
              {decision.intents.map((intent) => {
                const v = verdictFor(intent.ticker);
                const clipped = v?.action === "clipped";
                return (
                  <tr key={`${intent.ticker}-${intent.tag}`}>
                    <Td mono>{intent.ticker}</Td>
                    <Td align="right" mono className={clipped ? "text-amber" : "text-ash"}>
                      {delta(v?.requested_pct ?? intent.target_pct, 2)}
                    </Td>
                    <Td align="right" mono className={v?.action === "blocked" ? "text-ember" : "text-chalk"}>
                      {v ? delta(v.approved_pct, 2) : MISSING}
                    </Td>
                    <Td>
                      {!v ? (
                        <span className="text-ash-dim">—</span>
                      ) : v.action === "pass" ? (
                        <Badge>pass</Badge>
                      ) : (
                        <Badge kind="warn">{v.action}</Badge>
                      )}
                    </Td>
                    <Td mono className="text-ash-dim">
                      {v?.rule || null}
                    </Td>
                    {/* Model-authored reason string, rendered as text. */}
                    <Td className="text-ash">{intent.reason || v?.detail || null}</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Section>
    </div>
  );
}
