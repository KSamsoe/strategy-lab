/**
 * The decision tape.
 *
 * One row per `on_bar` call. Collapsed it is a log line; expanded it is the
 * entire input tape that produced the decision. The whole component exists to
 * collapse one distance: "that position looks wrong" -> "here is the exact
 * input that caused it", in one click and zero SQL. Anything that does not
 * serve that sentence does not belong here.
 *
 * The signals table is the load-bearing part. `event_time` and
 * `knowledge_time` sit next to each other with the gap spelled out, because
 * that gap is the answer to "why did it act now and not six weeks earlier" --
 * and a negative gap is lookahead, which is the difference between an honest
 * backtest and a fraudulent-looking one. It is never buried behind a tooltip.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { AgentCall, DecisionRow, GateVerdict, SignalInput } from "../lib/types";
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
import { nearestIndex, toMs, useTimeLock } from "../lib/timelock";
import { Badge, Button, Empty, ErrorNote, Loading, Table, Td, Th, Tr, cx } from "./ui";

/**
 * A long backtest journals thousands of decisions and the tape must stay
 * responsive, so the DOM is bounded to a window that follows the playhead. The
 * window is never silent: the strip says which slice is on screen and the
 * arrows page it, so no row is unreachable.
 */
const CAP = 500;

/** Our own name on the bus, so we can ignore the echo of our own seeks. */
const SOURCE = "decision-tape";

export interface DecisionTapeProps {
  scope: string;
  decisions: DecisionRow[];
  loading?: boolean;
  error?: unknown;
  /** Live mode pins the playhead to the newest row and auto-scrolls. */
  pinned?: boolean;
  /** Render the agent prompt/response block inline (Pattern-B strategies). */
  showAgent?: boolean;
  emptyCmd?: string;
}

export function DecisionTape({
  scope,
  decisions,
  loading,
  error,
  pinned = false,
  showAgent = false,
  emptyCmd = "lab backtest strategies/momo.py --journal",
}: DecisionTapeProps) {
  const { t, selection, source, select, seek, pin } = useTimeLock(scope);

  const [openIds, setOpenIds] = useState<ReadonlySet<string>>(() => new Set<string>());
  const [winStart, setWinStart] = useState(0);
  /** Live tape follows the newest row until the operator clicks one. */
  const [following, setFollowing] = useState(pinned);

  const rowEls = useRef(new Map<string, HTMLElement>());
  const anchorRef = useRef(-1);
  const newestRef = useRef<string | null>(null);

  const total = decisions.length;
  const times = useMemo(() => decisions.map((d) => toMs(d.at)), [decisions]);

  // An explicit decision selection wins over proximity: if the lock names a row
  // it is that row, even when two decisions share a timestamp.
  const activeIndex = useMemo(() => {
    if (selection.kind === "decision" && selection.id) {
      const i = decisions.findIndex((d) => d.id === selection.id);
      if (i >= 0) return i;
    }
    return nearestIndex(times, t);
  }, [decisions, times, selection.kind, selection.id, t]);

  const maxStart = Math.max(0, total - CAP);
  const start = total > CAP ? Math.min(Math.max(0, winStart), maxStart) : 0;
  const rows = total > CAP ? decisions.slice(start, start + CAP) : decisions;
  const capped = total > CAP;

  useEffect(() => {
    setFollowing(pinned);
  }, [pinned]);

  // Live: the playhead is now, and now is the newest decision.
  const newestId = total > 0 ? decisions[total - 1].id : null;
  useEffect(() => {
    if (!pinned || !following || total === 0) return;
    pin(toMs(decisions[total - 1].at));
  }, [pinned, following, total, decisions, pin, newestId]);

  // Keep the rendered window over the playhead, but only when the playhead
  // actually moved -- otherwise paging back would be undone on the next render.
  useEffect(() => {
    const anchor = pinned && following ? total - 1 : activeIndex;
    if (anchor < 0 || anchor === anchorRef.current) return;
    anchorRef.current = anchor;
    setWinStart((w) =>
      anchor >= w && anchor < w + CAP
        ? w
        : Math.min(Math.max(0, anchor - (CAP >> 1)), Math.max(0, total - CAP)),
    );
  }, [activeIndex, pinned, following, total]);

  // A move that came from a chart or the ledger scrolls the tape; a move we
  // emitted ourselves does not, because that row is already under the cursor.
  useEffect(() => {
    if (source === SOURCE) return;
    const d = decisions[activeIndex];
    if (!d) return;
    rowEls.current.get(d.id)?.scrollIntoView?.({ block: "nearest" });
  }, [activeIndex, decisions, source, start]);

  const onRow = useCallback(
    (d: DecisionRow) => {
      setOpenIds((prev) => {
        const next = new Set(prev);
        if (!next.delete(d.id)) next.add(d.id);
        return next;
      });
      // A deliberate click in live mode means "hold still and let me read this".
      if (pinned) {
        setFollowing(false);
        seek(toMs(d.at), SOURCE);
      }
      select({ kind: "decision", id: d.id }, toMs(d.at), SOURCE);
    },
    [pinned, seek, select],
  );

  const bindRow = useCallback((id: string, el: HTMLElement | null) => {
    if (el) rowEls.current.set(id, el);
    else rowEls.current.delete(id);
  }, []);

  if (error) return <ErrorNote error={error} />;
  if (loading) return <Loading what="reading decision journal" />;
  if (total === 0) {
    return (
      <Empty cmd={emptyCmd}>
        no decisions journalled for this {scope === "live" ? "session" : "run"}
      </Empty>
    );
  }

  const activeId = decisions[activeIndex]?.id ?? null;

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="flex items-center justify-between gap-3 px-2 h-6 shrink-0 border-b border-slate">
        <span className="flex items-center gap-2">
          <span className="unit">playhead</span>
          <span className="mono text-teal">{t === null ? MISSING : clock(t)}</span>
          {pinned ? (
            following ? (
              <Badge kind="accent" title="the tape is following the newest decision">
                live
              </Badge>
            ) : (
              <Button kind="accent" onClick={() => setFollowing(true)} title="resume following now">
                follow live
              </Button>
            )
          ) : null}
        </span>
        <span className="flex items-center gap-2">
          {capped ? (
            <>
              <Button
                onClick={() => setWinStart(Math.max(0, start - CAP))}
                disabled={start === 0}
                title="page the window earlier"
              >
                ←
              </Button>
              <Button
                onClick={() => setWinStart(Math.min(maxStart, start + CAP))}
                disabled={start >= maxStart}
                title="page the window later"
              >
                →
              </Button>
              <span className="unit text-amber" title="the tape is windowed, not filtered">
                showing {count(start + 1)}–{count(start + rows.length)} of {count(total)}
              </span>
            </>
          ) : (
            <span className="unit">{count(total)} decisions</span>
          )}
        </span>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {rows.map((d, i) => {
          const prev = i > 0 ? rows[i - 1] : undefined;
          const newDay = !prev || day(prev.at) !== day(d.at);
          return (
            <div key={d.id}>
              {newDay ? (
                <div className="sticky top-0 z-10 bg-gunmetal border-b border-slate px-2 h-5 flex items-center unit">
                  {day(d.at)}
                </div>
              ) : null}
              <TapeRow
                decision={d}
                open={openIds.has(d.id)}
                active={d.id === activeId}
                pulse={pinned && following && d.id === newestId && d.id !== newestRef.current}
                showAgent={showAgent}
                onClick={onRow}
                bindRow={bindRow}
              />
            </div>
          );
        })}
      </div>
      <PulseTracker id={newestId} slot={newestRef} />
    </div>
  );
}

/**
 * Records which row was newest on the previous commit so exactly one arriving
 * row pulses. A ref write in an effect, kept out of the render path.
 *
 * The box is called `slot`, not `ref`: React 18 reserves `ref` on a function
 * component, strips it from props, and the effect below then writes to
 * `undefined` and takes the whole tape down with it.
 */
function PulseTracker({ id, slot }: { id: string | null; slot: { current: string | null } }) {
  useEffect(() => {
    slot.current = id;
  }, [id, slot]);
  return null;
}

// --- one row -----------------------------------------------------------------

interface RowTone {
  glyph: string;
  cls: string;
  label: string;
}

/** Strongest gate action in the row, since that is what the eye should catch. */
function rowTone(d: DecisionRow): RowTone {
  if (d.verdicts.some((v) => v.action === "blocked")) {
    return { glyph: "⊘", cls: "text-amber", label: "gate blocked an intent" };
  }
  if (d.verdicts.some((v) => v.action === "clipped")) {
    return { glyph: "◑", cls: "text-amber", label: "gate clipped an intent" };
  }
  if (d.order_ids.length > 0) {
    return { glyph: "●", cls: "text-moss", label: "orders sent" };
  }
  return { glyph: "○", cls: "text-ash", label: "passed, no order" };
}

function TapeRow({
  decision,
  open,
  active,
  pulse,
  showAgent,
  onClick,
  bindRow,
}: {
  decision: DecisionRow;
  open: boolean;
  active: boolean;
  pulse: boolean;
  showAgent: boolean;
  onClick: (d: DecisionRow) => void;
  bindRow: (id: string, el: HTMLElement | null) => void;
}) {
  const tone = rowTone(decision);
  return (
    <div
      ref={(el) => bindRow(decision.id, el)}
      className={cx("border-b border-slate/50", active && "bg-teal-wash", pulse && "pulse")}
    >
      <button
        type="button"
        aria-expanded={open}
        onClick={() => onClick(decision)}
        style={{ height: "var(--row-h)" }}
        className="w-full flex items-center gap-2 px-2 text-left hover:bg-slate/40"
      >
        <span className="mono text-ash-dim w-2 shrink-0" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
        <span className={cx("mono w-3 text-center shrink-0", tone.cls)} title={tone.label}>
          {tone.glyph}
        </span>
        <span className="mono text-ash shrink-0">{clock(decision.at)}</span>
        <span className="mono text-chalk truncate flex-1 min-w-0">{decision.summary}</span>
        <span className="mono text-ash-dim shrink-0" title="decision latency">
          {duration(decision.duration_ms / 1000)}
        </span>
      </button>
      {open ? <TapeDetail decision={decision} showAgent={showAgent} /> : null}
    </div>
  );
}

// --- the expanded input tape -------------------------------------------------

function Section({
  label,
  right,
  children,
}: {
  label: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-baseline gap-2 mb-1">
        <span className="unit">{label}</span>
        {right}
      </div>
      {children}
    </div>
  );
}

function TapeDetail({ decision, showAgent }: { decision: DecisionRow; showAgent: boolean }) {
  const { indicators, signals, history, bars, prices } = decision.inputs;
  const p = decision.portfolio;

  return (
    <div className="px-3 pb-3 pt-2 border-t border-slate/50 bg-graphite/50">
      <Section label="input tape" right={<span className="text-xs text-ash-dim">verbatim</span>}>
        <div className="flex flex-col gap-2">
          {indicators && Object.keys(indicators).length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6">
              {Object.entries(indicators).map(([name, ind]) => (
                <div
                  key={name}
                  className="flex items-baseline justify-between gap-2 border-b border-slate/40 py-0.5"
                >
                  <span className="mono text-ash truncate">
                    {name}
                    {paramSuffix(ind.params)}
                  </span>
                  <span className="num text-chalk">{num(ind.value, 4)}</span>
                </div>
              ))}
            </div>
          ) : null}

          {history && Object.keys(history).length > 0
            ? Object.entries(history).map(([k, h]) => (
                <div key={k} className="flex items-baseline gap-2 text-xs min-w-0">
                  <span className="mono text-chalk shrink-0">{k}</span>
                  <span className="unit">n</span>
                  <span className="num text-ash">{count(h.n)}</span>
                  <span className="unit">last</span>
                  <span className="mono text-ash shrink-0">{stamp(h.last)}</span>
                  <span className="mono text-ash-dim truncate" title="tail of the slice the strategy saw">
                    {h.tail.map((pt) => num(pt.v, 2)).join("  ")}
                  </span>
                </div>
              ))
            : null}

          {bars && Object.keys(bars).length > 0
            ? Object.entries(bars).map(([k, b]) => (
                <div key={k} className="flex items-baseline gap-2 text-xs">
                  <span className="mono text-chalk shrink-0">{k}</span>
                  <span className="unit">bars</span>
                  <span className="num text-ash">{count(b.n)}</span>
                  <span className="unit">last</span>
                  <span className="mono text-ash">{stamp(b.last)}</span>
                  <span className="unit">close</span>
                  <span className="num text-chalk">{money(b.last_close, 2)}</span>
                </div>
              ))
            : null}

          {prices && Object.keys(prices).length > 0 ? (
            <div className="grid grid-cols-3 md:grid-cols-5 gap-x-6">
              {Object.entries(prices).map(([ticker, v]) => (
                <div key={ticker} className="flex items-baseline justify-between gap-2 py-0.5">
                  <span className="mono text-ash">{ticker}</span>
                  <span className="num text-chalk">{money(v, 2)}</span>
                </div>
              ))}
            </div>
          ) : null}

          <div className="flex items-baseline gap-3 text-xs border-t border-slate/40 pt-1">
            <span className="unit">portfolio</span>
            <span className="unit">equity</span>
            <span className="num text-chalk">{money(p.equity)}</span>
            <span className="unit">cash</span>
            <span className="num text-ash">{money(p.cash)}</span>
            <span className="unit">gross</span>
            <span className="num text-ash">{pct(p.gross_exposure)}</span>
            <span className="unit">net</span>
            <span className="num text-ash">{pct(p.net_exposure)}</span>
            <span className="unit">pos</span>
            <span className="num text-ash">{count(p.n_positions)}</span>
          </div>
        </div>
      </Section>

      {signals && signals.length > 0 ? <SignalsTable signals={signals} /> : null}

      {decision.intents.length > 0 ? (
        <Section label="intents">
          <Table className="text-xs">
            <thead>
              <tr>
                <Th>ticker</Th>
                <Th align="right">target</Th>
                <Th>tag</Th>
                <Th align="right">limit</Th>
                <Th>reason</Th>
              </tr>
            </thead>
            <tbody>
              {decision.intents.map((it) => (
                <Tr key={`${it.ticker}:${it.tag}:${it.reason}`}>
                  <Td mono>{it.ticker}</Td>
                  <Td align="right" className="num text-chalk">
                    {delta(it.target_pct)}
                  </Td>
                  <Td className="text-ash">{it.tag || null}</Td>
                  <Td align="right" className="num text-ash">
                    {money(it.limit_price, 2)}
                  </Td>
                  <Td className="text-ash">{it.reason || null}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        </Section>
      ) : null}

      {decision.verdicts.length > 0 ? (
        <Section label="gate verdicts">
          <Table className="text-xs">
            <thead>
              <tr>
                <Th>ticker</Th>
                <Th>action</Th>
                <Th>rule</Th>
                <Th align="right">requested → approved</Th>
                <Th>detail</Th>
              </tr>
            </thead>
            <tbody>
              {decision.verdicts.map((v) => (
                <Tr key={`${v.ticker}:${v.rule ?? "none"}:${v.action}`}>
                  <Td mono>{v.ticker}</Td>
                  <Td>
                    <VerdictBadge verdict={v} />
                  </Td>
                  <Td mono className="text-ash">
                    {v.rule}
                  </Td>
                  <Td align="right" className="mono">
                    {v.action === "pass" ? (
                      <span className="text-chalk">{pct(v.approved_pct)}</span>
                    ) : (
                      <>
                        <span className="text-ash line-through">{pct(v.requested_pct)}</span>
                        <span className="text-ash-dim"> → </span>
                        <span className="text-amber">{pct(v.approved_pct)}</span>
                      </>
                    )}
                  </Td>
                  <Td className="text-ash">{v.detail || null}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        </Section>
      ) : null}

      <Section
        label="orders"
        right={
          decision.order_ids.length === 0 ? (
            <span className="text-xs text-ash-dim">none sent</span>
          ) : null
        }
      >
        {decision.order_ids.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {/* The journal row carries ids only; side/qty/price live on the
                order and fill events the ledger and event tape render. */}
            {decision.order_ids.map((id) => (
              <code
                key={id}
                title={id}
                className="mono text-xs text-teal bg-graphite border border-slate rounded px-1.5"
              >
                {shortId(id, 12)}
              </code>
            ))}
          </div>
        ) : null}
      </Section>

      {decision.logs.length > 0 ? (
        <Section label="logs">
          <div className="flex flex-col">
            {decision.logs.map((l, i) => (
              <span key={i} className="mono text-xs text-ash-dim truncate">
                {JSON.stringify(l)}
              </span>
            ))}
          </div>
        </Section>
      ) : null}

      {showAgent && decision.agent ? <AgentBlock call={decision.agent} /> : null}
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: GateVerdict }) {
  if (verdict.action === "pass") return <Badge>pass</Badge>;
  return <Badge kind="warn">{verdict.action}</Badge>;
}

function paramSuffix(params: Record<string, unknown>): string {
  const vals = Object.values(params);
  return vals.length > 0 ? `(${vals.map(String).join(",")})` : "";
}

// --- signals: the reason this component exists -------------------------------

/** Seconds between the world event and the moment we could have known it. */
function lagSeconds(s: SignalInput): number {
  return (toMs(s.knowledge_time) - toMs(s.event_time)) / 1000;
}

function lagPhrase(seconds: number): string {
  if (!Number.isFinite(seconds)) return MISSING;
  if (Math.abs(seconds) < 1) return "same instant";
  return seconds > 0 ? `disclosed ${age(seconds)} later` : `known ${age(-seconds)} BEFORE event`;
}

function SignalsTable({ signals }: { signals: SignalInput[] }) {
  const maxLag = signals.reduce((m, s) => Math.max(m, lagSeconds(s)), 0);
  const leaks = signals.some((s) => lagSeconds(s) < 0);
  return (
    <Section
      label="signals"
      right={
        <>
          <span className="text-xs text-ash-dim">
            event_time is when it happened · knowledge_time is when this run could have known it
          </span>
          {maxLag >= 1 ? <Badge title="largest disclosure lag here">max lag {age(maxLag)}</Badge> : null}
          {leaks ? <Badge kind="bad">lookahead</Badge> : null}
        </>
      }
    >
      <Table className="text-xs">
        <thead>
          <tr>
            <Th>source</Th>
            <Th>ticker</Th>
            <Th>kind</Th>
            <Th>tier</Th>
            <Th>dir</Th>
            <Th align="right">score</Th>
            <Th>event_time</Th>
            <Th>knowledge_time</Th>
            <Th>lag</Th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => {
            const lag = lagSeconds(s);
            return (
              <Tr key={s.uid}>
                <Td className="text-ash">{s.source}</Td>
                <Td mono>
                  <span className="inline-flex items-center gap-1.5">
                    {s.ticker}
                    {s.fresh === false ? <Badge kind="warn">stale</Badge> : null}
                  </span>
                </Td>
                <Td className="text-ash">{s.kind}</Td>
                <Td className="text-ash">{s.tier}</Td>
                <Td className="text-ash">{s.direction}</Td>
                <Td align="right" className="num text-chalk">
                  {num(s.score, 2)}
                </Td>
                <Td mono className="text-ash">
                  {stamp(s.event_time)}
                </Td>
                <Td mono className="text-chalk">
                  {stamp(s.knowledge_time)}
                </Td>
                <Td className={cx("mono", lag < 0 ? "text-ember" : "text-ash")}>{lagPhrase(lag)}</Td>
              </Tr>
            );
          })}
        </tbody>
      </Table>
    </Section>
  );
}

// --- agent block (Pattern-B strategies) --------------------------------------

function asText(v: unknown): string {
  if (v === null || v === undefined) return MISSING;
  return typeof v === "string" ? v : JSON.stringify(v, null, 2);
}

function AgentBlock({ call }: { call: AgentCall }) {
  return (
    <Section
      label="agent call"
      right={
        <span className="flex items-baseline gap-3 text-xs">
          <span className="mono text-ash">{call.model}</span>
          <span className="unit">in</span>
          <span className="num text-ash">{count(call.input_tokens)}</span>
          <span className="unit">out</span>
          <span className="num text-ash">{count(call.output_tokens)}</span>
          <span className="unit">cost</span>
          <span className="num text-chalk">{money(call.cost_usd, 4)}</span>
          <span className="unit">latency</span>
          <span className="num text-ash">{duration(call.latency_ms / 1000)}</span>
        </span>
      }
    >
      {call.rationale ? (
        <p className="text-xs text-chalk mb-2 leading-relaxed">{call.rationale}</p>
      ) : null}
      <div className="grid md:grid-cols-2 gap-2">
        <figure className="m-0">
          <figcaption className="unit mb-1">prompt</figcaption>
          <pre className="mono text-xs text-ash bg-graphite border border-slate rounded p-2 m-0 max-h-64 overflow-auto whitespace-pre-wrap">
            {asText(call.prompt)}
          </pre>
        </figure>
        <figure className="m-0">
          <figcaption className="unit mb-1">response</figcaption>
          <pre className="mono text-xs text-chalk bg-graphite border border-slate rounded p-2 m-0 max-h-64 overflow-auto whitespace-pre-wrap">
            {asText(call.response)}
          </pre>
        </figure>
      </div>
    </Section>
  );
}
