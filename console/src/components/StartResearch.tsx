/**
 * The launch form.
 *
 * The console's rule is that it can only make the system safer, and the risk it
 * names is market exposure — which a research session cannot create. But it does
 * spend budget and, in freeform mode, execute model-authored Python, so starting
 * is opt-in server-side. When it is off this renders the reason and the command
 * to enable it rather than a disabled button with no explanation.
 *
 * Two things it deliberately cannot do: author a backtest config (it picks one
 * that already exists, because composing `limits` here would be raising a limit
 * from the UI), and exceed the server's ceilings on time, spend and calls.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api";
import type { ResearchLaunchOptions } from "../lib/types";
import { count, money } from "../lib/format";
import { Badge, Button, ErrorNote, Loading, LoudWarning, Panel, cx } from "./ui";

const FIELD =
  "h-6 px-2 bg-graphite border border-slate rounded text-xs text-chalk mono " +
  "focus:border-teal outline-none";

export interface StartResearchProps {
  onStarted?: (sessionId: string) => void;
}

export function StartResearch({ onStarted }: StartResearchProps) {
  const qc = useQueryClient();
  const opts = useQuery({
    queryKey: ["agent", "research", "options"],
    queryFn: api.researchOptions,
    retry: false,
  });

  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({
    config: "",
    brief: "",
    minutes: 30,
    budget_usd: 5,
    max_calls: 60,
    max_experiments: 40,
    metric: "oos_sharpe",
    periods: 4,
    neighbourhood_runs: 16,
    model: "",
    provider: "",
  });

  const start = useMutation({
    mutationFn: () =>
      api.startResearch({
        ...form,
        config: form.config || (opts.data?.configs[0] ?? ""),
        model: form.model || null,
        provider: form.provider || null,
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["agent", "research"] });
      setOpen(false);
      onStarted?.(r.session_id);
    },
  });

  if (opts.isLoading) return <Loading what="launch options" />;
  if (opts.error) return <ErrorNote error={opts.error} />;
  const o = opts.data as ResearchLaunchOptions;

  if (!o.enabled) {
    return (
      <Panel title="start a session">
        <div className="px-3 py-3 flex flex-col gap-2 text-xs text-ash">
          <p className="m-0">{o.reason}</p>
          <code className="mono text-teal bg-graphite border border-slate rounded px-2 py-1 self-start">
            lab ui --allow-research
          </code>
          <p className="m-0 text-ash-dim">
            Research cannot place an order or touch a broker. It can spend model budget and, in
            freeform mode, write and run Python — which is why starting it is a separate
            decision from pause, cancel and kill. Stopping a session never needs this.
          </p>
        </div>
      </Panel>
    );
  }

  const configs = o.configs.filter((c) => !c.includes("grid") && c !== "limits.yaml");
  const providers = o.providers.filter((p) => p.available);
  const lim = o.limits ?? {};

  return (
    <Panel
      title="start a session"
      right={
        <Button kind="accent" onClick={() => setOpen(!open)}>
          {open ? "cancel" : "+ new"}
        </Button>
      }
    >
      {!open ? (
        <p className="px-3 py-2 m-0 text-xs text-ash-dim">
          Hands the agent your strategy folder and a brief. It picks what to try, sees the clock
          and the budget every turn, and stops when it is satisfied or when a bound is reached.
        </p>
      ) : (
        <form
          className="px-3 py-3 flex flex-col gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            start.mutate();
          }}
        >
          {o.kill_switch_engaged ? (
            <LoudWarning>
              The kill switch is engaged ({o.kill_switch_reason}). Starting is refused until it is
              released, which is a CLI action.
            </LoudWarning>
          ) : null}

          <label className="flex flex-col gap-1">
            <span className="unit">brief — what to prioritise, and when to be satisfied</span>
            <textarea
              required
              rows={4}
              value={form.brief}
              onChange={(e) => setForm({ ...form, brief: e.target.value })}
              placeholder={
                "Beat buy-and-hold out of sample. Consistency across sub-periods matters more " +
                "than peak Sharpe. Be satisfied once you have something positive in 3 of 4 " +
                "periods, or once you can tell me nothing here beats the benchmark."
              }
              className={cx(FIELD, "h-auto py-1.5 leading-relaxed font-sans")}
            />
            <span className="text-micro text-ash-dim">
              This is the whole steering wheel. A vague brief gets a vague stopping point.
              The fitness metric alone will not stop a strategy that hides in cash — if you
              care about return, say so here as well as picking it below.
            </span>
          </label>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <label className="flex flex-col gap-1">
              <span className="unit">config</span>
              <select
                value={form.config || configs[0] || ""}
                onChange={(e) => setForm({ ...form, config: e.target.value })}
                className={FIELD}
              >
                {configs.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1">
              <span className="unit">provider</span>
              <select
                value={form.provider}
                onChange={(e) => setForm({ ...form, provider: e.target.value })}
                className={FIELD}
              >
                <option value="">{`configured (${o.default_provider})`}</option>
                {providers.map((p) => (
                  <option key={String(p.name)} value={String(p.name)}>
                    {String(p.name)} · {String(p.billing)}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1">
              <span className="unit">model</span>
              <input
                value={form.model}
                onChange={(e) => setForm({ ...form, model: e.target.value })}
                placeholder={o.default_model}
                className={FIELD}
              />
            </label>

            <label className="flex flex-col gap-1">
              <span className="unit">fitness metric</span>
              <select
                value={form.metric}
                onChange={(e) => setForm({ ...form, metric: e.target.value })}
                className={FIELD}
              >
                {/* Grouped by what a cash-heavy strategy can do to them. A ratio
                    divides by volatility or drawdown, so sitting out of the
                    market inflates it on almost no return; a return metric
                    cannot be gamed that way but says nothing about risk. */}
                <optgroup label="return — cannot be gamed by sitting in cash">
                  <option value="oos_cagr">oos_cagr — annualised return</option>
                  <option value="oos_total_return">oos_total_return</option>
                </optgroup>
                <optgroup label="risk-adjusted — a cash-heavy book flatters these">
                  <option value="oos_sharpe">oos_sharpe</option>
                  <option value="oos_sortino">oos_sortino</option>
                  <option value="oos_calmar">oos_calmar — return / max drawdown</option>
                </optgroup>
                <optgroup label="consistency — the weakest sub-period">
                  <option value="worst_sharpe">worst_sharpe</option>
                  <option value="worst_cagr">worst_cagr</option>
                </optgroup>
              </select>
            </label>
          </div>

          <fieldset className="border border-slate rounded px-3 pt-1 pb-2 m-0">
            <legend className="unit px-1">bounds — it stops at whichever comes first</legend>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField
                label="minutes"
                value={form.minutes}
                max={Number(lim.max_minutes ?? 180)}
                onChange={(v) => setForm({ ...form, minutes: v })}
              />
              <NumField
                label="budget"
                value={form.budget_usd}
                max={Number(lim.max_budget_usd ?? 25)}
                step={0.5}
                onChange={(v) => setForm({ ...form, budget_usd: v })}
              />
              <NumField
                label="max calls"
                value={form.max_calls}
                max={Number(lim.max_calls ?? 400)}
                onChange={(v) => setForm({ ...form, max_calls: v })}
              />
              <NumField
                label="max experiments"
                value={form.max_experiments}
                max={Number(lim.max_experiments ?? 200)}
                onChange={(v) => setForm({ ...form, max_experiments: v })}
              />
            </div>
            {/* Not a bound like the others — it spends backtests rather than
                capping them, and it is the only check that can tell a plateau
                from a spike. Sits here because it is the one number that costs
                wall clock without costing model budget. */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
              <NumField
                label="neighbourhood runs"
                value={form.neighbourhood_runs}
                max={40}
                onChange={(v) => setForm({ ...form, neighbourhood_runs: v })}
              />
              <p className="text-micro text-ash-dim m-0 col-span-1 md:col-span-3 self-end">
                A ceiling, not a quota: before finishing, the winning parameters are re-run ~10%
                either side of each numeric value — never more than two per parameter, so a
                small strategy costs less. Below full coverage the check says which parameters
                it could only nudge one way. Wall clock, not model budget — 0 disables it.
              </p>
            </div>
            <p className="text-micro text-ash-dim m-0 mt-2">
              Ceilings: {count(Number(lim.max_minutes))} min · {money(Number(lim.max_budget_usd), 0)}{" "}
              · {count(Number(lim.max_calls))} calls. On a subscription nothing reports remaining
              quota, so the call cap is the real guard.
            </p>
            {/* The prompt carries far more diagnostic detail than it used to, and
                input tokens are paid on every call — so the budget, not the call
                cap, is usually what ends a session. Say so here rather than
                letting it stop early and look like it finished. */}
            <p className="text-micro text-ash-dim m-0 mt-1">
              Budget is what usually stops a session first. On Opus a 60-call session runs about{" "}
              <span className="text-ash">$6</span> — roughly {money(0.1, 2)} a call — so a{" "}
              {money(Number(form.budget_usd), 0)} budget buys around{" "}
              {count(Math.floor((Number(form.budget_usd) / 6) * 60))} calls. Sonnet is about 40%
              cheaper.
            </p>
          </fieldset>

          {start.error ? <ErrorNote error={start.error} /> : null}

          <div className="flex items-center gap-3">
            <Button kind="accent" type="submit" disabled={start.isPending || !form.brief.trim()}>
              {start.isPending ? "starting…" : "start session"}
            </Button>
            <span className="text-xs text-ash-dim">
              It runs as a detached process — closing this tab does not stop it.
            </span>
          </div>
        </form>
      )}
    </Panel>
  );
}

function NumField({
  label,
  value,
  onChange,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  max: number;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="unit">{label}</span>
      <input
        type="number"
        min={step}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={cx(FIELD, "num")}
      />
    </label>
  );
}

export function StopSessionButton({ sessionId }: { sessionId: string }) {
  const qc = useQueryClient();
  const stop = useMutation({
    mutationFn: () => api.stopResearch(sessionId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agent", "research"] }),
  });
  return (
    <Button
      kind="danger"
      disabled={stop.isPending}
      onClick={() => {
        if (window.confirm("Stop after the current turn? The session stays resumable.")) {
          stop.mutate();
        }
      }}
    >
      {stop.isPending ? "stopping…" : "stop"}
    </Button>
  );
}

/** Exported for the badge in the sessions table. */
export function RunningBadge() {
  return <Badge kind="accent">running</Badge>;
}
