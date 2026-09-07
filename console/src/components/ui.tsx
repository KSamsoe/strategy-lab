/**
 * Shared primitives.
 *
 * Small on purpose. The console's visual identity lives in the token layer and
 * in how data is set, not in a component library, so these mostly exist to stop
 * eight screens inventing eight slightly different table headers.
 */

import type { ReactNode } from "react";
import { MISSING, age as fmtAge, toneClass } from "../lib/format";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// --- surfaces ----------------------------------------------------------------

export function Panel({
  title,
  right,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cx(
        "bg-gunmetal border border-slate rounded-md overflow-hidden flex flex-col min-h-0",
        className,
      )}
    >
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 px-3 h-8 shrink-0 border-b border-slate">
          <h2 className="unit text-chalk/90">{title}</h2>
          {right}
        </header>
      )}
      <div className={cx("min-h-0 flex-1", bodyClassName)}>{children}</div>
    </section>
  );
}

/** The one big number on a card. */
export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: number | null;
}) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2">
      <span className="unit">{label}</span>
      <span
        className={cx(
          "mono text-2xl leading-none",
          tone === undefined ? "text-chalk" : toneClass(tone),
        )}
      >
        {value}
      </span>
      {sub ? <span className="text-xs text-ash-dim mono">{sub}</span> : null}
    </div>
  );
}

// --- state -------------------------------------------------------------------

export type Health = "ok" | "warn" | "bad" | "idle";

const DOT: Record<Health, string> = {
  ok: "bg-moss",
  warn: "bg-amber",
  bad: "bg-ember",
  idle: "bg-ash-dim",
};

/**
 * Health dot plus an explicit age. The pairing is the point: "alpaca ● 2s"
 * says the connection is up *and* how long ago we last heard from it.
 * Freshness is displayed, never assumed.
 */
export function StatusDot({
  health,
  label,
  ageSeconds,
  detail,
}: {
  health: Health;
  label: string;
  ageSeconds?: number | null;
  detail?: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap" title={detail}>
      <span className={cx("size-1.5 rounded-full shrink-0", DOT[health])} aria-hidden />
      <span className="text-ash">{label}</span>
      {ageSeconds !== undefined && ageSeconds !== null ? (
        <span className="mono text-ash-dim">{fmtAge(ageSeconds)}</span>
      ) : null}
    </span>
  );
}

export function Badge({
  children,
  kind = "neutral",
  title,
}: {
  children: ReactNode;
  kind?: "neutral" | "good" | "warn" | "bad" | "accent";
  title?: string;
}) {
  const styles = {
    neutral: "border-slate text-ash",
    good: "border-moss/40 text-moss bg-moss-wash",
    warn: "border-amber/40 text-amber bg-amber-wash",
    bad: "border-ember/40 text-ember bg-ember-wash",
    accent: "border-teal/40 text-teal bg-teal-wash",
  }[kind];
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center h-4 px-1.5 rounded border text-micro uppercase tracking-wider",
        styles,
      )}
    >
      {children}
    </span>
  );
}

/**
 * A warning the reader must not be able to skim past. Reserved for the two
 * claims that would otherwise make a report quietly dishonest: optimistic fills
 * and training-data contamination.
 */
export function LoudWarning({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 border border-amber/40 bg-amber-wash rounded px-3 py-2 text-amber">
      <span aria-hidden className="mono">
        !
      </span>
      <div className="text-xs leading-relaxed">{children}</div>
    </div>
  );
}

// --- empties and errors ------------------------------------------------------

/**
 * An empty state says what to RUN to fill it. A blank panel that just says
 * "no data" wastes the one moment the user is actually asking what to do next.
 */
export function Empty({ children, cmd }: { children: ReactNode; cmd?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-ash-dim text-xs">
      <span>{children}</span>
      {cmd ? (
        <code className="mono text-teal bg-graphite border border-slate rounded px-2 py-1">
          {cmd}
        </code>
      ) : null}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="m-3 border border-ember/40 bg-ember-wash rounded px-3 py-2 text-ember text-xs">
      {msg}
    </div>
  );
}

export function Loading({ what = "loading" }: { what?: string }) {
  return <div className="py-8 text-center text-ash-dim text-xs">{what}…</div>;
}

// --- tables ------------------------------------------------------------------

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className="scroll-x h-full">
      <table className={cx("w-full border-collapse text-sm", className)}>{children}</table>
    </div>
  );
}

export function Th({
  children,
  align = "left",
  className,
  onClick,
  sorted,
}: {
  children: ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
  onClick?: () => void;
  sorted?: "asc" | "desc" | null;
}) {
  return (
    <th
      onClick={onClick}
      className={cx(
        "sticky top-0 z-10 bg-gunmetal border-b border-slate px-2 h-7 unit font-medium whitespace-nowrap",
        align === "right" && "text-right",
        align === "center" && "text-center",
        onClick && "cursor-pointer hover:text-chalk select-none",
        className,
      )}
    >
      {children}
      {sorted ? <span className="ml-1 text-teal">{sorted === "asc" ? "▲" : "▼"}</span> : null}
    </th>
  );
}

export function Td({
  children,
  align = "left",
  className,
  mono,
  title,
}: {
  children: ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
  mono?: boolean;
  title?: string;
}) {
  return (
    <td
      title={title}
      className={cx(
        "px-2 border-b border-slate/50 whitespace-nowrap",
        align === "right" && "text-right",
        align === "center" && "text-center",
        mono && "mono",
        className,
      )}
      style={{ height: "var(--row-h)" }}
    >
      {children ?? MISSING}
    </td>
  );
}

export function Tr({
  children,
  selected,
  onClick,
  className,
  pulse,
}: {
  children: ReactNode;
  selected?: boolean;
  onClick?: () => void;
  className?: string;
  pulse?: boolean;
}) {
  return (
    <tr
      onClick={onClick}
      className={cx(
        onClick && "cursor-pointer",
        selected ? "bg-teal-wash" : onClick && "hover:bg-slate/40",
        pulse && "pulse",
        className,
      )}
    >
      {children}
    </tr>
  );
}

// --- controls ----------------------------------------------------------------

export function Button({
  children,
  onClick,
  kind = "quiet",
  disabled,
  title,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: "quiet" | "danger" | "accent";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
}) {
  const styles = {
    quiet: "border-slate text-ash hover:text-chalk hover:border-slate-hi",
    danger: "border-ember/50 text-ember hover:bg-ember-wash",
    accent: "border-teal/50 text-teal hover:bg-teal-wash",
  }[kind];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cx(
        "inline-flex items-center gap-1.5 h-6 px-2 rounded border text-xs transition-colors",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        styles,
      )}
    >
      {children}
    </button>
  );
}
