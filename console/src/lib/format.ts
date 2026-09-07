/**
 * Formatters.
 *
 * One rule runs through all of these: **a number must not change width when it
 * ticks.** Fixed decimal places, explicit signs, and a fixed-width placeholder
 * for missing values, so a live column never jitters sideways. Combined with
 * tabular figures from the token layer, a table of updating values stays as
 * still as a printed one.
 */

const NBSP = " ";
/** Same visual width as a formatted number, so gaps do not collapse a column. */
export const MISSING = "—";

const nf = (min: number, max = min) =>
  new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max });

const f0 = nf(0);
const f1 = nf(1);
const f2 = nf(2);

export function num(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  return nf(decimals).format(v);
}

export function pct(v: number | null | undefined, decimals = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  return `${nf(decimals).format(v * 100)}%`;
}

/** Signed percent, for anything that reads as a change. */
export function delta(v: number | null | undefined, decimals = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  const s = nf(decimals).format(Math.abs(v) * 100);
  return `${v < 0 ? "−" : "+"}${s}%`;
}

export function money(v: number | null | undefined, decimals = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  const abs = Math.abs(v);
  const body = (decimals === 0 ? f0 : nf(decimals)).format(abs);
  return `${v < 0 ? "−" : ""}$${body}`;
}

export function signedMoney(v: number | null | undefined, decimals = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  const body = (decimals === 0 ? f0 : nf(decimals)).format(Math.abs(v));
  return `${v < 0 ? "−" : "+"}$${body}`;
}

export function ratio(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  return f2.format(v);
}

export function count(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  return f0.format(v);
}

export function shares(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  const s = Number.isInteger(v) ? f0.format(Math.abs(v)) : f2.format(Math.abs(v));
  return `${v < 0 ? "−" : "+"}${s}`;
}

// --- time --------------------------------------------------------------------

const timeFmt = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

const dateFmt = new Intl.DateTimeFormat("en-CA", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

export function clock(iso: string | number | Date | null | undefined): string {
  if (!iso) return MISSING;
  return timeFmt.format(new Date(iso));
}

export function day(iso: string | number | Date | null | undefined): string {
  if (!iso) return MISSING;
  return dateFmt.format(new Date(iso));
}

export function stamp(iso: string | number | Date | null | undefined): string {
  if (!iso) return MISSING;
  const d = new Date(iso);
  return `${dateFmt.format(d)}${NBSP}${timeFmt.format(d)}`;
}

/**
 * Compact age, for staleness. The console shows freshness rather than assuming
 * it, so this is on screen constantly and must stay narrow.
 */
export function age(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${f1.format(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m${NBSP}${s}s`;
}

export function days(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return MISSING;
  return `${f0.format(v)}d`;
}

// --- meaning -----------------------------------------------------------------

/**
 * The only place a number becomes a color. Zero is neutral on purpose: a
 * flat position is not a small win, and tinting it green would be a lie the eye
 * reads before the brain does.
 */
export function tone(v: number | null | undefined): "up" | "down" | "flat" {
  if (v === null || v === undefined || !Number.isFinite(v) || v === 0) return "flat";
  return v > 0 ? "up" : "down";
}

export const toneClass = (v: number | null | undefined): string =>
  ({ up: "text-moss", down: "text-ember", flat: "text-ash" })[tone(v)];

/** Short hash display: run ids and commits are for recognition, not reading. */
export function shortId(id: string | null | undefined, n = 8): string {
  if (!id) return MISSING;
  return id.length <= n ? id : id.slice(0, n);
}

export function titleize(s: string): string {
  return s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
