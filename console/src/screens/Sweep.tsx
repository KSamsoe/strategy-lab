/**
 * Sweep — the screen whose job is to make a good number look as suspicious as
 * it actually is.
 *
 * Three things here are deliberate and all three are about the same failure:
 *
 * 1. The heatmap colours by the OUT-OF-SAMPLE metric by default. An in-sample
 *    heatmap and an out-of-sample heatmap are pixel-identical in form and
 *    opposite in meaning — one shows where the model fit, the other shows where
 *    it generalized — so the colour source is stated in the panel header at
 *    full size, and switching it to in-sample turns the label amber.
 * 2. The attempt counter is the first number on the page, not a footnote. The
 *    best of 400 configurations is a sample maximum; presenting it the same way
 *    as the best of 4 is the quiet lie this whole screen exists to prevent.
 * 3. A capped grid says so. A silently truncated sweep reads as "we tried
 *    everything and this won", which is exactly what it is not.
 *
 * And a 1-D or 3-D grid gets a table. Folding three axes onto two makes a
 * picture that is easy to read and wrong; a sortable table is neither.
 */

import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { api } from "../lib/api";
import { EChart } from "../components/charts/EChart";
import {
  METRIC_BY_KEY,
  METRIC_COLUMNS,
  goodness,
  ink,
  metricValue,
  type MetricColumn,
} from "./Compare";
import type { Metrics, SweepResult, SweepRunRow, WalkForwardWindow } from "../lib/types";
import { MISSING, count, day, num, ratio, shortId, titleize, toneClass } from "../lib/format";
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
} from "../components/ui";

/** Same thresholds the run browser uses: one definition of "searched hard". */
const ATTEMPT_WARN = 25;
const ATTEMPT_LOUD = 100;

export type Source = "out_of_sample" | "in_sample";

export const SOURCE_LABEL: Record<Source, string> = {
  out_of_sample: "OOS",
  in_sample: "IN-SAMPLE",
};

/** Which metrics bundle a colour source reads. */
export function metricsFor(row: SweepRunRow, source: Source): Metrics {
  return source === "in_sample" ? row.is_metrics : row.oos_metrics;
}

/**
 * Grid axes that actually vary. A parameter pinned to one value is a constant,
 * not a dimension, and counting it would turn a 2-D sweep into a "3-D" one that
 * falls back to a table for no reason.
 */
export function gridAxes(grid: Record<string, unknown[]>): { key: string; values: unknown[] }[] {
  return Object.entries(grid)
    .filter(([, values]) => Array.isArray(values) && values.length > 1)
    .map(([key, values]) => ({ key, values: orderValues(values) }));
}

/** Numeric axes must be in numeric order or the surface is a lie about shape. */
export function orderValues(values: unknown[]): unknown[] {
  const nums = values.every((v) => typeof v === "number" && Number.isFinite(v));
  return nums ? [...(values as number[])].sort((a, b) => a - b) : [...values];
}

export const cellKey = (a: unknown, b: unknown): string => `${String(a)}|${String(b)}`;

/** Index the sweep's runs by their position on the two varying axes. */
export function indexCells(
  runs: readonly SweepRunRow[],
  xKey: string,
  yKey: string,
): Map<string, SweepRunRow> {
  const m = new Map<string, SweepRunRow>();
  for (const r of runs) m.set(cellKey(r.params[xKey], r.params[yKey]), r);
  return m;
}

export default function Sweep() {
  const { sweepId } = useParams({ from: "/sweeps/$sweepId" });
  return <SweepView sweepId={sweepId} />;
}

export function SweepView({ sweepId }: { sweepId: string }) {
  const navigate = useNavigate();

  // A finished sweep is as immutable as the runs inside it.
  const q = useQuery({
    queryKey: ["sweeps", sweepId],
    queryFn: () => api.sweep(sweepId),
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });

  const data = q.data ?? null;

  // Deliberate default, and not read from `ranked_on`: if the sweep ranked
  // itself in-sample, the honest first view is still the out-of-sample surface.
  const [source, setSource] = useState<Source>("out_of_sample");
  const [metric, setMetric] = useState<string | null>(null);

  const metricKey = metric ?? data?.metric ?? "sharpe";
  const axes = useMemo(() => gridAxes(data?.grid ?? {}), [data]);

  // If nothing ran out of sample there is no OOS surface to draw, and drawing
  // the in-sample one under an "OOS" label would be the exact confusion this
  // screen is built to prevent.
  const oosAvailable = useMemo(
    () => (data?.runs ?? []).some((r) => metricValue(r.oos_metrics, metricKey) !== null),
    [data, metricKey],
  );
  const effective: Source = source === "out_of_sample" && !oosAvailable ? "in_sample" : source;

  if (q.isPending) return <Loading what="reading sweep" />;
  if (q.error) {
    return (
      <div className="p-3">
        <Panel title="sweep">
          <ErrorNote error={q.error} />
          <Empty cmd="lab sweep strategies/momo.py --grid cfg/momo_grid.yaml --json">
            no sweep with id {shortId(sweepId, 12)}
          </Empty>
        </Panel>
      </div>
    );
  }
  if (!data) return <Empty cmd="lab sweep strategies/momo.py --grid cfg/momo_grid.yaml">no sweep</Empty>;

  return (
    <div className="p-3 flex flex-col gap-3">
      <SweepHeader data={data} sweepId={sweepId} />

      {data.truncated?.applied ? (
        <LoudWarning>
          <strong className="text-chalk">This grid was capped.</strong> The sweep requested{" "}
          <span className="mono">{count(data.truncated.requested)}</span> configurations and ran{" "}
          <span className="mono">{count(data.truncated.ran)}</span>. Every surface below is a
          slice of the grid, not the grid — the best cell shown is the best of what ran, and
          the cells that were never tried are not blank, they are unknown.
        </LoudWarning>
      ) : null}

      {data.ranked_on === "in_sample" ? (
        <LoudWarning>
          <strong className="text-chalk">Ranked in-sample.</strong> The winning cell was
          picked on the metric measured over the same data the parameters were fitted to, so
          "best" here means "fit this history hardest". That is the failure walk-forward
          exists to detect, not a result — re-rank on the out-of-sample column before taking
          any of it forward.
        </LoudWarning>
      ) : null}

      {axes.length === 2 ? (
        <Heatmap
          data={data}
          axes={axes}
          metricKey={metricKey}
          source={effective}
          requested={source}
          oosAvailable={oosAvailable}
          onSource={setSource}
          onMetric={setMetric}
          onOpen={(runId) => void navigate({ to: "/runs/$runId", params: { runId } })}
        />
      ) : (
        <Panel title={`parameter grid · ${count(axes.length)}-D`}>
          <p className="px-3 py-2 text-xs text-ash-dim">
            {axes.length === 1
              ? "A one-axis sweep is a list, not a surface — a heatmap would be a single row of colour carrying no more information than the column below."
              : `${count(axes.length)} axes vary. Projecting them onto two would average away whichever axis got collapsed and draw a surface the sweep never measured; the ranked table below is the honest form.`}
          </p>
        </Panel>
      )}

      <WalkForward windows={data.walk_forward} metricKey={metricKey} />

      <Results
        data={data}
        axes={axes}
        metricKey={metricKey}
        source={effective}
        onMetric={setMetric}
      />
    </div>
  );
}

// --- header -------------------------------------------------------------------

function SweepHeader({ data, sweepId }: { data: SweepResult; sweepId: string }) {
  const attempts = data.attempts || data.runs.length;
  const warn = attempts >= ATTEMPT_WARN;
  const loud = attempts >= ATTEMPT_LOUD;
  const axes = gridAxes(data.grid);

  return (
    <header className="flex flex-wrap items-stretch gap-x-6 gap-y-2 bg-gunmetal border border-slate rounded-md">
      {/* The attempt counter is the first thing on the page, at tile size. */}
      <div
        className={cx("shrink-0", warn && "bg-amber-wash")}
        title={
          warn
            ? "past this many attempts the top of the distribution is mostly sampling luck"
            : "configurations recorded for this sweep"
        }
      >
        <Stat
          label="attempts"
          value={<span className={warn ? "text-amber" : undefined}>{count(attempts)}</span>}
          sub={loud ? "treat the best cell as a sample maximum" : "configurations run"}
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 min-w-0">
        <span className="flex items-baseline gap-2">
          <span className="unit">sweep</span>
          <span className="mono text-chalk" title={sweepId}>
            {shortId(sweepId, 12)}
          </span>
        </span>
        <span className="mono text-chalk">{data.strategy}</span>
        <span className="flex items-baseline gap-2">
          <span className="unit">ranked on</span>
          {data.ranked_on === "in_sample" ? (
            <Badge kind="warn">in-sample</Badge>
          ) : (
            <Badge kind="good">out-of-sample</Badge>
          )}
        </span>
        <span className="flex items-baseline gap-2">
          <span className="unit">metric</span>
          <span className="mono text-ash">{data.metric}</span>
        </span>
        {data.truncated?.applied ? (
          <Badge kind="warn" title="the grid was capped before it finished">
            capped {count(data.truncated.ran)}/{count(data.truncated.requested)}
          </Badge>
        ) : null}
        <span className="flex items-baseline gap-2">
          <span className="unit">axes</span>
          <span className="mono text-ash">
            {axes.length === 0 ? MISSING : axes.map((a) => a.key).join(" × ")}
          </span>
        </span>
        {data.best ? (
          <Link
            to="/runs/$runId"
            params={{ runId: data.best.run_id }}
            className="mono text-xs text-teal no-underline hover:underline"
          >
            best {shortId(data.best.run_id, 8)} ↗
          </Link>
        ) : null}
      </div>
    </header>
  );
}

// --- heatmap ------------------------------------------------------------------

function Heatmap({
  data,
  axes,
  metricKey,
  source,
  requested,
  oosAvailable,
  onSource,
  onMetric,
  onOpen,
}: {
  data: SweepResult;
  axes: { key: string; values: unknown[] }[];
  metricKey: string;
  source: Source;
  requested: Source;
  oosAvailable: boolean;
  onSource: (s: Source) => void;
  onMetric: (m: string) => void;
  onOpen: (runId: string) => void;
}) {
  const [x, y] = axes;
  const cells = useMemo(() => indexCells(data.runs, x.key, y.key), [data.runs, x.key, y.key]);
  const col = METRIC_BY_KEY.get(metricKey);

  const points = useMemo(() => {
    const out: { xi: number; yi: number; value: number; runId: string }[] = [];
    x.values.forEach((xv, xi) => {
      y.values.forEach((yv, yi) => {
        const row = cells.get(cellKey(xv, yv));
        if (!row) return;
        const v = metricValue(metricsFor(row, source), metricKey);
        if (v === null) return;
        out.push({ xi, yi, value: v, runId: row.run_id });
      });
    });
    return out;
  }, [cells, x.values, y.values, source, metricKey]);

  const missing = x.values.length * y.values.length - points.length;
  const option = useMemo(
    () =>
      heatmapOption(
        points,
        x.values.map(String),
        y.values.map(String),
        col ?? null,
        `${SOURCE_LABEL[source]} ${col?.label ?? metricKey}`,
      ),
    [points, x.values, y.values, col, source, metricKey],
  );

  const isSample = source === "in_sample";

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          parameter heatmap
          <span className="mono normal-case tracking-normal text-ash">
            {x.key} × {y.key}
          </span>
        </span>
      }
      right={
        <div className="flex items-center gap-2">
          {/*
            The colour source, stated at label size rather than hidden inside a
            select. Two identical-looking surfaces mean opposite things and the
            reader has to know which one is on screen without going looking.
          */}
          <span className="unit">colour =</span>
          <Badge kind={isSample ? "warn" : "good"} title="the metric this surface is painted with">
            {SOURCE_LABEL[source]} {col?.label ?? metricKey}
          </Badge>
          <select
            value={requested}
            onChange={(e) => onSource(e.target.value as Source)}
            aria-label="colour source"
            className="mono h-5 rounded border border-slate bg-graphite px-1 text-xs text-chalk"
          >
            <option value="out_of_sample">out-of-sample</option>
            <option value="in_sample">in-sample</option>
          </select>
          <select
            value={metricKey}
            onChange={(e) => onMetric(e.target.value)}
            aria-label="colour metric"
            className="mono h-5 rounded border border-slate bg-graphite px-1 text-xs text-chalk"
          >
            {METRIC_COLUMNS.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
      }
    >
      {isSample ? (
        <p className="px-3 py-1.5 border-b border-slate text-xs text-amber">
          Painted with the in-sample metric: this shows where the parameters fitted the data
          they were chosen on, which says nothing about what any of them will do next.
          {requested === "out_of_sample" && !oosAvailable
            ? " No out-of-sample metric was recorded for this sweep, so there is no OOS surface to show."
            : ""}
        </p>
      ) : null}

      {points.length === 0 ? (
        <Empty cmd={`lab sweep --resume ${shortId(data.sweep_id, 12)} --json`}>
          no cell reports {col?.label ?? metricKey} on the {SOURCE_LABEL[source]} side
        </Empty>
      ) : (
        <div data-chart="sweep-heatmap" data-source={source} data-metric={metricKey}>
          <EChart
            option={option}
            height={Math.max(200, 40 + y.values.length * 26)}
            onEvent={{
              click: (params: unknown) => {
                const p = params as { data?: unknown };
                const arr = Array.isArray(p.data) ? p.data : null;
                const id = arr && typeof arr[3] === "string" ? arr[3] : null;
                if (id) onOpen(id);
              },
            }}
          />
        </div>
      )}

      {missing > 0 ? (
        <p className="px-3 py-1.5 border-t border-slate text-xs text-amber">
          {count(missing)} of {count(x.values.length * y.values.length)} cells have no{" "}
          {SOURCE_LABEL[source]} value. They are blank because they were never run or never
          scored, not because they scored zero.
        </p>
      ) : null}
    </Panel>
  );
}

function heatmapOption(
  points: { xi: number; yi: number; value: number; runId: string }[],
  xLabels: string[],
  yLabels: string[],
  col: MetricColumn | null,
  title: string,
): unknown {
  const c = ink();
  // One ramp, always oriented worse → better. `goodness` is what flips it for
  // the metrics that read backwards, so a deep drawdown never paints green.
  const scored = points.map((p) => ({ ...p, g: goodness(col ?? undefined, p.value) }));
  const gs = scored.map((p) => p.g);
  const lo = gs.length > 0 ? Math.min(...gs) : 0;
  const hi = gs.length > 0 ? Math.max(...gs) : 1;

  return {
    animation: false,
    grid: { left: 90, right: 70, top: 10, bottom: 42 },
    tooltip: {
      trigger: "item",
      formatter: (p: { data: unknown[] }) =>
        `${title}<br/>${xLabels[Number(p.data[0])]} × ${yLabels[Number(p.data[1])]} = ${num(
          Number(p.data[2]),
          3,
        )}`,
    },
    xAxis: { type: "category", data: xLabels, splitArea: { show: true } },
    yAxis: { type: "category", data: yLabels, splitArea: { show: true } },
    visualMap: {
      min: lo,
      max: hi === lo ? lo + 1 : hi,
      calculable: false,
      orient: "vertical",
      right: 4,
      top: "middle",
      itemWidth: 10,
      text: ["better", "worse"],
      textStyle: { color: c.ash, fontFamily: "IBM Plex Mono", fontSize: 10 },
      inRange: { color: [c.ember, c.slate, c.moss] },
      // Colour by the goodness projection, print the raw metric in the label.
      dimension: 4,
    },
    series: [
      {
        type: "heatmap",
        data: scored.map((p) => [p.xi, p.yi, p.value, p.runId, p.g]),
        label: {
          show: true,
          color: c.chalk,
          fontFamily: "IBM Plex Mono",
          fontSize: 10,
          formatter: (p: { data: unknown[] }) => num(Number(p.data[2]), 2),
        },
        itemStyle: { borderColor: c.slate, borderWidth: 1 },
      },
    ],
  };
}

// --- walk forward -------------------------------------------------------------

function WalkForward({
  windows,
  metricKey,
}: {
  windows: WalkForwardWindow[];
  metricKey: string;
}) {
  const col = METRIC_BY_KEY.get(metricKey);
  const rows = windows.filter((w) => w.is_metrics || w.oos_metrics);

  const option = useMemo(
    () => walkForwardOption(rows, metricKey),
    [rows, metricKey],
  );

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          walk-forward windows
          <span className="normal-case tracking-normal text-ash-dim">
            in-sample vs out-of-sample {col?.label ?? metricKey}, per window
          </span>
        </span>
      }
    >
      {rows.length === 0 ? (
        <Empty cmd="lab sweep strategies/momo.py --walk-forward 4 --json">
          this sweep was not run walk-forward, so every number in it is in-sample
        </Empty>
      ) : (
        <>
          <div data-chart="walk-forward">
            <EChart option={option} height={200} />
          </div>
          <div className="scroll-x border-t border-slate">
            <Table>
              <thead>
                <tr>
                  <Th>#</Th>
                  <Th>in-sample</Th>
                  <Th>out-of-sample</Th>
                  <Th align="right">is {col?.label ?? metricKey}</Th>
                  <Th align="right">oos {col?.label ?? metricKey}</Th>
                  <Th align="right">decay</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((w) => {
                  const isV = metricValue(w.is_metrics, metricKey);
                  const oosV = metricValue(w.oos_metrics, metricKey);
                  const decay = isV === null || oosV === null ? null : oosV - isV;
                  return (
                    <Tr key={w.index}>
                      <Td mono>{count(w.index)}</Td>
                      <Td mono className="text-ash">
                        {day(w.is_start)} → {day(w.is_end)}
                      </Td>
                      <Td mono className="text-ash">
                        {day(w.oos_start)} → {day(w.oos_end)}
                      </Td>
                      {/* The in-sample number is not a result, so it gets no
                          P&L colour: colouring it green would award the fit. */}
                      <Td align="right" mono className="text-ash">
                        {ratio(isV)}
                      </Td>
                      <Td align="right" mono className={toneClass(oosV)}>
                        {ratio(oosV)}
                      </Td>
                      <Td align="right" mono className={toneClass(decay)}>
                        {decay === null ? MISSING : num(decay, 2)}
                      </Td>
                    </Tr>
                  );
                })}
              </tbody>
            </Table>
          </div>
        </>
      )}
    </Panel>
  );
}

function walkForwardOption(rows: WalkForwardWindow[], metricKey: string): unknown {
  const c = ink();
  const labels = rows.map((w) => `w${w.index}`);
  const isVals = rows.map((w) => metricValue(w.is_metrics, metricKey));
  const oosVals = rows.map((w) => metricValue(w.oos_metrics, metricKey));

  return {
    animation: false,
    grid: { left: 48, right: 12, top: 24, bottom: 24 },
    tooltip: { trigger: "axis" },
    legend: {
      top: 0,
      textStyle: { color: c.ash, fontFamily: "IBM Plex Mono", fontSize: 10 },
      data: ["in-sample", "out-of-sample"],
    },
    xAxis: { type: "category", data: labels },
    yAxis: { type: "value", scale: true },
    series: [
      {
        name: "in-sample",
        type: "bar",
        data: isVals,
        itemStyle: { color: c.ashDim },
      },
      {
        name: "out-of-sample",
        type: "bar",
        // The OOS bar is the only one that earns a P&L hue.
        data: oosVals.map((v) => ({
          value: v,
          itemStyle: { color: v === null ? c.slate : v >= 0 ? c.moss : c.ember },
        })),
      },
    ],
  };
}

// --- results table --------------------------------------------------------------

type Dir = "asc" | "desc";

function Results({
  data,
  axes,
  metricKey,
  source,
  onMetric,
}: {
  data: SweepResult;
  axes: { key: string; values: unknown[] }[];
  metricKey: string;
  source: Source;
  onMetric: (m: string) => void;
}) {
  const [sortKey, setSortKey] = useState<string>("__score");
  const [dir, setDir] = useState<Dir>("desc");
  const col = METRIC_BY_KEY.get(metricKey);
  const paramKeys = axes.length > 0 ? axes.map((a) => a.key) : Object.keys(data.grid);

  const rows = useMemo(() => {
    const out = [...data.runs];
    out.sort((a, b) => {
      const av = sortField(a, sortKey, metricKey, source);
      const bv = sortField(b, sortKey, metricKey, source);
      const c =
        typeof av === "string" || typeof bv === "string"
          ? String(av).localeCompare(String(bv))
          : av - bv;
      return dir === "asc" ? c : -c;
    });
    return out;
  }, [data.runs, sortKey, dir, metricKey, source]);

  const sortBy = (key: string) => {
    if (key === sortKey) setDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setDir("desc");
    }
  };
  const sortedAs = (key: string): Dir | null => (key === sortKey ? dir : null);

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          every cell
          <span className="mono normal-case tracking-normal text-ash-dim">
            {count(data.runs.length)} runs
          </span>
        </span>
      }
      right={
        <label className="flex items-center gap-1.5">
          <span className="unit">metric</span>
          <select
            value={metricKey}
            onChange={(e) => onMetric(e.target.value)}
            aria-label="table metric"
            className="mono h-5 rounded border border-slate bg-graphite px-1 text-xs text-chalk"
          >
            {METRIC_COLUMNS.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      }
      bodyClassName="max-h-[520px] overflow-y-auto"
    >
      {data.runs.length === 0 ? (
        <Empty cmd="lab sweep strategies/momo.py --grid cfg/momo_grid.yaml --json">
          the sweep recorded no runs
        </Empty>
      ) : (
        <Table>
          <thead>
            <tr>
              <Th>run</Th>
              {paramKeys.map((k) => (
                <Th key={k} onClick={() => sortBy(`p:${k}`)} sorted={sortedAs(`p:${k}`)}>
                  {titleize(k)}
                </Th>
              ))}
              {/* In-sample first and deliberately dim: it is the number you are
                  not supposed to be shopping on. */}
              <Th
                align="right"
                className="text-ash-dim"
                onClick={() => sortBy("__is")}
                sorted={sortedAs("__is")}
              >
                is {col?.label ?? metricKey}
              </Th>
              <Th align="right" onClick={() => sortBy("__oos")} sorted={sortedAs("__oos")}>
                oos {col?.label ?? metricKey}
              </Th>
              <Th align="right" onClick={() => sortBy("__score")} sorted={sortedAs("__score")}>
                score
              </Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isV = metricValue(r.is_metrics, metricKey);
              const oosV = metricValue(r.oos_metrics, metricKey);
              const best = data.best?.run_id === r.run_id;
              return (
                <Tr key={r.run_id} selected={best}>
                  <Td mono title={r.run_id}>
                    <Link
                      to="/runs/$runId"
                      params={{ runId: r.run_id }}
                      className="text-teal no-underline hover:underline"
                    >
                      {shortId(r.run_id, 8)}
                    </Link>
                    {best ? (
                      <span className="ml-2">
                        <Badge kind="accent">best</Badge>
                      </span>
                    ) : null}
                  </Td>
                  {paramKeys.map((k) => (
                    <Td key={k} mono className="text-chalk">
                      {String(r.params[k] ?? MISSING)}
                    </Td>
                  ))}
                  <Td align="right" mono className="text-ash-dim">
                    {ratio(isV)}
                  </Td>
                  <Td
                    align="right"
                    mono
                    className={cx(source === "out_of_sample" && "text-chalk", toneClass(oosV))}
                  >
                    {ratio(oosV)}
                  </Td>
                  <Td align="right" mono>
                    {num(r.score, 3)}
                  </Td>
                </Tr>
              );
            })}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

function sortField(
  r: SweepRunRow,
  key: string,
  metricKey: string,
  source: Source,
): number | string {
  if (key === "__score") return Number.isFinite(r.score) ? r.score : Number.NEGATIVE_INFINITY;
  if (key === "__is") return metricValue(r.is_metrics, metricKey) ?? Number.NEGATIVE_INFINITY;
  if (key === "__oos") return metricValue(r.oos_metrics, metricKey) ?? Number.NEGATIVE_INFINITY;
  if (key.startsWith("p:")) {
    const v = r.params[key.slice(2)];
    return typeof v === "number" && Number.isFinite(v) ? v : String(v ?? "");
  }
  return metricValue(metricsFor(r, source), metricKey) ?? Number.NEGATIVE_INFINITY;
}
