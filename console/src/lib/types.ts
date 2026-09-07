/**
 * The JSON contract, typed.
 *
 * These mirror the FastAPI response models in `lab/api/models.py`, which in turn
 * mirror the CLI's `--json` payloads. That is the point of the architecture: the
 * console is a view over the same contract an agent or a shell script would
 * fetch, not a second API. If a field here has no counterpart in
 * `docs/CONTRACTS.md`, one of the two is wrong.
 */

export type RunKind = "backtest" | "paper" | "live" | "sweep";
export type RunOrigin = "human" | "agent-loop";
export type RunStatus = "running" | "ok" | "error";

export interface RunRecord {
  run_id: string;
  strategy: string;
  kind: RunKind;
  status: RunStatus;
  created_at: string;
  finished_at: string | null;
  git_commit: string | null;
  config_hash: string;
  data_version: string;
  params: Record<string, unknown>;
  config: Record<string, unknown>;
  metrics: Metrics;
  start: string | null;
  end: string | null;
  origin: RunOrigin;
  parent_run_id: string | null;
  sweep_id: string | null;
  notes: string;
  attempt: number;
  error: string;
  /** Predates the trade-ledger fix: trade stats undercount, returns are fine. */
  stale_ledger?: boolean;
  ledger_residual?: number | null;
}

export interface Metrics {
  total_return?: number;
  cagr?: number;
  sharpe?: number;
  sortino?: number;
  calmar?: number;
  volatility?: number;
  max_drawdown?: number;
  max_drawdown_duration_days?: number;
  exposure?: number;
  turnover?: number;
  trades?: number;
  hit_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  profit_factor?: number;
  final_equity?: number;
  starting_equity?: number;
  commission?: number;
  open_positions?: number;
  /** Loud flag: same-bar-close fills let a decision trade at a price it set. */
  optimistic_fills?: boolean;
  fill_mode?: string;
  /** Loud flag: an LLM strategy backtested inside its own training window. */
  contaminated?: boolean;
  attempt?: number;
  warnings?: string[];
  realized_pnl?: number;
  unrealized_pnl?: number;
  /** Non-zero means P&L no trade accounts for. */
  ledger_residual?: number | null;
  /** Present whenever the config named a `benchmark:` and its bars were found. */
  benchmark_total_return?: number;
  benchmark_cagr?: number;
  excess_return?: number;
  alpha?: number;
  beta?: number;
  /** Out-of-sample counterparts, present on walk-forward runs. */
  oos_sharpe?: number;
  oos_total_return?: number;
  is_sharpe?: number;
  [k: string]: unknown;
}

/** Columnar on purpose: the chart wants columns, and the payload gets large. */
export interface EquitySeries {
  run_id: string;
  n: number;
  t: string[];
  equity: (number | null)[];
  drawdown: (number | null)[];
  /** Per-point out-of-sample mask. What the equity chart shades. */
  is_oos: boolean[];
  exposure?: (number | null)[] | null;
  /** Rebased onto the same starting equity; absent when the run had no benchmark. */
  benchmark?: (number | null)[] | null;
}

export interface TradeRow {
  ticker: string;
  side: string;
  qty: number;
  entry_time: string;
  entry_price: number;
  exit_time: string | null;
  exit_price: number | null;
  pnl: number;
  pnl_pct: number;
  bars_held: number;
  commission: number;
  tag: string;
  exit_reason: string;
}

export type GateAction = "pass" | "clipped" | "blocked";

export interface GateVerdict {
  ticker: string;
  action: GateAction;
  requested_pct: number;
  approved_pct: number;
  rule: string | null;
  detail: string;
}

export interface Intent {
  ticker: string;
  target_pct: number;
  tag: string;
  reason: string;
  limit_price: number | null;
  meta: Record<string, unknown>;
}

/** One `on_bar` call, captured whole. The unit the decision tape renders. */
export interface DecisionRow {
  id: string;
  run_id: string;
  strategy: string;
  at: string;
  summary: string;
  inputs: {
    history?: Record<string, { n: number; last: string | null; tail: { t: string; v: number }[] }>;
    indicators?: Record<string, { params: Record<string, unknown>; value: number | null; n: number }>;
    signals?: SignalInput[];
    prices?: Record<string, number | null>;
    bars?: Record<string, { n: number; last: string | null; last_close: number | null }>;
  };
  intents: Intent[];
  verdicts: GateVerdict[];
  order_ids: string[];
  portfolio: PortfolioSnapshot;
  logs: Record<string, unknown>[];
  /** Present only for Pattern-B strategies: the prompt/response pair. */
  agent: AgentCall | null;
  duration_ms: number;
}

export interface SignalInput {
  source: string;
  ticker: string;
  kind: string;
  tier: string | null;
  direction: string | null;
  score: number | null;
  fresh: boolean | null;
  /** When the thing happened in the world. */
  event_time: string;
  /** When we could actually have known it. The gap is the whole point. */
  knowledge_time: string;
  uid: string;
}

export interface PortfolioSnapshot {
  at: string | null;
  cash: number;
  equity: number;
  gross_exposure: number;
  net_exposure: number;
  n_positions: number;
  positions: PositionRow[];
}

export interface PositionRow {
  ticker: string;
  qty: number;
  avg_price: number;
  last_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pct: number;
  opened_at: string | null;
  tag: string;
}

export interface AgentCall {
  model: string;
  prompt: unknown;
  response: unknown;
  rationale?: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
}

export type EventKind =
  | "decision"
  | "order"
  | "fill"
  | "reject"
  | "gate_block"
  | "breaker"
  | "reconcile"
  | "heartbeat"
  | "error"
  | "log"
  | "run_start"
  | "run_end";

export interface JournalEvent {
  seq: number;
  id: string;
  at: string;
  kind: EventKind;
  source: string;
  run_id: string | null;
  strategy: string | null;
  ticker: string | null;
  message: string;
  payload: Record<string, unknown>;
}

export interface LiveStrategy {
  strategy: string;
  mode: "paper" | "live";
  status: "running" | "paused" | "stopped" | "blocked";
  pnl_pct: number;
  equity: number;
  n_positions: number;
  gate_events_today: number;
  next_fire: string | null;
  started_at: string | null;
  spend_usd?: number;
  positions: PositionRow[];
}

/** Freshness is displayed, never assumed -- hence `age_s` on everything. */
export interface Heartbeat {
  source: string;
  at: string;
  age_s: number;
  meta: Record<string, unknown>;
}

export interface AdapterHealth {
  name: string;
  provides: string[];
  available: boolean;
  reason: string;
  quota_used: number | null;
  quota_limit: number | null;
  quota_tier: string | null;
  last_call: string | null;
}

export interface LiveHealth {
  adapters: AdapterHealth[];
  heartbeats: Heartbeat[];
  breaker: { tripped: boolean; reason: string };
  kill_switch: { engaged: boolean; reason: string | null };
  equity: number | null;
  day_change_pct: number | null;
}

export interface SweepResult {
  sweep_id: string;
  strategy: string;
  grid: Record<string, unknown[]>;
  metric: string;
  ranked_on: "out_of_sample" | "in_sample";
  attempts: number;
  truncated: { applied: boolean; requested: number; ran: number } | null;
  runs: SweepRunRow[];
  best: SweepRunRow | null;
  walk_forward: WalkForwardWindow[];
}

export interface SweepRunRow {
  run_id: string;
  params: Record<string, unknown>;
  is_metrics: Metrics;
  oos_metrics: Metrics;
  score: number;
}

export interface WalkForwardWindow {
  index: number;
  is_start: string;
  is_end: string;
  oos_start: string;
  oos_end: string;
  is_metrics?: Metrics;
  oos_metrics?: Metrics;
}

export interface LineageStep {
  n: number;
  run_id: string;
  params: Record<string, unknown>;
  diff: string;
  metrics: Metrics;
  rationale: string;
  cost_usd?: number;
}

export interface StrategyInfo {
  name: string;
  path: string;
  params?: Record<string, unknown>;
  doc?: string;
  source_hash?: string;
  loadable: boolean;
  error?: string;
}

// --- open-ended research sessions -------------------------------------------

export interface ResearchSummary {
  session_id: string;
  status: "running" | "finished" | "stopped";
  stopped_because: string;
  brief: string;
  created_at: string;
  updated_at: string;
  elapsed_minutes: number;
  calls: number;
  spend_usd: number;
  /** "api" = money; "subscription" = notional API-equivalent. */
  billing: string;
  experiments: number;
  best_run_id: string | null;
  best_score: number | null;
  satisfied_with: string | null;
}

export interface ResearchRunRow {
  n: number;
  run_id: string;
  strategy: string;
  params: Record<string, unknown>;
  score: number | null;
  oos_sharpe: number | null;
  oos_return: number | null;
  is_sharpe: number | null;
  worst_period_sharpe: number | null;
  /** Share of the OOS window invested. A cash-heavy strategy flatters its Sharpe. */
  oos_exposure: number | null;
  /** OOS return minus the benchmark's, over the same window. */
  oos_return_vs_market: number | null;
  /**
   * The whole tradeable period, warmup excluded. The OOS slice is the last fifth
   * of the range and answers "does it generalise"; these answer "what did it
   * actually do", and a strategy can win one while losing the other.
   */
  full_return: number | null;
  full_sharpe: number | null;
  full_return_vs_market: number | null;
  /**
   * Which attempt at this strategy this is. Every backtest reads the held-out
   * window, so a 4th variant's out-of-sample score is the best of four looks at
   * the test set — not an independent estimate.
   */
  variant_index: number | null;
  /**
   * One-sigma sampling error on `score`, from the length of the window it was
   * measured over. An annualised Sharpe from ~230 daily bars carries roughly
   * ±1.0 — wider than most parameter sweeps, so two scores closer together than
   * this are the same score however many decimals they are printed to.
   */
  fitness_one_sigma: number | null;
  trades: number | null;
  periods: { period: number; start: string; end: string; sharpe: number | null }[];
  rationale: string;
  ok: boolean;
  error: string;
  /** Pattern-B: refused rather than run, and never eligible as the answer. */
  contaminated: boolean;
}

export interface ResearchDetail extends ResearchSummary {
  verdict: string;
  next_steps: string;
  /**
   * What happened when the winning parameters were perturbed ~10% either side.
   * Part of the claim, not a footnote — a score that does not survive being
   * nudged is a property of the sample, not of the strategy.
   */
  neighbourhood: {
    checked: number;
    fragile?: boolean;
    verdict?: string;
    /** What the check could not reach. Absent coverage is not a clean result. */
    coverage?: {
      parameters: number;
      runs_for_full_coverage: number;
      runs_spent: number;
      complete: boolean;
      one_direction_only: string[];
      not_tested: string[];
    };
    chosen_score?: number | null;
    median_neighbour_score?: number | null;
    score_retention?: number | null;
    full_return_retention?: number | null;
    most_sensitive?: { param: string; from: unknown; to: unknown; costs: number } | null;
  } | null;
  notes: string[];
  market: { instrument?: string; oos?: Metrics; in_sample?: Metrics } | null;
  runs: ResearchRunRow[];
  /** One entry per turn, rejected actions included — the reasoning trail. */
  turns: Record<string, unknown>[];
  workspace: string;
  resumable: boolean;
}

export interface ResearchLaunchOptions {
  enabled: boolean;
  reason: string;
  configs: string[];
  providers: Record<string, unknown>[];
  default_model: string;
  default_provider: string;
  kill_switch_engaged: boolean;
  kill_switch_reason: string | null;
  limits: Record<string, number>;
}

export interface ResearchStartRequest {
  config: string;
  brief: string;
  minutes: number;
  budget_usd: number;
  max_calls: number;
  max_experiments: number;
  metric: string;
  periods: number;
  model?: string | null;
  provider?: string | null;
}

export interface ResearchStartResult {
  ok: boolean;
  session_id: string;
  pid: number | null;
  message: string;
}
