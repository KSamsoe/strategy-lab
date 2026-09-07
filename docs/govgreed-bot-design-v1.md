# GovGreed Signal Bot — v1 Design Doc

**Status:** Draft · **Cadence:** Daily · **Target:** Paper trading first
**Assumptions:** Solo developer, Python, GovGreed free tier at launch, Alpaca paper account for execution. All of these are swappable; they're stated here so the numbers below have something concrete to attach to.

## 1. Goal and scope

Build a small autonomous bot that, once per trading day, pulls composite signals from the GovGreed API, filters them into a candidate list, sizes positions under strict caps, and places paper trades through Alpaca. Every input and decision is logged so that after a burn-in period we can evaluate whether the signals carry any edge at all — measurement infrastructure is the real v1 deliverable; returns are a hypothesis to test.

Out of scope for v1: intraday or high-frequency operation, options, short selling, live capital, and multi-broker support. None of these make sense until the daily loop has proven itself on paper.

## 2. Constraints that shape the design

Two external constraints make almost every architectural decision for us.

**Quota.** The free tier is documented as 100 calls/day for the first week, then 20/day, with a 5 req/sec burst cap (their docs also mention a 250/day figure in one place, so treat exact numbers as in flux). The paid founders tier is 750/day. Consequence: the bot must run on a fixed, small call budget, must read the `X-RateLimit-*` response headers and `meta.quota` envelope field rather than hardcoding limits, and must degrade gracefully when quota runs out mid-run.

**Data cadence.** The underlying data moves slowly: conflict scores rebuild nightly, STOCK Act disclosures can lag the actual trade by up to 45 days, and signal freshness is flagged per-signal. There is nothing to gain from polling more than once a day, which conveniently is also all the quota allows.

A third, softer constraint: the API is in closed beta and the schema is actively changing (their changelog shows fields being renamed and tables quarantined). The client must tolerate unknown fields and log `request_id` on every call so breakage is diagnosable.

## 3. Architecture

```
cron (06:30 ET, weekdays)
   └─> run.py
        ├─ govgreed_client   — auth, envelope parsing, retries, quota tracking
        ├─ signal_engine     — merge + filter signals into candidates
        ├─ portfolio_manager — sizing, caps, exit checks against open positions
        ├─ alpaca_adapter    — idempotent paper orders
        └─ store (SQLite)    — snapshots, decisions, orders, quota log
             └─ alerts (email/Discord webhook) on errors, fills, circuit-breaker
```

**govgreed_client.** A thin wrapper along the lines of the ~30-line reference implementation in their docs: base URL `https://www.govgreed.com/api/v1`, key from `GOVGREED_API_KEY` env var sent as a Bearer token, unwrap the `{data, meta}` envelope, and persist `meta.request_id` plus quota fields on every call. Errors arrive as RFC 7807 problem+json; the client maps the documented codes to behavior (see §7). The key is shown once at issuance, so it lives only in the environment / a secrets file that never enters version control.

**signal_engine.** Merges the three signal-bearing endpoints into one candidate table per run, then applies filters: minimum composite tier, freshness required, herd corroboration or insider corroboration required, and a liquidity floor checked against Alpaca's asset data so we never act on a microcap that happens to appear in a political filing.

**portfolio_manager.** Pure function of (candidates, open positions, config). Enforces max position size as a % of equity, max concurrent positions, a per-sector concentration cap, exit rules, and a cool-down that blocks re-entering a ticker for N days after an exit.

**alpaca_adapter.** Places paper orders keyed by `(date, ticker, side)` so a crashed-and-rerun job can't double-order. Market-on-open for v1; limit orders are a later refinement.

**store.** SQLite is enough. It runs anywhere, backs up as one file, and the volumes are tiny.

## 4. Daily API call budget

The whole loop must fit inside the steady-state free tier with headroom for retries. Draft budget:

| Step | Endpoint | Calls/day |
|---|---|---|
| Composite signals | `GET /signals/top?tier=A&fresh=true&limit=25` | 1 |
| Herd corroboration | `GET /herd-signals?days=30` | 1 |
| Forward predictions | `GET /predictions/top?tier=A&status=ACTIVE` | 1 |
| Enrich top candidates | `GET /companies/{ticker}/insider-signal` × top 5 | 5 |
| Account/usage check | `GET /me/usage` (weekly, amortized) | ~0.2 |
| Retry buffer | — | 3 |
| **Total** | | **~11** |

That fits the 20/day steady state. During the first week's 100/day window, the surplus goes to one-time exploration: `GET /atlas` to map the catalog, and spot-checks of `/bills/{n}/timeline` and `/sectors/{sector}/positioning` to decide whether they earn a slot in the daily loop later. Upgrading to the founders tier (750/day) would unlock per-candidate bill timelines, sector positioning, and the Kalshi overlay as daily inputs — deferred until the evaluation in §8 justifies spending anything.

## 5. Data model

```
signals_raw   (run_date, endpoint, response_json)      -- full snapshot, verbatim
candidates    (run_date, ticker, composite_tier, herd_tier,
               insider_score, direction, action, reason_text)
positions     (ticker, side, qty, entry_date, entry_px, exit_date, exit_px, exit_reason)
orders        (order_key, run_date, ticker, side, qty, status, alpaca_id)
quota_log     (run_date, calls_used, limit_reported, tier)
```

`signals_raw` matters more than it looks: historical backfill is an institutional-tier feature, so the API gives us no cheap way to backtest signal history. Snapshotting every response from day one builds our own backtest dataset for free. This is also why M1 in the milestone plan starts logging before any trading logic exists.

## 6. Strategy v1 — parameters, not convictions

Entry: a ticker is a candidate when it appears in `/signals/top` at tier A or better with the fresh flag set, and is corroborated by either a herd signal (`herd_tier` at threshold or better, matching `net_direction`) or an insider-signal score above a configured floor. Exit: whichever comes first of a time stop (default 30 trading days), a fixed stop-loss %, or the signal flipping direction on a later run. Sizing: fixed-fractional, default ≤5% of paper equity per position, ≤8 concurrent, ≤2 per sector.

Every number above is a config value with a deliberately arbitrary default. The point of the paper phase is to tune them against our own logs. One honesty note that belongs in the doc rather than in anyone's head: the vendor's headline stats (e.g. "72.7% backtested win rate" on A+ signals) are unaudited marketing claims, congressional disclosure data is delayed by design, and this same data feeds several competing products, so any edge is shared and possibly crowded. GovGreed's own terms state all outputs are informational, not financial advice. The bot's forward logs are the only performance numbers we treat as real.

## 7. Failure handling

`429 DAILY_QUOTA_EXCEEDED` — abort the run cleanly, record how far it got, alert; never burn the retry buffer on a lost cause. `429 BURST_LIMIT_EXCEEDED` — honor `Retry-After: 1`; the client also self-throttles to ~2 req/sec so this shouldn't fire. `401 / 403` (invalid or revoked key) — halt everything and alert immediately; do not retry, since retries can't fix auth and the key may have been rotated. `5xx / INTERNAL_ERROR` — exponential backoff, three attempts, then skip the day and alert with the logged `request_id`. Schema drift — parse defensively, ignore unknown fields, alert on missing required ones.

Circuit breaker: a max-daily-loss threshold and a manual env-var kill switch both stop new order placement while leaving exit logic running.

## 8. Milestones

**M0 — client + plumbing (a weekend).** Wrapper, envelope/error handling, `/status` and `/me` round-trips, quota logging, SQLite schema.
**M1 — dry run (week 1, while quota is 100/day).** Daily job logs signals and would-be decisions; zero orders. Burn-in data starts accumulating from the first day of API access.
**M2 — paper trading.** Alpaca paper account wired in, small universe, alerts on fills and errors.
**M3 — evaluation (after ≥60 trading days of logs).** Compare against buy-and-hold SPY over the same window, including a check of how signal tiers correlated with outcomes. Only after this: decide whether the founders tier, richer endpoints, or shelving the project is the right next step.

## 9. Open questions

Which quota tier the beta approval lands us on, and whether the beta terms have any restriction on automated/trading use worth confirming once account docs are visible. Whether the free tier's post-week-one quota is 20 or 250 calls/day (docs disagree; the design works under the pessimistic number). What liquidity floor keeps political-filing microcaps out of the universe. Whether predictions (`/predictions/top`, including the DARK_WINDOW status) should gate entries or merely annotate them — v1 treats them as annotation only, since their forward accuracy is exactly what our logs will measure.
