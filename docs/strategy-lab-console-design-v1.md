# Strategy Lab Console — Frontend Design Doc

**Status:** Draft v1 · **Companion to:** `strategy-lab-design-v1.md`
**Assumptions:** Local-first, single-user web app launched with `lab ui` on the same machine as the lab; React + TypeScript SPA over a FastAPI read layer that queries the lab's existing SQLite/DuckDB/Parquet stores; one WebSocket for live events. No cloud deployment in scope.

## 1. Goal and thesis

The console answers one question — *what is my bot doing, and why* — and answers it identically whether the bot is trading live paper right now or ran a backtest over 2019 last night. It has three jobs, in priority order: **monitor the fleet** (is everything alive, flat-or-positioned as expected, inside its limits), **explain any single decision** (what the strategy saw, what it intended, what the gate and broker did with that intent), and **compare experiments** (runs, sweeps, walk-forward windows, agent iterations).

The architectural thesis carries over from the lab doc and becomes the UI's core trick: **one event schema, two sources**. Because backtest and live emit the same decision/order/fill/gate events, every screen is a renderer over an event stream and cannot tell replay from reality. Backtests get a live-quality inspector for free; live trading gets backtest-quality forensics for free; and "scrub back through what the bot did this morning" is the same component as "step through a 2021 backtest."

## 2. What the frontend imposes on the engine

Three small, explicit backend additions — worth listing because they're engine work, not UI work.

**Decision journal.** At every `on_bar`, the engine persists a decision record: timestamp, strategy, the *input tape* (every value the `Context` actually served — history slices summarized, indicator values verbatim, signals delivered), the emitted intents, the gate verdict per intent (pass / clipped / blocked, with the rule that fired), and resulting order ids. The `Context` already mediates all data access, so capture is a wrapper, not a strategy change. This journal is what makes "why does this position exist" a lookup instead of an investigation.

**Event journal.** The live runner appends every event (decision, order, fill, reject, gate block, breaker trip, reconciliation diff, heartbeat, error) to an append-only SQLite journal. The API tails it over the WebSocket; the alert webhooks become just another consumer. The UI never touches the broker or the runner directly — it reads journals. A runner crash therefore can't be caused by the UI, and the UI keeps working (as forensics) when the runner is down.

**Heartbeats.** Runner, data adapters, and broker connection each emit a periodic heartbeat event with staleness metadata. Freshness is displayed, never assumed.

## 3. Architecture

```
live runner ──append──> event journal (SQLite) ──tail──┐
backtester ──write───> registry + decision journal      ├─> FastAPI ──REST──> React SPA
equity/trades ───────> Parquet  <──DuckDB queries───────┘        └──WS (live events)──>
```

**Read-mostly, with asymmetric controls.** The UI's mutation surface follows one principle: *the console can only make the system safer, never riskier.* Pause strategy, cancel open orders, trip the kill switch — available, confirm-gated, and journaled. Start a live strategy, raise a limit, resume after a breaker — never in the UI; those remain CLI actions with the manual checklist from the lab doc. This keeps a stolen laptop, a misclick, or a compromised browser tab from *increasing* exposure.

**Serving.** `lab ui` starts FastAPI (uvicorn) bound to `127.0.0.1` and opens the SPA; an optional bearer token supports the "check from another machine on my LAN" case. REST endpoints mirror the CLI's JSON contract (`/runs`, `/runs/{id}/equity`, `/runs/{id}/decisions`, `/live/strategies`, `/live/events?since=`), so anything the UI can show, an agent or script can also fetch — the console is a view over the same contract, not a second API.

## 4. Screens

**Fleet overview (home).** The glanceable answer to "is everything okay."

```
┌ STRATEGY LAB ─────────────────────────────── [⏸ pause all] [⛔ kill] ┐
│ equity $103,412  ▁▂▃▅▆▅▇  +0.8% today      breaker ─ armed          │
│ alpaca ● 2s   govgreed ● 14/20 calls   runner ● hb 09:31:04         │
├──────────────────────────────────────────────────────────────────────┤
│ momo_v3        PAPER  +1.2%  4 pos  gate 0  next 16:00  ▸           │
│ govgreed_sig   PAPER  −0.3%  2 pos  gate 1  next 06:30  ▸           │
│ agent_daily    PAPER  +0.1%  1 pos  $0.42 spend  next 06:30  ▸      │
├──────────────────────────────────────────────────────────────────────┤
│ 09:31:02  momo_v3       FILL   AAPL  +120 @ 227.14                  │
│ 09:30:57  gate          BLOCK  govgreed_sig NVDA — sector cap 2/2   │
│ 09:30:00  runner        RECONCILE ok (0 diffs)                      │
└──────────────────────────────────────────────────────────────────────┘
```

Strategy cards, a global event tape, health strip with explicit staleness ("● 2s" means data 2 seconds old), and — because it earned a spot in the lab doc — the GovGreed quota meter as a first-class health item.

**Strategy detail (live).** Positions table with per-position P&L and age; price chart with entry/exit markers for the selected ticker; the decision tape (§5) for this strategy; open orders and recent gate events; a provenance panel (params, git commit, config hash, uptime); and for Pattern-B strategies an *Agent* tab (below). A "replay today" control scrubs the tape from the morning's first event — same components, historical source.

**Run browser.** The registry as a filterable, sortable table: strategy, date range, key metrics as columns, in-sample/out-of-sample badges, an origin tag (`human` / `agent-loop`), and an *attempt counter* per strategy+config family. Multi-select feeds Compare.

**Run detail (backtest).**

```
┌ run 8f3a · momo_v3 · 2018–2026 · git a1c2f · data v41 · sweep #12 ┐
│ CAGR 14.2   Sharpe 1.31   MaxDD −18.4   Trades 312   OOS ✓        │
│ [ equity + drawdown chart ── IS ▓▓▓▓▓ │ OOS ▓▓ shaded split ]     │
│ [ price chart, ▲ entries ▼ exits ]  ⇄ time-locked ⇄ decision tape │
│ [ trade ledger — click row → chart & tape jump to that trade ]    │
└───────────────────────────────────────────────────────────────────┘
```

Equity and drawdown with the walk-forward split shaded so out-of-sample performance is visually unmissable; per-ticker price charts with trade markers; the sortable trade ledger; the decision tape; and a provenance footer (git, config, data_version, fills model — including the loud flag when same-bar fills were enabled). Everything is time-locked: selecting a trade, a tape row, or a chart region moves the other two.

**Compare.** Overlaid normalized equity curves, a metrics table with per-column best highlighted, drawdown small-multiples, and a scatter (e.g., turnover vs. Sharpe) for larger selections.

**Sweep.** For 2-D grids, a parameter heatmap colored by the chosen metric with the out-of-sample metric available as the color source (defaulting to OOS, deliberately); per-window walk-forward bars showing IS vs OOS side by side; and the attempt counter front and center — the UI's contribution to the lab's overfitting honesty is making "you tried 400 things" impossible to miss.

**Agent activity.** Pattern A gets a *lineage view*: the iteration chain of an author loop as a metric trajectory (OOS Sharpe per iteration), with per-step diffs of the strategy file/params and the run each step produced — the "is the agent actually converging or just churning" chart. Pattern B gets the *ledger*: each decision's context bundle, the model's structured reply and rationale, the gate verdict, token/cost per call and cumulative spend, all linked into the same decision tape.

## 5. The decision tape (signature component)

One component, present on every detail screen, that renders the decision journal as a scrubbable timeline locked to the charts above it. Each row is a decision point; collapsed, it reads like a terse log line (`06:30 · saw 3 signals · intent +NVDA 4% · gate: clipped to 3.1% (position cap) · filled`). Expanded, it shows the full input tape (indicator values, signals with their `knowledge_time`, portfolio state), the intent list, per-intent gate reasoning, and the resulting orders and fills — and for agent strategies, the prompt/response pair inline. A playhead scrubs time; charts, ledger, and tape stay synchronized; live mode simply pins the playhead to now. This is the product's memorable object and its reason to exist: the distance from "position looks wrong" to "here is the exact input that caused it" is one click and zero SQL.

## 6. Stack and charting

| Concern | Choice | Why |
|---|---|---|
| App | Vite + React + TypeScript, TanStack Query + Router | Boring, fast, typed against the API's JSON contract |
| Time series | lightweight-charts | Candles, markers, and crosshair sync at 60fps without bundling a terminal |
| Heatmaps / scatter / bars | ECharts | Sweep grids and compare views need a general chart engine |
| Live | Native WebSocket hook feeding a ring buffer + Query cache updates | Event tape stays O(visible), not O(history) |
| Styling | Tailwind over a design-token layer (§7) | Tokens keep meaning-colors singular and enforceable |

Time-lock synchronization (chart ⇄ tape ⇄ ledger) is a tiny event bus keyed on `(run_id | live, t)` — worth writing once, carefully, since every screen uses it. The static HTML reports from the lab doc remain as portable, emailable artifacts; the console supersedes them for interactive work, and its run-detail screen links to the static report for archival.

## 7. Visual direction

Subject first: this is an **instrument panel a solo operator stares at for hours**, not a landing page and not a casino. The design brief that follows from that: high information density, quiet chrome, and color that is *reserved for meaning* — if a hue appears, it encodes P&L direction, warning state, or selection, never decoration.

**Palette (dark, but not terminal-cosplay).** Base `graphite #14171C`, panels `gunmetal #1C2128`, hairlines `slate #2B323C`, primary text `chalk #E9ECEF`, secondary `ash #98A2AE`. Meaning colors are deliberately muted so a screen full of them stays readable: gains `moss #6FBF73`, losses `ember #E0654F`, warnings/gate `amber #D9A441`, and a single identity accent `instrument teal #45B2A1` used only for selection, links, and the tape playhead. The near-black-plus-acid-green trading-terminal cliché is explicitly rejected; nothing on this screen should glow.

**Typography.** IBM Plex Sans for UI text and Plex Mono — tabular lining figures always — for every number, ticker, timestamp, and the tape. The pairing is chosen for its instrument-panel lineage and because in a dashboard the type personality lives in how data is set, not in display headlines: aligned decimal columns, fixed-width deltas that don't jitter as they update, uppercase-tracked micro-labels for units. Numbers never reflow when they tick.

**Density and motion.** Comfortable-dense by default (rows ~32px) with a compact toggle; empty states say what to run to fill them (`no runs yet — lab backtest <file>`); staleness is always visible rather than implied. Motion is spent in exactly one place: state *changes* pulse once (a fill row arriving, a breaker tripping) and the playhead moves; nothing idles, `prefers-reduced-motion` disables the pulses, and value updates change without animation. Keyboard focus visible throughout; the kill switch is reachable by keyboard from anywhere.

## 8. Milestones

**F0 — read the registry (with lab M1).** FastAPI over registry + run artifacts; run browser and run detail with equity/drawdown, ledger, and provenance. Immediately useful with zero live infrastructure.
**F1 — decision tape (needs the engine's decision journal).** Tape on run detail, time-lock bus, trade-marker charts. The signature lands here.
**F2 — live (with lab M3).** Event journal tail over WS, fleet overview, strategy live detail, replay-today, pause/kill controls.
**F3 — comparison science (with lab M2 sweeps).** Compare, sweep heatmaps, walk-forward panels, attempt counters.
**F4 — agent views (with lab M4/M5).** Lineage view for author loops; ledger tab and cost meters for in-loop strategies.

## 9. Non-goals

No order entry or manual trading (the broker's own app exists and is better audited); no chart drawing tools or indicator studio (this is forensics, not technical-analysis workspace); no multi-user, auth, or public deployment beyond the localhost bind and optional token; no mobile-first design — the fleet overview should merely *survive* a phone screen for a status glance. And no UI path that increases risk: that asymmetry is a feature, not a gap.
