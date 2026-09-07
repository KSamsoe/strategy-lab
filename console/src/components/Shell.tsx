/**
 * The frame every screen sits in: nav, the density toggle, and the two global
 * controls.
 *
 * Pause-all and the kill switch live here rather than on the fleet screen
 * because they must be reachable from anywhere, including by keyboard --
 * `Shift+K` from any screen. When the thing you need is a panic stop, "navigate
 * home first" is not an acceptable step.
 */

import { Suspense, useCallback, useEffect, useState, type ReactNode } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api";
import { Button, Loading, cx } from "./ui";

const NAV = [
  { to: "/", label: "Fleet" },
  { to: "/runs", label: "Runs" },
  { to: "/agent", label: "Agent" },
] as const;

function useDensity() {
  const [compact, setCompact] = useState(
    () => document.documentElement.dataset.density === "compact",
  );
  useEffect(() => {
    document.documentElement.dataset.density = compact ? "compact" : "comfortable";
  }, [compact]);
  return [compact, setCompact] as const;
}

export function Shell({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [compact, setCompact] = useDensity();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  const health = useQuery({
    queryKey: ["live", "health"],
    queryFn: api.liveHealth,
    refetchInterval: 5000,
    retry: false,
  });

  const live = useQuery({
    queryKey: ["live", "strategies"],
    queryFn: api.liveStrategies,
    refetchInterval: 5000,
    retry: false,
  });

  const kill = useMutation({
    mutationFn: (reason: string) => api.kill(reason),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["live"] }),
  });

  const pauseAll = useMutation({
    mutationFn: async () => {
      const names = live.data?.strategies.map((s) => s.strategy) ?? [];
      await Promise.all(names.map((n) => api.pause(n)));
      return names.length;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["live"] }),
  });

  // Both destructive-looking controls are confirm-gated. They can only ever
  // reduce exposure, but a misclick that halts a live book is still a bad day.
  const confirmKill = useCallback(() => {
    const engaged = health.data?.kill_switch.engaged;
    if (engaged) {
      window.alert(
        "Kill switch is already engaged.\n\nReleasing it is deliberately a CLI action:\n  lab kill --release",
      );
      return;
    }
    const reason = window.prompt(
      "Engage the kill switch?\n\nThis stops all new orders immediately. Exits keep running.\nReleasing it requires `lab kill --release` at a terminal.\n\nReason (recorded in the journal):",
      "manual stop from console",
    );
    if (reason !== null) kill.mutate(reason || "manual stop from console");
  }, [health.data, kill]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.shiftKey && (e.key === "K" || e.key === "k") && !e.metaKey && !e.ctrlKey) {
        const el = document.activeElement;
        if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return;
        e.preventDefault();
        confirmKill();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirmKill]);

  const killed = health.data?.kill_switch.engaged ?? false;
  const tripped = health.data?.breaker.tripped ?? false;

  return (
    <div className="h-full flex flex-col">
      <header className="shrink-0 flex items-center gap-4 h-9 px-3 border-b border-slate bg-gunmetal">
        <Link to="/" className="unit text-chalk tracking-[0.18em] no-underline">
          STRATEGY&nbsp;LAB
        </Link>

        <nav className="flex items-center gap-1">
          {NAV.map((n) => {
            const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
            return (
              <Link
                key={n.to}
                to={n.to}
                className={cx(
                  "px-2 h-6 inline-flex items-center rounded text-xs no-underline",
                  active ? "text-teal bg-teal-wash" : "text-ash hover:text-chalk",
                )}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex-1" />

        {killed ? (
          <span className="mono text-xs text-ember border border-ember/50 bg-ember-wash rounded px-2 py-0.5">
            KILL SWITCH ENGAGED
          </span>
        ) : tripped ? (
          <span className="mono text-xs text-amber border border-amber/50 bg-amber-wash rounded px-2 py-0.5">
            BREAKER TRIPPED
          </span>
        ) : null}

        <Button
          onClick={() => setCompact(!compact)}
          title="Row density"
        >
          {compact ? "compact" : "comfortable"}
        </Button>

        <Button
          onClick={() => {
            if (window.confirm("Pause every running strategy? Open positions are left alone.")) {
              pauseAll.mutate();
            }
          }}
          disabled={!live.data?.strategies.length}
          title="Stop all strategies from making new decisions"
        >
          ⏸ pause all
        </Button>

        <Button kind="danger" onClick={confirmKill} title="Kill switch — Shift+K from anywhere">
          ⛔ kill
        </Button>
      </header>

      <main className="flex-1 min-h-0 overflow-auto">
        <Suspense fallback={<Loading what="loading screen" />}>{children}</Suspense>
      </main>
    </div>
  );
}
