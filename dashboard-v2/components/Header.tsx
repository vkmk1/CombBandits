"use client";

import type { LiveResponse } from "@/lib/types";

export default function Header({
  experiment,
}: {
  experiment: LiveResponse["experiment"] | undefined;
}) {
  const total = experiment?.total_trials ?? 0;
  const done = experiment?.completed_trials ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

  return (
    <header>
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div>
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/40 bg-accent-dim px-2 py-[2px]">
              <span className="pulse-live h-1.5 w-1.5 rounded-full bg-accent" />
              <span className="text-[9.5px] font-mono tracking-[0.2em] text-accent uppercase">
                Live
              </span>
            </span>
            <span className="text-[10.5px] font-mono tracking-[0.2em] uppercase text-faint">
              Combinatorial Bandits · ICML 2026
            </span>
          </div>

          <h1 className="mt-4 text-[42px] md:text-[52px] leading-[1.02] tracking-tightest font-serif italic text-ink">
            Long-horizon
            <span className="font-sans not-italic font-medium"> validation</span>
            <span className="text-accent">.</span>
          </h1>

          <p className="mt-3 max-w-[580px] text-[13.5px] leading-[1.6] text-dim">
            Testing whether correlated Thompson sampling variants preserve
            their early-round advantage over the long horizon, and whether
            that advantage generalizes to held-out problem configurations.
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end gap-1 min-w-[220px]">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-faint">
            Run
          </span>
          <span className="font-mono text-[13px] text-ink">
            {experiment?.run_id ?? "—"}
          </span>
          <span className="mt-2 text-[10px] font-mono uppercase tracking-[0.2em] text-faint">
            Model
          </span>
          <span className="font-mono text-[13px] text-accent">
            {experiment?.model ?? "—"}
          </span>
        </div>
      </div>

      {/* progress bar */}
      <div className="mt-10 border-t border-border pt-5">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-faint">
            Progress
          </span>
          <span className="font-mono text-[12.5px] text-ink">
            <span className="text-accent">{done.toLocaleString()}</span>
            <span className="text-faint mx-2">/</span>
            {total.toLocaleString()}
            <span className="text-faint ml-3">{pct.toFixed(1)}%</span>
          </span>
        </div>
        <div className="relative h-[3px] w-full overflow-hidden rounded bg-border">
          <div
            className="absolute inset-y-0 left-0 bg-accent transition-all duration-700 ease-out"
            style={{ width: `${pct}%` }}
          />
          <div
            className="shimmer absolute inset-y-0 left-0"
            style={{ width: `${pct}%` }}
          />
        </div>

        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-x-10 gap-y-4">
          <Stat label="Horizon" value={experiment?.T ? `T=${(experiment.T).toLocaleString()}` : "—"} />
          <Stat label="Configs" value={experiment?.n_configs?.toString() ?? "—"} />
          <Stat label="Seeds/cfg" value={experiment?.n_seeds?.toString() ?? "—"} />
          <Stat label="Algorithms" value={(experiment?.variants?.length || 0).toString()} />
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-faint">
        {label}
      </div>
      <div className="mt-1 font-mono text-[15px] text-ink">{value}</div>
    </div>
  );
}
