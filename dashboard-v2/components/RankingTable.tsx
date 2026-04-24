"use client";

import type { Ranking } from "@/lib/types";
import { algoColor, algoDisplay, isOurs, isBaseline } from "@/lib/colors";

export default function RankingTable({ rankings }: { rankings: Ranking[] }) {
  const maxMean = Math.max(...rankings.map((r) => r.mean));
  return (
    <div className="rounded border border-border bg-panel overflow-hidden">
      <div className="grid grid-cols-[32px_minmax(220px,1.5fr)_90px_120px_110px_120px_70px] px-5 py-3 text-[10px] font-mono uppercase tracking-[0.16em] text-faint border-b border-border bg-[#0c0e11]">
        <div>#</div>
        <div>Algorithm</div>
        <div className="text-right">n</div>
        <div className="text-right">Mean regret</div>
        <div className="text-right">SE</div>
        <div className="text-right">Δ vs CTS</div>
        <div className="text-right">Median</div>
      </div>

      {rankings.map((r, i) => {
        const color = algoColor(r.algo);
        const ours = isOurs(r.algo);
        const base = isBaseline(r.algo);
        const pct = maxMean > 0 ? (r.mean / maxMean) * 100 : 0;
        return (
          <div
            key={r.algo}
            className="relative grid grid-cols-[32px_minmax(220px,1.5fr)_90px_120px_110px_120px_70px] items-center px-5 py-[14px] border-b border-border last:border-b-0 transition-colors hover:bg-[#11141a]"
          >
            {/* bar background */}
            <div
              className="absolute left-0 top-0 bottom-0 opacity-[0.04] pointer-events-none"
              style={{
                background: `linear-gradient(90deg, ${color} 0%, ${color}00 100%)`,
                width: `${pct}%`,
              }}
            />
            <div className="font-mono text-[11px] text-faint relative">
              {String(i + 1).padStart(2, "0")}
            </div>
            <div className="relative flex items-center gap-3">
              <span
                className="h-[10px] w-[3px] rounded-sm"
                style={{ backgroundColor: color }}
              />
              <span
                className="text-[14px] font-medium tracking-tight"
                style={{ color: ours ? "#fff" : base ? "#8a8f9a" : "#e6e8ec" }}
              >
                {algoDisplay(r.algo)}
              </span>
              {ours && (
                <span className="text-[9px] font-mono tracking-[0.2em] uppercase text-accent">
                  ours
                </span>
              )}
              {base && (
                <span className="text-[9px] font-mono tracking-[0.2em] uppercase text-faint">
                  baseline
                </span>
              )}
            </div>
            <div className="relative text-right font-mono text-[12.5px] text-dim">
              {r.n}
            </div>
            <div className="relative text-right font-mono text-[14px] text-ink tabular-nums">
              {r.mean.toFixed(1)}
            </div>
            <div className="relative text-right font-mono text-[12px] text-dim tabular-nums">
              ±{r.se.toFixed(1)}
            </div>
            <div
              className="relative text-right font-mono text-[13px] tabular-nums"
              style={{
                color:
                  r.vs_baseline_pct === null
                    ? "#4a4f5a"
                    : r.vs_baseline_pct > 0
                    ? "#7fd6a1"
                    : r.vs_baseline_pct < 0
                    ? "#ff7d7d"
                    : "#8a8f9a",
              }}
            >
              {r.vs_baseline_pct === null
                ? "—"
                : `${r.vs_baseline_pct > 0 ? "+" : ""}${r.vs_baseline_pct.toFixed(1)}%`}
            </div>
            <div className="relative text-right font-mono text-[12.5px] text-dim tabular-nums">
              {r.median.toFixed(0)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
