"use client";

import type { PairedTest } from "@/lib/types";
import { algoColor, algoDisplay } from "@/lib/colors";

export default function SignificancePanel({
  paired,
}: {
  paired: PairedTest[];
}) {
  const sorted = paired.slice().sort((a, b) => a.p - b.p);

  return (
    <div className="rounded border border-border bg-panel overflow-hidden">
      <div className="grid grid-cols-[minmax(160px,1fr)_70px_70px_90px_70px] px-5 py-3 text-[10px] font-mono uppercase tracking-[0.16em] text-faint border-b border-border bg-[#0c0e11]">
        <div>Algorithm</div>
        <div className="text-right">W/L</div>
        <div className="text-right">Δ̄</div>
        <div className="text-right">p-value</div>
        <div className="text-right">Sig</div>
      </div>
      {sorted.map((t) => {
        const color = algoColor(t.algo);
        const winFrac = t.n > 0 ? t.wins / t.n : 0;
        return (
          <div
            key={t.algo}
            className="grid grid-cols-[minmax(160px,1fr)_70px_70px_90px_70px] px-5 py-3 items-center border-b border-border last:border-b-0 transition-colors hover:bg-[#11141a]"
          >
            <div className="flex items-center gap-2.5">
              <span
                className="h-[8px] w-[3px] rounded-sm"
                style={{ backgroundColor: color }}
              />
              <span className="text-[13px] text-ink tracking-tight">
                {algoDisplay(t.algo)}
              </span>
            </div>
            <div className="text-right font-mono text-[11.5px] tabular-nums">
              <span className="text-good">{t.wins}</span>
              <span className="text-faint mx-[3px]">/</span>
              <span className="text-bad">{t.losses}</span>
            </div>
            <div
              className="text-right font-mono text-[12px] tabular-nums"
              style={{
                color:
                  t.mean_adv > 0 ? "#7fd6a1" : t.mean_adv < 0 ? "#ff7d7d" : "#8a8f9a",
              }}
            >
              {t.mean_adv > 0 ? "+" : ""}
              {t.mean_adv.toFixed(1)}
            </div>
            <div className="text-right font-mono text-[11.5px] text-dim tabular-nums">
              {formatP(t.p)}
            </div>
            <div className="text-right">
              {t.sig ? (
                <span className="inline-flex items-center gap-1 rounded-full border border-good/40 bg-good/10 px-2 py-[1px] text-[9px] font-mono uppercase tracking-[0.16em] text-good">
                  ★
                </span>
              ) : (
                <span className="text-faint font-mono text-[10px]">—</span>
              )}
            </div>
          </div>
        );
      })}
      <div className="px-5 py-3 text-[10.5px] font-mono text-faint bg-[#0c0e11] border-t border-border">
        Wilcoxon signed-rank · paired by (config, seed) · ★ = sig at α=0.05
        Holm-Bonferroni
      </div>
    </div>
  );
}

function formatP(p: number): string {
  if (p < 1e-4) return "<10⁻⁴";
  if (p < 1e-3) return p.toExponential(1);
  return p.toFixed(4);
}
