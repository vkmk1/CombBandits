"use client";

import type { SplitStats } from "@/lib/types";
import { algoColor, algoDisplay, isBaseline } from "@/lib/colors";

export default function GeneralizationGap({
  splits,
  baseline,
}: {
  splits: SplitStats;
  baseline: string;
}) {
  const orig = splits["original"];
  const held = splits["held_out"];
  if (!orig || !held) {
    return (
      <div className="rounded border border-border bg-panel px-6 py-14 text-center">
        <p className="text-[12.5px] text-dim font-mono">
          Need both <span className="text-ink">original</span> and{" "}
          <span className="text-ink">held-out</span> splits to compute generalization gap.
        </p>
      </div>
    );
  }
  const baseOrig = orig[baseline];
  const baseHeld = held[baseline];
  if (baseOrig == null || baseHeld == null) {
    return <div className="text-dim font-mono text-[12px]">No baseline data.</div>;
  }

  const algos = Object.keys(orig).filter(
    (a) => a !== baseline && held[a] != null && !isBaseline(a)
  );
  const items = algos
    .map((algo) => {
      const origAdv = baseOrig - orig[algo];
      const heldAdv = baseHeld - held[algo];
      const gap = origAdv - heldAdv;
      return { algo, origAdv, heldAdv, gap };
    })
    .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));

  const maxAbs = Math.max(
    ...items.map((i) => Math.max(Math.abs(i.origAdv), Math.abs(i.heldAdv), Math.abs(i.gap))),
    1
  );

  return (
    <div className="rounded border border-border bg-panel p-5">
      <div className="flex items-center gap-5 mb-5 text-[10px] font-mono uppercase tracking-[0.16em] text-faint">
        <Legend dot="#ffb64c" label="Original" />
        <Legend dot="#5ce1e6" label="Held-out" />
        <Legend dot="#ff7d7d" label="Gap" />
      </div>
      <div className="space-y-4">
        {items.map((it) => {
          const color = algoColor(it.algo);
          const overfit = it.gap > 20;
          return (
            <div key={it.algo} className="grid grid-cols-[minmax(140px,1fr)_1fr] gap-6 items-center">
              <div className="flex items-center gap-2.5">
                <span
                  className="h-[8px] w-[3px] rounded-sm"
                  style={{ backgroundColor: color }}
                />
                <span className="text-[12.5px] text-ink tracking-tight">
                  {algoDisplay(it.algo)}
                </span>
                {overfit && (
                  <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-bad">
                    overfit
                  </span>
                )}
              </div>
              <div className="space-y-1.5">
                <BarRow label="orig" value={it.origAdv} max={maxAbs} color="#ffb64c" />
                <BarRow label="held" value={it.heldAdv} max={maxAbs} color="#5ce1e6" />
                <BarRow label="gap"  value={it.gap}     max={maxAbs} color="#ff7d7d" />
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-6 text-[10.5px] font-mono text-faint leading-relaxed">
        Δ = mean regret of CTS minus mean regret of variant. Positive bars = variant
        beats CTS on that split. Large positive <span className="text-bad">gap</span>{" "}
        (orig − held) suggests overfitting to the configs we observed during design.
      </p>
    </div>
  );
}

function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: dot }} />
      {label}
    </span>
  );
}

function BarRow({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
}) {
  const pct = Math.min(100, (Math.abs(value) / max) * 50);
  const positive = value >= 0;
  return (
    <div className="flex items-center gap-3">
      <span className="w-9 text-[10px] font-mono uppercase tracking-[0.14em] text-faint">
        {label}
      </span>
      <div className="flex-1 relative h-[6px] rounded bg-[#15181d] overflow-hidden">
        <div
          className="absolute top-0 bottom-0 transition-all duration-700 ease-out"
          style={{
            left: positive ? "50%" : `${50 - pct}%`,
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}cc, ${color})`,
          }}
        />
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-faint/40" />
      </div>
      <span
        className="w-14 text-right font-mono text-[11px] tabular-nums"
        style={{ color: positive ? "#7fd6a1" : "#ff7d7d" }}
      >
        {value > 0 ? "+" : ""}
        {value.toFixed(1)}
      </span>
    </div>
  );
}
