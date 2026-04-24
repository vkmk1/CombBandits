"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { RegretCurve } from "@/lib/types";
import { algoColor, algoDisplay, isOurs, isBaseline } from "@/lib/colors";

export default function RegretCurves({ curves }: { curves: RegretCurve[] }) {
  // Build wide table: { x, algo1, algo2, ... }
  const allX = new Set<number>();
  curves.forEach((c) => c.x.forEach((x) => allX.add(x)));
  const xs = [...allX].sort((a, b) => a - b);

  const data = xs.map((x) => {
    const row: Record<string, number | null> = { x };
    curves.forEach((c) => {
      const i = c.x.indexOf(x);
      row[c.algo] = i >= 0 ? c.mean[i] : null;
    });
    return row;
  });

  return (
    <div className="rounded border border-border bg-panel p-5">
      <div className="h-[480px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 24, left: 12, bottom: 8 }}>
            <CartesianGrid stroke="#1d2026" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="x"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
              stroke="#2a2f38"
              tick={{ fill: "#6a6f78", fontSize: 10.5 }}
              tickLine={false}
            />
            <YAxis
              stroke="#2a2f38"
              tick={{ fill: "#6a6f78", fontSize: 10.5 }}
              tickLine={false}
              width={48}
            />
            <Tooltip
              cursor={{ stroke: "#2a2f38", strokeWidth: 1, strokeDasharray: "2 4" }}
              contentStyle={{
                background: "#0a0b0d",
                border: "1px solid #2a2f38",
                borderRadius: "4px",
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "11.5px",
                padding: "10px 12px",
              }}
              labelStyle={{ color: "#8a8f9a", marginBottom: 6, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.16em" }}
              labelFormatter={(v) => `t = ${(v as number).toLocaleString()}`}
              formatter={(value, name) => [
                (value as number).toFixed(1),
                algoDisplay(name as string),
              ]}
              itemSorter={(item) => (item.value as number) ?? 0}
            />
            {curves.map((c) => {
              const ours = isOurs(c.algo);
              const base = isBaseline(c.algo);
              return (
                <Line
                  key={c.algo}
                  type="monotone"
                  dataKey={c.algo}
                  stroke={algoColor(c.algo)}
                  strokeWidth={ours ? 1.8 : 1.2}
                  strokeOpacity={base ? 0.55 : 0.95}
                  strokeDasharray={base ? "4 3" : undefined}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                  isAnimationActive={false}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Custom legend */}
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 px-1">
        {curves
          .slice()
          .sort((a, b) => a.mean[a.mean.length - 1] - b.mean[b.mean.length - 1])
          .map((c) => (
            <div key={c.algo} className="flex items-center gap-2">
              <span
                className="h-[2px] w-4 rounded-full"
                style={{
                  backgroundColor: algoColor(c.algo),
                  opacity: isBaseline(c.algo) ? 0.5 : 1,
                }}
              />
              <span className="text-[11px] text-dim font-mono">
                {algoDisplay(c.algo)}
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}
