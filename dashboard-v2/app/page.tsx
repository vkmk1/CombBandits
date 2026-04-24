"use client";

import useSWR from "swr";
import { useMemo } from "react";
import type { LiveResponse } from "@/lib/types";
import Header from "@/components/Header";
import RankingTable from "@/components/RankingTable";
import RegretCurves from "@/components/RegretCurves";
import SignificancePanel from "@/components/SignificancePanel";
import GeneralizationGap from "@/components/GeneralizationGap";
import Connection from "@/components/Connection";

const fetcher = (url: string) =>
  fetch(url, { cache: "no-store" }).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  });

export default function Page() {
  const { data, error, isLoading } = useSWR<LiveResponse>(
    "/api/live",
    fetcher,
    { refreshInterval: 6000, keepPreviousData: true, revalidateOnFocus: true }
  );

  const hasData = data && data.stats.rankings.length > 0;

  const updated = useMemo(() => {
    if (!data?.updated_at) return "—";
    const d = new Date(data.updated_at * 1000);
    return d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }, [data?.updated_at]);

  return (
    <main className="relative z-10 min-h-screen">
      <div className="mx-auto max-w-[1440px] px-6 md:px-10 pt-10 pb-20">
        <Header experiment={data?.experiment} />

        <section className="mt-12 fade-in-up">
          <SectionTitle index="01" label="Live Ranking" />
          {hasData ? (
            <RankingTable rankings={data.stats.rankings} />
          ) : (
            <EmptyState loading={isLoading} error={!!error} />
          )}
        </section>

        <section className="mt-16 fade-in-up" style={{ animationDelay: "120ms" }}>
          <SectionTitle index="02" label="Cumulative Regret" />
          {hasData ? (
            <RegretCurves curves={data.stats.curves} />
          ) : (
            <EmptyState loading={isLoading} error={!!error} />
          )}
        </section>

        <div className="mt-16 grid grid-cols-1 lg:grid-cols-2 gap-10">
          <section className="fade-in-up" style={{ animationDelay: "220ms" }}>
            <SectionTitle
              index="03"
              label="Paired Significance"
              sub="Wilcoxon vs CTS · Holm-Bonferroni"
            />
            {hasData ? (
              <SignificancePanel paired={data.stats.paired} />
            ) : (
              <EmptyState loading={isLoading} error={!!error} />
            )}
          </section>

          <section className="fade-in-up" style={{ animationDelay: "320ms" }}>
            <SectionTitle
              index="04"
              label="Generalization Gap"
              sub="original − held-out advantage"
            />
            {hasData ? (
              <GeneralizationGap splits={data.stats.splits} baseline="CTS" />
            ) : (
              <EmptyState loading={isLoading} error={!!error} />
            )}
          </section>
        </div>

        <footer className="mt-20 pt-6 border-t border-border flex items-center justify-between text-[11px] text-faint font-mono uppercase tracking-[0.12em]">
          <span>CombBandits · ICML 2026 Submission</span>
          <Connection updated={updated} error={!!error} loading={isLoading} />
        </footer>
      </div>
    </main>
  );
}

function SectionTitle({
  index,
  label,
  sub,
}: {
  index: string;
  label: string;
  sub?: string;
}) {
  return (
    <div className="mb-5 flex items-baseline gap-4">
      <span className="font-mono text-[10px] tracking-[0.2em] text-faint">
        /{index}
      </span>
      <h2 className="text-[20px] font-medium tracking-tight text-ink">{label}</h2>
      {sub && (
        <span className="font-mono text-[10.5px] text-dim ml-1">{sub}</span>
      )}
    </div>
  );
}

function EmptyState({
  loading,
  error,
}: {
  loading: boolean;
  error: boolean;
}) {
  return (
    <div className="rounded border border-border bg-panel px-6 py-14 text-center">
      {error ? (
        <p className="text-[12.5px] text-bad font-mono">
          Unable to reach live API — retrying…
        </p>
      ) : (
        <p className="text-[12.5px] text-dim font-mono">
          {loading ? "Connecting to experiment…" : "Waiting for trials…"}
        </p>
      )}
    </div>
  );
}
