"use client";

export default function Connection({
  updated,
  error,
  loading,
}: {
  updated: string;
  error: boolean;
  loading: boolean;
}) {
  const status = error ? "error" : loading ? "syncing" : "connected";
  const color = error ? "bg-bad" : loading ? "bg-warn" : "bg-good";
  return (
    <span className="flex items-center gap-2.5">
      <span className={`relative flex h-1.5 w-1.5 ${error ? "" : "pulse-live"}`}>
        <span
          className={`absolute inset-0 rounded-full ${color}`}
        />
      </span>
      <span>{status}</span>
      <span className="text-faint">·</span>
      <span>last sync {updated}</span>
    </span>
  );
}
