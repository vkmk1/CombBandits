// Stable color per algorithm — palette tuned for dark terminal aesthetic
export const ALGO_COLORS: Record<string, string> = {
  CTS: "#6a6f78",
  CUCB: "#555a63",
  N1_corr_full: "#ffb64c",
  N4_robust_corr: "#e28cff",
  ABLATION_random_corr: "#5a6b85",
  ABLATION_random_kernel: "#9fa4af",
  V1_decay_kernel: "#5ce1e6",
  V2_requery_logspaced: "#a3e635",
  V3_blend_llm_data: "#f97171",
  V4_refine_topk: "#ffcf86",
  V5_ensemble_kernels: "#f0abfc",
  V6_edge_pruning: "#86efac",
  V7_per_arm_damping: "#c084fc",
};

export function algoColor(name: string): string {
  const base = name.split("@")[0]; // strip HP suffix
  return ALGO_COLORS[base] ?? "#8a8f9a";
}

export function algoDisplay(name: string): string {
  const parts = name.split("@");
  const base = parts[0]
    .replace(/^V(\d)_/, "V$1 · ")
    .replace(/^N(\d)_corr_full$/, "CorrCTS-Full")
    .replace(/^N(\d)_robust_corr$/, "RobustCorrCTS")
    .replace(/^ABLATION_/, "Ablation · ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return parts[1] ? `${base} · ${parts[1]}` : base;
}

export function isOurs(name: string): boolean {
  return (
    name.startsWith("N1_") ||
    name.startsWith("V") ||
    name.startsWith("N4_")
  );
}

export function isBaseline(name: string): boolean {
  return name === "CTS" || name === "CUCB";
}
