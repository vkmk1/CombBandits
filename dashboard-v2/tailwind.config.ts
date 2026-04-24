import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
        serif: ['"Instrument Serif"', "Georgia", "serif"],
      },
      colors: {
        bg: "#0a0b0d",
        panel: "#0f1114",
        border: "#1d2026",
        "border-hi": "#2a2f38",
        ink: "#e6e8ec",
        dim: "#8a8f9a",
        faint: "#4a4f5a",
        accent: "#ffb64c",
        "accent-dim": "rgba(255, 182, 76, 0.08)",
        good: "#7fd6a1",
        warn: "#ffb64c",
        bad: "#ff7d7d",
      },
      letterSpacing: {
        tightest: "-0.04em",
        tighter: "-0.02em",
      },
    },
  },
  plugins: [],
};
export default config;
