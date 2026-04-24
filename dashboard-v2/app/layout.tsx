import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CombBandits — Live Research Dashboard",
  description:
    "Real-time monitoring for combinatorial bandit experiments with LLM-guided correlation structure.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
