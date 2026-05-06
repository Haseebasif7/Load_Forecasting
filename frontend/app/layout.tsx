import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Electricity Load Forecast",
  description:
    "24-hour LSTM load forecasts with seasonal-naive baseline on UCI Electricity Load Diagrams 2011-2014.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
