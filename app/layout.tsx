import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EU-sjekk",
  description: "EU-kontroll risikosjekk for norske biler",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="nb">
      <body>{children}</body>
    </html>
  );
}
