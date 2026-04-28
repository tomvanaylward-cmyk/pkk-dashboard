import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import CookieBanner from "@/components/CookieBanner";
import Footer from "@/components/Footer";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title:       "EU-sjekk — EU-kontroll risikosjekk",
  description: "Sjekk risikoen for at bilen din stryker til EU-kontrollen. Basert på historiske PKK-data.",
  metadataBase: new URL("https://eu-sjekk.vercel.app"),
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="nb">
      <body className={inter.className}>
        {children}
        <Footer />
        <CookieBanner />
      </body>
    </html>
  );
}
