"use client";
import { useState, useEffect } from "react";

const STORAGE_KEY = "eu-sjekk-consent";

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

export function enableGA(measurementId: string) {
  if (typeof window === "undefined" || !measurementId) return;
  const script = document.createElement("script");
  script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
  script.async = true;
  document.head.appendChild(script);
  window.gtag = window.gtag || function (...args: unknown[]) {
    (window as unknown as { dataLayer: unknown[] }).dataLayer =
      (window as unknown as { dataLayer: unknown[] }).dataLayer || [];
    (window as unknown as { dataLayer: unknown[] }).dataLayer.push(args);
  };
  window.gtag("js", new Date());
  window.gtag("config", measurementId, { anonymize_ip: true });
}

export default function CookieBanner() {
  const [show, setShow] = useState(false);
  const measurementId = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID ?? "";

  useEffect(() => {
    const consent = localStorage.getItem(STORAGE_KEY);
    if (consent === "accepted") {
      enableGA(measurementId);
    } else if (!consent) {
      setShow(true);
    }
  }, [measurementId]);

  const accept = () => {
    localStorage.setItem(STORAGE_KEY, "accepted");
    enableGA(measurementId);
    setShow(false);
  };

  const decline = () => {
    localStorage.setItem(STORAGE_KEY, "declined");
    setShow(false);
  };

  if (!show) return null;

  return (
    <div
      role="dialog"
      aria-label="Informasjonskapsel-samtykke"
      aria-live="polite"
      className="fixed bottom-0 left-0 right-0 z-50 bg-dark text-white px-4 py-4 sm:px-6 sm:py-5"
    >
      <div className="max-w-2xl mx-auto flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <p className="text-sm text-mint flex-1">
          Vi bruker informasjonskapsler (Google Analytics) for å forstå hvordan tjenesten brukes.
          Ingen persondata deles.{" "}
          <a href="/faq#personvern" className="underline text-yellow">
            Les mer
          </a>
          .
        </p>
        <div className="flex gap-3 shrink-0">
          <button
            onClick={decline}
            className="text-sm text-mint underline hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-mint rounded"
          >
            Avvis
          </button>
          <button
            onClick={accept}
            className="bg-yellow text-dark font-semibold text-sm px-4 py-2 rounded-lg hover:bg-white transition-colors focus:outline-none focus:ring-2 focus:ring-yellow"
          >
            Godta
          </button>
        </div>
      </div>
    </div>
  );
}
