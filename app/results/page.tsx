"use client";
import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { predict, type PredictionResult } from "@/lib/predict";
import ChapterCard from "@/components/ChapterCard";
import RecallBanner from "@/components/RecallBanner";
import BookingCTA from "@/components/BookingCTA";
import type { SVVKjoretoy } from "@/lib/svv-types";
import type { RecallEntry } from "@/lib/recall";
import { fetchRecalls } from "@/lib/recall";

function ResultsContent() {
  const params = useSearchParams();
  const [result,  setResult]  = useState<PredictionResult | null>(null);
  const [recalls, setRecalls] = useState<RecallEntry[]>([]);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    const regnr    = params.get("regnr") ?? "";
    const km       = parseInt(params.get("km") ?? "0", 10);
    const aargangRaw = parseInt(params.get("aargang") ?? "", 10);
    const kjoretoy: SVVKjoretoy = {
      regnr,
      merke:     params.get("merke")     ?? "UKJENT",
      modell:    params.get("modell")    ?? "",
      aargang:   isNaN(aargangRaw) ? null : aargangRaw,
      drivstoff: (params.get("drivstoff") ?? "BENSIN") as SVVKjoretoy["drivstoff"],
      drivlinje: (params.get("drivlinje") ?? "FORHJUL") as SVVKjoretoy["drivlinje"],
      euFrist:   params.get("euFrist") || null,
      farge:     null,
    };

    if (!regnr || km <= 0) {
      setError("Manglende data. Gå tilbake og prøv igjen.");
      return;
    }

    predict(kjoretoy, km)
      .then(setResult)
      .catch(() => setError("Kunne ikke beregne risiko. Prøv igjen."));

    fetchRecalls(regnr).then(setRecalls);
  }, [params]);

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">{error}</p>
        <a href="/" className="mt-4 inline-block text-green underline">← Tilbake</a>
      </div>
    );
  }

  if (!result) {
    return <div className="text-center py-12 text-gray-500">Beregner risiko...</div>;
  }

  if (result.ingenHistorikk) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12">
        <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6">
          <h1 className="text-xl font-bold text-yellow-900 mb-2">Ikke nok historikk</h1>
          <p className="text-yellow-800">
            {result.merke} {result.modell} ({result.aargang ?? "ukjent årsmodell"}) er for ny til å ha PKK-historikk.
            Biler sjekkes til EU-kontroll første gang 4 år etter første gangs registrering.
          </p>
        </div>
        <a href="/" className="mt-4 inline-block text-green underline">← Sjekk en annen bil</a>
      </div>
    );
  }

  const riskBg  = result.overall >= 2.0 ? "bg-red-50"     : result.overall >= 1.3 ? "bg-yellow-50"   : "bg-pale";
  const riskTxt = result.overall >= 2.0 ? "text-red-800"  : result.overall >= 1.3 ? "text-yellow-900" : "text-dark";

  return (
    <div className="max-w-2xl mx-auto px-4 py-8">
      <div className={`rounded-2xl ${riskBg} border border-mint p-6 mb-6`}>
        <p className="text-sm font-medium text-gray-500 mb-1">
          {result.regnr} · {result.merke} {result.modell} ({result.aargang ?? "ukjent årsmodell"})
        </p>
        <h1 className={`text-3xl font-bold ${riskTxt} mb-1`}>
          {result.overall.toFixed(1)}×
          <span className="text-base font-normal ml-2">enn gjennomsnittet</span>
        </h1>
        <p className="text-sm text-gray-600">
          {result.drivlinje} · {result.kmBucket} km · EU-frist:{" "}
          {params.get("euFrist") || "ukjent"}
        </p>
      </div>

      <RecallBanner recalls={recalls} />

      <h2 className="font-bold text-dark mb-3">Risiko per kapittel</h2>
      <div role="list" className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
        {result.chapters.map((ch) => (
          <ChapterCard key={ch.chapter} chapter={ch} />
        ))}
      </div>

      <BookingCTA
        merke={result.merke}
        kapittel={result.chapters.filter(c => c.relativRisiko >= 1.3).map(c => c.chapter)}
      />

      <a href="/" className="block text-center text-green underline mt-6 text-sm">
        ← Sjekk en annen bil
      </a>

      <p className="text-xs text-gray-400 text-center mt-4">
        † = Kapittel justert med britiske MOT-data (DVSA). Se{" "}
        <a href="/modell" className="underline">Om modellen</a> for detaljer.
      </p>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <main className="min-h-screen bg-pale">
      <header className="bg-dark text-white px-6 py-4 flex items-center gap-3">
        <a href="/" className="font-bold text-lg tracking-tight hover:text-mint transition-colors">EU-sjekk</a>
        <span className="text-mint text-sm opacity-75">Beta</span>
      </header>
      <Suspense fallback={<div className="text-center py-12 text-gray-500">Laster...</div>}>
        <ResultsContent />
      </Suspense>
    </main>
  );
}
