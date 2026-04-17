import type { ChapterResult } from "@/lib/predict";

const CHAPTER_LABELS: Record<string, string> = {
  "Identification & documents": "Kap 0 — Dokumenter",
  "Brakes":                     "Kap 1 — Bremser",
  "Steering":                   "Kap 2 — Styring",
  "Visibility":                 "Kap 3 — Sikt",
  "Lights & electrical":        "Kap 4 — Lys og el.",
  "Axles, wheels & tyres":      "Kap 5 — Aksler og hjul",
  "Chassis & body":             "Kap 6 — Karosseri",
  "Other equipment":            "Kap 7 — Annet utstyr",
  "Noise & emissions":          "Kap 8 — Støy og utslipp",
  "Checks during drive":        "Kap 9 — Kjøretest",
  "Environment":                "Kap 10 — Miljø",
};

function riskLabel(rr: number): { label: string; color: string } {
  if (rr >= 2.0) return { label: "Høy risiko",    color: "bg-red-100 text-red-800 border-red-200" };
  if (rr >= 1.3) return { label: "Forhøyet",       color: "bg-yellow-100 text-yellow-800 border-yellow-200" };
  if (rr >= 0.7) return { label: "Gjennomsnitt",   color: "bg-pale text-green border-mint" };
  return           { label: "Lav risiko",    color: "bg-green/10 text-green border-green/20" };
}

export default function ChapterCard({ chapter }: { chapter: ChapterResult }) {
  const { label, color } = riskLabel(chapter.relativRisiko);
  const title = CHAPTER_LABELS[chapter.chapter] ?? chapter.chapter;

  return (
    <div className={`rounded-xl border p-4 ${color}`} role="listitem">
      <div className="flex justify-between items-start gap-2">
        <span className="font-medium text-sm">{title}</span>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full border shrink-0">
          {label}
        </span>
      </div>
      <div className="mt-2 text-xs opacity-75">
        {chapter.relativRisiko === 1.0
          ? "Gjennomsnittlig risiko"
          : `${chapter.relativRisiko.toFixed(1)}× ${chapter.relativRisiko > 1 ? "høyere" : "lavere"} enn snitt`}
      </div>
      {chapter.ukFaktorBrukt && (
        <div className="mt-1 text-xs opacity-50">† Justert med UK-data</div>
      )}
    </div>
  );
}
