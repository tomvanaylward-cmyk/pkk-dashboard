import type { RecallEntry } from "@/lib/recall";

export default function RecallBanner({ recalls }: { recalls: RecallEntry[] }) {
  if (recalls.length === 0) return null;
  return (
    <div role="alert" className="bg-red-50 border border-red-200 rounded-xl p-4 mb-4">
      <h2 className="font-bold text-red-800 mb-2">
        ⚠️ {recalls.length} åpen tilbakekalling
      </h2>
      <ul className="space-y-1">
        {recalls.map((r, i) => (
          <li key={i} className="text-sm text-red-700">
            {r.tittel} ({r.dato})
            {r.url && <a href={r.url} className="ml-2 underline" target="_blank" rel="noopener noreferrer">Les mer</a>}
          </li>
        ))}
      </ul>
    </div>
  );
}
