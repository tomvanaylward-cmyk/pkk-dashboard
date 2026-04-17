"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RegnrForm() {
  const router = useRouter();
  const [regnr, setRegnr]   = useState("");
  const [km, setKm]         = useState("");
  const [error, setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const cleanRegnr = regnr.trim().toUpperCase().replace(/\s/g, "");
    const cleanKm    = parseInt(km.replace(/\s/g, ""), 10);

    if (!/^[A-Z]{2}\d{4,5}$/.test(cleanRegnr)) {
      setError("Skriv inn et gyldig norsk registreringsnummer (f.eks. AB12345)");
      return;
    }
    if (isNaN(cleanKm) || cleanKm < 0 || cleanKm > 1_000_000) {
      setError("Kilometerstand må være et tall mellom 0 og 1 000 000");
      return;
    }
    setLoading(true);
    try {
      const resp = await fetch(`/api/kjoretoy?regnr=${cleanRegnr}`);
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error ?? "Noe gikk galt. Prøv igjen.");
        return;
      }
      router.push(
        `/results?regnr=${cleanRegnr}&km=${cleanKm}&merke=${encodeURIComponent(data.merke)}&modell=${encodeURIComponent(data.modell)}&aargang=${data.aargang}&drivstoff=${data.drivstoff}&drivlinje=${data.drivlinje}&euFrist=${data.euFrist ?? ""}`
      );
    } catch {
      setError("Kunne ikke kontakte SVV. Sjekk nettilkoblingen og prøv igjen.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full max-w-md">
      <div>
        <label htmlFor="regnr" className="block text-sm font-medium text-gray-700 mb-1">
          Registreringsnummer
        </label>
        <input
          id="regnr"
          type="text"
          value={regnr}
          onChange={(e) => setRegnr(e.target.value.toUpperCase())}
          placeholder="AB12345"
          maxLength={7}
          className="w-full px-4 py-3 text-lg font-mono tracking-widest border border-mint rounded-lg focus:outline-none focus:ring-2 focus:ring-green uppercase"
          aria-label="Registreringsnummer"
          autoComplete="off"
          inputMode="text"
        />
      </div>
      <div>
        <label htmlFor="km" className="block text-sm font-medium text-gray-700 mb-1">
          Kilometerstand
        </label>
        <input
          id="km"
          type="number"
          value={km}
          onChange={(e) => setKm(e.target.value)}
          placeholder="75000"
          min={0}
          max={1000000}
          className="w-full px-4 py-3 text-lg border border-mint rounded-lg focus:outline-none focus:ring-2 focus:ring-green"
          aria-label="Kilometerstand"
          inputMode="numeric"
        />
      </div>

      {error && (
        <p role="alert" className="text-red-600 text-sm bg-red-50 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="bg-green text-white font-semibold py-3 px-6 rounded-lg hover:bg-dark transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-green focus:ring-offset-2"
        aria-busy={loading}
      >
        {loading ? "Henter data..." : "Sjekk EU-kontroll risiko →"}
      </button>
    </form>
  );
}
