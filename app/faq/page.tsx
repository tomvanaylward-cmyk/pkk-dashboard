export const metadata = {
  title: "FAQ — EU-sjekk",
  description: "Vanlige spørsmål om EU-kontroll og risikomodellen.",
};

const FAQ_ITEMS = [
  {
    q: "Hva er EU-kontroll?",
    a: "EU-kontroll (periodisk kjøretøykontroll, PKK) er en obligatorisk teknisk kontroll av biler i Norge. Personbiler sjekkes første gang 4 år etter første registrering, deretter hvert 2. år.",
  },
  {
    q: "Hva betyr de ulike kapitlene?",
    a: "Kontrollene er delt inn i 11 kapitler: kap 0 (dokumenter), kap 1 (bremser), kap 2 (styring), kap 3 (sikt), kap 4 (lys), kap 5 (hjul/aksler), kap 6 (karosseri), kap 7 (annet utstyr), kap 8 (støy/utslipp), kap 9 (kjøretest), kap 10 (miljø).",
  },
  {
    q: "Hvorfor er noen kapitler justert med britiske data?",
    a: "Norsk PKK-data inneholder merke men ikke bilmodell. For å gi modell-spesifikke prediksjoner bruker vi britiske MOT-data som referanse. Britiske data brukes kun for klimauavhengige kapitler (dokumenter, lys) — for bremser, styring og fjæring brukes utelukkende norske data.",
  },
  {
    q: "Kan jeg stole på prediksjonen?",
    a: "Prediksjonen er et statistisk estimat basert på historiske mønstre — ikke en garanti. AUC-verdien (modelltreffsikkerhet) er tilgjengelig på Om modellen-siden. Se alltid på resultatet som en indikasjon, ikke en fasit.",
  },
  {
    q: "Hva lagrer dere om meg?",
    a: "Vi lagrer ikke eierinformasjon. Registreringsnummeret du søker på er ikke persondata i seg selv. Anonymisert statistikk kan brukes til å forbedre modellen. Se personvernerklæringen i bunnen av siden.",
  },
  {
    q: "Hvem er vi?",
    a: "EU-sjekk er et uavhengig prosjekt basert på åpne data fra Statens vegvesen. Vi er ikke tilknyttet NAF, Statens vegvesen eller andre organisasjoner.",
  },
];

export default function FaqPage() {
  return (
    <main className="min-h-screen bg-pale">
      <header className="bg-dark text-white px-6 py-4">
        <a href="/" className="font-bold text-lg tracking-tight hover:text-mint transition-colors">EU-sjekk</a>
      </header>
      <div className="max-w-2xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold text-dark mb-8">Ofte stilte spørsmål</h1>
        <dl className="space-y-6">
          {FAQ_ITEMS.map((item, i) => (
            <div key={i} className="bg-white rounded-xl border border-mint p-5">
              <dt className="font-semibold text-dark mb-2">{item.q}</dt>
              <dd className="text-gray-700 text-sm leading-relaxed">{item.a}</dd>
            </div>
          ))}
        </dl>
      </div>
    </main>
  );
}
