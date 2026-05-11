export const metadata = {
  title: "Om modellen — EU-sjekk",
  description: "Teknisk dokumentasjon av risikomodellen: datakilder, validering og begrensninger.",
};

export default function ModellPage() {
  return (
    <main className="min-h-screen bg-pale">
      <header className="bg-dark text-white px-6 py-4">
        <a href="/" className="font-bold text-lg tracking-tight hover:text-mint transition-colors">EU-sjekk</a>
      </header>
      <div className="max-w-2xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold text-dark mb-6">Om modellen</h1>

        <section className="mb-8">
          <h2 className="text-lg font-bold text-dark mb-2">Datagrunnlag</h2>
          <p className="text-gray-700 text-sm">
            Modellen er trent på <strong>~480 000 norske PKK-inspeksjoner</strong> fra Statens vegvesen
            (åpne data, NLOD-lisens, kvartalsvis oppdatert). Én logistisk regresjonsmodell per
            kapittel (kap 0–10 = 11 modeller).
          </p>
        </section>

        <section className="mb-8">
          <h2 className="text-lg font-bold text-dark mb-2">Validering</h2>
          <p className="text-gray-700 text-sm mb-2">
            Modellen valideres med <strong>5-fold StratifiedKFold kryssvalidering</strong>.
            AUC (Area Under ROC Curve) rapporteres som gjennomsnitt ± standardavvik over 5 folder.
            Modellen godkjennes ikke for deploy hvis gjennomsnittlig AUC er under 0,68.
          </p>
          <p className="text-gray-700 text-sm">
            Kalibrering (CalibratedClassifierCV) planlegges i Fase 2 — frem til da vises relativ
            risiko («X× høyere enn snitt»), ikke absolutte sannsynligheter.
          </p>
        </section>

        <section className="mb-8">
          <h2 className="text-lg font-bold text-dark mb-2">Modell-nivå justering (UK DVSA)</h2>
          <p className="text-gray-700 text-sm mb-2">
            Norsk PKK-data inneholder merke men ikke modell. For modell-nivå prediksjon bruker vi
            britiske MOT-data fra <strong>UK DVSA</strong> (100M+ inspeksjoner, åpen lisens) til å
            beregne relative justeringsfaktorer per modell og drivlinje.
          </p>
          <p className="text-gray-700 text-sm mb-2">
            UK-faktorer brukes kun for <strong>klimauavhengige kapitler</strong> (kap 0 og kap 4).
            For klimapåvirkede kapitler (bremser, styring, fjæring) brukes norsk merke-baseline.
          </p>
          <p className="text-gray-700 text-sm">
            Faktorene erstattes gradvis av norske observasjoner etter hvert som tjenesten vokser
            (Bayesiansk oppdatering, Fase 3).
          </p>
        </section>

        <section className="mb-8">
          <h2 className="text-lg font-bold text-dark mb-2">Begrensninger</h2>
          <ul className="text-gray-700 text-sm list-disc pl-5 space-y-1">
            <li>Prediksjon basert på statistiske mønstre — ikke garanti for individuelt resultat.</li>
            <li>Modell-faktorer for klimapåvirkede kapitler reflekterer norsk merke-gjennomsnitt, ikke din spesifikke modell.</li>
            <li>Biler under 4 år mangler PKK-historikk.</li>
            <li>Kilometerstand oppgis manuelt og er ikke verifisert.</li>
          </ul>
        </section>

        <section>
          <h2 className="text-lg font-bold text-dark mb-2">Oppdateringsfrekvens</h2>
          <p className="text-gray-700 text-sm">
            Modellen retrener kvartalsvis via GitHub Actions (januar, april, juli, oktober) mot de
            nyeste SVV PKK-filene.
          </p>
        </section>
      </div>
    </main>
  );
}
