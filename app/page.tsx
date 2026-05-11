import RegnrForm from "@/components/RegnrForm";

export const metadata = {
  title: "EU-kontroll Risikosjekk — Sjekk bilen din",
  description:
    "Sjekk risikoen for at bilen din stryker til EU-kontrollen. Basert på historiske PKK-data fra Statens vegvesen.",
};

export default function HomePage() {
  return (
    <main className="min-h-screen bg-pale">
      <header className="bg-dark text-white px-6 py-4 flex items-center gap-3">
        <span className="font-bold text-lg tracking-tight">EU-sjekk</span>
        <span className="text-mint text-sm opacity-75">Beta</span>
      </header>

      <div className="max-w-2xl mx-auto px-4 py-12 sm:py-20">
        <h1 className="text-3xl sm:text-4xl font-bold text-dark mb-3 leading-tight">
          Vil bilen din bestå<br />EU-kontrollen?
        </h1>
        <p className="text-gray-600 mb-8 text-base sm:text-lg">
          1 av 4 norske biler stryker ved første kontroll. Sjekk risikoen basert på historiske data — kapittel for kapittel.
        </p>

        <div className="bg-white rounded-2xl shadow-sm border border-mint p-6 sm:p-8">
          <RegnrForm />
        </div>

        <p className="text-xs text-gray-400 mt-6 text-center">
          Regnr slås opp mot Statens vegvesen. Vi lagrer ikke eierinformasjon.
        </p>
      </div>
    </main>
  );
}
