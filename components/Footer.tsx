export default function Footer() {
  return (
    <footer className="border-t border-mint bg-white mt-12 py-6 px-4">
      <div className="max-w-2xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-gray-500">
        <div className="flex gap-4">
          <a href="/modell" className="hover:text-green transition-colors">Om modellen</a>
          <a href="/faq"    className="hover:text-green transition-colors">FAQ</a>
        </div>
        <p className="text-center sm:text-right">
          Data: Statens vegvesen (NLOD) · UK DVSA (Open Government Licence)
        </p>
      </div>
      <div className="max-w-2xl mx-auto mt-3 text-xs text-gray-400 text-center">
        Prediksjoner er statistiske estimater og ikke garantier for individuelt utfall. EU-sjekk er ikke tilknyttet Statens vegvesen eller NAF.
        <span className="mx-2">·</span>
        <a href="/faq#personvern" className="underline hover:text-green transition-colors">Personvern</a>
      </div>
    </footer>
  );
}
