export default function BookingCTA() {
  return (
    <div className="bg-white rounded-2xl border border-mint p-6 mt-6">
      <h2 className="font-bold text-dark text-lg mb-1">Book verkstedstime</h2>
      <p className="text-gray-600 text-sm mb-4">
        Få fikset mangler før EU-kontrollen. Tilgjengelig hos partnerverksteder.
      </p>
      <div className="flex flex-col sm:flex-row gap-3">
        <button
          disabled
          className="flex-1 bg-green text-white font-semibold py-3 px-4 rounded-lg opacity-60 cursor-not-allowed text-sm"
          title="Kommer snart — vi inngår avtale med verksteder"
        >
          Mekonomen — Book tid
        </button>
        <button
          disabled
          className="flex-1 bg-dark text-white font-semibold py-3 px-4 rounded-lg opacity-60 cursor-not-allowed text-sm"
          title="Kommer snart"
        >
          Snap Drive — Book tid
        </button>
      </div>
      <p className="text-xs text-gray-400 mt-3">
        Partnerintegrasjon er under etablering. Booking-funksjon aktiveres snart.
      </p>
    </div>
  );
}
