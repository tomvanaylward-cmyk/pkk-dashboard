const RECALL_API = "https://www.vegvesen.no/ws/no/vegvesen/kjoretoy/tilbakekalling/liste/kjennemerke";

export interface RecallEntry {
  tittel:   string;
  dato:     string;
  url:      string | null;
}

export async function fetchRecalls(regnr: string): Promise<RecallEntry[]> {
  try {
    const resp = await fetch(`${RECALL_API}?kjennemerke=${regnr}`, {
      next: { revalidate: 3600 },
    });
    if (!resp.ok) return [];
    const data = await resp.json();
    return (data.tilbakekallinger ?? []).map((r: Record<string, unknown>) => ({
      tittel: String(r.tittel ?? ""),
      dato:   String(r.dato   ?? ""),
      url:    r.url ? String(r.url) : null,
    }));
  } catch {
    return [];
  }
}
