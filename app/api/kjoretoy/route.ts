import { NextRequest, NextResponse } from "next/server";
import type { SVVKjoretoy, SVVError } from "@/lib/svv-types";

export const runtime = "edge";

const SVV_BASE = "https://www.vegvesen.no/ws/no/vegvesen/kjoretoy/felles/datautlevering/enkeltoppslag/kjoretoydata";

function normalizeDrivlinje(raw: string): SVVKjoretoy["drivlinje"] {
  const r = raw.toUpperCase().replace(/\s/g, "");
  if (r.includes("4WD") || r.includes("AWD") || r.includes("FIREHJUL")) return "4WD";
  if (r.includes("FORHJUL") || r.includes("FRONT"))  return "FORHJUL";
  if (r.includes("BAKHJUL") || r.includes("REAR"))   return "BAKHJUL";
  return "UKJENT";
}

export async function GET(req: NextRequest): Promise<NextResponse<SVVKjoretoy | SVVError>> {
  const regnr = req.nextUrl.searchParams.get("regnr");
  if (!regnr || !/^[A-Z]{2}\d{4,5}$/i.test(regnr.trim())) {
    return NextResponse.json({ error: "Ugyldig registreringsnummer", status: 400 }, { status: 400 });
  }

  const apiKey = process.env.SVV_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "API key not configured", status: 500 }, { status: 500 });
  }

  const url = `${SVV_BASE}?kjennemerke=${regnr.toUpperCase().replace(/\s/g, "")}`;
  const resp = await fetch(url, {
    headers: {
      "SVV-Authorization": apiKey,
      "Accept": "application/json",
    },
    next: { revalidate: 86400 },  // Cache 24h in Vercel data cache
  });

  if (!resp.ok) {
    if (resp.status === 404) {
      return NextResponse.json({ error: "Kjøretøy ikke funnet", status: 404 }, { status: 404 });
    }
    return NextResponse.json({ error: `SVV API feil: ${resp.status}`, status: resp.status }, { status: 500 });
  }

  // SVV JSON structure (as documented at vegvesen.no/datautlevering)
  const raw = await resp.json();
  const k = raw.kjoretoydataListe?.[0];
  if (!k) {
    return NextResponse.json({ error: "Ingen data for dette kjøretøyet", status: 404 }, { status: 404 });
  }

  const result: SVVKjoretoy = {
    regnr:     regnr.toUpperCase(),
    merke:     k.godkjenning?.tekniskGodkjenning?.tekniskeData?.generelt?.merke?.[0]?.merke ?? "UKJENT",
    modell:    k.godkjenning?.tekniskGodkjenning?.tekniskeData?.generelt?.handelsbetegnelse?.[0] ?? "",
    aargang:   k.godkjenning?.tekniskGodkjenning?.tekniskeData?.generelt?.typegodkjenningsunderlag?.[0]?.sistEndretAr
               ?? new Date().getFullYear(),
    drivstoff: k.godkjenning?.tekniskGodkjenning?.tekniskeData?.motorOgDrivverk?.motor?.[0]?.drivstoff?.[0]?.kodeBeskrivelse?.toUpperCase() ?? "ANNET",
    drivlinje: normalizeDrivlinje(
      k.godkjenning?.tekniskGodkjenning?.tekniskeData?.motorOgDrivverk?.drivstoffOgGirkasse?.[0]?.drivhjulskode?.kodeBeskrivelse ?? ""
    ),
    euFrist:   k.periodiskKjoretoyKontroll?.kontrollfrist ?? null,
    farge:     k.godkjenning?.tekniskGodkjenning?.tekniskeData?.karosseriOgLasteplan?.rFarge?.[0]?.kodeBeskrivelse ?? null,
  };

  return NextResponse.json(result);
}
