import { NextRequest, NextResponse } from "next/server";
import { getDb } from "@/lib/db";

export const runtime = "edge";

interface BookingLeadBody {
  regnr:    string;
  verksted: string;
  kapitler: string[];
  merke?:   string;
  modell?:  string;
  aargang?: number;
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: BookingLeadBody;
  try {
    body = await req.json() as BookingLeadBody;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { regnr, verksted, kapitler = [], merke, modell, aargang } = body;
  if (!regnr || !verksted) {
    return NextResponse.json({ error: "regnr and verksted are required" }, { status: 400 });
  }

  try {
    const sql = getDb();
    await sql`
      INSERT INTO booking_leads (regnr, verksted, kapitler, merke, modell, aargang)
      VALUES (${regnr}, ${verksted}, ${kapitler}, ${merke ?? null}, ${modell ?? null}, ${aargang ?? null})
    `;
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("DB insert failed:", e);
    return NextResponse.json({ error: "Database error" }, { status: 500 });
  }
}
