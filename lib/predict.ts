import type { SVVKjoretoy } from "./svv-types";

export interface ChapterResult {
  chapter:      string;
  kapNr:        number;
  baseline:     number;
  prob:         number;
  relativRisiko: number;
  ukFaktorBrukt: boolean;
}

export interface PredictionResult {
  regnr:          string;
  merke:          string;
  modell:         string;
  aargang:        number | null;
  drivlinje:      string;
  km:             number;
  kmBucket:       string;
  kmPerYear:      number;
  overall:        number;
  chapters:       ChapterResult[];
  ingenHistorikk: boolean;
}

type CoefficientsJSON = {
  meta:             { pass_rate: number; [k: string]: unknown };
  intercept:        number;
  brand:            Record<string, number>;
  fuel:             Record<string, number>;
  ctrl_type:        Record<string, number>;
  weight:           Record<string, number>;
  km_bucket:        Record<string, number>;
  numeric:          { km_per_year_scaled: number; age_scaled: number; insp_scaled: number };
  scaler:           { km_per_year_mean: number; km_per_year_std: number;
                      age_mean: number; age_std: number;
                      insp_mean: number; insp_std: number };
  fingerprint: Record<string, {
    baseline:  number;
    intercept: number;
    brand:     Record<string, number>;
    fuel:      Record<string, number>;
    km_bucket: Record<string, number>;
    numeric:   { km_per_year_scaled: number; age_scaled: number; insp_scaled: number };
    scaler:    { km_per_year_mean: number; km_per_year_std: number;
                 age_mean: number; age_std: number;
                 insp_mean: number; insp_std: number };
  }>;
  model_adjustments: Record<string, Record<string, Record<string, Record<string, { factor: number; n: number }>>>>;
};

// Singleton — loaded once
let _coefs: CoefficientsJSON | null = null;

export async function loadCoefficients(): Promise<CoefficientsJSON> {
  if (_coefs) return _coefs;
  const resp = await fetch("/coefficients.json");
  if (!resp.ok) throw new Error("Failed to load model coefficients");
  _coefs = await resp.json() as CoefficientsJSON;
  return _coefs;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function kmToBucket(km: number): string {
  if (km < 50_000)  return "0-50k";
  if (km < 100_000) return "50-100k";
  if (km < 150_000) return "100-150k";
  return "150k+";
}

function normalizeFuel(drivstoff: string): string {
  const d = drivstoff.toUpperCase();
  if (d.includes("ELEKTR") && !d.includes("HYBRID")) return "BEV";
  if (d.includes("HYBRID"))  return "Hybrid";
  if (d.includes("DIESEL"))  return "Diesel";
  return "Petrol";
}

function drivlinjeToCategory(drivlinje: string): string {
  if (drivlinje === "4WD") return "4WD";
  return "OTHER";
}

const CHAPTER_KAP: Record<string, number> = {
  "Identification & documents": 0,
  "Brakes":                     1,
  "Steering":                   2,
  "Visibility":                 3,
  "Lights & electrical":        4,
  "Axles, wheels & tyres":      5,
  "Chassis & body":             6,
  "Other equipment":            7,
  "Noise & emissions":          8,
  "Checks during drive":        9,
  "Environment":                10,
};

const UK_VALID_CHAPTERS = new Set([
  "Identification & documents",
  "Lights & electrical",
]);

export async function predict(
  kjoretoy: SVVKjoretoy,
  km: number,
  currentYear: number = new Date().getFullYear(),
): Promise<PredictionResult> {
  const coefs = await loadCoefficients();

  const vehicleYear = kjoretoy.aargang ?? (currentYear - 10);
  const age       = Math.max(1, currentYear - vehicleYear);
  const kmBucket  = kmToBucket(km);
  const kmPerYear = Math.min(km / age, 80_000);
  const fuel      = normalizeFuel(kjoretoy.drivstoff);
  const brand     = kjoretoy.merke.toUpperCase().substring(0, 30);
  const modell    = kjoretoy.modell.toUpperCase().trim();

  const ingenHistorikk = age < 4;

  const chapters: ChapterResult[] = [];

  for (const [chapterName, fp] of Object.entries(coefs.fingerprint)) {
    const sc = fp.scaler;

    const kmPerYearScaled = (kmPerYear - sc.km_per_year_mean) / sc.km_per_year_std;
    const ageScaled       = (age       - sc.age_mean)         / sc.age_std;
    const inspNum         = Math.max(1, Math.floor((age - 4) / 2) + 1);
    const inspScaled      = (inspNum   - sc.insp_mean)        / sc.insp_std;

    let logit = fp.intercept
      + (fp.brand[brand]             ?? 0)
      + (fp.fuel[fuel]               ?? 0)
      + (fp.km_bucket?.[kmBucket]    ?? 0)
      + fp.numeric.km_per_year_scaled * kmPerYearScaled
      + fp.numeric.age_scaled         * ageScaled
      + fp.numeric.insp_scaled        * inspScaled;

    let ukFaktorBrukt = false;
    if (UK_VALID_CHAPTERS.has(chapterName)) {
      const makeFactors  = coefs.model_adjustments?.[brand];
      const modelFactors = makeFactors?.[modell];
      const dtCategory   = drivlinjeToCategory(kjoretoy.drivlinje);
      const factor       = modelFactors?.[dtCategory]?.[chapterName]?.factor;
      if (factor !== undefined) {
        logit += Math.log(factor);
        ukFaktorBrukt = true;
      }
    }

    const prob = sigmoid(logit);
    const relativRisiko = fp.baseline > 0 ? prob / fp.baseline : 1.0;

    chapters.push({
      chapter:       chapterName,
      kapNr:         CHAPTER_KAP[chapterName] ?? 0,
      baseline:      fp.baseline,
      prob:          Math.round(prob * 1000) / 1000,
      relativRisiko: Math.round(relativRisiko * 100) / 100,
      ukFaktorBrukt,
    });
  }

  chapters.sort((a, b) => b.relativRisiko - a.relativRisiko);

  const overall = chapters.length > 0
    ? Math.round((chapters.reduce((s, c) => s + c.relativRisiko, 0) / chapters.length) * 100) / 100
    : 1.0;

  return {
    regnr:          kjoretoy.regnr,
    merke:          kjoretoy.merke,
    modell:         kjoretoy.modell,
    aargang:        kjoretoy.aargang,
    drivlinje:      kjoretoy.drivlinje,
    km,
    kmBucket,
    kmPerYear:      Math.round(kmPerYear),
    overall,
    chapters,
    ingenHistorikk,
  };
}
