export interface SVVKjoretoy {
  regnr:     string;
  merke:     string;
  modell:    string;
  aargang:   number | null;
  drivstoff: "BENSIN" | "DIESEL" | "ELEKTRISK" | "HYBRID" | "ANNET";
  drivlinje: "4WD" | "FORHJUL" | "BAKHJUL" | "UKJENT";
  euFrist:   string | null;   // ISO date string, e.g. "2025-11-15"
  farge:     string | null;
}

export interface SVVError {
  error: string;
  status: number;
}
