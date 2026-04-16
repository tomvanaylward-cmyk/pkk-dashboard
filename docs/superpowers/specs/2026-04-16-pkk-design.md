# PKK Prediksjonsplattform — Design Spec

**Dato:** 2026-04-16  
**Status:** Klar for review  
**Fase:** Fase 1 (MVP)

---

## 1. Problemdefinisjon

~2,6 millioner personbiler i Norge gjennomgår EU-kontroll regelmessig. Omtrent 25 % stryker ved første kontroll. Det finnes ingen norsk tjeneste som gir bileieren forhåndsvarsel om risiko basert på historiske data. Statens vegvesen publiserer PKK-data (periodisk kjøretøykontroll) som åpne data kvartalsvis — men ingen har bygget et produkt som gjør dette tilgjengelig og handlingsrettet for vanlige bileiere.

---

## 2. Mål

**Primærmål (Fase 1):**
- Gi privatpersoner en regnr-basert EU-kontroll-risikoscore med kapittel-breakdown (kap 0–10)
- Vise hvilke konkrete feil som er vanligst for tilsvarende biler
- Lenke til booking hos partnerverksted (Mekonomen, Snap Drive)

**Sekundærmål (Fase 2–3):**
- B2B API for verksteder, forsikringsaktører og bilaktører
- Servicehistorikk som feature for bedre prediksjon
- Partnerintegrasjoner (Finn.no, forsikringsselskaper)

---

## 3. Brukerflyt

```
Landingsside
  └─ Skriv inn regnr
       └─ SVV Kjøretøyreg-API → hent merke, modell, årsmodell, km (bruker-input)
            └─ ML-modell (browser-side, coefficients.json)
                 └─ Resultatside
                      ├─ Relativ risikoscore ("2,4× høyere enn snitt")
                      ├─ Kapittelkort kap 0–10 (risiko per kapittel)
                      ├─ Recall-banner (hvis åpen tilbakekalling)
                      ├─ Anbefaling: hva bør fikses
                      └─ CTA: Book time → Mekonomen/Snap Drive deeplink
```

---

## 4. Datakilder

| Kilde | Type | Fase | Hva vi bruker det til |
|---|---|---|---|
| SVV PKK-data | Åpen, kvartalsvis | 1 | Historisk treningsdata, ~2M rader/år |
| SVV Kjøretøyreg-API | API, gratis (Altinn) | 1 | regnr → merke, modell, årsmodell, EU-frist |
| Transportstyrelsen SE | Åpen, CC0 | 1 | Modell-nivå justeringsfaktorer (triangulering) |
| SVV Tilbakekallingsregister | Åpen | 1 | Recall-banner per merke/modell/år |
| Servicehistorikk | Bruker-input | 2 | Bedre km_per_year-estimat, komponent-slitasje |
| Finn.no / NAF | Partneravtale | 3 | Bruktbil-historikk, egne EU-kontroll-rapporter |

**Merk:** km er ikke tilgjengelig via SVV-API og må oppgis av brukeren i Fase 1.

---

## 5. ML-modell

### 5.1 Valgt tilnærming: Approach B — Solid fundament

Vi beholder logistisk regresjon (én modell per kapittel, kap 0–10 = 11 modeller) men fikser de to kritiske feilene i nåværende pipeline:

1. **Validering:** Bytt fra train=eval til **5-fold StratifiedKFold**. Rapporter mean AUC ± std for hvert kapittel.
2. **Km-håndtering:** Bytt fra kontinuerlig StandardScaler til **km-buckets** (kategorisk): `0–50k`, `50–100k`, `100–150k`, `150k+`. PKK runder km til nærmeste 50 000 — kontinuerlig behandling er feil.

LightGBM vurderes i Fase 2 etter at fundamentet er stabilt.

### 5.2 Features (prioritert)

| Prioritet | Feature | Kilde | Notis |
|---|---|---|---|
| 1 | `km_per_year` | beregnet (km / alder) | Sterk prediktor for slitasje |
| 2 | `km_bucket` | PKK-data (avrundet) | Erstatter kontinuerlig km |
| 3 | `brand_model` | SVV API + PKK-data | Modell-nivå via triangulering med SE |
| 4 | `trafikkfarlig_feil` | PKK-data | Binær, høy prediktiv styrke |
| 5 | `fylke` | PKK-data | Geografisk variasjon (salt, klima) |
| 6 | `alder` | PKK-data / SVV API | Allerede i bruk |
| 7 | `merke` | PKK-data | Allerede i bruk |
| 8 | `drivstoff` | PKK-data | Allerede i bruk |
| 9 | `bruktimport` | fase 2 | Importerte biler har høyere feilrate |
| 10 | `historisk_feilrate_per_modell` | beregnet fra PKK | Fase 2 |

### 5.3 Output-format

`coefficients.json` eksporteres etter trening og lastes i browser for inference:

```json
{
  "failure_fingerprint": {
    "kap 1": { "intercept": -1.24, "coefs": { "km_bucket_100_150k": 0.43, ... }, "auc": 0.74, "auc_std": 0.02 },
    ...
  },
  "defect_analysis": { ... },
  "metadata": { "trained_at": "2026-04-16", "n_rows": 480000, "cv_folds": 5 }
}
```

**Kalibrering:** Legg til `CalibratedClassifierCV` etter k-fold for å sikre at "34 % risiko" faktisk betyr 34 %.

### 5.4 Presentasjon av risiko

Vis **relativ risiko** fremfor absolutt sannsynlighet:
- ✅ "2,4× høyere risiko enn snitt for tilsvarende biler"
- ❌ "34,2 % sjanse for stryk" (misvisende uten kalibrering)

---

## 6. Teknisk arkitektur

### 6.1 Stack

| Komponent | Teknologi |
|---|---|
| Frontend | Next.js 14, App Router, Tailwind CSS |
| Hosting | Vercel |
| API-lag | Vercel Edge Functions |
| Database | Neon PostgreSQL (booking leads) |
| ML inference | Browser-side (coefficients.json lastet som statisk asset) |
| ML trening | Python (train.py), GitHub Actions kvartalsvis |
| Data | Parquet (konvertert fra SVV ZIP/CSV ved første kjøring) |

### 6.2 Sider

**Side 1: Sjekk bil** (`/`)
- Hero med regnr-input
- SVV Kjøretøyreg-API oppslag (Edge Function)
- Km-input (bruker oppgir selv)
- Resultatvisning: relativ risikoscore, 11 kapittelkort, recall-banner, anbefaling, booking-CTA

**Side 2: Om modellen** (`/modell`)
- AUC per kapittel (med ± std fra k-fold)
- Features som brukes
- Datakilder og metodikk
- Begrensninger og forutsetninger

**Side 3: FAQ** (`/faq`)
- Hva er EU-kontroll?
- Hva betyr kapittel X?
- Kan jeg stole på prediksjonen?
- Hvem er vi?

### 6.3 API-endepunkter

```
GET  /api/kjoretoy?regnr=AB12345   → { merke, modell, aargang, euFrist, ... }
POST /api/booking-lead              → { regnr, verksted, kapittel[] } → lagres i Neon
```

B2B API (Fase 2):
```
POST /api/v1/predict   (API-nøkkel required)
Body: { regnr, km, servicehistorikk? }
Response: { risiko_score, relativ_risiko, kapitler: [...] }
```

### 6.4 GitHub Actions pipeline

```yaml
# Kvartalsvis (ikke daglig) — hopper over filer som allerede er lastet ned
schedule: "0 4 1 */3 *"   # 1. januar, april, juli, oktober

Steps:
  1. Sjekk om ny kvartalsfil finnes på SVV
  2. Hopp over hvis allerede lastet ned (smart incremental)
  3. Last ned ny fil → konverter til Parquet
  4. Kjør train.py med 5-fold CV
  5. Valider AUC (feiler hvis < 0.68)
  6. Eksporter coefficients.json
  7. Commit + push → Vercel auto-deploy
```

---

## 7. Branding & Design

- **Fargepalett:** NAF-inspirert uten NAF-logo eller NAF-navn
  - `--dark: #113824` (mørkegrønn, bakgrunn/header)
  - `--green: #267C4F` (primær grønn)
  - `--yellow: #FFD816` (aksent/highlight)
  - `--mint: #CBE0D5` (sekundær grønn, border)
  - `--pale: #E9F3E6` (lys bakgrunn)
- **Font:** System-UI / Inter
- **Tone:** Nøktern, faktabasert, trygg. Aldri alarmistisk.
- **Risikovisning:** "1 av 4 biler med tilsvarende profil stryker" fremfor prosent

---

## 8. Ikke i scope (Fase 1)

- Brukerkontoer / innlogging
- Push-varsling / e-postvarsling
- Servicehistorikk (Fase 2)
- B2B API (Fase 2)
- Finn.no / NAF-integrasjon (Fase 3)
- Survival-analyse / "km til forventet feil" (Fase 3)
- Flerspråklig støtte
- Mobilapp

---

## 9. Risikoer og mitigering

| Risiko | Sannsynlighet | Mitigering |
|---|---|---|
| AUC overvurdert (train=eval) | Høy — bekreftet | k-fold CV (dette er selve fiksen) |
| Km avrundingsfeil (50k-buckets) | Høy — bekreftet | km-buckets som kategorisk feature |
| SVV API-tilgang tar tid | Middels | Søk via Altinn umiddelbart; fallback: manuell input |
| Modell lover mer enn den holder | Middels | Relativ risiko-framing + kalibrering |
| Regulatorisk (B2B forsikring) | Lav | Juridisk avklaring i Fase 2 |

---

## 10. Suksesskriterier (Fase 1)

- [ ] AUC ≥ 0.70 (mean over 5-fold) for minst 7 av 11 kapitler
- [ ] Regnr-oppslag fungerer (SVV API integrert)
- [ ] Kapittel-breakdown er forståelig for ikke-teknisk bruker (intern brukertest)
- [ ] Booking-CTA sender til korrekt Mekonomen/Snap Drive deeplink
- [ ] ≥ 100 unike brukere i løpet av første uke etter lansering

---

## 11. Veivalg (oppsummert)

| # | Spørsmål | Valg | Begrunnelse |
|---|---|---|---|
| 1 | Primærbruker | Privatkunde | Bredest nedslagsfelt; B2B legges til |
| 2 | Resultat-layout | Kapittel-breakdown | Handlingsrettet; brukeren vet hva som må fikses |
| 3 | CTA | Booking hos partnerverksted | Inntektspotensial + brukerverdi |
| 4 | Arkitektur | Next.js / Vercel | Edge Functions + skalerbar til B2B |
| 5 | Branding | NAF-farger, ikke NAF-navn | Troverdighet uten juridisk binding |
| 6 | ML-validering | 5-fold StratifiedKFold | Ærlig AUC uten full ombygging |
| 7 | Datakilder | SVV + SE + Recall | Alle åpne og gratis; gir modell-nivå |
| 8 | Servicehistorikk | Fase 2 | Krever onboarding; utsettes til brukerbasen er etablert |

---

## 12. Roadmap

### Fase 1 — Uke 1–8: MVP & Solid fundament
- 5-fold StratifiedKFold CV
- km-buckets, km_per_year som features
- Parquet-konvertering av eksisterende data
- SVV API-integrasjon (regnr-lookup)
- Kapittel-breakdown UI (Next.js)
- Recall-banner
- Booking-CTA (Mekonomen / Snap Drive deeplinks)
- Om modellen-side og FAQ

### Fase 2 — Uke 9–20: Servicehistorikk & B2B alpha
- Servicehistorikk bruker-input
- CalibratedClassifierCV
- LightGBM A/B-test mot LogReg
- SHAP-forklaringer per kapittel
- B2B API v1 (API-nøkkel-basert)
- Verksted-dashboard (leads-oversikt)

### Fase 3 — Uke 21–52: Partnerintegrasjoner & skala
- Finn.no badge-integrasjon
- Forsikrings-API (enterprise)
- Fleet-screening (bulk regnr-oppslag)
- EU-varslings-push (SMS / e-post)
- Survival-analyse (km til forventet feil)
- Internasjonalisering (SE / DK)

---

## 13. Umiddelbare neste steg

1. **Søk SVV API-tilgang nå** — via Altinn, behandlingstid 1–3 uker
2. **Implementer k-fold CV** i `scripts/train.py` + bytt km til km-buckets
3. **Bootstrap Next.js-app** på Vercel med Neon PostgreSQL
4. **Bygg Sjekk-bil-siden** med statisk `coefficients.json` (SVV API kobles inn når tilgang er på plass)
