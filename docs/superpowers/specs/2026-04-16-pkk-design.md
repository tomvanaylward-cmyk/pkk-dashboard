# PKK Prediksjonsplattform — Design Spec

**Dato:** 2026-04-16
**Sist oppdatert:** 2026-04-16 (forhåndsanalyse — andre forhold)
**Status:** Under review
**Fase:** Fase 1 (MVP)

---

## 1. Problemdefinisjon

~2,6 millioner personbiler i Norge gjennomgår EU-kontroll regelmessig. Omtrent 25 % stryker ved første kontroll. Det finnes ingen norsk tjeneste som gir bileieren forhåndsvarsel om risiko basert på historiske data. Statens vegvesen publiserer PKK-data (periodisk kjøretøykontroll) som åpne data kvartalsvis — men ingen har bygget et produkt som gjør dette tilgjengelig og handlingsrettet for vanlige bileiere.

---

## 2. Mål

**Primærmål (Fase 1):**
- Gi privatpersoner en regnr-basert EU-kontroll-risikoscore med kapittel-breakdown (kap 0–10)
- Vise hvilke konkrete feil som er vanligst for tilsvarende biler
- Lenke til booking hos partnerverksted (Mekonomen, Snap Drive) — mockup CTA i Fase 1, reell deeplink i Fase 2

**Sekundærmål (Fase 2–3):**
- B2B API for verksteder, forsikringsaktører og bilaktører
- Servicehistorikk som feature for bedre prediksjon
- Partnerintegrasjoner (Finn.no, forsikringsselskaper)

---

## 3. Brukerflyt

```
Landingsside
  └─ Skriv inn regnr
       └─ SVV Enkeltoppslag-API → hent merke, modell, årsmodell, drivstoff, drivlinje
            └─ Bruker oppgir km manuelt
                 └─ ML-modell (browser-side, coefficients.json)
                      └─ Resultatside
                           ├─ Relativ risikoscore ("2,4× høyere enn snitt")
                           ├─ Kapittelkort kap 0–10 (risiko per kapittel)
                           ├─ Recall-banner (hvis åpen tilbakekalling)
                           ├─ Anbefaling: hva bør fikses
                           └─ CTA: Book time → Mekonomen/Snap Drive (mockup i Fase 1)
```

---

## 4. Datakilder

### 4.1 Oversikt

| Kilde | Type | Fase | Hva vi bruker det til |
|---|---|---|---|
| SVV PKK-data | Åpen, NLOD, kvartalsvis | 1 | Historisk treningsdata, ~2M rader/år |
| SVV Enkeltoppslag-API | Gratis, BankID, 50k/dag | 1 | regnr → merke, modell, årsmodell, drivlinje, EU-frist |
| UK DVSA MOT-data | Åpen, 100M+ rader, rad-nivå | 1 | Modell-nivå justeringsfaktorer (se §4.3) |
| SVV Tilbakekallingsregister | Åpen, NLOD | 1 | Recall-banner per merke/modell/år |
| Traficom Finland | Åpen, aggregert, årlig | 1 | Sekundær validering av modell-faktorer |
| Servicehistorikk | Bruker-input | 2 | Bedre km_per_year-estimat, komponent-slitasje |
| Finn.no / NAF | Partneravtale | 3 | Bruktbil-historikk, egne EU-kontroll-rapporter |

**Merk:** km er ikke tilgjengelig via SVV-API og må oppgis av brukeren i Fase 1.

### 4.2 SVV API — viktig avklaring

Det finnes to API-er fra SVV med ulike tilgangskrav:

| API | Tilgang | Krav |
|---|---|---|
| Tekniske opplysninger med eierinformasjon | Kun virksomheter | Maskinporten + org.nr + formell avtale |
| **Enkeltoppslag (teknisk info)** ← dette bruker vi | Alle inkl. privatpersoner | BankID → bestill API-nøkkel online |

Enkeltoppslag-APIet er gratis, krever ingen AS/org.nr, og har en grense på 50 000 kall per dag per nøkkel — mer enn tilstrekkelig for Fase 1. API-nøkkel bestilles på vegvesen.no med norsk BankID.

### 4.3 Modell-nivå prediksjon — metodikk

**Problemet:** Norske PKK-data inneholder `merke` men ikke `modell`. Vi kan dermed trene på merke-nivå (Volkswagen, Toyota osv.) men ikke modell-nivå (Golf, Corolla osv.).

**Opprinnelig plan (utgår):** Transportstyrelsen SE som trianguleringskilde. Avvist — rad-nivå data finnes ikke åpent; kun aggregert PDF/Excel. Tilgang krever kommersiell avtale.

**Valgt løsning: UK DVSA MOT-data som prior**

UK DVSA publiserer fullstendig rad-nivå inspeksjonsdata åpent:
- ✅ Merke + modell per inspeksjon
- ✅ Alle feilkategorier (tilsvarer PKK-kapitler)
- ✅ Kilometerstand
- ✅ 100M+ rader siden 2005, gratis, åpen lisens
- ✅ Nedlastbar fra Amazon S3

UK-data brukes til å beregne **relative modell-faktorer** (Golf vs VW-snitt), ikke absolutte sannsynligheter. Disse faktorene brukes som prior som justerer den norske merke-baselinjen.

### 4.4 Landforskjeller — begrensninger og mitigering

UK og Norge er ikke direkte sammenlignbare på alle kapitler. Viktige forskjeller:

| Faktor | Norge | UK |
|---|---|---|
| AWD-andel | ~50% av nybilsalg | ~20% |
| Veisalt | Høyt (vinter) | Middels |
| Temperatur | −20°C til +30°C | 0°C til +25°C |
| EV-andel nye biler | ~90% (2024) | ~25% |

**Mitigering — tre teknikker kombineres:**

**1. Drivetrain-segmentering:** SVV Enkeltoppslag-API returnerer drivlinje (AWD/FWD/RWD). UK-justeringsfaktorer beregnes separat per drivlinje, slik at norsk Golf AWD sammenlignes mot UK Golf AWD — ikke UK Golf FWD.

**2. Kapittelseleksjon:** UK-faktorer brukes kun på klimauavhengige kapitler:

| Kapittel | Klimapåvirket | UK-faktor gyldig |
|---|---|---|
| Kap 0 — Dokumenter | Nei | ✅ Ja |
| Kap 4 — Lys | Nei | ✅ Ja |
| Kap 7 — Støy/eksos | Delvis | 🟡 Med forbehold |
| Kap 1 — Bremser | Ja (salt, terreng) | ❌ Nei — norsk baseline |
| Kap 2 — Styring | Ja (rust, kulde) | ❌ Nei — norsk baseline |
| Kap 5 — Aksel/fjæring | Ja (salt) | ❌ Nei — norsk baseline |

For kapitler uten gyldig UK-faktor brukes norsk merke-baseline direkte.

**3. Bayesiansk oppdatering:** UK-faktorer er startpunktet (kald start). Etter hvert som norske brukere akkumuleres, vektes norsk observasjon gradvis tyngre:

```
faktor = (UK_prior × UK_vekt + NO_obs × NO_vekt) / (UK_vekt + NO_vekt)
```

Der `UK_vekt` starter høy og reduseres etter hvert som norske data bygges opp:
- Fase 1: UK-prior dominerer
- Fase 2 (~500+ norske Golf-oppslag): 50/50
- Fase 3 (~5000+ norske Golf-oppslag): Norsk data dominerer, UK-prior irrelevant

**Transparent kommunikasjon til bruker:**
> "Modell-faktorer er basert på britiske inspeksjonsdata (DVSA MOT), justert for drivlinje. Britiske data erstattes gradvis av norske data etter hvert som tjenesten vokser."

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
| 3 | `merke` | PKK-data | Basis for norsk baseline |
| 4 | `modell_faktor` | UK DVSA (se §4.3) | Justeringsfaktor per modell+drivlinje |
| 5 | `drivlinje` | SVV API | AWD/FWD/RWD — segmenterer UK-faktor |
| 6 | `trafikkfarlig_feil` | PKK-data | Binær, høy prediktiv styrke |
| 7 | `fylke` | PKK-data | Geografisk variasjon (salt, klima) |
| 8 | `alder` | PKK-data / SVV API | Allerede i bruk |
| 9 | `drivstoff` | PKK-data | Allerede i bruk |
| 10 | `bruktimport` | fase 2 | Importerte biler har høyere feilrate |
| 11 | `historisk_feilrate_per_modell_NO` | beregnet fra egne oppslag | Fase 3 — erstatter UK-prior |

### 5.3 Output-format

`coefficients.json` eksporteres etter trening og lastes i browser for inference:

```json
{
  "failure_fingerprint": {
    "kap 1": {
      "intercept": -1.24,
      "coefs": { "km_bucket_100_150k": 0.43, ... },
      "auc": 0.74,
      "auc_std": 0.02,
      "baseline": 0.18,
      "uk_factors_used": false
    },
    "kap 4": {
      "intercept": -1.51,
      "coefs": { ... },
      "auc": 0.71,
      "auc_std": 0.03,
      "baseline": 0.12,
      "uk_factors_used": true
    }
  },
  "model_adjustments": {
    "Golf": { "AWD": { "kap 4": 1.12 }, "FWD": { "kap 4": 0.94 } }
  },
  "defect_analysis": { ... },
  "metadata": {
    "trained_at": "2026-04-16",
    "n_rows": 480000,
    "cv_folds": 5,
    "calibrated": true,
    "uk_dvsa_rows_used": 2400000
  }
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
| Data (NO) | Parquet (konvertert fra SVV ZIP/CSV ved første kjøring) |
| Data (UK) | DVSA MOT CSV/ZIP fra Amazon S3, konvertert til Parquet |

### 6.2 Sider

**Side 1: Sjekk bil** (`/`)
- Hero med regnr-input
- SVV Enkeltoppslag-API oppslag (Edge Function, BankID API-nøkkel)
- Km-input (bruker oppgir selv)
- Resultatvisning: relativ risikoscore, 11 kapittelkort, recall-banner, anbefaling, booking-CTA

**Side 2: Om modellen** (`/modell`)
- AUC per kapittel (med ± std fra k-fold)
- Forklaring av UK DVSA-prior og Bayesiansk oppdatering
- Hvilke kapitler som bruker UK-faktorer vs norsk baseline
- Begrensninger: merke-nivå for klimapåvirkede kapitler

**Side 3: FAQ** (`/faq`)
- Hva er EU-kontroll?
- Hva betyr kapittel X?
- Hvorfor er modellen basert på britiske data?
- Kan jeg stole på prediksjonen?
- Hvem er vi?

### 6.3 API-endepunkter

```
GET  /api/kjoretoy?regnr=AB12345   → { merke, modell, aargang, drivstoff, drivlinje, euFrist, ... }
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
  1. Sjekk om ny norsk kvartalsfil finnes på SVV GitHub
  2. Hopp over hvis allerede lastet ned (smart incremental)
  3. Last ned ny fil → konverter til Parquet
  4. Last ned UK DVSA delta-filer (månedlig, Amazon S3)
  5. Beregn/oppdater modell_faktorer per modell+drivlinje
  6. Kjør train.py med 5-fold CV
  7. Valider AUC (feiler hvis < 0.68)
  8. Eksporter coefficients.json
  9. Commit + push → Vercel auto-deploy
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
- Maskinporten / enterprise SVV API (Fase 2 ved behov)
- Partneravtaler og booking-deeplinks (etter Fase 2 — CTA mockupes i MVP)
- Egendefinert domenenavn (utsettes — Vercel-subdomain til MVP er demonstrert)
- MC, tilhenger, varebil (viser feilmelding for ikke-personbil regnr)
- Utenlandsregistrerte biler (SVV API støtter kun norske regnr — tydelig feilmelding)

---

## 9. Risikoer og mitigering

| Risiko | Sannsynlighet | Mitigering |
|---|---|---|
| AUC overvurdert (train=eval) | Høy — bekreftet | k-fold CV (dette er selve fiksen) |
| Km avrundingsfeil (50k-buckets) | Høy — bekreftet | km-buckets som kategorisk feature |
| UK-faktorer overføres feil (klima/drivlinje) | Middels | Drivetrain-segmentering + kun klimauavhengige kapitler |
| SVV API-nøkkel tar tid | Lav | BankID-bestilling, typisk 1–2 dager; fallback: manuell merke/modell-dropdown |
| Modell lover mer enn den holder | Middels | Relativ risiko-framing + kalibrering + transparent UK-kommunikasjon |
| Regulatorisk (B2B forsikring) | Lav | Juridisk avklaring i Fase 2 |

---

## 10. Suksesskriterier (Fase 1)

- [ ] AUC ≥ 0.70 (mean over 5-fold) for minst 7 av 11 kapitler
- [ ] Regnr-oppslag fungerer (SVV Enkeltoppslag-API integrert)
- [ ] Drivlinje-segmentering implementert i UK-faktor-beregning
- [ ] Kapittel-breakdown er forståelig for ikke-teknisk bruker (intern brukertest)
- [ ] "Om modellen"-siden forklarer UK-prior og begrensninger
- [ ] Booking-CTA mockup vises på resultatsiden (reell deeplink tas i Fase 2 etter partneravtale)
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
| 7 | SVV API | Enkeltoppslag (ikke Maskinporten) | Tilgjengelig for privatpersoner med BankID |
| 8 | Modell-triangulering | UK DVSA (ikke Transportstyrelsen SE) | SE har ikke åpne rad-data; UK har 100M+ rader åpent |
| 9 | Landforskjeller | Drivetrain-segmentering + kapittelseleksjon + Bayesiansk prior | Nøytraliserer AWD/klima-bias; gradvis erstatning med norsk data |
| 10 | Servicehistorikk | Fase 2 | Krever onboarding; utsettes til brukerbasen er etablert |
| 11 | Analytics | Google Analytics 4 + cookie-samtykke-banner | Kjent verktøy; krever CookieYes/Cookiebot for GDPR-compliance |
| 12 | Partneravtaler | Utsatt til etter Fase 2 | MVP trenger ikke reelle deeplinks; mockup CTA holder for demo |
| 13 | Domenenavn | Utsatt (Vercel-subdomain for MVP) | pkk.no er tatt; avklar navn etter ledergruppe-demo |
| 14 | GDPR/personvern | Lagrer ikke eier — lav risiko | Regnr alene er ikke persondata; personvernerklæring i footer |

---

## 12. Roadmap

### Fase 1 — Uke 1–8: MVP & Solid fundament
- 5-fold StratifiedKFold CV
- km-buckets, km_per_year, drivlinje som features
- Parquet-konvertering (norsk + UK DVSA)
- UK DVSA modell-faktor-beregning (klimauavhengige kapitler, AWD/FWD/RWD-segmentert)
- SVV Enkeltoppslag-API integrasjon (regnr-lookup)
- Kapittel-breakdown UI (Next.js) — **mobil-first (375px primær)**
- WCAG 2.1 AA — kontrast, tastaturnavigasjon og skjermleser-støtte fra start
- Recall-banner
- Booking-CTA mockup (Mekonomen / Snap Drive — uten reell deeplink til Fase 2)
- «Ikke nok historikk»-visning for biler under 4 år (ingen PKK-data)
- Google Analytics 4 + cookie-samtykke-banner (CookieYes eller Cookiebot, gratis tier)
- Personvernerklæring og ansvarsfraskrivelse i footer
- Sentry (gratis tier) for frontend- og Edge Function-feilovervåking
- Om modellen-side (inkl. UK-prior forklaring) og FAQ

### Fase 2 — Uke 9–20: Servicehistorikk & B2B alpha
- Servicehistorikk bruker-input
- CalibratedClassifierCV
- LightGBM A/B-test mot LogReg
- SHAP-forklaringer per kapittel
- Bayesiansk oppdatering starter (norsk modell-data akkumuleres)
- B2B API v1 (API-nøkkel-basert)
- Verksted-dashboard (leads-oversikt)

### Fase 3 — Uke 21–52: Partnerintegrasjoner & skala
- Finn.no badge-integrasjon
- Forsikrings-API (enterprise)
- Fleet-screening (bulk regnr-oppslag)
- EU-varslings-push (SMS / e-post)
- Norsk modell-tabell erstatter UK-prior for populære modeller
- Survival-analyse (km til forventet feil)
- Internasjonalisering (SE / DK)

---

## 13. Åpne spørsmål (under avklaring)

**Avklart:**
- **GDPR / personvern:** Lagrer ikke eier — regnr alene er ikke persondata. Personvernerklæring legges i footer. ✅
- **Universell utforming:** WCAG 2.1 AA bygges inn fra start i Fase 1. ✅
- **Caching-strategi:** Server-side cache per regnr (24t TTL) innføres ved behov — OK å vente til etter MVP. ✅
- **Ansvarsfraskrivelse:** «Statistisk estimat, ikke garanti» — én setning i footer, nok for Fase 1. ✅
- **Analytics:** Google Analytics 4 + cookie-samtykke-banner (CookieYes gratis tier). ✅
- **Partneravtaler:** Utsatt til etter Fase 2. Booking-CTA mockupes uten reell deeplink i MVP. ✅
- **Domenenavn:** Utsatt. Vercel-subdomain brukes for MVP og ledergruppe-demo. ✅

**Fortsatt åpent:**
- **EV-spesifikk feilprofil:** Elbiler har annen inspeksjonsprofil (ingen motor/eksos). Bør modellen flagge EV separat, eller er drivstoff-featuret tilstrekkelig? Vurder i Fase 2.
- **coefficients.json-størrelse:** 11 modeller + UK-faktorer kan bli 500KB+. Vurder lazy-loading eller gzip-komprimering ved behov.
- **UK DVSA modellnavn-matching:** «Volkswagen Golf» (UK) vs «GOLF» (NO PKK). Trenger normalisering/alias-tabell i treningsskriptet.
- **Stalking-risiko:** Enhver kan slå opp andres bil. Vurder rate-limiting per IP (f.eks. 20 oppslag/time) i Edge Function.

---

## 14. Umiddelbare neste steg

1. **Bestill SVV Enkeltoppslag API-nøkkel** — vegvesen.no med BankID, typisk 1–2 dager
2. **Last ned UK DVSA MOT-data** fra Amazon S3, beregn modell-faktorer
3. **Implementer k-fold CV** i `scripts/train.py` + bytt km til km-buckets + legg til drivlinje
4. **Bootstrap Next.js-app** på Vercel med Neon PostgreSQL
5. **Bygg Sjekk-bil-siden** med statisk `coefficients.json`
