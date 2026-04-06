# PKK EU-kontroll — Norway Vehicle Inspection Predictor

A logistic regression model trained on Statens vegvesen's open PKK dataset, served as a static web page via GitHub Pages. The model retrains **automatically every night** via GitHub Actions.

## Live site
After setup: `https://tomvanaylward-cmyk.github.io/pkk-dashboard/`

## Architecture

```
vegvesen/periodisk-kjoretoy-kontroll (source data, updated quarterly)
        ↓
GitHub Actions (runs daily at 03:00 UTC)
        ↓
scripts/train.py  — downloads ZIPs, fits logistic regression
        ↓
docs/coefficients.json  — committed to repo
        ↓
GitHub Pages  — serves docs/index.html + coefficients.json
```

## Setup (10 minutes)

### 1. Fork or create this repo
Create a new repo on GitHub and push all these files.

### 2. Enable GitHub Pages
- Go to your repo → **Settings** → **Pages**
- Source: **Deploy from a branch**
- Branch: `gh-pages` / root
- Click **Save**

### 3. Enable Actions write permissions
- Go to **Settings** → **Actions** → **General**
- Under "Workflow permissions" → select **Read and write permissions**
- Click **Save**

### 4. Run the workflow manually (first time)
- Go to **Actions** → **Retrain PKK model & deploy**
- Click **Run workflow**
- Wait ~3–5 minutes for it to finish

### 5. Visit your site
`https://<your-username>.github.io/pkk-dashboard/`

After that, the model retrains every night automatically and re-deploys if the coefficients change.

## Local development

```bash
pip install -r scripts/requirements.txt
python scripts/train.py
# opens docs/coefficients.json with real coefficients

# serve locally
cd docs && python -m http.server 8000
# visit http://localhost:8000
```

## Data source
- **Publisher**: Statens vegvesen (Norwegian Public Roads Administration)
- **Dataset**: [periodisk-kjoretoy-kontroll](https://github.com/vegvesen/periodisk-kjoretoy-kontroll)
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0)
- **Update frequency**: Quarterly (new zip files added to the GitHub repo)

## Model details
Logistic regression with:
- Brand (one-hot encoded, top brands by volume)
- Fuel type (EV, Diesel, Bensin, Hybrid)
- Vehicle age (years since first registration)
- Odometer reading (km, standardised)
- Control type (Periodisk P / Etterkontroll E)
- Fylke (county, one-hot encoded)

Features are preprocessed with `ColumnTransformer` (OneHotEncoder + StandardScaler). Coefficients are exported to `docs/coefficients.json` and read directly by the frontend.

## Disclaimer
Not affiliated with Statens vegvesen. Predictions are statistical estimates based on historical data and should not be used as a guarantee of inspection outcomes.
