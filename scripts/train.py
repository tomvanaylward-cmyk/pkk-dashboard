"""
PKK EU-kontroll — logistic regression training pipeline
Confirmed format: latin-1, comma separator, quoted columns
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

GITHUB_API = "https://api.github.com/repos/vegvesen/periodisk-kjoretoy-kontroll/contents/"
RAW_BASE   = "https://raw.githubusercontent.com/vegvesen/periodisk-kjoretoy-kontroll/main/"

def list_zip_files():
    resp = requests.get(GITHUB_API, timeout=20)
    resp.raise_for_status()
    return sorted([f["name"] for f in resp.json() if f["name"].endswith(".zip")], reverse=True)

def download_zip(name):
    r = requests.get(RAW_BASE + name, timeout=120)
    r.raise_for_status()
    return r.content

def read_zip(content):
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for member in z.namelist():
            with z.open(member) as f:
                raw = f.read()
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=",",
                encoding="latin-1",
                on_bad_lines="skip",
                quotechar='"',
                skipinitialspace=True,
                usecols=["Kjøretøymerke", "Drivstofftype", "Kilometerstand",
                         "Første gang registrert", "PKK Kontrolltype",
                         "Kontrollorganets fylke",
                         "Tillatt totalvekt opp til og med 3500",
                         "Tillatt totalvekt 3501-7500",
                         "Tillatt totalvekt over 7500",
                         "Godkjent", "Trafikkfarlig feil"]
            )
            print(f"  {member}: {len(df):,} rows")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def load_all_data(max_files=6, sample_per_file=40000):
    zips = list_zip_files()[:max_files]
    print(f"Loading {len(zips)} files, ~{sample_per_file:,} rows each...")
    frames = []
    for name in zips:
        print(f"  {name}...")
        try:
            content = download_zip(name)
            df = read_zip(content)
            if df is not None:
                if len(df) > sample_per_file:
                    df = df.sample(n=sample_per_file, random_state=42)
                frames.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")
    if not frames:
        raise RuntimeError("No data loaded")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    return combined

def make_weight_col(df):
    """Combine the three weight boolean columns into one category."""
    w = pd.Series("Lette", index=df.index)
    if "Tillatt totalvekt 3501-7500" in df.columns:
        w[df["Tillatt totalvekt 3501-7500"].fillna(0).astype(str).str.strip() == "1"] = "Mellomtunge"
    if "Tillatt totalvekt over 7500" in df.columns:
        w[df["Tillatt totalvekt over 7500"].fillna(0).astype(str).str.strip() == "1"] = "Tunge"
    return w

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)

    out["brand"] = df["Kjøretøymerke"].astype(str).str.strip().str.upper().str[:30]

    fl = df["Drivstofftype"].astype(str).str.lower()
    out["fuel"] = "Other"
    out.loc[fl.str.contains("elektr", na=False), "fuel"] = "EV"
    out.loc[fl.str.contains("hybrid", na=False), "fuel"] = "Hybrid"
    out.loc[fl.str.contains("diesel", na=False), "fuel"] = "Diesel"
    out.loc[fl.str.contains("bensin|gasolin|petrol", na=False), "fuel"] = "Bensin"

    out["km"] = pd.to_numeric(df["Kilometerstand"], errors="coerce").clip(0, 500_000)

    reg = pd.to_numeric(df["Første gang registrert"], errors="coerce")
    age = (now_year - reg).where(lambda x: (x >= 0) & (x <= 50))
    out["age"] = age.clip(0, 30)

    ct = df["PKK Kontrolltype"].astype(str).str.strip().str.upper()
    out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")

    out["fylke"] = df["Kontrollorganets fylke"].astype(str).str.strip().str[:40]

    out["weight"] = make_weight_col(df)

    raw = df["Godkjent"].astype(str).str.strip().str.upper()
    approved = pd.Series(np.nan, index=df.index)
    approved[raw.isin(["1","JA","YES","TRUE","GODKJENT"])] = 1
    approved[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])] = 0
    num = pd.to_numeric(df["Godkjent"], errors="coerce")
    mask = approved.isna() & num.isin([0.0, 1.0])
    approved[mask] = num[mask]
    out["approved"] = approved

    before = len(out)
    out = out.dropna(subset=["km", "age", "approved"]).copy()
    out["approved"] = out["approved"].astype(int)
    out["brand"]   = out["brand"].fillna("UNKNOWN")
    out["fylke"]   = out["fylke"].fillna("UNKNOWN")
    print(f"Rows after cleaning: {len(out):,} (dropped {before-len(out):,})")
    print(f"Approved: {out['approved'].value_counts().to_dict()}")
    print(f"Fuel: {out['fuel'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]

    # Keep only top 50 brands to limit one-hot width
    top_brands = feat_df["brand"].value_counts().head(50).index
    X["brand"] = X["brand"].where(X["brand"].isin(top_brands), other="OTHER")

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
         ["brand", "fuel", "ctrl_type", "fylke", "weight"]),
        ("num", StandardScaler(), ["km", "age"]),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                   class_weight="balanced", solver="saga"))
    ])
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, model.predict(X)))
    print(f"AUC-ROC: {auc:.4f}")
    return model, auc, X

def extract_coefficients(model, feat_df, auc, X_train):
    clf   = model.named_steps["clf"]
    pre   = model.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_[0].tolist()

    def group(prefix):
        return {n[len(prefix):]: round(float(c), 4)
                for n, c in zip(names, coefs) if n.startswith(prefix)}

    top_brands = set(feat_df["brand"].value_counts().head(50).index)
    sc = pre.named_transformers_["num"]
    num_map = {n: c for n, c in zip(names, coefs) if n.startswith("num__")}

    return {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(float(auc), 4),
            "pass_rate":  round(float(feat_df["approved"].mean()), 4),
        },
        "intercept": round(float(clf.intercept_[0]), 4),
        "brand":     group("cat__brand_"),
        "fuel":      group("cat__fuel_"),
        "ctrl_type": group("cat__ctrl_type_"),
        "fylke":     group("cat__fylke_"),
        "weight":    group("cat__weight_"),
        "numeric": {
            "km_scaled":  round(float(num_map.get("num__km",  0)), 6),
            "age_scaled": round(float(num_map.get("num__age", 0)), 6),
        },
        "scaler": {
            "km_mean":  round(float(sc.mean_[0]),  2),
            "km_std":   round(float(sc.scale_[0]), 2),
            "age_mean": round(float(sc.mean_[1]),  2),
            "age_std":  round(float(sc.scale_[1]), 2),
        },
    }

def main():
    os.makedirs("docs", exist_ok=True)
    print("=== PKK model training pipeline ===")
    raw     = load_all_data(max_files=6, sample_per_file=40000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")
    model, auc, X_train = train_model(feat_df)
    coefs = extract_coefficients(model, feat_df, auc, X_train)
    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(coefs["meta"], indent=2))

if __name__ == "__main__":
    main()
