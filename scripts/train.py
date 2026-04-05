"""
PKK EU-kontroll — logistic regression training pipeline
Runs on GitHub Actions. Downloads latest zip files from vegvesen repo,
parses CSVs, fits a logistic regression, and writes docs/coefficients.json
"""

import os
import json
import zipfile
import io
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

GITHUB_API = "https://api.github.com/repos/vegvesen/periodisk-kjoretoy-kontroll/contents/"
RAW_BASE   = "https://raw.githubusercontent.com/vegvesen/periodisk-kjoretoy-kontroll/main/"

# ── column name normalisation ──────────────────────────────────────────────
COL_BRAND    = None   # resolved at runtime
COL_MODEL    = None
COL_FUEL     = None
COL_KM       = None
COL_AGE_REG  = None   # "Første gang registrert"
COL_AGE_NO   = None   # "Første gang registrert i Norge"
COL_TYPE     = None   # P / E
COL_APPROVED = None   # godkjent flag
COL_UNSAFE   = None   # trafikkfarlig feil
COL_FYLKE    = None
COL_DATE     = None   # control month/year
COL_WEIGHT   = None

CANDIDATE_MAP = {
    "brand":    ["kjøretøymerke", "merke", "brand"],
    "model":    ["kjøretøy modell", "modell", "model"],
    "fuel":     ["drivstofftype", "fuel"],
    "km":       ["kilometerstand", "km", "odometer"],
    "reg_first":["første gang registrert", "first registered"],
    "reg_no":   ["første gang registrert i norge", "first registered norway"],
    "ctrl_type":["pkk kontrolltype", "kontrolltype", "control type"],
    "approved": ["om kjøretøyet ble godkjent", "godkjent", "approved"],
    "unsafe":   ["om det ble avdekket trafikkfarlig feil", "trafikkfarlig", "unsafe"],
    "fylke":    ["fylke der kjøretøyet er kontrollert", "fylke", "county"],
    "date":     ["måned og år da kjøretøyet ble kontrollert", "kontrolldato", "date"],
    "weight":   ["tillatt totalvekt", "weight class", "vekt"],
}

def resolve_cols(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for key, candidates in CANDIDATE_MAP.items():
        for cand in candidates:
            if cand in cols_lower:
                mapping[key] = cols_lower[cand]
                break
    return mapping


def list_zip_files():
    resp = requests.get(GITHUB_API, timeout=20)
    resp.raise_for_status()
    files = resp.json()
    zips = [f["name"] for f in files if f["name"].endswith(".zip")]
    # sort newest first
    zips.sort(reverse=True)
    return zips


def download_zip(name):
    url = RAW_BASE + name
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def read_zip(content):
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for member in z.namelist():
            if member.lower().endswith(".csv"):
                with z.open(member) as f:
                    try:
                        df = pd.read_csv(f, sep=";", encoding="utf-8", low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, sep=";", encoding="latin-1", low_memory=False)
                    frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all_data(max_files=8):
    """Download up to max_files zip files (newest first) and concatenate."""
    zips = list_zip_files()[:max_files]
    print(f"Found {len(zips)} zip files, downloading {len(zips)}...")
    frames = []
    for name in zips:
        print(f"  Downloading {name}...")
        try:
            content = download_zip(name)
            df = read_zip(content)
            if df is not None:
                frames.append(df)
                print(f"    -> {len(df):,} rows")
        except Exception as e:
            print(f"    ERROR: {e}")
    if not frames:
        raise RuntimeError("No data could be loaded")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    return combined


def engineer_features(df, col):
    """Build model-ready feature dataframe."""
    rows = []
    now_year = datetime.now().year

    for _, r in df.iterrows():
        brand = str(r.get(col.get("brand",""), "")).strip().upper()
        model = str(r.get(col.get("model",""), "")).strip().upper()
        fuel  = str(r.get(col.get("fuel",""), "")).strip()
        km_raw = r.get(col.get("km",""), np.nan)
        reg_yr = r.get(col.get("reg_first",""), np.nan)
        ctrl_type = str(r.get(col.get("ctrl_type",""), "P")).strip().upper()
        approved_raw = r.get(col.get("approved",""), None)
        fylke = str(r.get(col.get("fylke",""), "")).strip()
        weight = str(r.get(col.get("weight",""), "Lette")).strip()

        # parse approved flag
        if approved_raw is None:
            continue
        approved_str = str(approved_raw).strip().upper()
        if approved_str in ("1","JA","YES","TRUE","GODKJENT"):
            approved = 1
        elif approved_str in ("0","NEI","NO","FALSE","IKKE GODKJENT"):
            approved = 0
        else:
            continue

        # km
        try:
            km = float(str(km_raw).replace(",",".").replace(" ",""))
        except:
            km = np.nan

        # age
        try:
            reg = int(float(str(reg_yr)))
            age = now_year - reg
            if age < 0 or age > 50:
                age = np.nan
        except:
            age = np.nan

        if pd.isna(km) or pd.isna(age):
            continue

        # normalise fuel
        fuel_norm = "Other"
        fl = fuel.lower()
        if "elektr" in fl:
            fuel_norm = "EV"
        elif "hybrid" in fl:
            fuel_norm = "Hybrid"
        elif "diesel" in fl:
            fuel_norm = "Diesel"
        elif "bensin" in fl or "gasolin" in fl or "petrol" in fl:
            fuel_norm = "Bensin"

        rows.append({
            "brand": brand[:30] if brand else "UNKNOWN",
            "fuel":  fuel_norm,
            "km":    min(km, 500_000),
            "age":   min(age, 30),
            "ctrl_type": "E" if ctrl_type.startswith("E") else "P",
            "fylke": fylke[:40] if fylke else "UNKNOWN",
            "weight": weight[:20],
            "approved": approved,
        })

    return pd.DataFrame(rows)


def train(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]

    cat_features = ["brand","fuel","ctrl_type","fylke","weight"]
    num_features = ["km","age"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("num", StandardScaler(), num_features),
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")),
    ])

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    print(classification_report(y, y_pred))
    auc = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    return model, X, auc


def extract_coefficients(model, feat_df, auc):
    """Pull brand, fuel, fylke effects and numeric slopes for the frontend."""
    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    feature_names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    def get_coefs(prefix):
        out = {}
        for name, coef in zip(feature_names, coefs):
            if name.startswith(prefix):
                label = name[len(prefix):]
                out[label] = round(float(coef), 4)
        return out

    brand_coefs = get_coefs("cat__brand_")
    fuel_coefs  = get_coefs("cat__fuel_")
    type_coefs  = get_coefs("cat__ctrl_type_")
    fylke_coefs = get_coefs("cat__fylke_")
    weight_coefs= get_coefs("cat__weight_")

    # numeric coefficients (StandardScaler-scaled — store both scaled and approx per-unit)
    num_idx = {n: i for i, n in enumerate(feature_names) if n.startswith("num__")}
    km_coef  = float(coefs[num_idx.get("num__km", 0)]) if "num__km" in num_idx else 0
    age_coef = float(coefs[num_idx.get("num__age", 0)]) if "num__age" in num_idx else 0

    # compute summary stats for the frontend
    brand_counts = feat_df["brand"].value_counts()
    top_brands = brand_counts[brand_counts >= 50].index.tolist()

    return {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(feat_df),
            "auc_roc": round(auc, 4),
            "pass_rate": round(float(feat_df["approved"].mean()), 4),
        },
        "intercept": round(intercept, 4),
        "brand": {k: v for k, v in brand_coefs.items() if k in top_brands},
        "fuel":  fuel_coefs,
        "ctrl_type": type_coefs,
        "fylke": fylke_coefs,
        "weight": weight_coefs,
        "numeric": {
            "km_scaled":  round(km_coef, 6),
            "age_scaled": round(age_coef, 6),
        },
        "scaler": {
            "km_mean":  round(float(pre.named_transformers_["num"].mean_[0]), 2),
            "km_std":   round(float(pre.named_transformers_["num"].scale_[0]), 2),
            "age_mean": round(float(pre.named_transformers_["num"].mean_[1]), 2),
            "age_std":  round(float(pre.named_transformers_["num"].scale_[1]), 2),
        }
    }


def main():
    os.makedirs("docs", exist_ok=True)
    print("=== PKK model training pipeline ===")
    raw = load_all_data(max_files=8)
    col = resolve_cols(raw)
    print("Column mapping:", col)

    feat_df = engineer_features(raw, col)
    print(f"Feature rows after cleaning: {len(feat_df):,}")

    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few usable rows ({len(feat_df)}) — check column mapping")

    model, X, auc = train(feat_df)
    coefs = extract_coefficients(model, feat_df, auc)

    out_path = "docs/coefficients.json"
    with open(out_path, "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print(f"Saved coefficients to {out_path}")
    print(json.dumps(coefs["meta"], indent=2))


if __name__ == "__main__":
    main()
