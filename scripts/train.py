"""
PKK EU-kontroll — XGBoost training pipeline (fast version)
- No data leakage: defect chapters excluded from features
- Fuel: BEV, Hybrid, Diesel, Petrol, Other
- 5 bootstrap resamples for CI (fast)
- 200 trees, 30k rows per file
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report

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
                sep=",", encoding="latin-1", on_bad_lines="skip",
                quotechar='"', skipinitialspace=True,
                usecols=["Kjøretøymerke","Drivstofftype","Kilometerstand",
                         "Første gang registrert","PKK Kontrolltype",
                         "Tillatt totalvekt opp til og med 3500",
                         "Tillatt totalvekt 3501-7500",
                         "Tillatt totalvekt over 7500",
                         "Godkjent"]
            )
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def load_all_data(max_files=6, sample_per_file=30000):
    zips = list_zip_files()[:max_files]
    print(f"Loading {len(zips)} files, {sample_per_file:,} rows each...")
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

def classify_fuel(s):
    s = str(s).lower().strip()
    if "elektr" in s and "hybrid" not in s:
        return "BEV"
    if "hybrid" in s:
        return "Hybrid"
    if "diesel" in s:
        return "Diesel"
    if "bensin" in s or "gasolin" in s or "petrol" in s:
        return "Petrol"
    return "Other"

def make_weight_col(df):
    w = pd.Series("Lette", index=df.index)
    if "Tillatt totalvekt 3501-7500" in df.columns:
        w[df["Tillatt totalvekt 3501-7500"].fillna(0).astype(str).str.strip()=="1"] = "Mellomtunge"
    if "Tillatt totalvekt over 7500" in df.columns:
        w[df["Tillatt totalvekt over 7500"].fillna(0).astype(str).str.strip()=="1"] = "Tunge"
    return w

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)
    out["brand"] = df["Kjøretøymerke"].astype(str).str.strip().str.upper().str[:30]
    out["fuel"]  = df["Drivstofftype"].apply(classify_fuel)
    out["km"]    = pd.to_numeric(df["Kilometerstand"], errors="coerce").clip(0, 500_000)
    reg = pd.to_numeric(df["Første gang registrert"], errors="coerce")
    age = (now_year - reg).where(lambda x: (x >= 0) & (x <= 50))
    out["age"]   = age.clip(0, 30)
    ct = df["PKK Kontrolltype"].astype(str).str.strip().str.upper()
    out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")
    out["weight"]    = make_weight_col(df)

    raw = df["Godkjent"].astype(str).str.strip().str.upper()
    approved = pd.Series(np.nan, index=df.index)
    approved[raw.isin(["1","JA","YES","TRUE","GODKJENT"])] = 1
    approved[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])] = 0
    num = pd.to_numeric(df["Godkjent"], errors="coerce")
    approved[approved.isna() & num.isin([0.0,1.0])] = num[approved.isna() & num.isin([0.0,1.0])]
    out["approved"] = approved

    out = out.dropna(subset=["km","age","approved"]).copy()
    out["approved"] = out["approved"].astype(int)
    out["brand"]    = out["brand"].fillna("UNKNOWN")
    top_brands = out["brand"].value_counts().head(50).index
    out["brand"] = out["brand"].where(out["brand"].isin(top_brands), "OTHER")

    print(f"Rows: {len(out):,} | Approved: {out['approved'].value_counts().to_dict()}")
    print(f"Fuel: {out['fuel'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def fit_xgb(X, y):
    from xgboost import XGBClassifier
    cat_cols = ["brand","fuel","ctrl_type","weight"]
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", enable_categorical=True,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model, X, cat_cols

def marginal_effect(model, X_enc, col, cat, baseline_cat):
    X_cat = X_enc.copy()
    X_bl  = X_enc.copy()
    X_cat[col] = pd.Categorical([cat]*len(X_cat), categories=X_enc[col].cat.categories)
    X_bl[col]  = pd.Categorical([baseline_cat]*len(X_bl), categories=X_enc[col].cat.categories)
    return float(model.predict_proba(X_cat)[:,1].mean() -
                 model.predict_proba(X_bl)[:,1].mean())

def get_effects(model, feat_df, X_enc, cat_cols):
    effects = {}
    for col in cat_cols:
        baseline = feat_df[col].mode()[0]
        effects[col] = {
            cat: round(marginal_effect(model, X_enc, col, cat, baseline), 4)
            for cat in feat_df[col].unique()
        }
    effects["numeric"] = {
        "km_corr":  round(float(feat_df["km"].corr(feat_df["approved"])), 4),
        "age_corr": round(float(feat_df["age"].corr(feat_df["approved"])), 4),
    }
    return effects

def bootstrap_ci(feat_df, cat_cols, n_boot=5):
    print(f"Bootstrap CI ({n_boot} resamples)...")
    boot = {col: {cat: [] for cat in feat_df[col].unique()} for col in cat_cols}
    for i in range(n_boot):
        idx = np.random.choice(len(feat_df), size=len(feat_df), replace=True)
        df_b = feat_df.iloc[idx].copy()
        X_b  = df_b.drop(columns=["approved"])
        y_b  = df_b["approved"]
        m, X_enc_b, _ = fit_xgb(X_b, y_b)
        for col in cat_cols:
            baseline = feat_df[col].mode()[0]
            for cat in feat_df[col].unique():
                try:
                    v = marginal_effect(m, X_enc_b, col, cat, baseline)
                    boot[col][cat].append(v)
                except Exception:
                    pass
        print(f"  Bootstrap {i+1}/{n_boot} done")

    ci = {}
    for col in cat_cols:
        ci[col] = {}
        for cat, vals in boot[col].items():
            if vals:
                arr = np.array(vals)
                ci[col][cat] = {
                    "coef": round(float(np.mean(arr)), 4),
                    "lo":   round(float(np.percentile(arr, 2.5)), 4),
                    "hi":   round(float(np.percentile(arr, 97.5)), 4),
                }
            else:
                ci[col][cat] = {"coef": 0.0, "lo": 0.0, "hi": 0.0}
    return ci

def main():
    os.makedirs("docs", exist_ok=True)
    print("=== PKK XGBoost pipeline (fast) ===")

    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])

    raw     = load_all_data(max_files=6, sample_per_file=30000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")

    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    model, X_enc, cat_cols = fit_xgb(X, y)

    # honest CV AUC
    from sklearn.metrics import roc_auc_score
    splits = np.array_split(np.random.permutation(len(feat_df)), 4)
    cv_aucs = []
    for fold in splits:
        mask = np.zeros(len(feat_df), dtype=bool)
        mask[fold] = True
        X_tr = X[~mask].copy(); y_tr = y[~mask]
        X_val = X[mask].copy();  y_val = y[mask]
        m, X_tr_enc, _ = fit_xgb(X_tr, y_tr)
        for col in cat_cols:
            X_val[col] = pd.Categorical(X_val[col], categories=X_tr_enc[col].cat.categories)
        cv_aucs.append(roc_auc_score(y_val, m.predict_proba(X_val)[:,1]))
    cv_auc = float(np.mean(cv_aucs))
    print(f"CV AUC (4-fold): {cv_auc:.4f}")

    effects = get_effects(model, feat_df, X_enc, cat_cols)
    ci      = bootstrap_ci(feat_df, cat_cols, n_boot=5)

    for col in cat_cols:
        for cat in effects.get(col, {}):
            if col in ci and cat in ci[col]:
                effects[col][cat] = ci[col][cat]
            else:
                v = effects[col][cat]
                effects[col][cat] = {"coef": v, "lo": v, "hi": v}

    output = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(cv_auc, 4),
            "pass_rate":  round(float(y.mean()), 4),
            "model":      "XGBoost (4-fold CV)",
        },
        "brand":     effects.get("brand", {}),
        "fuel":      effects.get("fuel", {}),
        "ctrl_type": effects.get("ctrl_type", {}),
        "weight":    effects.get("weight", {}),
        "numeric":   effects.get("numeric", {}),
        "pass_rate": round(float(y.mean()), 4),
    }

    with open("docs/coefficients.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(output["meta"], indent=2))

if __name__ == "__main__":
    main()
