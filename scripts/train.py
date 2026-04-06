"""
PKK EU-kontroll — XGBoost training pipeline
- Defect chapters excluded from features (post-inspection data = data leakage)
- Fuel: BEV, Hybrid, Diesel, Petrol, Other
- No fylke (national model)
- Bootstrap confidence intervals
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

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

def load_all_data(max_files=6, sample_per_file=50000):
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
    out["age"] = age.clip(0, 30)

    ct = df["PKK Kontrolltype"].astype(str).str.strip().str.upper()
    out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")
    out["weight"]    = make_weight_col(df)

    # approved
    raw = df["Godkjent"].astype(str).str.strip().str.upper()
    approved = pd.Series(np.nan, index=df.index)
    approved[raw.isin(["1","JA","YES","TRUE","GODKJENT"])] = 1
    approved[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])] = 0
    num = pd.to_numeric(df["Godkjent"], errors="coerce")
    mask = approved.isna() & num.isin([0.0, 1.0])
    approved[mask] = num[mask]
    out["approved"] = approved

    before = len(out)
    out = out.dropna(subset=["km","age","approved"]).copy()
    out["approved"] = out["approved"].astype(int)
    out["brand"]    = out["brand"].fillna("UNKNOWN")

    top_brands = out["brand"].value_counts().head(50).index
    out["brand"] = out["brand"].where(out["brand"].isin(top_brands), "OTHER")

    print(f"Rows after cleaning: {len(out):,} (dropped {before-len(out):,})")
    print(f"Approved: {out['approved'].value_counts().to_dict()}")
    print(f"Fuel: {out['fuel'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def build_model(X, y):
    from xgboost import XGBClassifier
    cat_cols = ["brand","fuel","ctrl_type","weight"]
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", enable_categorical=True,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model, X, cat_cols

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    model, X_enc, cat_cols = build_model(X, y)
    y_prob = model.predict_proba(X_enc)[:,1]
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, (y_prob>=0.5).astype(int)))
    print(f"Train AUC: {auc:.4f}")

    # cross-validated AUC for honest estimate
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    cv_scores = []
    kf_idx = np.array_split(np.random.permutation(len(feat_df)), 5)
    for fold_idx in kf_idx:
        mask = np.zeros(len(feat_df), dtype=bool)
        mask[fold_idx] = True
        X_tr = X[~mask].copy(); y_tr = y[~mask]
        X_val = X[mask].copy(); y_val = y[mask]
        m, X_tr_enc, _ = build_model(X_tr, y_tr)
        for col in ["brand","fuel","ctrl_type","weight"]:
            X_val[col] = pd.Categorical(X_val[col], categories=X_tr_enc[col].cat.categories)
        prob = m.predict_proba(X_val)[:,1]
        cv_scores.append(roc_auc_score(y_val, prob))
    cv_auc = np.mean(cv_scores)
    print(f"CV AUC (5-fold): {cv_auc:.4f}")
    return model, cv_auc, X_enc, cat_cols

def get_marginal_effects(model, feat_df, X_enc, cat_cols):
    """Marginal effect = mean prediction with category set vs baseline."""
    effects = {}
    for col in cat_cols:
        baseline = feat_df[col].mode()[0]
        categories = feat_df[col].unique().tolist()
        col_effects = {}
        for cat in categories:
            X_cat = X_enc.copy()
            X_bl  = X_enc.copy()
            X_cat[col] = pd.Categorical([cat]*len(X_cat), categories=X_enc[col].cat.categories)
            X_bl[col]  = pd.Categorical([baseline]*len(X_bl), categories=X_enc[col].cat.categories)
            diff = (model.predict_proba(X_cat)[:,1].mean() -
                    model.predict_proba(X_bl)[:,1].mean())
            col_effects[cat] = round(float(diff), 4)
        effects[col] = col_effects

    effects["numeric"] = {
        "km_corr":  round(float(feat_df["km"].corr(feat_df["approved"])), 4),
        "age_corr": round(float(feat_df["age"].corr(feat_df["approved"])), 4),
    }
    return effects

def bootstrap_ci(feat_df, cat_cols, n_boot=20):
    from xgboost import XGBClassifier
    print(f"Bootstrap CI ({n_boot} resamples)...")
    boot = {col: {cat: [] for cat in feat_df[col].unique()} for col in cat_cols}

    for i in range(n_boot):
        idx = np.random.choice(len(feat_df), size=len(feat_df), replace=True)
        df_b = feat_df.iloc[idx].copy()
        X_b = df_b.drop(columns=["approved"])
        y_b = df_b["approved"]
        m, X_enc_b, _ = build_model(X_b, y_b)

        for col in cat_cols:
            baseline = feat_df[col].mode()[0]
            for cat in feat_df[col].unique():
                X_cat = X_enc_b.copy()
                X_bl  = X_enc_b.copy()
                try:
                    X_cat[col] = pd.Categorical([cat]*len(X_cat), categories=X_enc_b[col].cat.categories)
                    X_bl[col]  = pd.Categorical([baseline]*len(X_bl), categories=X_enc_b[col].cat.categories)
                    diff = (m.predict_proba(X_cat)[:,1].mean() -
                            m.predict_proba(X_bl)[:,1].mean())
                    boot[col][cat].append(float(diff))
                except Exception:
                    pass
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_boot}")

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
    print("=== PKK XGBoost training pipeline ===")

    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])

    raw     = load_all_data(max_files=6, sample_per_file=50000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")

    model, auc, X_enc, cat_cols = train_model(feat_df)
    effects = get_marginal_effects(model, feat_df, X_enc, cat_cols)
    ci      = bootstrap_ci(feat_df, cat_cols, n_boot=20)

    # merge CI into effects
    for col in cat_cols:
        merged = {}
        for cat in effects.get(col, {}):
            if col in ci and cat in ci[col]:
                merged[cat] = ci[col][cat]
            else:
                v = effects[col][cat]
                merged[cat] = {"coef": v, "lo": v, "hi": v}
        effects[col] = merged

    # defect chapter importances (for display only — not used in prediction)
    feat_names = X_enc.columns.tolist()
    imp = dict(zip(feat_names, model.feature_importances_))
    defect_imp = {k: round(float(v), 4) for k, v in imp.items() if "kap" in k}

    output = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(float(auc), 4),
            "pass_rate":  round(float(feat_df["approved"].mean()), 4),
            "model":      "XGBoost (5-fold CV AUC)",
        },
        "brand":           effects.get("brand", {}),
        "fuel":            effects.get("fuel", {}),
        "ctrl_type":       effects.get("ctrl_type", {}),
        "weight":          effects.get("weight", {}),
        "numeric":         effects.get("numeric", {}),
        "defect_chapters": defect_imp,
        "pass_rate":       round(float(feat_df["approved"].mean()), 4),
    }

    with open("docs/coefficients.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(output["meta"], indent=2))

if __name__ == "__main__":
    main()
