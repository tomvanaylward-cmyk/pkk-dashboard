"""
PKK EU-kontroll — XGBoost training pipeline
- Removed fylke (national model)
- Fuel: BEV, PHEV, HEV, Diesel, Petrol, Other
- XGBoost for higher accuracy
- Bootstrap confidence intervals
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
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
                         "Ant 2-3er kap 0","Ant 2-3er kap 1","Ant 2-3er kap 2",
                         "Ant 2-3er kap 3","Ant 2-3er kap 4","Ant 2-3er kap 5",
                         "Ant 2-3er kap 6","Ant 2-3er kap 7","Ant 2-3er kap 8",
                         "Ant 2-3er kap 9","Ant 2-3er kap 10",
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
    """
    BEV  — battery electric
    PHEV — plug-in hybrid
    HEV  — non-plug hybrid
    Diesel, Petrol, Other
    """
    s = str(s).lower().strip()
    if "elektr" in s and "hybrid" not in s:
        return "BEV"
    if "plug" in s or "ladbar" in s or "phev" in s:
        return "PHEV"
    if "hybrid" in s:
        return "HEV"
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

DEFECT_CAPS = [f"Ant 2-3er kap {i}" for i in range(11)]

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)

    out["brand"] = df["Kjøretøymerke"].astype(str).str.strip().str.upper().str[:30]
    out["fuel"]  = df["Drivstofftype"].apply(classify_fuel)
    out["km"]    = pd.to_numeric(df["Kilometerstand"], errors="coerce").clip(0, 500_000)

    reg = pd.to_numeric(df["Første gang registrert"], errors="coerce")
    age = (now_year - reg).where(lambda x: (x>=0) & (x<=50))
    out["age"] = age.clip(0, 30)

    ct = df["PKK Kontrolltype"].astype(str).str.strip().str.upper()
    out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")
    out["weight"]    = make_weight_col(df)

    # defect chapter counts — powerful predictors
    for cap in DEFECT_CAPS:
        if cap in df.columns:
            out[cap] = pd.to_numeric(df[cap], errors="coerce").fillna(0).clip(0, 20)
        else:
            out[cap] = 0

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

    # keep only top 50 brands
    top_brands = out["brand"].value_counts().head(50).index
    out["brand"] = out["brand"].where(out["brand"].isin(top_brands), "OTHER")

    print(f"Rows after cleaning: {len(out):,} (dropped {before-len(out):,})")
    print(f"Approved: {out['approved'].value_counts().to_dict()}")
    print(f"Fuel: {out['fuel'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    from xgboost import XGBClassifier

    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]

    cat_cols = ["brand","fuel","ctrl_type","weight"]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, y_pred))
    print(f"AUC-ROC (train): {auc:.4f}")
    return model, auc, X, cat_cols

def get_feature_effects(model, feat_df, cat_cols):
    """
    Compute marginal effect of each category by averaging predictions
    with that category set vs the baseline (most common value).
    Returns {feature: {category: {coef, lo, hi}}} using prediction differences.
    """
    from xgboost import XGBClassifier

    X_base = feat_df.drop(columns=["approved"]).copy()
    for col in cat_cols:
        X_base[col] = X_base[col].astype("category")

    def sigmoid(x): return 1/(1+np.exp(-x))

    effects = {}

    for col in ["brand","fuel","ctrl_type","weight"]:
        categories = feat_df[col].unique().tolist()
        baseline = feat_df[col].mode()[0]
        col_effects = {}
        for cat in categories:
            X_tmp = X_base.copy()
            X_tmp[col] = pd.Categorical([cat]*len(X_tmp),
                                         categories=X_base[col].cat.categories)
            X_base_tmp = X_base.copy()
            X_base_tmp[col] = pd.Categorical([baseline]*len(X_base_tmp),
                                              categories=X_base[col].cat.categories)
            diff = model.predict_proba(X_tmp)[:,1].mean() - \
                   model.predict_proba(X_base_tmp)[:,1].mean()
            col_effects[cat] = round(float(diff), 4)
        effects[col] = col_effects

    # numeric: correlation direction
    effects["numeric"] = {
        "km_effect_dir":  "negative" if feat_df["km"].corr(feat_df["approved"]) < 0 else "positive",
        "age_effect_dir": "negative" if feat_df["age"].corr(feat_df["approved"]) < 0 else "positive",
        "km_corr":  round(float(feat_df["km"].corr(feat_df["approved"])), 4),
        "age_corr": round(float(feat_df["age"].corr(feat_df["approved"])), 4),
    }

    # defect chapter importances
    imp = model.feature_importances_
    feat_names = feat_df.drop(columns=["approved"]).columns.tolist()
    cap_imp = {}
    for name, score in zip(feat_names, imp):
        if "kap" in name:
            cap_imp[name] = round(float(score), 4)
    effects["defect_chapters"] = cap_imp

    return effects

def bootstrap_effects(feat_df, cat_cols, n_boot=20):
    """Bootstrap uncertainty on marginal effects."""
    from xgboost import XGBClassifier

    print(f"Running {n_boot} bootstrap resamples...")
    boot_results = {col: {cat: [] for cat in feat_df[col].unique()}
                    for col in ["brand","fuel","ctrl_type","weight"]}

    for i in range(n_boot):
        idx = np.random.choice(len(feat_df), size=len(feat_df), replace=True)
        df_b = feat_df.iloc[idx].copy()
        X_b = df_b.drop(columns=["approved"])
        y_b = df_b["approved"]
        for col in cat_cols:
            X_b[col] = X_b[col].astype("category")
        m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                          enable_categorical=True, tree_method="hist",
                          random_state=i, n_jobs=-1, eval_metric="auc")
        m.fit(X_b, y_b)

        for col in ["brand","fuel","ctrl_type","weight"]:
            baseline = feat_df[col].mode()[0]
            for cat in feat_df[col].unique():
                X_tmp = X_b.copy()
                X_tmp[col] = pd.Categorical([cat]*len(X_tmp), categories=X_b[col].cat.categories)
                X_bl = X_b.copy()
                X_bl[col] = pd.Categorical([baseline]*len(X_bl), categories=X_b[col].cat.categories)
                try:
                    diff = m.predict_proba(X_tmp)[:,1].mean() - m.predict_proba(X_bl)[:,1].mean()
                    boot_results[col][cat].append(float(diff))
                except Exception:
                    pass
        if (i+1) % 5 == 0:
            print(f"  Bootstrap {i+1}/{n_boot}")

    ci = {}
    for col in ["brand","fuel","ctrl_type","weight"]:
        ci[col] = {}
        for cat, vals in boot_results[col].items():
            if vals:
                arr = np.array(vals)
                ci[col][cat] = {
                    "coef": round(float(np.mean(arr)), 4),
                    "lo":   round(float(np.percentile(arr, 2.5)), 4),
                    "hi":   round(float(np.percentile(arr, 97.5)), 4),
                }
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

    model, auc, X_train, cat_cols = train_model(feat_df)
    effects = get_feature_effects(model, feat_df, cat_cols)
    ci      = bootstrap_effects(feat_df, cat_cols, n_boot=20)

    # merge CI into effects
    for col in ["brand","fuel","ctrl_type","weight"]:
        for cat in effects.get(col, {}):
            base_val = effects[col][cat]
            if col in ci and cat in ci[col]:
                effects[col][cat] = ci[col][cat]
            else:
                effects[col][cat] = {"coef": base_val, "lo": base_val, "hi": base_val}

    output = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(float(auc), 4),
            "pass_rate":  round(float(feat_df["approved"].mean()), 4),
            "model":      "XGBoost",
        },
        "brand":          effects.get("brand", {}),
        "fuel":           effects.get("fuel", {}),
        "ctrl_type":      effects.get("ctrl_type", {}),
        "weight":         effects.get("weight", {}),
        "numeric":        effects.get("numeric", {}),
        "defect_chapters":effects.get("defect_chapters", {}),
        "pass_rate":      round(float(feat_df["approved"].mean()), 4),
    }

    with open("docs/coefficients.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(output["meta"], indent=2))

if __name__ == "__main__":
    main()
