"""
PKK EU-kontroll — logistic regression pipeline (reliable version)
- Fuel: BEV, Hybrid, Diesel, Petrol, Other
- No fylke (national model)
- Bootstrap CI via logistic regression standard errors
- Fast: ~4 minutes on GitHub Actions
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

def load_all_data(max_files=6, sample_per_file=40000):
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
    mask = approved.isna() & num.isin([0.0,1.0])
    approved[mask] = num[mask]
    out["approved"] = approved

    out = out.dropna(subset=["km","age","approved"]).copy()
    out["approved"] = out["approved"].astype(int)
    out["brand"]    = out["brand"].fillna("UNKNOWN")
    top_brands = out["brand"].value_counts().head(50).index
    out["brand"] = out["brand"].where(out["brand"].isin(top_brands), "OTHER")

    print(f"Rows: {len(out):,} | Pass rate: {out['approved'].mean():.3f}")
    print(f"Fuel: {out['fuel'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]

    cat_features = ["brand","fuel","ctrl_type","weight"]
    num_features = ["km","age"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500, C=1.0,
                                   class_weight="balanced", solver="saga"))
    ])
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, (y_prob>=0.5).astype(int)))
    print(f"Train AUC: {auc:.4f}")
    return model, auc, X

def bootstrap_ci(feat_df, n_boot=8):
    """Bootstrap standard errors on logistic regression coefficients."""
    print(f"Bootstrap CI ({n_boot} resamples)...")
    cat_features = ["brand","fuel","ctrl_type","weight"]
    num_features = ["km","age"]

    pre0 = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    m0 = Pipeline([("pre", pre0),
                   ("clf", LogisticRegression(max_iter=300, C=1.0,
                                              class_weight="balanced", solver="saga"))])
    X0 = feat_df.drop(columns=["approved"])
    y0 = feat_df["approved"]
    m0.fit(X0, y0)
    names = list(m0.named_steps["pre"].get_feature_names_out())
    n_feat = len(names)

    boot_coefs = np.zeros((n_boot, n_feat))
    for i in range(n_boot):
        idx = np.random.choice(len(feat_df), size=len(feat_df), replace=True)
        df_b = feat_df.iloc[idx]
        X_b = df_b.drop(columns=["approved"])
        y_b = df_b["approved"]
        pre_b = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ])
        m_b = Pipeline([("pre", pre_b),
                        ("clf", LogisticRegression(max_iter=300, C=1.0,
                                                   class_weight="balanced", solver="saga"))])
        m_b.fit(X_b, y_b)
        names_b = list(m_b.named_steps["pre"].get_feature_names_out())
        coefs_b = m_b.named_steps["clf"].coef_[0]
        for j, nm in enumerate(names):
            if nm in names_b:
                boot_coefs[i, j] = coefs_b[names_b.index(nm)]
        print(f"  Bootstrap {i+1}/{n_boot}")

    stds = boot_coefs.std(axis=0)
    return dict(zip(names, stds.tolist())), names

def extract_coefficients(model, feat_df, auc, stds, names):
    clf  = model.named_steps["clf"]
    pre  = model.named_steps["pre"]
    coefs = clf.coef_[0]
    z = 1.96

    def group_with_ci(prefix):
        out = {}
        for n, c in zip(names, coefs):
            if n.startswith(prefix):
                label = n[len(prefix):]
                se = stds.get(n, 0)
                out[label] = {
                    "coef": round(float(c), 4),
                    "lo":   round(float(c) - z*se, 4),
                    "hi":   round(float(c) + z*se, 4),
                }
        return out

    sc = pre.named_transformers_["num"]
    num_map = {n: c for n, c in zip(names, coefs) if n.startswith("num__")}

    return {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(float(auc), 4),
            "pass_rate":  round(float(feat_df["approved"].mean()), 4),
            "model":      "Logistic Regression",
        },
        "intercept": round(float(clf.intercept_[0]), 4),
        "brand":     group_with_ci("cat__brand_"),
        "fuel":      group_with_ci("cat__fuel_"),
        "ctrl_type": group_with_ci("cat__ctrl_type_"),
        "weight":    group_with_ci("cat__weight_"),
        "numeric": {
            "km_scaled":  round(float(num_map.get("num__km",  0)), 6),
            "age_scaled": round(float(num_map.get("num__age", 0)), 6),
            "km_corr":    round(float(feat_df["km"].corr(feat_df["approved"])), 4),
            "age_corr":   round(float(feat_df["age"].corr(feat_df["approved"])), 4),
        },
        "scaler": {
            "km_mean":  round(float(sc.mean_[0]),  2),
            "km_std":   round(float(sc.scale_[0]), 2),
            "age_mean": round(float(sc.mean_[1]),  2),
            "age_std":  round(float(sc.scale_[1]), 2),
        },
        "pass_rate": round(float(feat_df["approved"].mean()), 4),
    }

def main():
    os.makedirs("docs", exist_ok=True)
    print("=== PKK Logistic Regression pipeline ===")
    raw     = load_all_data(max_files=6, sample_per_file=40000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")
    model, auc, X = train_model(feat_df)
    stds, names   = bootstrap_ci(feat_df, n_boot=8)
    coefs         = extract_coefficients(model, feat_df, auc, stds, names)

    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(coefs["meta"], indent=2))

if __name__ == "__main__":
    main()
