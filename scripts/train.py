"""
PKK EU-kontroll — logistic regression pipeline v4
- 80k rows per file (~480k total)
- inspection_number feature (which inspection is this for the vehicle)
- Fuel: BEV, Hybrid, Diesel, Petrol, Other
- No fylke (national model)
- 5-fold StratifiedKFold CV (replaces bootstrap CI)
"""

import os, json, zipfile, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

GITHUB_API = "https://api.github.com/repos/vegvesen/periodisk-kjoretoy-kontroll/contents/"
RAW_BASE   = "https://raw.githubusercontent.com/vegvesen/periodisk-kjoretoy-kontroll/main/"

# Use GITHUB_TOKEN to avoid rate limiting (60/hr unauthenticated → 5000/hr authenticated)
_GH_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
_GH_HEADERS = {"Authorization": f"Bearer {_GH_TOKEN}"} if _GH_TOKEN else {}

# Feature lists — shared by train_model(), cross_validate_pipeline(), failure_fingerprint()
# Updated in Task 2 to include km_bucket and km_per_year
_CAT_FEATURES = ["brand", "fuel", "ctrl_type", "weight"]
_NUM_FEATURES = ["km", "age", "insp_num"]

def list_zip_files():
    resp = requests.get(GITHUB_API, headers=_GH_HEADERS, timeout=20)
    resp.raise_for_status()
    return sorted([f["name"] for f in resp.json() if f["name"].endswith(".zip")], reverse=True)

def download_zip(name):
    r = requests.get(RAW_BASE + name, headers=_GH_HEADERS, timeout=120)
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
                         "Første gang registrert","Første gang registrert i Norge",
                         "PKK Kontrolltype","PKK Intervall",
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

def load_all_data(max_files=6, sample_per_file=80000):
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

def compute_inspection_number(df, now_year):
    reg_no  = pd.to_numeric(df.get("Første gang registrert i Norge"), errors="coerce")
    reg_wld = pd.to_numeric(df.get("Første gang registrert"), errors="coerce")
    reg     = reg_no.fillna(reg_wld)
    age = (now_year - reg).clip(0, 50)
    interval = pd.to_numeric(df.get("PKK Intervall"), errors="coerce").fillna(2).clip(1, 4)
    insp_num = 1 + ((age - 4) / interval).clip(lower=0).apply(np.floor)
    return insp_num.clip(1, 15).fillna(1)

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)
    out["brand"] = df["Kjøretøymerke"].astype(str).str.strip().str.upper().str[:30]
    out["fuel"]  = df["Drivstofftype"].apply(classify_fuel)
    out["km"]    = pd.to_numeric(df["Kilometerstand"], errors="coerce").clip(0, 500_000)
    reg_no  = pd.to_numeric(df.get("Første gang registrert i Norge"), errors="coerce")
    reg_wld = pd.to_numeric(df.get("Første gang registrert"), errors="coerce")
    reg     = reg_no.fillna(reg_wld)
    age     = (now_year - reg).where(lambda x: (x >= 0) & (x <= 50))
    out["age"] = age.clip(0, 30)
    out["insp_num"] = compute_inspection_number(df, now_year)
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
    cat_features = _CAT_FEATURES
    num_features = _NUM_FEATURES
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

def cross_validate_pipeline(feat_df):
    """5-fold stratified CV. Returns (mean_auc, std_auc, fold_scores)."""
    cat_features = _CAT_FEATURES
    num_features = _NUM_FEATURES
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    pipeline = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500, C=1.0,
                                   class_weight="balanced", solver="saga")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = int(os.environ.get("CV_N_JOBS", "-1"))
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
    print(f"5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {[round(s, 4) for s in scores.tolist()]}")
    return float(scores.mean()), float(scores.std()), scores.tolist()

def extract_coefficients(model, feat_df, cv_mean, cv_std, cv_scores):
    clf   = model.named_steps["clf"]
    pre   = model.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_[0]

    def group_coefs(prefix):
        return {n[len(prefix):]: round(float(c), 4)
                for n, c in zip(names, coefs) if n.startswith(prefix)}

    sc = pre.named_transformers_["num"]
    insp_pass = feat_df.groupby("insp_num")["approved"].agg(["mean", "count"])
    insp_rates = {
        int(k): {"pass_rate": round(float(v["mean"]), 4), "n": int(v["count"])}
        for k, v in insp_pass.iterrows()
        if v["count"] >= 100
    }
    brand_fuel = {}
    for brand in feat_df["brand"].unique():
        sub = feat_df[feat_df["brand"] == brand]
        dominant = sub["fuel"].value_counts()
        if len(dominant) > 0:
            top = dominant.index[0]
            brand_fuel[brand] = {
                "dominant": top,
                "exclusive": bool(dominant.iloc[0] / len(sub) > 0.95),
            }
    return {
        "meta": {
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "n_samples":    int(len(feat_df)),
            "auc_mean":     round(cv_mean, 4),
            "auc_std":      round(cv_std,  4),
            "auc_folds":    [round(s, 4) for s in cv_scores],
            "pass_rate":    round(float(feat_df["approved"].mean()), 4),
            "model":        "LogReg v4 — 5-fold CV",
            "calibrated":   False,
        },
        "intercept":       round(float(clf.intercept_[0]), 4),
        "brand":           group_coefs("cat__brand_"),
        "fuel":            group_coefs("cat__fuel_"),
        "ctrl_type":       group_coefs("cat__ctrl_type_"),
        "weight":          group_coefs("cat__weight_"),
        "numeric": {
            "km_scaled":    round(float(dict(zip(names, coefs)).get("num__km",       0)), 6),
            "age_scaled":   round(float(dict(zip(names, coefs)).get("num__age",      0)), 6),
            "insp_scaled":  round(float(dict(zip(names, coefs)).get("num__insp_num", 0)), 6),
        },
        "scaler": {
            "km_mean":    round(float(sc.mean_[0]),  2),
            "km_std":     round(float(sc.scale_[0]), 2),
            "age_mean":   round(float(sc.mean_[1]),  2),
            "age_std":    round(float(sc.scale_[1]), 2),
            "insp_mean":  round(float(sc.mean_[2]),  2),
            "insp_std":   round(float(sc.scale_[2]), 2),
        },
        "insp_pass_rates": insp_rates,
        "brand_fuel":      brand_fuel,
    }

def failure_fingerprint(feat_df, raw_df):
    CAP_COLS = {
        "Ant 2-3er kap 0":  "Identification & documents",
        "Ant 2-3er kap 1":  "Brakes",
        "Ant 2-3er kap 2":  "Steering",
        "Ant 2-3er kap 3":  "Visibility",
        "Ant 2-3er kap 4":  "Lights & electrical",
        "Ant 2-3er kap 5":  "Axles, wheels & tyres",
        "Ant 2-3er kap 6":  "Chassis & body",
        "Ant 2-3er kap 7":  "Other equipment",
        "Ant 2-3er kap 8":  "Noise & emissions",
        "Ant 2-3er kap 9":  "Checks during drive",
        "Ant 2-3er kap 10": "Environment",
    }
    cat_features = _CAT_FEATURES
    num_features = _NUM_FEATURES
    raw_aligned = raw_df.iloc[:len(feat_df)].copy()
    raw_aligned.index = feat_df.index
    fingerprint = {}
    for col, name in CAP_COLS.items():
        if col not in raw_aligned.columns:
            print(f"  Skipping {name} — column not found")
            continue
        y_chap = (pd.to_numeric(raw_aligned[col], errors="coerce")
                    .fillna(0).reindex(feat_df.index).fillna(0) > 0).astype(int)
        if y_chap.sum() < 50:
            print(f"  Skipping {name} — too few positives ({y_chap.sum()})")
            continue
        baseline = round(float(y_chap.mean()), 4)
        if baseline < 0.02:
            print(f"  Skipping {name} — baseline too low ({baseline:.3f})")
            continue
        print(f"  Training {name}: baseline={baseline:.3f}, positives={y_chap.sum()}")
        X = feat_df.drop(columns=["approved"])
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ])
        m = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, C=1.0, solver="saga"))
        ])
        m.fit(X, y_chap)
        feat_names = list(m.named_steps["pre"].get_feature_names_out())
        coefs      = m.named_steps["clf"].coef_[0]
        coef_map   = dict(zip(feat_names, coefs))
        sc         = m.named_steps["pre"].named_transformers_["num"]
        def g(prefix):
            return {k[len(prefix):]: round(float(v), 4)
                    for k, v in coef_map.items() if k.startswith(prefix)}
        fingerprint[name] = {
            "baseline":  baseline,
            "intercept": round(float(m.named_steps["clf"].intercept_[0]), 4),
            "brand":     g("cat__brand_"),
            "fuel":      g("cat__fuel_"),
            "ctrl_type": g("cat__ctrl_type_"),
            "numeric": {
                "km_scaled":   round(float(coef_map.get("num__km",      0)), 6),
                "age_scaled":  round(float(coef_map.get("num__age",     0)), 6),
                "insp_scaled": round(float(coef_map.get("num__insp_num",0)), 6),
            },
            "scaler": {
                "km_mean":   round(float(sc.mean_[0]),  2),
                "km_std":    round(float(sc.scale_[0]), 2),
                "age_mean":  round(float(sc.mean_[1]),  2),
                "age_std":   round(float(sc.scale_[1]), 2),
                "insp_mean": round(float(sc.mean_[2]),  2),
                "insp_std":  round(float(sc.scale_[2]), 2),
            },
        }
    return fingerprint


def defect_analysis(raw_df):
    CAP_NAMES = {
        "Ant 2-3er kap 0":  "Identification & documents",
        "Ant 2-3er kap 1":  "Brakes",
        "Ant 2-3er kap 2":  "Steering",
        "Ant 2-3er kap 3":  "Visibility",
        "Ant 2-3er kap 4":  "Lights & electrical",
        "Ant 2-3er kap 5":  "Axles, wheels & tyres",
        "Ant 2-3er kap 6":  "Chassis & body",
        "Ant 2-3er kap 7":  "Other equipment",
        "Ant 2-3er kap 8":  "Noise & emissions",
        "Ant 2-3er kap 9":  "Checks during drive",
        "Ant 2-3er kap 10": "Environment",
    }
    raw = raw_df["Godkjent"].astype(str).str.strip().str.upper()
    failed = raw_df[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])].copy()
    total_inspections = len(raw_df)
    chapter_stats = []
    for col, name in CAP_NAMES.items():
        if col not in raw_df.columns:
            continue
        has_fault_all  = (pd.to_numeric(raw_df[col], errors="coerce").fillna(0) > 0)
        has_fault_fail = (pd.to_numeric(failed[col],  errors="coerce").fillna(0) > 0)
        chapter_stats.append({
            "chapter":     col,
            "name":        name,
            "rate_all":    round(float(has_fault_all.mean()),  4),
            "rate_failed": round(float(has_fault_fail.mean()), 4),
        })
    chapter_stats.sort(key=lambda x: x["rate_all"], reverse=True)
    by_fuel = {}
    if "Drivstofftype" in raw_df.columns:
        raw_df2 = raw_df.copy()
        raw_df2["fuel_norm"] = raw_df2["Drivstofftype"].apply(classify_fuel)
        for fuel in ["BEV","Hybrid","Diesel","Petrol"]:
            sub = raw_df2[raw_df2["fuel_norm"]==fuel]
            if len(sub) < 100:
                continue
            sub_raw  = sub["Godkjent"].astype(str).str.strip().str.upper()
            sub_fail = sub[sub_raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])]
            top = {}
            for col, name in CAP_NAMES.items():
                if col in sub_fail.columns:
                    rate = float((pd.to_numeric(sub_fail[col], errors="coerce").fillna(0)>0).mean())
                    top[name] = round(rate, 4)
            by_fuel[fuel] = dict(sorted(top.items(), key=lambda x: x[1], reverse=True))
    return {
        "total_inspections": int(total_inspections),
        "total_failed":      int(len(failed)),
        "fail_rate":         round(len(failed)/total_inspections, 4),
        "chapters":          chapter_stats,
        "by_fuel":           by_fuel,
    }

def main():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("public", exist_ok=True)
    print("=== PKK LogReg pipeline v4 — 5-fold CV ===")
    raw     = load_all_data(max_files=6, sample_per_file=80000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")

    print("\n=== Cross-validation ===")
    cv_mean, cv_std, cv_scores = cross_validate_pipeline(feat_df)

    # AUC gate — fail fast if model quality is too low
    if cv_mean < 0.68:
        raise RuntimeError(f"AUC too low: {cv_mean:.4f} < 0.68 threshold")
    print(f"AUC gate passed: {cv_mean:.4f} >= 0.68")

    print("\n=== Training final model on full data ===")
    model, _, X = train_model(feat_df)

    coefs = extract_coefficients(model, feat_df, cv_mean, cv_std, cv_scores)

    print("\n=== Failure fingerprint (11 chapter models) ===")
    coefs["fingerprint"] = failure_fingerprint(feat_df, raw)
    coefs["defects"]     = defect_analysis(raw)

    out_path = "public/coefficients.json"
    with open(out_path, "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")
    print(json.dumps(coefs["meta"], indent=2))

if __name__ == "__main__":
    main()
