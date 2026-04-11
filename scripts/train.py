"""
PKK EU-kontroll — logistic regression pipeline v3
- 80k rows per file (~480k total)
- inspection_number feature (which inspection is this for the vehicle)
- Fuel: BEV, Hybrid, Diesel, Petrol, Other
- No fylke (national model)
- Bootstrap CI (8 resamples)
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
    """
    Estimate which inspection number this is for the vehicle.
    Norway rules: first inspection at 4 years, then every 2 years.
    inspection_number = 1 means first-ever inspection.
    """
    # use Norwegian registration year preferentially
    reg_no  = pd.to_numeric(df.get("Første gang registrert i Norge"), errors="coerce")
    reg_wld = pd.to_numeric(df.get("Første gang registrert"), errors="coerce")
    reg     = reg_no.fillna(reg_wld)

    age = (now_year - reg).clip(0, 50)

    # interval from PKK Intervall column (years between inspections, typically 2)
    interval = pd.to_numeric(df.get("PKK Intervall"), errors="coerce").fillna(2).clip(1, 4)

    # first inspection deadline = 4 years after registration
    # subsequent = every `interval` years
    # inspection_number = 1 + max(0, floor((age - 4) / interval))
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

    # inspection number — key new feature
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
    print(f"Inspection number distribution:\n{out['insp_num'].value_counts().sort_index().head(8)}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    cat_features = ["brand","fuel","ctrl_type","weight"]
    num_features = ["km","age","insp_num"]
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
    print(f"Bootstrap CI ({n_boot} resamples)...")
    cat_features = ["brand","fuel","ctrl_type","weight"]
    num_features = ["km","age","insp_num"]

    def fit_lr(X, y):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ])
        m = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=300, C=1.0,
                                                 class_weight="balanced", solver="saga"))])
        m.fit(X, y)
        return m

    X0 = feat_df.drop(columns=["approved"])
    y0 = feat_df["approved"]
    m0 = fit_lr(X0, y0)
    names = list(m0.named_steps["pre"].get_feature_names_out())
    n_feat = len(names)
    boot_coefs = np.zeros((n_boot, n_feat))

    for i in range(n_boot):
        idx = np.random.choice(len(feat_df), size=len(feat_df), replace=True)
        df_b = feat_df.iloc[idx]
        m_b = fit_lr(df_b.drop(columns=["approved"]), df_b["approved"])
        names_b = list(m_b.named_steps["pre"].get_feature_names_out())
        coefs_b = m_b.named_steps["clf"].coef_[0]
        for j, nm in enumerate(names):
            if nm in names_b:
                boot_coefs[i, j] = coefs_b[names_b.index(nm)]
        print(f"  Bootstrap {i+1}/{n_boot}")

    stds = boot_coefs.std(axis=0)
    return dict(zip(names, stds.tolist())), names

def extract_coefficients(model, feat_df, auc, stds, names):
    clf   = model.named_steps["clf"]
    pre   = model.named_steps["pre"]
    coefs = clf.coef_[0]
    z     = 1.96

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

    # pass rate by inspection number for frontend calibration
    insp_pass = feat_df.groupby("insp_num")["approved"].agg(["mean","count"])
    insp_rates = {
        int(k): {"pass_rate": round(float(v["mean"]),4), "n": int(v["count"])}
        for k, v in insp_pass.iterrows()
        if v["count"] >= 100
    }

    # also compute brand→fuel mapping from data
    brand_fuel = {}
    for brand in feat_df["brand"].unique():
        sub = feat_df[feat_df["brand"]==brand]
        dominant = sub["fuel"].value_counts()
        if len(dominant) > 0:
            # store top fuel and whether brand is exclusively one type
            top = dominant.index[0]
            top_pct = dominant.iloc[0] / len(sub)
            brand_fuel[brand] = {
                "dominant": top,
                "exclusive": bool(top_pct > 0.95)
            }

    return {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples":  int(len(feat_df)),
            "auc_roc":    round(float(auc), 4),
            "pass_rate":  round(float(feat_df["approved"].mean()), 4),
            "model":      "Logistic Regression v3",
        },
        "intercept":  round(float(clf.intercept_[0]), 4),
        "brand":      group_with_ci("cat__brand_"),
        "fuel":       group_with_ci("cat__fuel_"),
        "ctrl_type":  group_with_ci("cat__ctrl_type_"),
        "weight":     group_with_ci("cat__weight_"),
        "numeric": {
            "km_scaled":    round(float(num_map.get("num__km",      0)), 6),
            "age_scaled":   round(float(num_map.get("num__age",     0)), 6),
            "insp_scaled":  round(float(num_map.get("num__insp_num",0)), 6),
            "km_corr":      round(float(feat_df["km"].corr(feat_df["approved"])),       4),
            "age_corr":     round(float(feat_df["age"].corr(feat_df["approved"])),      4),
            "insp_corr":    round(float(feat_df["insp_num"].corr(feat_df["approved"])), 4),
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
        "pass_rate":       round(float(feat_df["approved"].mean()), 4),
    }

def failure_fingerprint(feat_df, raw_df):
    """
    Train 11 logistic regressions, one per defect chapter.
    For each chapter, output:
    - national baseline fault rate
    - coefficients for brand, fuel, age, km
    So the frontend can predict per-vehicle chapter risk.
    """
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

    cat_features = ["brand", "fuel", "ctrl_type", "weight"]
    num_features = ["km", "age", "insp_num"]

    # align raw_df index with feat_df
    raw_aligned = raw_df.iloc[:len(feat_df)].copy()
    raw_aligned.index = feat_df.index

    fingerprint = {}
    for col, name in CAP_COLS.items():
        if col not in raw_aligned.columns:
            print(f"  Skipping {name} — column not found")
            continue

        # binary target: did this inspection have at least one fault in this chapter?
        y_chap = (pd.to_numeric(raw_aligned[col], errors="coerce")
                    .fillna(0).reindex(feat_df.index).fillna(0) > 0).astype(int)

        if y_chap.sum() < 50:
            print(f"  Skipping {name} — too few positives ({y_chap.sum()})")
            continue

        baseline = round(float(y_chap.mean()), 4)

        # Skip chapters with very low national fault rate — not enough signal
        # and balanced weighting makes predictions nonsensical for rare events
        if baseline < 0.02:
            print(f"  Skipping {name} — baseline too low ({baseline:.3f}), not enough signal")
            continue

        print(f"  Training {name}: baseline={baseline:.3f}, positives={y_chap.sum()}")

        X = feat_df.drop(columns=["approved"])

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features),
        ])
        # No class_weight="balanced" here — we want realistic probability estimates
        # anchored to the actual fault rate, not artificially centred at 50%
        m = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=300, C=1.0, solver="saga"))
        ])
        m.fit(X, y_chap)

        feat_names = list(m.named_steps["pre"].get_feature_names_out())
        coefs      = m.named_steps["clf"].coef_[0]
        coef_map   = dict(zip(feat_names, coefs))
        sc         = m.named_steps["pre"].named_transformers_["num"]

        def g(prefix):
            return {
                k[len(prefix):]: round(float(v), 4)
                for k, v in coef_map.items() if k.startswith(prefix)
            }

        fingerprint[name] = {
            "baseline":   baseline,
            "intercept":  round(float(m.named_steps["clf"].intercept_[0]), 4),
            "brand":      g("cat__brand_"),
            "fuel":       g("cat__fuel_"),
            "ctrl_type":  g("cat__ctrl_type_"),
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
    print("=== PKK Logistic Regression pipeline v3 ===")
    raw     = load_all_data(max_files=6, sample_per_file=80000)
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few rows ({len(feat_df)})")
    model, auc, X = train_model(feat_df)
    stds, names   = bootstrap_ci(feat_df, n_boot=8)
    coefs         = extract_coefficients(model, feat_df, auc, stds, names)
    print("\n=== Failure fingerprint (11 chapter models) ===")
    coefs["fingerprint"] = failure_fingerprint(feat_df, raw)
    coefs["defects"]     = defect_analysis(raw)

    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(coefs["meta"], indent=2))
    d = coefs["defects"]
    print(f"Top defect: {d['chapters'][0]['name']} ({d['chapters'][0]['rate_all']*100:.1f}% of all inspections)")
    print(f"Brand→fuel mappings: {len(coefs['brand_fuel'])} brands")
    print(f"Fingerprint chapters trained: {len(coefs['fingerprint'])}/11")

if __name__ == "__main__":
    main()
