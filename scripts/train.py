"""
PKK EU-kontroll — logistic regression training pipeline
Column names sourced directly from vegvesen/periodisk-kjoretoy-kontroll README
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

# Exact column names from the README (trimmed, lowercased for matching)
# readme lists them comma-separated with spaces: "Kjøretøymerke , Kjøretøy Modell , ..."
KNOWN_COLS = {
    "brand":    ["kjøretøymerke", "kj\u00f8ret\u00f8ymerke"],
    "fuel":     ["drivstofftype"],
    "km":       ["kilometerstand"],
    "reg_world":["første gang registrert"],
    "reg_no":   ["første gang registrert i norge"],
    "ctrl_type":["pkk kontrolltype"],
    "approved": ["om kjøretøyet ble godkjent"],
    "unsafe":   ["om det ble avdekket trafikkfarlig feil"],
    "fylke":    ["fylke der kjøretøyet er kontrollert"],
    "weight":   ["tillatt totalvekt"],
}

def list_zip_files():
    resp = requests.get(GITHUB_API, timeout=20)
    resp.raise_for_status()
    return sorted([f["name"] for f in resp.json() if f["name"].endswith(".zip")], reverse=True)

def download_zip(name):
    r = requests.get(RAW_BASE + name, timeout=120)
    r.raise_for_status()
    return r.content

def read_zip(content):
    """
    The PKK files use comma as separator with spaces around column names.
    Try comma first, then semicolon. Use latin-1 for Norwegian chars.
    """
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for member in z.namelist():
            print(f"    Member: {member}")
            with z.open(member) as f:
                raw = f.read()

            # Print first 500 bytes raw to see exact format
            print(f"    First 300 bytes hex: {raw[:300].hex()}")
            for enc in ["latin-1", "utf-8", "utf-8-sig", "cp1252"]:
                try:
                    preview = raw[:500].decode(enc)
                    print(f"    Preview ({enc}):\n{repr(preview[:300])}")
                    break
                except Exception:
                    pass

            # Now try to parse
            parsed = False
            for enc in ["latin-1", "utf-8", "utf-8-sig", "cp1252"]:
                for sep in [",", ";", "\t", "|"]:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(raw), sep=sep, encoding=enc,
                            low_memory=False, on_bad_lines="skip",
                            engine="python", skipinitialspace=True
                        )
                        print(f"    sep={repr(sep)} enc={enc}: {len(df.columns)} cols, {len(df):,} rows")
                        print(f"    Columns: {list(df.columns)}")
                        if len(df.columns) >= 5:
                            frames.append(df)
                            parsed = True
                            break
                    except Exception as e:
                        print(f"    sep={repr(sep)} enc={enc}: ERROR {e}")
                if parsed:
                    break

    return pd.concat(frames, ignore_index=True) if frames else None

def load_all_data(max_files=1):
    """Load only 1 file for diagnosis."""
    zips = list_zip_files()[:max_files]
    print(f"Using: {zips}")
    frames = []
    for name in zips:
        print(f"\nDownloading {name}...")
        try:
            content = download_zip(name)
            print(f"  {len(content):,} bytes downloaded")
            df = read_zip(content)
            if df is not None:
                frames.append(df)
                print(f"  Loaded {len(df):,} rows")
        except Exception as e:
            import traceback
            print(f"  FATAL ERROR: {traceback.format_exc()}")
    if not frames:
        raise RuntimeError("No data loaded")
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nFINAL: {len(combined):,} rows")
    print(f"FINAL COLUMNS: {list(combined.columns)}")
    return combined

def find_col(df, *keywords):
    for col in df.columns:
        cl = col.lower().strip()
        for kw in keywords:
            if kw.lower() in cl:
                return col
    return None

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)

    brand_col    = find_col(df, "kjøretøymerke", "merke")
    fuel_col     = find_col(df, "drivstofftype", "drivstoff")
    km_col       = find_col(df, "kilometerstand", "kilometer")
    reg_col      = find_col(df, "første gang registrert i norge", "registrert i nor", "første gang registrert")
    type_col     = find_col(df, "pkk kontrolltype", "kontrolltype")
    fylke_col    = find_col(df, "fylke der kjøretøyet", "fylke")
    weight_col   = find_col(df, "tillatt totalvekt", "totalvekt", "vektgruppe")
    approved_col = find_col(df, "om kjøretøyet ble godkjent", "godkjent")

    print(f"\nColumn detection:")
    for k, v in [("brand",brand_col),("fuel",fuel_col),("km",km_col),
                 ("reg",reg_col),("type",type_col),("fylke",fylke_col),
                 ("weight",weight_col),("approved",approved_col)]:
        print(f"  {k}: {v}")

    if approved_col is None:
        raise RuntimeError("Cannot find approved column. All columns: " + str(list(df.columns)))

    out["brand"] = (df[brand_col].astype(str).str.strip().str.upper().str[:30]
                    if brand_col else "UNKNOWN")
    if fuel_col:
        fl = df[fuel_col].astype(str).str.lower()
        out["fuel"] = "Other"
        out.loc[fl.str.contains("elektr", na=False), "fuel"] = "EV"
        out.loc[fl.str.contains("hybrid", na=False), "fuel"] = "Hybrid"
        out.loc[fl.str.contains("diesel", na=False), "fuel"] = "Diesel"
        out.loc[fl.str.contains("bensin|gasolin|petrol", na=False), "fuel"] = "Bensin"
    else:
        out["fuel"] = "Other"
    if km_col:
        out["km"] = pd.to_numeric(
            df[km_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
            errors="coerce").clip(0, 500_000)
    else:
        out["km"] = np.nan
    if reg_col:
        reg = pd.to_numeric(df[reg_col], errors="coerce")
        age = (now_year - reg).where(lambda x: (x >= 0) & (x <= 50))
        out["age"] = age.clip(0, 30)
    else:
        out["age"] = np.nan
    if type_col:
        ct = df[type_col].astype(str).str.strip().str.upper()
        out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")
    else:
        out["ctrl_type"] = "P"
    out["fylke"]  = (df[fylke_col].astype(str).str.strip().str[:40] if fylke_col else "UNKNOWN")
    out["weight"] = (df[weight_col].astype(str).str.strip().str[:20] if weight_col else "Lette")

    raw = df[approved_col].astype(str).str.strip().str.upper()
    approved = pd.Series(np.nan, index=df.index)
    approved[raw.isin(["1","JA","YES","TRUE","GODKJENT"])] = 1
    approved[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])] = 0
    num = pd.to_numeric(df[approved_col], errors="coerce")
    mask = approved.isna() & num.isin([0.0, 1.0])
    approved[mask] = num[mask]
    out["approved"] = approved

    before = len(out)
    out = out.dropna(subset=["km", "age", "approved"]).copy()
    out["approved"] = out["approved"].astype(int)
    out["brand"]   = out["brand"].fillna("UNKNOWN")
    out["fylke"]   = out["fylke"].fillna("UNKNOWN")
    print(f"\nRows after cleaning: {len(out):,} (dropped {before-len(out):,})")
    print(f"Approved: {out['approved'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["brand","fuel","ctrl_type","fylke","weight"]),
        ("num", StandardScaler(), ["km","age"]),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))])
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, model.predict(X)))
    print(f"AUC-ROC: {auc:.4f}")
    return model, auc

def extract_coefficients(model, feat_df, auc):
    clf   = model.named_steps["clf"]
    pre   = model.named_steps["pre"]
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    def group(prefix):
        return {n[len(prefix):]: round(float(c), 4)
                for n, c in zip(names, coefs) if n.startswith(prefix)}
    top_brands = set(feat_df["brand"].value_counts()[lambda x: x >= 50].index)
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
        "brand":     {k: v for k, v in group("cat__brand_").items() if k in top_brands},
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
    raw     = load_all_data(max_files=1)   # 1 file for diagnosis
    feat_df = engineer_features(raw)
    if len(feat_df) < 100:
        raise RuntimeError(f"Too few rows ({len(feat_df)}) — check column detection above")
    model, auc = train_model(feat_df)
    coefs      = extract_coefficients(model, feat_df, auc)
    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(coefs["meta"], indent=2))

if __name__ == "__main__":
    main()
            for enc in ["utf-8", "latin-1", "utf-8-sig"]:
                try:
                    text = raw.decode(enc)
                    lines = text.splitlines()
                    print(f"\n  --- First 5 lines ({enc}) ---")
                    for line in lines[:5]:
                        print(f"  {repr(line)}")
                    # detect separator
                    first = lines[0] if lines else ""
                    for sep in [";", ",", "\t", "|"]:
                        count = first.count(sep)
                        if count > 0:
                            print(f"  Separator '{sep}' appears {count} times in header")
                    break
                except Exception as e:
                    print(f"  {enc} decode failed: {e}")

def try_parse_csv(raw_bytes, member_name):
    """Try every combination to parse, with full error reporting."""
    for enc in ["utf-8", "latin-1", "utf-8-sig", "cp1252"]:
        for sep in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="skip",
                    engine="python"
                )
                if len(df.columns) >= 5:
                    print(f"      Parsed with enc={enc} sep={repr(sep)}: {len(df):,} rows, {len(df.columns)} cols")
                    print(f"      Columns: {list(df.columns)[:8]}...")
                    return df
                else:
                    print(f"      enc={enc} sep={repr(sep)}: only {len(df.columns)} columns, skipping")
            except Exception as e:
                print(f"      enc={enc} sep={repr(sep)}: {type(e).__name__}: {str(e)[:80]}")
    return None

def read_zip(content, zip_name):
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for member in z.namelist():
            print(f"    Reading {member}...")
            with z.open(member) as f:
                raw = f.read()
            ext = member.lower().split(".")[-1]
            if ext == "csv":
                df = try_parse_csv(raw, member)
                if df is not None:
                    frames.append(df)
                else:
                    print(f"    FAILED to parse {member} with any encoding/separator")
            else:
                print(f"    Skipping non-CSV: {member} (extension: {ext})")
    return pd.concat(frames, ignore_index=True) if frames else None

def load_all_data(max_files=2):
    """Load only 2 files for speed during diagnosis."""
    zips = list_zip_files()[:max_files]
    print(f"Found zip files, using first {len(zips)}: {zips}")
    frames = []
    for name in zips:
        print(f"\nDownloading {name}...")
        try:
            content = download_zip(name)
            print(f"  Downloaded {len(content):,} bytes")
            # always diagnose first
            diagnose_zip(content, name)
            df = read_zip(content, name)
            if df is not None:
                frames.append(df)
                print(f"  SUCCESS: {len(df):,} rows")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    if not frames:
        raise RuntimeError("No data loaded — see diagnostic output above")
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined):,}")
    print(f"ALL COLUMNS: {list(combined.columns)}")
    return combined

def find_col(df, *keywords):
    for col in df.columns:
        cl = col.lower().strip()
        for kw in keywords:
            if kw in cl:
                return col
    return None

def engineer_features(df):
    now_year = datetime.now().year
    out = pd.DataFrame(index=df.index)

    brand_col    = find_col(df, "merke")
    fuel_col     = find_col(df, "drivstoff")
    km_col       = find_col(df, "kilometer")
    reg_col      = find_col(df, "registrert i nor", "f\u00f8rste gang registrert")
    type_col     = find_col(df, "kontrolltype")
    fylke_col    = find_col(df, "fylke")
    weight_col   = find_col(df, "totalvekt", "vektgruppe")
    approved_col = find_col(df, "om kj\u00f8ret\u00f8yet ble godkjent", "godkjent")

    print(f"\nDetected columns:")
    print(f"  brand={brand_col}, fuel={fuel_col}, km={km_col}")
    print(f"  reg={reg_col}, type={type_col}, fylke={fylke_col}")
    print(f"  weight={weight_col}, approved={approved_col}")

    if approved_col is None:
        raise RuntimeError("Cannot find approved/godkjent column.\nAll columns: " + str(list(df.columns)))

    out["brand"] = (df[brand_col].astype(str).str.strip().str.upper().str[:30]
                    if brand_col else "UNKNOWN")
    if fuel_col:
        fl = df[fuel_col].astype(str).str.lower()
        out["fuel"] = "Other"
        out.loc[fl.str.contains("elektr", na=False), "fuel"] = "EV"
        out.loc[fl.str.contains("hybrid", na=False), "fuel"] = "Hybrid"
        out.loc[fl.str.contains("diesel", na=False), "fuel"] = "Diesel"
        out.loc[fl.str.contains("bensin|gasolin|petrol", na=False), "fuel"] = "Bensin"
    else:
        out["fuel"] = "Other"
    if km_col:
        out["km"] = pd.to_numeric(
            df[km_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
            errors="coerce").clip(0, 500_000)
    else:
        out["km"] = np.nan
    if reg_col:
        reg = pd.to_numeric(df[reg_col], errors="coerce")
        age = (now_year - reg).where(lambda x: (x >= 0) & (x <= 50))
        out["age"] = age.clip(0, 30)
    else:
        out["age"] = np.nan
    if type_col:
        ct = df[type_col].astype(str).str.strip().str.upper()
        out["ctrl_type"] = np.where(ct.str.startswith("E"), "E", "P")
    else:
        out["ctrl_type"] = "P"
    out["fylke"]  = (df[fylke_col].astype(str).str.strip().str[:40] if fylke_col else "UNKNOWN")
    out["weight"] = (df[weight_col].astype(str).str.strip().str[:20] if weight_col else "Lette")

    raw = df[approved_col].astype(str).str.strip().str.upper()
    approved = pd.Series(np.nan, index=df.index)
    approved[raw.isin(["1","JA","YES","TRUE","GODKJENT"])] = 1
    approved[raw.isin(["0","NEI","NO","FALSE","IKKE GODKJENT"])] = 0
    num = pd.to_numeric(df[approved_col], errors="coerce")
    mask = approved.isna() & num.isin([0.0, 1.0])
    approved[mask] = num[mask]
    out["approved"] = approved

    before = len(out)
    out = out.dropna(subset=["km", "age", "approved"]).copy()
    out["approved"]  = out["approved"].astype(int)
    out["brand"]     = out["brand"].fillna("UNKNOWN")
    out["fylke"]     = out["fylke"].fillna("UNKNOWN")
    print(f"\nRows after cleaning: {len(out):,} (dropped {before-len(out):,})")
    print(f"Approved: {out['approved'].value_counts().to_dict()}")
    return out.reset_index(drop=True)

def train_model(feat_df):
    X = feat_df.drop(columns=["approved"])
    y = feat_df["approved"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["brand","fuel","ctrl_type","fylke","weight"]),
        ("num", StandardScaler(), ["km","age"]),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))])
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_prob)
    print(classification_report(y, model.predict(X)))
    print(f"AUC-ROC: {auc:.4f}")
    return model, auc

def extract_coefficients(model, feat_df, auc):
    clf   = model.named_steps["clf"]
    pre   = model.named_steps["pre"]
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    def group(prefix):
        return {n[len(prefix):]: round(float(c), 4)
                for n, c in zip(names, coefs) if n.startswith(prefix)}
    top_brands = set(feat_df["brand"].value_counts()[lambda x: x >= 50].index)
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
        "brand":     {k: v for k, v in group("cat__brand_").items() if k in top_brands},
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
    raw     = load_all_data(max_files=2)   # only 2 files for diagnosis
    feat_df = engineer_features(raw)
    if len(feat_df) < 1000:
        raise RuntimeError(f"Too few usable rows ({len(feat_df)})")
    model, auc = train_model(feat_df)
    coefs      = extract_coefficients(model, feat_df, auc)
    with open("docs/coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print("\nSaved docs/coefficients.json")
    print(json.dumps(coefs["meta"], indent=2))

if __name__ == "__main__":
    main()
