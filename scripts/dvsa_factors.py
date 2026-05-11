"""
UK DVSA MOT data → relative model-level failure factors.

Downloads anonymised MOT test results from:
  s3://anonymised-mot-test/  (public, no credentials needed)

Computes: for each (make, model, drivetrain_category), how much more or less
likely is this model to fail a given chapter compared to the make average?

factor > 1.0 → worse than make average
factor < 1.0 → better than make average

Output: scripts/dvsa_factors.json
"""

import os, json, io
import boto3
import pandas as pd
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config

BUCKET        = "anonymised-mot-test"
REGION        = "eu-west-1"
YEARS         = [2022, 2023]
MIN_SAMPLES   = 100
OUT_PATH      = "scripts/dvsa_factors.json"
NAME_MAP_PATH = "scripts/dvsa_name_map.json"

# PKK chapter → DVSA test_item_group prefix mapping
# DVSA groups failures by test item group (e.g. "1.1 Service brake...")
# We match the leading N.N prefix to our 11 PKK chapters
CHAPTER_GROUPS = {
    "Identification & documents": ["8.1", "8.2"],
    "Brakes":                     ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"],
    "Steering":                   ["2.1", "2.2", "2.3", "2.4"],
    "Visibility":                 ["4.1", "4.2", "4.3"],
    "Lights & electrical":        ["4.4", "4.5", "4.6", "4.7", "6.1", "6.2"],
    "Axles, wheels & tyres":      ["5.1", "5.2", "5.3"],
    "Chassis & body":             ["6.3", "6.4", "6.5"],
    "Other equipment":            ["7.1", "7.2"],
    "Noise & emissions":          ["7.3", "8.3"],
    "Environment":                ["8.4", "8.5"],
}


def fuel_to_drivetrain_category(fuel: str) -> str:
    """Map DVSA fuel type to drivetrain category."""
    fuel = str(fuel).upper()
    if "ELECTRIC" in fuel:  return "EV"
    if "HYBRID" in fuel:    return "HYBRID"
    if "DIESEL" in fuel:    return "DIESEL"
    return "PETROL"


def load_name_map() -> dict:
    with open(NAME_MAP_PATH) as f:
        return json.load(f)


def make_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED),
    )


def download_dvsa_year(year: int, s3_client) -> pd.DataFrame:
    """Try common key patterns for a given year. Returns empty DataFrame on failure."""
    key_candidates = [
        f"test_result_{year}.csv.gz",
        f"dft_test_result_{year}.csv.gz",
        f"dft-test-result-{year}.csv.gz",
    ]
    for key in key_candidates:
        print(f"  Trying s3://{BUCKET}/{key} ...")
        try:
            obj = s3_client.get_object(Bucket=BUCKET, Key=key)
            raw = obj["Body"].read()
            df = pd.read_csv(
                io.BytesIO(raw),
                compression="gzip",
                usecols=["make", "model", "fuel_type", "test_result", "test_item_group"],
                dtype=str,
                low_memory=False,
            )
            print(f"    Loaded {len(df):,} rows from {key}")
            return df
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception as e:
            print(f"    Error on {key}: {e}")
            continue
    print(f"  WARNING: Could not load year {year} — no matching key found")
    return pd.DataFrame()


def group_to_chapter(group_str: str) -> str | None:
    """Map DVSA test_item_group string to a PKK chapter name."""
    g = str(group_str).strip()
    for chapter, prefixes in CHAPTER_GROUPS.items():
        for p in prefixes:
            if g.startswith(p):
                return chapter
    return None


def compute_factors(df: pd.DataFrame, name_map: dict) -> dict:
    """
    For each (make, model, drivetrain_cat, chapter), compute relative factor
    vs. make-level average for that chapter.
    """
    make_map = name_map["make_map"]
    df = df.copy()
    df["make_norm"]  = df["make"].str.upper().str.strip().map(make_map)
    df = df[df["make_norm"].notna()].copy()
    df["drivetrain"] = df["fuel_type"].apply(fuel_to_drivetrain_category)
    df["failed"]     = (df["test_result"].str.upper() == "F").astype(int)
    df["chapter"]    = df["test_item_group"].apply(group_to_chapter)
    df = df[df["chapter"].notna()].copy()

    factors: dict = {}
    for make in df["make_norm"].unique():
        make_df = df[df["make_norm"] == make]
        factors[make] = {}

        # Make-level baseline per chapter (average across all drivetrain types)
        make_baseline = make_df.groupby("chapter")["failed"].mean()

        for model_raw in make_df["model"].dropna().unique():
            model = str(model_raw).upper().strip()
            model_df = make_df[make_df["model"].str.upper().str.strip() == model]
            if len(model_df) < MIN_SAMPLES:
                continue

            model_entry: dict = {}
            for drivetrain in ["PETROL", "DIESEL", "HYBRID", "EV"]:
                dt_df = model_df[model_df["drivetrain"] == drivetrain]
                if len(dt_df) < MIN_SAMPLES:
                    continue
                dt_entry: dict = {}
                for chapter in CHAPTER_GROUPS.keys():
                    ch_df = dt_df[dt_df["chapter"] == chapter]
                    if len(ch_df) < 30:
                        continue
                    model_rate = ch_df["failed"].mean()
                    make_rate  = make_baseline.get(chapter, np.nan)
                    if pd.isna(make_rate) or make_rate < 0.01:
                        continue
                    factor = round(float(model_rate / make_rate), 4)
                    dt_entry[chapter] = {
                        "factor": factor,
                        "n":      int(len(ch_df)),
                    }
                if dt_entry:
                    model_entry[drivetrain] = dt_entry
            if model_entry:
                factors[make][model] = model_entry

    return factors


def main() -> None:
    name_map = load_name_map()
    s3 = make_s3_client()

    all_frames: list[pd.DataFrame] = []
    for year in YEARS:
        df = download_dvsa_year(year, s3)
        if len(df) > 0:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError(
            "No DVSA data loaded. Check S3 bucket/key names at "
            "https://data.dft.gov.uk/anonymised-mot/ for current paths."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal DVSA rows: {len(combined):,}")
    print("Computing relative model factors ...")
    factors = compute_factors(combined, name_map)

    n_makes  = len(factors)
    n_models = sum(len(v) for v in factors.values())
    print(f"  {n_makes} makes, {n_models} models with factors")

    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "years_used":   YEARS,
        "n_rows":       int(len(combined)),
        "factors":      factors,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
