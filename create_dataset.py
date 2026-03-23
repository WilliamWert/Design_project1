"""
create_dataset.py
-----------------
Data acquisition, validation, and normalization script for the
Corporate Credit Rating dataset.

Project : DS 4320 – Corporate Bond Default Risk Prediction
Dataset : Kaggle Corporate Credit Rating
Source  : https://www.kaggle.com/datasets/agewerc/corporate-credit-rating
File    : corporate_rating.csv
Records : 2,029  |  Columns: 31  |  No missing values
Period  : January 2014 – September 2016

Output (4 normalized CSV files):
  companies.csv  – 593 rows × 3 cols   (Symbol PK, Name, Sector)
  agencies.csv   –   5 rows × 2 cols   (AgencyID PK, AgencyName)
  ratings.csv    – 2,029 rows × 5 cols (RatingID PK, Symbol FK, AgencyID FK, Date, Rating)
  financials.csv – 2,029 rows × 26 cols (RatingID PK/FK, + 25 financial ratio columns)
"""

# ── Step 1: Download from Kaggle ───────────────────────────────────────────────
# The dataset is downloaded directly from Kaggle — no programmatic generation.
#
# Option A (Kaggle CLI — requires kaggle.json credentials):
#   pip install kaggle
#   kaggle datasets download -d agewerc/corporate-credit-rating
#   unzip corporate-credit-rating.zip
#
# Option B (Manual):
#   Visit https://www.kaggle.com/datasets/agewerc/corporate-credit-rating
#   Click "Download" and save corporate_rating.csv to your working directory.

# ── Step 2: Validate the downloaded file ──────────────────────────────────────
import pandas as pd
import os


def validate_dataset(path="corporate_rating.csv"):
    """
    Load and validate the Corporate Credit Rating CSV.
    Raises AssertionError if any check fails.
    Returns the loaded DataFrame.
    """
    df = pd.read_csv(path)

    # Shape
    assert df.shape == (2029, 31), \
        f"Expected (2029, 31), got {df.shape}"

    # No missing values
    assert df.isnull().sum().sum() == 0, \
        f"Unexpected nulls: {df.isnull().sum()[df.isnull().sum() > 0]}"

    # Required columns present
    required = [
        "Rating", "Name", "Symbol", "Rating Agency Name", "Date", "Sector",
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
        "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
        "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed",
        "returnOnEquity", "assetTurnover", "fixedAssetTurnover",
        "debtEquityRatio", "debtRatio", "effectiveTaxRate",
        "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare",
        "cashPerShare", "companyEquityMultiplier", "ebitPerRevenue",
        "enterpriseValueMultiple", "operatingCashFlowPerShare",
        "operatingCashFlowSalesRatio", "payablesTurnover",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"

    # Primary key is unique: (Symbol, Date, Rating Agency Name)
    pk = ["Symbol", "Date", "Rating Agency Name"]
    assert not df.duplicated(subset=pk).any(), \
        "Composite primary key (Symbol, Date, Rating Agency Name) is not unique"

    # Expected rating values
    expected_ratings = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}
    actual_ratings   = set(df["Rating"].str.strip().str.upper().unique())
    assert actual_ratings == expected_ratings, \
        f"Unexpected rating values: {actual_ratings - expected_ratings}"

    # Expected rating agencies
    assert df["Rating Agency Name"].nunique() == 5, \
        f"Expected 5 rating agencies, got {df['Rating Agency Name'].nunique()}"

    print("All validation checks passed.")
    print(f"  Shape    : {df.shape}")
    print(f"  Companies: {df['Symbol'].nunique()}")
    print(f"  Agencies : {df['Rating Agency Name'].nunique()}")
    print(f"  Date range: {df['Date'].min()} — {df['Date'].max()}")
    print()
    print("Rating distribution:")
    order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    print(df["Rating"].value_counts().reindex(order).to_string())

    return df


# ── Step 3: Normalize into 4 relational tables ────────────────────────────────
RATIO_COLS = [
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
    "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed",
    "returnOnEquity", "assetTurnover", "fixedAssetTurnover",
    "debtEquityRatio", "debtRatio", "effectiveTaxRate",
    "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare",
    "cashPerShare", "companyEquityMultiplier", "ebitPerRevenue",
    "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "operatingCashFlowSalesRatio", "payablesTurnover",
]


def normalize_dataset(df, output_dir="."):
    """
    Normalize the flat corporate_rating DataFrame into 4 relational tables
    and write them as CSV files.

    Tables produced
    ---------------
    companies.csv  : Symbol (PK), Name, Sector
    agencies.csv   : AgencyID (PK), AgencyName
    ratings.csv    : RatingID (PK), Symbol (FK→companies), AgencyID (FK→agencies),
                     Date, Rating
    financials.csv : RatingID (PK/FK→ratings), + 25 financial ratio columns

    FK integrity is verified before writing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── companies ──────────────────────────────────────────────────────────────
    companies = (
        df[["Symbol", "Name", "Sector"]]
        .drop_duplicates(subset="Symbol")
        .reset_index(drop=True)
    )

    # ── agencies ───────────────────────────────────────────────────────────────
    agency_names = sorted(df["Rating Agency Name"].unique())
    agencies = pd.DataFrame({
        "AgencyID":   range(1, len(agency_names) + 1),
        "AgencyName": agency_names,
    })
    agency_map = dict(zip(agencies["AgencyName"], agencies["AgencyID"]))

    # ── ratings ────────────────────────────────────────────────────────────────
    ratings = df[["Symbol", "Rating Agency Name", "Date", "Rating"]].copy()
    ratings["AgencyID"] = ratings["Rating Agency Name"].map(agency_map)
    ratings = ratings.drop(columns=["Rating Agency Name"])
    ratings.insert(0, "RatingID", range(1, len(ratings) + 1))

    # ── financials ─────────────────────────────────────────────────────────────
    financials = df[RATIO_COLS].copy()
    financials.insert(0, "RatingID", range(1, len(financials) + 1))

    # ── FK integrity checks ────────────────────────────────────────────────────
    assert set(ratings["Symbol"]).issubset(set(companies["Symbol"])), \
        "FK violation: ratings.Symbol → companies.Symbol"
    assert set(ratings["AgencyID"]).issubset(set(agencies["AgencyID"])), \
        "FK violation: ratings.AgencyID → agencies.AgencyID"
    assert set(financials["RatingID"]).issubset(set(ratings["RatingID"])), \
        "FK violation: financials.RatingID → ratings.RatingID"

    # ── write CSVs ─────────────────────────────────────────────────────────────
    companies.to_csv(os.path.join(output_dir, "companies.csv"),  index=False)
    agencies.to_csv( os.path.join(output_dir, "agencies.csv"),   index=False)
    ratings.to_csv(  os.path.join(output_dir, "ratings.csv"),    index=False)
    financials.to_csv(os.path.join(output_dir, "financials.csv"), index=False)

    print("\nNormalization complete — 4 relational tables written:")
    print(f"  companies.csv  : {companies.shape[0]:,} rows × {companies.shape[1]} cols")
    print(f"  agencies.csv   : {agencies.shape[0]:,} rows × {agencies.shape[1]} cols")
    print(f"  ratings.csv    : {ratings.shape[0]:,} rows × {ratings.shape[1]} cols")
    print(f"  financials.csv : {financials.shape[0]:,} rows × {financials.shape[1]} cols")
    print("All FK integrity checks passed.")

    return companies, agencies, ratings, financials


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw = validate_dataset("corporate_rating.csv")
    normalize_dataset(raw, output_dir=".")
