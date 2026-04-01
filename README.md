# DS 4320 Project 1: Predicting Corporate Credit Ratings from Financial Ratios

## Executive Summary

This repository contains the complete DS 4320 Project 1 submission. The project addresses corporate bond default risk prediction: given 25 financial ratios derived from a company's balance sheet, income statement, and cash flow data, can a machine learning classifier predict the credit rating grade assigned by a rating agency — and can it do so without the conflict-of-interest that affects agency ratings? The dataset is the Kaggle Corporate Credit Rating dataset (2,029 rating observations for 593 publicly traded US companies from 5 rating agencies, 2014–2016), normalized into a four-table relational schema. Four classifiers are trained and evaluated (Logistic Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors); the best models (Random Forest and Gradient Boosting) achieve approximately 51% macro-averaged F1 across 7 rating classes, well above the 14% random baseline. The full pipeline — data normalization, DuckDB loading, SQL feature engineering, model training, and publication-quality visualization — is documented in `pipeline.ipynb`.

**Name:** William Wert

**NetID:** dxg9tt

**DOI:** `[10.5281/zenodo.19363237]`

**Press Release:** [press_release.md](press_release.md)

**Data:** [UVA OneDrive Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/dxg9tt_virginia_edu/IgAHpnTba-X-Trud1TZDWY44ARDRxI37PFLqHGhRr7vc4XY?e=S0GoOx)

**Pipeline:** [pipeline.ipynb](pipeline.ipynb) · [pipeline.md](pipeline.md)

**License:** [LICENSE](LICENSE)

---

## Problem Definition

### General and Specific Problem Statement

**General Problem:** Investors and financial institutions struggle to accurately assess which corporate bonds are at elevated risk of default before traditional credit rating agencies downgrade them, leading to mispriced portfolios, unexpected losses, and reactive rather than proactive risk management.

**Specific Problem:** Using the Kaggle Corporate Credit Rating dataset — which pairs 2,029 publicly traded US companies with their credit ratings from five agencies and 25 financial ratios — can we train a machine learning classifier that predicts the full letter-grade rating (AAA through CCC/D) from financial statement data alone, and do tree-based ensemble models capture non-linear risk signals that a linear baseline misses?

### Rationale for Refinement

The refinement from the general problem to the specific one is driven by two constraints: what the data contains and what is analytically actionable. The general problem could be addressed by sentiment analysis of earnings calls, insider trading signals, or macroeconomic models — but the dataset at hand contains financial ratio snapshots for publicly traded US companies paired with their assigned credit ratings. This makes a classification approach the natural choice: use the 25 financial ratios as features and the rating grade as the label, asking whether financial statement signals alone can recover the grade boundaries that agencies assign. This framing is also grounded in a well-established literature — the Altman Z-score model has been predicting corporate distress from financial ratios since 1968 — which gives the specific problem both legitimacy and a natural benchmark to test against.

### Motivation

Corporate bond defaults are not rare events — they are a recurring feature of credit markets that accelerate dramatically during economic downturns, with global corporate default rates reaching 12% during the 2009 financial crisis and 6% at the peak of the COVID-19 shock in 2020. When a large issuer defaults, the consequences ripple through institutional portfolios, pension funds, and insurance companies that hold those bonds as "safe" assets based on their assigned credit ratings. The problem is that rating agencies — Moody's, S&P, and Fitch — are paid by the issuers they rate, a structural conflict of interest that has been documented to produce ratings inflation, delayed downgrades, and systematically optimistic assessments for large clients. A data-driven classifier trained on objective financial ratios has no such conflict. If financial ratios contain forward-looking distress signals that agency ratings underweight — and the literature suggests they do — then a machine learning model could serve as an independent early-warning system, giving investors a second opinion before the agencies act.

### Press Release

**Headline:** New Machine Learning Model Spots Corporate Bond Defaults Months Before Rating Agencies Do

See full press release: [press_release.md](press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|------------|
| Corporate Bond | A debt security issued by a corporation to raise capital; the issuer promises to pay periodic interest (coupon) and return principal at maturity |
| Default | Failure by a bond issuer to make a scheduled interest or principal payment; the terminal credit event |
| Credit Rating | A letter-grade assessment of an issuer's creditworthiness assigned by a ratings agency (AAA = safest, D = default) |
| Investment Grade | Bonds rated BBB/Baa or above; considered lower default risk and eligible for many institutional portfolios |
| Speculative Grade | Bonds rated BB/Ba or below (also called "junk" or "high yield"); higher default risk, higher yield |
| Yield Spread | The difference between a corporate bond's yield and an equivalent-maturity US Treasury yield; widens as default risk increases |
| Debt/Equity Ratio | Total debt divided by shareholders' equity; measures financial leverage and capital structure risk |
| Interest Coverage Ratio | EBIT divided by annual interest expense; measures ability to service debt from operating earnings |
| Current Ratio | Current assets divided by current liabilities; a measure of short-term liquidity |
| Altman Z-Score | A composite of five financial ratios developed in 1968 to predict corporate bankruptcy; still widely used as a baseline model |
| ROE | Return on Equity — net income divided by shareholders' equity; a measure of profitability relative to capital |
| Coupon Rate | The fixed annual interest rate paid by a bond issuer to bondholders, expressed as a percentage of face value |
| OAS | Option-Adjusted Spread — the yield spread of a corporate bond above equivalent Treasuries, adjusted for any embedded options |
| Rating Agency | An independent firm (Moody's, S&P, Fitch) that assigns credit ratings to bond issuers; paid by the issuers being rated |
| Issuer-Pay Model | The dominant business model for rating agencies, where the company being rated pays for its own rating; a documented source of ratings inflation |
| Class Imbalance | In default prediction, actual defaults are rare relative to non-defaults, requiring special modeling techniques (e.g., SMOTE, weighted loss) |
| SMOTE | Synthetic Minority Over-sampling Technique — a method for addressing class imbalance by generating synthetic examples of the minority class |

### Domain Overview

This project sits at the intersection of quantitative finance, credit risk management, and machine learning. Credit risk — the risk that a borrower will fail to meet their obligations — is one of the oldest and most consequential problems in finance, and the corporate bond market is where it is most consequentially priced. With over $10 trillion in US corporate bonds outstanding, even small improvements in default prediction accuracy translate into billions of dollars of better-allocated capital. The domain has evolved from rule-based ratio analysis (Altman's Z-score) through logistic regression and discriminant analysis to modern ensemble methods and neural networks. The key domain-specific challenge is class imbalance: actual defaults are rare even in stressed markets, which means any classifier trained on historical data must handle a heavily skewed label distribution. The data science layer involves not just model training but careful feature engineering from financial statements, understanding of how accounting ratios relate to economic solvency, and calibration of probability outputs that can be acted on by portfolio managers and risk officers.

### Background Readings

Background reading files are stored in the [project OneDrive folder](https://myuva-my.sharepoint.com/:f:/r/personal/dxg9tt_virginia_edu/Documents/DS4320_project1?csf=1&web=1&e=sXAo77).

| Title | Description |
|-------|-------------|
| A Bibliometric Study on Intelligent Techniques of Bankruptcy Prediction for Corporate Firms | Comprehensive survey of bankruptcy prediction methods from Altman Z-score through modern ML, tracing the evolution of the field |
| Financial Distress Prediction Using Integrated Z-Score and Multilayer Perceptron Neural Networks | Empirical comparison of the Altman Z-score baseline against neural networks on financial ratio data |
| Detecting Conflicts of Interest in Credit Rating Changes: A Distribution Dynamics Approach | Quantitative analysis of rating agency incentive biases and delayed downgrades in the issuer-pay model |
| Datasets for Advanced Bankruptcy Prediction: A Survey and Taxonomy | Survey of publicly available datasets for corporate default prediction including feature descriptions and benchmarks |
| Testing Conflicts of Interest at Bond Ratings Agencies | Federal Reserve empirical paper testing whether rating agency conflicts of interest produce systematically inflated ratings |

---

## Data Creation

### Provenance

The project dataset is the **Kaggle Corporate Credit Rating** dataset, available at https://www.kaggle.com/datasets/agewerc/corporate-credit-rating, downloaded in Spring 2026. The dataset aggregates credit ratings issued by five major rating agencies — Standard & Poor's Ratings Services (n=744), Egan-Jones Ratings Company (n=603), Moody's Investors Service (n=579), Fitch Ratings (n=100), and DBRS (n=3) — for 593 publicly traded US companies across 12 business sectors. Ratings span the period January 2014 through September 2016. For each rating observation, the dataset pairs the agency's credit rating (on the S&P scale from AAA to D) with 25 financial ratios computed from the company's publicly available financial statements (balance sheet, income statement, and cash flow). The resulting flat CSV contains 2,029 records and 31 fields, with no missing values.

The dataset was downloaded directly from Kaggle as a single CSV file. No merging, web scraping, or API calls were required. The raw file was uploaded to the notebook environment without modification. The financial ratios were pre-computed by the Kaggle dataset author from public SEC filings and financial data providers; the specific sources and computation dates for each ratio are not provided in the dataset metadata, which is itself a source of uncertainty documented below.

### Code

| File | Description |
|------|-------------|
| [`create_dataset.py`](create_dataset.py) | Downloads the raw dataset from Kaggle, validates its integrity (shape, nulls, PK uniqueness, rating values), and normalizes the flat CSV into the four-table relational schema: `companies`, `agencies`, `ratings`, `financials`. |
| [`companies.csv`](companies.csv) | Dimension table of 593 unique publicly traded companies. Generated by `create_dataset.py`. |
| [`agencies.csv`](agencies.csv) | Dimension table of 5 rating agencies. Generated by `create_dataset.py`. |
| [`ratings.csv`](ratings.csv) | Fact table of 2,029 credit rating observations. Generated by `create_dataset.py`. |
| [`financials.csv`](financials.csv) | Fact table of 25 financial ratios, one row per rating observation. Generated by `create_dataset.py`. |
| [`pipeline.ipynb`](pipeline.ipynb) | Full problem solution pipeline: DuckDB loading, SQL EDA, feature matrix construction, model training, and visualization. |

### Bias Identification

Several sources of bias affect the corporate credit rating dataset. First, **rating agency incentive bias**: four of the five agencies in this dataset (S&P, Moody's, Fitch, DBRS) operate under the issuer-pay model, in which the company being rated pays for its own rating — a structural conflict of interest documented to produce inflated ratings and delayed downgrades for large clients. Egan-Jones operates on an investor-pay model and is generally regarded as more timely. Pooling ratings from these agencies without accounting for this structural difference introduces systematic bias in the label variable. Second, **temporal snapshot bias**: all ratings fall within January 2014 – September 2016, a period of relative macroeconomic stability. Models trained on this window may not generalize to distressed market conditions. Third, **class imbalance**: the rating distribution is heavily skewed toward BBB (n=671) and BB (n=490), with very few observations in the most extreme classes (AAA: 7, CC: 5, C: 2, D: 1). Any classifier trained on this data will be biased toward the majority classes. Fourth, **financial ratio data quality**: many financial ratio columns contain extreme outliers inconsistent with typical corporate financials, suggesting data entry errors, non-standard financial reporting, or financial institution effects that make ratios non-comparable across sectors. Fifth, **selection bias**: the Kaggle dataset does not document how the 593 companies were selected, making it difficult to assess the representativeness of the sample.

### Bias Mitigation

The following strategies can address or quantify the biases described above. (1) **Agency stratification**: analyses should be run separately by rating agency and results compared; if agencies produce systematically different financial ratio profiles for the same rating, the issuer-pay bias is detectable and can be quantified. (2) **Class balancing**: for any classifier, address imbalance using oversampling of minority classes (SMOTE), undersampling of the majority class, or class-weighted loss functions; report performance metrics separately for each class (precision, recall, F1) rather than relying on overall accuracy. (3) **Outlier treatment**: winsorize financial ratio columns at the 1st and 99th percentiles before modeling; document which records are affected and perform sensitivity analyses with and without winsorization to quantify the impact on results. (4) **Sector stratification**: financial ratios are not directly comparable across sectors; sector-adjusted z-scores or sector-fixed effects can reduce this confounding. (5) **Temporal cross-validation**: use time-aware train/test splits (e.g., train on 2014–2015, test on 2016) rather than random splits, to avoid data leakage and better simulate real-world deployment conditions.

### Rationale for Critical Decisions

**Dataset selection.** The Kaggle Corporate Credit Rating dataset was chosen because it directly pairs financial ratio snapshots with credit rating labels from multiple agencies, enabling a supervised classification approach without requiring additional data joins. Alternative sources (e.g., WRDS Compustat) would offer broader coverage but require institutional access not available for this project.

**Normalization into four tables.** The raw Kaggle CSV is a fully denormalized flat file that repeats company name and sector on every row and embeds agency names as strings. Decomposing it into four tables — `companies`, `agencies`, `ratings`, `financials` — eliminates this redundancy and satisfies first, second, and third normal form (3NF). The `companies` table (593 rows) stores each company once; the `agencies` table (5 rows) stores each agency once; the `ratings` table (2,029 rows) records each (company, agency, date, rating) observation; and the `financials` table (2,029 rows, linked 1-to-1 to ratings via RatingID) stores the 25 financial ratios separately from the rating label, enabling independent querying of financial features without loading rating metadata.

**Primary key design.** Each table uses a purpose-appropriate key: `companies` uses the natural key `Symbol`; `agencies` uses a synthetic integer `AgencyID`; `ratings` uses a synthetic `RatingID` that also serves as the FK anchor for `financials`; `financials` reuses `RatingID` as both its PK and FK, enforcing the 1-to-1 relationship with `ratings`.

**Outlier retention.** Extreme values in financial ratio columns are retained in the raw tables and documented in the data dictionary. They are addressed during the modeling pipeline (class-weighted loss functions, StandardScaler normalization) rather than in the stored data, keeping preprocessing decisions explicit and reproducible.

---

## Metadata

### Schema — ER Diagram

Four-table relational schema. `companies` and `agencies` are dimension tables; `ratings` and `financials` are fact tables. All foreign keys are enforced programmatically (verified in `pipeline.ipynb` and `create_dataset.py`).

```
companies                    agencies
─────────────────────        ─────────────────
Symbol  VARCHAR  PK          AgencyID  INT  PK
Name    VARCHAR              AgencyName VARCHAR
Sector  VARCHAR
        │                           │
        │ (Symbol FK)               │ (AgencyID FK)
        ▼                           ▼
        ┌────────────── ratings ───────────────┐
        │  RatingID  INT        PK             │
        │  Symbol    VARCHAR    FK → companies │
        │  AgencyID  INT        FK → agencies  │
        │  Date      VARCHAR                   │
        │  Rating    VARCHAR                   │
        └──────────────────────┬───────────────┘
                               │ (RatingID FK, 1:1)
                               ▼
                    financials
                    ─────────────────────────────
                    RatingID  INT   PK / FK
                    currentRatio          DOUBLE
                    quickRatio            DOUBLE
                    cashRatio             DOUBLE
                    ... (25 ratio columns total)
```

### Data Table

| Table | Description | Rows | Cols | Link to CSV |
|-------|-------------|------|------|-------------|
| `companies` | Dimension table. One row per unique publicly traded company. Contains company identifier, full name, and business sector. | 593 | 3 | [companies.csv](https://drive.google.com/file/d/1RQ7rxnbCFkIFTmOgugIz3iYh3de4kUxT/view?usp=drive_link) |
| `agencies` | Dimension table. One row per rating agency. Contains synthetic AgencyID and agency name. | 5 | 2 | [agencies.csv](https://drive.google.com/file/d/1EMDsU4ETnpxEVZo7kJrlesjCqyT8yxeQ/view?usp=drive_link) |
| `ratings` | Fact table. One row per (company, agency, date) rating observation. Links companies and agencies via FKs; stores the S&P-scale rating label. | 2,029 | 5 | [ratings.csv](https://drive.google.com/file/d/1Wlihvk4jxBDD1AK6_C082-ti-wlhHWj6/view?usp=drive_link) |
| `financials` | Fact table. One row per rating observation (1-to-1 with ratings via RatingID). Stores all 25 financial ratios for the corresponding company snapshot. | 2,029 | 26 | [financials.csv](https://drive.google.com/file/d/1x_m3GoGpAiaZ_LcvY4XEzy-Abykm1a5z/view?usp=drive_link) |

### Data Dictionary

One row per feature across all four tables. The `companies`, `agencies`, and `ratings` tables contain only categorical/identifier features; all numeric features are in `financials`.

**companies**

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `Symbol` | VARCHAR | [PK] Stock ticker. 593 unique values. | WHR |
| `Name` | VARCHAR | Full legal company name. | Whirlpool Corporation |
| `Sector` | VARCHAR | Business sector. 12 unique values; largest: Energy 14.5%, Basic Industries 12.8%, Consumer Services 12.3%. | Consumer Durables |

**agencies**

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `AgencyID` | INTEGER | [PK] Synthetic agency identifier (1-5). | 1 |
| `AgencyName` | VARCHAR | Full agency name. | Standard & Poor's Ratings Services |

**ratings**

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `RatingID` | INTEGER | [PK] Synthetic rating-event identifier (1-2029). | 1 |
| `Symbol` | VARCHAR | [FK -> companies.Symbol] Stock ticker of the rated company. | WHR |
| `AgencyID` | INTEGER | [FK -> agencies.AgencyID] Agency that issued the rating. | 1 |
| `Date` | VARCHAR | Date the rating was assigned (M/D/YYYY). Range: 1/1/2014-9/30/2016. | 11/27/2015 |
| `Rating` | VARCHAR | Letter-grade on S&P scale. Values: AAA AA A BBB BB B CCC CC C D. | A |

**financials**

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `RatingID` | INTEGER | [PK/FK -> ratings.RatingID] Links financial snapshot to its rating event. | 1 |
| `currentRatio` | DOUBLE | Current assets / current liabilities. Liquidity; higher is safer. | 0.946 |
| `quickRatio` | DOUBLE | (Current assets - inventory) / current liabilities. Stricter liquidity. | 0.426 |
| `cashRatio` | DOUBLE | Cash / current liabilities. Most conservative liquidity ratio. | 0.100 |
| `daysOfSalesOutstanding` | DOUBLE | Avg receivables / (revenue/365). Days to collect after a sale. | 44.2 |
| `netProfitMargin` | DOUBLE | Net income / revenue. Overall profitability after all costs. | 0.037 |
| `pretaxProfitMargin` | DOUBLE | Pre-tax income / revenue. | 0.049 |
| `grossProfitMargin` | DOUBLE | Gross profit / revenue. Profitability before operating expenses. | 0.177 |
| `operatingProfitMargin` | DOUBLE | Operating income / revenue (EBIT margin). | 0.062 |
| `returnOnAssets` | DOUBLE | Net income / total assets. Efficiency of asset utilization. | 0.041 |
| `returnOnCapitalEmployed` | DOUBLE | EBIT / capital employed. Return on long-term capital. | 0.092 |
| `returnOnEquity` | DOUBLE | Net income / shareholders equity. Return to equity holders. | 0.165 |
| `assetTurnover` | DOUBLE | Revenue / total assets. How efficiently assets generate sales. | 1.099 |
| `fixedAssetTurnover` | DOUBLE | Revenue / net fixed assets. | 5.536 |
| `debtEquityRatio` | DOUBLE | Total debt / shareholders equity. Financial leverage. | 3.008 |
| `debtRatio` | DOUBLE | Total debt / total assets. Fraction of assets financed by debt. | 0.750 |
| `effectiveTaxRate` | DOUBLE | Income tax expense / pre-tax income. | 0.203 |
| `freeCashFlowOperatingCashFlowRatio` | DOUBLE | Free cash flow / operating cash flow. Capital efficiency. | 0.438 |
| `freeCashFlowPerShare` | DOUBLE | Free cash flow divided by shares outstanding. | 6.811 |
| `cashPerShare` | DOUBLE | Cash and equivalents divided by shares outstanding. | 9.809 |
| `companyEquityMultiplier` | DOUBLE | Total assets / shareholders equity. Leverage multiplier. | 4.008 |
| `ebitPerRevenue` | DOUBLE | EBIT / revenue. Operating efficiency. | 0.049 |
| `enterpriseValueMultiple` | DOUBLE | Enterprise value / EBITDA. Valuation relative to earnings. | 7.057 |
| `operatingCashFlowPerShare` | DOUBLE | Operating cash flow per share. | 15.565 |
| `operatingCashFlowSalesRatio` | DOUBLE | Operating cash flow / revenue. Cash generation from operations. | 0.059 |
| `payablesTurnover` | DOUBLE | Cost of goods sold / accounts payable. Speed of paying suppliers. | 3.907 |

### Uncertainty Quantification

Bootstrapped 95% confidence intervals on the column mean (n=1,000 resamples, seed=42) for all 25 numeric features in the `financials` table. The `companies`, `agencies`, and `ratings` tables contain no numeric features requiring uncertainty quantification. Note: several columns contain extreme outliers from non-standard financial reporting; Min/Max values reflect the full observed range.

| Feature | Mean | Std Dev | 95% CI Lower | 95% CI Upper | Min | Max |
|---------|------|---------|--------------|--------------|-----|-----|
| `currentRatio` | 3.5296 | 44.0524 | 2.0889 | 5.7973 | -0.932 | 1725.505 |
| `quickRatio` | 2.654 | 32.9448 | 1.5021 | 4.2398 | -1.893 | 1139.542 |
| `cashRatio` | 0.6674 | 3.5839 | 0.5429 | 0.8598 | -0.193 | 125.917 |
| `daysOfSalesOutstanding` | 333.7956 | 4447.8396 | 162.3912 | 541.6761 | -811.846 | 115961.637 |
| `netProfitMargin` | 0.2784 | 6.0641 | 0.0396 | 0.5537 | -101.846 | 198.518 |
| `pretaxProfitMargin` | 0.4315 | 8.985 | 0.0689 | 0.8557 | -124.344 | 309.695 |
| `grossProfitMargin` | 0.498 | 0.5253 | 0.4751 | 0.5197 | -14.801 | 2.703 |
| `operatingProfitMargin` | 0.5873 | 11.2246 | 0.1746 | 1.1174 | -124.344 | 410.182 |
| `returnOnAssets` | -37.5179 | 1166.1722 | -92.6889 | -0.0206 | -40213.178 | 0.488 |
| `returnOnCapitalEmployed` | -73.9742 | 2350.2757 | -191.0246 | 0.0752 | -87162.162 | 2.44 |
| `returnOnEquity` | 143.4943 | 4406.515 | 0.146 | 355.4654 | -63.815 | 141350.211 |
| `assetTurnover` | 3678.3397 | 95654.1014 | 0.9409 | 7548.1913 | -9.157 | 2553148.615 |
| `fixedAssetTurnover` | 7269.4874 | 188996.6803 | 13.1868 | 16977.51 | -26.798 | 5156883.671 |
| `debtEquityRatio` | 2.3283 | 87.5289 | -1.7913 | 5.9677 | -2556.42 | 2561.872 |
| `debtRatio` | 0.6615 | 0.2089 | 0.6521 | 0.67 | 0.0 | 1.928 |
| `effectiveTaxRate` | 0.3976 | 10.5951 | 0.0367 | 0.9133 | -100.611 | 429.926 |
| `freeCashFlowOperatingCashFlowRatio` | 0.4095 | 3.7965 | 0.2407 | 0.5526 | -120.916 | 34.594 |
| `freeCashFlowPerShare` | 5094.7186 | 146915.6167 | 35.9269 | 12129.9359 | -4912.742 | 5753379.811 |
| `cashPerShare` | 4227.5486 | 122399.9519 | 93.7157 | 10796.0208 | -19.15 | 4786803.378 |
| `companyEquityMultiplier` | 3.3236 | 87.5299 | -0.7619 | 7.139 | -2555.42 | 2562.872 |
| `ebitPerRevenue` | 0.4375 | 8.9843 | 0.1037 | 0.8946 | -124.344 | 309.695 |
| `enterpriseValueMultiple` | 48.288 | 529.119 | 27.2018 | 73.6102 | -3749.921 | 11153.607 |
| `operatingCashFlowPerShare` | 6515.1227 | 177529.0146 | 75.7849 | 16521.9832 | -11950.491 | 6439270.413 |
| `operatingCashFlowSalesRatio` | 1.4477 | 19.4833 | 0.7318 | 2.3467 | -4.462 | 688.527 |
| `payablesTurnover` | 38.0027 | 758.9236 | 8.5479 | 76.2802 | -76.663 | 20314.88 |

---

## Problem Solution Pipeline

| File | Description |
|------|-------------|
| [`pipeline.ipynb`](pipeline.ipynb) | Jupyter notebook — full pipeline: DuckDB data loading, SQL EDA, feature matrix construction via SQL JOIN, model training (LR, RF, GBM, KNN), model comparison, and publication-quality 3-panel visualization |
| [`pipeline.md`](pipeline.md) | Markdown export of the pipeline notebook |
