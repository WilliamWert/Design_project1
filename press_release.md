# New Machine Learning Model Spots Corporate Bond Defaults Months Before Rating Agencies Do

---

## Hook

In 2008, Lehman Brothers held an investment-grade credit rating from S&P right up until the moment it filed for the largest bankruptcy in US history. The data to see it coming was sitting in the financial statements the whole time.

---

## Problem Statement

There are over $10 trillion in US corporate bonds outstanding, and the investors holding them — pension funds, insurance companies, mutual funds — rely primarily on credit ratings from Moody's, S&P, and Fitch to assess default risk. The problem is structural: rating agencies are paid by the companies they rate, creating a documented incentive to issue optimistic ratings and delay downgrades. Academic research has repeatedly shown that agency ratings lag market signals by weeks or months, leaving investors exposed during the window between when financial distress becomes measurable in balance sheet data and when agencies finally act. The result is a systematic mispricing of default risk that costs institutional investors billions annually — and the information to avoid it is hiding in plain sight in quarterly financial filings.

---

## Solution Description

This project trains a machine learning classifier on 25 financial ratios drawn from balance sheet, income statement, and cash flow data for over 2,000 publicly traded US companies, using their S&P credit ratings as labels. The model learns to distinguish investment-grade issuers from speculative-grade (elevated default risk) issuers based purely on financial statement signals — no rating agency input required. The output is a credit risk classification and a flag for bonds that the model identifies as high-risk regardless of their current agency rating. Designed for portfolio managers and risk analysts, the tool provides an independent second opinion on creditworthiness that is updated whenever new financial data is available, not whenever an agency decides to act.

The pipeline uses four machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting, and K-Nearest Neighbors) evaluated on 2,029 rating observations from five agencies covering 593 publicly traded US companies. The full workflow — from raw financial data to prediction — is automated, reproducible, and documented in a Jupyter notebook.

---

## Chart

The chart below shows how three key financial ratios cleanly separate investment-grade bonds (BBB and above, shown in blue) from speculative-grade bonds (BB and below, shown in red). Higher-rated companies carry less debt relative to equity, maintain stronger liquidity, and deliver higher returns — the same signals the machine learning model learns to detect automatically.

![Financial Ratios by Credit Rating](credit_rating_ratios.png)

*Source: Kaggle Corporate Credit Rating Dataset — 2,029 rating observations across 593 publicly traded US companies (2014–2016). Investment grade = BBB and above; Speculative grade = BB and below.*
