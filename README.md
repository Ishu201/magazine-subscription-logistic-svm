# Magazine Subscription Decline Analysis: Logistic Regression vs SVM

Predicting customer subscription behavior using logistic regression and
Support Vector Machine (SVM) classifiers on a marketing campaign dataset.
Built to help a media company identify high-probability subscription
targets and reverse declining subscription rates.

Built as part of ALY 6020: Predictive Analytics at Northeastern University.

---

## Business Context

A media company experienced declining magazine subscriptions during its
most recent campaign cycle. This analysis builds two classification
models to identify which customer behaviors and demographics predict
subscription, and recommends a targeting strategy for future campaigns.

---

## Dataset

**Source:** Marketing campaign dataset (UCI / Kaggle)
**Records:** 2,240 customer records
**Features:** 29 variables covering demographics, spending across 6
product categories, purchase frequency across 4 channels, web
engagement, recency, and responses to 5 prior campaigns
**Target:** Response (1 = subscribed, 0 = did not subscribe)
**Class distribution:** 85.1% non-subscriber, 14.9% subscriber

> The dataset is not included in this repository. Place
> `marketing_campaign.csv` in a `/data` folder before running the
> notebook.

---

## Methodology

### Part 1: Data Cleansing
- Median imputation for 24 missing Income values (1.07% of records)
- Removed 4 irrelevant columns: ID, Dt_Customer, Z_CostContact,
  Z_Revenue
- Ordinal encoding for Education (Basic=0 to PhD=4)
- Merged rare Marital_Status categories (Alone, Absurd, YOLO) into
  Single, then one-hot encoded with drop_first=True
- IQR-based Winsorization on all numeric columns for outlier treatment
- class_weight='balanced' and stratify=y applied to handle 85/15 class
  imbalance
- StandardScaler normalization fit on training set only to prevent data
  leakage
- 80/20 stratified train/test split (random_state=42)

### Part 2: Logistic Regression
- max_iter=1000, class_weight='balanced', random_state=42
- Coefficients extracted directly for feature importance

### Part 3: SVM
- RBF kernel, C=1.0, gamma='scale', class_weight='balanced'
- Permutation importance used for feature contribution (RBF kernel
  produces no direct coefficients)

---

## Results

| Metric | Logistic Regression | SVM |
|---|---|---|
| Accuracy | 76.6% | 80.6% |
| AUC | 0.874 | 0.876 |
| Precision (Class 1) | 0.372 | 0.417 |
| Recall (Class 1) | 0.821 | 0.746 |
| F1 Score (Class 1) | 0.512 | 0.535 |

**Recommendation:** SVM is the primary model for high-cost outreach
campaigns (higher precision, fewer wasted contacts). Logistic regression
is preferred when recall is the priority or when business stakeholders
need coefficient-level interpretability.

---

## Key Findings

Both models agreed on the same core predictors, providing strong
confidence in these drivers:

**Positive drivers of subscription:** web visit frequency
(NumWebVisitsMonth), catalog purchase behavior (NumCatalogPurchases),
wine spending (MntWines), and income

**Negative drivers:** in-store purchase preference
(NumStorePurchases), low recency (disengaged customers), presence of
teenagers in the household, and partnered/married marital status

**Ideal target profile:** recently active, high-income, web-engaged
customers who purchase through catalogs and do not have teenagers at
home

---

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## Files

| File | Description |
|---|---|
| `magazine_subscription_logistic_svm.ipynb` | Full analysis notebook |
| `Isuri_Module3_SVM_Report.pdf` | Written report with interpretation |

---

## Author

**Isuri Nawodya**  
Master of Professional Studies in Analytics, Northeastern University Vancouver  
