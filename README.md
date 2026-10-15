# Credit Card Fraud Detection — Cost-Aware Machine Learning

## Problem
A bank processes ~285k transactions across 2 days. Only 0.17% are fraud. Manual review capacity is limited. Build a model that flags risky transactions while minimizing expected cost (missed fraud × $ loss + false positives × review cost).

## Data
- **Source:** Kaggle ULB Credit Card Fraud Detection
- **Raw:** 284,807 transactions, 492 frauds (0.172%)
- **After dedup:** 283,726 transactions, 473 frauds (0.167%)

![Class imbalance — fraud is only 0.17% of transactions](outputs/01_class_imbalance.png)

## Results

| Model | Recall | False Positives | Expected Cost |
|---|---|---|---|
| Logistic @ 0.5 | 89.7% | 2,915 | $16,775 |
| **XGBoost (optimal)** | **82.2%** | **151** | **$4,555** |
| **Δ** | −7.5 pts | **−95%** | **−72.8%** |

The cost curve below shows why threshold selection matters more than model choice: the optimal operating point cuts expected cost by **73%** versus the logistic baseline.

![Expected cost vs decision threshold](outputs/05_cost_threshold.png)

## Explainability

Every alert is accompanied by a SHAP explanation so analysts can justify decisions.

![SHAP summary — global feature impact](outputs/08_shap_beeswarm.png)

**Example — single alert explanation (highest-risk fraud in the test set):**

![SHAP force plot for one fraud alert](outputs/09_shap_single_alert.png)

## Approach
1. **EDA** — confirmed extreme class imbalance, analyzed amount & hour patterns
2. **Cleaning** — removed 1,081 duplicates (19 fraud rows — documented as a judgment call)
3. **Feature engineering**
   - `Is_night` (11pm–5am flag) — night fraud rate was **2.7× higher** (0.38% vs 0.14%)
   - `Amount_deviation` (per-hour z-score of amount)
   - `Txn_count_last_hour` (velocity proxy)
4. **Validation** — time-based 70/30 split (fraud patterns drift)
5. **Models** — Logistic Regression (baseline) vs XGBoost
6. **Threshold tuning** — cost-based ($200 per missed fraud, $5 per review)
7. **Explainability** — SHAP for global + per-alert explanation

## Results

| Model | Recall | False Positives | Expected Cost |
|---|---|---|---|
| Logistic @ 0.5 | 89.7% | 2,915 | $16,775 |
| **XGBoost (optimal)** | **82.2%** | **151** | **$4,555** |
| **Δ** | −7.5 pts | **−95%** | **−72.8%** |

**Operating point:** At 200 alerts/day, the model catches **81% of fraud at 44% precision** — realistic for a fraud ops team.

## Top SHAP Features
V14, V4, V12, V10 (PCA features), plus **`Amount_scaled`** and **`Amount_deviation`** (engineered).

## Business Impact
- **72.8% reduction** in expected fraud cost vs logistic baseline
- **95% fewer false positives** → dramatically less analyst friction
- **Explainable alerts** via SHAP → analysts can justify decisions

## Limitations
- Only 2 days of data — no customer IDs, no seasonality
- Velocity is a weak proxy (no per-card history)
- `scale_pos_weight` distorts raw probabilities (cost-based threshold corrects this)
- 19 duplicate frauds dropped (could represent repeated attacks, not data errors)

## Next Steps
- Per-customer rolling features (1h, 24h, 7d windows)
- Probability calibration (Platt scaling)
- Real-time scoring API
- Drift monitoring and retraining cadence

## Repo Structure