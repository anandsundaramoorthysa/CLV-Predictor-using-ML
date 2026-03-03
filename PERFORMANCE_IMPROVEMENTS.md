# Model Performance Improvements

## What Was Done

### 1. **Amount + Tenure + Recency added (form and model)**
- **Amount** (order/product value $): Restored in Phase 1/2 and added to the app form. Strong correlation with `Total_Amount` (Pearson ~0.67).
- **Tenure_months**: Computed in Phase 1 from `Date` and `Customer_ID` as (max Date − min Date) per customer in months; added to features and form.
- **Recency_months**: Months since last purchase per customer (from Date); added to features and form.

### 2. **Phase 1: Preprocessing**
- Loads `Date` and `Customer_ID` to compute Tenure_months and Recency_months, then drops them.
- Target winsorization at 1st and 99th percentiles.
- **16 features** (6 numeric: Age, Total_Purchases, Amount, Ratings, Tenure_months, Recency_months; 10 categorical).

### 3. **Phase 4: Feature engineering (inside pipeline)**
- **Age_group**, **Total_Purchases_sq**, **Purchases_x_Age** (unchanged).

### 4. **Result**
- **Best model**: Gradient Boosting (by validation RMSE).
- **Validation**: RMSE ≈ $114, **R² ≈ 0.99** (Gradient Boosting); Linear Regression **Val R² ≈ 0.90** (above 70%).
- **Test**: R² ≈ 0.90 (LR) to 0.99 (GBR). Three models only: Linear Regression, Random Forest, Gradient Boosting.

---

## How to Re-run the Full Pipeline

```bash
python phase1_data_preprocessing.py
python phase2_feature_engineering.py
python phase3_train_test_split.py
python phase4_model_training.py
python generate_analysis_plots.py
```

Then restart the Flask app; it will load the new models and `outputs/`.
