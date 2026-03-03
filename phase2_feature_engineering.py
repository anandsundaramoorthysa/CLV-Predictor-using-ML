"""
PHASE 2: FEATURE ENGINEERING
Customer Lifetime Value (CLV) Predictor

Steps:
- Compute Pearson & Spearman correlations between features and target
- Select and save feature columns (raw, encoding done in model pipeline)
- Save column metadata (numeric/categorical lists)
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from scipy import stats as scipy_stats

print("=" * 80)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 80)

input_dir  = Path("outputs")
output_dir = Path("outputs")

FEATURE_COLS = [
    'Age', 'Gender', 'Income',
    'Total_Purchases', 'Amount',
    'Product_Category', 'Payment_Method', 'Shipping_Method',
    'Customer_Segment', 'Ratings', 'Feedback',
    'Month', 'Product_Brand', 'Product_Type'
]
TARGET_COL   = 'Total_Amount'
NUMERIC_COLS = ['Age', 'Total_Purchases', 'Amount', 'Ratings']
CAT_COLS     = [
    'Gender', 'Income', 'Product_Category', 'Payment_Method', 'Shipping_Method',
    'Customer_Segment', 'Feedback', 'Month', 'Product_Brand', 'Product_Type'
]

print("\n[1] Loading clean dataset...")
df = pd.read_csv(input_dir / 'clean_df.csv')
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 1: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("[2] Computing feature correlations with target...")
print("=" * 80)

# Label-encode categoricals for correlation computation only
df_enc = df.copy()
for col in CAT_COLS:
    df_enc[col] = df_enc[col].astype('category').cat.codes

correlation_data = []
for feat in FEATURE_COLS:
    if feat not in df_enc.columns:
        continue

    x = df_enc[feat].values.astype(float)
    y = df_enc[TARGET_COL].values.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    pearson_r,  _ = scipy_stats.pearsonr(x, y)
    spearman_r, _ = scipy_stats.spearmanr(x, y)

    abs_p = abs(pearson_r)
    if abs_p >= 0.7:
        strength = 'Strong'
    elif abs_p >= 0.4:
        strength = 'Moderate'
    elif abs_p >= 0.2:
        strength = 'Weak'
    else:
        strength = 'Very Weak'

    direction = 'Positive' if pearson_r >= 0 else 'Negative'

    correlation_data.append({
        'feature':        feat,
        'pearson':        round(float(pearson_r),  4),
        'spearman':       round(float(spearman_r), 4),
        'interpretation': f'{strength} {direction}'
    })

correlation_data.sort(key=lambda x: abs(x['pearson']), reverse=True)

print(f"\n  Correlation results (sorted by |Pearson|):")
for row in correlation_data:
    print(f"    {row['feature']:25s} Pearson={row['pearson']:+.4f}  Spearman={row['spearman']:+.4f}  ({row['interpretation']})")

with open(output_dir / 'correlation_data.json', 'w') as f:
    json.dump(correlation_data, f, indent=2)
print(f"\n  ✓ Correlation data saved → outputs/correlation_data.json")

# ============================================================================
# STEP 2: SELECT FEATURES (raw, encoding happens inside model pipeline)
# ============================================================================
print("\n" + "=" * 80)
print("[3] Selecting and saving feature columns...")
print("=" * 80)

features_df = df[FEATURE_COLS + [TARGET_COL]].copy()
print(f"\n  Feature columns: {FEATURE_COLS}")
print(f"  Target column:   {TARGET_COL}")
print(f"  Final shape:     {features_df.shape}")
print(f"  Missing values:  {features_df.isnull().sum().sum()}")

features_df.to_csv(output_dir / 'features_df.csv', index=False)
print(f"\n  ✓ Features saved          → outputs/features_df.csv")

with open(output_dir / 'feature_columns.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
print(f"  ✓ Feature columns saved   → outputs/feature_columns.pkl")

with open(output_dir / 'numeric_columns.pkl', 'wb') as f:
    pickle.dump(NUMERIC_COLS, f)
print(f"  ✓ Numeric columns saved   → outputs/numeric_columns.pkl")

with open(output_dir / 'categorical_columns.pkl', 'wb') as f:
    pickle.dump(CAT_COLS, f)
print(f"  ✓ Categorical columns saved → outputs/categorical_columns.pkl")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE!")
print(f"  {len(FEATURE_COLS)} features selected: {len(NUMERIC_COLS)} numeric + {len(CAT_COLS)} categorical")
print("=" * 80)
