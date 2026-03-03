"""
PHASE 1: DATA PREPROCESSING
Customer Lifetime Value (CLV) Predictor

Steps:
- Load and inspect dataset
- Select relevant columns
- Convert data types
- Handle missing values
- Validate data
- Save clean dataset and statistics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("PHASE 1: DATA PREPROCESSING")
print("=" * 80)

data_file  = Path("new_retail_data.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

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

# ============================================================================
# STEP 1: LOAD & INSPECT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING & INSPECTING DATASET")
print("=" * 80)

df = pd.read_csv(data_file)
print(f"\n  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Data types:\n{df.dtypes.to_string()}")
print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
print(f"\n  Sample (first 3 rows):\n{df.head(3).to_string()}")

# ============================================================================
# STEP 2: COLUMN SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: COLUMN SELECTION")
print("=" * 80)

all_needed = FEATURE_COLS + [TARGET_COL]
missing_cols = [c for c in all_needed if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df_model = df[all_needed].copy()
print(f"\n  Selected {len(all_needed)} columns: {all_needed}")
print(f"  Shape after selection: {df_model.shape}")

# ============================================================================
# STEP 3: DATA TYPE CONVERSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA TYPE CONVERSION")
print("=" * 80)

for c in NUMERIC_COLS + [TARGET_COL]:
    df_model[c] = pd.to_numeric(df_model[c], errors='coerce')
    print(f"  {c}: converted to numeric")

for c in CAT_COLS:
    df_model[c] = df_model[c].astype(str).str.strip()
    df_model[c] = df_model[c].replace('nan', np.nan)   # re-expose NaN for dropna()
    print(f"  {c}: cleaned string")

# ============================================================================
# STEP 4: MISSING VALUE HANDLING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MISSING VALUE HANDLING")
print("=" * 80)

before_rows = len(df_model)
df_model = df_model.dropna().reset_index(drop=True)
after_rows  = len(df_model)
print(f"\n  Rows before:  {before_rows:,}")
print(f"  Rows dropped: {before_rows - after_rows:,}")
print(f"  Rows after:   {after_rows:,}")

# ============================================================================
# STEP 5: VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: VALIDATION")
print("=" * 80)

# Check target values
neg_target = (df_model[TARGET_COL] < 0).sum()
if neg_target > 0:
    print(f"  WARNING: {neg_target} negative target values, removing...")
    df_model = df_model[df_model[TARGET_COL] >= 0].reset_index(drop=True)
else:
    print(f"  PASSED: No negative target values")

# Check age range
invalid_age = ((df_model['Age'] < 1) | (df_model['Age'] > 120)).sum()
if invalid_age > 0:
    print(f"  WARNING: {invalid_age} invalid age values, removing...")
    df_model = df_model[(df_model['Age'] >= 1) & (df_model['Age'] <= 120)].reset_index(drop=True)
else:
    print(f"  PASSED: All age values valid")

# Check Total_Purchases
neg_purchases = (df_model['Total_Purchases'] < 0).sum()
if neg_purchases > 0:
    print(f"  WARNING: {neg_purchases} negative purchase counts, removing...")
    df_model = df_model[df_model['Total_Purchases'] >= 0].reset_index(drop=True)
else:
    print(f"  PASSED: All purchase counts non-negative")

print(f"\n  Categorical value summary:")
for c in CAT_COLS:
    vals = df_model[c].unique().tolist()
    print(f"    {c}: {vals}")

print(f"\n  Final dataset shape: {df_model.shape}")
print(f"\n  Target ({TARGET_COL}) distribution:")
print(f"    Min:    ${df_model[TARGET_COL].min():.2f}")
print(f"    Max:    ${df_model[TARGET_COL].max():.2f}")
print(f"    Mean:   ${df_model[TARGET_COL].mean():.2f}")
print(f"    Median: ${df_model[TARGET_COL].median():.2f}")
print(f"    Std:    ${df_model[TARGET_COL].std():.2f}")
print(f"    P25:    ${df_model[TARGET_COL].quantile(0.25):.2f}")
print(f"    P75:    ${df_model[TARGET_COL].quantile(0.75):.2f}")

# ============================================================================
# STEP 6: SAVE DATA STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: SAVING DATA STATISTICS")
print("=" * 80)

category_values = {}
for c in CAT_COLS:
    vals = sorted(df_model[c].dropna().unique().tolist())
    category_values[c] = vals

data_stats = {
    'total_rows':             int(len(df_model)),
    'target_mean':            round(float(df_model[TARGET_COL].mean()),            2),
    'target_std':             round(float(df_model[TARGET_COL].std()),             2),
    'target_min':             round(float(df_model[TARGET_COL].min()),             2),
    'target_max':             round(float(df_model[TARGET_COL].max()),             2),
    'target_p25':             round(float(df_model[TARGET_COL].quantile(0.25)),    2),
    'target_p50':             round(float(df_model[TARGET_COL].quantile(0.50)),    2),
    'target_p75':             round(float(df_model[TARGET_COL].quantile(0.75)),    2),
    'source':                 str(data_file),
    'features':               FEATURE_COLS,
    'target':                 TARGET_COL,
    'numeric_features':       NUMERIC_COLS,
    'categorical_features':   CAT_COLS,
    'category_values':        category_values,
}

with open(output_dir / 'data_stats.json', 'w') as f:
    json.dump(data_stats, f, indent=2)
print(f"\n  ✓ Statistics saved  → outputs/data_stats.json")

# ============================================================================
# STEP 7: SAVE CLEAN DATASET
# ============================================================================
df_model.to_csv(output_dir / 'clean_df.csv', index=False)
print(f"  ✓ Clean dataset saved → outputs/clean_df.csv")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE!")
print("=" * 80)
