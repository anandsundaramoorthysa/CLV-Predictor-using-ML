"""
PHASE 3: TRAIN / VALIDATION / TEST SPLIT
Customer Lifetime Value (CLV) Predictor

Split: 70% Train | 15% Validation | 15% Test (random, seed=42)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 80)
print("PHASE 3: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

input_dir  = Path("outputs")
output_dir = Path("outputs")
TARGET_COL = 'Total_Amount'

print("\n[1] Loading features dataset...")
features_df = pd.read_csv(input_dir / 'features_df.csv')
print(f"  Shape: {features_df.shape}")

with open(input_dir / 'feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
print(f"  Feature columns: {feature_cols}")

# ============================================================================
# SPLIT: 70 / 15 / 15
# ============================================================================
print("\n[2] Creating Train / Validation / Test splits (70/15/15)...")

X = features_df[feature_cols].copy()
y = features_df[TARGET_COL].copy()

# Step 1: 70% train, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 2: 15% val, 15% test from the remaining 30%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

n_total = len(X)
print(f"\n  Total samples:  {n_total:,}")
print(f"  Train:          {len(X_train):,} rows  ({len(X_train)/n_total*100:.1f}%)")
print(f"  Validation:     {len(X_val):,} rows   ({len(X_val)/n_total*100:.1f}%)")
print(f"  Test:           {len(X_test):,} rows   ({len(X_test)/n_total*100:.1f}%)")

print(f"\n  Train target:      min=${y_train.min():.2f}  max=${y_train.max():.2f}  mean=${y_train.mean():.2f}")
print(f"  Validation target: min=${y_val.min():.2f}  max=${y_val.max():.2f}  mean=${y_val.mean():.2f}")
print(f"  Test target:       min=${y_test.min():.2f}  max=${y_test.max():.2f}  mean=${y_test.mean():.2f}")

# ============================================================================
# SAVE SPLITS
# ============================================================================
print("\n[3] Saving splits to outputs/...")

X_train.to_csv(output_dir / 'X_train.csv', index=False)
y_train.to_csv(output_dir / 'y_train.csv', index=False)
X_val.to_csv(  output_dir / 'X_val.csv',   index=False)
y_val.to_csv(  output_dir / 'y_val.csv',   index=False)
X_test.to_csv( output_dir / 'X_test.csv',  index=False)
y_test.to_csv( output_dir / 'y_test.csv',  index=False)

print("  ✓ X_train.csv, y_train.csv saved")
print("  ✓ X_val.csv,   y_val.csv   saved")
print("  ✓ X_test.csv,  y_test.csv  saved")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE!")
print("=" * 80)
