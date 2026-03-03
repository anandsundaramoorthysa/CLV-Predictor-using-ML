"""
PHASE 4: MODEL TRAINING
Customer Lifetime Value (CLV) Predictor

Models trained (each as a sklearn Pipeline with preprocessing):
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
  4. Extra Trees Regressor
  5. LightGBM Regressor

Preprocessing (inside each pipeline):
  - Categorical features → OrdinalEncoder
  - Numeric features     → StandardScaler (LR) / passthrough (trees)

Evaluation metrics: RMSE, MAE, R² on Train, Validation, Test sets.
Best model selected by lowest Validation RMSE.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: MODEL TRAINING")
print("=" * 80)

input_dir  = Path("outputs")
output_dir = Path("outputs")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading train/validation/test splits...")

X_train = pd.read_csv(input_dir / 'X_train.csv')
y_train = pd.read_csv(input_dir / 'y_train.csv').iloc[:, 0]
X_val   = pd.read_csv(input_dir / 'X_val.csv')
y_val   = pd.read_csv(input_dir / 'y_val.csv').iloc[:, 0]
X_test  = pd.read_csv(input_dir / 'X_test.csv')
y_test  = pd.read_csv(input_dir / 'y_test.csv').iloc[:, 0]

with open(input_dir / 'feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open(input_dir / 'numeric_columns.pkl', 'rb') as f:
    numeric_cols = pickle.load(f)
with open(input_dir / 'categorical_columns.pkl', 'rb') as f:
    cat_cols = pickle.load(f)

print(f"  Train:      {X_train.shape}  | target mean=${y_train.mean():.2f}")
print(f"  Validation: {X_val.shape}    | target mean=${y_val.mean():.2f}")
print(f"  Test:       {X_test.shape}   | target mean=${y_test.mean():.2f}")
print(f"  Numeric features:     {numeric_cols}")
print(f"  Categorical features: {cat_cols}")

y_train_raw, y_val_raw, y_test_raw = y_train.copy(), y_val.copy(), y_test.copy()


def evaluate_model(y_true_raw, y_pred, model_name, dataset_name):
    """Compute RMSE, MAE, R² in dollar-space."""
    y_pred = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true_raw, y_pred)))
    mae  = float(mean_absolute_error(y_true_raw, y_pred))
    r2   = float(r2_score(y_true_raw, y_pred))
    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE: ${rmse:.2f}  MAE: ${mae:.2f}  R²: {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# ============================================================================
# PREPROCESSORS
# ============================================================================
print("\n[2] Building preprocessors...")

tree_preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', numeric_cols)
], remainder='drop')

linear_preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', StandardScaler(), numeric_cols)
], remainder='drop')

print("  ✓ Preprocessors built")

# Feature names after encoding: cat_cols first, then numeric_cols
feature_names_encoded = cat_cols + numeric_cols

# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("[3] Training Linear Regression...")
print("=" * 80)

lr_pipeline = Pipeline([
    ('preprocessor', linear_preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

lr_train = evaluate_model(y_train_raw, lr_pipeline.predict(X_train), "Linear Regression", "Train")
lr_val   = evaluate_model(y_val_raw,   lr_pipeline.predict(X_val),   "Linear Regression", "Validation")
lr_test  = evaluate_model(y_test_raw,  lr_pipeline.predict(X_test),  "Linear Regression", "Test")

with open(models_dir / 'linear_regression_pipeline.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)
print("\n  ✓ Saved → models/linear_regression_pipeline.pkl")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("[4] Training Random Forest Regressor...")
print("=" * 80)

rf_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)

rf_train = evaluate_model(y_train_raw, rf_pipeline.predict(X_train), "Random Forest", "Train")
rf_val   = evaluate_model(y_val_raw,   rf_pipeline.predict(X_val),   "Random Forest", "Validation")
rf_test  = evaluate_model(y_test_raw,  rf_pipeline.predict(X_test),  "Random Forest", "Test")

with open(models_dir / 'random_forest_pipeline.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)
print("\n  ✓ Saved → models/random_forest_pipeline.pkl")

# ============================================================================
# MODEL 3: GRADIENT BOOSTING
# ============================================================================
print("\n" + "=" * 80)
print("[5] Training Gradient Boosting Regressor...")
print("=" * 80)

gbr_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    ))
])
gbr_pipeline.fit(X_train, y_train)

gbr_train = evaluate_model(y_train_raw, gbr_pipeline.predict(X_train), "Gradient Boosting", "Train")
gbr_val   = evaluate_model(y_val_raw,   gbr_pipeline.predict(X_val),   "Gradient Boosting", "Validation")
gbr_test  = evaluate_model(y_test_raw,  gbr_pipeline.predict(X_test),  "Gradient Boosting", "Test")

with open(models_dir / 'gradient_boosting_pipeline.pkl', 'wb') as f:
    pickle.dump(gbr_pipeline, f)
print("\n  ✓ Saved → models/gradient_boosting_pipeline.pkl")

# ============================================================================
# MODEL 4: EXTRA TREES
# ============================================================================
print("\n" + "=" * 80)
print("[6] Training Extra Trees Regressor...")
print("=" * 80)

et_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])
et_pipeline.fit(X_train, y_train)

et_train = evaluate_model(y_train_raw, et_pipeline.predict(X_train), "Extra Trees", "Train")
et_val   = evaluate_model(y_val_raw,   et_pipeline.predict(X_val),   "Extra Trees", "Validation")
et_test  = evaluate_model(y_test_raw,  et_pipeline.predict(X_test),  "Extra Trees", "Test")

with open(models_dir / 'extra_trees_pipeline.pkl', 'wb') as f:
    pickle.dump(et_pipeline, f)
print("\n  ✓ Saved → models/extra_trees_pipeline.pkl")

# ============================================================================
# MODEL 5: LIGHTGBM
# ============================================================================
print("\n" + "=" * 80)
print("[7] Training LightGBM Regressor...")
print("=" * 80)

lgbm_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=63,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
])
lgbm_pipeline.fit(X_train, y_train)

lgbm_train = evaluate_model(y_train_raw, lgbm_pipeline.predict(X_train), "LightGBM", "Train")
lgbm_val   = evaluate_model(y_val_raw,   lgbm_pipeline.predict(X_val),   "LightGBM", "Validation")
lgbm_test  = evaluate_model(y_test_raw,  lgbm_pipeline.predict(X_test),  "LightGBM", "Test")

with open(models_dir / 'lightgbm_pipeline.pkl', 'wb') as f:
    pickle.dump(lgbm_pipeline, f)
print("\n  ✓ Saved → models/lightgbm_pipeline.pkl")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[8] MODEL COMPARISON (Best Model by Validation RMSE)")
print("=" * 80)

val_rmse_map = {
    'Linear Regression': lr_val['rmse'],
    'Random Forest':     rf_val['rmse'],
    'Gradient Boosting': gbr_val['rmse'],
    'Extra Trees':       et_val['rmse'],
    'LightGBM':          lgbm_val['rmse'],
}
sorted_models = sorted(val_rmse_map.items(), key=lambda x: x[1])

print("\n  Ranked by Validation RMSE (lowest = best):")
for i, (name, rmse) in enumerate(sorted_models, 1):
    print(f"    {i}. {name}: {rmse:.4f}")

best_model_name    = sorted_models[0][0]
best_model_val_rmse = sorted_models[0][1]
print(f"\n  Best Model: {best_model_name}  (Val RMSE: {best_model_val_rmse:.4f})")

# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================
print("\n[9] Saving evaluation results...")

results_df = pd.DataFrame({
    'Model':      ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'LightGBM'],
    'Train_RMSE': [lr_train['rmse'],  rf_train['rmse'],  gbr_train['rmse'],  et_train['rmse'],  lgbm_train['rmse']],
    'Train_MAE':  [lr_train['mae'],   rf_train['mae'],   gbr_train['mae'],   et_train['mae'],   lgbm_train['mae']],
    'Train_R2':   [lr_train['r2'],    rf_train['r2'],    gbr_train['r2'],    et_train['r2'],    lgbm_train['r2']],
    'Val_RMSE':   [lr_val['rmse'],    rf_val['rmse'],    gbr_val['rmse'],    et_val['rmse'],    lgbm_val['rmse']],
    'Val_MAE':    [lr_val['mae'],     rf_val['mae'],     gbr_val['mae'],     et_val['mae'],     lgbm_val['mae']],
    'Val_R2':     [lr_val['r2'],      rf_val['r2'],      gbr_val['r2'],      et_val['r2'],      lgbm_val['r2']],
    'Test_RMSE':  [lr_test['rmse'],   rf_test['rmse'],   gbr_test['rmse'],   et_test['rmse'],   lgbm_test['rmse']],
    'Test_MAE':   [lr_test['mae'],    rf_test['mae'],    gbr_test['mae'],    et_test['mae'],    lgbm_test['mae']],
    'Test_R2':    [lr_test['r2'],     rf_test['r2'],     gbr_test['r2'],     et_test['r2'],     lgbm_test['r2']],
})
results_df = results_df.sort_values('Val_RMSE').reset_index(drop=True)
results_df.to_csv(output_dir / 'model_evaluation_results.csv', index=False)
print("  ✓ Saved → outputs/model_evaluation_results.csv")

with open(output_dir / 'best_model.json', 'w') as f:
    json.dump({
        'best_model':    best_model_name,
        'best_val_rmse': round(best_model_val_rmse, 4),
        'best_val_mae':  round(val_rmse_map[best_model_name], 4),
    }, f, indent=2)
print("  ✓ Saved → outputs/best_model.json")

# ============================================================================
# FEATURE IMPORTANCE (tree-based models)
# ============================================================================
print("\n[10] Computing feature importance...")

tree_pipelines = {
    'Random Forest':     rf_pipeline,
    'Gradient Boosting': gbr_pipeline,
    'Extra Trees':       et_pipeline,
    'LightGBM':          lgbm_pipeline,
}

feature_importance_all = {}
for model_name, pipeline in tree_pipelines.items():
    estimator = pipeline.named_steps['model']
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        total = importances.sum()
        fi = sorted(
            zip(feature_names_encoded, importances),
            key=lambda x: x[1], reverse=True
        )
        feature_importance_all[model_name] = [
            {
                'feature':    str(feat),
                'importance': round(float(val), 4),
                'pct':        round(float(val / total * 100), 1) if total > 0 else 0.0
            }
            for feat, val in fi
        ]
        top = fi[0]
        print(f"  ✓ {model_name}: top feature = '{top[0]}' ({top[1]:.4f})")

with open(output_dir / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance_all, f, indent=2)
print("  ✓ Saved → outputs/feature_importance.json")

# ============================================================================
# SAVE TEST SET PREDICTIONS (sample of 50 for about page)
# ============================================================================
print("\n[11] Saving test set predictions (sample)...")

all_pipelines = {
    'Linear Regression': lr_pipeline,
    'Random Forest':     rf_pipeline,
    'Gradient Boosting': gbr_pipeline,
    'Extra Trees':       et_pipeline,
    'LightGBM':          lgbm_pipeline,
}

sample_size       = min(50, len(X_test))
X_test_sample     = X_test.sample(n=sample_size, random_state=42)
y_test_raw_sample = y_test_raw.loc[X_test_sample.index]

test_pred_df = pd.DataFrame({'Actual': [round(float(v), 2) for v in y_test_raw_sample.values]})
for model_name, pipeline in all_pipelines.items():
    preds = np.clip(pipeline.predict(X_test_sample), 0, None)
    test_pred_df[model_name] = [round(float(p), 2) for p in preds]

test_pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
print(f"  ✓ Saved {sample_size} test predictions → outputs/test_predictions.csv")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE!")
print(f"  Best Model: {best_model_name}  (Val RMSE: {best_model_val_rmse:.4f})")
print("=" * 80)
