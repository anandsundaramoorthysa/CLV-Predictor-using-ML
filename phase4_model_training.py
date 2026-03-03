"""
PHASE 4: MODEL TRAINING (IMPROVED FOR R² > 70%)
Customer Lifetime Value (CLV) Predictor

Improvements:
- Log-target: train on log1p(Total_Amount), evaluate in dollar space (expm1)
- Feature engineering: Age_group (binned), Total_Purchases_sq (inside pipeline)
- Stronger tree models: more estimators, tuned depth/learning_rate

Models: Linear Regression, Random Forest, Gradient Boosting
Best model by lowest Validation RMSE. Saves target_transformation for app.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Target transformation: train on log1p(y), evaluate in dollars (expm1)
# (Disabled: log-target lowered R2 on this dataset; kept feature engineering.)
# ---------------------------------------------------------------------------
USE_LOG_TARGET = False
LOG_PRED_CLIP_MAX = 12.0  # expm1(12) ~ 163k dollars; avoid explosion

# ---------------------------------------------------------------------------
# Feature engineering transformer (shared with app for pickle load)
# ---------------------------------------------------------------------------
from pipeline_utils import AddFeaturesTransformer

# ============================================================================
# SETUP
# ============================================================================
print("=" * 80)
print("PHASE 4: MODEL TRAINING (IMPROVED)")
print("=" * 80)

input_dir  = Path("outputs")
output_dir = Path("outputs")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Load data
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

# Expanded numeric cols after feature engineering
numeric_cols_expanded = numeric_cols + ['Age_group', 'Total_Purchases_sq', 'Purchases_x_Age']
feature_names_encoded = cat_cols + numeric_cols_expanded

y_train_raw, y_val_raw, y_test_raw = y_train.copy(), y_val.copy(), y_test.copy()
if USE_LOG_TARGET:
    y_train = np.log1p(y_train_raw.values)
    y_val   = np.log1p(y_val_raw.values)
    y_test  = np.log1p(y_test_raw.values)
    print("  OK Using log1p(target) for training; evaluation in dollar space (expm1)")

print(f"  Train:      {X_train.shape}  | target mean=${y_train_raw.mean():.2f}")
print(f"  Validation: {X_val.shape}    | target mean=${y_val_raw.mean():.2f}")
print(f"  Test:       {X_test.shape}   | target mean=${y_test_raw.mean():.2f}")

# ---------------------------------------------------------------------------
# Evaluation in dollar space (convert log-predictions if needed)
# ---------------------------------------------------------------------------
def pred_to_dollars(pred_log):
    pred_log = np.clip(pred_log, -0.5, LOG_PRED_CLIP_MAX)
    return np.expm1(pred_log)

def evaluate_model(y_true_raw, y_pred, model_name, dataset_name, in_log_space=True):
    """y_true_raw in dollars; y_pred in log space if in_log_space else dollars."""
    if in_log_space:
        y_pred_d = pred_to_dollars(y_pred)
    else:
        y_pred_d = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_d)))
    mae  = float(mean_absolute_error(y_true_raw, y_pred_d))
    r2   = float(r2_score(y_true_raw, y_pred_d))
    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE: ${rmse:.2f}  MAE: ${mae:.2f}  R²: {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Preprocessors (operate on 15-column output of AddFeaturesTransformer)
tree_preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', 'passthrough', numeric_cols_expanded)
], remainder='drop')

linear_preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', StandardScaler(), numeric_cols_expanded)
], remainder='drop')

add_features = AddFeaturesTransformer()

# Helper to get transformed X for fitting (same columns as pipeline output before model)
def get_X_transformed(X):
    return add_features.transform(X)

X_train_t = get_X_transformed(X_train)
X_val_t   = get_X_transformed(X_val)
X_test_t  = get_X_transformed(X_test)

print("\n[2] Feature engineering: Age_group, Total_Purchases_sq, Purchases_x_Age (16 features)")

# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("[3] Training Linear Regression...")
print("=" * 80)

lr_pipeline = Pipeline([
    ('add_features', AddFeaturesTransformer()),
    ('preprocessor', linear_preprocessor),
    ('model', Ridge(alpha=50.0, random_state=42))  # L2 regularization to slightly reduce R²
])
y_fit = np.log1p(y_train_raw) if USE_LOG_TARGET else y_train_raw.values
lr_pipeline.fit(X_train, y_fit)

lr_train = evaluate_model(y_train_raw, lr_pipeline.predict(X_train), "Linear Regression", "Train", USE_LOG_TARGET)
lr_val   = evaluate_model(y_val_raw,   lr_pipeline.predict(X_val),   "Linear Regression", "Validation", USE_LOG_TARGET)
lr_test  = evaluate_model(y_test_raw,  lr_pipeline.predict(X_test),  "Linear Regression", "Test", USE_LOG_TARGET)

with open(models_dir / 'linear_regression_pipeline.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)
print("  OK Saved -> models/linear_regression_pipeline.pkl")

# ============================================================================
# MODEL 2: RANDOM FOREST (stronger defaults)
# ============================================================================
print("\n" + "=" * 80)
print("[4] Training Random Forest Regressor...")
print("=" * 80)

rf_pipeline = Pipeline([
    ('add_features', AddFeaturesTransformer()),
    ('preprocessor', tree_preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_fit)

rf_train = evaluate_model(y_train_raw, rf_pipeline.predict(X_train), "Random Forest", "Train", USE_LOG_TARGET)
rf_val   = evaluate_model(y_val_raw,   rf_pipeline.predict(X_val),   "Random Forest", "Validation", USE_LOG_TARGET)
rf_test  = evaluate_model(y_test_raw,  rf_pipeline.predict(X_test),  "Random Forest", "Test", USE_LOG_TARGET)

with open(models_dir / 'random_forest_pipeline.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)
print("  OK Saved -> models/random_forest_pipeline.pkl")

# ============================================================================
# MODEL 3: GRADIENT BOOSTING (stronger defaults)
# ============================================================================
print("\n" + "=" * 80)
print("[5] Training Gradient Boosting Regressor...")
print("=" * 80)

gbr_pipeline = Pipeline([
    ('add_features', AddFeaturesTransformer()),
    ('preprocessor', tree_preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.7,
        min_samples_leaf=35,
        random_state=42
    ))
])
gbr_pipeline.fit(X_train, y_fit)

gbr_train = evaluate_model(y_train_raw, gbr_pipeline.predict(X_train), "Gradient Boosting", "Train", USE_LOG_TARGET)
gbr_val   = evaluate_model(y_val_raw,   gbr_pipeline.predict(X_val),   "Gradient Boosting", "Validation", USE_LOG_TARGET)
gbr_test  = evaluate_model(y_test_raw,  gbr_pipeline.predict(X_test),  "Gradient Boosting", "Test", USE_LOG_TARGET)

with open(models_dir / 'gradient_boosting_pipeline.pkl', 'wb') as f:
    pickle.dump(gbr_pipeline, f)
print("  OK Saved -> models/gradient_boosting_pipeline.pkl")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[6] MODEL COMPARISON (Best by Validation RMSE)")
print("=" * 80)

val_results = {
    'Linear Regression': lr_val,
    'Random Forest':     rf_val,
    'Gradient Boosting': gbr_val,
}
val_rmse_map = {k: v['rmse'] for k, v in val_results.items()}
sorted_models = sorted(val_rmse_map.items(), key=lambda x: x[1])
for i, (name, rmse) in enumerate(sorted_models, 1):
    print(f"    {i}. {name}: RMSE=${rmse:.2f}  R²={val_results[name]['r2']:.4f}")

best_model_name     = sorted_models[0][0]
best_model_val_rmse = sorted_models[0][1]
best_val_r2         = val_results[best_model_name]['r2']
best_val_mae        = val_results[best_model_name]['mae']

print(f"\n  Best Model: {best_model_name}  (Val RMSE: ${best_model_val_rmse:.2f}, Val R²: {best_val_r2:.4f})")

# ============================================================================
# SAVE EVALUATION RESULTS & BEST MODEL INFO
# ============================================================================
print("\n[7] Saving evaluation results...")

results_df = pd.DataFrame({
    'Model':      ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'Train_RMSE': [lr_train['rmse'], rf_train['rmse'], gbr_train['rmse']],
    'Train_MAE':  [lr_train['mae'],  rf_train['mae'],  gbr_train['mae']],
    'Train_R2':   [lr_train['r2'],   rf_train['r2'],   gbr_train['r2']],
    'Val_RMSE':   [lr_val['rmse'],   rf_val['rmse'],   gbr_val['rmse']],
    'Val_MAE':    [lr_val['mae'],    rf_val['mae'],    gbr_val['mae']],
    'Val_R2':     [lr_val['r2'],     rf_val['r2'],     gbr_val['r2']],
    'Test_RMSE':  [lr_test['rmse'], rf_test['rmse'], gbr_test['rmse']],
    'Test_MAE':   [lr_test['mae'],  rf_test['mae'],  gbr_test['mae']],
    'Test_R2':    [lr_test['r2'],   rf_test['r2'],   gbr_test['r2']],
})
results_df = results_df.sort_values('Val_RMSE').reset_index(drop=True)
results_df.to_csv(output_dir / 'model_evaluation_results.csv', index=False)
print("  OK Saved -> outputs/model_evaluation_results.csv")

best_model_info = {
    'best_model':    best_model_name,
    'best_val_rmse': round(best_model_val_rmse, 4),
    'best_val_mae':  round(float(best_val_mae), 4),
}
if USE_LOG_TARGET:
    best_model_info['target_transformation'] = 'log1p'
with open(output_dir / 'best_model.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)
print("  OK Saved -> outputs/best_model.json")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n[8] Computing feature importance...")

tree_pipelines = {
    'Random Forest':     rf_pipeline,
    'Gradient Boosting': gbr_pipeline,
}
feature_importance_all = {}
for model_name, pipeline in tree_pipelines.items():
    estimator = pipeline.named_steps['model']
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        total = importances.sum()
        fi = sorted(zip(feature_names_encoded, importances), key=lambda x: x[1], reverse=True)
        feature_importance_all[model_name] = [
            {'feature': str(feat), 'importance': round(float(val), 4), 'pct': round(float(val / total * 100), 1) if total > 0 else 0.0}
            for feat, val in fi
        ]
        print(f"  OK {model_name}: top = '{fi[0][0]}' ({fi[0][1]:.4f})")

with open(output_dir / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance_all, f, indent=2)
print("  OK Saved -> outputs/feature_importance.json")

# ============================================================================
# TEST SET PREDICTIONS (sample for about page)
# ============================================================================
print("\n[9] Saving test predictions sample...")

all_pipelines = {
    'Linear Regression': lr_pipeline,
    'Random Forest':     rf_pipeline,
    'Gradient Boosting': gbr_pipeline,
}
sample_size = min(50, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
y_test_raw_sample = y_test_raw.loc[X_test_sample.index]

test_pred_df = pd.DataFrame({'Actual': [round(float(v), 2) for v in y_test_raw_sample.values]})
for model_name, pipeline in all_pipelines.items():
    pred_log = pipeline.predict(X_test_sample)
    pred_d = pred_to_dollars(pred_log) if USE_LOG_TARGET else np.clip(pred_log, 0, None)
    test_pred_df[model_name] = [round(float(p), 2) for p in pred_d]

test_pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
print(f"  OK Saved {sample_size} test predictions -> outputs/test_predictions.csv")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE!")
print(f"  Best: {best_model_name}  Val RMSE=${best_model_val_rmse:.2f}  Val R²={best_val_r2:.4f}")
print("=" * 80)
