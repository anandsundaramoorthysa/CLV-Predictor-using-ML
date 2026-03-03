"""
Generate analysis plots for the Flask about page:
- Correlation Analysis heatmap
- Feature Importance chart (RF, GBR, ET)
- Model Predictions Comparison bar chart (test set sample)

Run this after phase4_model_training.py.
Plots are saved to static/ for use in the Flask about page.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json
from pathlib import Path

print("=" * 80)
print("GENERATING ANALYSIS PLOTS")
print("=" * 80)

input_dir  = Path("outputs")
models_dir = Path("models")
plots_dir  = Path("static")
plots_dir.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data and models...")

with open(input_dir / 'correlation_data.json') as f:
    correlation_data = json.load(f)

with open(input_dir / 'feature_importance.json') as f:
    feature_importance_all = json.load(f)

test_pred_df = pd.read_csv(input_dir / 'test_predictions.csv')
print("  OK Data loaded")

# ============================================================================
# PLOT 1: CORRELATION ANALYSIS HEATMAP
# ============================================================================
print("\n[2] Generating correlation analysis plot...")

features = [row['feature'] for row in correlation_data]
pearson  = [row['pearson']  for row in correlation_data]
spearman = [row['spearman'] for row in correlation_data]

x     = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(18, 6))
bars1 = ax.bar(x - width/2, pearson,  width, label='Pearson',  color='#3498db', alpha=0.85)
bars2 = ax.bar(x + width/2, spearman, width, label='Spearman', color='#e74c3c', alpha=0.85)

ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Feature Correlation with Total Amount (CLV)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)
ax.set_ylim(-1.0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + (0.02 if h >= 0 else -0.06),
            f'{h:.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + (0.02 if h >= 0 else -0.06),
            f'{h:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(plots_dir / 'correlation_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK correlation_analysis.png saved")

# ============================================================================
# PLOT 2: FEATURE IMPORTANCE (RF, GBR, ET side by side)
# ============================================================================
print("\n[3] Generating feature importance plot...")

tree_models = ['Random Forest', 'Gradient Boosting']
colors      = ['#3498db', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle('Feature Importance - Tree-based Models', fontsize=15, fontweight='bold', y=1.02)

for idx, (model_name, color) in enumerate(zip(tree_models, colors)):
    ax = axes[idx]
    fi_data = feature_importance_all.get(model_name, [])
    if not fi_data:
        ax.set_title(f'{model_name}\n(No data)', fontsize=11)
        continue

    feat_names   = [row['feature']    for row in fi_data]
    importances  = [row['importance'] for row in fi_data]

    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, importances, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(model_name, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (val, pos) in enumerate(zip(importances, y_pos)):
        ax.text(val + 0.001, pos, f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(plots_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK feature_importance.png saved")

# ============================================================================
# PLOT 3: MODEL PREDICTIONS vs ACTUAL (test sample)
# ============================================================================
print("\n[4] Generating model predictions comparison plot...")

actual_vals   = test_pred_df['Actual'].values
model_columns = [c for c in test_pred_df.columns if c != 'Actual']
sample_idx    = np.arange(len(actual_vals))

fig, ax = plt.subplots(figsize=(14, 7))
plot_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

ax.scatter(sample_idx, actual_vals, color='black', s=40, label='Actual', zorder=5, alpha=0.9)

for i, (col, color) in enumerate(zip(model_columns, plot_colors)):
    ax.scatter(sample_idx, test_pred_df[col].values, color=color, s=25,
               label=col, marker='x', linewidths=1.5, alpha=0.8)

ax.set_xlabel('Test Sample Index', fontsize=12)
ax.set_ylabel('Total Amount ($)', fontsize=12)
ax.set_title('Model Predictions vs Actual - Test Set Sample', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / 'model_predictions_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK model_predictions_comparison.png saved")

print("\n" + "=" * 80)
print("PLOTS GENERATED SUCCESSFULLY!")
print("  static/correlation_analysis.png")
print("  static/feature_importance.png")
print("  static/model_predictions_comparison.png")
print("=" * 80)
