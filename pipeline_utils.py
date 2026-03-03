"""
Shared pipeline components for CLV predictor.
Used by phase4_model_training.py and app.py (for pickle deserialization).
"""
import numpy as np


class AddFeaturesTransformer:
    """Add Age_group, Total_Purchases_sq, Purchases_x_Age. Input: 16 cols in, Output: 19 cols out."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        age = np.asarray(X['Age'], dtype=float).reshape(-1)
        purchases = np.asarray(X['Total_Purchases'], dtype=float).reshape(-1)
        # Bins: 0-25 -> 0, 26-40 -> 1, 41-55 -> 2, 56+ -> 3
        age_group = np.digitize(age, bins=[0, 26, 41, 56, 121]) - 1
        X['Age_group'] = np.clip(age_group, 0, 3)
        X['Total_Purchases_sq'] = purchases ** 2
        # Interaction: scale by 1/100 to keep magnitude similar to other numerics
        X['Purchases_x_Age'] = (purchases * age) / 100.0
        return X

    def get_feature_names_out(self, input_features=None):
        out = list(input_features) if input_features is not None else []
        return out + ['Age_group', 'Total_Purchases_sq', 'Purchases_x_Age']
