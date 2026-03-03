"""
FLASK WEB APPLICATION
Customer Lifetime Value (CLV) Predictor
"""

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Required for unpickling pipelines that use AddFeaturesTransformer
from pipeline_utils import AddFeaturesTransformer  # noqa: F401

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clv-predictor-2024')

models_dir  = Path("models")
outputs_dir = Path("outputs")

print("Loading models and data...")

print("  Loading Linear Regression...")
with open(models_dir / 'linear_regression_pipeline.pkl', 'rb') as f:
    lr_pipeline = pickle.load(f)

print("  Loading Random Forest...")
with open(models_dir / 'random_forest_pipeline.pkl', 'rb') as f:
    rf_pipeline = pickle.load(f)

print("  Loading Gradient Boosting...")
with open(models_dir / 'gradient_boosting_pipeline.pkl', 'rb') as f:
    gbr_pipeline = pickle.load(f)

models_dict = {
    'Linear Regression': lr_pipeline,
    'Random Forest':     rf_pipeline,
    'Gradient Boosting': gbr_pipeline,
}

# Evaluation results
eval_results     = pd.read_csv(outputs_dir / 'model_evaluation_results.csv')
model_comparison = eval_results.to_dict('records')

# Best model info (and optional log-target flag for prediction)
with open(outputs_dir / 'best_model.json') as f:
    best_info = json.load(f)
best_model_name = best_info['best_model']
best_val_rmse   = best_info['best_val_rmse']
target_transformation = best_info.get('target_transformation', None)  # 'log1p' if models predict log1p(dollars)

best_row      = eval_results[eval_results['Model'] == best_model_name].iloc[0]
best_val_mae  = float(best_row['Val_MAE'])
best_val_r2   = float(best_row['Val_R2'])

# Analytics data + form options
with open(outputs_dir / 'data_stats.json') as f:
    data_stats = json.load(f)

# Build form dropdown options dynamically from data_stats (avoids hardcoding brands/types)
FORM_OPTIONS = dict(data_stats.get('category_values', {}))
FORM_OPTIONS['Gender'] = ['Male', 'Female']
FORM_OPTIONS['Income'] = ['Low', 'Medium', 'High']

with open(outputs_dir / 'correlation_data.json') as f:
    correlation_data = json.load(f)

with open(outputs_dir / 'feature_importance.json') as f:
    feature_importance_all = json.load(f)

fi_rf  = feature_importance_all.get('Random Forest',     [])
fi_gbr = feature_importance_all.get('Gradient Boosting', [])

# Test predictions for about page
test_pred_df     = pd.read_csv(outputs_dir / 'test_predictions.csv')
test_predictions = test_pred_df.to_dict('records')

print("  OK Models and data loaded")
print(f"  Best Model: {best_model_name}  (Val RMSE: {best_val_rmse}, Val R2: {best_val_r2:.4f})")


def interpret_clv(value):
    """Classify predicted CLV into Low / Medium / High."""
    p25 = data_stats.get('target_p25', 0)
    p75 = data_stats.get('target_p75', 9e9)
    if value >= p75:
        return 'High Value Customer', '#27ae60'
    elif value <= p25:
        return 'Low Value Customer', '#e74c3c'
    else:
        return 'Medium Value Customer', '#f39c12'


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    result = session.get('last_result')
    return render_template(
        'index.html',
        best_model=best_model_name,
        best_model_rmse=best_val_rmse,
        best_model_r2=best_val_r2,
        options=FORM_OPTIONS,
        result=result
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age              = request.form.get('age', '').strip()
        gender           = request.form.get('gender', '').strip()
        income           = request.form.get('income', '').strip()
        total_purchases  = request.form.get('total_purchases', '').strip()
        amount           = request.form.get('amount', '').strip()
        tenure_months    = request.form.get('tenure_months', '').strip()
        recency_months   = request.form.get('recency_months', '').strip()
        product_category = request.form.get('product_category', '').strip()
        payment_method   = request.form.get('payment_method', '').strip()
        shipping_method  = request.form.get('shipping_method', '').strip()
        customer_segment = request.form.get('customer_segment', '').strip()
        ratings          = request.form.get('ratings', '').strip()
        feedback         = request.form.get('feedback', '').strip()
        month            = request.form.get('month', '').strip()
        product_brand    = request.form.get('product_brand', '').strip()
        product_type     = request.form.get('product_type', '').strip()

        if not all([age, gender, income, total_purchases, amount, tenure_months, recency_months,
                    product_category, payment_method, shipping_method,
                    customer_segment, ratings, feedback, month,
                    product_brand, product_type]):
            session['form_error'] = 'Please fill in all fields.'
            return redirect(url_for('index'))

        # Column order must match feature_columns.pkl (Phase 2)
        input_df = pd.DataFrame([{
            'Age':              int(age),
            'Gender':           gender,
            'Income':           income,
            'Total_Purchases':  int(total_purchases),
            'Amount':           float(amount),
            'Product_Category': product_category,
            'Payment_Method':   payment_method,
            'Shipping_Method':  shipping_method,
            'Customer_Segment': customer_segment,
            'Ratings':          float(ratings),
            'Feedback':         feedback,
            'Month':            month,
            'Product_Brand':    product_brand,
            'Product_Type':     product_type,
            'Tenure_months':    float(tenure_months),
            'Recency_months':   float(recency_months),
        }])

        # Run all models (convert from log space if trained with log1p target)
        all_preds = []
        for model_name, pipeline in models_dict.items():
            pred_raw = float(pipeline.predict(input_df)[0])
            if target_transformation == 'log1p':
                pred_raw = np.clip(pred_raw, -0.5, 12.0)
                pred = max(0.0, float(np.expm1(pred_raw)))
            else:
                pred = max(0.0, pred_raw)
            all_preds.append({'name': model_name, 'prediction': round(pred, 2)})

        # Sort to match evaluation results order (best model first)
        order = list(eval_results['Model'])
        all_preds.sort(key=lambda x: order.index(x['name']) if x['name'] in order else 99)

        best_pred    = next((p['prediction'] for p in all_preds if p['name'] == best_model_name),
                            all_preds[0]['prediction'])
        label, color = interpret_clv(best_pred)

        session['last_result'] = {
            'best_prediction': best_pred,
            'label':           label,
            'label_color':     color,
            'all_predictions': all_preds,
            'inputs': {
                'Age':              age,
                'Gender':           gender,
                'Income':           income,
                'Total Purchases':  total_purchases,
                'Amount':           amount,
                'Tenure (months)':  tenure_months,
                'Recency (months)': recency_months,
                'Product Category': product_category,
                'Payment Method':   payment_method,
                'Shipping Method':  shipping_method,
                'Customer Segment': customer_segment,
                'Ratings':          ratings,
                'Feedback':         feedback,
                'Month':            month,
                'Product Brand':    product_brand,
                'Product Type':     product_type,
            }
        }
        return redirect(url_for('index'))

    except Exception as e:
        session['form_error'] = f'Prediction error: {str(e)}'
        return redirect(url_for('index'))


@app.route('/about')
def about():
    result = session.get('last_result')
    return render_template(
        'about.html',
        model_comparison=model_comparison,
        correlation_data=correlation_data,
        feature_importance_rf=fi_rf,
        feature_importance_gbr=fi_gbr,
        data_stats=data_stats,
        test_predictions=test_predictions,
        best_model=best_model_name,
        best_model_rmse=best_val_rmse,
        best_model_r2=best_val_r2,
        best_model_mae=best_val_mae,
        result=result
    )


@app.route('/reset')
def reset():
    session.pop('last_result', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
