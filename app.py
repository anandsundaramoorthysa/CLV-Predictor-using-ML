import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import warnings
import threading
import time
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Association Rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Removed file size limit - no MAX_CONTENT_LENGTH restriction
app.config['STATIC_PLOTS'] = 'static/plots'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_PLOTS'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# ==================== PROGRESS TRACKING ====================

# Shared progress state (thread-safe dictionary)
progress_state = {}
progress_lock = threading.Lock()

# Shared results storage (thread-safe dictionary)
results_storage = {}
results_lock = threading.Lock()

def init_progress(session_id):
    """Initialize progress state for a session"""
    with progress_lock:
        progress_state[session_id] = {
            'current_step': 0,
            'total_steps': 11,
            'steps': {
                1: {'name': 'File uploaded', 'status': 'pending'},
                2: {'name': 'Data validation', 'status': 'pending'},
                3: {'name': 'Data preprocessing', 'status': 'pending'},
                4: {'name': 'Feature engineering', 'status': 'pending'},
                5: {'name': 'Train model 1 – Linear Regression', 'status': 'pending'},
                6: {'name': 'Train model 2 – Random Forest', 'status': 'pending'},
                7: {'name': 'Train model 3 – Gradient Boosting', 'status': 'pending'},
                8: {'name': 'Model evaluation', 'status': 'pending'},
                9: {'name': 'Best model selection', 'status': 'pending'},
                10: {'name': 'Final prediction generation', 'status': 'pending'},
                11: {'name': 'Business insights ready', 'status': 'pending'}
            },
            'error': None,
            'completed': False
        }

def update_progress(session_id, step_num, status='completed'):
    """Update progress for a specific step"""
    with progress_lock:
        if session_id in progress_state:
            if step_num in progress_state[session_id]['steps']:
                progress_state[session_id]['steps'][step_num]['status'] = status
                if status == 'in_progress':
                    progress_state[session_id]['current_step'] = step_num
                elif status == 'completed' and step_num == progress_state[session_id]['current_step']:
                    progress_state[session_id]['current_step'] = step_num + 1

def set_error(session_id, error_message):
    """Set error state"""
    with progress_lock:
        if session_id in progress_state:
            progress_state[session_id]['error'] = error_message

def mark_completed(session_id):
    """Mark processing as completed"""
    with progress_lock:
        if session_id in progress_state:
            progress_state[session_id]['completed'] = True

def get_progress(session_id):
    """Get current progress state"""
    with progress_lock:
        if session_id in progress_state:
            return progress_state[session_id].copy()
        return None

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== PREPROCESSING MODULE ====================

def load_dataset(filepath):
    """Load CSV or Excel file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        return df
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def detect_column_types(df):
    """Automatically detect numerical, categorical, and date columns"""
    numerical = []
    categorical = []
    date_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Double-check it's not actually a date stored as number
            if not _is_date_column(df[col]):
                numerical.append(col)
        elif df[col].dtype == 'object':
            # Check if it's a date column by trying to parse multiple values
            if _is_date_column(df[col]):
                date_cols.append(col)
            else:
                categorical.append(col)
        elif 'datetime' in str(df[col].dtype):
            date_cols.append(col)
    
    return numerical, categorical, date_cols

def _is_date_column(series):
    """Check if a column contains date values"""
    if len(series.dropna()) == 0:
        return False
    
    # Try to parse a sample of non-null values
    sample_size = min(10, len(series.dropna()))
    sample = series.dropna().head(sample_size)
    
    date_count = 0
    for value in sample:
        try:
            pd.to_datetime(value)
            date_count += 1
        except:
            pass
    
    # If more than 50% of sample can be parsed as dates, consider it a date column
    return date_count >= (sample_size * 0.5)

def handle_missing_values(df, numerical_cols, categorical_cols):
    """Handle missing values: mean for numerical, mode for categorical"""
    df_clean = df.copy()
    
    # Numerical: fill with mean
    for col in numerical_cols:
        if col in df_clean.columns:
            try:
                # Ensure column is numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            except Exception as e:
                # If conversion fails, treat as categorical
                print(f"Warning: Column {col} could not be converted to numeric: {e}")
                if col not in categorical_cols:
                    categorical_cols.append(col)
    
    # Categorical: fill with mode
    for col in categorical_cols:
        if col in df_clean.columns and col not in numerical_cols:
            mode_value = df_clean[col].mode()
            if len(mode_value) > 0:
                df_clean[col].fillna(mode_value[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

def remove_duplicates(df):
    """Remove duplicate rows"""
    return df.drop_duplicates()

def convert_date_columns(df, date_cols):
    """Convert date columns to numerical features"""
    df_converted = df.copy()
    new_numerical_cols = []
    
    for col in date_cols:
        if col in df_converted.columns:
            try:
                # Convert to datetime
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                
                # Extract numerical features
                df_converted[f'{col}_year'] = df_converted[col].dt.year
                df_converted[f'{col}_month'] = df_converted[col].dt.month
                df_converted[f'{col}_day'] = df_converted[col].dt.day
                df_converted[f'{col}_dayofweek'] = df_converted[col].dt.dayofweek
                
                # Add to numerical columns list
                new_numerical_cols.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
                
                # Drop original date column
                df_converted.drop(columns=[col], inplace=True)
            except Exception as e:
                # If conversion fails, drop the column
                print(f"Warning: Could not convert date column {col}: {e}")
                df_converted.drop(columns=[col], inplace=True, errors='ignore')
    
    return df_converted, new_numerical_cols

def encode_categorical(df, categorical_cols):
    """Encode categorical variables using LabelEncoder"""
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders

def scale_features(X_train, X_test, numerical_cols):
    """Scale numerical features using StandardScaler"""
    # Only scale if we have numerical columns
    if len(numerical_cols) > 0:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Use a single scaler for all columns (more efficient and correct)
        # Get only the numerical columns that exist in the dataframe
        cols_to_scale = [col for col in numerical_cols if col in X_train.columns]
        
        if len(cols_to_scale) > 0:
            scaler = StandardScaler()
            # Fit and transform training data
            X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            # Transform test data using training scaler
            X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train, X_test, None
    
    return X_train, X_test, None

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    # Detect column types
    numerical, categorical, date_cols = detect_column_types(df)
    
    # Convert date columns to numerical features
    df_clean, new_numerical_cols = convert_date_columns(df, date_cols)
    
    # Add new numerical columns from date conversion
    numerical.extend(new_numerical_cols)
    
    # Re-detect column types after date conversion (in case types changed)
    numerical, categorical, _ = detect_column_types(df_clean)
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, numerical, categorical)
    
    # Remove duplicates
    df_clean = remove_duplicates(df_clean)
    
    # Encode categorical
    df_encoded, encoders = encode_categorical(df_clean, categorical)
    
    return df_encoded, numerical, categorical, date_cols, encoders

# ==================== FEATURE ENGINEERING ====================

def feature_engineering(df, numerical_cols):
    """Create derived features if applicable"""
    df_fe = df.copy()
    derived_features = []
    
    # Ensure numerical columns are actually numeric
    for col in numerical_cols:
        if col in df_fe.columns:
            try:
                df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
            except:
                pass
    
    # Create interaction features for numerical columns (if we have at least 2)
    if len(numerical_cols) >= 2:
        # Example: product of first two numerical columns
        try:
            col1, col2 = numerical_cols[0], numerical_cols[1]
            if col1 in df_fe.columns and col2 in df_fe.columns:
                try:
                    df_fe[f'{col1}_x_{col2}'] = pd.to_numeric(df_fe[col1], errors='coerce') * pd.to_numeric(df_fe[col2], errors='coerce')
                    derived_features.append(f'{col1}_x_{col2}')
                except:
                    pass
        except (IndexError, ValueError):
            pass
    
    # Create squared features for first numerical column
    if len(numerical_cols) >= 1:
        try:
            col = numerical_cols[0]
        except IndexError:
            col = None
        
        if col and col in df_fe.columns:
            try:
                df_fe[f'{col}_squared'] = pd.to_numeric(df_fe[col], errors='coerce') ** 2
                derived_features.append(f'{col}_squared')
            except:
                pass
    
    return df_fe, derived_features

# ==================== MODEL TRAINING MODULE ====================

def prepare_target_variable(df, numerical_cols):
    """Automatically select target variable (last numerical column or create synthetic)"""
    if len(numerical_cols) == 0:
        # Create synthetic target if no numerical columns
        df['target'] = np.random.randn(len(df))
        return 'target', df
    
    # Use last numerical column as target
    target_col = numerical_cols[-1]
    return target_col, df

def train_models(X_train, X_test, y_train, y_test, session_id=None):
    """Train three models: Linear Regression, Random Forest, Gradient Boosting"""
    models = {}
    results = {}
    
    # Ensure no NaN values in training data
    # Convert to numpy arrays and handle NaN
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Check and handle NaN values
    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
        # Fill NaN with column means for X, median for y
        imputer_X = SimpleImputer(strategy='mean')
        X_train = imputer_X.fit_transform(X_train)
        X_test = imputer_X.transform(X_test)
        
        imputer_y = SimpleImputer(strategy='median')
        # Convert to numpy array if it's a Series
        y_train_arr = np.array(y_train).reshape(-1, 1) if not isinstance(y_train, np.ndarray) else y_train.reshape(-1, 1)
        y_test_arr = np.array(y_test).reshape(-1, 1) if not isinstance(y_test, np.ndarray) else y_test.reshape(-1, 1)
        y_train = imputer_y.fit_transform(y_train_arr).ravel()
        y_test = imputer_y.transform(y_test_arr).ravel()
    
    # Model 1: Linear Regression
    if session_id:
        update_progress(session_id, 5, 'in_progress')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr)
    }
    if session_id:
        update_progress(session_id, 5, 'completed')
    
    # Model 2: Random Forest
    if session_id:
        update_progress(session_id, 6, 'in_progress')
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf)
    }
    if session_id:
        update_progress(session_id, 6, 'completed')
    
    # Model 3: Gradient Boosting
    if session_id:
        update_progress(session_id, 7, 'in_progress')
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'r2': r2_score(y_test, y_pred_gb)
    }
    if session_id:
        update_progress(session_id, 7, 'completed')
    
    return models, results

def select_best_model(results):
    """Select best model based on lowest RMSE, lowest MAE, and highest R²"""
    best_model_name = None
    best_score = float('inf')
    
    for name, metrics in results.items():
        # Combined score: lower is better (normalize R² by subtracting from 1)
        score = metrics['rmse'] + metrics['mae'] - metrics['r2']
        if score < best_score:
            best_score = score
            best_model_name = name
    
    return best_model_name, results[best_model_name]

# ==================== BUSINESS INSIGHTS MODULE ====================

def customer_segmentation(df, numerical_cols, n_clusters=3):
    """Perform KMeans clustering for customer segmentation"""
    # Get all numerical columns from dataframe (after encoding, categorical become numerical too)
    df_numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                        and col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk']]
    
    # Use passed numerical_cols if available, otherwise use detected numerical columns
    if len(numerical_cols) >= 2:
        available_cols = [col for col in numerical_cols if col in df.columns]
    else:
        available_cols = df_numerical_cols
    
    if len(available_cols) < 2:
        return None
    
    # Use first 2-3 numerical columns for clustering
    features = available_cols[:min(3, len(available_cols))]
    X = df[features].values
    
    # Handle NaN values before clustering
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Final NaN check after scaling (shouldn't happen, but safety)
    if np.isnan(X_scaled).any():
        X_scaled = np.nan_to_num(X_scaled, nan=np.nanmean(X_scaled, axis=0))
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['segment'] = clusters
    
    # Segment analysis
    segment_info = []
    for i in range(n_clusters):
        segment_df = df[df['segment'] == i]
        size = len(segment_df)
        segment_info.append({
            'segment': f'Segment {i+1}',
            'size': size,
            'percentage': round(size / len(df) * 100, 2),
            'interpretation': f'Customer group with distinct characteristics in {", ".join(features[:min(2, len(features))])}'
        })
    
    # Visualization
    if len(features) >= 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Customer Segmentation')
        plt.legend()
        plt.colorbar(scatter)
        plot_path = os.path.join(app.config['STATIC_PLOTS'], 'segmentation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return segment_info

def clv_prediction(df, numerical_cols, best_model, feature_cols=None):
    """Predict Customer Lifetime Value"""
    # Use the same features that the model was trained on
    if feature_cols is not None and len(feature_cols) > 0:
        # Use the exact features from training (don't filter - model expects all of them)
        # Exclude only columns that are added during insights (segment, clv, etc.)
        features = [col for col in feature_cols if col in df.columns]
    else:
        # Fallback: Get all numerical columns from dataframe
        df_numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                            and col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk']]
        
        # Use passed numerical_cols if available, otherwise use detected numerical columns
        if len(numerical_cols) >= 1:
            available_cols = [col for col in numerical_cols if col in df.columns]
        else:
            available_cols = df_numerical_cols
        
        if len(available_cols) == 0:
            return None
        
        features = [col for col in df.columns if col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk'] and col in available_cols]
        if len(features) == 0:
            features = available_cols[:min(5, len(available_cols))]
    
    # Double-check all features exist
    features = [col for col in features if col in df.columns]
    if len(features) == 0:
        return None
    
    # Ensure we have all features the model expects (in the same order)
    # If feature_cols was provided, use those exact features in that order
    if feature_cols is not None:
        # Use exact training features in exact training order
        features = feature_cols.copy()
        
        # Check if any features are missing from dataframe
        missing_features = [col for col in features if col not in df.columns]
        if len(missing_features) > 0:
            # Add missing features with 0 values
            for col in missing_features:
                df[col] = 0
    
    # Create feature matrix in the exact order the model expects
    X = df[features].values
    
    # Verify feature count matches model expectation
    if hasattr(best_model, 'n_features_in_'):
        expected_features = best_model.n_features_in_
        if X.shape[1] != expected_features:
            # Pad or trim to match expected features
            if X.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.hstack([X, padding])
            elif X.shape[1] > expected_features:
                # Trim to expected size (shouldn't happen, but safety)
                X = X[:, :expected_features]
    
    # Handle NaN values before prediction
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    clv_predictions = best_model.predict(X)
    df['clv'] = clv_predictions
    
    # Top high-value customers
    top_customers = df.nlargest(10, 'clv')[['clv']].to_dict('records')
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(clv_predictions, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Customer Lifetime Value')
    plt.ylabel('Frequency')
    plt.title('CLV Distribution')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(app.config['STATIC_PLOTS'], 'clv_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'distribution': {
            'mean': float(np.mean(clv_predictions)),
            'median': float(np.median(clv_predictions)),
            'std': float(np.std(clv_predictions))
        },
        'top_customers': top_customers
    }

def churn_prediction(df, numerical_cols):
    """Predict churn probability using classification"""
    # Get all numerical columns from dataframe (after encoding, categorical become numerical too)
    df_numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                        and col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk']]
    
    # Use passed numerical_cols if available, otherwise use detected numerical columns
    if len(numerical_cols) >= 1:
        available_cols = [col for col in numerical_cols if col in df.columns]
    else:
        available_cols = df_numerical_cols
    
    if len(available_cols) == 0:
        return None
    
    # Create synthetic churn target (in real scenario, this would be provided)
    # Use a threshold on one of the numerical columns
    if len(available_cols) == 0:
        return None
    
    threshold_col = available_cols[0]
    if threshold_col not in df.columns:
        return None
    
    threshold = df[threshold_col].median()
    y_churn = (df[threshold_col] < threshold).astype(int)
    
    # Train classifier
    features = [col for col in available_cols if col != threshold_col][:5]
    if len(features) == 0:
        features = available_cols[:min(5, len(available_cols))]
    
    # Double-check all features exist
    features = [col for col in features if col in df.columns]
    if len(features) == 0:
        return None
    
    X = df[features].values
    
    # Handle NaN values before training
    if np.isnan(X).any() or np.isnan(y_churn).any():
        imputer_X = SimpleImputer(strategy='mean')
        X = imputer_X.fit_transform(X)
        # For y_churn, fill with mode (most common class)
        imputer_y = SimpleImputer(strategy='most_frequent')
        # Convert to numpy array if it's a Series
        y_churn_arr = np.array(y_churn).reshape(-1, 1) if not isinstance(y_churn, np.ndarray) else y_churn.reshape(-1, 1)
        y_churn = imputer_y.fit_transform(y_churn_arr).ravel().astype(int)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y_churn)
    
    churn_probs = clf.predict_proba(X)[:, 1]
    df['churn_probability'] = churn_probs
    
    # Categorize risk
    df['churn_risk'] = pd.cut(churn_probs, bins=[0, 0.3, 0.7, 1.0], 
                              labels=['Low', 'Medium', 'High'])
    
    risk_counts = df['churn_risk'].value_counts().to_dict()
    high_risk = df[df['churn_risk'] == 'High'].head(10)[['churn_probability']].to_dict('records')
    
    return {
        'risk_counts': risk_counts,
        'high_risk_customers': high_risk
    }

def market_basket_analysis(df, categorical_cols):
    """Perform market basket analysis using Apriori"""
    if len(categorical_cols) < 2:
        return None
    
    try:
        # Filter to only include columns that actually exist in the dataframe
        available_cols = [col for col in categorical_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return None
        
        # Use first few categorical columns that exist
        cols = available_cols[:min(3, len(available_cols))]
        
        # Check if we have any data in these columns
        df_subset = df[cols].dropna()
        if len(df_subset) == 0:
            return None
        
        # Check if columns have any non-null values and exist in df_subset
        valid_cols = []
        for col in cols:
            if col in df_subset.columns and df_subset[col].notna().sum() > 0:
                # Check if column has at least 2 unique values (needed for dummies)
                if df_subset[col].nunique() >= 2:
                    valid_cols.append(col)
        
        if len(valid_cols) < 2:
            return None
        
        cols = valid_cols[:min(3, len(valid_cols))]
        
        # Double-check all columns exist in df_subset
        cols = [col for col in cols if col in df_subset.columns]
        if len(cols) < 2:
            return None
        
        # Create binary matrix and convert to boolean
        # Don't use prefix parameter - let pandas auto-generate column names
        # This avoids prefix length mismatch errors
        try:
            df_basket = pd.get_dummies(df_subset[cols], prefix_sep='_')
        except Exception as e:
            print(f"Error creating dummies: {e}")
            return None
        
        if df_basket.empty or len(df_basket.columns) == 0:
            return None
        
        df_basket = df_basket.astype(bool)
        
        # Apply Apriori
        # Adjust min_support based on dataset size
        min_support = max(0.05, min(0.2, 10.0 / len(df_basket)))
        frequent_itemsets = apriori(df_basket, min_support=min_support, use_colnames=True, max_len=2)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
            
            if len(rules) > 0:
                # Get top combinations
                top_rules = rules.nlargest(5, 'confidence')[['antecedents', 'consequents', 'support', 'confidence']]
                
                combinations = []
                for idx, row in top_rules.iterrows():
                    try:
                        antecedents = list(row['antecedents'])
                        consequents = list(row['consequents'])
                        if len(antecedents) > 0 and len(consequents) > 0:
                            combinations.append({
                                'combination': f"{antecedents[0]} → {consequents[0]}",
                                'support': round(row['support'], 3),
                                'confidence': round(row['confidence'], 3)
                            })
                    except:
                        continue
                
                if len(combinations) > 0:
                    return combinations
    except Exception as e:
        print(f"Market basket analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def sales_forecasting(df, numerical_cols, best_model, feature_cols=None):
    """Predict future sales/demand"""
    # Use the same features that the model was trained on
    if feature_cols is not None and len(feature_cols) > 0:
        # Use the exact features from training (don't filter - model expects all of them)
        # Exclude only columns that are added during insights (segment, clv, etc.)
        features = [col for col in feature_cols if col in df.columns]
    else:
        # Fallback: Get all numerical columns from dataframe
        df_numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                            and col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk']]
        
        # Use passed numerical_cols if available, otherwise use detected numerical columns
        if len(numerical_cols) >= 1:
            available_cols = [col for col in numerical_cols if col in df.columns]
        else:
            available_cols = df_numerical_cols
        
        if len(available_cols) == 0:
            return None
        
        features = [col for col in df.columns if col not in ['segment', 'target', 'clv', 'churn_probability', 'churn_risk'] and col in available_cols]
        if len(features) == 0:
            features = available_cols[:min(5, len(available_cols))]
    
    # Double-check all features exist
    features = [col for col in features if col in df.columns]
    if len(features) == 0:
        return None
    
    # Ensure we have all features the model expects (in the same order)
    # If feature_cols was provided, use those exact features in that order
    if feature_cols is not None:
        # Use exact training features in exact training order
        features = feature_cols.copy()
        
        # Check if any features are missing from dataframe
        missing_features = [col for col in features if col not in df.columns]
        if len(missing_features) > 0:
            # Add missing features with 0 values
            for col in missing_features:
                df[col] = 0
    
    # Create feature matrix in the exact order the model expects
    X = df[features].values
    
    # Verify feature count matches model expectation
    if hasattr(best_model, 'n_features_in_'):
        expected_features = best_model.n_features_in_
        if X.shape[1] != expected_features:
            # Pad or trim to match expected features
            if X.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.hstack([X, padding])
            elif X.shape[1] > expected_features:
                # Trim to expected size (shouldn't happen, but safety)
                X = X[:, :expected_features]
    
    # Handle NaN values before prediction
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    predictions = best_model.predict(X)
    
    # Create forecast table
    forecast_data = []
    for i in range(min(10, len(predictions))):
        forecast_data.append({
            'period': f'Period {i+1}',
            'predicted': round(float(predictions[i]), 2)
        })
    
    # Visualization: Actual vs Predicted
    if len(predictions) > 0:
        plt.figure(figsize=(12, 6))
        # Use first available numerical column or first feature for actual values
        if len(numerical_cols) > 0 and numerical_cols[0] in df.columns:
            actual = df[numerical_cols[0]].values[:len(predictions)]
        elif len(features) > 0 and features[0] in df.columns:
            actual = df[features[0]].values[:len(predictions)]
        else:
            # Use predictions as actual if no suitable column found
            actual = predictions.copy()
        
        plt.plot(range(len(actual)), actual, label='Actual', marker='o', alpha=0.7)
        plt.plot(range(len(predictions)), predictions, label='Predicted', marker='s', alpha=0.7)
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.title('Sales/Demand Forecasting: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(app.config['STATIC_PLOTS'], 'forecasting.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return forecast_data

# ==================== BACKGROUND PROCESSING ====================

def process_dataset_background(filepath, session_id):
    """Background processing function that runs all ML tasks with progress updates"""
    try:
        # Step 1: File uploaded (already done, just mark it)
        update_progress(session_id, 1, 'completed')
        
        # Step 2: Data validation
        update_progress(session_id, 2, 'in_progress')
        df = load_dataset(filepath)
        if len(df) < 10:
            raise Exception('Dataset too small. Please upload a dataset with at least 10 rows.')
        update_progress(session_id, 2, 'completed')
        
        # Step 3: Data preprocessing
        update_progress(session_id, 3, 'in_progress')
        df_processed, numerical, categorical, date_cols, encoders = preprocess_data(df)
        update_progress(session_id, 3, 'completed')
        
        # Step 4: Feature engineering
        update_progress(session_id, 4, 'in_progress')
        df_fe, derived_features = feature_engineering(df_processed, numerical)
        update_progress(session_id, 4, 'completed')
        
        # Prepare target variable
        target_col, df_fe = prepare_target_variable(df_fe, numerical)
        
        # Prepare features
        feature_cols = [col for col in df_fe.columns if col != target_col]
        X = df_fe[feature_cols]
        y = df_fe[target_col]
        
        # Ensure all features are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop rows with NaN values that resulted from conversion
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise Exception("No valid numeric data found after preprocessing. Please check your dataset.")
        
        # Final check: Fill any remaining NaN values (shouldn't happen, but safety check)
        X = X.fillna(X.mean())
        y = y.fillna(y.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Final NaN check after split (in case split introduced issues)
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use training mean for test
        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_train.median())  # Use training median for test
        
        # Scale features - use all numerical columns in X_train (includes derived features)
        # Get all numerical columns from X_train (not just original numerical list)
        all_numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, all_numerical_cols)
        
        # Final NaN check after scaling
        if isinstance(X_train_scaled, pd.DataFrame):
            X_train_scaled = X_train_scaled.fillna(X_train_scaled.mean())
            X_test_scaled = X_test_scaled.fillna(X_train_scaled.mean())
        elif isinstance(X_train_scaled, np.ndarray):
            if np.isnan(X_train_scaled).any():
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmean(X_train_scaled, axis=0))
            if np.isnan(X_test_scaled).any():
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmean(X_train_scaled, axis=0))
        
        # Steps 5-7: Train models (progress updated inside train_models)
        models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, session_id)
        
        # Step 8: Model evaluation
        update_progress(session_id, 8, 'in_progress')
        # Evaluation is already done in train_models, just mark as completed
        time.sleep(0.1)  # Small delay for UI update
        update_progress(session_id, 8, 'completed')
        
        # Step 9: Best model selection
        update_progress(session_id, 9, 'in_progress')
        best_model_name, best_model_result = select_best_model(results)
        best_model = models[best_model_name]
        time.sleep(0.1)
        update_progress(session_id, 9, 'completed')
        
        # Step 10: Final prediction generation
        update_progress(session_id, 10, 'in_progress')
        # Predictions are already generated, just mark as completed
        time.sleep(0.1)
        update_progress(session_id, 10, 'completed')
        
        # Step 11: Business insights ready
        update_progress(session_id, 11, 'in_progress')
        segment_info = customer_segmentation(df_fe, numerical, n_clusters=3)
        clv_info = clv_prediction(df_fe, numerical, best_model, feature_cols)
        churn_info = churn_prediction(df_fe, numerical)
        basket_info = market_basket_analysis(df_fe, categorical)
        forecast_info = sales_forecasting(df_fe, numerical, best_model, feature_cols)
        update_progress(session_id, 11, 'completed')
        
        # Store results in shared storage
        with results_lock:
            results_storage[session_id] = {
                'df_processed': df_fe.to_json(),
                'numerical_cols': numerical,
                'categorical_cols': categorical,
                'date_cols': date_cols,
                'derived_features': derived_features,
                'target_col': target_col,
                'feature_cols': feature_cols,
                'best_model_name': best_model_name,
                'results': {k: {'rmse': float(v['rmse']), 'mae': float(v['mae']), 'r2': float(v['r2'])} 
                           for k, v in results.items()},
                'segment_info': segment_info,
                'clv_info': clv_info,
                'churn_info': churn_info,
                'basket_info': basket_info,
                'forecast_info': forecast_info,
                'filename': os.path.basename(filepath)
            }
        
        mark_completed(session_id)
        
    except Exception as e:
        set_error(session_id, str(e))
        print(f"Error in background processing: {e}")

# ==================== FLASK ROUTES ====================

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate unique session ID for progress tracking
            session_id = session.get('session_id', str(time.time()))
            session['session_id'] = session_id
            
            # Initialize progress
            init_progress(session_id)
            
            # Start background processing
            thread = threading.Thread(target=process_dataset_background, args=(filepath, session_id))
            thread.daemon = True
            thread.start()
            
            flash('File uploaded successfully. Processing started...', 'success')
            return redirect(url_for('home'))
        
        else:
            flash('Invalid file format. Please upload CSV or Excel file.', 'error')
            return redirect(request.url)
    
    # GET request - display results if available
    session_id = session.get('session_id')
    insights = {}
    
    # Check if processing is completed and load results
    if session_id:
        with results_lock:
            if session_id in results_storage and get_progress(session_id) and get_progress(session_id).get('completed'):
                stored_results = results_storage[session_id]
                # Store only small metadata in session (NOT the large DataFrame)
                # This prevents ERR_RESPONSE_HEADERS_TOO_BIG error
                session['numerical_cols'] = stored_results.get('numerical_cols', [])
                session['categorical_cols'] = stored_results.get('categorical_cols', [])
                session['date_cols'] = stored_results.get('date_cols', [])
                session['derived_features'] = stored_results.get('derived_features', [])
                session['target_col'] = stored_results.get('target_col', '')
                session['feature_cols'] = stored_results.get('feature_cols', [])
                session['best_model_name'] = stored_results.get('best_model_name', '')
                session['results'] = stored_results.get('results', {})
                session['segment_info'] = stored_results.get('segment_info')
                session['clv_info'] = stored_results.get('clv_info')
                session['churn_info'] = stored_results.get('churn_info')
                session['basket_info'] = stored_results.get('basket_info')
                session['forecast_info'] = stored_results.get('forecast_info')
                session['filename'] = stored_results.get('filename', '')
                # DO NOT store df_processed in session - it's too large
                # It remains in results_storage for access if needed
                
                # Use for current display
                insights = {
                    'segmentation': stored_results.get('segment_info'),
                    'clv': stored_results.get('clv_info'),
                    'churn': stored_results.get('churn_info'),
                    'basket': stored_results.get('basket_info'),
                    'forecast': stored_results.get('forecast_info')
                }
            elif 'segment_info' in session:
                # Fallback to session if results storage not available
                insights = {
                    'segmentation': session.get('segment_info'),
                    'clv': session.get('clv_info'),
                    'churn': session.get('churn_info'),
                    'basket': session.get('basket_info'),
                    'forecast': session.get('forecast_info')
                }
    
    return render_template('home.html', insights=insights, session_id=session_id)

@app.route('/progress')
def progress():
    """API endpoint to get current progress status"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'error': 'No session found', 
                'completed': False,
                'steps': {},
                'current_step': 0,
                'total_steps': 11
            }), 200
        
        progress_data = get_progress(session_id)
        if not progress_data:
            return jsonify({
                'error': 'Progress not found', 
                'completed': False,
                'steps': {},
                'current_step': 0,
                'total_steps': 11
            }), 200
        
        return jsonify(progress_data)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'completed': False,
            'steps': {},
            'current_step': 0,
            'total_steps': 11
        }), 200

@app.route('/analysis')
def analysis():
    if 'results' not in session:
        flash('Please upload and analyze a dataset first.', 'error')
        return redirect(url_for('home'))
    
    # Get technical details
    technical_data = {
        'numerical_cols': session.get('numerical_cols', []),
        'categorical_cols': session.get('categorical_cols', []),
        'date_cols': session.get('date_cols', []),
        'derived_features': session.get('derived_features', []),
        'target_col': session.get('target_col', ''),
        'feature_cols': session.get('feature_cols', []),
        'results': session.get('results', {}),
        'best_model_name': session.get('best_model_name', ''),
        'filename': session.get('filename', '')
    }
    
    return render_template('analysis.html', data=technical_data)

if __name__ == '__main__':
    app.run(debug=True)

