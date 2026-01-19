# app.py - Main Flask Application Documentation

## Overview
`app.py` is the core Flask application that handles all backend logic, machine learning processing, and API endpoints for the ML Analytics System.

## Table of Contents
1. [Application Setup](#application-setup)
2. [Progress Tracking System](#progress-tracking-system)
3. [Data Preprocessing Module](#data-preprocessing-module)
4. [Feature Engineering](#feature-engineering)
5. [Model Training Module](#model-training-module)
6. [Business Insights Module](#business-insights-module)
7. [Background Processing](#background-processing)
8. [Flask Routes](#flask-routes)

---

## Application Setup

### Imports and Configuration
```python
- Flask framework for web application
- Pandas/NumPy for data manipulation
- Scikit-learn for ML models (including SimpleImputer for NaN handling)
- Matplotlib/Seaborn for visualizations
- Threading for background processing
- MLxtend for association rules (market basket analysis)
```

### Key Configuration
- **UPLOAD_FOLDER**: Directory for uploaded files (`uploads/`)
- **STATIC_PLOTS**: Directory for generated visualizations (`static/plots/`)
- **Secret Key**: Session management (should be changed in production)
- **No File Size Limit**: Removed MAX_CONTENT_LENGTH restriction for large datasets

---

## Progress Tracking System

### Purpose
Manages real-time progress updates for background ML processing tasks.

### Components

#### Shared State
- `progress_state`: Thread-safe dictionary storing progress for each session
- `results_storage`: Thread-safe dictionary storing completed results
- `progress_lock` & `results_lock`: Thread synchronization locks

#### Functions

**`init_progress(session_id)`**
- Initializes progress state for a new session
- Creates 11-step progress structure
- All steps start as 'pending'

**`update_progress(session_id, step_num, status)`**
- Updates status of a specific step
- Status options: 'pending', 'in_progress', 'completed'
- Updates current_step counter

**`get_progress(session_id)`**
- Retrieves current progress state
- Returns copy of progress dictionary

**`set_error(session_id, error_message)`**
- Sets error state when processing fails
- Stores error message for frontend display

**`mark_completed(session_id)`**
- Marks entire processing as completed
- Signals frontend to display results

---

## Data Preprocessing Module

### Purpose
Handles data loading, cleaning, validation, and transformation.

### Key Functions

#### `load_dataset(filepath)`
- Loads CSV or Excel files
- Returns pandas DataFrame
- Handles file format errors

#### `detect_column_types(df)`
- Automatically detects column types:
  - **Numerical**: int64, float64 columns
  - **Categorical**: object columns (non-date)
  - **Date**: Columns containing date values
- Uses `_is_date_column()` helper to identify dates

#### `handle_missing_values(df, numerical_cols, categorical_cols)`
- **Numerical columns**: Fills with mean (after converting to numeric)
- **Categorical columns**: Fills with mode
- Converts columns to numeric with error handling
- Handles conversion failures gracefully (treats as categorical if conversion fails)

#### `convert_date_columns(df, date_cols)`
- Converts date columns to numerical features:
  - Year, Month, Day, Day of Week
- Drops original date columns
- Returns new numerical column names

#### `encode_categorical(df, categorical_cols)`
- Uses LabelEncoder for categorical variables
- Converts text categories to numerical labels
- Stores encoders for potential inverse transformation

#### `scale_features(X_train, X_test, numerical_cols)`
- Applies StandardScaler to numerical features
- Uses single scaler for all columns (efficient and correct)
- Normalizes features to mean=0, std=1
- Only scales columns that exist in the dataframe
- Returns scaled training and test sets

#### `preprocess_data(df)`
- **Complete preprocessing pipeline**:
  1. Detect column types
  2. Convert date columns to numerical
  3. Handle missing values
  4. Remove duplicates
  5. Encode categorical variables
- Returns processed DataFrame and column type lists

---

## Feature Engineering

### Purpose
Creates derived features to improve model performance.

### `feature_engineering(df, numerical_cols)`

#### Features Created
1. **Interaction Features**: Product of first two numerical columns
   - Example: `column1_x_column2`
2. **Polynomial Features**: Squared values of first numerical column
   - Example: `column1_squared`

#### Process
- Ensures all numerical columns are numeric type
- Creates features only if sufficient columns exist
- Uses try-except to handle IndexError (prevents list index out of range)
- Returns DataFrame with derived features and feature names list

#### Error Handling
- Wrapped in try-except blocks to prevent crashes
- Handles cases where numerical columns list is empty or insufficient

---

## Model Training Module

### Purpose
Trains and evaluates three regression models.

### `prepare_target_variable(df, numerical_cols)`
- Automatically selects target variable
- Uses last numerical column as target
- Creates synthetic target if no numerical columns exist

### `train_models(X_train, X_test, y_train, y_test, session_id=None)`

#### Data Preparation
- Converts pandas DataFrames/Series to numpy arrays
- Handles NaN values using SimpleImputer:
  - X (features): Fills with mean
  - y (target): Fills with median
- Ensures clean data before training

#### Models Trained

1. **Linear Regression**
   - Baseline linear model
   - Fast training, interpretable
   - Updates progress step 5

2. **Random Forest Regressor**
   - 100 trees, max_depth=10
   - Handles non-linearity
   - Updates progress step 6

3. **Gradient Boosting Regressor**
   - 100 estimators, max_depth=5
   - Sequential ensemble learning
   - Updates progress step 7

#### Evaluation Metrics
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R² Score**: Coefficient of Determination (higher is better, 0-1)

### `select_best_model(results)`
- **Selection Criteria**: Combined score
  - Formula: `RMSE + MAE - R²`
  - Lower score = better model
- Returns best model name and metrics

---

## Business Insights Module

### Purpose
Generates five business insights from processed data.

### 1. Customer Segmentation (`customer_segmentation`)
- **Method**: KMeans Clustering
- **Process**:
  - Detects numerical columns from dataframe (includes encoded categorical)
  - Uses 2-3 numerical columns
  - Handles NaN values before clustering
  - Standardizes features
  - Creates 3 customer segments
  - Generates scatter plot visualization
- **Output**: Segment sizes, percentages, interpretations
- **Error Handling**: Returns None if insufficient columns (< 2)

### 2. CLV Prediction (`clv_prediction`)
- **Method**: Uses best regression model
- **Process**:
  - Uses exact training features (same as model was trained on)
  - Ensures feature count matches model expectation
  - Handles NaN values before prediction
  - Predicts Customer Lifetime Value for all rows
  - Creates distribution histogram
  - Identifies top high-value customers
- **Output**: CLV statistics, top customers table
- **Feature Matching**: Automatically uses all training features in correct order

### 3. Churn Prediction (`churn_prediction`)
- **Method**: Random Forest Classifier
- **Process**:
  - Detects numerical columns from dataframe (includes encoded categorical)
  - Creates synthetic churn target (threshold-based)
  - Handles NaN values before training
  - Trains binary classifier
  - Predicts churn probability
  - Categorizes: Low/Medium/High risk
- **Output**: Risk distribution, high-risk customers
- **Error Handling**: Returns None if insufficient columns

### 4. Market Basket Analysis (`market_basket_analysis`)
- **Method**: Apriori Algorithm (Association Rules)
- **Process**:
  - Validates categorical columns exist in dataframe
  - Checks columns have sufficient unique values (≥2)
  - Creates binary matrix from categorical columns (no prefix to avoid errors)
  - Finds frequent itemsets with adaptive min_support
  - Generates association rules
  - Calculates support and confidence
- **Output**: Top product combinations with metrics
- **Error Handling**: Comprehensive validation and error handling, returns None if can't perform

### 5. Sales Forecasting (`sales_forecasting`)
- **Method**: Uses best regression model
- **Process**:
  - Uses exact training features (same as model was trained on)
  - Ensures feature count matches model expectation
  - Handles NaN values before prediction
  - Predicts future sales/demand for all rows
  - Creates actual vs predicted plot (with fallback if numerical_cols empty)
  - Generates forecast table
- **Output**: Forecast values, visualization
- **Feature Matching**: Automatically uses all training features in correct order

---

## Background Processing

### Purpose
Runs ML pipeline in background thread without blocking UI.

### `process_dataset_background(filepath, session_id)`

#### Processing Steps (11 Total)

1. **File uploaded** ✅
   - Marks file upload as complete

2. **Data validation** ⏳→✅
   - Loads dataset
   - Validates minimum rows (≥10)

3. **Data preprocessing** ⏳→✅
   - Runs complete preprocessing pipeline

4. **Feature engineering** ⏳→✅
   - Creates derived features

5. **Train model 1 – Linear Regression** ⏳→✅
   - Trains and evaluates

6. **Train model 2 – Random Forest** ⏳→✅
   - Trains and evaluates

7. **Train model 3 – Gradient Boosting** ⏳→✅
   - Trains and evaluates

8. **Model evaluation** ⏳→✅
   - Metrics already calculated

9. **Best model selection** ⏳→✅
   - Selects optimal model

10. **Final prediction generation** ⏳→✅
    - Predictions already generated

11. **Business insights ready** ⏳→✅
    - Generates all 5 insights
    - Creates visualizations

#### Error Handling
- Wraps entire process in try-except
- Sets error state on failure
- Stores error message for frontend

#### Result Storage
- Stores all results in `results_storage` dictionary (in-memory)
- Includes: processed data (as JSON), model results, insights
- **Session Optimization**: Only stores small metadata in Flask session
- **Large Data**: DataFrame JSON stored in `results_storage`, NOT in session (prevents ERR_RESPONSE_HEADERS_TOO_BIG)
- Accessible via session ID

---

## Flask Routes

### `@app.route('/', methods=['GET', 'POST'])` - Home Page

#### POST Request (File Upload)
1. Validates file format (CSV/Excel)
2. Saves file to uploads folder
3. Generates unique session ID
4. Initializes progress tracking
5. Starts background processing thread
6. Redirects to home page

#### GET Request (Display Results)
1. Checks for completed processing
2. Loads results from storage
3. Stores in session for persistence
4. Renders home.html with insights

### `@app.route('/progress')` - Progress API

**Purpose**: Returns current processing status as JSON

**Response Format**:
```json
{
  "current_step": 5,
  "total_steps": 11,
  "steps": {
    "1": {"name": "File uploaded", "status": "completed"},
    "2": {"name": "Data validation", "status": "completed"},
    ...
  },
  "completed": false,
  "error": null
}
```

**Error Handling**:
- Returns empty progress structure if no session
- Handles missing progress gracefully

### `@app.route('/analysis')` - Analysis Page

**Purpose**: Displays technical model evaluation

**Requirements**:
- User must have completed analysis
- Redirects to home if no results

**Data Passed**:
- Column types (numerical, categorical, date)
- Derived features
- Model performance metrics
- Best model selection
- Feature engineering details

---

## Key Design Patterns

### 1. Thread-Safe Operations
- All shared state uses locks
- Prevents race conditions
- Ensures data consistency

### 2. Error Handling
- Try-except blocks throughout
- Graceful degradation
- User-friendly error messages
- NaN value handling at all stages
- List index protection (prevents IndexError)
- Feature mismatch prevention

### 3. Modular Design
- Separate functions for each task
- Reusable components
- Easy to test and maintain

### 4. Background Processing
- Non-blocking UI
- Real-time progress updates
- Session-based tracking

---

## Data Flow

```
File Upload
    ↓
Background Thread Started
    ↓
Progress Tracking Initialized
    ↓
Data Loading & Validation
    ↓
Preprocessing Pipeline
    ↓
Feature Engineering
    ↓
Model Training (3 models)
    ↓
Model Evaluation
    ↓
Best Model Selection
    ↓
Business Insights Generation
    ↓
Results Storage
    ↓
Frontend Display
```

---

## Important Notes

1. **Session Management**: Uses Flask sessions for user state (optimized - no large DataFrames)
2. **Thread Safety**: All shared data uses locks
3. **Error Recovery**: Errors don't crash the application
4. **Scalability**: Background processing allows concurrent users
5. **Data Persistence**: Results stored in memory (`results_storage`) and small metadata in session
6. **NaN Handling**: Comprehensive NaN handling at all stages (preprocessing, training, predictions)
7. **Feature Matching**: Business insights use exact training features to prevent mismatch errors
8. **Date Handling**: Automatic date detection and conversion to numerical features
9. **List Safety**: All list indexing protected with try-except and length checks

---

## Recent Improvements

### NaN Handling
- Added SimpleImputer for comprehensive NaN handling
- NaN filling at preprocessing, training, and prediction stages
- Prevents "Input X contains NaN" errors

### Feature Matching
- Business insights now use exact training features
- Prevents "X has N features, but model expects M" errors
- Automatic feature ordering and padding

### Error Prevention
- List index protection (prevents IndexError)
- Column existence validation
- Graceful fallbacks for missing data

### Session Optimization
- Large DataFrames NOT stored in Flask session
- Prevents ERR_RESPONSE_HEADERS_TOO_BIG errors
- Only metadata stored in session

### Date Column Handling
- Improved date detection (checks multiple values)
- Automatic conversion to numerical features
- Handles various date formats

## Future Enhancements

- Database integration for result persistence
- Model persistence (pickle/joblib)
- Batch processing for multiple files
- Advanced feature engineering options
- Custom model hyperparameter tuning
- Export results to various formats
- File-based storage for large datasets

