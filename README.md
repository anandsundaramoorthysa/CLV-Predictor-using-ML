# Machine Learning Analytics System

A comprehensive Flask-based web application for automated machine learning analysis and business insights generation.

## Features

- **Automated Data Preprocessing**: Handles missing values, encoding, scaling, and feature engineering
- **Multiple ML Models**: Trains Linear Regression, Random Forest, and Gradient Boosting models
- **Automatic Model Selection**: Selects the best model based on RMSE, MAE, and R² scores
- **Business Insights**: Generates customer segmentation, CLV prediction, churn analysis, market basket analysis, and sales forecasting
- **Technical Analysis**: Detailed model evaluation and performance metrics

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload Dataset**: Click "Choose CSV or Excel file" and select your dataset
2. **Analyze**: Click "Upload & Analyze" to process your data
3. **View Insights**: Business insights will appear on the home page
4. **Technical Analysis**: Click "View Technical Analysis" to see detailed model evaluation

## Dataset Requirements

- **Format**: CSV (.csv) or Excel (.xlsx, .xls)
- **Size**: Maximum 16MB
- **Minimum Rows**: At least 10 rows
- **Columns**: Should contain both numerical and categorical columns for best results

## Project Structure

```
Marketing_Project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   ├── home.html         # Home page template
│   └── analysis.html     # Analysis page template
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── plots/            # Generated visualizations
└── uploads/              # Uploaded files (created automatically)
```

## Technical Details

### Models Trained
1. **Linear Regression**: Baseline linear model
2. **Random Forest**: Ensemble method with 100 trees
3. **Gradient Boosting**: Sequential ensemble with 100 estimators

### Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of Determination

### Business Insights Generated
1. Customer Segmentation (KMeans clustering)
2. Customer Lifetime Value (CLV) Prediction
3. Churn Prediction
4. Market Basket Analysis (Apriori algorithm)
5. Sales/Demand Forecasting

## Notes

- The application automatically detects column types (numerical, categorical, date)
- Missing values are handled automatically (mean for numerical, mode for categorical)
- The best model is selected automatically based on combined performance metrics
- All visualizations are saved in `static/plots/` directory

## License

This project is created for academic/educational purposes.

