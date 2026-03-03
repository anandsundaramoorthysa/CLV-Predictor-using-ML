---
title: Customer Lifetime Value (CLV) Predictor
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Customer Lifetime Value (CLV) Predictor

A Flask web app that predicts **Customer Lifetime Value** (total amount a customer is likely to spend) using pre-trained machine learning models. Enter customer and transaction details to get an instant CLV prediction and segment (Low / Medium / High value).

## Features

- **CLV prediction**: Predict total spend from age, gender, income, purchases, product category, payment/shipping, segment, ratings, feedback, month, brand, and product type.
- **Five models**: Linear Regression, Random Forest, Gradient Boosting, Extra Trees, LightGBM — best model (by validation RMSE) is used for the main prediction.
- **Segment label**: Automatically classifies prediction as Low / Medium / High value customer using data percentiles.
- **About page**: Model comparison, correlation analysis, feature importance, and test-set predictions.

## Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK.
3. Clone your Space repo and push this project (including `models/`, `outputs/`, `saved_model/`, and generated files in `static/`):

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   cp -r /path/to/Marketing_Project/* .
   git add .
   git commit -m "Add CLV Predictor app"
   git push
   ```

4. Ensure these are **committed** (not in `.gitignore`) so the Space can run:
   - `models/*.pkl`
   - `outputs/` (all JSON, CSV, pkl)
   - `saved_model/`
   - `static/style.css`, `static/correlation_analysis.png`, `static/feature_importance.png`, `static/model_predictions_comparison.png`
   - `templates/`, `app.py`, `requirements.txt`, `Dockerfile`

5. The Space will build the Docker image and serve the app on port **7860**.

## Project structure (for deployment)

```
├── app.py              # Flask app
├── requirements.txt    # Python deps (includes gunicorn)
├── Dockerfile          # For Hugging Face Spaces
├── .dockerignore       # Keeps build context small
├── models/             # Trained pipelines (.pkl) — committed for HF
├── outputs/            # Evaluation, stats, form options — committed for HF
├── saved_model/        # Legacy stats — committed for HF
├── templates/          # HTML
└── static/             # CSS and generated PNGs
```

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt` (Flask, pandas, scikit-learn, LightGBM, gunicorn, etc.)

## License

MIT / academic use.
