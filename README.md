# Smart Resource Guardian

## Overview
Smart Resource Guardian is an AI-powered electricity consumption prediction and anomaly detection system designed for institutional resource monitoring.

## Features
- Time-series feature engineering
- Model comparison (Linear Regression vs Random Forest)
- Automatic best model selection
- Isolation Forest anomaly detection
- REST API using FastAPI
- Swagger documentation

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn

## Model Performance
Random Forest:
- R² Score: 0.86
- MAE: 1.396
- RMSE: 1.773

## API Endpoints
- GET /health
- POST /predict
- POST /anomaly

## Future Improvements
- Real-time data streaming
- Cloud deployment
- Frontend dashboard
- Hyperparameter tuning
