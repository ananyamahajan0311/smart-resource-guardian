from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import joblib
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained models
regression_model = joblib.load("models/best_model.pkl")
anomaly_model = joblib.load("models/anomaly_model.pkl")

print("Model features:", regression_model.feature_names_in_)


@app.get("/health")
def health_check():
    return {"status": "API is running"}

from fastapi import HTTPException

@app.post("/predict")
def predict(data: dict):

    required_columns = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "block_Block_B",
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "rolling_std_24"
]


    df = pd.DataFrame([data])

    # Add missing columns if any
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Force exact column order
    df = df[required_columns]

    prediction = regression_model.predict(df)

    return {"predicted_units": float(prediction[0])}



@app.post("/anomaly")
def detect_anomaly(data: dict):

    df = pd.DataFrame([data])

    prediction = anomaly_model.predict(df[["units"]])

    is_anomaly = 1 if prediction[0] == -1 else 0

    return {"anomaly": is_anomaly}
