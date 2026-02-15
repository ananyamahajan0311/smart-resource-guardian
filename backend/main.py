from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained models
regression_model = joblib.load("models/best_model.pkl")
anomaly_model = joblib.load("models/anomaly_model.pkl")


@app.get("/health")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = regression_model.predict(df)

    return {"predicted_units": float(prediction[0])}


@app.post("/anomaly")
def detect_anomaly(data: dict):

    df = pd.DataFrame([data])

    prediction = anomaly_model.predict(df[["units"]])

    is_anomaly = 1 if prediction[0] == -1 else 0

    return {"anomaly": is_anomaly}
