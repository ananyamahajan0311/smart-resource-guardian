from sklearn.ensemble import IsolationForest
import joblib


def train_anomaly_model(df):

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    model.fit(df[["units"]])

    joblib.dump(model, "models/anomaly_model.pkl")

    return model


def detect_anomalies(model, df):

    df["anomaly_score"] = model.decision_function(df[["units"]])
    df["anomaly"] = model.predict(df[["units"]])

    # Convert -1 to 1 (anomaly), 1 to 0 (normal)
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    return df
