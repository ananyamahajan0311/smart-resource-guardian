import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import load_data
from src.feature_engineering import create_features
from src.train import train_models, save_model
from src.evaluate import evaluate_model
from src.anomaly import train_anomaly_model, detect_anomalies


def run_pipeline():

    print("Loading data...")
    df = load_data()

    print("Creating features...")
    df = create_features(df)

    X = df.drop(columns=["units", "datetime", "day_type"])
    y = df["units"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training models...")
    models = train_models(X_train, y_train)

    best_model = None
    best_r2 = -1

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        print(f"{name} Performance:")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name}")
    save_model(best_model, "best_model")

    # ✅ ANOMALY SECTION INSIDE FUNCTION
    print("\nTraining anomaly detection model...")
    anomaly_model = train_anomaly_model(df)

    df_with_anomalies = detect_anomalies(anomaly_model, df)
    anomaly_count = df_with_anomalies["anomaly"].sum()

    print(f"Total anomalies detected: {anomaly_count}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
