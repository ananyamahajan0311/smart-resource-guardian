import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import load_data
from src.feature_engineering import create_features
from src.train import train_models, save_model
from src.evaluate import evaluate_model
from src.anomaly import train_anomaly_model, detect_anomalies
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


def run_pipeline():

    print("Loading data...")
    df = load_data()

    print("Creating features...")
    df = create_features(df)

    X = df.drop(columns=["units", "datetime", "day_type"])
    y = df["units"]

    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    import numpy as np

    # Initial Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training models...")
    models = train_models(X_train, y_train)

    # 🔥 TimeSeries Cross Validation
    print("\nPerforming Time Series Cross Validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    for name, model in models.items():
        r2_scores = []

        for train_index, test_index in tscv.split(X):
            X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
            y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train_cv, y_train_cv)
            _, _, r2 = evaluate_model(model, X_test_cv, y_test_cv)

            r2_scores.append(r2)

        print(f"{name} Average R2 (CV): {np.mean(r2_scores):.3f}")

    # 🔥 Final Evaluation on Holdout Set
    best_model = None
    best_r2 = -1

    for name, model in models.items():
        print(f"\nEvaluating {name} on Holdout Set...")
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

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        import matplotlib.pyplot as plt

        importances = best_model.feature_importances_
        feature_names = X.columns

        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, importances)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png")
        plt.close()

        print("Feature importance plot saved in reports/")

    # Anomaly Detection
    print("\nTraining anomaly detection model...")
    anomaly_model = train_anomaly_model(df)

    df_with_anomalies = detect_anomalies(anomaly_model, df)
    anomaly_count = df_with_anomalies["anomaly"].sum()

    print(f"Total anomalies detected: {anomaly_count}")

    print("\nPipeline completed successfully!")
if __name__ == "__main__":
    run_pipeline()
