from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib


def train_models(X_train, y_train):

    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["LinearRegression"] = lr

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    return models


def save_model(model, name):
    joblib.dump(model, f"models/{name}.pkl")
