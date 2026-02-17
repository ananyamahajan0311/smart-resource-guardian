import pandas as pd

def create_features(df):

    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # One-hot encode block column
    df = pd.get_dummies(df, columns=["block"], drop_first=True)

    # Lag features
    df["lag_1"] = df["units"].shift(1)
    df["lag_24"] = df["units"].shift(24)

    # 🔥 NEW Rolling Features
    df["rolling_mean_24"] = df["units"].rolling(24).mean()
    df["rolling_std_24"] = df["units"].rolling(24).std()

    df = df.dropna()

    return df
