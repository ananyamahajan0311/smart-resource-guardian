import pandas as pd

def load_data(path="data/raw/electricity_usage.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.drop_duplicates()
    return df
