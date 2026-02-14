import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/electricity_usage.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())
# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicates (if any)
df = df.drop_duplicates()

print("\nCleaned Shape:", df.shape)
print("\nBasic Statistics:")
print(df["units"].describe())
plt.figure()
df[df["block"] == "Block_A"].set_index("datetime")["units"].resample("D").mean().plot()
plt.title("Daily Average Electricity Usage - Block A")
plt.xlabel("Date")
plt.ylabel("Units Consumed")
plt.tight_layout()
plt.show()
weekday_avg = df.groupby("day_type")["units"].mean()
print("\nAverage usage by day type:")
print(weekday_avg)

weekday_avg.plot(kind="bar", title="Weekday vs Weekend Usage")
plt.ylabel("Average Units")
plt.tight_layout()
plt.show()
