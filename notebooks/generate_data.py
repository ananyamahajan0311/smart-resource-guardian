import pandas as pd
import numpy as np

# Date range (6 months hourly data)
date_range = pd.date_range(
    start="2025-01-01",
    end="2025-06-30",
    freq="H"
)

blocks = ["Block_A", "Block_B"]
data = []

for block in blocks:
    base_load = 20 if block == "Block_A" else 15

    for dt in date_range:
        hour = dt.hour
        day_type = "Weekend" if dt.weekday() >= 5 else "Weekday"

        # Higher usage during day & evening
        if 6 <= hour <= 10:
            usage = base_load + np.random.uniform(5, 10)
        elif 18 <= hour <= 22:
            usage = base_load + np.random.uniform(8, 15)
        else:
            usage = base_load + np.random.uniform(1, 4)

        # Add noise
        usage += np.random.normal(0, 1)

        data.append([
            dt, block, round(max(usage, 1), 2), day_type, hour
        ])

df = pd.DataFrame(
    data,
    columns=["datetime", "block", "units", "day_type", "hour"]
)

df.to_csv("data/raw/electricity_usage.csv", index=False)

print("Dataset generated successfully!")
