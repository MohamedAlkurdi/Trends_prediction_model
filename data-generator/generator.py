import pandas as pd
import numpy as np
from datetime import timedelta


def generate_fake_data(real_data: pd.DataFrame, num_days: int) -> pd.DataFrame:

    real_data["date"] = pd.to_datetime(real_data["date"])

    start_date = real_data["date"].max() + timedelta(days=1)
    end_date = start_date + timedelta(days=num_days - 1)
    date_range = pd.date_range(start=start_date, end=end_date)

    synthetic_data = {
        "date": [],
        "general_label": [],
        "traffic_rate": [],
        "total_traffic": [],
    }

    general_label = real_data["general_label"].iloc[0]

    mean_traffic_rate = real_data["traffic_rate"].mean()
    std_traffic_rate = real_data["traffic_rate"].std()
    mean_total_traffic = real_data["total_traffic"].mean()
    std_total_traffic = real_data["total_traffic"].std()

    for date in date_range:
        seasonal_variation = np.sin(2 * np.pi * (date.dayofyear / 365))
        traffic_rate = np.clip(np.random.normal(mean_traffic_rate + seasonal_variation * 0.1, std_traffic_rate * 0.5),0,1,)
        total_traffic = np.clip(np.random.normal(mean_total_traffic * traffic_rate, std_total_traffic * 0.5),0,None,)

        synthetic_data["date"].append(date)
        synthetic_data["general_label"].append(general_label)
        synthetic_data["traffic_rate"].append(traffic_rate)
        synthetic_data["total_traffic"].append(total_traffic)

    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df


path = "/Classification/output/regions/africa/genral_labeled_data_with_relative_traffic_rates/Entertainment/real_data/Kenya_with_relative_traffic_rates.csv"
df = pd.read_csv(path)

fake_data = generate_fake_data(df, 10)

print("testing generate_fake_data: ", fake_data)
