import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np

google_trend_path = "../north_america_australia/canada/Beauty_Fitness/original_parallel.csv"
original_path = "../../Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Lifestyle/real_data/Canada_with_relative_traffic_rates.csv"

google_trend_df = pd.read_csv(google_trend_path)
original_df = pd.read_csv(original_path)

thresholds = [0.0, 0.1, 0.25, 0.5, 0.75, 1]

data_column = google_trend_df.columns[1]

min_value = google_trend_df[data_column].min()
# print("min_value: ",min_value)
# print("max_value: ",google_trend_df[data_column].max())

google_trend_df[data_column] = google_trend_df[data_column] - min_value

max_value = google_trend_df[data_column].max()
# print("normalized max: ",max_value)

def relativeValue(value, max):
    if pd.isna(value):
        return 0.1
        
    rate = int(value * 100 / max)
    if rate > 0 and rate < 50:
        return 0.1
    elif rate >= 50 and rate < 70:
        return 0.25
    elif rate >= 70 and rate < 80:
        return 0.5
    elif rate >= 80:
        return 1
    else:
        return 0.0

google_trend_df[data_column] = google_trend_df[data_column].apply(
    lambda x: relativeValue(x, max_value)
)

def compute_similarity(df1, df2):
    common_dates = set(df1["date"]).intersection(df2["Day"])
    df1 = df1[df1["date"].isin(common_dates)]
    df2 = df2[df2["Day"].isin(common_dates)]

    data_column1 = df1.columns[2]
    data_column2 = df2.columns[1]

    df1 = df1.sort_values("date").reset_index(drop=True)
    df2 = df2.sort_values("Day").reset_index(drop=True)

    if len(df1) < 2 or len(df2) < 2:
        return None
    
    vector1 = list(df1[data_column1])
    vector2 = list(df2[data_column2])
    vector2 = [0.1 if pd.isna(x) else x for x in vector2]
    
    print("data_column1: ", data_column1)
    print("data_column2: ", data_column2)
    print("df1[data_column1]: ", vector1)
    print("df2[data_column2]: ", vector2)

    if all(not pd.isna(x) for x in vector1) and all(not pd.isna(x) for x in vector2):
        traffic_rate_similarity = 1 - cosine(vector1, vector2)
        return traffic_rate_similarity
    else:
        print("Warning: Still have NaN values after cleaning")
        return 0

similarity = compute_similarity(original_df, google_trend_df)
print("similarity: ", similarity)