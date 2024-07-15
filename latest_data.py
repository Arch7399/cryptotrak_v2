import pandas as pd


def filter_latest(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get the latest timestamp
    latest_timestamps = df["timestamp"].drop_duplicates().nlargest(1)

    # Filter data for the latest timestamp
    latest_df = df[df["timestamp"].isin(latest_timestamps)]

    return latest_df
