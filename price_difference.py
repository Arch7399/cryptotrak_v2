import pandas as pd
import os


def calculate_latest_price_difference(df, diff_csv_path):
    """
    This function calculates the difference in `quote.USD.price` between the latest and the second-latest timestamps
    for each cryptocurrency, and appends the new timestamp column with the price difference
    to an existing CSV file.
    """
    # Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get the two latest timestamps
    latest_timestamps = df["timestamp"].drop_duplicates().nlargest(2)

    # Filter data for the two latest timestamps
    latest_df = df[df["timestamp"].isin(latest_timestamps)]

    # Pivot the data to have cryptocurrencies as rows and timestamps as columns
    pivot_df = latest_df.pivot(
        index="slug", columns="timestamp", values="quote.USD.price"
    )

    # Ensure we have exactly two columns (latest and second-latest timestamps)
    if pivot_df.shape[1] < 2:
        raise ValueError("Not enough timestamps to calculate the difference.")

    # Calculate the difference between the latest and second-latest timestamps
    diff_df = pivot_df.iloc[:, 0].subtract(pivot_df.iloc[:, 1])

    # Rename the difference column to reflect the latest timestamp
    latest_timestamp = pivot_df.columns[1]
    diff_df.name = latest_timestamp

    # If the diff CSV file already exists, read it and append the new column
    if os.path.exists(diff_csv_path):
        existing_diff_df = pd.read_csv(diff_csv_path, index_col="slug")

        # Merge the new diff column with the existing CSV
        combined_df = pd.concat([existing_diff_df, diff_df], axis=1)
    else:
        # If the file doesn't exist, create a new one with the current diff column
        combined_df = pd.DataFrame(diff_df)

    # Save the updated DataFrame to the CSV (overwrite with new data)
    combined_df.to_csv(diff_csv_path, mode="w", header=True, index=True)
