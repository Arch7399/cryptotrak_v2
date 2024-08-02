import pandas as pd
import os


def calculate_latest_price_difference(df, diff_csv_path, df2):
    """
    Calculate and append price differences for cryptocurrencies listed in the price difference CSV
    or in the current top 20 predictions (df).
    """
    # Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])

    # Drop any duplicate rows for the same `slug` and `timestamp`
    df = df.drop_duplicates(subset=["slug", "timestamp"], keep="last")
    df2 = df2.drop_duplicates(subset=["slug", "timestamp"], keep="last")

    # Get the two latest timestamps
    latest_timestamps = df["timestamp"].drop_duplicates().nlargest(2)

    # Filter data for the two latest timestamps
    latest_df = df[df["timestamp"].isin(latest_timestamps)]
    latest_df2 = df2[df2["timestamp"].isin(latest_timestamps)]

    # Load the existing slugs from the price difference CSV, if it exists
    if os.path.exists(diff_csv_path):
        existing_diff_df = pd.read_csv(diff_csv_path, index_col="slug")
        existing_slugs = existing_diff_df.index
    else:
        existing_diff_df = None
        existing_slugs = pd.Index([])

    # Combine slugs from the top 20 predictions (df) and the existing slugs in the diff CSV
    combined_slugs = pd.Index(df["slug"]).union(existing_slugs)

    # Filter df2 to only include rows where slug is in the combined slugs list
    filtered_df2 = latest_df2[latest_df2["slug"].isin(combined_slugs)]

    # Pivot the filtered data to have cryptocurrencies as rows and timestamps as columns
    pivot_df2 = filtered_df2.pivot(
        index="slug", columns="timestamp", values="quote.USD.price"
    )

    # Ensure we have exactly two columns (latest and second-latest timestamps)
    if pivot_df2.shape[1] < 2:
        raise ValueError("Not enough timestamps to calculate the difference.")

    # Calculate the difference between the latest and second-latest timestamps
    diff_df = pivot_df2.iloc[:, 0].subtract(pivot_df2.iloc[:, 1])

    # Rename the difference column to reflect the latest timestamp
    latest_timestamp = pivot_df2.columns[1]
    diff_df.name = latest_timestamp

    # Convert timestamp columns to strings before saving
    pivot_df2.columns = [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in pd.to_datetime(pivot_df2.columns)]

    # If the diff CSV file already exists, append the new column to it
    if existing_diff_df is not None:
        # Merge the new diff column with the existing CSV
        combined_df = pd.concat([existing_diff_df, diff_df], axis=1)
    else:
        # If the file doesn't exist, create a new one with the current diff column
        combined_df = pd.DataFrame(diff_df)

    

    # Save the updated DataFrame to the CSV (overwrite with new data)
    combined_df.to_csv(diff_csv_path, mode="w", header=True, index=True)

    print(f"Price differences updated in {diff_csv_path}")
