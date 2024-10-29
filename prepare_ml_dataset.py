import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def process_price_data(df1, df2):
    """
    Process price data and calculate differences.

    Args:
        df1: Price data DataFrame
        df2: Difference data DataFrame (with timestamp columns)
    """
    # Create copies of input DataFrames to avoid modifications to originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Convert the column headers (except 'slug') to datetime
    timestamp_columns = [col for col in df2.columns if col != "slug"]
    df2.columns = ["slug"] + [pd.to_datetime(col) for col in timestamp_columns]

    # Melt the second dataframe to convert columns to rows, with timestamps as columns
    df2_melted = df2.melt(
        id_vars=["slug"], var_name="timestamp", value_name="difference"
    )

    # Convert df1 timestamp to datetime
    df1["timestamp"] = pd.to_datetime(df1["timestamp"])

    # Create a mapping dictionary for shifted timestamps
    unique_timestamps = sorted(df1["timestamp"].unique())
    timestamp_mapping = {}

    for i in range(len(unique_timestamps) - 1):
        timestamp_mapping[unique_timestamps[i]] = unique_timestamps[i + 1]

    # Create an efficient lookup dictionary for differences
    diff_dict = df2_melted.set_index(["timestamp", "slug"])["difference"].to_dict()

    def get_difference(row):
        if row["timestamp"] in timestamp_mapping:
            mapped_time = timestamp_mapping[row["timestamp"]]
            diff_value = diff_dict.get((mapped_time, row["slug"]))
            if diff_value is not None and diff_value != 0:  # Check for non-zero value
                return diff_value + row["quote.USD.price"]
            elif diff_value == 0:  # If difference is 0, use quote price
                return row["quote.USD.price"]
        return None  # Return None for the last timestamp

    # Add the target column
    df1["target"] = df1.apply(get_difference, axis=1)

    # Sort by timestamp and slug for consistency
    return df1.sort_values(["timestamp", "slug"]).reset_index(drop=True)


def construct_ml_data():
    """
    Load new data, process it, and save to output file.
    """
    output_path = rf"C:/Users/{os.getenv('USER')}/Desktop/ml_training_data.csv"
    try:
        # Load new data
        file1 = pd.read_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
        )
        file2 = pd.read_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_PromisingCurrencies.csv"
        )

        # Process data
        result = process_price_data(file1, file2)

        # Save the result (overwrite mode)
        result.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}")
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
