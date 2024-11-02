import pandas as pd
import os
from price_difference import calculate_latest_price_difference
import glob


def append_price_changes(Crypto_API_df):

    csv_files = {
        "PromisingCurrencies",
    }

    # Loop through each file and calculate the price differences
    for csv_file in csv_files:

        price_diff_output_file = rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_{csv_file}.csv"

        # Read the CSV file
        price_diff_df = pd.read_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{csv_file}.csv"
        )

        # Calculate and append price differences between consecutive timestamps
        calculate_latest_price_difference(
            price_diff_df, price_diff_output_file, Crypto_API_df
        )

        print(
            f"Processed file: {csv_file}, differences saved to: {price_diff_output_file}"
        )
