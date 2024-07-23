import pandas as pd
import os
from price_difference import calculate_latest_price_difference
import glob


def append_price_changes(df2):

    csv_files = {
        "Bullish",
        "HighRSI",
        "LowRSI",
        "PriceSpikes",
        "LowLiquidity",
        "VolumeSurges",
        "PromisingCurrencies",
    }

    # Loop through each file and calculate the price differences
    for csv_file in csv_files:

        output_file = rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_{csv_file}.csv"

        # Read the CSV file
        df = pd.read_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{csv_file}.csv"
        )

        # Calculate and append price differences between consecutive timestamps
        calculate_latest_price_difference(df, output_file, df2)

        print(f"Processed file: {csv_file}, differences saved to: {output_file}")
